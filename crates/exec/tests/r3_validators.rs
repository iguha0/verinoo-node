//! Phase R3 — multi-validator consensus tests.
//!
//! Covers: leader rotation determinism, wrong-producer rejection,
//! pubkey-binding rejection, staked register/unregister lifecycle,
//! stake accounting, and a live two-producer alternating chain.

use verinoo_exec::engine::{
    current_head, produce_block, set_head, validate_block, Block, BlockHeader, ProducerKey,
};
use verinoo_exec::{LeaderPolicy, RoundRobin};
use verinoo_state::StateStore;
use verinoo_types::tx::{TxData, TxEnvelope, TxKind};

use ed25519_dalek::Signer as _;
use sha2::Digest as _;

fn keypair(name: &str) -> ed25519_dalek::SigningKey {
    let mut h = sha2::Sha256::new();
    h.update(format!("key:{name}").as_bytes());
    let digest: [u8; 32] = h.finalize().into();
    ed25519_dalek::SigningKey::from_bytes(&digest)
}

fn addr(kp: &ed25519_dalek::SigningKey) -> String {
    verinoo_exec::engine::public_key_to_address(&hex::encode(kp.verifying_key().to_bytes()))
        .unwrap()
}

fn make_tx(
    signer: &ed25519_dalek::SigningKey,
    from_addr: &str,
    to: &str,
    value: u64,
    kind: TxKind,
    data: serde_json::Value,
    nonce: u64,
) -> TxEnvelope {
    let data = TxData { r#type: kind, data };
    let payload = verinoo_types::canonical::UnsignedPayload {
        data: serde_json::to_value(&data).unwrap(),
        from: from_addr.into(),
        nonce,
        to: to.into(),
        value,
        gas_limit: None,
        gas_price: None,
    };
    let tx_id = payload.tx_id().unwrap();
    let sig = signer.sign(tx_id.as_bytes());
    TxEnvelope {
        tx_id,
        from: from_addr.into(),
        to: to.into(),
        value,
        nonce,
        data,
        gas_limit: None,
        gas_price: None,
        public_key: hex::encode(signer.verifying_key().to_bytes()),
        signature: hex::encode(sig.to_bytes()),
    }
}

/// Testnet harness: genesis validators seeded, empty blocks produced up
/// to `height`, keypairs addressable by test name.
struct Chain {
    store: StateStore,
    keys: std::collections::HashMap<String, ed25519_dalek::SigningKey>,
    addresses: std::collections::HashMap<String, String>,
    nonces: std::collections::HashMap<String, u64>,
    now_ms: u64,
}

impl Chain {
    fn fresh(validator_names: &[&str]) -> Self {
        Self::with_keys(validator_names, &[])
    }

    fn with_keys(validators: &[&str], extra_keys: &[&str]) -> Self {
        let store = StateStore::open_temp().unwrap();

        let mut keys = std::collections::HashMap::new();
        let mut addresses = std::collections::HashMap::new();
        for name in validators.iter().chain(extra_keys.iter()) {
            let kp = keypair(name);
            addresses.insert(name.to_string(), addr(&kp));
            keys.insert(name.to_string(), kp);
        }

        // Genesis block.
        let mut h = BlockHeader {
            hash: format!("genesis-{}", 0),
            version: 1,
            index: 0,
            timestamp: 0,
            previous_hash: "f".repeat(64),
            validator: "genesis".into(),
            validator_pub_key: String::new(),
            validator_signature: String::new(),
            state_root: String::new(),
            tx_root: String::new(),
            inference_tasks_root: String::new(),
            compute_root: String::new(),
        };
        h.hash = h.hash.clone(); // opaque marker for tests
        let genesis = Block {
            header: h.clone(),
            transactions: vec![],
        };
        set_head(&store, &genesis);

        // Seed valset.
        let vals: Vec<String> = validators.iter().map(|n| addresses[*n].clone()).collect();
        store.seed_genesis_validators(&vals, 0);

        Self {
            store,
            keys,
            addresses,
            nonces: std::collections::HashMap::new(),
            now_ms: 1_000_000,
        }
    }

    fn addr_of(&self, name: &str) -> String {
        self.addresses[name].clone()
    }

    fn fund(&self, name_or_addr: &str, amount: i64) {
        let addr = self
            .addresses
            .get(name_or_addr)
            .cloned()
            .unwrap_or_else(|| name_or_addr.to_string());
        let mut acc = self.store.account_or_default(&addr);
        acc.balance = amount;
        self.store.set_account(&acc);
    }

    /// Produce an EMPTY block by whatever validator is scheduled.
    fn tick(&mut self) -> Result<Block, String> {
        self.produce_by_scheduled(vec![])
    }

    /// Produce a block by whoever is SCHEDULED at the next height,
    /// including the given transactions.
    fn produce_scheduled(&mut self, txs: Vec<TxEnvelope>) -> Result<Block, String> {
        self.produce_by_scheduled(txs)
    }

    fn produce_by_scheduled(&mut self, txs: Vec<TxEnvelope>) -> Result<Block, String> {
        let prev = current_head(&self.store).unwrap();
        let next_height = prev.header.index + 1;
        let vals = self.store.validator_addresses();
        let policy = RoundRobin;
        let leader = policy
            .leader_at(next_height, &vals)
            .ok_or("empty validator set")?;

        // Find the keypair whose address matches the leader.
        let (leader_name, leader_kp) = self
            .keys
            .iter()
            .find(|(_, kp)| addr(kp) == leader)
            .map(|(n, kp)| (n.clone(), kp.clone()))
            .ok_or_else(|| format!("leader {leader} has no test keypair"))?;

        let producer = ProducerKey::from_parts(leader, leader_kp);
        let block = produce_block(&self.store, &txs, &producer, self.now_ms).map_err(|e| {
            eprintln!("[produce err] {e}");
            e.to_string()
        })?;
        validate_block(&block, &prev.header, &vals, self.now_ms).map_err(|e| {
            eprintln!("[validate err] {e}");
            e.to_string()
        })?;
        self.now_ms += 10_000;
        set_head(&self.store, &block);

        // Track nonces of executed senders so subsequent txs stay sequential.
        for t in &block.transactions {
            for (name, kp) in &self.keys {
                if addr(kp) == t.from {
                    self.nonces.insert(name.clone(), t.nonce);
                }
            }
        }
        let _ = &leader_name;
        Ok(block)
    }

    /// Produce by a SPECIFIC validator (may violate schedule — for rejection tests).
    fn produce_by_name(&mut self, name: &str, txs: Vec<TxEnvelope>) -> Result<Block, String> {
        let kp = match self.keys.get(name) {
            Some(k) => k.clone(),
            None => {
                // Lazily register any named keypair so tests can reference
                // non-validator identities.
                let kp = keypair(name);
                let a = addr(&kp);
                self.keys.insert(name.to_string(), kp.clone());
                self.addresses.insert(name.to_string(), a.clone());
                kp
            }
        };
        let producer = ProducerKey::from_parts(addr(&kp), kp);
        let block =
            produce_block(&self.store, &txs, &producer, self.now_ms).map_err(|e| e.to_string())?;
        let prev = current_head(&self.store).unwrap();
        validate_block(
            &block,
            &prev.header,
            &self.store.validator_addresses(),
            self.now_ms,
        )
        .map_err(|e| e.to_string())?;
        self.now_ms += 10_000;
        set_head(&self.store, &block);
        Ok(block)
    }
}

// ================= rotation =================

#[test]
fn round_robin_rotation_is_deterministic() {
    let vals = vec!["c".into(), "a".into(), "b".into()];
    // sorted = [a, b, c]; height % 3 cycles a -> b -> c -> a ...
    assert_eq!(RoundRobin.leader_at(1, &vals).as_deref(), Some("a"));
    assert_eq!(RoundRobin.leader_at(2, &vals).as_deref(), Some("b"));
    assert_eq!(RoundRobin.leader_at(3, &vals).as_deref(), Some("c"));
    assert_eq!(RoundRobin.leader_at(4, &vals).as_deref(), Some("a"));
    assert_eq!(
        RoundRobin.leader_at(0, &vals),
        None,
        "height 0: no producer"
    );
    assert_eq!(RoundRobin.leader_at(5, &[]), None);
}

#[test]
fn three_validators_alternate_production() {
    let mut chain = Chain::fresh(&["alpha", "beta", "gamma"]);

    let expected = ["alpha", "beta", "gamma", "alpha", "beta", "gamma"];
    for (i, name) in expected.iter().enumerate() {
        let block = chain
            .produce_by_scheduled(vec![])
            .unwrap_or_else(|e| panic!("block {} failed: {e}", i + 1));
        assert_eq!(block.header.index, (i + 1) as u64);
        assert_eq!(
            block.header.validator,
            chain.addr_of(name),
            "height {} must be produced by {name}",
            i + 1
        );
    }
}

// ================= enforcement =================

#[test]
fn wrong_producer_rejected_in_wrong_slot() {
    let mut chain = Chain::fresh(&["alpha", "beta"]);
    // height 1 = alpha's slot; beta must fail...
    assert!(chain.produce_by_name("beta", vec![]).is_err());
    // ...and alpha succeeds at the same (still-unadvanced) height.
    assert!(chain.produce_by_name("alpha", vec![]).is_ok());
}

#[test]
fn non_validator_rejected_entirely() {
    // "beta" is a real keypair but NOT in the validator set.
    let mut chain = Chain::fresh(&["alpha"]);
    let beta_kp = keypair("beta");
    let beta_addr = addr(&beta_kp);
    chain.fund(&beta_addr, 5000);

    let err = chain.produce_by_name("beta", vec![]).unwrap_err();
    assert!(
        err.contains("unauthorized") || err.contains("wrong producer") || err.contains("expected"),
        "{err}"
    );
}

#[test]
fn rogue_pubkey_binding_rejected() {
    // Alpha's ADDRESS claims the slot, but the header carries OUTSIDER's
    // pubkey+signature. Pubkey->address binding must reject even though
    // membership passes.
    let mut chain = Chain::fresh(&["alpha", "beta"]);

    // Advance so we're past genesis.
    chain.tick().unwrap();

    let outsider_kp = keypair("outsider");
    let alpha_addr = chain.addr_of("alpha");

    let rogue_producer = ProducerKey::from_parts(alpha_addr.clone(), outsider_kp.clone());
    let block = produce_block(&chain.store, &[], &rogue_producer, chain.now_ms)
        .expect("produce mechanically works with any signing key");

    let prev = current_head(&chain.store).unwrap();
    let err = validate_block(
        &block,
        &prev.header,
        &chain.store.validator_addresses(),
        chain.now_ms,
    )
    .unwrap_err();

    // Either the pubkey-binding fires or the signature check does — both are
    // correct rejections for a forged producer identity.
    let msg = err.to_string();
    assert!(
        msg.contains("does not derive")
            || msg.contains("invalid validator signature")
            || msg.contains("wrong producer"),
        "unexpected rejection reason: {msg}"
    );
}

// ================= staking lifecycle =================

#[test]
fn staked_validator_registration_and_exit() {
    let mut chain = Chain::fresh(&["genesis-val"]);

    // New participant.
    let kp = keypair("joiner");
    let joiner_addr = addr(&kp);
    chain.fund(&joiner_addr, 10_000);

    // Under-minimum stake -> execution reverted -> no entry, no bond debit.
    let under_min = make_tx(
        &kp,
        &joiner_addr,
        "",
        0,
        TxKind::RegisterValidator,
        serde_json::json!({ "stakedAmount": 999 }),
        1,
    );
    let before = chain.store.get_account(&joiner_addr).unwrap().balance;
    chain.produce_scheduled(vec![under_min]).unwrap();
    assert!(
        chain.store.get_validator(&joiner_addr).is_none(),
        "under-min rejected"
    );
    // Gas consumed on revert (blockchain semantics): 10000 - 100(gas)
    let after_revert = chain.store.get_account(&joiner_addr).unwrap().balance;
    assert_eq!(after_revert, before - 100, "gas charged on reverted tx");

    // Proper stake >= 1000.
    let reg = make_tx(
        &kp,
        &joiner_addr,
        "",
        0,
        TxKind::RegisterValidator,
        serde_json::json!({ "stakedAmount": 1000 }),
        2,
    );
    let block = chain.produce_scheduled(vec![reg]).unwrap();
    eprintln!(
        "[debug] register block #{} executed={} skipped={:?}",
        block.header.index,
        block.transactions.len(),
        {
            // can't access skipped from here; just check state
        }
    );
    let entry = chain.store.get_validator(&joiner_addr).expect("registered");
    assert_eq!(entry.staked_amount, 1000);
    assert!(!entry.genesis);
    assert!(
        chain.store.validator_addresses().contains(&joiner_addr),
        "joiner in election order"
    );

    // Balance: 10000 - 100(under-min gas) - 100(proper gas) - 1000(bond) = 8800
    // Nonce: 2 (under-min consumed 1, proper used 2)
    let acc = chain.store.get_account(&joiner_addr).unwrap();
    assert_eq!(acc.balance, 10_000 - 100 - 100 - 1_000);
    assert_eq!(acc.nonce, 2);

    // Unregister refunds bond (minus unregister fee 50).
    let unreg = make_tx(
        &kp,
        &joiner_addr,
        "",
        0,
        TxKind::UnregisterValidator,
        serde_json::json!({}),
        3,
    );
    chain.produce_scheduled(vec![unreg]).unwrap();

    assert!(chain.store.get_validator(&joiner_addr).is_none(), "removed");
    // refund: prior 8800 + 1000 refund - 50 fee = 9750
    let acc = chain.store.get_account(&joiner_addr).unwrap();
    assert_eq!(acc.balance, 8_800 + 1_000 - 50, "bond refunded");
}

#[test]
fn last_validator_cannot_leave() {
    let mut chain = Chain::fresh(&["solo"]);
    let solo_addr = chain.addr_of("solo");
    chain.fund(&solo_addr, 20_000);

    let unreg = make_tx(
        &keypair("solo"),
        &solo_addr,
        "",
        0,
        TxKind::UnregisterValidator,
        serde_json::json!({}),
        1,
    );
    chain.produce_scheduled(vec![unreg]).unwrap();

    assert!(
        chain.store.get_validator(&solo_addr).is_some(),
        "last validator cannot leave"
    );
}
