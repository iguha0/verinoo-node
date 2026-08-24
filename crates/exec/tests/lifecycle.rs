//! Full-lifecycle integration test — port of the reference client's
//! engine.test.ts "full task lifecycle" + gas-market tests, running against
//! the Rust execution core with real Ed25519 identities.

use ed25519_dalek::Signer;
use verinoo_exec::engine::{current_head, produce_block, validate_block, Block, ProducerKey};
use verinoo_state::genesis::CHAIN_ID;
use verinoo_state::StateStore;
use verinoo_types::canonical::UnsignedPayload;
use verinoo_types::tx::{TxData, TxEnvelope, TxKind};

fn keypair(name: &str) -> ed25519_dalek::SigningKey {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(format!("key:{name}").as_bytes());
    let digest: [u8; 32] = h.finalize().into();
    ed25519_dalek::SigningKey::from_bytes(&digest)
}

/// Build a canonical, correctly-signed transaction (Rust-native path).
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
    let payload = UnsignedPayload {
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

struct Chain {
    store: StateStore,
    producer: ProducerKey,
    now_ms: u64,
    height: u64,
}

impl Chain {
    fn fresh() -> Self {
        let store = StateStore::open_temp().unwrap();
        let producer = ProducerKey::deterministic("validator-a", 1);
        // genesis
        let genesis = Block {
            header: verinoo_exec::engine::BlockHeader {
                hash: "0".repeat(64),
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
            },
            transactions: vec![],
        };
        verinoo_exec::engine::set_head(&store, &genesis);
        assert_eq!(CHAIN_ID, "verinoo-test-1");
        Self {
            store,
            producer,
            now_ms: 1_000_000,
            height: 0,
        }
    }

    fn fund(&self, address: &str, amount: i64) {
        let mut acc = self.store.account_or_default(address);
        acc.balance = amount;
        self.store.set_account(&acc);
    }

    /// Produce with automatic validation of the produced block.
    fn produce(&mut self, txs: &[TxEnvelope]) -> Block {
        let block = produce_block(&self.store, txs, &self.producer, self.now_ms).unwrap();
        let head = current_head(&self.store).unwrap();
        validate_block(
            &block,
            &head.header,
            std::slice::from_ref(&self.producer.address),
            self.now_ms,
        )
        .expect("produced block must pass its own gate");
        self.height = block.header.index;
        self.now_ms += 10_000;
        verinoo_exec::engine::set_head(&self.store, &block);
        block
    }
}

#[test]
fn mandatory_gas_fee_split_and_transfer() {
    let mut chain = Chain::fresh();
    let kp = keypair("sender");
    chain.fund(&hex_key_address(&kp), 1000);

    let val_addr = chain.producer.address.clone();
    let tx = make_tx(
        &kp,
        &hex_key_address(&kp),
        "recipient",
        10,
        TxKind::Transfer,
        serde_json::json!({}),
        1,
    );
    chain.produce(&[tx]);

    // transfer cost = 50 * baseFee(1); split 75/37.5->37 | 25%->12.5->13? ints:
    // fee=50, validator=50*3/4=37, treasury=13
    let sender = chain.store.account_or_default(&hex_key_address(&kp));
    assert_eq!(sender.balance, 1000 - 10 - 50);
    assert_eq!(chain.store.account_or_default("recipient").balance, 10);
    assert_eq!(
        chain.store.account_or_default(&val_addr).balance as u64,
        50 * 3 / 4
    );
    assert_eq!(
        chain.store.account_or_default("treasury").balance as u64,
        50 - 50 * 3 / 4
    );
}

fn hex_key_address(kp: &ed25519_dalek::SigningKey) -> String {
    verinoo_exec::engine::public_key_to_address(&hex::encode(kp.verifying_key().to_bytes()))
        .unwrap()
}

#[test]
fn full_task_lifecycle_with_dispute_game() {
    let mut chain = Chain::fresh();
    let treasury = keypair("treasury");
    let user = keypair("user");
    let node = keypair("node");
    let challenger = keypair("challenger");

    for (name, bal) in [
        ("treasury", 10_000),
        ("user", 5_000),
        ("node", 5_000),
        ("challenger", 2_000),
    ] {
        let kp = keypair(name);
        chain.fund(&hex_key_address(&kp), bal);
    }

    // registerModel
    chain.produce(&[make_tx(
        &treasury,
        &hex_key_address(&treasury),
        "",
        0,
        TxKind::RegisterModel,
        serde_json::json!({
            "architecture": "Tiny-Test-Net", "parameterCount": 256000,
            "weightsHash": format!("w_{}", "a".repeat(32)),
            "runtimeHash": format!("r_{}", "b".repeat(32)),
            "stakingRequirement": 50, "description": "rust test model",
        }),
        1,
    )]);
    let models = chain
        .store
        .iter_prefix::<verinoo_state::models::ModelRegistryEntry>("model:");
    assert_eq!(models.len(), 1, "model registered");
    let model_id = models[0].model_id.clone();

    // registerNode
    chain.produce(&[make_tx(
        &node,
        &hex_key_address(&node),
        "",
        0,
        TxKind::RegisterNode,
        serde_json::json!({ "stakedAmount": 800, "availableCapacity": 4, "maxCapacity": 4,
            "activeTasks": 0, "reputation": 75, "successfulInferences": 0, "failedInferences": 0,
            "supportedModels": [model_id] }),
        1,
    )]);
    let registered = chain.store.get_node(&hex_key_address(&node)).unwrap();
    assert_eq!(registered.reputation, 75);

    // submitInference -> auto-matched
    chain.produce(&[make_tx(
        &user,
        &hex_key_address(&user),
        "",
        100,
        TxKind::SubmitInference,
        serde_json::json!({ "requester": hex_key_address(&user), "targetModel": model_id,
            "inputCommitment": "inp_commit", "maxFee": 100, "deadline": 1000,
            "verificationType": "optimistic" }),
        1,
    )]);
    let tasks = chain.store.iter_tasks_by_status("assigned");
    assert_eq!(tasks.len(), 1, "task auto-matched");
    assert_eq!(
        tasks[0].assigned_to.as_deref(),
        Some(hex_key_address(&node).as_str())
    );
    let task_id = tasks[0].task_id.clone();

    // submitResult
    chain.produce(&[make_tx(
        &node, &hex_key_address(&node), "", 0, TxKind::SubmitResult,
        serde_json::json!({ "taskId": task_id, "resultHash": "res_h", "resultOutput": "out", "proofData": "" }),
        2,
    )]);
    let done = chain.store.get_task(&task_id).unwrap();
    assert_eq!(done.status, "completed");

    // challengeResult -> game opens
    chain.produce(&[make_tx_challenge(&challenger, &task_id, 1)]);
    let challenged = chain.store.get_task(&task_id).unwrap();
    assert_eq!(challenged.status, "challenged");
    let game_id = challenged.game_id.clone().unwrap();

    // bisection: disagree at L2, then agree at L1 -> narrows [0,3] to [1,2]
    chain.produce(&[make_bisect(&challenger, &game_id, 2, "trace_ch_2", 2)]);
    chain.produce(&[make_bisect(&node, &game_id, 2, "trace_df_2_DIFFERENT", 3)]);
    assert_eq!(
        chain.store.get_game(&game_id).unwrap()["status"],
        "bisecting",
        "mid-game"
    );
    chain.produce(&[make_bisect(&challenger, &game_id, 1, "trace_ch_1", 3)]);
    chain.produce(&[make_bisect(&node, &game_id, 1, "trace_ch_1", 4)]);

    let game = chain.store.get_game(&game_id).unwrap();
    assert_eq!(game["status"], "proving", "bisection must narrow");
    assert_eq!(game["disputedLayer"], 1);
}

fn make_tx_challenge(kp: &ed25519_dalek::SigningKey, task_id: &str, nonce: u64) -> TxEnvelope {
    make_tx(
        kp,
        &hex_key_address(kp),
        "",
        0,
        TxKind::ChallengeResult,
        serde_json::json!({ "taskId": task_id, "reason": "rust dispute test" }),
        nonce,
    )
}
fn make_bisect(
    kp: &ed25519_dalek::SigningKey,
    game_id: &str,
    layer: i64,
    root: &str,
    nonce: u64,
) -> TxEnvelope {
    make_tx(
        kp,
        &hex_key_address(kp),
        "",
        0,
        TxKind::Bisect,
        serde_json::json!({ "gameId": game_id, "layerIndex": layer, "traceRoot": root }),
        nonce,
    )
}

#[test]
fn hostile_block_rejections() {
    let mut chain = Chain::fresh();
    let kp = keypair("honest");
    chain.fund(&hex_key_address(&kp), 5000);

    let good = make_tx(
        &kp,
        &hex_key_address(&kp),
        "r",
        5,
        TxKind::Transfer,
        serde_json::json!({}),
        1,
    );
    let head = current_head(&chain.store).unwrap();
    let block = chain.produce(&[good]);

    // 1. tampered header hash
    let mut evil = block.clone();
    evil.header.hash = "deadbeef".repeat(8);
    assert!(validate_block(
        &evil,
        &head.header,
        &[chain.producer.address.clone()],
        chain.now_ms
    )
    .is_err());

    // 2. wrong producer
    let rogue = ProducerKey::deterministic("rogue", 9);
    let signed_by_rogue = produce_block_for(
        &chain.store,
        &[make_tx(
            &kp,
            &hex_key_address(&kp),
            "r",
            5,
            TxKind::Transfer,
            serde_json::json!({}),
            2,
        )],
        &rogue,
        chain.now_ms + 10_000,
    );
    assert!(validate_block(
        &signed_by_rogue,
        &block.header,
        &[chain.producer.address.clone()],
        chain.now_ms
    )
    .is_err());

    // 3. duplicate transaction inside one block
    let dup_block = Block {
        header: block.header.clone(),
        transactions: vec![block.transactions[0].clone(), block.transactions[0].clone()],
    };
    // re-sign not needed: txRoot check catches duplicates before signature check ordering matters
    assert!(validate_block(
        &dup_block,
        &head.header,
        &[chain.producer.address.clone()],
        chain.now_ms
    )
    .is_err());
}

fn produce_block_for(
    store: &StateStore,
    txs: &[TxEnvelope],
    producer: &ProducerKey,
    now: u64,
) -> Block {
    produce_block(store, txs, producer, now).unwrap()
}

#[test]
fn malleated_transaction_is_rejected() {
    let kp = keypair("victim");
    let tx = make_tx(
        &kp,
        &hex_key_address(&kp),
        "bob",
        10,
        TxKind::Transfer,
        serde_json::json!({}),
        1,
    );
    let mut redirected = tx.clone();
    redirected.to = "attacker".into();
    let store = StateStore::open_temp().unwrap();
    let v = verinoo_exec::Validator {
        store: &store,
        base_fee: 1,
    };
    // signature itself is over the ORIGINAL txId — but canonical binding fails first order-wise?
    // Reference order: sig check passes (sig covers original txId), then canonical mismatch.
    assert!(v.validate_transaction(&redirected).is_err());
    let err = v.validate_transaction(&redirected).unwrap_err();
    assert!(
        err.contains("does not match") || err.contains("invalid sig"),
        "{err}"
    );
}
