//! Phase R6 — dispute games, stake-weighted rotation, slashing enforcement.
use ed25519_dalek::Signer as _;
use serde_json::json;
use sha2::Digest as _;
use verinoo_exec::engine::{
    current_head, produce_block, set_head, validate_block, Block, BlockHeader, ProducerKey,
};
use verinoo_exec::leader::stake_weighted_leader;
use verinoo_exec::{LeaderPolicy, RoundRobin};
use verinoo_state::StateStore;
use verinoo_types::tx::{TxData, TxEnvelope, TxKind};

fn keypair(name: &str) -> ed25519_dalek::SigningKey {
    let mut h = sha2::Sha256::new();
    h.update(format!("key:{name}").as_bytes());
    ed25519_dalek::SigningKey::from_bytes(&h.finalize().into())
}
fn addr(kp: &ed25519_dalek::SigningKey) -> String {
    verinoo_exec::engine::public_key_to_address(&hex::encode(kp.verifying_key().to_bytes()))
        .unwrap()
}

struct Chain {
    store: StateStore,
    keys: std::collections::HashMap<String, ed25519_dalek::SigningKey>,
    addresses: std::collections::HashMap<String, String>,
    nonces: std::collections::HashMap<String, u64>,
    now_ms: u64,
}

impl Chain {
    fn fresh(validators: &[&str]) -> Self {
        let store = StateStore::open_temp().unwrap();
        let mut keys = std::collections::HashMap::new();
        let mut addresses = std::collections::HashMap::new();
        for name in validators {
            let kp = keypair(name);
            addresses.insert(name.to_string(), addr(&kp));
            keys.insert(name.to_string(), kp);
        }
        let mut h = BlockHeader {
            hash: format!("genesis-0"),
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
        h.hash = format!("genesis-hash-{}", 0);
        set_head(
            &store,
            &Block {
                header: h.clone(),
                transactions: vec![],
            },
        );
        let vals: Vec<String> = validators.iter().map(|n| addresses[*n].clone()).collect();
        store.seed_genesis_validators(&vals, 0);
        Self {
            store,
            keys,
            addresses,
            nonces: Default::default(),
            now_ms: 1_000_000,
        }
    }

    fn addr_of(&self, name: &str) -> String {
        self.addresses[name].clone()
    }
    fn fund(&self, addr: &str, amount: i64) {
        let mut acc = self.store.account_or_default(addr);
        acc.balance = amount;
        self.store.set_account(&acc);
    }
    fn next_nonce(&self, name: &str) -> u64 {
        let addr = self.addr_of(name);
        self.store.account_or_default(&addr).nonce + 1
    }
    fn make_tx(
        &mut self,
        signer_name: &str,
        kind: TxKind,
        data: serde_json::Value,
        to: &str,
        value: u64,
    ) -> TxEnvelope {
        let kp = self.keys.get(signer_name).unwrap().clone();
        let from_addr = addr(&kp);
        let nonce = self.next_nonce(signer_name);
        let data = TxData { r#type: kind, data };
        let payload = verinoo_types::canonical::UnsignedPayload {
            data: serde_json::to_value(&data).unwrap(),
            from: from_addr.clone(),
            nonce,
            to: to.into(),
            value,
            gas_limit: None,
            gas_price: None,
        };
        let tx_id = payload.tx_id().unwrap();
        let sig = kp.sign(tx_id.as_bytes());
        TxEnvelope {
            tx_id,
            from: from_addr,
            to: to.into(),
            value,
            nonce,
            data,
            gas_limit: None,
            gas_price: None,
            public_key: hex::encode(kp.verifying_key().to_bytes()),
            signature: hex::encode(sig.to_bytes()),
        }
    }
    fn produce_validated(&mut self, txs: Vec<TxEnvelope>) -> Result<Block, String> {
        let prev = current_head(&self.store).unwrap();
        let next_height = prev.header.index + 1;
        let vals = self.store.validator_addresses();
        let leader = RoundRobin
            .leader_at(next_height, &vals)
            .ok_or("empty valset")?;
        let (lname, lkp) = self
            .keys
            .iter()
            .find(|(_, kp)| addr(kp) == leader)
            .ok_or_else(|| format!("leader {leader} no key"))?;
        let lname = lname.clone();
        let producer = ProducerKey::from_parts(leader, lkp.clone());
        let block =
            produce_block(&self.store, &txs, &producer, self.now_ms).map_err(|e| e.to_string())?;
        validate_block(&block, &prev.header, &vals, self.now_ms).map_err(|e| e.to_string())?;
        self.now_ms += 10_000;
        set_head(&self.store, &block);
        for t in &block.transactions {
            for (name, kp) in &self.keys {
                if addr(kp) == t.from {
                    self.nonces.insert(name.clone(), t.nonce);
                }
            }
        }
        void(lname);
        Ok(block)
    }
    fn produce_by(&mut self, name: &str, txs: Vec<TxEnvelope>) -> Result<Block, String> {
        let kp = self
            .keys
            .get(name)
            .cloned()
            .ok_or(format!("no key for {name}"))?;
        let a = addr(&kp);
        let producer = ProducerKey::from_parts(a, kp);
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

fn void<T>(_: T) {}

// ================= stake-weighted rotation =================

#[test]
fn stake_weighted_favors_larger_stakes() {
    let vals = vec![("ai_small".into(), 1000u64), ("ai_whale".into(), 9000u64)];
    let mut small_count = 0;
    let mut whale_count = 0;
    for height in 1..=100u64 {
        match stake_weighted_leader(height, &vals).unwrap().as_str() {
            "ai_small" => small_count += 1,
            "ai_whale" => whale_count += 1,
            _ => {}
        }
    }
    assert!(
        whale_count > small_count * 3,
        "whale ({whale_count}) vs small ({small_count})"
    );
}

#[test]
fn stake_weighted_deterministic() {
    let vals = vec![("ai_a".into(), 500u64), ("ai_b".into(), 3000u64)];
    assert_eq!(
        stake_weighted_leader(42, &vals),
        stake_weighted_leader(42, &vals)
    );
}

#[test]
fn round_robin_still_works() {
    assert_eq!(
        RoundRobin
            .leader_at(1, &["x".into(), "y".into()])
            .as_deref(),
        Some("x")
    );
}

// ================= dispute game full lifecycle =================

#[test]
fn dispute_defender_wins_with_honest_output() {
    let mut c = Chain::fresh(&["val", "compute", "user", "challenger"]);
    let compute_addr = c.addr_of("compute");
    let user_addr = c.addr_of("user");
    let challenger_addr = c.addr_of("challenger");

    c.fund(&c.addr_of("val"), 20_000);
    c.fund(&user_addr, 10_000);
    c.fund(&challenger_addr, 5_000);
    c.fund(&compute_addr, 5_000);

    let tx1 = c.make_tx(
        "val",
        TxKind::RegisterModel,
        json!({
        "architecture": "Relu-Test-Net", "parameterCount": 64,
        "weightsHash": "w", "runtimeHash": "r", "stakingRequirement": 1 }),
        "",
        0,
    );
    c.produce_validated(vec![tx1]).unwrap();
    let models = c
        .store
        .iter_prefix::<verinoo_state::models::ModelRegistryEntry>("model:");
    assert!(!models.is_empty(), "model must be registered");
    let model_id = models[0].model_id.clone();

    let tx2 = c.make_tx(
        "compute",
        TxKind::RegisterNode,
        json!({
        "stakedAmount": 800, "availableCapacity": 2, "maxCapacity": 2,
        "reputation": 50, "supportedModels": [model_id] }),
        "",
        0,
    );
    c.produce_validated(vec![tx2]).unwrap();
    let models = c
        .store
        .iter_prefix::<verinoo_state::models::ModelRegistryEntry>("model:");
    eprintln!("[dbg] models in store: {}", models.len());
    assert!(!models.is_empty(), "model must exist");
    let model_id = models[0].model_id.clone();

    let tx3 = c.make_tx(
        "user",
        TxKind::SubmitInference,
        json!({
        "requester": user_addr, "targetModel": model_id, "inputCommitment": "c",
        "maxFee": 100, "deadline": 9999, "verificationType": "optimistic" }),
        "",
        0,
    );
    c.produce_validated(vec![tx3]).unwrap();

    let tasks = c.store.iter_tasks_by_status("assigned");
    assert_eq!(tasks.len(), 1, "task auto-matched");
    let task_id = tasks[0].task_id.clone();

    let tx4 = c.make_tx(
        "compute",
        TxKind::SubmitResult,
        json!({
        "taskId": task_id, "resultHash": "rh", "resultOutput": "ro" }),
        "",
        0,
    );
    c.produce_validated(vec![tx4]).unwrap();
    assert_eq!(c.store.get_task(&task_id).unwrap().status, "completed");

    let tx5 = c.make_tx(
        "challenger",
        TxKind::ChallengeResult,
        json!({
        "taskId": task_id, "reason": "dispute" }),
        "",
        0,
    );
    c.produce_validated(vec![tx5]).unwrap();
    let game_id = c.store.get_task(&task_id).unwrap().game_id.clone().unwrap();

    // proveStep: HONEST relu output
    let input_f: Vec<i32> = [0.5f64, -0.25, 1.0, -1.5]
        .iter()
        .map(|&v| verinoo_zk::f2i(v))
        .collect();
    let honest_output = verinoo_zk::relu(&input_f);

    // Bisect: both agree at layer 0 -> narrows single-layer spec to proving
    let bx1 = c.make_tx(
        "challenger",
        TxKind::Bisect,
        json!({ "gameId": game_id, "layerIndex": 0, "traceRoot": "t0" }),
        "",
        0,
    );
    c.produce_validated(vec![bx1]).unwrap();
    let bx2 = c.make_tx(
        "compute",
        TxKind::Bisect,
        json!({ "gameId": game_id, "layerIndex": 0, "traceRoot": "t0" }),
        "",
        0,
    );
    c.produce_validated(vec![bx2]).unwrap();
    eprintln!(
        "[dbg] after bisect: {}",
        serde_json::to_string(&c.store.get_game(&game_id).unwrap()).unwrap_or_default()
    );

    let n = c.next_nonce("compute");
    let tx6 = c.make_tx(
        "compute",
        TxKind::ProveStep,
        json!({
        "gameId": game_id, "layerWeights": [], "layerInput": input_f,
        "layerBias": [], "layerOutput": honest_output,
        "actualTraceRoot": "t0" }),
        "",
        0,
    );
    void(n);
    c.produce_validated(vec![tx6])
        .unwrap_or_else(|e| panic!("prove failed: {e}"));

    let game = c.store.get_game(&game_id).unwrap();
    assert_eq!(game["status"], "resolved_valid");
    assert_eq!(game["winner"], json!(compute_addr));

    let df_bal = c.store.get_account(&compute_addr).unwrap().balance;
    assert!(df_bal > 4_800, "defender should have bonds: {df_bal}");
}

#[test]
fn dishonest_output_slashes_defender() {
    let mut c = Chain::fresh(&["val", "compute", "user", "challenger"]);
    let compute_addr = c.addr_of("compute");
    let challenger_addr = c.addr_of("challenger");
    let user_addr = c.addr_of("user");

    c.fund(&c.addr_of("val"), 20_000);
    c.fund(&user_addr, 10_000);
    c.fund(&challenger_addr, 5_000);
    c.fund(&compute_addr, 5_000);

    let tx1 = c.make_tx(
        "val",
        TxKind::RegisterModel,
        json!({
        "architecture": "Relu-Test-Net", "parameterCount": 64,
        "weightsHash": "w", "runtimeHash": "r", "stakingRequirement": 1 }),
        "",
        0,
    );
    c.produce_validated(vec![tx1]).unwrap();
    let models = c
        .store
        .iter_prefix::<verinoo_state::models::ModelRegistryEntry>("model:");
    let model_id = models[0].model_id.clone();

    let tx2 = c.make_tx(
        "compute",
        TxKind::RegisterNode,
        json!({
        "stakedAmount": 800, "availableCapacity": 2, "maxCapacity": 2,
        "reputation": 50, "supportedModels": [model_id] }),
        "",
        0,
    );
    c.produce_validated(vec![tx2]).unwrap();
    let models = c
        .store
        .iter_prefix::<verinoo_state::models::ModelRegistryEntry>("model:");
    eprintln!("[dbg] models in store: {}", models.len());
    assert!(!models.is_empty(), "model must exist");
    let model_id = models[0].model_id.clone();

    let tx3 = c.make_tx(
        "user",
        TxKind::SubmitInference,
        json!({
        "requester": user_addr, "targetModel": model_id, "inputCommitment": "c",
        "maxFee": 100, "deadline": 9999 }),
        "",
        0,
    );
    c.produce_validated(vec![tx3]).unwrap();

    let task = c
        .store
        .iter_tasks_by_status("assigned")
        .first()
        .cloned()
        .unwrap();
    let task_id = task.task_id.clone();

    let tx4 = c.make_tx(
        "compute",
        TxKind::SubmitResult,
        json!({
        "taskId": task_id, "resultHash": "h", "resultOutput": "o" }),
        "",
        0,
    );
    c.produce_validated(vec![tx4]).unwrap();

    let tx5 = c.make_tx(
        "challenger",
        TxKind::ChallengeResult,
        json!({
        "taskId": task_id, "reason": "fraud check" }),
        "",
        0,
    );
    c.produce_validated(vec![tx5]).unwrap();
    let game_id = c.store.get_task(&task_id).unwrap().game_id.clone().unwrap();

    for name in ["challenger", "compute"] {
        let n = c.next_nonce(name);
        let tx = c.make_tx(
            name,
            TxKind::Bisect,
            json!({ "gameId": game_id, "layerIndex": 0, "traceRoot": "t" }),
            "",
            0,
        );
        void(n);
        c.produce_validated(vec![tx]).unwrap();
    }

    // Dishonest output: all zeros instead of actual relu values.
    // Bisect: both agree at layer 0 -> narrows single-layer spec to proving
    let bx1 = c.make_tx(
        "challenger",
        TxKind::Bisect,
        json!({ "gameId": game_id, "layerIndex": 0, "traceRoot": "t0" }),
        "",
        0,
    );
    c.produce_validated(vec![bx1]).unwrap();
    let bx2 = c.make_tx(
        "compute",
        TxKind::Bisect,
        json!({ "gameId": game_id, "layerIndex": 0, "traceRoot": "t0" }),
        "",
        0,
    );
    c.produce_validated(vec![bx2]).unwrap();

    let n = c.next_nonce("compute");
    let tx6 = c.make_tx(
        "compute",
        TxKind::ProveStep,
        json!({
        "gameId": game_id, "layerWeights": [],
        "layerInput": [32768, -16384, 65536, -98304],
        "layerBias": [], "layerOutput": [0, 0, 0, 0], // WRONG
        "actualTraceRoot": "t", "snark": null }),
        "",
        0,
    );
    void(n);
    c.produce_validated(vec![tx6]).unwrap();

    let game = c.store.get_game(&game_id).unwrap();
    assert_eq!(
        game["status"], "resolved_slash",
        "dishonest output must slash defender"
    );
    assert_eq!(game["winner"], json!(challenger_addr));
}

#[test]
fn last_validator_cannot_leave() {
    let mut c = Chain::fresh(&["solo"]);
    c.fund(&c.addr_of("solo"), 20_000);
    let tx = c.make_tx("solo", TxKind::UnregisterValidator, json!({}), "", 0);
    c.produce_validated(vec![tx]).unwrap();
    assert!(
        c.store.get_validator(&c.addr_of("solo")).is_some(),
        "last validator cannot leave"
    );
}
