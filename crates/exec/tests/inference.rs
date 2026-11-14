//! The built-in model must be executed by consensus, not by the caller:
//! two independent nodes applying the same block have to derive the same
//! output, and the receipt must be anchored to the block that carries it.

use ed25519_dalek::Signer;
use verinoo_exec::engine::{
    current_head, produce_block, set_head, Block, BlockHeader, ProducerKey,
};
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

fn addr(sk: &ed25519_dalek::SigningKey) -> String {
    verinoo_exec::engine::public_key_to_address(&hex::encode(sk.verifying_key().to_bytes()))
        .unwrap()
}

fn make_tx(
    signer: &ed25519_dalek::SigningKey,
    kind: TxKind,
    data: serde_json::Value,
    nonce: u64,
) -> TxEnvelope {
    let from = addr(signer);
    let data = TxData { r#type: kind, data };
    let payload = UnsignedPayload {
        data: serde_json::to_value(&data).unwrap(),
        from: from.clone(),
        nonce,
        to: String::new(),
        value: 0,
        gas_limit: None,
        gas_price: None,
    };
    let tx_id = payload.tx_id().unwrap();
    let sig = signer.sign(tx_id.as_bytes());
    TxEnvelope {
        tx_id,
        from,
        to: String::new(),
        value: 0,
        nonce,
        data,
        gas_limit: None,
        gas_price: None,
        public_key: hex::encode(signer.verifying_key().to_bytes()),
        signature: hex::encode(sig.to_bytes()),
    }
}

fn fresh_chain() -> StateStore {
    let store = StateStore::open_temp().unwrap();
    let genesis = Block {
        header: BlockHeader {
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
    set_head(&store, &genesis);
    store
}

fn run_inference_example(store: &StateStore, producer: &ProducerKey, input: &str) -> Block {
    let user = keypair("requester");
    let mut acc = store.account_or_default(&addr(&user));
    acc.balance = 10_000;
    store.set_account(&acc);

    let tx = make_tx(
        &user,
        TxKind::SubmitInference,
        serde_json::json!({
            "requester": addr(&user),
            "targetModel": verinoo_model::MODEL_ID,
            "input": input,
            "inputCommitment": verinoo_model::sha256_hex(input),
            "maxFee": 100,
            "deadline": 1_000,
            "verificationType": "optimistic",
        }),
        1,
    );
    let block = produce_block(store, &[tx], producer, 1_000_000).unwrap();
    let head = current_head(store).unwrap();
    verinoo_exec::engine::validate_block(
        &block,
        &head.header,
        std::slice::from_ref(&producer.address),
        1_000_000,
    )
    .expect("produced block must pass its own gate");
    set_head(store, &block);
    block
}

#[test]
fn task_is_executed_by_consensus_and_receipt_is_anchored() {
    let store = fresh_chain();
    let producer = ProducerKey::deterministic("validator-a", 1);
    let block = run_inference_example(&store, &producer, "where is my package");

    let tasks: Vec<verinoo_state::models::InferenceTask> = store.iter_prefix("task:");
    assert_eq!(tasks.len(), 1);
    let task = &tasks[0];

    assert_eq!(task.status, "completed", "task must complete in the block");
    assert_eq!(
        task.executed_at, block.header.index,
        "result must be produced while applying the block"
    );
    let output = task.result_output.clone().expect("output stored");
    assert!(
        output.contains("\"shipping\""),
        "model classified incorrectly: {output}"
    );
    assert_eq!(
        Some(verinoo_model::sha256_hex(&output)),
        task.result_hash,
        "resultHash must commit to the exact output bytes"
    );

    let receipt =
        verinoo_exec::engine::receipt_for(&store, &task.task_id).expect("receipt stamped");
    assert_eq!(receipt["blockHash"], block.header.hash);
    assert_eq!(receipt["height"], block.header.index);
    assert_eq!(receipt["validator"], producer.address);
    assert_eq!(receipt["weightsHash"], verinoo_model::weights_hash());
    assert_eq!(receipt["outputHash"], task.result_hash.clone().unwrap());
}

#[test]
fn two_nodes_derive_identical_results() {
    let input = "there is an unknown charge on my invoice";
    let a = fresh_chain();
    let b = fresh_chain();
    let block_a = run_inference_example(&a, &ProducerKey::deterministic("validator-a", 1), input);
    let block_b = run_inference_example(&b, &ProducerKey::deterministic("validator-a", 1), input);

    let ta: Vec<verinoo_state::models::InferenceTask> = a.iter_prefix("task:");
    let tb: Vec<verinoo_state::models::InferenceTask> = b.iter_prefix("task:");
    assert_eq!(ta[0].result_output, tb[0].result_output);
    assert_eq!(ta[0].result_hash, tb[0].result_hash);
    assert_eq!(block_a.header.state_root, block_b.header.state_root);
}

#[test]
fn fee_is_settled_to_the_executor() {
    let store = fresh_chain();
    let producer = ProducerKey::deterministic("validator-a", 1);
    let user = keypair("requester");
    let mut acc = store.account_or_default(&addr(&user));
    acc.balance = 1_000;
    store.set_account(&acc);

    let tx = make_tx(
        &user,
        TxKind::SubmitInference,
        serde_json::json!({
            "requester": addr(&user),
            "targetModel": verinoo_model::MODEL_ID,
            "input": "please cancel my subscription",
            "maxFee": 100,
            "deadline": 1_000,
        }),
        1,
    );
    produce_block(&store, &[tx], &producer, 1_000_000).unwrap();

    let requester = store.account_or_default(&addr(&user));
    let executor = store.account_or_default(&producer.address);
    // gas fee 60 on submitInference, plus the 100 withheld for the task.
    // Split: 75% of gas (45) + 80% of the task fee (80) to the executor,
    // 25% of gas (15) burned to treasury.
    assert_eq!(requester.balance, 1_000 - 60 - 100);
    assert_eq!(store.account_or_default("treasury").balance, 15);
    assert_eq!(executor.balance, 45 + 80);
}

/// A model the chain was not compiled with — loaded from an operator's model
/// set — must execute in consensus and derive the same scores as the built-in
/// classifier it was exported from.
#[test]
fn operator_model_runs_through_the_generic_runtime() {
    let input = "my parcel was never delivered";
    let spec = verinoo_model::export_graph("acme-support-v1");
    let model = verinoo_runtime::Model::from_value(&spec).expect("exported graph is valid");
    let mut registry = verinoo_runtime::Registry::empty();
    registry.insert(model.clone()).expect("unique model id");
    verinoo_runtime::install_once(registry);

    let store = fresh_chain();
    let producer = ProducerKey::deterministic("validator-a", 1);
    let user = keypair("graph-requester");
    let mut acc = store.account_or_default(&addr(&user));
    acc.balance = 10_000;
    store.set_account(&acc);

    let tx = make_tx(
        &user,
        TxKind::SubmitInference,
        serde_json::json!({
            "requester": addr(&user),
            "targetModel": "acme-support-v1",
            "input": input,
            "inputCommitment": verinoo_model::sha256_hex(input),
            "maxFee": 100,
            "deadline": 1_000,
        }),
        1,
    );
    produce_block(&store, &[tx], &producer, 1_000_000).unwrap();

    let tasks: Vec<verinoo_state::models::InferenceTask> = store.iter_prefix("task:");
    assert_eq!(tasks.len(), 1);
    let task = &tasks[0];
    assert_eq!(
        task.status, "completed",
        "runtime model must execute on-chain"
    );
    let output = task.result_output.clone().expect("output stored");
    assert!(output.contains("\"shipping\""), "wrong label: {output}");
    assert!(
        output.contains("acme-support-v1"),
        "wrong model id: {output}"
    );

    let receipt =
        verinoo_exec::engine::receipt_for(&store, &task.task_id).expect("receipt stamped");
    assert_eq!(receipt["weightsHash"], model.weights_hash());
    assert_eq!(receipt["modelId"], "acme-support-v1");

    // Parity: same parameters, same integer path, same scores.
    let builtin = verinoo_model::infer(input);
    let graph = model.infer(input);
    assert_eq!(graph.label, builtin.label);
    assert_eq!(graph.scores, builtin.scores);
}

/// Two nodes running the same model set must agree on the state root, which
/// now commits to every executed output.
#[test]
fn runtime_results_are_part_of_the_state_root() {
    let input = "please cancel my subscription";
    let spec = verinoo_model::export_graph("acme-support-v1");
    let model = verinoo_runtime::Model::from_value(&spec).unwrap();
    let mut registry = verinoo_runtime::Registry::empty();
    registry.insert(model).unwrap();
    verinoo_runtime::install_once(registry);

    let mut roots = Vec::new();
    for _ in 0..2 {
        let store = fresh_chain();
        let producer = ProducerKey::deterministic("validator-a", 1);
        let user = keypair("root-requester");
        let mut acc = store.account_or_default(&addr(&user));
        acc.balance = 10_000;
        store.set_account(&acc);
        let tx = make_tx(
            &user,
            TxKind::SubmitInference,
            serde_json::json!({
                "requester": addr(&user),
                "targetModel": "acme-support-v1",
                "input": input,
                "inputCommitment": verinoo_model::sha256_hex(input),
                "maxFee": 100,
                "deadline": 1_000,
            }),
            1,
        );
        let block = produce_block(&store, &[tx], &producer, 1_000_000).unwrap();
        roots.push(block.header.state_root);
    }
    assert_eq!(
        roots[0], roots[1],
        "same inputs and models must produce one root"
    );
}

fn install_test_input_key() -> String {
    let key = verinoo_crypto::NetworkKey::from_secret([42u8; 32]);
    let _ = verinoo_crypto::install_input_key(key);
    verinoo_crypto::input_public_key_hex().expect("input key installed")
}

#[test]
fn encrypted_input_is_executed_and_never_appears_in_the_block() {
    let public_key = install_test_input_key();
    let store = fresh_chain();
    let producer = ProducerKey::deterministic("validator-enc", 3);
    let user = keypair("encrypted-requester");
    let mut acc = store.account_or_default(&addr(&user));
    acc.balance = 10_000;
    store.set_account(&acc);

    let plaintext = "my parcel was never delivered";
    let commitment = verinoo_crypto::sha256_hex(plaintext);
    let encrypted = verinoo_crypto::encrypt(&public_key, plaintext, &commitment).unwrap();
    let tx = make_tx(
        &user,
        TxKind::SubmitInference,
        serde_json::json!({
            "requester": addr(&user),
            "targetModel": verinoo_model::MODEL_ID,
            "encryptedInput": encrypted,
            "inputCommitment": commitment,
            "maxFee": 100,
            "deadline": 1_000,
            "verificationType": "optimistic",
        }),
        1,
    );
    let block = produce_block(&store, &[tx], &producer, 1_000_000).unwrap();
    let block_json = serde_json::to_string(&block).unwrap();
    assert!(
        !block_json.contains(plaintext),
        "plaintext leaked into the block payload"
    );

    let head = current_head(&store).unwrap();
    verinoo_exec::engine::validate_block(
        &block,
        &head.header,
        std::slice::from_ref(&producer.address),
        1_000_000,
    )
    .unwrap();
    set_head(&store, &block);

    let tasks: Vec<verinoo_state::models::InferenceTask> = store.iter_prefix("task:");
    assert_eq!(tasks.len(), 1);
    let task = &tasks[0];
    assert_eq!(task.status, "completed");
    assert!(task.input.is_empty(), "plaintext must not be stored");
    assert!(
        task.encrypted_input.is_some(),
        "ciphertext must be retained"
    );
    assert_eq!(
        verinoo_exec::engine::plaintext_input(task).unwrap(),
        plaintext
    );
    let output = task.result_output.clone().unwrap();
    assert!(output.contains("\"shipping\""), "{output}");
    assert_eq!(Some(verinoo_model::sha256_hex(&output)), task.result_hash);
}

#[test]
fn an_encrypted_task_with_a_wrong_commitment_does_not_execute() {
    let public_key = install_test_input_key();
    let store = fresh_chain();
    let producer = ProducerKey::deterministic("validator-bad", 4);
    let user = keypair("bad-commitment");
    let mut acc = store.account_or_default(&addr(&user));
    acc.balance = 10_000;
    store.set_account(&acc);

    let plaintext = "please cancel my subscription";
    let wrong = verinoo_crypto::sha256_hex("different text");
    let encrypted = verinoo_crypto::encrypt(&public_key, plaintext, &wrong).unwrap();
    let tx = make_tx(
        &user,
        TxKind::SubmitInference,
        serde_json::json!({
            "requester": addr(&user),
            "targetModel": verinoo_model::MODEL_ID,
            "encryptedInput": encrypted,
            "inputCommitment": wrong,
            "maxFee": 100,
            "deadline": 1_000,
        }),
        1,
    );
    let block = produce_block(&store, &[tx], &producer, 1_000_000).unwrap();
    set_head(&store, &block);
    let tasks: Vec<verinoo_state::models::InferenceTask> = store.iter_prefix("task:");
    if let Some(task) = tasks.first() {
        assert_eq!(task.status, "pending", "wrong commitment must not complete");
    }
}
