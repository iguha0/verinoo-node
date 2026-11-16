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

static INSTALL_MODELS: std::sync::Once = std::sync::Once::new();

/// Install one process-wide model set for every test. `Once` serialises the
/// installs so test order cannot change which registry wins.
fn install_test_models() {
    INSTALL_MODELS.call_once(|| {
        let spec = verinoo_model::export_graph("acme-support-v1");
        let model = verinoo_runtime::Model::from_value(&spec).expect("exported graph is valid");
        let mut graph = verinoo_runtime::Registry::empty();
        graph.insert(model).expect("unique graph model");
        let claims_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/claims-triage-v1.graph.json");
        let bytes = std::fs::read(claims_path).expect("committed claims model");
        let model = verinoo_runtime::Model::from_bytes(&bytes).expect("valid claims model");
        graph.insert(model).expect("unique claims model");
        verinoo_runtime::install_once(graph);

        let transformer_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/intent-transformer-v1.json");
        let bytes = std::fs::read(transformer_path).expect("committed transformer model");
        let model = verinoo_runtime::TransformerModel::from_bytes(&bytes).expect("valid model");
        let mut transformers = verinoo_runtime::TransformerRegistry::empty();
        transformers.insert(model).unwrap();
        verinoo_runtime::transformer::install_transformers_once(transformers);

        let decoder_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt-v1.json");
        let bytes = std::fs::read(decoder_path).expect("committed decoder model");
        let model = verinoo_runtime::GptModel::from_bytes(&bytes).expect("valid decoder");
        let mut decoders = verinoo_runtime::GptRegistry::empty();
        decoders.insert(model).unwrap();
        let stories_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/tiny-stories-1m.json");
        if stories_path.exists() {
            let bytes = std::fs::read(stories_path).expect("TinyStories model");
            let model = verinoo_runtime::GptModel::from_bytes(&bytes).expect("valid TinyStories");
            decoders.insert(model).unwrap();
        }
        verinoo_runtime::gpt::install_gpt_once(decoders);
    });
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
    install_test_models();
    let spec = verinoo_model::export_graph("acme-support-v1");
    let model = verinoo_runtime::Model::from_value(&spec).expect("exported graph is valid");
    let input = "my parcel was never delivered";
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
    install_test_models();
    let input = "please cancel my subscription";
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

#[test]
fn transformer_task_runs_by_consensus_and_receipt_binds_weights() {
    install_test_models();
    let store = fresh_chain();
    let producer = ProducerKey::deterministic("validator-transformer", 6);
    let user = keypair("transformer-requester");
    let mut acc = store.account_or_default(&addr(&user));
    acc.balance = 10_000;
    store.set_account(&acc);

    let input = "my parcel is delayed again";
    let tx = make_tx(
        &user,
        TxKind::SubmitInference,
        serde_json::json!({
            "requester": addr(&user),
            "targetModel": "intent-transformer-v1",
            "input": input,
            "inputCommitment": verinoo_model::sha256_hex(input),
            "maxFee": 100,
            "deadline": 1_000,
        }),
        1,
    );
    let block = produce_block(&store, &[tx], &producer, 1_000_000).unwrap();
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
    let output = task.result_output.clone().unwrap();
    assert!(output.contains("\"shipping\""), "{output}");
    let receipt = verinoo_exec::engine::receipt_for(&store, &task.task_id).unwrap();
    assert_eq!(receipt["modelId"], "intent-transformer-v1");
    assert_eq!(
        receipt["weightsHash"].as_str().unwrap().len(),
        64,
        "receipt must bind the transformer weights"
    );
    assert_eq!(
        receipt["outputHash"].as_str().unwrap(),
        verinoo_model::sha256_hex(&output)
    );
}

fn decoder_fixture() -> serde_json::Value {
    let fixture_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt-v1.test.json");
    serde_json::from_slice(&std::fs::read(fixture_path).expect("decoder fixture"))
        .expect("fixture JSON")
}

#[test]
fn decoder_task_runs_by_consensus_and_matches_the_float_generation() {
    install_test_models();
    let fixture = decoder_fixture();
    let case = &fixture["cases"][0];
    let prompt = case["prompt"].as_str().unwrap();
    let expected_text = case["text"].as_str().unwrap();

    let store = fresh_chain();
    let producer = ProducerKey::deterministic("validator-decoder", 7);
    let user = keypair("decoder-requester");
    let mut acc = store.account_or_default(&addr(&user));
    acc.balance = 10_000;
    store.set_account(&acc);

    let tx = make_tx(
        &user,
        TxKind::SubmitInference,
        serde_json::json!({
            "requester": addr(&user),
            "targetModel": "tiny-gpt-v1",
            "input": prompt,
            "inputCommitment": verinoo_model::sha256_hex(prompt),
            "maxFee": 100,
            "deadline": 1_000,
        }),
        1,
    );
    let block = produce_block(&store, &[tx], &producer, 1_000_000).unwrap();
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
    let output: serde_json::Value =
        serde_json::from_str(task.result_output.as_deref().unwrap()).unwrap();
    assert_eq!(output["generatedText"], expected_text);
    assert_eq!(output["modelId"], "tiny-gpt-v1");
    assert!(!output["tokenIds"].as_array().unwrap().is_empty());

    let receipt = verinoo_exec::engine::receipt_for(&store, &task.task_id).unwrap();
    assert_eq!(receipt["modelId"], "tiny-gpt-v1");
    assert_eq!(
        receipt["weightsHash"].as_str().unwrap().len(),
        64,
        "receipt must bind the decoder weights"
    );
}

#[test]
fn real_decoder_task_runs_by_consensus_and_matches_the_float_generation() {
    install_test_models();
    if verinoo_runtime::gpt::gpt_global()
        .get("tiny-stories-1m")
        .is_none()
    {
        eprintln!("skip: run scripts/import_tiny_stories.py to create the TinyStories model");
        return;
    }
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/tiny-stories-1m.test.json");
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_path).expect("TinyStories fixture"))
            .expect("fixture JSON");
    let case = &fixture["cases"][0];
    let prompt = case["prompt"].as_str().unwrap();
    let expected_text = case["text"].as_str().unwrap();
    let expected_tokens: Vec<u64> = case["tokenIds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();

    let store = fresh_chain();
    let producer = ProducerKey::deterministic("validator-stories", 8);
    let user = keypair("stories-requester");
    let mut acc = store.account_or_default(&addr(&user));
    acc.balance = 10_000;
    store.set_account(&acc);

    let tx = make_tx(
        &user,
        TxKind::SubmitInference,
        serde_json::json!({
            "requester": addr(&user),
            "targetModel": "tiny-stories-1m",
            "input": prompt,
            "inputCommitment": verinoo_model::sha256_hex(prompt),
            "maxFee": 100,
            "deadline": 1_000,
        }),
        1,
    );
    let block = produce_block(&store, &[tx], &producer, 1_000_000).unwrap();
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
    let output: serde_json::Value =
        serde_json::from_str(task.result_output.as_deref().unwrap()).unwrap();
    assert_eq!(output["generatedText"], expected_text);
    assert_eq!(output["modelId"], "tiny-stories-1m");
    let tokens: Vec<u64> = output["tokenIds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert_eq!(tokens, expected_tokens);

    let receipt = verinoo_exec::engine::receipt_for(&store, &task.task_id).unwrap();
    assert_eq!(receipt["modelId"], "tiny-stories-1m");
    assert_eq!(receipt["weightsHash"].as_str().unwrap().len(), 64);
}

#[test]
fn claims_triage_task_runs_by_consensus() {
    install_test_models();
    let store = fresh_chain();
    let producer = ProducerKey::deterministic("validator-claims", 9);
    let user = keypair("claims-requester");
    let mut acc = store.account_or_default(&addr(&user));
    acc.balance = 10_000;
    store.set_account(&acc);

    let input = "a tree fell on the roof and broke two windows";
    let tx = make_tx(
        &user,
        TxKind::SubmitInference,
        serde_json::json!({
            "requester": addr(&user),
            "targetModel": "claims-triage-v1",
            "input": input,
            "inputCommitment": verinoo_model::sha256_hex(input),
            "maxFee": 100,
            "deadline": 1_000,
        }),
        1,
    );
    let block = produce_block(&store, &[tx], &producer, 1_000_000).unwrap();
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
    let output: serde_json::Value =
        serde_json::from_str(task.result_output.as_deref().unwrap()).unwrap();
    assert_eq!(output["label"], "property_damage");
    assert_eq!(output["modelId"], "claims-triage-v1");
    let receipt = verinoo_exec::engine::receipt_for(&store, &task.task_id).unwrap();
    assert_eq!(receipt["modelId"], "claims-triage-v1");
    assert_eq!(
        receipt["outputHash"].as_str().unwrap(),
        verinoo_model::sha256_hex(task.result_output.as_deref().unwrap())
    );
}

fn sampling_spec() -> verinoo_types::SamplingSpec {
    verinoo_types::SamplingSpec {
        temperature_milli: 800,
        top_k: 40,
        top_p_milli: 900,
        seed: 77,
    }
}

#[test]
fn decoder_task_uses_per_task_sampling_and_receipt_binds_it() {
    install_test_models();
    let store = fresh_chain();
    let producer = ProducerKey::deterministic("validator-sampling", 10);
    let user = keypair("sampling-requester");
    let mut acc = store.account_or_default(&addr(&user));
    acc.balance = 10_000;
    store.set_account(&acc);

    let prompt = "please cancel";
    let spec = sampling_spec();
    let tx = make_tx(
        &user,
        TxKind::SubmitInference,
        serde_json::json!({
            "requester": addr(&user),
            "targetModel": "tiny-gpt-v1",
            "input": prompt,
            "inputCommitment": verinoo_model::sha256_hex(prompt),
            "maxFee": 100,
            "deadline": 1_000,
            "sampling": {
                "temperatureMilli": spec.temperature_milli,
                "topK": spec.top_k,
                "topPMilli": spec.top_p_milli,
                "seed": spec.seed,
            },
        }),
        1,
    );
    let block = produce_block(&store, &[tx], &producer, 1_000_000).unwrap();
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
    assert_eq!(task.sampling, Some(spec));

    let model = verinoo_exec::engine::ExecModel::resolve("tiny-gpt-v1").unwrap();
    let plaintext = verinoo_exec::engine::plaintext_input(task).unwrap();
    let recomputed = model
        .result_json_with_sampling(&plaintext, task.sampling.as_ref())
        .unwrap();
    assert_eq!(task.result_output.as_deref(), Some(recomputed.as_str()));
    assert_eq!(
        task.result_hash.as_deref(),
        Some(verinoo_model::sha256_hex(&recomputed).as_str())
    );

    let receipt = verinoo_exec::engine::receipt_for(&store, &task.task_id).unwrap();
    let expected = serde_json::to_value(spec).unwrap();
    let canonical = verinoo_types::canonical::canonical_json(&expected).unwrap();
    assert_eq!(receipt["sampling"], expected);
    assert_eq!(
        receipt["samplingHash"],
        verinoo_exec::engine::sha256_hex(&canonical)
    );
    assert_eq!(receipt["samplingSource"], "task");

    let default_task = model.result_json(&plaintext).unwrap();
    assert!(serde_json::from_str::<serde_json::Value>(&default_task).is_ok());
}

#[test]
fn decoder_task_without_sampling_keeps_the_model_default() {
    install_test_models();
    let store = fresh_chain();
    let producer = ProducerKey::deterministic("validator-default-sampling", 11);
    let user = keypair("default-sampling-requester");
    let mut acc = store.account_or_default(&addr(&user));
    acc.balance = 10_000;
    store.set_account(&acc);

    let prompt = "please cancel";
    let tx = make_tx(
        &user,
        TxKind::SubmitInference,
        serde_json::json!({
            "requester": addr(&user),
            "targetModel": "tiny-gpt-v1",
            "input": prompt,
            "inputCommitment": verinoo_model::sha256_hex(prompt),
            "maxFee": 100,
            "deadline": 1_000,
        }),
        1,
    );
    let block = produce_block(&store, &[tx], &producer, 1_000_000).unwrap();

    let tasks: Vec<verinoo_state::models::InferenceTask> = store.iter_prefix("task:");
    assert_eq!(tasks.len(), 1);
    let task = &tasks[0];
    assert_eq!(task.status, "completed");
    assert!(task.sampling.is_none());

    let model = verinoo_exec::engine::ExecModel::resolve("tiny-gpt-v1").unwrap();
    let plaintext = verinoo_exec::engine::plaintext_input(task).unwrap();
    let expected = model.result_json(&plaintext).unwrap();
    assert_eq!(task.result_output.as_deref(), Some(expected.as_str()));

    let receipt = verinoo_exec::engine::receipt_for(&store, &task.task_id).unwrap();
    let decoder = verinoo_runtime::gpt::gpt_global()
        .get("tiny-gpt-v1")
        .expect("decoder installed");
    assert_eq!(
        serde_json::to_value(decoder.default_sampling()).unwrap(),
        serde_json::json!({
            "temperatureMilli": 0,
            "topK": 0,
            "topPMilli": 0,
            "seed": 0,
        })
    );
    assert!(receipt.get("sampling").is_none());
    assert!(receipt.get("samplingHash").is_none());
    assert!(receipt.get("samplingSource").is_none());
    assert_eq!(
        receipt,
        serde_json::json!({
            "taskId": task.task_id,
            "modelId": "tiny-gpt-v1",
            "weightsHash": model.weights_hash(),
            "inputHash": task.input_commitment,
            "outputHash": task.result_hash,
            "height": 1,
            "blockHash": block.header.hash,
            "validator": block.header.validator,
            "validatorSignature": block.header.validator_signature,
        })
    );
}

#[test]
fn sampling_on_a_non_decoder_is_rejected_before_inclusion() {
    let spec = sampling_spec();
    let model = verinoo_exec::engine::ExecModel::resolve(verinoo_model::MODEL_ID).unwrap();
    let err = model
        .result_json_with_sampling("hello", Some(&spec))
        .unwrap_err();
    assert!(err.contains("does not support per-task sampling"), "{err}");

    let store = fresh_chain();
    let producer = ProducerKey::deterministic("validator-unsupported", 12);
    let user = keypair("unsupported-sampling");
    let mut acc = store.account_or_default(&addr(&user));
    acc.balance = 10_000;
    store.set_account(&acc);

    let input = "my parcel was never delivered";
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
            "sampling": {
                "temperatureMilli": spec.temperature_milli,
                "topK": spec.top_k,
                "topPMilli": spec.top_p_milli,
                "seed": spec.seed,
            },
        }),
        1,
    );
    let err = verinoo_exec::engine::Validator {
        store: &store,
        base_fee: 1,
    }
    .validate_transaction(&tx)
    .unwrap_err();
    assert!(err.contains("does not support per-task sampling"), "{err}");

    let block = produce_block(&store, &[tx], &producer, 1_000_000).unwrap();
    assert!(block.transactions.is_empty());
    let tasks: Vec<verinoo_state::models::InferenceTask> = store.iter_prefix("task:");
    assert!(tasks.is_empty());
    assert_eq!(store.account_or_default(&addr(&user)).balance, 10_000);
}

#[test]
fn invalid_sampling_is_rejected_by_consensus_validation_without_state_changes() {
    install_test_models();
    let store = fresh_chain();
    let user = keypair("bad-sampling");
    let mut acc = store.account_or_default(&addr(&user));
    acc.balance = 10_000;
    store.set_account(&acc);
    let before = verinoo_exec::engine::compute_state_root(&store);

    let cases = [
        serde_json::json!(42),
        serde_json::json!({"temperatureMilli": -1}),
        serde_json::json!({"seed": "1.5"}),
        serde_json::json!({"temperatureMilli": 100_001}),
        serde_json::json!({"topPMilli": 1_001}),
        serde_json::json!({"topK": 1_000_001}),
        serde_json::json!({"topK": 321}),
        serde_json::json!({"temperature": 800}),
    ];
    for sampling in cases {
        let input = "please cancel";
        let tx = make_tx(
            &user,
            TxKind::SubmitInference,
            serde_json::json!({
                "requester": addr(&user),
                "targetModel": "tiny-gpt-v1",
                "input": input,
                "inputCommitment": verinoo_model::sha256_hex(input),
                "maxFee": 100,
                "deadline": 1_000,
                "sampling": sampling,
            }),
            1,
        );
        let err = verinoo_exec::engine::Validator {
            store: &store,
            base_fee: 1,
        }
        .validate_transaction(&tx)
        .unwrap_err();
        assert!(err.contains("sampling"), "{sampling}: {err}");
    }

    // A float cannot be encoded by canonical JSON, so no signed transaction can
    // carry one. The raw envelope is still rejected by the validator, and the
    // parser rejects it directly.
    let float_sampling = serde_json::json!({"seed": 1.5});
    assert!(verinoo_exec::engine::parse_sampling(Some(&float_sampling))
        .unwrap_err()
        .contains("malformed sampling"));
    let input = "please cancel";
    let data = TxData {
        r#type: TxKind::SubmitInference,
        data: serde_json::json!({
            "requester": addr(&user),
            "targetModel": "tiny-gpt-v1",
            "input": input,
            "inputCommitment": verinoo_model::sha256_hex(input),
            "maxFee": 100,
            "deadline": 1_000,
            "sampling": float_sampling,
        }),
    };
    let tx_id = "00".repeat(32);
    let signature = user.sign(tx_id.as_bytes());
    let float_tx = TxEnvelope {
        tx_id: tx_id.clone(),
        from: addr(&user),
        to: String::new(),
        value: 0,
        nonce: 1,
        data,
        gas_limit: None,
        gas_price: None,
        public_key: hex::encode(user.verifying_key().to_bytes()),
        signature: hex::encode(signature.to_bytes()),
    };
    assert!(verinoo_exec::engine::Validator {
        store: &store,
        base_fee: 1,
    }
    .validate_transaction(&float_tx)
    .is_err());

    assert_eq!(verinoo_exec::engine::compute_state_root(&store), before);
    assert_eq!(store.account_or_default(&addr(&user)).balance, 10_000);
    let tasks: Vec<verinoo_state::models::InferenceTask> = store.iter_prefix("task:");
    assert!(tasks.is_empty());
}

#[test]
fn invalid_sampling_is_skipped_without_debiting_gas() {
    install_test_models();
    let store = fresh_chain();
    let producer = ProducerKey::deterministic("validator-bad-sampling", 13);
    let user = keypair("bad-sampling-gas");
    let mut acc = store.account_or_default(&addr(&user));
    acc.balance = 10_000;
    store.set_account(&acc);

    let input = "please cancel";
    let tx = make_tx(
        &user,
        TxKind::SubmitInference,
        serde_json::json!({
            "requester": addr(&user),
            "targetModel": "tiny-gpt-v1",
            "input": input,
            "inputCommitment": verinoo_model::sha256_hex(input),
            "maxFee": 100,
            "deadline": 1_000,
            "sampling": {"temperatureMilli": 100_001},
        }),
        1,
    );
    let block = produce_block(&store, &[tx], &producer, 1_000_000).unwrap();
    assert!(block.transactions.is_empty());
    assert_eq!(store.account_or_default(&addr(&user)).balance, 10_000);
    let tasks: Vec<verinoo_state::models::InferenceTask> = store.iter_prefix("task:");
    assert!(tasks.is_empty());
}

#[test]
fn a_claimed_non_decoder_sampling_task_does_not_mint_or_release_capacity() {
    install_test_models();
    let store = fresh_chain();
    let producer = ProducerKey::deterministic("validator-mint", 14);
    let requester = keypair("mint-requester");
    let node_key = keypair("mint-node");
    let node_address = addr(&node_key);
    for signer in [&requester, &node_key] {
        let mut acc = store.account_or_default(&addr(signer));
        acc.balance = 10_000;
        store.set_account(&acc);
    }

    let input = "my parcel was never delivered";
    let spec = sampling_spec();
    let task_id = "sampling-mint-task";
    store.set_task(&verinoo_state::models::InferenceTask {
        task_id: task_id.into(),
        requester: addr(&requester),
        target_model: verinoo_model::MODEL_ID.into(),
        input_commitment: verinoo_model::sha256_hex(input),
        input: input.into(),
        encrypted_input: None,
        max_fee: 100,
        deadline: 1_000,
        sampling: Some(spec),
        verification_type: "optimistic".into(),
        status: "pending".into(),
        assigned_to: None,
        collateral: 0,
        result_hash: None,
        result_output: None,
        proof_data: None,
        challenge_window_end: None,
        game_id: None,
        executed_at: 0,
    });
    let pending_id = "sampling-pending-task";
    let mut pending = store.get_task(task_id).unwrap();
    pending.task_id = pending_id.into();
    store.set_task(&pending);

    let register = make_tx(
        &node_key,
        TxKind::RegisterNode,
        serde_json::json!({
            "supportedModels": [verinoo_model::MODEL_ID],
            "availableCapacity": 1,
            "maxCapacity": 1,
            "stakedAmount": 0,
        }),
        1,
    );
    let claim = make_tx(
        &node_key,
        TxKind::ClaimTask,
        serde_json::json!({"taskId": task_id}),
        2,
    );
    produce_block(&store, &[register, claim], &producer, 1_000_000).unwrap();

    let task = store.get_task(task_id).unwrap();
    assert_eq!(task.status, "assigned");
    assert_eq!(task.assigned_to.as_deref(), Some(node_address.as_str()));
    assert!(verinoo_exec::engine::receipt_for(&store, task_id).is_none());

    let node = store.get_node(&node_address).unwrap();
    assert_eq!(node.available_capacity, 0);
    assert_eq!(node.active_tasks, 1);
    assert_eq!(node.failed_inferences, 0);
    assert_eq!(store.account_or_default(&addr(&requester)).balance, 10_000);

    produce_block(&store, &[], &producer, 2_000_000).unwrap();

    let task = store.get_task(task_id).unwrap();
    assert_eq!(task.status, "assigned");
    assert_eq!(task.proof_data, None);
    let node = store.get_node(&node_address).unwrap();
    assert_eq!(node.available_capacity, 0);
    assert_eq!(node.active_tasks, 1);
    assert_eq!(node.failed_inferences, 0);
    assert_eq!(store.account_or_default(&addr(&requester)).balance, 10_000);
    let pending = store.get_task(pending_id).unwrap();
    assert_eq!(pending.status, "pending");
    assert!(verinoo_exec::engine::receipt_for(&store, pending_id).is_none());
}

fn fund(store: &StateStore, signer: &ed25519_dalek::SigningKey, balance: i64) {
    let mut account = store.account_or_default(&addr(signer));
    account.balance = balance;
    store.set_account(&account);
}

fn run_decoder_example(
    store: &StateStore,
    producer: &ProducerKey,
    requester: &str,
    prompt: &str,
    sampling: Option<verinoo_types::SamplingSpec>,
) -> Block {
    install_test_models();
    let user = keypair(requester);
    fund(store, &user, 10_000);
    let mut data = serde_json::json!({
        "requester": addr(&user),
        "targetModel": "tiny-gpt-v1",
        "input": prompt,
        "inputCommitment": verinoo_model::sha256_hex(prompt),
        "maxFee": 100,
        "deadline": 1_000,
    });
    if let Some(spec) = sampling {
        data["sampling"] = serde_json::to_value(spec).unwrap();
    }
    let tx = make_tx(&user, TxKind::SubmitInference, data, 1);
    let head = current_head(store).unwrap();
    let block = produce_block(store, &[tx], producer, 1_000_000).unwrap();
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

fn with_version(mut block: Block, version: u32, producer: &ProducerKey) -> Block {
    block.header.version = version;
    block.header.hash = verinoo_exec::engine::hash_header(&block.header);
    let signature = producer.signing.sign(block.header.hash.as_bytes());
    block.header.validator_signature = hex::encode(signature.to_bytes());
    block
}

#[test]
fn sampling_block_replays_identically_on_two_stores() {
    let spec = sampling_spec();
    let prompt = "please cancel";
    let a = fresh_chain();
    let b = fresh_chain();
    let block_a = run_decoder_example(
        &a,
        &ProducerKey::deterministic("validator-replay-sampling", 15),
        "replay-sampling",
        prompt,
        Some(spec),
    );
    let block_b = run_decoder_example(
        &b,
        &ProducerKey::deterministic("validator-replay-sampling", 15),
        "replay-sampling",
        prompt,
        Some(spec),
    );
    assert_eq!(block_a.header.state_root, block_b.header.state_root);
    assert_eq!(block_a.header.hash, block_b.header.hash);

    let task_a = a.iter_prefix::<verinoo_state::models::InferenceTask>("task:")[0].clone();
    let task_b = b.iter_prefix::<verinoo_state::models::InferenceTask>("task:")[0].clone();
    assert_eq!(task_a.result_output, task_b.result_output);
    assert_eq!(task_a.result_hash, task_b.result_hash);
    let receipt_a = verinoo_exec::engine::receipt_for(&a, &task_a.task_id).unwrap();
    let receipt_b = verinoo_exec::engine::receipt_for(&b, &task_b.task_id).unwrap();
    assert_eq!(receipt_a, receipt_b);
    assert_eq!(receipt_a["sampling"], serde_json::to_value(spec).unwrap());
}

#[test]
fn different_seeds_produce_different_token_ids() {
    install_test_models();
    let model = verinoo_exec::engine::ExecModel::resolve("tiny-gpt-v1").unwrap();
    let prompt = "please cancel";
    let seed_one = verinoo_types::SamplingSpec {
        temperature_milli: 1_000,
        top_k: 0,
        top_p_milli: 0,
        seed: 1,
    };
    let seed_two = verinoo_types::SamplingSpec {
        seed: 2,
        ..seed_one
    };
    let first = model
        .result_json_with_sampling(prompt, Some(&seed_one))
        .unwrap();
    let second = model
        .result_json_with_sampling(prompt, Some(&seed_two))
        .unwrap();
    let ids = |output: &str| {
        serde_json::from_str::<serde_json::Value>(output).unwrap()["tokenIds"].clone()
    };
    assert_ne!(ids(&first), ids(&second));
}

#[test]
fn v1_blocks_replay_without_sampling_and_reject_sampling() {
    install_test_models();
    let producer = ProducerKey::deterministic("validator-v1", 16);

    let source = fresh_chain();
    let genesis = current_head(&source).unwrap();
    let user = keypair("v1-replay");
    fund(&source, &user, 10_000);
    let input = "my parcel was never delivered";
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
        }),
        1,
    );
    let v2 = produce_block(&source, std::slice::from_ref(&tx), &producer, 1_000_000).unwrap();
    assert_eq!(
        v2.header.version,
        verinoo_exec::engine::HEADER_VERSION,
        "new blocks must use the current format"
    );

    let v1 = with_version(v2, 1, &producer);
    verinoo_exec::engine::validate_block(
        &v1,
        &genesis.header,
        std::slice::from_ref(&producer.address),
        1_000_000,
    )
    .expect("historical v1 block must still validate");

    let replay = fresh_chain();
    fund(&replay, &user, 10_000);
    verinoo_exec::engine::apply_block(&replay, &v1, 1, &producer.address).unwrap();
    assert_eq!(
        verinoo_exec::engine::compute_state_root(&replay),
        v1.header.state_root
    );
    let tasks = replay.iter_prefix::<verinoo_state::models::InferenceTask>("task:");
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].status, "completed");

    let sampled_source = fresh_chain();
    let sampled_genesis = current_head(&sampled_source).unwrap();
    let sampled = run_decoder_example(
        &sampled_source,
        &producer,
        "v1-sampled",
        "please cancel",
        Some(sampling_spec()),
    );
    let sampled_v1 = with_version(sampled, 1, &producer);
    let err = verinoo_exec::engine::validate_block(
        &sampled_v1,
        &sampled_genesis.header,
        std::slice::from_ref(&producer.address),
        1_000_000,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("version 1 block cannot contain sampling"),
        "{err}"
    );
    let apply_err =
        verinoo_exec::engine::apply_block(&fresh_chain(), &sampled_v1, 1, &producer.address)
            .unwrap_err();
    assert!(
        apply_err
            .to_string()
            .contains("version 1 block cannot contain sampling"),
        "{apply_err}"
    );
}
