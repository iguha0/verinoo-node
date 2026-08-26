//! Transaction + block execution engine.
//! Port of reference `src/core/engine.ts`, with real Ed25519 signatures
//! and canonical txId binding enforced from the start.

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde_json::json;
use sha2::{Digest, Sha256};
use verinoo_state::models::{
    Account, AgentAccount, ComputeNode, InferenceTask, ModelRegistryEntry,
};
use verinoo_state::StateStore;
use verinoo_types::tx::TxEnvelope;

use crate::gas;

pub fn sha256_hex(input: &str) -> String {
    let mut h = Sha256::new();
    h.update(input.as_bytes());
    hex::encode(h.finalize())
}

/// Ed25519 address: `ai_` + sha256(publicKey)[0..20] hex — mirrors reference.
pub fn public_key_to_address(public_key_hex: &str) -> Result<String, EngineError> {
    let bytes = hex::decode(public_key_hex)
        .map_err(|_| EngineError::Validation("publicKey is not hex".into()))?;
    if bytes.len() != 32 {
        return Err(EngineError::Validation("publicKey must be 32 bytes".into()));
    }
    let mut h = Sha256::new();
    h.update(&bytes);
    Ok(format!("ai_{}", hex::encode(&h.finalize()[..20])))
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("{0}")]
    Validation(String),
    #[error("execution reverted: {0}")]
    Reverted(String),
    #[error(transparent)]
    Canonical(#[from] verinoo_types::canonical::CanonicalError),
    #[error("database error: {0}")]
    Db(String),
}

impl EngineError {
    pub fn is_rejection(&self) -> bool {
        matches!(self, EngineError::Validation(_))
    }
}

// ED scheme transports signatures as HEX (spec: src/core/canonical.ts).
fn verify_ed25519(message: &str, sig_hex: &str, pk_hex: &str) -> bool {
    let (Ok(pk), Ok(sig)) = (hex::decode(pk_hex), hex::decode(sig_hex)) else {
        return false;
    };
    let (Ok(pk_arr), Ok(sig_arr)) = (<[u8; 32]>::try_from(pk), <[u8; 64]>::try_from(sig)) else {
        return false;
    };
    match (
        VerifyingKey::from_bytes(&pk_arr),
        Signature::from_slice(&sig_arr),
    ) {
        (Ok(vk), Ok(sig)) => vk.verify(message.as_bytes(), &sig).is_ok(),
        _ => false,
    }
}

// ============================ validation ============================

pub struct Validator<'a> {
    pub store: &'a StateStore,
    pub base_fee: u64,
}

impl<'a> Validator<'a> {
    /// Full reference-parity transaction validation. Order matters: it must
    /// match the TS engine's failure precedence for golden vectors.
    pub fn validate_transaction(&self, tx: &TxEnvelope) -> Result<(), String> {
        if tx.public_key.is_empty() || tx.signature.is_empty() {
            return Err("missing sig".into());
        }
        let addr =
            public_key_to_address(&tx.public_key).map_err(|_| "sig/from mismatch".to_string())?;
        if addr != tx.from {
            return Err("sig/from mismatch".into());
        }
        if !verify_ed25519(&tx.tx_id, &tx.signature, &tx.public_key) {
            return Err("invalid sig".into());
        }
        // Canonical binding — the anti-malleation check.
        let payload = verinoo_types::canonical::UnsignedPayload {
            data: serde_json::to_value(&tx.data).unwrap(),
            from: tx.from.clone(),
            nonce: tx.nonce,
            to: tx.to.clone(),
            value: tx.value,
            gas_limit: tx.gas_limit,
            gas_price: tx.gas_price,
        };
        let required = payload.tx_id().map_err(|e| e.to_string())?;
        if tx.tx_id != required {
            return Err("txId does not match transaction contents".into());
        }

        // Policy validation (submitInference only).
        if as_kind_str(&tx.data) == "submitInference" {
            let declared = tx.data.data["verificationType"].as_str();
            if let Some(d) = declared {
                gas::policy_multiplier(Some(d))?;
            }
        }

        let acc = self.store.get_account(&tx.from);
        if let Some(a) = &acc {
            if tx.nonce != a.nonce + 1 {
                return Err(format!("bad nonce (have {})", a.nonce));
            }
        }
        let bal = acc.as_ref().map(|a| a.balance).unwrap_or(0);
        let data_value = serde_json::to_value(&tx.data).unwrap();
        let max_fee = gas::gas_cost_for(&data_value)
            * self.base_fee
            * match gas::policy_of_tx(&data_value) {
                Some(p) => gas::policy_multiplier(Some(p))?,
                None => 1,
            };
        if (bal as u64) < tx.value + max_fee {
            return Err("insufficient balance for value + gas".into());
        }
        Ok(())
    }
}

fn as_kind_str(tx_data: &verinoo_types::tx::TxData) -> String {
    match tx_data.r#type {
        verinoo_types::tx::TxKind::Transfer => "transfer",
        verinoo_types::tx::TxKind::RegisterModel => "registerModel",
        verinoo_types::tx::TxKind::RegisterNode => "registerNode",
        verinoo_types::tx::TxKind::RegisterAgent => "registerAgent",
        verinoo_types::tx::TxKind::SubmitInference => "submitInference",
        verinoo_types::tx::TxKind::ClaimTask => "claimTask",
        verinoo_types::tx::TxKind::SubmitResult => "submitResult",
        verinoo_types::tx::TxKind::ChallengeResult => "challengeResult",
        verinoo_types::tx::TxKind::Bisect => "bisect",
        verinoo_types::tx::TxKind::ProveStep => "proveStep",
        verinoo_types::tx::TxKind::AgentPayment => "agentPayment",
        verinoo_types::tx::TxKind::RegisterValidator => "registerValidator",
        verinoo_types::tx::TxKind::UnregisterValidator => "unregisterValidator",
    }
    .to_string()
}

// ============================ blocks ============================

use verinoo_types::canonical::canonical_json;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BlockHeader {
    pub hash: String,
    pub version: u32,
    pub index: u64,
    pub timestamp: u64,
    #[serde(rename = "previousHash")]
    pub previous_hash: String,
    pub validator: String,
    #[serde(rename = "validatorPubKey")]
    pub validator_pub_key: String,
    #[serde(rename = "validatorSignature")]
    pub validator_signature: String,
    #[serde(rename = "stateRoot")]
    pub state_root: String,
    #[serde(rename = "txRoot")]
    pub tx_root: String,
    #[serde(rename = "inferenceTasksRoot")]
    pub inference_tasks_root: String,
    #[serde(rename = "computeRoot")]
    pub compute_root: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Block {
    pub header: BlockHeader,
    pub transactions: Vec<TxEnvelope>,
}

/// Port of reference hashHeader(): canonical JSON over the identity subset.
pub fn hash_header(h: &BlockHeader) -> String {
    let payload = serde_json::json!({
        "v": h.version, "i": h.index, "t": h.timestamp,
        "p": h.previous_hash, "s": h.state_root, "tx": h.tx_root,
        "inf": h.inference_tasks_root, "comp": h.compute_root,
        "val": h.validator,
    });
    sha256_hex(&canonical_json(&payload).unwrap())
}

/// Port of merkleRoot().
pub fn merkle_root(hashes: &[String]) -> String {
    if hashes.is_empty() {
        return sha256_hex("empty-merkle");
    }
    let mut row: Vec<String> = hashes.to_vec();
    while row.len() > 1 {
        let mut next = Vec::with_capacity(row.len() / 2);
        for pair in row.chunks(2) {
            let a = &pair[0];
            let b = pair.get(1).unwrap_or(&pair[0]);
            next.push(sha256_hex(&format!("{a}{b}")));
        }
        row = next;
    }
    row[0].clone()
}

pub const HEADER_VERSION: u32 = 1;

/// Structural + cryptographic block gate. Stateless checks that MUST hold
/// before anything touches state. Stateful rules (nonce/balance) are
/// enforced per-tx during apply_block with skip-on-failure, matching the
/// reference engine's semantics exactly.
pub fn validate_block(
    block: &Block,
    prev_head: &BlockHeader,
    allowed_validators: &[String],
    now_ms: u64,
) -> Result<(), EngineError> {
    validate_block_with_policy(
        block,
        prev_head,
        allowed_validators,
        now_ms,
        &crate::leader::RoundRobin,
    )
}

/// Same gate with an injectable leader policy (Malachite adapter point).
pub fn validate_block_with_policy(
    block: &Block,
    prev_head: &BlockHeader,
    allowed_validators: &[String],
    now_ms: u64,
    policy: &dyn crate::leader::LeaderPolicy,
) -> Result<(), EngineError> {
    if block.header.version != HEADER_VERSION {
        return Err(EngineError::Validation(format!(
            "unknown block version {}",
            block.header.version
        )));
    }
    // Hash binding.
    let recomputed = hash_header(&block.header);
    if recomputed != block.header.hash {
        return Err(EngineError::Validation("header hash mismatch".into()));
    }
    // Producer signature (Ed25519 over header hash string; hex transport).
    let vkey_bytes = hex::decode(&block.header.validator_pub_key)
        .ok()
        .and_then(|b| <[u8; 32]>::try_from(b).ok());
    let sig_bytes = hex::decode(&block.header.validator_signature)
        .ok()
        .and_then(|b| <[u8; 64]>::try_from(b).ok());
    let producer_ok = match (vkey_bytes, sig_bytes) {
        (Some(vk_bytes), Some(sig_bytes)) => {
            match (
                VerifyingKey::from_bytes(&vk_bytes),
                Signature::from_slice(&sig_bytes),
            ) {
                (Ok(vk), Ok(sig)) => vk.verify(block.header.hash.as_bytes(), &sig).is_ok(),
                _ => false,
            }
        }
        _ => false,
    };
    if !producer_ok {
        return Err(EngineError::Validation(
            "invalid validator signature".into(),
        ));
    }
    // Producer authorization: must be in the active set...
    if !allowed_validators.contains(&block.header.validator) {
        return Err(EngineError::Validation(format!(
            "unauthorized producer {}",
            block.header.validator
        )));
    }
    // ...and the pubkey must derive to that validator's address.
    let derived = public_key_to_address(&block.header.validator_pub_key)
        .map_err(|_| EngineError::Validation("bad validatorPubKey".into()))?;
    if derived != block.header.validator {
        return Err(EngineError::Validation(
            "validatorPubKey does not derive to validator".into(),
        ));
    }
    // ...and this height is the producer's scheduled slot.
    match policy.leader_at(block.header.index, allowed_validators) {
        Some(leader) if leader == block.header.validator => {}
        Some(leader) => {
            return Err(EngineError::Validation(format!(
                "wrong producer for height {}: expected {leader}, got {}",
                block.header.index, block.header.validator
            )));
        }
        None => return Err(EngineError::Validation("empty validator set".into())),
    }
    // Chain linkage.
    if block.header.index != prev_head.index + 1 || block.header.previous_hash != prev_head.hash {
        return Err(EngineError::Validation("block does not extend head".into()));
    }
    // Timestamp sanity: monotonic vs parent, max 2h in the future.
    if block.header.timestamp < prev_head.timestamp {
        return Err(EngineError::Validation("timestamp went backwards".into()));
    }
    if block.header.timestamp > now_ms.saturating_add(2 * 3600 * 1000) {
        return Err(EngineError::Validation(
            "timestamp too far in future".into(),
        ));
    }
    // Tx root + duplicate rejection.
    let ids: Vec<String> = block.transactions.iter().map(|t| t.tx_id.clone()).collect();
    if block.header.tx_root != merkle_root(&ids) {
        return Err(EngineError::Validation("txRoot mismatch".into()));
    }
    let unique: std::collections::HashSet<&String> = ids.iter().collect();
    if unique.len() != ids.len() {
        return Err(EngineError::Validation(
            "duplicate transaction in block".into(),
        ));
    }
    // Per-tx signature + canonical binding (stateless subset).
    for tx in &block.transactions {
        let addr = public_key_to_address(&tx.public_key)
            .map_err(|_| EngineError::Validation("bad publicKey encoding".into()))?;
        if addr != tx.from {
            return Err(EngineError::Validation("sig/from mismatch".into()));
        }
        if !verify_ed25519(&tx.tx_id, &tx.signature, &tx.public_key) {
            return Err(EngineError::Validation("invalid sig".into()));
        }
        let payload = verinoo_types::canonical::UnsignedPayload {
            data: serde_json::to_value(&tx.data).unwrap(),
            from: tx.from.clone(),
            nonce: tx.nonce,
            to: tx.to.clone(),
            value: tx.value,
            gas_limit: tx.gas_limit,
            gas_price: tx.gas_price,
        };
        if payload
            .tx_id()
            .map_err(|e| EngineError::Validation(e.to_string()))?
            != tx.tx_id
        {
            return Err(EngineError::Validation(
                "txId does not match transaction contents".into(),
            ));
        }
    }
    Ok(())
}

// ============================ execution ============================

pub struct BlockContext {
    pub height: u64,
    pub validator_address: String,
    pub base_fee: u64,
}

/// Execute one transaction against state, including the mandatory gas fee
/// (25% burn to treasury / 75% to block producer). Mirrors the reference
/// engine's failure semantics: Err means "skipped" at block level.
pub fn execute_transaction(
    store: &StateStore,
    tx: &TxEnvelope,
    ctx: &BlockContext,
) -> Result<(), EngineError> {
    let fee = gas::gas_cost_for(&serde_json::to_value(&tx.data).unwrap())
        * ctx.base_fee
        * gas::policy_multiplier(gas::policy_of_tx(&serde_json::to_value(&tx.data).unwrap()))
            .unwrap_or(1);

    let mut sender = store.account_or_default(&tx.from);
    let total = tx.value + fee;
    if (sender.balance as u128) < total as u128 {
        return Err(EngineError::Reverted("overdraft".into()));
    }
    sender.balance -= total as i64;
    sender.nonce = tx.nonce;
    sender.updated_at = ctx.height;
    store.set_account(&sender);

    // Fee split.
    let validator_share = fee * 3 / 4;
    if validator_share > 0 {
        let mut v = store.account_or_default(&ctx.validator_address);
        v.balance += validator_share as i64;
        v.updated_at = ctx.height;
        store.set_account(&v);
    }
    let burn = fee - validator_share;
    if burn > 0 {
        let mut t = store.account_or_default("treasury");
        t.balance += burn as i64;
        store.set_account(&t);
    }

    // value transfer
    if !tx.to.is_empty() && tx.to != tx.from {
        let mut recv = store.account_or_default(&tx.to);
        recv.balance += tx.value as i64;
        recv.updated_at = ctx.height;
        store.set_account(&recv);
    }

    let data = &tx.data.data;

    match tx.data.r#type {
        verinoo_types::tx::TxKind::Transfer => {}

        verinoo_types::tx::TxKind::RegisterModel => {
            let model_id = sha256_hex(&format!(
                "{}{}",
                verinoo_types::canonical::canonical_json(data).map_err(EngineError::Canonical)?,
                tx.tx_id
            ))[..32]
                .to_string();
            store.set_model(&ModelRegistryEntry {
                model_id,
                architecture: data["architecture"].as_str().unwrap_or("").into(),
                parameter_count: data["parameterCount"].as_u64().unwrap_or(0),
                weights_hash: data["weightsHash"].as_str().unwrap_or("").into(),
                runtime_hash: data["runtimeHash"].as_str().unwrap_or("").into(),
                staking_requirement: data["stakingRequirement"].as_u64().unwrap_or(0),
                description: data["description"].as_str().unwrap_or("").into(),
                owner: tx.from.clone(),
                registered_at: ctx.height,
                is_active: true,
            });
        }

        verinoo_types::tx::TxKind::RegisterNode => {
            store.set_node(&ComputeNode {
                node_id: tx.from.clone(),
                public_key: tx.public_key.clone(),
                staked_amount: data["stakedAmount"].as_u64().unwrap_or(0),
                available_capacity: data["availableCapacity"].as_u64().unwrap_or(0),
                max_capacity: data["maxCapacity"].as_u64().unwrap_or(0),
                active_tasks: data["activeTasks"].as_u64().unwrap_or(0),
                reputation: data["reputation"].as_i64().unwrap_or(0),
                successful_inferences: 0,
                failed_inferences: 0,
                supported_models: data["supportedModels"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default(),
                registered_at: ctx.height,
                last_heartbeat_at: 0,
                is_active: true,
                total_slashed: 0,
            });
        }

        verinoo_types::tx::TxKind::RegisterAgent => {
            let agent = AgentAccount {
                address: tx.from.clone(),
                public_key: tx.public_key.clone(),
                nonce: 0,
                balance: 0,
                reputation: 50,
                total_inferences: 0,
                total_fees_paid: 0,
                is_autonomous: true,
                services_provided: serde_json::json!({}),
                updated_at: ctx.height,
            };
            store.set_agent(&agent);
        }

        verinoo_types::tx::TxKind::SubmitInference => {
            let task_id = sha256_hex(&format!(
                "{}{}",
                verinoo_types::canonical::canonical_json(data).map_err(EngineError::Canonical)?,
                tx.tx_id
            ))[..32]
                .to_string();
            store.set_task(&InferenceTask {
                task_id,
                requester: data["requester"].as_str().unwrap_or(&tx.from).into(),
                target_model: data["targetModel"].as_str().unwrap_or("").into(),
                input_commitment: data["inputCommitment"].as_str().unwrap_or("").into(),
                max_fee: data["maxFee"].as_u64().unwrap_or(0),
                deadline: data["deadline"].as_u64().unwrap_or(0),
                verification_type: data["verificationType"]
                    .as_str()
                    .unwrap_or("optimistic")
                    .into(),
                status: "pending".into(),
                assigned_to: None,
                collateral: 0,
                result_hash: None,
                result_output: None,
                proof_data: None,
                challenge_window_end: None,
                game_id: None,
            });
        }

        verinoo_types::tx::TxKind::ClaimTask => {
            let task_id = data["taskId"]
                .as_str()
                .ok_or_else(|| EngineError::Reverted("missing taskId".into()))?;
            let mut task = store
                .get_task(task_id)
                .ok_or_else(|| EngineError::Reverted("task unavailable".into()))?;
            if task.status != "pending" {
                return Err(EngineError::Reverted("task unavailable".into()));
            }
            let mut node = store
                .get_node(&tx.from)
                .ok_or_else(|| EngineError::Reverted("unsupported model".into()))?;
            if !node.supported_models.contains(&task.target_model) {
                return Err(EngineError::Reverted("unsupported model".into()));
            }
            if node.available_capacity == 0 {
                return Err(EngineError::Reverted("at capacity".into()));
            }
            task.status = "assigned".into();
            task.assigned_to = Some(node.node_id.clone());
            task.collateral = task.max_fee / 5; // 20%
            store.set_task(&task);
            node.available_capacity -= 1;
            node.active_tasks += 1;
            store.set_node(&node);
        }

        verinoo_types::tx::TxKind::SubmitResult => {
            let task_id = data["taskId"]
                .as_str()
                .ok_or_else(|| EngineError::Reverted("missing taskId".into()))?;
            let mut task = store
                .get_task(task_id)
                .ok_or_else(|| EngineError::Reverted("task not assigned".into()))?;
            if task.status != "assigned" {
                return Err(EngineError::Reverted("task not assigned".into()));
            }
            if task.assigned_to.as_deref() != Some(tx.from.as_str()) {
                return Err(EngineError::Reverted("not assignee".into()));
            }
            task.status = "completed".into();
            task.result_hash = data["resultHash"].as_str().map(String::from);
            task.result_output = data["resultOutput"].as_str().map(String::from);
            task.proof_data = data["proofData"]
                .as_str()
                .map(String::from)
                .or(Some(String::new()));
            task.challenge_window_end = Some(ctx.height + 100);
            store.set_task(&task);

            let node_id = task.assigned_to.clone().unwrap();
            let mut node = store.get_node(&node_id).expect("assignee node exists");
            node.available_capacity += 1;
            node.active_tasks = node.active_tasks.saturating_sub(1);
            node.successful_inferences += 1;
            node.reputation = (node.reputation + 1).min(100);
            store.set_node(&node);

            let mut acc = store.account_or_default(&node_id);
            acc.balance += (task.max_fee * 4 / 5) as i64;
            acc.updated_at = ctx.height;
            store.set_account(&acc);
        }

        verinoo_types::tx::TxKind::ChallengeResult => {
            let task_id = data["taskId"]
                .as_str()
                .ok_or_else(|| EngineError::Reverted("missing taskId".into()))?;
            let mut task = store
                .get_task(task_id)
                .ok_or_else(|| EngineError::Reverted("cannot challenge".into()))?;
            if task.status != "completed" {
                return Err(EngineError::Reverted("cannot challenge".into()));
            }
            if ctx.height > task.challenge_window_end.unwrap_or(0) {
                return Err(EngineError::Reverted("challenge window closed".into()));
            }
            // Challenger bond.
            let bond = task.max_fee;
            let mut ch = store.account_or_default(&tx.from);
            ch.balance -= bond as i64;
            if ch.balance < 0 {
                return Err(EngineError::Reverted("insufficient challenge bond".into()));
            }
            store.set_account(&ch);

            let model_arch = store
                .get_model(&task.target_model)
                .map(|m| m.architecture)
                .unwrap_or_else(|| task.target_model.clone());
            let game_id =
                sha256_hex(&format!("game{task_id}{}{}", tx.from, ctx.height))[..32].to_string();
            let layer_spec = crate::spec::get_layer_spec(&model_arch);

            let game = serde_json::json!({
                "gameId": game_id,
                "taskId": task_id,
                "challenger": tx.from,
                "defender": task.assigned_to,
                "status": "open",
                "low": 0,
                "high": layer_spec.len(),
                "currentStep": 0,
                "maxSteps": (layer_spec.len() as f64).log2().ceil() as u64 + 2,
                "challengerCommitments": {},
                "defenderCommitments": {},
                "disputedLayer": -1,
                "challengerBond": bond,
                "defenderBond": json!(if task.collateral > 0 { task.collateral } else { task.max_fee / 5 }),
                "architecture": task.target_model,
                "layerSpec": serde_json::to_value(&layer_spec).unwrap(),
                "openedAt": ctx.height,
                "lastMoveAt": ctx.height,
                "moveTimeout": 5,
            });
            store.set_game(&game_id, &game);
            task.status = "challenged".into();
            task.game_id = Some(game_id.clone());
            store.set_task(&task);

            if let Some(node) = task
                .assigned_to
                .as_deref()
                .and_then(|id| store.get_node(id))
            {
                let mut n = node;
                n.reputation = (n.reputation - 10).max(0);
                store.set_node(&n);
            }
        }

        verinoo_types::tx::TxKind::Bisect => {
            let game_id = data["gameId"]
                .as_str()
                .ok_or_else(|| EngineError::Reverted("missing gameId".into()))?;
            let mut game = store
                .get_game(game_id)
                .ok_or_else(|| EngineError::Reverted("game not found".into()))?;
            let status = game["status"].as_str().unwrap_or("");
            if status != "open" && status != "bisecting" {
                return Err(EngineError::Reverted("game not active".into()));
            }
            let last_move = game["lastMoveAt"].as_u64().unwrap_or(0);
            let timeout = game["moveTimeout"].as_u64().unwrap_or(5);
            if ctx.height > last_move + timeout {
                return Err(EngineError::Reverted("move timeout".into()));
            }
            let challenger = game["challenger"].as_str().unwrap_or("");
            let defender = game["defender"].as_str().unwrap_or("");
            if tx.from != challenger && tx.from != defender {
                return Err(EngineError::Reverted("not a participant".into()));
            }
            let layer_index = data["layerIndex"].as_i64().unwrap_or(-1);
            let trace_root = data["traceRoot"].as_str().unwrap_or("");

            let side_key = if tx.from == challenger {
                "challengerCommitments"
            } else {
                "defenderCommitments"
            };
            game[side_key][&layer_index.to_string()] = json!(trace_root);

            let ch_root = game["challengerCommitments"][&layer_index.to_string()]
                .as_str()
                .map(String::from);
            let df_root = game["defenderCommitments"][&layer_index.to_string()]
                .as_str()
                .map(String::from);

            let mut low = game["low"].as_i64().unwrap_or(0);
            let mut high = game["high"].as_i64().unwrap_or(0);
            let mut step = game["currentStep"].as_i64().unwrap_or(0);

            if let (Some(c), Some(d)) = (&ch_root, &df_root) {
                if c == d {
                    low = layer_index;
                } else {
                    high = layer_index;
                    // Reference engine marks the disputed layer immediately
                    // upon disagreement (engine.ts bisect case).
                    game["disputedLayer"] = json!(layer_index);
                }
                step += 1;
            }
            game["low"] = json!(low);
            game["high"] = json!(high);
            game["currentStep"] = json!(step);

            if high - low <= 1 || step >= game["maxSteps"].as_i64().unwrap_or(0) {
                game["status"] = json!("proving");
                game["disputedLayer"] = json!((low + high) / 2);
            } else {
                game["status"] = json!("bisecting");
            }
            game["lastMoveAt"] = json!(ctx.height);
            store.set_game(game_id, &game);
        }

        verinoo_types::tx::TxKind::ProveStep => {
            let game_id = data["gameId"]
                .as_str()
                .ok_or_else(|| EngineError::Reverted("missing gameId".into()))?;
            let mut game = store
                .get_game(game_id)
                .ok_or_else(|| EngineError::Reverted("game not found".into()))?;
            if game["status"].as_str() != Some("proving") {
                return Err(EngineError::Reverted("not at proving stage".into()));
            }
            if tx.from != game["defender"].as_str().unwrap_or("") {
                return Err(EngineError::Reverted("only defender can prove".into()));
            }
            let last_move = game["lastMoveAt"].as_u64().unwrap_or(0);
            let timeout = game["moveTimeout"].as_u64().unwrap_or(5);
            if ctx.height > last_move + timeout {
                return Err(EngineError::Reverted("prove timeout".into()));
            }

            let disputed_layer = game["disputedLayer"].as_i64().unwrap_or(0) as usize;
            let spec = &game["layerSpec"][disputed_layer];
            let op_type = spec["opType"].as_str().unwrap_or("");
            let _input_shape = &spec["inputShape"];

            let layer_weights = data["layerWeights"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_i64().map(|x| x as i32))
                        .collect::<Vec<i32>>()
                })
                .unwrap_or_default();
            let layer_input = data["layerInput"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_i64().map(|x| x as i32))
                        .collect::<Vec<i32>>()
                })
                .unwrap_or_default();
            let claimed_output = data["layerOutput"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_i64().map(|x| x as i32))
                        .collect::<Vec<i32>>()
                })
                .unwrap_or_default();

            // Deterministic recompute using native Q16.16 witnesses.
            let honest_output =
                verinoo_zk::execute_layer(op_type, &layer_weights, &layer_input, None);
            let is_valid = claimed_output == honest_output;

            let challenger_addr = game["challenger"].as_str().unwrap_or("").to_string();
            let defender_addr = tx.from.clone();
            let ch_bond = game["challengerBond"].as_u64().unwrap_or(0);
            let df_bond = game["defenderBond"].as_u64().unwrap_or(0);

            if is_valid {
                // DEFENDER WINS: gets both bonds.
                game["status"] = json!("resolved_valid");
                game["winner"] = json!(defender_addr);
                game["loser"] = json!(challenger_addr);

                let mut df_acc = store.account_or_default(&defender_addr);
                df_acc.balance += (ch_bond + df_bond) as i64;
                df_acc.updated_at = ctx.height;
                store.set_account(&df_acc);

                let mut ch_acc = store.account_or_default(&challenger_addr);
                ch_acc.balance -= ch_bond as i64;
                ch_acc.updated_at = ctx.height;
                store.set_account(&ch_acc);

                if let Some(mut node) = store.get_node(&defender_addr) {
                    node.reputation = (node.reputation + 15).min(100);
                    store.set_node(&node);
                }
                eprintln!("[verif] DEFENDER WINS — challenger slashed {ch_bond}");
            } else {
                // CHALLENGER WINS: defender slashed.
                game["status"] = json!("resolved_slash");
                game["winner"] = json!(challenger_addr);
                game["loser"] = json!(defender_addr);

                let mut df_acc = store.account_or_default(&defender_addr);
                df_acc.balance -= df_bond as i64;
                df_acc.updated_at = ctx.height;
                store.set_account(&df_acc);

                let mut ch_acc = store.account_or_default(&challenger_addr);
                ch_acc.balance += (df_bond + ch_bond) as i64;
                ch_acc.updated_at = ctx.height;
                store.set_account(&ch_acc);

                if let Some(mut node) = store.get_node(&defender_addr) {
                    node.reputation = (node.reputation - 30).max(0);
                    node.total_slashed += df_bond;
                    store.set_node(&node);
                }
                // If the defender is also a validator: remove from set + slash stake.
                if let Some(mut val) = store.get_validator(&defender_addr) {
                    let slash_amount = val.staked_amount / 2; // 50% slash
                    val.staked_amount -= slash_amount;
                    if val.staked_amount < 1000 {
                        // Below minimum: remove entirely
                        store.remove_validator(&defender_addr);
                        tracing::info!(
                            "validator {defender_addr} removed (insufficient post-slash stake)"
                        );
                    }
                    store.set_validator(&val);
                }
                eprintln!("[verif] CHALLENGER WINS — defender slashed {df_bond}");
            }

            game["resolvedAt"] = json!(ctx.height);
            game["lastMoveAt"] = json!(ctx.height);
            store.set_game(game_id, &game);

            // Update task status based on outcome.
            if let Some(task_id) = game["taskId"].as_str() {
                if let Some(mut task) = store.get_task(task_id) {
                    task.status = if is_valid {
                        "completed".into()
                    } else {
                        "failed".into()
                    };
                    store.set_task(&task);
                }
            }
        }

        verinoo_types::tx::TxKind::RegisterValidator => {
            use verinoo_state::validators::{ValidatorEntry, MIN_VALIDATOR_STAKE};
            let stake = data["stakedAmount"].as_u64().unwrap_or(0);
            if store.get_validator(&tx.from).is_some() {
                return Err(EngineError::Reverted("already a validator".into()));
            }
            if stake < MIN_VALIDATOR_STAKE {
                return Err(EngineError::Reverted(format!(
                    "stake below minimum ({stake} < {MIN_VALIDATOR_STAKE})"
                )));
            }
            // Bond: debit sender.
            let mut acc = store.account_or_default(&tx.from);
            acc.balance -= stake as i64;
            if acc.balance < 0 {
                return Err(EngineError::Reverted(
                    "insufficient balance for stake".into(),
                ));
            }
            acc.updated_at = ctx.height;
            store.set_account(&acc);
            store.set_validator(&ValidatorEntry {
                address: tx.from.clone(),
                public_key: tx.public_key.clone(),
                staked_amount: stake,
                registered_at_height: ctx.height,
                genesis: false,
            });
        }

        verinoo_types::tx::TxKind::UnregisterValidator => {
            let entry = store
                .get_validator(&tx.from)
                .ok_or_else(|| EngineError::Reverted("not a validator".into()))?;
            // Guard: never leave the set empty.
            let active = store.active_validators();
            if active.len() <= 1 {
                return Err(EngineError::Reverted("cannot remove last validator".into()));
            }
            // Refund bond.
            let mut acc = store.account_or_default(&tx.from);
            acc.balance += entry.staked_amount as i64;
            acc.updated_at = ctx.height;
            store.set_account(&acc);
            store.remove_validator(&tx.from);
        }

        verinoo_types::tx::TxKind::AgentPayment => {
            let agent_addr = tx.from.clone();
            let mut agent = store
                .get_agent(&agent_addr)
                .ok_or_else(|| EngineError::Reverted("not an agent".into()))?;
            let fee_amt = data["fee"].as_u64().unwrap_or(0);
            let provider = data["provider"]
                .as_str()
                .ok_or_else(|| EngineError::Reverted("missing provider".into()))?;
            let mut prov = store.account_or_default(provider);
            prov.balance += fee_amt as i64;
            prov.updated_at = ctx.height;
            store.set_account(&prov);
            agent.total_inferences += 1;
            agent.total_fees_paid += fee_amt;
            agent.updated_at = ctx.height;
            store.set_agent(&agent);
        }
    }
    Ok(())
}

// ============================ apply / produce ============================

use verinoo_state::genesis::INITIAL_BASE_FEE;

/// Compute a deterministic fingerprint of the observable state.
/// NOTE: intentionally NOT the reference client's quirky sHash; conformance
/// compares observable entities instead. Documented divergence.
pub fn compute_state_root(store: &StateStore) -> String {
    let mut accounts: Vec<String> = store
        .iter_prefix::<Account>("acc:")
        .into_iter()
        .map(|a| serde_json::to_string(&a).unwrap())
        .collect();
    accounts.sort();
    let mut models: Vec<String> = store
        .iter_prefix::<ModelRegistryEntry>("model:")
        .into_iter()
        .map(|m| m.model_id)
        .collect();
    models.sort();
    let mut pending: Vec<String> = store
        .iter_tasks_by_status("pending")
        .into_iter()
        .map(|t| t.task_id)
        .collect();
    pending.sort();
    let nodes: Vec<String> = store
        .iter_nodes_by_reputation()
        .into_iter()
        .map(|n| n.node_id)
        .collect();

    let _ = &mut accounts;
    let payload = serde_json::json!({
        "accounts": accounts,
        "models": models,
        "pending": pending,
        "nodes": {
            "list": { "nodes": nodes },
        },
    });
    sha256_hex(&canonical_json(&payload).unwrap())
}

/// Deterministic auto-matching of pending tasks to capable nodes.
fn match_pending_tasks(store: &StateStore, height: u64) {
    let mut tasks = store.iter_tasks_by_status("pending");
    tasks.sort_by(|a, b| a.task_id.cmp(&b.task_id));
    for mut task in tasks {
        for node in store.iter_nodes_by_reputation() {
            if node.supported_models.contains(&task.target_model) && node.available_capacity > 0 {
                let requester_ok = store
                    .get_account(&task.requester)
                    .map(|a| a.balance as u64 >= task.max_fee)
                    .unwrap_or(false);
                if !requester_ok {
                    continue;
                }
                task.status = "assigned".into();
                task.assigned_to = Some(node.node_id.clone());
                task.collateral = task.max_fee / 5;
                store.set_task(&task);

                let mut acc = store.account_or_default(&task.requester);
                acc.balance -= task.max_fee as i64;
                acc.updated_at = height;
                store.set_account(&acc);

                let mut n = node;
                n.available_capacity -= 1;
                n.active_tasks += 1;
                store.set_node(&n);
                break;
            }
        }
    }
}

/// Execute a validated block body. Caller MUST have run validate_block.
pub fn apply_block(
    store: &StateStore,
    block: &Block,
    base_fee: u64,
    validator_address: &str,
) -> Result<AppliedBlock, EngineError> {
    let height = block.header.index;
    let ctx = BlockContext {
        height,
        validator_address: validator_address.to_string(),
        base_fee,
    };

    let mut executed: Vec<&TxEnvelope> = Vec::new();
    let mut gas_left = crate::gas::block_gas_limit();
    let mut skipped: Vec<(String, String)> = Vec::new();
    eprintln!(
        "[apply_block] height={} txs={} validator={}",
        height,
        block.transactions.len(),
        validator_address
    );

    for tx in &block.transactions {
        let cost = crate::gas::gas_cost_for(&serde_json::to_value(&tx.data).unwrap());
        if cost > gas_left {
            skipped.push((tx.tx_id.clone(), "block gas limit".into()));
            continue;
        }
        // Stateful validation + execution; failures skip the tx (reference semantics).
        let validator = Validator { store, base_fee };
        if let Err(e) = validator.validate_transaction(tx) {
            skipped.push((tx.tx_id.clone(), e));
            continue;
        }
        match execute_transaction(store, tx, &ctx) {
            Ok(()) => {
                executed.push(tx);
                gas_left -= cost;
            }
            Err(e) => skipped.push((tx.tx_id.clone(), e.to_string())),
        }
    }

    match_pending_tasks(store, height);

    let gas_used: u64 = executed
        .iter()
        .map(|t| crate::gas::gas_cost_for(&serde_json::to_value(&t.data).unwrap()))
        .sum();

    Ok(AppliedBlock {
        executed_ids: executed.iter().map(|t| t.tx_id.clone()).collect(),
        skipped,
        gas_used,
        next_base_fee: crate::gas::compute_base_fee(base_fee, gas_used),
    })
}

#[derive(Debug)]
pub struct AppliedBlock {
    pub executed_ids: Vec<String>,
    pub skipped: Vec<(String, String)>,
    pub gas_used: u64,
    pub next_base_fee: u64,
}

/// A block producer's signing identity.
pub struct ProducerKey {
    pub address: String,
    pub public_key_hex: String,
    pub signing: ed25519_dalek::SigningKey,
}

impl ProducerKey {
    pub fn from_parts(address: String, signing: ed25519_dalek::SigningKey) -> Self {
        let public_key_hex = hex::encode(signing.verifying_key().to_bytes());
        Self {
            address,
            public_key_hex,
            signing,
        }
    }

    /// seed = sha256(name) — deterministic test identities.
    pub fn deterministic(name: &str, index: u32) -> Self {
        let mut h = Sha256::new();
        h.update(format!("{name}:{index}").as_bytes());
        let digest: [u8; 32] = h.finalize().into();
        let sk = ed25519_dalek::SigningKey::from_bytes(&digest);
        let vk = sk.verifying_key();
        Self {
            address: public_key_to_address(&hex::encode(vk.to_bytes())).unwrap(),
            public_key_hex: hex::encode(vk.to_bytes()),
            signing: sk,
        }
    }
}

/// Produce a block: gas-limited execution loop + header construction +
/// Ed25519 signature over the final header hash. Mirrors produceBlock.
pub fn produce_block(
    store: &StateStore,
    txs: &[TxEnvelope],
    producer: &ProducerKey,
    now_ms: u64,
) -> Result<Block, EngineError> {
    let prev = current_head(store).ok_or(EngineError::Validation("no head".into()))?;
    let height = prev.header.index + 1;
    let base_fee = store.meta_u64_or("baseFee", INITIAL_BASE_FEE);
    let ctx = BlockContext {
        height,
        validator_address: producer.address.clone(),
        base_fee,
    };

    let mut executed: Vec<TxEnvelope> = Vec::new();
    let mut gas_left = crate::gas::block_gas_limit();
    eprintln!("[produce_block] height={} input_txs={}", height, txs.len());
    for tx in txs {
        let cost = crate::gas::gas_cost_for(&serde_json::to_value(&tx.data).unwrap());
        if cost > gas_left {
            continue;
        }
        let v = Validator { store, base_fee };
        match v.validate_transaction(tx) {
            Err(e) => eprintln!("[produce] VALIDATE FAIL tx={} err={e}", tx.tx_id),
            Ok(()) => match execute_transaction(store, tx, &ctx) {
                Ok(()) => {
                    executed.push(tx.clone());
                    gas_left -= cost;
                }
                Err(e) => eprintln!("[produce] EXEC REVERT tx={} err={e}", tx.tx_id),
            },
        }
    }

    match_pending_tasks(store, height);

    // Roots over observable state (Rust-defined; see compute_state_root note).
    let state_root = compute_state_root(store);
    let ids: Vec<String> = executed.iter().map(|t| t.tx_id.clone()).collect();
    let tx_root = merkle_root(&ids);
    let pending_ids: Vec<String> = store
        .iter_tasks_by_status("pending")
        .iter()
        .map(|t| t.task_id.clone())
        .collect();
    let inf_root = merkle_root(&pending_ids);
    let node_ids: Vec<String> = store
        .iter_nodes_by_reputation()
        .iter()
        .map(|n| n.node_id.clone())
        .collect();
    let comp_root = merkle_root(&node_ids);

    let mut header = BlockHeader {
        hash: String::new(),
        version: HEADER_VERSION,
        index: height,
        timestamp: now_ms,
        previous_hash: prev.header.hash.clone(),
        validator: producer.address.clone(),
        validator_pub_key: producer.public_key_hex.clone(),
        validator_signature: String::new(),
        state_root,
        tx_root,
        inference_tasks_root: inf_root,
        compute_root: comp_root,
    };
    header.hash = hash_header(&header);
    use ed25519_dalek::Signer;
    let sig = producer.signing.sign(header.hash.as_bytes());
    header.validator_signature = hex::encode(sig.to_bytes());

    store.put_block_by_hash(
        &header.hash,
        &serde_json::to_value(&Block {
            header: header.clone(),
            transactions: executed.clone(),
        })
        .expect("block serialize"),
    );

    let next_base_fee = crate::gas::compute_base_fee(
        base_fee,
        executed
            .iter()
            .map(|t| crate::gas::gas_cost_for(&serde_json::to_value(&t.data).unwrap()))
            .sum(),
    );
    store.set_meta_u64("baseFee", next_base_fee);

    Ok(Block {
        header,
        transactions: executed,
    })
}

/// Current head pointer stored in meta.
pub fn current_head(store: &StateStore) -> Option<Block> {
    let h = store.get_meta("head")?;
    serde_json::from_value(h["block"].clone()).ok()
}

pub fn head_height(store: &StateStore) -> u64 {
    store
        .get_meta("head")
        .and_then(|h| h["height"].as_u64())
        .unwrap_or(0)
}

/// Persist a block as the new canonical head.
pub fn set_head(store: &StateStore, block: &Block) {
    store.set_meta(
        "head",
        &serde_json::json!({
            "height": block.header.index,
            "hash": block.header.hash,
            "block": serde_json::to_value(block).unwrap(),
        }),
    );
}
