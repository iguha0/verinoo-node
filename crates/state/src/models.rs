//! State models — serde types persisted in RocksDB.
//! Field-for-field port of the reference client's state entities.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Account {
    pub address: String,
    #[serde(rename = "publicKey")]
    pub public_key: String,
    pub nonce: u64,
    pub balance: i64,
    #[serde(rename = "updatedAt")]
    pub updated_at: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRegistryEntry {
    #[serde(rename = "modelId")]
    pub model_id: String,
    pub architecture: String,
    #[serde(rename = "parameterCount")]
    pub parameter_count: u64,
    #[serde(rename = "weightsHash")]
    pub weights_hash: String,
    #[serde(rename = "runtimeHash")]
    pub runtime_hash: String,
    #[serde(rename = "stakingRequirement")]
    pub staking_requirement: u64,
    pub description: String,
    pub owner: String,
    #[serde(rename = "registeredAt")]
    pub registered_at: u64,
    #[serde(rename = "isActive")]
    pub is_active: bool,
}

/// Inference task — status lifecycle mirrors the reference engine:
/// pending → assigned → completed | challenged | failed
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceTask {
    #[serde(rename = "taskId")]
    pub task_id: String,
    pub requester: String,
    #[serde(rename = "targetModel")]
    pub target_model: String,
    #[serde(rename = "inputCommitment", default)]
    pub input_commitment: String,
    /// Clear-text input. Committed by hash in `input_commitment`; carried so
    /// every node can re-execute the task deterministically at block apply.
    /// Empty for encrypted tasks (the `encrypted_input` payload is used
    /// instead). Public API responses must redact both.
    #[serde(default)]
    pub input: String,
    /// JSON-encoded `verinoo_crypto::EncryptedInput`. Only the network key can
    /// decrypt it; blocks and API responses carry ciphertext, not plaintext.
    #[serde(
        rename = "encryptedInput",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub encrypted_input: Option<String>,
    #[serde(rename = "maxFee")]
    pub max_fee: u64,
    pub deadline: u64,
    #[serde(rename = "verificationType", default = "default_verification_type")]
    pub verification_type: String,
    pub status: String,
    #[serde(rename = "assignedTo", default)]
    pub assigned_to: Option<String>,
    #[serde(default)]
    pub collateral: u64,
    #[serde(rename = "resultHash", default)]
    pub result_hash: Option<String>,
    #[serde(rename = "resultOutput", default)]
    pub result_output: Option<String>,
    #[serde(rename = "proofData", default)]
    pub proof_data: Option<String>,
    #[serde(rename = "challengeWindowEnd", default)]
    pub challenge_window_end: Option<u64>,
    #[serde(rename = "gameId", default)]
    pub game_id: Option<String>,
    /// Block height at which a node produced the result.
    #[serde(rename = "executedAt", default)]
    pub executed_at: u64,
}

fn default_verification_type() -> String {
    "optimistic".into()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputeNode {
    #[serde(rename = "nodeId")]
    pub node_id: String,
    #[serde(rename = "publicKey")]
    pub public_key: String,
    #[serde(rename = "stakedAmount")]
    pub staked_amount: u64,
    #[serde(rename = "availableCapacity")]
    pub available_capacity: u64,
    #[serde(rename = "maxCapacity")]
    pub max_capacity: u64,
    #[serde(rename = "activeTasks", default)]
    pub active_tasks: u64,
    pub reputation: i64,
    #[serde(rename = "successfulInferences", default)]
    pub successful_inferences: u64,
    #[serde(rename = "failedInferences", default)]
    pub failed_inferences: u64,
    #[serde(rename = "supportedModels")]
    pub supported_models: Vec<String>,
    #[serde(rename = "registeredAt", default)]
    pub registered_at: u64,
    #[serde(rename = "lastHeartbeatAt", default)]
    pub last_heartbeat_at: u64,
    #[serde(rename = "isActive", default = "bool_true")]
    pub is_active: bool,
    #[serde(rename = "totalSlashed", default)]
    pub total_slashed: u64,
}

fn bool_true() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentAccount {
    pub address: String,
    #[serde(rename = "publicKey")]
    pub public_key: String,
    pub nonce: u64,
    pub balance: i64,
    pub reputation: i64,
    #[serde(rename = "totalInferences", default)]
    pub total_inferences: u64,
    #[serde(rename = "totalFeesPaid", default)]
    pub total_fees_paid: u64,
    #[serde(rename = "isAutonomous", default = "bool_true")]
    pub is_autonomous: bool,
    #[serde(rename = "servicesProvided", default)]
    pub services_provided: serde_json::Value,
    #[serde(rename = "updatedAt")]
    pub updated_at: u64,
}
