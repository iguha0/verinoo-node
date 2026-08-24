//! Transaction model — strict enums for every op the engine understands.
//! Ported from the TypeScript reference client `src/core/types.ts` +
//! `src/core/engine.ts` dispatch.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TxKind {
    Transfer,
    RegisterModel,
    RegisterNode,
    RegisterAgent,
    SubmitInference,
    ClaimTask,
    SubmitResult,
    ChallengeResult,
    Bisect,
    ProveStep,
    AgentPayment,
    RegisterValidator,
    UnregisterValidator,
}

/// The `data` field of a transaction: `{ type: TxKind, data: <op-specific> }`.
/// Op-specific payloads are typed in later phases (R1 execution core); at the
/// types layer they remain canonical JSON values validated per-kind.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TxData {
    pub r#type: TxKind,
    pub data: serde_json::Value,
}

/// Full signed transaction envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TxEnvelope {
    pub tx_id: String,
    pub from: String,
    pub to: String,
    pub value: u64,
    pub nonce: u64,
    pub data: TxData,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_limit: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_price: Option<u64>,
    pub public_key: String,
    pub signature: String,
}
