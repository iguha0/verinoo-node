//! Validator set state — staked, open-entry (testnet policy).
//!
//! Storage: `val:<address>` → ValidatorEntry. Leader election iterates
//! ACTIVE entries sorted by address for determinism.

use serde::{Deserialize, Serialize};

pub const MIN_VALIDATOR_STAKE: u64 = 1_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidatorEntry {
    pub address: String,
    #[serde(rename = "publicKey")]
    pub public_key: String,
    /// Bonded amount — locked at register, refunded at unregister.
    #[serde(rename = "stakedAmount")]
    pub staked_amount: u64,
    #[serde(rename = "registeredAtHeight")]
    pub registered_at_height: u64,
    /// Genesis-seeded validators are exempt from the stake minimum.
    pub genesis: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidatorSetInfo {
    pub validators: Vec<ValidatorSetInfoEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidatorSetInfoEntry {
    pub address: String,
    #[serde(rename = "stakedAmount")]
    pub staked_amount: u64,
    pub genesis: bool,
}
