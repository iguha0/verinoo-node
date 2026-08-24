//! Genesis / chain parameters for verinoo-test-1.

use serde::{Deserialize, Serialize};

pub const CHAIN_ID: &str = "verinoo-test-1";
pub const BLOCK_TIME_MS: u64 = 10_000;
pub const BLOCK_GAS_TARGET: u64 = 500;
pub const BLOCK_GAS_LIMIT: u64 = 1_000;
pub const INITIAL_BASE_FEE: u64 = 1;
/// Max +/- change of baseFee per block (EIP-1559 elasticity denominator).
pub const BASE_FEE_DENOM: u64 = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisParams {
    pub chain_id: String,
    pub block_time_ms: u64,
    pub initial_base_fee: u64,
    pub block_gas_limit: u64,
    /// Addresses allowed to produce blocks until Phase R3 opens staking.
    pub validators: Vec<String>,
}

impl Default for GenesisParams {
    fn default() -> Self {
        Self {
            chain_id: CHAIN_ID.into(),
            block_time_ms: BLOCK_TIME_MS,
            initial_base_fee: INITIAL_BASE_FEE,
            block_gas_limit: BLOCK_GAS_LIMIT,
            validators: vec![],
        }
    }
}
