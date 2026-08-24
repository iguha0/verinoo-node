//! Gas market — port of reference `src/core/gas.ts` (testnet-scaled table).

use serde_json::Value;
use verinoo_state::genesis::BLOCK_GAS_LIMIT;

pub fn gas_cost_for(data: &Value) -> u64 {
    let kind = data["type"].as_str().unwrap_or("");
    match kind {
        "registerNode" => 50,
        "registerModel" => 80,
        "registerAgent" => 50,
        "submitInference" => 60,
        "submitResult" => 40,
        "challengeResult" => 120,
        "bisect" => 100,
        "proveStep" => 80,
        "agentPayment" => 50,
        _ => 50, // transfer + unknown default
    }
}

/// Verification-policy multiplier for submitInference tasks.
pub fn policy_multiplier(policy: Option<&str>) -> Result<u64, String> {
    match policy {
        None | Some("optimistic") => Ok(1),
        Some("sampled") => Ok(2),
        Some("full-zk") => Ok(4),
        Some(other) => Err(format!("invalid verificationType: {other}")),
    }
}

/// The policy a transaction carries, if any (submitInference only).
pub fn policy_of_tx(data: &Value) -> Option<&str> {
    if data["type"].as_str() != Some("submitInference") {
        return None;
    }
    data["data"]["verificationType"].as_str()
}

pub fn block_gas_limit() -> u64 {
    BLOCK_GAS_LIMIT
}

/// EIP-1559-style baseFee adjustment. Port of computeBaseFee.
pub fn compute_base_fee(prev_base_fee: u64, gas_used: u64) -> u64 {
    const TARGET: i128 = 500; // BLOCK_GAS_TARGET
    const MAX_CHANGE_BPS: i128 = 1250; // ±12.5%

    let prev = prev_base_fee as i128;
    let used = gas_used as i128;
    let change = ((used - TARGET) * prev) / TARGET / 8; // BASE_FEE_DENOM
    let rate_bps = 10_000 + change.clamp(-MAX_CHANGE_BPS, MAX_CHANGE_BPS);
    let next = (prev * rate_bps) / 10_000;
    next.max(1) as u64
}
