//! Leader election — deterministic producer selection per height.
//!
//! Phase R3 ships [`RoundRobin`]. The trait boundary exists so a BFT
//! implementation (Malachite adapter) can replace the policy without
//! touching the engine or networking layers. ADR: docs/adr/001-leader-policy.md

pub trait LeaderPolicy {
    /// Which validator address must produce the block at `height`?
    fn leader_at(&self, height: u64, validators: &[String]) -> Option<String>;
}

/// Round-robin over the ACTIVE validator set sorted by address.
/// Deterministic across nodes by construction.
#[derive(Debug, Clone, Copy, Default)]
pub struct RoundRobin;

impl LeaderPolicy for RoundRobin {
    fn leader_at(&self, height: u64, validators: &[String]) -> Option<String> {
        if validators.is_empty() {
            return None;
        }
        let mut sorted = validators.to_vec();
        sorted.sort();
        // Height 1 (first real block after genesis) -> sorted[0].
        if height == 0 {
            return None; // genesis has no producer
        }
        let idx = ((height - 1) % sorted.len() as u64) as usize;
        Some(sorted[idx].clone())
    }
}
