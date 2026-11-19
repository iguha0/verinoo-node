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

/// `bft-v2` round election: the leader of `(height, round)` over the active
/// set sorted by address — the same formula the consensus crate applies.
/// [`RoundLeader(0)`] agrees with [`RoundRobin`] on every height, so
/// round-0 proposals pass both gates.
///
/// The engine's block gate cannot see consensus rounds, so the node passes
/// this policy with the round from the proposal envelope or commit
/// certificate (`validate_block_with_policy`).
#[derive(Debug, Clone, Copy)]
pub struct RoundLeader(pub u64);

impl LeaderPolicy for RoundLeader {
    fn leader_at(&self, height: u64, validators: &[String]) -> Option<String> {
        if validators.is_empty() || height == 0 {
            return None;
        }
        let mut sorted = validators.to_vec();
        sorted.sort();
        sorted.dedup();
        let idx = ((height - 1).saturating_add(self.0) as usize) % sorted.len();
        Some(sorted[idx].clone())
    }
}

/// Stake-weighted leader selection: validators with more stake produce
/// proportionally more blocks. Uses cumulative-weight sampling with a
/// deterministic seed derived from the height.
#[derive(Debug, Clone, Copy, Default)]
pub struct StakeWeighted;

impl LeaderPolicy for StakeWeighted {
    fn leader_at(&self, height: u64, validators: &[String]) -> Option<String> {
        // This trait method doesn't have access to stake amounts.
        // Use `leader_at_weighted` instead when stakes are available.
        // Fallback to round-robin for compatibility.
        RoundRobin.leader_at(height, validators)
    }
}

/// Stake-weighted leader selection with explicit stake data.
pub fn stake_weighted_leader(
    height: u64,
    validators: &[(String, u64)], // (address, stake)
) -> Option<String> {
    if validators.is_empty() || height == 0 {
        return None;
    }

    let mut sorted = validators.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));

    let total_stake: u64 = sorted.iter().map(|(_, s)| s).sum();
    if total_stake == 0 {
        return Some(sorted[0].0.clone());
    }

    // Deterministic pseudo-random sample point based on height.
    let seed = height.wrapping_mul(2654435761); // Knuth's multiplicative hash
    let target = seed % total_stake;

    let mut cumulative = 0u64;
    for (addr, stake) in &sorted {
        cumulative += stake;
        if target < cumulative {
            return Some(addr.clone());
        }
    }
    sorted.last().map(|(a, _)| a.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addresses(n: u64) -> Vec<String> {
        (0..n).map(|i| format!("ai_{i:02}")).collect()
    }

    #[test]
    fn round_zero_agrees_with_round_robin() {
        let validators = vec![addresses(4)[3].clone(), addresses(4)[0].clone()];
        for height in 1..=6 {
            assert_eq!(
                RoundLeader(0).leader_at(height, &validators),
                RoundRobin.leader_at(height, &validators)
            );
        }
    }

    #[test]
    fn rounds_rotate_one_validator_per_round() {
        let validators = addresses(4);
        for height in 1..=3 {
            for round in 0..4u64 {
                let expected = &validators[((height - 1 + round) % 4) as usize];
                assert_eq!(
                    RoundLeader(round).leader_at(height, &validators).as_deref(),
                    Some(expected.as_str())
                );
            }
        }
        // Rounds past the set size wrap around.
        assert_eq!(
            RoundLeader(4).leader_at(1, &validators),
            RoundLeader(0).leader_at(1, &validators)
        );
    }
}
