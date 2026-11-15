//! Deterministic readiness decisions and validator-set canonicalisation.
//!
//! The rule is a pure function of an injected head age so tests do not depend on
//! wall-clock timing. `/ready` and `/status` share the same canonical set hash.

use sha2::{Digest, Sha256};

/// A head older than this multiple of the configured block time is stale.
pub const STALE_BLOCK_MULTIPLIER: u64 = 3;

#[derive(Debug, Clone, Copy)]
pub struct ReadinessInput {
    pub serving: bool,
    pub syncing: bool,
    pub height: u64,
    pub head_age_ms: u64,
    pub block_time_ms: u64,
    pub best_peer_height: Option<u64>,
    pub peer_lag_blocks: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Readiness {
    pub ready: bool,
    pub reason: Option<&'static str>,
}

impl Readiness {
    fn not_ready(reason: &'static str) -> Self {
        Self {
            ready: false,
            reason: Some(reason),
        }
    }
}

/// Apply the readiness rule with an already-measured head age.
///
/// A node is ready when it has a head, is not applying catch-up blocks, the
/// head is younger than three block intervals, and (when a peer height is
/// known) it is within the configured lag tolerance of the best peer.
pub fn evaluate(input: ReadinessInput) -> Readiness {
    if !input.serving {
        return Readiness::not_ready("no_head");
    }
    if input.syncing {
        return Readiness::not_ready("syncing");
    }
    let stale_after_ms = input.block_time_ms.saturating_mul(STALE_BLOCK_MULTIPLIER);
    if input.head_age_ms >= stale_after_ms {
        return Readiness::not_ready("stale_head");
    }
    if let Some(peer_height) = input.best_peer_height {
        if input.height.saturating_add(input.peer_lag_blocks) < peer_height {
            return Readiness::not_ready("behind_peer");
        }
    }
    Readiness {
        ready: true,
        reason: None,
    }
}

/// Canonical validator set: addresses sorted and deduplicated.
pub fn canonical_validator_set(addresses: &[String]) -> Vec<String> {
    let mut set = addresses.to_vec();
    set.sort();
    set.dedup();
    set
}

/// Lowercase hex SHA-256 over the canonical set joined with `\n`.
///
/// The digest covers addresses only, not public keys, stake, or registration
/// height, so nodes with the same active addresses agree even if their local
/// `val:` records differ. An empty set hashes the empty string.
pub fn validator_set_hash(addresses: &[String]) -> String {
    let canonical = canonical_validator_set(addresses);
    let mut hasher = Sha256::new();
    hasher.update(canonical.join("\n").as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK_TIME_MS: u64 = 10_000;

    fn input(serving: bool, syncing: bool, head_age_ms: u64) -> ReadinessInput {
        ReadinessInput {
            serving,
            syncing,
            height: 100,
            head_age_ms,
            block_time_ms: BLOCK_TIME_MS,
            best_peer_height: None,
            peer_lag_blocks: 2,
        }
    }

    #[test]
    fn fresh_head_is_ready() {
        let decision = evaluate(input(true, false, 3 * BLOCK_TIME_MS - 1));
        assert!(decision.ready);
        assert_eq!(decision.reason, None);
    }

    #[test]
    fn stale_head_is_not_ready() {
        let decision = evaluate(input(true, false, 3 * BLOCK_TIME_MS));
        assert!(!decision.ready);
        assert_eq!(decision.reason, Some("stale_head"));
    }

    #[test]
    fn syncing_is_not_ready_even_with_fresh_head() {
        let decision = evaluate(input(true, true, 0));
        assert!(!decision.ready);
        assert_eq!(decision.reason, Some("syncing"));
    }

    #[test]
    fn missing_head_is_not_ready() {
        let decision = evaluate(input(false, false, 0));
        assert!(!decision.ready);
        assert_eq!(decision.reason, Some("no_head"));
    }

    #[test]
    fn lag_beyond_peer_tolerance_is_not_ready() {
        let mut decision_input = input(true, false, 0);
        decision_input.best_peer_height = Some(103);
        let decision = evaluate(decision_input);
        assert!(!decision.ready);
        assert_eq!(decision.reason, Some("behind_peer"));
    }

    #[test]
    fn lag_within_peer_tolerance_is_ready() {
        let mut decision_input = input(true, false, 0);
        decision_input.best_peer_height = Some(102);
        assert!(evaluate(decision_input).ready);
    }

    #[test]
    fn validator_set_hash_is_order_and_duplicate_insensitive() {
        let with_duplicates = vec!["ai_b".to_string(), "ai_a".to_string(), "ai_b".to_string()];
        let canonical = vec!["ai_a".to_string(), "ai_b".to_string()];
        assert_eq!(
            validator_set_hash(&with_duplicates),
            validator_set_hash(&canonical)
        );
        assert_eq!(
            validator_set_hash(&canonical),
            "51601458f4bce4547570cc8bb3dc800150d0da61e558100097b69fd07c32b70a"
        );
    }

    #[test]
    fn empty_validator_set_hashes_the_empty_string() {
        assert_eq!(
            validator_set_hash(&[]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
