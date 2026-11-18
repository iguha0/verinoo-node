//! Deterministic round/timeout schedule for view changes.
//!
//! The pacemaker owns only `(height, round)` and the timeout policy. It has no
//! wall clock, no sockets, and no background tasks: the node arms a timer from
//! [`Schedule`] and calls [`Pacemaker::on_timeout`] when the timer fires.
//! Higher layers turn the resulting schedule into an [`Action::StartRound`]
//! (see `state.rs`).
//!
//! Backoff is `base * 1.5^round` with integer arithmetic, capped at
//! `max_timeout_ms`. `R_MAX` is the largest multiplier exponent that is
//! applied: rounds above it keep the timeout from `R_MAX` but still rotate
//! leaders and keep retrying liveness. This matches ADR 004 "Timeouts".

/// Number of rounds after which the backoff exponent stops growing.
pub const R_MAX: u64 = 8;

/// Backwards-compatible alias for [`R_MAX`].
pub const DEFAULT_MAX_ROUND: u64 = R_MAX;

/// Default upper bound for a round timeout.
pub const DEFAULT_MAX_TIMEOUT_MS: u64 = 60_000;

/// Default base timeout when a caller does not specify one.
pub const DEFAULT_BASE_TIMEOUT_MS: u64 = 1_000;

const TIMEOUT_NUMERATOR: u64 = 3;
const TIMEOUT_DENOMINATOR: u64 = 2;

/// Timeout policy shared by the pacemaker and the consensus state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacemakerConfig {
    pub base_timeout_ms: u64,
    pub max_timeout_ms: u64,
    pub max_round: u64,
}

impl Default for PacemakerConfig {
    fn default() -> Self {
        Self::new(DEFAULT_BASE_TIMEOUT_MS)
    }
}

impl PacemakerConfig {
    pub fn new(base_timeout_ms: u64) -> Self {
        Self {
            base_timeout_ms: base_timeout_ms.max(1),
            max_timeout_ms: DEFAULT_MAX_TIMEOUT_MS,
            max_round: R_MAX,
        }
    }

    #[must_use]
    pub fn with_timeout_cap(mut self, max_timeout_ms: u64) -> Self {
        self.max_timeout_ms = max_timeout_ms.max(1);
        self
    }

    #[must_use]
    pub fn with_max_round(mut self, max_round: u64) -> Self {
        self.max_round = max_round;
        self
    }
}

/// Timer instruction for the node's timer actor.
///
/// `Timeout` means "arm or replace the timer for this `(height, round)`".
/// `Cancel` means "cancel the timer for this `(height, round)`", for example
/// after a proposal or a commit makes the phase timer irrelevant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    Timeout {
        height: u64,
        round: u64,
        timeout_ms: u64,
    },
    Cancel {
        height: u64,
        round: u64,
    },
}

impl Schedule {
    pub fn timeout_ms(self) -> Option<u64> {
        match self {
            Schedule::Timeout { timeout_ms, .. } => Some(timeout_ms),
            Schedule::Cancel { .. } => None,
        }
    }
}

/// Pure round clock and timeout calculator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pacemaker {
    height: u64,
    round: u64,
    config: PacemakerConfig,
}

impl Pacemaker {
    pub fn new(base_timeout_ms: u64) -> Self {
        Self::with_config(PacemakerConfig::new(base_timeout_ms))
    }

    pub fn with_config(config: PacemakerConfig) -> Self {
        Self {
            height: 1,
            round: 0,
            config,
        }
    }

    pub fn height(&self) -> u64 {
        self.height
    }

    pub fn current_round(&self) -> u64 {
        self.round
    }

    pub fn round(&self) -> u64 {
        self.round
    }

    pub fn max_round(&self) -> u64 {
        self.config.max_round
    }

    pub fn config(&self) -> PacemakerConfig {
        self.config
    }

    /// Timeout for the current round.
    pub fn timeout_ms(&self) -> u64 {
        self.timeout_ms_at(self.round)
    }

    /// Timeout for an explicit round, using `base * 1.5^min(round, R_MAX)`
    /// capped at `max_timeout_ms`.
    pub fn timeout_ms_at(&self, round: u64) -> u64 {
        let exponent = round.min(self.config.max_round);
        let mut out = self.config.base_timeout_ms;
        for _ in 0..exponent {
            let next = out.saturating_mul(TIMEOUT_NUMERATOR) / TIMEOUT_DENOMINATOR;
            if next >= self.config.max_timeout_ms {
                return self.config.max_timeout_ms;
            }
            // Base 1 cannot grow under integer arithmetic; stop early so a
            // large configured `R_MAX` cannot make this loop unbounded.
            if next <= out {
                break;
            }
            out = next;
        }
        out.min(self.config.max_timeout_ms)
    }

    /// Arm (or re-arm) the timer for the current round.
    pub fn schedule(&self) -> Schedule {
        Schedule::Timeout {
            height: self.height,
            round: self.round,
            timeout_ms: self.timeout_ms(),
        }
    }

    /// Cancel the timer for the current round.
    pub fn cancel(&self) -> Schedule {
        Schedule::Cancel {
            height: self.height,
            round: self.round,
        }
    }

    /// Enter an explicit `(height, round)`, resetting the round clock to that
    /// position and returning the timer to arm.
    pub fn start_round(&mut self, height: u64, round: u64) -> Schedule {
        self.height = height;
        self.round = round;
        self.schedule()
    }

    /// Enter `(height, 0)`.
    pub fn start_height(&mut self, height: u64) -> Schedule {
        self.start_round(height, 0)
    }

    /// Advance exactly one round and return the next timer. There is no stop
    /// at `R_MAX`: rounds keep advancing with a capped timeout so a live set
    /// can recover from a long sequence of offline leaders.
    pub fn on_timeout(&mut self) -> Schedule {
        self.round = self.round.saturating_add(1);
        self.schedule()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_is_exponential_until_r_max() {
        let pacemaker = Pacemaker::new(1_000);
        assert_eq!(pacemaker.timeout_ms_at(0), 1_000);
        assert_eq!(pacemaker.timeout_ms_at(1), 1_500);
        assert_eq!(pacemaker.timeout_ms_at(2), 2_250);
        assert_eq!(pacemaker.timeout_ms_at(3), 3_375);
        assert_eq!(pacemaker.timeout_ms_at(4), 5_062);
    }

    #[test]
    fn timeout_caps_at_the_r_max_value_and_rounds_keep_advancing() {
        let mut pacemaker = Pacemaker::new(1_000);
        let capped = pacemaker.timeout_ms_at(R_MAX);
        assert!(capped > pacemaker.timeout_ms_at(R_MAX - 1));
        assert_eq!(pacemaker.timeout_ms_at(R_MAX + 1), capped);
        assert_eq!(pacemaker.timeout_ms_at(R_MAX + 100), capped);

        for _ in 0..(R_MAX + 5) {
            pacemaker.on_timeout();
        }
        assert_eq!(pacemaker.current_round(), R_MAX + 5);
        assert_eq!(pacemaker.timeout_ms(), capped);
        assert_eq!(
            pacemaker.schedule(),
            Schedule::Timeout {
                height: 1,
                round: R_MAX + 5,
                timeout_ms: capped,
            }
        );
    }

    #[test]
    fn explicit_timeout_cap_wins_over_the_formula() {
        let config = PacemakerConfig::new(1_000)
            .with_timeout_cap(5_000)
            .with_max_round(10);
        let pacemaker = Pacemaker::with_config(config);
        assert_eq!(pacemaker.timeout_ms_at(3), 3_375);
        assert_eq!(pacemaker.timeout_ms_at(4), 5_000);
        assert_eq!(pacemaker.timeout_ms_at(10), 5_000);
    }

    #[test]
    fn start_round_and_timeout_produce_schedules_without_a_clock() {
        let mut pacemaker = Pacemaker::new(2_000);
        assert_eq!(
            pacemaker.start_round(9, 2),
            Schedule::Timeout {
                height: 9,
                round: 2,
                timeout_ms: 4_500,
            }
        );
        assert_eq!(
            pacemaker.on_timeout(),
            Schedule::Timeout {
                height: 9,
                round: 3,
                timeout_ms: 6_750,
            }
        );
        assert_eq!(
            pacemaker.cancel(),
            Schedule::Cancel {
                height: 9,
                round: 3,
            }
        );
    }

    #[test]
    fn tiny_base_does_not_loop_forever() {
        let pacemaker = Pacemaker::with_config(
            PacemakerConfig::new(1)
                .with_timeout_cap(u64::MAX)
                .with_max_round(u64::MAX),
        );
        assert_eq!(pacemaker.timeout_ms_at(u64::MAX), 1);
    }
}
