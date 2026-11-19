//! Quorum finality over validated blocks.
//!
//! The block producer is authenticated by the normal block-header signature.
//! Validators sign a prevote for a block after applying and checking its state
//! root, then a precommit once a prevote quorum agrees. The block is final only
//! when a quorum of precommits is collected. This is the certificate layer for
//! the existing single-producer chain; round/view-change liveness is not part
//! of this integration.

use ed25519_dalek::SigningKey;
use std::collections::{BTreeMap, HashMap};
use verinoo_consensus::{
    round_leader, Action, CommitCertificate, ConsensusState, Evidence, Proposal, Schedule, Vote,
};
use verinoo_exec::engine::Block;

/// Wire envelope for `bft-v2` proposal gossip: the signed round proposal plus
/// the block it binds, so a receiver can verify the leader envelope, admit the
/// block, and check the state root on a checkpoint clone before voting.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProposalMessage {
    pub proposal: Proposal,
    pub block: Block,
}

/// Heights above the finalized tip for which inputs may be buffered.
const VOTE_HEIGHT_WINDOW: u64 = 64;
/// Rounds ahead of the current one from which proposals and votes may be
/// buffered until the pacemaker reaches them.
const FUTURE_ROUND_WINDOW: u64 = 32;
/// Distinct rounds of buffered proposals per height.
const PROPOSALS_BUFFERED_PER_HEIGHT: usize = 16;
/// Buffered votes per (height, round).
const VOTES_BUFFERED_PER_ROUND: usize = 128;

/// Cheap pre-filter for proposal gossip: is a proposal worth validating at
/// all? The node checks this before running admission and the checkpoint
/// clone, so garbage costs one signature check at most.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposalRelevance {
    /// The proposal is for the active (height, round): validate and vote.
    Current,
    /// The proposal is for a reachable future round or a not-yet-active
    /// height: validate once, then buffer until the round starts.
    Future,
    /// Wrong chain, wrong leader, stale round, or out of window: drop.
    Irrelevant,
}

#[derive(Default)]
pub struct FinalityOutput {
    pub votes: Vec<Vote>,
    pub finalized: Option<(u64, String, CommitCertificate)>,
    pub evidence: Vec<Evidence>,
    /// Timer instruction for the round the state machine is now in
    /// (`Action::StartRound`): `(height, round, timeout_ms)` for the node's
    /// timer actor to arm.
    pub start_round: Option<Schedule>,
    /// Finality before this output was produced. When the caller cannot
    /// commit the finalized block, it rolls the manager back to this
    /// `(height, hash)` so in-memory finality never runs ahead of the
    /// canonical head.
    pub previous_finalized: Option<(u64, String)>,
}

impl FinalityOutput {
    fn merge(&mut self, other: FinalityOutput) {
        self.votes.extend(other.votes);
        self.evidence.extend(other.evidence);
        if other.finalized.is_some() {
            self.finalized = other.finalized;
        }
        if other.start_round.is_some() {
            self.start_round = other.start_round;
        }
        if self.previous_finalized.is_none() {
            self.previous_finalized = other.previous_finalized;
        }
    }
}

pub struct FinalityManager {
    chain_id: String,
    local: String,
    signing: SigningKey,
    validators: Vec<String>,
    states: BTreeMap<u64, ConsensusState>,
    pending_votes: HashMap<u64, Vec<Vote>>,
    /// Proposals already validated on arrival for rounds this node has not
    /// entered yet (timer skew across peers): height -> round ->
    /// `(proposal, block_valid)`.
    pending_proposals: HashMap<u64, HashMap<u64, (Proposal, bool)>>,
    /// Votes for reachable future rounds, keyed `(height, round)`.
    pending_round_votes: HashMap<(u64, u64), Vec<Vote>>,
    finalized_height: u64,
    finalized_hash: String,
    base_timeout_ms: u64,
}

impl FinalityManager {
    pub fn new(
        chain_id: &str,
        validators: Vec<String>,
        local: &str,
        signing: SigningKey,
        finalized_height: u64,
        finalized_hash: &str,
        base_timeout_ms: u64,
    ) -> Result<Self, String> {
        let mut validators = validators;
        validators.sort();
        validators.dedup();
        if validators.is_empty() {
            return Err("BFT mode requires a non-empty validator set".into());
        }
        if !validators.iter().any(|v| v == local) {
            return Err("BFT mode requires this node to be a member of the validator set".into());
        }
        Ok(Self {
            chain_id: chain_id.to_string(),
            local: local.to_string(),
            signing,
            validators,
            states: BTreeMap::new(),
            pending_votes: HashMap::new(),
            pending_proposals: HashMap::new(),
            pending_round_votes: HashMap::new(),
            finalized_height,
            finalized_hash: finalized_hash.to_string(),
            base_timeout_ms: base_timeout_ms.max(500),
        })
    }

    pub fn finalized_height(&self) -> u64 {
        self.finalized_height
    }
    pub fn finalized_hash(&self) -> &str {
        &self.finalized_hash
    }
    pub fn validators(&self) -> &[String] {
        &self.validators
    }
    pub fn local(&self) -> &str {
        &self.local
    }

    /// Round-leader election over the canonical (sorted) validator set —
    /// the same rule the consensus state applies, usable by the timer actor
    /// before any round state exists.
    pub fn round_leader(&self, height: u64, round: u64) -> Option<String> {
        round_leader(&self.validators, height, round).map(str::to_owned)
    }

    /// Adopt finalized history learned from a verified peer certificate.
    ///
    /// A late node must not sign votes for heights that are already final.
    /// Dropping state and buffered votes at or below the adopted height makes
    /// the next vote start at the current head instead of replaying history.
    pub fn adopt_finalized(&mut self, height: u64, block_hash: &str) {
        if height <= self.finalized_height {
            return;
        }
        self.finalized_height = height;
        self.finalized_hash = block_hash.to_string();
        self.states.retain(|h, _| *h > height);
        self.pending_votes.retain(|h, _| *h > height);
        self.pending_proposals.retain(|h, _| *h > height);
        self.pending_round_votes.retain(|(h, _), _| *h > height);
    }

    /// Undo an in-memory finalization whose block could not be committed, so
    /// the manager never reports a finalized height the canonical head does
    /// not have. `to_height`/`to_hash` are the pre-finalization values (see
    /// [`FinalityOutput::previous_finalized`]).
    pub fn rollback_finalization(&mut self, to_height: u64, to_hash: &str) {
        if to_height >= self.finalized_height {
            return;
        }
        self.finalized_height = to_height;
        self.finalized_hash = to_hash.to_string();
        self.states.retain(|h, _| *h <= to_height);
        self.pending_votes.retain(|h, _| *h <= to_height);
        self.pending_proposals.retain(|h, _| *h <= to_height);
        self.pending_round_votes.retain(|(h, _), _| *h <= to_height);
        tracing::warn!(
            height = to_height,
            "finality rolled back to the last committable height"
        );
    }

    /// Activate `height` for consensus: create its round state at round 0 if
    /// missing (idempotent), replay anything buffered for round 0, and report
    /// the current round's timer schedule via `start_round`. The timer actor
    /// re-arms only when `(height, round)` changes.
    pub fn start_height(&mut self, height: u64) -> FinalityOutput {
        let mut output = FinalityOutput::default();
        if height <= self.finalized_height {
            return output;
        }
        if !self.states.contains_key(&height) {
            let mut state = match ConsensusState::new(
                &self.chain_id,
                self.validators.clone(),
                &self.local,
                height,
                self.base_timeout_ms,
            ) {
                Ok(state) => state,
                Err(e) => {
                    tracing::error!("BFT state creation failed: {e}");
                    return output;
                }
            };
            state.restore_finalized(self.finalized_height, &self.finalized_hash);
            self.states.insert(height, state);
            tracing::debug!(height, "bft round state activated");
            output.merge(self.replay_round_inputs(height, 0));
            if let Some(votes) = self.pending_votes.remove(&height) {
                for vote in votes {
                    output.merge(self.on_vote_at(height, &vote));
                }
            }
        }
        if let Some(state) = self.states.get(&height) {
            output.start_round = Some(state.schedule());
        }
        output
    }

    /// Fire the round timer for `(height, round)`. A stale timer (the round
    /// already advanced or finalized) is ignored. Otherwise the timed-out
    /// round is nil-prevoted, the pacemaker advances exactly one round, and
    /// inputs buffered for the new round (a leader proposal or votes that
    /// arrived during timer skew) are replayed. `start_round` carries the
    /// new schedule; the next round leader is
    /// `round_leader(height, start_round.round)`.
    pub fn on_timeout(&mut self, height: u64, round: u64) -> FinalityOutput {
        let mut output = FinalityOutput::default();
        let Some(state) = self.states.get_mut(&height) else {
            return output;
        };
        if state.current_round() != round {
            return output;
        }
        let actions = state.on_timeout();
        output.merge(self.drive(height, actions));
        let Some(Schedule::Timeout {
            round: new_round, ..
        }) = output.start_round
        else {
            return output;
        };
        output.merge(self.replay_round_inputs(height, new_round));
        output
    }

    /// Feed a gossiped proposal whose block was validated (or not) by the
    /// node. A proposal for the active round is voted on now; one for a
    /// reachable future round is buffered with its validation verdict until
    /// the pacemaker gets there; anything else is dropped.
    pub fn on_proposal(&mut self, proposal: &Proposal, block_valid: bool) -> FinalityOutput {
        if proposal.height <= self.finalized_height
            || proposal.height > self.finalized_height.saturating_add(VOTE_HEIGHT_WINDOW)
        {
            return FinalityOutput::default();
        }
        // The height is not active yet: only small rounds are worth
        // buffering until the round state exists.
        let Some(current) = self
            .states
            .get(&proposal.height)
            .map(|state| state.current_round())
        else {
            if proposal.round <= FUTURE_ROUND_WINDOW {
                self.buffer_proposal(proposal, block_valid);
            }
            return FinalityOutput::default();
        };
        if proposal.round == current {
            let actions = self
                .states
                .get_mut(&proposal.height)
                .expect("the round state was just read")
                .on_proposal(proposal, block_valid);
            return self.drive(proposal.height, actions);
        }
        if proposal.round > current && proposal.round <= current.saturating_add(FUTURE_ROUND_WINDOW)
        {
            self.buffer_proposal(proposal, block_valid);
        }
        // Stale rounds and rounds beyond the window are ignored.
        FinalityOutput::default()
    }

    fn buffer_proposal(&mut self, proposal: &Proposal, block_valid: bool) {
        let rounds = self.pending_proposals.entry(proposal.height).or_default();
        if rounds.len() >= PROPOSALS_BUFFERED_PER_HEIGHT {
            tracing::debug!(
                height = proposal.height,
                "proposal buffer full — dropping proposal"
            );
            return;
        }
        // First proposal for a round wins; later equivocations are dropped.
        rounds
            .entry(proposal.round)
            .or_insert_with(|| (proposal.clone(), block_valid));
    }

    /// Cheap pre-filter for the node's proposal handler. Only run admission
    /// and the checkpoint clone for proposals that can matter.
    pub fn classify_proposal(&self, proposal: &Proposal) -> ProposalRelevance {
        use ProposalRelevance::*;
        if proposal.chain_id != self.chain_id
            || proposal.height <= self.finalized_height
            || proposal.height > self.finalized_height.saturating_add(VOTE_HEIGHT_WINDOW)
        {
            return Irrelevant;
        }
        if self
            .round_leader(proposal.height, proposal.round)
            .as_deref()
            != Some(proposal.proposer.as_str())
        {
            return Irrelevant;
        }
        match self.states.get(&proposal.height) {
            Some(state) => {
                let current = state.current_round();
                if proposal.round == current {
                    Current
                } else if proposal.round > current
                    && proposal.round <= current.saturating_add(FUTURE_ROUND_WINDOW)
                {
                    Future
                } else {
                    Irrelevant
                }
            }
            // The height is not active yet; only small rounds are worth
            // validating and buffering.
            None => {
                if proposal.round <= FUTURE_ROUND_WINDOW {
                    Future
                } else {
                    Irrelevant
                }
            }
        }
    }

    /// Feed the buffered proposal and votes for `round` after a round
    /// change or a fresh height activation.
    fn replay_round_inputs(&mut self, height: u64, round: u64) -> FinalityOutput {
        let mut output = FinalityOutput::default();
        let replayed = self
            .pending_proposals
            .get_mut(&height)
            .and_then(|rounds| rounds.remove(&round));
        if let Some((proposal, block_valid)) = replayed {
            let actions = self
                .states
                .get_mut(&height)
                .map(|state| state.on_proposal(&proposal, block_valid))
                .unwrap_or_default();
            output.merge(self.drive(height, actions));
        }
        if let Some(votes) = self.pending_round_votes.remove(&(height, round)) {
            for vote in votes {
                output.merge(self.on_vote_at(height, &vote));
            }
        }
        output
    }

    /// Restore a persisted own vote after a restart so the node rebroadcasts
    /// it verbatim instead of signing a second vote. Creates the height's
    /// round state if needed and refuses conflicting restores.
    pub fn restore_own_vote(&mut self, vote: Vote) -> Result<(), String> {
        if vote.validator != self.local {
            return Err("restored vote is not from the local validator".into());
        }
        if vote.height <= self.finalized_height {
            return Err("restored vote is at or below the finalized height".into());
        }
        if vote.height > self.finalized_height.saturating_add(VOTE_HEIGHT_WINDOW) {
            return Err("restored vote is outside the height window".into());
        }
        vote.verify_with_chain(&self.chain_id, &self.validators)
            .map_err(|e| e.to_string())?;
        if !self.states.contains_key(&vote.height) {
            let mut state = ConsensusState::new(
                &self.chain_id,
                self.validators.clone(),
                &self.local,
                vote.height,
                self.base_timeout_ms,
            )?;
            state.restore_finalized(self.finalized_height, &self.finalized_hash);
            self.states.insert(vote.height, state);
        }
        self.states
            .get_mut(&vote.height)
            .expect("round state was just inserted")
            .restore_own_vote(vote)
    }

    /// Reload every persisted own vote (`consensusOwnVote:*`) after a
    /// restart. Historical or malformed entries are skipped with a debug
    /// log; the returned count is for the startup log.
    pub fn restore_own_votes_from_store(&mut self, store: &verinoo_state::StateStore) -> usize {
        let votes: Vec<Vote> = store
            .iter_prefix::<serde_json::Value>("meta:consensusOwnVote:")
            .into_iter()
            .filter_map(|value| serde_json::from_value(value).ok())
            .collect();
        let mut restored = 0usize;
        for vote in votes {
            match self.restore_own_vote(vote) {
                Ok(()) => restored += 1,
                Err(e) => tracing::debug!("own vote restore skipped: {e}"),
            }
        }
        restored
    }

    pub fn observe_block(&mut self, height: u64, block_hash: &str) -> FinalityOutput {
        if height <= self.finalized_height || block_hash.is_empty() {
            return FinalityOutput::default();
        }
        if !self.states.contains_key(&height) {
            match ConsensusState::new(
                &self.chain_id,
                self.validators.clone(),
                &self.local,
                height,
                self.base_timeout_ms,
            ) {
                Ok(mut state) => {
                    state.restore_finalized(self.finalized_height, &self.finalized_hash);
                    self.states.insert(height, state);
                }
                Err(e) => {
                    tracing::error!("BFT state creation failed: {e}");
                    return FinalityOutput::default();
                }
            }
        }
        let actions = self
            .states
            .get_mut(&height)
            .map(|state| state.on_validated_block(block_hash))
            .unwrap_or_default();
        let mut output = self.drive(height, actions);
        if let Some(votes) = self.pending_votes.remove(&height) {
            for vote in votes {
                output.merge(self.on_vote_at(height, &vote));
            }
        }
        output
    }

    pub fn on_vote(&mut self, vote: &Vote) -> FinalityOutput {
        if vote.height <= self.finalized_height {
            return FinalityOutput::default();
        }
        if !self.states.contains_key(&vote.height) {
            if vote.height > self.finalized_height.saturating_add(VOTE_HEIGHT_WINDOW) {
                return FinalityOutput::default();
            }
            let pending = self.pending_votes.entry(vote.height).or_default();
            if pending.len() < VOTES_BUFFERED_PER_ROUND
                && !pending.iter().any(|existing| {
                    existing.round == vote.round
                        && existing.kind == vote.kind
                        && existing.validator == vote.validator
                })
            {
                pending.push(vote.clone());
            }
            return FinalityOutput::default();
        }
        // A vote for a reachable future round (peers whose timer fired first)
        // is buffered and replayed when the pacemaker enters that round.
        if let Some(state) = self.states.get(&vote.height) {
            let current = state.current_round();
            if vote.round > current && vote.round <= current.saturating_add(FUTURE_ROUND_WINDOW) {
                let key = (vote.height, vote.round);
                let pending = self.pending_round_votes.entry(key).or_default();
                if pending.len() < VOTES_BUFFERED_PER_ROUND
                    && !pending.iter().any(|existing| {
                        existing.kind == vote.kind && existing.validator == vote.validator
                    })
                {
                    pending.push(vote.clone());
                }
                return FinalityOutput::default();
            }
        }
        self.on_vote_at(vote.height, vote)
    }

    fn on_vote_at(&mut self, height: u64, vote: &Vote) -> FinalityOutput {
        let actions = self
            .states
            .get_mut(&height)
            .map(|state| state.on_vote(vote))
            .unwrap_or_default();
        self.drive(height, actions)
    }

    fn drive(&mut self, height: u64, initial: Vec<Action>) -> FinalityOutput {
        use std::collections::VecDeque;
        let mut queue: VecDeque<Action> = initial.into();
        let mut output = FinalityOutput {
            previous_finalized: Some((self.finalized_height, self.finalized_hash.clone())),
            ..FinalityOutput::default()
        };
        let chain_id = self.chain_id.clone();
        let local = self.local.clone();
        let signing = self.signing.clone();
        let mut guard = 0usize;
        while let Some(action) = queue.pop_front() {
            guard += 1;
            if guard > 128 {
                tracing::warn!("BFT action loop exceeded; dropping remaining actions");
                break;
            }
            match action {
                Action::Vote(intent) => {
                    let vote = Vote::sign(
                        &chain_id,
                        intent.kind,
                        intent.height,
                        intent.round,
                        &intent.block_hash,
                        &local,
                        &signing,
                    );
                    if let Some(state) = self.states.get_mut(&height) {
                        queue.extend(state.on_vote(&vote));
                    }
                    output.votes.push(vote);
                }
                Action::Rebroadcast(vote) => output.votes.push(vote),
                Action::StartRound {
                    height,
                    round,
                    timeout_ms,
                } => {
                    output.start_round = Some(Schedule::Timeout {
                        height,
                        round,
                        timeout_ms,
                    });
                }
                Action::Evidence(evidence) => output.evidence.push(*evidence),
                Action::Finalized {
                    height: finalized_height,
                    block_hash,
                    certificate,
                } => {
                    if finalized_height > self.finalized_height {
                        self.finalized_height = finalized_height;
                        self.finalized_hash = block_hash.clone();
                        self.states.retain(|h, _| *h > finalized_height);
                        self.pending_votes.retain(|h, _| *h > finalized_height);
                        self.pending_proposals.retain(|h, _| *h > finalized_height);
                        self.pending_round_votes
                            .retain(|(h, _), _| *h > finalized_height);
                    }
                    output.finalized = Some((finalized_height, block_hash, certificate));
                }
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validators(n: u8) -> (Vec<SigningKey>, Vec<String>) {
        let keys: Vec<SigningKey> = (0..n)
            .map(|i| SigningKey::from_bytes(&[i.saturating_add(1); 32]))
            .collect();
        let addresses = keys
            .iter()
            .map(|key| {
                verinoo_consensus::public_key_to_address(&hex::encode(
                    key.verifying_key().to_bytes(),
                ))
                .unwrap()
            })
            .collect();
        (keys, addresses)
    }

    #[test]
    fn adopt_finalized_stops_historical_votes_and_restarts_at_the_head() {
        let (keys, addresses) = validators(4);
        let mut manager = FinalityManager::new(
            "test-chain",
            addresses.clone(),
            &addresses[0],
            keys[0].clone(),
            2,
            "hash-2",
            1_000,
        )
        .unwrap();

        let before = manager.observe_block(3, "hash-3");
        assert!(before.finalized.is_none());
        assert!(!before.votes.is_empty());
        assert!(before.votes.iter().all(|vote| vote.height == 3));

        manager.adopt_finalized(5, "hash-5");
        assert_eq!(manager.finalized_height(), 5);
        assert_eq!(manager.finalized_hash(), "hash-5");

        let stale = manager.observe_block(3, "hash-3");
        assert!(stale.votes.is_empty());
        assert!(stale.finalized.is_none());

        let fresh = manager.observe_block(6, "hash-6");
        assert!(!fresh.votes.is_empty());
        assert!(fresh.votes.iter().all(|vote| vote.height == 6));
    }

    fn manager(keys: &[SigningKey], addresses: &[String], local: usize) -> FinalityManager {
        FinalityManager::new(
            "test-chain",
            addresses.to_vec(),
            &addresses[local],
            keys[local].clone(),
            0,
            "",
            1_000,
        )
        .unwrap()
    }

    fn sign_vote(
        keys: &[SigningKey],
        addresses: &[String],
        index: usize,
        kind: verinoo_consensus::VoteKind,
        height: u64,
        round: u64,
        block_hash: &str,
    ) -> Vote {
        Vote::sign(
            "test-chain",
            kind,
            height,
            round,
            block_hash,
            &addresses[index],
            &keys[index],
        )
    }

    /// Feed votes the way the node does: every own signature the state
    /// machine produced during `drive` is already self-fed, so gossiped
    /// votes are simply handed to `on_vote` and outputs merged.
    fn apply_votes(manager: &mut FinalityManager, votes: Vec<Vote>) -> FinalityOutput {
        let mut merged = FinalityOutput::default();
        for vote in votes {
            let output = manager.on_vote(&vote);
            merged.votes.extend(output.votes);
            if output.finalized.is_some() {
                merged.finalized = output.finalized;
            }
            if output.start_round.is_some() {
                merged.start_round = output.start_round;
            }
            if merged.previous_finalized.is_none() {
                merged.previous_finalized = output.previous_finalized;
            }
        }
        merged
    }

    #[test]
    fn offline_leader_view_change_finalizes_at_round_one() {
        let (keys, addresses) = validators(4);
        // Pick a local validator that is neither the round-0 leader (the one
        // that stays offline) nor the round-1 leader, so the view change
        // happens around this node rather than through it.
        let mut sorted = addresses.clone();
        sorted.sort();
        let round0_leader = round_leader(&sorted, 1, 0).unwrap().to_string();
        let round1_leader = round_leader(&sorted, 1, 1).unwrap().to_string();
        let local = (0..4)
            .find(|i| addresses[*i] != round0_leader && addresses[*i] != round1_leader)
            .expect("four validators leave two neutral choices");
        let mut manager = manager(&keys, &addresses, local);

        // Height activation arms the round-0 timer.
        let output = manager.start_height(1);
        assert_eq!(
            output.start_round,
            Some(Schedule::Timeout {
                height: 1,
                round: 0,
                timeout_ms: 1_000,
            })
        );

        // The round-1 leader proposes while this node is still in round 0
        // (timer skew): the proposal is validated and buffered, not voted.
        let leader_1_index = addresses.iter().position(|a| *a == round1_leader).unwrap();
        assert_ne!(leader_1_index, local);
        let early = Proposal::sign(
            "test-chain",
            1,
            1,
            "hash-b",
            &round1_leader,
            &keys[leader_1_index],
        );
        assert_eq!(manager.classify_proposal(&early), ProposalRelevance::Future);
        let output = manager.on_proposal(&early, true);
        assert!(output.votes.is_empty(), "nothing is voted before round 1");

        // The round-0 timer fires: nil prevote for the timed-out round,
        // the pacemaker advances to round 1, and the buffered round-1
        // proposal is replayed and prevoted for.
        let output = manager.on_timeout(1, 0);
        let nil = output
            .votes
            .iter()
            .find(|v| v.round == 0 && v.kind == verinoo_consensus::VoteKind::Prevote)
            .expect("nil prevote for the timed-out round");
        assert!(nil.block_hash.is_empty());
        assert_eq!(
            output.start_round,
            Some(Schedule::Timeout {
                height: 1,
                round: 1,
                timeout_ms: 1_500,
            })
        );
        let prevote = output
            .votes
            .iter()
            .find(|v| v.round == 1 && v.kind == verinoo_consensus::VoteKind::Prevote)
            .expect("replayed prevote for the buffered round-1 proposal");
        assert_eq!(prevote.block_hash, "hash-b");
        assert_eq!(prevote.validator, addresses[local]);

        // The peers whose timers fired first join round 1: with this
        // node's own prevote they reach the 3-of-4 quorum, which emits the
        // own precommit (self-fed by the manager).
        let peers: Vec<usize> = (0..4)
            .filter(|i| *i != local && *i != leader_1_index)
            .collect();
        let votes = vec![
            sign_vote(
                &keys,
                &addresses,
                leader_1_index,
                verinoo_consensus::VoteKind::Prevote,
                1,
                1,
                "hash-b",
            ),
            sign_vote(
                &keys,
                &addresses,
                peers[0],
                verinoo_consensus::VoteKind::Prevote,
                1,
                1,
                "hash-b",
            ),
        ];
        let output = apply_votes(&mut manager, votes);
        assert!(output
            .votes
            .iter()
            .any(|v| v.kind == verinoo_consensus::VoteKind::Precommit && v.block_hash == "hash-b"));

        // Precommit quorum: the height finalizes in round 1 even though the
        // round-0 leader never proposed.
        let votes = vec![
            sign_vote(
                &keys,
                &addresses,
                leader_1_index,
                verinoo_consensus::VoteKind::Precommit,
                1,
                1,
                "hash-b",
            ),
            sign_vote(
                &keys,
                &addresses,
                peers[0],
                verinoo_consensus::VoteKind::Precommit,
                1,
                1,
                "hash-b",
            ),
        ];
        let output = apply_votes(&mut manager, votes);
        let (height, hash, certificate) = output.finalized.expect("round-1 finality");
        assert_eq!(height, 1);
        assert_eq!(hash, "hash-b");
        assert_eq!(certificate.round, 1);
        assert_eq!(manager.finalized_height(), 1);
        assert_eq!(
            output.previous_finalized,
            Some((0, String::new())),
            "rollback target is the pre-finalization state"
        );
    }

    #[test]
    fn wrong_state_root_proposal_is_prevoted_nil() {
        let (keys, addresses) = validators(4);
        let mut manager = manager(&keys, &addresses, 2);
        manager.start_height(1);

        // The round-0 leader's envelope, but the node's execution check
        // rejected the block (block_valid = false).
        let leader = manager.round_leader(1, 0).unwrap();
        let leader_index = addresses.iter().position(|a| a == &leader).unwrap();
        let proposal = Proposal::sign("test-chain", 1, 0, "hash-bad", &leader, &keys[leader_index]);
        assert_eq!(
            manager.classify_proposal(&proposal),
            ProposalRelevance::Current
        );

        let output = manager.on_proposal(&proposal, false);
        let [vote] = output.votes.as_slice() else {
            panic!("a failed proposal still emits a nil prevote");
        };
        assert_eq!(vote.kind, verinoo_consensus::VoteKind::Prevote);
        assert_eq!(vote.height, 1);
        assert_eq!(vote.round, 0);
        assert!(vote.block_hash.is_empty(), "nil prevote");
    }

    #[test]
    fn classify_proposal_drops_wrong_chain_wrong_leader_and_stale_rounds() {
        let (keys, addresses) = validators(4);
        let mut manager = manager(&keys, &addresses, 0);
        manager.start_height(1);

        let leader = manager.round_leader(1, 0).unwrap();
        let leader_index = addresses.iter().position(|a| a == &leader).unwrap();
        let good = Proposal::sign("test-chain", 1, 0, "hash-a", &leader, &keys[leader_index]);
        assert_eq!(manager.classify_proposal(&good), ProposalRelevance::Current);

        // Foreign chain ids never reach validation.
        let foreign = Proposal::sign("other-chain", 1, 0, "hash-a", &leader, &keys[leader_index]);
        assert_eq!(
            manager.classify_proposal(&foreign),
            ProposalRelevance::Irrelevant
        );

        // A non-leader envelope is not a proposal.
        let not_leader = addresses.iter().position(|a| *a != leader).unwrap();
        let forged = Proposal::sign(
            "test-chain",
            1,
            0,
            "hash-a",
            &addresses[not_leader],
            &keys[not_leader],
        );
        assert_eq!(
            manager.classify_proposal(&forged),
            ProposalRelevance::Irrelevant
        );

        // Reachable future rounds are buffered; rounds beyond the window,
        // and rounds below the current one, are dropped.
        let proposal_from = |manager: &FinalityManager, round: u64| -> (String, usize) {
            let leader = manager.round_leader(1, round).unwrap();
            let index = addresses.iter().position(|a| *a == leader).unwrap();
            (leader, index)
        };
        let (leader_5, index_5) = proposal_from(&manager, 5);
        let future = Proposal::sign("test-chain", 1, 5, "hash-a", &leader_5, &keys[index_5]);
        assert_eq!(
            manager.classify_proposal(&future),
            ProposalRelevance::Future
        );
        let (leader_40, index_40) = proposal_from(&manager, 40);
        let too_far = Proposal::sign("test-chain", 1, 40, "hash-a", &leader_40, &keys[index_40]);
        assert_eq!(
            manager.classify_proposal(&too_far),
            ProposalRelevance::Irrelevant
        );

        // After the round advances to 1, a round-0 proposal is stale.
        manager.on_timeout(1, 0);
        assert_eq!(
            manager.classify_proposal(&good),
            ProposalRelevance::Irrelevant
        );
    }

    #[test]
    fn restart_restore_rebroadcasts_the_own_vote_without_resigning() {
        let (keys, addresses) = validators(4);
        let mut first = manager(&keys, &addresses, 0);
        first.start_height(1);

        // The local validator is the round-0 leader; its proposal makes the
        // manager sign and emit a prevote.
        let proposal = Proposal::sign("test-chain", 1, 0, "hash-a", &addresses[0], &keys[0]);
        let output = first.on_proposal(&proposal, true);
        let original = output
            .votes
            .iter()
            .find(|v| v.kind == verinoo_consensus::VoteKind::Prevote)
            .expect("signed prevote")
            .clone();

        // The node persists own votes before broadcasting (the same key
        // scheme `apply_finality_output` uses).
        let store = verinoo_state::StateStore::open_temp().unwrap();
        store.set_meta(
            "consensusOwnVote:1:0:prevote",
            &serde_json::to_value(&original).unwrap(),
        );

        // Restart: a fresh manager reloads the persisted vote.
        let mut restarted = manager(&keys, &addresses, 0);
        assert_eq!(restarted.restore_own_votes_from_store(&store), 1);

        // The same proposal again rebroadcasts the stored vote verbatim —
        // the signature is byte-identical, so there is no second signature.
        let output = restarted.on_proposal(&proposal, true);
        let [vote] = output.votes.as_slice() else {
            panic!("expected exactly the rebroadcast prevote");
        };
        assert_eq!(vote, &original);

        // A conflicting proposal for the same round still cannot make this
        // node sign a different prevote.
        let conflicting = Proposal::sign("test-chain", 1, 0, "hash-c", &addresses[0], &keys[0]);
        let output = restarted.on_proposal(&conflicting, true);
        assert!(output
            .votes
            .iter()
            .all(|v| v.signature == original.signature));

        // And a conflicting restore is refused outright.
        let mut forged = original.clone();
        forged.block_hash = "hash-c".into();
        let payload = forged.signing_payload().unwrap();
        use ed25519_dalek::Signer as _;
        forged.signature = hex::encode(keys[0].sign(&payload).to_bytes());
        assert!(restarted.restore_own_vote(forged).is_err());
    }

    #[test]
    fn commit_failure_rollback_keeps_the_manager_at_the_last_committable_height() {
        let (keys, addresses) = validators(1);
        let mut manager = manager(&keys, &addresses, 0);
        manager.start_height(1);

        // Single validator: its own proposal finalizes immediately.
        let output = manager.observe_block(1, "hash-x");
        let (height, hash, certificate) = output.finalized.expect("single-validator finality");
        assert_eq!(height, 1);
        assert_eq!(hash, "hash-x");
        assert_eq!(manager.finalized_height(), 1);
        assert_eq!(
            output.previous_finalized,
            Some((0, String::new())),
            "the rollback target is the pre-finalization finality"
        );
        assert!(certificate.verify(&addresses).is_ok());

        // The node cannot commit the finalized block (unknown block, model
        // divergence, ...). Rolling back to `previous_finalized` keeps the
        // manager consistent with the canonical head instead of stranding
        // it at a height that can never commit.
        let (previous_height, previous_hash) = output.previous_finalized.unwrap();
        manager.rollback_finalization(previous_height, &previous_hash);
        assert_eq!(manager.finalized_height(), previous_height);
        assert_eq!(manager.finalized_hash(), previous_hash);
        assert_eq!(manager.finalized_hash(), "");

        // The manager is usable again: the height can be re-observed and
        // re-finalized (for example after the divergence is fixed).
        let output = manager.observe_block(1, "hash-y");
        let (height, hash, _) = output.finalized.expect("re-finalization after rollback");
        assert_eq!(height, 1);
        assert_eq!(hash, "hash-y");
        assert_eq!(manager.finalized_height(), 1);
    }
}
