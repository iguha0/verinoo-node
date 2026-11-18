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
use verinoo_consensus::{Action, CommitCertificate, ConsensusState, Evidence, Vote};

#[derive(Default)]
pub struct FinalityOutput {
    pub votes: Vec<Vote>,
    pub finalized: Option<(u64, String, CommitCertificate)>,
    pub evidence: Vec<Evidence>,
}

impl FinalityOutput {
    fn merge(&mut self, other: FinalityOutput) {
        self.votes.extend(other.votes);
        self.evidence.extend(other.evidence);
        if other.finalized.is_some() {
            self.finalized = other.finalized;
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
            if vote.height > self.finalized_height.saturating_add(64) {
                return FinalityOutput::default();
            }
            let pending = self.pending_votes.entry(vote.height).or_default();
            if pending.len() < 128
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
        let mut output = FinalityOutput::default();
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
                Action::StartRound { .. } => {}
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
                    }
                    output.finalized = Some((finalized_height, block_hash, certificate));
                }
            }
        }
        output
    }
}
