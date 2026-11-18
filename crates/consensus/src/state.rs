//! Deterministic round/lock state machine for the finality gadget.
//!
//! The state machine emits intents; the node signs them, persists the signed
//! vote, broadcasts it, and feeds it back through `on_vote`. This keeps the
//! transition logic pure and testable without sockets or wall clocks.

use super::{public_key_to_address, quorum, CommitCertificate, Evidence, Proposal, Vote, VoteKind};
use std::collections::{HashMap, HashSet};

pub const DEFAULT_MAX_ROUND: u64 = 8;
pub const DEFAULT_MAX_TIMEOUT_MS: u64 = 60_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoteIntent {
    pub kind: VoteKind,
    pub height: u64,
    pub round: u64,
    pub block_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Sign, persist, broadcast and feed back through `on_vote`.
    Vote(VoteIntent),
    /// Re-broadcast a previously persisted own vote verbatim.
    Rebroadcast(Vote),
    StartRound {
        height: u64,
        round: u64,
        timeout_ms: u64,
    },
    Finalized {
        height: u64,
        block_hash: String,
        certificate: CommitCertificate,
    },
    Evidence(Box<Evidence>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lock {
    pub height: u64,
    pub block_hash: String,
    pub round: u64,
}

#[derive(Debug, Clone)]
pub struct ConsensusState {
    chain_id: String,
    validators: Vec<String>,
    local: String,
    quorum: usize,
    height: u64,
    round: u64,
    finalized_height: u64,
    finalized_hash: String,
    locked: Option<Lock>,
    votes: HashMap<(u64, u64, VoteKind, String), Vote>,
    own_votes: HashMap<(u64, u64, VoteKind), Vote>,
    pending: HashSet<(u64, u64, VoteKind)>,
    evidence: Vec<Evidence>,
    evidence_ids: HashSet<String>,
    certificate: Option<CommitCertificate>,
    base_timeout_ms: u64,
    max_round: u64,
}

impl ConsensusState {
    pub fn new(
        chain_id: &str,
        validators: Vec<String>,
        local: &str,
        start_height: u64,
        base_timeout_ms: u64,
    ) -> Result<Self, String> {
        if chain_id.is_empty() {
            return Err("chain id is empty".into());
        }
        let mut validators = validators;
        validators.sort();
        validators.dedup();
        if validators.is_empty() {
            return Err("validator set is empty".into());
        }
        if !validators.iter().any(|v| v == local) {
            return Err("local validator is not in the consensus set".into());
        }
        let q = quorum(validators.len());
        Ok(Self {
            chain_id: chain_id.to_string(),
            validators,
            local: local.to_string(),
            quorum: q,
            height: start_height.max(1),
            round: 0,
            finalized_height: 0,
            finalized_hash: String::new(),
            locked: None,
            votes: HashMap::new(),
            own_votes: HashMap::new(),
            pending: HashSet::new(),
            evidence: Vec::new(),
            evidence_ids: HashSet::new(),
            certificate: None,
            base_timeout_ms: base_timeout_ms.max(1),
            max_round: DEFAULT_MAX_ROUND,
        })
    }

    pub fn chain_id(&self) -> &str {
        &self.chain_id
    }
    pub fn validators(&self) -> &[String] {
        &self.validators
    }
    pub fn local(&self) -> &str {
        &self.local
    }
    pub fn quorum(&self) -> usize {
        self.quorum
    }
    pub fn height(&self) -> u64 {
        self.height
    }
    pub fn round(&self) -> u64 {
        self.round
    }
    pub fn finalized_height(&self) -> u64 {
        self.finalized_height
    }
    pub fn finalized_hash(&self) -> &str {
        &self.finalized_hash
    }
    pub fn lock(&self) -> Option<&Lock> {
        self.locked.as_ref()
    }
    pub fn certificate(&self) -> Option<&CommitCertificate> {
        self.certificate.as_ref()
    }
    pub fn evidence(&self) -> &[Evidence] {
        &self.evidence
    }

    pub fn leader_for(&self, height: u64, round: u64) -> &str {
        let index =
            (height.saturating_sub(1).saturating_add(round) as usize) % self.validators.len();
        &self.validators[index]
    }

    pub fn is_leader(&self) -> bool {
        self.local == self.leader_for(self.height, self.round)
    }

    pub fn timeout_ms(&self) -> u64 {
        let mut out = self.base_timeout_ms;
        for _ in 0..self.round {
            out = out.saturating_mul(3) / 2;
            if out >= DEFAULT_MAX_TIMEOUT_MS {
                return DEFAULT_MAX_TIMEOUT_MS;
            }
        }
        out
    }

    /// Restore a signed own vote after restart. The node persists these before
    /// broadcasting; the state machine will not sign a conflicting vote for the
    /// same `(height, round, kind)`.
    pub fn restore_own_vote(&mut self, vote: Vote) -> Result<(), String> {
        vote.verify_with_chain(&self.chain_id, &self.validators)
            .map_err(|e| e.to_string())?;
        if vote.validator != self.local {
            return Err("restored vote is not from the local validator".into());
        }
        self.pending.remove(&(vote.height, vote.round, vote.kind));
        self.own_votes
            .insert((vote.height, vote.round, vote.kind), vote.clone());
        self.votes.insert(
            (vote.height, vote.round, vote.kind, vote.validator.clone()),
            vote,
        );
        Ok(())
    }

    pub fn own_vote(&self, height: u64, round: u64, kind: VoteKind) -> Option<&Vote> {
        self.own_votes.get(&(height, round, kind))
    }

    /// Advance to the next height after finality. Round/votes/locks reset.
    pub fn advance_height(&mut self, height: u64) -> Result<(), String> {
        if height <= self.finalized_height {
            return Err("next height must be above the finalized height".into());
        }
        if height <= self.height {
            return Err("next height must increase".into());
        }
        self.height = height;
        self.round = 0;
        self.locked = None;
        self.votes.clear();
        self.own_votes.clear();
        self.pending.clear();
        self.certificate = None;
        Ok(())
    }

    pub fn on_proposal(&mut self, proposal: &Proposal, block_valid: bool) -> Vec<Action> {
        if proposal.height != self.height || proposal.round != self.round {
            return Vec::new();
        }
        if proposal.proposer != self.leader_for(proposal.height, proposal.round) {
            return self.emit_vote(VoteKind::Prevote, self.height, self.round, String::new());
        }
        if proposal
            .verify_with_chain(&self.chain_id, &self.validators)
            .is_err()
        {
            return self.emit_vote(VoteKind::Prevote, self.height, self.round, String::new());
        }
        let hash = if block_valid && !proposal.block_hash.is_empty() {
            let locked_blocks = self.locked.as_ref().is_some_and(|lock| {
                lock.height == self.height
                    && lock.block_hash != proposal.block_hash
                    && proposal.round <= lock.round
            });
            if locked_blocks {
                String::new()
            } else {
                proposal.block_hash.clone()
            }
        } else {
            String::new()
        };
        self.emit_vote(VoteKind::Prevote, self.height, self.round, hash)
    }

    pub fn on_vote(&mut self, vote: &Vote) -> Vec<Action> {
        if vote.height != self.height || vote.round != self.round {
            return Vec::new();
        }
        if vote
            .verify_with_chain(&self.chain_id, &self.validators)
            .is_err()
        {
            return Vec::new();
        }
        let key = (vote.height, vote.round, vote.kind, vote.validator.clone());
        if let Some(existing) = self.votes.get(&key) {
            if existing.block_hash == vote.block_hash {
                return Vec::new();
            }
            if vote.validator != self.local {
                if let Ok(evidence) = Evidence::new(existing.clone(), vote.clone()) {
                    let id = evidence.id();
                    if self.evidence_ids.insert(id) {
                        self.evidence.push(evidence.clone());
                        return vec![Action::Evidence(Box::new(evidence))];
                    }
                }
            }
            return Vec::new();
        }
        self.votes.insert(key, vote.clone());
        if vote.validator == self.local {
            self.pending.remove(&(vote.height, vote.round, vote.kind));
            self.own_votes
                .insert((vote.height, vote.round, vote.kind), vote.clone());
        }

        let mut actions = Vec::new();
        if vote.kind == VoteKind::Prevote {
            self.apply_unlock_rule();
            if let Some(hash) = self.quorum_hash(VoteKind::Prevote, vote.round) {
                if !hash.is_empty() {
                    let should_lock = self
                        .locked
                        .as_ref()
                        .is_none_or(|lock| lock.block_hash != hash || lock.round < vote.round);
                    if should_lock {
                        self.locked = Some(Lock {
                            height: self.height,
                            block_hash: hash.clone(),
                            round: vote.round,
                        });
                    }
                }
                actions.extend(self.emit_vote(VoteKind::Precommit, self.height, vote.round, hash));
            }
        } else if vote.kind == VoteKind::Precommit {
            if let Some(hash) = self.quorum_hash(VoteKind::Precommit, vote.round) {
                if !hash.is_empty()
                    && self
                        .locked
                        .as_ref()
                        .is_none_or(|lock| lock.block_hash == hash)
                {
                    let certificate = self.build_certificate(&hash, vote.round);
                    self.finalized_height = self.height;
                    self.finalized_hash = hash.clone();
                    self.certificate = Some(certificate.clone());
                    actions.push(Action::Finalized {
                        height: self.height,
                        block_hash: hash,
                        certificate,
                    });
                }
            }
        }
        actions
    }

    pub fn on_timeout(&mut self) -> Vec<Action> {
        if self.round >= self.max_round {
            return Vec::new();
        }
        self.round += 1;
        self.pending.clear();
        let mut actions = vec![Action::StartRound {
            height: self.height,
            round: self.round,
            timeout_ms: self.timeout_ms(),
        }];
        actions.extend(self.emit_vote(VoteKind::Prevote, self.height, self.round, String::new()));
        actions
    }

    fn apply_unlock_rule(&mut self) {
        let lock = match &self.locked {
            Some(lock) => lock.clone(),
            None => return,
        };
        if let Some(hash) = self.quorum_hash(VoteKind::Prevote, self.round) {
            if self.round > lock.round && !hash.is_empty() && hash != lock.block_hash {
                self.locked = None;
            }
        }
    }

    fn quorum_hash(&self, kind: VoteKind, round: u64) -> Option<String> {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for ((height, vote_round, vote_kind, _), vote) in &self.votes {
            if *height == self.height && *vote_round == round && *vote_kind == kind {
                *counts.entry(vote.block_hash.as_str()).or_default() += 1;
            }
        }
        counts
            .into_iter()
            .find(|(_, count)| *count >= self.quorum)
            .map(|(hash, _)| hash.to_string())
    }

    fn build_certificate(&self, hash: &str, round: u64) -> CommitCertificate {
        let mut votes: Vec<Vote> = self
            .votes
            .iter()
            .filter(|((height, vote_round, kind, _), vote)| {
                *height == self.height
                    && *vote_round == round
                    && *kind == VoteKind::Precommit
                    && vote.block_hash == hash
            })
            .map(|(_, vote)| vote.clone())
            .collect();
        votes.sort_by(|a, b| a.validator.cmp(&b.validator));
        CommitCertificate {
            chain_id: self.chain_id.clone(),
            height: self.height,
            round,
            block_hash: hash.to_string(),
            votes,
        }
    }

    fn emit_vote(
        &mut self,
        kind: VoteKind,
        height: u64,
        round: u64,
        block_hash: String,
    ) -> Vec<Action> {
        if height <= self.finalized_height {
            return Vec::new();
        }
        if let Some(existing) = self.own_votes.get(&(height, round, kind)) {
            return vec![Action::Rebroadcast(existing.clone())];
        }
        if !self.pending.insert((height, round, kind)) {
            return Vec::new();
        }
        vec![Action::Vote(VoteIntent {
            kind,
            height,
            round,
            block_hash,
        })]
    }

    pub fn local_address_for_key(public_key: &str) -> Result<String, String> {
        public_key_to_address(public_key).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn key(seed: u8) -> (SigningKey, String) {
        let sk = SigningKey::from_bytes(&[seed; 32]);
        let pk = hex::encode(sk.verifying_key().to_bytes());
        let addr = public_key_to_address(&pk).unwrap();
        (sk, addr)
    }

    fn setup() -> (ConsensusState, Vec<(SigningKey, String)>) {
        let keys: Vec<_> = (1..=4).map(key).collect();
        let mut validators: Vec<String> = keys.iter().map(|(_, a)| a.clone()).collect();
        validators.sort();
        let local = validators[0].clone();
        (
            ConsensusState::new("chain", validators, &local, 1, 1000).unwrap(),
            keys,
        )
    }

    fn find_key<'a>(keys: &'a [(SigningKey, String)], validator: &str) -> &'a SigningKey {
        &keys
            .iter()
            .find(|(_, address)| address == validator)
            .unwrap()
            .0
    }

    fn sign_intent(keys: &[(SigningKey, String)], local: &str, intent: &VoteIntent) -> Vote {
        Vote::sign(
            "chain",
            intent.kind,
            intent.height,
            intent.round,
            &intent.block_hash,
            local,
            find_key(keys, local),
        )
    }

    fn intent(actions: &[Action]) -> Option<VoteIntent> {
        actions.iter().find_map(|action| match action {
            Action::Vote(intent) => Some(intent.clone()),
            _ => None,
        })
    }

    #[test]
    fn happy_path_finalizes_with_three_precommits() {
        let (mut state, keys) = setup();
        let local = state.local().to_string();
        let leader = state.leader_for(1, 0).to_string();
        let proposal = Proposal::sign("chain", 1, 0, "hash-a", &leader, find_key(&keys, &leader));
        let actions = state.on_proposal(&proposal, true);
        let prevote = intent(&actions).expect("prevote intent");
        assert_eq!(prevote.block_hash, "hash-a");
        state.on_vote(&sign_intent(&keys, &local, &prevote));

        for (_, address) in keys.iter().filter(|(_, a)| *a != local && *a != leader) {
            let vote = Vote::sign(
                "chain",
                VoteKind::Prevote,
                1,
                0,
                "hash-a",
                address,
                find_key(&keys, address),
            );
            let actions = state.on_vote(&vote);
            if let Some(precommit) = intent(&actions) {
                assert_eq!(precommit.kind, VoteKind::Precommit);
                state.on_vote(&sign_intent(&keys, &local, &precommit));
            }
        }

        let mut finalized = false;
        for (_, address) in keys.iter().filter(|(_, a)| *a != local) {
            let vote = Vote::sign(
                "chain",
                VoteKind::Precommit,
                1,
                0,
                "hash-a",
                address,
                find_key(&keys, address),
            );
            let actions = state.on_vote(&vote);
            if actions
                .iter()
                .any(|a| matches!(a, Action::Finalized { .. }))
            {
                finalized = true;
            }
        }
        assert!(finalized, "expected finality");
        assert_eq!(state.finalized_height(), 1);
        assert_eq!(state.finalized_hash(), "hash-a");
        let cert = state.certificate().expect("certificate");
        assert!(cert.verify(state.validators()).is_ok());
        assert!(cert.votes.len() >= 3);
    }

    #[test]
    fn timeout_starts_next_round_with_nil_prevote() {
        let (mut state, keys) = setup();
        let local = state.local().to_string();
        let actions = state.on_timeout();
        assert_eq!(state.round(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::StartRound { round: 1, .. })));
        let prevote = intent(&actions).expect("nil prevote");
        assert_eq!(prevote.kind, VoteKind::Prevote);
        assert_eq!(prevote.round, 1);
        assert_eq!(prevote.block_hash, "");
        state.on_vote(&sign_intent(&keys, &local, &prevote));
    }

    #[test]
    fn conflicting_vote_emits_evidence() {
        let (mut state, keys) = setup();
        let voter = state.validators()[1].clone();
        let vote_a = Vote::sign(
            "chain",
            VoteKind::Prevote,
            1,
            0,
            "a",
            &voter,
            find_key(&keys, &voter),
        );
        let vote_b = Vote::sign(
            "chain",
            VoteKind::Prevote,
            1,
            0,
            "b",
            &voter,
            find_key(&keys, &voter),
        );
        state.on_vote(&vote_a);
        let actions = state.on_vote(&vote_b);
        assert!(actions
            .iter()
            .any(|action| matches!(action, Action::Evidence(_))));
        assert_eq!(state.evidence().len(), 1);
    }

    #[test]
    fn lock_rejects_conflicting_same_round_proposal() {
        let (mut state, keys) = setup();
        let local = state.local().to_string();
        let leader = state.leader_for(1, 0).to_string();
        let proposal_a = Proposal::sign("chain", 1, 0, "a", &leader, find_key(&keys, &leader));
        let actions = state.on_proposal(&proposal_a, true);
        let prevote = intent(&actions).unwrap();
        state.on_vote(&sign_intent(&keys, &local, &prevote));
        // Build a prevote quorum for a and precommit it, setting the lock.
        for (_, addr) in keys.iter().filter(|(_, a)| *a != local) {
            let vote = Vote::sign(
                "chain",
                VoteKind::Prevote,
                1,
                0,
                "a",
                addr,
                find_key(&keys, addr),
            );
            let actions = state.on_vote(&vote);
            if let Some(p) = intent(&actions) {
                state.on_vote(&sign_intent(&keys, &local, &p));
            }
        }
        let proposal_b = Proposal::sign("chain", 1, 0, "b", &leader, find_key(&keys, &leader));
        let actions = state.on_proposal(&proposal_b, true);
        assert!(actions.iter().any(|a| matches!(a, Action::Rebroadcast(_))));
    }

    #[test]
    fn own_vote_replay_does_not_resign() {
        let (mut state, keys) = setup();
        let local = state.local().to_string();
        let leader = state.leader_for(1, 0).to_string();
        let proposal = Proposal::sign("chain", 1, 0, "a", &leader, find_key(&keys, &leader));
        let actions = state.on_proposal(&proposal, true);
        let prevote = intent(&actions).unwrap();
        let signed = sign_intent(&keys, &local, &prevote);
        state.restore_own_vote(signed).unwrap();
        let actions = state.on_proposal(&proposal, true);
        assert!(actions.iter().any(|a| matches!(a, Action::Rebroadcast(_))));
        assert!(!actions.iter().any(|a| matches!(a, Action::Vote(_))));
    }
}
