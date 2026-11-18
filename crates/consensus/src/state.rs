//! Deterministic round/lock state machine for the finality gadget.
//!
//! The state machine emits intents; the node signs them, persists the signed
//! vote, broadcasts it, and feeds it back through `on_vote`. This keeps the
//! transition logic pure and testable without sockets or wall clocks.

use super::{
    public_key_to_address, quorum, CommitCertificate, Evidence, Pacemaker, Proposal, Schedule,
    Vote, VoteKind,
};
use std::collections::{HashMap, HashSet};

const MAX_BLOCK_HASH_LEN: usize = 128;

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
    /// Arm the round timer for `(height, round)` with `timeout_ms`.
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
    pacemaker: Pacemaker,
    finalized_height: u64,
    finalized_hash: String,
    locked: Option<Lock>,
    votes: HashMap<(u64, u64, VoteKind, String), Vote>,
    own_votes: HashMap<(u64, u64, VoteKind), Vote>,
    pending: HashSet<(u64, u64, VoteKind)>,
    evidence: Vec<Evidence>,
    evidence_ids: HashSet<String>,
    certificate: Option<CommitCertificate>,
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
        let mut pacemaker = Pacemaker::new(base_timeout_ms);
        pacemaker.start_height(start_height.max(1));
        Ok(Self {
            chain_id: chain_id.to_string(),
            validators,
            local: local.to_string(),
            quorum: q,
            pacemaker,
            finalized_height: 0,
            finalized_hash: String::new(),
            locked: None,
            votes: HashMap::new(),
            own_votes: HashMap::new(),
            pending: HashSet::new(),
            evidence: Vec::new(),
            evidence_ids: HashSet::new(),
            certificate: None,
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
        self.pacemaker.height()
    }
    pub fn round(&self) -> u64 {
        self.pacemaker.current_round()
    }
    pub fn current_round(&self) -> u64 {
        self.pacemaker.current_round()
    }
    pub fn max_round(&self) -> u64 {
        self.pacemaker.max_round()
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
        self.local == self.leader_for(self.height(), self.round())
    }

    /// Timeout for the current round.
    pub fn timeout_ms(&self) -> u64 {
        self.pacemaker.timeout_ms()
    }

    /// Timeout for an explicit round under the same deterministic backoff.
    pub fn timeout_ms_for(&self, round: u64) -> u64 {
        self.pacemaker.timeout_ms_at(round)
    }

    /// Timer instruction for the node's timer actor.
    pub fn schedule(&self) -> Schedule {
        self.pacemaker.schedule()
    }

    /// Explicitly enter `(height, round)` after a restart, catch-up, or sync.
    /// This does not emit a vote; the node waits for the round leader's
    /// proposal or calls `on_timeout`.
    pub fn start_round(&mut self, height: u64, round: u64) -> Result<Vec<Action>, String> {
        if height <= self.finalized_height {
            return Err("cannot start a round at or below the finalized height".into());
        }
        self.reset_for_height(height, round);
        Ok(vec![Action::StartRound {
            height,
            round,
            timeout_ms: self.timeout_ms(),
        }])
    }

    /// Restore a signed own vote after restart. The node persists these before
    /// broadcasting; the state machine will not sign a conflicting vote for the
    /// same `(height, round, kind)` and refuses conflicting restores.
    pub fn restore_own_vote(&mut self, vote: Vote) -> Result<(), String> {
        vote.verify_with_chain(&self.chain_id, &self.validators)
            .map_err(|e| e.to_string())?;
        if vote.validator != self.local {
            return Err("restored vote is not from the local validator".into());
        }
        if vote.height <= self.finalized_height {
            return Err("restored vote is at or below the finalized height".into());
        }
        let key = (vote.height, vote.round, vote.kind);
        if let Some(existing) = self.own_votes.get(&key) {
            if existing.block_hash != vote.block_hash
                || existing.signature != vote.signature
                || existing.public_key != vote.public_key
                || existing.chain_id != vote.chain_id
            {
                return Err("conflicting own vote for the same height, round, and kind".into());
            }
        }
        self.pending.remove(&key);
        self.own_votes.insert(key, vote.clone());
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
        if height <= self.height() {
            return Err("next height must increase".into());
        }
        self.reset_for_height(height, 0);
        Ok(())
    }

    /// A block already validated/applied by the node acts as the round-0
    /// proposal. The block header signature is the producer authentication;
    /// no separate proposal envelope is needed for this single-producer
    /// finality layer. Returns the prevote intent (or nil when locked).
    pub fn on_validated_block(&mut self, block_hash: &str) -> Vec<Action> {
        if !valid_block_hash(block_hash) {
            return Vec::new();
        }
        let round = self.round();
        let hash = if self.lock_allows(block_hash, round) {
            block_hash.to_string()
        } else {
            String::new()
        };
        self.emit_vote(VoteKind::Prevote, self.height(), round, hash)
    }

    /// Restore finality metadata from the canonical store on restart.
    pub fn restore_finalized(&mut self, height: u64, hash: &str) {
        if height < self.finalized_height {
            return;
        }
        self.finalized_height = height;
        self.finalized_hash = hash.to_string();
        self.votes.retain(|(h, _, _, _), _| *h > height);
        self.own_votes.retain(|(h, _, _), _| *h > height);
        self.pending.retain(|(h, _, _)| *h > height);
        if self
            .locked
            .as_ref()
            .is_some_and(|lock| lock.height <= height)
        {
            self.locked = None;
        }
    }

    /// Handle a round-change proposal. Only the elected leader for the current
    /// `(height, round)` can propose, and the envelope signature, chain id,
    /// height, round, and block hash must all be valid. An invalid or locked-out
    /// proposal produces a nil prevote; a stale or future proposal is ignored.
    pub fn on_proposal(&mut self, proposal: &Proposal, block_valid: bool) -> Vec<Action> {
        let height = self.height();
        let round = self.round();
        if proposal.height <= self.finalized_height || proposal.height != height {
            return Vec::new();
        }
        if proposal.round != round {
            return Vec::new();
        }
        let valid = block_valid
            && valid_block_hash(&proposal.block_hash)
            && proposal.proposer == self.leader_for(height, round)
            && proposal
                .verify_with_chain(&self.chain_id, &self.validators)
                .is_ok();
        let hash = if valid && self.lock_allows(&proposal.block_hash, round) {
            proposal.block_hash.clone()
        } else {
            String::new()
        };
        self.emit_vote(VoteKind::Prevote, height, round, hash)
    }

    pub fn on_vote(&mut self, vote: &Vote) -> Vec<Action> {
        if vote.height <= self.finalized_height {
            return Vec::new();
        }
        if vote.height != self.height() || vote.round != self.round() {
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

        match vote.kind {
            VoteKind::Prevote => self.on_prevote_quorum(vote.round),
            VoteKind::Precommit => self.on_precommit_quorum(vote.round),
        }
    }

    /// Phase/round timeout: nil-prevote the round that timed out, then advance
    /// exactly one round and schedule its timer. Rounds keep advancing past
    /// `R_MAX` with the capped timeout.
    pub fn on_timeout(&mut self) -> Vec<Action> {
        if self.height() <= self.finalized_height {
            return Vec::new();
        }
        let timed_out_round = self.round();
        let height = self.height();
        let mut actions = self.emit_vote(VoteKind::Prevote, height, timed_out_round, String::new());
        self.pending.clear();
        self.pacemaker.on_timeout();
        actions.push(Action::StartRound {
            height,
            round: self.round(),
            timeout_ms: self.timeout_ms(),
        });
        actions
    }

    fn reset_for_height(&mut self, height: u64, round: u64) {
        self.pacemaker.start_round(height, round);
        self.votes.clear();
        self.pending.clear();
        self.locked = None;
        self.certificate = None;
        self.own_votes
            .retain(|(h, _, _), _| *h > self.finalized_height);
    }

    /// Tendermint lock rule: a proposal at `round` may be prevoted when there
    /// is no lock, the lock is for a different (older) height, the hash
    /// matches, or the new round is strictly above the locked round.
    fn lock_allows(&self, block_hash: &str, round: u64) -> bool {
        match &self.locked {
            None => true,
            Some(lock) if lock.height < self.height() => true,
            Some(lock) if lock.height == self.height() => {
                lock.block_hash == block_hash || round > lock.round
            }
            Some(_) => false,
        }
    }

    /// A precommit certificate is authoritative. A lock from an earlier round
    /// is unlocked by a certificate from a later round; only a same-round
    /// conflicting lock blocks finality (which would imply more than f faulty
    /// validators).
    fn finality_allowed(&self, block_hash: &str, round: u64) -> bool {
        match &self.locked {
            None => true,
            Some(lock) if lock.height < self.height() => true,
            Some(lock) if lock.height == self.height() => {
                lock.block_hash == block_hash || round > lock.round
            }
            Some(_) => false,
        }
    }

    fn on_prevote_quorum(&mut self, round: u64) -> Vec<Action> {
        let Some(hash) = self.quorum_hash(VoteKind::Prevote, round) else {
            return Vec::new();
        };
        let height = self.height();
        let own = self
            .own_votes
            .get(&(height, round, VoteKind::Prevote))
            .cloned();
        match own {
            Some(vote) if vote.block_hash == hash => {
                if !hash.is_empty() && self.lock_allows(&hash, round) {
                    self.locked = Some(Lock {
                        height,
                        block_hash: hash.clone(),
                        round,
                    });
                    return self.emit_vote(VoteKind::Precommit, height, round, hash);
                }
                if hash.is_empty() {
                    return self.emit_vote(VoteKind::Precommit, height, round, hash);
                }
                Vec::new()
            }
            // Never precommit a hash that differs from our own prevote.
            Some(_) => Vec::new(),
            None => {
                let hash = if hash.is_empty() || self.lock_allows(&hash, round) {
                    hash
                } else {
                    return Vec::new();
                };
                self.emit_vote(VoteKind::Prevote, height, round, hash)
            }
        }
    }

    fn on_precommit_quorum(&mut self, round: u64) -> Vec<Action> {
        let Some(hash) = self.quorum_hash(VoteKind::Precommit, round) else {
            return Vec::new();
        };
        if hash.is_empty() || !self.finality_allowed(&hash, round) {
            return Vec::new();
        }
        let height = self.height();
        if self.lock_allows(&hash, round) {
            self.locked = Some(Lock {
                height,
                block_hash: hash.clone(),
                round,
            });
        }
        let certificate = self.build_certificate(&hash, round);
        self.finalized_height = height;
        self.finalized_hash = hash.clone();
        self.certificate = Some(certificate.clone());
        vec![Action::Finalized {
            height,
            block_hash: hash,
            certificate,
        }]
    }

    fn quorum_hash(&self, kind: VoteKind, round: u64) -> Option<String> {
        let height = self.height();
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for ((vote_height, vote_round, vote_kind, _), vote) in &self.votes {
            if *vote_height == height && *vote_round == round && *vote_kind == kind {
                *counts.entry(vote.block_hash.as_str()).or_default() += 1;
            }
        }
        counts
            .into_iter()
            .find(|(_, count)| *count >= self.quorum)
            .map(|(hash, _)| hash.to_string())
    }

    fn build_certificate(&self, hash: &str, round: u64) -> CommitCertificate {
        let height = self.height();
        let mut votes: Vec<Vote> = self
            .votes
            .iter()
            .filter(|((vote_height, vote_round, kind, _), vote)| {
                *vote_height == height
                    && *vote_round == round
                    && *kind == VoteKind::Precommit
                    && vote.block_hash == hash
            })
            .map(|(_, vote)| vote.clone())
            .collect();
        votes.sort_by(|a, b| a.validator.cmp(&b.validator));
        CommitCertificate {
            chain_id: self.chain_id.clone(),
            height,
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

fn valid_block_hash(block_hash: &str) -> bool {
    !block_hash.is_empty() && block_hash.len() <= MAX_BLOCK_HASH_LEN
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
    fn timeout_nil_prevotes_the_timed_out_round_and_starts_the_next() {
        let (mut state, keys) = setup();
        let local = state.local().to_string();
        let actions = state.on_timeout();
        assert_eq!(state.round(), 1);
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::StartRound { round: 1, .. })));
        let prevote = intent(&actions).expect("nil prevote");
        assert_eq!(prevote.kind, VoteKind::Prevote);
        assert_eq!(prevote.round, 0);
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
        assert_eq!(state.lock().unwrap().block_hash, "a");
        assert_eq!(state.lock().unwrap().round, 0);
        let proposal_b = Proposal::sign("chain", 1, 0, "b", &leader, find_key(&keys, &leader));
        let actions = state.on_proposal(&proposal_b, true);
        assert!(actions.iter().any(|a| matches!(a, Action::Rebroadcast(_))));
        assert!(!actions.iter().any(|a| matches!(a, Action::Vote(_))));
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
