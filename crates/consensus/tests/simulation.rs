use ed25519_dalek::SigningKey;
use std::collections::VecDeque;
use verinoo_consensus::{
    public_key_to_address, Action, CommitCertificate, ConsensusState, Evidence, Proposal, Vote,
    VoteKind,
};

const CHAIN: &str = "verinoo-sim-1";
const BASE_TIMEOUT_MS: u64 = 1_000;

enum Msg {
    Proposal(Proposal),
    Vote(Vote),
}

struct Node {
    state: ConsensusState,
    finalized: Vec<(u64, String)>,
    certificates: Vec<CommitCertificate>,
    evidence: Vec<Evidence>,
    rounds_started: Vec<(u64, u64, u64)>,
}

struct Network {
    nodes: Vec<Node>,
    bus: VecDeque<Msg>,
    keys: Vec<(SigningKey, String)>,
    validators: Vec<String>,
}

fn key(seed: u8) -> (SigningKey, String) {
    let sk = SigningKey::from_bytes(&[seed; 32]);
    let addr = public_key_to_address(&hex::encode(sk.verifying_key().to_bytes())).unwrap();
    (sk, addr)
}

impl Network {
    fn new(honest_count: usize) -> Self {
        let keys: Vec<_> = (1..=4u8).map(key).collect();
        let mut validators: Vec<String> = keys.iter().map(|(_, addr)| addr.clone()).collect();
        validators.sort();
        let nodes = (0..honest_count)
            .map(|index| {
                let local = validators[index].clone();
                Node {
                    state: ConsensusState::new(
                        CHAIN,
                        validators.clone(),
                        &local,
                        1,
                        BASE_TIMEOUT_MS,
                    )
                    .expect("consensus state"),
                    finalized: Vec::new(),
                    certificates: Vec::new(),
                    evidence: Vec::new(),
                    rounds_started: Vec::new(),
                }
            })
            .collect();
        Self {
            nodes,
            bus: VecDeque::new(),
            keys,
            validators,
        }
    }

    fn key_of(&self, validator: &str) -> &SigningKey {
        &self
            .keys
            .iter()
            .find(|(_, addr)| addr == validator)
            .expect("validator key")
            .0
    }

    fn apply(&mut self, index: usize, actions: Vec<Action>) {
        for action in actions {
            match action {
                Action::Vote(intent) => {
                    let local = self.nodes[index].state.local().to_string();
                    let vote = Vote::sign(
                        CHAIN,
                        intent.kind,
                        intent.height,
                        intent.round,
                        &intent.block_hash,
                        &local,
                        self.key_of(&local),
                    );
                    self.bus.push_back(Msg::Vote(vote));
                }
                Action::Rebroadcast(vote) => self.bus.push_back(Msg::Vote(vote)),
                Action::StartRound {
                    height,
                    round,
                    timeout_ms,
                } => self.nodes[index]
                    .rounds_started
                    .push((height, round, timeout_ms)),
                Action::Finalized {
                    height,
                    block_hash,
                    certificate,
                } => {
                    let node = &mut self.nodes[index];
                    node.finalized.push((height, block_hash));
                    node.certificates.push(certificate);
                }
                Action::Evidence(evidence) => self.nodes[index].evidence.push(*evidence),
            }
        }
    }

    fn deliver(&mut self, msg: &Msg) {
        for index in 0..self.nodes.len() {
            let actions = match msg {
                Msg::Proposal(proposal) => self.nodes[index].state.on_proposal(proposal, true),
                Msg::Vote(vote) => self.nodes[index].state.on_vote(vote),
            };
            self.apply(index, actions);
        }
    }

    fn drive(&mut self) {
        while let Some(msg) = self.bus.pop_front() {
            self.deliver(&msg);
        }
    }

    fn tick(&mut self) {
        for index in 0..self.nodes.len() {
            let actions = self.nodes[index].state.on_timeout();
            self.apply(index, actions);
        }
    }

    fn propose(&mut self, height: u64, round: u64, block_hash: &str) {
        let leader = self.nodes[0].state.leader_for(height, round).to_string();
        let proposal = Proposal::sign(
            CHAIN,
            height,
            round,
            block_hash,
            &leader,
            self.key_of(&leader),
        );
        self.bus.push_back(Msg::Proposal(proposal));
    }

    fn inject_vote(
        &mut self,
        validator: &str,
        kind: VoteKind,
        height: u64,
        round: u64,
        block_hash: &str,
    ) {
        let vote = Vote::sign(
            CHAIN,
            kind,
            height,
            round,
            block_hash,
            validator,
            self.key_of(validator),
        );
        self.bus.push_back(Msg::Vote(vote));
    }
}

fn assert_certificate(
    cert: &CommitCertificate,
    validators: &[String],
    height: u64,
    block_hash: &str,
) {
    cert.verify(validators).expect("certificate verifies");
    assert!(cert.votes.len() >= 3, "quorum of precommits");
    assert!(cert
        .votes
        .iter()
        .all(|vote| vote.kind == VoteKind::Precommit));
    let mut signers: Vec<&str> = cert
        .votes
        .iter()
        .map(|vote| vote.validator.as_str())
        .collect();
    signers.sort_unstable();
    signers.dedup();
    assert_eq!(signers.len(), cert.votes.len(), "distinct signers");
    assert_eq!(cert.height, height);
    assert_eq!(cert.block_hash, block_hash);
}

#[test]
fn happy_path_finalizes_multiple_heights() {
    let mut net = Network::new(4);
    let hashes = ["hash-h1", "hash-h2", "hash-h3"];
    for (index, hash) in hashes.iter().enumerate() {
        let height = index as u64 + 1;
        net.propose(height, 0, hash);
        net.drive();
        for node in &net.nodes {
            assert_eq!(node.finalized.last(), Some(&(height, hash.to_string())));
            assert_eq!(node.state.finalized_height(), height);
            assert_eq!(node.state.finalized_hash(), *hash);
            let cert = node.certificates.last().expect("commit certificate");
            assert_certificate(cert, &net.validators, height, hash);
        }
        for node in &mut net.nodes {
            node.state
                .advance_height(height + 1)
                .expect("advance height");
        }
    }
    let expected: Vec<(u64, String)> = (1..=3).map(|h| (h, format!("hash-h{h}"))).collect();
    for node in &net.nodes {
        assert_eq!(node.finalized, expected);
        assert_eq!(node.certificates.len(), 3);
    }
}

#[test]
fn byzantine_equivocator_records_evidence_and_honest_quorum_finalizes() {
    let mut net = Network::new(3);
    let byzantine = net.validators[3].clone();
    assert_ne!(net.nodes[0].state.leader_for(1, 0), byzantine.as_str());

    net.inject_vote(&byzantine, VoteKind::Prevote, 1, 0, "hash-a");
    net.inject_vote(&byzantine, VoteKind::Prevote, 1, 0, "hash-b");
    net.propose(1, 0, "hash-a");
    net.drive();

    let honest_with_evidence = net
        .nodes
        .iter()
        .filter(|node| !node.evidence.is_empty())
        .count();
    assert!(honest_with_evidence >= 1, "equivocation evidence recorded");
    for node in &net.nodes {
        assert_eq!(node.state.finalized_height(), 1);
        assert_eq!(node.state.finalized_hash(), "hash-a");
        assert_eq!(node.evidence.len(), 1);
        let evidence = node.evidence.first().expect("recorded evidence");
        assert_eq!(evidence.validator, byzantine);
        assert_eq!(evidence.height, 1);
        assert_eq!(evidence.round, 0);
        assert_eq!(evidence.kind, VoteKind::Prevote);
        assert_ne!(evidence.a.block_hash, evidence.b.block_hash);
        evidence.verify(&net.validators).expect("evidence verifies");
        let cert = node.certificates.last().expect("commit certificate");
        assert_certificate(cert, &net.validators, 1, "hash-a");
        assert!(cert.votes.iter().all(|vote| vote.validator != byzantine));
    }
}

#[test]
fn offline_leader_round_one_finalizes() {
    let mut net = Network::new(4);
    let offline = net.validators[0].clone();
    assert_eq!(net.nodes[0].state.leader_for(1, 0), offline.as_str());

    net.tick();
    let round_one_timeout = BASE_TIMEOUT_MS * 3 / 2;
    for node in &net.nodes {
        assert_eq!(node.state.round(), 1);
        assert!(
            node.rounds_started.contains(&(1, 1, round_one_timeout)),
            "pacemaker advanced to round 1 with 1.5x backoff"
        );
    }
    net.drive();

    let round_one_leader = net.nodes[0].state.leader_for(1, 1).to_string();
    assert_eq!(round_one_leader, net.validators[1]);
    assert_ne!(round_one_leader, offline);
    net.propose(1, 1, "hash-a");
    net.drive();

    for node in &net.nodes {
        assert_eq!(node.finalized, vec![(1, "hash-a".to_string())]);
        assert_eq!(node.state.finalized_height(), 1);
        assert_eq!(node.state.finalized_hash(), "hash-a");
        let cert = node.certificates.last().expect("commit certificate");
        assert_certificate(cert, &net.validators, 1, "hash-a");
        assert_eq!(cert.round, 1);
    }
}
