use ed25519_dalek::SigningKey;
use verinoo_consensus::{
    public_key_to_address, Error, Evidence, Proposal, ProposalEvidence, Vote, VoteKind,
};

const CHAIN: &str = "verinoo-test-1";

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn addr(sk: &SigningKey) -> String {
    public_key_to_address(&hex::encode(sk.verifying_key().to_bytes())).unwrap()
}

fn proposal(sk: &SigningKey, height: u64, round: u64, hash: &str) -> Proposal {
    Proposal::sign(CHAIN, height, round, hash, &addr(sk), sk)
}

fn proposal_evidence(
    sk: &SigningKey,
    height: u64,
    round: u64,
    hashes: (&str, &str),
) -> ProposalEvidence {
    ProposalEvidence::new(
        proposal(sk, height, round, hashes.0),
        proposal(sk, height, round, hashes.1),
    )
    .unwrap()
}

#[test]
fn double_proposal_evidence_verifies_and_id_is_stable() {
    let sk = key(1);
    let vals = vec![addr(&sk)];
    let evidence = proposal_evidence(&sk, 5, 0, ("aaa", "bbb"));
    evidence.verify(&vals).unwrap();
    evidence.verify_with_chain(CHAIN, &vals).unwrap();

    let same = proposal_evidence(&sk, 5, 0, ("aaa", "bbb"));
    assert_eq!(evidence.id(), same.id());
    assert_eq!(evidence.id().len(), 64);

    let swapped =
        ProposalEvidence::new(proposal(&sk, 5, 0, "bbb"), proposal(&sk, 5, 0, "aaa")).unwrap();
    assert_eq!(evidence.id(), swapped.id());

    let other = proposal_evidence(&sk, 5, 0, ("aaa", "ccc"));
    assert_ne!(evidence.id(), other.id());
}

#[test]
fn proposal_evidence_rejects_same_hash() {
    let sk = key(1);
    let a = proposal(&sk, 5, 0, "aaa");
    let b = proposal(&sk, 5, 0, "aaa");
    assert_eq!(ProposalEvidence::new(a, b), Err(Error::NotDoubleProposal));

    let vals = vec![addr(&sk)];
    let mut bad = proposal_evidence(&sk, 5, 0, ("aaa", "bbb"));
    bad.b.block_hash = bad.a.block_hash.clone();
    assert_eq!(bad.verify(&vals), Err(Error::NotDoubleProposal));
}

#[test]
fn proposal_evidence_rejects_unknown_validator() {
    let sk = key(1);
    let vals = vec![addr(&key(2))];
    let evidence = proposal_evidence(&sk, 5, 0, ("aaa", "bbb"));
    assert_eq!(
        evidence.verify(&vals),
        Err(Error::UnknownValidator(addr(&sk)))
    );
}

#[test]
fn proposal_evidence_rejects_wrong_chain() {
    let sk = key(1);
    let vals = vec![addr(&sk)];
    let evidence = proposal_evidence(&sk, 5, 0, ("aaa", "bbb"));
    assert_eq!(
        evidence.verify_with_chain("other-chain", &vals),
        Err(Error::ChainIdMismatch {
            expected: "other-chain".into(),
            actual: CHAIN.into(),
        })
    );

    let other_chain = Proposal::sign("other-chain", 5, 0, "bbb", &addr(&sk), &sk);
    assert_eq!(
        ProposalEvidence::new(proposal(&sk, 5, 0, "aaa"), other_chain),
        Err(Error::ProposalEvidenceMismatch)
    );
}

#[test]
fn proposal_evidence_rejects_mismatched_tuple() {
    let sk = key(1);
    let vals = vec![addr(&sk)];

    assert_eq!(
        ProposalEvidence::new(proposal(&sk, 5, 0, "aaa"), proposal(&sk, 5, 1, "bbb"),),
        Err(Error::ProposalEvidenceMismatch)
    );

    assert_eq!(
        ProposalEvidence::new(proposal(&sk, 5, 0, "aaa"), proposal(&sk, 6, 0, "bbb"),),
        Err(Error::ProposalEvidenceMismatch)
    );

    let other_proposer = Proposal::sign(CHAIN, 5, 0, "bbb", &addr(&key(2)), &key(2));
    assert_eq!(
        ProposalEvidence::new(proposal(&sk, 5, 0, "aaa"), other_proposer),
        Err(Error::ProposalEvidenceMismatch)
    );

    let mut other_inner_chain = proposal_evidence(&sk, 5, 0, ("aaa", "bbb"));
    other_inner_chain.a.chain_id = "other-chain".into();
    assert_eq!(
        other_inner_chain.verify(&vals),
        Err(Error::ProposalEvidenceMismatch)
    );

    let mut other_outer_height = proposal_evidence(&sk, 5, 0, ("aaa", "bbb"));
    other_outer_height.height += 1;
    assert_eq!(
        other_outer_height.verify(&vals),
        Err(Error::ProposalEvidenceMismatch)
    );
}

#[test]
fn proposal_evidence_rejects_bad_signature() {
    let sk = key(1);
    let vals = vec![addr(&sk)];
    let mut bad = proposal_evidence(&sk, 5, 0, ("aaa", "bbb"));
    bad.a.signature = "00".repeat(64);
    assert_eq!(bad.verify(&vals), Err(Error::InvalidSignature));

    let mut malformed = proposal_evidence(&sk, 5, 0, ("aaa", "bbb"));
    malformed.b.signature = "zz".into();
    assert_eq!(malformed.verify(&vals), Err(Error::MalformedSignature));
}

#[test]
fn proposal_evidence_serializes_camel_case() {
    let evidence = proposal_evidence(&key(1), 5, 0, ("aaa", "bbb"));
    let value = serde_json::to_value(&evidence).unwrap();
    for key in ["chainId", "proposer", "height", "round", "a", "b"] {
        assert!(value.get(key).is_some(), "missing field {key}");
    }
    assert!(value
        .get("a")
        .expect("inner proposal a")
        .get("publicKey")
        .is_some());
    let back: ProposalEvidence = serde_json::from_value(value).unwrap();
    assert_eq!(back, evidence);
}

#[test]
fn vote_evidence_still_passes() {
    let sk = key(1);
    let vals = vec![addr(&sk)];
    let a = Vote::sign(CHAIN, VoteKind::Precommit, 5, 0, "aaa", &addr(&sk), &sk);
    let b = Vote::sign(CHAIN, VoteKind::Precommit, 5, 0, "bbb", &addr(&sk), &sk);
    let evidence = Evidence::new(a, b.clone()).unwrap();
    evidence.verify(&vals).unwrap();
    assert_eq!(evidence.id().len(), 64);

    let swapped = Evidence::new(b, evidence.a.clone()).unwrap();
    assert_eq!(evidence.id(), swapped.id());
    assert_eq!(evidence.kind, VoteKind::Precommit);
}
