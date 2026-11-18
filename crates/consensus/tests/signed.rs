use ed25519_dalek::SigningKey;
use verinoo_consensus::{
    public_key_to_address, quorum, verify_certificate, CommitCertificate, Error, Evidence,
    Proposal, Vote, VoteKind,
};

const CHAIN: &str = "verinoo-test-1";

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn addr(sk: &SigningKey) -> String {
    public_key_to_address(&hex::encode(sk.verifying_key().to_bytes())).unwrap()
}

fn validators(count: u8) -> Vec<String> {
    (1..=count).map(|seed| addr(&key(seed))).collect()
}

fn vote(sk: &SigningKey, kind: VoteKind, height: u64, round: u64, hash: &str) -> Vote {
    Vote::sign(CHAIN, kind, height, round, hash, &addr(sk), sk)
}

#[test]
fn vote_and_proposal_sign_and_verify() {
    let sk = key(1);
    let vals = vec![addr(&sk)];
    let v = vote(&sk, VoteKind::Prevote, 7, 2, "aa");
    v.verify(&vals).unwrap();
    v.verify_with_chain(CHAIN, &vals).unwrap();

    let p = Proposal::sign(CHAIN, 7, 2, "aa", &addr(&sk), &sk);
    p.verify(&vals).unwrap();
    p.verify_with_chain(CHAIN, &vals).unwrap();
}

#[test]
fn vote_rejects_wrong_chain() {
    let sk = key(1);
    let vals = vec![addr(&sk)];
    let v = vote(&sk, VoteKind::Prevote, 1, 0, "aa");
    assert_eq!(
        v.verify_with_chain("other-chain", &vals),
        Err(Error::ChainIdMismatch {
            expected: "other-chain".into(),
            actual: CHAIN.into(),
        })
    );
}

#[test]
fn vote_rejects_unknown_validator() {
    let sk = key(1);
    let vals = vec![addr(&key(2))];
    assert!(matches!(
        vote(&sk, VoteKind::Prevote, 1, 0, "aa").verify(&vals),
        Err(Error::UnknownValidator(_))
    ));

    let p = Proposal::sign(CHAIN, 1, 0, "aa", &addr(&sk), &sk);
    assert!(matches!(p.verify(&vals), Err(Error::UnknownValidator(_))));
}

#[test]
fn vote_rejects_bad_signature() {
    let sk = key(1);
    let vals = vec![addr(&sk)];
    let mut v = vote(&sk, VoteKind::Prevote, 1, 0, "aa");
    v.signature = "00".repeat(64);
    assert_eq!(v.verify(&vals), Err(Error::InvalidSignature));

    v.signature = "zz".into();
    assert_eq!(v.verify(&vals), Err(Error::MalformedSignature));
}

#[test]
fn vote_rejects_wrong_version() {
    let sk = key(1);
    let vals = vec![addr(&sk)];
    let mut v = vote(&sk, VoteKind::Prevote, 1, 0, "aa");
    v.v = 2;
    assert_eq!(v.verify(&vals), Err(Error::UnsupportedVersion(2)));
}

#[test]
fn vote_rejects_address_public_key_mismatch() {
    let sk1 = key(1);
    let sk2 = key(2);
    let vals = vec![addr(&sk1)];
    let mut v = vote(&sk1, VoteKind::Prevote, 1, 0, "aa");
    v.public_key = hex::encode(sk2.verifying_key().to_bytes());
    assert_eq!(v.verify(&vals), Err(Error::AddressMismatch(addr(&sk1))));
}

#[test]
fn vote_rejects_malformed_public_key() {
    let sk = key(1);
    let vals = vec![addr(&sk)];
    let mut v = vote(&sk, VoteKind::Prevote, 1, 0, "aa");
    v.public_key = "not-hex".into();
    assert_eq!(v.verify(&vals), Err(Error::MalformedPublicKey));

    v.public_key = "00".repeat(31);
    assert_eq!(v.verify(&vals), Err(Error::InvalidPublicKeyLength));
}

#[test]
fn quorum_values_for_small_sets() {
    assert_eq!(
        (1..=7).map(quorum).collect::<Vec<_>>(),
        vec![1, 2, 3, 3, 4, 5, 5]
    );
}

#[test]
fn certificate_with_three_precommits_for_four_validators_passes() {
    let vals = validators(4);
    let votes = (1..=3)
        .map(|seed| vote(&key(seed), VoteKind::Precommit, 9, 1, "block-a"))
        .collect();
    let cert = CommitCertificate {
        chain_id: CHAIN.into(),
        height: 9,
        round: 1,
        block_hash: "block-a".into(),
        votes,
    };
    verify_certificate(&cert, &vals).unwrap();
    cert.verify(&vals).unwrap();
}

#[test]
fn certificate_rejects_duplicate_signer() {
    let vals = validators(4);
    let mut cert = certificate();
    cert.votes[1] = cert.votes[0].clone();
    assert!(matches!(
        verify_certificate(&cert, &vals),
        Err(Error::DuplicateValidator(_))
    ));
}

#[test]
fn certificate_rejects_insufficient_votes() {
    let vals = validators(4);
    let mut cert = certificate();
    cert.votes.truncate(2);
    assert_eq!(
        verify_certificate(&cert, &vals),
        Err(Error::InsufficientQuorum { got: 2, need: 3 })
    );

    let mut empty = cert.clone();
    empty.votes.clear();
    assert_eq!(
        verify_certificate(&empty, &vals),
        Err(Error::InsufficientQuorum { got: 0, need: 3 })
    );
}

#[test]
fn certificate_rejects_mixed_hash_or_round() {
    let vals = validators(4);
    let mut cert = certificate();

    let mut other_hash = cert.clone();
    other_hash.votes[2] = vote(&key(3), VoteKind::Precommit, 9, 1, "block-b");
    assert_eq!(
        verify_certificate(&other_hash, &vals),
        Err(Error::BindingMismatch)
    );

    let mut other_round = cert.clone();
    other_round.votes[2] = vote(&key(3), VoteKind::Precommit, 9, 2, "block-a");
    assert_eq!(
        verify_certificate(&other_round, &vals),
        Err(Error::BindingMismatch)
    );

    let mut other_chain = cert.clone();
    other_chain.votes[2] = Vote::sign(
        "other-chain",
        VoteKind::Precommit,
        9,
        1,
        "block-a",
        &addr(&key(3)),
        &key(3),
    );
    assert_eq!(
        verify_certificate(&other_chain, &vals),
        Err(Error::BindingMismatch)
    );

    cert.votes[2] = vote(&key(3), VoteKind::Precommit, 10, 1, "block-a");
    assert_eq!(
        verify_certificate(&cert, &vals),
        Err(Error::BindingMismatch)
    );
}

#[test]
fn certificate_rejects_wrong_kind() {
    let vals = validators(4);
    let mut cert = certificate();
    cert.votes[2] = vote(&key(3), VoteKind::Prevote, 9, 1, "block-a");
    assert_eq!(verify_certificate(&cert, &vals), Err(Error::NotPrecommit));
}

#[test]
fn certificate_rejects_nil_block_hash() {
    let vals = validators(4);
    let mut cert = certificate();
    cert.block_hash.clear();
    assert_eq!(verify_certificate(&cert, &vals), Err(Error::NilBlockHash));
}

#[test]
fn certificate_rejects_bad_signature() {
    let vals = validators(4);
    let mut cert = certificate();
    cert.votes[0].signature = "00".repeat(64);
    assert_eq!(
        verify_certificate(&cert, &vals),
        Err(Error::InvalidSignature)
    );
}

#[test]
fn certificate_rejects_unknown_validator() {
    let vals = validators(4);
    let mut cert = certificate();
    let sk = key(9);
    cert.votes[0] = vote(&sk, VoteKind::Precommit, 9, 1, "block-a");
    assert!(matches!(
        verify_certificate(&cert, &vals),
        Err(Error::UnknownValidator(_))
    ));
}

#[test]
fn equivocation_evidence_verifies_and_id_is_stable() {
    let sk = key(1);
    let vals = vec![addr(&sk)];
    let a = vote(&sk, VoteKind::Prevote, 5, 0, "aaa");
    let b = vote(&sk, VoteKind::Prevote, 5, 0, "bbb");
    let evidence = Evidence::new(a.clone(), b.clone()).unwrap();
    evidence.verify(&vals).unwrap();

    let same = Evidence::new(a.clone(), b.clone()).unwrap();
    assert_eq!(evidence.id(), same.id());
    assert_eq!(evidence.id().len(), 64);

    let other = Evidence::new(
        vote(&sk, VoteKind::Prevote, 5, 0, "aaa"),
        vote(&sk, VoteKind::Prevote, 5, 0, "ccc"),
    )
    .unwrap();
    assert_ne!(evidence.id(), other.id());
}

#[test]
fn evidence_rejects_same_hash() {
    let sk = key(1);
    let a = vote(&sk, VoteKind::Prevote, 5, 0, "aaa");
    let b = vote(&sk, VoteKind::Prevote, 5, 0, "aaa");
    assert_eq!(Evidence::new(a, b), Err(Error::NotEquivocation));
}

#[test]
fn evidence_rejects_different_tuple_or_bad_signature() {
    let sk = key(1);
    let vals = vec![addr(&sk)];
    let a = vote(&sk, VoteKind::Prevote, 5, 0, "aaa");
    let b = vote(&sk, VoteKind::Prevote, 5, 0, "bbb");

    let mut other_round = b.clone();
    other_round.round = 1;
    assert_eq!(
        Evidence::new(a.clone(), other_round),
        Err(Error::EvidenceMismatch)
    );

    let mut other_kind = b.clone();
    other_kind.kind = VoteKind::Precommit;
    assert_eq!(
        Evidence::new(a.clone(), other_kind),
        Err(Error::EvidenceMismatch)
    );

    let mut bad = Evidence::new(a, b).unwrap();
    bad.b.signature = "00".repeat(64);
    assert_eq!(bad.verify(&vals), Err(Error::InvalidSignature));
}

#[test]
fn canonical_vote_payload_bytes_are_stable() {
    let sk = key(1);
    let v = vote(&sk, VoteKind::Prevote, 42, 0, "abc123");
    let payload = String::from_utf8(v.signing_payload().unwrap()).unwrap();
    assert_eq!(
        payload,
        "verinoo/vote/v1\n{\"blockHash\":\"abc123\",\"chainId\":\"verinoo-test-1\",\"height\":42,\"round\":0,\"type\":\"prevote\",\"v\":1}"
    );
}

#[test]
fn canonical_proposal_payload_bytes_are_stable() {
    let sk = key(1);
    let p = Proposal::sign(CHAIN, 42, 0, "abc123", &addr(&sk), &sk);
    let payload = String::from_utf8(p.signing_payload().unwrap()).unwrap();
    assert_eq!(
        payload,
        "verinoo/proposal/v1\n{\"blockHash\":\"abc123\",\"chainId\":\"verinoo-test-1\",\"height\":42,\"round\":0,\"v\":1}"
    );
}

fn certificate() -> CommitCertificate {
    let votes = (1..=3)
        .map(|seed| vote(&key(seed), VoteKind::Precommit, 9, 1, "block-a"))
        .collect();
    CommitCertificate {
        chain_id: CHAIN.into(),
        height: 9,
        round: 1,
        block_hash: "block-a".into(),
        votes,
    }
}
