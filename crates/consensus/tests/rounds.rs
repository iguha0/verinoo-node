use ed25519_dalek::SigningKey;
use verinoo_consensus::{
    public_key_to_address, Action, ConsensusState, Pacemaker, PacemakerConfig, Proposal, Schedule,
    Vote, VoteIntent, VoteKind, R_MAX,
};

const CHAIN: &str = "verinoo-test-1";
const HASH_A: &str = "block-hash-a";
const HASH_B: &str = "block-hash-b";

fn key(seed: u8) -> (SigningKey, String) {
    let sk = SigningKey::from_bytes(&[seed; 32]);
    let addr = public_key_to_address(&hex::encode(sk.verifying_key().to_bytes())).unwrap();
    (sk, addr)
}

fn setup(local_index: usize) -> (ConsensusState, Vec<(SigningKey, String)>) {
    let keys: Vec<_> = (1..=4).map(key).collect();
    let mut validators: Vec<String> = keys.iter().map(|(_, address)| address.clone()).collect();
    validators.sort();
    let local = validators[local_index].clone();
    (
        ConsensusState::new(CHAIN, validators, &local, 1, 1_000).unwrap(),
        keys,
    )
}

fn find_key<'a>(keys: &'a [(SigningKey, String)], validator: &str) -> &'a SigningKey {
    &keys
        .iter()
        .find(|(_, address)| address == validator)
        .expect("validator key")
        .0
}

fn sign_vote(
    keys: &[(SigningKey, String)],
    validator: &str,
    kind: VoteKind,
    height: u64,
    round: u64,
    block_hash: &str,
) -> Vote {
    Vote::sign(
        CHAIN,
        kind,
        height,
        round,
        block_hash,
        validator,
        find_key(keys, validator),
    )
}

fn sign_intent(keys: &[(SigningKey, String)], local: &str, intent: &VoteIntent) -> Vote {
    sign_vote(
        keys,
        local,
        intent.kind,
        intent.height,
        intent.round,
        &intent.block_hash,
    )
}

fn vote_intent(actions: &[Action]) -> Option<VoteIntent> {
    actions.iter().find_map(|action| match action {
        Action::Vote(intent) => Some(intent.clone()),
        _ => None,
    })
}

fn finalized(actions: &[Action]) -> Option<(u64, String)> {
    actions.iter().find_map(|action| match action {
        Action::Finalized {
            height, block_hash, ..
        } => Some((*height, block_hash.clone())),
        _ => None,
    })
}

#[test]
fn offline_round_zero_leader_times_out_then_round_one_proposal_finalizes() {
    let (mut state, keys) = setup(1);
    let local = state.local().to_string();
    let offline = state.leader_for(1, 0).to_string();
    assert_eq!(offline, state.validators()[0]);
    assert_ne!(local, offline);

    let actions = state.on_timeout();
    assert_eq!(state.round(), 1);
    assert!(actions.iter().any(|action| matches!(
        action,
        Action::StartRound {
            height: 1,
            round: 1,
            ..
        }
    )));
    let nil = vote_intent(&actions).expect("nil prevote for the timed-out round");
    assert_eq!(nil.kind, VoteKind::Prevote);
    assert_eq!(nil.round, 0);
    assert_eq!(nil.block_hash, "");
    state.on_vote(&sign_intent(&keys, &local, &nil));

    let leader = state.leader_for(1, 1).to_string();
    assert_eq!(leader, local, "round 1 leader is the next sorted validator");
    let proposal = Proposal::sign(CHAIN, 1, 1, HASH_A, &leader, find_key(&keys, &leader));
    let actions = state.on_proposal(&proposal, true);
    let prevote = vote_intent(&actions).expect("prevote for the round 1 proposal");
    assert_eq!(prevote.round, 1);
    assert_eq!(prevote.block_hash, HASH_A);
    state.on_vote(&sign_intent(&keys, &local, &prevote));

    let online: Vec<String> = state
        .validators()
        .iter()
        .filter(|validator| **validator != offline && **validator != local)
        .cloned()
        .collect();
    assert_eq!(online.len(), 2, "three validators are online for quorum");

    let mut precommit = None;
    for validator in &online {
        let actions = state.on_vote(&sign_vote(
            &keys,
            validator,
            VoteKind::Prevote,
            1,
            1,
            HASH_A,
        ));
        if let Some(intent) = vote_intent(&actions) {
            precommit = Some(intent);
        }
    }
    let precommit = precommit.expect("precommit after the prevote quorum");
    assert_eq!(precommit.kind, VoteKind::Precommit);
    assert_eq!(precommit.block_hash, HASH_A);
    state.on_vote(&sign_intent(&keys, &local, &precommit));

    let mut result = None;
    for validator in &online {
        let actions = state.on_vote(&sign_vote(
            &keys,
            validator,
            VoteKind::Precommit,
            1,
            1,
            HASH_A,
        ));
        if let Some(finalized) = finalized(&actions) {
            result = Some(finalized);
        }
    }

    assert_eq!(result, Some((1, HASH_A.to_string())));
    assert_eq!(state.finalized_height(), 1);
    assert_eq!(state.finalized_hash(), HASH_A);
    let certificate = state.certificate().expect("commit certificate");
    assert_eq!(certificate.votes.len(), 3);
    certificate.verify(state.validators()).unwrap();
}

#[test]
fn repeated_rounds_backoff_and_keep_advancing_past_r_max() {
    let (mut state, keys) = setup(0);
    let local = state.local().to_string();

    for _ in 0..(R_MAX + 3) {
        let actions = state.on_timeout();
        assert!(actions.iter().any(|action| matches!(
            action,
            Action::StartRound {
                height: 1,
                round,
                ..
            } if *round == state.round()
        )));
        if let Some(intent) = vote_intent(&actions) {
            state.on_vote(&sign_intent(&keys, &local, &intent));
        }
    }

    assert_eq!(state.round(), R_MAX + 3);
    assert_eq!(state.timeout_ms(), state.timeout_ms_for(R_MAX));
    let actions = state.on_timeout();
    assert_eq!(state.round(), R_MAX + 4);
    let schedule = actions
        .iter()
        .find_map(|action| match action {
            Action::StartRound { timeout_ms, .. } => Some(*timeout_ms),
            _ => None,
        })
        .expect("StartRound action");
    assert_eq!(schedule, state.timeout_ms_for(R_MAX));
    assert!(schedule > state.timeout_ms_for(0));
}

#[test]
fn pacemaker_backoff_is_deterministic_and_capped() {
    let mut pacemaker = Pacemaker::new(1_000);
    assert_eq!(pacemaker.timeout_ms_at(0), 1_000);
    assert_eq!(pacemaker.timeout_ms_at(1), 1_500);
    assert_eq!(pacemaker.timeout_ms_at(2), 2_250);
    assert_eq!(pacemaker.timeout_ms_at(3), 3_375);

    let at_max = pacemaker.timeout_ms_at(R_MAX);
    assert_eq!(pacemaker.timeout_ms_at(R_MAX + 1), at_max);
    for _ in 0..(R_MAX + 10) {
        pacemaker.on_timeout();
    }
    assert_eq!(pacemaker.current_round(), R_MAX + 10);
    assert_eq!(
        pacemaker.schedule(),
        Schedule::Timeout {
            height: 1,
            round: R_MAX + 10,
            timeout_ms: at_max,
        }
    );

    let capped = Pacemaker::with_config(
        PacemakerConfig::new(1_000)
            .with_timeout_cap(5_000)
            .with_max_round(10),
    );
    assert_eq!(capped.timeout_ms_at(4), 5_000);
    assert_eq!(capped.timeout_ms_at(10), 5_000);
}

#[test]
fn lock_blocks_conflicting_same_round_proposal_and_higher_round_unlocks() {
    let (mut state, keys) = setup(1);
    let local = state.local().to_string();
    let round_zero_leader = state.leader_for(1, 0).to_string();

    let proposal_a = Proposal::sign(
        CHAIN,
        1,
        0,
        HASH_A,
        &round_zero_leader,
        find_key(&keys, &round_zero_leader),
    );
    let actions = state.on_proposal(&proposal_a, true);
    let prevote_a = vote_intent(&actions).expect("prevote a");
    state.on_vote(&sign_intent(&keys, &local, &prevote_a));

    for validator in state.validators().to_vec() {
        if validator == local {
            continue;
        }
        let actions = state.on_vote(&sign_vote(
            &keys,
            &validator,
            VoteKind::Prevote,
            1,
            0,
            HASH_A,
        ));
        if let Some(precommit) = vote_intent(&actions) {
            state.on_vote(&sign_intent(&keys, &local, &precommit));
        }
    }
    assert_eq!(state.lock().unwrap().block_hash, HASH_A);
    assert_eq!(state.lock().unwrap().round, 0);

    let conflicting = Proposal::sign(
        CHAIN,
        1,
        0,
        HASH_B,
        &round_zero_leader,
        find_key(&keys, &round_zero_leader),
    );
    let actions = state.on_proposal(&conflicting, true);
    assert!(!actions
        .iter()
        .any(|action| matches!(action, Action::Vote(_))));
    match actions.first() {
        Some(Action::Rebroadcast(vote)) => {
            assert_eq!(vote.kind, VoteKind::Prevote);
            assert_eq!(vote.block_hash, HASH_A);
        }
        other => panic!("expected rebroadcast of the round-0 prevote, got {other:?}"),
    }

    let actions = state.on_timeout();
    assert_eq!(state.round(), 1);
    if let Some(intent) = vote_intent(&actions) {
        state.on_vote(&sign_intent(&keys, &local, &intent));
    }

    let round_one_leader = state.leader_for(1, 1).to_string();
    let proposal_b = Proposal::sign(
        CHAIN,
        1,
        1,
        HASH_B,
        &round_one_leader,
        find_key(&keys, &round_one_leader),
    );
    let actions = state.on_proposal(&proposal_b, true);
    let prevote_b = vote_intent(&actions).expect("higher round unlocks and prevotes b");
    assert_eq!(prevote_b.round, 1);
    assert_eq!(prevote_b.block_hash, HASH_B);
    state.on_vote(&sign_intent(&keys, &local, &prevote_b));

    let mut precommitted = false;
    for validator in state.validators().to_vec() {
        if validator == local {
            continue;
        }
        let actions = state.on_vote(&sign_vote(
            &keys,
            &validator,
            VoteKind::Prevote,
            1,
            1,
            HASH_B,
        ));
        if actions
            .iter()
            .any(|action| matches!(action, Action::Vote(_)))
        {
            precommitted = true;
        }
    }
    assert!(precommitted, "expected a precommit at the higher round");
    assert_eq!(state.lock().unwrap().block_hash, HASH_B);
    assert_eq!(state.lock().unwrap().round, 1);
}

#[test]
fn higher_round_precommit_certificate_unlocks_a_stale_lock() {
    let (mut state, keys) = setup(1);
    let local = state.local().to_string();
    let round_zero_leader = state.leader_for(1, 0).to_string();
    let proposal_a = Proposal::sign(
        CHAIN,
        1,
        0,
        HASH_A,
        &round_zero_leader,
        find_key(&keys, &round_zero_leader),
    );
    let actions = state.on_proposal(&proposal_a, true);
    let prevote_a = vote_intent(&actions).expect("prevote a");
    state.on_vote(&sign_intent(&keys, &local, &prevote_a));
    for validator in state.validators().to_vec() {
        if validator == local {
            continue;
        }
        let actions = state.on_vote(&sign_vote(
            &keys,
            &validator,
            VoteKind::Prevote,
            1,
            0,
            HASH_A,
        ));
        if let Some(precommit) = vote_intent(&actions) {
            state.on_vote(&sign_intent(&keys, &local, &precommit));
        }
    }
    assert_eq!(state.lock().unwrap().block_hash, HASH_A);

    let actions = state.on_timeout();
    assert_eq!(state.round(), 1);
    if let Some(intent) = vote_intent(&actions) {
        state.on_vote(&sign_intent(&keys, &local, &intent));
    }

    let mut result = None;
    for validator in state.validators().to_vec() {
        if validator == local {
            continue;
        }
        let actions = state.on_vote(&sign_vote(
            &keys,
            &validator,
            VoteKind::Precommit,
            1,
            1,
            HASH_B,
        ));
        if let Some(finalized) = finalized(&actions) {
            result = Some(finalized);
        }
    }
    assert_eq!(result, Some((1, HASH_B.to_string())));
    assert_eq!(state.finalized_hash(), HASH_B);
    assert_eq!(state.lock().unwrap().block_hash, HASH_B);
}

#[test]
fn restart_restores_own_vote_and_refuses_conflicting_votes() {
    let (mut state, keys) = setup(1);
    let local = state.local().to_string();
    let leader = state.leader_for(1, 0).to_string();
    let proposal = Proposal::sign(CHAIN, 1, 0, HASH_A, &leader, find_key(&keys, &leader));
    let actions = state.on_proposal(&proposal, true);
    let prevote = vote_intent(&actions).expect("prevote intent");
    let signed = sign_intent(&keys, &local, &prevote);

    let mut restarted =
        ConsensusState::new(CHAIN, state.validators().to_vec(), &local, 1, 1_000).unwrap();
    restarted.restore_finalized(0, "");
    restarted.restore_own_vote(signed.clone()).unwrap();

    let actions = restarted.on_proposal(&proposal, true);
    assert!(!actions
        .iter()
        .any(|action| matches!(action, Action::Vote(_))));
    match actions.first() {
        Some(Action::Rebroadcast(vote)) => {
            assert_eq!(vote.signature, signed.signature);
            assert_eq!(vote.block_hash, HASH_A);
        }
        other => panic!("expected the persisted vote to be rebroadcast, got {other:?}"),
    }

    let other_proposal = Proposal::sign(CHAIN, 1, 0, HASH_B, &leader, find_key(&keys, &leader));
    let actions = restarted.on_proposal(&other_proposal, true);
    assert!(!actions
        .iter()
        .any(|action| matches!(action, Action::Vote(_))));
    match actions.first() {
        Some(Action::Rebroadcast(vote)) => assert_eq!(vote.signature, signed.signature),
        other => panic!("expected the stored vote, got {other:?}"),
    }

    let conflicting = sign_vote(&keys, &local, VoteKind::Prevote, 1, 0, HASH_B);
    assert!(restarted.restore_own_vote(conflicting).is_err());
    assert_eq!(
        restarted
            .own_vote(1, 0, VoteKind::Prevote)
            .unwrap()
            .block_hash,
        HASH_A
    );
}

#[test]
fn invalid_or_wrong_leader_proposals_prevote_nil() {
    let (mut state, keys) = setup(1);
    let leader = state.leader_for(1, 0).to_string();
    let not_leader = state
        .validators()
        .iter()
        .find(|validator| **validator != leader)
        .unwrap()
        .clone();

    let wrong_leader = Proposal::sign(
        CHAIN,
        1,
        0,
        HASH_A,
        &not_leader,
        find_key(&keys, &not_leader),
    );
    let actions = state.on_proposal(&wrong_leader, true);
    let nil = vote_intent(&actions).expect("nil prevote for a non-leader proposal");
    assert_eq!(nil.block_hash, "");

    let (mut state, keys) = setup(1);
    let leader = state.leader_for(1, 0).to_string();
    let mut bad_signature = Proposal::sign(CHAIN, 1, 0, HASH_A, &leader, find_key(&keys, &leader));
    bad_signature.signature = "00".repeat(64);
    let actions = state.on_proposal(&bad_signature, true);
    let nil = vote_intent(&actions).expect("nil prevote for a bad envelope signature");
    assert_eq!(nil.block_hash, "");

    let (mut state, keys) = setup(1);
    let leader = state.leader_for(1, 0).to_string();
    let empty_hash = Proposal::sign(CHAIN, 1, 0, "", &leader, find_key(&keys, &leader));
    let actions = state.on_proposal(&empty_hash, true);
    let nil = vote_intent(&actions).expect("nil prevote for an empty proposal hash");
    assert_eq!(nil.block_hash, "");

    let (mut state, keys) = setup(1);
    let leader = state.leader_for(1, 0).to_string();
    let valid = Proposal::sign(CHAIN, 1, 0, HASH_A, &leader, find_key(&keys, &leader));
    assert!(state
        .on_proposal(&valid, false)
        .iter()
        .any(|action| matches!(action, Action::Vote(intent) if intent.block_hash.is_empty())));
}

#[test]
fn never_votes_at_or_below_the_finalized_height() {
    let (mut state, keys) = setup(1);
    let local = state.local().to_string();
    state.restore_finalized(1, "final-hash");

    let leader = state.leader_for(1, 0).to_string();
    let proposal = Proposal::sign(CHAIN, 1, 0, HASH_A, &leader, find_key(&keys, &leader));
    assert!(state.on_proposal(&proposal, true).is_empty());
    assert!(state.on_validated_block(HASH_A).is_empty());
    assert!(state.on_timeout().is_empty());
    assert!(state.own_vote(1, 0, VoteKind::Prevote).is_none());
    assert!(state.start_round(1, 0).is_err());

    let stale = sign_vote(&keys, &local, VoteKind::Prevote, 1, 0, HASH_A);
    assert!(state.restore_own_vote(stale).is_err());
}
