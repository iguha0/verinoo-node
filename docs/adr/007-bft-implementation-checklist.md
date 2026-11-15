# ADR 007: BFT Implementation Checklist

## Status

**Proposed checklist — not implemented.** ADR 004
(`docs/adr/004-bft-finality.md`) is the design; this file is the execution
order, review checklist, and claim boundary. It does not replace ADR 004.
Date: 2026-11-15. Scope: `crates/consensus` (new), `crates/node`,
`crates/exec`, `crates/network`, `crates/state`.

## Phase 0 — prerequisites

- [ ] `StateStore::checkpoint_to` via `rocksdb::checkpoint::Checkpoint`; build
      and validate proposals on a scratch checkpoint and compare
      `compute_state_root` before any vote.
- [ ] Move `baseFee` advancement out of `produce_block`
      (`crates/exec/src/engine.rs:1467`) into the commit path; split
      `commit_block` (`crates/node/src/main.rs:130`) into
      `verify_certificate` and `commit_finalized` (only the latter calls
      `set_head`).
- [ ] Add `RoundAware { round }` leader policy beside `RoundRobin`
      (`crates/exec/src/leader.rs`).
- [ ] Add `validatorSetHash` to `/status`; refuse to vote on a mismatch.
- [ ] Keep `VNN_CONSENSUS=legacy|bft`; default `legacy`.

## Phase 1 — vote types and signing

- [ ] New `crates/consensus` crate: `Proposal`, `Vote` (`prevote` /
      `precommit`), `CommitCertificate`, `Evidence`, `VoteKey`.
- [ ] Canonical signed payloads and domain separators exactly as ADR 004;
      fields `chainId`, `height`, `round`, `blockHash`, `validator`,
      `publicKey`, `signature`, `v`; empty `blockHash` means nil.
- [ ] Reject unknown `v`, unknown `type`, wrong chain id, and a public key
      that does not derive to the validator address.
- [ ] Unit tests for canonical bytes and signature verification.

## Phase 2 — round state machine and persistence

- [ ] `propose -> prevote -> precommit` phases; nil votes on timeout.
- [ ] Tendermint lock/unlock rules exactly as ADR 004 "Lock and unlock rules".
- [ ] Persist `finalizedHeight`, `finalizedHash`, `consensusLock`,
      `consensusOwnVote:<H>:<r>:<kind>`, `consensusRound`, and
      `commit:<H>` before broadcasting.
- [ ] On restart, reload and re-broadcast the stored own vote verbatim; refuse
      a second signature for the same `(H, r, kind)`.
- [ ] Manual clock; deterministic timers (`min(base * 1.5^r, cap)`),
      `R_MAX = 8`; never vote at a height <= `finalizedHeight`.

## Phase 3 — wire and actor

- [ ] Add `PROPOSALS_TOPIC = "verinoo/proposals/1"` and
      `VOTES_TOPIC = "verinoo/votes/1"`, chain-prefixed like the existing
      topics (`crates/network/src/lib.rs:23-24,217-220`).
- [ ] `NetworkEvent::Proposal` and `NetworkEvent::Vote`; publish commands;
      topic-suffix mapping in the swarm loop.
- [ ] A single consensus actor owns vote/lock/round state; the event pump only
      forwards. Vote cache keyed by `(height, round, kind, validator)`: same
      key + same hash is a duplicate, same key + different hash is evidence.
- [ ] Window: `finalizedHeight < H <= finalizedHeight + 64`; at most 32 future
      rounds queued.
- [ ] Per-peer token bucket (200 msg/s, burst 400) before signature
      verification; size caps (votes <= 4 KB, evidence <= 16 KB, proposals
      <= 1 MB).
- [ ] `flood_publish(true)` is accepted for N <= 10; document O(N^2).

## Phase 4 — finality gating and certificates

- [ ] Only `commit_finalized` applies a block and advances the head; remove
      `set_head` from the production path
      (`crates/node/src/main.rs:214`).
- [ ] `verify_certificate`: quorum `floor(2N/3)+1`, distinct validators, each
      precommit verified over the reconstructed payload, all bound to the same
      `(height, round, blockHash)`.
- [ ] Commit-time state-root mismatch logs an error and halts; never advance an
      unverified head.
- [ ] Persist `meta:commit:<H>`; expose `finalizedHeight`/`finalizedHash` on
      `/status`; export `verinoo_finalized_height`,
      `verinoo_consensus_round`, `verinoo_consensus_votes_total`.
- [ ] Legacy `commit_block` remains reachable behind the flag for rollback.

## Phase 5 — sync, evidence, and operations

- [ ] `GET /finalized/since/:index` returning `{block, certificate}` pairs;
      `sync_once` verifies certificates before apply.
- [ ] Keep `/blocks/since` unchanged for legacy mode; in BFT mode it serves
      finalized blocks only.
- [ ] `GET /evidence` and `GET /evidence/:validator`; store
      `evid:<sha256(canonical evidence)>` only after both signatures verify.
- [ ] Detect: two proposals for one `(H, r)`, two votes by one validator for
      one `(H, r)` with different hashes, and a precommit that differs from the
      validator's own prevote.
- [ ] Evidence gossip is deduplicated and rate-limited; no automatic removal
      or fund movement in v1.
- [ ] Document that removal is still a manual `VNN_VALIDATORS` change.

## Phase 6 — migration and activation

- [ ] `VNN_BFT_ACTIVATION_HEIGHT`; all validators restart together at a
      finalized height.
- [ ] Mixed legacy/BFT sets are unsupported; refuse to vote if peers disagree
      on `validatorSetHash`.
- [ ] Publish the activation height, binary version, and `validatorSetHash`
      before the restart.
- [ ] Rollback: restart all validators in `legacy` mode; certificates become
      advisory.
- [ ] Keep the legacy path compiled for one release.

## Test matrix

| Test | Setup | Pass condition |
|---|---|---|
| Happy path | 4 validators, no faults, 5+ heights | same `finalizedHeight`/hash; 3 distinct precommits per certificate; head advances once per height |
| One Byzantine | 4 validators, 1 equivocating leader or voter | honest 3 finalize one hash; evidence stored; `verinoo_equivocations_total == 1` |
| Partition/rejoin | 2/2 split, several rounds, heal | no finality while split; next leader finalizes after heal; old minority votes dropped |
| One offline | 4 validators, 1 offline | remaining 3 finalize (quorum 3); one offline + one Byzantine stalls (documented) |
| Late join | fresh store, sync from finalized height 6 | 6 certificates verified, 6 blocks applied, participates at height 7 |
| Invalid proposal | leader signs a wrong state root | honest nodes prevote nil; round 1 honest leader finalizes; canonical store unchanged |
| Restart/amnesia | sign a vote, restart, retry same tuple | stored vote returned; no second signature |
| Double vote | two signed prevotes, same `(H, r)`, different hashes | one `evid:` key; both signatures verify; validator remains in quorum (v1) |
| Safety property | randomized schedules, <= 1 Byzantine, >= 3 honest | no two nodes finalize different hashes at the same height |
| Sync with certificate | late node gets blocks + certificates | applies without replaying votes; rejects a bad quorum or bad signature |

Run the matrix in-process with a manual clock and a bus supporting drop,
duplicate, reorder, and partition; no sockets in unit tests.

## What not to claim until each phase is done

- After Phase 0: "consensus refactor ready", not "BFT".
- After Phase 1: "vote format implemented", not "network finality".
- After Phase 2: "round state machine tested in-process", not "live liveness".
- After Phase 3: "votes propagate", not "safe under partition".
- After Phase 4: "finality gating and certificates implemented", not "audited".
- After Phase 5: "evidence recorded", not "slashing" or "validator removal".
- After Phase 6: "BFT activated on this testnet", not "mainnet finality".
- Never claim "trustless", "instant finality", or "Byzantine fault tolerant" for
  N <= 3; N=4 (quorum 3) is the minimum that tolerates one Byzantine fault.
  Never claim validator-set changes, automatic HA, or snapshot transfer from
  this ADR; those are ADR 006, and they are not implemented either.
