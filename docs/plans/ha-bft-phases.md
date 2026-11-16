# HA, Snapshot, Validator Set, and BFT Phases

Status: implementation plan only. No snapshot transfer, automatic failover, on-chain validator-set change, or BFT finality exists in this repository. Date: 2026-11-16. Scope: `crates/node`, `crates/state`, `crates/exec`, `crates/network`, a new `crates/consensus`, `scripts/`, and operations docs. ADR 004, 006, and 007 are the design source; this plan orders work, separates implementable-now items from protocol work, and defines gates. It does not rewrite the ADRs.

## Baseline

- Consensus membership is the configured list: `Shared::allowed_validators` (`crates/node/src/main.rs:115`) returns `params.validators`. `RegisterValidator`/`UnregisterValidator` mutate `val:` (`crates/exec/src/engine.rs:905-950`) but do not change consensus.
- Genesis entries are seeded with `genesis: true`, `staked_amount: 0`, and an empty `public_key` (`crates/state/src/lib.rs:212-225`; `ValidatorEntry`, `crates/state/src/validators.rs:11-22`).
- `compute_state_root` (`crates/exec/src/engine.rs:981`) covers accounts, models, pending tasks, completed result hashes, and compute nodes. It does not cover `val:` or `meta:` set parameters.
- A block is final because one producer signed it: `validate_block` (`engine.rs:222`), `commit_block` (`crates/node/src/main.rs:179`), `set_head`. There is no prevote, precommit, or certificate.
- Catch-up replays blocks: `sync_once` (`main.rs:780`) polls `/status`, fetches `/blocks/since/:index`, and calls `admit_block`/`commit_block`. A fresh node starts at genesis, so join time is O(chain length).
- HA is manual. `/ready` (`crates/node/src/rpc.rs:136`) exposes syncing, peer lag, and validator-set consistency, but no lease or fencing.
- There is no snapshot endpoint, epoch activation, `BindValidatorKey`, distributed lease, or BFT code.

## Workstreams

| Workstream | Implementable now? | Depends on | Estimate |
|---|---|---|---|
| State-root coverage and set params | Yes, gated hard fork | None | 1-1.5 wk |
| Snapshot create/verify/apply | Yes | Checkpoint API | 1-2 wk |
| Snapshot serving endpoints | Yes | Snapshot format | 3-5 d |
| HA preflight, promotion script, duplicate-key check | Yes | `/ready` | 1 wk |
| Validator key binding and activation/deactivation | Partly; needs hard fork | State-root coverage | 2-3 wk |
| Distributed lease and automatic promotion | No; new dependency | Lease backend | 2-3 wk |
| BFT finality | No; protocol work | ADR 004/007, state refactor | 5.5-7 wk |
| Evidence-driven removal/slashing | No; v2 | BFT + bound keys + bonds | 3-6 wk after BFT |

## Phase 1 — State-root coverage (prerequisite)

Do this before any set change or snapshot trust claim. Without it, two nodes can disagree about the validator set without a state-root mismatch.

- Add sorted `val:<address>` entries to `compute_state_root`: address, public key, stake, `genesis`, `registered_at_height`, and the new activation/deactivation/jail/key fields. Add `meta:validatorSetParams` (epoch length, activation delay, minimum set, minimum stake, version).
- Keep the existing `validatorSetHash` observability value (`crates/node/src/main.rs:148`) separate until the root covers the set. `/ready` mismatch remains a fail-closed signal, not a consensus rule.
- Decide the root-version mechanism before coding. Changing the root changes every block hash immediately. Options: (a) `HEADER_VERSION = 2` with a `validatorRoot`, or (b) a `VNN_STATE_ROOT_VERSION` switch activated at a coordinated height. Pick one and make old/new root selection a pure function of height so mixed binaries cannot disagree silently.
- Do not include `meta:head` or anything derived from the head (circular). Include `game:`/`claim:` only when ADR 005 is enabled.
- Tests: two stores with different `val:` entries produce different roots; a node omitting `val:` is rejected; pre-activation root equals the old root; activation height switches exactly once; conformance vectors for both versions.

## Phase 2 — Snapshot sync

### Manifest and layout

```text
snapshot-<height>/
  manifest.json
  manifest.sig
  state/                    # RocksDB checkpoint, uncompressed in v1
  blocks/recent.json        # headers for the last challenge/finality window
```

`manifest.json` is canonical JSON: `{v, kind:"verinoo-state-snapshot", chainId, protocolVersion, height, blockHash, parentHash, stateRoot, baseFee, finalized, commitCertificate, modelSetHash, modelHashes, validatorSetHash, validatorSet, createdAt, compression:"none", files:[{path,size,sha256}], signers, signatures}`. `snapshotId = sha256(canonical manifest without signatures)`. Sign domain `verinoo/snapshot/v1`. `stateRoot` is recomputed, never trusted. Reject unknown fields/files and duplicate signers.

### Trust model without BFT

- The manifest is operator-attested. Pin one or more Ed25519 keys in `VNN_SNAPSHOT_SIGNERS`; require the configured threshold (v1: 1-of-1 or 1-of-N). Publish the key out of band. This is not trustless: a verifier can recompute the state root and check the block/parent hashes, but it cannot prove the state is canonical without a trusted checkpoint.
- Set `finalized: false` and `commitCertificate: null` until BFT exists. A fresh node still verifies every block after the snapshot normally.
- Pin `chainId` and the genesis hash in the node config. Reject a snapshot for another chain or a parent that does not chain to the pinned genesis history.
- With BFT (Phase 5), require a validator quorum over the manifest and a commit certificate for `height`; verify both exactly as the finalized sync path does. Only then may `finalized: true`.

### Checkpoint, hashes, verification

- Add `StateStore::checkpoint_to(&Path)` using `rocksdb::checkpoint::Checkpoint` (the `db` field at `crates/state/src/lib.rs:25` is private). Run `backfill_height_index` (`crates/node/src/main.rs:691`) or verify `blockheight:<height>` exists before checkpointing.
- Capture the head before and after the checkpoint. If it moved, retry; a snapshot must bind exactly one `height`/`blockHash`. Prefer a stopped-service snapshot for v1; a live checkpoint is allowed only if this check is enforced.
- Compute per-file SHA-256 and size after the checkpoint is closed. The manifest lists every file; a file not listed is rejected; a missing file is rejected. Verify all hashes before opening the snapshot as a database.
- Verification order: manifest signature/threshold, chain id/version, `snapshotId`, files, `stateRoot` recompute, `height`/`blockHash`/`parentHash`, `baseFee`, `modelSetHash`/`modelHashes`, `validatorSetHash`/`validatorSet`, then BFT certificate if present. Any failure leaves the live database untouched.

### Download, apply, rollback

- Add CLI subcommands in `crates/node/src/main.rs`: `snapshot create`, `snapshot verify`, `snapshot apply`. Do not add an apply RPC; applying requires stopping the node.
- Download into `snap-<height>.partial`; support resume by file; never open a partial directory as a live database. Verify each file before it is moved into place.
- Stop the node. Rename the live DB to `data.pre-snapshot-<height>` (do not delete), rename the verified directory into `VNN_DB`, start the node, and confirm `/status` height/hash. Keep the old directory until at least one new block applies and the state root matches.
- Reject a snapshot at or below the local finalized/head height. Same height with a different hash is an incident, not a sync choice. Never roll back to a lower height; rollback means restoring the retained old directory after stopping the node.
- After apply, `sync_once` starts at the snapshot height. `/blocks/since/:index` remains the steady-state path; add `/finalized/since/:index` in BFT mode.
- Malicious omission of history is allowed; a snapshot cannot silently change state at `height` if the root, model set, and validator set are checked. It can still serve a non-canonical branch unless the signer/quorum is trusted.

### Endpoints and operations

- `GET /snapshot/manifest` and `GET /snapshot/files/:name` (range support, path sanitization, no traversal) in `crates/node/src/rpc.rs`; optional `GET /snapshot/status`.
- Serve only when `/ready` is 200, `syncing == false`, and the node has the snapshot files. Add `VNN_SNAPSHOT_DIR`, `VNN_SNAPSHOT_TOKEN`, and rate limits. Do not serve while applying.
- Runbook: create on a stopped or quiesced node; verify locally; publish manifest + files; on a fresh node download, verify, stop, swap, start, watch `/status` and `/ready`; if anything fails, restore the retained directory.
- Tests: checkpoint/apply round-trip; tampered file; tampered `stateRoot`; wrong `modelSetHash`; wrong `validatorSetHash`; lower/stale height; same-height different hash; interrupted download/resume; open `game:`/`claim:` preserved; post-snapshot block replay matches a from-genesis node; unauthorized/threshold-missing signature rejected.

## Phase 3 — On-chain validator activation and removal

### State schema

Extend `ValidatorEntry` with serde defaults for backward compatibility: `activation_height: u64`, `deactivation_height: Option<u64>`, `jailed: bool`, `key_epoch: u64`, `bound_at_height: u64`. Existing genesis entries read as active at height 0 with an empty key. Store epoch parameters in `meta:validatorSetParams`; they are consensus state once included in the root.

### Transactions and rules

- `RegisterValidator`: `public_key_to_address(tx.public_key) == tx.from` (same rule as blocks, `engine.rs:23-32`), `stakedAmount >= MIN_VALIDATOR_STAKE` (`crates/state/src/validators.rs:8`), the address is not already active/pending, and `activationHeight >= currentHeight + activationDelay` aligned to an epoch boundary. Debit the bond at inclusion.
- `BindValidatorKey { publicKey }` (new `TxKind`): required for genesis entries before they may vote in BFT or be slashed. One key per address; a second binding is rejected unless it is a rotation signed by both old and new keys with a delay. Store the key, `bound_at_height`, and `key_epoch`.
- `UnregisterValidator`: only the validator's own address may sign; mark `deactivating` instead of deleting; keep the bond locked until `deactivationHeight` and until every open challenge/game involving it resolves. Refund at deactivation, not inclusion. Reject if the active set would fall below `minValidatorSet` or below the quorum needed to finalize the change.
- Activation/deactivation happen only at epoch boundaries. The active set for height `H` is computed deterministically from entries with `activation_height <= H` and no deactivation at or before `H`, sorted by address. Quorum for `H` is computed from the set committed before the epoch that contains `H`; a transaction in a block must not change the quorum that validates it.
- Bonds: genesis validators have zero bond and cannot be economically slashed. New validators bond real stake. Do not claim slashing until BFT evidence, bound keys, and bonded stake all exist.

### State root, migration, and compatibility

- Include sorted `val:` entries (all fields) and `meta:validatorSetParams` in the root at the activation height, per Phase 1. If a header `validatorRoot` is chosen instead, bump the header version and document the hard fork.
- Add `VNN_VALIDATOR_SET_MODE=genesis|state` and `VNN_VALIDATOR_SET_ACTIVATION_HEIGHT`. Before activation, bind every genesis key and confirm `validatorSetHash` on all nodes. At activation, the state-derived set becomes authoritative; mixed binaries or mixed modes are unsupported.
- Adding a `TxKind` variant changes canonical tx ids and makes old nodes reject the new kind. Ship the binary everywhere before the activation height; update conformance fixtures and SDK enum handling.
- Rollback is a coordinated incident: stop the set, restore the pre-activation database and binary, restart in genesis mode. Do not roll back after a set change has finalized.
- Tests: key/address derivation; minimum stake; activation boundary; deactivation boundary; last-validator rejection; below-quorum rejection; genesis key binding and rotation; bond lock/refund; open game liability; reorg around activation; two nodes with different `val:` roots reject each other; old root before activation, new root after.

## Phase 4 — Fencing and HA

The invariant is one validator key, one live process. The node does not enforce it today: `validator_loop` (`crates/node/src/main.rs:210`) produces in its scheduled slot, and two hosts with the same `VNN_KEY_SEED` can sign different blocks at the same height. Under BFT that becomes attributable equivocation evidence; it is still a safety violation.

- **Startup duplicate check (implement now).** As a validator, poll `VNN_SYNC_URLS`/configured peers' `/status` before producing. Refuse to start as a validator if a peer reports the same validator address at a different height or head hash. This is a heuristic, not a lock; an old process at the same height is invisible.
- **Per-height production record (implement now).** Persist `produced:<height>` with the signed block hash before broadcasting; refuse to sign a different hash at the same height after a restart. This catches amnesia, not two concurrent processes.
- **Manual fencing (implement now).** Promotion requires: `/ready` 200; replica height and head hash equal to the active node; `syncing == false`; `validatorSetConsistent == true`; stop the old process and disable its supervisor; fence the host (power/network/instance stop) so it cannot restart. Add `scripts/promote-replica.sh` that refuses while the old process answers or heights differ. Update DNS, the load balancer, or `VNN_SEEDS`; watch the first produced block.
- **Distributed lease (multi-week).** Acquire a TTL lease keyed by validator address (etcd/Consul/cloud instance lock) with renewal; produce only while held; stop producing on lease loss. This is the first safe basis for automatic promotion. It is a new operational dependency and is not implemented.
- **Split-brain detection (implement now).** Alert if two hosts report the same validator at different head hashes, or if a node logs `wrong producer` for a height it did not produce. Under BFT, record the evidence; detection is not prevention.
- Safe automation now: read-only load-balancer failover, replica catch-up, promotion preflight, alerting, backups. Not safe yet: automatic write promotion, key handoff, or un-fenced restart. Do not call a manual promotion "automatic failover."
- Tests: promotion script refuses while the old process is alive; refuses a lagging replica; succeeds after fencing; duplicate seed on two hosts is detected or the test documents the observed fork; stale replica is removed from `/ready`; lease expiry stops production (after the lease exists).

## Phase 5 — BFT finality

Use ADR 004 option (b), the minimal 2/3 prevote/precommit gadget over existing gossipsub and `validate_block`; ADR 007 is the execution checklist. This phase does not replace libp2p or RocksDB.

### Execution order and integration

- **Phase 0: state refactor.** Add `StateStore::checkpoint_to`; build/verify proposals on a scratch checkpoint; move `baseFee` advancement out of `produce_block` (`crates/exec/src/engine.rs:1467`) into commit; split `commit_block` (`crates/node/src/main.rs:179`) into `verify_certificate` and `commit_finalized`; add `RoundAware { round }` beside `RoundRobin` (`crates/exec/src/leader.rs:15`); only `commit_finalized` calls `set_head`.
- **Phase 1: vote types.** New `crates/consensus`: `Proposal`, `Vote` (`prevote`/`precommit`), `CommitCertificate`, `Evidence`, `VoteKey`; canonical signed payloads and domain separators exactly as ADR 004; reject unknown version/type, wrong chain id, and keys that do not derive to the validator address.
- **Phase 2: state machine.** Propose/prevote/precommit with nil votes on timeout; Tendermint lock/unlock; persist `consensusLock`, `consensusOwnVote:<H>:<r>:<kind>`, and `consensusRound` before broadcasting; reload and re-broadcast the stored vote verbatim after restart; never vote at a height at or below `finalizedHeight`.
- **Phase 3: wire and actor.** Add `PROPOSALS_TOPIC` and `VOTES_TOPIC` (`crates/network/src/lib.rs:23-24`), `NetworkEvent` variants, and publish commands; one consensus actor owns vote/lock/round state; vote cache keyed by `(height, round, kind, validator)`; height window 64, future-round window 32; per-peer token bucket and size caps before signature verification.
- **Phase 4: finality gating.** Quorum `floor(2N/3)+1` over distinct validators, all precommits bound to the same `(height, round, blockHash)`; only `commit_finalized` applies and advances; persist `finalizedHeight`, `finalizedHash`, and `meta:commit:<H>` on finality; a commit-time state-root mismatch logs and halts rather than advancing; add `/finalized/since/:index`, `finalizedHeight`/`finalizedHash` on `/status`, and metrics.
- **Phase 5: evidence and sync.** Store `evid:<sha256(canonical evidence)>` only after both signatures verify; expose `/evidence` and `/evidence/:validator`; certificate-verified `sync_once`; no automatic removal or fund movement in v1.
- **Phase 6: activation.** `VNN_BFT_ACTIVATION_HEIGHT`; all validators restart together at a finalized height; mixed legacy/BFT sets unsupported; keep the legacy path for one release; rollback restarts everyone in legacy mode and certificates become advisory.

### Receipts and block production

Under BFT the canonical store represents finalized state. `apply_block` stamps receipts (`crates/exec/src/engine.rs:1314`), and `produce_block` also stamps them (`:1458`). Proposal execution must run on a scratch checkpoint and must not write canonical `receipt:` keys or move the head. Only `commit_finalized` applies the block, compares the root, calls `set_head`, and stamps receipts. Add `finalized: true`, `commitHeight`, and a certificate hash/reference to the receipt (or keep the certificate under `commit:<H>` and reference it). `/v1/verify` must return `finalized: false` before finality and under legacy mode, and `finalized: true` only with a verified certificate.

### Testnet plan

- T0 in-process: 4 deterministic validators, manual clock, a bus supporting drop/duplicate/reorder/partition; run the ADR 007 matrix (happy path, one Byzantine, partition/rejoin, one offline, late join, invalid proposal, restart/amnesia, double vote, safety property). No sockets.
- T1 4-node local devnet: real gossipsub, 1 s blocks, kill one node, partition with firewall rules, restart a validator, join a fresh node from a snapshot plus certificates.
- T2 public testnet: 4 independent operators, two-week soak, evidence drill, key-rotation drill, snapshot join, rolling restart. Publish `validatorSetHash` and activation height before activation.
- T3 pilot: only after T0-T2 are green and an independent review of the lock/persistence rules. Do not claim "BFT" or "final" for N<=3; N=4 (quorum 3) is the minimum that tolerates one Byzantine fault.

### Safety invariants

1. No two nodes finalize different hashes at the same height.
2. At most one prevote and one precommit per `(height, round, kind)` per validator.
3. Never vote at a height at or below `finalizedHeight`; never precommit a hash not prevoted.
4. A certificate has a quorum of distinct validators, all bound to one `(height, round, blockHash)`.
5. A block is applied only after the certificate verifies and the recomputed state root matches.
6. Own votes are persisted before broadcast and re-broadcast verbatim after restart.
7. Receipts are written only from the finalized commit path.
8. Validator-set changes, if enabled, take effect only at an epoch boundary and quorum for the boundary comes from the prior committed set.

## Implementable now vs multi-week

| Item | Wave | Estimate | Gate/risk |
|---|---|---|---|
| State-root coverage and root versioning | 1 | 1-1.5 wk | Hard fork; old/new root must be height-deterministic |
| Snapshot create/verify/apply CLI | 1 | 1-2 wk | RocksDB checkpoint + atomic swap |
| Snapshot serving endpoints | 1 | 3-5 d | Path sanitization, auth, ready-only |
| Promotion preflight, duplicate-key check, production record | 1 | 1 wk | Heuristic only; no lease |
| BFT Phase 0 state refactor | 1-2 | 1 wk | Touches `produce_block`/`commit_block`; legacy path must stay green |
| Validator key binding + activation/deactivation | 2 | 2-3 wk | New `TxKind`; hard fork |
| Distributed lease + guarded automatic promotion | 2 | 2-3 wk | New etcd/Consul dependency |
| BFT minimal finality | 3 | 5.5-7 wk | Lock/persistence correctness; ADR 007 matrix |
| BFT evidence + snapshot certificate integration | 3 | 1-2 wk | No slashing |
| Evidence-driven removal/slashing v2 | 4 | 3-6 wk | Needs bound keys, real bonds, BFT |
| Automatic slashing of genesis validators | Not scheduled | — | Zero bond and no bound key today |

## Risks and non-claims

- **State-root hard fork.** Adding `val:`/params to the root changes every block hash. A mixed rollout forks or halts the chain; gate by height and test both root versions.
- **Snapshot trust without BFT is operator attested, not trustless.** A pinned signer can sign a non-canonical branch. The state root check proves internal consistency, not canonicality.
- **Checkpoint/apply is filesystem-sensitive.** A crash between renames, a partial download, or an open DB in the wrong directory can lose the node. Keep the old directory and never delete it until a new block applies.
- **Validator-set activation is a hard fork.** New `TxKind`, root schema, and quorum rules must ship everywhere before the activation height. Rollback after a finalized set change is not safe.
- **No automatic HA.** A lease is the first safe primitive; without it, promotion is manual and fenced. Same-key double signing is detected, at best, after the fact.
- **BFT liveness is not guaranteed.** Gossipsub is unreliable; ≥2/3 must be online; N=2 needs both, N=3 tolerates zero faults. A partition stalls finality by design.
- **No slashing or automatic removal in v1.** Evidence is stored and exposed; funds do not move and quorum does not change. Genesis validators have no bond and no bound key, so there is nothing to burn.
- **No snapshot transfer today, no on-chain set change today, no BFT finality today.** Do not cite this plan as shipped behavior.
