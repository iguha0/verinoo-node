# ADR 006: HA, Snapshot Sync, and On-Chain Validator Removal

## Status

**Partially implemented — readiness observability and the state-layer snapshot
format.** `GET /ready`, the `syncing` flag, `peerCount`, `validatorSetHash`, and
`validatorSetConsistent` are implemented as described in "API failover, health
checks, DNS, and load balancer". The state layer also implements local snapshot
create/verify/apply (`crates/state/src/snapshot.rs`, documented in
`docs/snapshots.md`): RocksDB checkpointing, a hashed file manifest, strict
path and file verification, and atomic apply with rollback and a backup when
replacing an existing database. Node snapshot endpoints, manifest signing and
signature verification, state-root recomputation for snapshots using
`crates/exec`, and BFT certificates remain unimplemented. The rest of this
document remains a target design: no code in this repository implements
automatic failover, split-brain
protection, state snapshot transfer over the network, or a consensus
validator-set change. Do not cite anything else here as shipped behavior.

Date: 2026-11-15 (readiness surface implemented 2026-11-15; hardened 2026-11-16;
state-layer snapshot create/verify/apply added 2026-11-17)
Scope: `crates/node`, `crates/state`, `crates/exec`, `crates/network`,
`scripts/`, and operations documentation.

## Context

### Current constraints

- **The validator set is genesis-seeded.** `GenesisParams::validators`
  (`crates/state/src/genesis.rs:19-20`) is copied into `val:` entries by
  `seed_genesis_validators` (`crates/state/src/lib.rs:212-225`). Those entries
  have `genesis: true`, `staked_amount: 0`, and an **empty `public_key`**
  (`crates/state/src/lib.rs:217-223`; `ValidatorEntry` at
  `crates/state/src/validators.rs:11-22`).
- **Consensus membership is the config list, not state.**
  `Shared::allowed_validators` returns `params.validators`
  (`crates/node/src/main.rs:105-107`). `StateStore::active_validators()`
  (`crates/state/src/lib.rs:194-205`) exists but the live block path does not
  use it for admission or leader selection.
- **There is no removal.** `RegisterValidator` and `UnregisterValidator`
  mutate `val:` state (`crates/exec/src/engine.rs:905-950`), but they do not
  change `params.validators`, so a running network cannot add or remove a
  consensus participant without a coordinated restart and, today, a
  re-genesis. The only guard is "cannot remove last validator"
  (`crates/exec/src/engine.rs:939-943`).
- **The state root does not cover the validator set.**
  `compute_state_root` (`crates/exec/src/engine.rs:981-1027`) covers accounts,
  models, pending tasks, completed result hashes, and compute nodes. It does
  not cover `val:` entries, `meta:`, or `game:` records. A validator-set
  change is therefore not committed by the state root.
- **There is no snapshot transfer.** `sync_once`
  (`crates/node/src/main.rs:671-713`) asks a peer for `/status`, then replays
  `/blocks/since/:index` through `admit_block` and `commit_block`
  (`crates/node/src/main.rs:118-158`). A fresh node starts at the genesis
  block and replays every block, so join time is O(chain length).
- **Production is slot-based and stalls on a missed leader.** The validator
  loop (`crates/node/src/main.rs:161-229`) produces one block per tick when
  `RoundRobin::leader_at(height)` matches this validator. A block is rejected
  unless it comes from the scheduled producer
  (`crates/exec/src/engine.rs:295-305`). If that producer is offline, no other
  validator may produce height `H`, and height `H+1` cannot exist because
   `previous_hash` must equal the missing block. The chain stalls until the
   leader returns or the network is re-genesised with a different set. ADR 004's
   rounds/view changes are the protocol fix; this ADR covers the operational
   and set-management parts.
- **HA is manual.** `docs/operations.md` documents a catch-up replica and a
  manual promotion. There is no automatic failover, no leader lease, and no
  protection against running the same validator key on two hosts.
  `docs/readiness.md:72-78` and `docs/pilot.md:159-165` list these as gaps.

### What breaks today

1. A validator host that dies stops the chain at its next slot. A replica can
   be promoted manually, but promotion is a stop/start procedure with no
   fencing.
2. A node that is far behind replays from genesis. On a long chain, the join
   time and disk churn grow without bound.
3. A validator cannot be removed for compromise, downtime, or decommissioning
   without a coordinated restart. Genesis entries have no bound public key and
   no bond, so even evidence-based removal has nothing to act on.
4. Two API nodes can serve different heights. There is no readiness signal
   that says "this replica is caught up and safe to promote."

## HA and failover

### Active/passive replica

The supported shape is one active validator and one passive replica:

- The replica runs with `VNN_VALIDATOR=false`, the same `VNN_VALIDATORS` list,
  the same `VNN_MODEL_DIR` (or a verified model set), and
  `VNN_SYNC_URLS=<active API>`. It validates every replayed block and refuses
  a divergent state root, as `sync_once` does today.
- The replica may serve read APIs (`/status`, `/blocks`, `/account`,
  `/v1/tasks`, `/v1/verify`) and accept transactions. It must not produce
  blocks until promoted.
- A load balancer may send reads to either node only if it checks readiness and
  lag (`GET /ready`). A stale read is not a correctness failure, but it is a bad
  receipt experience.
- Inference submission is a transaction, so any node can accept it and gossip
  it (`NetworkEvent::Tx` handling at `crates/node/src/main.rs:546-559`). The
  block producer includes it if it has the tx. A replica that is behind may
  accept a task whose result is not yet visible; clients should poll a node at
  the chain head or wait for the receipt.

### Promotion and fencing

Promotion is a manual, fenced procedure. The exact commands belong in
`docs/operations.md`; the invariants are:

1. Confirm the replica's height and head hash equal the active node's height
   and head hash. Do not promote a replica that is behind.
2. Stop the active validator process and prevent it from restarting. Stopping
   the process is not enough if a supervisor or a second host can restart it.
   Use a service lock, remove it from the supervisor, fence the host's network
   or power, or use a cloud instance stop. The key invariant is: exactly one
   process with the validator key is alive.
3. Start the replica with `VNN_VALIDATOR=true` and the validator key. The
   replica's `VNN_SYNC_URLS` should point at another node, not itself.
4. Update DNS, the load balancer, or the static seed list so peers and clients
   reach the new host.
5. Watch the first produced block and `/status` on the other nodes.

Without a fencing step, a promotion is a split-brain hazard, not a failover.

### Split-brain and same-key double signing

The node has no leader lease and no persisted production lock. If the same
`VNN_KEY_SEED` runs on two hosts:

- Both can believe they are the scheduled leader at the same height. They build
  blocks from their local state, sign different headers (different timestamps,
  possibly different mempools), and gossip both. Peers see two blocks at the
  same height and one branch wins arbitrarily.
- Both can sign the same height with the same block hash if their state and
  timestamp are identical. That is still two signatures from one key on one
  height, and it is indistinguishable from an operational mistake or an
  attack.
- With ADR 004 BFT, two different prevotes/precommits at the same
  `(height, round)` are equivocation evidence. Evidence makes the fault
  attributable; it does not prevent the safety violation if both signatures
  reach the network. The safe rule remains "one key, one process."
- Without BFT, there is no automatic detection. An operator sees two head
  hashes or a `wrong producer` rejection and has to reconcile manually.

Recommended mitigations, in order:

1. A startup check that refuses to start a validator if a peer reports the
   same validator address at a different height or head hash. This is a
   heuristic, not a lock.
2. A distributed lease keyed by validator address (etcd, Consul, or a cloud
   instance lock) that the process must hold before producing. The lease must
   be renewed and must expire if the process dies. This is the correct fix and
   is not implemented.
3. A per-height production record (`produced:<height>` or an own-vote record
   from ADR 004) persisted before signing, and a refusal to sign a different
   block at a height already produced. This catches restart amnesia but not
   two live processes.
4. Operator policy: the validator seed lives in one secret manager, one host
   identity, and one deployment. No copied `.env` files.

Do not describe a manual promotion as "automatic failover." It is a planned
stop/start with a fencing requirement.

### API failover, health checks, DNS, and load balancer

- `GET /ready` (`crates/node/src/rpc.rs`) is the readiness surface. It returns
  200 only when a head exists (`serving`), the state-derived validator set
  matches the configured set (`validatorSetConsistent`), the node is not
  applying catch-up blocks (`syncing == false`), `headAgeMs < 3 *
  block_time_ms`, and the local height is within `VNN_PEER_LAG_BLOCKS` of the
  best peer height observed through `VNN_SYNC_URLS`. It returns 503 with a
  stable `reason`: `no_head`, `validator_set_mismatch`, `syncing`,
  `stale_head`, or `behind_peer`. The response also includes `height`,
  `headAgeMs`, `peerCount` (live libp2p connections), `syncing`,
  `validatorSetHash`, `validatorSetConsistent`, and `reason` (null when ready).
  With no head, `headAgeMs` is 0 rather than the current epoch time.
- Peer height is only available when catch-up sync is configured. Each poll
  accepts a peer's height only after its `/status` reports this node's
  `chainId`, and replaces the stored best height with the maximum from the
  valid peers that responded (clearing it when none did). A wrong-chain or
  vanished peer therefore cannot pin `/ready` in `behind_peer` forever. A node
  with no `VNN_SYNC_URLS` (for example the live single validator) has no peer
  height, so the `behind_peer` condition is skipped rather than fabricated.
  `peerCount` is always the real connection count, including 0 when isolated.
- `syncing` is set only while replayed catch-up blocks are being applied, not
  during the routine `/status` poll or block fetch. The sync HTTP client uses a
  2 s connect timeout, 5 s read timeout, and 5 s total request timeout, so a
  silent peer cannot hold the loop open indefinitely.
- `GET /health` remains liveness only: `ok: head.is_some()` with `height` and
  `headAgeMs`. A lagging replica with an old head still returns `ok: true`, so
  a load balancer must use `/ready`, not `ok` alone.
- `validatorSetHash` is lowercase hex SHA-256 over the canonical JSON of the
  active validator addresses sorted and deduplicated (a JSON array such as
  `["ai_a","ai_b"]`, not newline-joined). It comes from `val:` state when
  non-empty, otherwise from the genesis/`VNN_VALIDATORS` list. It covers
  addresses only, not keys or stake, and never panics on an empty set. This is
  an observability hash: it is not part of `compute_state_root` and does not
  make set changes consensus state.
- `validatorSetConsistent` compares the state-derived set with the configured
  set and fails closed: `/ready` returns 503 `validator_set_mismatch` when they
  differ. It does not rewrite state or change consensus membership.
  `/status` exposes the same `syncing`, `validatorSetHash`, and
  `validatorSetConsistent` values alongside its existing fields.
  `finalizedHeight` still waits on ADR 004.
- API failover: DNS name or load balancer in front of the API ports. Use
  active/passive for writes that depend on a specific validator (there are
  none today: `POST /tx` and `/v1/infer` just gossip transactions) and
  health-checked read balancing for reads. If the LB cannot compare peer
  heights, send all reads to the active node and keep the replica as a warm
  standby.
- P2P failover: peers dial `VNN_SEEDS` multiaddrs. If the validator host
  changes, the new address must be reachable. Use a stable DNS name in the
  multiaddr, or publish multiple seeds, or update `VNN_SEEDS` on every node.
  mDNS is not a production discovery mechanism.
- Split-brain detection: expose the validator address and head hash on
  `/status`; alert if two hosts report the same validator at different head
  hashes, or if a node logs `wrong producer` for a height it did not produce.

## Snapshot sync

### Snapshot format

A snapshot is a RocksDB checkpoint at a finalized head plus a signed manifest.
The state-layer v1 implementation covers the checkpoint, file hashes, and the
manifest fields below except `finalized`, `commitCertificate`, `validatorSet`,
`compression`, and `signers`; `docs/snapshots.md` records the exact implemented
schema. Proposed layout:

```text
snapshot-<height>/
  manifest.json
  manifest.sig
  state/                 # RocksDB checkpoint of the canonical store
  blocks/recent.json     # headers for the last challengeWindow + finalityDepth
```

`manifest.json` (canonical JSON):

```json
{
  "v": 1,
  "kind": "verinoo-state-snapshot",
  "chainId": "verinoo-test-1",
  "height": 12345,
  "blockHash": "<hash at height>",
  "parentHash": "<hash at height-1>",
  "stateRoot": "<compute_state_root of the snapshot store>",
  "baseFee": 3,
  "finalized": true,
  "commitCertificate": null,
  "modelSetHash": "<runtime + transformer + gpt + builtin weights>",
  "modelHashes": { "<modelId>": "<weightsHash>" },
  "validatorSetHash": "<sha256 of canonical JSON sorted address array>",
  "validatorSet": [
    { "address": "ai_...", "publicKey": "<hex or empty>",
      "stakedAmount": 0, "genesis": true, "activationHeight": 0 }
  ],
  "createdAt": 1760000000000,
  "compression": "zstd",
  "files": [
    { "path": "state/000123.sst", "size": 123456, "sha256": "<hex>" }
  ],
  "signers": ["ai_..."],
  "signatures": [{ "address": "ai_...", "publicKey": "<hex>", "signature": "<hex>" }]
}
```

The manifest signature covers the canonical JSON with `signatures` removed,
domain `verinoo/snapshot/v1`. `stateRoot` is recomputed after download, not
trusted. `modelSetHash` must equal the local registry hash; otherwise the node
will fail at the first block that runs a model. `validatorSetHash` must equal
the local expected set (or the on-chain set after activation). `blocks/recent`
lets the verifier check `blockHash` and `parentHash` without replaying from
genesis; it is not a substitute for the commit certificate.

### Signed manifest

- **With BFT (ADR 004):** include the commit certificate for `height` and
  require a quorum of validator signatures over the manifest. The snapshot is
  then as trustworthy as the finalized block it commits to. Verify the
  certificate exactly as `sync_once` verifies a finalized block.
- **Without BFT:** the manifest is operator-attested. A single operator key or
  a configured `VNN_SNAPSHOT_SIGNERS` set signs it. A third party can recompute
  the state root and check it against the manifest, but without a trusted
  checkpoint it cannot know that this state is the canonical chain. Do not call
  this trustless. For a pilot, pin the snapshot signer and publish its public
  key out of band.
- Reject unknown/invalid signatures, duplicate signers, and a signer set that
  differs from the expected set. Record the accepted manifest hash in the
  node's logs and `/status`.

### Download, verify, apply

1. Fetch `manifest.json` and `manifest.sig` from
   `GET /snapshot/manifest` (proposed).
2. Verify the manifest signature threshold, chain id, protocol version, and
   `validatorSetHash`. Reject a snapshot whose `height` is not greater than the
   local head.
3. Download each file listed in `files` (or a range endpoint) into a temporary
   directory. Verify size and SHA-256 per file before writing it into place.
   Support resume; a partial snapshot is never opened as a live database.
4. Open the checkpoint as a `StateStore` and recompute
   `compute_state_root`. It must equal `manifest.stateRoot`. Open the head
   block and verify `head.hash == manifest.blockHash`,
   `head.index == manifest.height`, and `head.previous_hash ==
   manifest.parentHash`.
5. Verify `modelSetHash` and every `modelHashes` entry against the local model
   registry (`verinoo_runtime::global()`, transformer, GPT, built-in). A
   mismatch is a configuration error, not a reason to overwrite the local
   database.
6. Verify `validatorSetHash` against the expected set (the genesis list before
   activation; the state-derived set after activation).
7. If BFT: verify the commit certificate and require the manifest height to be
   a finalized height.
8. Stop the node. Move the live database directory aside (do not delete it),
   rename the verified temporary directory into `VNN_DB`, and start the node.
9. Set the head from the snapshot, ensure `blockheight:<height>` exists, and
   start catch-up sync from `height`. `sync_once` continues with
   `/blocks/since/:height`.

The apply step must be atomic from the node's perspective: a crash before the
rename leaves the old database intact; a crash after the rename leaves the new
snapshot. The database must be closed before apply and the snapshot directory
must be immutable and trusted for the whole verify/apply operation. Keep the
old directory until the node has applied at least one new block and produced a
matching state root.

State-layer scope today: `verify_snapshot` performs the local file-set checks
from steps 3-4 (size/SHA-256, path safety, metadata comparison, and
`baseFee` when the caller supplies `SnapshotExpectations.base_fee`) for an
already-downloaded snapshot. `apply_snapshot` implements the rename and
rollback portion of step 8 and consumes `state/`, leaving the backup in place
when it replaces an existing database. Apply re-runs integrity or full
expectations immediately before the state rename and is bound to the manifest
bytes verified at entry, but the snapshot directory must still be immutable and
trusted; the target database must be closed. Network download/resume,
state-root recomputation (step 4), signature verification (step 2), certificate
verification (step 7), and head/catch-up setup (step 9) are not implemented.
The state layer treats `stateRoot`, `modelSetHash`, `modelHashes`,
`validatorSetHash`, and `baseFee` as caller/node-level assertions; it does not
derive or cryptographically verify them.

### Trust and rollback

- **Trust anchor.** With BFT, the commit certificate and validator quorum.
  Without BFT, the pinned snapshot signer plus the genesis hash. A fresh node
  has no local history, so the snapshot is the trust anchor for the state at
  `height`; it still verifies every subsequent block normally.
- **Wrong state.** If the recomputed state root differs from the manifest, the
  snapshot is invalid; delete the temporary directory and fall back to
  catch-up. Do not "repair" the live database.
- **Wrong model/validator set.** Reject before touching the live database. A
  model set mismatch is almost always a missing or different model file;
  `docs/operations.md` already has the `state root mismatch` runbook.
- **Rollback.** Never roll back to a lower height. If the snapshot height
  equals the local head but the hash differs, reject (a same-height fork is an
  incident, not a sync choice). If the local node has a higher finalized
  height, reject the snapshot. Keep the old database for manual recovery.
- **Malicious omission.** A snapshot can omit history; that is the point. It
  cannot silently change the state at `height` if the state root, model set
  hash, and validator set hash are checked. It can serve a valid state from a
  non-canonical branch; the manifest signature/quorum and the block hash chain
  to a trusted checkpoint are what prevent that.
- **Challenges and pending tasks.** The state includes `task:`, `game:`, and
  `claim:` records. A snapshot taken while a challenge is open preserves it;
  the new node must be able to resolve it. Include the recent block headers
  needed to compute the challenge deadline and the claim's block hash. Do not
  prune game state from a snapshot until the challenge window and any appeal
  period have passed.

### Interaction with catch-up sync

Snapshot sync is a base, not a replacement for block validation:

- After applying a snapshot at height `H`, `sync_once` starts at `H` and
  verifies every block from `H+1` through the peer's head using the existing
  `admit_block`/`commit_block` path. The snapshot does not bypass
  `validate_block` for future blocks.
- `/blocks/since/:index` remains the steady-state catch-up endpoint. Add
  `/snapshot/manifest` and a file/range endpoint; keep the block endpoint
  unchanged.
- With BFT, add `/finalized/since/:index` (ADR 004) and verify certificates
  after the snapshot. A snapshot at a finalized height plus certificates is
  the fast join path.
- A node that is only a few blocks behind should use catch-up, not a snapshot.
  Snapshots are for fresh nodes and for nodes beyond a configured gap (e.g.,
  more than 10,000 blocks behind).

## On-chain validator removal

### Transactions

Use the existing transaction kinds, with explicit fields and validation:

- `RegisterValidator { stakedAmount, publicKey, activationHeight? }`
  - `public_key_to_address(publicKey) == tx.from` (same rule as blocks,
    `crates/exec/src/engine.rs:23-32`).
  - `stakedAmount >= MIN_VALIDATOR_STAKE` (`crates/state/src/validators.rs:8`).
  - The address must not already be active; re-registration after removal is
    allowed after a cooldown.
  - `activationHeight` must be at least `currentHeight + activationDelay` and
    aligned to an epoch boundary.
- `UnregisterValidator { }`
  - Only the validator's own address may sign it.
  - The validator is marked `deactivating`, not deleted; it remains liable for
    open challenges and past claims until `deactivationHeight`.
  - Refund the bond at `deactivationHeight`, not at transaction inclusion.
  - Reject if the removal would leave fewer than `minValidatorSet` active
    validators, or fewer than the quorum required to finalize the transition.
- `BindValidatorKey { publicKey }`
  - For genesis entries with an empty `public_key`.
  - Signed by the key that derives to the validator address.
  - Binds one key to one address; a second binding is rejected unless it is a
    rotation signed by both old and new keys and subject to a delay.
  - Until bound, a genesis validator may be listed but cannot vote in BFT
    mode or be slashed; the safe rule is to require binding before activation
    of BFT.
- Optional `JailValidator { evidence }` or an evidence-driven rule. ADR 004 v1
  only records evidence; this ADR does not add automatic jailing.

### Activation heights and quorum

- Define `EPOCH_LENGTH` (e.g., 1,000 blocks) and `SET_ACTIVATION_DELAY` (e.g.,
  2 epochs). A set change submitted in epoch `E` takes effect at the start of
  epoch `E + 2`.
- The active set for height `H` is computed deterministically from `val:`
  entries with `activation_height <= H` and not `deactivating` before `H`.
  The set is sorted by address.
- `quorum = floor(2N/3) + 1` over the active set for `H`; one validator, one
  vote. ADR 004's N=1/2/3 caveats apply. N=4 (quorum 3) is the minimum that
  tolerates one Byzantine fault; reject a set change that claims BFT while
  leaving N < 4.
- A validator added at epoch `E` cannot vote before `E`. A validator removed
  at `E` cannot vote at `E` or later. There is no mid-epoch change, so every
  node computes the same quorum for the same height.
- Publish `validatorSetHash = sha256(canonical_json(sorted set))` in `/status`
  and in the block header or state root. Refuse to vote or produce if a peer's
  value differs (this is also proposed in ADR 004 for the genesis set).
- On a set change, the quorum for the activation height must be computable
  from the previous set's committed state. Do not let a transaction in the
  same block change the quorum that validates it.

### Public key binding

Genesis entries have an empty `public_key` (`crates/state/src/lib.rs:217-223`),
so evidence cannot be checked against a stored key. ADR 004 works around this
by carrying `publicKey` in each vote and deriving the address. For an on-chain
set, that is not enough: an attacker could register an address with one key and
vote with another, or a removed validator could keep voting.

- Require a bound key for any address that votes, produces, or is slashable.
- `BindValidatorKey` must be a transaction in a finalized block before the
  validator's activation height.
- Key rotation requires signatures from the old and new keys and a delay of at
  least `SET_ACTIVATION_DELAY`; the old key must not vote after the rotation
  height.
- Store `public_key`, `bound_at_height`, and `key_epoch` in `ValidatorEntry`.
  Include them in the validator-set root.
- Verify every vote/evidence message against the bound key, not just the
  self-carried key. If the message key differs from the bound key, reject it
  even if the address derivation matches.

### Removal during a challenge

A validator can be the executor, challenger, or verifier in an open LLM or
dispute game (ADR 005). Removal must not erase liability:

- Mark `deactivating`; keep the bond locked until every game involving the
  validator is resolved or its challenge window has closed.
- If the validator is a verifier in a panel, replace it with the next
  eligible verifier for future votes; votes already cast remain valid.
- If the validator is the executor or challenger, the game continues; the
  bond is paid or slashed according to the game result, even after
  deactivation.
- If removal would leave the active set below the quorum needed to resolve
  the game, delay the deactivation until a replacement is active.
- A removed validator's past receipts remain valid. Removal is not a
  retroactive statement about its earlier work.

### State root coverage

`compute_state_root` must cover the validator set before consensus membership
can be derived from state. Otherwise two nodes can disagree about who is
allowed to vote without a state-root mismatch.

Add, sorted:

- `val:<address>` records: address, public key, stake, `genesis`, activation
  height, deactivation height, `jailed`, `key_epoch`;
- `meta:validatorSetHash` and the active-set computation parameters
  (`epochLength`, `activationDelay`, `minValidatorSet`);
- `game:` and `claim:` records (or their hashes), so challenges are consensus
  state;
- `meta:llmParams` if ADR 005 is enabled.

Do not include `meta:head` or any value derived from the head itself
(circular). A header field `validatorRoot` is an alternative to including
`val:` in the state root; it is cleaner for light clients but is a header
version bump and a coordinated hard fork. For v1, including sorted `val:`
entries in the state root is the smaller change; document the activation
height.

### Genesis migration

Moving from a genesis list to a state-derived set is a hard fork, not a
restart toggle:

1. Ship the code with both modes and `VNN_VALIDATOR_SET_MODE=genesis|state`.
2. At `VNN_VALIDATOR_SET_ACTIVATION_HEIGHT`, every node must run the same
   binary, the same model set, and the same genesis list.
3. Before activation, bind every genesis validator's key with
   `BindValidatorKey` and confirm `validatorSetHash` matches on all nodes.
4. At activation, the state-derived set becomes authoritative. A node that
   still uses `VNN_VALIDATORS` for consensus will compute a different set and
   its blocks will be rejected (or worse, it will vote with the wrong quorum).
5. Rollback requires stopping the set and restarting in genesis mode at a
   finalized height; treat it as a coordinated incident.

## Implementation order

Do the prerequisites before the visible features:

1. **State-root coverage and set hash (1 wk).** `validatorSetHash` and
   `validatorSetConsistent` are now exposed on `/status` and `/ready` (addresses
   only), and `/ready` fails closed on a mismatch. But `val:` and `meta:` set
   parameters are not yet in `compute_state_root`, and consensus membership is
   still the configured list, so the node does not refuse to run with a
   mismatched set. This coverage is a prerequisite for on-chain removal and for
   BFT.
2. **Snapshot format, manifest, and verifier (1-2 wk).** The state layer now has
   `StateStore::checkpoint`, a manifest builder/verifier, and atomic
   apply/rollback (`crates/state/src/snapshot.rs`; `docs/snapshots.md`).
   Signing, node download/verify orchestration, state-root recomputation, and
   serving endpoints remain. Do not serve snapshots until the node verifier
   rejects tampered files, stale heights, and unauthorized signatures.
3. **Catch-up from a snapshot (0.5-1 wk).** Start `sync_once` at the snapshot
   height; keep `/blocks/since` unchanged. Add `/snapshot/manifest` and a file
   endpoint behind a flag.
4. **HA runbook and fencing (1 wk).** `/ready`, height/lag checks, `peerCount`,
   `syncing`, and `validatorSetHash` are implemented. A promotion script that
   refuses to run while the old process is alive and split-brain alerts remain.
   Automated failover is a separate 2-3 wk item and should not be promised
   before the fencing primitive exists.
5. **Validator key binding and registration/unregistration (2-3 wk).** Add
   `BindValidatorKey`, epoch activation, deactivation, bond locking, and
   validator-set root coverage. Reject changes that break the quorum or the
   minimum set.
6. **Removal during challenges (1 wk).** Integrate with the ADR 005 game state
   and bond accounting.
7. **BFT integration (depends on ADR 004).** Certificates in snapshots,
   `finalizedHeight` in `/status`, round-aware leader selection, and
   equivocation evidence for same-key double signing. Do not claim BFT
   finality before ADR 004's test matrix is green.

The readiness document budgets 2-3 weeks for "automatic HA / validator
removal" and 1-2 weeks for snapshot transfer (`docs/readiness.md:250-251`).
Those are optimistic if done independently and without the state-root
prerequisite. A realistic combined estimate is **6-9 weeks** for snapshot,
HA/fencing, state-root coverage, and on-chain removal, plus **+2-4 weeks** for
automated failover and BFT certificate integration. The estimate excludes an
external audit and a multi-operator testnet.

## Test plan

Snapshot tests (the state-layer cases run in `crates/state/src/snapshot.rs`;
the remaining cases are integration or node-level):

The state-layer suite covers create/verify/apply, tampered files (including a
manifest or state file changed between verify and apply), metadata mismatches
(including `baseFee` expectations), path traversal/absolute paths/duplicates/
missing files, successful replacement with an intact readable backup, rejected
overlapping snapshot/target/backup paths, backup/rollback, and stable signing
bytes. The cases below that require a network endpoint, signature policy,
state-root recomputation, or BFT remain unimplemented.

1. Build a store at height H, checkpoint it, write a manifest, apply it to a
   fresh temp store. Assert the state root, head hash, height, base fee, model
   set hash, and validator set hash all match.
2. Tamper one file; assert the SHA-256 check fails and the live database is
   untouched.
3. Tamper the manifest's `stateRoot`; assert the recomputed root rejects it.
4. Tamper `modelSetHash` or `validatorSetHash`; assert rejection before apply.
5. Serve a snapshot at a lower height than the local head; assert refusal. Serve
   a same-height different-hash snapshot; assert refusal and an incident log.
6. Interrupt a download and resume; assert no partial file is opened.
7. Take a snapshot with an open `game:`/`claim:` record; apply it; assert the
   game is still resolvable and the deadline is preserved.
8. After applying a snapshot at H, sync blocks H+1..H+k through
   `admit_block`/`commit_block`; assert the same head as a node that replayed
   from genesis.
9. With BFT enabled, require the commit certificate; a manifest without a
   quorum is rejected; a certificate with a wrong height/hash is rejected.
10. Snapshot signed by an unauthorized key; assert rejection. Snapshot with
    fewer than the threshold signatures; assert rejection.

HA tests (scripts/integration, no consensus changes):

1. Passive replica with `VNN_SYNC_URLS` catches up to the active node; assert
   identical height/head hash and state root.
2. Promote the replica while the old process is still running; the promotion
   script must refuse. Kill the old process, promote, produce a block, assert
   the network follows the new host.
3. Start the same `VNN_KEY_SEED` on two hosts; assert the startup check
   detects the duplicate (if implemented) or the test documents the observed
   fork and evidence. Do not claim prevention without the lease.
4. Load balancer health check: a stale replica (`headAgeMs` high, height
   behind) is marked not ready; reads route to the active node; after catch-up
   it becomes ready.
5. Split-brain simulation: partition the two hosts, let both produce, heal;
   assert the chain converges on one head in non-BFT mode and that BFT mode
   records evidence (once ADR 004 exists).
6. API key file and model set are identical on both nodes; a mismatch is
   detected before promotion.

Validator-set tests (`crates/exec/tests/` and a consensus harness):

1. Register a validator with a valid key and stake; assert it is inactive
   before `activationHeight` and active at the boundary; it can produce/vote
   only at/after activation.
2. Register with a public key that does not derive to the address; reject.
3. Register below `MIN_VALIDATOR_STAKE`; reject.
4. Unregister a validator; assert it remains liable for an open game, its bond
   is locked until resolution, and it is removed from the active set at the
   deactivation height.
5. Remove a validator that would leave the set below quorum or below
   `minValidatorSet`; reject.
6. Genesis entry with an empty public key: bind a key, vote, and verify the
   vote against the bound key. A second binding without rotation is rejected;
   rotation requires both signatures and a delay.
7. Two nodes with different `VNN_VALIDATORS` or different `val:` state produce
   different state roots and reject each other's blocks; `/status`
   `validatorSetHash` mismatches are detected.
8. Change the set at an epoch boundary; assert the quorum for the activation
   height is computed from the old set and the next height uses the new set;
   no mid-epoch change is possible.
9. Remove a validator that is a verifier in an ADR 005 panel; assert a
   replacement is selected and already-cast votes remain valid.
10. Reorg around the activation height: without BFT, a change in an orphaned
    block is not active; with BFT, the activation height is finalized and
    cannot be reorged.

## Non-claims

- **Readiness only.** `/ready`, `syncing`, `peerCount`, `validatorSetHash`, and
  `validatorSetConsistent` are implemented as observability. A mismatch makes
  `/ready` return 503, but the node still runs and consensus membership remains
  the configured list. These signals do not elect a leader, fence a host,
  transfer state, or change consensus membership.
- **Not implemented.** There is no automatic failover, leader lease, fencing
  primitive, snapshot HTTP endpoint, node-level manifest signing/verification,
  state-root recomputation during snapshot verification, epoch activation,
  `BindValidatorKey`, or state-derived consensus set in this repository.
- **No automatic HA.** Promotion is manual and requires an external fencing
  step. Running the same validator key on two hosts is not prevented by the
  node; it is detected, at best, after the fact.
- **No split-brain protection.** There is no lease, lock, or quorum-based
  leader election. `/ready`, `peerCount`, and `validatorSetHash` make a
  divergent head or set visible, but they do not prevent it. ADR 004's evidence
  collection is not a prevention mechanism.
- **No snapshot transfer over the network today.** The state layer can create,
  verify, and apply a local snapshot (`crates/state/src/snapshot.rs`,
  `docs/snapshots.md`), but no endpoint serves one and no node downloads one. A
  late node still replays from genesis via `VNN_SYNC_URLS`; join time is
  O(chain length).
- **No signature or state-root verification in the state layer.** Snapshot
  metadata is caller-provided. The state layer compares `stateRoot`,
  `modelSetHash`, `modelHashes`, and `validatorSetHash` to expected values and
  hashes the files; it does not recompute the state root and does not verify
  that a signature belongs to a signer. Those checks remain node/`exec` work.
- **No trustless snapshot without a trust anchor.** With BFT, a commit
  certificate and quorum signatures; without BFT, a pinned operator signer.
  A snapshot does not replace block validation for blocks after its height.
- **No on-chain validator removal today.** `RegisterValidator` /
  `UnregisterValidator` mutate `val:` but do not affect consensus membership
  (`crates/exec/src/engine.rs:905-950`). The live set is still
  `VNN_VALIDATORS` (`crates/node/src/main.rs:105-107`).
- **No slashing of genesis validators.** Genesis entries have
  `staked_amount: 0` and an empty public key
  (`crates/state/src/lib.rs:217-223`), so there is nothing to burn and no bound
  key to verify evidence against until `BindValidatorKey` exists.
- **No BFT finality.** ADR 004 is a design. Snapshot certificates, quorum
  removal rules, and round-aware production are not shipped.
- **No claim that a missed leader is fixed.** The slot-based producer rule
  still stalls the chain if the scheduled validator is offline. Only ADR 004's
  rounds/view changes address that.
- **No light-client proof.** A snapshot manifest proves state at a height to
  someone who trusts its signer or checks the commit certificate; it is not a
  succinct proof of the whole chain.
- **No state pruning.** The snapshot format does not define a pruning policy;
  old `hash:`, `blockheight:`, `task:`, and `game:` records remain unless a
  separate retention rule is added.
