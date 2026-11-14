# ADR 004: BFT Finality for the Verinoo Node

## Status

**Proposed — not implemented.** This document describes a target design and
the integration work required. No code in this repository implements
prevotes, precommits, commit certificates, finality, or equivocation handling
today. Nothing here may be cited as shipped behavior.

Date: 2026-11-14
Scope: `crates/node`, `crates/exec`, `crates/network`, `crates/state`, plus a
new `crates/consensus`.

## Context

The chain currently produces blocks with deterministic single-producer
authority and no finality gadget:

- Leader selection is `RoundRobin` over the genesis-configured active set
  (`crates/exec/src/leader.rs:17`); `ValidatorPolicy` is applied at
  `crates/exec/src/engine.rs:296`. `StakeWeighted`
  (`crates/exec/src/leader.rs:37`) is a compatibility fallback; the live path
  uses `RoundRobin`.
- A block is authenticated by exactly one Ed25519 signature from
  `block.header.validator` over the header hash
  (`crates/exec/src/engine.rs:256-294`, header construction at
  `crates/exec/src/engine.rs:1323-1341`). There is no prevote, precommit, or
  quorum certificate.
- Receiving nodes admit a block with `validate_block`
  (`crates/exec/src/engine.rs:222`), apply it, compare the state root, and
  immediately advance the head (`commit_block` at
  `crates/node/src/main.rs:130-158`). "Final" means "the producer signed it".
- The validator loop (`crates/node/src/main.rs:161-229`) produces one block
  per tick when the node is the scheduled leader, writes the head itself, and
  gossips the block. There is no round, timeout, or view change.
- Catch-up sync exists (`sync_loop`/`sync_once`,
  `crates/node/src/main.rs:599-656`): a lagging node polls
  `GET /blocks/since/:index` (`crates/node/src/rpc.rs:118-145`) and replays
  blocks through the same `admit_block`/`commit_block` path.
- The validator set is genesis-seeded: `GenesisParams::validators`
  (`crates/state/src/genesis.rs:14-21`) is copied into `val:` entries with
  `genesis: true` and an **empty public key**
  (`crates/state/src/lib.rs:212-225`; `ValidatorEntry` at
  `crates/state/src/validators.rs:11-22`). `Shared::allowed_validators()`
  (`crates/node/src/main.rs:105-107`) returns `params.validators`, not the
  state-backed set. `RegisterValidator`/`UnregisterValidator` mutate state
  (`crates/exec/src/engine.rs:854-899`) but do not affect consensus
  membership.
- `compute_state_root` (`crates/exec/src/engine.rs:930-976`) covers accounts,
  models, pending tasks, completed result hashes, and compute nodes. It does
  **not** cover `val:` entries, `meta:`, or games. Validator-set changes are
  therefore not committed by the state root.
- The deployment is a single operator. `docs/readiness.md:50-52` names "no
  finality" as a launch blocker, and `docs/readiness.md:228-231` recommends
  not building BFT before a pilot asks for it.
- `docs/adr/002-malachite-bft.md` records the earlier decision to defer a
  Malachite adapter because it brings its own P2P and storage and may conflict
  with libp2p + RocksDB.

The design goal is therefore narrow: give the existing block format a 2/3+1
quorum certificate, make finality explicit and persistent, collect
equivocation evidence, and do so without replacing libp2p, RocksDB, or the
execution engine.

## Options

### (a) Malachite / Tendermint adapter

Integrate `malachite-consensus` behind a context/codec and implement
Malachite's networking, block-store, and state-machine adapters.

- **Buys:** a mature, independently tested Tendermint implementation with
  rounds, locks, WAL, timeouts, and evidence plumbing.
- **Integration cost:** high. Malachite expects to own consensus networking
  and block/WAL persistence. Bridging means writing adapters from
  `StateStore`/RocksDB and from gossipsub/libp2p to Malachite's network
  traits, then resolving who owns the head, the mempool, and block storage.
  The current producer executes against state during production
  (`crates/exec/src/engine.rs:1263-1362`), which also has to be reshaped.
  Estimated 8-12 engineer-weeks, with a large diff and a new source of truth
  for block ordering.
- **Risk:** high. Two storage layers, two networking stacks, and a
  three-party race between the existing `chain_lock`, Malachite's consensus
  state, and catch-up sync. API churn in an evolving crate. Hard to test the
  node end-to-end without also testing the adapter layer.
- **What it does not buy:** it does not remove the genesis-set limitation.
  Validator-set changes still need a set-change mechanism covered by the
  state root.

### (b) Minimal 2/3 prevote/precommit gadget on gossipsub + `validate_block`

Add a Tendermint-style round state machine as a new pure crate
(`crates/consensus`), carry proposals and signed votes on new gossipsub
topics, and reuse the existing block hash, header signature, and
`validate_block_with_policy` gate. The canonical store advances only when
2/3+1 precommits for the same `(height, round, block_hash)` are verified.

- **Buys:** explicit finality certificates, equivocation evidence, no
  single-signature finality, no new P2P or storage dependency, and a
  safety argument that is the standard Tendermint lock rule. With four
  validators it tolerates one Byzantine fault (`f=1`, quorum 3).
- **Integration cost:** medium. About 5.5-7 engineer-weeks including tests.
  Main costs are (i) moving proposal execution off the canonical store,
  (ii) the vote/lock state machine and persistence, (iii) node wiring and
  timers, (iv) certificate-aware sync.
- **Risk:** medium. Liveness requires ≥2/3 online; gossipsub is not a
  reliable transport; the lock/persistence rules must be implemented
  exactly; the current "apply during produce" behavior must be refactored.
  There is no set-change or automatic slashing in v1.
- **What it does not buy:** no responsiveness or pipelining, no light-client
  protocol, no validator-set rotation.

### (c) HotStuff-style chained finality

Replace rounds with a chained three-phase (prepare/pre-commit/commit) or
two-phase pipelined HotStuff variant with a pacemaker and QC chain.

- **Buys:** linear view changes, pipelined block production, responsiveness,
  and a path toward larger validator sets.
- **Integration cost:** 7-10 engineer-weeks. Requires a new safety/liveness
  argument, a pacemaker, leader staking/weights wired to real stake (the
  live set currently has no stake weighting), and a QC-chain that changes
  the block/header model more deeply than option (b).
- **Risk:** high for the current goal. More novel code to audit, and the
  liveness benefits are irrelevant at N≤10 with a 10 s block time. It also
  does not solve slashing or set changes by itself.

### Comparison

| | (a) Malachite | (b) Minimal gadget | (c) HotStuff-style |
|---|---|---|---|
| New P2P | yes (adapter) | no | no |
| New storage/WAL | yes | no (RocksDB keys) | no |
| Reuses block hash + producer sig | partially | yes | mostly |
| Finality certificate | yes | yes | yes (QC) |
| Equivocation evidence | built in | explicit, small surface | explicit |
| Tolerates 1 Byzantine | yes | yes with N=4 | yes |
| Engineer-weeks | 8-12 | 5.5-7 | 7-10 |
| Main risk | adapter conflict | lock/persistence correctness | protocol complexity |

## Recommendation

**Choose option (b): a minimal 2/3 prevote/precommit finality gadget over
the existing gossipsub topics and block validation.**

Reasons, in order of weight:

1. It reuses the two things that already work: the canonical block hash
   (`hash_header`, `crates/exec/src/engine.rs:188-196`) and Ed25519
   signatures verified by `validate_block_with_policy`
   (`crates/exec/src/engine.rs:238-360`). A vote is a signed tuple over
   `(chain_id, type, height, round, block_hash)`; verification is the same
   primitive, not a new one.
2. It does not fight the existing stack. Malachite would introduce a second
   P2P and storage layer next to libp2p and RocksDB — the exact conflict
   ADR 002 deferred. Option (b) adds two gossipsub topics and state keys.
3. The network is small and single-operator. The realistic near-term goal is
   an honest finality claim and evidence collection, not pipelined consensus
   for 100 validators. Option (b) reaches that with a standard safety
   argument and a small audit surface.
4. It is testable entirely in-process: the state machine is pure, messages
   are JSON, and four deterministic validators can exercise byzantine,
   partition, restart, and catch-up scenarios without opening sockets.
5. It leaves a clean upgrade path. The round-aware leader policy keeps the
   `LeaderPolicy` seam (`crates/exec/src/leader.rs:7-10`); a later Malachite
   or HotStuff implementation can replace the driver without changing block
   validation or execution.

Option (a) should be revisited if the network grows beyond a handful of
independent operators or if an audited consensus implementation becomes a
procurement requirement. Option (c) should be revisited if block time needs
to drop below ~2 s or the set grows past ~20 validators.

## Protocol

### Parameters and terminology

- `N` = size of the consensus validator set. For v1 this is the sorted
  `GenesisParams::validators` list, fixed for the life of the chain unless
  the node is restarted with a different list. State-backed
  `RegisterValidator` entries do not affect consensus in v1.
- `quorum(N) = floor(2N/3) + 1` (strictly more than 2/3). Examples: N=1 → 1,
  N=2 → 2, N=3 → 3, N=4 → 3, N=7 → 5.
- One validator, one vote. Stake weighting is not used in v1 because the
  live validator set is genesis-seeded and `stake_weighted_leader`
  (`crates/exec/src/leader.rs:49`) is not called anywhere in the node.
- `H` = height. `r` = round, starting at 0 for each height.
- `finalized_height`/`finalized_hash` are persisted meta values. A node
  never votes at a height ≤ `finalized_height`, except to gossip stored
  evidence.
- `locked` = `(locked_height, locked_hash, locked_round)`, persisted.
- Round-0 leader is the existing `RoundRobin` leader for that height:
  `sorted[(height - 1) % N]`. Round `r` leader is
  `sorted[(height - 1 + r) % N]`. This can be implemented without changing
  the `LeaderPolicy` trait by adding a `RoundAware { round: u32 }` policy in
  `crates/exec/src/leader.rs` that calls the same formula; pass it to the
  existing `validate_block_with_policy`.

### Messages and phases

For height `H` and round `r`:

1. **Proposal.** The round leader builds a candidate block extending the
   node's locked block (and therefore the finalized parent), signs the
   normal block header, wraps the block in a signed proposal envelope that
   binds `(H, r, block_hash)`, and publishes it. The envelope signature is
   required because the block header does not contain `round`; without it a
   valid block could be replayed into a different round.
2. **Proposal receipt.** A validator:
   - verifies the proposal envelope signature and that `proposer ==
     block.header.validator`;
   - runs `validate_block_with_policy(block, parent_header,
     active_validators, now, &RoundAware { round: r })`;
   - executes the block against a **checkpoint clone** of the canonical
     store and checks that `compute_state_root(clone) ==
     block.header.state_root`;
   - checks the prevote rule below.
   If all pass, it broadcasts `prevote(H, r, block_hash)`; otherwise it
   broadcasts `prevote(H, r, "")` (nil).
3. **Prevote.** On collecting `quorum(N)` prevotes for the same
   `(H, r, block_hash)` (a "polka"), a validator broadcasts
   `precommit(H, r, block_hash)` and updates its lock. It must never
   precommit a hash it did not prevote.
4. **Precommit / commit.** On collecting `quorum(N)` precommits for the same
   `(H, r, block_hash)`, the block is final. The node:
   - assembles a commit certificate from the precommit signatures;
   - applies the block to the canonical store exactly once (see below);
   - sets `finalizedHeight = H`, `finalizedHash = block_hash`;
   - persists the certificate under `meta:commit:<H>`;
   - broadcasts the certificate (`type: "commit"`);
   - shuts down the round state for `H` and starts `H+1` at round 0.
5. **Nil rounds.** If the proposal/prevote/precommit phase times out, the
   validator sends a nil vote of the corresponding kind and moves to the
   next round. A nil prevote does not lock. A nil precommit does not finalize
   anything.

### Lock and unlock rules (Tendermint rule)

Persist the lock as `meta:consensusLock = {height, hash, round}`.

- On entering a new height, `locked = (finalized_height, finalized_hash,
  -1)`.
- Before prevoting for a proposal `P` at round `r`:
  - prevote `P` if `locked_round == -1`, or `P.hash == locked_hash`, or
    `r > locked_round`;
  - otherwise prevote nil.
- When a polka for `(H, r, P)` is observed and `r > locked_round`, unlock:
  set `locked = (H, P.hash, r)` and precommit `P`.
- When a polka for `(H, r, P)` is observed at the round where the node
  prevoted `P`, set `locked = (H, P.hash, r)` and precommit `P`.
- Never sign two different precommits at the same `(H, r)`. Never sign a
  precommit for a hash that differs from the node's own prevote at that
  `(H, r)`.
- Persist the signed prevote/precommit under
  `meta:consensusOwnVote:<H>:<r>:<kind>` **before** broadcasting. On restart,
  reload the stored vote and re-broadcast it verbatim; refuse to produce a
  different signature for the same tuple. This prevents "amnesia" double
  votes after a crash.

### Timeouts

Manual-clock friendly and bounded:

- `timeout_propose(r) = min(3s * 1.5^r, 15s)`
- `timeout_prevote(r) = min(2s * 1.5^r, 10s)`
- `timeout_precommit(r) = min(2s * 1.5^r, 10s)`

Round 0 is roughly 7 s, under the existing 10 s block time. After
`R_MAX = 8`, keep the capped timeouts and keep rotating leaders; do not stop
liveness attempts. Timers start when the phase begins and are cancelled by
the event that ends the phase (proposal received → cancel propose timer;
prevote sent → start prevote timer; polka observed → start precommit timer;
quorum precommits → cancel all and finalize).

A node that is not in the consensus set is an observer: it never votes, runs
no timers, and applies blocks only on a verified commit certificate.

### Quorum and validator-set assumptions

- The quorum is computed over the sorted genesis list. For N=1, quorum=1;
  finality is self-attestation and must be described as such.
- For N=2, quorum=2: one unavailable node halts finality.
- For N=3, quorum=3: no fault tolerance.
- For N=4, quorum=3: tolerates exactly one Byzantine **or** one crash fault,
  not both. This is the minimum configuration that supports the "one
  byzantine node" test and claim.
- If nodes disagree on `VNN_VALIDATORS`, safety is undefined. Add
  `validatorSetHash = sha256(canonical_json(sorted(addresses)))` to
  `/status` and refuse to vote if a peer's value differs.

### Execution and state handling

The canonical store must represent finalized state only. This requires a
small refactor because `produce_block` currently mutates the store it is
given (`crates/exec/src/engine.rs:1291`, `1303-1307`, `1346-1350`,
`1359`), and receivers apply on admission (`commit_block` at
`crates/node/src/main.rs:130-158`).

Target flow:

- **Proposal build (leader):** take a RocksDB checkpoint of the canonical
  store into a temp directory
  (`rocksdb::checkpoint::Checkpoint::new(&db).create_checkpoint(path)`;
  available in `rocksdb 0.22`), open it as a scratch `StateStore`, and call
  the existing `produce_block(scratch, txs, producer, now)`. Discard the
  scratch. Persist the block body in the canonical store under
  `hash:<block_hash>` (hash-addressed, does not move the head or the height
  index).
- **Proposal verification (all validators):** checkpoint the canonical
  store, run `validate_block_with_policy` against the parent header, then
  `apply_block(scratch, &block, base_fee, producer)` and compare
  `compute_state_root(scratch)` with the header. Discard the scratch. Only
  a matching root yields a prevote.
- **Commit (all nodes):** under `chain_lock`, verify the certificate, then
  `apply_block` against the canonical store, compare the root, persist
  `baseFee = applied.next_base_fee`, call `set_head`, and write
  `finalizedHeight`/`finalizedHash`/`meta:commit:<H>`. A commit-time root
  mismatch is a fatal local fault (nondeterminism or corruption), not a
  reason to silently stop: log an error and halt the node rather than
  advance an unverified head.

This gives the correct product property: receipts are only stamped into the
canonical store on commit (`stamp_receipts` is already called by
`apply_block`, `crates/exec/src/engine.rs:1206`), so a receipt always refers
to a finalized block.

**Prerequisite bug to fix in Phase 0:** `produce_block` advances `baseFee`
on its store (`crates/exec/src/engine.rs:1359`), but `commit_block` discards
`AppliedBlock.next_base_fee` (`crates/node/src/main.rs:138-143`). Today a
follower keeps the genesis base fee while a producer advances it, which can
produce different fee deductions and therefore different state roots. Under
finality-gated execution, move the `baseFee` update into `commit_block` only
and remove it from the proposal/scratch path.

### Catch-up sync and missed votes

- `chain_lock` remains the single writer to canonical state. Only the
  consensus actor (commit path) and `sync_once` take it.
- `set_head` is called only on commit, so `blockheight:<H>` and
  `GET /blocks/since/:index` (`crates/node/src/rpc.rs:118-145`) serve
  finalized blocks only.
- Add `GET /finalized/since/:index` returning
  `[{ "block": Block, "certificate": CommitCertificate }]`. `sync_once`
  verifies each certificate (quorum, distinct validators, signatures,
  `height/round/hash` binding) and each block against its parent before
  applying. Keep `/blocks/since` unchanged for compatibility, but do not use
  it as the sole source of truth in BFT mode.
- A node that missed only votes but has the block: the commit certificate is
  sufficient proof; it applies the block without replaying votes.
- A node that missed votes and the block: fetch the block bundle from
  `/finalized/since` or ask a peer for `hash:<block_hash>` via the existing
  block fetch path.
- A node that was partitioned: on rejoin, sync from its `finalizedHeight`,
  verify certificates in order, apply, set `finalizedHeight`, discard all
  buffered votes at heights ≤ the new finalized height, and start the state
  machine at `finalized + 1, round 0`. It never tries to "vote its way back"
  into an old height.
- A node that has votes for a future height but no proposal: buffer them in
  a bounded `BTreeMap<height, RoundVotes>` (see limits below). A commit
  certificate short-circuits the buffer: fetch and apply.
- `sync_once` currently trusts the remote `/status.height`; in BFT mode,
  `/status` must report `height` as the finalized head and may add
  `pendingHeight` separately. Never advance past a certificate that verifies.

## Wire format and gossipsub topics

Keep `<chain_id>/verinoo/blocks/1` and `<chain_id>/verinoo/txs/1` as-is
(`crates/network/src/lib.rs:23-24,217-220`). Add two topics:

```rust
pub const PROPOSALS_TOPIC: &str = "verinoo/proposals/1";
pub const VOTES_TOPIC: &str = "verinoo/votes/1";
```

The full topic names are prefixed with the chain id in `NetworkNode::new`,
matching the existing convention. Two topics rather than one because
proposals can be up to the 1 MB gossipsub limit while votes are small and
latency-critical; separating meshes keeps a large proposal from crowding
votes in the same flood-publish path.

All payloads are UTF-8 JSON. `v` is a protocol version (start at 1). Signatures
are hex-encoded Ed25519 over the canonical JSON of the stated payload using
`verinoo_types::canonical::canonical_json`. `blockHash` is a string; the
empty string means nil.

Proposal (`verinoo/proposals/1`):

```json
{
  "v": 1,
  "chainId": "verinoo-test-1",
  "type": "proposal",
  "height": 42,
  "round": 0,
  "proposer": "ai_...",
  "block": { "header": { "...": "..." }, "transactions": [] },
  "signature": "<hex>"
}
```

Signed payload:

```json
{"domain":"verinoo/proposal/v1","chainId":"verinoo-test-1",
 "height":42,"round":0,"blockHash":"<block.header.hash>"}
```

Vote (`verinoo/votes/1`, `type` is `prevote` or `precommit`):

```json
{
  "v": 1,
  "chainId": "verinoo-test-1",
  "type": "prevote",
  "height": 42,
  "round": 0,
  "blockHash": "<hash or empty string for nil>",
  "validator": "ai_...",
  "publicKey": "<hex>",
  "signature": "<hex>"
}
```

Signed payload:

```json
{"domain":"verinoo/vote/v1","chainId":"verinoo-test-1",
 "type":"prevote","height":42,"round":0,"blockHash":"..."}
```

Commit certificate (`verinoo/votes/1`, `type: "commit"`):

```json
{
  "v": 1,
  "chainId": "verinoo-test-1",
  "type": "commit",
  "height": 42,
  "round": 0,
  "blockHash": "<hash>",
  "precommits": [
    { "validator": "ai_...", "publicKey": "<hex>", "signature": "<hex>" }
  ]
}
```

Each entry is verified by reconstructing the precommit payload above for the
certificate's `(height, round, blockHash)`.

Equivocation evidence (`verinoo/votes/1`, `type: "evidence"`):

```json
{
  "v": 1,
  "chainId": "verinoo-test-1",
  "type": "evidence",
  "kind": "double-vote",
  "validator": "ai_...",
  "height": 42,
  "round": 0,
  "a": { "type": "prevote", "blockHash": "...", "publicKey": "...", "signature": "..." },
  "b": { "type": "prevote", "blockHash": "...", "publicKey": "...", "signature": "..." }
}
```

`a` and `b` are the signed vote payloads (without the envelope) so that both
signatures verify over exactly the bytes that were signed.

### Bounding vote traffic

- **Semantic dedup.** The existing `mark_seen` (`crates/node/src/main.rs:64-80`)
  hashes raw payloads; use it only as a first filter. Add a vote cache keyed
  by `(height, round, kind, validator)`. Same key + same hash is a duplicate
  (drop); same key + different hash is evidence, not a vote.
- **Height window.** Accept votes only for
  `finalized_height < height <= finalized_height + 64`. Drop anything else
  without storage. Retain per-height state for at most 32 heights; evict the
  oldest.
- **Round window.** Process rounds `0..=current_round + 2`; queue at most 32
  future rounds. Votes beyond that are counted and dropped. This lets an
  honest node jump rounds without letting a peer force unbounded memory.
- **Per-peer rate limit.** Token bucket per peer: 200 consensus messages/s,
  burst 400, applied in the network task before the event channel. A
  sustained excess closes the connection; repeated violations go through the
  existing `ban` path (`crates/node/src/main.rs:92-98`). Signature
  verification is CPU-bound; rate limiting happens before verification.
- **Message size caps.** Votes ≤ 4 KB, evidence ≤ 16 KB, proposals ≤ 1 MB
  (the existing gossipsub max).
- **One signature per tuple.** A validator signs at most one prevote and one
  precommit per `(height, round)`; own-vote persistence enforces this across
  restarts.
- **Evidence gossip.** Relay each evidence hash at most once per 60 s and do
  not re-gossip beyond two hops (`MAX_SEEN` already caps the seen set).
- **Flood publish.** The current gossipsub config uses `flood_publish(true)`
  (`crates/network/src/lib.rs:207`) for tiny meshes. This makes vote traffic
  O(N²) per height. At N≤10 and 10 s blocks this is acceptable (hundreds of
  small messages per second); document it and revisit if the set grows.

## Integration points

### `crates/network/src/lib.rs`

- Add `PROPOSALS_TOPIC`, `VOTES_TOPIC`; subscribe in `NetworkNode::new`
  alongside `blocks_topic`/`txs_topic` (lines 217-220).
- Add `Command::PublishProposal`, `Command::PublishVote`, and
  `publish_proposal(&str)` / `publish_vote(&str)` (lines 169-171, 387-396).
- Add `NetworkEvent::Proposal(String)` and `NetworkEvent::Vote(String)` to
  the enum (line 27). In the swarm message match (lines 316-330), map by
  topic suffix, mirroring the existing block/tx mapping.
- Keep the topics chain-prefixed: `format!("{chain_id}/{PROPOSALS_TOPIC}")`.

### `crates/node/src/main.rs`

- `Shared` (lines 31-45): add a consensus handle, e.g.
  `consensus: tokio::sync::Mutex<ConsensusState>` or an
  `mpsc::UnboundedSender<ConsensusInput>` owned by a dedicated actor task.
  Keep `chain_lock` as the canonical-state writer lock.
- Event pump (lines 488-536): add match arms for `Proposal` and `Vote` that
  forward to the consensus actor. Do not touch state directly from the pump.
  Keep block admission handling for legacy mode and observer catch-up.
- `admit_block` (lines 118-127): keep as the structural gate, but add a
  round-aware wrapper `admit_proposal(shared, envelope)` that calls
  `validate_block_with_policy(..., &RoundAware { round })` and additionally
  verifies the proposal envelope signature. The existing
  `validate_block` call assumes the round-0 leader; round > 0 proposals must
  use the round-aware policy.
- `commit_block` (lines 130-158): split into:
  - `verify_certificate(cert, block)` — quorum count over distinct
    validators, each precommit signature verified against the reconstructed
    payload, all bound to the same `(height, round, blockHash)`;
  - `commit_finalized(shared, block, cert)` — takes `chain_lock`, applies,
    compares the root, sets `baseFee` from `AppliedBlock.next_base_fee`,
    calls `set_head`, writes `finalizedHeight`, `finalizedHash`,
    `meta:commit:<height>`, and optionally prunes old vote state.
  Only `commit_finalized` may advance the head. The legacy `commit_block`
  remains for non-BFT mode behind the feature flag.
- `validator_loop` (lines 161-229): replace the tick → produce → set head
  sequence with the round driver:
  - on a height start, compute the leader for `(H, 0)`;
  - if leader, build and publish a proposal from a scratch store;
  - otherwise wait for a proposal;
  - drive propose/prevote/precommit timers, broadcast votes, and call
    `commit_finalized` on quorum;
  - do not call `set_head` after producing (the current line 214 must go
    away in BFT mode).
  Keep the "no peers → don't produce" guard
  (lines 200-207) as an additional liveness guard, but it must not replace
  timeout-driven rounds.
- `sync_loop`/`sync_once` (lines 599-656): switch to
  `/finalized/since/{height}`, verify certificates before `commit_finalized`,
  and never advance the head without a certificate. Reset the consensus
  state machine to `finalized+1, round 0` after syncing.
- Add a `VNN_CONSENSUS` mode switch (`legacy` default, `bft` opt-in) and a
  startup check that `validatorSetHash` matches peers.

### `crates/exec/src/engine.rs`

- `validate_block` / `validate_block_with_policy` (lines 222-360): no
  signature change. Proposals use the round-aware policy; everything else
  stays. This is the main reason option (b) is cheap.
- `produce_block` (lines 1263-1362): keep the function but document and
  enforce that it may run on a scratch store. Remove the `baseFee` write
  (line 1359) from the proposal path; base fee becomes a commit-side effect.
  The block body persistence (line 1346) and receipt stamping (line 1350)
  on a scratch store are harmless but must be repeated on commit.
- `apply_block` (lines 1160-1219): unchanged; the returned
  `next_base_fee` must now be persisted by `commit_finalized`.
- `set_head` (lines 1378-1393): unchanged; only called from the commit path.
  It already writes `blockheight:<height>`, which is what makes
  `/blocks/since` finalized-only.
- Add `pub fn parent_header(store) -> Option<BlockHeader>` or reuse
  `current_head`; proposal validation needs the parent.

### `crates/exec/src/leader.rs`

- Add a round-aware policy without changing the `LeaderPolicy` signature:
  `pub struct RoundAware { pub round: u32 }` whose `leader_at` implements
  `sorted[((height - 1 + round) % N)]`. Keep `RoundRobin` as the round-0
  implementation for legacy mode and for tests.

### `crates/state/src/lib.rs`

Add methods and keys:

- `fn checkpoint_to(&self, path: &Path) -> Result<(), String>` using
  `rocksdb::checkpoint::Checkpoint` (rocksdb 0.22 exposes `checkpoint`).
- Meta keys:
  - `finalizedHeight` (u64), `finalizedHash` (string)
  - `commit:<height>` (commit certificate JSON)
  - `consensusLock` (`{height, hash, round}`)
  - `consensusOwnVote:<height>:<round>:<kind>` (signed vote JSON)
  - `consensusRound` (`{height, round}`; best-effort, reconstructable)
  - `evid:<sha256(canonical evidence)>` (evidence JSON)
  - `jailed:<address>` only if/when automatic jailing is enabled (not v1)
- Reuse `hash:<block_hash>` for block bodies so a proposal can be stored
  before finality without moving the head. `blockheight:<height>` continues
  to be written only by `set_head` on commit.

### `crates/node/src/rpc.rs`

- `/status` (lines 99-113): add `finalizedHeight`, `finalizedHash`,
  `consensusRound`, and `validatorSetHash`. Keep `height` equal to the
  finalized head in BFT mode.
- Add `GET /finalized/since/:index` returning `{block, certificate}` pairs
  for sync. Keep `/blocks/since` as-is.
- Add `GET /evidence` and `GET /evidence/:validator` for operations.
- `/metrics` (existing handler) should export
  `verinoo_finalized_height`, `verinoo_consensus_round`,
  `verinoo_consensus_votes_total{kind}`, and
  `verinoo_equivocations_total`.

### `chain_lock`

- `chain_lock` is taken by `commit_finalized` around the whole apply + state
  root check + `set_head` + finality metadata write.
- `sync_once` takes it per applied block, as today.
- Proposal construction and proposal verification do **not** take it; they
  only read a consistent head/checkpoint. Take it briefly to snapshot the
  head pointer if needed.
- The consensus actor is the only writer of vote/lock/round state; do not
  duplicate that state in `Shared`.

## Equivocation evidence

### What is stored

For each detected conflict, store a self-contained evidence record:

```json
{
  "kind": "double-vote",
  "validator": "ai_...",
  "height": 42,
  "round": 0,
  "a": { "type": "prevote", "blockHash": "A", "publicKey": "hex", "signature": "hex" },
  "b": { "type": "prevote", "blockHash": "B", "publicKey": "hex", "signature": "hex" }
}
```

Detected conflicts:

- two different proposal envelope signatures for the same `(height, round)`;
- two prevotes or two precommits by the same validator for the same
  `(height, round)` with different `blockHash` values;
- a precommit whose hash differs from that validator's own prevote at the
  same `(height, round)` (this is also a double-vote and is captured by the
  two-message form).
- A byzantine leader proposing different blocks to different peers is
  covered by proposal-envelope evidence; the envelope signature binds the
  round and hash.

### How it is verified

Genesis-seeded `ValidatorEntry` records have an empty `public_key`
(`crates/state/src/lib.rs:217-223`), so evidence cannot be checked against a
stored key. It does not need to be: each message carries `publicKey`, and
verification is:

1. `public_key_to_address(publicKey) == validator` using the existing
   address derivation (`crates/exec/src/engine.rs:23-32`).
2. Both signatures verify over the canonical vote payload reconstructed from
   the message fields.
3. The two messages have the same `(validator, height, round, type)` and
   different `blockHash`.

This is exactly the check `validate_block_with_policy` already performs for
producer keys. Store the evidence only after both signatures verify.
Evidence is gossiped on the votes topic and deduplicated by
`sha256(canonical_json(evidence))`.

### Slashing and removal under the current set limitation

Two independent limitations make live slashing/removal out of scope for v1:

1. Consensus membership is `params.validators`, a startup config list, not
   `store.active_validators()`. Removing a `val:` entry or executing
   `UnregisterValidator` does not change who may produce or vote. A running
   network can only change the set by a coordinated restart with a new
   `VNN_VALIDATORS` value.
2. `compute_state_root` does not cover `val:` entries
   (`crates/exec/src/engine.rs:930-976`). If consensus membership were
   derived from state, two nodes could disagree about the set without a
   state-root mismatch. That must be fixed (add a `validatorRoot` to the
   header or include sorted `val:` entries in the state root) before a
   state-derived set is safe.

Therefore:

- **v1 (this ADR): evidence only.** Store and gossip evidence, expose it
  over RPC, log it, and increment a metric. Do not alter the quorum, do not
  remove the validator, do not move funds. Operators must manually remove a
  misbehaving validator from `VNN_VALIDATORS` and restart the set. This is a
  documented operational procedure, not automatic slashing.
- **v2 (separate ADR): set changes.** Make the consensus set a function of
  finalized state at a deterministic height boundary (e.g., `setEpoch =
  floor(height / 1000)`), add sorted validator entries to the state root,
  and define a `jail` transaction/evidence application rule. Quorum for
  height `H` would be computed from the set committed at
  `setEpoch(H) - 1`. Until that exists, any "slashing" language is a
  non-claim.
- Even with v2, a genesis-seeded validator can be jailed but its bond is
  zero (`staked_amount: 0` for genesis entries), so there is nothing to
  burn. Economic slashing requires real bonded stake; that is a separate
  staking/registration change.

## Test plan

Build an in-process harness in `crates/consensus/tests/` (or
`crates/node/tests/` if the harness needs `Shared`). It must not open
sockets and must use a manual clock.

Harness shape:

- Four `TestNode`s, each with `StateStore::open_temp()`, a
  `ProducerKey::deterministic("validator", i)`
  (`crates/exec/src/engine.rs:1247-1258`), and a `ConsensusState`.
- Genesis: sorted addresses from the four keys, seeded via
  `seed_genesis_validators` (`crates/state/src/lib.rs:212`), then a genesis
  head as in `crates/exec/tests/lifecycle.rs:63-93`.
- A `Bus` that delivers `Msg` values to selected nodes, supports drop,
  duplication, reordering, and partition; `advance(dt)` fires timeouts.
- Assertions read `finalizedHeight`/`finalizedHash`, certificates, evidence
  keys, and own-vote keys directly from each store.

Required tests:

1. **Four validators finalize (happy path).** No faults. All four reach the
   same `finalizedHeight` and `finalizedHash`; each stored certificate has
   exactly 3 distinct precommits; `set_head` advanced only once per height;
   `/blocks/since` is finalized-only (assert `blockheight:` exists only for
   committed heights). Run for at least 5 heights to cover leader rotation.
2. **One byzantine node (equivocating leader or voter).** N=4, quorum=3.
   The Byzantine node sends two proposals (or two prevotes) for the same
   `(H, r)` with different hashes to different halves. The three honest
   nodes must finalize exactly one hash (whichever gathers a polka), never
   two different finalized hashes, and all must store the same evidence
   record. Assert `verinoo_equivocations_total == 1` and that the evidence
   verifies from the stored bytes alone.
3. **Double-vote evidence and restart.** Craft two correctly signed
   prevotes for the same `(H, r)` with different hashes. Feed them to one
   node; assert one `evid:` key appears, both signatures verify, and the
   validator is not removed from the quorum (v1 behavior). Restart the
   store, reload, and assert the evidence persists. Separately, sign a
   prevote, restart, and attempt to sign a different prevote at the same
   tuple: the node must return the stored vote and never emit a second
   signature.
4. **Partition and rejoin.** Split 2/2. Advance several rounds; assert no
   finality (quorum 3 impossible). Heal. Assert the next elected leader's
   proposal finalizes, the previously finalized block is never reorged, and
   stale votes from the minority's round are dropped. Also cover N=4 with
   one node offline: the remaining three (quorum 3) still finalize; with one
   offline and one byzantine, finality stalls (document expected liveness
   loss).
5. **Late start catch-up.** Run a cluster to finalized height 6. Start a
   fresh node with an empty store and `sync_once` against a peer. Assert it
   verifies 6 certificates, applies 6 blocks in order, sets
   `finalizedHeight = 6`, and then participates at height 7. Also cover the
   "has votes but not the block" case by delivering only votes and then the
   certificate; the block is fetched from the bundle endpoint and applied
   without replaying votes.
6. **Invalid proposal yields no prevote.** A byzantine leader proposes a
   block with a correct header signature but a wrong state root (or an
   unwinding transaction set). Honest nodes prevote nil; the height does not
   finalize at round 0; round 1's honest leader proposes a valid block and
   finality advances. Assert state was not mutated by the rejected proposal
   (canonical store head unchanged).

Invariant/property test: over randomized schedules with ≤1 Byzantine and
≥3 honest online, assert "no two nodes finalize different hashes at the same
height" is never violated. This is the safety property that matters; a
passing happy-path test alone does not validate the lock rules.

## Migration and versioning

- **Protocol version.** Envelope `v: 1`; topics end in `/1`. Unknown `v` or
  unknown `type` is dropped and counted, not parsed liberally.
- **Block version.** `HEADER_VERSION` stays 1. Round/finality live in the
  envelope and certificate, not the header. A future `HEADER_VERSION = 2`
  could add `commitRoot`/`validatorRoot`; that is a coordinated hard fork
  and is out of scope.
- **Activation.** Add `VNN_CONSENSUS=legacy|bft`, default `legacy` until the
  full test matrix is green. Activate BFT at a coordinated finalized height
  (`VNN_BFT_ACTIVATION_HEIGHT`) with all validators restarted together. A
  mixed legacy/BFT set is not supported: legacy nodes commit on proposal and
  BFT nodes wait for quorum, so they will disagree on head timing.
- **Data compatibility.** `hash:<hash>` bodies, `blockheight:<height>`, and
  the `head` meta format are unchanged. New keys are additive. A legacy
  database can start in `legacy` mode and later switch to `bft` at a
  finalized height; the reverse (BFT → legacy) is allowed for rollback but
  abandons finality guarantees from that height.
- **RPC.** `/status` additions are backwards compatible. `/finalized/since`
  is new. `/blocks/since` keeps its shape and semantics (finalized blocks
  only, because the head is finalized).
- **Rollback plan.** Keep the legacy commit path compiled and reachable
  behind the flag for one release. If a BFT bug is found, restart all
  validators in `legacy` mode; finality certificates become advisory.
- **Validator set migration.** Because the set is immutable in v1, changing
  it is a config rollout: publish the new `VNN_VALIDATORS` list and
  `validatorSetHash`, stop the set at a finalized height, restart together,
  verify `/status` agreement, then resume. Document that this is manual and
  not atomic.

## Effort estimate

One engineer familiar with the codebase; calendar weeks, including tests and
review but not external audit.

| Phase | Work | Estimate |
|---|---|---|
| 0 | State refactor: `StateStore::checkpoint_to`; proposal build/verify on scratch; move `baseFee` to commit; split `commit_block`; round-aware `LeaderPolicy` | 1 wk |
| 1 | `crates/consensus`: messages, canonical signing, round state machine, lock/unlock, own-vote persistence, unit tests | 1.5 wk |
| 2 | Network topics, `NetworkEvent` variants, consensus actor, event-pump wiring, timers, feature flag | 1 wk |
| 3 | Certificate persistence, `/finalized/since`, `/status`, certificate-verified `sync_once`, evidence storage + `/evidence` | 1 wk |
| 4 | In-process 4-validator harness, byzantine/partition/restart/late-start tests, state-root-mismatch test, invariant test | 1.5 wk |
| 5 | Ops docs, metrics, migration/rollback runbook, load sanity on a 4-node testnet | 0.5-1 wk |
| **Total** | | **5.5-7 wk** |

For comparison: option (a) 8-12 weeks; option (c) 7-10 weeks. The readiness
document currently budgets 4-6 weeks for "finality or honest framing"
(`docs/readiness.md:215`); 5.5-7 is the realistic range once the execution
refactor and the test matrix are included.

## Non-claims

- **Not implemented.** This is a proposal. No prevote/precommit, certificate,
  finality, lock, or evidence code exists in the repository.
- **No slashing in the live path.** Evidence is recorded, verified, stored,
  and exposed. No funds are moved, no validator is removed, and no quorum is
  changed automatically. Removal is a manual `VNN_VALIDATORS` change and a
  coordinated restart.
- **No validator-set change mechanism.** `RegisterValidator` /
  `UnregisterValidator` exist as transactions but do not affect consensus in
  v1. Genesis-seeded validators have zero bond, so even a future jail rule
  has nothing to burn.
- **No automatic HA.** There is no failover, no split-brain protection, and
  no leader lease. Running the same validator key on two hosts produces
  equivocation evidence and, if both sign, violates safety assumptions; the
  design detects it but does not prevent it.
- **No token.** Finality does not change fees, gas, or settlement. Fees
  remain testnet balances.
- **No BFT claim for N≤3.** N=3 has quorum 3 and tolerates zero faults;
  N=1 finality is self-attestation. N=4 (quorum 3) is the minimum that
  tolerates one Byzantine fault.
- **No light-client proof.** Certificates prove validator agreement on a
  block hash; they are not a succinct state proof and do not replace
  independent state-root verification.
- **No snapshot transfer.** Catch-up still replays finalized blocks from the
  caller's height and remains O(chain length).
- **No reorg protection for legacy nodes.** In `legacy` mode, a
  single-signature block is still treated as head. Only BFT mode gets
  finality.
- **Not an audit.** The lock/unlock and persistence rules are standard
  Tendermint rules, but they must be implemented exactly and reviewed
  independently before any "BFT" or "final" marketing claim.
