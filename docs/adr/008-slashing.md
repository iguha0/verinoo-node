# ADR 008: Equivocation Evidence Taxonomy and Slashing Groundwork

## Status

**Partially implemented — verifiable evidence primitives only; no penalties.**
`crates/consensus` implements the two-message evidence records for vote
equivocation (`Evidence`: two conflicting signed votes by one validator at one
`(chain_id, height, round, kind)`) and, as of this ADR, for double proposals
(`ProposalEvidence`: two signed proposals by one proposer at one
`(chain_id, height, round)` with different block hashes). Both verify
deterministically against only the evidence bytes and the known validator
set, both have a stable id (`sha256` over the canonical JSON of the record),
and both ids are now independent of discovery order. The node stores verified
vote evidence received on the evidence topic under `evid:<id>` keys. Not
implemented: node-side detection and relay of double-proposal evidence, the
mixed-kind record for "precommit differing from own prevote", the evidence
RPC/metrics surface (ADR 007 Phase 5), and every form of penalty — slashing,
jailing, removal, and fund movement remain unimplemented by design.

Date: 2026-11-19
Scope: `crates/consensus` (evidence types and verification), this document.
Node wiring is future work in `crates/node`/`crates/network`; see
"Remaining work".

## Context

ADR 004 defines equivocation evidence as part of the BFT finality design and
defers slashing to a separate ADR with the phrase "v2 (separate ADR)". This
is that ADR, narrowed to what is safe to build now:

- **Why a separate record for proposals.** A byzantine leader can propose two
  different blocks for the same `(height, round)` to different peers. The
  block header does not contain the round, so the *proposal envelope*
  signature over `(v, chainId, height, round, blockHash)` is what binds the
  tuple; two valid envelope signatures over different hashes are the offense.
  ADR 004 lists this conflict ("two different proposal envelope signatures
  for the same `(height, round)`") but until now nothing could record it:
  the only evidence type was vote-shaped.
- **Why no penalties yet.** Three independent limitations (from ADR 004
  "Slashing and removal under the current set limitation" and ADR 006):
  1. Consensus membership is the startup config list
     (`VNN_VALIDATORS` → `params.validators`), not state; removing a `val:`
     entry or executing `UnregisterValidator` does not change who may vote.
  2. `compute_state_root` does not cover `val:` entries, so a state-derived
     set could disagree across nodes without a state-root mismatch.
  3. Genesis validators have zero bond (`staked_amount: 0`); there is
     nothing to burn and no registration that bonds stake.
  Given those, any "automatic slashing" would be theater. What the chain
  *can* do honestly today is detect, verify, store, and surface signed
  proof of misbehavior. This ADR fixes that primitive and the policy around
  it, and specifies — without claiming — the future on-chain design.

## Evidence taxonomy

Four offense classes. All are defined over signed consensus messages; the
messages themselves are the evidence. Replaying a byte-identical message is
never an offense (same signature, dropped as a duplicate), and an empty
(`nil`) hash is just another value: `nil` plus a hash at the same tuple is a
valid conflict pair.

| # | Offense | Definition | Record | Status |
|---|---|---|---|---|
| 1 | Double prevote | two prevotes by one validator at one `(chainId, height, round)` with different `blockHash` | `Evidence`, `kind: prevote` | implemented |
| 2 | Double precommit | two precommits by one validator at one `(chainId, height, round)` with different `blockHash` | `Evidence`, `kind: precommit` | implemented |
| 3 | Precommit differing from own prevote | a precommit whose `blockHash` differs from that validator's prevote at the same `(chainId, height, round)` | signed prevote + precommit pair | not implemented — the current `Evidence` constructor requires matching kinds, so this needs a mixed-kind record (see "Remaining work") |
| 4 | Double proposal | two proposals by one proposer at one `(chainId, height, round)` with different `blockHash` | `ProposalEvidence` | primitive implemented; node detection and relay not wired |

ADR 004's note that offense #3 "is captured by the two-message form" is
directionally right — both signed messages exist and verify independently —
but the concrete `Evidence` struct enforces same-kind pairs, so it cannot
record #3 as-is. This ADR records that gap rather than papering over it.

## Deterministic verification rules

Verification is pure, stateless, and needs only: the evidence bytes, the
known validator set (sorted addresses), and optionally the expected chain id.
Nothing about the reporter is trusted; any observer with the same inputs
reaches the same verdict. Checks run in a fixed order and fail closed.

### `ProposalEvidence` (double proposal)

1. **Tuple match.** Both proposals and the envelope agree on
   `(chainId, proposer, height, round)`; otherwise
   `Error::ProposalEvidenceMismatch`.
2. **Differing hashes.** `a.blockHash != b.blockHash`; identical hashes are a
   duplicate proposal, not an offense (`Error::NotDoubleProposal`).
3. **Chain.** Envelope `chainId` is non-empty, and under
   `verify_with_chain` equal to the expected chain id
   (`Error::ChainIdMismatch` / `Error::EmptyChainId`).
4. **Per-proposal verification** (both `a` and `b`, each independently):
   version `v == 1`; `proposer` is in the known validator set
   (`Error::UnknownValidator`); `public_key_to_address(publicKey) ==
   proposer` — address derivation, so a forged `publicKey` field cannot
   frame a validator (`Error::AddressMismatch`); and the signature verifies
   over `PROPOSAL_DOMAIN || canonical_json({v, chainId, height, round,
   blockHash})` (`Error::InvalidSignature`).

### `Evidence` (double prevote / double precommit)

Same shape with the vote tuple `(chainId, validator, height, round, kind)`:

1. **Tuple match** across both votes and the envelope — including the same
   `kind` (`Error::EvidenceMismatch`).
2. **Differing hashes** (`Error::NotEquivocation`).
3. **Per-vote verification**: version, chain (non-empty; expected chain
   under `verify_with_chain`), known validator, address derivation, and the
   signature over `VOTE_DOMAIN || canonical_json({v, chainId, type,
   height, round, blockHash})`.

### Record identity

`id = sha256(canonical_json(record))`, hex-encoded. The constructor orders
the pair canonically — by `(blockHash, signature)`, which necessarily
differs for a valid pair — so the id does not depend on which message a node
observed first. Two honest nodes that see the same conflict store the same
`evid:` key, which is what ADR 004's "all must store the same evidence
record" test requires. (Before this ADR, discovery order leaked into the
id; constructing the pair in the other order produced a second key for the
same offense.)

### Storage and gossip

- **Store after verify, only.** The node writes `evid:<id>` → record JSON
  into its meta store only after `verify` passes
  (`crates/node/src/main.rs`, evidence event arm and the finality output
  path). Unverifiable payloads are dropped.
- **Gossip.** Evidence travels on the evidence topic
  (`verinoo/evidence/1`, chain-prefixed like the other consensus topics)
  with a 16 KiB payload cap in `crates/network`. Today the node *receives*
  vote evidence on that topic; it does not yet *publish* locally-detected
  evidence (the `PublishEvidence` command exists in `crates/network`, the
  node never calls it) — closing that loop is listed under "Remaining
  work".
- **Wire shape.** The evidence topic currently carries the vote record as
  the bare `Evidence` JSON (camelCase: `chainId`, `validator`, `height`,
  `round`, `kind`, `a`, `b`). `ProposalEvidence` adds a second shape
  (`chainId`, `proposer`, `height`, `round`, `a`, `b`). When node wiring
  lands, payloads must become distinguishable; the target is the ADR 004
  envelope `{"v":1,"kind":"double-vote"|"double-proposal", ...}` rather
  than try-both deserialization. Until then, a `ProposalEvidence` payload
  parsed as `Evidence` fails on the missing `validator`/`kind` fields and
  is dropped — acceptable, because no producer emits it yet.

## v1 policy: evidence only, operator-manual response

Unchanged from ADR 004's "v1 (this ADR): evidence only", now with the
complete offense taxonomy:

- **No automatic penalties.** No funds are moved, no validator is removed,
  jailed, or banned from voting, and no quorum is recomputed on evidence.
  Verification, storage, and gossip are the entire effect.
- **Operator review is the response path.** A misbehaving validator is
  handled by a human: read the stored `evid:` records (each re-verifiable
  from bytes alone), confirm the offense, then remove the validator by
  editing `VNN_VALIDATORS` and coordinating a set-wide restart, as
  documented in ADR 004 "Migration and versioning". This is a manual
  operational procedure, and it is the only one v1 offers.
- **Evidence is advisory to consensus.** Safety comes from the quorum and
  lock rules, not from evidence. A node does not change its voting behavior
  because it saw evidence; with N=4 and one byzantine validator the
  honest quorum of 3 finalizes safely with or without a recorded record.

## Future on-chain slashing and removal (design, not a claim)

This section is the target design for v2+ so that "slashing" language has a
meaning to grow into. **None of it is implemented**, and none of it may be
cited as shipped behavior. Prerequisites are ordered: later steps depend on
earlier ones.

1. **Bonded stake (prerequisite for any burning).** A registration path
   where a validator posts a real bond (`ValidatorEntry.staked_amount > 0`
   via a bond transaction, not genesis seeding). Genesis-set validators
   have nothing to burn; economic slashing is meaningless until bonds exist.
2. **State-root coverage (prerequisite for state-derived membership).**
   Include sorted `val:` entries in `compute_state_root`, or add a
   `validatorRoot` to the header (`HEADER_VERSION = 2`, a coordinated
   fork). Until the set is committed by the root, two nodes can disagree
   about membership without detection (ADR 004, ADR 006).
3. **Set-change epochs (prerequisite for deterministic penalties).** Make
   the consensus set a function of finalized state:
   `setEpoch = floor(height / 1000)`; quorum for height `H` is computed
   from the set committed at `setEpoch(H) - 1`. An offense at height `H`
   is judged by the set active at `H`.
4. **Evidence window.** Evidence for an offense at height `H` must be
   submitted (as a transaction) before `H + W` with `W` bounded — e.g.
   100k heights (~12 days at 10 s blocks). Expired evidence is pruned from
   the pool and can no longer be applied. This keeps the evidence set
   bounded and prevents "sit on evidence, slash years later" griefing.
5. **Evidence application.** Anyone submits the record as a transaction;
   execution verifies it with the same pure checks (known set for the
   offense's epoch, chain id, both signatures, address derivation) and
   applies a deterministic outcome per offense class:
   - double prevote / double precommit: burn a fixed fraction of the bond
     (e.g. 1%) and jail (remove from the active set until unbonded);
   - precommit differing from own prevote: same as a double vote;
   - double proposal: same as a double vote, plus leader-specific handling;
   - repeated offenses within the window: tombstone — permanent removal
     with a partial burn. Slashed amounts burn outright (no recipient), so
     evidence submission has no profit motive beyond honest reporting.
6. **Activation.** The penalty rules activate at a coordinated finalized
   height, only after (1)–(3) exist; offenses before activation are never
   punished retroactively. Mixed legacy/BFT or bonded/unbonded
   configurations are not supported.

### Explicit non-claims

- No slashing, jailing, removal, or fund movement of any kind exists in this
  repository. "Evidence" ≠ "slashed"; the v1 policy is operator review.
- Evidence does not *prevent* equivocation; it detects it after the fact.
  Two validator keys on two hosts (ADR 006's HA hazard) still produce
  evidence only *after* both have signed.
- Evidence delivery is best-effort gossip. A partitioned minority may never
  see or relay a record; safety never depends on evidence existing.
- Double-proposal detection is not wired into the node: the proposals-topic
  handler is a no-op in the current single-producer integration and
  `ConsensusState::on_proposal` validates one proposal per round without a
  per-`(height, round)` proposal cache. Only the verifiable record exists.
- The "precommit differing from own prevote" record is not implemented; the
  current constructor requires same-kind pairs.
- No evidence RPC, metrics export, or retention policy is implemented
  (ADR 007 Phase 5 remains unchecked). `GET /evidence`,
  `verinoo_equivocations_total`, and evidence relay (publishing locally
  detected records) are future node work.
- Nothing here implements validator-set changes, automatic HA, or snapshot
  transfer (ADR 006), and no "BFT" claim beyond ADR 007's phase gates.

## Remaining work

Ordered, node-side unless noted:

1. **Proposal cache + detection.** Keep the first valid proposal per
   `(height, round, proposer)`; on a second, differing one, construct
   `ProposalEvidence`, verify, store under `evid:<id>`, and relay. Extends
   ADR 004's vote-cache rule ("same key + different hash is evidence") to
   proposals, and requires the node to actually consume the proposals
   topic.
2. **Evidence relay.** Call the existing `PublishEvidence` network command
   for locally-detected records (both kinds) so evidence propagates beyond
   direct observers; keep ADR 004's relay limits (once per 60 s, capped
   hops).
3. **Tagged wire envelope.** Emit and parse the `{"v":1,"kind":...}`
   evidence envelope from ADR 004 on the evidence topic so vote records and
   proposal records are distinguishable.
4. **RPC/metrics.** `GET /evidence` and `GET /evidence/:validator`
   re-verifying stored bytes; `verinoo_equivocations_total` (ADR 007
   Phase 5).
5. **Mixed-kind record.** Generalize the vote record (or add a third type)
   to cover "precommit differing from own prevote" so offense #3 is
   recordable; mirror the verification rules above.
6. **Bond + state-root + epochs.** The v2 prerequisites in "Future on-chain
   slashing and removal" — a separate implementation effort, and a
   prerequisite for any penalty claim.

## Test coverage

`crates/consensus/tests/proposal_evidence.rs` (new) and
`tests/signed.rs` (existing, unchanged) cover:

- valid double-proposal evidence verifies under `verify` and
  `verify_with_chain`; id is stable across reconstruction and independent
  of pair order; different pairs produce different ids;
- same-hash pairs rejected at construction and at verification;
- unknown validator, wrong chain (`verify_with_chain` and cross-chain
  pairs), mismatched tuple (height/round/proposer/inner-chain, both at
  construction and via tampered envelope fields), and bad or malformed
  signatures rejected;
- camelCase serialization round-trip;
- vote evidence (double prevote and double precommit) still constructs,
  verifies, and dedups identically after this change.
