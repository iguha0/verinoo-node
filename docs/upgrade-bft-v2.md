# Activation and operations runbook: BFT v2 (`bft-v2`)

Scope: move a verinoo network from the shipped BFT certificate layer
(`VNN_CONSENSUS=bft`, v1) to full BFT (`VNN_CONSENSUS=bft-v2`): proposal-on-clone
execution, certificate-gated commit, and pacemaker-driven view changes for an
offline leader.

**Status: planned, not shipped.** At the time of writing the released binary
accepts `VNN_CONSENSUS=legacy|bft` only; the live testnet is a single validator
in `bft` mode; `bft-v2` node wiring is in progress (ADR 007, Phases 4-6). This
runbook becomes operative when a release that accepts `VNN_CONSENSUS=bft-v2`
ships. Nothing described here is deployed, and nothing here has been
independently audited.

Design source: `docs/adr/004-bft-finality.md` (protocol, wire format, timeouts,
non-claims), `docs/adr/007-bft-implementation-checklist.md` (phase gates and
the test matrix), `docs/plans/ha-bft-phases.md` (Phase 5). General node
operations (backups, `/ready`, snapshots, promotion) are in
[`docs/operations.md`](operations.md); this document only covers the v2
activation and the three BFT-specific incident runbooks.

## 1. What changes from v1 to v2

| Area | `legacy` (shipped) | `bft` v1 (shipped) | `bft-v2` (target) |
|---|---|---|---|
| Block production | single-producer round-robin | same; leader must be online | leader proposes; execution on a scratch checkpoint clone |
| Head advance | on block admission | on admission; certificates are a parallel finality layer | only via `commit_finalized` after a verified quorum certificate |
| Offline leader | chain stalls | chain stalls (no view changes) | view change to the next round leader (pacemaker) |
| `/status` `height` | produced head | produced head; `finalizedHeight` trails | finalized head (`height` == `finalizedHeight` once committed) |
| Finality data | none | `finalizedHeight`, `finalizedHash`, `meta:commit:<H>` | same; head itself is certificate-gated |
| Catch-up | `/blocks/since` | + certificate-verified `/finalized/since` | same; serves finalized history only |
| Evidence | none | stored under `evid:` keys, logged; no RPC | same, plus `/evidence` endpoints per the release |
| Slashing / removal | none | none (manual `VNN_VALIDATORS` change) | none (unchanged) |

With N validators the quorum is `floor(2N/3) + 1`. N=4 gives quorum 3 and
tolerates exactly one Byzantine **or** one crash fault — not both. N<=3
tolerates zero Byzantine faults; N=1 finality is self-attestation. Liveness
requires at least 2/3 of validators online and connected; a partition stalls
finality by design.

## 2. Prerequisites (gate for Phase A)

Do not start Phase A until every item holds. Each is a hard stop, not advice.

1. **At least four independent validators.** Four distinct hosts, four
   validator keys, four operator-controlled seeds, no shared infrastructure
   that one failure can take out. Four nodes run by one operator are
   self-attestation regardless of N; the certificate claim is only as strong
   as the independence of the key holders.
2. **Identical `VNN_VALIDATORS`** on every node: the same sorted, deduplicated
   address list. Verify `validatorSetHash` is byte-identical on every
   `/status` and `validatorSetConsistent` is true on every `/ready`.
3. **Same models and state root.** `/v1/models` lists the same ids and
   `weightsHash` everywhere; `modelSetHash` matches; `VNN_MODEL_DIR` points at
   the same set. A missing model file diverges state roots at the first
   execution.
4. **Synced chains.** All nodes report the same head hash and height (and, in
   `bft` mode, the same `finalizedHeight`/`finalizedHash`) before each phase
   boundary. A node behind must finish catch-up first.
5. **Identical timing.** Same `VNN_BLOCK_TIME_MS` (or genesis default) and
   same chain id on every node. NTP sane on every host.
6. **Backups.** A fresh stopped-service backup
   (`scripts/backup.sh`) plus its checksum on **every** node immediately
   before Phases B and C. The pre-Phase-B archive is the only honest rollback
   artifact once v2 certificates exist (section 5).
7. **Monitoring.** `/ready` and `/status` polled on every node with alerting
   on a flat height, `validatorSetHash` divergence, and (in the v2 release)
   `verinoo_finalized_height` / `verinoo_consensus_round` /
   `verinoo_equivocations_total` from `/metrics`. Journals watched for
   `equivocation evidence recorded` and round-change messages.
8. **A pinned release.** All binaries built from the same tagged commit with
   a published SHA-256; the release notes name the exact environment
   variables for the mode switch and the timer controls. Mixed binary
   versions during a phase are unsupported.
9. **No mixed consensus modes, ever.** `legacy`, `bft` and `bft-v2` nodes
   disagree on when the head advances; a mixed set is undefined behavior
   (ADR 004 "Activation"). Every mode change is a coordinated
   stop-all/restart-all, in one window.

## 3. Phased activation plan

Three phases, strictly in order. Do not skip a phase or run two switches in
one window: Phase B and Phase C change different invariants and must be
observed separately.

### Phase A — binary soak (v2 binaries, v1 semantics)

Deploy the `bft-v2` release binary on **every** node while leaving
`VNN_CONSENSUS` at its current value (`legacy` or `bft`). No configuration
changes; chain behavior must be identical to before. This phase proves
binary, database, model-set and monitoring compatibility without touching
consensus semantics.

Soak for at least 48 hours (or the pilot's agreed window) with client traffic
at normal levels.

**Acceptance criteria — all must hold before Phase B:**

- [ ] `/ready` is 200 on every node for the whole window; height advances on
      schedule; no unexpected service restarts.
- [ ] No `state root mismatch` or `wrong producer for height` in any journal.
- [ ] `validatorSetHash`, `modelSetHash` and `chainId` identical across nodes
      at every spot check.
- [ ] In `bft` mode: `finalizedHeight`/`finalizedHash` present and advancing;
      `meta:commit:<H>` certificates persisting.
- [ ] Zero `evid:` records created during the soak (no equivocation evidence).
- [ ] Catch-up check: restart one replica mid-soak; it catches up via
      `/finalized/since`, adopts the finalized height, and never votes below
      it.
- [ ] A post-soak stopped-service backup exists on every node.

**Rollback:** reinstall the previous binary per node (section 5.1). Nothing
protocol-level has changed; this phase is always reversible.

### Phase B — coordinated mode switch (`VNN_CONSENSUS=bft-v2`)

One maintenance window, all validators, no exceptions. From here on the head
advances only on a verified quorum certificate.

1. Announce the window; pause client submissions.
2. Record on every node: head height, head hash, `finalizedHeight`,
   `finalizedHash` (they must agree across nodes before proceeding).
3. Stop **all** validators. Take and verify a stopped-service backup on every
   node. Retain the old binary.
4. Set `VNN_CONSENSUS=bft-v2` and `VNN_BFT_TIMERS=0` on every validator, and
   the activation height (`VNN_BFT_ACTIVATION_HEIGHT`, if the release uses
   it) to the next height above the recorded head. Values must be identical
   on every node.
5. Start all validators together. Confirm the same `chainId`,
   `validatorSetHash` and model set on every `/status`.
6. Watch the first heights live before resuming client traffic.

**Acceptance criteria:**

- [ ] Every node reports `"consensus": "bft-v2"`.
- [ ] The head advances **only** with a certificate: for every new height H,
      the head hash equals `finalizedHash` and a `meta:commit:<H>`
      certificate exists. No height advances on a proposal alone.
- [ ] At every observed height, all nodes report the same
      `finalizedHeight`/`finalizedHash` (safety).
- [ ] A late-restarted node syncs via `/finalized/since` and resumes voting
      at the current head, never at historical heights.
- [ ] Negative check (drill): a test block without a valid certificate is
      refused by every honest node; the head does not move.
- [ ] Commit-time state-root mismatch (if drilled) halts the node with an
      error rather than advancing an unverified head.

**Known limitation during Phase B:** with `VNN_BFT_TIMERS=0`, an offline
scheduled leader still stalls production, exactly as in v1. That is expected
and is the reason Phase C exists. Restore the leader; do not treat this stall
as a v2 failure and do not downgrade a node to `legacy` to "keep producing"
— that creates a mixed-mode network.

**Rollback window:** from the Phase B restart until the **first v2
certificate** is created, the switch is reversible: stop all validators,
revert `VNN_CONSENSUS`, restart together (section 5.2). After the first v2
certificate, section 5.3 applies.

### Phase C — enable timers and view changes

After Phase B has soaked cleanly (at least 24 hours), enable the
view-change timers on all validators in one coordinated restart by setting
`VNN_BFT_TIMERS=1` (the default in `bft-v2`; Phase B ran with
`VNN_BFT_TIMERS=0`). The schedule is ADR 004's:
`timeout_propose = min(3s * 1.5^r, 15s)`,
`timeout_prevote = min(2s * 1.5^r, 10s)`,
`timeout_precommit = min(2s * 1.5^r, 10s)`, `R_MAX = 8` with capped timeouts
and continued leader rotation. Identical settings on every node.

**Acceptance criteria (drills on the pilot cluster, in this order):**

- [ ] **Offline-leader view change:** stop the scheduled leader. The chain
      must keep finalizing: rounds advance, the next round leader's proposal
      commits with the remaining quorum (N=4: the three online validators
      still reach quorum 3). Head advances within a bounded number of
      rounds, not never.
- [ ] **Partition/heal safety:** firewall the network into 2/2. No
      finality while split (quorum 3 is impossible) — a flat
      `finalizedHeight` during the split is the **correct** outcome. Heal.
      The next leader's proposal finalizes; every node reports the same
      `finalizedHash` history; at no height do two nodes finalize different
      hashes; stale minority-round votes are discarded after the heal.
- [ ] **Restart/amnesia:** restart a validator mid-round. It reloads its
      stored own vote, re-broadcasts it verbatim, and never signs a second,
      different vote for the same `(height, round, kind)`. No new `evid:`
      record names the restarted validator.
- [ ] **Evidence drill:** inject a correctly signed double vote from a test
      validator. Honest nodes record it under `evid:`, log `equivocation
      evidence recorded`, and the validator is **not** removed from the
      quorum.
- [ ] Phase B criteria still hold: head advances only with certificates.
- [ ] Soak at least 48 hours with timers on, including at least one planned
      validator restart.

### Evidence recording (all phases)

Record every acceptance check: date/time, heights and hashes observed,
commands run, journal excerpts, `evid:` ids where relevant, the binary
SHA-256, and the operator sign-off. Store this with the ops log. The claim
"bft-v2 activated on this testnet" is only honest with this record; without
it, say "bft-v2 binaries deployed" at most. Per ADR 007, activation is also
not a "mainnet finality" claim.

## 4. Monitoring during and after activation

| Signal | Where | Alarm |
|---|---|---|
| consensus mode | `/status` `consensus` | any node not reporting the expected mode |
| finalized head | `/status` `finalizedHeight`/`finalizedHash`, `/metrics` `verinoo_finalized_height` | flat while >= 2/3 validators online; divergent hashes across nodes |
| rounds | `/metrics` `verinoo_consensus_round` | rounds advancing without commits, or stuck at one round |
| votes | `/metrics` `verinoo_consensus_votes_total{kind}` | prevotes without matching precommits |
| equivocation | journals, `verinoo_equivocations_total`, `evid:` keys | any value above zero |
| validator set | `/status` `validatorSetHash`/`validatorSetConsistent` | divergence across nodes |
| readiness | `/ready` | HTTP 503 or a non-null `reason` |
| peers | `/status` `peerCount` | below N-1 during a drill or in steady state |

## 5. Rollback rules and irreversibility

### 5.1 Phase A (binary only)

Per-node binary rollback at any time: stop the node, reinstall the previous
binary from its saved copy, start, confirm `/ready`. No protocol state has
changed.

### 5.2 Phase B before the first v2 certificate

The switch is reversible as a set: stop **all** validators, revert
`VNN_CONSENSUS` (and the activation height) on every node, restart together,
confirm all nodes agree on the head. No v2-certified history exists, so
nothing is discarded. This window may be minutes long — from the restart
until the first certificate — and it cannot be extended.

### 5.3 After the first v2 certificate

**v2-certified history is irreversible with the same validator set.** Once
`floor(2N/3)+1` validators have signed a precommit certificate for height H:

- Honest validators refuse to vote at any height <= `finalizedHeight` and
  refuse to sign a second, different value for the same
  `(height, round, kind)`. The same set cannot be made to certify different
  blocks at certified heights.
- Restarting a single node from a pre-certificate backup against a chain
  that has v2 certificates is undefined: it will disagree about the
  finalized head, may manufacture evidence, and can stall the network.
  Never do this.
- "Rolling back" below the first v2 certificate means discarding blocks that
  a quorum signed. The only honest procedure is a **coordinated network
  reset**: stop every node; restore the pre-Phase-B backups everywhere (or
  re-genesis); accept that every receipt anchored to the discarded heights
  becomes unverifiable; disclose the incident and the reset height to every
  pilot participant.

### 5.4 Downgrade is not rollback

Restarting some or all validators into `bft` (v1) or `legacy` **after** v2
certificates exist is not a rollback; it is a mixed-mode network. v1 nodes
advance the head on block admission and ignore certificate gating, so v1 and
v2 nodes disagree about the head by construction. ADR 007 keeps the legacy
and `bft` paths compiled and reachable for one release after activation —
that escape hatch exists for the coordinated reset above, not for a live
per-node downgrade.

## 6. Runbook: stuck chain (no finality, head flat)

Symptoms: `height` or `finalizedHeight` flat, `/ready` 503 `stale_head`,
rounds stuck or cycling.

1. Read `/status` on **every** validator: `consensus`, `height`,
   `finalizedHeight`, `finalizedHash`, `validatorSetHash`, `peerCount`.
   Two nodes reporting different `finalizedHash` values at the same height is
   a **safety incident** — stop and go to section 7 immediately.
2. `validatorSetHash` differs across nodes: configuration drift. Do not
   restart into a mode change. Fix `VNN_VALIDATORS` on the divergent node(s),
   then restart them; confirm hashes match before anything else.
3. `peerCount` low or zero: network partition. A partition must not finalize
   (quorum is unreachable) — a flat `finalizedHeight` during a split is
   correct behavior. Restore connectivity, then confirm every node reports
   the same `finalizedHash` before resuming traffic.
4. Fewer than quorum validators online (e.g. two of four down): the stall is
   **by design**. Do not "restore liveness" by downgrading a node to
   `legacy` or `bft` — that creates a mixed-mode network. Restore a third
   validator (host fix or backup restore onto a fenced replacement host)
   instead.
5. Quorum online, timers on, but flat: check journals for round-change
   messages. Rounds should advance (`R_MAX` caps the timeout growth but
   leader rotation continues). Rounds advancing without commits points at a
   proposal/verification fault (often a model or state-root divergence —
   compare `/v1/models` and `modelSetHash`, see the `state root mismatch`
   entry in `docs/operations.md`). Rounds not advancing at all is a pacemaker
   bug: collect logs from all nodes and halt the set pending triage.
6. One node stuck while the others finalize: a laggard, not an emergency.
   Restart it; it must catch up through `/finalized/since`. If it refuses to
   adopt certified history with a state-root mismatch, fix its model set —
   do not delete its database.
7. Quorum restored and still stalled for several round schedules: this is a
   liveness bug in v2. Coordinated halt (stop all validators), preserve all
   data, and decide between holding for a fix and the network reset in
   section 5.3. Never restart individual nodes into different consensus
   modes to "keep things moving".

## 7. Runbook: Byzantine validator (equivocation observed or suspected)

1. **Verify the evidence from its stored bytes** (`evid:` records, or
   `GET /evidence` when the release ships it): both signatures verify over
   the reconstructed canonical payloads, both derive to the validator
   address, same `(validator, height, round, kind)`, different
   `blockHash`. Only then is it equivocation.
2. **Check safety first.** Every honest node must report the same
   `finalizedHash` at every height. With N=4 and one Byzantine validator,
   the three honest nodes still reach quorum 3, so finality must continue
   and must never fork. Divergent finalized hashes at one height means the
   safety argument has failed: coordinated halt, preserve everything,
   network reset per section 5.3 — this is the worst case, not a routine
   incident.
3. **Expect slower liveness, not a halt.** A Byzantine validator that
   withholds votes or sends conflicting proposals delays rounds; view
   changes rotate past a Byzantine round leader. Do not remove validators
   mid-incident unless safety is threatened; removing one node from a
   four-node set leaves N=3, which tolerates zero faults.
4. **Contain the cause if operational.** The most common source of
   "Byzantine" behavior is the same `VNN_KEY_SEED` running on two hosts
   (for example an unfenced promotion). If two hosts answer for the same
   validator address, fence one immediately: stop the process, disable its
   supervisor, block it at the hypervisor/network layer.
5. **Removal is manual, never automatic.** No slashing exists in any mode:
   evidence is recorded, funds never move, and the quorum never changes by
   itself. To remove a validator: publish the new `VNN_VALIDATORS` list and
   its `validatorSetHash`; stop the whole set at a finalized height in a
   new window; change the list on every node; restart together; confirm
   `/status` agreement. Add a replacement validator in the same window
   where possible — a three-validator set tolerates zero Byzantine faults.
6. **Record the incident:** evidence ids, affected heights and rounds, the
   decision and its rationale, in the ops log with the phase evidence
   (section 3). Do not prune or wipe any validator database while the
   investigation is open; the store is the evidence.

## 8. Runbook: evidence incident (equivocation reported, suspected compromise)

1. **Preserve before anything else.** Copy the `evid:` records, the
   relevant journal window, and `/status` snapshots from every node, with
   timestamps. Do not wipe, prune or "clean" any database; do not restart
   nodes into other consensus modes.
2. **Independently verify.** Re-verify both signatures over the canonical
   payloads, the key-to-address derivation, and that the two messages
   conflict (same tuple, different hash). An evidence record that fails
   verification is itself an incident (someone forged or corrupted a
   record) and should be kept, not deleted.
3. **Classify the cause.** Evidence alone cannot distinguish: (a) a
   duplicate host running the same seed (operational), (b) a compromised
   key, (c) malicious modified software. Inspect the hosts: process lists,
   seed storage, login history, deploy logs.
4. **Duplicate host:** fence the extra host (section 7.4), restart the
   surviving host from its own database, and confirm no further evidence
   appears over several heights.
5. **Suspected compromise:** treat the validator key as burned. Follow the
   coordinated rotation procedure in `docs/operations.md` ("Validator key
   rotation"), and note the hard limit: the compromised address keeps its
   voting power until the manual set change in section 7.5 completes. There
   is no emergency on-chain revocation.
6. **Disclosure.** Certified heights stay final and their receipts stay
   verifiable — equivocation does not retroactively void finality. Record
   the incident in the ops log and report it to pilot participants with the
   evidence bytes. Any "penalty" is contractual or legal and off-chain;
   there is no slashing to point to.

## 9. Non-claims

Say none of these in any material that references bft-v2:

- **"Automatic slashing or validator removal."** Evidence is recorded and
  exposed; funds never move automatically; the quorum never changes on its
  own. Removal is a manual, coordinated `VNN_VALIDATORS` change.
- **"Trustless" or a multi-operator guarantee.** Certificates are exactly as
  trustworthy as the independence of the operators holding the keys. One
  operator running all validators gets self-attestation with extra steps,
  regardless of N.
- **"Light client."** Certificates prove validator agreement on a block
  hash. They are not a succinct state proof and do not replace
  state-root recomputation by a full node.
- **"Guaranteed liveness."** Liveness requires at least 2/3 of validators
  online and connected. View changes recover from a bad or offline leader;
  they do not recover from the loss of quorum. Partitions stall finality by
  design.
- **"Audited."** The lock and persistence rules are standard Tendermint
  rules, but no independent review has happened. Activation evidence
  (section 3) is the only supportable claim, and it covers this testnet
  only.
- **"Byzantine fault tolerant" for N<=3.** N=4 (quorum 3) is the minimum
  that tolerates one Byzantine fault, and it tolerates exactly one — one
  crash plus one Byzantine validator still stalls the chain.
- **"Validator set changes from consensus."** `RegisterValidator` /
  `UnregisterValidator` do not affect the consensus set in any mode; the
  set is the configured `VNN_VALIDATORS` list, changed only by coordinated
  restart.
- **"HA / failover."** No automatic failover, no split-brain protection, no
  fencing primitive; snapshots do not carry certificates. Promotion stays
  manual per `docs/operations.md`.

## 10. Assumptions this runbook makes

These must hold for the plan above to be executable; the shipping release
notes supersede any working names.

1. The v2 release accepts `VNN_CONSENSUS=bft-v2`, requires
   `VNN_VALIDATOR=true` for it (as `bft` does today), and rejects mixed
   deployments loudly (`validatorSetHash` mismatch, mode disagreement).
2. The coordinated switch uses `VNN_BFT_ACTIVATION_HEIGHT` (ADR 007 Phase 6)
   or an equivalent pinned-height mechanism named by the release.
3. Timer/view-change controls exist and default to the ADR 004 schedule,
   with a release-documented way to keep timers off for Phase B and enable
   them for Phase C.
4. `/status` keeps reporting `consensus`, `finalizedHeight` and
   `finalizedHash` with the same meanings; `/metrics` adds
   `verinoo_finalized_height`, `verinoo_consensus_round`,
   `verinoo_consensus_votes_total{kind}` and `verinoo_equivocations_total`;
   `/evidence` ships per ADR 007 Phase 5 or evidence stays store-only.
5. Four independent validators exist before Phase A (section 2.1) and remain
   four through Phase C; the 4-validator local cluster result (v1 mode)
   is treated as a smoke test, not activation evidence.
6. Upgrade order is fixed: Phase A (binaries) -> Phase B (mode) -> Phase C
   (timers), each with its own acceptance record; the rollback windows are
   exactly those in section 5.
