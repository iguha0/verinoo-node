# ADR 005: Verification for Production LLM Inference

## Status

**Proposed — not implemented.** This document describes a target design and
the integration work required. No code in this repository implements an
optimistic LLM executor, a bonded challenger, a TEE attestation path, or
integer sampling. Nothing here may be cited as shipped behavior.

Date: 2026-11-15
Scope: `crates/exec`, `crates/runtime`, `crates/node`, `crates/network`,
`crates/state`, plus a new verification harness.

## Context

The current inference path is consensus re-execution. `execute_deterministic_tasks`
(`crates/exec/src/engine.rs:1162`) runs every pending task on every validator
during `apply_block` (`crates/exec/src/engine.rs:1312-1313`), and again during
`produce_block` (`crates/exec/src/engine.rs:1411-1412`). `stamp_receipts`
(`crates/exec/src/engine.rs:1225`) anchors the output hash to the signed block,
and `/v1/verify` (`crates/node/src/rpc.rs:616`) recomputes the output to check
it. This is honest verification: a receipt is accepted because the output was
computed as part of consensus state, not because an operator asserted it.

That model has a hard size limit. The largest model that runs today is
`tiny-stories-1m`, a 3.75M-parameter GPT-Neo decoder with greedy decoding
(`docs/decoder.md`). A 7B model is about 1,900 times larger; a 70B model is
about 18,700 times larger. Re-executing a 7B/70B forward pass on every
validator is not economically or operationally viable:

- 7B fp16 weights are ~14 GB; 70B fp16 weights are ~140 GB. Every validator
  would need that memory and the GPUs to use it.
- A 70B generation of 256 tokens takes tens of seconds on a multi-GPU host.
  The current block time is 10 s (`crates/state/src/genesis.rs:6`) and
  `apply_block` is synchronous. The chain cannot wait for N validators to
  generate the same 256 tokens.
- Cost scales by N. With four validators, every task pays for four full
  generations, whether or not anyone disputes the result.
- Every validator must run the same LLM kernels bit-for-bit. GPU libraries
  are not generally bitwise deterministic across batch sizes, hardware, or
  driver versions. The current Q16.16 runtime avoids this by using integer
  arithmetic on CPU; that is not a practical 70B serving path.

The existing dispute game does not solve this. `ChallengeResult` opens a
layer-bisection game (`crates/exec/src/engine.rs:631`) whose layer shapes come
from `get_layer_spec` (`crates/exec/src/spec.rs:44`), which fabricates shapes
from a model name. `docs/readiness.md:64-68` already says exposing that game
for a real model would be misleading. For an LLM, committing to every
intermediate activation would be gigabytes per task, so layer bisection is not
the answer either.

The design goal is narrow: keep the receipt meaningful for 7B/70B models
without making every validator execute the full model, and without claiming a
trust model that the deployment does not have.

## Options

### (a) Optimistic executor + bonded challenger

One selected executor runs the model and posts a claim with a bond. A
challenge window opens. If nobody challenges, the claim is paid. If someone
challenges, one or more verifiers re-execute the task; the losing side loses
its bond.

- **Buys:** execution cost is 1x in the honest case, plus Kx only on dispute.
  No new hardware trust. It fits the existing `task`, `game`, and bond
  vocabulary (`InferenceTask` at `crates/state/src/models.rs:40`,
  `ChallengeResult` at `crates/exec/src/engine.rs:631`).
- **Costs:** a credible challenge requires at least one independent party
  with the same model and a deterministic runtime. With a single operator,
  the "challenger" is the operator, so the fraud proof is not credible. The
  challenge window also needs finality or a depth assumption; a reorg can
  otherwise erase a claim or a challenge.
- **Risk:** high if presented as trustless before independent verifiers
  exist. Medium technically, because the state machine and bond accounting
  are straightforward but need exact tests.

### (b) TEE attestation

Run the model inside a confidential VM or confidential GPU and put a hardware
quote in the receipt. The quote binds the model hash, tokenizer hash, runtime
hash, input commitment, sampling parameters, output hash, and a nonce.

- **Buys:** one execution, no challenge game, no independent challenger.
  Verification is a quote check against vendor roots and a measurement
  allowlist.
- **Costs:** trust in the hardware vendor and the attestation service; side
  channels; limited GPU support and memory; a performance overhead; a new
  revocation and freshness story. A TEE proves that the attested code ran,
  not that the model is semantically correct. A buggy runtime inside a TEE
  still produces an attested wrong answer.
- **Risk:** medium-high operationally, low protocol complexity. It is a
  different trust claim: "trust the enclave" instead of "recompute or
  challenge".

### (c) ZK proof of inference

Prove the forward pass in zero knowledge.

- **Buys:** the strongest verification story: a succinct proof that the
  committed model produced the committed output.
- **Costs:** proving a 7B transformer forward pass is many orders of
  magnitude more expensive than re-executing it today; 70B is not close.
  Memory, prover time, and circuit/tooling maturity are all blockers. Even
  small-model zkML is experimental.
- **Risk:** not viable for this product in the relevant time frame. Do not
  schedule it as an implementation phase.

### Comparison

| | (a) Optimistic | (b) TEE | (c) ZK |
|---|---|---|---|
| Executions per undisputed task | 1 | 1 | 1 + proof |
| Executions per dispute | K verifiers | 0 | 0 |
| New hardware trust | no | yes (vendor) | no |
| Independent challenger required | yes | no | no |
| Deterministic runtime required | yes | no (attested) | yes (circuit) |
| Works with a single operator today | no (not credibly) | yes, if hardware exists | no |
| 7B/70B viable now | yes, with caveats | limited availability | no |
| Integration cost | medium | medium-high | very high |

## Recommendation

**Use option (a) as the default path for models too large for consensus
re-execution, keep full re-execution for small deterministic models, and treat
option (b) as an optional attested fast path for deployments that have
confidential hardware. Do not schedule option (c).**

Reasons, in order of weight:

1. It preserves the product claim that matters: a receipt binds a specific
   model, tokenizer, sampling configuration, and output, and a third party can
   check it without trusting the operator's API. The check is a re-execution
   by a verifier, not a signature on a claim.
2. It does not require every validator to own a 70B model. Only the selected
   executor and the verifier panel need the weights; the rest of the network
   validates the claim transaction, bond, deadline, and signatures.
3. It reuses concepts already in the repository: `SubmitResult`,
   `ChallengeResult`, `game:` state, `cnode:` reputation, and task fees. The
   implementation still has to change the semantics (payment is escrowed, not
   immediate), but it is not a new subsystem.
4. It is honest about the weak point. With one operator and no independent
   verifier, optimistic verification is an audit trail, not a trustless
   guarantee. The ADR says so explicitly and keeps full re-execution as the
   default for models where it works.

Option (b) should be added when a customer requires a lower-latency receipt
and can accept hardware trust, or when a 70B task has no economically rational
independent challenger. Option (c) should be revisited only if a production
zkLLM system changes the cost curve by orders of magnitude.

For the current single-operator pilot, full re-execution remains the only
honest verification for models that fit in the integer runtime. Optimistic LLM
verification becomes meaningful only when an independent party can afford to
re-execute and a producer cannot censor its challenge. Until then, describe it
as an audit trail and keep the "verified inference" claim scoped to the
deterministic model set.

## Protocol

### Parameters

These are genesis/state parameters, not hardcoded constants:

| Parameter | Meaning | Initial value (proposal) |
|---|---|---|
| `verificationType` | Task verification mode | `optimistic-llm-v1` |
| `challengeWindowBlocks` | Window after claim inclusion | 100 blocks (~16 min at 10 s; ~100 s at 1 s) |
| `finalityDepth` | Blocks before a non-BFT claim is treated as settled | 12 |
| `verifierQuorumLow` | Verifiers for tasks below `highValueFee` | 1 |
| `verifierQuorumHigh` | Verifiers for tasks at/above `highValueFee` | 3 |
| `highValueFee` | Threshold for the larger panel | 10,000 fee units |
| `executorBondMultiple` | Bond as a multiple of `maxFee` | 20 |
| `challengerBondMultiple` | Challenger bond as a multiple of `maxFee` | 2 |
| `verifierFeeMultiple` | Verifier fee as a multiple of one re-execution cost estimate | 2 |
| `maxClaimBytes` | Receipt size cap | 64 KB |

The chain stores these under `meta:llmParams` and includes the value in the
state root (see "State keys and state root"). Changing them is a transaction
or a genesis change with an activation height, not a node restart setting.

### Task lifecycle

The existing lifecycle is `pending -> assigned -> completed | challenged |
failed` (`crates/state/src/models.rs:38-39`). The LLM path adds explicit claim
and verification states. It must not mark a claim as `completed`, because
`compute_state_root` currently treats `completed` as a committed output
(`crates/exec/src/engine.rs:1003-1009`).

1. **Submit.** The requester submits `submitInference` with
   `verificationType: "optimistic-llm-v1"`, a model id, an input commitment,
   and canonical sampling parameters. The task is `pending`. The requester's
   `maxFee` is locked or escrowed; it is not paid to an executor yet.
2. **Assign.** At the next block, `match_pending_tasks` assigns the task to a
   deterministic eligible executor (see below). The task becomes `assigned`.
   If no eligible executor exists, it stays `pending`; a deadline timeout
   refunds the requester.
3. **Execute and claim.** The executor runs the model off-chain, builds a
   receipt (below), and submits `submitLlmResult` with the receipt, output
   commitment, and executor bond. The transaction is included in a block; the
   task becomes `claimed`; the challenge window opens at
   `claimHeight + challengeWindowBlocks`.
4. **Challenge.** Before the deadline, any address may submit
   `challengeLlmResult` with a challenger bond and an optional alternative
   output hash. The task becomes `challenged`; a `game:` record is created.
   A challenge without an alternative hash is allowed; the verifier performs
   the comparison.
5. **Verify.** A deterministic verifier panel is selected from the eligible
   verifier set, excluding the executor and challenger. Each selected verifier
   re-executes the task and submits `submitVerifierVote` with the output hash
   and a signature. The majority output hash wins. If the executor's claim
   matches the majority, the challenger loses its bond; otherwise the executor
   loses its bond and is slashed.
6. **Finalize.** If the window closes with no challenge, anyone may submit
   `finalizeLlmResult`; if a challenge was resolved, the resolution
   transaction finalizes the task. The task becomes `verified` or `slashed`.
   Only then is `receipt:<task_id>` written and the output hash included in
   the verified-results commitment.
7. **Timeout.** If the executor does not claim by its deadline, anyone may
   submit `timeoutLlmTask`. The task is reassigned or refunded, and a
   configurable part of the executor's assignment bond is slashed. If a
   verifier does not vote by its deadline, the next verifier in the sorted
   panel is used; if no panel can reach quorum, the challenge is resolved by
   the fallback rule in "Failure and timeout rules".

### Executor selection

Selection must be deterministic from finalized state, not from a node's local
configuration. Proposed:

```text
eligible = sorted active compute nodes (`cnode:`) whose `supported_models`
           contains the target model and whose `available_capacity` > 0
if eligible is empty:
    eligible = sorted active validators (fallback for a small network)
seed = sha256("verinoo/llm-executor/v1" || chain_id || task_id ||
              assignment_parent_hash || epoch_le_u64)
executor = eligible[seed_as_u64 % eligible.len()]
```

`assignment_parent_hash` is the hash of the finalized block before the
assignment block. The executor cannot grind the seed by choosing the task id
(the task id commits the transaction contents) or the parent hash after the
fact. A producer could still bias assignment by censoring transactions; that
is a liveness/censorship risk, not a safety one, and it is the same risk the
current mempool already has. With a single producer, do not claim
unbiased assignment.

A node advertises executor capability at registration (`RegisterNode`) and in
its model set. It must not be selected if it cannot produce a receipt for the
exact `modelHash` and `tokenizerHash`. The current `ComputeNode` record
(`crates/state/src/models.rs:92`) has `supported_models`, `available_capacity`,
and `reputation`; add `executor_key` or reuse `public_key`, plus
`max_model_bytes` and `backend`.

### Receipt fields

The receipt is canonical JSON, signed by the executor. The transaction that
carries it is also signed, but a standalone receipt signature lets a third
party verify the claim without reconstructing the transaction envelope.

```json
{
  "v": 1,
  "kind": "llm-result-claim",
  "chainId": "verinoo-test-1",
  "taskId": "<32 hex>",
  "modelId": "llm-7b-v1",
  "modelHash": "<sha256 of weights manifest>",
  "tokenizerHash": "<sha256 of vocab/merges or tokenizer.json>",
  "runtimeHash": "<sha256 of runtime binary + kernel version>",
  "backend": "cpu-q16.16 | cuda-deterministic | tee-sev-snp | tee-tdx | tee-nvidia-cc",
  "inputCommitment": "<64 hex>",
  "promptTokenIdsHash": "<sha256 of canonical prompt token ids>",
  "sampling": {
    "temperatureMilli": 700,
    "topK": 40,
    "topPMilli": 950,
    "maxNewTokens": 256,
    "repetitionPenaltyMilli": 1000,
    "clientSeed": "<64 hex or empty>",
    "paramsHash": "<sha256 of the canonical sampling object>"
  },
  "seed": "<64 hex, derived; see below>",
  "tokenIds": [11, 612, "..."],
  "tokenIdsRoot": "<sha256 of canonical token id list>",
  "outputHash": "<sha256 of canonical output object>",
  "outputTextHash": "<sha256 of decoded text, optional>",
  "executor": { "address": "ai_...", "publicKey": "<hex>" },
  "claimedAtHeight": 42,
  "challengeDeadlineHeight": 142,
  "executorBond": 2000,
  "attestation": null,
  "signature": "<hex over canonical claim payload>"
}
```

`outputHash` is authoritative. It is
`sha256(canonical_json({tokenIds, tokenizerHash, modelHash}))`. `outputTextHash`
is optional and is not used for consensus, because UTF-8 decoding, special
tokens, and normalization can differ. A verifier compares token ids first and
text second. For long outputs, `tokenIds` may be replaced by a Merkle root plus
a chunked availability proof; v1 keeps the full list and caps it at
`maxClaimBytes`.

The signed payload is the canonical JSON of the receipt with `signature`
removed, with domain `verinoo/llm-claim/v1`. The executor's public key must
derive to its address using `public_key_to_address`
(`crates/exec/src/engine.rs:23`).

### Sampling commitment and determinism

The current decoder is greedy only (`docs/decoder.md:161-166`). Sampling must
be integer and reproducible before any claim is meaningful.

**Sampling parameters.** Encode all parameters as integers in canonical JSON:

- `temperatureMilli` in `1..=2000`, where 1000 is 1.0;
- `topK` in `0..=vocab_size`, where 0 disables top-k;
- `topPMilli` in `1..=1000`, where 1000 keeps all tokens;
- `maxNewTokens` in `1..=max_context - prompt_len`;
- `repetitionPenaltyMilli` in `1000..=2000`, where 1000 disables it;
- `clientSeed` is optional 32 bytes; empty means the requester does not choose
  a seed.

`paramsHash = sha256(canonical_json(sampling_object))`. The object is included
in the task transaction, so the executor cannot change it after the fact.

**Task-derived seed.** The executor does not choose the seed. Both the
executor and a challenger derive:

```text
seed = sha256("verinoo/llm-sampling/v1" || chain_id || task_id ||
              inputCommitment || paramsHash || modelHash || tokenizerHash)
```

`task_id` already commits the transaction contents (`task_id_for`,
`crates/exec/src/engine.rs:1259`), and `inputCommitment` binds the input. If
`clientSeed` is non-empty, append it as the last field; it is then part of the
committed parameters. A changed seed changes the token sequence, so it is not
a private executor choice.

**Per-step RNG.** Use a counter-based stream, not a stateful PRNG with
platform-dependent behavior:

```text
u_t = first 8 bytes (little-endian) of
      sha256(seed || step_le_u32 || 0x00)
```

Reject modulo bias: `threshold = u64::MAX - (u64::MAX % total_mass)`; draw
until `u < threshold`, then `r = u % total_mass`.

**Integer sampling.**

1. Run the final layer to Q16.16 logits.
2. Subtract the maximum logit (integer) for numerical range.
3. Temperature: `scaled_i = round(logit_i * 1000 / temperatureMilli)` with
   saturating i64 arithmetic. Define round-half-to-even.
4. Exponentiation: `p_i = exp_q16(scaled_i)` using a fixed 4096-entry table
   over `[-16, 0]` with linear interpolation and a documented rounding rule.
   Values below -16 are zero. This is approximate, but it is the same
   approximation on every node; do not use `f32`/`f64` `exp`.
5. Repetition penalty: for token ids already in the context, multiply positive
   logits by `1000 / penaltyMilli` or divide negative logits, in integer
   arithmetic with a documented rule.
6. Top-k: stable sort by `(p_i descending, token_id ascending)`, keep the
   first `k` entries.
7. Top-p: sort the same way, accumulate `p_i` in `u64`; keep the shortest
   prefix whose cumulative mass satisfies `cum * 1000 >= topPMilli * total`;
   always keep at least one token.
8. Renormalize by using `total_mass = sum(kept p_i)`; sample
   `r = u_t % total_mass`, walk the cumulative list, and return the token at
   the first cumulative value greater than `r`.

The exact `exp_q16` table, rounding, sort tie-break, and rejection-sampling
rule are part of the protocol version. A challenger that uses a different
table is running a different protocol and must be rejected, not argued with.

**Tokenizer.** The tokenizer hash covers `vocab.json`, `merges.txt`, the
special-token map, and the pre-tokenization regex version. BPE merges are
applied by rank; ties are broken by rank order. The prompt token ids are
computed with the same tokenizer and included as `promptTokenIdsHash`.
Decoding for `outputTextHash` reverses the byte-to-Unicode table, skips
special tokens, and uses UTF-8 lossy replacement; the token ids remain the
authoritative output.

**Model and runtime identity.** `modelHash` is a Merkle root over the sorted
shard names and per-shard SHA-256 values plus the config and tokenizer hashes.
`runtimeHash` covers the deterministic runtime version and the arithmetic
kernels. If GPU kernels are used, the receipt also carries `backend`,
`gpuArch`, `driver`, and `deterministicKernels: true`; a verifier must use the
same backend or a documented reference CPU path. This is the hardest
determinism constraint: many GPU matmul kernels are not bitwise reproducible
across batch sizes or hardware. If exact reproduction cannot be guaranteed,
the optimistic path is not valid for that backend and the task must use a TEE
attestation or a deterministic CPU implementation.

**Challenger reproduction.** A challenger:

1. checks the receipt signature, model/tokenizer/runtime hashes, and sampling
   commitment against the on-chain task;
2. loads the exact model, tokenizer, and runtime version;
3. recomputes the prompt token ids and `promptTokenIdsHash`;
4. derives `seed` from the committed fields;
5. runs prefill and per-token decode with the same KV-cache semantics and the
   integer sampler above;
6. compares each token id at each step, not just the final text;
7. if any token differs, records the first divergent step, both token
   sequences, and the per-step hashes as evidence; if all match, the challenge
   fails and the challenger's bond is forfeited.

### Challenge and verification

- A challenge is a transaction, not a new gossip protocol. It carries
  `taskId`, `challenger`, `challengerBond`, an optional
  `challengerOutputHash`, and a reason code (`invalid-token-sequence`,
  `wrong-seed`, `wrong-model`, `wrong-output`, `withheld-output`).
- The verifier panel is selected deterministically:
  `seed = sha256("verinoo/llm-verifier/v1" || taskId || challengeBlockHash)`;
  sort eligible verifiers by address, exclude the executor and challenger, and
  take the first `K`. If fewer than `K` remain, use all remaining; if none
  remain, the challenge is invalid and the challenger's bond is returned (the
  task remains claimed and can be challenged again).
- Each verifier re-executes independently and submits
  `submitVerifierVote {gameId, outputHash, tokenIdsRoot, signature}`. Votes
  are transactions, so they are ordered and finalized like any other state
  change.
- Resolution: majority of the `K` votes. With `K=1`, the single verifier's
  result decides; the challenger's bond must be large enough to make a
  frivolous challenge unprofitable, and the verifier must be independent of
  the executor. With `K=3`, a single bribed verifier cannot decide.
- If the verifier's output matches the executor's claim, the executor wins;
  the challenger loses `challengerBond`. If it differs, the executor is
  slashed and the challenger is rewarded. The verifier is paid from the losing
  bond.
- The verifier's re-execution is the expensive step. Do not use the existing
  layer-bisection `Bisect`/`ProveStep` game for LLM tasks: committing to every
  intermediate activation is too large, and `get_layer_spec` is not a real
  model description. Use full re-execution by the verifier.

### Bonds, slashing, and payouts

Bond sizes must cover the cost of verification and make fraud unprofitable.
They are economic parameters, not consensus constants; the chain only enforces
the arithmetic.

- `executorBond >= max(executorBondMultiple * maxFee, cost_estimate(K))`.
  `cost_estimate(K)` is an operator-set oracle for K re-executions. It is not
  trustless; document that.
- `challengerBond >= max(challengerBondMultiple * maxFee, cost_estimate(1))`.
- `verifierFee = verifierFeeMultiple * cost_estimate(1)`, paid from the losing
  bond.
- Example in fee units: `maxFee = 1,000`, `cost_estimate(1) = 500`,
  `K = 3`. Then `executorBond >= max(20,000, 1,500) = 20,000`,
  `challengerBond >= max(2,000, 500) = 2,000`, and
  `verifierFee = 1,000`. The absolute numbers are placeholders; the important
  property is that a losing bond covers the verifier work and still leaves a
  penalty.
- If the executor is honest and the challenge fails: the executor receives the
  escrowed `maxFee`, the challenger's bond is split between the executor
  (50%), the verifiers (30%), and the burn address (20%).
- If the executor is fraudulent: the executor's bond is split between the
  challenger (50%), the verifiers (30%), and the burn address (20%); the
  executor's escrowed fee is returned to the requester; the executor's
  `cnode:` reputation is reduced and repeated fraud removes it from the
  eligible set.
- If the executor times out before claiming: an assignment bond (a fraction of
  `executorBond`, e.g., 10%) is slashed to the requester, the rest is
  returned, and the task is reassigned. This prevents an executor from
  accepting work and stalling.
- If a verifier fails to vote by its deadline: a small verifier bond is
  slashed (or the verifier is removed from the eligible panel) and the next
  verifier is used. If the panel cannot reach quorum, the fallback rule is:
  the claim remains `claimed`, the challenge bond is returned, and the task is
  escalated to a manual/operator resolution. This is a liveness gap; do not
  claim automatic resolution in that case.

The current `SubmitResult` pays the executor immediately
(`crates/exec/src/engine.rs:625-628`). The LLM path must escrow instead, and
only pay on finalization. The current `ChallengeResult` takes
`challengerBond = max_fee` (`crates/exec/src/engine.rs:645`); the LLM path uses
its own bond formula and its own game kind.

### Finality of a verified result

A verified result is final when the resolution transaction is in a finalized
block. With ADR 004 BFT, that means a commit certificate with a 2/3+ quorum;
the challenge window is measured in finalized heights, so a reorg cannot erase
a claim or a challenge. Without BFT, use `finalityDepth` (e.g., 12 blocks) as
an explicit, weaker assumption; a reorg deeper than that invalidates claims
and challenges in orphaned blocks, and the operator must reconcile them
manually. Do not call a depth-based result final.

Before finality, `/v1/verify` must return `verified: false` with a status such
as `claimed`, `challenged`, or `awaiting-finality`, even if the output hash
matches a recomputation. After finality, `receipt:<task_id>` is written and
`verified: true` is allowed.

### Failure and timeout rules

- **Withheld output.** The executor claims nothing by the deadline; timeout
  slashes the assignment bond and reassigns. A malicious executor cannot
  collect the fee and withhold the output.
- **Withheld claim, output leaked.** If the output is available but no valid
  claim exists, the task is still pending; the requester can cancel and
  resubmit. Do not treat an off-chain output as settled.
- **Invalid token sequence.** The challenger submits the first divergent step
  and both sequences; the verifier confirms by re-execution; the executor is
  slashed.
- **Wrong seed or sampling parameters.** The verifier recomputes the seed from
  the committed fields. A receipt whose seed does not derive from the task is
  invalid without re-execution.
- **Wrong model/tokenizer/runtime hash.** The claim is rejected at admission;
  no bond is at risk if the claim is structurally invalid, but a valid-signature
  claim with a mismatched hash is fraud.
- **Challenge after the window.** Rejected. The window is a consensus rule,
  not an API preference.
- **Challenge during a reorg.** Without BFT, a challenge included in an
  orphaned block is not valid; the task reverts to its prior state. With BFT,
  a finalized challenge cannot be orphaned.
- **Executor and verifier are the same operator.** Allowed by the protocol but
  reduces credibility. The API should expose verifier identities, and the
  documentation should not call a self-verified result independent.

## Integration points

### `ExecModel` and runtime

`ExecModel` (`crates/exec/src/engine.rs:1069`) is an enum of in-process
runtimes resolved from `VNN_MODEL_DIR`. It is used by block execution, so it
must not resolve a 7B/70B model into a synchronous `apply_block` call.

- Keep `ExecModel` for deterministic replay models (built-in classifier,
  graph runtime, transformer encoder, small decoder).
- Add a separate executor interface, e.g. `LlmExecutor`, with
  `model_identity()`, `generate(task, params) -> Receipt`, and
  `verify_receipt(receipt) -> Result<(), String>`. It is used by the executor
  host and verifier hosts, not by `apply_block`.
- `ExecModel::resolve` should return `None` for a model whose
  `verificationType` is `optimistic-llm-v1`, or the call site should skip
  those tasks before resolving. The current call at
  `crates/exec/src/engine.rs:1167` would otherwise try to execute the LLM on
  every validator.
- Add model metadata: `modelHash`, `tokenizerHash`, `runtimeHash`, `backend`,
  `maxContext`, `maxNewTokens`, `deterministic` (bool), and `verification`
  (`deterministic-replay` or `optimistic-llm-v1`). `/v1/models`
  (`crates/node/src/rpc.rs:494`) should expose it.

### `execute_deterministic_tasks`

`execute_deterministic_tasks` (`crates/exec/src/engine.rs:1162`) is the
consensus execution loop. It must filter on verification type:

```text
if task.verification_type != "deterministic-replay" { continue; }
```

The LLM path is driven by explicit transactions (`submitLlmResult`,
`challengeLlmResult`, `submitVerifierVote`, `finalizeLlmResult`), not by
`apply_block` running the model. `match_pending_tasks`
(`crates/exec/src/engine.rs:1030`) can still assign LLM tasks, but assignment
must not require the model to be present on every validator.

The state-root computation (`crates/exec/src/engine.rs:981`) currently covers
accounts, models, pending tasks, completed result hashes, and compute nodes.
It must add claims, games, verifier votes, and validator-set entries before
claim status is consensus state. Otherwise two nodes can disagree about
whether a claim was challenged without a state-root mismatch.

### `stamp_receipts`

`stamp_receipts` (`crates/exec/src/engine.rs:1225`) currently stamps every
task with `status == "completed"` and `executed_at == height`. For the LLM
path:

- Do not stamp a claim as a final receipt.
- Stamp `receipt:<task_id>` only when the task is `verified` and the
  resolution is final (BFT finalized, or `finalityDepth` reached).
- Include the challenge resolution, verifier identities, and finality
  evidence in the receipt, or store those under `game:<game_id>` and reference
  them by hash.
- Keep the existing deterministic receipt format for small models so existing
  `/v1/verify` behavior and tests do not change.

### `v1_verify` and API

`/v1/verify` (`crates/node/src/rpc.rs:616`) recomputes the output locally.
That is correct for deterministic models and impossible for a 70B model on
every API node.

- Branch on `task.verification_type`.
- For `deterministic-replay`: keep the existing recompute-and-compare
  behavior.
- For `optimistic-llm-v1`: return the claim, receipt, challenge/game status,
  verifier votes, and whether the result is final. Do not return
  `verified: true` merely because the node has a local model that produced
  the same output; that is an off-chain opinion, not the protocol result.
- Add an opt-in `?recompute=1` path for operators or verifiers who have the
  model; rate-limit it and require an API key. A public endpoint must not
  trigger a 70B generation on demand.
- `/v1/models` should say `verification: "optimistic-llm-v1"` and list the
  executor/verifier capabilities separately from the consensus model set.
- `/v1/tasks/:id` should expose `status`, `claim`, `challenge`, `receipt`, and
  `finalized` separately.

### Dispute operations

The existing `ChallengeResult` (`crates/exec/src/engine.rs:631`), `Bisect`
(`:698`), and `ProveStep` (`:765`) are a layer-bisection game built on
`get_layer_spec`. They are test-only for a real model and must not be presented
as LLM verification.

- Add a new transaction kind for the LLM game, or extend `ChallengeResult`
  with `gameKind: "llm-recompute"`. Do not overload the layer fields.
- Reuse the `game:` KV store (`crates/state/src/lib.rs:127`) for the game
  record, but define a typed JSON schema rather than an opaque `Value` for the
  LLM fields.
- Reuse the bond/account arithmetic, but do not reuse the immediate-payout
  behavior of `SubmitResult`.
- The `ProveStep` path's `verinoo_zk::execute_layer` (`crates/exec/src/engine.rs:815`)
  is not an LLM verifier. Do not call it for a transformer forward pass.

### Network

The current topics are blocks and transactions
(`crates/network/src/lib.rs:23-24`), chain-prefixed in `NetworkNode::new`
(`:217-220`). Claims and challenges are transactions, so the default path needs
no new topic: they ride `TXS_TOPIC` and are ordered into blocks like any other
tx.

Add topics only if the payload does not fit a transaction or needs priority:

- `RESULTS_TOPIC = "verinoo/results/1"` for a claim/receipt broadcast before
  block inclusion;
- `EVIDENCE_TOPIC = "verinoo/evidence/1"` for verifier votes or divergent-token
  evidence.

Follow the ADR 004 pattern: chain-prefixed names, `NetworkEvent` variants, a
`Command::Publish*`, and a topic-suffix match in the swarm loop
(`crates/network/src/lib.rs:316-330`). Do not gossip full token sequences for
a 70B task if the receipt can be fetched by hash; gossip the claim hash and
serve the body over RPC. The `flood_publish(true)` setting (`:207`) is
acceptable for a small set but makes claim traffic O(N^2); size it or move
large receipts to request/response.

### State keys and state root

Add:

| Key | Value |
|---|---|
| `task:<task_id>` | existing task record; status includes `claimed`, `challenged`, `verified`, `slashed` |
| `claim:<task_id>` | canonical receipt JSON plus executor bond |
| `game:<game_id>` | typed LLM challenge game |
| `vote:<game_id>:<verifier>` | verifier vote + signature |
| `receipt:<task_id>` | final receipt, written only on verification |
| `meta:llmParams` | protocol parameters and their activation height |
| `cnode:<node_id>` | executor capability, bond, reputation, slashing history |
| `val:<address>` | verifier eligibility and bond, if verifiers are validators |

Extend `compute_state_root` to include, sorted:

- `claim:<task_id>` hashes, not the full output;
- `game:<game_id>` status and winner;
- verifier votes (or a Merkle root of them);
- `val:` entries (address, public key, stake, activation height, jailed);
- `meta:llmParams`.

Do not include `meta:head`, `meta:receipt` for unverified claims, or anything
that is not deterministic across nodes. If a claim is not in the state root,
it is not consensus state and a node can silently disagree about whether a
task was challenged.

## Throughput and economics

The following are estimates, not measurements. They are here to show the
shape, not to set prices.

Let `C_re` be the cost of one 256-token re-execution of the model on the
verifier's hardware, and `p` the probability that a claim is challenged.

| Model | Full re-execution (N=4) | Optimistic | TEE |
|---|---|---|---|
| 7B | 4 x C_re per task; 4 GPUs with ~16 GB each; every validator must serve | 1 x C_re + p x K x C_re; one executor GPU, K verifier GPUs on dispute | ~1.0-1.3 x C_re; one confidential GPU; attestation overhead |
| 70B | 4 x C_re; 4 multi-GPU hosts with ~160 GB each; not practical for a testnet | 1 x C_re + p x K x C_re; executor and verifiers need multi-GPU hosts | ~1.0-1.3 x C_re; confidential 70B serving is capacity-constrained |

Why full re-execution fails: `C_total = N x C_re`, and the latency is the
slowest validator. A 70B model at 5-15 tokens/s on a multi-GPU host needs
17-50 s for 256 tokens, before network and block overhead. The 10 s block time
cannot wait, and requiring every validator to own 70B weights turns a small
validator set into a GPU cluster. It also does not remove the determinism
problem; it multiplies it by N.

Optimistic cost is `C_re x (1 + p x K)`. With `p = 1%` and `K = 3`, that is
1.03 x `C_re`. The challenge cost is paid by the losing bond, not by every
task. The unclosed risk is that no independent party can afford a 70B
re-execution, so `p` is effectively zero because fraud is not detected. TEE
avoids that at the cost of hardware trust. ZK would be best on paper but is
not in the same cost regime.

Liveness and fraud assumptions:

- Optimistic assumes at least one honest verifier with the model, the runtime,
  and enough time within the challenge window, and that the challenge
  transaction can be included by a producer that is not colluding with the
  executor. With one operator and one producer, these assumptions are not
  met; the design is an audit trail, not a trustless guarantee.
- TEE assumes the hardware vendor's attestation roots, the measurement
  allowlist, and no side-channel break. It does not assume an independent
  challenger.
- Full re-execution assumes every validator can run the model; that is the
  current model and the reason this ADR exists.
- None of these assumes the model is correct. They verify that the committed
  model produced the committed output, not that the output is good.

## Test plan

Build an in-process harness with a small deterministic decoder (the
`tiny-stories-1m` test model, not a 7B model) and a mock slow executor. Do not
put a real 7B/70B run in CI.

Required tests:

1. **Honest executor, no challenge.** Submit, assign, claim, advance past the
   window, finalize. Assert the task is `verified`, the receipt is written, the
   output hash is in the verified-results root, and the executor is paid the
   escrowed fee exactly once.
2. **Honest executor, invalid challenge.** A challenger posts a wrong
   alternative hash; the verifier reproduces the executor's output; the
   challenger loses the bond; the executor is paid; the verifier is paid.
3. **Fraudulent executor, valid challenge.** The executor flips one token id
   in the claim; the verifier reproduces the true sequence; the executor's
   bond is slashed, the requester is refunded, and the verifier is paid.
4. **Withheld output.** The executor claims nothing by the deadline; timeout
   slashes the assignment bond, refunds or reassigns the task, and the
   executor cannot collect.
5. **Bad sampling seed.** The claim's `seed` does not derive from the task
   fields; the claim is rejected without a full re-execution; the executor's
   bond is not rewarded.
6. **Invalid token sequence.** The claim's `tokenIdsRoot` does not match the
   canonical token list, or the first divergent step is recorded; the
   verifier's evidence is stored and the bond is slashed.
7. **Model/tokenizer/runtime mismatch.** The claim carries a different
   `modelHash`, `tokenizerHash`, or `runtimeHash`; admission rejects it;
   `/v1/verify` reports the mismatch.
8. **Bond and slashing accounting.** Run the four outcomes (honest/no
   challenge, honest/challenge, fraud, timeout) and assert every balance
   change, burn, verifier fee, and `total_slashed` is exact; assert no path
   pays the executor twice.
9. **Timeout and reassignment.** The executor accepts but does not claim; the
   task is reassigned to the next executor; the first executor's bond is
   partially slashed; the second can claim.
10. **Challenge-window boundaries.** A challenge at the last allowed height is
    accepted; one block later is rejected. With a reorg before finality, the
    challenge reverts with the block; with BFT finality, it cannot.
11. **Verifier panel.** Selection is deterministic from the task and challenge
    block; the executor and challenger are excluded; a missing verifier is
    replaced; a `K=3` panel with one dissenting vote still resolves by
    majority.
12. **State-root coverage.** Two stores with different claim or challenge
    status produce different state roots; a node that omits `claim:` or
    `game:` from the root is rejected by a peer.
13. **Reorg/challenge edge.** A claim in an orphaned block is not final; a
    challenge in an orphaned block is not counted; with `finalityDepth`
    exceeded, a deeper reorg is surfaced as an incident, not silently
    accepted.
14. **Determinism vectors.** Fixed prompt, model, tokenizer, sampling
    parameters, and seed produce the same token ids on two independent
    processes; changing `topK`, `topPMilli`, `temperatureMilli`, or the seed
    changes the sequence in a documented way; float and integer samplers are
    not mixed.

## Phased estimate

One engineer familiar with the codebase; calendar weeks including tests and
review, not external audit.

| Phase | Work | Estimate |
|---|---|---|
| 0 | Freeze the receipt and sampling spec; implement the integer sampler and test vectors; mock executor | 1 wk |
| 1 | Detached LLM execution path; task lifecycle (`claimed`/`verified`); claim transaction; state-root coverage for claims/games | 2-3 wk |
| 2 | Challenge game; verifier panel; re-execution harness; bonds, escrow, slashing; `/v1/verify` branch | 3-4 wk |
| 3 | Finality gating (BFT or `finalityDepth`), timeout/reassignment, reorg handling | 1-2 wk |
| 4 | 4-validator + independent verifier testnet, economics calibration, operator docs | 1-2 wk |
| **Total** | Optimistic path | **8-12 wk** |
| + | TEE attestation path | +6-10 wk |
| + | 7B/70B serving runtime (separate from verification; see `docs/pilot.md:154-160`) | +4-8 wk |
| + | ZK | not scheduled |

Dependencies: ADR 004 for finality; ADR 006 for validator removal and snapshot
distribution of large weights; a production LLM runtime with deterministic
kernels. The estimate does not include model licensing, GPU procurement, or a
security audit.

## Non-claims

- **Not implemented.** No optimistic LLM executor, verifier panel, bond,
  challenge window, TEE attestation, ZK proof, or integer sampler exists in
  this repository.
- **No production LLM support.** The largest model that runs in consensus is
  the 3.75M-parameter `tiny-stories-1m`; 7B/70B serving, long context,
  batching, streaming, and a production tokenizer are not implemented.
- **No sampling today.** The decoder is greedy (`docs/decoder.md:161-166`).
  This ADR specifies sampling; it does not ship it.
- **No trustless verification with one operator.** Optimistic verification
  requires an independent verifier and a producer that cannot censor
  challenges. The current deployment has neither.
- **No BFT finality.** ADR 004 is a design. Without it, the challenge window
  is a depth assumption, not finality.
- **No live dispute game for a real model.** The existing `ChallengeResult` /
  `Bisect` / `ProveStep` path is test-only, and `get_layer_spec` fabricates
  layer shapes from a model name (`docs/readiness.md:64-68`).
- **No TEE or ZK claim.** TEE attestation is a hardware trust claim and is not
  a correctness proof. ZK is not viable for 7B/70B in this time frame.
- **No slashing of live validators.** Bonds and slashing described here are a
  protocol design; the live path does not move funds automatically.
- **No token settlement.** Fees remain testnet balances; this ADR does not
  change settlement.
- **No claim that a receipt means the answer is good.** It means the committed
  model, tokenizer, runtime, sampling parameters, and seed produced the
  committed token sequence. Model quality and safety are separate.
