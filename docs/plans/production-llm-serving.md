# Production LLM Serving Plan

Status: implementation plan only. No GPU executor, batching, streaming endpoint, or policy crate exists in this repository. Date: 2026-11-16. Scope: `crates/node`, `crates/exec`, `crates/runtime`, `crates/state`, a new `crates/policy`, both SDKs, and deployment scripts. Design sources: ADR 004, 005, 006; this plan orders work and defines gates.

## Baseline

- Consensus re-executes every pending/assigned task on every validator: `execute_deterministic_tasks` (`crates/exec/src/engine.rs:1162`) runs from `apply_block` (`:1268`) and `produce_block` (`:1371`). `compute_state_root` includes completed result hashes (`:1003-1009`), so one divergent token changes the root.
- Decoder: Q16.16, seeded SplitMix64 sampler (`GptModel::generate_with_sampling`, `crates/runtime/src/gpt.rs:834`; `sample_token`, `:881`), sampling fixed in the model JSON (`Sampling`, `:146`), mmap'd `.vnnw` weights pinned by SHA-256 (`from_file`, `:549`).
- DistilGPT-2: 81.9M parameters, 1,024 context, 16 new tokens by default (`gpt.rs:99-100`). Dev Mac ~2.62 s for load + 48 tokens (~40 ms/token); the 1-vCPU VPS is slower. Local-only, not public.
- Receipts bind final token ids/text (`stamp_receipts`, `crates/exec/src/engine.rs:1225`; `receipt_for`, `:1254`). No consensus token streaming. `/ready` exposes `syncing`, peer lag, and validator-set consistency (`crates/node/src/rpc.rs:136`).

## Decision summary

1. Keep CPU Q16.16 as the only consensus executor until a GPU kernel passes byte-for-byte parity. Never put float GPU output into `compute_state_root`.
2. Ship API batching and status/final streaming first; both are non-consensus.
3. Treat GPU inference as an off-consensus optimistic executor per ADR 005: GPU preview, CPU verification of final token ids, receipt only after verification. No GPU implementation exists today.
4. Put input policy at the API boundary and output redaction on the read path; a filter must never silently mutate the consensus result.
5. Defer runtime batching and deterministic GPU kernels behind parity gates.

## GPU execution

### Options and decision criteria

| Option | What it buys | Main cost | Status |
|---|---|---|---|
| Stay CPU | No new trust/determinism risk | DistilGPT-2 misses a 1 s block on 1 vCPU | Default |
| Off-consensus GPU executor (ADR 005) | GPU speed, 1 execution per undisputed task | Claim/challenge lifecycle, bonds, verifier, model distribution | Recommended |
| CUDA worker inside consensus | No separate executor | Valid only if integer kernels are bitwise identical to `gpt.rs` | Research |
| Metal/wgpu worker | Local Apple Silicon dev/demos | No production determinism claim | Research |
| TEE GPU | Attested execution without a challenger | Hardware, side channels, vendor roots, revocation | Optional later |

Decide with: model/context/tokens; latency SLO (block-bound vs request-bound); whether an independent operator can run the model and verifier; GPU budget and utilization; attestation requirement; tokenizer support (only GPT-2/GPT-Neo, `docs/decoder.md`).

### Determinism strategy

The consensus boundary is `ExecModel::result_json` (`crates/exec/src/engine.rs:1115`); its SHA-256 is `result_hash`, which is in the state root. Therefore:

- Do not route GPU output through `ExecModel` for a consensus task. A float or non-identical kernel can flip one argmax and fork the chain.
- Three stages for large models:
  1. **GPU preview.** Load the same `.vnnw`, tokenize identically, derive the seed from committed task fields (ADR 005), generate token ids; label `preview`, `verified: false`.
  2. **CPU verification.** Re-run Q16.16 `generate_with_sampling` with the same model, prompt ids, sampling, and seed. Compare **per token id at each step**, not text. On match, the CPU output is canonical; otherwise discard the preview and record the first divergent step.
  3. **Consensus receipt.** Only verified CPU token ids become the claim and `receipt:<task_id>`. Never claim the GPU output was proven.
- A divergent preview is not partial success; it is discarded.
- A future integer CUDA/Metal kernel can become `ExecModel` only after `distilgpt2_binary` and cross-hardware bitwise parity pass. Until then "GPU consensus execution" is a non-claim.
- Per-task sampling is missing (`docs/decoder.md`: model-file fixed). ADR 005's `paramsHash` and task-derived seed must land before a claim is meaningful; use model-file sampling and say so until then.

### Hardware and benchmark requirements

Add `scripts/bench-llm.sh` and an ignored `crates/runtime/tests/llm_bench.rs`. Record model id, `weightsHash`, load/prefill/decode p50/p95, RSS/VRAM, context, batch, host/GPU/driver. Measure context 128/512/1024; batch 1/8/32; cold and warm load; 16/32/64 new tokens; on the 1-vCPU VPS, dev Mac, and each candidate GPU. The ~40 ms/token Mac number is not a VPS or GPU measurement.

Provisional gates: optimistic 16-token p95 < 750 ms on target GPU; CPU verifier p95 < 3 s for DistilGPT-2; no OOM at batch 8; warm load < 5 s. A consensus GPU path must fit the whole block, including generation and state root, inside the configured block time. Do not set that gate from a vendor benchmark.

### Security and attestation boundaries

- The GPU worker must not hold `VNN_KEY_SEED` or the validator key. Give it a separate executor key that signs claims only.
- A remote worker should not hold `VNN_INPUT_KEY`; that key decrypts every encrypted task. Keep the worker inside the validator boundary, or deliver only the specific plaintext to the worker out of band and document that it sees plaintext.
- Pin `weightsSha256`, `tokenizerHash`, `runtimeHash`, and `backend`; reject a worker whose model hash differs from the registered model.
- TEE is optional: a quote binds model, tokenizer, runtime, input commitment, sampling, output hash, and nonce. It proves attested code ran, not model correctness. Do not claim TEE before implementation and review.
- Preview events must be task-scoped; never expose another tenant's prompt or output.

## Batching

### API-level batch submission (implement first)

Add `POST /v1/infer/batch` beside `submit_tx` (`crates/node/src/rpc.rs:452`) and `v1_models` (`:573`). Request: `{"items":[{"model":"...","input":"...","maxFee":100}, ...]}` or `encryptedInput` per item; cap at 16. Each item fans out through the existing submit path as an independent `SubmitInference` transaction with its own `task_id` (`task_id_for`, `crates/exec/src/engine.rs:1259`) and receipt.

Response is HTTP 200 when at least one item is accepted: `{"accepted":2,"rejected":1,"results":[{"index":0,"taskId":"...","txId":"...","status":"accepted"},{"index":1,"status":"rejected","error":"model not allowed"}]}`. If every item is invalid, return 400.

- Partial failure is per item; do not roll back accepted siblings, and never create a batch-level task or receipt.
- Limits: 16 items, 4 KiB plaintext each, quota counted per item, total `maxFee` checked against the requester balance before any submit. Encrypted items are opaque: check structure, size, model allowlist, and commitment only.
- Fee/gas: each task pays its own `maxFee` and uses the same gas as a single submission. Batching is not a gas discount and must not bypass the block gas limit. Do not add a `SubmitInferenceBatch` tx kind unless gas is summed per item and the block limit still applies.
- Acceptance: 16 valid items -> 16 distinct tasks and receipts; one invalid item leaves siblings accepted; duplicate submission does not duplicate task ids; quota exhaustion is reported at the item index; every receipt verifies.

### Consensus/runtime batching (optional, parity-gated)

`execute_deterministic_tasks` loops tasks sorted by `task_id` (`crates/exec/src/engine.rs:1163-1165`). A batched decoder needs a padded KV cache and per-row attention masking in `crates/runtime/src/gpt.rs` (`KVCache`, `:442-483`).

- Batch only the same model, tokenizer, and sampling parameters; deterministic order from finalized state, never arrival order.
- Preserve per-task receipts and result hashes; one row failing must not change another row.
- Arithmetic, accumulation order, and masking must be row-independent. For fixed vectors, batched token ids must equal sequential token ids byte-for-byte and the state root must match a sequential re-execution.
- Do not enable for consensus until that passes. It is not needed on the 1-vCPU VPS, where the block budget is the bottleneck.

## Streaming

Consensus commits a whole result; there is no consensus token stream. Split streaming into two honest layers.

### Status and final stream (no executor required)

Add SSE in `crates/node/src/rpc.rs` (axum SSE; `futures` is already a dependency): `GET /v1/tasks/:task_id/events` returns `text/event-stream`.

- `status`: `{"taskId":"...","status":"pending|assigned|completed","height":null,"syncing":false}`.
- `final`: `{"taskId":"...","receipt":{...},"anchored":true,"finalized":false,"finality":"single-producer"}`. Emit only after `receipt:<task_id>` exists (`crates/exec/src/engine.rs:1250`) and the task is `completed`. Under ADR 004 add `finalized: true` and the certificate height/hash; without BFT, `finalized` stays false. Never call a depth-based result final.
- `error`: `{"taskId":"...","code":"...","message":"..."}`; `done` terminates.
- Resume with `Last-Event-ID`; heartbeat every 15 s; cap streams per IP/API key; close on completion.

### Preview stream (only with an off-consensus executor)

- Emit `preview` after generation but before CPU verification: `{"taskId":"...","text":"...","tokens":[...],"verified":false,"source":"executor-preview"}`.
- A preview never carries `receipt`, `blockHash`, or `finalized: true`. If CPU verification diverges, emit `preview_invalidated` and do not emit `final` from the preview.
- CPU-only deployments emit `status` then `final`; that is the honest no-token-streaming behavior.

### WebSocket and SDKs

- WebSocket is optional and later; it needs the axum `ws` feature in `crates/node/Cargo.toml` (currently `axum = "0.7"` defaults). Use `/v1/ws` with `{"subscribe":"task","taskId":"..."}` only if bidirectional control is needed; prefer SSE through nginx.
- TypeScript: add `Client.streamTask(id, {signal})` as an async iterator; keep `waitForTask` (`sdk/typescript/src/index.js:445`) as the fallback.
- Python: add `Client.stream_task(id)` as a generator; keep `wait_for_task` (`sdk/python/verinoo/client.py:479`).
- Expose `anchor`/`finality` so clients distinguish "receipt exists" from "BFT-finalized"; do not silently change `infer()`'s return shape.

## Safety

### Input policy at the API boundary

- Plaintext: enforce max length, required fields, model allowlist, and an optional operator denylist/PII policy before creating a transaction. Reject at `POST /v1/infer` and `POST /tx` with a stable error code.
- Encrypted: treat ciphertext as opaque. Check structure, size, `inputCommitment`, model allowlist, and quota — not content. Do not decrypt at the API and do not claim screening. `VNN_REQUIRE_ENCRYPTED_INPUT` (`crates/node/src/rpc.rs:584`) already supports encrypted-only deployments.
- Model allowlist: add `VNN_ALLOWED_MODELS` or a registry `production` flag and check it in the API path. `ExecModel::resolve` (`crates/exec/src/engine.rs:1079`) returns `None` for unknown models, leaving tasks pending; reject up front instead.
- No overclaim: a denylist is not a guarantee, an encrypted input is not screened, and a model can still produce harmful text.

### Output filtering and redaction

1. **Read-path redaction (implement now).** Keep `task.result_output`, `task.result_hash`, and `receipt:<task_id>` byte-identical to consensus. `GET /v1/tasks/:task_id` (`crates/node/src/rpc.rs:657`) returns a `display`/`redactedText` view plus `outputHash` and `redacted: true`. `/v1/verify` (`:695`) recomputes the canonical output, compares the hash, then returns the hash or redacted view. A third party can still verify the full result by recomputation; the API does not rewrite state.
2. **Consensus-level block (later, explicit hard fork).** If a task must never commit unsafe text, define a deterministic versioned policy (`policyId`, `policyHash`) that every validator runs before `stamp_receipts`. The committed result becomes `status: "blocked"` with no text, and the receipt binds that decision plus the policy hash. This is not a silent filter.

### Policy crate

New `crates/policy` with pure functions: `check_input(&InputView) -> Decision`, `classify_output(&str) -> Decision`, `redact(&str) -> RedactedView`; version with `PolicyVersion { id, sha256 }`. No network, downloads, time, or randomness. Log the version at the API edge; a consensus-level policy commits its hash. Integrate at `submit_tx`/`v1_infer` first; never call an external moderation API from `execute_deterministic_tasks`. Tests: denied plaintext creates no tx; structurally valid encrypted input is accepted without content inspection; allowlist enforced; redaction preserves the canonical output hash; version changes are visible.

## Milestones

One engineer familiar with the codebase; includes tests/review, excludes GPU procurement and audit.

### Wave 1 — non-consensus serving (3-4 weeks)

| Milestone | Files/functions | Tests | Gate | Est. |
|---|---|---|---|---|
| M1 Batch API | `crates/node/src/rpc.rs` (`infer_batch`, route in `app`, `:43`), both SDKs | Accept/partial/limit/quota; per-item receipts | No state or receipt change; suite green | 3-5 d |
| M2 SSE status/final | `crates/node/src/rpc.rs` (`task_events`), `crates/node/Cargo.toml` | Final only after receipt; resume; heartbeat; auth scoping | `/v1/tasks`, `/v1/verify` unchanged | 4-6 d |
| M3 Safety boundary | new `crates/policy/src/lib.rs`; `rpc.rs` (`submit_tx`); `main.rs` config | Plaintext reject; opaque encrypted; allowlist; redaction | No consensus result mutation | 5-7 d |
| M4 Benchmark harness | `scripts/bench-llm.sh`, `crates/runtime/tests/llm_bench.rs` | VPS/Mac/GPU vectors; load/prefill/decode | Baseline before GPU spend | 3-4 d |

### Wave 2 — off-consensus GPU path (6-10 weeks)

| Milestone | Files/functions | Tests | Gate | Est. |
|---|---|---|---|---|
| M5 Executor/claim lifecycle (ADR 005 subset) | new `crates/executor` or `node/src/executor.rs`; `claim:`/`receipt:` in `crates/state/src/lib.rs`; claim admission in `engine.rs` | Claim signature, model hash, sampling commitment, window | Claims never stamped final before verification | 2-4 wk |
| M6 GPU worker prototype | new `workers/cuda/` outside the consensus workspace; executor client | Load `.vnnw`, tokenize, generate, report hashes; CPU parity | Worker cannot sign blocks or hold validator keys | 3-6 wk + hardware |
| M7 Preview + CPU verification | `rpc.rs` preview SSE; executor verifier | Per-token compare; divergence evidence; invalidation | Only CPU-verified output becomes a claim | 2-3 wk |
| M8 Decoder batch parity (optional) | `gpt.rs` (`KVCache`, `generate_with_sampling`), `engine.rs` (`execute_deterministic_tasks`) | Batched == sequential token ids and root | Not enabled unless byte-identical | 2-4 wk |

### Wave 3 — production claims (multi-week)

| Milestone | Files/functions | Tests | Gate | Est. |
|---|---|---|---|---|
| M9 Production GPU service | worker deployment, model distribution, metrics | Soak, failure, rotation, no OOM | SLO/cost met | 4-8 wk |
| M10 Deterministic GPU kernels | `gpt.rs` parity path; CUDA/Metal kernels | Cross-hardware bitwise parity; `distilgpt2_binary` | Independent review | 8+ wk research |
| M11 Consensus policy activation | `crates/policy`, `engine.rs`, task/header versioning | Same policy hash; state-root equality | Coordinated hard fork | 3-4 wk |
| M12 TEE attestation | worker + claim `attestation` | Quote, revocation, freshness | Vendor roots + allowlist | +6-10 wk |

Dependencies: ADR 005 (claim/sampling), ADR 004 (`finalized` labels), ADR 006 (model distribution), API keys/quotas for batch abuse, GPU hardware, model license.

## Acceptance tests

1. 16 valid batch items create 16 task ids and 16 independently verifiable receipts; one invalid item does not reject the others.
2. Over-cap or over-quota batches return per-item errors without double-debit.
3. SSE `final` is byte-equal to `GET /v1/tasks/:id`'s receipt and is emitted only after the receipt key exists.
4. A preview with a flipped token is labelled unverified, emits `preview_invalidated` when CPU verification disagrees, and never creates a receipt.
5. CPU verification compares token ids per step, not decoded text.
6. GPU preview never appears as a canonical result in `/v1/tasks` or `/v1/verify`.
7. Denied plaintext is rejected before any transaction is created.
8. Encrypted input is checked only for structure/size/model/quota.
9. Redaction changes only the read view; stored output, hash, and receipt are unchanged.
10. Batched decoder execution (if enabled) matches sequential token ids and state root for fixed vectors.

## Risks and non-claims

- No GPU implementation exists: no CUDA/Metal worker, GPU executor, or GPU benchmark. A preview is not verification; CPU re-execution is.
- Optimistic verification with one operator is an audit trail, not a trustless guarantee (ADR 005).
- API batching is a fan-out, not a throughput guarantee; runtime batching is parity-gated and not scheduled.
- Consensus has no token streaming. No denylist or redaction view makes a model safe; encrypted inputs are opaque to the API; no ML moderation runs in consensus.
- Without ADR 004, `finalized` is always false; a receipt is anchored to a single-producer block. DistilGPT-2 is not served on the 1-vCPU public node.
