# Go-to-market readiness

Assessment date: 2026-11-15. Scope: the Rust node in this repository.

## Verdict

**Ready to sell as a paid pilot for the reference model. Not ready as a
general inference service.**

A complete vertical slice now exists and was verified live: tasks are executed
by consensus against a real trained model, receipts are anchored to signed
blocks, an independent node verifies them by recomputation, and usage is
metered for invoicing. What is missing for a general offer is a runtime for
models the customer actually cares about, and Byzantine finality. Both are
claims a technical buyer will test in the first diligence call.

## What works today (verified)

| Area | State | Evidence |
|---|---|---|
| Workspace build | green | `cargo build --workspace --locked` |
| Test suite | green | `cargo test --workspace --locked` (124 tests) |
| Node boot + block production | works | block height advances every 10 s |
| Ed25519 tx/block signing | works | enforced in `validate_block` |
| Canonical tx-id binding | works | conformance vectors |
| Gas market / base fee | works | EIP-1559-style adjustment, tested |
| P2P block + tx gossip | works | 2-node testnet converges, blocks every 10 s |
| RPC read surface | works | `/status`, `/ready`, `/metrics`, `/blocks`, `/account`, `/validators` |
| Wallet CLI | works | generate / balance / send |
| Validator set | works | genesis-seeded, round-robin leader schedule |
| Inference execution | works | consensus-level, `exec/tests/inference.rs` |
| Model | works (small) | `intent-classifier-v1`, 8 196 params, 98.4% train / 100% held-out |
| Operator model set | works | `models/acme-support-v1.graph.json`, executed on 2 nodes |
| ONNX classifier import | works | `verinoo-onnx`, parity vs f32, 10 tests |
| GGUF classifier import | works (experimental) | `verinoo-gguf`, Q4/Q5/Q8 + Q2_K–Q6_K dequant, 15 tests |
| Input encryption | works (validator-confidential) | X25519 + HKDF + ChaCha20-Poly1305; task/block redaction; `docs/privacy.md` |
| SDKs | works | `sdk/typescript` (8 tests), `sdk/python` (8 tests), live-tested against an encrypted node |
| Block-time override | works | `VNN_BLOCK_TIME_MS=1000` gives ~1–2 s task latency; network-wide setting |
| Attention kernel | prototype | Q16.16 single/multi-head, 6 tests; not wired into model loading (`docs/transformer.md`) |
| BFT finality | design only | `docs/adr/004-bft-finality.md`; no code |
| Result commitment | works | state root covers every executed output |
| Receipts + verification | works | `/v1/verify` recompute, live 2-node run |
| Metering | works | `/v1/usage`, CSV export |
| Dispute ops | implemented | `exec/tests/r6_disputes.rs` (in-memory) |
| Transformer encoder | works (experimental) | `intent-transformer-v1`, 5 508 params, Q16.16 attention, 80/80 label parity |
| GPT decoder | works (demonstration) | `tiny-gpt-v1`, 110 400 params, char-BPE, KV cache, greedy, token parity on 8 prompts |
| Real pretrained decoder | works (experimental) | `tiny-stories-1m`, 3.75M-param GPT-Neo, GPT-2 BPE, 32-token parity on 8 prompts |
| Binary decoder weights | works | `.vnnw` header + SHA-256 payload, tamper test; DistilGPT-2 328 MB |
| Deterministic sampling | works | temperature/top-k/top-p + SplitMix64 seed; cold sampling equals greedy |
| Licensed large decoder import | works (local) | Apache-2.0 DistilGPT-2, 81.9M params, 1,024 context, 16-token parity on 3 prompts; not public |
| ZK op vectors | works | Q16.16 conformance vs WASM runtime |

## What blocks a paid launch

1. **No production LLM serving.** A real pretrained 3.75M-parameter GPT-Neo
   decoder (`tiny-stories-1m`) runs in consensus with the GPT-2 byte-level BPE
   tokenizer, causal attention, a KV cache and greedy decoding. The runtime now
   also has deterministic temperature/top-k/top-p sampling and a SHA-256-bound
   binary `.vnnw` weight format, and a licensed 81.9M-parameter DistilGPT-2 with
   1,024-token context is imported and matches its float reference locally. What
   is missing is serving: GPU execution, batching, streaming, safety filters, a
   request queue, arbitrary-checkpoint import (LLaMA/SentencePiece) and 7B+
   models. On the 1-CPU VPS DistilGPT-2 is slower than a 1 s consensus block, so
   the public node still serves the small demo. Dense classifiers from ONNX/GGUF
   continue to work. This is the main product gap.
2. **No finality.** `RoundRobin` selects producers; there are no prevote /
   precommit rounds, so a block is only as final as the single producer's
   signature. No equivocation slashing in the live path.
3. **The dispute game is not live over RPC for a real model.** With consensus
   re-execution, verification is recomputation and there is nothing to dispute
   in the deterministic path. The game that exists targets the manual-claim
   path and `exec::spec::get_layer_spec` still fabricates layer shapes from a
   model *name* (ADR 003), so exposing it now would be misleading. Test-only.
4. **No token or on-chain fiat settlement.** Fees use a testnet balance and a
   faucet. Usage is metered and exportable (`/v1/usage/:addr/csv`); invoicing is
   manual. A token is a legal/strategy decision, not an engineering task.
5. **Operational HA is manual.** Backups, restore, API-key rotation, a catch-up
   replica and an incident runbook now exist ([`docs/operations.md`](operations.md)),
   but there is no automatic failover, no split-brain protection and no way to
   remove a validator from the active set without a coordinated re-genesis.
   `/ready` exposes lag, sync activity and validator-set consistency for a load
   balancer; it does not fence a host or promote a replica.
6. **No state snapshot transfer.** Catch-up replays blocks from the caller's
   height; a node far behind replays from genesis, which is O(chain length).
   There is no snapshot download yet.

## Prepared in this pass

- Genesis validator set is configurable (`VNN_VALIDATORS`) and seeded into
  state, so `/validators` is accurate and multi-node testnets can accept peer
  blocks (previously the allowed set was always empty and every peer block was
  rejected with "empty validator set").
- Validators produce only in their scheduled slot, preventing same-height forks
  on a multi-validator network.
- P2P now actually works between nodes. Three separate faults were fixed:
  mDNS discovery was never dialed, a dial that raced a peer's startup was
  never retried, and every connection was closed with `KeepAliveTimeout` ~2 ms
  after opening because gossipsub only requests keep-alive once a peer is in a
  mesh — a keep-alive behaviour now holds links open.
- Validators wait one block time before their first production and refuse to
  produce while isolated (multi-validator set, zero peers), so the first block
  is not published into a non-existent mesh.
- `POST /tx` now gossips accepted transactions; previously a tx submitted to a
  non-validator node stayed in that process's memory and was lost.
- The faucet now emits a signed transfer from a deterministic, genesis-funded
  faucet account. It previously mutated local state only, so drips never
  replicated and balances diverged between nodes.
- Validators refuse to start without an explicit `VNN_KEY_SEED` (the old
  default produced a publicly derivable signing key and colliding PeerIds).
- Added `GET /health` and permissive CORS for browser clients.
- Removed debug `eprintln!` noise from the execution path; block/tx publish
  failures with no peers are logged at debug, not warn.
- `deploy/setup-vps.sh` now fails on build errors (it previously ran
  `cargo build || true` and then copied a possibly stale binary) and installs
  the unit, env file and nginx site.
- CI no longer swallows failures with `|| true`.
- `verinoo-node address --seed <string>` derives a validator address before it
  boots, so `VNN_VALIDATORS` can be assembled without starting a node.
- `scripts/demo-2node.sh` reproduces the two-validator proof (11 checks) and
  `scripts/verify.sh <taskId>` verifies any task against any node.
- Address-bound API keys with per-key daily quotas, hashed at rest, issued and
  revoked with `verinoo-node apikey`, reloaded with SIGHUP. `/v1/usage` is
  tenant-scoped when keys are enabled.
- GGUF K-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K) dequantise with the
  llama.cpp layouts and are checked against reference vectors from the official
  `gguf` Python package.
- Catch-up sync: `GET /blocks/since/:index` plus `VNN_SYNC_URLS`. A late
  replica replays blocks through the same validation and state-root check;
  `scripts/demo-sync.sh` proves a three-block catch-up.
- `/ready` hardening: catch-up HTTP has 2 s connect / 5 s read / 5 s total
  timeouts, `syncing` is set only while replayed blocks are applied, peer
  heights are accepted only from peers with a matching `chainId` and are
  refreshed on every poll, and `validatorSetConsistent` fails closed with
  `validator_set_mismatch` when state disagrees with `VNN_VALIDATORS`. The
  validator-set hash is SHA-256 over canonical JSON of the sorted address
  array; it is an observability value, not part of the state root.
- `scripts/backup.sh` takes a consistent stopped-service backup, and
  [`docs/operations.md`](operations.md) covers restore, API-key rotation,
  validator-key rotation, a catch-up replica/HA recipe, monitoring and
  incident response.
- Input encryption (`verinoo-crypto`): X25519 + HKDF-SHA256 + ChaCha20-Poly1305,
  network-key decryption at execution, ciphertext in blocks/state, redacted
  task responses, `VNN_INPUT_KEY` / `VNN_REQUIRE_ENCRYPTED_INPUT`, and a
  documented threat model in [`docs/privacy.md`](privacy.md).
- TypeScript and Python SDKs with canonical JSON, Ed25519 signing, input
  encryption, faucet, task polling and verification; both live-tested against
  an encrypted node.
- `VNN_BLOCK_TIME_MS` lets a pilot run 1 s blocks (single validator) for ~1–2 s
  task latency; every validator must use the same value.
- A deterministic Q16.16 transformer encoder classifier
  (`crates/runtime/src/transformer.rs`) running in consensus. It uses a real
  trained model, fixed-point exponential attention softmax, encrypted inputs
  and a weights-bound receipt. 80/80 label parity with the float reference on
  all 80 corpus examples. See [`docs/transformer.md`](transformer.md).
- A deterministic GPT-style decoder runtime (`crates/runtime/src/gpt.rs`) with
  char-BPE and GPT-2 byte-level BPE tokenizers, causal attention, a per-layer
  KV cache, greedy or seeded sampling and binary `.vnnw` weights (48-byte
  header, SHA-256 payload, tamper detection). `tiny-gpt-v1` is a
  110,400-parameter synthetic demo; `tiny-stories-1m` is a real 3.75M-parameter
  GPT-Neo decoder matching its float reference for 32 tokens on 8 prompts. An
  Apache-2.0 81.9M-parameter DistilGPT-2 with 1,024-token context is imported
  locally and matches its float reference for 16 greedy tokens on 3 prompts.
  See [`docs/decoder.md`](decoder.md). DistilGPT-2 and TinyStories weights are
  downloaded locally and are not redistributed; TinyStories has no stated
  license, DistilGPT-2 is Apache 2.0 per its model card.
- A BFT finality design (`docs/adr/004-bft-finality.md`). Not implemented.

## Verified end-to-end (2-node product run)

```
PASS model registry lists intent-classifier-v1 (8 196 params)
PASS funded requester on both nodes (10 000)
PASS inference submitted, task cd7b41c1…
PASS task executed by consensus on both nodes (completed)
PASS both nodes derived an identical result hash (2b22e4f6…)
PASS label is shipping (correct intent)
PASS receipt verifies on the producing node
PASS receipt verifies on an independent node
PASS metering counted the task (feesCommitted 100)
PASS CSV export for invoicing
PASS chains converged (height 4)
```

Reproduce with `./scripts/demo.sh` (single node) or the two-node setup below.

## Verified end-to-end (operator model set, 2 nodes)

```
PASS operator model registered from model set (acme-support-v1)
PASS built-in model still registered
PASS model set hash 510eb42bc26754d3…
PASS task submitted against acme-support-v1
PASS executed by consensus on both nodes
PASS identical result hash on both nodes
PASS label shipping on both nodes
PASS independent node verifies the receipt
PASS receipt binds the operator model's weights hash
PASS built-in model still executes on both nodes
PASS chains converged (height 5)
PASS identical head hash
PASS no state root mismatch
PASS metering counted both tasks
```

`models/acme-support-v1.graph.json` is the reference classifier exported
through the generic runtime: a second `modelId`, real 8,196-parameter weights,
no compiled-in code path. Reproduce with `VNN_MODEL_DIR=./models` on both nodes
and `verinoo-node infer --model acme-support-v1`.

## Verified end-to-end (2-node testnet)

```
PASS validator set exposed: ai_1735... / ai_04df...
PASS connection stable (0 disconnects in 12s)
PASS both nodes advanced past genesis (heights 1 / 1)
PASS faucet drip replicated to both nodes (10 000 / 10 000)
PASS cross-node transfer mined and replicated (100 / 100)
PASS chains converged (final heights 6 / 6)
```

## Verified end-to-end (`scripts/demo-2node.sh`, 2026-11-12)

```
PASS both nodes healthy
PASS validator set exposed on both nodes
PASS chains converged (height 2)
PASS faucet drip replicated to both nodes
PASS task executed by consensus on both nodes
PASS both nodes derived an identical result hash (2b22e4f6…)
PASS receipt verifies on the producing node
PASS receipt verifies on the independent node
PASS metering counted the task (feesCommitted 100)
PASS CSV export for invoicing
PASS chains stayed converged (height 4/4)
✓ all 11 checks passed
```

This is the reproducible design-partner proof: two validators share a set,
execute the same task during block application, derive the same result hash, and
the receipt verifies on the node that did not produce the block.

## Live deployment (2026-11-15)

`https://verinoo.com/v1/*` is served by this node (`v0.9.0-testnet`, single
validator, systemd unit `verinoo-node`, data in `/var/lib/verinoo`). Nginx
proxies `/v1/`, `/faucet` and `/account/` to `127.0.0.1:7100` with rate limits;
the API and P2P ports are firewalled. Verified against the public endpoint:
faucet drip mined, inference completed in a block, `/v1/verify` returned
`verified: true`, and usage/CSV matched. The root `/status` and `/health` on the
domain still belong to the older reference node.

## Minimum bar to charge money

| Gate | Work | Est. |
|---|---|---|
| ~~Real forward pass on one small model with committed weights~~ | **done** for the built-in classifier | — |
| ~~Receipt API (signed receipt + block anchor)~~ | **done**: `/v1/infer`, `/v1/tasks`, `/v1/verify` | — |
| ~~Runtime for an operator-supplied model~~ | **done**: `verinoo-runtime`, dense/relu/softmax over Q16.16 | — |
| ~~ONNX classifier importer with a parity harness~~ | **done**: `verinoo-onnx`, `onnx-import` | — |
| ~~GGUF dense classifier importer~~ | **done**: `verinoo-gguf`, `gguf-import`, Q4/Q5/Q8 + K-quants | — |
| ~~Metering + billing export~~ | **done**: `/v1/usage`, `/v1/usage/:addr/csv` | — |
| ~~API keys, quotas, tenant isolation~~ | **done**: address-bound keys, per-key daily quota, SIGHUP rotation (`verinoo-node apikey`) | — |
| ~~Block catch-up sync~~ | **done**: `/blocks/since`, `VNN_SYNC_URLS`, validated replay (`scripts/demo-sync.sh`) | — |
| ~~Backups, restore, key rotation, incident runbook~~ | **done**: `scripts/backup.sh`, [`docs/operations.md`](operations.md) | — |
| Dispute game live for a real model | commit layer specs/weights at model registration; replace `get_layer_spec` | 2–3 wk |
| Finality or honest framing | Malachite adapter **or** document "operator-attested with economic recourse" | 4–6 wk / 1 wk |
| Public multi-party ceremony | external contributors to powers-of-tau | 2 wk + coordination |
| Automatic HA / validator removal | failover, split-brain protection, on-chain set changes | 2–3 wk |
| State snapshot transfer | serve and verify a signed state snapshot | 1–2 wk |

## Recommended sequence

1. ~~Finish the testnet launch and publish the honest "what is proven vs
   asserted" page.~~ Done: `docs/pilot.md`, `scripts/demo.sh`,
   `scripts/demo-2node.sh`.
2. Run three paid design partners at $5k for eight weeks against a classifier
   they care about (ONNX or GGUF, dense/relu/softmax). Issue each one an API key
   bound to its funded address with a daily quota.
3. Decide from those pilots whether the next engineering investment is a
   transformer runtime or deeper audit/ops features. Do not build BFT or a
   token before a pilot asks for it.
4. Use fiat invoicing from the CSV export; a token is a separate legal and
   product decision.

## Do not claim

- "Trustless" — there is one operator and no BFT.
- "Verified inference for any model" — only the supported runtimes execute
  deterministically: dense/relu/softmax graphs, a transformer encoder
  classifier, and GPT-2/GPT-Neo decoders including an imported 81.9M
  DistilGPT-2. Arbitrary checkpoints and LLaMA/SentencePiece tokenizers are not
  supported.
- "ZK-proven" — attention disputes fall back to hash equality.
- "Inputs are private from the operator" — inputs are encrypted to the
  validator set; every validator, including the operator, can decrypt. Pre-
  encryption tasks remain public.
- "LLM support" — a 3.75M GPT-Neo demo runs in consensus with GPT-2 BPE and a
  128-token context; a licensed 81.9M DistilGPT-2 with 1,024-token context,
  binary weights and deterministic sampling runs locally. It is not deployed or
  benchmarked as a service, and there is no GPU executor, batching, streaming,
  safety layer, 7B+ serving or arbitrary-checkpoint import. "BFT" — the finality
  design is not implemented.
