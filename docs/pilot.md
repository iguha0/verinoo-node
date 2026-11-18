# Design partner pilot — what you are evaluating

Verinoo returns a model's answer **together with a receipt a third party can
check without trusting the operator**. This page is the diligence summary: what
is proven today, how to reproduce it yourself, and what is explicitly not
claimed yet.

Assessment date: 2026-11-12. Scope: the Rust node in this repository.

## Download and live demo

Prebuilt `x86_64` Linux binary (OpenSSL 3):
<https://github.com/iguha0/verinoo-node/releases/tag/v0.11.0-testnet>

The `/v1` API is live on the testnet node behind `https://verinoo.com`
(single validator, restarts possible, no SLA):

```bash
curl -s https://verinoo.com/v1/models | jq
```

Run a full request against it:

```bash
verinoo-node wallet generate --output key.json
ADDR=$(jq -r .address key.json)
curl -s -X POST https://verinoo.com/faucet \
  -H 'content-type: application/json' -d "{\"address\":\"$ADDR\"}"
sleep 12
verinoo-node infer --key-file key.json \
  --input "my parcel was never delivered" --node https://verinoo.com
./scripts/verify.sh <taskId> --node https://verinoo.com
```

The root `/status` and `/health` on `verinoo.com` still belong to the older
reference node; the verifiable-inference product is the `/v1` surface.

## Run the proof yourself

Two commands, no hosted service, no token:

```bash
cargo build --workspace --locked
./scripts/demo-2node.sh
```

The two-validator run prints 11 checks. The important ones:

- the task completes on **both** nodes, not just the producer;
- both derive an **identical result hash**;
- `GET /v1/verify/:taskId` returns `verified: true` on the independent node;
- metering and the invoice CSV record the task.

Single-node quickstart: `./scripts/demo.sh`.

Verify any task from any node:

```bash
./scripts/verify.sh <taskId> --node http://localhost:7100
# or inspect the JSON directly
curl -s http://localhost:7100/v1/verify/<taskId> | jq '{verified, outputMatches, receipt}'
```

## What is proven

| Claim | Evidence |
|---|---|
| Output is consensus state, not a claim | every node re-runs the model while applying the block (`crates/exec/src/engine.rs`) |
| Deterministic across machines | Q16.16 integer math, no float/threads; identical hash on two nodes |
| Receipt binds task, model, weights, input, output, block, signature | `/v1/tasks/:id` and `/v1/verify/:id` |
| A third party can recheck without trusting us | `/v1/verify` recomputes and compares to the committed hash |
| Usage is metered for invoicing | `/v1/usage/:addr`, `/v1/usage/:addr/csv` |
| A customer classifier can run | ONNX (`onnx-import`) and GGUF (`gguf-import`) import to a runtime graph with f32 parity checks |
| A real transformer runs in consensus | `intent-transformer-v1`: Q16.16 attention encoder, 80/80 label parity, weights-bound receipt |
| A causal decoder runs in consensus | `tiny-gpt-v1`: char-BPE, KV cache, greedy, token parity on 8 prompts |
| A real pretrained decoder runs in consensus | `tiny-stories-1m`: 3.75M GPT-Neo, GPT-2 BPE, 32-token parity on 8 prompts |
| A licensed larger decoder can be imported | `distilgpt2`: 81.9M GPT-2, 1,024 context, binary `.vnnw`, 16-token parity on 3 prompts; local only |
| Deterministic sampling | temperature/top-k/top-p with a seeded integer RNG; cold sampling equals greedy |
| On-chain settlement of fees | signed transfers, gas market, on-chain fee split |

## What is not claimed

- **Not trustless.** One operator runs both validators. Round-robin leader
  election with one producer signature per block; no BFT, no finality gadget,
  no equivocation slashing in the live path.
- **Not a production LLM host.** A real pretrained 3.75M-parameter GPT-Neo
  decoder (`tiny-stories-1m`) runs in consensus with the GPT-2 byte-level BPE
  tokenizer, KV cache and greedy decoding. The runtime also supports 1,024-token
  context, deterministic sampling and binary weights; an Apache-2.0 81.9M
  DistilGPT-2 model is imported and parity-tested locally. It is not deployed on
  the public 1-CPU node and is slower than a 1 s block there. There is still no
  GPU executor, batching, streaming, safety layer, large-model (7B+) serving or
  LLaMA/SentencePiece and other checkpoint import, so a Llama-class checkpoint cannot be served yet.
- **Not ZK-proven.** The attention/matmul dispute path falls back to hash
  equality (ADR 003); `full-zk` is not what it sounds like.
- **Catch-up, not snapshot sync.** A late node replays blocks from genesis via
  `VNN_SYNC_URLS`; there is no state snapshot transfer yet, so a very long
  chain takes proportionally longer to join.
- **Auth is gateway-local.** API keys are address-bound and enforce a daily
  quota, but quotas are billing/abuse control, not a consensus rule. Keys can be
  issued self-serve with `POST /v1/keys` when `VNN_SELF_SERVE_KEYS=1` (one key
  per address, default 1000 tasks/day), or by the operator with
  `verinoo-node apikey`. The public demo runs without keys; paid deployments
  enable `VNN_API_KEYS_FILE` and/or self-serve issuance.
- **Input privacy is validator-confidential, not operator-blind.** Inputs are
  encrypted to a network X25519 key, so API/block readers see only ciphertext
  (`docs/privacy.md`). Every validator, including the operator, can decrypt.
  Pre-encryption tasks remain public; start a fresh chain for a pilot.

## Vertical demo: claims triage

A concrete example of the design-partner shape is in
[`docs/claims-pilot.md`](claims-pilot.md): an eight-queue claims classifier with
98.75% held-out accuracy, 100% integer parity, a one-command receipt demo and a
pilot report you can send to a model-risk team.

```bash
./scripts/demo-claims.sh
```

## Wire in your own classifier

The model must score the hashed-unigram feature vector (2048 by default) and
end in `softmax`. Conversion is offline; the on-chain path never touches floats.

```bash
# ONNX (Gemm/MatMul+Add/Relu/Softmax)
cargo run -p verinoo-onnx --bin onnx-import -- \
  --onnx your-model.onnx --model-id your-model-v1 \
  --labels class_a,class_b,class_c --out models/your-model-v1.graph.json

# GGUF dense classifier (Q4/Q5/Q8 and Q2_K–Q6_K dequantised)
cargo run -p verinoo-gguf --bin gguf-import -- \
  --gguf your-model.gguf --model-id your-model-v1 \
  --labels class_a,class_b,class_c --out models/your-model-v1.graph.json
```

Both importers run a float-vs-integer parity check by default and refuse to write
a file if the predicted class changes. Put the graph in `VNN_MODEL_DIR` on
**every** node; `/v1/models` exposes the set and its hash. A mismatch changes the
state root and peers reject the block instead of forking.

## Pilot shape

- **Design partner: $5,000 / 8 weeks.** Dedicated validator set, your classifier
  wired in, a joint verification run, and a weekly sync. Billed in fiat; usage
  reported from the CSV export.
- **Usage: $5 per 1,000 tasks** on a self-serve key with a daily quota.
- **Enterprise: from $60k / year** for a private network and custom runtime work.

Charge for the receipt, not the token count: the buyer is paying to prove what a
model returned.

## Known engineering gaps before self-serve

1. Production LLM serving. A transformer encoder and a real 3.75M GPT-Neo
   decoder with GPT-2 BPE, KV cache, sampling and 128-token context run
   end-to-end; an Apache-2.0 81.9M DistilGPT-2 with 1,024-token context and
   binary weights is imported locally. GPU execution, batching, streaming,
   safety, LLaMA tokenizers and ONNX LLM import are not implemented
   (`docs/decoder.md`). 4–8 weeks.
2. Byzantine finality. `docs/adr/004-bft-finality.md` is a design only; no
   prevote/precommit, certificate or slashing code exists. 5.5–7 weeks.
3. Self-serve key signup exists behind `VNN_SELF_SERVE_KEYS=1`, but there is
   no automated billing/payment or key-revocation UI yet.
4. Automatic HA/failover and on-chain validator removal (2–3 weeks). Manual
   replica promotion and coordinated key rotation are documented in
   [`docs/operations.md`](operations.md).
5. State snapshot transfer for fast joins (1–2 weeks).
6. Dispute games for a real model; the deterministic path verifies by
   recomputation, so there is currently nothing to dispute.

API keys, quotas, SDKs (TypeScript + Python), input encryption, catch-up sync,
backups, restore, API-key rotation and an incident runbook now exist. Nothing
in the list above blocks a design-partner pilot against a classifier.
