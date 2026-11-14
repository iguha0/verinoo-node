# Design partner pilot — what you are evaluating

Verinoo returns a model's answer **together with a receipt a third party can
check without trusting the operator**. This page is the diligence summary: what
is proven today, how to reproduce it yourself, and what is explicitly not
claimed yet.

Assessment date: 2026-11-12. Scope: the Rust node in this repository.

## Download and live demo

Prebuilt `x86_64` Linux binary (OpenSSL 3):
<https://github.com/iguha0/verinoo-node/releases/tag/v0.3.0-testnet>

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
| On-chain settlement of fees | signed transfers, gas market, on-chain fee split |

## What is not claimed

- **Not trustless.** One operator runs both validators. Round-robin leader
  election with one producer signature per block; no BFT, no finality gadget,
  no equivocation slashing in the live path.
- **Not an LLM host.** Only dense / relu / softmax classifiers over the
  hashed-unigram featuriser execute. A deterministic Q16.16 attention kernel
  exists as a prototype (`crates/runtime/src/attention.rs`) but is not wired
  into model loading or consensus. Attention, convolutions, arbitrary
  tokenizers and sampling are not supported, so a Llama-class checkpoint cannot
  run yet.
- **Not ZK-proven.** The attention/matmul dispute path falls back to hash
  equality (ADR 003); `full-zk` is not what it sounds like.
- **Catch-up, not snapshot sync.** A late node replays blocks from genesis via
  `VNN_SYNC_URLS`; there is no state snapshot transfer yet, so a very long
  chain takes proportionally longer to join.
- **Auth is gateway-local.** API keys are address-bound and enforce a daily
  quota, but quotas are billing/abuse control, not a consensus rule. Keys are
  issued by the operator (`verinoo-node apikey`); there is no public signup yet.
  The public demo runs without keys; paid deployments enable `VNN_API_KEYS_FILE`.
- **Input privacy is validator-confidential, not operator-blind.** Inputs are
  encrypted to a network X25519 key, so API/block readers see only ciphertext
  (`docs/privacy.md`). Every validator, including the operator, can decrypt.
  Pre-encryption tasks remain public; start a fresh chain for a pilot.

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
- **Usage: $0.50 per 1,000 tasks** on a self-serve key with a daily quota.
- **Enterprise: from $60k / year** for a private network and custom runtime work.

Charge for the receipt, not the token count: the buyer is paying to prove what a
model returned.

## Known engineering gaps before self-serve

1. Transformer runtime. The Q16.16 attention kernel is a prototype; embeddings,
   tokenizer, layernorm, FFN, KV cache, sampling and model loading are not
   implemented (`docs/transformer.md`). 4–8 weeks.
2. Byzantine finality. `docs/adr/004-bft-finality.md` is a design only; no
   prevote/precommit, certificate or slashing code exists. 5.5–7 weeks.
3. Self-serve key signup; keys are issued manually today.
4. Automatic HA/failover and on-chain validator removal (2–3 weeks). Manual
   replica promotion and coordinated key rotation are documented in
   [`docs/operations.md`](operations.md).
5. State snapshot transfer for fast joins (1–2 weeks).
6. Dispute games for a real model; the deterministic path verifies by
   recomputation, so there is currently nothing to dispute.

API keys, quotas, SDKs (TypeScript + Python), input encryption, catch-up sync,
backups, restore, API-key rotation and an incident runbook now exist. Nothing
in the list above blocks a design-partner pilot against a classifier.
