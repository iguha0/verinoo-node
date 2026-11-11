# Design partner pilot — what you are evaluating

Verinoo returns a model's answer **together with a receipt a third party can
check without trusting the operator**. This page is the diligence summary: what
is proven today, how to reproduce it yourself, and what is explicitly not
claimed yet.

Assessment date: 2026-11-12. Scope: the Rust node in this repository.

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
  hashed-unigram featuriser execute. Attention, convolutions, arbitrary
  tokenizers and sampling are not supported, so a Llama-class checkpoint cannot
  run yet.
- **Not ZK-proven.** The attention/matmul dispute path falls back to hash
  equality (ADR 003); `full-zk` is not what it sounds like.
- **No catch-up sync.** Nodes must start together and stay online. A node that
  falls behind stalls at that height; there is no snapshot transfer yet.
- **No multi-tenant auth.** Any caller with a funded testnet key can submit; no
  API keys, quotas or tenant isolation yet.

## Wire in your own classifier

The model must score the hashed-unigram feature vector (2048 by default) and
end in `softmax`. Conversion is offline; the on-chain path never touches floats.

```bash
# ONNX (Gemm/MatMul+Add/Relu/Softmax)
cargo run -p verinoo-onnx --bin onnx-import -- \
  --onnx your-model.onnx --model-id your-model-v1 \
  --labels class_a,class_b,class_c --out models/your-model-v1.graph.json

# GGUF dense classifier (Q4/Q5/Q8 dequantised; K-quants refused)
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
- **Usage: $0.50 per 1,000 tasks** once self-serve API keys and quotas ship.
- **Enterprise: from $60k / year** for a private network and custom runtime work.

Charge for the receipt, not the token count: the buyer is paying to prove what a
model returned.

## Known engineering gaps before self-serve

1. API keys, quotas, per-tenant isolation (1–2 weeks).
2. Live dispute-game RPC; today the challenge path is exercised in tests
   (2–3 weeks).
3. Byzantine finality (4–6 weeks) or continued honest "operator-attested with
   economic recourse" framing.
4. Operations: HA pair, backups, key rotation, incident runbook (2–3 weeks).

Nothing in that list blocks a design-partner pilot against a classifier.
