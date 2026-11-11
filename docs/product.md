# Verinoo — verifiable inference API

**Status: working product vertical slice on a testnet chain (November 2026).**
Not a frontier-model host, and not trustless. Read the limits before selling it.

## The product

A paid HTTP API that returns a model's answer **together with a receipt that a
third party can check without trusting the operator**.

Every request is a transaction. When a block is produced, each node applying it
re-runs the model over the same input with the same weights, as part of state
transition. The output is therefore consensus state, not a claim attached to a
response. The receipt binds:

| Field | Meaning |
|---|---|
| `taskId` | the on-chain task |
| `modelId`, `weightsHash` | exactly which parameters produced the output |
| `inputHash`, `outputHash` | commitment to input and to the result bytes |
| `height`, `blockHash` | the block that carries the execution |
| `validator`, `validatorSignature` | the Ed25519 signature over that block header |

Verification is a recomputation, not a proof: `GET /v1/verify/:taskId` re-runs
the model and compares against the value the chain committed. If they differ,
the chain itself is inconsistent — that is the invariant a validator protects.

## What is actually true today

- **Real execution.** `intent-classifier-v1` is a multinomial Naive Bayes
  classifier over hashed unigrams: 8,196 integer parameters, trained from the
  80-example corpus in `crates/model/data/intents.jsonl` by
  `cargo run --release -p verinoo-model --bin train`.
- **Honest accuracy.** 98.4% on the training split, 100% on the held-out fifth
  (16 examples), one `billing` → `cancel` confusion. The corpus is small; treat
  the number as a demonstration, not a benchmark.
- **Operator-supplied models.** A node loads graph models from
  `VNN_MODEL_DIR` and executes them in consensus the same way: dense / relu /
  softmax over Q16.16 integers. `models/acme-support-v1.graph.json` is a second
  model executed that way on two nodes — the reference classifier exported
  through the generic runtime, so its scores match byte-for-byte.
- **Deterministic across machines.** Inference is Q16.16 integer arithmetic with
  no floating point and no threading. Two independent nodes produced the same
  result hash for the same task in the verification run below.
- **Signatures and state are real.** Ed25519 over canonical tx ids and block
  headers, RocksDB state, gas market, faucet, on-chain fee settlement.

## What is not true — do not claim it

- **Not trustless.** One operator runs both validators. There is no BFT
  consensus: round-robin leader election with a single producer signature per
  block, no prevote/precommit, no equivocation slashing in the live path.
- **Not a general model host.** Dense / relu / softmax classifiers over the
  hashed-unigram featuriser convert from ONNX (`onnx-import`) or GGUF
  (`gguf-import`, dequantising Q4/Q5/Q8) — see
  [`docs/runtime.md`](runtime.md). Transformer graphs — attention, convolutions,
  arbitrary tokenizers, sampling — are not supported, so a customer's LLM
  checkpoint cannot run yet. Tasks for models a node has not loaded fall back to
  the manual claim → submit result flow and are **not** verified by the chain.
- **Not ZK-proven.** The attention/matmul dispute path still falls back to hash
  equality (ADR 003); `full-zk` is not what it sounds like.
- **No catch-up sync.** Nodes must start together and stay online. A node that
  misses a block stalls the network at that height.
- **No multi-tenant auth.** Any caller with a funded key can submit; there are
  no API keys, quotas or per-tenant isolation yet.

## API

Base URL is the node's RPC port (default `:7100`).

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/models` | models this node can execute, with `weightsHash` and parameter count |
| `POST` | `/v1/infer` | submit a signed inference tx (same body as `/tx`); returns `txId` + `taskId` |
| `GET` | `/v1/tasks/:taskId` | task state, output, and the receipt once anchored |
| `GET` | `/v1/verify/:taskId` | recompute and compare; returns `verified`, both hashes, the output and the receipt |
| `GET` | `/v1/usage/:address` | metering: tasks, completed, pending, failed, fees committed, balance |
| `GET` | `/v1/usage/:address/csv` | the same as a one-row CSV for invoicing |

Read endpoints (existing): `/health`, `/status`, `/metrics`, `/blocks`,
`/account/:address`, `/validators`, `/mempool`, `POST /tx`, `POST /faucet`.

```bash
# submit
verinoo-node infer --key-file key.json --input "my parcel was never delivered" \
  --node http://localhost:7100
# → task: 0f3c…   tx: 9ab1…

curl -s localhost:7100/v1/verify/0f3c… | jq '{verified, outputMatches, receipt}'
```

The customer-facing receipt is self-contained: hand `receipt` plus the output to
an auditor and they can re-run `GET /v1/verify` on any node in the network.

## Bringing a customer model

Place a graph file in `VNN_MODEL_DIR` and every node in the network loads it at
startup. `/v1/models` exposes the set and its hash; the graph, its parameters
and the parity requirement are documented in [`docs/runtime.md`](runtime.md).

```bash
# from the reference classifier (compiled-in path, exported as a graph)
cargo run -p verinoo-model --bin export -- \
  --model-id acme-support-v1 --out models/acme-support-v1.graph.json

# from a customer ONNX classifier (parity-checked against f32)
cargo run -p verinoo-onnx --bin onnx-import -- \
  --onnx customer.onnx --model-id customer-v1 --labels a,b,c \
  --out models/customer-v1.graph.json

# from a customer GGUF dense classifier (Q4/Q5/Q8 dequantised)
cargo run -p verinoo-gguf --bin gguf-import -- \
  --gguf customer.gguf --model-id customer-v1 --labels a,b,c \
  --out models/customer-v1.graph.json

verinoo-node infer --model customer-v1 --input "my parcel was never delivered"
```

Model files must be identical across nodes: a mismatch changes the state root
and peers reject blocks rather than forking.

## Run it

```bash
cargo build --workspace
./scripts/demo.sh          # one node: faucet → inference → verification, ~45s
./scripts/demo-2node.sh    # two validators: consensus + independent verify, ~2min
```

Or manually:

```bash
VNN_KEY_SEED="$(openssl rand -hex 32)" VNN_VALIDATOR=1 ./target/debug/verinoo-node
```

See the top-level `README.md` for multi-node setup and `docs/readiness.md` for
the full readiness matrix.

## Pricing (proposed, fiat — no token)

| Plan | Price | Includes |
|---|---|---|
| Usage | **$0.50 per 1,000 tasks** | metered from `/v1/usage`, invoiced monthly from the CSV export |
| Design partner | **$5,000 / 8 weeks** | dedicated node, your model wired in, joint verification run, weekly sync |
| Enterprise | from **$60k / year** | private network, custom model runtime, audit support |

Charge for the *receipt*, not the token count: the buyer is paying to be able to
prove what a model returned, which is worth more than the compute.

## The honest pitch

> "We re-execute your inference on independent nodes and give you a receipt any
> third party can verify. Today that is proven for a small classifier on our
> testnet. If you need Llama-class models or Byzantine fault tolerance, we are
> not there yet — and we will tell you that in the first call."

The first thing a technical buyer will test is a model they care about. A
dense classifier converts from ONNX or GGUF in one command, executes in
consensus, and comes with a measured parity figure against its float reference
(see [`docs/runtime.md`](runtime.md)); what is missing is transformer support, so
an LLM checkpoint still cannot run. Everything around it — receipts,
verification, metering, settlement — already works.
