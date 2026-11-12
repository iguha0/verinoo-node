# Model runtime — running an operator's weights in consensus

The chain treats an inference result as state, so every validator must derive
the *same bytes* from the same weights. `crates/runtime` is therefore a small,
integer-only executor: Q16.16 fixed point, `i64` accumulators, no floating
point, no threads, no framework runtime. A graph that cannot be executed
exactly this way is rejected at load time instead of producing a divergent
result.

## Loading models

Models are JSON files in `VNN_MODEL_DIR` (default `./models`). Every `*.json`
file is parsed and validated at startup; a bad file is reported and skipped.

```bash
VNN_MODEL_DIR=./models VNN_KEY_SEED=... VNN_VALIDATOR=1 ./target/debug/verinoo-node
# INFO model loaded | acme-support-v1
# INFO model set: 1 model(s) | hash 510eb42bc26754d3
```

`GET /v1/models` lists the compiled-in classifier plus everything loaded, and
returns `modelSetHash` — a fingerprint of the whole set.

**Every node must load the same set.** Execution is part of state transition,
so a missing or different model changes the state root; peers then reject each
other's blocks (logged as `state root mismatch at #N`). That failure mode is
deliberate: refusing to advance beats silently forking into two chains that
disagree about what a model returned.

## File format

```json
{
  "modelId": "acme-support-v1",
  "architecture": "dense-2048-32-4",
  "description": "Acme support triage",
  "features": 2048,
  "labels": ["refund", "shipping", "billing", "cancel"],
  "layers": [
    {"op": "dense", "input": 2048, "output": 32,
     "weights": [0, 65536, -32768], "bias": [0]},
    {"op": "relu"},
    {"op": "dense", "input": 32, "output": 4,
     "weights": [0], "bias": [0]},
    {"op": "softmax"}
  ]
}
```

| Field | Meaning |
|---|---|
| `modelId` | the identifier tasks name in `targetModel` |
| `features` | width of the hashed-unigram feature vector |
| `labels` | class names; must equal the graph's final width |
| `layers[].op` | `dense`, `relu` or `softmax` |
| `layers[].weights` | row-major, `output × input`, Q16.16 signed integers |
| `layers[].bias` | `output` Q16.16 integers |

Ops are exactly the integer kernels the dispute game already reasons about:

* `dense` — `y[j] = bias[j] + Σ_f (x[f] * W[j][f]) >> 16`
* `relu` — `x = max(x, 0)`
* `softmax` — integer approximation (same one the reference classifier uses)

Validation rejects: unknown ops, a layer whose `input` does not match the
previous width, a weight count that does not equal `input × output`, a bias of
the wrong length, a final width that does not match the label count, and
oversized layers.

## Identity

`weightsHash` is sha256 over the **canonical definition** (the parsed model
re-serialised), not the raw file bytes. Reformatting the JSON or reordering
keys keeps receipts valid; changing a parameter, a label or the feature width
changes the hash and invalidates old receipts. That hash is what receipts and
`/v1/verify` report, so a customer can prove which parameters produced an
answer.

## Converting a model

Today the featuriser is fixed: hashed unigrams scaled to Q16.16 shares of the
token total, exactly as `crates/model` does it. A model therefore needs to be a
classification graph over *those* features. Producing the file is a translation
step you run offline:

```bash
# the reference classifier, exported through the generic runtime
cargo run -p verinoo-model --bin export -- \
  --model-id acme-support-v1 --out models/acme-support-v1.graph.json
```

`models/acme-support-v1.graph.json` is committed and is exactly that: the
8,196-parameter classifier with a different `modelId`. It executes through the
generic path and produces byte-identical scores to the compiled-in model, which
is the parity guarantee a customer's own weights rely on:

```bash
verinoo-node infer --key-file key.json --model acme-support-v1 \
  --input "my parcel was never delivered"
curl -s localhost:7100/v1/verify/<taskId> | jq '{verified, weightsHash, label}'
```

The JSON above is what the node loads. For a customer model you do not write it
by hand: `onnx-import` produces it from their ONNX checkpoint.

## Importing ONNX

```bash
cargo run -p verinoo-onnx --bin onnx-import -- \
  --onnx acme-support-v2.onnx \
  --model-id acme-support-v2 \
  --labels refund,shipping,billing,cancel \
  --out models/acme-support-v2.graph.json
```

```text
read acme-support-v2.onnx: ir_version 8, producer pytorch, opset 13, 2 node(s), 2 initializer(s)
ops: Gemm -> Softmax
runtime layers: dense 2048->4, softmax
parameters: 8196, weightsHash: 57d93cd52b97541f...
parity: 64 sample(s), max |Δlogit| = 1.615e-04, argmax 64/64
wrote models/acme-support-v2.graph.json
```

Parameters are quantised to Q16.16 with round-half-away-from-zero; a value
outside ±32768 fails the import rather than being clamped, because a clamped
weight is a model the customer did not train. ONNX does not store class names,
so `--labels` supplies them in output order; `--features` is needed only when
the graph's input width is dynamic.

**Parity runs by default.** The importer evaluates the graph twice on the same
inputs — once in f32 with the original parameters, once in Q16.16 with the
converted ones — over 32 samples (real sentences through the production
featuriser plus pseudo-random token bags). It prints the largest logit
divergence and refuses to write the file if the predicted class ever changes.
`--check N` changes the sample count; `--check 0` skips the check, which means
shipping a conversion nobody measured.

| ONNX | Becomes |
|---|---|
| `Gemm` with `alpha = beta = 1` (activation in input 0, weights in B, bias in C) | `dense` with the bias folded in |
| `MatMul` (either operand order) followed by `Add` | `dense` |
| `Relu`, `Softmax` on the class axis | `relu`, `softmax` |
| `Identity`, `Dropout`, `Flatten` (axis 1), `Reshape`, `Squeeze`, `Unsqueeze` | pass-through |

Anything else is a hard error naming the node, as is a reshape that does not
preserve the feature vector, a bias of the wrong width, or an output width that
disagrees with `--labels`. The batch dimension is ignored — the runtime scores
one feature vector at a time.

The importer is checked against files written by the official `onnx` Python
package, not only against its own hand-encoded fixtures:

```bash
pip install onnx numpy
cargo build -p verinoo-onnx --bin onnx-import
python3 crates/onnx/verify_official.py   # exports the reference model, reimports it, diffs parameters
```

## Importing GGUF

GGUF is a container, not a graph: it stores named tensors and their
quantisation, so a transformer checkpoint still cannot be executed here. What it
*can* carry is a plain dense classifier, which `gguf-import` extracts from the
tensor table:

```bash
cargo run -p verinoo-gguf --bin gguf-import -- \
  --gguf acme-support-v3.gguf \
  --model-id acme-support-v3 \
  --labels refund,shipping,billing,cancel \
  --out models/acme-support-v3.graph.json

# inspect metadata and tensor names first
cargo run -p verinoo-gguf --bin gguf-import -- --gguf acme-support-v3.gguf --list
```

The importer needs a *plan*: which tensors are dense layers, in order. Pass
`--layers fc1,fc2` explicitly, or let the convention pick every 2-D tensor whose
name ends in `.weight` (or `/weight`), sorted by name. A plan whose widths do
not form a chain is refused before any tensor data is read. Missing bias tensors
become zeros and are reported as notes.

| GGML type | Handling |
|---|---|
| `F32`, `F16`, `BF16` | read directly, then quantised to Q16.16 |
| `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0` | dequantised with the llama.cpp block layouts |
| `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` | dequantised with the llama.cpp super-block layouts; checked against reference vectors from the official `gguf` Python package |
| `Q8_1`, IQ-quants | refused by name — no reference layout in this importer |

`--activation relu` (default) inserts a `relu` between dense layers; `none`
leaves the chain linear. `--labels` is required because GGUF stores no class
names. `--features` is checked against the first layer. `--check N` runs the same
f32-vs-Q16.16 parity harness as the ONNX importer over `N` samples (default 32)
and refuses to write the file if the predicted class ever changes;
`--check 0` skips it.

## Not supported

* Transformer checkpoints and any graph outside the ops above — attention,
  convolutions, embedding lookups, arbitrary tokenizers. GGUF support is limited
  to extracting dense classifier layers; it is not an LLM runtime.
* Sampling — the runtime returns scores and an argmax label, not generated text.
* Float weights — parameters are signed integers; conversion happens offline so
  the on-chain path never touches floating point.
