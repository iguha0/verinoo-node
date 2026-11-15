# Transformer encoder runtime

Status: **first slice implemented and running in consensus.** This is a small
transformer **encoder classifier**, not a generative LLM runtime.

`crates/runtime/src/transformer.rs` executes a pre-norm transformer encoder over
a word-level vocabulary and a classifier head. Every arithmetic operation is
integer or Q16.16 fixed point, so every node derives the same bytes.

## What runs

`models/intent-transformer-v1.json`:

| Field | Value |
|---|---|
| architecture | `transformer-encoder-1x16-2h` |
| parameters | 5,508 |
| hidden / heads / head_dim | 16 / 2 / 8 |
| encoder layers | 1 (pre-norm, ReLU FFN, width 32) |
| vocabulary | 187 word-level tokens |
| max sequence | 12 tokens |
| pooling | CLS |
| labels | refund, shipping, billing, cancel |

Forward pass:

```text
x = embedding(tokens) + positional
for each layer:
  y = layer_norm(x)
  q, k, v = dense(y)
  a = multi_head_attention(q, k, v, heads, head_dim)
  x = x + dense(a)
  y = layer_norm(x)
  x = x + dense(relu(dense(y)))
x = final_layer_norm(x)
logits = dense(x[cls])
scores = softmax(logits)
```

Layer norm uses integer sums, biased variance, an integer square root and a
Q16.16 reciprocal. Attention uses the Q16.16 kernel in
`crates/runtime/src/attention.rs`, including a fixed-point exponential softmax
(range-reduced `exp(-z) = 2^-n * exp(-r)`, five-term Taylor). No floats, threads
or hash-map iteration are used.

## Accuracy and parity

- Train 1.000 (60 examples), validation 0.900 (10), held-out test 0.800 (10).
- Overall 0.963 on all 80 corpus examples.
- **80/80 label parity** between the Q16.16 runtime and the float PyTorch
  reference on every corpus example (`crates/runtime/src/transformer.rs` test).
- The corpus is 80 examples. Treat these numbers as a demonstration, not a
  benchmark. The Naive Bayes `intent-classifier-v1` remains the more accurate
  classifier on this corpus.

The transformer is registered as a second model. Tasks for
`intent-transformer-v1` are executed by consensus like any other model; the
receipt binds its `weightsHash` (`8b5aad95…`).

## Model format

The JSON file contains:

| Field | Meaning |
|---|---|
| `kind` | must be `transformer-classifier` |
| `vocab` | token strings in id order; must include `<pad>`, `<unk>`, `<cls>` |
| `maxSeqLen`, `hidden`, `heads`, `ff`, `layers` | shape metadata |
| `labels` | class names, in output order |
| `embedding` | `vocabSize × hidden` Q16.16, row-major |
| `positional` | `maxSeqLen × hidden` Q16.16 |
| `encoder[]` | per layer: layernorm, Q/K/V/O, FFN weights and biases |
| `lnfGamma`, `lnfBeta` | final layernorm |
| `classifierWeights`, `classifierBias` | output head |

All weights are signed Q16.16 integers. The runtime validates every length and
rejects unknown/missing fields instead of guessing. `weightsHash` is SHA-256
over the canonical parsed model, not the file bytes.

## Train and export

```bash
python3 scripts/train_transformer.py
# train 1.000 (60)  val 0.900 (10)  test 0.800 (10)  params 5508
# wrote models/intent-transformer-v1.json
# wrote models/intent-transformer-v1.test.json
```

The script uses PyTorch (CPU is fine), a deterministic seed, early stopping on
a validation split, and writes both the quantised model and float-reference
test vectors. `--seed`, `--epochs` and `--out-dir` are configurable.

## What is still missing

- **Generation in this runtime.** No decoder or causal mask here. A separate
  decoder runtime now exists with BPE, a KV cache and greedy decoding; see
  [`docs/decoder.md`](decoder.md). It is a tiny synthetic demo, not LLM support.
- **Tokenizers.** ASCII words only. No BPE, WordPiece, SentencePiece, byte-level
  mapping or Unicode normalisation.
- **Scale.** One tiny layer, context 12, vocabulary 187. Long context, multi-layer
  models and large vocabularies are untested.
- **ONNX attention import.** `onnx-import` still accepts only dense/relu/softmax
  classifiers. A transformer checkpoint must be trained/exported through this
  format for now.
- **Parity harness beyond labels.** The test checks the predicted label on every
  corpus example, not full logit equality. Quantisation is label-preserving on
  this corpus only.
- **A real customer model.** The next milestone is a customer encoder classifier
  trained on their data, not a larger synthetic demo.

## Why this matters

Until this slice, the chain could only execute dense classifiers. It now runs a
real attention encoder in consensus, with encrypted inputs, a weights-bound
receipt and independent recomputation. That is the first step toward the models
buyers actually ask about; it is not the whole way there.
