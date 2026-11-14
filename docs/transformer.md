# Transformer roadmap — Q16.16 attention prototype

`crates/runtime/src/attention.rs` adds a deterministic, integer-only
scaled-dot-product attention kernel and a multi-head wrapper. It is a
**prototype kernel, not transformer support**. The runtime still executes only
`dense` / `relu` / `softmax` classifier graphs; nothing in `Model::forward`, the
graph loader, `crates/exec`, `crates/node`, or the dispute game calls this
module. A model file containing an `attention` layer is still rejected at load
time.

## What the kernel does

Both entry points use Q16.16 (`ONE = 65_536`) and are row-major.

| Function | Layout |
|---|---|
| `scaled_dot_product_attention` / `scaled_dot_product_attention_shaped` | `q`, `k`, `v`: `[seq_len, head_dim]`, one head |
| `multi_head_attention` | `q`, `k`, `v`: `[seq_len, heads, head_dim]`; output identical |

For each query row `i` and key row `j`:

```text
scores[i][j] = (Σ_d q[i][d] * k[j][d]) >> 16
scores[i][j] = (scores[i][j] * scale) >> 16       scale ≈ 1 / sqrt(head_dim)
p[i]         = softmax(scores[i])                 runtime integer softmax
out[i][d]    = (Σ_j p[i][j] * v[j][d]) >> 16
```

* `scale = floor(sqrt(2^32 / head_dim))`, a Q16.16 approximation within one
  least-significant unit of the real `1 / sqrt(head_dim)`. The square root is
  integer-only.
* Score and output accumulators are `i64` with saturating arithmetic; the
  scaled score and the output are clamped to `i32`. The scale multiplication
  uses an `i128` intermediate to avoid overflow before the final shift.
* Heads are evaluated sequentially and concatenated back into
  `[seq_len, heads, head_dim]`. `multi_head_attention` returns
  `Result<Vec<i32>, String>` and rejects zero `heads` / `head_dim`, mismatched
  lengths, a length that is not a multiple of `heads * head_dim`, and an empty
  sequence.
* There is no floating point, threading, hash-map iteration, or dependence on
  allocation addresses. Loop order is fixed, so repeated runs are byte-identical.
  Any future blocking, SIMD, or parallel version must preserve the same
  summation order and the single final shift.
* The flat `scaled_dot_product_attention` cannot recover `head_dim` from three
  equal-length slices. It infers the largest divisor of the total length not
  exceeding its integer square root (so 6 values become `3 x 2`), and returns an
  empty vector on mismatched or empty input. Use the `_shaped` variant when the
  shape is known and may be wider than it is long.

The softmax is the classifier softmax from `lib.rs`, not `exp`: an offset below
the row maximum becomes `0`, an offset `d >= 0` becomes `d + ONE`. It is
monotone but piecewise linear, rows sum to at most `ONE`, and every score
strictly below the row maximum receives exactly zero probability. That is a
coarse approximation for attention scores.

## What it does not do

* No tokenizer or detokenizer. No BPE, byte-level mapping, Unicode
  normalisation, special tokens, or vocabulary file.
* No embeddings or positional encodings (learned, RoPE, ALiBi).
* No Q/K/V/O projections, residual connections, LayerNorm/RMSNorm, or FFN.
* No causal or padding mask. The kernel attends to every position.
* No KV cache, prefill, or incremental decoding.
* No batch dimension, grouped/multi-query attention, dropout, or context-length
  policy.
* No logits-to-token decoding or sampling (greedy, temperature, top-k, top-p).
* No transformer model format, tensor loader, or `weightsHash` covering a
  transformer checkpoint.
* No parity harness against a float reference and no ONNX/GGUF transformer
  import.
* No integration with receipts, consensus, `crates/exec`, or `crates/node`.

## Remaining work, in implementation order

### 1. Tokenizer and embeddings

A tokenizer is consensus-critical: two nodes that split text differently
produce different token ids and therefore different outputs. Start with one
frozen tokenizer version, ideally a byte-level BPE with a checked-in vocabulary
and merge table. Token lookup should use a sorted vector or `BTreeMap`, never
`HashMap` iteration. Pin the vocab/merge file by hash and reject a model whose
tokenizer hash does not match.

Add an `embedding` op that maps `[seq_len]` token ids to a
`[seq_len, d_model]` Q16.16 matrix, plus a fixed positional embedding (learned
positional embeddings first; RoPE is a separate design decision because it
changes attention itself). Validate every id against `vocab_size`, cap
`seq_len`, and reject unknown ids rather than silently clamping.

Tests: fixed strings to token ids, the same ids after reload, out-of-range id
rejection, embedding row lookup, and positional-add saturation.

Estimate: 1.5–2.5 weeks, dominated by tokenizer edge cases and fixtures.

### 2. LayerNorm / RMSNorm in Q16.16

Implement integer mean/variance (LayerNorm) or mean-square (RMSNorm), integer
square root, and an integer reciprocal. Define floor/rounding behaviour
explicitly, fix epsilon as a Q16.16 constant, and clamp only at specified
points. `crates/zk` has an integer layernorm reference used by the dispute
simulation, but it seeds its square root from `f64`; that is not acceptable for
the on-chain runtime, so port the algorithm with a pure integer `isqrt` and
golden vectors. Do not share code by modifying `crates/zk`.

Tests: known vectors including constant rows, near-zero variance, and negative
inputs; exact repeatability; offline float comparison with a stated tolerance.

Estimate: 3–5 days.

### 3. FFN

Compose the existing integer `dense` with an activation and a second `dense`,
then add a saturating residual. ReLU can reuse the existing op. GELU/SiLU need
a deterministic integer approximation: either a fixed polynomial or a pinned
lookup table. A table is acceptable only if its contents and interpolation are
part of the model hash and versioned. Keep the same `i64` accumulator and
`>> 16` convention as `dense`, or document and test a deliberately different
version.

Tests: shape validation, per-layer parity against a float reference, saturation
at the extremes, and residual overflow.

Estimate: 1–1.5 weeks for ReLU-only, plus about 1 week for GELU/SiLU parity.

### 4. Causal mask and KV cache

Autoregressive decoding needs a causal mask and, for practical speed, a cache.
Add a mask before softmax: masked scores must produce zero probability without
changing the unmasked order of operations. A fixed integer sentinel is safer
than relying on "very negative" scores. Be careful with fully masked rows: the
current softmax would turn an all-equal sentinel row into a uniform
distribution, so reject or define that case.

Cache post-projection K and V at the same Q16.16 precision the model would
recompute; caching higher-precision values makes recomputation and incremental
decoding diverge. Define the layout per layer/head as an append-only
`[layer][head][position][head_dim]` buffer, a hard context cap, and the
prefill-versus-decode transition. Validate the cache offset and sequence length
on every call.

Tests: masked attention equals an offline masked reference; incremental decode
with a cache equals full recomputation logits for the same prefix; appending
past the cap is rejected; cache contents are identical after a reload.

Estimate: 2–3 weeks.

### 5. Decoding and sampling

Start with greedy decoding: argmax over logits with the lowest index winning
ties, a fixed stop-token set, and a maximum token budget. Return token ids, not
text, so the tokenizer/detokenizer stays a separate deterministic step.

If sampling is required, the PRNG, seed source, temperature, top-k, top-p and
tie-breaking must all be consensus-defined. Deriving the seed from a task id or
block hash makes sampling reproducible, but it also makes every sampled token a
consensus-bearing value; a fixed integer implementation is mandatory. Greedy
decoding is the safer first target.

Tests: deterministic argmax, stop conditions, and byte-identical output across
repeated runs. For sampling, pin golden PRNG streams.

Estimate: 3–5 days for greedy; roughly 1 more week for sampling policies.

### 6. Model file format with `weightsHash`

The current `weightsHash` is sha256 over the canonical JSON `ModelSpec`. A
transformer needs a versioned config (layers, heads, `d_model`, `d_ff`,
`vocab_size`, context cap, positional-encoding kind, activation kind), the
tokenizer hash, and a tensor table with shapes. A JSON file holding millions of
weights is impractical, so use a binary container or sharded tensors plus a
canonical manifest, while keeping the public hash semantics: the hash covers
every semantic input, including tokenizer data and all tensor bytes, and is
stable under file reordering or reformatting. Sort tensor names before hashing;
never rely on map iteration order.

Reject missing, extra, mis-shaped, or out-of-range tensors at load time. Keep
existing dense graph models loadable and hash-compatible. Document endianness
and quantisation so another implementation can reproduce the bytes.

Tests: hash stability under reordering, a one-byte tensor edit changing the
hash, a tokenizer change changing the hash, and malformed files failing closed.

Estimate: 2–3 weeks, more if sharded weights and memory limits are included.

### 7. Parity harness vs a float reference

Build an offline reference (f32 or f64, never on the on-chain path) and compare
per-kernel outputs first, then full-model logits and token sequences. Report max
absolute logit error, mean error, top-1 agreement, and exact-token agreement
over fixed prompts. Keep the integer outputs as golden vectors so a future
optimization that changes accumulation order fails loudly.

The harness is most useful if it grows with steps 1–6, but the final gate should
run after the model format and decoding are fixed. It should also compare a
known-good converted checkpoint against the original checkpoint, not only two
copies of the integer runtime.

Estimate: 2–3 weeks including fixtures and CI wiring.

## Accuracy and determinism risks

* **Q16.16 range and resolution.** The range is roughly ±32,768 with a
  resolution of 1/65,536. Large attention scores and FFN activations can
  saturate; clamping is part of the observable output, so changing a clamp bound
  or a saturation point changes consensus results. Validate or reject values
  before they overflow rather than relying on silent saturation where possible.
* **Score scaling.** `scale` is a floor approximation with up to one LSB of
  error. That is small per score but can cross a softmax threshold in a
  near-tie.
* **The classifier softmax is coarse for attention.** Below-max scores become
  exactly zero, probabilities are floor-divided, and rows sum to at most `ONE`,
  not exactly `ONE`. Long rows accumulate more truncation, and attention can
  look much sharper or more one-hot than the float model. Because the same
  softmax is part of existing classifier receipts, changing it would change
  historical outputs; a transformer path should version its softmax rather than
  redefine the classifier one.
* **Accumulation order.** Integer shifts discard low bits. Different orderings
  of the same products produce different scores even without overflow, and the
  non-linear softmax can amplify a one-LSB difference into a different token.
  Fix the order (`i`, `j`, `d` for scores; `i`, `d`, `j` for outputs) and keep
  it in tests. A parallel or SIMD reduction is only safe if it uses a fixed
  tree order and applies the shift at the same point.
* **Saturation is observable.** `i64` products of two `i32` values can overflow
  when summed over a large head. Saturating addition is deterministic but
  lossy. Define the maximum supported `head_dim` and activation range, and
  reject or test at the boundary.
* **Debug versus release.** Overflow checks differ between profiles if any
  non-saturating arithmetic slips in. The on-chain path must use explicit
  `saturating_*` or `checked_*` operations so both profiles agree.
* **Masking.** An all-masked row is undefined by the current softmax (it becomes
  uniform). Define padding and empty-prefix behaviour, and never let a mask
  change the summation order of unmasked entries.
* **Tokenizer drift.** Unicode handling, normalisation, merge tie-breaking and
  special-token rules must be frozen and hashed. A tokenizer "bug fix" is a
  model change.
* **Cache precision.** Cached K/V must be the same Q16.16 values that a full
  forward pass would produce. Caching the pre-shift accumulator or a wider
  intermediate changes later logits.
* **Platform details.** Avoid `usize` overflow on 32-bit builds, define tensor
  byte order, and validate dimensions before multiplying them.

### Why accumulation order matters

Two mathematically equal expressions are not equal after integer truncation:

```text
(a*b >> 16) + (c*d >> 16)  !=  (a*b + c*d) >> 16
```

The left side drops the low 16 bits twice; the right side drops them once.
Right-shifting a negative value also rounds toward negative infinity. In
attention the score feeds a softmax whose below-max branch is exactly zero, so a
few low bits can decide whether a key gets any probability at all. Consensus
requires byte-identical outputs from every node, so "numerically close" is not
enough: the order, the shift point, and the clamp point are all part of the
kernel's definition.

## Realistic effort

For a deliberately constrained first model — tiny GPT-2-style, small fixed
tokenizer, short context, greedy decoding, single-threaded CPU — the remaining
work is roughly **8–13 engineer-weeks**, depending on how much of the tokenizer
and model-conversion tooling already exists. The attention kernel in this change
is about half a day to a day including tests and this document.

That estimate does not include performance work, GPU support, large-vocabulary
tokenizers, RoPE/GQA, production hardening, adversarial testing, or an external
audit. An on-chain transformer that can serve real models is a multi-month
project, not a follow-up sprint.

## What to validate first

1. Per-kernel golden tests: embedding lookup, RMSNorm/LayerNorm, FFN, attention,
   masked attention, and a cache-versus-recompute check.
2. A tiny GPT-2-style model: 2 layers, `d_model = 64`, 4 heads
   (`head_dim = 16`), `d_ff = 256`, vocabulary around 1,000 tokens, context
   16–32. Use a fixed byte-level BPE snapshot and greedy decoding.
3. Fixed prompts with golden token ids and logits. Require exact token-sequence
   agreement, and record max logit delta rather than pretending it is zero.
4. Repeatability across processes and machines, then extend context length,
   then layer count, then vocabulary.
5. Only after that: larger models, longer context, KV-cache performance, and
   optional sampling.

## Status

This is a prototype. `attention.rs` is compiled and unit-tested, but it is not
reachable from a model file, the node, receipts, or consensus. The `attention`
labels in `crates/exec` layer specifications are challenge-shaped generic
layers, not this kernel. Do not describe the runtime as transformer-capable
until the items above are implemented, integrated, and parity-tested.
