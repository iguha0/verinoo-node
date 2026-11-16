# Generative decoder runtime

Status: **three decoder slices, binary weights and deterministic sampling.**
`DistilGPT-2` is a licensed 82M-parameter model with 1,024-token context. It is
imported and parity-tested locally, but is **not served on the public
`verinoo.com` node** and is not yet a production LLM service.

Sampling can be fixed in the model JSON or chosen per task. A per-task choice
is signed with the transaction, executed by every validator and exposed in the
receipt.

| Model | What it is | Source |
|---|---|---|
| `tiny-gpt-v1` | synthetic two-layer GPT-style decoder, char-BPE, 110,400 params | trained by `scripts/train_gpt.py` |
| `tiny-stories-1m` | real pretrained GPT-Neo decoder, GPT-2 byte-level BPE, 3.75M params | `roneneldan/TinyStories-1M` |
| `distilgpt2` | real pretrained GPT-2 decoder, GPT-2 byte-level BPE, 81.9M params, 1,024 context | `distilgpt2` (Apache 2.0) |

All use causal attention, a per-layer KV cache and greedy or seeded sampling.
Every op is integer or Q16.16, so two nodes with the same model file and seed
generate the same token ids from the same prompt. A config-driven importer,
`scripts/import_hf_gpt.py`, handles other checkpoints in the same GPT-2/GPT-Neo
subset (it is not arbitrary LLM checkpoint support).

## What runs

### `tiny-gpt-v1` (synthetic)

| Field | Value |
|---|---|
| architecture | `gpt-decoder-2x64-4h` |
| parameters | 110,400 |
| hidden / heads / head_dim | 64 / 4 / 16 |
| decoder layers | 2 (pre-norm, causal attention, ReLU FFN width 128) |
| vocabulary | 320 char-BPE tokens |
| context / generation | 32 tokens / at most 24 new tokens |
| training data | 3,847-line synthetic support corpus |

### `tiny-stories-1m` (real pretrained)

| Field | Value |
|---|---|
| architecture | `gpt-neo-8x64-16h` |
| parameters | 3,745,984 (tied embeddings) |
| hidden / heads / head_dim | 64 / 16 / 4 |
| decoder layers | 8 (pre-norm, causal attention, `gelu_new` FFN width 256) |
| vocabulary | 50,257 GPT-2 byte-level BPE tokens |
| context / generation | 128 tokens (truncated from 2048) / 32 new tokens |
| attention | no score scaling (GPT-Neo), tied embeddings, LN epsilon 1e-5 |
| local attention | 8 layers alternate global/local with window 256; context 128 means the window never matters |

### `distilgpt2` (real pretrained, licensed)

| Field | Value |
|---|---|
| architecture | `gpt2-6x768-12h` |
| parameters | 81,912,576 (tied embeddings) |
| hidden / heads / head_dim | 768 / 12 / 64 |
| decoder layers | 6 (pre-norm, causal attention, `gelu_new` FFN width 3,072) |
| vocabulary | 50,257 GPT-2 byte-level BPE tokens |
| context / generation | 1,024 tokens / 16 new tokens by default |
| attention | GPT-2-style sqrt scaling, tied embeddings, LN epsilon 1e-5 |
| weights file | `models/distilgpt2.vnnw`, 327.7 MB binary (gitignored) |
| license | Apache 2.0 per the model card; weights downloaded, not redistributed |
| deployment | local/GPU path only; not served on the public 1-CPU node |

Generation:

```text
ids = [BOS if configured] + tokenize(prompt)
logits = prefill(ids, cache)
repeat up to maxNewTokens:
  next = argmax(logits) if temperatureMilli == 0
         else sample(logits, temperature, topK, topP, seeded RNG)
  append next
  if next == EOS: stop
  logits = step(next, cache)
text = decode(generated ids)
```

`prefill` runs the prompt through every layer, fills the per-layer K/V cache,
and returns the last position's logits. Each `step` embeds one token, projects
Q/K/V, appends K/V to the cache, attends only to cached keys (causal), and runs
the FFN. No full-sequence recomputation per token.

## Sampling

Sampling defaults are part of the model spec and are deterministic across
nodes:

```json
"sampling": {
  "temperatureMilli": 800,
  "topK": 40,
  "topPMilli": 900,
  "seed": 7
}
```

| Field | Meaning |
|---|---|
| `temperatureMilli` | `0` = greedy; otherwise logits are divided by `temperatureMilli / 1000` in Q16.16 |
| `topK` | keep the `K` highest-scoring tokens; `0` keeps all |
| `topPMilli` | nucleus cutoff; `0` or `1000` disables it |
| `seed` | SplitMix64 seed; the same prompt, weights and seed give the same token ids |

The sampler scales logits in fixed point, applies top-k, truncates the nucleus,
quantises the softmax to Q16.16 and draws with a SplitMix64 integer PRNG. There
is no floating-point random number generation. Temperature `0` keeps the old
greedy path, so existing models and fixtures are unchanged.

Sampling defaults come from the model JSON. A client may override it per task
by adding the same object to `submitInference` data:

```json
{
  "type": "submitInference",
  "data": {
    "targetModel": "tiny-gpt-v1",
    "input": "Once upon a time",
    "sampling": {
      "temperatureMilli": 800,
      "topK": 40,
      "topPMilli": 900,
      "seed": 7
    }
  }
}
```

The optional `sampling` object is canonical transaction data, so it changes
`txId` and is bound to the derived task id. The task stores it, every validator
uses it during deterministic execution, and `/v1/verify` recomputes with it.
The receipt includes `sampling` and `samplingHash = sha256(canonical_json(
sampling))`; `samplingSource` is `task` for an override and `model-default`
when no override was supplied. Tasks without sampling keep the model default,
so old tasks and receipts are unchanged.

Range validation is enforced before execution: `temperatureMilli` `0..=100000`,
`topK` `0..=1000000`, `topPMilli` `0..=1000`, and `seed` is any `u64`. Only
decoder models accept sampling; submitting one for a classifier fails instead
of being silently ignored. Every node and verifier must still load the same
model file. If you change the model's default sampling for a published model,
treat it as a new model version.

## Binary weights (`.vnnw`)

Large models are not practical as inline JSON. A model JSON may point at a
binary tensor file:

```json
"weightsFile": "distilgpt2.vnnw",
"weightsSha256": "…64 hex characters…"
```

The file layout is:

| Offset | Size | Meaning |
|---|---|---|
| 0 | 4 | magic `VNNW` |
| 4 | 4 | format version (`1`), little-endian `u32` |
| 8 | 8 | payload length in bytes, little-endian `u64` |
| 16 | 32 | SHA-256 of the payload |
| 48 | n | Q16.16 tensors as little-endian `i32` |

Tensors are stored in the same order as the JSON fields: embedding,
positional, then each decoder layer (layernorm, Q/K/V/O, FFN), then the final
layernorm and optional untied head. `weightsSha256` is mandatory when
`weightsFile` is set. The loader verifies the magic, version, length, payload
hash and declared `weightsSha256` before using any tensor; a single tampered
byte is rejected. A model may use either inline tensors or `weightsFile`, not
both: any non-empty inline tensor field is rejected.

Binary payloads are memory-mapped and the tensors borrow from the mapping, so
loading DistilGPT-2 does not copy the 328 MB payload into `Vec<i32>`. The
loader still verifies the magic, version, length and both hashes before any
tensor is used. Mapped tensors require a 4-byte-aligned payload and a
little-endian host; if either check fails, the loader falls back to an owned
copy instead of reinterpreting the bytes unsafely. If the platform cannot map
the file at all, the loader reads it into an owned buffer and performs the same
header and hash checks. The mapping is read-only, and model files must not be
modified in place while a node is running. Writes go to a temporary file in the
same directory, are fsynced, and are renamed into place so replacing a weight
file does not truncate a live mapping.

## Tokenizers

### Char-BPE (synthetic model)

- Start from the sorted character alphabet.
- Merge the most frequent adjacent pairs in order.
- Encoding applies the learned merges in order.
- The Python trainer and Rust runtime use the same merge order.

### GPT-2 byte-level BPE (real models)

- `vocab.json` and `merges.txt` come from the checkpoint.
- The standard GPT-2 ByteLevel regex pre-tokenizes the prompt.
- Each matched piece is mapped through the GPT-2 byte-to-Unicode table.
- Adjacent pairs are merged by rank (lowest rank first).
- Decoding reverses the byte mapping and skips special tokens.
- The Python exporter and Rust runtime produce the same token ids; the parity
  tests check the reference prompts.

The byte-level tokenizer is GPT-2 only. There is no LLaMA/SentencePiece support
and no arbitrary tokenizer file format.

## Parity and determinism

- **`tiny-gpt-v1`:** greedy token parity with the float PyTorch reference on
  8 prompts; KV cache equals full forward.
- **`tiny-stories-1m`:** prompt tokenizer parity on 8 prompts, **32-token greedy
  parity on all 8 prompts**, and KV cache equals full forward.
- **`distilgpt2`:** binary-weight load, prompt tokenizer parity and **16-token
  greedy parity on 3 prompts** against the official float checkpoint, exercised
  at 1,024-token context. The opt-in test is:
  `VNN_TEST_DISTILGPT2=1 cargo test -p verinoo-runtime distilgpt2_binary`.
- **Sampling:** deterministic across repeated runs; temperature `0` equals
  greedy, and top-k `1` with a positive temperature equals greedy.
- Repeated generation is byte-identical.
- The real models required four runtime details to match their references:
  1. GPT-2 sqrt attention scaling, or none for GPT-Neo;
  2. `gelu_new` (tanh approximation), not ReLU;
  3. layer-norm epsilon 1e-5 and rounded normalization;
  4. dense layers that accumulate in Q32.32 and round once at the end.
- The synthetic model remains a small demo; its output is often template-like
  or garbled. `tiny-stories-1m` generates simple children's stories.
  `distilgpt2` generates recognisably fluent English, but it is still a small
  2019-era model.

**Performance caveat:** on the development Mac, release mode loads the 328 MB
binary and generates 48 tokens in 2.62 s (about 40 ms/token, including load).
The public 1-CPU VPS is materially slower. A 32-token generation can exceed a
1 s consensus block, so DistilGPT-2 is not enabled on the live node. GPU
execution or an optimistic-execution path is required before serving it.

## Model format

| Field | Meaning |
|---|---|
| `kind` | must be `gpt-decoder` |
| `tokenizerKind` | `char-bpe` or `byte-bpe` |
| `vocab`, `merges` | token strings in id order; ordered BPE merges |
| `maxContext`, `maxNewTokens` | context and generation limits |
| `hidden`, `heads`, `ff`, `layers` | shape metadata |
| `activation` | `relu` or `gelu_new` |
| `attentionScale` | `sqrt` (GPT-2-style) or `none` (GPT-Neo) |
| `layerNormEpsilon` | Q32.32 epsilon added to the variance |
| `tieWordEmbeddings` | reuse the token embedding as the output head |
| `prependBos` | whether to prepend `bosId` |
| `bosId`, `eosId`, `padId`, `unkId` | special token ids |
| `sampling` | temperature, top-k, top-p and seed |
| `weightsFile`, `weightsSha256` | optional binary tensor file and its required payload hash |
| `embedding`, `positional` | Q16.16 token and position embeddings (inline mode) |
| `decoder[]` | per layer: layernorm, Q/K/V/O, FFN |
| `lnfGamma`, `lnfBeta` | final layernorm |
| `headWeights`, `headBias` | output head (empty when tied) |

The runtime validates every tensor length, special-token mapping, vocabulary
uniqueness, head dimension, context limit and sampling range.

`result_json` returns:

```json
{
  "generatedText": "...",
  "prompt": "...",
  "promptTokens": [7454, 2402, 257, 640],
  "tokenIds": [11, 612, ...],
  "modelId": "tiny-stories-1m",
  "weightsHash": "4d9ffafb..."
}
```

The receipt binds `modelId`, `weightsHash` and the effective `sampling`
configuration; `/v1/verify` regenerates the text with the same sampling and
compares the output hash. A verifier must load the same model file.

## Import or train

```bash
# generic GPT-2/GPT-Neo importer: HuggingFace model id or local directory
python3 scripts/import_hf_gpt.py --model distilgpt2 --skip-generation
python3 scripts/import_hf_gpt.py --model roneneldan/TinyStories-1M \
  --max-context 256 --out-name tiny-stories-1m
# models/<out-name>.json, models/<out-name>.vnnw,
# models/<out-name>.test.json (unless --skip-generation)

# Apache 2.0 DistilGPT-2 convenience wrapper: same CLI shape and outputs
python3 scripts/import_distilgpt2.py
# models/distilgpt2.json (1.7 MB), models/distilgpt2.vnnw (328 MB),
# models/distilgpt2.test.json

# real TinyStories-1M checkpoint (downloads weights, not redistributed)
python3 scripts/import_tiny_stories.py
# models/tiny-stories-1m.json (25 MB), models/tiny-stories-1m.test.json

# synthetic demo model
python3 scripts/train_gpt.py
# models/tiny-gpt-v1.json, models/tiny-gpt-v1.test.json
```

The generic importer is config-driven and covers the same decoder subset as the
runtime:

- **GPT-2** (`GPT2LMHeadModel` or model type `gpt2`): Conv1D weights are
  `[in, out]` and are transposed to the runtime's `[out, in]` layout; attention
  uses sqrt scaling.
- **GPT-Neo** (`GPTNeoForCausalLM` or model type `gpt_neo`): Linear weights are
  already `[out, in]` and are not transposed; attention has no score scaling.
- Tied and untied `lm_head`; an untied head gets the required bias (zeros when
  the checkpoint has none).
- `gelu_new` or `relu` activation only, and the GPT-2 byte-level BPE tokenizer
  (`vocab.json` + `merges.txt`).
- Attention scaling configs outside the runtime subset are rejected instead of
  importing logits that differ: GPT-2 `scale_attn_weights=False`,
  `scale_attn_by_inverse_layer_idx=True`, `add_cross_attention=True`, and
  GPT-Neo configs that expose `scale_attn_weights=True`.

`--max-context` defaults to `min(1024, config positions)` and is capped by the
model's positional embeddings. GPT-Neo local-attention checkpoints are rejected
when the requested context exceeds their window because the runtime treats
local layers as global attention; choose `--max-context` at or below the
window. LLaMA/SentencePiece tokenizers, non-`gelu_new`/`relu` activations,
rotary embeddings, other architectures, and shapes outside the runtime limits
fail with a clear error.

Model weight files are not committed. `models/*.vnnw` is gitignored, so both
the specific outputs above and generic `models/<name>.vnnw` imports are never
committed or redistributed. The small metadata JSON (tokenizer and sampling
config) and optional float-reference fixtures may be committed. The importers
quantise the checkpoint to Q16.16, write the `.vnnw` header and SHA-256, extract
the GPT-2 tokenizer and optionally record float-reference greedy vectors.
Weights are downloaded from HuggingFace or read from a local directory; they
are not redistributed.

**Licensing:** DistilGPT-2 is Apache 2.0 according to its model card; verify the
card before redistribution. The model files are downloaded locally and are not
committed or included in release artifacts. The TinyStories-1M model card does
not state a license for the weights; they are also downloaded locally, not
redistributed, and `models/NOTICE-tiny-stories-1m.txt` records the caveat. The
generic importer does not change the checkpoint's license: verify the model card
before using or redistributing any imported weights.

## What is still missing

- **Production LLM serving.** 82M parameters and 1,024-token context are a
  foundation, not a service. There is no GPU executor, batching, streaming,
  safety layer, request queue or large-model (7B+) support, and DistilGPT-2 is
  not on the public node.
- **Per-task generation controls.** Temperature, top-k, top-p and seed can be
  chosen per task and are bound in the receipt. There is no stop sequence,
  repetition penalty or beam search, and sampling is only accepted for decoder
  models.
- **Arbitrary checkpoints.** The generic importer covers GPT-2/GPT-Neo shapes
  and the GPT-2 byte-BPE tokenizer, but there is no LLaMA/SentencePiece support,
  no rotary embeddings, no arbitrary tokenizer formats and no ONNX LLM import.
- **Long-context quality.** The runtime supports up to 8,192 positions of
  validation, but only DistilGPT-2's 1,024 positions are tested; local attention
  beyond GPT-Neo's window is not implemented.
- **Continuation parity for very long generations.** Parity is measured on
  16–32 tokens; small numerical differences may flip an argmax later.

## Why this matters

The chain executes a real pretrained 82M-parameter GPT-2 decoder with a
1,024-token context, the GPT-2 byte-level BPE tokenizer, causal attention, a KV
cache, deterministic sampling and a SHA-256-bound binary weight file. Inputs are
encrypted, the result is consensus state, and the receipt binds the exact
weights. That is the runtime shape an LLM needs. It is not an LLM product yet.
