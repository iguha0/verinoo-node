# Generative decoder runtime

Status: **two decoder slices implemented and running in consensus.** This is not
production LLM support.

| Model | What it is | Source |
|---|---|---|
| `tiny-gpt-v1` | synthetic two-layer GPT-style decoder, char-BPE, 110,400 params | trained by `scripts/train_gpt.py` |
| `tiny-stories-1m` | real pretrained GPT-Neo decoder, GPT-2 byte-level BPE, 3.75M params | `roneneldan/TinyStories-1M` |

Both use causal attention, a per-layer KV cache and greedy decoding. Every op
is integer or Q16.16, so two nodes generate the same token ids from the same
prompt.

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

Generation:

```text
ids = [BOS if configured] + tokenize(prompt)
logits = prefill(ids, cache)
repeat up to maxNewTokens:
  next = argmax(logits)
  append next
  if next == EOS: stop
  logits = step(next, cache)
text = decode(generated ids)
```

`prefill` runs the prompt through every layer, fills the per-layer K/V cache,
and returns the last position's logits. Each `step` embeds one token, projects
Q/K/V, appends K/V to the cache, attends only to cached keys (causal), and runs
the FFN. No full-sequence recomputation per token.

## Tokenizers

### Char-BPE (synthetic model)

- Start from the sorted character alphabet.
- Merge the most frequent adjacent pairs in order.
- Encoding applies the learned merges in order.
- The Python trainer and Rust runtime use the same merge order.

### GPT-2 byte-level BPE (real model)

- `vocab.json` and `merges.txt` come from the checkpoint.
- The standard GPT-2 ByteLevel regex pre-tokenizes the prompt.
- Each matched piece is mapped through the GPT-2 byte-to-Unicode table.
- Adjacent pairs are merged by rank (lowest rank first).
- Decoding reverses the byte mapping and skips special tokens.
- The Python exporter and Rust runtime produce the same token ids; the parity
  test checks all eight reference prompts.

The byte-level tokenizer is GPT-2 only. There is no LLaMA/SentencePiece support
and no arbitrary tokenizer file format.

## Parity and determinism

- **`tiny-gpt-v1`:** greedy token parity with the float PyTorch reference on
  8 prompts; KV cache equals full forward.
- **`tiny-stories-1m`:** prompt tokenizer parity on 8 prompts, **32-token greedy
  parity on all 8 prompts**, and KV cache equals full forward.
- Repeated generation is byte-identical.
- The real model required four runtime details to match GPT-Neo:
  1. no attention-score scaling (`attentionScale: "none"`);
  2. `gelu_new` (tanh approximation), not ReLU;
  3. layer-norm epsilon 1e-5 and rounded normalization;
  4. dense layers that accumulate in Q32.32 and round once at the end.
- The synthetic model remains a small demo; its output is often template-like
  or garbled. `tiny-stories-1m` generates simple, mostly coherent children's
  stories.

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
| `embedding`, `positional` | Q16.16 token and position embeddings |
| `decoder[]` | per layer: layernorm, Q/K/V/O, FFN |
| `lnfGamma`, `lnfBeta` | final layernorm |
| `headWeights`, `headBias` | output head (empty when tied) |

The runtime validates every tensor length, special-token mapping, vocabulary
uniqueness, head dimension and context limit.

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

The receipt binds `modelId` and `weightsHash`; `/v1/verify` regenerates the
text and compares the output hash.

## Import or train

```bash
# real TinyStories-1M checkpoint (downloads weights, not redistributed)
python3 scripts/import_tiny_stories.py
# models/tiny-stories-1m.json (25 MB), models/tiny-stories-1m.test.json

# synthetic demo model
python3 scripts/train_gpt.py
# models/tiny-gpt-v1.json, models/tiny-gpt-v1.test.json
```

**Licensing:** the TinyStories-1M model card does not state a license for the
weights. They are not committed or included in release artifacts. The import
script downloads them from Hugging Face; review the model card and dataset
terms before use. See `models/NOTICE-tiny-stories-1m.txt`.

## What is still missing

- **Production LLM support.** 3.75M parameters, 128-token context, one GPT-2
  tokenizer, no sampling, batching, streaming, safety filters or long context.
- **Local attention beyond the window.** GPT-Neo's local layers are treated as
  global because the runtime context is 128 and the window is 256.
- **Tokenizers beyond GPT-2.** No LLaMA/SentencePiece, byte fallback or
  arbitrary tokenizer formats.
- **Sampling.** Greedy only. No temperature, top-k, top-p, beam search,
  repetition penalty or stop sequences.
- **Model quality and licensing.** The real demo is a small children's-story
  model with unresolved license metadata.
- **Continuation parity for very long generations.** Parity is measured on 32
  tokens; small numerical differences may flip an argmax later.

## Why this matters

The chain executes a real pretrained 3.75M-parameter GPT-Neo decoder with the
GPT-2 byte-level BPE tokenizer, causal attention, a KV cache and greedy
generation during block application. Inputs are encrypted, the result is
consensus state, and the receipt binds the exact weights. That is the runtime
shape an LLM needs. It is not an LLM product yet.
