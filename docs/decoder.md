# Generative decoder runtime

Status: **first decoder slice implemented and running in consensus.** This is a
tiny GPT-style demonstration decoder, not production LLM support.

`crates/runtime/src/gpt.rs` executes a two-layer causal transformer with a
learned char-BPE tokenizer, a KV cache and greedy decoding. Every op is integer
or Q16.16, so two nodes generate the same token ids from the same prompt.

## What runs

`models/tiny-gpt-v1.json`:

| Field | Value |
|---|---|
| architecture | `gpt-decoder-2x64-4h` |
| parameters | 110,400 |
| hidden / heads / head_dim | 64 / 4 / 16 |
| decoder layers | 2 (pre-norm, causal attention, ReLU FFN width 128) |
| vocabulary | 320 char-BPE tokens |
| context | 32 tokens |
| generation | greedy argmax, at most 24 new tokens |
| training data | 3,847-line synthetic support corpus |

Generation:

```text
ids = [BOS] + BPE(prompt)
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

## BPE tokenizer

- Training starts from the sorted character alphabet of the corpus.
- The 273 most frequent adjacent pairs are merged in order.
- Encoding applies the learned merges in order to the prompt characters.
- Decoding concatenates token strings and skips `<pad>`, `<bos>`, `<eos>` and
  `<unk>`.
- The Python trainer and the Rust runtime use the same merge order. The parity
  test checks the encoded prompt tokens and the generated token ids.

The vocabulary is ASCII char-BPE trained on the synthetic corpus. It is not a
GPT-2/LLaMA BPE tokenizer and does not handle arbitrary Unicode vocabularies.

## Parity and determinism

- **Token-for-token greedy parity** with the float PyTorch reference on 8
  prompts (`models/tiny-gpt-v1.test.json`).
- **KV cache equals full forward**: prefill + one step produces exactly the
  same logits as a full-sequence forward pass.
- Repeated generation is byte-identical.
- Synthetic model quality: validation loss 0.799, token accuracy about 74%.
  Output is often template-like or garbled. This is a runtime demonstration,
  not a model-quality claim.

## Model format

The JSON file contains:

| Field | Meaning |
|---|---|
| `kind` | must be `gpt-decoder` |
| `vocab` | token strings in id order; must include the four special tokens |
| `merges` | ordered BPE merge pairs |
| `maxContext`, `maxNewTokens` | context and generation limits |
| `hidden`, `heads`, `ff`, `layers` | shape metadata |
| `bosId`, `eosId`, `padId`, `unkId` | special token ids |
| `embedding`, `positional` | Q16.16 token and position embeddings |
| `decoder[]` | per layer: layernorm, Q/K/V/O, FFN |
| `lnfGamma`, `lnfBeta` | final layernorm |
| `headWeights`, `headBias` | output head |

The runtime validates every tensor length, special-token mapping, vocabulary
uniqueness, head dimension and context limit.

`result_json` returns:

```json
{
  "generatedText": "...",
  "prompt": "...",
  "promptTokens": [85, 157],
  "tokenIds": [277, 8, ...],
  "modelId": "tiny-gpt-v1",
  "weightsHash": "19b8f15f..."
}
```

The receipt binds `modelId` and `weightsHash`; `/v1/verify` regenerates the
text and compares the output hash.

## Train and export

```bash
python3 scripts/train_gpt.py
# BPE: 320 tokens, 273 merges
# train 3462  val 385
# best val loss 0.7994
# wrote models/tiny-gpt-v1.json
# wrote models/tiny-gpt-v1.test.json
```

The script uses PyTorch on CPU, a deterministic seed, padding masks, dropout,
early stopping and a train/validation split. `--epochs`, `--seed`,
`--vocab-size` and `--out-dir` are configurable.

## What is still missing

- **Real LLM tokenizers.** No GPT-2/LLaMA BPE, SentencePiece, byte fallback or
  arbitrary Unicode vocabularies.
- **Sampling.** Greedy only. No temperature, top-k, top-p, beam search,
  repetition penalty or stop sequences.
- **Context and batching.** 32-token context, batch size one at inference, no
  paged attention or cache eviction.
- **Production model quality.** 110k parameters on a synthetic corpus. This is
  not a useful assistant.
- **Safety and operations.** No content filters, red-team evaluation, rate
  limits per token, streaming responses or long-output controls.
- **A real customer checkpoint.** The next step is importing a small real
  decoder (for example a tiny GPT-2) with a matching tokenizer and measuring
  parity.

## Why this matters

The chain now executes a causal decoder with a BPE tokenizer, a KV cache and
greedy generation during block application, with encrypted inputs and a
weights-bound receipt. That is the runtime shape an LLM needs. It is not an LLM
product yet.
