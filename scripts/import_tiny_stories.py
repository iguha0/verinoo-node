#!/usr/bin/env python3
"""Import TinyStories-1M into the Verinoo Q16.16 decoder format.

TinyStories-1M is a real 3.75M-parameter GPT-Neo decoder trained on simple
children's stories. It uses the GPT-2 byte-level BPE tokenizer, `gelu_new`,
pre-norm layers, tied embeddings and alternating global/local attention. This
import keeps the first 128 positions, so the local window (256) never matters.

    python3 scripts/import_tiny_stories.py
    python3 scripts/import_tiny_stories.py --out-dir models --max-context 128

Outputs:
  models/tiny-stories-1m.json       quantised decoder (runtime input)
  models/tiny-stories-1m.test.json  float-reference tokenizer + greedy vectors
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "roneneldan/TinyStories-1M"
MAX_NEW_TOKENS = 32
Q = 65536
PROMPTS = [
    "Once upon a time",
    "The little dog",
    "Lily found a",
    "One day, a cat",
    "Tom was happy",
    "The sun was",
    "Mia and her mom",
    "A small bird",
]


def quantise(value: float) -> int:
    scaled = value * Q
    rounded = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    if abs(value) >= 32768:
        raise ValueError(f"weight {value} is outside the Q16.16 real range")
    return int(rounded)


def flatten(tensor: torch.Tensor) -> list[int]:
    return [quantise(float(x)) for x in tensor.detach().cpu().reshape(-1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--max-context", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()
    config = model.config
    hidden = config.hidden_size
    heads = config.num_heads
    layers = config.num_layers
    ff = config.intermediate_size or hidden * 4
    max_context = min(args.max_context, config.max_position_embeddings)

    vocab_path = hf_hub_download(MODEL, "vocab.json")
    merges_path = hf_hub_download(MODEL, "merges.txt")
    vocab_map = json.load(open(vocab_path))
    vocab: list[str | None] = [None] * len(vocab_map)
    for token, idx in vocab_map.items():
        vocab[idx] = token
    if any(tok is None for tok in vocab):
        raise ValueError("vocab.json has holes")
    merges: list[list[str]] = []
    for line in Path(merges_path).read_text().splitlines():
        if not line or line.startswith("#version"):
            continue
        a, b = line.split(" ", 1)
        merges.append([a, b])
    print(f"tokenizer: {len(vocab)} tokens, {len(merges)} merges", flush=True)

    state = model.state_dict()
    wte = state["transformer.wte.weight"]
    wpe = state["transformer.wpe.weight"][:max_context]
    spec = {
        "kind": "gpt-decoder",
        "modelId": "tiny-stories-1m",
        "architecture": f"gpt-neo-{layers}x{hidden}-{heads}h",
        "description": (
            "TinyStories-1M: a 3.75M-parameter GPT-Neo decoder with the GPT-2 "
            "byte-level BPE tokenizer, gelu_new and tied embeddings. "
            "Demonstration model, not production LLM support."
        ),
        "tokenizerKind": "byte-bpe",
        "activation": "gelu_new",
        "attentionScale": "none",
        "layerNormEpsilon": quantise_eps(config.layer_norm_epsilon),
        "tieWordEmbeddings": bool(config.tie_word_embeddings),
        "prependBos": False,
        "vocab": vocab,
        "merges": merges,
        "maxContext": max_context,
        "maxNewTokens": args.max_new_tokens,
        "hidden": hidden,
        "heads": heads,
        "ff": ff,
        "layers": layers,
        "bosId": tokenizer.bos_token_id,
        "eosId": tokenizer.eos_token_id,
        "padId": tokenizer.eos_token_id,
        "unkId": tokenizer.unk_token_id,
        "embedding": flatten(wte),
        "positional": flatten(wpe),
        "decoder": [],
        "lnfGamma": flatten(state["transformer.ln_f.weight"]),
        "lnfBeta": flatten(state["transformer.ln_f.bias"]),
        "headWeights": [],
        "headBias": [],
    }

    for i in range(layers):
        prefix = f"transformer.h.{i}"
        attn = f"{prefix}.attn.attention"
        q = state[f"{attn}.q_proj.weight"]
        k = state[f"{attn}.k_proj.weight"]
        v = state[f"{attn}.v_proj.weight"]
        o = state[f"{attn}.out_proj.weight"]
        o_bias = state[f"{attn}.out_proj.bias"]
        zeros = torch.zeros(hidden)
        spec["decoder"].append(
            {
                "ln1Gamma": flatten(state[f"{prefix}.ln_1.weight"]),
                "ln1Beta": flatten(state[f"{prefix}.ln_1.bias"]),
                "qWeights": flatten(q),
                "qBias": flatten(zeros),
                "kWeights": flatten(k),
                "kBias": flatten(zeros),
                "vWeights": flatten(v),
                "vBias": flatten(zeros),
                "oWeights": flatten(o),
                "oBias": flatten(o_bias),
                "ln2Gamma": flatten(state[f"{prefix}.ln_2.weight"]),
                "ln2Beta": flatten(state[f"{prefix}.ln_2.bias"]),
                "ff1Weights": flatten(state[f"{prefix}.mlp.c_fc.weight"]),
                "ff1Bias": flatten(state[f"{prefix}.mlp.c_fc.bias"]),
                "ff2Weights": flatten(state[f"{prefix}.mlp.c_proj.weight"]),
                "ff2Bias": flatten(state[f"{prefix}.mlp.c_proj.bias"]),
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "tiny-stories-1m.json"
    model_path.write_text(json.dumps(spec) + "\n")
    print(f"wrote {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")

    cases = []
    with torch.no_grad():
        for prompt in PROMPTS:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long)
            output = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated = output[0, input_ids.shape[1] :].tolist()
            text = tokenizer.decode(generated, skip_special_tokens=True)
            cases.append(
                {
                    "prompt": prompt,
                    "promptTokens": prompt_ids,
                    "tokenIds": generated,
                    "text": text,
                }
            )
            print(f"{prompt!r} -> {text!r}  tokens {generated}", flush=True)

    fixture = {
        "modelId": "tiny-stories-1m",
        "note": (
            "Float-reference tokenizer and greedy generations from the "
            "official TinyStories-1M checkpoint. The Q16.16 runtime must match "
            "the prompt tokens and generated token ids."
        ),
        "cases": cases,
    }
    fixture_path = out_dir / "tiny-stories-1m.test.json"
    fixture_path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {fixture_path}")


def quantise_eps(value: float) -> int:
    # LayerNorm epsilon is added to a Q32.32 variance, not a Q16.16 weight.
    return int(round(value * (1 << 32)))


if __name__ == "__main__":
    main()
