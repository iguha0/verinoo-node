#!/usr/bin/env python3
"""Import DistilGPT-2 into the Verinoo Q16.16 decoder format.

DistilGPT-2 is a licensed (Apache 2.0) 82M-parameter GPT-2 distillation with
the GPT-2 byte-level BPE tokenizer. Weights are written to a binary `.vnnw`
file; the JSON file carries only metadata, tokenizer and sampling config.

    python3 scripts/import_distilgpt2.py
    python3 scripts/import_distilgpt2.py --max-context 1024 --max-new-tokens 16

Outputs (both gitignored):
  models/distilgpt2.json       metadata + tokenizer + sampling
  models/distilgpt2.vnnw       82M Q16.16 parameters (328 MB)
  models/distilgpt2.test.json  float-reference greedy vectors (optional)

The model is too large for the current single-CPU consensus demo; it is the
foundation for a GPU/optimistic-execution deployment. Do not run it on the
public verinoo.com node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "distilgpt2"
Q = 65536
MAGIC = b"VNNW"
VERSION = 1
PROMPTS = [
    "Once upon a time",
    "The little dog",
    "In a world where",
]


def quantise(tensor: torch.Tensor) -> np.ndarray:
    values = tensor.detach().cpu().numpy().astype(np.float64)
    if np.max(np.abs(values)) >= 32768:
        raise ValueError("weight outside the Q16.16 real range")
    scaled = values * Q
    rounded = np.where(scaled >= 0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    return rounded.astype("<i4").reshape(-1)


def write_weights(path: Path, arrays: list[np.ndarray]) -> str:
    payload = np.concatenate(arrays).astype("<i4").tobytes()
    digest = hashlib.sha256(payload).digest()
    header = struct.pack("<4sIQ32s", MAGIC, VERSION, len(payload), digest)
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(payload)
    return digest.hex()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()
    config = model.config
    hidden = config.n_embd
    heads = config.n_head
    layers = config.n_layer
    ff = config.n_inner or hidden * 4
    max_context = min(args.max_context, config.n_positions)
    if hidden % heads:
        raise ValueError("n_embd must be divisible by n_head")

    vocab_path = hf_hub_download(MODEL, "vocab.json")
    merges_path = hf_hub_download(MODEL, "merges.txt")
    vocab_map = json.load(open(vocab_path))
    vocab: list[str | None] = [None] * len(vocab_map)
    for token, idx in vocab_map.items():
        vocab[idx] = token
    if any(tok is None for tok in vocab):
        raise ValueError("vocab.json has holes")
    merges = []
    for line in Path(merges_path).read_text().splitlines():
        if not line or line.startswith("#version"):
            continue
        a, b = line.split(" ", 1)
        merges.append([a, b])
    print(f"tokenizer: {len(vocab)} tokens, {len(merges)} merges", flush=True)

    state = model.state_dict()
    arrays: list[np.ndarray] = []
    arrays.append(quantise(state["transformer.wte.weight"]))
    arrays.append(quantise(state["transformer.wpe.weight"][:max_context]))
    for i in range(layers):
        p = f"transformer.h.{i}"
        attn = f"{p}.attn"
        mlp = f"{p}.mlp"
        arrays.append(quantise(state[f"{p}.ln_1.weight"]))
        arrays.append(quantise(state[f"{p}.ln_1.bias"]))
        # GPT-2 Conv1D: c_attn.weight is [in, 3*out]; our dense is [out, in].
        c_attn = state[f"{attn}.c_attn.weight"]
        c_bias = state[f"{attn}.c_attn.bias"]
        for part in range(3):
            block = c_attn[:, part * hidden : (part + 1) * hidden].T.contiguous()
            arrays.append(quantise(block))
            arrays.append(quantise(c_bias[part * hidden : (part + 1) * hidden]))
        arrays.append(quantise(state[f"{attn}.c_proj.weight"].T.contiguous()))
        arrays.append(quantise(state[f"{attn}.c_proj.bias"]))
        arrays.append(quantise(state[f"{p}.ln_2.weight"]))
        arrays.append(quantise(state[f"{p}.ln_2.bias"]))
        arrays.append(quantise(state[f"{mlp}.c_fc.weight"].T.contiguous()))
        arrays.append(quantise(state[f"{mlp}.c_fc.bias"]))
        arrays.append(quantise(state[f"{mlp}.c_proj.weight"].T.contiguous()))
        arrays.append(quantise(state[f"{mlp}.c_proj.bias"]))
    arrays.append(quantise(state["transformer.ln_f.weight"]))
    arrays.append(quantise(state["transformer.ln_f.bias"]))
    tied = bool(config.tie_word_embeddings)
    if not tied:
        arrays.append(quantise(state["lm_head.weight"]))
        arrays.append(np.zeros(len(vocab), dtype="<i4"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "distilgpt2.vnnw"
    payload_sha = write_weights(weights_path, arrays)
    parameters = int(sum(arr.size for arr in arrays))
    print(f"wrote {weights_path} ({weights_path.stat().st_size / 1e6:.1f} MB, {parameters} params)")

    spec = {
        "kind": "gpt-decoder",
        "modelId": "distilgpt2",
        "architecture": f"gpt2-{layers}x{hidden}-{heads}h",
        "description": (
            "DistilGPT-2, an Apache 2.0 82M-parameter GPT-2 decoder. "
            "Demonstration/production foundation; not served on the public node."
        ),
        "tokenizerKind": "byte-bpe",
        "activation": "gelu_new",
        "attentionScale": "sqrt",
        "layerNormEpsilon": int(round(config.layer_norm_epsilon * (1 << 32))),
        "tieWordEmbeddings": tied,
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
        "sampling": {
            "temperatureMilli": 0,
            "topK": 0,
            "topPMilli": 0,
            "seed": 0,
        },
        "weightsFile": "distilgpt2.vnnw",
        "weightsSha256": payload_sha,
    }
    metadata_path = out_dir / "distilgpt2.json"
    metadata_path.write_text(json.dumps(spec) + "\n")
    print(f"wrote {metadata_path} ({metadata_path.stat().st_size / 1e6:.2f} MB)")

    if args.skip_generation:
        print("skipping float-reference generation")
        return

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
            print(f"{prompt!r} -> {text!r}", flush=True)
    fixture = {
        "modelId": "distilgpt2",
        "note": "Float-reference greedy generation from the official DistilGPT-2 checkpoint.",
        "cases": cases,
    }
    (out_dir / "distilgpt2.test.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {out_dir / 'distilgpt2.test.json'}")


if __name__ == "__main__":
    main()
