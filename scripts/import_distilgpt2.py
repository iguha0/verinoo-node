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
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from hf_gpt_common import (
    export_checkpoint,
    load_tokenizer_data,
    resolve_max_context,
    resolve_max_new_tokens,
    shape_from_config,
    write_generation_fixture,
)

MODEL = "distilgpt2"
PROMPTS = [
    "Once upon a time",
    "The little dog",
    "In a world where",
]
DESCRIPTION = (
    "DistilGPT-2, an Apache 2.0 82M-parameter GPT-2 decoder. "
    "Demonstration/production foundation; not served on the public node."
)


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
    shape = shape_from_config(model.config)
    max_context = resolve_max_context(shape, args.max_context)
    max_new_tokens = resolve_max_new_tokens(args.max_new_tokens, max_context)
    tokenizer_data = load_tokenizer_data(MODEL)
    print(
        f"tokenizer: {len(tokenizer_data.vocab)} tokens, "
        f"{len(tokenizer_data.merges)} merges",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    export_checkpoint(
        state=model.state_dict(),
        shape=shape,
        tokenizer=tokenizer_data,
        model_id=MODEL,
        architecture=f"gpt2-{shape.layers}x{shape.hidden}-{shape.heads}h",
        description=DESCRIPTION,
        out_dir=out_dir,
        out_name=MODEL,
        max_context=max_context,
        max_new_tokens=max_new_tokens,
    )

    if args.skip_generation:
        print("skipping float-reference generation")
        return

    write_generation_fixture(
        model,
        tokenizer,
        PROMPTS,
        out_dir / "distilgpt2.test.json",
        model_id=MODEL,
        max_new_tokens=max_new_tokens,
        note="Float-reference greedy generation from the official DistilGPT-2 checkpoint.",
    )


if __name__ == "__main__":
    main()
