#!/usr/bin/env python3
"""Import a GPT-2/GPT-Neo checkpoint from HuggingFace into Verinoo format.

This is a generic importer for the decoder subset the Verinoo runtime actually
implements: GPT-2 (`GPT2LMHeadModel`) and GPT-Neo (`GPTNeoForCausalLM`) with the
GPT-2 byte-level BPE tokenizer. It is not arbitrary LLM checkpoint support;
LLaMA/SentencePiece, rotary embeddings and other architectures fail with a
clear error.

    python3 scripts/import_hf_gpt.py --model distilgpt2
    python3 scripts/import_hf_gpt.py --model roneneldan/TinyStories-1M \
        --max-context 256 --max-new-tokens 16
    python3 scripts/import_hf_gpt.py --model ./local-checkpoint --out-name my-gpt2

Outputs (gitignored model files, weights downloaded not redistributed):
  models/<out-name>.json       metadata + tokenizer + sampling
  models/<out-name>.vnnw       Q16.16 weights with the VNNW header and SHA-256
  models/<out-name>.test.json  float-reference greedy vectors (optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from hf_gpt_common import (
    UnsupportedModelError,
    UnsupportedTokenizerError,
    default_model_name,
    export_checkpoint,
    load_tokenizer_data,
    resolve_max_context,
    resolve_max_new_tokens,
    shape_from_config,
    validate_local_attention,
    write_generation_fixture,
)

PROMPTS = [
    "Once upon a time",
    "The little dog",
    "In a world where",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a GPT-2/GPT-Neo HuggingFace checkpoint for the Verinoo decoder"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model id or local directory")
    parser.add_argument("--out-name", default=None, help="output file stem")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="context length (default: min(1024, config positions))",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    out_name = args.out_name or default_model_name(args.model)
    if Path(out_name).name != out_name:
        parser.error("--out-name must be a file stem without path separators")

    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    shape = shape_from_config(config)
    max_context = resolve_max_context(shape, args.max_context)
    validate_local_attention(shape, max_context)
    max_new_tokens = resolve_max_new_tokens(args.max_new_tokens, max_context)

    tokenizer_data = load_tokenizer_data(args.model, revision=args.revision)
    print(
        f"tokenizer: {len(tokenizer_data.vocab)} tokens, "
        f"{len(tokenizer_data.merges)} merges",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision
    )

    family = "GPT-2" if shape.family == "gpt2" else "GPT-Neo"
    architecture = (
        f"{shape.family}-{shape.layers}x{shape.hidden}-{shape.heads}h"
    )
    description = (
        f"{args.model}: a {family} causal decoder imported with the GPT-2 "
        "byte-level BPE tokenizer. Weights are loaded locally and not redistributed."
    )
    out_dir = Path(args.out_dir)
    export_checkpoint(
        state=model.state_dict(),
        shape=shape,
        tokenizer=tokenizer_data,
        model_id=out_name,
        architecture=architecture,
        description=description,
        out_dir=out_dir,
        out_name=out_name,
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
        out_dir / f"{out_name}.test.json",
        model_id=out_name,
        max_new_tokens=max_new_tokens,
        note=(
            f"Float-reference greedy generation from the HuggingFace checkpoint "
            f"`{args.model}`."
        ),
    )


if __name__ == "__main__":
    try:
        main()
    except (UnsupportedModelError, UnsupportedTokenizerError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None
