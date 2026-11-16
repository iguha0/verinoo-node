#!/usr/bin/env python3
"""Shared import helpers for GPT-2/GPT-Neo checkpoints.

The Verinoo decoder format supports GPT-2 and GPT-Neo decoder shapes with the
GPT-2 byte-level BPE tokenizer. This module owns the Q16.16 quantisation,
tensor ordering, metadata shape and float-reference generation so the specific
DistilGPT-2 importer and the generic HuggingFace importer stay in sync.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from huggingface_hub import hf_hub_download

Q = 65536
MAGIC = b"VNNW"
VERSION = 1
RUNTIME_MAX_CONTEXT = 8192
RUNTIME_MAX_HIDDEN = 2048
RUNTIME_MAX_FF = RUNTIME_MAX_HIDDEN * 8
RUNTIME_MAX_VOCAB = 1_000_000
SENTENCEPIECE_FILES = ("tokenizer.model", "sentencepiece.bpe.model", "spiece.model")
DEFAULT_SAMPLING = {
    "temperatureMilli": 0,
    "topK": 0,
    "topPMilli": 0,
    "seed": 0,
}


class UnsupportedModelError(ValueError):
    """The checkpoint is outside the GPT-2/GPT-Neo runtime subset."""


class UnsupportedTokenizerError(ValueError):
    """The checkpoint does not use the GPT-2 byte-level BPE tokenizer."""


@dataclass(frozen=True)
class ModelShape:
    family: str
    hidden: int
    heads: int
    layers: int
    ff: int
    max_positions: int
    activation: str
    attention_scale: str
    layer_norm_epsilon: float
    tie_word_embeddings: bool
    window_size: int | None = None
    attention_types: tuple[Any, ...] = ()


@dataclass(frozen=True)
class TokenizerData:
    vocab: list[str]
    merges: list[list[str]]
    bos_id: int
    eos_id: int
    pad_id: int
    unk_id: int
    prepend_bos: bool


def quantise(tensor: torch.Tensor) -> np.ndarray:
    values = tensor.detach().cpu().float().numpy().astype(np.float64)
    if values.size == 0:
        return np.zeros(0, dtype="<i4")
    if np.max(np.abs(values)) >= 32768:
        raise ValueError("weight outside the Q16.16 real range")
    scaled = values * Q
    rounded = np.where(scaled >= 0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    return rounded.astype("<i4").reshape(-1)


def quantise_epsilon(value: float) -> int:
    return int(round(value * (1 << 32)))


def write_weights(path: Path, arrays: Sequence[np.ndarray]) -> str:
    payload = np.concatenate(arrays).astype("<i4").tobytes()
    digest = hashlib.sha256(payload).digest()
    header = struct.pack("<4sIQ32s", MAGIC, VERSION, len(payload), digest)
    path = Path(path)
    temporary = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with temporary as handle:
            handle.write(header)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary.name, path)
    except BaseException:
        try:
            os.unlink(temporary.name)
        except FileNotFoundError:
            pass
        raise
    return digest.hex()


def default_model_name(model: str) -> str:
    name = Path(model).name
    if name.endswith(".git"):
        name = name[:-4]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._").lower()
    if not slug:
        raise ValueError(f"cannot derive an output name from `{model}`; pass --out-name")
    return slug


def architecture_name(shape: ModelShape) -> str:
    return f"{shape.family}-{shape.layers}x{shape.hidden}-{shape.heads}h"


def describe_model(model: str, shape: ModelShape) -> str:
    family = "GPT-2" if shape.family == "gpt2" else "GPT-Neo"
    return (
        f"{model}: a {family} causal decoder imported with the GPT-2 "
        "byte-level BPE tokenizer. Weights are loaded locally and not redistributed."
    )


def detect_family(config: Any) -> str:
    model_type = str(getattr(config, "model_type", "") or "").lower()
    architectures = " ".join(
        str(item) for item in (getattr(config, "architectures", None) or [])
    )
    if model_type == "gpt_neo" or "GPTNeoForCausalLM" in architectures:
        return "gpt_neo"
    if model_type == "gpt2" or "GPT2LMHeadModel" in architectures:
        return "gpt2"
    detail = f"model_type={model_type!r}"
    if architectures:
        detail += f", architectures={architectures!r}"
    raise UnsupportedModelError(
        f"unsupported model architecture ({detail}); this importer supports GPT-2 "
        "(GPT2LMHeadModel / model_type `gpt2`) and GPT-Neo "
        "(GPTNeoForCausalLM / model_type `gpt_neo`) with the GPT-2 byte-level "
        "BPE tokenizer. LLaMA/SentencePiece, rotary embeddings and other "
        "architectures are not supported."
    )


def validate_activation(value: Any) -> str:
    activation = str(value or "gelu_new").lower()
    if activation in ("gelu_new", "relu"):
        return activation
    raise UnsupportedModelError(
        f"unsupported activation `{value}`; the Verinoo decoder supports only "
        "`gelu_new` and `relu` (not `gelu`, `gelu_fast`, `silu`/SiLU, `swiglu`, ...)."
    )


def _validate_attention_config(config: Any, family: str) -> None:
    scale_weights = getattr(config, "scale_attn_weights", None)
    if scale_weights is not None:
        if family == "gpt2" and not bool(scale_weights):
            raise UnsupportedModelError(
                "unsupported config scale_attn_weights=False; the Verinoo GPT-2 "
                "decoder always applies sqrt attention scaling"
            )
        if family == "gpt_neo" and bool(scale_weights):
            raise UnsupportedModelError(
                "unsupported config scale_attn_weights=True for GPT-Neo; the "
                "Verinoo GPT-Neo decoder applies no attention-score scaling"
            )
    if bool(getattr(config, "scale_attn_by_inverse_layer_idx", False)):
        raise UnsupportedModelError(
            "unsupported config scale_attn_by_inverse_layer_idx=True; the "
            "Verinoo decoder does not rescale attention by layer index"
        )
    if bool(getattr(config, "add_cross_attention", False)):
        raise UnsupportedModelError(
            "unsupported config add_cross_attention=True; the Verinoo decoder "
            "does not implement cross-attention"
        )


def _require_int(config: Any, names: Iterable[str], label: str) -> int:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                return number
    raise UnsupportedModelError(f"config is missing a usable {label}")


def shape_from_config(config: Any) -> ModelShape:
    family = detect_family(config)
    _validate_attention_config(config, family)
    if family == "gpt2":
        hidden = _require_int(config, ("n_embd", "hidden_size"), "hidden size")
        heads = _require_int(config, ("n_head", "num_heads"), "head count")
        layers = _require_int(config, ("n_layer", "num_layers"), "layer count")
        max_positions = _require_int(
            config, ("n_positions", "max_position_embeddings"), "position count"
        )
        ff = getattr(config, "n_inner", None) or hidden * 4
    else:
        hidden = _require_int(config, ("hidden_size", "n_embd"), "hidden size")
        heads = _require_int(config, ("num_heads", "n_head"), "head count")
        layers = _require_int(config, ("num_layers", "n_layer"), "layer count")
        max_positions = _require_int(
            config, ("max_position_embeddings", "n_positions"), "position count"
        )
        ff = getattr(config, "intermediate_size", None) or hidden * 4
    ff = int(ff)
    if ff <= 0:
        raise UnsupportedModelError("config has an invalid feed-forward width")
    if hidden % heads:
        raise UnsupportedModelError(
            f"hidden size {hidden} is not divisible by head count {heads}"
        )
    epsilon = getattr(config, "layer_norm_epsilon", None)
    if epsilon is None:
        raise UnsupportedModelError("config is missing layer_norm_epsilon")
    window_size = None
    attention_types: tuple[Any, ...] = ()
    if family == "gpt_neo":
        raw_window = getattr(config, "window_size", None)
        if raw_window is not None:
            try:
                window_size = int(raw_window)
            except (TypeError, ValueError):
                window_size = None
        raw_types = getattr(config, "attention_types", None)
        if raw_types:
            attention_types = tuple(raw_types)
    return ModelShape(
        family=family,
        hidden=hidden,
        heads=heads,
        layers=layers,
        ff=ff,
        max_positions=max_positions,
        activation=validate_activation(getattr(config, "activation_function", None)),
        attention_scale="sqrt" if family == "gpt2" else "none",
        layer_norm_epsilon=float(epsilon),
        tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        window_size=window_size,
        attention_types=attention_types,
    )


def resolve_max_context(shape: ModelShape, requested: int | None) -> int:
    if requested is None:
        context = min(1024, shape.max_positions)
    else:
        requested = int(requested)
        if requested <= 0:
            raise ValueError("--max-context must be positive")
        context = min(requested, shape.max_positions)
        if requested > shape.max_positions:
            print(
                f"warning: --max-context {requested} exceeds {shape.max_positions} "
                f"positional embeddings; using {context}",
                flush=True,
            )
    if context > RUNTIME_MAX_CONTEXT:
        raise UnsupportedModelError(
            f"maxContext {context} exceeds the runtime limit of {RUNTIME_MAX_CONTEXT}"
        )
    return context


def resolve_max_new_tokens(requested: int, max_context: int) -> int:
    value = int(requested)
    if value <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if value > max_context:
        print(
            f"warning: --max-new-tokens {value} exceeds maxContext {max_context}; "
            f"using {max_context}",
            flush=True,
        )
        value = max_context
    return value


def normalise_attention_types(attention_types: Any) -> list[str]:
    if not attention_types:
        return []
    if isinstance(attention_types, str):
        return [attention_types]
    result: list[str] = []
    if all(isinstance(item, str) for item in attention_types):
        return [str(item) for item in attention_types]
    for entry in attention_types:
        if isinstance(entry, str):
            result.append(entry)
        elif isinstance(entry, (list, tuple)):
            if (
                len(entry) == 2
                and isinstance(entry[0], (list, tuple))
                and isinstance(entry[1], int)
            ):
                pattern, count = entry
                result.extend(str(item) for item in list(pattern) * count)
            else:
                result.extend(str(item) for item in entry)
    return result


def has_local_attention(shape: ModelShape) -> bool:
    return any(
        item.lower() == "local" for item in normalise_attention_types(shape.attention_types)
    )


def validate_local_attention(shape: ModelShape, max_context: int) -> None:
    if not has_local_attention(shape):
        return
    window = shape.window_size
    if window is None or window <= 0:
        raise UnsupportedModelError(
            "model uses GPT-Neo local attention but has no usable window_size; "
            "the Verinoo runtime treats local layers as global attention"
        )
    if window < max_context:
        raise UnsupportedModelError(
            f"max-context {max_context} exceeds the GPT-Neo local attention window "
            f"{window}; the Verinoo runtime treats local layers as global attention, "
            f"so this checkpoint would produce different tokens. Use --max-context "
            f"{window} or less."
        )


def validate_runtime_limits(
    shape: ModelShape, vocab_size: int, max_context: int, max_new_tokens: int
) -> None:
    if shape.hidden > RUNTIME_MAX_HIDDEN:
        raise UnsupportedModelError(
            f"hidden size {shape.hidden} exceeds the runtime limit of {RUNTIME_MAX_HIDDEN}"
        )
    if shape.ff > RUNTIME_MAX_FF:
        raise UnsupportedModelError(
            f"feed-forward width {shape.ff} exceeds the runtime limit of {RUNTIME_MAX_FF}"
        )
    if not 0 < vocab_size <= RUNTIME_MAX_VOCAB:
        raise UnsupportedModelError(
            f"vocabulary size {vocab_size} is outside the runtime range"
        )
    if not 0 < max_context <= RUNTIME_MAX_CONTEXT:
        raise UnsupportedModelError(
            f"maxContext {max_context} is outside the runtime range"
        )
    if not 0 < max_new_tokens <= max_context:
        raise ValueError("maxNewTokens must be between 1 and maxContext")


def read_vocab_merges(vocab_path: Path, merges_path: Path) -> tuple[list[str], list[list[str]]]:
    try:
        vocab_map = json.loads(Path(vocab_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UnsupportedTokenizerError(
            f"could not read GPT-2 vocab.json from {vocab_path}: {exc}"
        ) from exc
    if not isinstance(vocab_map, dict) or not vocab_map:
        raise UnsupportedTokenizerError(f"{vocab_path} is not a GPT-2 vocab.json map")
    vocab: list[str | None] = [None] * (max(int(index) for index in vocab_map.values()) + 1)
    for token, index in vocab_map.items():
        vocab[int(index)] = token
    if any(token is None for token in vocab):
        raise UnsupportedTokenizerError("vocab.json has holes in the id sequence")
    merges: list[list[str]] = []
    try:
        lines = Path(merges_path).read_text().splitlines()
    except OSError as exc:
        raise UnsupportedTokenizerError(
            f"could not read GPT-2 merges.txt from {merges_path}: {exc}"
        ) from exc
    for line in lines:
        if not line or line.startswith("#version"):
            continue
        parts = line.split(" ", 1)
        if len(parts) != 2:
            raise UnsupportedTokenizerError(f"malformed merge line: {line!r}")
        merges.append([parts[0], parts[1]])
    return [str(token) for token in vocab], merges


def tokenizer_file_paths(model: str, revision: str | None = None) -> tuple[Path, Path]:
    local = Path(model)
    if local.is_dir():
        vocab_path = local / "vocab.json"
        merges_path = local / "merges.txt"
        if vocab_path.is_file() and merges_path.is_file():
            return vocab_path, merges_path
        sentencepiece = [
            name for name in SENTENCEPIECE_FILES if (local / name).is_file()
        ]
        if sentencepiece:
            raise UnsupportedTokenizerError(
                f"`{model}` contains a SentencePiece tokenizer ({', '.join(sentencepiece)}) "
                "but no GPT-2 vocab.json/merges.txt; LLaMA/SentencePiece tokenizers "
                "are not supported by the Verinoo decoder."
            )
        raise UnsupportedTokenizerError(
            f"`{model}` does not contain GPT-2 byte-level BPE tokenizer files "
            "(vocab.json and merges.txt); SentencePiece/LLaMA-style tokenizers are "
            "not supported."
        )
    try:
        vocab_path = Path(hf_hub_download(model, "vocab.json", revision=revision))
        merges_path = Path(hf_hub_download(model, "merges.txt", revision=revision))
        return vocab_path, merges_path
    except Exception as exc:
        sentencepiece = []
        for name in SENTENCEPIECE_FILES:
            try:
                hf_hub_download(model, name, revision=revision)
                sentencepiece.append(name)
            except Exception:
                continue
        if sentencepiece:
            raise UnsupportedTokenizerError(
                f"`{model}` uses a SentencePiece tokenizer ({', '.join(sentencepiece)}) "
                "instead of GPT-2 byte-level BPE; LLaMA/SentencePiece tokenizers are "
                "not supported by the Verinoo decoder."
            ) from exc
        raise UnsupportedTokenizerError(
            f"could not find GPT-2 byte-level BPE tokenizer files (vocab.json and "
            f"merges.txt) for `{model}`. The Verinoo decoder supports only the GPT-2 "
            "byte-level BPE tokenizer; SentencePiece/LLaMA-style tokenizers are not "
            "supported."
        ) from exc


def determine_prepend_bos(tokenizer: Any) -> bool:
    if bool(getattr(tokenizer, "add_bos_token", False)):
        return True
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        return False
    try:
        with_special = tokenizer.encode("hello", add_special_tokens=True)
        without_special = tokenizer.encode("hello", add_special_tokens=False)
    except Exception:
        return False
    return bool(
        with_special
        and with_special != without_special
        and with_special[0] == bos_id
    )


def load_tokenizer_data(model: str, revision: str | None = None) -> TokenizerData:
    vocab_path, merges_path = tokenizer_file_paths(model, revision)
    vocab, merges = read_vocab_merges(vocab_path, merges_path)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise UnsupportedTokenizerError(
            f"tokenizer for `{model}` has no EOS token id; cannot build a decoder spec"
        )
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else eos_id
    for name, value in (
        ("bos", bos_id),
        ("eos", eos_id),
        ("pad", pad_id),
        ("unk", unk_id),
    ):
        if not 0 <= int(value) < len(vocab):
            raise UnsupportedTokenizerError(
                f"tokenizer {name} id {value} is outside the vocabulary of {len(vocab)}"
            )
    return TokenizerData(
        vocab=vocab,
        merges=merges,
        bos_id=int(bos_id),
        eos_id=int(eos_id),
        pad_id=int(pad_id),
        unk_id=int(unk_id),
        prepend_bos=determine_prepend_bos(tokenizer),
    )


def orient_dense_weight(weight: torch.Tensor, family: str) -> torch.Tensor:
    if family == "gpt2":
        return weight.T.contiguous()
    if family == "gpt_neo":
        return weight.contiguous()
    raise UnsupportedModelError(f"unsupported weight orientation for `{family}`")


def _tensor(
    state: dict[str, torch.Tensor], key: str, *expected: int
) -> torch.Tensor:
    value = state.get(key)
    if value is None:
        raise UnsupportedModelError(f"checkpoint is missing tensor `{key}`")
    if expected and tuple(value.shape) != tuple(expected):
        raise UnsupportedModelError(
            f"tensor `{key}` has shape {tuple(value.shape)}, expected {tuple(expected)}"
        )
    return value


def _bias_or_zeros(
    state: dict[str, torch.Tensor], key: str, length: int
) -> torch.Tensor:
    value = state.get(key)
    if value is None:
        return torch.zeros(length, dtype=torch.float32)
    if tuple(value.shape) != (length,):
        raise UnsupportedModelError(
            f"tensor `{key}` has shape {tuple(value.shape)}, expected {(length,)}"
        )
    return value


def _gpt2_layer_arrays(
    state: dict[str, torch.Tensor], layer: int, hidden: int, ff: int
) -> list[np.ndarray]:
    prefix = f"transformer.h.{layer}"
    attn = f"{prefix}.attn"
    mlp = f"{prefix}.mlp"
    arrays = [
        quantise(_tensor(state, f"{prefix}.ln_1.weight", hidden)),
        quantise(_tensor(state, f"{prefix}.ln_1.bias", hidden)),
    ]
    c_attn = _tensor(state, f"{attn}.c_attn.weight", hidden, 3 * hidden)
    c_bias = _tensor(state, f"{attn}.c_attn.bias", 3 * hidden)
    for part in range(3):
        block = c_attn[:, part * hidden : (part + 1) * hidden].T.contiguous()
        arrays.append(quantise(block))
        arrays.append(quantise(c_bias[part * hidden : (part + 1) * hidden]))
    arrays.append(
        quantise(
            orient_dense_weight(
                _tensor(state, f"{attn}.c_proj.weight", hidden, hidden), "gpt2"
            )
        )
    )
    arrays.append(quantise(_tensor(state, f"{attn}.c_proj.bias", hidden)))
    arrays.append(quantise(_tensor(state, f"{prefix}.ln_2.weight", hidden)))
    arrays.append(quantise(_tensor(state, f"{prefix}.ln_2.bias", hidden)))
    arrays.append(
        quantise(
            orient_dense_weight(
                _tensor(state, f"{mlp}.c_fc.weight", hidden, ff), "gpt2"
            )
        )
    )
    arrays.append(quantise(_tensor(state, f"{mlp}.c_fc.bias", ff)))
    arrays.append(
        quantise(
            orient_dense_weight(
                _tensor(state, f"{mlp}.c_proj.weight", ff, hidden), "gpt2"
            )
        )
    )
    arrays.append(quantise(_tensor(state, f"{mlp}.c_proj.bias", hidden)))
    return arrays


def _gpt_neo_layer_arrays(
    state: dict[str, torch.Tensor], layer: int, hidden: int, ff: int
) -> list[np.ndarray]:
    prefix = f"transformer.h.{layer}"
    attn = f"{prefix}.attn.attention"
    mlp = f"{prefix}.mlp"
    arrays = [
        quantise(_tensor(state, f"{prefix}.ln_1.weight", hidden)),
        quantise(_tensor(state, f"{prefix}.ln_1.bias", hidden)),
    ]
    for name in ("q_proj", "k_proj", "v_proj"):
        arrays.append(
            quantise(
                orient_dense_weight(
                    _tensor(state, f"{attn}.{name}.weight", hidden, hidden), "gpt_neo"
                )
            )
        )
        arrays.append(
            quantise(_bias_or_zeros(state, f"{attn}.{name}.bias", hidden))
        )
    arrays.append(
        quantise(
            orient_dense_weight(
                _tensor(state, f"{attn}.out_proj.weight", hidden, hidden), "gpt_neo"
            )
        )
    )
    arrays.append(
        quantise(_bias_or_zeros(state, f"{attn}.out_proj.bias", hidden))
    )
    arrays.append(quantise(_tensor(state, f"{prefix}.ln_2.weight", hidden)))
    arrays.append(quantise(_tensor(state, f"{prefix}.ln_2.bias", hidden)))
    arrays.append(
        quantise(
            orient_dense_weight(
                _tensor(state, f"{mlp}.c_fc.weight", ff, hidden), "gpt_neo"
            )
        )
    )
    arrays.append(quantise(_tensor(state, f"{mlp}.c_fc.bias", ff)))
    arrays.append(
        quantise(
            orient_dense_weight(
                _tensor(state, f"{mlp}.c_proj.weight", hidden, ff), "gpt_neo"
            )
        )
    )
    arrays.append(quantise(_tensor(state, f"{mlp}.c_proj.bias", hidden)))
    return arrays


def extract_arrays(
    state: dict[str, torch.Tensor], shape: ModelShape, max_context: int
) -> list[np.ndarray]:
    hidden = shape.hidden
    embedding = _tensor(state, "transformer.wte.weight")
    if embedding.ndim != 2 or int(embedding.shape[1]) != hidden:
        raise UnsupportedModelError(
            f"tensor `transformer.wte.weight` has shape {tuple(embedding.shape)}, "
            f"expected [vocab, {hidden}]"
        )
    vocab_size = int(embedding.shape[0])
    positional = _tensor(state, "transformer.wpe.weight")
    if positional.ndim != 2 or int(positional.shape[1]) != hidden:
        raise UnsupportedModelError(
            f"tensor `transformer.wpe.weight` has shape {tuple(positional.shape)}, "
            f"expected [positions, {hidden}]"
        )
    if int(positional.shape[0]) < max_context:
        raise UnsupportedModelError(
            f"maxContext {max_context} exceeds positional embeddings "
            f"({int(positional.shape[0])})"
        )
    arrays = [
        quantise(embedding),
        quantise(positional[:max_context]),
    ]
    for layer in range(shape.layers):
        if shape.family == "gpt2":
            arrays.extend(_gpt2_layer_arrays(state, layer, hidden, shape.ff))
        else:
            arrays.extend(_gpt_neo_layer_arrays(state, layer, hidden, shape.ff))
    arrays.append(quantise(_tensor(state, "transformer.ln_f.weight", hidden)))
    arrays.append(quantise(_tensor(state, "transformer.ln_f.bias", hidden)))
    if not shape.tie_word_embeddings:
        arrays.append(
            quantise(_tensor(state, "lm_head.weight", vocab_size, hidden))
        )
        arrays.append(
            quantise(_bias_or_zeros(state, "lm_head.bias", vocab_size))
        )
    return arrays


def expected_parameter_count(
    shape: ModelShape, vocab_size: int, max_context: int
) -> int:
    hidden = shape.hidden
    per_layer = (
        4 * hidden * hidden
        + 9 * hidden
        + 2 * shape.ff * hidden
        + shape.ff
    )
    total = (
        vocab_size * hidden
        + max_context * hidden
        + shape.layers * per_layer
        + 2 * hidden
    )
    if not shape.tie_word_embeddings:
        total += vocab_size * hidden + vocab_size
    return total


def build_metadata(
    *,
    model_id: str,
    architecture: str,
    description: str,
    shape: ModelShape,
    tokenizer: TokenizerData,
    max_context: int,
    max_new_tokens: int,
    weights_file: str,
    weights_sha256: str,
) -> dict[str, Any]:
    return {
        "kind": "gpt-decoder",
        "modelId": model_id,
        "architecture": architecture,
        "description": description,
        "tokenizerKind": "byte-bpe",
        "activation": shape.activation,
        "attentionScale": shape.attention_scale,
        "layerNormEpsilon": quantise_epsilon(shape.layer_norm_epsilon),
        "tieWordEmbeddings": shape.tie_word_embeddings,
        "prependBos": tokenizer.prepend_bos,
        "vocab": tokenizer.vocab,
        "merges": tokenizer.merges,
        "maxContext": max_context,
        "maxNewTokens": max_new_tokens,
        "hidden": shape.hidden,
        "heads": shape.heads,
        "ff": shape.ff,
        "layers": shape.layers,
        "bosId": tokenizer.bos_id,
        "eosId": tokenizer.eos_id,
        "padId": tokenizer.pad_id,
        "unkId": tokenizer.unk_id,
        "sampling": dict(DEFAULT_SAMPLING),
        "weightsFile": weights_file,
        "weightsSha256": weights_sha256,
    }


def export_checkpoint(
    *,
    state: dict[str, torch.Tensor],
    shape: ModelShape,
    tokenizer: TokenizerData,
    model_id: str,
    architecture: str,
    description: str,
    out_dir: Path,
    out_name: str,
    max_context: int,
    max_new_tokens: int,
) -> tuple[Path, Path, str, int]:
    vocab_size = int(_tensor(state, "transformer.wte.weight").shape[0])
    if vocab_size != len(tokenizer.vocab):
        raise UnsupportedTokenizerError(
            f"embedding has {vocab_size} rows but the tokenizer has "
            f"{len(tokenizer.vocab)} tokens"
        )
    validate_runtime_limits(shape, vocab_size, max_context, max_new_tokens)
    arrays = extract_arrays(state, shape, max_context)
    parameters = int(sum(array.size for array in arrays))
    expected = expected_parameter_count(shape, vocab_size, max_context)
    if parameters != expected:
        raise UnsupportedModelError(
            f"serialised {parameters} parameters, expected {expected}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / f"{out_name}.vnnw"
    payload_sha = write_weights(weights_path, arrays)
    print(
        f"wrote {weights_path} "
        f"({weights_path.stat().st_size / 1e6:.1f} MB, {parameters} params)"
    )
    spec = build_metadata(
        model_id=model_id,
        architecture=architecture,
        description=description,
        shape=shape,
        tokenizer=tokenizer,
        max_context=max_context,
        max_new_tokens=max_new_tokens,
        weights_file=weights_path.name,
        weights_sha256=payload_sha,
    )
    metadata_path = out_dir / f"{out_name}.json"
    metadata_path.write_text(json.dumps(spec) + "\n")
    print(f"wrote {metadata_path} ({metadata_path.stat().st_size / 1e6:.2f} MB)")
    return weights_path, metadata_path, payload_sha, parameters


def write_generation_fixture(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    out_path: Path,
    *,
    model_id: str,
    max_new_tokens: int,
    note: str,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with torch.no_grad():
        for prompt in prompts:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long)
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
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
        "modelId": model_id,
        "note": note,
        "cases": cases,
    }
    out_path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {out_path}")
    return cases
