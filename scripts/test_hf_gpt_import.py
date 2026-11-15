#!/usr/bin/env python3
"""Offline unit tests for the shared GPT-2/GPT-Neo import logic."""

from __future__ import annotations

import hashlib
import math
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hf_gpt_common import (
    ModelShape,
    TokenizerData,
    UnsupportedModelError,
    UnsupportedTokenizerError,
    build_metadata,
    default_model_name,
    expected_parameter_count,
    extract_arrays,
    load_tokenizer_data,
    orient_dense_weight,
    quantise,
    quantise_epsilon,
    resolve_max_context,
    shape_from_config,
    validate_activation,
    validate_local_attention,
    write_weights,
)

VOCAB = 8
HIDDEN = 4
HEADS = 2
FF = 8
POSITIONS = 16
CONTEXT = 8


def make_shape(family: str = "gpt2", **overrides) -> ModelShape:
    values = {
        "family": family,
        "hidden": HIDDEN,
        "heads": HEADS,
        "layers": 1,
        "ff": FF,
        "max_positions": POSITIONS,
        "activation": "gelu_new",
        "attention_scale": "sqrt" if family == "gpt2" else "none",
        "layer_norm_epsilon": 1e-5,
        "tie_word_embeddings": True,
        "window_size": None,
        "attention_types": (),
    }
    values.update(overrides)
    return ModelShape(**values)


def make_state(
    family: str,
    *,
    hidden: int = HIDDEN,
    layers: int = 1,
    ff: int = FF,
    vocab: int = VOCAB,
    positions: int = POSITIONS,
    tied: bool = True,
) -> dict[str, torch.Tensor]:
    counter = [0]

    def tensor(*shape: int) -> torch.Tensor:
        size = math.prod(shape)
        values = torch.arange(
            counter[0], counter[0] + size, dtype=torch.float32
        )
        counter[0] += size
        return values.reshape(shape) * 0.001 - 0.5

    state: dict[str, torch.Tensor] = {
        "transformer.wte.weight": tensor(vocab, hidden),
        "transformer.wpe.weight": tensor(positions, hidden),
        "transformer.ln_f.weight": tensor(hidden),
        "transformer.ln_f.bias": tensor(hidden),
    }
    for layer in range(layers):
        prefix = f"transformer.h.{layer}"
        state[f"{prefix}.ln_1.weight"] = tensor(hidden)
        state[f"{prefix}.ln_1.bias"] = tensor(hidden)
        state[f"{prefix}.ln_2.weight"] = tensor(hidden)
        state[f"{prefix}.ln_2.bias"] = tensor(hidden)
        if family == "gpt2":
            attn = f"{prefix}.attn"
            mlp = f"{prefix}.mlp"
            state[f"{attn}.c_attn.weight"] = tensor(hidden, 3 * hidden)
            state[f"{attn}.c_attn.bias"] = tensor(3 * hidden)
            state[f"{attn}.c_proj.weight"] = tensor(hidden, hidden)
            state[f"{attn}.c_proj.bias"] = tensor(hidden)
            state[f"{mlp}.c_fc.weight"] = tensor(hidden, ff)
            state[f"{mlp}.c_fc.bias"] = tensor(ff)
            state[f"{mlp}.c_proj.weight"] = tensor(ff, hidden)
            state[f"{mlp}.c_proj.bias"] = tensor(hidden)
        else:
            attn = f"{prefix}.attn.attention"
            mlp = f"{prefix}.mlp"
            for name in ("q_proj", "k_proj", "v_proj"):
                state[f"{attn}.{name}.weight"] = tensor(hidden, hidden)
            state[f"{attn}.out_proj.weight"] = tensor(hidden, hidden)
            state[f"{attn}.out_proj.bias"] = tensor(hidden)
            state[f"{mlp}.c_fc.weight"] = tensor(ff, hidden)
            state[f"{mlp}.c_fc.bias"] = tensor(ff)
            state[f"{mlp}.c_proj.weight"] = tensor(hidden, ff)
            state[f"{mlp}.c_proj.bias"] = tensor(hidden)
    if tied:
        state["lm_head.weight"] = state["transformer.wte.weight"]
    else:
        state["lm_head.weight"] = tensor(vocab, hidden)
    return state


def make_tokenizer() -> TokenizerData:
    vocab = ["<eos>", "a", "b", "c", "d", "e", "f", "g"]
    return TokenizerData(
        vocab=vocab,
        merges=[["a", "b"]],
        bos_id=0,
        eos_id=0,
        pad_id=0,
        unk_id=0,
        prepend_bos=False,
    )


class OrientationTests(unittest.TestCase):
    def test_conv1d_is_transposed_and_linear_is_not(self):
        weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        self.assertTrue(torch.equal(orient_dense_weight(weight, "gpt2"), weight.T))
        self.assertTrue(torch.equal(orient_dense_weight(weight, "gpt_neo"), weight))

    def test_gpt2_attention_is_split_then_transposed(self):
        shape = make_shape(
            "gpt2", hidden=2, heads=1, ff=3, layers=1, max_positions=4
        )
        state = make_state(
            "gpt2", hidden=2, layers=1, ff=3, vocab=4, positions=4
        )
        arrays = extract_arrays(state, shape, 2)
        c_attn = state["transformer.h.0.attn.c_attn.weight"]
        expected_q = quantise(c_attn[:, 0:2].T.contiguous())
        expected_k = quantise(c_attn[:, 2:4].T.contiguous())
        expected_v = quantise(c_attn[:, 4:6].T.contiguous())
        self.assertEqual(arrays[4].tolist(), expected_q.tolist())
        self.assertEqual(arrays[6].tolist(), expected_k.tolist())
        self.assertEqual(arrays[8].tolist(), expected_v.tolist())

    def test_gpt_neo_linear_weights_keep_their_orientation(self):
        shape = make_shape("gpt_neo")
        state = make_state("gpt_neo")
        arrays = extract_arrays(state, shape, CONTEXT)
        q = state["transformer.h.0.attn.attention.q_proj.weight"]
        self.assertEqual(arrays[4].tolist(), quantise(q).tolist())


class HeadTests(unittest.TestCase):
    def test_tied_head_has_no_head_tensors(self):
        shape = make_shape("gpt2", layers=2)
        state = make_state("gpt2", layers=2, tied=True)
        arrays = extract_arrays(state, shape, CONTEXT)
        self.assertEqual(len(arrays), 2 + 16 * 2 + 2)
        self.assertEqual(
            sum(array.size for array in arrays),
            expected_parameter_count(shape, VOCAB, CONTEXT),
        )
        self.assertEqual(
            expected_parameter_count(shape, VOCAB, CONTEXT),
            VOCAB * HIDDEN
            + CONTEXT * HIDDEN
            + 2 * (4 * HIDDEN**2 + 9 * HIDDEN + 2 * FF * HIDDEN + FF)
            + 2 * HIDDEN,
        )

    def test_untied_head_appends_weights_and_zero_bias(self):
        shape = make_shape("gpt2", layers=2, tie_word_embeddings=False)
        state = make_state("gpt2", layers=2, tied=False)
        arrays = extract_arrays(state, shape, CONTEXT)
        self.assertEqual(len(arrays), 2 + 16 * 2 + 4)
        self.assertEqual(
            sum(array.size for array in arrays),
            expected_parameter_count(shape, VOCAB, CONTEXT),
        )
        self.assertEqual(
            arrays[-2].tolist(),
            quantise(state["lm_head.weight"]).tolist(),
        )
        self.assertEqual(arrays[-1].tolist(), [0] * VOCAB)

    def test_untied_gpt_neo_head_also_appends_zero_bias(self):
        shape = make_shape(
            "gpt_neo", layers=1, tie_word_embeddings=False
        )
        state = make_state("gpt_neo", layers=1, tied=False)
        arrays = extract_arrays(state, shape, CONTEXT)
        self.assertEqual(
            sum(array.size for array in arrays),
            expected_parameter_count(shape, VOCAB, CONTEXT),
        )
        self.assertEqual(arrays[-1].tolist(), [0] * VOCAB)


class ValidationTests(unittest.TestCase):
    def test_activation_rejects_unsupported_functions(self):
        for activation in ("gelu", "gelu_fast", "silu", "swiglu"):
            with self.assertRaises(UnsupportedModelError):
                validate_activation(activation)
        self.assertEqual(validate_activation("gelu_new"), "gelu_new")
        self.assertEqual(validate_activation("relu"), "relu")

    def test_sentencepiece_tokenizer_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "tokenizer.model").write_bytes(b"fake sentencepiece")
            with self.assertRaises(UnsupportedTokenizerError) as caught:
                load_tokenizer_data(str(directory))
            self.assertIn("SentencePiece", str(caught.exception))

    def test_missing_byte_bpe_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UnsupportedTokenizerError) as caught:
                load_tokenizer_data(tmp)
            self.assertIn("vocab.json", str(caught.exception))

    def test_local_attention_window_smaller_than_context_is_rejected(self):
        shape = make_shape(
            "gpt_neo",
            window_size=4,
            attention_types=((["global", "local"], 2),),
        )
        with self.assertRaises(UnsupportedModelError) as caught:
            validate_local_attention(shape, 5)
        self.assertIn("local attention window", str(caught.exception))
        validate_local_attention(shape, 4)

    def test_global_only_attention_ignores_window(self):
        shape = make_shape(
            "gpt_neo",
            window_size=4,
            attention_types=((["global", "global"], 2),),
        )
        validate_local_attention(shape, CONTEXT)

    def test_default_context_is_min_1024_and_positions(self):
        long_shape = make_shape(max_positions=2048)
        short_shape = make_shape(max_positions=512)
        self.assertEqual(resolve_max_context(long_shape, None), 1024)
        self.assertEqual(resolve_max_context(short_shape, None), 512)

    def test_default_output_name(self):
        self.assertEqual(default_model_name("distilgpt2"), "distilgpt2")
        self.assertEqual(
            default_model_name("roneneldan/TinyStories-1M"), "tinystories-1m"
        )
        self.assertEqual(default_model_name("/tmp/My-Model"), "my-model")

    def test_shape_from_config_for_gpt2_and_gpt_neo(self):
        gpt2 = types.SimpleNamespace(
            model_type="gpt2",
            architectures=["GPT2LMHeadModel"],
            n_embd=8,
            n_head=2,
            n_layer=2,
            n_inner=None,
            n_positions=16,
            activation_function="gelu_new",
            layer_norm_epsilon=1e-5,
            tie_word_embeddings=True,
        )
        shape = shape_from_config(gpt2)
        self.assertEqual(shape.family, "gpt2")
        self.assertEqual(shape.ff, 32)
        self.assertEqual(shape.attention_scale, "sqrt")
        self.assertTrue(shape.tie_word_embeddings)

        gpt_neo = types.SimpleNamespace(
            model_type="gpt_neo",
            architectures=["GPTNeoForCausalLM"],
            hidden_size=8,
            num_heads=2,
            num_layers=1,
            intermediate_size=None,
            max_position_embeddings=16,
            activation_function="relu",
            layer_norm_epsilon=1e-5,
            tie_word_embeddings=False,
            window_size=4,
            attention_types=[[["global", "local"], 1]],
        )
        shape = shape_from_config(gpt_neo)
        self.assertEqual(shape.family, "gpt_neo")
        self.assertEqual(shape.ff, 32)
        self.assertEqual(shape.attention_scale, "none")
        self.assertEqual(shape.activation, "relu")
        self.assertFalse(shape.tie_word_embeddings)
        self.assertEqual(shape.window_size, 4)

    def test_llama_architecture_is_rejected(self):
        llama = types.SimpleNamespace(
            model_type="llama", architectures=["LlamaForCausalLM"]
        )
        with self.assertRaises(UnsupportedModelError) as caught:
            shape_from_config(llama)
        self.assertIn("LLaMA", str(caught.exception))


class MetadataAndBinaryTests(unittest.TestCase):
    def test_metadata_fields_and_tensor_lengths(self):
        shape = make_shape(
            "gpt2", hidden=HIDDEN, layers=2, tie_word_embeddings=False
        )
        tokenizer = make_tokenizer()
        metadata = build_metadata(
            model_id="synthetic-gpt2",
            architecture="gpt2-2x4-2h",
            description="synthetic fixture",
            shape=shape,
            tokenizer=tokenizer,
            max_context=CONTEXT,
            max_new_tokens=4,
            weights_file="synthetic-gpt2.vnnw",
            weights_sha256="ab" * 32,
        )
        self.assertEqual(
            list(metadata),
            [
                "kind",
                "modelId",
                "architecture",
                "description",
                "tokenizerKind",
                "activation",
                "attentionScale",
                "layerNormEpsilon",
                "tieWordEmbeddings",
                "prependBos",
                "vocab",
                "merges",
                "maxContext",
                "maxNewTokens",
                "hidden",
                "heads",
                "ff",
                "layers",
                "bosId",
                "eosId",
                "padId",
                "unkId",
                "sampling",
                "weightsFile",
                "weightsSha256",
            ],
        )
        self.assertEqual(metadata["kind"], "gpt-decoder")
        self.assertEqual(metadata["modelId"], "synthetic-gpt2")
        self.assertEqual(metadata["architecture"], "gpt2-2x4-2h")
        self.assertEqual(metadata["tokenizerKind"], "byte-bpe")
        self.assertEqual(metadata["activation"], "gelu_new")
        self.assertEqual(metadata["attentionScale"], "sqrt")
        self.assertEqual(
            metadata["layerNormEpsilon"], quantise_epsilon(1e-5)
        )
        self.assertFalse(metadata["tieWordEmbeddings"])
        self.assertFalse(metadata["prependBos"])
        self.assertEqual(metadata["vocab"], tokenizer.vocab)
        self.assertEqual(metadata["merges"], tokenizer.merges)
        self.assertEqual(metadata["maxContext"], CONTEXT)
        self.assertEqual(metadata["maxNewTokens"], 4)
        self.assertEqual(metadata["hidden"], HIDDEN)
        self.assertEqual(metadata["heads"], HEADS)
        self.assertEqual(metadata["ff"], FF)
        self.assertEqual(metadata["layers"], 2)
        self.assertEqual(metadata["sampling"]["temperatureMilli"], 0)
        self.assertEqual(metadata["weightsFile"], "synthetic-gpt2.vnnw")
        self.assertEqual(metadata["weightsSha256"], "ab" * 32)

        state = make_state("gpt2", layers=2, tied=False)
        arrays = extract_arrays(state, shape, CONTEXT)
        lengths = [array.size for array in arrays]
        per_layer = 4 * HIDDEN**2 + 9 * HIDDEN + 2 * FF * HIDDEN + FF
        expected_lengths = [VOCAB * HIDDEN, CONTEXT * HIDDEN]
        for _ in range(2):
            expected_lengths.extend(
                [
                    HIDDEN,
                    HIDDEN,
                    HIDDEN**2,
                    HIDDEN,
                    HIDDEN**2,
                    HIDDEN,
                    HIDDEN**2,
                    HIDDEN,
                    HIDDEN**2,
                    HIDDEN,
                    HIDDEN,
                    HIDDEN,
                    FF * HIDDEN,
                    FF,
                    HIDDEN * FF,
                    HIDDEN,
                ]
            )
        expected_lengths.extend([HIDDEN, HIDDEN, VOCAB * HIDDEN, VOCAB])
        self.assertEqual(lengths, expected_lengths)
        self.assertEqual(sum(lengths), expected_parameter_count(shape, VOCAB, CONTEXT))
        self.assertEqual(
            per_layer,
            4 * HIDDEN**2 + 9 * HIDDEN + 2 * FF * HIDDEN + FF,
        )

    def test_binary_header_and_payload_hash(self):
        arrays = [
            np.array([1, 2, 3], dtype="<i4"),
            np.array([-4, 5], dtype="<i4"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.vnnw"
            digest_hex = write_weights(path, arrays)
            raw = path.read_bytes()
        magic, version, length, digest = struct.unpack("<4sIQ32s", raw[:48])
        payload = raw[48:]
        self.assertEqual(magic, b"VNNW")
        self.assertEqual(version, 1)
        self.assertEqual(length, len(payload))
        self.assertEqual(length, 20)
        self.assertEqual(digest, hashlib.sha256(payload).digest())
        self.assertEqual(digest_hex, digest.hex())


if __name__ == "__main__":
    unittest.main()
