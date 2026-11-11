#!/usr/bin/env python3
"""Cross-check the importer against a file written by the official ONNX library.

The Rust tests build their fixtures with a hand-written encoder, so they prove
the converter works against *our* understanding of the format. This script
closes that gap: it takes the committed reference classifier, exports it as a
real `.onnx` with `onnx.helper`, runs `onnx-import`, and checks that the
converted graph comes back with byte-identical integer parameters.

Needs: pip install onnx numpy

    cargo build -p verinoo-onnx --bin onnx-import
    python3 crates/onnx/verify_official.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "models" / "acme-support-v1.graph.json"
BIN = ROOT / "target" / "debug" / "onnx-import"


def export_reference(tmp: Path) -> Path:
    """Export the committed graph as Gemm + Softmax, the layout torch.onnx uses."""
    graph = json.loads(SOURCE.read_text())
    layer = graph["layers"][0]
    features = graph["features"]
    classes = len(graph["labels"])

    weights = np.array(layer["weights"], dtype=np.int32).reshape(classes, features)
    bias = np.array(layer["bias"], dtype=np.int32)
    # Q16.16 -> f32 exactly the way the importer will quantise it back.
    w = (weights.astype(np.float64) / 65536.0).astype(np.float32)
    b = (bias.astype(np.float64) / 65536.0).astype(np.float32)

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, features])
    y = helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, classes])
    nodes = [
        helper.make_node("Gemm", ["X", "W", "B"], ["logits"], name="fc", transB=1),
        helper.make_node("Softmax", ["logits"], ["probs"], name="prob", axis=1),
    ]
    graph_proto = helper.make_graph(
        nodes,
        "acme-support-v1",
        [x],
        [y],
        initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")],
    )
    model = helper.make_model(
        graph_proto,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="verinoo-verify",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    out = tmp / "acme-support-v1.onnx"
    onnx.save(model, out)
    return out


def main() -> int:
    if not BIN.exists():
        print(f"missing {BIN}; run: cargo build -p verinoo-onnx --bin onnx-import")
        return 1
    source = json.loads(SOURCE.read_text())
    labels = ",".join(source["labels"])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        onnx_path = export_reference(tmp)
        print(f"wrote {onnx_path} ({onnx_path.stat().st_size} bytes, onnx {onnx.__version__})")

        imported = tmp / "imported.graph.json"
        result = subprocess.run(
            [
                str(BIN),
                "--onnx",
                str(onnx_path),
                "--model-id",
                "acme-onnx-v1",
                "--labels",
                labels,
                "--description",
                "imported from an official-library ONNX export",
                "--out",
                str(imported),
                "--check",
                "64",
            ],
            capture_output=True,
            text=True,
        )
        print(result.stdout.strip())
        if result.returncode != 0:
            print(result.stderr.strip(), file=sys.stderr)
            return result.returncode

        converted = json.loads(imported.read_text())

    original_layer = source["layers"][0]
    converted_layer = converted["layers"][0]
    checks = {
        "features": converted["features"] == source["features"],
        "labels": converted["labels"] == source["labels"],
        "architecture": converted_layer["op"] == "dense"
        and converted_layer["input"] == original_layer["input"]
        and converted_layer["output"] == original_layer["output"],
        "weights round-trip exactly": converted_layer["weights"] == original_layer["weights"],
        "bias round-trip exactly": converted_layer["bias"] == original_layer["bias"],
        "softmax tail": converted["layers"][-1]["op"] == "softmax",
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
