#!/usr/bin/env python3
"""Train the claims-triage demo classifier over the production featuriser.

This is the classifier shape a design partner actually runs: hashed unigrams
into a small linear model, a small label set, and a decision that later needs
to be defended. The exported graph executes in consensus.

    python3 scripts/train_claims_classifier.py
    python3 scripts/train_claims_classifier.py --features 2048 --out-dir models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ONE = 65536
LABELS = [
    "auto_collision",
    "property_damage",
    "theft",
    "medical_injury",
    "liability",
    "weather",
    "fraud_review",
    "subrogation",
]


def tokens(text: str) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            current.append(ch)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


def bucket(token: str, features: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % features


def counts(text: str, features: int) -> np.ndarray:
    vec = np.zeros(features, dtype=np.int64)
    for token in tokens(text):
        vec[bucket(token, features)] += 1
    return vec


def features_float(text: str, features: int) -> np.ndarray:
    vec = counts(text, features)
    total = int(vec.sum())
    if total == 0:
        return np.zeros(features, dtype=np.float64)
    return vec / total


def features_q16(text: str, features: int) -> list[int]:
    vec = counts(text, features)
    total = int(vec.sum())
    if total == 0:
        return [0] * features
    return [int(c * ONE // total) for c in vec]


def quantise(value: float) -> int:
    scaled = value * ONE
    rounded = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    if abs(value) >= 32768:
        raise ValueError(f"weight {value} outside Q16.16 real range")
    return int(rounded)


def flatten(matrix) -> list[int]:
    return [quantise(float(x)) for x in np.asarray(matrix).reshape(-1)]


def dense_q(x: list[int], weights: list[int], bias: list[int], inp: int, out: int) -> list[int]:
    result = []
    for j in range(out):
        acc = bias[j]
        row = j * inp
        for f in range(inp):
            acc += (x[f] * weights[row + f]) >> 16
        result.append(max(-(2**31), min(2**31 - 1, acc)))
    return result


def softmax_q(x: list[int]) -> list[int]:
    maximum = max(x)
    e = [0 if v - maximum < 0 else (v - maximum) + ONE for v in x]
    denom = max(1, sum(e))
    return [int((ei * ONE) // denom) for ei in e]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="crates/model/data/claims.jsonl")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--features", type=int, default=2048)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    print("classes", Counter(labels), flush=True)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=7, stratify=labels
    )
    x_train = np.stack([features_float(t, args.features) for t in train_texts])
    x_test = np.stack([features_float(t, args.features) for t in test_texts])
    clf = LogisticRegression(max_iter=5000, C=2.0)
    clf.fit(x_train, train_labels)
    train_acc = clf.score(x_train, train_labels)
    test_acc = clf.score(x_test, test_labels)
    print(f"train accuracy {train_acc:.4f}  test accuracy {test_acc:.4f}", flush=True)

    classes = list(clf.classes_)
    weights = [quantise(float(v)) for v in clf.coef_.reshape(-1)]
    bias = [quantise(float(v)) for v in clf.intercept_.reshape(-1)]
    # Reorder rows to LABELS if sklearn sorted them differently.
    order = [classes.index(label) for label in LABELS]
    reordered = []
    for row in order:
        reordered.extend(weights[row * args.features : (row + 1) * args.features])
    reordered_bias = [bias[i] for i in order]

    integer_correct = 0
    float_correct = 0
    parity = 0
    for text, truth in zip(test_texts, test_labels):
        q = features_q16(text, args.features)
        logits = dense_q(q, reordered, reordered_bias, args.features, len(LABELS))
        scores = softmax_q(logits)
        predicted = LABELS[int(np.argmax(scores))]
        float_pred = clf.predict([features_float(text, args.features)])[0]
        integer_correct += predicted == truth
        float_correct += float_pred == truth
        parity += predicted == float_pred
    n = len(test_texts)
    print(
        f"integer test accuracy {integer_correct / n:.4f}  "
        f"float test accuracy {float_correct / n:.4f}  "
        f"label parity {parity / n:.4f}",
        flush=True,
    )

    spec = {
        "modelId": "claims-triage-v1",
        "architecture": f"linear-{args.features}-{len(LABELS)}",
        "description": (
            "Synthetic claims-triage classifier over hashed unigrams. "
            "Eight queues: auto collision, property damage, theft, medical "
            "injury, liability, weather, fraud review, subrogation."
        ),
        "features": args.features,
        "labels": LABELS,
        "layers": [
            {
                "op": "dense",
                "input": args.features,
                "output": len(LABELS),
                "weights": reordered,
                "bias": reordered_bias,
            },
            {"op": "softmax"},
        ],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "claims-triage-v1.graph.json"
    model_path.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"wrote {model_path} ({model_path.stat().st_size / 1e3:.1f} kB)")

    cases = []
    for text, truth in list(zip(test_texts, test_labels))[:20]:
        q = features_q16(text, args.features)
        logits = dense_q(q, reordered, reordered_bias, args.features, len(LABELS))
        scores = softmax_q(logits)
        cases.append(
            {
                "text": text,
                "label": truth,
                "predicted": LABELS[int(np.argmax(scores))],
                "scores": scores,
            }
        )
    fixture = {
        "modelId": "claims-triage-v1",
        "note": "Q16.16 reference labels and scores from the exported graph.",
        "testAccuracy": integer_correct / n,
        "parityWithFloat": parity / n,
        "cases": cases,
    }
    (out_dir / "claims-triage-v1.test.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {out_dir / 'claims-triage-v1.test.json'}")


if __name__ == "__main__":
    main()
