#!/usr/bin/env python3
"""Train and export the tiny intent transformer.

This is a demonstration model, not a benchmark: 80 examples, four classes.
The point is a real attention encoder whose weights can be quantised to Q16.16
and executed deterministically by `verinoo-runtime`.

    python3 scripts/train_transformer.py
    python3 scripts/train_transformer.py --out-dir models --epochs 500 --seed 1

Outputs:
  models/intent-transformer-v1.json       quantised model (runtime input)
  models/intent-transformer-v1.test.json  float-reference parity vectors
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = ["refund", "shipping", "billing", "cancel"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
MAX_SEQ_LEN = 12
HIDDEN = 16
HEADS = 2
FF = 32
LAYERS = 1
PAD = "<pad>"
UNK = "<unk>"
CLS = "<cls>"
TOKEN_RE = re.compile(r"[a-z0-9]+")
Q = 65536


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def token_ids(text: str, vocab: dict[str, int]) -> list[int]:
    ids = [vocab[CLS]] + [vocab.get(tok, vocab[UNK]) for tok in tokens(text)]
    return ids[:MAX_SEQ_LEN]


def quantise(value: float) -> int:
    scaled = value * Q
    rounded = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    if abs(value) >= 32768:
        raise ValueError(f"weight {value} is outside the Q16.16 real range")
    return int(rounded)


def flatten(tensor: torch.Tensor) -> list[int]:
    return [quantise(float(x)) for x in tensor.detach().cpu().reshape(-1)]


class EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(HIDDEN)
        self.attn = nn.MultiheadAttention(
            HIDDEN, HEADS, batch_first=True, dropout=0.0
        )
        self.ln2 = nn.LayerNorm(HIDDEN)
        self.ff1 = nn.Linear(HIDDEN, FF)
        self.ff2 = nn.Linear(FF, HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln1(x)
        attn, _ = self.attn(y, y, y, need_weights=False)
        x = x + attn
        x = x + self.ff2(F.relu(self.ff1(self.ln2(x))))
        return x


class Model(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, HIDDEN)
        self.pos = nn.Parameter(torch.randn(MAX_SEQ_LEN, HIDDEN) * 0.02)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(LAYERS)])
        self.lnf = nn.LayerNorm(HIDDEN)
        self.head = nn.Linear(HIDDEN, len(LABELS))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids) + self.pos[: len(ids)]
        for layer in self.layers:
            x = layer(x)
        x = self.lnf(x)
        return self.head(x[0])


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def accuracy(model: Model, rows: list[dict], vocab: dict[str, int]) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for row in rows:
            logits = model(torch.tensor(token_ids(row["text"], vocab)))
            correct += int(logits.argmax()) == LABEL_TO_ID[row["label"]]
    model.train()
    return correct / len(rows)


def export_layer(layer: EncoderLayer) -> dict:
    in_weight = layer.attn.in_proj_weight.detach()
    in_bias = layer.attn.in_proj_bias.detach()
    h = HIDDEN
    return {
        "ln1Gamma": flatten(layer.ln1.weight),
        "ln1Beta": flatten(layer.ln1.bias),
        "qWeights": flatten(in_weight[0:h]),
        "qBias": flatten(in_bias[0:h]),
        "kWeights": flatten(in_weight[h : 2 * h]),
        "kBias": flatten(in_bias[h : 2 * h]),
        "vWeights": flatten(in_weight[2 * h : 3 * h]),
        "vBias": flatten(in_bias[2 * h : 3 * h]),
        "oWeights": flatten(layer.attn.out_proj.weight),
        "oBias": flatten(layer.attn.out_proj.bias),
        "ln2Gamma": flatten(layer.ln2.weight),
        "ln2Beta": flatten(layer.ln2.bias),
        "ff1Weights": flatten(layer.ff1.weight),
        "ff1Bias": flatten(layer.ff1.bias),
        "ff2Weights": flatten(layer.ff2.weight),
        "ff2Bias": flatten(layer.ff2.bias),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="crates/model/data/intents.jsonl")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows = load_rows(Path(args.data))
    vocab_tokens = sorted({tok for row in rows for tok in tokens(row["text"])})
    vocab_list = [PAD, UNK, CLS] + vocab_tokens
    vocab = {tok: i for i, tok in enumerate(vocab_list)}

    train = [row for i, row in enumerate(rows) if i % 8 not in (0, 1)]
    val = [row for i, row in enumerate(rows) if i % 8 == 1]
    test = [row for i, row in enumerate(rows) if i % 8 == 0]

    model = Model(len(vocab_list))
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.05)
    best_val = -1.0
    best_state = None
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train)
        for row in train:
            ids = torch.tensor(token_ids(row["text"], vocab))
            target = torch.tensor([LABEL_TO_ID[row["label"]]])
            loss = F.cross_entropy(model(ids).unsqueeze(0), target)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        if epoch % 20 == 19:
            score = accuracy(model, val, vocab)
            if score > best_val + 1e-9:
                best_val = score
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)

    train_acc = accuracy(model, train, vocab)
    val_acc = accuracy(model, val, vocab)
    test_acc = accuracy(model, test, vocab)
    parameters = sum(p.numel() for p in model.parameters())
    print(
        f"train {train_acc:.3f} ({len(train)})  "
        f"val {val_acc:.3f} ({len(val)})  test {test_acc:.3f} ({len(test)})  "
        f"params {parameters}"
    )

    spec = {
        "kind": "transformer-classifier",
        "modelId": "intent-transformer-v1",
        "architecture": f"transformer-encoder-{LAYERS}x{HIDDEN}-{HEADS}h",
        "description": (
            "Tiny transformer encoder for support-intent classification over a "
            "word-level vocabulary. Demonstration model, not a benchmark."
        ),
        "pooling": "cls",
        "vocab": vocab_list,
        "maxSeqLen": MAX_SEQ_LEN,
        "hidden": HIDDEN,
        "heads": HEADS,
        "ff": FF,
        "layers": LAYERS,
        "labels": LABELS,
        "embedding": flatten(model.emb.weight),
        "positional": flatten(model.pos),
        "encoder": [export_layer(layer) for layer in model.layers],
        "lnfGamma": flatten(model.lnf.weight),
        "lnfBeta": flatten(model.lnf.bias),
        "classifierWeights": flatten(model.head.weight),
        "classifierBias": flatten(model.head.bias),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "intent-transformer-v1.json"
    model_path.write_text(json.dumps(spec, indent=2) + "\n")

    cases = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            logits = model(torch.tensor(token_ids(row["text"], vocab)))
            probs = torch.softmax(logits, dim=-1)
            cases.append(
                {
                    "text": row["text"],
                    "label": row["label"],
                    "predicted": LABELS[int(logits.argmax())],
                    "tokens": token_ids(row["text"], vocab),
                    "logits": [float(x) for x in logits],
                    "scores": [float(x) for x in probs],
                }
            )
    fixture = {
        "modelId": "intent-transformer-v1",
        "note": (
            "Float-reference outputs from scripts/train_transformer.py. "
            "The Q16.16 runtime must preserve the predicted label."
        ),
        "trainAccuracy": train_acc,
        "valAccuracy": val_acc,
        "testAccuracy": test_acc,
        "cases": cases,
    }
    (out_dir / "intent-transformer-v1.test.json").write_text(
        json.dumps(fixture, indent=2) + "\n"
    )
    print(f"wrote {model_path}")
    print(f"wrote {out_dir / 'intent-transformer-v1.test.json'}")


if __name__ == "__main__":
    main()
