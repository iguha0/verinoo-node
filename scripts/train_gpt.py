#!/usr/bin/env python3
"""Train and export the tiny GPT-style decoder.

Demonstration model: a byte/char BPE tokenizer over a synthetic support corpus
and a two-layer causal decoder. The point is a deterministic decoder with a KV
cache in consensus, not generation quality.

    python3 scripts/train_gpt.py
    python3 scripts/train_gpt.py --epochs 150 --seed 4

Outputs:
  models/tiny-gpt-v1.json       quantised decoder (runtime input)
  models/tiny-gpt-v1.test.json  float-reference greedy-generation vectors
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
MAX_CONTEXT = 32
HIDDEN = 64
HEADS = 4
LAYERS = 2
FF = 128
MAX_NEW_TOKENS = 24
Q = 65536
PROMPTS = [
    "i want a",
    "my package",
    "where is my order",
    "please cancel",
    "i was charged",
    "how do i",
    "the tracking",
    "please refund",
]


def quantise(value: float) -> int:
    scaled = value * Q
    rounded = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    if abs(value) >= 32768:
        raise ValueError(f"weight {value} is outside the Q16.16 real range")
    return int(rounded)


def flatten(tensor: torch.Tensor) -> list[int]:
    return [quantise(float(x)) for x in tensor.detach().cpu().reshape(-1)]


def apply_merge(tokens: list[str], a: str, b: str, new: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
            out.append(new)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def train_bpe(lines: list[str], vocab_size: int):
    vocab = list(SPECIALS) + sorted({ch for line in lines for ch in line})
    merges: list[tuple[str, str]] = []
    seqs = [list(line) for line in lines]
    while len(vocab) < vocab_size:
        counts: Counter[tuple[str, str]] = Counter()
        for seq in seqs:
            for a, b in zip(seq, seq[1:]):
                counts[(a, b)] += 1
        if not counts:
            break
        best_pair = None
        best_count = -1
        for pair, count in counts.items():
            if count > best_count or (
                count == best_count and (best_pair is None or pair < best_pair)
            ):
                best_pair, best_count = pair, count
        assert best_pair is not None
        a, b = best_pair
        new = a + b
        merges.append((a, b))
        vocab.append(new)
        seqs = [apply_merge(seq, a, b, new) for seq in seqs]
    return vocab, merges


def encode(text: str, merges: list[tuple[str, str]], vocab: dict[str, int], unk: int):
    tokens = list(text)
    for a, b in merges:
        tokens = apply_merge(tokens, a, b, a + b)
    return [vocab.get(tok, unk) for tok in tokens]


def decode(ids: list[int], vocab_list: list[str]) -> str:
    specials = set(SPECIALS)
    return "".join(vocab_list[i] for i in ids if 0 <= i < len(vocab_list) and vocab_list[i] not in specials)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(HIDDEN)
        self.attn = nn.MultiheadAttention(HIDDEN, HEADS, batch_first=True, dropout=0.05)
        self.ln2 = nn.LayerNorm(HIDDEN)
        self.ff1 = nn.Linear(HIDDEN, FF)
        self.ff2 = nn.Linear(FF, HIDDEN)
        self.drop = nn.Dropout(0.05)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        y = self.ln1(x)
        a, _ = self.attn(
            y, y, y, attn_mask=mask, key_padding_mask=pad_mask, need_weights=False
        )
        x = x + self.drop(a)
        x = x + self.drop(self.ff2(F.relu(self.ff1(self.ln2(x)))))
        return x


class Model(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, HIDDEN)
        self.pos = nn.Parameter(torch.randn(MAX_CONTEXT, HIDDEN) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.lnf = nn.LayerNorm(HIDDEN)
        self.head = nn.Linear(HIDDEN, vocab_size)

    def forward(
        self, ids: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        length = ids.shape[1]
        x = self.emb(ids) + self.pos[:length]
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            x = block(x, mask, pad_mask)
        x = self.lnf(x)
        return self.head(x)


def generate_float(model: Model, prompt: str, merges, vocab, vocab_list, bos, eos):
    prompt_ids = encode(prompt, merges, vocab, vocab["<unk>"])
    ids = [bos] + prompt_ids
    generated: list[int] = []
    model.eval()
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            if len(ids) >= MAX_CONTEXT:
                break
            logits = model(torch.tensor(ids))[0, -1]
            nxt = int(logits.argmax())
            ids.append(nxt)
            generated.append(nxt)
            if nxt == eos:
                break
    return ids, generated, decode(generated, vocab_list)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="crates/model/data/decoder-corpus.txt")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=320)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    lines = [line.strip() for line in Path(args.data).read_text().splitlines() if line.strip()]
    vocab_list, merges = train_bpe(lines, args.vocab_size)
    vocab = {tok: i for i, tok in enumerate(vocab_list)}
    bos, eos = vocab["<bos>"], vocab["<eos>"]
    print(f"BPE: {len(vocab_list)} tokens, {len(merges)} merges", flush=True)

    encoded = []
    for line in lines:
        ids = [bos] + encode(line, merges, vocab, vocab["<unk>"])
        ids = ids[:MAX_CONTEXT]
        if len(ids) >= 2:
            encoded.append(ids)
    train = [ids for i, ids in enumerate(encoded) if i % 10 != 0]
    val = [ids for i, ids in enumerate(encoded) if i % 10 == 0]
    print(f"train {len(train)}  val {len(val)}", flush=True)

    batch_size = 16
    pad_id = vocab["<pad>"]

    def make_batch(seqs):
        xs = [seq[:-1] for seq in seqs]
        ys = [seq[1:] for seq in seqs]
        length = max(len(x) for x in xs)
        x = torch.full((len(seqs), length), pad_id, dtype=torch.long)
        y = torch.full((len(seqs), length), pad_id, dtype=torch.long)
        for row, (xi, yi) in enumerate(zip(xs, ys)):
            x[row, : len(xi)] = torch.tensor(xi)
            y[row, : len(yi)] = torch.tensor(yi)
        return x, y, x == pad_id

    def evaluate(seqs):
        model.eval()
        loss_sum = 0.0
        correct = 0
        count = 0
        with torch.no_grad():
            for start in range(0, len(seqs), batch_size):
                x, y, pad_mask = make_batch(seqs[start : start + batch_size])
                logits = model(x, pad_mask)
                loss_sum += float(
                    F.cross_entropy(
                        logits.reshape(-1, len(vocab_list)),
                        y.reshape(-1),
                        ignore_index=pad_id,
                    )
                )
                pred = logits.argmax(-1)
                valid = y != pad_id
                correct += int(((pred == y) & valid).sum())
                count += int(valid.sum())
        return loss_sum / max(1, math.ceil(len(seqs) / batch_size)), correct / max(1, count)

    model = Model(len(vocab_list))
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    best_val = float("inf")
    best_state = None
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train)
        total = 0.0
        batches = 0
        for start in range(0, len(train), batch_size):
            x, y, pad_mask = make_batch(train[start : start + batch_size])
            logits = model(x, pad_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, len(vocab_list)),
                y.reshape(-1),
                ignore_index=pad_id,
            )
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss.detach())
            batches += 1
        if epoch % 10 == 9:
            vloss, acc = evaluate(val)
            print(
                f"epoch {epoch + 1} train {total / max(1, batches):.4f} "
                f"val {vloss:.4f} token-acc {acc:.3f}",
                flush=True,
            )
            if vloss < best_val - 1e-5:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 4:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"best val loss {best_val:.4f}", flush=True)

    spec = {
        "kind": "gpt-decoder",
        "modelId": "tiny-gpt-v1",
        "architecture": f"gpt-decoder-{LAYERS}x{HIDDEN}-{HEADS}h",
        "description": (
            "Tiny GPT-style decoder over a BPE tokenizer, trained on a synthetic "
            "support corpus. Demonstration model, not a benchmark."
        ),
        "vocab": vocab_list,
        "merges": [[a, b] for a, b in merges],
        "maxContext": MAX_CONTEXT,
        "maxNewTokens": MAX_NEW_TOKENS,
        "hidden": HIDDEN,
        "heads": HEADS,
        "ff": FF,
        "layers": LAYERS,
        "bosId": bos,
        "eosId": eos,
        "padId": vocab["<pad>"],
        "unkId": vocab["<unk>"],
        "embedding": flatten(model.emb.weight),
        "positional": flatten(model.pos),
        "decoder": [],
        "lnfGamma": flatten(model.lnf.weight),
        "lnfBeta": flatten(model.lnf.bias),
        "headWeights": flatten(model.head.weight),
        "headBias": flatten(model.head.bias),
    }
    for block in model.blocks:
        in_weight = block.attn.in_proj_weight.detach()
        in_bias = block.attn.in_proj_bias.detach()
        h = HIDDEN
        spec["decoder"].append(
            {
                "ln1Gamma": flatten(block.ln1.weight),
                "ln1Beta": flatten(block.ln1.bias),
                "qWeights": flatten(in_weight[0:h]),
                "qBias": flatten(in_bias[0:h]),
                "kWeights": flatten(in_weight[h : 2 * h]),
                "kBias": flatten(in_bias[h : 2 * h]),
                "vWeights": flatten(in_weight[2 * h : 3 * h]),
                "vBias": flatten(in_bias[2 * h : 3 * h]),
                "oWeights": flatten(block.attn.out_proj.weight),
                "oBias": flatten(block.attn.out_proj.bias),
                "ln2Gamma": flatten(block.ln2.weight),
                "ln2Beta": flatten(block.ln2.bias),
                "ff1Weights": flatten(block.ff1.weight),
                "ff1Bias": flatten(block.ff1.bias),
                "ff2Weights": flatten(block.ff2.weight),
                "ff2Bias": flatten(block.ff2.bias),
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tiny-gpt-v1.json").write_text(json.dumps(spec, indent=2) + "\n")

    cases = []
    for prompt in PROMPTS:
        ids, generated, text = generate_float(
            model, prompt, merges, vocab, vocab_list, bos, eos
        )
        cases.append(
            {
                "prompt": prompt,
                "promptTokens": encode(prompt, merges, vocab, vocab["<unk>"]),
                "tokenIds": generated,
                "text": text,
                "fullIds": ids,
            }
        )
        print(f"prompt {prompt!r} -> {text!r}  tokens {generated}", flush=True)

    fixture = {
        "modelId": "tiny-gpt-v1",
        "note": (
            "Float-reference greedy generations from scripts/train_gpt.py. "
            "The Q16.16 runtime must produce the same token ids."
        ),
        "bestValLoss": best_val,
        "cases": cases,
    }
    (out_dir / "tiny-gpt-v1.test.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {out_dir / 'tiny-gpt-v1.json'}")
    print(f"wrote {out_dir / 'tiny-gpt-v1.test.json'}")


if __name__ == "__main__":
    main()
