# Claims-triage pilot kit

This is the first vertical demo built for the design-partner profile: a
mid-market insurer or insurtech with a text classifier that routes claim notes
into a small number of queues, and a compliance or model-risk team that has
been asked to explain a decision.

## The classifier

| Field | Value |
|---|---|
| modelId | `claims-triage-v1` |
| task | claim note → queue |
| labels | auto collision, property damage, theft, medical injury, liability, weather, fraud review, subrogation |
| features | 2,048 hashed unigrams, Q16.16 shares |
| model | multinomial logistic regression, 16,392 parameters |
| train accuracy | 99.53% |
| held-out accuracy | 98.75% (158/160) |
| integer/float parity | 100% (160/160) |
| model file | 197 kB |

It is trained on a synthetic 800-example claims corpus
(`crates/model/data/claims.jsonl`) with `scripts/train_claims_classifier.py`.
The point is the shape, not the corpus: a real partner would train the same
graph on their own labelled claims.

## Run the demo

```bash
cargo build --workspace
./scripts/demo-claims.sh
```

The script starts a validator, registers `claims-triage-v1`, funds a wallet,
submits three claim notes, waits for block execution, verifies each receipt and
writes a report to `reports/claims-pilot-<timestamp>.md`.

Against the live testnet:

```bash
CLAIMS_NODE=https://verinoo.com ./scripts/demo-claims.sh
```

## What the report contains

- the model, labels, parameter count and weights hash
- each claim note, the returned queue, the task id and whether the receipt verified
- the full receipt for the first claim: task, model, weights, input hash, output hash, block hash and validator signature
- a `curl` command that recomputes the result on any node
- usage and the invoice CSV
- an explicit “what this proves / what this does not prove” section

Send the report to a prospect's ML lead and model-risk owner. It is the artifact
they can take into an internal review; it is not a marketing claim.

## Bringing a partner's classifier

The current import path supports:

- ONNX classifiers built from `Gemm` / `MatMul` + `Add` / `Relu` / `Softmax`
  over the 2,048-dimension hashed-unigram featuriser (`onnx-import`);
- GGUF dense classifiers (`gguf-import`);
- a transformer encoder trained/exported through the Verinoo format.

A partner with a TF-IDF + logistic regression or small MLP text classifier is
the easiest fit. A partner with a large fine-tuned BERT cannot be imported
faithfully yet; the honest answer is that we would need to retrain or distil a
compatible model.

## What it does not claim

- **Synthetic demo data.** The classifier is not trained on the partner's claims.
- **Not trustless.** One operator runs the validator set; no BFT finality.
- **Not operator-blind.** Inputs are encrypted to the validators, but the
  operator can decrypt.
- **Not a production LLM host.** The pilot uses a classifier and the
  transformer encoder. A licensed 81.9M DistilGPT-2 decoder with 1,024-token
  context, sampling and binary weights runs locally as a decoder demo, but it is
  not deployed or offered as a service in this pilot.
- **Not a general model host.** The model must be in the supported runtime set.

## Pilot shape

One classifier, one decision workflow, eight weeks. We wire the partner's model
into a dedicated validator set, run a joint verification, and hand over receipts
their auditor can recheck. The pilot is $5,000; production usage is priced per
verified task.
