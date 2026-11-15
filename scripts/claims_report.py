#!/usr/bin/env python3
"""Render the claims-triage pilot evidence report.

The demo script writes `/v1/models`, `/v1/verify/:taskId` and usage responses
into a work directory; this turns them into a markdown artifact a design
partner can hand to their model-risk or audit team.

    python3 scripts/claims_report.py \
      --work-dir /tmp/demo --node https://verinoo.com --out report.md \
      --claim "claim text|expected_label"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def load(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--claim", action="append", default=[])
    args = parser.parse_args()

    work = Path(args.work_dir)
    models = load(work / "models.json", {})
    model = next(
        (m for m in models.get("models", []) if m.get("modelId") == "claims-triage-v1"),
        {},
    )
    usage = load(work / "usage.json", {})
    csv_text = (work / "usage.csv").read_text() if (work / "usage.csv").exists() else ""

    claims = []
    for i, raw in enumerate(args.claim):
        text, _, expected = raw.partition("|")
        verify = load(work / f"verify{i}.json", {})
        output = verify.get("output")
        try:
            output = json.loads(output) if isinstance(output, str) else output
        except Exception:
            output = None
        claims.append(
            {
                "text": text,
                "expected": expected,
                "verify": verify,
                "output": output or {},
            }
        )

    lines: list[str] = []
    lines.append("# Claims-triage pilot evidence")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"Node: `{args.node}`")
    lines.append("")
    lines.append("## Model")
    lines.append("")
    if model:
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| modelId | `{model.get('modelId')}` |")
        lines.append(f"| architecture | `{model.get('architecture')}` |")
        lines.append(f"| labels | {', '.join(model.get('labels', []))} |")
        lines.append(f"| parameters | {model.get('parameterCount')} |")
        lines.append(f"| runtime | `{model.get('runtime')}` |")
        lines.append(f"| weightsHash | `{model.get('weightsHash')}` |")
    else:
        lines.append("The node did not report `claims-triage-v1`.")
    lines.append("")
    lines.append("## Claims")
    lines.append("")
    lines.append("| Claim note | Expected | Returned | Verified | Task |")
    lines.append("|---|---|---|---|---|")
    for claim in claims:
        output = claim["output"]
        verify = claim["verify"]
        label = output.get("label", "n/a")
        verified = "yes" if verify.get("verified") else "no"
        task = verify.get("taskId", "n/a")
        short_task = f"`{task[:16]}...`" if len(task) > 16 else f"`{task}`"
        lines.append(
            f"| {claim['text']} | `{claim['expected']}` | `{label}` | {verified} | {short_task} |"
        )
    lines.append("")
    if claims and claims[0]["verify"].get("receipt"):
        lines.append("## Receipt for the first claim")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(claims[0]["verify"]["receipt"], indent=2))
        lines.append("```")
        lines.append("")
        task = claims[0]["verify"].get("taskId", "")
        lines.append("Recompute it on any node:")
        lines.append("")
        lines.append("```bash")
        lines.append(f"curl -s {args.node}/v1/verify/{task} | jq '{{verified, output, receipt}}'")
        lines.append("```")
        lines.append("")
    lines.append("## Metering")
    lines.append("")
    if usage:
        lines.append("```json")
        lines.append(json.dumps(usage, indent=2))
        lines.append("```")
        lines.append("")
    if csv_text:
        lines.append("```csv")
        lines.append(csv_text.strip())
        lines.append("```")
        lines.append("")
    lines.append("## What this proves")
    lines.append("")
    lines.append("- The classifier ran during block application; the protocol requires every validator to re-execute it.")
    lines.append("- The output is consensus state, not a response attached to a log.")
    lines.append("- The receipt binds the task, model, weights hash, input hash, output hash, block and validator signature.")
    lines.append("- `GET /v1/verify/:taskId` recomputes the output and compares it with the committed value.")
    lines.append("- The model has 100% label parity between its float reference and the Q16.16 graph on the committed fixture.")
    lines.append("")
    lines.append("## What this does not prove")
    lines.append("")
    lines.append("- **Not trustless.** One operator runs the validator set; there is no BFT finality.")
    lines.append("- **Not operator-blind.** Inputs are encrypted to the validators, but the operator can decrypt.")
    lines.append("- **Synthetic demo data.** The classifier is trained on a synthetic claims corpus, not a customer's data.")
    lines.append("- **Not a general model host.** Dense/relu/softmax classifiers and the committed transformer encoder only; no LLM checkpoint.")
    lines.append("")
    lines.append("## Fit")
    lines.append("")
    lines.append(
        "This is the shape of a design-partner pilot: one text classifier, one decision "
        "workflow, and a receipt an auditor can recheck. Production usage is priced per "
        "verified task; the pilot is scoped to eight weeks."
    )
    lines.append("")

    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
