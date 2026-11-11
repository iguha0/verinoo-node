#!/usr/bin/env bash
# Verify a task's receipt against any node: recompute and compare.
#
#   ./scripts/verify.sh <taskId> [--node http://localhost:7100]
set -euo pipefail

TASK="${1:-}"
NODE="http://localhost:7100"

if [ "$TASK" = "--help" ] || [ "$TASK" = "-h" ] || [ -z "$TASK" ]; then
  echo "Usage: $0 <taskId> [--node http://localhost:7100]" >&2
  exit 1
fi
shift || true
while [ "$#" -gt 0 ]; do
  case "$1" in
    --node) NODE="${2:?--node needs a URL}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

RESP="$(curl -fsS -m 10 "$NODE/v1/verify/$TASK")" || {
  echo "error: could not fetch $NODE/v1/verify/$TASK" >&2
  exit 1
}

printf '%s' "$RESP" | python3 -c '
import json, sys
v = json.load(sys.stdin)
print("task:      ", v.get("taskId"))
print("verified:  ", v.get("verified"))
print("output:    ", v.get("output"))
print("on-chain:  ", v.get("onChainHash"))
print("recomputed:", v.get("recomputedHash"))
r = v.get("receipt")
if r:
    print("anchored:   height", r.get("height"), "block", str(r.get("blockHash"))[:16] + "…")
    print("weights:   ", r.get("weightsHash"))
    print("validator: ", r.get("validator"))
else:
    print("anchored:   false (task not yet completed by a block)")
sys.exit(0 if v.get("verified") else 1)
'
