#!/usr/bin/env bash
# End-to-end claims-triage pilot demo and evidence report.
#
#   cargo build --workspace && ./scripts/demo-claims.sh
#   CLAIMS_NODE=https://verinoo.com ./scripts/demo-claims.sh   # use a running node
#
# Runs three claim notes through the claims-triage-v1 classifier, verifies each
# receipt, and writes a markdown report to reports/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/target/debug/verinoo-node"
API="$(echo "${CLAIMS_API_PORT:-7960}")"
P2P="$(echo "${CLAIMS_P2P_PORT:-7961}")"
REPORT_DIR="${CLAIMS_REPORT_DIR:-$ROOT/reports}"
EXTERNAL_NODE="${CLAIMS_NODE:-}"
WORK="$(mktemp -d)"
PASS=0
FAIL=0
PIDS=()

pass() { PASS=$((PASS + 1)); echo "PASS $*"; }
fail() { FAIL=$((FAIL + 1)); echo "FAIL $*" >&2; }

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    [ -n "$pid" ] && wait "$pid" 2>/dev/null || true
  done
  rm -rf "$WORK"
}
trap cleanup EXIT

CLAIMS=(
  "my car was rear ended at a red light and the bumper is cracked"
  "a tree fell on the roof and broke two windows"
  "my laptop was stolen from the hotel room"
)
EXPECTED=(auto_collision property_damage theft)

if [ -n "$EXTERNAL_NODE" ]; then
  NODE="${EXTERNAL_NODE%/}"
  echo "→ using running node $NODE"
else
  if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found — run: cargo build --workspace" >&2
    exit 1
  fi
  NODE="http://localhost:$API"
  echo "→ starting validator on :$API (p2p :$P2P)"
  VNN_DB="$WORK/data" VNN_API_PORT="$API" VNN_P2P_PORT="$P2P" \
  VNN_KEY_SEED="claims-demo-$(date +%s)" VNN_VALIDATOR=1 \
  VNN_MODEL_DIR="$ROOT/models" VNN_BLOCK_TIME_MS=1000 \
    "$BIN" >"$WORK/node.log" 2>&1 &
  PIDS+=("$!")
  for _ in $(seq 1 60); do
    curl -fsS -m1 "$NODE/health" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -fsS -m2 "$NODE/health" >/dev/null || {
    echo "error: node did not become healthy; log:" >&2
    tail -30 "$WORK/node.log" >&2
    exit 1
  }
fi

echo "→ model"
curl -fsS "$NODE/v1/models" >"$WORK/models.json"
python3 - "$WORK/models.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
m = next((x for x in d.get("models", []) if x.get("modelId") == "claims-triage-v1"), None)
if not m:
    print("claims-triage-v1 is not loaded on this node", file=sys.stderr)
    sys.exit(1)
print(f"  {m['modelId']} | {m['parameterCount']} params | {m['weightsHash'][:16]}…")
PY
pass "claims-triage-v1 is registered"

echo "→ wallet and faucet"
"$BIN" wallet generate --output "$WORK/key.json" >/dev/null
ADDR="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["address"])' "$WORK/key.json")"
curl -fsS -XPOST "$NODE/faucet" -H 'content-type: application/json' \
  -d "{\"address\":\"$ADDR\"}" >/dev/null
for _ in $(seq 1 60); do
  BAL="$(curl -fsS -m2 "$NODE/account/$ADDR" 2>/dev/null |
    python3 -c 'import json,sys;print(json.load(sys.stdin).get("balance",0))' 2>/dev/null || echo 0)"
  [ "${BAL:-0}" -ge 10000 ] && break
  sleep 1
done
[ "${BAL:-0}" -ge 10000 ] && pass "wallet funded ($BAL)" || fail "faucet did not fund the wallet"

for i in "${!CLAIMS[@]}"; do
  TEXT="${CLAIMS[$i]}"
  WANT="${EXPECTED[$i]}"
  echo "→ claim $((i + 1)): $TEXT"
  if ! "$BIN" infer --key-file "$WORK/key.json" --model claims-triage-v1 \
      --input "$TEXT" --node "$NODE" >"$WORK/infer$i.txt" 2>"$WORK/infer$i.err"; then
    fail "inference submission failed for claim $((i + 1))"
    continue
  fi
  TASK="$(sed -n 's/^task: //p' "$WORK/infer$i.txt")"
  [ -n "$TASK" ] || { fail "no task id for claim $((i + 1))"; continue; }

  for _ in $(seq 1 60); do
    CODE="$(curl -sS -m2 -o "$WORK/task$i.json" -w '%{http_code}' "$NODE/v1/tasks/$TASK" 2>/dev/null || echo 000)"
    if [ "$CODE" = "200" ]; then
      STATUS="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["task"]["status"])' "$WORK/task$i.json")"
      [ "$STATUS" = "completed" ] && break
    fi
    sleep 1
  done

  curl -fsS "$NODE/v1/verify/$TASK" >"$WORK/verify$i.json"
  LABEL="$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));o=d.get("output");o=json.loads(o) if isinstance(o,str) else o;print((o or {}).get("label",""))' "$WORK/verify$i.json")"
  VERIFIED="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("verified"))' "$WORK/verify$i.json")"
  if [ "$LABEL" = "$WANT" ]; then
    pass "claim $((i + 1)) returned $LABEL"
  else
    fail "claim $((i + 1)) returned $LABEL, expected $WANT"
  fi
  if [ "$VERIFIED" = "True" ]; then
    pass "claim $((i + 1)) receipt verified"
  else
    fail "claim $((i + 1)) receipt did not verify"
  fi
done

curl -fsS "$NODE/v1/usage/$ADDR" >"$WORK/usage.json" 2>/dev/null || true
curl -fsS "$NODE/v1/usage/$ADDR/csv" >"$WORK/usage.csv" 2>/dev/null || true

mkdir -p "$REPORT_DIR"
REPORT="$REPORT_DIR/claims-pilot-$(date -u +%Y%m%d-%H%M%S).md"
REPORT_ARGS=()
for i in "${!CLAIMS[@]}"; do
  REPORT_ARGS+=(--claim "${CLAIMS[$i]}|${EXPECTED[$i]}")
done
python3 "$ROOT/scripts/claims_report.py" \
  --work-dir "$WORK" --node "$NODE" --out "$REPORT" "${REPORT_ARGS[@]}"

echo
echo "report: $REPORT"
if [ "$FAIL" -ne 0 ]; then
  echo "✗ $FAIL check(s) failed ($PASS passed)" >&2
  exit 1
fi
echo "✓ all $PASS checks passed"
