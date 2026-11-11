#!/usr/bin/env bash
# End-to-end product demo: fund a wallet, run an inference, verify the receipt.
#
#   cargo build --workspace && ./scripts/demo.sh
#
# Runs one validator locally on 7100/7101 with a throwaway database.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/target/debug/verinoo-node"
API_PORT="${VNN_API_PORT:-7100}"
P2P_PORT="${VNN_P2P_PORT:-7101}"
WORK="$(mktemp -d)"
DB="$WORK/data"
KEY="$WORK/key.json"
LOG="$WORK/node.log"

if [ ! -x "$BIN" ]; then
  echo "error: $BIN not found — run: cargo build --workspace" >&2
  exit 1
fi

cleanup() {
  if [ -n "${NODE_PID:-}" ] && kill -0 "$NODE_PID" 2>/dev/null; then
    kill "$NODE_PID" 2>/dev/null || true
    wait "$NODE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "→ starting validator (api :$API_PORT, p2p :$P2P_PORT)"
VNN_DB="$DB" VNN_API_PORT="$API_PORT" VNN_P2P_PORT="$P2P_PORT" \
VNN_KEY_SEED="demo-$(date +%s)" VNN_VALIDATOR=1 \
  "$BIN" >"$LOG" 2>&1 &
NODE_PID=$!

for _ in $(seq 1 30); do
  if curl -fsS -m 1 "http://localhost:$API_PORT/health" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS -m 2 "http://localhost:$API_PORT/health" >/dev/null || {
  echo "error: node did not become healthy; log:" >&2; tail -20 "$LOG" >&2; exit 1;
}

echo "→ model"
curl -fsS "http://localhost:$API_PORT/v1/models" | python3 -m json.tool | head -20

echo "→ creating wallet and funding it from the faucet"
"$BIN" wallet generate --output "$KEY" >/dev/null
ADDR="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["address"])' "$KEY")"
curl -fsS -X POST "http://localhost:$API_PORT/faucet" \
  -H 'content-type: application/json' -d "{\"address\":\"$ADDR\"}" >/dev/null
echo "  waiting for the drip to be mined…"
sleep 13

echo "→ inference"
"$BIN" infer --key-file "$KEY" --input "my parcel was never delivered" \
  --node "http://localhost:$API_PORT" | tee "$WORK/infer.txt"
TASK="$(sed -n 's/^task: //p' "$WORK/infer.txt")"
[ -n "$TASK" ] || { echo "error: no task id" >&2; exit 1; }

echo "→ waiting for the task to be executed in a block…"
sleep 13
curl -fsS "http://localhost:$API_PORT/v1/tasks/$TASK" | python3 -m json.tool

echo "→ independent verification"
curl -fsS "http://localhost:$API_PORT/v1/verify/$TASK" | python3 -m json.tool

echo "→ metering"
curl -fsS "http://localhost:$API_PORT/v1/usage/$ADDR"
echo
curl -fsS "http://localhost:$API_PORT/v1/usage/$ADDR/csv"

echo "✓ done — data in $WORK"
