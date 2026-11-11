#!/usr/bin/env bash
# Two-validator testnet: consensus execution, replication, and independent
# verification. This is the design-partner proof: the same task is executed by
# both nodes and the receipt verifies on the node that did not produce it.
#
#   cargo build --workspace && ./scripts/demo-2node.sh
#
# Both nodes share one validator set and dial each other explicitly, so this
# does not depend on mDNS discovery. Block time is the default 10 s.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/target/debug/verinoo-node"
API_A="${VNN_API_A:-7200}"
P2P_A="${VNN_P2P_A:-7300}"
API_B="${VNN_API_B:-7201}"
P2P_B="${VNN_P2P_B:-7301}"
WORK="$(mktemp -d)"
KEY="$WORK/key.json"

if [ ! -x "$BIN" ]; then
  echo "error: $BIN not found — run: cargo build --workspace" >&2
  exit 1
fi

PASS=0
FAIL=0
pass() { PASS=$((PASS + 1)); echo "PASS $*"; }
fail() { FAIL=$((FAIL + 1)); echo "FAIL $*" >&2; }

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
PIDS=()

ADDR_A="$("$BIN" address --seed demo-two-node-a)"
ADDR_B="$("$BIN" address --seed demo-two-node-b)"
VALS="$ADDR_A,$ADDR_B"

echo "→ validator set"
echo "  A: $ADDR_A (api :$API_A, p2p :$P2P_A)"
echo "  B: $ADDR_B (api :$API_B, p2p :$P2P_B)"

echo "→ starting both validators"
VNN_DB="$WORK/a" VNN_API_PORT="$API_A" VNN_P2P_PORT="$P2P_A" \
VNN_KEY_SEED="demo-two-node-a" VNN_VALIDATOR=1 VNN_VALIDATORS="$VALS" \
VNN_SEEDS="/ip4/127.0.0.1/tcp/$P2P_B" VNN_MODEL_DIR="$ROOT/models" \
  "$BIN" >"$WORK/a.log" 2>&1 &
PIDS+=("$!")

VNN_DB="$WORK/b" VNN_API_PORT="$API_B" VNN_P2P_PORT="$P2P_B" \
VNN_KEY_SEED="demo-two-node-b" VNN_VALIDATOR=1 VNN_VALIDATORS="$VALS" \
VNN_SEEDS="/ip4/127.0.0.1/tcp/$P2P_A" VNN_MODEL_DIR="$ROOT/models" \
  "$BIN" >"$WORK/b.log" 2>&1 &
PIDS+=("$!")

api() { echo "http://localhost:$1"; }

# wait_for <seconds> <description> <command...>
wait_for() {
  local deadline=$1 desc=$2
  shift 2
  local end=$((SECONDS + deadline))
  while [ "$SECONDS" -lt "$end" ]; do
    if "$@" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  echo "error: timed out waiting for $desc; node A log:" >&2
  tail -20 "$WORK/a.log" >&2 || true
  echo "...node B log:" >&2
  tail -20 "$WORK/b.log" >&2 || true
  return 1
}

healthy() { curl -fsS -m 1 "$(api "$1")/health" >/dev/null; }
ready() { healthy "$1" && healthy "$2"; }

echo "→ waiting for both nodes"
wait_for 60 "both nodes to become healthy" ready "$API_A" "$API_B"
pass "both nodes healthy"

wait_for 60 "validator set on both nodes" bash -c \
  "curl -fsS '$(api "$API_A")/validators' | grep -q '$ADDR_A' && \
   curl -fsS '$(api "$API_B")/validators' | grep -q '$ADDR_B'"
pass "validator set exposed on both nodes"

height() { curl -fsS -m 2 "$(api "$1")/status" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("height",0))'; }
converged() {
  local a b
  a="$(height "$API_A")"; b="$(height "$API_B")"
  [ "$a" -ge 2 ] && [ "$a" = "$b" ]
}

echo "→ waiting for both chains to advance and converge (round-robin slots)"
wait_for 120 "chains to converge at height ≥ 2" converged
H_HEIGHT="$(height "$API_A")"
pass "chains converged (height $H_HEIGHT)"

echo "→ creating and funding a wallet"
"$BIN" wallet generate --output "$KEY" >/dev/null
ADDR="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["address"])' "$KEY")"
curl -fsS -X POST "$(api "$API_A")/faucet" \
  -H 'content-type: application/json' -d "{\"address\":\"$ADDR\"}" >/dev/null

drip_replicated() {
  local bal
  bal="$(curl -fsS -m 2 "$(api "$API_B")/account/$ADDR" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("balance",0))')"
  [ "$bal" -ge 10000 ]
}
wait_for 90 "faucet drip to replicate to node B" drip_replicated
pass "faucet drip replicated to both nodes"

echo "→ inference on node A"
"$BIN" infer --key-file "$KEY" --input "my parcel was never delivered" \
  --node "$(api "$API_A")" | tee "$WORK/infer.txt"
TASK="$(sed -n 's/^task: //p' "$WORK/infer.txt")"
[ -n "$TASK" ] || { echo "error: no task id" >&2; exit 1; }

completed() {
  curl -fsS -m 2 "$(api "$1")/v1/tasks/$TASK" |
    python3 -c 'import json,sys; t=json.load(sys.stdin).get("task",{}); print(t.get("status",""))' |
    grep -qx completed
}
echo "→ waiting for the task to execute in a block"
wait_for 90 "task to complete on node A" completed "$API_A"
wait_for 90 "task to complete on node B" completed "$API_B"
pass "task executed by consensus on both nodes"

hash_on() {
  curl -fsS -m 2 "$(api "$1")/v1/tasks/$TASK" |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["task"].get("resultHash",""))'
}
HASH_A="$(hash_on "$API_A")"
HASH_B="$(hash_on "$API_B")"
if [ -n "$HASH_A" ] && [ "$HASH_A" = "$HASH_B" ]; then
  pass "both nodes derived an identical result hash (${HASH_A:0:8}…)"
else
  fail "result hashes differ: A=${HASH_A:0:16} B=${HASH_B:0:16}"
fi

verified_on() {
  curl -fsS -m 2 "$(api "$1")/v1/verify/$TASK" |
    python3 -c 'import json,sys; print(json.load(sys.stdin).get("verified"))' |
    grep -qx True
}
if verified_on "$API_A"; then pass "receipt verifies on the producing node"; else fail "receipt does not verify on node A"; fi
if verified_on "$API_B"; then pass "receipt verifies on the independent node"; else fail "receipt does not verify on node B"; fi

usage_ok() {
  curl -fsS -m 2 "$(api "$API_A")/v1/usage/$ADDR" |
    python3 -c 'import json,sys; u=json.load(sys.stdin); print(u.get("completed"), u.get("feesCommitted"))' |
    grep -qx "1 100"
}
if usage_ok; then pass "metering counted the task (feesCommitted 100)"; else fail "metering did not record the task"; fi

CSV="$(curl -fsS -m 2 "$(api "$API_A")/v1/usage/$ADDR/csv")"
if printf '%s' "$CSV" | grep -q '^address,tasks,completed'; then
  pass "CSV export for invoicing"
else
  fail "CSV export malformed"
fi

if converged; then
  pass "chains stayed converged (height $(height "$API_A")/$(height "$API_B"))"
else
  fail "chains diverged"
fi

echo
echo "→ artifacts in $WORK"
if [ "$FAIL" -ne 0 ]; then
  echo "✗ $FAIL check(s) failed ($PASS passed)" >&2
  exit 1
fi
echo "✓ all $PASS checks passed"
