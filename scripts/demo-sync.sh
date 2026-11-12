#!/usr/bin/env bash
# Prove catch-up sync: a late-starting replica replays the validator's blocks.
#
#   cargo build --workspace && ./scripts/demo-sync.sh
#
# The replica is a non-validator with the same validator set and model set. It
# starts at genesis after the validator has already produced several blocks,
# then must converge to the validator's height and head hash via /blocks/since.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/target/debug/verinoo-node"
A_API="${VNN_A_API:-7600}"
A_P2P="${VNN_A_P2P:-7601}"
B_API="${VNN_B_API:-7610}"
B_P2P="${VNN_B_P2P:-7611}"
W="$(mktemp -d)"
PASS=0
FAIL=0
pass() { PASS=$((PASS + 1)); echo "PASS $*"; }
fail() { FAIL=$((FAIL + 1)); echo "FAIL $*" >&2; }

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    [ -n "$pid" ] && wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT
PIDS=()

if [ ! -x "$BIN" ]; then
  echo "error: $BIN not found — run: cargo build --workspace" >&2
  exit 1
fi

A_ADDR="$("$BIN" address --seed sync-validator)"
height() {
  curl -fsS -m 2 "http://localhost:$1/status" 2>/dev/null |
    python3 -c 'import json,sys;print(json.load(sys.stdin).get("height",0))' 2>/dev/null || echo 0
}
echo "→ validator $A_ADDR on :$A_API (p2p :$A_P2P)"
VNN_DB="$W/a" VNN_API_PORT="$A_API" VNN_P2P_PORT="$A_P2P" \
VNN_KEY_SEED="sync-validator" VNN_VALIDATOR=1 VNN_VALIDATORS="$A_ADDR" \
VNN_MODEL_DIR="$ROOT/models" "$BIN" >"$W/a.log" 2>&1 &
PIDS+=("$!")

for _ in $(seq 1 30); do curl -fsS -m1 "http://localhost:$A_API/health" >/dev/null 2>&1 && break; sleep 1; done
for _ in $(seq 1 60); do H="$(height "$A_API")"; [ "$H" -ge 3 ] && break; sleep 2; done
[ "${H:-0}" -ge 3 ] || { echo "error: validator did not reach height 3" >&2; tail -20 "$W/a.log" >&2; exit 1; }
pass "validator produced $H block(s)"

SERVED="$(curl -fsS "http://localhost:$A_API/blocks/since/0" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)))')"
[ "$SERVED" -ge 3 ] && pass "/blocks/since served $SERVED block(s)" || fail "/blocks/since served $SERVED"

echo "→ starting late replica on :$B_API (sync from :$A_API)"
VNN_DB="$W/b" VNN_API_PORT="$B_API" VNN_P2P_PORT="$B_P2P" \
VNN_KEY_SEED="sync-replica" VNN_VALIDATOR=false VNN_VALIDATORS="$A_ADDR" \
VNN_SYNC_URLS="http://127.0.0.1:$A_API" VNN_MODEL_DIR="$ROOT/models" \
  "$BIN" >"$W/b.log" 2>&1 &
PIDS+=("$!")

for _ in $(seq 1 60); do BH="$(height "$B_API")"; [ "$BH" -ge "$H" ] && break; sleep 2; done
BH="$(height "$B_API")"
[ "$BH" -ge "$H" ] && pass "replica caught up (height $BH ≥ $H)" || fail "replica stuck at $BH (validator $H)"

block_at() {
  curl -fsS -m 2 "http://localhost:$1/blocks/since/$(($2 - 1))" 2>/dev/null |
    python3 -c 'import json,sys
b=json.load(sys.stdin)
print(b[0]["header"]["hash"] if b else "")' 2>/dev/null || true
}
AHASH="$(block_at "$A_API" "$H")"
BHASH="$(block_at "$B_API" "$H")"
if [ -n "$AHASH" ] && [ "$BHASH" = "$AHASH" ]; then
  pass "replica block #$H matches validator (${AHASH:0:8}…)"
else
  fail "block #$H mismatch: validator ${AHASH:0:16} replica ${BHASH:0:16}"
fi

grep -q "synced" "$W/b.log" && pass "sync logged by replica" || fail "no sync log line"

echo
if [ "$FAIL" -ne 0 ]; then
  echo "✗ $FAIL check(s) failed ($PASS passed)" >&2
  exit 1
fi
echo "✓ all $PASS checks passed"
