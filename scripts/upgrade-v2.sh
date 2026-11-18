#!/usr/bin/env bash
# Operator helper for the v0.10.0-testnet protocol-v2 upgrade.
#
#   sudo scripts/upgrade-v2.sh --binary /root/verinoo-node-v0.10.0-testnet \
#     --sha256 <expected-sha256>
#   sudo scripts/upgrade-v2.sh --binary /root/verinoo-node-v0.10.0-testnet \
#     --sums-file /root/SHA256SUMS.v0.10.0-testnet \
#     --service verinoo-node --db /var/lib/verinoo/data \
#     --backup-dir /root/verinoo-backups/20261118-120000
#
# The script preflights the host (including /ready), backs up the database and
# old binary, stops the service, installs the new binary, waits for /health and
# /ready, prints /v1/models, and parses /blocks to confirm the first
# post-upgrade block is v2. It exits non-zero and prints "installed but v2
# unconfirmed" when it cannot prove that.
#
# It does not upgrade peers, change VNN_VALIDATORS, fence a host, promote a
# replica or roll back automatically. Backups are never deleted. Rollback
# commands for the old v0.9.0 binary are only printed with --force-rollback.
# Read docs/upgrade-v2.md before running it.
set -euo pipefail

SERVICE="verinoo-node"
DB="/var/lib/verinoo/data"
BACKUP_DIR=""
NEW_BINARY=""
SHA256_ARG=""
SUMS_FILE=""
FORCE_ROLLBACK=0
API_PORT="${VNN_API_PORT:-7100}"
API_BASE="http://127.0.0.1:${API_PORT}"
V2_TIMEOUT="${VNN_UPGRADE_V2_TIMEOUT:-90}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_ARCHIVE=""
INSTALL_TMP=""

usage() {
  cat >&2 <<'EOF'
usage: scripts/upgrade-v2.sh --binary <new-binary>
       (--sha256 <hex> | --sums-file <path>)
       [--service <unit>] [--db <path>] [--backup-dir <path>]
       [--force-rollback]

  --binary          new verinoo-node binary (required)
  --sha256          expected SHA-256 of the new binary (required unless
                    --sums-file is used)
  --sums-file       SHA256SUMS file containing an entry for the new binary
  --service         systemd unit (default: verinoo-node)
  --db              RocksDB path (default: /var/lib/verinoo/data)
  --backup-dir      backup directory (default: /root/verinoo-backups/<timestamp>)
  --force-rollback  allow the failure report to print old-binary rollback
                    commands; only use when no v2 block exists

Run as root on the server. The node must be live, /health must be reachable, and the new binary checksum
must be known before the upgrade starts. A legacy v0.9.0 node may return 404
for /ready; the post-upgrade check still requires /ready 200. VNN_API_PORT
overrides the local API port (7100).
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

cleanup() {
  [ -n "${INSTALL_TMP:-}" ] && rm -f "$INSTALL_TMP"
  return 0
}
trap cleanup EXIT INT TERM

while [ "$#" -gt 0 ]; do
  case "$1" in
    --binary)
      [ "$#" -ge 2 ] || die "--binary needs a path"
      NEW_BINARY="$2"
      shift 2
      ;;
    --binary=*)
      NEW_BINARY="${1#*=}"
      shift
      ;;
    --sha256)
      [ "$#" -ge 2 ] || die "--sha256 needs a value"
      SHA256_ARG="$2"
      shift 2
      ;;
    --sha256=*)
      SHA256_ARG="${1#*=}"
      shift
      ;;
    --sums-file)
      [ "$#" -ge 2 ] || die "--sums-file needs a path"
      SUMS_FILE="$2"
      shift 2
      ;;
    --sums-file=*)
      SUMS_FILE="${1#*=}"
      shift
      ;;
    --service)
      [ "$#" -ge 2 ] || die "--service needs a unit name"
      SERVICE="$2"
      shift 2
      ;;
    --service=*)
      SERVICE="${1#*=}"
      shift
      ;;
    --db)
      [ "$#" -ge 2 ] || die "--db needs a path"
      DB="$2"
      shift 2
      ;;
    --db=*)
      DB="${1#*=}"
      shift
      ;;
    --backup-dir)
      [ "$#" -ge 2 ] || die "--backup-dir needs a path"
      BACKUP_DIR="$2"
      shift 2
      ;;
    --backup-dir=*)
      BACKUP_DIR="${1#*=}"
      shift
      ;;
    --force-rollback)
      FORCE_ROLLBACK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      die "unknown argument: $1"
      ;;
  esac
done

[ -n "$NEW_BINARY" ] || { usage; die "--binary is required"; }
[ "$(id -u)" -eq 0 ] || die "run as root"

for cmd in systemctl curl tar install df du python3; do
  command -v "$cmd" >/dev/null 2>&1 || die "required command not found: $cmd"
done

[ -f "$NEW_BINARY" ] || die "new binary not found: $NEW_BINARY"
[ -x "$NEW_BINARY" ] || die "new binary is not executable: $NEW_BINARY"
NEW_BINARY="$(cd "$(dirname "$NEW_BINARY")" && pwd)/$(basename "$NEW_BINARY")"
BINARY_BASENAME="$(basename "$NEW_BINARY")"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

# Resolve the expected checksum before touching the service. A missing or
# ambiguous checksum is a hard failure: never install an unverified binary.
SHA256_EXPECTED=""
if [ -n "$SHA256_ARG" ]; then
  printf '%s' "$SHA256_ARG" | grep -Eq '^[0-9A-Fa-f]{64}$' ||
    die "--sha256 must be exactly 64 hex characters"
  SHA256_EXPECTED="$(lower "$SHA256_ARG")"
fi
if [ -n "$SUMS_FILE" ]; then
  [ -f "$SUMS_FILE" ] || die "sums file not found: $SUMS_FILE"
  SUMS_FILE="$(cd "$(dirname "$SUMS_FILE")" && pwd)/$(basename "$SUMS_FILE")"
  SUM_HASHES=()
  while IFS= read -r sum_hash; do
    [ -n "$sum_hash" ] && SUM_HASHES+=("$sum_hash")
  done < <(awk -v base="$BINARY_BASENAME" '
    {
      name = $2
      if (name == "") next
      sub(/^\*/, "", name)
      sub(/^\.\//, "", name)
      if (name == base || name ~ ("(^|/)" base "$")) print $1
    }
  ' "$SUMS_FILE")
  [ "${#SUM_HASHES[@]}" -gt 0 ] ||
    die "no checksum entry for $BINARY_BASENAME in $SUMS_FILE"
  FIRST_HASH="$(lower "${SUM_HASHES[0]}")"
  printf '%s' "$FIRST_HASH" | grep -Eq '^[0-9a-f]{64}$' ||
    die "invalid checksum entry for $BINARY_BASENAME in $SUMS_FILE"
  i=1
  while [ "$i" -lt "${#SUM_HASHES[@]}" ]; do
    next_hash="$(lower "${SUM_HASHES[$i]}")"
    [ "$next_hash" = "$FIRST_HASH" ] ||
      die "ambiguous checksum entries for $BINARY_BASENAME in $SUMS_FILE"
    i=$((i + 1))
  done
  if [ -n "$SHA256_EXPECTED" ] && [ "$SHA256_EXPECTED" != "$FIRST_HASH" ]; then
    die "--sha256 ($SHA256_EXPECTED) does not match $SUMS_FILE ($FIRST_HASH)"
  fi
  SHA256_EXPECTED="$FIRST_HASH"
fi
[ -n "$SHA256_EXPECTED" ] ||
  die "refusing to upgrade without a published checksum: pass --sha256 <hex> or --sums-file <path>"

COMPUTED_SHA="$(lower "$(sha256_file "$NEW_BINARY")")"
if [ "$COMPUTED_SHA" != "$SHA256_EXPECTED" ]; then
  die "checksum mismatch for $NEW_BINARY: expected $SHA256_EXPECTED, got $COMPUTED_SHA"
fi
echo "✓ new binary checksum verified: $SHA256_EXPECTED"

[ -d "$DB" ] || die "database not found: $DB"
DB="$(cd "$DB" && pwd)"

systemctl cat "$SERVICE" >/dev/null 2>&1 || die "systemd service not found: $SERVICE"

if [ -z "$BACKUP_DIR" ]; then
  BACKUP_DIR="/root/verinoo-backups/$(date -u +%Y%m%d-%H%M%S)"
fi
mkdir -p "$BACKUP_DIR"
BACKUP_DIR="$(cd "$BACKUP_DIR" && pwd)"
chmod 700 "$BACKUP_DIR" 2>/dev/null || true
if find "$BACKUP_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  die "backup directory is not empty: $BACKUP_DIR (choose a new directory)"
fi

case "$BACKUP_DIR/" in
  "$DB"/*) die "backup directory must not be inside the database: $BACKUP_DIR" ;;
esac
case "$DB/" in
  "$BACKUP_DIR"/*) die "database must not be inside the backup directory: $DB" ;;
esac

echo "→ preflight"
echo "  service:  $SERVICE"
echo "  database: $DB"
echo "  backup:   $BACKUP_DIR"

if ! curl -fsS -m 5 "$API_BASE/health" >/dev/null 2>&1; then
  die "/health is not reachable at $API_BASE/health; fix the node before upgrading"
fi
echo "  /health: reachable at $API_BASE/health"

READY_CODE="$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$API_BASE/ready" 2>/dev/null || true)"
if [ "$READY_CODE" = "200" ]; then
  echo "  /ready: HTTP 200 at $API_BASE/ready"
elif [ "$READY_CODE" = "404" ]; then
  echo "  /ready: 404 (legacy v0.9.0 binary has no readiness endpoint); /health is reachable"
else
  READY_DETAIL="$(curl -sS -m 5 "$API_BASE/ready" 2>/dev/null || true)"
  die "/ready is not ready at $API_BASE/ready before the upgrade (HTTP ${READY_CODE:-unknown})${READY_DETAIL:+: $READY_DETAIL}"
fi

PRE_HEALTH="$(curl -fsS -m 5 "$API_BASE/health" 2>/dev/null || true)"
PRE_HEIGHT="$(printf '%s' "$PRE_HEALTH" |
  python3 -c 'import json,sys; d=json.load(sys.stdin); h=d.get("height"); print(h if isinstance(h,int) and h >= 0 else "")' 2>/dev/null || true)"
[ -n "$PRE_HEIGHT" ] ||
  die "could not read the pre-upgrade height from $API_BASE/health; refusing to upgrade without a comparison point"
echo "  pre-upgrade height: $PRE_HEIGHT"

DB_KB="$(du -sk "$DB" | awk '{print $1}')"
AVAIL_KB="$(df -Pk "$BACKUP_DIR" | awk 'NR==2 {print $4}')"
NEED_KB=$((DB_KB + DB_KB / 5 + 262144))
if [ "$AVAIL_KB" -lt "$NEED_KB" ]; then
  die "not enough free space: need ~${NEED_KB} KiB for database backup + 256 MiB slack, have ${AVAIL_KB} KiB at $BACKUP_DIR"
fi
echo "  disk: OK (${AVAIL_KB} KiB free, need ~${NEED_KB} KiB)"

# systemd may report one ExecStart entry with path=..., or several commands.
# Require exactly one executable path, named verinoo-node, so we never guess
# and overwrite an arbitrary system binary.
EXEC_START="$(systemctl show -p ExecStart --value "$SERVICE" 2>/dev/null || true)"
[ -n "$EXEC_START" ] ||
  die "systemctl returned an empty ExecStart for $SERVICE; refusing to guess the install path"

EXEC_PATHS=()
while IFS= read -r exec_path; do
  [ -n "$exec_path" ] && EXEC_PATHS+=("$exec_path")
done < <(printf '%s\n' "$EXEC_START" | grep -o 'path=[^;[:space:]]*' | sed 's/^path=//')

if [ "${#EXEC_PATHS[@]}" -eq 0 ]; then
  die "could not find any path= entry in ExecStart for $SERVICE; refusing to guess: $EXEC_START"
fi
if [ "${#EXEC_PATHS[@]}" -ne 1 ]; then
  die "ambiguous ExecStart for $SERVICE: found ${#EXEC_PATHS[@]} executable paths; refusing to guess: $EXEC_START"
fi
INSTALL_PATH="${EXEC_PATHS[0]}"
[ -n "$INSTALL_PATH" ] || die "resolved ExecStart path is empty for $SERVICE"
if [ "$(basename "$INSTALL_PATH")" != "verinoo-node" ]; then
  die "ExecStart for $SERVICE resolves to '$INSTALL_PATH', which is not a verinoo-node binary (for example, /usr/bin/env); fix the unit or upgrade manually"
fi
[ -f "$INSTALL_PATH" ] || die "current binary not found at $INSTALL_PATH"
[ "$NEW_BINARY" != "$INSTALL_PATH" ] || die "new binary is already the install target: $INSTALL_PATH"
echo "  install:  $INSTALL_PATH"

OLD_BINARY="$BACKUP_DIR/verinoo-node.pre-upgrade"
cp -p "$INSTALL_PATH" "$OLD_BINARY"
sha256_file "$OLD_BINARY" >"$OLD_BINARY.sha256"
echo "  old binary saved: $OLD_BINARY"

BACKUP_SCRIPT=""
for candidate in "$SCRIPT_DIR/backup.sh" "/opt/verinoo/scripts/backup.sh"; do
  if [ -n "$candidate" ] && [ -f "$candidate" ] && [ -x "$candidate" ]; then
    BACKUP_SCRIPT="$candidate"
    break
  fi
done

if [ -n "$BACKUP_SCRIPT" ]; then
  echo "→ backing up database with $BACKUP_SCRIPT"
  if ! VNN_SERVICE="$SERVICE" VNN_DB="$DB" BACKUP_DIR="$BACKUP_DIR" \
      VNN_API_PORT="$API_PORT" "$BACKUP_SCRIPT"; then
    die "database backup failed; no binary was installed and backups were not deleted"
  fi
  DB_ARCHIVE="$(ls -1t "$BACKUP_DIR"/verinoo-*.tar.gz 2>/dev/null | head -n 1 || true)"
  [ -n "$DB_ARCHIVE" ] || die "backup script finished but no archive was found in $BACKUP_DIR"
  echo "  database archive: $DB_ARCHIVE"
else
  echo "→ scripts/backup.sh not found; using a stopped-service copy"
  if ! systemctl stop "$SERVICE"; then
    die "failed to stop $SERVICE for the database copy; no binary was installed"
  fi
  DB_ARCHIVE="$BACKUP_DIR/verinoo-db-pre-upgrade.tar.gz"
  if ! tar czf "$DB_ARCHIVE" -C "$(dirname "$DB")" "$(basename "$DB")"; then
    systemctl start "$SERVICE" >/dev/null 2>&1 || true
    die "database copy failed; no binary was installed and backups were not deleted"
  fi
  sha256_file "$DB_ARCHIVE" >"$DB_ARCHIVE.sha256"
  echo "  database archive: $DB_ARCHIVE"
fi
[ -s "$DB_ARCHIVE" ] || die "database archive is empty: $DB_ARCHIVE"

echo "→ stopping $SERVICE"
if ! systemctl stop "$SERVICE"; then
  die "failed to stop $SERVICE; no binary was installed and backups were not deleted"
fi
if systemctl is-active --quiet "$SERVICE"; then
  die "$SERVICE is still active after stop; refusing to replace its binary"
fi

echo "→ installing $NEW_BINARY"
INSTALL_TMP="${INSTALL_PATH}.new.$$"
if ! install -m 0755 "$NEW_BINARY" "$INSTALL_TMP"; then
  echo "error: failed to stage the new binary; $SERVICE is stopped and the old binary is still at $INSTALL_PATH" >&2
  echo "  systemctl start \"$SERVICE\"" >&2
  exit 1
fi
STAGED_SHA="$(lower "$(sha256_file "$INSTALL_TMP")")"
if [ "$STAGED_SHA" != "$SHA256_EXPECTED" ]; then
  rm -f "$INSTALL_TMP"
  INSTALL_TMP=""
  echo "error: staged binary checksum mismatch: expected $SHA256_EXPECTED, got $STAGED_SHA" >&2
  echo "  $SERVICE is stopped and the old binary is still at \"$INSTALL_PATH\"" >&2
  echo "  old binary backup: \"$OLD_BINARY\"" >&2
  exit 1
fi
if ! mv -f "$INSTALL_TMP" "$INSTALL_PATH"; then
  rm -f "$INSTALL_TMP"
  INSTALL_TMP=""
  echo "error: failed to move the new binary into place; $SERVICE is stopped" >&2
  echo "  old binary backup: \"$OLD_BINARY\"" >&2
  echo "  systemctl start \"$SERVICE\"" >&2
  exit 1
fi
INSTALL_TMP=""
INSTALLED_SHA="$(lower "$(sha256_file "$INSTALL_PATH")")"
if [ "$INSTALLED_SHA" != "$SHA256_EXPECTED" ]; then
  echo "error: installed binary checksum mismatch: expected $SHA256_EXPECTED, got $INSTALLED_SHA" >&2
  echo "  $SERVICE is stopped; the old binary backup is \"$OLD_BINARY\"" >&2
  exit 1
fi
echo "  installed binary checksum verified"

# ---------------------------------------------------------------------------
# Helpers used after the binary swap.
# ---------------------------------------------------------------------------

get_blocks() {
  curl -fsS -m 5 "$API_BASE/blocks" 2>/dev/null || true
}

# Print "<index> <version>" for the oldest block above the given height.
first_post_upgrade_block() {
  local from="$1" blocks
  if [ -n "$from" ]; then
    blocks="$(curl -fsS -m 5 "$API_BASE/blocks/since/$from" 2>/dev/null || true)"
  else
    blocks="$(get_blocks)"
  fi
  [ -n "$blocks" ] || return 1
  printf '%s' "$blocks" | python3 -c '
import json
import sys

raw = sys.stdin.read()
start = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "" else None
try:
    blocks = json.loads(raw)
except Exception:
    sys.exit(2)
if not isinstance(blocks, list):
    sys.exit(2)
rows = []
for block in blocks:
    if not isinstance(block, dict):
        continue
    header = block.get("header")
    if not isinstance(header, dict):
        continue
    index = header.get("index")
    version = header.get("version")
    if isinstance(index, int) and isinstance(version, int):
        rows.append((index, version))
if start is None:
    rows = [row for row in rows if row[0] > 0]
else:
    rows = [row for row in rows if row[0] > start]
if not rows:
    sys.exit(3)
rows.sort()
print(rows[0][0], rows[0][1])
' "$from"
}

v2_present() {
  local blocks
  blocks="$(get_blocks)"
  [ -n "$blocks" ] || return 1
  printf '%s' "$blocks" | python3 -c '
import json
import sys

try:
    blocks = json.load(sys.stdin)
except Exception:
    sys.exit(1)
if not isinstance(blocks, list):
    sys.exit(1)
for block in blocks:
    if not isinstance(block, dict):
        continue
    header = block.get("header")
    if isinstance(header, dict) and isinstance(header.get("version"), int) and header["version"] >= 2:
        sys.exit(0)
sys.exit(1)
'
}

wait_http() {
  local url="$1" timeout="$2" deadline
  deadline=$((SECONDS + timeout))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -fsS -m 3 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

running_binary_matches() {
  local pid exe expected
  pid="$(systemctl show -p MainPID --value "$SERVICE" 2>/dev/null || true)"
  case "$pid" in
    ''|*[!0-9]*) return 2 ;;
  esac
  [ "$pid" -gt 0 ] || return 2
  [ -d /proc ] || return 2
  [ -r "/proc/$pid/exe" ] || return 2
  exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
  [ -n "$exe" ] || return 2
  case "$exe" in
    *' (deleted)') exe="${exe% (deleted)}" ;;
  esac
  expected="$(readlink -f "$INSTALL_PATH" 2>/dev/null || printf '%s' "$INSTALL_PATH")"
  if [ "$exe" != "$expected" ]; then
    echo "running process $pid uses $exe, expected $expected" >&2
    return 1
  fi
  return 0
}

rollback_help() {
  local in_v2="unknown"
  if v2_present; then
    in_v2="yes"
  else
    in_v2="no"
  fi

  echo
  echo "UPGRADE FAILED — no backup was deleted."
  echo
  if [ "$in_v2" = "yes" ]; then
    echo "A v2 block exists on this node. DO NOT roll back to v0.9.0: it rejects v2."
    echo "Roll forward with v0.10.0 on every node, or restore the whole network from"
    echo "pre-upgrade backups and re-genesis. See docs/upgrade-v2.md section 8."
  elif [ "$FORCE_ROLLBACK" -eq 1 ]; then
    echo "No v2 block was observed. You passed --force-rollback, so the old-binary"
    echo "rollback is:"
    echo
    echo "  systemctl stop \"$SERVICE\""
    echo "  install -m 0755 \"$OLD_BINARY\" \"$INSTALL_PATH\""
    if [ -n "$DB_ARCHIVE" ]; then
      echo "  # only if the database must also be restored (move it aside, do not delete it):"
      echo "  # mv \"$DB\" \"${DB}.failed\""
      echo "  # tar xzf \"$DB_ARCHIVE\" -C \"$(dirname "$DB")\""
      echo "  # chown -R verinoo:verinoo \"$DB\""
    fi
    echo "  systemctl start \"$SERVICE\""
    echo "  curl -fsS \"$API_BASE/ready\""
  else
    echo "No v2 block was observed, but this script will not print v0.9.0 rollback"
    echo "commands unless you explicitly pass --force-rollback. If you are certain"
    echo "that no v2 block exists and the node is not serving traffic, stop the"
    echo "service and restore the saved binary:"
    echo
    echo "  old binary: \"$OLD_BINARY\""
    echo "  install path: \"$INSTALL_PATH\""
    echo
    echo "If you cannot tell whether a v2 block exists, assume it does and do not"
    echo "restart a v0.9.0 validator. Check /blocks for a header version >= 2."
  fi
  echo
  echo "Backups (keep these):"
  echo "  database: \"$DB_ARCHIVE\""
  echo "  binary:   \"$OLD_BINARY\""
  echo
  if command -v journalctl >/dev/null 2>&1; then
    echo "Last $SERVICE log lines:"
    journalctl -u "$SERVICE" -n 40 --no-pager 2>/dev/null || true
  fi
}

echo "→ starting $SERVICE"
if ! systemctl start "$SERVICE"; then
  rollback_help
  exit 1
fi

echo "→ waiting for /health"
if ! wait_http "$API_BASE/health" 60; then
  echo "error: /health did not become reachable at $API_BASE/health" >&2
  rollback_help
  exit 1
fi
echo "  /health: OK"

if ! wait_http "$API_BASE/ready" 120; then
  echo "error: /ready did not return 200 at $API_BASE/ready" >&2
  curl -sS -m 3 "$API_BASE/ready" 2>/dev/null || true
  echo
  rollback_help
  exit 1
fi
echo "  /ready: OK"

# On Linux, confirm the running process is the binary we just installed. If
# /proc or a MainPID is unavailable, skip rather than fail a valid host.
sleep 1
if running_binary_matches; then
  echo "  running binary: $INSTALL_PATH"
else
  running_status=$?
  if [ "$running_status" -eq 2 ]; then
    echo "  warning: could not read the running process path; checksum is verified" >&2
  else
    echo "error: the service is not running the installed binary" >&2
    rollback_help
    exit 1
  fi
fi

echo "→ /v1/models"
if command -v python3 >/dev/null 2>&1; then
  curl -fsS -m 10 "$API_BASE/v1/models" | python3 -m json.tool 2>/dev/null ||
    curl -fsS -m 10 "$API_BASE/v1/models"
else
  curl -fsS -m 10 "$API_BASE/v1/models"
fi
echo

echo "→ checking the first post-upgrade block"
VERSION_OK=0
VERSION_RESULT=""
deadline=$((SECONDS + V2_TIMEOUT))
while [ "$SECONDS" -lt "$deadline" ]; do
  if latest="$(first_post_upgrade_block "$PRE_HEIGHT")"; then
    height="${latest%% *}"
    version="${latest##* }"
    VERSION_RESULT="$latest"
    if [ "$version" -eq 2 ]; then
      VERSION_OK=1
    fi
    break
  fi
  sleep 2
done

if [ "$VERSION_OK" -eq 1 ]; then
  echo "  height $height (above pre-upgrade $PRE_HEIGHT) header version $version"
  echo "✓ upgrade verified: the node is producing/validating v2 blocks"
  echo
  echo "✓ upgrade complete"
  echo "  backups: \"$BACKUP_DIR\" (do not delete)"
  exit 0
fi

echo "error: installed but v2 unconfirmed" >&2
if [ -n "$VERSION_RESULT" ]; then
  echo "  first post-upgrade block: height ${VERSION_RESULT%% *}, header version ${VERSION_RESULT##* } (expected 2)" >&2
else
  echo "  no block above pre-upgrade height $PRE_HEIGHT was observed within ${V2_TIMEOUT} seconds" >&2
fi
echo "  The new binary is installed and $SERVICE is running, but this script cannot prove the chain is on v2." >&2
echo "  Check GET /blocks for the first block above height $PRE_HEIGHT:" >&2
echo "    curl -fsS \"$API_BASE/blocks/since/$PRE_HEIGHT\"" >&2
echo "  Or inspect the journal for produced/applied blocks and version errors:" >&2
echo "    journalctl -u \"$SERVICE\" -n 100 --no-pager | grep -iE 'version|block'" >&2
echo "  Do not start a v0.9.0 validator once a v2 block exists." >&2
rollback_help
exit 1
