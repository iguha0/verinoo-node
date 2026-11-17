#!/usr/bin/env bash
# Operator helper for the v0.10.0-testnet protocol-v2 upgrade.
#
#   sudo scripts/upgrade-v2.sh --binary /root/verinoo-node-v0.10.0-testnet
#   sudo scripts/upgrade-v2.sh --binary /root/verinoo-node-v0.10.0-testnet \
#     --service verinoo-node --db /var/lib/verinoo/data \
#     --backup-dir /root/verinoo-backups/20261118-120000
#
# The script preflights the host, backs up the database and old binary, stops
# the service, installs the new binary, waits for /health and /ready, prints
# /v1/models, and checks the first post-upgrade block's header version.
#
# It does not upgrade peers, change VNN_VALIDATORS, fence a host, promote a
# replica or roll back automatically. Backups are never deleted. Read
# docs/upgrade-v2.md before running it.
set -euo pipefail

SERVICE="verinoo-node"
DB="/var/lib/verinoo/data"
BACKUP_DIR=""
NEW_BINARY=""
API_PORT="${VNN_API_PORT:-7100}"
API_BASE="http://127.0.0.1:${API_PORT}"
DEFAULT_INSTALL="/usr/local/bin/verinoo-node"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_ARCHIVE=""
INSTALL_TMP=""

usage() {
  cat >&2 <<'EOF'
usage: scripts/upgrade-v2.sh --binary <new-binary>
       [--service <unit>] [--db <path>] [--backup-dir <path>]

  --binary      new verinoo-node binary (required)
  --service     systemd unit (default: verinoo-node)
  --db          RocksDB path (default: /var/lib/verinoo/data)
  --backup-dir  backup directory (default: /root/verinoo-backups/<timestamp>)

Run as root on the server. The node must be live and /health must answer
before the upgrade starts. VNN_API_PORT overrides the local API port (7100).
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

for cmd in systemctl curl tar install df du; do
  command -v "$cmd" >/dev/null 2>&1 || die "required command not found: $cmd"
done

[ -f "$NEW_BINARY" ] || die "new binary not found: $NEW_BINARY"
[ -x "$NEW_BINARY" ] || die "new binary is not executable: $NEW_BINARY"
NEW_BINARY="$(cd "$(dirname "$NEW_BINARY")" && pwd)/$(basename "$NEW_BINARY")"

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
echo "  /health: OK at $API_BASE"

PRE_HEALTH="$(curl -fsS -m 5 "$API_BASE/health" 2>/dev/null || true)"
PRE_HEIGHT="$(printf '%s' "$PRE_HEALTH" |
  grep -o '"height"[[:space:]]*:[[:space:]]*[0-9]*' |
  head -n 1 | grep -o '[0-9]*$' || true)"
if [ -n "$PRE_HEIGHT" ]; then
  echo "  pre-upgrade height: $PRE_HEIGHT"
else
  echo "  warning: could not read the pre-upgrade height from /health"
fi

DB_KB="$(du -sk "$DB" | awk '{print $1}')"
AVAIL_KB="$(df -Pk "$BACKUP_DIR" | awk 'NR==2 {print $4}')"
NEED_KB=$((DB_KB + DB_KB / 5 + 262144))
if [ "$AVAIL_KB" -lt "$NEED_KB" ]; then
  die "not enough free space: need ~${NEED_KB} KiB for database backup + 256 MiB slack, have ${AVAIL_KB} KiB at $BACKUP_DIR"
fi
echo "  disk: OK (${AVAIL_KB} KiB free, need ~${NEED_KB} KiB)"

EXEC_START="$(systemctl show -p ExecStart --value "$SERVICE" 2>/dev/null || true)"
INSTALL_PATH="$(printf '%s' "$EXEC_START" |
  sed -n 's/.*path=\([^ ;]*\).*/\1/p' | head -n 1)"
if [ -z "$INSTALL_PATH" ]; then
  INSTALL_PATH="$DEFAULT_INSTALL"
fi
[ -f "$INSTALL_PATH" ] || die "current binary not found at $INSTALL_PATH"
[ "$NEW_BINARY" != "$INSTALL_PATH" ] || die "new binary is already the install target: $INSTALL_PATH"
echo "  install:  $INSTALL_PATH"

OLD_BINARY="$BACKUP_DIR/verinoo-node.pre-upgrade"
cp -p "$INSTALL_PATH" "$OLD_BINARY"
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}
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
  echo "  systemctl start $SERVICE" >&2
  exit 1
fi
if ! mv -f "$INSTALL_TMP" "$INSTALL_PATH"; then
  echo "error: failed to move the new binary into place; $SERVICE is stopped" >&2
  echo "  old binary backup: $OLD_BINARY" >&2
  echo "  systemctl start $SERVICE" >&2
  exit 1
fi
INSTALL_TMP=""

# ---------------------------------------------------------------------------
# Helpers used after the binary swap.
# ---------------------------------------------------------------------------

get_blocks() {
  curl -fsS -m 5 "$API_BASE/blocks" 2>/dev/null || true
}

latest_index_version() {
  local blocks idx ver
  blocks="$(get_blocks)"
  [ -n "$blocks" ] || return 1
  idx="$(printf '%s' "$blocks" |
    grep -o '"index"[[:space:]]*:[[:space:]]*[0-9]*' |
    head -n 1 | grep -o '[0-9]*$' || true)"
  ver="$(printf '%s' "$blocks" |
    grep -o '"version"[[:space:]]*:[[:space:]]*[0-9]*' |
    head -n 1 | grep -o '[0-9]*$' || true)"
  [ -n "$idx" ] && [ -n "$ver" ] || return 1
  printf '%s %s\n' "$idx" "$ver"
}

v2_seen() {
  local blocks
  blocks="$(get_blocks)"
  [ -n "$blocks" ] || return 1
  printf '%s' "$blocks" | grep -Eq '"version"[[:space:]]*:[[:space:]]*2([^0-9]|$)'
}

wait_http() {
  local url="$1" timeout="$2" deadline body
  deadline=$((SECONDS + timeout))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if body="$(curl -fsS -m 3 "$url" 2>/dev/null)"; then
      printf '%s' "$body"
      return 0
    fi
    sleep 2
  done
  return 1
}

rollback_help() {
  local in_v2="no"
  if v2_seen; then
    in_v2="yes"
  fi

  echo
  echo "UPGRADE FAILED — no backup was deleted."
  echo
  if [ "$in_v2" = "yes" ]; then
    echo "A v2 block exists on this node. DO NOT roll back to v0.9.0: it rejects v2."
    echo "Roll forward with v0.10.0 on every node, or restore the whole network from"
    echo "pre-upgrade backups and re-genesis. See docs/upgrade-v2.md section 8."
  else
    echo "No v2 block was observed. If the first v2 block has not been produced,"
    echo "the exact rollback is:"
    echo
    echo "  systemctl stop $SERVICE"
    echo "  install -m 0755 $OLD_BINARY $INSTALL_PATH"
    if [ -n "$DB_ARCHIVE" ]; then
      echo "  # only if the database must also be restored (move it aside, do not delete it):"
      echo "  # mv $DB ${DB}.failed"
      echo "  # tar xzf $DB_ARCHIVE -C $(dirname "$DB")"
      echo "  # chown -R verinoo:verinoo $DB"
    fi
    echo "  systemctl start $SERVICE"
    echo "  curl -fsS $API_BASE/ready"
    echo
    echo "If you cannot tell whether a v2 block exists, assume it does and do not"
    echo "restart a v0.9.0 validator. Check /blocks for \"version\": 2."
  fi
  echo
  echo "Backups (keep these):"
  echo "  database: $DB_ARCHIVE"
  echo "  binary:   $OLD_BINARY"
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
if ! wait_http "$API_BASE/health" 60 >/dev/null; then
  echo "error: /health did not become reachable at $API_BASE/health" >&2
  rollback_help
  exit 1
fi
echo "  /health: OK"

echo "→ waiting for /ready"
if ! wait_http "$API_BASE/ready" 120 >/dev/null; then
  echo "error: /ready did not return 200 at $API_BASE/ready" >&2
  curl -sS -m 3 "$API_BASE/ready" 2>/dev/null || true
  echo
  rollback_help
  exit 1
fi
echo "  /ready: OK"

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
if [ -n "$PRE_HEIGHT" ]; then
  deadline=$((SECONDS + 90))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if latest="$(latest_index_version)"; then
      height="${latest%% *}"
      version="${latest##* }"
      if [ "$height" -gt "$PRE_HEIGHT" ]; then
        echo "  height $height header version $version"
        if [ "$version" -eq 2 ]; then
          VERSION_OK=1
        fi
        break
      fi
    fi
    sleep 2
  done
else
  if latest="$(latest_index_version)"; then
    height="${latest%% *}"
    version="${latest##* }"
    echo "  latest height $height header version $version"
    if [ "$version" -eq 2 ]; then
      VERSION_OK=1
    fi
  fi
fi

if [ "$VERSION_OK" -eq 1 ]; then
  echo "✓ upgrade verified: the node is producing/validating v2 blocks"
else
  echo "warning: could not confirm a new v2 block yet" >&2
  echo "  Check GET /blocks for the first block above the pre-upgrade height:" >&2
  echo "    curl -fsS $API_BASE/blocks" >&2
  echo "  Or inspect the journal for produced/applied blocks and version errors:" >&2
  echo "    journalctl -u $SERVICE -n 100 --no-pager | grep -iE 'version|block'" >&2
  echo "  Do not start a v0.9.0 validator once a v2 block exists." >&2
fi

echo
echo "✓ upgrade complete"
echo "  backups: $BACKUP_DIR (do not delete)"
