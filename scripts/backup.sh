#!/usr/bin/env bash
# Consistent backup of the node database.
#
# RocksDB must not be copied while it is being written, so this stops the
# service, archives the database, restarts it, and prunes old copies.
#
#   sudo ./scripts/backup.sh
#   sudo VNN_DB=/var/lib/verinoo/data BACKUP_DIR=/var/backups/verinoo ./scripts/backup.sh
#
# Restore: stop the service, extract the archive over VNN_DB, start the
# service, then check GET /health and GET /status.
set -euo pipefail

SERVICE="${VNN_SERVICE:-verinoo-node}"
DB="${VNN_DB:-/var/lib/verinoo/data}"
OUT="${BACKUP_DIR:-/var/backups/verinoo}"
KEEP="${KEEP:-7}"
API="${VNN_API_PORT:-7100}"

if [ ! -d "$DB" ]; then
  echo "error: database not found at $DB" >&2
  exit 1
fi

if [ "$(id -u)" -ne 0 ] && systemctl is-active "$SERVICE" >/dev/null 2>&1; then
  echo "error: run as root so the service can be stopped safely" >&2
  exit 1
fi

mkdir -p "$OUT"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
ARCHIVE="$OUT/verinoo-$STAMP.tar.gz"
WAS_ACTIVE=0
if systemctl is-active "$SERVICE" >/dev/null 2>&1; then
  WAS_ACTIVE=1
  echo "→ stopping $SERVICE"
  systemctl stop "$SERVICE"
fi

cleanup() {
  if [ "$WAS_ACTIVE" -eq 1 ] && ! systemctl is-active "$SERVICE" >/dev/null 2>&1; then
    echo "→ restarting $SERVICE"
    systemctl start "$SERVICE"
  fi
}
trap cleanup EXIT

echo "→ archiving $DB"
tar czf "$ARCHIVE" -C "$(dirname "$DB")" "$(basename "$DB")"
sha256sum "$ARCHIVE" >"$ARCHIVE.sha256"
echo "wrote $ARCHIVE"
cat "$ARCHIVE.sha256"

if [ "$WAS_ACTIVE" -eq 1 ]; then
  systemctl start "$SERVICE"
  trap - EXIT
  echo "→ waiting for /health"
  for _ in $(seq 1 30); do
    if curl -fsS -m 1 "http://127.0.0.1:$API/health" >/dev/null 2>&1; then
      echo "✓ node healthy"
      break
    fi
    sleep 1
  done
fi

# Keep the newest N archives (and their checksums).
ls -1t "$OUT"/verinoo-*.tar.gz 2>/dev/null | tail -n +"$((KEEP + 1))" | while read -r old; do
  echo "pruning $old"
  rm -f "$old" "$old.sha256"
done
