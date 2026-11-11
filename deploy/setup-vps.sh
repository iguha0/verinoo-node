#!/usr/bin/env bash
# Verinoo testnet node — VPS bootstrap.
# Run from a checkout: sudo REPO_DIR=/opt/verinoo ./deploy/setup-vps.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVICE_USER="${SERVICE_USER:-verinoo}"

if [[ $EUID -ne 0 ]]; then
  echo "error: run as root (needed to install the service and firewall rules)" >&2
  exit 1
fi

echo "==> Verinoo Rust testnet VPS setup"
echo "    repo: $REPO_DIR"

apt-get update -qq
apt-get install -y -qq git curl nginx ufw rsync build-essential clang libclang-dev >/dev/null

if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y >/dev/null 2>&1
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi

echo "==> Creating service user + dirs"
id -u "$SERVICE_USER" &>/dev/null ||
  useradd -r -m -d "/opt/$SERVICE_USER" -s /usr/sbin/nologin "$SERVICE_USER"
mkdir -p /var/lib/verinoo /etc/verinoo /var/www/verinoo
chown -R "$SERVICE_USER:$SERVICE_USER" /var/lib/verinoo

if [[ ! -f "$REPO_DIR/Cargo.toml" ]]; then
  echo "error: no Rust workspace at $REPO_DIR (clone the repo there first)" >&2
  exit 1
fi

echo "==> Building verinoo-node (release, locked)"
(cd "$REPO_DIR" && cargo build --release --locked)

install -m 0755 "$REPO_DIR/target/release/verinoo-node" /usr/local/bin/verinoo-node
/usr/local/bin/verinoo-node wallet >/dev/null || {
  echo "error: built binary does not run" >&2
  exit 1
}

echo "==> Installing systemd unit + env"
if [[ ! -f /etc/verinoo/node.env ]]; then
  install -m 0600 "$REPO_DIR/deploy/node.env.example" /etc/verinoo/node.env
  echo "    wrote /etc/verinoo/node.env — EDIT IT (unique VNN_KEY_SEED, VNN_VALIDATORS)"
else
  echo "    keeping existing /etc/verinoo/node.env"
fi
install -m 0644 "$REPO_DIR/deploy/verinoo-node.service" /etc/systemd/system/verinoo-node.service
chown -R "$SERVICE_USER:$SERVICE_USER" /etc/verinoo

echo "==> Firewall"
ufw allow 22/tcp >/dev/null
ufw allow 80/tcp >/dev/null
ufw allow 443/tcp >/dev/null
ufw allow 7101/tcp >/dev/null # p2p
ufw --force enable >/dev/null

echo "==> Nginx"
sed "s/DOMAIN/${DOMAIN:-_}/" "$REPO_DIR/deploy/nginx-verinoo.conf" >/etc/nginx/sites-available/verinoo
ln -sf /etc/nginx/sites-available/verinoo /etc/nginx/sites-enabled/verinoo
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx

systemctl daemon-reload
systemctl enable verinoo-node

cat <<'EOF'

==> Done. Remaining manual steps:
  1. Edit /etc/verinoo/node.env:
       VNN_KEY_SEED=<unique random string, never reuse across nodes>
       VNN_VALIDATORS=<comma-separated ai_ addresses of the whole set>
       VNN_SEEDS=/ip4/<peer-ip>/tcp/7101
  2. systemctl restart verinoo-node
  3. curl -fsS http://127.0.0.1:7100/health
  4. tail -f journalctl -u verinoo-node
EOF
