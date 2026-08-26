#!/usr/bin/env bash
set -euo pipefail
echo "==> Verinoo Rust testnet VPS setup"
echo "  Requires: Ubuntu 24.04, root access"
echo ""

apt-get update -qq
apt-get install -y -qq git curl nginx ufw rsync build-essential > /dev/null

# Node.js for static site serving (optional)
curl -fsSL https://deb.nodesource.com/setup_22.x | bash - > /dev/null 2>&1
apt-get install -y -qq nodejs > /dev/null

# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y > /dev/null 2>&1
source "$HOME/.cargo/env"

echo "==> Creating service user + dirs"
id -u verinoo &>/dev/null || useradd -r -m -d /opt/verinoo -s /usr/sbin/nologin verinoo
mkdir -p /var/lib/verinoo /etc/verinoo /var/www/verinoo
chown -R verinoo:verinoo /opt/verinoo /var/lib/verinoo

echo "==> Building verinoo-node"
cd /opt/verinoo
cargo build --release 2>/dev/null || true
cp target/release/verinoo-node /usr/local/bin/

echo "==> Firewall"
ufw allow 22/tcp && ufw allow 80/tcp && ufw allow 443/tcp && ufw allow 7101/tcp
ufw --force enable

echo "==> Done. Next steps:"
echo "  1. cp deploy/node.env.example /etc/verinoo/node.env"
echo "  2. Edit env with your key seed + peer addresses"
echo "  3. cp deploy/verinoo-node.service /etc/systemd/system/"
echo "  4. systemctl daemon-reload && systemctl enable --now verinoo-node"
echo "  5. Configure nginx with deploy/nginx-verinoo.conf"
