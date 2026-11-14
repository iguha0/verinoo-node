# Operations runbook

Scope: a single-validator testnet or a small dedicated pilot network. This is
the operational layer around `verinoo-node`; it does not claim BFT or HA.

## Layout on a VPS

| Path | Purpose |
|---|---|
| `/usr/local/bin/verinoo-node` | binary |
| `/opt/verinoo` | checkout (models in `models/`, deploy assets) |
| `/etc/verinoo/node.env` | environment (`VNN_*`), mode `0600` |
| `/etc/verinoo/api-keys.json` | tenant keys (hashes only), optional |
| `/var/lib/verinoo/data` | RocksDB state |
| `/etc/systemd/system/verinoo-node.service` | unit |
| `/etc/nginx/sites-available/ainative` | TLS + reverse proxy (this host) |

`deploy/setup-vps.sh` installs the unit, env file and nginx site on Ubuntu
24.04. The live host serves `/v1/`, `/faucet` and `/account/` from the Rust
node and leaves the older reference node on `/status` and `/health`.

## Deploy or upgrade

```bash
cd /opt/verinoo
git fetch --tags && git checkout v0.3.0-testnet
cargo build --release --locked -p verinoo-node
install -m 0755 target/release/verinoo-node /usr/local/bin/verinoo-node
systemctl restart verinoo-node
curl -fsS localhost:7100/health
curl -fsS localhost:7100/v1/models
```

State is forward-compatible within a release; take a backup before upgrading.

## Backups

RocksDB must not be copied while it is being written. `scripts/backup.sh`
stops the service, archives the database, restarts it, and prunes old copies.

```bash
sudo VNN_DB=/var/lib/verinoo/data BACKUP_DIR=/var/backups/verinoo KEEP=14 \
  /opt/verinoo/scripts/backup.sh
```

Suggested cron (03:17 UTC daily):

```cron
17 3 * * * root VNN_DB=/var/lib/verinoo/data BACKUP_DIR=/var/backups/verinoo KEEP=14 /opt/verinoo/scripts/backup.sh >>/var/log/verinoo-backup.log 2>&1
```

Restore:

```bash
systemctl stop verinoo-node
tar xzf /var/backups/verinoo/verinoo-YYYYMMDD-HHMMSS.tar.gz -C /var/lib/verinoo
chown -R verinoo:verinoo /var/lib/verinoo/data
systemctl start verinoo-node
curl -fsS localhost:7100/health
```

Also back up `/etc/verinoo` and `/opt/verinoo/models`; without the same model set
the state root diverges and the node refuses peer blocks.

## API key rotation (no downtime)

Keys are address-bound, stored as SHA-256 hashes, and reloaded with SIGHUP.

```bash
verinoo-node apikey new --tenant acme --address ai_... --quota 5000 \
  --file /etc/verinoo/api-keys.json
systemctl kill -s HUP verinoo-node          # reload without restart

verinoo-node apikey list --file /etc/verinoo/api-keys.json
verinoo-node apikey revoke --tenant oldco --file /etc/verinoo/api-keys.json
systemctl kill -s HUP verinoo-node
```

`--quota 0` means unlimited. Quotas are gateway-local and reset per UTC day.
They are a billing/abuse control, not a consensus rule.

## Input encryption key

Inputs are encrypted to one network X25519 key so blocks and API responses carry
ciphertext. The key is validator-confidential, not operator-blind.

```bash
openssl rand -hex 32          # write to VNN_INPUT_KEY on every validator
curl -fsS localhost:7100/v1/input-key
```

- Back up the key with the database. Without it, encrypted tasks cannot be
  re-executed or verified.
- Add `VNN_REQUIRE_ENCRYPTED_INPUT=true` to reject plaintext submissions on a
  paid deployment.
- Rotate only after draining pending encrypted tasks, then restart every
  validator with the new key. There is no online rotation or re-encryption.
- Pre-encryption tasks and blocks remain plaintext permanently.

Full threat model: [`docs/privacy.md`](privacy.md).

## Block time and latency

`VNN_BLOCK_TIME_MS` overrides the chain's default interval and must be identical
on every validator. A single-validator demo at `1000` gives roughly 1–2 s task
latency; the two- and multi-validator safety story is unchanged (still
round-robin, no BFT). Changing it does not change state.

## SDKs

- `sdk/typescript` — zero-dependency Node >= 20 client.
- `sdk/python` — `cryptography` + stdlib client.

Both build canonical transactions, sign with Ed25519, encrypt inputs to
`/v1/input-key`, and poll/verify tasks. Use them instead of hand-rolling the
canonical JSON.

## Validator key rotation (coordinated)

There is no live validator-set governance yet. `VNN_VALIDATORS` is seeded into
state at startup and the set only grows; removing an old validator requires a
coordinated restart and, today, a re-genesis. For a pilot:

1. Derive the new address: `verinoo-node address --seed <new-seed>`.
2. Add it to `VNN_VALIDATORS` on **every** node.
3. Restart the new node and let it sync.
4. Do not retire the old node until the old address is removed from the set in
   a future release.

Treat the validator seed as a long-lived secret; store it in the host secret
manager, not in shell history.

## Catch-up replica / HA recipe

Catch-up sync lets a node replay blocks it missed. A read replica can run in
front of the API:

```ini
# /etc/verinoo/node.env on the replica
VNN_VALIDATOR=false
VNN_VALIDATORS=ai_<validator>
VNN_SYNC_URLS=http://10.0.0.10:7100
VNN_API_KEYS_FILE=/etc/verinoo/api-keys.json
```

The replica validates every replayed block and refuses a divergent state root.
Point a load balancer health check at `GET /health` and send reads to either
node. Promotion is manual: stop the old validator, start the replica with the
validator seed and `VNN_VALIDATOR=true`, and update `VNN_SYNC_URLS` on the other
nodes. There is no automatic failover and no split-brain protection.

## Monitoring

| Signal | Where | Alarm |
|---|---|---|
| liveness | `GET /health` (`ok`, `height`, `headAgeMs`) | `ok=false` or head age > 3× block time |
| chain | `GET /status` | height not advancing |
| mempool | `GET /metrics` | sustained growth |
| service | `systemctl status verinoo-node` | restarts |
| disk | `df -h /var/lib/verinoo` | < 20% free |
| sync | journal `synced N block(s)` | repeated failures on a replica |

## Incident runbook

**No blocks / head age growing.**
1. `systemctl status verinoo-node`, `journalctl -u verinoo-node -n 100`.
2. Check `VNN_VALIDATORS` matches every peer and the validator is in its slot.
3. Multi-validator: confirm peer count is non-zero (`/status` or logs). An
   isolated validator intentionally does not produce.
4. Restart; if the node is behind, a replica with `VNN_SYNC_URLS` catches up.

**`state root mismatch`.**
The node derived a different state root than the producer. Almost always a
different or missing model file. Compare `/v1/models` and `modelSetHash` across
nodes, fix `VNN_MODEL_DIR`, restart. Do not delete the database to "fix" it.

**`wrong producer for height`.**
The validator sets differ. Make `VNN_VALIDATORS` identical on every node.

**Disk full.**
Stop the node, move `/var/lib/verinoo/data` to a larger volume (or restore the
latest backup), start, verify height. Prune old backups.

**Suspected key compromise.**
API key: revoke and SIGHUP. Validator key: stop the node, rotate per the
coordinated procedure above; assume an attacker can produce until the old
address is removed from the set.

## Security notes

- UFW should allow only 22/80/443; 7100 and 7101 stay private.
- Run as the unprivileged `verinoo` user; the unit already sets
  `NoNewPrivileges`, `ProtectSystem=full`, `ProtectHome` and `PrivateTmp`.
- `/etc/verinoo/node.env` is `0600`; never commit it.
- Nginx rate-limits `/v1/` and `/faucet`.
- The public demo intentionally has no API key; enable `VNN_API_KEYS_FILE` for
  a paid or private deployment.
