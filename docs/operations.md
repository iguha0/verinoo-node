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
git fetch --tags && git checkout v0.9.0-testnet
cargo build --release --locked -p verinoo-node
install -m 0755 target/release/verinoo-node /usr/local/bin/verinoo-node
systemctl restart verinoo-node
curl -fsS localhost:7100/health
curl -fsS localhost:7100/ready
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
curl -fsS localhost:7100/ready
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

## Self-serve API keys

Set `VNN_SELF_SERVE_KEYS=1` to let clients issue their own address-bound key:

```bash
curl -s -X POST localhost:7100/v1/keys \
  -H 'content-type: application/json' -d '{"address":"ai_..."}'
```

| Variable | Default | Notes |
|---|---|---|
| `VNN_SELF_SERVE_KEYS` | unset (`404`) | enable `POST /v1/keys` |
| `VNN_SELF_SERVE_QUOTA` | `1000` | tasks per UTC day; `0` = unlimited |

- The body must be `{"address":"ai_<40 hex>"}`; anything else is `400`.
- The response includes the raw `apiKey` **once**. Only its SHA-256 hash is
  stored, in RocksDB meta `selfserve:key:<sha256hex>`; the address index is
  `selfserve:addr:<address>`. Nothing is written to `/etc`.
- One active key per address. A second request returns `409` with the existing
  tenant (`self-serve`) and address, never the raw key.
- The key only authorizes API access for a signer that already holds the
  address's private key: `tx.from` must match the bound address.
- Quota is per address per UTC day, so keys cannot multiply an allowance.
- Issuance is rate-limited best-effort to **5 requests per hour per IP**
  (`429` with `retryAfterSecs`). The window is in memory and resets on restart;
  behind the proxy the limiter honours `X-Forwarded-For` / `X-Real-IP`.
- Self-serve keys live in the database and are **not** listed by
  `verinoo-node apikey list`, which only reads the operator key file. They are
  included in normal RocksDB backups. Unset `VNN_SELF_SERVE_KEYS` to stop new
  issuance; existing self-serve keys stop authorizing while it is off.

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

## Request policy

`VNN_POLICY_FILE` points at the deterministic plaintext policy described in
[`docs/policy.md`](policy.md). The node loads it once at startup:

```bash
# /etc/verinoo/policy.json
{
  "maxInputBytes": 8192,
  "allowedModels": ["intent-classifier-v1", "tiny-gpt-v1"],
  "blockedTerms": ["confidential-project"],
  "blockedPatterns": ["(?i)account\\s+number"],
  "requirePlaintext": false
}
```

- A missing or invalid file fails startup. There is no implicit fallback to a
  permissive policy; fix the file or unset the variable deliberately.
- Policy changes require a restart. The file is not reloaded on SIGHUP.
- Plaintext `input` is inspected for blocked content. `maxInputBytes` applies
  to plaintext and to the serialized `encryptedInput` envelope, so ciphertext
  cannot bypass the byte limit. Encrypted and commitment-only submissions are
  not content-searched: with `requirePlaintext: false` they pass model/size
  checks, with `requirePlaintext: true` they are rejected.
- Rejections are `422` with `reason: "policy"` and a stable `policyReason`
  (`blocked_term`, `blocked_pattern`, `model_not_allowed`, `input_too_long`,
  `encrypted_input`). Blocked terms are never echoed in responses or logs.
- If redaction would change the input, the node rejects with `422` /
  `redaction_required` and asks the client to redact before signing. It never
  mutates a signed transaction.
- Policy is a submission-boundary control. It does not run in consensus, does
  not alter block validity, results, receipts, or verification.

## Batch and task event streams

- `POST /v1/infer/batch` accepts 1–32 signed transaction envelopes and applies
  the same per-item checks as `/v1/infer`. The 32-item count is a transport
  cap, not the real capacity: the whole batch must fit one block's gas
  (`1000`). `submitInference` costs 60 gas (120 for `sampled`, 240 for
  `full-zk`), so at most 16 optimistic inference items fit in one batch. A
  batch over the gas limit is rejected with `400` before any item is admitted.
- Nonce and balance use a batch-local overlay seeded from committed state, so
  consecutive nonces from one sender are admitted in order; the overlay
  advances only for accepted items. A rejected middle item therefore cascades:
  later items from the same sender fail their nonce check until the missing
  nonce is submitted. Other senders are independent.
- After request-level checks (shape, size, auth, gas capacity), the endpoint
  returns `200` with per-item results even if every item is rejected. Accepted
  items are pushed and gossiped; rejected items are reported by index.
  Admission is not a reservation: block production re-validates against
  evolving state. Empty, oversized or over-gas batches are `400`; an
  unauthenticated request is `401`.
- The mempool admits at most **5000 transactions** across RPC and gossip. A
  single submission returns `429` when full; a batch reports a per-item
  `mempool is full` error and still returns `200`.
- Gas-limited transactions left out of a produced block are re-queued at the
  front of the mempool in their original order; invalid or reverted
  transactions are dropped.
- `GET /v1/tasks/:id/events` is a Server-Sent Events stream with `status`,
  `result`, `done`, and `timeout` events. It polls about once per second and
  ends after 60 seconds. The task row is created only when the block carrying
  the `submitInference` is applied, so both the task read and the stream return
  `404` until then; retry after the transaction is included.
- At most **64 concurrent event streams** are allowed per process. A request
  beyond that limit returns `429`; the slot is released when the stream ends.
- The stream exposes only committed task state and receipts. It does not emit
  token-level previews; unverified execution previews are not implemented.
- SSE needs proxy buffering disabled (for example nginx
  `proxy_buffering off;` on that location) if you put it behind an external
  proxy.

## Block time and latency

`VNN_BLOCK_TIME_MS` overrides the chain's default interval and must be identical
on every validator. A single-validator demo at `1000` gives roughly 1–2 s task
latency; the two- and multi-validator safety story is unchanged (still
round-robin, no BFT). Changing it does not change state. The same effective
interval drives the `/ready` staleness threshold.

## Readiness (`/ready`)

`GET /ready` is the load-balancer and promotion probe. It returns HTTP 200 only
when all of these hold:

- `serving`: a head exists.
- `validatorSetConsistent`: the state-derived active validator set matches the
  genesis/`VNN_VALIDATORS` set.
- `syncing` is false: the node is not applying catch-up blocks.
- `headAgeMs < 3 * blockTimeMs`, using `VNN_BLOCK_TIME_MS` or the genesis
  default.
- When a peer height is known, `height + VNN_PEER_LAG_BLOCKS >= peerHeight`.

Otherwise it returns HTTP 503 with a stable `reason`:

| `reason` | Meaning |
|---|---|
| `no_head` | The node has no head block yet. |
| `validator_set_mismatch` | The state-derived set differs from `VNN_VALIDATORS`. |
| `syncing` | The node is applying catch-up blocks. |
| `stale_head` | The head is at least three block intervals old. |
| `behind_peer` | The node is more than `VNN_PEER_LAG_BLOCKS` behind the best peer. |

```json
{
  "ready": false,
  "serving": true,
  "height": 12345,
  "headAgeMs": 4100,
  "peerCount": 2,
  "syncing": false,
  "validatorSetHash": "613d100e31f0fdbd765986c9544186b43dd5b8652b1eafefc0ad545efb9d8ecb",
  "validatorSetConsistent": true,
  "reason": "behind_peer"
}
```

- `peerCount` is the live libp2p connection count (0 on an isolated or
  single-validator node).
- Peer height comes from the `/status` response of the URLs in
  `VNN_SYNC_URLS`. A peer is only used when its `chainId` matches this node.
  The best height is recomputed on every sync poll (every 3 s) from the valid
  peers that respond; if none respond it is cleared. A stale or wrong-chain
  height therefore cannot keep `/ready` in `behind_peer` indefinitely. A node
  without `VNN_SYNC_URLS` (for example the live single validator) has no peer
  height, so the `behind_peer` condition is skipped rather than fabricated.
- `syncing` is true only while replayed catch-up blocks are being applied. The
  routine `/status` poll and block fetch do not set it. The sync HTTP client
  has a 2 s connect timeout, 5 s read timeout, and 5 s total request timeout,
  so a silent peer cannot hold the sync loop open.
- `validatorSetHash` is lowercase hex SHA-256 over the canonical JSON of the
  active validator addresses sorted and deduplicated (for example
  `["ai_a","ai_b"]`, not newline-joined). It is derived from `val:` state when
  non-empty, otherwise from the genesis/`VNN_VALIDATORS` list. Addresses only;
  public keys and stake are not included. This is an observability hash: it is
  not part of the state root and does not make set changes consensus state.
- `validatorSetConsistent` is false when the state-derived set differs from the
  configured set. `/ready` fails closed with `validator_set_mismatch`; the node
  does not change consensus membership or rewrite state to "fix" it. Compare
  the value across nodes and fix `VNN_VALIDATORS` or the database.
- `/status` exposes the same `syncing`, `validatorSetHash`, and
  `validatorSetConsistent` values alongside its existing fields.
- `GET /health` remains a liveness probe: `ok` only means a head exists. Do not
  route traffic on `ok` alone.
- `deploy/nginx-verinoo.conf` proxies `/ready`; add the same `location` to the
  live nginx site before pointing an external load balancer at it.

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
Point a load balancer health check at `GET /ready` and send reads only to a node
that returns 200. A replica returns 503 while it is syncing, stale, behind the
best peer, or has a divergent validator set. Promotion is manual: stop the old validator, start the replica
with the validator seed and `VNN_VALIDATOR=true`, and update `VNN_SYNC_URLS` on
the other nodes. There is no automatic failover and no split-brain protection.

## Monitoring

| Signal | Where | Alarm |
|---|---|---|
| readiness | `GET /ready` | HTTP 503 or a non-null `reason` |
| liveness | `GET /health` (`ok`, `height`, `headAgeMs`) | `ok=false` |
| chain | `GET /status` (`height`, `syncing`, `validatorSetHash`, `validatorSetConsistent`) | height not advancing, unexpected `syncing`, hash differs across nodes, or `validatorSetConsistent=false` |
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
