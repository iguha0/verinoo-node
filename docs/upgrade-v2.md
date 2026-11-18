# Upgrade runbook: protocol v2 (v0.10.0-testnet)

Scope: move a verinoo testnet node from v0.9.0 (header version 1) to
v0.10.0-testnet (header version 2). This is a **coordinated protocol upgrade**,
not a rolling-compatible minor release.

- v0.10.0 accepts and replays historical v1 blocks.
- v0.10.0 produces v2 blocks.
- v0.9.0 rejects v2 blocks (`unknown block version 2`).
- A v1 block may not contain a `sampling` transaction; sampling is v2-only.

The supported upgrade helper is `scripts/upgrade-v2.sh`. It is an operator aid,
not an orchestrator: it does not upgrade peers, fence hosts, promote replicas or
roll back automatically.

## 1. Who must upgrade

| Node role | Required action |
|---|---|
| Validator (single-validator testnet) | Upgrade before restarting production. |
| Validator (multi-validator) | Stop all validators, upgrade all, restart together. |
| Read replica / API node | Upgrade before or with the validators; it accepts v1 until v2 appears. |
| Archival / observer node | Upgrade before the first v2 block if it must follow the head. |

Do not rely on a mixed-version window. Upgrade **all validators and replicas**
before any upgraded validator produces a block. There is no automatic failover
and no split-brain protection: a v0.9.0 validator that keeps producing after a
v0.10.0 validator emits v2 creates two incompatible heads.

## 2. What changes

| Area | v0.9.0 | v0.10.0 |
|---|---|---|
| Block header | `version: 1` | `version: 2` (v1 replay still accepted) |
| Per-task sampling | not available | signed `sampling` object; v2 gate |
| Receipts | historical shape | adds `sampling`, `samplingHash`, `samplingSource` only for overrides |
| Batch API | no | `/v1/infer/batch`, 1–32 items, per-item results |
| Task events | polling | final-only SSE with a 60 s cap and 64-stream limit |
| Policy | no | `VNN_POLICY_FILE` submission gate |
| Snapshots | no | signed operator `snapshot create/verify/apply` CLI, state-layer library, read-only serving |
| Runtime | owned `.vnnw` loading | mmap `.vnnw`, generic GPT-2/GPT-Neo importer |

Older tasks, receipts and v1 blocks without sampling remain valid and keep
their historical encoding.

## 3. Preflight

Run these on every node before changing anything.

```bash
# 1. Identify the installed binary and service. This tree has no
#    `verinoo-node --version`; use the unit's ExecStart path and its checksum.
systemctl status verinoo-node --no-pager
systemctl show -p ExecStart --no-pager verinoo-node
journalctl -u verinoo-node -n 50 --no-pager

# 2. The node must be live and ready, not syncing or stale. /health is liveness
#    only and returns 200 with "ok": false before a head exists; /ready is the
#    gate.
curl -fsS http://127.0.0.1:7100/health
curl -fsS http://127.0.0.1:7100/ready
curl -fsS http://127.0.0.1:7100/status

# 3. Record the pre-upgrade head. You will compare it after restart.
curl -fsS http://127.0.0.1:7100/blocks |
  python3 -c 'import json,sys; b=json.load(sys.stdin); print(b[0]["header"]["index"], b[0]["header"]["version"])'

# 4. Check disk space. The database plus its backup needs room.
df -h /var/lib/verinoo /root
du -sh /var/lib/verinoo/data

# 5. Confirm the model set and config are identical on all nodes.
curl -fsS http://127.0.0.1:7100/v1/models
diff -u /etc/verinoo/node.env /path/to/reference/node.env
```

Stop if `/ready` is not 200. Fix the reason first (`no_head`,
`validator_set_mismatch`, `syncing`, `stale_head`, `behind_peer`). Upgrading a
node that is already unhealthy makes the failure harder to attribute.

Confirm the new binary is the expected x86_64 Linux build and executable:

```bash
file /root/verinoo-node-v0.10.0-testnet
ls -l /root/verinoo-node-v0.10.0-testnet
sha256sum /root/verinoo-node-v0.10.0-testnet
# Compare the hash with the value you pass to --sha256/--sums-file.
# The release sums file covers the archive; verify it first, then hash the
# extracted bin/verinoo-node.
```

## 4. Backup and optional snapshot

The database must not be copied while RocksDB is writing. The supported backup
is `scripts/backup.sh`, which stops the service, archives `VNN_DB`, restarts the
service, and writes a `.sha256` file:

```bash
sudo VNN_DB=/var/lib/verinoo/data \
     BACKUP_DIR=/root/verinoo-backups/pre-v0.10.0 \
     KEEP=7 \
     /opt/verinoo/scripts/backup.sh
```

Keep the archive and its checksum. Do not delete backups during or after the
upgrade; they are the recovery path for the pre-v2 state.

If the shipped snapshot CLI is present (`verinoo-node snapshot --help`), you may
additionally take a signed state snapshot. Stop the service first: `create`
opens RocksDB and fails while the live node holds the lock. `--out` must be
absent or an empty directory, and `create` always signs with a seed:

```bash
sudo systemctl stop verinoo-node

sudo verinoo-node snapshot create \
  --db /var/lib/verinoo/data \
  --out /root/snapshots/snapshot-<height> \
  --key-seed "<operator-seed>" \
  --signer <signer-pubkey-hex>

sudo verinoo-node snapshot verify \
  --snapshot /root/snapshots/snapshot-<height> \
  --signer <signer-pubkey-hex>

sudo systemctl start verinoo-node
```

`create` requires `--key-seed` (or `VNN_KEY_SEED`) and always signs the
manifest; `--signer` pins the derived signing key. `verify` and `apply` require
at least one `--signer` (or `VNN_SNAPSHOT_SIGNERS`). A failed `create` can leave
a partial directory; delete it before retrying.

Snapshot caveats:

- A snapshot is not a backup substitute. Keep the stopped-service tarball too.
- The state layer verifies file integrity against caller-supplied expectations.
  The `verinoo-node snapshot verify` command additionally recomputes
  `stateRoot`, the validator-set hash, the checkpoint head fields, and the local
  model hashes.
- Manifest signatures are **operator-attested**, not BFT certificates. There is
  no automatic snapshot download, trust-on-first-use, sync or failover.
- `apply` requires the node to be stopped and never deletes the existing
  database; it moves it to a backup directory.

If the CLI is absent from the binary, skip snapshots and use
`scripts/backup.sh`.

## 5. Upgrade order

### 5.1 Single-validator testnet

There is no coordination problem, but there is still a rollback boundary.

```bash
# Preflight, backup and record the head first (sections 3 and 4).
NEW_BIN=/root/verinoo-node-v0.10.0-testnet
sudo scripts/upgrade-v2.sh \
  --binary "$NEW_BIN" \
  --sha256 "$(sha256sum "$NEW_BIN" | awk '{print $1}')" \
  --service verinoo-node \
  --db /var/lib/verinoo/data \
  --backup-dir /root/verinoo-backups/$(date -u +%Y%m%d-%H%M%S)
# --sums-file <path> works instead of --sha256 when the file contains an entry
# for the binary (the extracted package SHA256SUMS does; the release archive
# sums file does not).
```

The script:

1. Requires root, an executable new binary with a published SHA-256
   (`--sha256` or `--sums-file`), an existing service, `/ready` returning 200,
   exactly one `ExecStart` path whose basename is `verinoo-node`, and enough
   free space for a database copy. It never falls back to a guessed install
   path.
2. Backs up the database with `scripts/backup.sh` when available, otherwise
   stops the service and copies the database.
3. Backs up the old binary and its hash.
4. Stops the service, stages the new binary, verifies its checksum again,
   installs it at the service's `ExecStart` path, starts the service, and
   checks the running process where `/proc` is available.
5. Waits for `/health` and `/ready`, prints `/v1/models`, and parses
   `/blocks/since/<pre-height>` as JSON. The first block above the recorded
   height must have header version 2.
6. Exits non-zero with `installed but v2 unconfirmed` when it cannot prove
   that. It leaves backups in place, does not delete them, and only prints
   old-binary rollback commands when `--force-rollback` is passed and no v2
   block is observed.

It does **not** upgrade peers, rotate keys, change `VNN_VALIDATORS`, fence a
host or roll back automatically.

### 5.2 Multi-validator testnet

Use a maintenance window. The chain intentionally has no automatic failover;
stopping all validators is safer than letting old and new producers race.

1. Announce the window and stop client submissions.
2. On every node: preflight (section 3) and backup (section 4).
3. Stop all validators:

   ```bash
   sudo systemctl stop verinoo-node
   ```

4. Upgrade all replicas and validators. You may use the helper per node, but
   it expects `/ready` to be 200; for an already-stopped node use the manual
   path in section 5.3.
5. Start all validators together. Confirm the same `chainId`, validator set and
   model set.
6. Verify the first new block is `version: 2` (section 6). Only then resume
   client traffic.
7. Do not restart any v0.9.0 validator against the v2 chain.

Alternative, only if you can pause production without stopping every process:
upgrade every replica first (v0.10.0 accepts v1 and v2), then stop all
validators, upgrade them, and restart together. Never let an old validator
produce after the first v2 block.

### 5.3 Manual upgrade path

Use this when the service is already stopped or the helper cannot reach
`/health`:

```bash
sudo systemctl stop verinoo-node

# Database backup while stopped (if scripts/backup.sh was not used).
sudo mkdir -p /root/verinoo-backups/manual
sudo tar czf /root/verinoo-backups/manual/verinoo-db-pre-v0.10.0.tar.gz \
  -C /var/lib/verinoo data
sudo sha256sum /root/verinoo-backups/manual/verinoo-db-pre-v0.10.0.tar.gz \
  > /root/verinoo-backups/manual/verinoo-db-pre-v0.10.0.tar.gz.sha256

# Old binary backup.
sudo cp -p /usr/local/bin/verinoo-node \
  /root/verinoo-backups/manual/verinoo-node.pre-v0.10.0

# Install and start.
sudo install -m 0755 /root/verinoo-node-v0.10.0-testnet \
  /usr/local/bin/verinoo-node
sudo systemctl start verinoo-node

# Verify.
curl -fsS http://127.0.0.1:7100/health
curl -fsS http://127.0.0.1:7100/ready
curl -fsS http://127.0.0.1:7100/v1/models
```

Restore the database only when you actually need to roll back (section 8).

## 6. Verify the upgrade

### 6.1 Liveness and readiness

```bash
curl -fsS http://127.0.0.1:7100/health
curl -fsS http://127.0.0.1:7100/ready
```

`/health` is liveness only: it returns HTTP 200 even when no head exists, with
`"ok": false`. `/ready` must return 200 with `"ready": true`. A 503 body gives
the reason:

| `reason` | Action |
|---|---|
| `no_head` | Wait for the first block; check the producer and peer count. |
| `validator_set_mismatch` | Fix `VNN_VALIDATORS` or the database; do not edit state. |
| `syncing` | Wait for catch-up to finish. |
| `stale_head` | Check block production, peer connectivity and the clock. |
| `behind_peer` | Let the node catch up, or fix the peer URL. |

### 6.2 Header version

`GET /blocks` exposes the header version. Compare the head with the height you
recorded before the upgrade:

```bash
curl -fsS http://127.0.0.1:7100/blocks |
  python3 -c 'import json,sys; b=json.load(sys.stdin); h=b[0]["header"]; print("height", h["index"], "version", h["version"])'
```

The genesis block remains `version 1`; the first block produced after the
upgrade (the first height above the value you recorded) must be `version 2`.
If the RPC is unavailable, inspect the journal:

```bash
journalctl -u verinoo-node -n 100 --no-pager
# Look for produced/applied block logs and "unknown block version" errors.
```

A v0.9.0 node logs `unknown block version 2` when it receives a v2 block. That
is a mixed-version alarm, not a reason to restart the old node into service.

### 6.3 Models and API

```bash
curl -fsS http://127.0.0.1:7100/v1/models
curl -fsS http://127.0.0.1:7100/status
```

Every node must report the same model set and `modelSetHash`; otherwise state
roots will diverge at the first model execution.

### 6.4 Sampling, receipts and claims demo

Exercise a decoder task with an explicit override:

```bash
verinoo-node infer --key-file key.json --model tiny-gpt-v1 \
  --input "Once upon a time" --node http://127.0.0.1:7100
# Then inspect the receipt with /v1/tasks/<id>; it must contain "sampling",
# "samplingHash" and "samplingSource": "task".
```

Run the receipt check:

```bash
./scripts/verify.sh <taskId> --node http://127.0.0.1:7100
```

Run the claims-triage demo against the local node:

```bash
CLAIMS_NODE=http://127.0.0.1:7100 ./scripts/demo-claims.sh
```

Batch and SSE can be checked with the SDK tests:

```bash
cd sdk/typescript && node --test test-batch-sse.mjs
```

## 7. Monitoring during the window

| Signal | Where | What to watch |
|---|---|---|
| readiness | `/ready` | Must stay 200; a reason is an incident. |
| height | `/status` | Must advance on schedule; a flat height means no producer or a stalled network. |
| header version | `/blocks` | First post-upgrade block must be 2. |
| mixed versions | journals | `unknown block version 2` on an old node means stop it and upgrade. |
| validator set | `/status` | `validatorSetConsistent` must be true and hashes equal across nodes. |
| models | `/v1/models` | Same ids, `weightsHash` and `modelSetHash` everywhere. |
| disk | `df -h` | Keep at least 20% free; backups are large. |
| mempool | `/metrics` | Sustained growth after a restart can mean admission is ahead of production. |

## 8. Rollback rules

There are two different rollback cases. Do not confuse them.

### 8.1 Before the first v2 block

This is a normal binary/config rollback.

```bash
sudo systemctl stop verinoo-node
sudo install -m 0755 /root/verinoo-backups/<timestamp>/verinoo-node.pre-upgrade \
  /usr/local/bin/verinoo-node

# Only if the database was also changed or is suspect. Move the failed
# database aside instead of deleting it, then restore the backup:
# sudo mv /var/lib/verinoo/data /var/lib/verinoo/data.failed
# sudo tar xzf /root/verinoo-backups/<timestamp>/verinoo-<stamp>.tar.gz \
#   -C /var/lib/verinoo
# sudo chown -R verinoo:verinoo /var/lib/verinoo/data

sudo systemctl start verinoo-node
curl -fsS http://127.0.0.1:7100/ready
```

Do not delete the backup. If the old binary cannot read the database, restore
the pre-upgrade archive instead of experimenting on the live directory.

### 8.2 After the first v2 block

**Do not roll back to v0.9.0.** It rejects v2 blocks and cannot follow the
chain. Rolling back one node while others are on v2 produces a stall or a fork.

Your options are:

1. **Roll forward:** install v0.10.0 everywhere and let the network converge.
2. **Restore the whole network:** stop every node, restore each pre-upgrade
   database backup or snapshot, install v0.9.0, and re-genesis if the restored
   heights are inconsistent. This is a network-wide operation, not a per-node
   rollback.

There is no automatic recovery, no BFT finality and no quorum certificate to
fall back on. Treat a post-v2 rollback as a deliberate testnet reset.

## 9. Failure cases

| Symptom | Likely cause | Action |
|---|---|---|
| New binary exits at startup | Wrong binary/arch, bad config, unreadable DB | Check `journalctl -u verinoo-node -n 100`. Restore the old binary only if no v2 block exists. |
| `/health` 200 but `/ready` 503 `stale_head` | No block produced, no peers, wrong clock | Check validator flag, peer count, producer slot and clock. |
| `/ready` 503 `validator_set_mismatch` | `VNN_VALIDATORS` differs from state | Make the configured set identical on all nodes; do not edit state. |
| `/ready` 503 `behind_peer` | Replica is catching up | Wait; check `VNN_SYNC_URLS` and peer `/status`. |
| `unknown block version 2` on an old node | Mixed-version network | Stop the old validator immediately; upgrade it before restarting. |
| `state root mismatch` | Different/missing model file or divergent state | Compare `/v1/models` and `modelSetHash`; fix config. Do not delete the DB. |
| `wrong producer for height` | Different validator sets | Make `VNN_VALIDATORS` identical on every node. |
| Service restart loop | Unit path, permissions, port conflict | `systemctl status`, `journalctl`, `ss -ltnp`. |
| Disk full during backup | Not enough free space | Free space or use another backup volume; do not delete the only good backup. |

## 10. Failover, fencing and promotion

There is **no automatic failover**. `/ready` and `/status` are observability
signals; they do not elect a leader, acquire a lease, fence a host or move
state.

Manual promotion of a replica:

1. Confirm the replica is at the same chain id, height, state root, validator
   set and model set as the validators.
2. Stop the old validator and remove it from the API/load-balancer rotation.
3. Confirm the old process has exited. Do not run the same `VNN_KEY_SEED` on
   two hosts: nothing in the node prevents a duplicate signer, and there is no
   split-brain protection.
4. Configure the replica with `VNN_VALIDATOR=1` and the validator seed, and
   update `VNN_VALIDATORS`/`VNN_SYNC_URLS` on the other nodes.
5. Start it and confirm `/ready` is 200 and the height advances.

Fencing is external: stop the old process, detach its volume, or block it at
the network/hypervisor layer. The node does not provide a lease or a fencing
primitive.

## 11. Checklist

- [ ] Read the v2 upgrade warning and chose a window.
- [ ] Every node: preflight healthy, `/ready` 200, model/config identical.
- [ ] Every node: database backup plus checksum; old binary saved.
- [ ] Optional: snapshot created and verified (CLI present).
- [ ] All validators stopped, or production suspended.
- [ ] v0.10.0 installed on all validators and replicas.
- [ ] All nodes started; `/health` and `/ready` 200.
- [ ] First new block shows `"version": 2`.
- [ ] `/v1/models` and `modelSetHash` match across nodes.
- [ ] Sampling receipt verified; claims demo passes.
- [ ] Backups retained; no backup deleted.
- [ ] Monitoring restored and client traffic resumed.
