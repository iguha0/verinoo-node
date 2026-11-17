# State snapshots

Status: the state layer can create, verify, and apply a snapshot; the node has
operator CLI commands for all three, signs manifests with Ed25519 operator
keys, recomputes the checkpoint state root read-only, and serves snapshots over
HTTP. Snapshot signatures are **operator-attested**, not BFT certificates. The
node does not automatically download or apply a snapshot, does not set its head
from one, and does not provide automatic failover. See ADR 006 and "What
remains" below.

## Scope

`verinoo-state` (`crates/state/src/snapshot.rs`) owns:

- `StateStore::checkpoint`: a RocksDB checkpoint of the canonical database.
- `StateStore::open_read_only`: opens a checkpoint without taking the write
  lock or replaying writes, for state-root verification.
- Deterministic file listing, SHA-256 hashing, and atomic manifest writing.
- Manifest parsing and structural validation.
- File-set completeness, path, symlink, size, and hash checks.
- Comparison against caller-supplied trusted metadata.
- Atomic rename-based apply with rollback and a backup when the target exists.

`verinoo-node` (`crates/node/src/snapshot.rs`) owns:

- `verinoo-node snapshot create|verify|apply`.
- Ed25519 manifest signatures and signer threshold checks.
- `stateRoot` recomputation with `verinoo_exec::engine::compute_state_root`.
- Validator-set and model-set hash derivation and comparison.
- Checkpoint head consistency checks.
- The read-only `/snapshot/manifest` and `/snapshot/files/*path` endpoints.

The state layer does **not** know chain truth, model registries, or signer
identity. The node does not own BFT finality, automatic catch-up, or head setup
after apply.

## Layout

```text
snapshot-<height>/
  manifest.json          # canonical JSON, no whitespace
  state/                 # RocksDB checkpoint
```

Version 1 records only state files. Every file path in the manifest and every
served path starts with `state/`; there is no `blocks/recent.json`,
`manifest.sig`, or compressed archive yet. The Ed25519 signatures live in
`manifest.json.signatures`.

## Manifest

`SnapshotManifest` uses camelCase. Unknown fields are rejected on read.

```json
{
  "v": 1,
  "kind": "verinoo-state-snapshot",
  "chainId": "verinoo-test-1",
  "height": 12345,
  "blockHash": "<hash at height>",
  "parentHash": "<hash at height-1>",
  "stateRoot": "<compute_state_root of the snapshot store>",
  "baseFee": 3,
  "modelSetHash": "<sha256 of sorted modelId:weightsHash lines>",
  "modelHashes": { "<modelId>": "<weightsHash>" },
  "validatorSetHash": "<sha256 of canonical JSON sorted address array>",
  "createdAt": 1760000000000,
  "files": [
    { "path": "state/000123.sst", "size": 123456, "sha256": "<lowercase hex>" }
  ],
  "signatures": [
    { "address": "ai_...", "publicKey": "<hex>", "signature": "<hex>" }
  ]
}
```

Field notes:

- `v` must be `1`; `kind` must be `verinoo-state-snapshot`.
- `baseFee` is optional. `snapshot create` copies `meta:baseFee` when present;
  a database that has never written a block may omit it. It is not trusted
  implicitly: callers that care must set `SnapshotExpectations.base_fee` to
  `Some(value)`, which requires an exact match (an absent manifest value is a
  mismatch). `None` skips the check.
- `files` lists every regular file under `state/`, sorted by path. Paths are
  relative, slash-separated, and must begin with `state/`.
- `size` and `sha256` describe the bytes on disk, not the manifest.
- `signatures` defaults to an empty array if the field is absent. A signature
  entry contains exactly `address`, `publicKey`, and `signature`; extra fields
  are rejected.
- The manifest is canonical JSON: sorted keys, no whitespace, integers only.
  The same encoder is used for signing bytes.

### Hash schemes

- `validatorSetHash` is lowercase hex SHA-256 over the canonical JSON of the
  sorted, deduplicated active validator addresses, for example
  `["ai_a","ai_b"]`. This is the same scheme as `/ready` and `/status`
  (`crates/node/src/readiness.rs`). It covers addresses only; it is not part
  of `compute_state_root`.
- `modelHashes` maps every locally executable model id to its `weightsHash`
  from the builtin model plus the graph, transformer, and GPT registries.
- `modelSetHash` is lowercase hex SHA-256 over the sorted newline-joined
  `modelId:weightsHash` lines, matching the registry fingerprint scheme.
- `stateRoot` is `compute_state_root` over the checkpoint, never a copied head
  field.

### Signing bytes

`manifest_signing_bytes` produces:

```text
verinoo/snapshot/v1\n
<canonical JSON with the top-level "signatures" member removed>
```

The domain prefix is a literal byte prefix, not a JSON field. The output is
deterministic across signature changes; an empty or populated `signatures`
array does not affect it. The state layer exposes these bytes but does not sign
or validate them.

## CLI

Run `verinoo-node snapshot --help` for the full syntax.

```bash
verinoo-node snapshot create --db <path> --out <dir> \
  [--key-seed <string>] [--signer <pubkey-hex>]...

verinoo-node snapshot verify --snapshot <dir> --signer <pubkey-hex> \
  [--signer ...] [--threshold N] [--chain-id <id>]

verinoo-node snapshot apply --snapshot <dir> --db <target> --backup <dir> \
  --signer <pubkey-hex> [--signer ...] [--threshold N]
```

Environment:

| Variable | Used by | Meaning |
|---|---|---|
| `VNN_KEY_SEED` | create | Default seed when `--key-seed` is omitted. |
| `VNN_SNAPSHOT_SIGNERS` | verify, apply | Comma-separated signer public keys when no `--signer` is given. |
| `VNN_MODEL_DIR` | create, verify, apply | Model directory. Create defaults to `./models`; verify/apply compare only this directory when set, otherwise builtin-only and fail closed on a mismatch. |
| `VNN_SNAPSHOT_DIR` | node serving | Directory containing `manifest.json` and `state/`. |

### create

`create` requires the database to be stopped. It opens the store read-write to
take the checkpoint, so a running node's RocksDB lock makes the command fail
with a clear "is the node still running?" error. It also requires a chain head.
Steps:

1. Read the head block and compute `compute_state_root` over the store.
2. Copy `baseFee` from `meta:baseFee` when present.
3. Load the model directory and derive `modelHashes`/`modelSetHash`.
4. Hash the state's canonical sorted validator addresses.
5. Create the checkpoint and unsigned manifest with `create_snapshot`.
6. Sign `manifest_signing_bytes` with the seed-derived Ed25519 key, append the
   signature (`address`, `publicKey`, `signature`), and atomically rewrite
   `manifest.json`.
7. Print the height, manifest hash (SHA-256 over the exact manifest file
   bytes), state root, and signer.

The seed is hashed once with SHA-256 to obtain the 32-byte Ed25519 key, exactly
like `verinoo-node address`. The address is `public_key_to_address(publicKey)`.
`--signer` on create is a pin: when supplied, the derived signing key must be
one of those public keys. It does not create multiple signatures.

The destination must be absent or empty. A failed create can leave a partial
directory; delete it before retrying.

### verify

`verify` requires at least one explicit `--signer` or
`VNN_SNAPSHOT_SIGNERS`. Checks run in this order:

1. Parse signer public keys; reject malformed, non-32-byte, or duplicate keys.
2. Read `manifest.json` and reject an unsupported version or kind.
3. When `--chain-id` is given, require an exact chain id match.
4. Reject a threshold below 1 or above the configured signer count.
5. For every manifest signature: the public key must be one of the configured
   signers (unknown keys are rejected), its address must equal
   `public_key_to_address(publicKey)`, it must be the only signature from that
   address, and the Ed25519 signature must verify over the signing bytes for
   domain `verinoo/snapshot/v1`. A signature made for a different domain fails.
   Require at least `--threshold` distinct valid signers (default 1).
6. Run the state layer's file-set verification: paths, symlinks, completeness,
   size, SHA-256, version, kind, and manifest metadata. The manifest returned
   by this step must equal the signed manifest, so a manifest swapped during
   verification is rejected.
7. Open `state/` with `StateStore::open_read_only` and recompute
   `compute_state_root`. It must equal `stateRoot`.
8. Recompute `validatorSetHash` from the checkpoint's active validator
   addresses. It must equal the manifest.
9. Require the checkpoint head index, hash, and previous hash to equal
   `height`, `blockHash`, and `parentHash`.
10. Derive the local model hashes and set hash. Both must match exactly. With
    `VNN_MODEL_DIR` unset only builtin models are loaded, so a manifest that
    lists installed models fails closed rather than skipping the check.

Signatures are checked before the checkpoint is opened. The checkpoint is never
mutated by verification. The manifest hash printed by `verify` is SHA-256 over
the exact manifest bytes.

### apply

`apply` performs the full `verify` sequence, then:

1. If the target database exists, opens it read-write without create to probe
   the RocksDB lock. A running node (or any other writer) makes apply fail.
2. Calls `apply_verified_snapshot` with expectations derived from the verified
   manifest (`stateRoot`, `modelSetHash`, `modelHashes`, `validatorSetHash`,
   chain id, height, block hashes, and `baseFee`).
3. Moves an existing target to `--backup` before moving `state/` into place.
   The old database is **never deleted**. If the target did not exist, no
   backup is created.

The snapshot directory must be immutable and trusted for the whole verify and
apply operation. Apply re-verifies integrity and expectations immediately
before the state rename and binds to the manifest bytes seen at entry, but it
cannot stop an attacker who controls the snapshot directory from racing the
final check against the rename. Stop the node before apply and keep the backup
until the new database has served blocks and reproduced the expected state
root.

## Trust boundaries

The state layer verifies **file integrity**, not chain truth. `stateRoot`,
`modelSetHash`, `modelHashes`, `validatorSetHash`, and `baseFee` are
caller/node-level trust assertions: the state layer only checks that the
manifest bytes equal the expected values supplied by the caller. It does not
recompute the state root, derive model or validator hashes, or authenticate the
source of those values. Signatures are never verified there.

The node verifier recomputes the state root, validator-set hash, and head
consistency from the checkpoint, and compares the local model set. That proves
the signed manifest matches these bytes and this configuration. It does **not**
prove the state belongs to the canonical chain.

| Value | Source of truth for verify |
|---|---|
| `chainId`, `height`, `blockHash`, `parentHash` | Signed manifest plus the checkpoint's own head |
| `stateRoot` | Recomputed from the read-only checkpoint by `verinoo-exec` |
| `baseFee` | Manifest value; caller expectation when set |
| `modelSetHash`, `modelHashes` | Local builtin/graph/transformer/GPT registries |
| `validatorSetHash` | Recomputed from checkpoint `val:` addresses |
| `signatures` | Ed25519 operator keys configured by the verifier |

Without BFT, a signature is **operator attestation**. A third party can
recompute the state root and verify that the files match the manifest, but
without a trusted checkpoint or a BFT certificate it cannot independently know
that the state is the canonical chain. Pin the signer public key and publish it
out of band. Do not call this trustless. A snapshot does not replace block
validation for blocks after its height.

The operator is responsible for rejecting a snapshot at or below the local
head, a same-height snapshot with a different hash, or a snapshot older than
the local finalized state. The CLI does not read the target database's old head
for that comparison; it verifies the snapshot in isolation.

## Serving

Set `VNN_SNAPSHOT_DIR` to the snapshot directory. Reads are public chain state
and do not require an API key:

- `GET /snapshot/manifest` returns the manifest JSON, or 404 when serving is
  disabled, the directory is missing, or the manifest is absent/not a regular
  file.
- `GET /snapshot/files/*path` serves one checkpoint file. Only relative paths
  beginning with `state/` are accepted; `..`, absolute paths, backslashes,
  other top-level directories, and symlinks are rejected. Traversal attempts
  return 400, symlinks 403, and missing files or directories 404.
- The file endpoint supports a single `Range: bytes=...` range. It returns 206
  with `Content-Range` and `Accept-Ranges: bytes`, or 416 with
  `Content-Range: bytes */<size>` for an invalid/unsatisfiable/multi-range
  request. Without a range it streams the whole file with 200.
- Files are streamed from `tokio::fs` in 64 KiB chunks. The snapshot directory
  is expected to be immutable for the duration of a download; do not rewrite a
  snapshot in place.

`deploy/nginx-verinoo.conf` proxies `/snapshot/`. Snapshot reads can be large
and should be rate-limited or egress-capped at the proxy if exposed to the
public internet. Path safety is enforced by the node, but the proxy must not
rewrite or normalize paths into a different directory.

## Operations runbook

Create on a stopped validator or a dedicated snapshot host with the same
`VNN_MODEL_DIR` as the network:

```bash
systemctl stop verinoo-node
VNN_KEY_SEED="$SNAPSHOT_SIGNER_SEED" VNN_MODEL_DIR=/opt/verinoo/models \
  verinoo-node snapshot create \
    --db /var/lib/verinoo/data \
    --out /var/lib/verinoo/snapshots/snapshot-$(date -u +%Y%m%dT%H%M%SZ) \
    --key-seed "$SNAPSHOT_SIGNER_SEED"
systemctl start verinoo-node
```

Serve the directory by setting `VNN_SNAPSHOT_DIR` on a node and restarting, or
copy it to a static file host. Download and verify on the target host **before**
touching the live database:

```bash
curl -fsS -o manifest.json "$SNAPSHOT_URL/snapshot/manifest"
# Then download each files[].path with Range/curl -C - into the same layout.
verinoo-node snapshot verify \
  --snapshot /var/lib/verinoo/incoming/snapshot-<height> \
  --signer "$SNAPSHOT_SIGNER_PUBKEY" \
  --chain-id verinoo-test-1
```

Apply only with the target node stopped, keeping the old database:

```bash
systemctl stop verinoo-node
verinoo-node snapshot apply \
  --snapshot /var/lib/verinoo/incoming/snapshot-<height> \
  --db /var/lib/verinoo/data \
  --backup /var/lib/verinoo/data.pre-snapshot-$(date -u +%Y%m%dT%H%M%SZ) \
  --signer "$SNAPSHOT_SIGNER_PUBKEY"
systemctl start verinoo-node
curl -fsS localhost:7100/health
curl -fsS localhost:7100/ready
```

If verification or apply fails, leave the live database alone and fall back to
normal catch-up. If apply succeeded but the node later diverges, stop it and
restore the backup path manually. Never delete the backup until the node has
advanced and reproduced the expected state root.

Promotion/fencing is unchanged: applying a snapshot is not failover. Stop the
old validator and ensure only one process holds the validator key before
starting another host. See `docs/operations.md`.

## What remains

- **BFT certificates.** Version 1 has no commit certificate or finalized-height
  proof. A quorum signature scheme over finalized blocks from ADR 004 is not
  implemented.
- **Automatic snapshot sync.** The node serves and verifies but does not
  download, resume, verify, and apply a snapshot automatically. Operators run
  the CLI.
- **Head setup after apply.** Apply moves the database but does not restart the
  node or wire the snapshot height into catch-up sync. The checkpoint already
  contains the head and `blockheight:<height>` index, so a restart resumes from
  it; a fresh node still needs its normal genesis/startup path.
- **Stale/same-height policy.** The CLI does not compare against the target's
  old head. Enforce "height must be greater" and "same height must have the
  same hash" in the promotion runbook or a future node integration.
- **Download resume tooling.** Range serving exists, but there is no built-in
  downloader, temporary-directory manager, or partial-download verifier.
- **Retention/pruning.** No snapshot retention policy is defined. Old `hash:`,
  `blockheight:`, `task:`, and `game:` records remain in a checkpoint.
- **Manifest hash in `/status`.** The accepted manifest hash is printed by the
  CLI but not recorded in node status.
