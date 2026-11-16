# State snapshots

Status: the state layer can create, verify, and apply a snapshot. The node does
not serve snapshots, sign manifests, recompute state roots, or verify BFT
certificates yet. See ADR 006 and "What remains" below.

## Scope

`verinoo-state` (`crates/state/src/snapshot.rs`) owns:

- `StateStore::checkpoint`: a RocksDB checkpoint of the canonical database.
- Deterministic file listing, SHA-256 hashing, and manifest writing.
- Manifest parsing and structural validation.
- File-set completeness, path, symlink, size, and hash checks.
- Comparison against caller-supplied trusted metadata.
- Atomic rename-based apply with rollback and a backup when the target exists.

It does **not** own:

- Computing `stateRoot` — that is `compute_state_root` in `crates/exec`.
- Deriving `modelSetHash`, `modelHashes`, or `validatorSetHash` — callers
  supply these from their registries/state.
- Signing or verifying manifest signatures — those are node-level concerns.
- Moving the node's head, catch-up sync, or block validation after apply.

## Layout

```text
snapshot-<height>/
  manifest.json          # canonical JSON, no whitespace
  state/                 # RocksDB checkpoint
```

Version 1 records only state files. It does not write `manifest.sig`,
`blocks/recent.json`, or a compressed archive; compression and recent headers
remain part of the node-level design in ADR 006.

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
  "modelSetHash": "<runtime + transformer + gpt + builtin weights>",
  "modelHashes": { "<modelId>": "<weightsHash>" },
  "validatorSetHash": "<sha256 of canonical JSON sorted address array>",
  "createdAt": 1760000000000,
  "files": [
    { "path": "state/000123.sst", "size": 123456, "sha256": "<lowercase hex>" }
  ],
  "signatures": [
    { "address": "ai_...", "publicKey": "<hex or empty>", "signature": "<hex>" }
  ]
}
```

Field notes:

- `v` must be `1`; `kind` must be `verinoo-state-snapshot`.
- `baseFee` is optional. When absent it is omitted from serialization and
  signing bytes. It is not trusted implicitly: callers that care must set
  `SnapshotExpectations.base_fee` to `Some(value)`, which requires an exact
  match (an absent manifest value is a mismatch). `None` skips the check.
- `files` lists every regular file under `state/`, sorted by path. Paths are
  relative, slash-separated, and must begin with `state/`.
- `size` and `sha256` describe the bytes on disk, not the manifest.
- `signatures` defaults to an empty array if the field is absent. A signature
  entry contains exactly `address`, `publicKey`, and `signature`; extra fields
  are rejected.
- The manifest is canonical JSON: sorted keys, no whitespace, integers only.
  The same encoder is used for signing bytes.

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

## Trust boundaries

The state layer verifies **file integrity**, not chain truth. `stateRoot`,
`modelSetHash`, `modelHashes`, `validatorSetHash`, and `baseFee` are
caller/node-level trust assertions: the state layer only checks that the
manifest bytes equal the expected values supplied by the caller. It does not
recompute the state root, derive model or validator hashes, or authenticate the
source of those values. Signatures are never verified here.

| Value | Source of truth for verify |
|---|---|
| `chainId`, `height`, `blockHash`, `parentHash` | Caller's expected checkpoint/head |
| `stateRoot` | Caller-supplied value from `verinoo-exec`; never recomputed here |
| `baseFee` | Caller-supplied expected value when `SnapshotExpectations.base_fee` is `Some`; not trusted implicitly |
| `modelSetHash`, `modelHashes` | Caller's local model registry |
| `validatorSetHash` | Caller's expected genesis or state-derived set |
| `signatures` | Not trusted or verified by the state layer |

`verify_snapshot` compares the manifest to `SnapshotExpectations`. A matching
manifest means "these bytes match these claimed metadata values", not "this is
the canonical chain". A node must still:

1. Establish that the expected height/hash/state root are trustworthy (its own
   finalized head, a pinned checkpoint, or a BFT certificate).
2. Recompute the state root from the applied database and compare it to the
   manifest.
3. Verify signatures against the expected signer set or a BFT quorum.

The caller may omit `modelHashes` from expectations to check only
`modelSetHash`; when `Some`, both maps must match exactly, including missing and
extra model ids.

## Create

```rust
let manifest = create_snapshot(&store, &snapshot_dir, metadata)?;
```

- The destination must be absent or an empty directory. A non-empty destination
  is rejected; existing snapshots and databases are never overwritten.
- `snapshot_dir/state` must not already exist. RocksDB creates it.
- Every regular file under `state/` is hashed with SHA-256 in a recursive,
  sorted walk. Symlinks and non-file entries are rejected.
- `manifest.json` is written via a temporary file and rename. The source store
  may stay open while the checkpoint is taken.
- A failed create can leave a partial directory; delete it before retrying.

## Verify

```rust
let manifest = verify_snapshot(&snapshot_dir, &expected)?;
```

Checks performed before returning success:

- `manifest.json` exists, is a regular file, is below 16 MiB, and is valid
  JSON with no unknown fields.
- Version is 1 and kind is `verinoo-state-snapshot`.
- Every `files[].path` is relative, slash-separated, starts with `state/`, and
  contains no `.` or `..` components.
- No duplicate paths.
- `state/` exists, is a real directory, and contains only regular files and
  directories (no symlinks or special files).
- Every on-disk file is listed, and every listed file exists.
- Size and SHA-256 match for every file.
- Caller expectations match for chain id, height, block hash, parent hash,
  state root, base fee (when supplied), model set hash, model hashes (when
  supplied), and validator set hash.

Signatures are returned in the manifest but are ignored by verification. A
manifest with missing, empty, or arbitrary signatures can pass; signer identity
is a node-level decision.

## Apply and rollback

```rust
apply_snapshot(&snapshot_dir, &target_db, &backup_dir)?;
// or, to verify trusted metadata first:
apply_verified_snapshot(&snapshot_dir, &expected, &target_db, &backup_dir)?;
```

Requirements and semantics:

- The target database **must not be open** by the node or any other process
  that can read or write it. RocksDB keeps locks and in-memory state; renaming
  an open database is unsafe. Stop the node before calling apply.
- The snapshot directory **must be immutable and trusted** for the whole verify
  and apply operation. The state layer re-verifies and binds apply to the exact
  manifest bytes seen at entry, but it cannot stop an attacker who controls the
  snapshot directory from racing the final check against the rename.
- `apply_snapshot` re-runs full file-integrity verification, and
  `apply_verified_snapshot` re-runs the full caller-expectations verification,
  immediately before `snapshot_dir/state` is moved. Both also re-read the
  manifest before touching the target and reject a changed manifest, so a
  snapshot swapped for a different valid snapshot is not applied. A failure in
  the final check after the target was moved to the backup rolls the target
  back.
- Snapshot, target, and backup paths must not overlap; a target that is or
  contains `snapshot_dir/state`, a backup inside the snapshot or target, and
  equal target/backup paths are rejected.
- If `target_db` does not exist, `snapshot_dir/state` is renamed directly to
  `target_db`. No backup is created and `backup_dir` is not touched.
- If `target_db` exists, it must be a real directory. `backup_dir` must not
  exist. The target is renamed to the backup first, then
  `snapshot_dir/state` is renamed to the target.
- If the second rename fails, the backup is renamed back to the target and the
  error reports `rolled_back: true`. If the rollback itself fails, both errors
  are returned; no directory is deleted.
- The old database is **never deleted**. The backup remains for manual
  recovery even after a successful apply.
- Apply is atomic per rename, not a single transaction across both renames. A
  crash between them leaves the old database at the backup path and no target;
  an operator must restore or roll forward manually. Callers should keep the
  backup until the node has applied new blocks and reproduced the expected
  state root.
- `snapshot_dir/state` is moved, so a successful apply consumes the state
  directory. The manifest remains in the snapshot directory.
- All three paths should be on the same filesystem; rename across filesystems
  fails.

## What remains

- **Node endpoints.** `GET /snapshot/manifest`, a file/range endpoint, resumable
  download into a temporary directory, and a flag to enable serving. Nothing in
  `crates/node` serves or fetches snapshots today.
- **Signed manifests.** Operator-key or validator-set signatures, threshold
  rules, duplicate/unauthorized signer rejection, and recording the accepted
  manifest hash. The state layer only provides deterministic signing bytes.
- **State-root recomputation.** Open the checkpoint through `verinoo-exec`,
  recompute `compute_state_root`, and compare it to `manifest.stateRoot`.
- **Head and block checks.** Verify the head block hash/index/parent hash, and
  reject snapshots at or below the local finalized head or at the same height
  with a different hash.
- **Model and validator derivation.** Compute the local model-set hash from the
  runtime/transformer/GPT/builtin registries and the expected validator-set
  hash from genesis or state.
- **BFT certificate.** Include and verify a commit certificate, require the
  snapshot height to be finalized, and apply quorum rules from ADR 004.
- **Catch-up integration.** Set the head at the snapshot height and start
  `sync_once` from there; keep `/blocks/since/:index` unchanged.
- **Operational tooling.** Snapshot retention/pruning, resume, checksum
  manifests for the manifest itself, and a runbook for backup restore.
