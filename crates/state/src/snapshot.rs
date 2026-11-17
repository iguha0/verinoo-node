//! State snapshot format: RocksDB checkpoint plus a hashed file manifest.
//!
//! The state layer owns file integrity and safe apply/rollback. It does not
//! recompute `compute_state_root` (that lives in `verinoo-exec`) and it does not
//! verify signer identity. Callers supply the trusted metadata and verify the
//! manifest signatures at the node layer.
//!
//! A snapshot directory must be immutable and trusted for the whole verify and
//! apply operation. The state layer re-checks integrity and caller expectations
//! immediately before the destructive rename and binds the apply to the exact
//! verified manifest bytes, but path-based checks cannot close the final race
//! against an attacker who can rewrite the snapshot directory during apply.

use crate::StateStore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use thiserror::Error;
use verinoo_types::canonical_json;

pub const SNAPSHOT_VERSION: u64 = 1;
pub const SNAPSHOT_KIND: &str = "verinoo-state-snapshot";
pub const SIGNING_DOMAIN: &str = "verinoo/snapshot/v1";

const MANIFEST_NAME: &str = "manifest.json";
const STATE_DIR: &str = "state";
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const HASH_BUFFER_BYTES: usize = 64 * 1024;

pub type ModelHashes = BTreeMap<String, String>;

type ManifestDigest = [u8; 32];

/// Caller-supplied metadata for a snapshot. The state layer records these
/// values but cannot derive or validate their truth; `state_root` must come
/// from `verinoo-exec` and the hashes from the local model/validator registries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotMetadata {
    pub chain_id: String,
    pub height: u64,
    pub block_hash: String,
    pub parent_hash: String,
    pub state_root: String,
    pub base_fee: Option<u64>,
    pub model_set_hash: String,
    pub model_hashes: ModelHashes,
    pub validator_set_hash: String,
    pub created_at: u64,
}

/// Trusted values a verifier requires the manifest to match.
///
/// Every field is a caller-supplied assertion. The state layer compares the
/// manifest to these values but cannot prove they reflect the canonical chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotExpectations {
    pub chain_id: String,
    pub height: u64,
    pub block_hash: String,
    pub parent_hash: String,
    pub state_root: String,
    /// When `Some`, the manifest `baseFee` must equal this value (an absent
    /// manifest `baseFee` is a mismatch). When `None`, `baseFee` is not
    /// verified; do not treat that as implicit acceptance of the manifest
    /// value.
    pub base_fee: Option<u64>,
    pub model_set_hash: String,
    /// When set, the manifest model map must match exactly. When `None`, only
    /// `model_set_hash` is compared.
    pub model_hashes: Option<ModelHashes>,
    pub validator_set_hash: String,
}

impl SnapshotExpectations {
    pub fn from_metadata(metadata: &SnapshotMetadata) -> Self {
        Self {
            chain_id: metadata.chain_id.clone(),
            height: metadata.height,
            block_hash: metadata.block_hash.clone(),
            parent_hash: metadata.parent_hash.clone(),
            state_root: metadata.state_root.clone(),
            base_fee: metadata.base_fee,
            model_set_hash: metadata.model_set_hash.clone(),
            model_hashes: Some(metadata.model_hashes.clone()),
            validator_set_hash: metadata.validator_set_hash.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotFile {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SnapshotSignature {
    pub address: String,
    pub public_key: String,
    pub signature: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SnapshotManifest {
    pub v: u64,
    pub kind: String,
    pub chain_id: String,
    pub height: u64,
    pub block_hash: String,
    pub parent_hash: String,
    pub state_root: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_fee: Option<u64>,
    pub model_set_hash: String,
    pub model_hashes: ModelHashes,
    pub validator_set_hash: String,
    pub created_at: u64,
    pub files: Vec<SnapshotFile>,
    /// Carried for node-level signature checks. The state layer never treats
    /// these entries as proof of signer identity.
    #[serde(default)]
    pub signatures: Vec<SnapshotSignature>,
}

#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("snapshot io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("rocksdb checkpoint failed: {0}")]
    RocksDb(#[from] rocksdb::Error),
    #[error("manifest is not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("canonical JSON encoding failed: {0}")]
    Canonical(#[from] verinoo_types::CanonicalError),
    #[error("snapshot destination is not empty: {0}")]
    DestinationNotEmpty(PathBuf),
    #[error("snapshot destination already has a state directory: {0}")]
    DestinationStateExists(PathBuf),
    #[error("snapshot destination is not a directory: {0}")]
    DestinationNotDirectory(PathBuf),
    #[error("manifest is not a regular file: {0}")]
    ManifestNotFile(PathBuf),
    #[error("manifest is too large: {0} bytes")]
    ManifestTooLarge(u64),
    #[error("unsupported snapshot version {actual}, expected {expected}")]
    UnsupportedVersion { expected: u64, actual: u64 },
    #[error("wrong snapshot kind {actual}, expected {expected}")]
    WrongKind {
        expected: &'static str,
        actual: String,
    },
    #[error("snapshot chain id mismatch: expected {expected}, got {actual}")]
    ChainIdMismatch { expected: String, actual: String },
    #[error("snapshot height mismatch: expected {expected}, got {actual}")]
    HeightMismatch { expected: u64, actual: u64 },
    #[error("snapshot block hash mismatch: expected {expected}, got {actual}")]
    BlockHashMismatch { expected: String, actual: String },
    #[error("snapshot parent hash mismatch: expected {expected}, got {actual}")]
    ParentHashMismatch { expected: String, actual: String },
    #[error("snapshot state root mismatch: expected {expected}, got {actual}")]
    StateRootMismatch { expected: String, actual: String },
    #[error("snapshot base fee mismatch: expected {expected}, got {actual:?}")]
    BaseFeeMismatch { expected: u64, actual: Option<u64> },
    #[error("snapshot model set hash mismatch: expected {expected}, got {actual}")]
    ModelSetHashMismatch { expected: String, actual: String },
    #[error("snapshot model hash mismatch for {model_id}: expected {expected}, got {actual}")]
    ModelHashMismatch {
        model_id: String,
        expected: String,
        actual: String,
    },
    #[error("snapshot validator set hash mismatch: expected {expected}, got {actual}")]
    ValidatorSetHashMismatch { expected: String, actual: String },
    #[error("snapshot manifest is invalid: {0}")]
    InvalidManifest(String),
    #[error("snapshot manifest changed during apply: expected {expected}, got {actual}")]
    ManifestChanged { expected: String, actual: String },
    #[error("invalid snapshot path {path}: {reason}")]
    InvalidPath { path: String, reason: &'static str },
    #[error("snapshot path is a symlink: {0}")]
    Symlink(PathBuf),
    #[error("snapshot path is not a regular file or directory: {0}")]
    UnsupportedFileType(PathBuf),
    #[error("snapshot path is not valid UTF-8: {0}")]
    NonUtf8Path(PathBuf),
    #[error("duplicate snapshot file path: {0}")]
    DuplicateFile(String),
    #[error("snapshot file is missing from the manifest: {0}")]
    FileNotInManifest(String),
    #[error("snapshot file listed in the manifest is missing on disk: {0}")]
    FileMissingFromDisk(String),
    #[error("snapshot file size mismatch for {path}: expected {expected}, got {actual}")]
    SizeMismatch {
        path: String,
        expected: u64,
        actual: u64,
    },
    #[error("snapshot file hash mismatch for {path}: expected {expected}, got {actual}")]
    HashMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    #[error("snapshot has no state files")]
    EmptySnapshot,
    #[error("snapshot state directory is missing or not a directory: {0}")]
    StateDirMissing(PathBuf),
    #[error("apply target is not a directory: {0}")]
    TargetNotDirectory(PathBuf),
    #[error("apply backup path already exists: {0}")]
    BackupExists(PathBuf),
    #[error("snapshot, target, and backup paths overlap: {0}")]
    OverlappingPaths(PathBuf),
    #[error("snapshot apply failed (rolled back: {rolled_back}): {source}")]
    ApplyFailed {
        rolled_back: bool,
        #[source]
        source: io::Error,
    },
    #[error("snapshot apply failed and rollback failed: apply={apply}; rollback={rollback}")]
    RollbackFailed {
        apply: io::Error,
        rollback: io::Error,
    },
}

fn io_err(path: &Path, source: io::Error) -> SnapshotError {
    SnapshotError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn invalid_path(path: &str, reason: &'static str) -> SnapshotError {
    SnapshotError::InvalidPath {
        path: path.to_string(),
        reason,
    }
}

/// Read and structurally validate `manifest.json` without checking files or
/// signatures.
///
/// Node-level verifiers use this to obtain the signed bytes before opening the
/// checkpoint, then pass the same manifest to [`verify_snapshot`] and compare
/// the returned value for equality.
pub fn read_snapshot_manifest(snapshot_dir: &Path) -> Result<SnapshotManifest, SnapshotError> {
    Ok(read_manifest(snapshot_dir)?.0)
}

/// Atomically rewrite `manifest.json` with canonical JSON.
///
/// Intended for node-level signing after [`create_snapshot`]. The file is
/// written to `manifest.json.tmp` (must not exist), fsynced, and renamed over
/// the manifest. A manifest created by this call is still not trusted until a
/// verifier checks its signatures and recomputes the state root.
pub fn write_snapshot_manifest(
    snapshot_dir: &Path,
    manifest: &SnapshotManifest,
) -> Result<(), SnapshotError> {
    write_manifest(snapshot_dir, manifest)
}

/// Validate a manifest-relative snapshot path.
///
/// The path must be relative, slash-separated, start with `state/` followed by
/// a file component, and must not contain `..` or absolute components. HTTP
/// file handlers use the same rule as manifest verification so a served path
/// can never escape the snapshot directory.
pub fn validate_snapshot_path(path: &str) -> Result<(), SnapshotError> {
    validate_manifest_path(path)
}

/// Create a RocksDB checkpoint plus `manifest.json` in `snapshot_dir`.
///
/// The destination must be absent or an empty directory; a non-empty
/// destination is rejected so an existing snapshot or database is never
/// overwritten. The source store may remain open.
pub fn create_snapshot(
    store: &StateStore,
    snapshot_dir: &Path,
    metadata: SnapshotMetadata,
) -> Result<SnapshotManifest, SnapshotError> {
    prepare_destination(snapshot_dir)?;
    let state_dir = snapshot_dir.join(STATE_DIR);
    store.checkpoint(&state_dir)?;
    let files = collect_state_files(&state_dir)?;
    if files.is_empty() {
        return Err(SnapshotError::EmptySnapshot);
    }
    let manifest = SnapshotManifest {
        v: SNAPSHOT_VERSION,
        kind: SNAPSHOT_KIND.to_string(),
        chain_id: metadata.chain_id,
        height: metadata.height,
        block_hash: metadata.block_hash,
        parent_hash: metadata.parent_hash,
        state_root: metadata.state_root,
        base_fee: metadata.base_fee,
        model_set_hash: metadata.model_set_hash,
        model_hashes: metadata.model_hashes,
        validator_set_hash: metadata.validator_set_hash,
        created_at: metadata.created_at,
        files,
        signatures: Vec::new(),
    };
    write_manifest(snapshot_dir, &manifest)?;
    Ok(manifest)
}

/// Verify manifest structure and every state file against the manifest's own
/// hashes, then compare caller-provided trust metadata.
///
/// Signatures are returned as data but are never validated here; signer
/// identity and thresholds belong to the node layer. `state_root` is compared
/// to `expected.state_root`, not recomputed (recomputation lives in
/// `verinoo-exec`).
pub fn verify_snapshot(
    snapshot_dir: &Path,
    expected: &SnapshotExpectations,
) -> Result<SnapshotManifest, SnapshotError> {
    let (manifest, _) = verify_snapshot_with_digest(snapshot_dir, expected)?;
    Ok(manifest)
}

fn verify_snapshot_with_digest(
    snapshot_dir: &Path,
    expected: &SnapshotExpectations,
) -> Result<(SnapshotManifest, ManifestDigest), SnapshotError> {
    let (manifest, digest) = verify_snapshot_integrity(snapshot_dir)?;
    verify_expectations(&manifest, expected)?;
    Ok((manifest, digest))
}

fn verify_expectations(
    manifest: &SnapshotManifest,
    expected: &SnapshotExpectations,
) -> Result<(), SnapshotError> {
    if manifest.chain_id != expected.chain_id {
        return Err(SnapshotError::ChainIdMismatch {
            expected: expected.chain_id.clone(),
            actual: manifest.chain_id.clone(),
        });
    }
    if manifest.height != expected.height {
        return Err(SnapshotError::HeightMismatch {
            expected: expected.height,
            actual: manifest.height,
        });
    }
    if manifest.block_hash != expected.block_hash {
        return Err(SnapshotError::BlockHashMismatch {
            expected: expected.block_hash.clone(),
            actual: manifest.block_hash.clone(),
        });
    }
    if manifest.parent_hash != expected.parent_hash {
        return Err(SnapshotError::ParentHashMismatch {
            expected: expected.parent_hash.clone(),
            actual: manifest.parent_hash.clone(),
        });
    }
    if manifest.state_root != expected.state_root {
        return Err(SnapshotError::StateRootMismatch {
            expected: expected.state_root.clone(),
            actual: manifest.state_root.clone(),
        });
    }
    if let Some(expected_fee) = expected.base_fee {
        if manifest.base_fee != Some(expected_fee) {
            return Err(SnapshotError::BaseFeeMismatch {
                expected: expected_fee,
                actual: manifest.base_fee,
            });
        }
    }
    if manifest.model_set_hash != expected.model_set_hash {
        return Err(SnapshotError::ModelSetHashMismatch {
            expected: expected.model_set_hash.clone(),
            actual: manifest.model_set_hash.clone(),
        });
    }
    if manifest.validator_set_hash != expected.validator_set_hash {
        return Err(SnapshotError::ValidatorSetHashMismatch {
            expected: expected.validator_set_hash.clone(),
            actual: manifest.validator_set_hash.clone(),
        });
    }
    if let Some(expected_hashes) = &expected.model_hashes {
        if &manifest.model_hashes != expected_hashes {
            return Err(first_model_hash_mismatch(
                expected_hashes,
                &manifest.model_hashes,
            ));
        }
    }
    Ok(())
}

/// Apply a snapshot to `target_db` after re-verifying file integrity.
///
/// `target_db` must not be open by any process (RocksDB keeps locks and
/// in-memory state). If `target_db` exists it is renamed to `backup_dir`, which
/// must not already exist and is never deleted. A failure after the backup move
/// renames the backup back. This does not verify trust metadata; use
/// [`verify_snapshot`] or [`apply_verified_snapshot`] first.
///
/// The snapshot directory must be immutable and trusted during apply. Integrity
/// is re-checked immediately before the state directory is moved, and apply is
/// bound to the manifest bytes verified at entry.
pub fn apply_snapshot(
    snapshot_dir: &Path,
    target_db: &Path,
    backup_dir: &Path,
) -> Result<(), SnapshotError> {
    let (_, digest) = verify_snapshot_integrity(snapshot_dir)?;
    verify_manifest_unchanged(snapshot_dir, &digest)?;
    apply_verified_state_dir(snapshot_dir, target_db, backup_dir, || {
        let (_, actual) = verify_snapshot_integrity(snapshot_dir).map_err(io::Error::other)?;
        ensure_manifest_unchanged(&digest, &actual).map_err(io::Error::other)?;
        Ok(())
    })
}

/// Verify with caller expectations and then apply. Returns the verified
/// manifest that was applied.
///
/// Full expectations are re-verified immediately before the state directory is
/// moved, not just file integrity. Apply is bound to the exact manifest bytes
/// seen by the first verification, so a snapshot swapped for a different valid
/// snapshot is rejected. The snapshot directory must still be immutable and
/// trusted: an attacker who can rewrite it between the final check and the
/// rename can still race the apply.
pub fn apply_verified_snapshot(
    snapshot_dir: &Path,
    expected: &SnapshotExpectations,
    target_db: &Path,
    backup_dir: &Path,
) -> Result<SnapshotManifest, SnapshotError> {
    let (manifest, digest) = verify_snapshot_with_digest(snapshot_dir, expected)?;
    verify_manifest_unchanged(snapshot_dir, &digest)?;
    apply_verified_state_dir(snapshot_dir, target_db, backup_dir, || {
        let (_, actual) =
            verify_snapshot_with_digest(snapshot_dir, expected).map_err(io::Error::other)?;
        ensure_manifest_unchanged(&digest, &actual).map_err(io::Error::other)?;
        Ok(())
    })?;
    Ok(manifest)
}

fn verify_manifest_unchanged(
    snapshot_dir: &Path,
    expected_digest: &ManifestDigest,
) -> Result<(), SnapshotError> {
    let (_, actual_digest) = read_manifest(snapshot_dir)?;
    ensure_manifest_unchanged(expected_digest, &actual_digest)
}

fn ensure_manifest_unchanged(
    expected: &ManifestDigest,
    actual: &ManifestDigest,
) -> Result<(), SnapshotError> {
    if expected != actual {
        return Err(SnapshotError::ManifestChanged {
            expected: hex::encode(expected),
            actual: hex::encode(actual),
        });
    }
    Ok(())
}

/// Deterministic node-signing bytes: [`SIGNING_DOMAIN`], a newline, then
/// canonical JSON with `signatures` removed recursively at the top level.
pub fn manifest_signing_bytes(manifest: &SnapshotManifest) -> Result<Vec<u8>, SnapshotError> {
    let mut value = serde_json::to_value(manifest)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| SnapshotError::InvalidManifest("manifest is not a JSON object".into()))?;
    object.remove("signatures");
    let canonical = canonical_json(&value)?;
    let mut out = Vec::with_capacity(SIGNING_DOMAIN.len() + 1 + canonical.len());
    out.extend_from_slice(SIGNING_DOMAIN.as_bytes());
    out.push(b'\n');
    out.extend_from_slice(canonical.as_bytes());
    Ok(out)
}

fn prepare_destination(snapshot_dir: &Path) -> Result<(), SnapshotError> {
    match fs::symlink_metadata(snapshot_dir) {
        Ok(meta) => {
            if meta.file_type().is_symlink() {
                return Err(SnapshotError::Symlink(snapshot_dir.to_path_buf()));
            }
            if !meta.is_dir() {
                return Err(SnapshotError::DestinationNotDirectory(
                    snapshot_dir.to_path_buf(),
                ));
            }
            let mut entries = fs::read_dir(snapshot_dir).map_err(|e| io_err(snapshot_dir, e))?;
            if entries
                .next()
                .transpose()
                .map_err(|e| io_err(snapshot_dir, e))?
                .is_some()
            {
                return Err(SnapshotError::DestinationNotEmpty(
                    snapshot_dir.to_path_buf(),
                ));
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            fs::create_dir_all(snapshot_dir).map_err(|e| io_err(snapshot_dir, e))?;
        }
        Err(e) => return Err(io_err(snapshot_dir, e)),
    }

    let state_dir = snapshot_dir.join(STATE_DIR);
    match fs::symlink_metadata(&state_dir) {
        Ok(_) => Err(SnapshotError::DestinationStateExists(state_dir)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(io_err(&state_dir, e)),
    }
}

fn write_manifest(snapshot_dir: &Path, manifest: &SnapshotManifest) -> Result<(), SnapshotError> {
    let value = serde_json::to_value(manifest)?;
    let json = canonical_json(&value)?;
    let target = snapshot_dir.join(MANIFEST_NAME);
    let tmp = snapshot_dir.join(format!("{MANIFEST_NAME}.tmp"));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .map_err(|e| io_err(&tmp, e))?;
    file.write_all(json.as_bytes())
        .map_err(|e| io_err(&tmp, e))?;
    file.sync_all().map_err(|e| io_err(&tmp, e))?;
    drop(file);
    fs::rename(&tmp, &target).map_err(|e| io_err(&target, e))
}

fn read_manifest(snapshot_dir: &Path) -> Result<(SnapshotManifest, ManifestDigest), SnapshotError> {
    let path = snapshot_dir.join(MANIFEST_NAME);
    let meta = fs::symlink_metadata(&path).map_err(|e| io_err(&path, e))?;
    if meta.file_type().is_symlink() || !meta.is_file() {
        return Err(SnapshotError::ManifestNotFile(path));
    }
    if meta.len() > MAX_MANIFEST_BYTES {
        return Err(SnapshotError::ManifestTooLarge(meta.len()));
    }
    let bytes = fs::read(&path).map_err(|e| io_err(&path, e))?;
    let digest = sha256_digest(&bytes);
    let manifest: SnapshotManifest = serde_json::from_slice(&bytes)?;
    if manifest.v != SNAPSHOT_VERSION {
        return Err(SnapshotError::UnsupportedVersion {
            expected: SNAPSHOT_VERSION,
            actual: manifest.v,
        });
    }
    if manifest.kind != SNAPSHOT_KIND {
        return Err(SnapshotError::WrongKind {
            expected: SNAPSHOT_KIND,
            actual: manifest.kind,
        });
    }
    Ok((manifest, digest))
}

fn sha256_digest(bytes: &[u8]) -> ManifestDigest {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn verify_snapshot_integrity(
    snapshot_dir: &Path,
) -> Result<(SnapshotManifest, ManifestDigest), SnapshotError> {
    let (manifest, digest) = read_manifest(snapshot_dir)?;
    let state_dir = snapshot_dir.join(STATE_DIR);
    let actual = collect_state_files(&state_dir)?;
    if actual.is_empty() {
        return Err(SnapshotError::EmptySnapshot);
    }

    let mut listed = BTreeMap::new();
    for file in &manifest.files {
        validate_manifest_path(&file.path)?;
        if listed.insert(file.path.clone(), file).is_some() {
            return Err(SnapshotError::DuplicateFile(file.path.clone()));
        }
    }

    for actual_file in actual {
        let Some(listed_file) = listed.remove(&actual_file.path) else {
            return Err(SnapshotError::FileNotInManifest(actual_file.path));
        };
        if listed_file.size != actual_file.size {
            return Err(SnapshotError::SizeMismatch {
                path: actual_file.path,
                expected: listed_file.size,
                actual: actual_file.size,
            });
        }
        if listed_file.sha256 != actual_file.sha256 {
            return Err(SnapshotError::HashMismatch {
                path: actual_file.path,
                expected: listed_file.sha256.clone(),
                actual: actual_file.sha256,
            });
        }
    }

    if let Some((path, _)) = listed.into_iter().next() {
        return Err(SnapshotError::FileMissingFromDisk(path));
    }
    Ok((manifest, digest))
}

fn first_model_hash_mismatch(expected: &ModelHashes, actual: &ModelHashes) -> SnapshotError {
    let ids: BTreeSet<&String> = expected.keys().chain(actual.keys()).collect();
    for id in ids {
        let expected_hash = expected.get(id);
        let actual_hash = actual.get(id);
        if expected_hash != actual_hash {
            return SnapshotError::ModelHashMismatch {
                model_id: id.clone(),
                expected: expected_hash
                    .cloned()
                    .unwrap_or_else(|| "<missing>".to_string()),
                actual: actual_hash
                    .cloned()
                    .unwrap_or_else(|| "<missing>".to_string()),
            };
        }
    }
    SnapshotError::InvalidManifest("model hash maps differ without a key".into())
}

fn collect_state_files(state_dir: &Path) -> Result<Vec<SnapshotFile>, SnapshotError> {
    match fs::symlink_metadata(state_dir) {
        Ok(meta) if meta.file_type().is_symlink() => {
            return Err(SnapshotError::Symlink(state_dir.to_path_buf()));
        }
        Ok(meta) if !meta.is_dir() => {
            return Err(SnapshotError::StateDirMissing(state_dir.to_path_buf()));
        }
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(SnapshotError::StateDirMissing(state_dir.to_path_buf()));
        }
        Err(e) => return Err(io_err(state_dir, e)),
    }

    let mut files = Vec::new();
    walk_state(state_dir, Path::new(STATE_DIR), &mut files)?;
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(files)
}

fn walk_state(
    dir: &Path,
    relative: &Path,
    out: &mut Vec<SnapshotFile>,
) -> Result<(), SnapshotError> {
    let mut entries = fs::read_dir(dir)
        .map_err(|e| io_err(dir, e))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| io_err(dir, e))?;
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        let entry_path = entry.path();
        let file_type = entry.file_type().map_err(|e| io_err(&entry_path, e))?;
        let relative_path = relative.join(entry.file_name());

        if file_type.is_symlink() {
            return Err(SnapshotError::Symlink(relative_path));
        }
        if file_type.is_dir() {
            walk_state(&entry_path, &relative_path, out)?;
        } else if file_type.is_file() {
            let path = manifest_path(&relative_path)?;
            let (size, sha256) = hash_file(&entry_path, &relative_path)?;
            out.push(SnapshotFile { path, size, sha256 });
        } else {
            return Err(SnapshotError::UnsupportedFileType(relative_path));
        }
    }
    Ok(())
}

fn manifest_path(path: &Path) -> Result<String, SnapshotError> {
    let path = path
        .to_str()
        .ok_or_else(|| SnapshotError::NonUtf8Path(path.to_path_buf()))?;
    Ok(path.replace(std::path::MAIN_SEPARATOR, "/"))
}

fn hash_file(path: &Path, relative: &Path) -> Result<(u64, String), SnapshotError> {
    let mut file = File::open(path).map_err(|e| io_err(relative, e))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; HASH_BUFFER_BYTES];
    let mut size = 0u64;
    loop {
        let read = file.read(&mut buffer).map_err(|e| io_err(relative, e))?;
        if read == 0 {
            break;
        }
        size += read as u64;
        hasher.update(&buffer[..read]);
    }
    Ok((size, hex::encode(hasher.finalize())))
}

fn validate_manifest_path(path: &str) -> Result<(), SnapshotError> {
    if path.is_empty() {
        return Err(invalid_path(path, "empty path"));
    }
    if path.contains('\0') {
        return Err(invalid_path(path, "contains a NUL byte"));
    }
    if path.starts_with('/') || path.contains('\\') {
        return Err(invalid_path(path, "must be relative and slash-separated"));
    }

    let mut components = Path::new(path).components();
    match components.next() {
        Some(Component::Normal(part)) if part == STATE_DIR => {}
        _ => return Err(invalid_path(path, "must start with state/")),
    }

    let mut names_file = false;
    for component in components {
        match component {
            Component::Normal(_) => names_file = true,
            Component::CurDir => return Err(invalid_path(path, "contains .")),
            Component::ParentDir => return Err(invalid_path(path, "contains ..")),
            Component::RootDir | Component::Prefix(_) => {
                return Err(invalid_path(path, "must be relative"));
            }
        }
    }
    if !names_file {
        return Err(invalid_path(path, "does not name a file"));
    }
    Ok(())
}

fn resolved_path(path: &Path) -> Result<PathBuf, SnapshotError> {
    match fs::canonicalize(path) {
        Ok(resolved) => Ok(resolved),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let parent = path
                .parent()
                .ok_or_else(|| io_err(path, io::Error::other("path has no parent")))?;
            let name = path
                .file_name()
                .ok_or_else(|| io_err(path, io::Error::other("path has no file name")))?;
            let parent = fs::canonicalize(parent).map_err(|pe| io_err(parent, pe))?;
            Ok(parent.join(name))
        }
        Err(e) => Err(io_err(path, e)),
    }
}

fn apply_verified_state_dir<F>(
    snapshot_dir: &Path,
    target_db: &Path,
    backup_dir: &Path,
    before_state_move: F,
) -> Result<(), SnapshotError>
where
    F: FnOnce() -> io::Result<()>,
{
    let state_src = snapshot_dir.join(STATE_DIR);
    let state_meta = fs::symlink_metadata(&state_src).map_err(|e| io_err(&state_src, e))?;
    if state_meta.file_type().is_symlink() {
        return Err(SnapshotError::Symlink(state_src));
    }
    if !state_meta.is_dir() {
        return Err(SnapshotError::StateDirMissing(state_src));
    }

    let target_meta = match fs::symlink_metadata(target_db) {
        Ok(meta) => Some(meta),
        Err(e) if e.kind() == io::ErrorKind::NotFound => None,
        Err(e) => return Err(io_err(target_db, e)),
    };

    let state_abs = resolved_path(&state_src)?;
    let target_abs = resolved_path(target_db)?;
    let backup_abs = resolved_path(backup_dir)?;

    if target_abs == state_abs
        || target_abs.starts_with(&state_abs)
        || state_abs.starts_with(&target_abs)
    {
        return Err(SnapshotError::OverlappingPaths(target_db.to_path_buf()));
    }
    if backup_abs.starts_with(&state_abs)
        || state_abs.starts_with(&backup_abs)
        || backup_abs == target_abs
        || backup_abs.starts_with(&target_abs)
        || target_abs.starts_with(&backup_abs)
    {
        return Err(SnapshotError::OverlappingPaths(backup_dir.to_path_buf()));
    }

    if let Some(meta) = &target_meta {
        if meta.file_type().is_symlink() || !meta.is_dir() {
            return Err(SnapshotError::TargetNotDirectory(target_db.to_path_buf()));
        }
        match fs::symlink_metadata(backup_dir) {
            Ok(_) => return Err(SnapshotError::BackupExists(backup_dir.to_path_buf())),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(io_err(backup_dir, e)),
        }
        fs::rename(target_db, backup_dir).map_err(|e| io_err(target_db, e))?;
    }

    if let Err(apply_error) = before_state_move() {
        if target_meta.is_some() {
            return match fs::rename(backup_dir, target_db) {
                Ok(()) => Err(SnapshotError::ApplyFailed {
                    rolled_back: true,
                    source: apply_error,
                }),
                Err(rollback_error) => Err(SnapshotError::RollbackFailed {
                    apply: apply_error,
                    rollback: rollback_error,
                }),
            };
        }
        return Err(SnapshotError::ApplyFailed {
            rolled_back: false,
            source: apply_error,
        });
    }

    if let Err(apply_error) = fs::rename(&state_src, target_db) {
        if target_meta.is_some() {
            return match fs::rename(backup_dir, target_db) {
                Ok(()) => Err(SnapshotError::ApplyFailed {
                    rolled_back: true,
                    source: apply_error,
                }),
                Err(rollback_error) => Err(SnapshotError::RollbackFailed {
                    apply: apply_error,
                    rollback: rollback_error,
                }),
            };
        }
        return Err(SnapshotError::ApplyFailed {
            rolled_back: false,
            source: apply_error,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Account, InferenceTask};
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir()
                .join(format!("verinoo-snapshot-{tag}-{}-{n}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn metadata() -> SnapshotMetadata {
        SnapshotMetadata {
            chain_id: "verinoo-test-1".into(),
            height: 42,
            block_hash: "aa".into(),
            parent_hash: "bb".into(),
            state_root: "cc".into(),
            base_fee: Some(3),
            model_set_hash: "dd".into(),
            model_hashes: BTreeMap::from([("model-1".into(), "ee".into())]),
            validator_set_hash: "ff".into(),
            created_at: 1_760_000_000_000,
        }
    }

    fn populated_store(path: &Path) -> StateStore {
        let store = StateStore::open(path).unwrap();
        store.set_account(&Account {
            address: "ai_alice".into(),
            public_key: "pk".into(),
            nonce: 2,
            balance: 500,
            updated_at: 9,
        });
        let task: InferenceTask = serde_json::from_value(json!({
            "taskId": "task-1",
            "requester": "ai_alice",
            "targetModel": "model-1",
            "maxFee": 10,
            "deadline": 20,
            "status": "pending"
        }))
        .unwrap();
        store.set_task(&task);
        store.set_meta_u64("baseFee", 3);
        store
    }

    fn fixture(tag: &str) -> (TempDir, SnapshotMetadata, PathBuf) {
        let root = TempDir::new(tag);
        let db = root.path().join("db");
        let snapshot_dir = root.path().join("snapshot");
        let meta = metadata();
        {
            let store = populated_store(&db);
            create_snapshot(&store, &snapshot_dir, meta.clone()).unwrap();
        }
        (root, meta, snapshot_dir)
    }

    #[test]
    fn create_verify_apply_roundtrip() {
        let root = TempDir::new("roundtrip");
        let db = root.path().join("db");
        let snapshot_dir = root.path().join("snapshot");
        let applied = root.path().join("applied");
        let backup = root.path().join("backup");
        let meta = metadata();

        {
            let store = populated_store(&db);
            let created = create_snapshot(&store, &snapshot_dir, meta.clone()).unwrap();
            assert_eq!(created.v, SNAPSHOT_VERSION);
            assert_eq!(created.kind, SNAPSHOT_KIND);
            assert_eq!(created.height, meta.height);
            assert_eq!(created.base_fee, Some(3));
            assert!(created.signatures.is_empty());
            assert!(!created.files.is_empty());
            assert!(created.files.iter().all(|f| f.path.starts_with("state/")));
        }

        let expected = SnapshotExpectations::from_metadata(&meta);
        let verified = verify_snapshot(&snapshot_dir, &expected).unwrap();
        assert_eq!(verified.state_root, meta.state_root);
        let applied_manifest =
            apply_verified_snapshot(&snapshot_dir, &expected, &applied, &backup).unwrap();
        assert_eq!(applied_manifest, verified);
        assert!(!snapshot_dir.join(STATE_DIR).exists());
        assert!(!backup.exists());

        let store = StateStore::open(&applied).unwrap();
        assert_eq!(store.get_account("ai_alice").unwrap().balance, 500);
        assert_eq!(store.get_task("task-1").unwrap().status, "pending");
        assert_eq!(store.meta_u64_or("baseFee", 0), 3);
    }

    #[test]
    fn tampered_state_file_fails_verification() {
        let (_root, meta, snapshot_dir) = fixture("tamper");
        let expected = SnapshotExpectations::from_metadata(&meta);
        let manifest = verify_snapshot(&snapshot_dir, &expected).unwrap();
        let victim = manifest
            .files
            .iter()
            .find(|file| file.size > 0)
            .expect("checkpoint has a non-empty file");
        let path = snapshot_dir.join(&victim.path);
        let mut bytes = fs::read(&path).unwrap();
        bytes[0] ^= 0x01;
        fs::write(&path, &bytes).unwrap();

        let err = verify_snapshot(&snapshot_dir, &expected).unwrap_err();
        assert!(matches!(
            err,
            SnapshotError::HashMismatch { .. } | SnapshotError::SizeMismatch { .. }
        ));
    }

    #[test]
    fn metadata_mismatches_fail_verification() {
        let (_root, meta, snapshot_dir) = fixture("mismatch");
        let base = SnapshotExpectations::from_metadata(&meta);

        let mut wrong = base.clone();
        wrong.chain_id = "other-chain".into();
        assert!(matches!(
            verify_snapshot(&snapshot_dir, &wrong),
            Err(SnapshotError::ChainIdMismatch { .. })
        ));

        let mut wrong = base.clone();
        wrong.height += 1;
        assert!(matches!(
            verify_snapshot(&snapshot_dir, &wrong),
            Err(SnapshotError::HeightMismatch { .. })
        ));

        let mut wrong = base.clone();
        wrong.block_hash = "00".into();
        assert!(matches!(
            verify_snapshot(&snapshot_dir, &wrong),
            Err(SnapshotError::BlockHashMismatch { .. })
        ));

        let mut wrong = base.clone();
        wrong.parent_hash = "00".into();
        assert!(matches!(
            verify_snapshot(&snapshot_dir, &wrong),
            Err(SnapshotError::ParentHashMismatch { .. })
        ));

        let mut wrong = base.clone();
        wrong.state_root = "00".into();
        assert!(matches!(
            verify_snapshot(&snapshot_dir, &wrong),
            Err(SnapshotError::StateRootMismatch { .. })
        ));

        let mut wrong = base.clone();
        wrong.base_fee = Some(4);
        assert!(matches!(
            verify_snapshot(&snapshot_dir, &wrong),
            Err(SnapshotError::BaseFeeMismatch { .. })
        ));

        let mut unchecked = base.clone();
        unchecked.base_fee = None;
        verify_snapshot(&snapshot_dir, &unchecked).unwrap();

        let mut wrong = base.clone();
        wrong.model_set_hash = "00".into();
        assert!(matches!(
            verify_snapshot(&snapshot_dir, &wrong),
            Err(SnapshotError::ModelSetHashMismatch { .. })
        ));

        let mut wrong = base.clone();
        wrong.model_hashes = Some(BTreeMap::from([("model-2".into(), "00".into())]));
        assert!(matches!(
            verify_snapshot(&snapshot_dir, &wrong),
            Err(SnapshotError::ModelHashMismatch { .. })
        ));

        let mut wrong = base;
        wrong.validator_set_hash = "00".into();
        assert!(matches!(
            verify_snapshot(&snapshot_dir, &wrong),
            Err(SnapshotError::ValidatorSetHashMismatch { .. })
        ));
    }

    #[test]
    fn absent_manifest_base_fee_fails_a_some_expectation() {
        let (_root, meta, snapshot_dir) = fixture("base-fee-absent");
        let path = snapshot_dir.join(MANIFEST_NAME);
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value.as_object_mut().unwrap().remove("baseFee");
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();

        let expected = SnapshotExpectations::from_metadata(&meta);
        assert!(matches!(
            verify_snapshot(&snapshot_dir, &expected),
            Err(SnapshotError::BaseFeeMismatch {
                expected: 3,
                actual: None
            })
        ));
    }

    #[test]
    fn crafted_manifest_paths_and_files_are_rejected() {
        let (_root, meta, snapshot_dir) = fixture("paths");
        let expected = SnapshotExpectations::from_metadata(&meta);
        let manifest_path = snapshot_dir.join(MANIFEST_NAME);
        let original = fs::read_to_string(&manifest_path).unwrap();

        let mutate = |change: &dyn Fn(&mut serde_json::Value)| {
            let mut value: serde_json::Value = serde_json::from_str(&original).unwrap();
            change(&mut value);
            fs::write(&manifest_path, serde_json::to_vec(&value).unwrap()).unwrap();
        };

        mutate(&|v| v["files"][0]["path"] = "../escape".into());
        assert!(matches!(
            verify_snapshot(&snapshot_dir, &expected),
            Err(SnapshotError::InvalidPath { .. })
        ));

        mutate(&|v| v["files"][0]["path"] = "/etc/passwd".into());
        assert!(matches!(
            verify_snapshot(&snapshot_dir, &expected),
            Err(SnapshotError::InvalidPath { .. })
        ));

        mutate(&|v| v["files"][0]["path"] = "state\\..\\..\\escape".into());
        assert!(matches!(
            verify_snapshot(&snapshot_dir, &expected),
            Err(SnapshotError::InvalidPath { .. })
        ));

        mutate(&|v| {
            let first = v["files"][0].clone();
            v["files"].as_array_mut().unwrap().push(first);
        });
        assert!(matches!(
            verify_snapshot(&snapshot_dir, &expected),
            Err(SnapshotError::DuplicateFile(_))
        ));

        mutate(&|v| {
            v["files"].as_array_mut().unwrap().remove(0);
        });
        assert!(matches!(
            verify_snapshot(&snapshot_dir, &expected),
            Err(SnapshotError::FileNotInManifest(_))
        ));

        mutate(&|v| {
            v["files"].as_array_mut().unwrap().push(json!({
                "path": "state/missing.sst",
                "size": 1,
                "sha256": "00"
            }));
        });
        assert!(matches!(
            verify_snapshot(&snapshot_dir, &expected),
            Err(SnapshotError::FileMissingFromDisk(_))
        ));

        mutate(&|v| v["unexpected"] = json!(true));
        assert!(matches!(
            verify_snapshot(&snapshot_dir, &expected),
            Err(SnapshotError::Json(_))
        ));

        fs::write(&manifest_path, original).unwrap();
        verify_snapshot(&snapshot_dir, &expected).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_in_state_are_rejected() {
        use std::os::unix::fs::symlink;

        let (root, meta, snapshot_dir) = fixture("symlink");
        let expected = SnapshotExpectations::from_metadata(&meta);
        let outside = root.path().join("outside");
        fs::write(&outside, b"outside").unwrap();
        symlink(&outside, snapshot_dir.join(STATE_DIR).join("escape.sst")).unwrap();

        let err = verify_snapshot(&snapshot_dir, &expected).unwrap_err();
        assert!(matches!(err, SnapshotError::Symlink(_)));
    }

    #[test]
    fn create_refuses_non_empty_destination() {
        let root = TempDir::new("non-empty");
        let db = root.path().join("db");
        let snapshot_dir = root.path().join("snapshot");
        fs::create_dir_all(&snapshot_dir).unwrap();
        fs::write(snapshot_dir.join("keep"), b"data").unwrap();

        let store = populated_store(&db);
        let err = create_snapshot(&store, &snapshot_dir, metadata()).unwrap_err();
        assert!(matches!(err, SnapshotError::DestinationNotEmpty(_)));
        assert!(snapshot_dir.join("keep").exists());
    }

    #[test]
    fn apply_refuses_existing_backup_and_leaves_target_intact() {
        let (root, _meta, snapshot_dir) = fixture("backup-exists");
        let target = root.path().join("target");
        let backup = root.path().join("backup");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("sentinel"), b"old").unwrap();
        fs::create_dir_all(&backup).unwrap();
        fs::write(backup.join("sentinel"), b"backup").unwrap();

        let err = apply_snapshot(&snapshot_dir, &target, &backup).unwrap_err();
        assert!(matches!(err, SnapshotError::BackupExists(_)));
        assert_eq!(fs::read(target.join("sentinel")).unwrap(), b"old");
        assert_eq!(fs::read(backup.join("sentinel")).unwrap(), b"backup");
        assert!(snapshot_dir.join(STATE_DIR).exists());
    }

    #[test]
    fn apply_restores_target_when_state_move_fails() {
        let (root, _meta, snapshot_dir) = fixture("rollback");
        let target = root.path().join("target");
        let backup = root.path().join("backup");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("sentinel"), b"old").unwrap();

        verify_snapshot_integrity(&snapshot_dir).unwrap();
        let err = apply_verified_state_dir(&snapshot_dir, &target, &backup, || {
            Err(io::Error::other("injected move failure"))
        })
        .unwrap_err();

        assert!(matches!(
            err,
            SnapshotError::ApplyFailed {
                rolled_back: true,
                ..
            }
        ));
        assert_eq!(fs::read(target.join("sentinel")).unwrap(), b"old");
        assert!(!backup.exists());
        assert!(snapshot_dir.join(STATE_DIR).exists());
    }

    #[test]
    fn apply_without_existing_target_does_not_create_backup() {
        let (root, _meta, snapshot_dir) = fixture("fresh-target");
        let target = root.path().join("target");
        let backup = root.path().join("backup");

        apply_snapshot(&snapshot_dir, &target, &backup).unwrap();
        assert!(target.join("CURRENT").exists());
        assert!(!backup.exists());
    }

    #[test]
    fn apply_replaces_existing_target_and_keeps_backup_intact() {
        let (root, meta, snapshot_dir) = fixture("replace-target");
        let target = root.path().join("target");
        let backup = root.path().join("backup");
        {
            let old = StateStore::open(&target).unwrap();
            old.set_account(&Account {
                address: "ai_old".into(),
                public_key: "old-pk".into(),
                nonce: 7,
                balance: 11,
                updated_at: 12,
            });
            old.set_meta_u64("baseFee", 99);
        }

        let expected = SnapshotExpectations::from_metadata(&meta);
        let applied = apply_verified_snapshot(&snapshot_dir, &expected, &target, &backup).unwrap();
        assert_eq!(applied.height, meta.height);
        assert!(!snapshot_dir.join(STATE_DIR).exists());
        assert!(backup.is_dir());

        {
            let old = StateStore::open(&backup).unwrap();
            assert_eq!(old.get_account("ai_old").unwrap().balance, 11);
            assert!(old.get_account("ai_alice").is_none());
            assert_eq!(old.meta_u64_or("baseFee", 0), 99);
        }

        let new = StateStore::open(&target).unwrap();
        assert_eq!(new.get_account("ai_alice").unwrap().balance, 500);
        assert!(new.get_account("ai_old").is_none());
        assert_eq!(new.get_task("task-1").unwrap().status, "pending");
        assert_eq!(new.meta_u64_or("baseFee", 0), 3);
    }

    #[test]
    fn apply_rejects_overlapping_snapshot_target_and_backup_paths() {
        let (root, _meta, snapshot_dir) = fixture("overlap-paths");
        let target = root.path().join("target");
        let backup = root.path().join("backup");
        fs::create_dir_all(&target).unwrap();
        let state_dir = snapshot_dir.join(STATE_DIR);

        let cases = [
            (state_dir.clone(), backup.clone()),
            (snapshot_dir.clone(), backup.clone()),
            (target.clone(), state_dir.join("backup")),
            (target.clone(), target.clone()),
            (target.clone(), target.join("backup")),
        ];
        for (target_path, backup_path) in cases {
            let err = apply_snapshot(&snapshot_dir, &target_path, &backup_path).unwrap_err();
            assert!(
                matches!(err, SnapshotError::OverlappingPaths(_)),
                "{target_path:?} / {backup_path:?}: {err:?}"
            );
        }

        let nested = root.path().join("backup-parent");
        fs::create_dir_all(&nested).unwrap();
        let err = apply_snapshot(&snapshot_dir, &nested.join("target"), &nested).unwrap_err();
        assert!(matches!(err, SnapshotError::OverlappingPaths(_)));

        assert!(state_dir.exists());
        assert!(target.is_dir());
    }

    #[test]
    fn apply_verified_snapshot_rejects_manifest_changed_after_verify() {
        let (root, meta, snapshot_dir) = fixture("toctou-manifest");
        let target = root.path().join("target");
        let backup = root.path().join("backup");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("sentinel"), b"old").unwrap();

        let expected = SnapshotExpectations::from_metadata(&meta);
        verify_snapshot(&snapshot_dir, &expected).unwrap();

        let manifest_path = snapshot_dir.join(MANIFEST_NAME);
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        value["stateRoot"] = json!("00");
        fs::write(&manifest_path, serde_json::to_vec(&value).unwrap()).unwrap();

        let err = apply_verified_snapshot(&snapshot_dir, &expected, &target, &backup).unwrap_err();
        assert!(matches!(
            err,
            SnapshotError::ManifestChanged { .. } | SnapshotError::StateRootMismatch { .. }
        ));
        assert_eq!(fs::read(target.join("sentinel")).unwrap(), b"old");
        assert!(!backup.exists());
        assert!(snapshot_dir.join(STATE_DIR).exists());
    }

    #[test]
    fn apply_verified_snapshot_rejects_state_changed_after_verify() {
        let (root, meta, snapshot_dir) = fixture("toctou-state");
        let target = root.path().join("target");
        let backup = root.path().join("backup");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("sentinel"), b"old").unwrap();

        let expected = SnapshotExpectations::from_metadata(&meta);
        let manifest = verify_snapshot(&snapshot_dir, &expected).unwrap();
        let victim = manifest
            .files
            .iter()
            .find(|file| file.size > 0)
            .expect("checkpoint has a non-empty file");
        let path = snapshot_dir.join(&victim.path);
        let mut bytes = fs::read(&path).unwrap();
        bytes[0] ^= 0x01;
        fs::write(&path, &bytes).unwrap();

        let err = apply_verified_snapshot(&snapshot_dir, &expected, &target, &backup).unwrap_err();
        assert!(
            matches!(
                err,
                SnapshotError::HashMismatch { .. } | SnapshotError::SizeMismatch { .. }
            ),
            "{err:?}"
        );
        assert_eq!(fs::read(target.join("sentinel")).unwrap(), b"old");
        assert!(!backup.exists());
        assert!(snapshot_dir.join(STATE_DIR).exists());
    }

    #[test]
    fn signing_bytes_are_stable_and_exclude_signatures() {
        let (_root, meta, snapshot_dir) = fixture("signing");
        let expected = SnapshotExpectations::from_metadata(&meta);
        let manifest = verify_snapshot(&snapshot_dir, &expected).unwrap();

        let bytes = manifest_signing_bytes(&manifest).unwrap();
        assert_eq!(bytes, manifest_signing_bytes(&manifest).unwrap());
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.starts_with(&format!("{SIGNING_DOMAIN}\n")));
        assert!(!text.contains("\"signatures\""));

        let mut signed = manifest;
        signed.signatures.push(SnapshotSignature {
            address: "ai_a".into(),
            public_key: "pk".into(),
            signature: "00".into(),
        });
        assert_eq!(manifest_signing_bytes(&signed).unwrap(), bytes);

        let mut other_signature = signed.clone();
        other_signature.signatures[0].signature = "ff".into();
        assert_eq!(manifest_signing_bytes(&other_signature).unwrap(), bytes);
    }

    #[test]
    fn read_only_checkpoint_open_does_not_mutate_the_snapshot() {
        let (_root, _meta, snapshot_dir) = fixture("read-only");
        let state_dir = snapshot_dir.join(STATE_DIR);
        let before = collect_state_files(&state_dir).unwrap();
        {
            let store = StateStore::open_read_only(&state_dir).unwrap();
            assert_eq!(store.get_account("ai_alice").unwrap().balance, 500);
            assert_eq!(store.meta_u64_or("baseFee", 0), 3);
        }
        let after = collect_state_files(&state_dir).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn missing_or_empty_signatures_do_not_break_verification() {
        let (_root, meta, snapshot_dir) = fixture("signatures");
        let expected = SnapshotExpectations::from_metadata(&meta);
        let path = snapshot_dir.join(MANIFEST_NAME);
        let original = fs::read_to_string(&path).unwrap();

        let mut value: serde_json::Value = serde_json::from_str(&original).unwrap();
        value.as_object_mut().unwrap().remove("signatures");
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let legacy = verify_snapshot(&snapshot_dir, &expected).unwrap();
        assert!(legacy.signatures.is_empty());

        value.as_object_mut().unwrap().insert(
            "signatures".into(),
            json!([{"address": "ai_a", "publicKey": "pk", "signature": "00"}]),
        );
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let signed = verify_snapshot(&snapshot_dir, &expected).unwrap();
        assert_eq!(signed.signatures.len(), 1);
    }
}
