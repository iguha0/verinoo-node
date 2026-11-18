//! Operator snapshot CLI: create, verify, and apply a signed state snapshot.
//!
//! Signatures are operator-attested Ed25519 signatures over the state-layer
//! [`manifest_signing_bytes`]. This is not a BFT certificate and does not prove
//! that the state belongs to a finalized block. The verifier independently
//! recomputes the state root and validator-set hash from the checkpoint and
//! compares the local model set before an apply.

use ed25519_dalek::{Signature, Signer as _, Verifier as _, VerifyingKey};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use verinoo_exec::engine::{Block, HEADER_VERSION, MIN_HEADER_VERSION};
use verinoo_state::snapshot::{
    apply_verified_snapshot_bound, create_snapshot, manifest_signing_bytes, read_snapshot_manifest,
    verify_snapshot_verified, write_snapshot_manifest, ModelHashes, SnapshotExpectations,
    SnapshotManifest, SnapshotMetadata, SnapshotSignature, VerifiedSnapshot, SIGNING_DOMAIN,
};
use verinoo_state::StateStore;

use crate::readiness;

const MANIFEST_NAME: &str = "manifest.json";
const DEFAULT_MODEL_DIR: &str = "./models";
const SIGNERS_ENV: &str = "VNN_SNAPSHOT_SIGNERS";
const MAX_SNAPSHOT_SIGNATURES: usize = 64;

type CliResult<T> = Result<T, Box<dyn Error>>;

/// `verinoo-node snapshot <command>` entry point.
pub fn cli(args: &[String]) -> CliResult<()> {
    let Some(command) = args.first() else {
        print_help();
        return Ok(());
    };
    match command.as_str() {
        "create" => create_cmd(&args[1..]),
        "verify" => verify_cmd(&args[1..]),
        "apply" => apply_cmd(&args[1..]),
        "--help" | "-h" | "help" => {
            print_help();
            Ok(())
        }
        other => Err(format!("unknown snapshot command `{other}`; try `snapshot --help`").into()),
    }
}

fn print_help() {
    println!("Usage: verinoo-node snapshot <command> [options]");
    println!();
    println!("Commands:");
    println!("  create --db <path> --out <dir> [--key-seed <string>] [--signer <pubkey-hex>]...");
    println!("  verify --snapshot <dir> --signer <pubkey-hex> [--signer ...] [--threshold N] [--chain-id <id>]");
    println!("  apply  --snapshot <dir> --db <target> --backup <dir> --chain-id <id> --signer <pubkey-hex> [--signer ...] [--threshold N]");
    println!();
    println!("Environment:");
    println!("  VNN_KEY_SEED          default signing seed for `create`");
    println!("  VNN_SNAPSHOT_SIGNERS  comma-separated signer public keys for verify/apply");
    println!("  VNN_MODEL_DIR         model directory (create default: ./models)");
    println!();
    println!("Trust boundary: snapshot signatures are operator-attested, not BFT.");
    println!("verify recomputes the state root and validator hash from the checkpoint");
    println!("and compares the local model set. create requires a model directory that");
    println!("loads at least one installed model. apply requires --chain-id, rejects a");
    println!("snapshot at or below the target head, holds the target database lock");
    println!("across the rename, and moves the old database to --backup (never deleted).");
}

fn create_cmd(args: &[String]) -> CliResult<()> {
    if help_requested(args) {
        print_help();
        return Ok(());
    }
    let mut db: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut key_seed: Option<String> = None;
    let mut signers = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--db" if i + 1 < args.len() => {
                db = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--out" if i + 1 < args.len() => {
                out = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--key-seed" if i + 1 < args.len() => {
                key_seed = Some(args[i + 1].clone());
                i += 1;
            }
            "--signer" if i + 1 < args.len() => {
                signers.push(args[i + 1].clone());
                i += 1;
            }
            other => return Err(format!("unknown or incomplete argument `{other}`").into()),
        }
        i += 1;
    }

    let db = db.ok_or("create requires --db <path>")?;
    let out = out.ok_or("create requires --out <dir>")?;
    let seed = key_seed
        .or_else(|| std::env::var("VNN_KEY_SEED").ok())
        .filter(|seed| !seed.trim().is_empty())
        .ok_or("create requires --key-seed or VNN_KEY_SEED")?;
    let model_dir = std::env::var("VNN_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_MODEL_DIR));

    let report = create_at(&db, &out, &seed, &signers, &model_dir)?;
    println!("snapshot created: {}", out.display());
    println!("height: {}", report.manifest.height);
    println!("manifest hash: {}", report.manifest_hash);
    println!("state root: {}", report.manifest.state_root);
    println!("validator set hash: {}", report.manifest.validator_set_hash);
    println!("model set hash: {}", report.manifest.model_set_hash);
    println!(
        "signer: {} (public key {})",
        report.signer_address, report.signer_public_key
    );
    Ok(())
}

fn verify_cmd(args: &[String]) -> CliResult<()> {
    if help_requested(args) {
        print_help();
        return Ok(());
    }
    let mut snapshot_dir: Option<PathBuf> = None;
    let mut signers = Vec::new();
    let mut threshold = 1usize;
    let mut chain_id: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--snapshot" if i + 1 < args.len() => {
                snapshot_dir = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--signer" if i + 1 < args.len() => {
                signers.push(args[i + 1].clone());
                i += 1;
            }
            "--threshold" if i + 1 < args.len() => {
                threshold = args[i + 1]
                    .parse()
                    .map_err(|_| format!("invalid --threshold `{}`", args[i + 1]))?;
                i += 1;
            }
            "--chain-id" if i + 1 < args.len() => {
                chain_id = Some(args[i + 1].clone());
                i += 1;
            }
            other => return Err(format!("unknown or incomplete argument `{other}`").into()),
        }
        i += 1;
    }

    let snapshot_dir = snapshot_dir.ok_or("verify requires --snapshot <dir>")?;
    let signers = with_env_signers(signers)?;
    let model_dir = std::env::var("VNN_MODEL_DIR").ok().map(PathBuf::from);
    let report = verify_at(
        &snapshot_dir,
        &signers,
        threshold,
        chain_id.as_deref(),
        model_dir.as_deref(),
    )?;

    println!("snapshot verified: {}", snapshot_dir.display());
    println!("height: {}", report.manifest.height);
    println!("manifest hash: {}", report.manifest_hash);
    println!("state root: {}", report.manifest.state_root);
    println!("validator set hash: {}", report.manifest.validator_set_hash);
    println!("model set hash: {}", report.manifest.model_set_hash);
    println!("signers: {}", report.signers.join(","));
    Ok(())
}

fn apply_cmd(args: &[String]) -> CliResult<()> {
    if help_requested(args) {
        print_help();
        return Ok(());
    }
    let mut snapshot_dir: Option<PathBuf> = None;
    let mut target: Option<PathBuf> = None;
    let mut backup: Option<PathBuf> = None;
    let mut signers = Vec::new();
    let mut threshold = 1usize;
    let mut chain_id: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--snapshot" if i + 1 < args.len() => {
                snapshot_dir = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--db" if i + 1 < args.len() => {
                target = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--backup" if i + 1 < args.len() => {
                backup = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--chain-id" if i + 1 < args.len() => {
                chain_id = Some(args[i + 1].clone());
                i += 1;
            }
            "--signer" if i + 1 < args.len() => {
                signers.push(args[i + 1].clone());
                i += 1;
            }
            "--threshold" if i + 1 < args.len() => {
                threshold = args[i + 1]
                    .parse()
                    .map_err(|_| format!("invalid --threshold `{}`", args[i + 1]))?;
                i += 1;
            }
            other => return Err(format!("unknown or incomplete argument `{other}`").into()),
        }
        i += 1;
    }

    let snapshot_dir = snapshot_dir.ok_or("apply requires --snapshot <dir>")?;
    let target = target.ok_or("apply requires --db <target>")?;
    let backup = backup.ok_or("apply requires --backup <dir>")?;
    let chain_id = chain_id.ok_or("apply requires --chain-id <id>")?;
    let signers = with_env_signers(signers)?;
    let model_dir = std::env::var("VNN_MODEL_DIR").ok().map(PathBuf::from);

    let report = apply_at(
        &snapshot_dir,
        &target,
        &backup,
        &signers,
        threshold,
        &chain_id,
        model_dir.as_deref(),
    )?;
    println!("snapshot applied: {}", snapshot_dir.display());
    println!("height: {}", report.verify.manifest.height);
    println!("manifest hash: {}", report.verify.manifest_hash);
    println!("target: {}", target.display());
    if let Some(path) = &report.backup {
        println!("backup: {}", path.display());
    } else {
        println!("backup: none (target did not exist)");
    }
    println!("signers: {}", report.verify.signers.join(","));
    Ok(())
}

fn help_requested(args: &[String]) -> bool {
    args.iter()
        .any(|arg| arg == "--help" || arg == "-h" || arg == "help")
}

fn with_env_signers(mut signers: Vec<String>) -> Result<Vec<String>, String> {
    if signers.is_empty() {
        if let Ok(raw) = std::env::var(SIGNERS_ENV) {
            signers.extend(
                raw.split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(String::from),
            );
        }
    }
    if signers.is_empty() {
        return Err(format!(
            "at least one --signer or {SIGNERS_ENV} entry is required"
        ));
    }
    Ok(signers)
}

#[derive(Debug)]
struct CreateReport {
    manifest: SnapshotManifest,
    manifest_hash: String,
    signer_address: String,
    signer_public_key: String,
}

#[derive(Debug)]
struct VerifyReport {
    manifest: SnapshotManifest,
    manifest_hash: String,
    signers: Vec<String>,
    verified: VerifiedSnapshot,
}

#[derive(Debug)]
struct ApplyReport {
    verify: VerifyReport,
    backup: Option<PathBuf>,
}

fn create_at(
    db: &Path,
    out: &Path,
    key_seed: &str,
    expected_signer_hex: &[String],
    model_dir: &Path,
) -> Result<CreateReport, String> {
    if !db.exists() {
        return Err(format!("database path does not exist: {}", db.display()));
    }
    if !db.is_dir() {
        return Err(format!(
            "database path is not a directory: {}",
            db.display()
        ));
    }
    let expected_signers = parse_signer_keys(expected_signer_hex)?;
    let (signing, public_key, address) = signing_key_from_seed(key_seed)?;
    if !expected_signers.is_empty() && !expected_signers.contains_key(&public_key) {
        return Err(format!(
            "signing key {public_key} is not one of the --signer public keys"
        ));
    }

    require_model_directory(model_dir)?;

    // Opening read-write also requires the node to be stopped: RocksDB refuses
    // a second writer while the live process holds the lock.
    let store = StateStore::open(db).map_err(|error| {
        format!(
            "cannot open database {} (is the node still running?): {error}",
            db.display()
        )
    })?;
    let state_root = verinoo_exec::engine::compute_state_root(&store);
    let head = validate_checkpoint_head(&store, &state_root, None)?;
    let base_fee = store.get_meta("baseFee").and_then(|value| value.as_u64());
    let (model_hashes, model_set_hash, problems) = local_model_hashes(Some(model_dir))?;
    for problem in problems {
        eprintln!("model warning: {problem}");
    }
    if model_hashes.len() <= 1 {
        return Err(format!(
            "model directory {} loaded no models beyond the builtin; refusing to sign a builtin-only snapshot",
            model_dir.display()
        ));
    }
    let chain_id = ensure_database_chain_id(&store, &format!("database {}", db.display()))?;
    let validator_set_hash = readiness::validator_set_hash(&store.validator_addresses());
    let metadata = SnapshotMetadata {
        chain_id,
        height: head.header.index,
        block_hash: head.header.hash.clone(),
        parent_hash: head.header.previous_hash.clone(),
        state_root,
        base_fee,
        model_set_hash,
        model_hashes,
        validator_set_hash,
        created_at: now_ms(),
    };

    let mut manifest = create_snapshot(&store, out, metadata)
        .map_err(|error| format!("create snapshot failed: {error}"))?;
    let signature = sign_manifest(&manifest, &signing)?;
    manifest.signatures.push(SnapshotSignature {
        address: address.clone(),
        public_key: public_key.clone(),
        signature,
    });
    write_snapshot_manifest(out, &manifest)
        .map_err(|error| format!("writing signed manifest failed: {error}"))?;
    let manifest_hash = hash_manifest_file(out)?;
    Ok(CreateReport {
        manifest,
        manifest_hash,
        signer_address: address,
        signer_public_key: public_key,
    })
}

fn verify_at(
    snapshot_dir: &Path,
    signer_hex: &[String],
    threshold: usize,
    expected_chain_id: Option<&str>,
    model_dir: Option<&Path>,
) -> Result<VerifyReport, String> {
    let allowed = parse_signer_keys(signer_hex)?;
    let candidate = read_snapshot_manifest(snapshot_dir)
        .map_err(|error| format!("reading snapshot manifest failed: {error}"))?;
    if let Some(expected) = expected_chain_id {
        if candidate.chain_id != expected {
            return Err(format!(
                "snapshot chain id mismatch: expected {expected}, got {}",
                candidate.chain_id
            ));
        }
    }
    let signers = verify_manifest_signatures(&candidate, &allowed, threshold)?;

    // The state layer re-reads the manifest and returns a digest-bound handle.
    // Require the manifest it verified to equal the one the signatures covered,
    // then pass the digest into apply so a later swap is rejected.
    let expectations = SnapshotExpectations::from_metadata(&manifest_metadata(&candidate));
    let verified = verify_snapshot_verified(snapshot_dir, &expectations)
        .map_err(|error| format!("snapshot file verification failed: {error}"))?;
    if verified.manifest() != &candidate {
        return Err("snapshot manifest changed during verification".into());
    }
    let manifest = verified.manifest();
    if let Some(expected) = expected_chain_id {
        if manifest.chain_id != expected {
            return Err(format!(
                "snapshot chain id mismatch: expected {expected}, got {}",
                manifest.chain_id
            ));
        }
    }

    let state_dir = snapshot_dir.join("state");
    let store = StateStore::open_read_only(&state_dir).map_err(|error| {
        format!(
            "opening checkpoint {} read-only failed: {error}",
            state_dir.display()
        )
    })?;
    let recomputed_root = verinoo_exec::engine::compute_state_root(&store);
    if recomputed_root != manifest.state_root {
        return Err(format!(
            "snapshot state root mismatch: manifest {}, recomputed {recomputed_root}",
            manifest.state_root
        ));
    }
    validate_checkpoint_head(&store, &recomputed_root, Some(manifest))?;
    let validator_set_hash = readiness::validator_set_hash(&store.validator_addresses());
    if validator_set_hash != manifest.validator_set_hash {
        return Err(format!(
            "snapshot validator set hash mismatch: manifest {}, recomputed {validator_set_hash}",
            manifest.validator_set_hash
        ));
    }
    drop(store);

    if let Some(dir) = model_dir {
        require_model_directory(dir)?;
    }
    let (local_hashes, local_set_hash, problems) = local_model_hashes(model_dir)?;
    for problem in problems {
        eprintln!("model warning: {problem}");
    }
    if local_set_hash != manifest.model_set_hash {
        return Err(format!(
            "snapshot model set hash mismatch: manifest {}, local {local_set_hash}; set VNN_MODEL_DIR to the same model directory",
            manifest.model_set_hash
        ));
    }
    if local_hashes != manifest.model_hashes {
        return Err(first_model_hash_mismatch(
            &manifest.model_hashes,
            &local_hashes,
        ));
    }

    let manifest_hash = verified.manifest_digest();
    Ok(VerifyReport {
        manifest: verified.manifest().clone(),
        manifest_hash,
        signers,
        verified,
    })
}

fn apply_at(
    snapshot_dir: &Path,
    target: &Path,
    backup: &Path,
    signer_hex: &[String],
    threshold: usize,
    chain_id: &str,
    model_dir: Option<&Path>,
) -> Result<ApplyReport, String> {
    let report = verify_at(
        snapshot_dir,
        signer_hex,
        threshold,
        Some(chain_id),
        model_dir,
    )?;
    if report.manifest.chain_id != chain_id {
        return Err(format!(
            "snapshot chain id mismatch: expected {chain_id}, got {}",
            report.manifest.chain_id
        ));
    }

    // Hold the target's RocksDB write lock (not just a probe) for the whole
    // rename. A node started after the probe therefore cannot open the old
    // database while it is being moved aside.
    let target_exists = match std::fs::symlink_metadata(target) {
        Ok(meta) => {
            if meta.file_type().is_symlink() || !meta.is_dir() {
                return Err(format!(
                    "apply target is not a directory: {}",
                    target.display()
                ));
            }
            true
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => {
            return Err(format!(
                "cannot inspect apply target {}: {error}",
                target.display()
            ))
        }
    };

    let _target_lock = if target_exists {
        let lock = StateStore::open_existing(target).map_err(|error| {
            format!(
                "target database {} is not closed or cannot be opened (stop the node before apply): {error}",
                target.display()
            )
        })?;
        let target_chain_id =
            database_chain_id(&lock, &format!("target database {}", target.display()))?;
        if target_chain_id != chain_id {
            return Err(format!(
                "target chain id mismatch: snapshot {}, target {target_chain_id}",
                chain_id
            ));
        }
        if let Some(head) = verinoo_exec::engine::current_head(&lock) {
            if report.manifest.height <= head.header.index {
                return Err(format!(
                    "snapshot height {} is not greater than target head height {}; refusing to apply a stale or same-height snapshot",
                    report.manifest.height, head.header.index
                ));
            }
        }
        Some(lock)
    } else {
        None
    };

    let expectations = SnapshotExpectations::from_metadata(&manifest_metadata(&report.manifest));
    apply_verified_snapshot_bound(
        snapshot_dir,
        &report.verified,
        &expectations,
        target,
        backup,
    )
    .map_err(|error| format!("snapshot apply failed: {error}"))?;
    drop(_target_lock);

    Ok(ApplyReport {
        backup: target_exists.then(|| backup.to_path_buf()),
        verify: report,
    })
}

fn validate_checkpoint_head(
    store: &StateStore,
    computed_state_root: &str,
    expected: Option<&SnapshotManifest>,
) -> Result<Block, String> {
    let head = verinoo_exec::engine::current_head(store).ok_or("checkpoint has no chain head")?;
    if head.header.version < MIN_HEADER_VERSION || head.header.version > HEADER_VERSION {
        return Err(format!(
            "checkpoint head version {} is outside {}..={}",
            head.header.version, MIN_HEADER_VERSION, HEADER_VERSION
        ));
    }
    let recomputed = verinoo_exec::engine::hash_header(&head.header);
    if recomputed != head.header.hash {
        return Err(format!(
            "checkpoint head hash mismatch: header {}, recomputed {recomputed}",
            head.header.hash
        ));
    }
    if head.header.state_root != computed_state_root {
        return Err(format!(
            "checkpoint head state root mismatch: header {}, recomputed {computed_state_root}",
            head.header.state_root
        ));
    }
    if let Some(manifest) = expected {
        if head.header.index != manifest.height {
            return Err(format!(
                "snapshot head height mismatch: manifest {}, checkpoint {}",
                manifest.height, head.header.index
            ));
        }
        if head.header.hash != manifest.block_hash {
            return Err(format!(
                "snapshot head block hash mismatch: manifest {}, checkpoint {}",
                manifest.block_hash, head.header.hash
            ));
        }
        if head.header.previous_hash != manifest.parent_hash {
            return Err(format!(
                "snapshot parent hash mismatch: manifest {}, checkpoint {}",
                manifest.parent_hash, head.header.previous_hash
            ));
        }
        if head.header.state_root != manifest.state_root {
            return Err(format!(
                "snapshot head state root mismatch: manifest {}, checkpoint {}",
                manifest.state_root, head.header.state_root
            ));
        }
    }
    Ok(head)
}

/// Read the database chain id, recording the compiled chain id on databases
/// that predate the metadata. Create opens the database read-write anyway.
fn ensure_database_chain_id(store: &StateStore, label: &str) -> Result<String, String> {
    if let Some(id) = store
        .get_meta("chainId")
        .and_then(|value| value.as_str().map(str::to_owned))
        .filter(|id| !id.trim().is_empty())
    {
        return Ok(id);
    }
    let id = verinoo_state::genesis::CHAIN_ID.to_string();
    store.set_meta("chainId", &serde_json::json!(id));
    eprintln!("{label} had no chain id; recorded `{id}` before snapshotting");
    Ok(id)
}

fn database_chain_id(store: &StateStore, label: &str) -> Result<String, String> {
    store
        .get_meta("chainId")
        .and_then(|value| value.as_str().map(str::to_owned))
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| {
            format!(
                "{label} has no chain id; start the node once with this release to record it before creating or applying a snapshot"
            )
        })
}

fn require_model_directory(model_dir: &Path) -> Result<(), String> {
    let meta = std::fs::metadata(model_dir).map_err(|error| {
        format!(
            "model directory {} is missing or unreadable: {error}",
            model_dir.display()
        )
    })?;
    if !meta.is_dir() {
        return Err(format!(
            "model directory {} is not a directory",
            model_dir.display()
        ));
    }
    Ok(())
}

fn manifest_metadata(manifest: &SnapshotManifest) -> SnapshotMetadata {
    SnapshotMetadata {
        chain_id: manifest.chain_id.clone(),
        height: manifest.height,
        block_hash: manifest.block_hash.clone(),
        parent_hash: manifest.parent_hash.clone(),
        state_root: manifest.state_root.clone(),
        base_fee: manifest.base_fee,
        model_set_hash: manifest.model_set_hash.clone(),
        model_hashes: manifest.model_hashes.clone(),
        validator_set_hash: manifest.validator_set_hash.clone(),
        created_at: manifest.created_at,
    }
}

fn parse_signer_keys(values: &[String]) -> Result<BTreeMap<String, VerifyingKey>, String> {
    let mut keys = BTreeMap::new();
    for value in values {
        let public_key = normalize_public_key(value)?;
        let bytes: [u8; 32] = hex::decode(&public_key)
            .map_err(|_| format!("signer public key is not hex: {value}"))?
            .try_into()
            .map_err(|_| format!("signer public key must be 32 bytes: {value}"))?;
        let verifying = VerifyingKey::from_bytes(&bytes)
            .map_err(|error| format!("invalid signer public key {public_key}: {error}"))?;
        if keys.insert(public_key.clone(), verifying).is_some() {
            return Err(format!("duplicate signer public key {public_key}"));
        }
    }
    Ok(keys)
}

fn normalize_public_key(value: &str) -> Result<String, String> {
    let value = value.trim();
    if value.is_empty() {
        return Err("signer public key is empty".into());
    }
    let bytes = hex::decode(value).map_err(|_| format!("signer public key is not hex: {value}"))?;
    if bytes.len() != 32 {
        return Err(format!(
            "signer public key must be 32 bytes, got {}: {value}",
            bytes.len()
        ));
    }
    Ok(hex::encode(bytes))
}

fn signing_key_from_seed(raw: &str) -> Result<(ed25519_dalek::SigningKey, String, String), String> {
    if raw.trim().is_empty() {
        return Err("snapshot signing seed is empty".into());
    }
    let mut hasher = Sha256::new();
    hasher.update(raw.as_bytes());
    let seed: [u8; 32] = hasher.finalize().into();
    let signing = ed25519_dalek::SigningKey::from_bytes(&seed);
    let public_key = hex::encode(signing.verifying_key().to_bytes());
    let address = verinoo_exec::engine::public_key_to_address(&public_key)
        .map_err(|error| format!("deriving signer address failed: {error}"))?;
    Ok((signing, public_key, address))
}

fn sign_manifest(
    manifest: &SnapshotManifest,
    signing: &ed25519_dalek::SigningKey,
) -> Result<String, String> {
    let bytes = manifest_signing_bytes(manifest)
        .map_err(|error| format!("building manifest signing bytes failed: {error}"))?;
    Ok(hex::encode(signing.sign(&bytes).to_bytes()))
}

fn verify_manifest_signatures(
    manifest: &SnapshotManifest,
    allowed: &BTreeMap<String, VerifyingKey>,
    threshold: usize,
) -> Result<Vec<String>, String> {
    if threshold == 0 {
        return Err("signature threshold must be at least 1".into());
    }
    if threshold > allowed.len() {
        return Err(format!(
            "signature threshold {threshold} exceeds {} configured signer(s)",
            allowed.len()
        ));
    }
    if manifest.signatures.len() > MAX_SNAPSHOT_SIGNATURES {
        return Err(format!(
            "snapshot has too many signatures: {} (max {MAX_SNAPSHOT_SIGNATURES})",
            manifest.signatures.len()
        ));
    }
    let bytes = manifest_signing_bytes(manifest)
        .map_err(|error| format!("building manifest signing bytes failed: {error}"))?;
    let domain = format!("{SIGNING_DOMAIN}\n");
    if !bytes.starts_with(domain.as_bytes()) {
        return Err("manifest signing bytes have the wrong domain".into());
    }

    let mut seen = BTreeSet::new();
    for signature in &manifest.signatures {
        let public_key = normalize_public_key(&signature.public_key)?;
        let Some(verifying) = allowed.get(&public_key) else {
            return Err(format!(
                "unknown snapshot signer {public_key} (address {})",
                signature.address
            ));
        };
        let address = verinoo_exec::engine::public_key_to_address(&public_key)
            .map_err(|error| format!("deriving signer address failed: {error}"))?;
        if signature.address != address {
            return Err(format!(
                "snapshot signature address {} does not match public key {public_key} (derived {address})",
                signature.address
            ));
        }
        if !seen.insert(address.clone()) {
            return Err(format!("duplicate snapshot signer {address}"));
        }
        let signature_bytes: [u8; 64] = hex::decode(&signature.signature)
            .map_err(|_| format!("snapshot signature is not hex for {address}"))?
            .try_into()
            .map_err(|_| format!("snapshot signature must be 64 bytes for {address}"))?;
        let signature = Signature::from_bytes(&signature_bytes);
        if verifying.verify(&bytes, &signature).is_err() {
            return Err(format!("invalid snapshot signature from {address}"));
        }
    }
    if seen.len() < threshold {
        return Err(format!(
            "snapshot has {} valid signer(s), threshold is {threshold}",
            seen.len()
        ));
    }
    Ok(seen.into_iter().collect())
}

fn local_model_hashes(
    model_dir: Option<&Path>,
) -> Result<(ModelHashes, String, Vec<String>), String> {
    let mut descriptors = vec![verinoo_model::descriptor()];
    let mut problems = Vec::new();
    if let Some(dir) = model_dir {
        let (graph, graph_problems) = verinoo_runtime::Registry::load_dir(dir);
        problems.extend(graph_problems);
        descriptors.extend(graph.descriptors());

        let (transformers, transformer_problems) =
            verinoo_runtime::transformer::TransformerRegistry::load_dir(dir);
        problems.extend(transformer_problems);
        descriptors.extend(transformers.descriptors());

        let (decoders, decoder_problems) = verinoo_runtime::gpt::GptRegistry::load_dir(dir);
        problems.extend(decoder_problems);
        descriptors.extend(decoders.descriptors());
    }
    let (hashes, set_hash) = model_hashes_from_descriptors(&descriptors)?;
    Ok((hashes, set_hash, problems))
}

fn model_hashes_from_descriptors(
    descriptors: &[serde_json::Value],
) -> Result<(ModelHashes, String), String> {
    let mut hashes = BTreeMap::new();
    for descriptor in descriptors {
        let model_id = descriptor
            .get("modelId")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "local model descriptor is missing modelId".to_string())?;
        let weights_hash = descriptor
            .get("weightsHash")
            .and_then(|value| value.as_str())
            .ok_or_else(|| format!("local model `{model_id}` is missing weightsHash"))?;
        if hashes
            .insert(model_id.to_string(), weights_hash.to_string())
            .is_some()
        {
            return Err(format!(
                "duplicate model id `{model_id}` in the local model registries"
            ));
        }
    }
    let mut lines: Vec<String> = hashes
        .iter()
        .map(|(model_id, weights_hash)| format!("{model_id}:{weights_hash}"))
        .collect();
    lines.sort();
    Ok((hashes, sha256_hex(lines.join("\n").as_bytes())))
}

fn first_model_hash_mismatch(expected: &ModelHashes, local: &ModelHashes) -> String {
    let ids: BTreeSet<&String> = expected.keys().chain(local.keys()).collect();
    for id in ids {
        if expected.get(id) != local.get(id) {
            return format!(
                "snapshot model hash mismatch for {id}: manifest {}, local {}; set VNN_MODEL_DIR to the same model directory",
                expected.get(id).map(String::as_str).unwrap_or("<missing>"),
                local.get(id).map(String::as_str).unwrap_or("<missing>")
            );
        }
    }
    "snapshot model hash maps differ".into()
}

fn hash_manifest_file(snapshot_dir: &Path) -> Result<String, String> {
    let path = snapshot_dir.join(MANIFEST_NAME);
    let bytes = std::fs::read(&path)
        .map_err(|error| format!("reading {} failed: {error}", path.display()))?;
    Ok(sha256_hex(&bytes))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir().join(format!(
                "verinoo-node-snapshot-{tag}-{}-{n}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn empty_model_dir(&self) -> PathBuf {
            let path = self.0.join("models-empty");
            fs::create_dir_all(&path).unwrap();
            path
        }

        fn model_dir(&self) -> PathBuf {
            let path = self.0.join("models");
            fs::create_dir_all(&path).unwrap();
            let source = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../models/intent-transformer-v1.json");
            fs::copy(source, path.join("intent-transformer-v1.json")).unwrap();
            path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn valid_block(store: &StateStore, index: u64, parent: &str) -> Block {
        let mut header = verinoo_exec::engine::BlockHeader {
            hash: String::new(),
            version: HEADER_VERSION,
            index,
            timestamp: 1_000,
            previous_hash: parent.into(),
            validator: "validator".into(),
            validator_pub_key: String::new(),
            validator_signature: "sig".into(),
            state_root: verinoo_exec::engine::compute_state_root(store),
            tx_root: "00".repeat(32),
            inference_tasks_root: "00".repeat(32),
            compute_root: "00".repeat(32),
        };
        header.hash = verinoo_exec::engine::hash_header(&header);
        Block {
            header,
            transactions: vec![],
        }
    }

    fn install_head(store: &StateStore, index: u64) {
        let block = valid_block(store, index, &"bb".repeat(32));
        verinoo_exec::engine::set_head(store, &block);
    }

    fn populated_store(path: &Path) -> StateStore {
        let store = StateStore::open(path).unwrap();
        store.set_account(&verinoo_state::models::Account {
            address: "ai_alice".into(),
            public_key: "pk".into(),
            nonce: 2,
            balance: 500,
            updated_at: 9,
        });
        store.set_meta_u64("baseFee", 3);
        store.set_meta(
            "chainId",
            &serde_json::json!(verinoo_state::genesis::CHAIN_ID),
        );
        install_head(&store, 7);
        store
    }

    fn target_store(path: &Path, chain_id: &str, height: Option<u64>) -> StateStore {
        let store = StateStore::open(path).unwrap();
        store.set_meta("chainId", &serde_json::json!(chain_id));
        store.set_meta_u64("baseFee", 99);
        if let Some(height) = height {
            install_head(&store, height);
        }
        store
    }

    fn signer_seed(seed: &str) -> (String, String) {
        let (_, public_key, address) = signing_key_from_seed(seed).unwrap();
        (public_key, address)
    }

    fn fixture(tag: &str) -> (TempDir, PathBuf, PathBuf, String) {
        let root = TempDir::new(tag);
        let db = root.path().join("db");
        let snapshot_dir = root.path().join("snapshot");
        let models = root.model_dir();
        {
            let _store = populated_store(&db);
        }
        let (public_key, _) = signer_seed("operator-seed");
        create_at(&db, &snapshot_dir, "operator-seed", &[], &models).unwrap();
        (root, snapshot_dir, models, public_key)
    }

    #[test]
    fn create_verify_and_apply_roundtrip() {
        let (root, snapshot_dir, models, public_key) = fixture("roundtrip");
        let verified = verify_at(
            &snapshot_dir,
            std::slice::from_ref(&public_key),
            1,
            Some(verinoo_state::genesis::CHAIN_ID),
            Some(&models),
        )
        .unwrap();
        assert_eq!(verified.manifest.height, 7);
        assert_eq!(verified.manifest.base_fee, Some(3));

        let target = root.path().join("target");
        let backup = root.path().join("backup");
        drop(target_store(
            &target,
            verinoo_state::genesis::CHAIN_ID,
            Some(3),
        ));
        let report = apply_at(
            &snapshot_dir,
            &target,
            &backup,
            &[public_key],
            1,
            verinoo_state::genesis::CHAIN_ID,
            Some(&models),
        )
        .unwrap();
        assert_eq!(report.backup.as_deref(), Some(backup.as_path()));
        assert!(backup.is_dir());
        let applied = StateStore::open(&target).unwrap();
        assert_eq!(applied.get_account("ai_alice").unwrap().balance, 500);
        assert_eq!(applied.meta_u64_or("baseFee", 0), 3);
    }

    #[test]
    fn signature_checks_reject_unknown_duplicate_invalid_and_wrong_domain() {
        let (_root, snapshot_dir, _models, public_key) = fixture("signatures");
        let manifest = read_snapshot_manifest(&snapshot_dir).unwrap();
        let allowed = parse_signer_keys(std::slice::from_ref(&public_key)).unwrap();
        verify_manifest_signatures(&manifest, &allowed, 1).unwrap();

        let (unknown_key, _) = signer_seed("unknown-seed");
        let unknown = parse_signer_keys(&[unknown_key]).unwrap();
        assert!(verify_manifest_signatures(&manifest, &unknown, 1)
            .unwrap_err()
            .contains("unknown snapshot signer"));

        let mut duplicate = manifest.clone();
        let first_signature = duplicate.signatures[0].clone();
        duplicate.signatures.push(first_signature);
        assert!(verify_manifest_signatures(&duplicate, &allowed, 1)
            .unwrap_err()
            .contains("duplicate snapshot signer"));

        let mut invalid = manifest.clone();
        invalid.signatures[0].signature = "00".repeat(64);
        assert!(verify_manifest_signatures(&invalid, &allowed, 1)
            .unwrap_err()
            .contains("invalid snapshot signature"));

        let mut wrong_address = manifest.clone();
        wrong_address.signatures[0].address = "ai_".to_string() + &"11".repeat(20);
        assert!(verify_manifest_signatures(&wrong_address, &allowed, 1)
            .unwrap_err()
            .contains("does not match public key"));

        let (signing, _, address) = signing_key_from_seed("operator-seed").unwrap();
        let mut wrong_domain = manifest.clone();
        let mut bytes = b"verinoo/not-snapshot/v1\n".to_vec();
        bytes.extend_from_slice(
            String::from_utf8(manifest_signing_bytes(&manifest).unwrap())
                .unwrap()
                .split_once('\n')
                .unwrap()
                .1
                .as_bytes(),
        );
        wrong_domain.signatures[0] = SnapshotSignature {
            address,
            public_key: public_key.clone(),
            signature: hex::encode(signing.sign(&bytes).to_bytes()),
        };
        assert!(verify_manifest_signatures(&wrong_domain, &allowed, 1)
            .unwrap_err()
            .contains("invalid snapshot signature"));

        assert!(verify_manifest_signatures(&manifest, &allowed, 2)
            .unwrap_err()
            .contains("threshold"));
        assert!(verify_manifest_signatures(&manifest, &allowed, 0)
            .unwrap_err()
            .contains("at least 1"));
    }

    #[test]
    fn threshold_requires_enough_distinct_signers() {
        let (_root, snapshot_dir, models, _) = fixture("threshold");
        let mut manifest = read_snapshot_manifest(&snapshot_dir).unwrap();
        let (second_key, second_address) = signer_seed("second-seed");
        let (second_signing, _, _) = signing_key_from_seed("second-seed").unwrap();
        let bytes = manifest_signing_bytes(&manifest).unwrap();
        manifest.signatures.push(SnapshotSignature {
            address: second_address,
            public_key: second_key.clone(),
            signature: hex::encode(second_signing.sign(&bytes).to_bytes()),
        });
        write_snapshot_manifest(&snapshot_dir, &manifest).unwrap();
        let (first_key, _) = signer_seed("operator-seed");

        verify_at(
            &snapshot_dir,
            &[first_key.clone(), second_key.clone()],
            2,
            None,
            Some(&models),
        )
        .unwrap();
        assert!(
            verify_at(&snapshot_dir, &[first_key], 2, None, Some(&models))
                .unwrap_err()
                .contains("exceeds")
        );
    }

    #[test]
    fn create_pins_the_signing_key_to_explicit_signers() {
        let root = TempDir::new("pin");
        let db = root.path().join("db");
        let snapshot_dir = root.path().join("snapshot");
        let models = root.model_dir();
        drop(populated_store(&db));
        let (other_key, _) = signer_seed("other-seed");
        let error =
            create_at(&db, &snapshot_dir, "operator-seed", &[other_key], &models).unwrap_err();
        assert!(error.contains("is not one of the --signer"));
    }

    #[test]
    fn read_only_probe_rejects_a_running_target() {
        let (root, snapshot_dir, models, public_key) = fixture("locked");
        let target = root.path().join("target");
        let backup = root.path().join("backup");
        drop(target_store(
            &target,
            verinoo_state::genesis::CHAIN_ID,
            Some(1),
        ));
        let live = StateStore::open(&target).unwrap();
        let error = apply_at(
            &snapshot_dir,
            &target,
            &backup,
            &[public_key],
            1,
            verinoo_state::genesis::CHAIN_ID,
            Some(&models),
        )
        .unwrap_err();
        assert!(error.contains("is not closed"), "{error}");
        drop(live);
        let (public_key, _) = signer_seed("operator-seed");
        apply_at(
            &snapshot_dir,
            &target,
            &backup,
            &[public_key],
            1,
            verinoo_state::genesis::CHAIN_ID,
            Some(&models),
        )
        .unwrap();
    }

    #[test]
    fn model_mismatch_fails_closed() {
        let (_root, snapshot_dir, _models, public_key) = fixture("models");
        // The fixture created the snapshot with one installed model; loading
        // the repository models (or a different dir) must not be trusted.
        let repo_models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        let error =
            verify_at(&snapshot_dir, &[public_key], 1, None, Some(&repo_models)).unwrap_err();
        assert!(error.contains("model"), "{error}");
    }

    #[test]
    fn apply_rejects_chain_mismatch_stale_and_same_height_targets() {
        let (root, snapshot_dir, models, public_key) = fixture("stale");
        let signers = [public_key];
        let chain = verinoo_state::genesis::CHAIN_ID;

        let target = root.path().join("target-chain");
        let backup = root.path().join("backup-chain");
        drop(target_store(&target, "other-chain", Some(1)));
        let error = apply_at(
            &snapshot_dir,
            &target,
            &backup,
            &signers,
            1,
            chain,
            Some(&models),
        )
        .unwrap_err();
        assert!(error.contains("target chain id mismatch"), "{error}");
        assert!(!backup.exists());
        assert!(target.exists());

        for height in [7, 8] {
            let target = root.path().join(format!("target-{height}"));
            let backup = root.path().join(format!("backup-{height}"));
            drop(target_store(&target, chain, Some(height)));
            let error = apply_at(
                &snapshot_dir,
                &target,
                &backup,
                &signers,
                1,
                chain,
                Some(&models),
            )
            .unwrap_err();
            assert!(error.contains("not greater"), "{error}");
            assert!(!backup.exists());
        }

        let target = root.path().join("target-no-chain");
        let backup = root.path().join("backup-no-chain");
        drop(StateStore::open(&target).unwrap());
        let error = apply_at(
            &snapshot_dir,
            &target,
            &backup,
            &signers,
            1,
            chain,
            Some(&models),
        )
        .unwrap_err();
        assert!(error.contains("has no chain id"), "{error}");
        assert!(!backup.exists());
        assert!(snapshot_dir.join("state").exists());
    }

    #[test]
    fn apply_refuses_an_existing_backup_even_without_a_target() {
        let (root, snapshot_dir, models, public_key) = fixture("backup-absent");
        let target = root.path().join("target-absent");
        let backup = root.path().join("backup");
        fs::create_dir_all(&backup).unwrap();
        fs::write(backup.join("sentinel"), b"keep").unwrap();

        let error = apply_at(
            &snapshot_dir,
            &target,
            &backup,
            &[public_key],
            1,
            verinoo_state::genesis::CHAIN_ID,
            Some(&models),
        )
        .unwrap_err();
        assert!(error.contains("backup path already exists"), "{error}");
        assert!(!target.exists());
        assert_eq!(fs::read(backup.join("sentinel")).unwrap(), b"keep");
        assert!(snapshot_dir.join("state").exists());
    }

    #[test]
    fn tampered_manifest_after_verify_is_not_applied() {
        let (root, snapshot_dir, models, public_key) = fixture("bound-apply");
        let target = root.path().join("target");
        let backup = root.path().join("backup");
        drop(target_store(
            &target,
            verinoo_state::genesis::CHAIN_ID,
            Some(3),
        ));
        let report = verify_at(
            &snapshot_dir,
            &[public_key],
            1,
            Some(verinoo_state::genesis::CHAIN_ID),
            Some(&models),
        )
        .unwrap();

        // `createdAt` is signed but not part of the apply expectations, so a
        // swapped manifest still passes expectation checks. The digest bound
        // into `report.verified` is what rejects it.
        let manifest_path = snapshot_dir.join(MANIFEST_NAME);
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        value["createdAt"] = serde_json::json!(value["createdAt"].as_u64().unwrap() + 1);
        fs::write(&manifest_path, serde_json::to_vec(&value).unwrap()).unwrap();

        let expectations =
            SnapshotExpectations::from_metadata(&manifest_metadata(&report.manifest));
        let error = apply_verified_snapshot_bound(
            &snapshot_dir,
            &report.verified,
            &expectations,
            &target,
            &backup,
        )
        .unwrap_err();
        assert!(
            matches!(
                error,
                verinoo_state::snapshot::SnapshotError::ManifestChanged { .. }
            ),
            "{error:?}"
        );
        assert!(!backup.exists());
        let untouched = StateStore::open(&target).unwrap();
        assert!(untouched.get_account("ai_alice").is_none());
        assert_eq!(untouched.meta_u64_or("baseFee", 0), 99);
    }

    #[test]
    fn create_requires_a_model_directory_with_installed_models() {
        let root = TempDir::new("model-required");
        let db = root.path().join("db");
        drop(populated_store(&db));

        let missing = root.path().join("missing-models");
        let out = root.path().join("snapshot-missing");
        let error = create_at(&db, &out, "operator-seed", &[], &missing).unwrap_err();
        assert!(error.contains("model directory"), "{error}");
        assert!(!out.exists());

        let empty = root.empty_model_dir();
        let out = root.path().join("snapshot-empty");
        let error = create_at(&db, &out, "operator-seed", &[], &empty).unwrap_err();
        assert!(error.contains("loaded no models"), "{error}");
        assert!(!out.exists());
    }

    #[test]
    fn checkpoint_head_validation_rejects_forged_headers() {
        let root = TempDir::new("forged-head");
        let db = root.path().join("db");
        let store = populated_store(&db);
        let computed = verinoo_exec::engine::compute_state_root(&store);
        validate_checkpoint_head(&store, &computed, None).unwrap();

        let mut block = valid_block(&store, 7, &"bb".repeat(32));
        block.header.hash = "aa".repeat(32);
        verinoo_exec::engine::set_head(&store, &block);
        let error = validate_checkpoint_head(&store, &computed, None).unwrap_err();
        assert!(error.contains("hash mismatch"), "{error}");

        let mut block = valid_block(&store, 7, &"bb".repeat(32));
        block.header.version = HEADER_VERSION + 1;
        block.header.hash = verinoo_exec::engine::hash_header(&block.header);
        verinoo_exec::engine::set_head(&store, &block);
        let error = validate_checkpoint_head(&store, &computed, None).unwrap_err();
        assert!(error.contains("version"), "{error}");

        let mut block = valid_block(&store, 7, &"bb".repeat(32));
        block.header.state_root = "00".repeat(32);
        block.header.hash = verinoo_exec::engine::hash_header(&block.header);
        verinoo_exec::engine::set_head(&store, &block);
        let error = validate_checkpoint_head(&store, &computed, None).unwrap_err();
        assert!(error.contains("state root mismatch"), "{error}");
    }

    #[test]
    fn signature_list_is_capped() {
        let (_root, snapshot_dir, _models, public_key) = fixture("signature-cap");
        let mut manifest = read_snapshot_manifest(&snapshot_dir).unwrap();
        let first = manifest.signatures[0].clone();
        while manifest.signatures.len() <= MAX_SNAPSHOT_SIGNATURES {
            manifest.signatures.push(first.clone());
        }
        let allowed = parse_signer_keys(&[public_key]).unwrap();
        let error = verify_manifest_signatures(&manifest, &allowed, 1).unwrap_err();
        assert!(error.contains("too many signatures"), "{error}");
    }

    #[test]
    fn signer_parsing_rejects_duplicates_and_bad_hex() {
        let (public_key, _) = signer_seed("operator-seed");
        assert!(parse_signer_keys(&[public_key.clone(), public_key])
            .unwrap_err()
            .contains("duplicate"));
        assert!(parse_signer_keys(&["zz".into()])
            .unwrap_err()
            .contains("not hex"));
        assert!(parse_signer_keys(&["00".into()])
            .unwrap_err()
            .contains("32 bytes"));
    }
}
