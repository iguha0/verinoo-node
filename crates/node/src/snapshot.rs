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

use verinoo_state::snapshot::{
    apply_verified_snapshot, create_snapshot, manifest_signing_bytes, read_snapshot_manifest,
    verify_snapshot, write_snapshot_manifest, ModelHashes, SnapshotExpectations, SnapshotManifest,
    SnapshotMetadata, SnapshotSignature, SIGNING_DOMAIN,
};
use verinoo_state::StateStore;

use crate::readiness;

const MANIFEST_NAME: &str = "manifest.json";
const DEFAULT_MODEL_DIR: &str = "./models";
const SIGNERS_ENV: &str = "VNN_SNAPSHOT_SIGNERS";

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
    println!("  apply  --snapshot <dir> --db <target> --backup <dir> --signer <pubkey-hex> [--signer ...] [--threshold N]");
    println!();
    println!("Environment:");
    println!("  VNN_KEY_SEED          default signing seed for `create`");
    println!("  VNN_SNAPSHOT_SIGNERS  comma-separated signer public keys for verify/apply");
    println!("  VNN_MODEL_DIR         model directory (create default: ./models)");
    println!();
    println!("Trust boundary: snapshot signatures are operator-attested, not BFT.");
    println!("verify recomputes the state root and validator hash from the checkpoint");
    println!("and compares the local model set. The target database must be stopped for");
    println!("apply; the old database is moved to --backup and never deleted.");
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
    let signers = with_env_signers(signers)?;
    let model_dir = std::env::var("VNN_MODEL_DIR").ok().map(PathBuf::from);

    let report = apply_at(
        &snapshot_dir,
        &target,
        &backup,
        &signers,
        threshold,
        model_dir.as_deref(),
    )?;
    println!("snapshot applied: {}", snapshot_dir.display());
    println!("height: {}", report.manifest.height);
    println!("manifest hash: {}", report.manifest_hash);
    println!("target: {}", target.display());
    if backup.exists() {
        println!("backup: {}", backup.display());
    } else {
        println!("backup: none (target did not exist)");
    }
    println!("signers: {}", report.signers.join(","));
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

    // Opening read-write also requires the node to be stopped: RocksDB refuses
    // a second writer while the live process holds the lock.
    let store = StateStore::open(db).map_err(|error| {
        format!(
            "cannot open database {} (is the node still running?): {error}",
            db.display()
        )
    })?;
    let head = verinoo_exec::engine::current_head(&store)
        .ok_or_else(|| format!("database {} has no chain head", db.display()))?;
    let state_root = verinoo_exec::engine::compute_state_root(&store);
    let base_fee = store.get_meta("baseFee").and_then(|value| value.as_u64());
    let (model_hashes, model_set_hash, problems) = local_model_hashes(Some(model_dir))?;
    for problem in problems {
        eprintln!("model warning: {problem}");
    }
    let validator_set_hash = readiness::validator_set_hash(&store.validator_addresses());
    let metadata = SnapshotMetadata {
        chain_id: verinoo_state::genesis::CHAIN_ID.to_string(),
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
    let manifest = read_snapshot_manifest(snapshot_dir)
        .map_err(|error| format!("reading snapshot manifest failed: {error}"))?;
    if let Some(expected) = expected_chain_id {
        if manifest.chain_id != expected {
            return Err(format!(
                "snapshot chain id mismatch: expected {expected}, got {}",
                manifest.chain_id
            ));
        }
    }
    let signers = verify_manifest_signatures(&manifest, &allowed, threshold)?;

    // The state layer re-reads the manifest. Compare the result to the signed
    // manifest so a swapped manifest cannot be verified under the old
    // signatures.
    let expectations = SnapshotExpectations::from_metadata(&manifest_metadata(&manifest));
    let verified = verify_snapshot(snapshot_dir, &expectations)
        .map_err(|error| format!("snapshot file verification failed: {error}"))?;
    if verified != manifest {
        return Err("snapshot manifest changed during verification".into());
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
    let validator_set_hash = readiness::validator_set_hash(&store.validator_addresses());
    if validator_set_hash != manifest.validator_set_hash {
        return Err(format!(
            "snapshot validator set hash mismatch: manifest {}, recomputed {validator_set_hash}",
            manifest.validator_set_hash
        ));
    }
    let head = verinoo_exec::engine::current_head(&store)
        .ok_or_else(|| "snapshot checkpoint has no chain head".to_string())?;
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
    drop(store);

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

    let manifest_hash = hash_manifest_file(snapshot_dir)?;
    Ok(VerifyReport {
        manifest,
        manifest_hash,
        signers,
    })
}

fn apply_at(
    snapshot_dir: &Path,
    target: &Path,
    backup: &Path,
    signer_hex: &[String],
    threshold: usize,
    model_dir: Option<&Path>,
) -> Result<VerifyReport, String> {
    let report = verify_at(snapshot_dir, signer_hex, threshold, None, model_dir)?;

    // A read-only probe would not detect a live writer. Opening read-write
    // without create takes the RocksDB lock, so a running node fails here.
    if target.exists() {
        if !target.is_dir() {
            return Err(format!(
                "apply target is not a directory: {}",
                target.display()
            ));
        }
        let probe = StateStore::open_existing(target).map_err(|error| {
            format!(
                "target database {} is not closed or cannot be opened (stop the node before apply): {error}",
                target.display()
            )
        })?;
        drop(probe);
    }

    let expectations = SnapshotExpectations::from_metadata(&manifest_metadata(&report.manifest));
    apply_verified_snapshot(snapshot_dir, &expectations, target, backup)
        .map_err(|error| format!("snapshot apply failed: {error}"))?;
    Ok(report)
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
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn head(index: u64, hash: &str, parent: &str) -> verinoo_exec::engine::Block {
        verinoo_exec::engine::Block {
            header: verinoo_exec::engine::BlockHeader {
                hash: hash.into(),
                version: verinoo_exec::engine::HEADER_VERSION,
                index,
                timestamp: 1_000,
                previous_hash: parent.into(),
                validator: "validator".into(),
                validator_pub_key: String::new(),
                validator_signature: "sig".into(),
                state_root: "00".repeat(32),
                tx_root: "00".repeat(32),
                inference_tasks_root: "00".repeat(32),
                compute_root: "00".repeat(32),
            },
            transactions: vec![],
        }
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
        let block = head(7, &"aa".repeat(32), &"bb".repeat(32));
        verinoo_exec::engine::set_head(&store, &block);
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
        let models = root.empty_model_dir();
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
        {
            let old = StateStore::open(&target).unwrap();
            old.set_meta_u64("baseFee", 99);
        }
        apply_at(
            &snapshot_dir,
            &target,
            &backup,
            &[public_key],
            1,
            Some(&models),
        )
        .unwrap();
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
        let models = root.empty_model_dir();
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
        let live = StateStore::open(&target).unwrap();
        let error = apply_at(
            &snapshot_dir,
            &target,
            &backup,
            &[public_key],
            1,
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
            Some(&models),
        )
        .unwrap();
    }

    #[test]
    fn model_mismatch_fails_closed() {
        let (_root, snapshot_dir, _models, public_key) = fixture("models");
        // The fixture created the snapshot with the builtin-only set; loading
        // the repository models (or a different dir) must not be trusted.
        let repo_models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        let error =
            verify_at(&snapshot_dir, &[public_key], 1, None, Some(&repo_models)).unwrap_err();
        assert!(error.contains("model"), "{error}");
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
