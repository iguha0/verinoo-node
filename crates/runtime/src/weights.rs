//! Binary weight files for the decoder runtimes.
//!
//! A JSON model is fine for a 5,000-parameter classifier and painful for an
//! 82M-parameter LM. Large decoders keep their metadata (tokenizer, shapes,
//! sampling) in JSON and their parameters in a `.vnnw` file:
//!
//! ```text
//! magic "VNNW"        4 bytes
//! version u32 LE      4 bytes
//! payload length u64  8 bytes
//! payload sha256     32 bytes
//! payload             N * i32 little-endian, in canonical tensor order
//! ```
//!
//! The caller knows the canonical order from the model metadata, so the file
//! does not repeat tensor names. The payload hash is checked on every load and
//! is part of the model identity used in receipts.
//!
//! Binary payloads are mapped and exposed as `&[i32]` without copying. Hosts
//! that are not little-endian, or mappings whose payload is not 4-byte
//! aligned, fall back to an owned copy.

use memmap2::Mmap;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const MAGIC: &[u8; 4] = b"VNNW";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 4 + 4 + 8 + 32;

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn payload_sha256(weights: &[i32]) -> String {
    let mut h = Sha256::new();
    for value in weights {
        h.update(value.to_le_bytes());
    }
    hex::encode(h.finalize())
}

pub fn write(path: &Path, weights: &[i32]) -> Result<String, String> {
    let mut payload = Vec::with_capacity(weights.len() * 4);
    for value in weights {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    let digest = Sha256::digest(&payload);
    let mut bytes = Vec::with_capacity(HEADER_LEN + payload.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&digest);
    bytes.extend_from_slice(&payload);
    write_atomic(path, &bytes)?;
    Ok(hex::encode(digest))
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let temp = temp_path(path);
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|e| format!("writing {}: {e}", temp.display()))?;
        file.write_all(bytes)
            .map_err(|e| format!("writing {}: {e}", temp.display()))?;
        file.sync_all()
            .map_err(|e| format!("syncing {}: {e}", temp.display()))?;
        drop(file);
        std::fs::rename(&temp, path).map_err(|e| format!("replacing {}: {e}", path.display()))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

fn temp_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| "weights".into());
    let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    path.with_file_name(format!(".{name}.tmp-{}-{counter}", std::process::id()))
}

/// Tensor storage for a model: an owned flat buffer for inline JSON, or a
/// read-only mapping whose payload is reinterpreted as little-endian `i32`.
#[derive(Debug, Clone)]
pub struct WeightsData {
    backing: WeightsBacking,
}

#[derive(Debug, Clone)]
enum WeightsBacking {
    Owned(Vec<i32>),
    Mapped {
        mmap: Arc<Mmap>,
        offset: usize,
        len: usize,
    },
}

impl WeightsData {
    pub fn owned(values: Vec<i32>) -> Self {
        Self {
            backing: WeightsBacking::Owned(values),
        }
    }

    fn mapped(mmap: Arc<Mmap>, offset: usize, len: usize) -> Self {
        Self {
            backing: WeightsBacking::Mapped { mmap, offset, len },
        }
    }

    pub fn is_mapped(&self) -> bool {
        matches!(self.backing, WeightsBacking::Mapped { .. })
    }

    pub fn len(&self) -> usize {
        match &self.backing {
            WeightsBacking::Owned(values) => values.len(),
            WeightsBacking::Mapped { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> &[i32] {
        match &self.backing {
            WeightsBacking::Owned(values) => values,
            // SAFETY: `load` only constructs this variant after verifying that
            // the payload starts at a 4-byte-aligned address, is a multiple of
            // four bytes, and that the host is little-endian. Every i32 bit
            // pattern is valid, and the `Arc<Mmap>` keeps the bytes alive for
            // at least as long as `&self`.
            WeightsBacking::Mapped { mmap, offset, len } => unsafe {
                std::slice::from_raw_parts(mmap.as_ptr().add(*offset).cast::<i32>(), *len)
            },
        }
    }

    pub fn tensor(&self, offset: usize, len: usize) -> &[i32] {
        let end = offset
            .checked_add(len)
            .expect("weight tensor range overflow");
        &self.as_slice()[offset..end]
    }
}

pub struct LoadedWeights {
    pub weights: WeightsData,
    pub sha256: String,
}

/// Map and verify a weight file. The mapping is preferred; an owned copy is
/// returned on big-endian hosts, if the payload alignment is unsuitable, or if
/// the platform cannot map the file at all.
pub fn load(path: &Path) -> Result<LoadedWeights, String> {
    let file = File::open(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let file_len = file
        .metadata()
        .map_err(|e| format!("reading {}: {e}", path.display()))?
        .len();
    let file_len = usize::try_from(file_len)
        .map_err(|_| format!("{}: file is too large to map", path.display()))?;
    if file_len < HEADER_LEN {
        return Err(format!(
            "{}: not a Verinoo weight file (expected magic VNNW)",
            path.display()
        ));
    }
    // SAFETY: The mapping is read-only and is not exposed beyond this module
    // except as `&[i32]`. Operators are expected to keep model files immutable
    // while a node is running.
    match unsafe { Mmap::map(&file) } {
        Ok(mmap) => {
            let (payload_len, sha256) = verify(&mmap, path)?;
            let aligned =
                (mmap.as_ptr() as usize + HEADER_LEN).is_multiple_of(std::mem::align_of::<i32>());
            let weights = if cfg!(target_endian = "little") && aligned {
                WeightsData::mapped(Arc::new(mmap), HEADER_LEN, payload_len / 4)
            } else {
                owned_from_payload(&mmap[HEADER_LEN..])
            };
            Ok(LoadedWeights { weights, sha256 })
        }
        Err(map_error) => {
            let bytes = std::fs::read(path).map_err(|e| {
                format!(
                    "mapping {}: {map_error}; reading {}: {e}",
                    path.display(),
                    path.display()
                )
            })?;
            let (_, sha256) = verify(&bytes, path)?;
            Ok(LoadedWeights {
                weights: owned_from_payload(&bytes[HEADER_LEN..]),
                sha256,
            })
        }
    }
}

fn verify(bytes: &[u8], path: &Path) -> Result<(usize, String), String> {
    if bytes.len() < HEADER_LEN {
        return Err(format!(
            "{}: not a Verinoo weight file (expected magic VNNW)",
            path.display()
        ));
    }
    if &bytes[0..4] != MAGIC {
        return Err(format!(
            "{}: not a Verinoo weight file (expected magic VNNW)",
            path.display()
        ));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(format!(
            "{}: unsupported weight version {version}",
            path.display()
        ));
    }
    let payload_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let payload_len = usize::try_from(payload_len)
        .map_err(|_| format!("{}: payload is too large", path.display()))?;
    let expected_sha = hex::encode(&bytes[16..48]);
    let payload = &bytes[HEADER_LEN..];
    if payload.len() != payload_len {
        return Err(format!(
            "{}: header says {payload_len} payload bytes, file has {}",
            path.display(),
            payload.len()
        ));
    }
    if payload.is_empty() {
        return Err(format!(
            "{}: weight file has an empty payload",
            path.display()
        ));
    }
    let digest = Sha256::digest(payload);
    if hex::encode(digest) != expected_sha {
        return Err(format!("{}: payload hash mismatch", path.display()));
    }
    if !payload.len().is_multiple_of(4) {
        return Err(format!(
            "{}: payload is not a multiple of 4 bytes",
            path.display()
        ));
    }
    Ok((payload_len, expected_sha))
}

fn owned_from_payload(payload: &[u8]) -> WeightsData {
    let mut out = Vec::with_capacity(payload.len() / 4);
    for chunk in payload.as_chunks::<4>().0 {
        out.push(i32::from_le_bytes(*chunk));
    }
    WeightsData::owned(out)
}

/// Read and verify a weight file into an owned buffer.
pub fn read(path: &Path) -> Result<(Vec<i32>, String), String> {
    let loaded = load(path)?;
    Ok((loaded.weights.as_slice().to_vec(), loaded.sha256))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(label: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("verinoo-weights-{label}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn file_bytes(version: u32, declared_len: u64, payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(HEADER_LEN + payload.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&declared_len.to_le_bytes());
        bytes.extend_from_slice(&Sha256::digest(payload));
        bytes.extend_from_slice(payload);
        bytes
    }

    fn write_file(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn load_error(path: &Path) -> String {
        match load(path) {
            Ok(_) => panic!("expected {} to be rejected", path.display()),
            Err(err) => err,
        }
    }

    #[test]
    fn roundtrip_and_tamper_detection() {
        let dir = scratch("roundtrip");
        let path = dir.join("test.vnnw");
        let weights = vec![0, 1, -1, 65_536, i32::MIN, i32::MAX];
        let sha = write(&path, &weights).unwrap();
        let (got, read_sha) = read(&path).unwrap();
        assert_eq!(got, weights);
        assert_eq!(sha, read_sha);

        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&path, &bytes).unwrap();
        let err = read(&path).unwrap_err();
        assert!(err.contains("hash mismatch"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_replaces_an_existing_file() {
        let dir = scratch("replace");
        let path = dir.join("weights.vnnw");
        write(&path, &[1, 2, 3]).unwrap();
        let sha = write(&path, &[4, 5, 6]).unwrap();
        let (got, read_sha) = read(&path).unwrap();
        assert_eq!(got, vec![4, 5, 6]);
        assert_eq!(sha, read_sha);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_maps_the_payload_without_copying() {
        let dir = scratch("mmap");
        let path = dir.join("mapped.vnnw");
        let weights = vec![7, -7, 42, 65_536, i32::MIN, i32::MAX];
        let sha = write(&path, &weights).unwrap();
        let loaded = load(&path).unwrap();
        assert!(loaded.weights.is_mapped());
        assert_eq!(loaded.weights.as_slice(), weights);
        assert_eq!(loaded.weights.tensor(2, 3), &weights[2..5]);
        assert_eq!(loaded.sha256, sha);
        drop(loaded);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_an_empty_file() {
        let dir = scratch("negative-empty");
        let path = write_file(&dir, "empty.vnnw", &[]);
        let err = load_error(&path);
        assert!(err.contains("not a Verinoo weight file"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_a_header_only_file() {
        let dir = scratch("negative-header");
        let path = write_file(&dir, "header.vnnw", &file_bytes(VERSION, 0, &[]));
        let err = load_error(&path);
        assert!(err.contains("empty payload"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_a_truncated_payload() {
        let dir = scratch("negative-truncated");
        let payload = [1u8, 0, 0, 0, 2, 0, 0, 0];
        let mut bytes = file_bytes(VERSION, payload.len() as u64, &payload);
        bytes.truncate(bytes.len() - 4);
        let path = write_file(&dir, "truncated.vnnw", &bytes);
        let err = load_error(&path);
        assert!(err.contains("header says 8 payload bytes"), "{err}");
        assert!(err.contains("file has 4"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_wrong_magic() {
        let dir = scratch("negative-magic");
        let mut bytes = file_bytes(VERSION, 4, &[1, 0, 0, 0]);
        bytes[..4].copy_from_slice(b"XXXX");
        let path = write_file(&dir, "magic.vnnw", &bytes);
        let err = load_error(&path);
        assert!(err.contains("not a Verinoo weight file"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_unsupported_versions() {
        let dir = scratch("negative-version");
        let path = write_file(
            &dir,
            "version.vnnw",
            &file_bytes(VERSION + 1, 4, &[1, 0, 0, 0]),
        );
        let err = load_error(&path);
        assert!(err.contains("unsupported weight version"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_payload_length_mismatches() {
        let dir = scratch("negative-length");
        let path = write_file(
            &dir,
            "length.vnnw",
            &file_bytes(VERSION, 4, &[1, 0, 0, 0, 2, 0, 0, 0]),
        );
        let err = load_error(&path);
        assert!(err.contains("header says 4 payload bytes"), "{err}");
        assert!(err.contains("file has 8"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
