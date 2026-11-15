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
use std::path::Path;
use std::sync::Arc;

const MAGIC: &[u8; 4] = b"VNNW";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 4 + 4 + 8 + 32;

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
    std::fs::write(path, &bytes).map_err(|e| format!("writing {}: {e}", path.display()))?;
    Ok(hex::encode(digest))
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
/// returned on big-endian hosts or if the payload alignment is unsuitable.
pub fn load(path: &Path) -> Result<LoadedWeights, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
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
    let mmap =
        unsafe { Mmap::map(&file) }.map_err(|e| format!("mapping {}: {e}", path.display()))?;
    if mmap.len() < HEADER_LEN {
        return Err(format!(
            "{}: not a Verinoo weight file (expected magic VNNW)",
            path.display()
        ));
    }
    let bytes = &mmap[..];
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
    let digest = Sha256::digest(payload);
    if hex::encode(digest) != expected_sha {
        return Err(format!("{}: payload hash mismatch", path.display()));
    }
    if payload.len() % 4 != 0 {
        return Err(format!(
            "{}: payload is not a multiple of 4 bytes",
            path.display()
        ));
    }
    let aligned =
        (bytes.as_ptr() as usize + HEADER_LEN).is_multiple_of(std::mem::align_of::<i32>());
    let weights = if cfg!(target_endian = "little") && aligned {
        WeightsData::mapped(Arc::new(mmap), HEADER_LEN, payload_len / 4)
    } else {
        let mut out = Vec::with_capacity(payload.len() / 4);
        for chunk in payload.as_chunks::<4>().0 {
            out.push(i32::from_le_bytes(*chunk));
        }
        WeightsData::owned(out)
    };
    Ok(LoadedWeights {
        weights,
        sha256: expected_sha,
    })
}

/// Read and verify a weight file into an owned buffer.
pub fn read(path: &Path) -> Result<(Vec<i32>, String), String> {
    let loaded = load(path)?;
    Ok((loaded.weights.as_slice().to_vec(), loaded.sha256))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_tamper_detection() {
        let dir = std::env::temp_dir().join(format!("verinoo-weights-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
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
    fn load_maps_the_payload_without_copying() {
        let dir = std::env::temp_dir().join(format!("verinoo-weights-mmap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mapped.vnnw");
        let weights = vec![7, -7, 42, 65_536, i32::MIN, i32::MAX];
        let sha = write(&path, &weights).unwrap();
        let loaded = load(&path).unwrap();
        assert!(loaded.weights.is_mapped());
        assert_eq!(loaded.weights.as_slice(), weights);
        assert_eq!(loaded.weights.tensor(2, 3), &weights[2..5]);
        assert_eq!(loaded.sha256, sha);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
