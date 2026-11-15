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
//! The current loader reads the payload into a `Vec<i32>` once at startup.
//! Zero-copy mmap-backed tensors are the next step; the on-disk format does not
//! change.

use sha2::{Digest, Sha256};
use std::path::Path;

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

/// Read and verify a weight file. Returns the parameters and the payload hash.
pub fn read(path: &Path) -> Result<(Vec<i32>, String), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    if bytes.len() < HEADER_LEN || &bytes[0..4] != MAGIC {
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
    let payload_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
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
    let mut out = Vec::with_capacity(payload.len() / 4);
    for chunk in payload.as_chunks::<4>().0 {
        out.push(i32::from_le_bytes(*chunk));
    }
    Ok((out, expected_sha))
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
}
