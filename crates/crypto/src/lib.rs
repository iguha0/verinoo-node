//! Validator-confidential input encryption.
//!
//! Inference inputs must be readable by every validator (they are re-executed
//! during block application) but must not be readable by API clients, block
//! explorers, or anyone with the task id. Clients encrypt to a network X25519
//! public key; validators hold the matching secret and decrypt at execution.
//!
//! This is **validator-confidential, not operator-blind**: the operator runs
//! the validators and can decrypt. It removes inputs from public blocks and
//! API responses; it does not provide TEE- or ZK-level confidentiality.
//!
//! Scheme (v1):
//!   * ephemeral X25519 keypair per request
//!   * HKDF-SHA256(shared, salt = network_pk || ephemeral_pk,
//!     info = "verinoo-input-v1", 32 bytes)
//!   * ChaCha20-Poly1305, 12-byte nonce, AAD = inputCommitment bytes
//!   * ciphertext is base64; public keys and nonce are hex

use base64::Engine as _;
use chacha20poly1305::aead::{Aead, Payload};
use chacha20poly1305::{ChaCha20Poly1305, KeyInit, Nonce};
use rand::RngCore as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;
use x25519_dalek::{PublicKey, StaticSecret};

pub const ALGORITHM: &str = "x25519-hkdf-sha256-chacha20poly1305";
const INFO: &[u8] = b"verinoo-input-v1";
const NONCE_LEN: usize = 12;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EncryptedInput {
    /// Sender's ephemeral X25519 public key, hex.
    pub ephemeral_public_key: String,
    /// 12-byte ChaCha20-Poly1305 nonce, hex.
    pub nonce: String,
    /// Ciphertext (including the Poly1305 tag), base64.
    pub ciphertext: String,
}

#[derive(Clone)]
pub struct NetworkKey {
    secret: StaticSecret,
    public: PublicKey,
}

impl std::fmt::Debug for NetworkKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NetworkKey")
            .field("public", &hex::encode(self.public.as_bytes()))
            .finish_non_exhaustive()
    }
}

impl NetworkKey {
    pub fn from_secret(secret: [u8; 32]) -> Self {
        let secret = StaticSecret::from(secret);
        let public = PublicKey::from(&secret);
        Self { secret, public }
    }

    pub fn from_hex(secret_hex: &str) -> Result<Self, String> {
        let bytes =
            hex::decode(secret_hex.trim()).map_err(|_| "input key is not hex".to_string())?;
        let secret: [u8; 32] = bytes
            .try_into()
            .map_err(|_| "input key must be exactly 32 bytes (64 hex chars)".to_string())?;
        Ok(Self::from_secret(secret))
    }

    pub fn public_key_hex(&self) -> String {
        hex::encode(self.public.as_bytes())
    }

    pub fn decrypt(
        &self,
        encrypted: &EncryptedInput,
        commitment_hex: &str,
    ) -> Result<String, String> {
        let ephemeral = parse_32(&encrypted.ephemeral_public_key, "ephemeralPublicKey")?;
        let ephemeral = PublicKey::from(ephemeral);
        let nonce = hex::decode(&encrypted.nonce).map_err(|_| "nonce is not hex".to_string())?;
        if nonce.len() != NONCE_LEN {
            return Err(format!("nonce must be {NONCE_LEN} bytes"));
        }
        let ciphertext = base64::engine::general_purpose::STANDARD
            .decode(encrypted.ciphertext.as_bytes())
            .map_err(|_| "ciphertext is not valid base64".to_string())?;
        let aad = commitment_bytes(commitment_hex)?;

        let shared = self.secret.diffie_hellman(&ephemeral);
        let key = derive_key(
            shared.as_bytes(),
            self.public.as_bytes(),
            ephemeral.as_bytes(),
        )?;
        let cipher = ChaCha20Poly1305::new((&key).into());
        let plaintext = cipher
            .decrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: &ciphertext,
                    aad: &aad,
                },
            )
            .map_err(|_| {
                "input decryption failed (wrong key, tampered ciphertext or commitment)"
            })?;
        let text = String::from_utf8(plaintext).map_err(|_| "decrypted input is not UTF-8")?;
        if sha256_hex(&text) != commitment_hex.to_ascii_lowercase() {
            return Err("decrypted input does not match inputCommitment".into());
        }
        Ok(text)
    }
}

pub fn encrypt(
    recipient_public_key_hex: &str,
    plaintext: &str,
    commitment_hex: &str,
) -> Result<EncryptedInput, String> {
    let recipient = parse_32(recipient_public_key_hex, "recipient public key")?;
    let recipient = PublicKey::from(recipient);
    let mut secret_bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut secret_bytes);
    let ephemeral_secret = StaticSecret::from(secret_bytes);
    let ephemeral_public = PublicKey::from(&ephemeral_secret);

    let shared = ephemeral_secret.diffie_hellman(&recipient);
    let key = derive_key(
        shared.as_bytes(),
        recipient.as_bytes(),
        ephemeral_public.as_bytes(),
    )?;
    let aad = commitment_bytes(commitment_hex)?;
    let mut nonce = [0u8; NONCE_LEN];
    rand::rngs::OsRng.fill_bytes(&mut nonce);

    let cipher = ChaCha20Poly1305::new((&key).into());
    let ciphertext = cipher
        .encrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: plaintext.as_bytes(),
                aad: &aad,
            },
        )
        .map_err(|_| "input encryption failed".to_string())?;

    Ok(EncryptedInput {
        ephemeral_public_key: hex::encode(ephemeral_public.as_bytes()),
        nonce: hex::encode(nonce),
        ciphertext: base64::engine::general_purpose::STANDARD.encode(ciphertext),
    })
}

fn derive_key(
    shared: &[u8; 32],
    recipient: &[u8; 32],
    ephemeral: &[u8; 32],
) -> Result<[u8; 32], String> {
    let mut salt = [0u8; 64];
    salt[..32].copy_from_slice(recipient);
    salt[32..].copy_from_slice(ephemeral);
    let hk = hkdf::Hkdf::<Sha256>::new(Some(&salt), shared);
    let mut key = [0u8; 32];
    hk.expand(INFO, &mut key)
        .map_err(|_| "HKDF expand failed".to_string())?;
    Ok(key)
}

fn parse_32(hex_value: &str, field: &str) -> Result<[u8; 32], String> {
    let bytes = hex::decode(hex_value.trim()).map_err(|_| format!("{field} is not hex"))?;
    bytes
        .try_into()
        .map_err(|_| format!("{field} must be 32 bytes"))
}

fn commitment_bytes(commitment_hex: &str) -> Result<[u8; 32], String> {
    parse_32(commitment_hex, "inputCommitment")
}

pub fn sha256_hex(input: &str) -> String {
    let mut h = Sha256::new();
    h.update(input.as_bytes());
    hex::encode(h.finalize())
}

// ---- installed network key (one per process) ----

static INSTALLED: OnceLock<NetworkKey> = OnceLock::new();

/// Install the network key once at startup. Returns the key back if another
/// key was already installed.
pub fn install_input_key(key: NetworkKey) -> Result<(), NetworkKey> {
    INSTALLED.set(key)
}

pub fn input_key() -> Option<&'static NetworkKey> {
    INSTALLED.get()
}

pub fn input_public_key_hex() -> Option<String> {
    INSTALLED.get().map(NetworkKey::public_key_hex)
}

pub fn decrypt_with_installed(
    encrypted: &EncryptedInput,
    commitment_hex: &str,
) -> Result<String, String> {
    let key = INSTALLED
        .get()
        .ok_or_else(|| "this node has no input key configured".to_string())?;
    key.decrypt(encrypted, commitment_hex)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> NetworkKey {
        NetworkKey::from_secret([7u8; 32])
    }

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let network = key();
        let text = "my parcel was never delivered";
        let commitment = sha256_hex(text);
        let encrypted = encrypt(&network.public_key_hex(), text, &commitment).unwrap();
        assert_eq!(encrypted.ephemeral_public_key.len(), 64);
        assert_eq!(encrypted.nonce.len(), 24);
        assert_eq!(network.decrypt(&encrypted, &commitment).unwrap(), text);
    }

    #[test]
    fn ciphertext_is_randomised() {
        let network = key();
        let text = "same input";
        let commitment = sha256_hex(text);
        let a = encrypt(&network.public_key_hex(), text, &commitment).unwrap();
        let b = encrypt(&network.public_key_hex(), text, &commitment).unwrap();
        assert_ne!(a.ciphertext, b.ciphertext);
        assert_ne!(a.ephemeral_public_key, b.ephemeral_public_key);
    }

    #[test]
    fn tampering_is_detected() {
        let network = key();
        let text = "secret";
        let commitment = sha256_hex(text);
        let mut encrypted = encrypt(&network.public_key_hex(), text, &commitment).unwrap();
        let mut bytes = base64::engine::general_purpose::STANDARD
            .decode(encrypted.ciphertext.as_bytes())
            .unwrap();
        bytes[0] ^= 1;
        encrypted.ciphertext = base64::engine::general_purpose::STANDARD.encode(bytes);
        assert!(network.decrypt(&encrypted, &commitment).is_err());
    }

    #[test]
    fn wrong_commitment_or_key_is_rejected() {
        let network = key();
        let text = "secret";
        let commitment = sha256_hex(text);
        let encrypted = encrypt(&network.public_key_hex(), text, &commitment).unwrap();
        let wrong = sha256_hex("other");
        assert!(network.decrypt(&encrypted, &wrong).is_err());
        let other = NetworkKey::from_secret([9u8; 32]);
        assert!(other.decrypt(&encrypted, &commitment).is_err());
    }

    #[test]
    fn commitment_must_be_a_hash() {
        let network = key();
        assert!(encrypt(&network.public_key_hex(), "x", "not-a-hash").is_err());
    }
}
