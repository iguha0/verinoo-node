//! Node-local request policy loading and inference submission checks.

use std::path::Path;

use verinoo_policy::{InputKind, Policy, PolicyError};
use verinoo_types::tx::{TxEnvelope, TxKind};

pub const POLICY_REASON: &str = "policy";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateError {
    Policy(PolicyError),
    RedactionRequired,
}

impl GateError {
    pub fn message(&self) -> String {
        match self {
            Self::Policy(error) => error.to_string(),
            Self::RedactionRequired => "input requires client-side redaction".into(),
        }
    }

    pub fn code(&self) -> &'static str {
        match self {
            Self::Policy(PolicyError::TooLong { .. }) => "input_too_long",
            Self::Policy(PolicyError::ModelNotAllowed) => "model_not_allowed",
            Self::Policy(PolicyError::BlockedTerm) => "blocked_term",
            Self::Policy(PolicyError::BlockedPattern) => "blocked_pattern",
            Self::Policy(PolicyError::EncryptedInput) => "encrypted_input",
            Self::Policy(PolicyError::InvalidConfig { .. }) => "invalid_config",
            Self::RedactionRequired => "redaction_required",
        }
    }
}

/// Load the policy path from `VNN_POLICY_FILE`. Unset or empty means the
/// permissive default; a configured path must load and validate.
pub fn load_from_env() -> Result<Policy, String> {
    load_from_var(std::env::var("VNN_POLICY_FILE").ok().as_deref())
}

pub fn load_from_var(value: Option<&str>) -> Result<Policy, String> {
    match value.map(str::trim).filter(|value| !value.is_empty()) {
        Some(path) => load_from_path(Path::new(path)),
        None => Ok(Policy::default()),
    }
}

pub fn load_from_path(path: &Path) -> Result<Policy, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("reading policy {}: {e}", path.display()))?;
    Policy::from_json(&raw).map_err(|e| format!("invalid policy {}: {e}", path.display()))
}

/// Check a transaction at the submission boundary. Only `submitInference`
/// carries request content; encrypted payloads are byte-limited but not
/// inspected, and commitment-only payloads are opaque.
pub fn check_tx(policy: &Policy, tx: &TxEnvelope) -> Result<(), GateError> {
    if tx.data.r#type != TxKind::SubmitInference {
        return Ok(());
    }

    let data = &tx.data.data;
    let target_model = data
        .get("targetModel")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    // Serialize the envelope so the byte limit covers the actual ciphertext and
    // its metadata, not just an empty placeholder.
    let encrypted_payload = data
        .get("encryptedInput")
        .filter(|value| !value.is_null())
        .map(|value| serde_json::to_string(value).unwrap_or_default());
    let plaintext = data
        .get("input")
        .and_then(|value| value.as_str())
        .unwrap_or("");

    let (kind, input) = if let Some(payload) = encrypted_payload.as_deref() {
        (InputKind::Encrypted, payload)
    } else if !plaintext.is_empty() {
        (InputKind::Plaintext, plaintext)
    } else {
        (InputKind::Opaque, "")
    };

    let checked = policy
        .check_input_kind(target_model, input, kind)
        .map_err(GateError::Policy)?;
    if kind == InputKind::Plaintext && (checked.was_redacted() || checked.text != input) {
        return Err(GateError::RedactionRequired);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    use ed25519_dalek::{Signer as _, SigningKey};
    use verinoo_types::canonical::canonical_json;

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_path(name: &str) -> std::path::PathBuf {
        let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "verinoo-policy-gate-{}-{id}-{name}",
            std::process::id()
        ))
    }

    fn signed_inference(data: serde_json::Value) -> TxEnvelope {
        let signing = SigningKey::from_bytes(&[3u8; 32]);
        let address = verinoo_exec::engine::public_key_to_address(&hex::encode(
            signing.verifying_key().to_bytes(),
        ))
        .unwrap();
        let data = json!({ "type": "submitInference", "data": data });
        let payload = json!({
            "data": data,
            "from": address,
            "nonce": 1,
            "to": "",
            "value": 0,
        });
        let tx_id = verinoo_model::sha256_hex(&canonical_json(&payload).unwrap());
        let signature = signing.sign(tx_id.as_bytes());
        serde_json::from_value(json!({
            "txId": tx_id,
            "from": address,
            "to": "",
            "value": 0,
            "nonce": 1,
            "data": data,
            "publicKey": hex::encode(signing.verifying_key().to_bytes()),
            "signature": hex::encode(signature.to_bytes()),
        }))
        .unwrap()
    }

    fn inference_data(model: &str, input: &str) -> serde_json::Value {
        json!({
            "targetModel": model,
            "input": input,
            "inputCommitment": verinoo_model::sha256_hex(input),
            "maxFee": 100,
            "deadline": 1000,
        })
    }

    #[test]
    fn default_policy_is_permissive() {
        let policy = load_from_var(None).unwrap();
        let tx = signed_inference(inference_data("m", "anything at all"));
        assert!(check_tx(&policy, &tx).is_ok());
    }

    #[test]
    fn blocked_terms_are_rejected_without_echoing_them() {
        let path = temp_path("blocked.json");
        std::fs::write(&path, r#"{ "blockedTerms": ["secretproject"] }"#).unwrap();
        let policy = load_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let tx = signed_inference(inference_data("m", "summarize secretPROJECT notes"));
        let error = check_tx(&policy, &tx).unwrap_err();
        assert_eq!(error.code(), "blocked_term");
        let rendered = format!("{} {:?}", error.message(), error);
        assert!(!rendered.to_lowercase().contains("secretproject"));
    }

    #[test]
    fn require_plaintext_rejects_encrypted_input() {
        let path = temp_path("plaintext.json");
        std::fs::write(&path, r#"{ "requirePlaintext": true }"#).unwrap();
        let policy = load_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let tx = signed_inference(json!({
            "targetModel": "m",
            "encryptedInput": {
                "ephemeralPublicKey": "ab".repeat(32),
                "nonce": "00".repeat(12),
                "ciphertext": "AAAA",
            },
            "inputCommitment": "ab".repeat(32),
            "maxFee": 100,
            "deadline": 1000,
        }));
        assert_eq!(
            check_tx(&policy, &tx).unwrap_err().code(),
            "encrypted_input"
        );
    }

    #[test]
    fn encrypted_input_obeys_the_byte_limit() {
        let path = temp_path("encrypted-size.json");
        std::fs::write(&path, r#"{ "maxInputBytes": 256 }"#).unwrap();
        let policy = load_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let oversized = signed_inference(json!({
            "targetModel": "m",
            "encryptedInput": {
                "ephemeralPublicKey": "ab".repeat(32),
                "nonce": "00".repeat(12),
                "ciphertext": "AA".repeat(256),
            },
            "inputCommitment": "ab".repeat(32),
            "maxFee": 100,
            "deadline": 1000,
        }));
        let error = check_tx(&policy, &oversized).unwrap_err();
        assert_eq!(error.code(), "input_too_long");
        assert!(error.message().contains("256"));

        let small = signed_inference(json!({
            "targetModel": "m",
            "encryptedInput": {
                "ephemeralPublicKey": "ab".repeat(32),
                "nonce": "00".repeat(12),
                "ciphertext": "AAAA",
            },
            "inputCommitment": "ab".repeat(32),
            "maxFee": 100,
            "deadline": 1000,
        }));
        assert!(check_tx(&policy, &small).is_ok());
    }

    #[test]
    fn redaction_rejects_instead_of_rewriting_the_signed_input() {
        let path = temp_path("redact.json");
        std::fs::write(&path, r#"{ "redact": { "email": true } }"#).unwrap();
        let policy = load_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let tx = signed_inference(inference_data("m", "mail alice@example.com"));
        let error = check_tx(&policy, &tx).unwrap_err();
        assert_eq!(error, GateError::RedactionRequired);
        assert_eq!(error.message(), "input requires client-side redaction");
    }

    #[test]
    fn invalid_or_missing_policy_files_fail_closed() {
        let path = temp_path("invalid.json");
        std::fs::write(&path, r#"{ "blockedPatterns": ["("] }"#).unwrap();
        let error = load_from_path(&path).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(error.contains("invalid policy"));

        let missing = temp_path("missing.json");
        assert!(load_from_path(&missing).is_err());

        let path = temp_path("valid.json");
        std::fs::write(&path, r#"{ "maxInputBytes": 64 }"#).unwrap();
        let policy = load_from_var(Some(path.to_str().unwrap())).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(policy.config().max_input_bytes, 64);
    }
}
