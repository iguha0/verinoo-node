//! Deterministic plaintext request policy for Verinoo.
//!
//! `verinoo-policy` applies a small, configurable allow/deny/redact pass to
//! plaintext inference inputs before a transaction is built, and an optional
//! view pass to outputs before they are displayed. It performs no network
//! calls, loads no models, and has no randomness or time dependence: the same
//! policy and input always produce the same result.
//!
//! This is **not ML moderation** and it is **not a PII guarantee**. It matches
//! configured terms, regular expressions and a few common identifiers; it
//! cannot understand intent or find every sensitive value.
//!
//! The policy can only inspect plaintext. Encrypted and opaque inputs are
//! marked by the caller and are either rejected (when [`PolicyConfig::require_plaintext`]
//! is set) or passed through untouched. A node-side policy sees only the
//! plaintext it is given; it cannot inspect ciphertext.
//!
//! [`Policy::check_output`] is a **non-consensus view helper**. Its result must
//! never replace the inference output, result hash, receipt, or any other
//! committed state. Applying or skipping output policy on one node cannot be
//! allowed to change consensus.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use thiserror::Error;

/// Default maximum plaintext input size in bytes.
pub const DEFAULT_MAX_INPUT_BYTES: usize = 8192;

/// Stable placeholder used when email redaction is enabled.
pub const EMAIL_PLACEHOLDER: &str = "[EMAIL]";
/// Stable placeholder used when phone redaction is enabled.
pub const PHONE_PLACEHOLDER: &str = "[PHONE]";
/// Stable placeholder used when US SSN redaction is enabled.
pub const SSN_PLACEHOLDER: &str = "[SSN]";

const MAX_CONFIGURED_INPUT_BYTES: usize = 16 * 1024 * 1024;
const MAX_ALLOWED_MODELS: usize = 128;
const MAX_MODEL_BYTES: usize = 256;
const MAX_BLOCKED_TERMS: usize = 128;
const MAX_BLOCKED_TERM_BYTES: usize = 256;
const MAX_BLOCKED_PATTERNS: usize = 64;
const MAX_BLOCKED_PATTERN_BYTES: usize = 1024;

const EMAIL_PATTERN: &str = r"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b";
const PHONE_PATTERN: &str = r"(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\b\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b";
const SSN_PATTERN: &str = r"\b\d{3}-\d{2}-\d{4}\b";

/// Stable policy failure reasons.
///
/// Blocked content is never echoed. [`PolicyError::InvalidConfig`] carries a
/// short reason describing what failed, not the offending value.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PolicyError {
    #[error("input exceeds the configured limit of {limit} bytes")]
    TooLong { limit: usize },
    #[error("model is not allowed by policy")]
    ModelNotAllowed,
    #[error("input contains a blocked term")]
    BlockedTerm,
    #[error("input matches a blocked pattern")]
    BlockedPattern,
    #[error("input is encrypted or opaque and plaintext is required")]
    EncryptedInput,
    #[error("invalid policy config: {reason}")]
    InvalidConfig { reason: String },
}

/// Whether a caller-supplied payload is plaintext or marked as protected.
///
/// Encrypted and opaque payloads are never searched or redacted because doing
/// so would corrupt them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum InputKind {
    #[default]
    Plaintext,
    Encrypted,
    Opaque,
}

/// Which redaction classes are enabled.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct RedactRules {
    pub email: bool,
    pub phone: bool,
    pub ssn: bool,
}

/// JSON policy configuration.
///
/// Every field is optional. An empty object produces the permissive default:
/// an 8192-byte limit, no allowlist, no blocks, no redaction, and encrypted
/// inputs accepted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct PolicyConfig {
    pub max_input_bytes: usize,
    /// Exact model ids. Empty means every model is allowed.
    pub allowed_models: Vec<String>,
    /// Case-insensitive substring matches.
    pub blocked_terms: Vec<String>,
    /// Regular expressions matched against the input as written.
    pub blocked_patterns: Vec<String>,
    pub redact: RedactRules,
    /// Reject payloads marked [`InputKind::Encrypted`] or [`InputKind::Opaque`].
    pub require_plaintext: bool,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            max_input_bytes: DEFAULT_MAX_INPUT_BYTES,
            allowed_models: Vec::new(),
            blocked_terms: Vec::new(),
            blocked_patterns: Vec::new(),
            redact: RedactRules::default(),
            require_plaintext: false,
        }
    }
}

/// Redaction classes applied to sanitized text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedactionKind {
    Email,
    Phone,
    Ssn,
}

impl RedactionKind {
    pub const fn placeholder(self) -> &'static str {
        match self {
            Self::Email => EMAIL_PLACEHOLDER,
            Self::Phone => PHONE_PLACEHOLDER,
            Self::Ssn => SSN_PLACEHOLDER,
        }
    }
}

/// A plaintext input after policy checks and optional redaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SanitizedInput {
    pub text: String,
    pub redactions: Vec<RedactionKind>,
}

impl SanitizedInput {
    pub fn was_redacted(&self) -> bool {
        !self.redactions.is_empty()
    }
}

/// A display view of an output after policy checks and optional redaction.
///
/// This type is not consensus data. Never write its text back into a task,
/// block, receipt, or result commitment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SanitizedOutput {
    pub text: String,
    pub redactions: Vec<RedactionKind>,
}

impl SanitizedOutput {
    pub fn was_redacted(&self) -> bool {
        !self.redactions.is_empty()
    }
}

/// A validated policy with compiled blocked patterns.
#[derive(Debug, Clone)]
pub struct Policy {
    config: PolicyConfig,
    blocked_terms: Vec<String>,
    blocked_patterns: Vec<Regex>,
}

impl Default for Policy {
    fn default() -> Self {
        Self::from_config(PolicyConfig::default())
            .expect("the built-in default policy config is valid")
    }
}

impl Policy {
    /// Parse and validate a JSON policy config.
    ///
    /// Invalid regexes, empty terms or patterns, oversized config lists, and
    /// paths that can match the empty string are rejected at load time.
    pub fn from_json(json: &str) -> Result<Self, PolicyError> {
        let config: PolicyConfig =
            serde_json::from_str(json).map_err(|_| PolicyError::InvalidConfig {
                reason: "config does not match the policy schema".into(),
            })?;
        Self::from_config(config)
    }

    /// Validate an in-memory policy config.
    pub fn from_config(config: PolicyConfig) -> Result<Self, PolicyError> {
        if config.max_input_bytes == 0 || config.max_input_bytes > MAX_CONFIGURED_INPUT_BYTES {
            return Err(invalid_config(format!(
                "maxInputBytes must be between 1 and {MAX_CONFIGURED_INPUT_BYTES}"
            )));
        }
        if config.allowed_models.len() > MAX_ALLOWED_MODELS {
            return Err(invalid_config(format!(
                "allowedModels may contain at most {MAX_ALLOWED_MODELS} entries"
            )));
        }

        let mut allowed_models = Vec::with_capacity(config.allowed_models.len());
        for (index, model) in config.allowed_models.iter().enumerate() {
            let model = model.trim();
            if model.is_empty() || model.len() > MAX_MODEL_BYTES {
                return Err(invalid_config(format!(
                    "allowedModels[{index}] must be non-empty and at most {MAX_MODEL_BYTES} bytes"
                )));
            }
            allowed_models.push(model.to_owned());
        }

        if config.blocked_terms.len() > MAX_BLOCKED_TERMS {
            return Err(invalid_config(format!(
                "blockedTerms may contain at most {MAX_BLOCKED_TERMS} entries"
            )));
        }

        let mut blocked_terms = Vec::with_capacity(config.blocked_terms.len());
        let mut configured_terms = Vec::with_capacity(config.blocked_terms.len());
        for (index, term) in config.blocked_terms.iter().enumerate() {
            let term = term.trim();
            if term.is_empty() || term.len() > MAX_BLOCKED_TERM_BYTES {
                return Err(invalid_config(format!(
                    "blockedTerms[{index}] must be non-empty and at most {MAX_BLOCKED_TERM_BYTES} bytes"
                )));
            }
            configured_terms.push(term.to_owned());
            blocked_terms.push(term.to_lowercase());
        }

        if config.blocked_patterns.len() > MAX_BLOCKED_PATTERNS {
            return Err(invalid_config(format!(
                "blockedPatterns may contain at most {MAX_BLOCKED_PATTERNS} entries"
            )));
        }

        let mut blocked_patterns = Vec::with_capacity(config.blocked_patterns.len());
        let mut configured_patterns = Vec::with_capacity(config.blocked_patterns.len());
        for (index, pattern) in config.blocked_patterns.iter().enumerate() {
            let pattern = pattern.trim();
            if pattern.is_empty() || pattern.len() > MAX_BLOCKED_PATTERN_BYTES {
                return Err(invalid_config(format!(
                    "blockedPatterns[{index}] must be non-empty and at most {MAX_BLOCKED_PATTERN_BYTES} bytes"
                )));
            }
            let compiled = Regex::new(pattern).map_err(|_| {
                invalid_config(format!("blockedPatterns[{index}] is not a valid regex"))
            })?;
            if compiled.is_match("") {
                return Err(invalid_config(format!(
                    "blockedPatterns[{index}] matches the empty string and is too broad"
                )));
            }
            configured_patterns.push(pattern.to_owned());
            blocked_patterns.push(compiled);
        }

        let config = PolicyConfig {
            max_input_bytes: config.max_input_bytes,
            allowed_models,
            blocked_terms: configured_terms,
            blocked_patterns: configured_patterns,
            redact: config.redact,
            require_plaintext: config.require_plaintext,
        };

        Ok(Self {
            config,
            blocked_terms,
            blocked_patterns,
        })
    }

    pub fn config(&self) -> &PolicyConfig {
        &self.config
    }

    /// Check a plaintext input before it is placed into a transaction.
    pub fn check_input(&self, model: &str, input: &str) -> Result<SanitizedInput, PolicyError> {
        self.check_input_kind(model, input, InputKind::Plaintext)
    }

    /// Check a caller-labelled payload.
    ///
    /// The byte limit and model allowlist apply to every kind. Content checks
    /// and redaction run only for [`InputKind::Plaintext`]; encrypted and
    /// opaque payloads are either rejected by `requirePlaintext` or returned
    /// unchanged.
    pub fn check_input_kind(
        &self,
        model: &str,
        input: &str,
        kind: InputKind,
    ) -> Result<SanitizedInput, PolicyError> {
        self.check_size(input)?;
        self.check_model(model)?;

        if kind != InputKind::Plaintext {
            if self.config.require_plaintext {
                return Err(PolicyError::EncryptedInput);
            }
            return Ok(SanitizedInput {
                text: input.to_owned(),
                redactions: Vec::new(),
            });
        }

        self.check_content(input)?;
        let (text, redactions) = self.sanitize(input);
        Ok(SanitizedInput { text, redactions })
    }

    /// Build a non-consensus view of an output.
    ///
    /// This never enforces the input byte limit and must never change committed
    /// inference output, result hashes, or receipts.
    pub fn check_output(&self, model: &str, output: &str) -> Result<SanitizedOutput, PolicyError> {
        self.check_model(model)?;
        self.check_content(output)?;
        let (text, redactions) = self.sanitize(output);
        Ok(SanitizedOutput { text, redactions })
    }

    fn check_size(&self, input: &str) -> Result<(), PolicyError> {
        if input.len() > self.config.max_input_bytes {
            return Err(PolicyError::TooLong {
                limit: self.config.max_input_bytes,
            });
        }
        Ok(())
    }

    fn check_model(&self, model: &str) -> Result<(), PolicyError> {
        if self.config.allowed_models.is_empty()
            || self.config.allowed_models.iter().any(|m| m == model)
        {
            return Ok(());
        }
        Err(PolicyError::ModelNotAllowed)
    }

    fn check_content(&self, text: &str) -> Result<(), PolicyError> {
        if !self.blocked_terms.is_empty() {
            let lower = text.to_lowercase();
            if self.blocked_terms.iter().any(|term| lower.contains(term)) {
                return Err(PolicyError::BlockedTerm);
            }
        }
        if self
            .blocked_patterns
            .iter()
            .any(|pattern| pattern.is_match(text))
        {
            return Err(PolicyError::BlockedPattern);
        }
        Ok(())
    }

    fn sanitize(&self, text: &str) -> (String, Vec<RedactionKind>) {
        let mut out = text.to_owned();
        let mut redactions = Vec::new();

        if self.config.redact.email {
            apply_redaction(
                &mut out,
                email_regex(),
                RedactionKind::Email,
                &mut redactions,
            );
        }
        if self.config.redact.ssn {
            apply_redaction(&mut out, ssn_regex(), RedactionKind::Ssn, &mut redactions);
        }
        if self.config.redact.phone {
            apply_redaction(
                &mut out,
                phone_regex(),
                RedactionKind::Phone,
                &mut redactions,
            );
        }

        (out, redactions)
    }
}

fn invalid_config(reason: impl Into<String>) -> PolicyError {
    PolicyError::InvalidConfig {
        reason: reason.into(),
    }
}

fn apply_redaction(
    text: &mut String,
    pattern: &Regex,
    kind: RedactionKind,
    redactions: &mut Vec<RedactionKind>,
) {
    if pattern.is_match(text) {
        let replacement = pattern.replace_all(text, kind.placeholder()).into_owned();
        *text = replacement;
        redactions.push(kind);
    }
}

fn email_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(EMAIL_PATTERN).expect("email redaction regex is valid"))
}

fn phone_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(PHONE_PATTERN).expect("phone redaction regex is valid"))
}

fn ssn_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(SSN_PATTERN).expect("SSN redaction regex is valid"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(json: &str) -> Policy {
        Policy::from_json(json).unwrap()
    }

    fn all_redactions() -> &'static str {
        r#"{
            "redact": { "email": true, "phone": true, "ssn": true }
        }"#
    }

    #[test]
    fn default_is_permissive() {
        let policy = Policy::default();
        let input = "Contact alice@example.com or 555-123-4567, SSN 123-45-6789.";
        let checked = policy.check_input("any-model", input).unwrap();
        assert_eq!(checked.text, input);
        assert!(checked.redactions.is_empty());
        assert_eq!(policy.config().max_input_bytes, DEFAULT_MAX_INPUT_BYTES);
        assert!(!policy.config().require_plaintext);
    }

    #[test]
    fn empty_config_object_uses_defaults() {
        let policy = policy("{}");
        assert_eq!(policy.config(), &PolicyConfig::default());
    }

    #[test]
    fn rejects_input_over_byte_limit() {
        let policy = policy(r#"{ "maxInputBytes": 4 }"#);
        assert_eq!(
            policy.check_input("m", "hello").unwrap_err(),
            PolicyError::TooLong { limit: 4 }
        );
        assert_eq!(policy.check_input("m", "éé").unwrap().text, "éé");
        assert_eq!(
            policy.check_input("m", "ééé").unwrap_err(),
            PolicyError::TooLong { limit: 4 }
        );
    }

    #[test]
    fn enforces_model_allowlist() {
        let policy = policy(r#"{ "allowedModels": ["gpt-2", "tiny"] }"#);
        assert!(policy.check_input("gpt-2", "hello").is_ok());
        assert!(policy.check_input("tiny", "hello").is_ok());
        assert_eq!(
            policy.check_input("gpt-3", "hello").unwrap_err(),
            PolicyError::ModelNotAllowed
        );
        assert_eq!(
            policy.check_output("gpt-3", "hello").unwrap_err(),
            PolicyError::ModelNotAllowed
        );
    }

    #[test]
    fn blocks_terms_case_insensitively_without_echoing_them() {
        let policy = policy(r#"{ "blockedTerms": ["SecretProject"] }"#);
        let error = policy
            .check_input("m", "summarize secretPROJECT notes")
            .unwrap_err();
        assert_eq!(error, PolicyError::BlockedTerm);
        let rendered = format!("{error} {error:?}");
        let lower = rendered.to_lowercase();
        assert!(!lower.contains("secretproject"));
        assert!(!lower.contains("secretproject notes"));
    }

    #[test]
    fn blocks_regex_patterns() {
        let policy = policy(r#"{ "blockedPatterns": ["(?i)forbidden\\s+word"] }"#);
        assert_eq!(
            policy
                .check_input("m", "a FORBIDDEN   word here")
                .unwrap_err(),
            PolicyError::BlockedPattern
        );
        assert!(policy.check_input("m", "forbidden things").is_ok());
    }

    #[test]
    fn rejects_invalid_or_too_broad_config_patterns() {
        assert!(matches!(
            Policy::from_json(r#"{ "blockedPatterns": ["("] }"#).unwrap_err(),
            PolicyError::InvalidConfig { .. }
        ));
        assert!(matches!(
            Policy::from_json(r#"{ "blockedPatterns": [""] }"#).unwrap_err(),
            PolicyError::InvalidConfig { .. }
        ));
        assert!(matches!(
            Policy::from_json(r#"{ "blockedPatterns": [".*"] }"#).unwrap_err(),
            PolicyError::InvalidConfig { .. }
        ));
        assert!(matches!(
            Policy::from_json(r#"{ "blockedTerms": ["   "] }"#).unwrap_err(),
            PolicyError::InvalidConfig { .. }
        ));
        assert!(matches!(
            Policy::from_json(r#"{ "maxInputBytes": 0 }"#).unwrap_err(),
            PolicyError::InvalidConfig { .. }
        ));
    }

    #[test]
    fn redacts_identifiers_and_is_idempotent() {
        let policy = policy(all_redactions());
        let input = "Mail alice@example.com, call 555-123-4567, SSN 123-45-6789.";
        let checked = policy.check_input("m", input).unwrap();
        assert_eq!(checked.text, "Mail [EMAIL], call [PHONE], SSN [SSN].");
        assert_eq!(
            checked.redactions,
            vec![
                RedactionKind::Email,
                RedactionKind::Ssn,
                RedactionKind::Phone
            ]
        );

        let again = policy.check_input("m", &checked.text).unwrap();
        assert_eq!(again.text, checked.text);
        assert!(!again.was_redacted());
    }

    #[test]
    fn redacts_parenthesised_and_country_code_phones() {
        let policy = policy(all_redactions());
        for input in ["(555) 123-4567", "+1 555-123-4567", "555.123.4567"] {
            let checked = policy.check_input("m", input).unwrap();
            assert_eq!(checked.text, PHONE_PLACEHOLDER, "{input}");
        }
    }

    #[test]
    fn requires_plaintext_when_configured() {
        let policy = policy(r#"{ "requirePlaintext": true }"#);
        assert_eq!(
            policy
                .check_input_kind("m", "ciphertext", InputKind::Encrypted)
                .unwrap_err(),
            PolicyError::EncryptedInput
        );
        assert_eq!(
            policy
                .check_input_kind("m", "opaque", InputKind::Opaque)
                .unwrap_err(),
            PolicyError::EncryptedInput
        );
        assert!(policy
            .check_input_kind("m", "plain", InputKind::Plaintext)
            .is_ok());

        let permissive = Policy::default();
        assert_eq!(
            permissive
                .check_input_kind("m", "ciphertext", InputKind::Encrypted)
                .unwrap()
                .text,
            "ciphertext"
        );
    }

    #[test]
    fn output_helper_blocks_and_redacts_without_consensus_effects() {
        let policy = policy(
            r#"{
                "blockedTerms": ["internal-only"],
                "redact": { "email": true, "phone": true, "ssn": true }
            }"#,
        );
        let view = policy
            .check_output("m", "reply to alice@example.com or 555-123-4567")
            .unwrap();
        assert_eq!(view.text, "reply to [EMAIL] or [PHONE]");
        assert_eq!(
            view.redactions,
            vec![RedactionKind::Email, RedactionKind::Phone]
        );
        assert_eq!(
            policy.check_output("m", "INTERNAL-ONLY notes").unwrap_err(),
            PolicyError::BlockedTerm
        );
    }

    #[test]
    fn repeated_checks_are_deterministic() {
        let policy = policy(
            r#"{
                "maxInputBytes": 64,
                "allowedModels": ["m"],
                "blockedTerms": ["blocked"],
                "blockedPatterns": ["bad\\s+word"],
                "redact": { "email": true, "phone": true, "ssn": true }
            }"#,
        );
        let input = "alice@example.com 555-123-4567 123-45-6789";
        let first = policy.check_input("m", input).unwrap();
        for _ in 0..64 {
            assert_eq!(policy.check_input("m", input).unwrap(), first);
            assert_eq!(
                policy.check_input("m", "a blocked value").unwrap_err(),
                PolicyError::BlockedTerm
            );
        }

        let first_output = policy.check_output("m", input).unwrap();
        for _ in 0..64 {
            assert_eq!(policy.check_output("m", input).unwrap(), first_output);
        }
    }
}
