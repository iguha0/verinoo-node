//! Canonical JSON serialization — the byte-level contract between the
//! TypeScript reference client and this implementation.
//!
//! Rules (mirroring src/core/canonical.ts in the reference repo):
//!   - object keys sorted lexicographically, recursively
//!   - no whitespace
//!   - arrays keep order
//!   - numbers: INTEGERS ONLY within i64/u64 bounds. Non-integer or
//!     out-of-range numbers are a hard error (Verinoo Canonical Data
//!     Format forbids floats — "zero floating point" is doctrine).

use serde_json::Value;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum CanonicalError {
    FloatNotAllowed(f64),
    NumberOutOfRange(String),
}

impl fmt::Display for CanonicalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CanonicalError::FloatNotAllowed(v) => {
                write!(f, "float not allowed in canonical data: {v}")
            }
            CanonicalError::NumberOutOfRange(s) => {
                write!(f, "number out of range for canonical encoding: {s}")
            }
        }
    }
}
impl std::error::Error for CanonicalError {}

/// Serialize a JSON value into Verinoo canonical form.
pub fn canonical_json(value: &Value) -> Result<String, CanonicalError> {
    let mut out = String::new();
    write_canonical(value, &mut out)?;
    Ok(out)
}

fn write_canonical(value: &Value, out: &mut String) -> Result<(), CanonicalError> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                // f64 that isn't an exact integer — forbidden by spec
                return Err(CanonicalError::NumberOutOfRange(n.to_string()));
            }
        }
        Value::String(s) => {
            out.push_str(&serde_json::to_string(s).expect("string serialize infallible"));
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            // serde_json maps are BTreeMap-backed by default? No — preserve
            // deterministic order by sorting keys explicitly.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).expect("string serialize infallible"));
                out.push(':');
                write_canonical(&map[*k], out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// Unsigned transaction payload — the exact field set hashed into txId.
/// Field presence/order is irrelevant; canonical encoding defines identity.
#[derive(Debug, Clone, PartialEq)]
pub struct UnsignedPayload {
    pub data: Value,
    pub from: String,
    pub nonce: u64,
    pub to: String,
    pub value: u64,
    pub gas_limit: Option<u64>,
    pub gas_price: Option<u64>,
}

impl UnsignedPayload {
    pub fn to_canonical_string(&self) -> Result<String, CanonicalError> {
        // Build via serde map then canonicalize (keys get sorted anyway).
        use serde_json::Map;
        let mut m = Map::new();
        m.insert("data".into(), self.data.clone());
        m.insert("from".into(), Value::String(self.from.clone()));
        m.insert("nonce".into(), Value::from(self.nonce));
        m.insert("to".into(), Value::String(self.to.clone()));
        m.insert("value".into(), Value::from(self.value));
        if let Some(g) = self.gas_limit {
            m.insert("gasLimit".into(), Value::from(g));
        }
        if let Some(g) = self.gas_price {
            m.insert("gasPrice".into(), Value::from(g));
        }
        canonical_json(&Value::Object(m))
    }

    /// txId = SHA-256(canonical string), hex-encoded lowercase.
    pub fn tx_id(&self) -> Result<String, CanonicalError> {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(self.to_canonical_string()?.as_bytes());
        Ok(hex::encode(hasher.finalize()))
    }
}
