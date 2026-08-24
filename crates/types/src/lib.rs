//! Verinoo primitive types: transactions, blocks, and the canonical
//! serialization that binds txIds to transaction contents.
//!
//! SPEC SOURCE: the TypeScript reference client (iguha0/Verinoo),
//! `src/core/canonical.ts`. Conformance fixtures in
//! `crates/conformance/fixtures` pin this implementation byte-for-byte.

pub mod canonical;
pub mod tx;

pub use canonical::{canonical_json, CanonicalError, UnsignedPayload};
pub use tx::{TxData, TxEnvelope, TxKind};
