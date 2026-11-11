//! Import an ONNX classifier into a graph the Verinoo runtime can execute.
//!
//! The runtime executes `dense`/`relu`/`softmax` over Q16.16 integers (see
//! `verinoo-runtime`). This crate reads a real `.onnx` file, folds its linear
//! chain into that form, quantises the parameters, and measures how far the
//! integer execution lands from an f32 evaluation of the same graph.
//!
//! ```no_run
//! use verinoo_onnx::{convert, ImportOptions};
//! # fn main() -> Result<(), String> {
//! let bytes = std::fs::read("model.onnx").map_err(|e| e.to_string())?;
//! let model = verinoo_onnx::onnx::parse_model(&bytes)?;
//! let opts = ImportOptions::new("acme-support-v2", vec!["billing".into(), "refund".into()]);
//! let imported = convert(&model, &opts)?;
//! println!("{} layers", imported.spec.layers.len());
//! # Ok(()) }
//! ```
//!
//! What is supported, and what is refused, is documented in `docs/runtime.md`.
//! Refusing is deliberate: a guessed op would yield a model whose answers differ
//! from the customer's reference implementation, and the difference would only
//! surface after the parameters were committed on-chain.

pub mod check;
pub mod convert;
pub mod onnx;
pub mod quant;
pub mod wire;

pub use check::{compare, sample_features, Parity};
pub use convert::{convert, op_chain, FloatLayer, ImportOptions, Imported};
pub use onnx::{parse_model, Model};
pub use quant::to_q16;
