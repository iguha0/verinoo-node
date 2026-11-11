//! Import a GGUF file's feed-forward layers into a Verinoo runtime graph.
//!
//! GGUF is a container, not a graph: it stores named tensors and their
//! quantisation, so a transformer checkpoint cannot be executed by this runtime
//! (no attention, no tokenizer, no sampling). What it *can* run is a plain
//! dense/relu/softmax classifier, which is what this crate extracts.
//!
//! ```no_run
//! use verinoo_gguf::{convert, Gguf, GgufOptions};
//! # fn main() -> Result<(), String> {
//! let mut file = Gguf::open("classifier.gguf")?;
//! let opts = GgufOptions::new("acme-support-v3", vec!["billing".into(), "refund".into()]);
//! let imported = convert(&mut file, &opts)?;
//! println!("{} layers", imported.spec.layers.len());
//! # Ok(()) }
//! ```
//!
//! Quantised weights (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0) are dequantised with the
//! llama.cpp block layouts and then requantised to Q16.16; F32, F16 and BF16
//! convert directly. K-quants are refused by name — a wrong dequantisation
//! would silently change every answer the chain records.

pub mod convert;
pub mod gguf;

pub use convert::{convert, plan_layers, GgufOptions, Imported, LayerPlan};
pub use gguf::{
    block_elements, dequantize, f16_to_f32, type_name, type_size, Gguf, TensorInfo, Value,
};
