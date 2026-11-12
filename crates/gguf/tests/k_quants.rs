//! K-quant dequantisation against reference vectors produced by the official
//! `gguf` Python package (`gguf.quants.dequantize`).
//!
//! The fixture contains two super-blocks per type with known f16 scales and
//! pseudo-random payload bytes. A wrong scale unpacking, byte offset or
//! accumulation order shows up immediately.

use serde::Deserialize;
use verinoo_gguf::{block_elements, dequantize, type_size};

#[derive(Deserialize)]
struct Fixture {
    types: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    kind: u32,
    blocks: String,
    expected: Vec<f32>,
}

#[test]
fn k_quants_match_the_reference_dequantisation() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/k_quants.json");
    let raw = std::fs::read_to_string(&path).expect("fixture readable");
    let fixture: Fixture = serde_json::from_str(&raw).expect("fixture is valid JSON");
    assert!(!fixture.types.is_empty());

    for case in &fixture.types {
        let bytes = hex::decode(&case.blocks).expect("fixture hex");
        assert_eq!(
            bytes.len(),
            2 * type_size(case.kind).expect("known type"),
            "{}: two blocks",
            case.name
        );
        assert_eq!(block_elements(case.kind), 256, "{}: QK_K", case.name);
        let got = dequantize(case.kind, &bytes, 512).expect("dequantise");
        assert_eq!(got.len(), case.expected.len(), "{}: length", case.name);
        for (i, (g, e)) in got.iter().zip(&case.expected).enumerate() {
            let tolerance = 1e-6 * (1.0 + e.abs());
            assert!(
                (g - e).abs() <= tolerance,
                "{}: value {i}: got {g}, expected {e}",
                case.name
            );
        }
    }
}

#[test]
fn q8_k_uses_the_f32_delta() {
    let mut block = vec![0u8; 292];
    block[0..4].copy_from_slice(&1.5f32.to_le_bytes());
    for i in 0..256 {
        block[4 + i] = i as u8;
    }
    let got = dequantize(15, &block, 256).expect("dequantise");
    assert_eq!(got.len(), 256);
    for (i, v) in got.iter().enumerate() {
        let q = (i as u8) as i8;
        assert_eq!(*v, 1.5 * q as f32, "value {i}");
    }
}

#[test]
fn k_quant_sizes_are_the_llama_cpp_block_sizes() {
    for (kind, size) in [
        (10, 84),
        (11, 110),
        (12, 144),
        (13, 176),
        (14, 210),
        (15, 292),
    ] {
        assert_eq!(type_size(kind), Some(size), "type {kind}");
        assert_eq!(block_elements(kind), 256, "type {kind}");
    }
}
