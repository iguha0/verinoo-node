//! Golden-vector conformance: Rust op witnesses must produce bit-identical
//! outputs to the WASM runtime (inference.wat) for every test case.

use verinoo_zk::{argmax, f2i, i2f, layernorm, relu, softmax};

#[derive(serde::Deserialize)]
struct OpVector {
    op: String,
    #[serde(rename = "inputFixed")]
    input_fixed: Vec<i32>,
    #[serde(rename = "weightsFixed")]
    weights_fixed: Vec<i32>,
    #[serde(rename = "biasFixed")]
    bias_fixed: Vec<i32>,
    #[serde(rename = "expectedFixed")]
    expected_fixed: Vec<i32>,
}

fn load_vectors() -> Vec<OpVector> {
    let path = format!("{}/tests/op_vectors.json", env!("CARGO_MANIFEST_DIR"));
    let raw = std::fs::read_to_string(&path).expect("op_vectors.json present");
    serde_json::from_str(&raw).expect("parse")
}

#[test]
fn all_op_vectors_match_wasm_runtime() {
    let vectors = load_vectors();
    assert!(vectors.len() >= 8, "expected at least 8 vectors");

    for v in &vectors {
        let got = match v.op.as_str() {
            "relu" => relu(&v.input_fixed),
            "softmax" => softmax(&v.input_fixed, v.input_fixed.len()),
            "layernorm" => layernorm(&v.input_fixed, v.input_fixed.len()),
            "attention" | "matmul" => verinoo_zk::execute_layer(
                "attention",
                &v.weights_fixed,
                &v.input_fixed,
                Some(&v.bias_fixed),
            ),
            other => panic!("unknown op: {other}"),
        };
        assert_eq!(
            got, v.expected_fixed,
            "MISMATCH for op={} input={:?}",
            v.op, v.input_fixed
        );
    }
}

#[test]
fn argmax_basics_and_ties() {
    assert_eq!(argmax(&[1, 3, 2]), 1);
    assert_eq!(argmax(&[5, 5, 1]), 0, "ties -> smallest index");
    assert_eq!(argmax(&[0, -1, -2, 0]), 0);
}

#[test]
fn q16_16_conversion_roundtrip() {
    for v in [-8.0, -1.5, 0.0, 0.001, 1.0, 3.14159, 100.0] {
        let fixed = f2i(v);
        let back = i2f(fixed);
        assert!(
            (back - v).abs() < 1.0 / 65536.0,
            "roundtrip {v} -> {fixed} -> {back}"
        );
    }
}
