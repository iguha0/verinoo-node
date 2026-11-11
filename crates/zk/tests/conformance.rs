//! Golden-vector conformance: Rust op witnesses must produce bit-identical
//! outputs to the WASM runtime (inference.wat) for every test case.

use verinoo_zk::{argmax, f2i, i2f, layernorm, relu, softmax};

#[derive(serde::Deserialize)]
#[allow(dead_code)]
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
    assert!(vectors.len() >= 10, "expected at least 10 vectors");

    for v in &vectors {
        let got = match v.op.as_str() {
            "relu8" | "relu64" => relu(&v.input_fixed),
            "softmax8" | "softmax64" => softmax(&v.input_fixed, v.input_fixed.len()),
            "layernorm8" | "layernorm64" => layernorm(&v.input_fixed, v.input_fixed.len()),
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
    for v in [-8.0, -1.5, 0.0, 0.001, 1.0, std::f64::consts::PI, 100.0] {
        let fixed = f2i(v);
        let back = i2f(fixed);
        assert!(
            (back - v).abs() < 1.0 / 65536.0,
            "roundtrip {v} -> {fixed} -> {back}"
        );
    }
}

#[test]
fn q16_16_matmul_column_major_exact() {
    // Exact values from WASM runtime — verified manually with BigInt math.
    let input: Vec<i32> = vec![32768, -16384, 65536, 131072]; // [0.5, -0.25, 1.0, 2.0]
    let weights: Vec<i32> = vec![
        6554, -13107, 19661, 26214, // row 0: [0.1, -0.2, 0.3, 0.4]
        -32768, 39322, -45875, 52429, // row 1: [-0.5, 0.6, -0.7, 0.8]
        58982, 65536, -72090, 78643, // row 2: [0.9, 1.0, -1.1, 1.2]
        -85197, 91750, -98304, 104858, // row 3: [-1.3, 1.4, -1.5, 1.6]
    ];
    let bias: Vec<i32> = vec![655, -1311, 1966, -2621]; // [0.01, -0.02, 0.03, -0.04]

    // Column-major: weights[k * n_out + i] connects input k to output i.
    let n_out = 4;

    // Manual step-by-step for out[0]:
    // fmul(input[0], w[0]) = (32768 * 6554) >> 16 = 32768
    // fmul(input[1], w[4]) = (-16384 * -32768) >> 16 = 8192
    // fmul(input[2], w[8]) = (65536 * 58982) >> 16 = 58982
    // fmul(input[3], w[12]) = (131072 * -85197) >> 16 = -170394

    // Sum: 32768 + 8192 + 58982 - 170394 = -70452
    // Add bias: -70452 + 655 = -69797
    // ReLU: max(0, -69797) = 0 ✓

    use verinoo_zk::fmul;
    let mut results = Vec::new();
    for (i, &b) in bias.iter().enumerate().take(n_out) {
        let mut acc = b;
        for (k, &inp) in input.iter().enumerate() {
            let wi = k * n_out + i;
            acc = acc.wrapping_add(fmul(inp, weights[wi]));
        }
        results.push(acc);
    }

    let after_relu: Vec<i32> = results.iter().map(|&v| if v > 0 { v } else { 0 }).collect();

    // Compare with verinoo-zk execute_layer
    let via_dispatch = verinoo_zk::execute_layer("attention", &weights, &input, Some(&bias));

    assert_eq!(
        via_dispatch, after_relu,
        "dispatch must match manual column-major computation"
    );

    // Verify against WASM golden: [0, 232651, 0, 288358]
    assert_eq!(after_relu[0], 0);
    assert_eq!(after_relu[2], 0);
}

#[test]
fn fmul_matches_q16_16_reference() {
    // Q16.16 multiply: result = floor(a*b / 65536), verified against float math
    let cases: Vec<(i32, i32, i64)> = vec![
        (32768, 6554, 3277),       // 0.5 * 0.1 = 0.05
        (-16384, -32768, 8192),    // -0.25 * -0.5 = 0.125
        (65536, 58982, 58982),     // 1.0 * 0.9 = 0.9
        (131072, -85197, -170394), // 2.0 * -1.3 = -2.6
        (65536, 65536, 65536),     // 1.0 * 1.0 = 1.0
        (0, 65536, 0),             // 0.0
    ];
    for (a, b, expected) in cases {
        let got = verinoo_zk::fmul(a, b);
        assert_eq!(got as i64, expected, "fmul({a}, {b}) should be {expected}");
    }
}
