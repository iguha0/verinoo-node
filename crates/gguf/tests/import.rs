mod support;

use support::*;
use verinoo_gguf::{convert, plan_layers, Gguf, GgufOptions};
use verinoo_onnx::check::{compare, sample_features};

fn write_tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    // A unique directory per call: tests run in parallel and several fixtures
    // share a filename, so a fixed path would truncate another test's file.
    let dir = std::env::temp_dir().join(format!(
        "verinoo-gguf-tests-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

fn opts() -> GgufOptions {
    GgufOptions::new("gguf-unit", vec!["negative".into(), "positive".into()])
}

#[test]
fn f32_gguf_imports_a_dense_chain_with_parity() {
    let path = write_tmp("f32-mlp.gguf", &f32_mlp(true));
    let mut file = Gguf::open(&path).expect("fixture opens");
    assert_eq!(file.architecture(), Some("acme"));

    let plan = plan_layers(&file, &opts()).expect("plan is inferred");
    let names: Vec<String> = plan.iter().map(|l| l.name.clone()).collect();
    assert_eq!(names, ["fc1", "fc2"]);

    let imported = convert(&mut file, &opts()).expect("conversion succeeds");
    let ops: Vec<&str> = imported.spec.layers.iter().map(|l| l.op.as_str()).collect();
    assert_eq!(ops, ["dense", "relu", "dense", "softmax"]);
    assert_eq!(imported.spec.features, FEATURES);
    assert_eq!(imported.spec.labels.len(), CLASSES);

    let first = &imported.spec.layers[0];
    assert_eq!((first.input, first.output), (FEATURES, HIDDEN));
    assert_eq!(first.weights[0], (w1()[0] * 65_536.0).round() as i32);
    assert_eq!(
        first.bias,
        b1().iter()
            .map(|v| (v * 65_536.0).round() as i32)
            .collect::<Vec<_>>()
    );

    let samples = sample_features(imported.spec.features, 32);
    let parity = compare(&imported.spec, &imported.float, &samples).expect("parity runs");
    assert!(
        parity.is_exact_enough(),
        "class changed: {:?}",
        parity.mismatches
    );
    assert_eq!(parity.argmax_agree, parity.samples);
}

#[test]
fn q8_0_weights_use_the_reference_block_layout() {
    let path = write_tmp("q8.gguf", &q8_mlp());
    let mut file = Gguf::open(&path).expect("fixture opens");
    let imported = convert(&mut file, &opts()).expect("conversion succeeds");

    let layer = &imported.spec.layers[0];
    assert_eq!((layer.input, layer.output), (Q_FEATURES, Q_CLASSES));
    for row in 0..Q_CLASSES {
        for i in 0..Q_FEATURES {
            let expected = (q8_value(row, i) * 65_536.0).round() as i32;
            assert_eq!(
                layer.weights[row * Q_FEATURES + i],
                expected,
                "row {row} column {i}"
            );
        }
    }
    // d = 1/8 and q = -8 gives exactly -1.0, which also proves the f16 decode.
    assert_eq!(layer.weights[0], -65_536);
    assert_eq!(layer.weights[8], 0);
    assert!(
        imported.notes.iter().any(|n| n.contains("Q8_0")),
        "{:?}",
        imported.notes
    );
}

#[test]
fn a_plan_that_is_not_a_chain_is_refused() {
    let path = write_tmp("f32-mlp.gguf", &f32_mlp(true));
    let mut file = Gguf::open(&path).unwrap();
    let mut options = opts();
    options.layers = Some(vec!["fc2".into(), "fc1".into()]);
    let err = convert(&mut file, &options).expect_err("reversed plan must fail");
    assert!(err.contains("not a chain"), "{err}");
}

#[test]
fn missing_bias_tensors_become_zeros_and_are_reported() {
    let path = write_tmp("f32-nobias.gguf", &f32_mlp(false));
    let mut file = Gguf::open(&path).unwrap();
    let imported = convert(&mut file, &opts()).expect("conversion succeeds");
    assert!(imported.spec.layers[0].bias.iter().all(|b| *b == 0));
    assert!(
        imported.notes.iter().any(|n| n.contains("no bias tensor")),
        "{:?}",
        imported.notes
    );
}

#[test]
fn k_quants_are_refused_by_name() {
    let path = write_tmp("k-quant.gguf", &k_quant());
    let mut file = Gguf::open(&path).unwrap();
    let mut options = opts();
    options.layers = Some(vec!["quant".into()]);
    let err = convert(&mut file, &options).expect_err("Q4_K must be refused");
    assert!(err.contains("Q4_K"), "{err}");
    assert!(err.contains("quant.weight"), "{err}");
}

#[test]
fn label_count_must_match_the_last_layer() {
    let path = write_tmp("f32-mlp.gguf", &f32_mlp(true));
    let mut file = Gguf::open(&path).unwrap();
    let mut options = opts();
    options.labels = vec!["a".into(), "b".into(), "c".into()];
    let err = convert(&mut file, &options).expect_err("3 labels for 2 outputs must fail");
    assert!(err.contains("2 outputs"), "{err}");
}

#[test]
fn a_non_gguf_file_names_the_format_mismatch() {
    let path = write_tmp("not-gguf.bin", b"ONNX\x00\x00\x00\x00");
    let err = Gguf::open(&path).expect_err("must not parse");
    assert!(err.contains("not a GGUF file"), "{err}");
    assert!(err.contains("onnx-import"), "{err}");
}

#[test]
fn a_truncated_file_is_rejected_before_any_read() {
    let full = f32_mlp(true);
    let cut = &full[..full.len() - 64];
    let path = write_tmp("truncated.gguf", cut);
    let err = Gguf::open(&path).expect_err("truncated file must fail");
    assert!(err.contains("truncated"), "{err}");
}
