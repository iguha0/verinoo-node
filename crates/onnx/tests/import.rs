mod support;

use support::*;
use verinoo_onnx::check::{compare, sample_features};
use verinoo_onnx::{convert, parse_model, ImportOptions, Imported};

fn labels() -> Vec<String> {
    LABELS.iter().map(|s| s.to_string()).collect()
}

fn import(bytes: &[u8], features: Option<usize>) -> Result<Imported, String> {
    let model = parse_model(bytes)?;
    let mut opts = ImportOptions::new("unit-mlp", labels());
    opts.features = features;
    convert(&model, &opts)
}

#[test]
fn fixture_parses_into_the_expected_graph() {
    let model = parse_model(&mlp_gemm()).expect("fixture parses");
    let mut names: Vec<String> = model.graph.initializers.keys().cloned().collect();
    names.sort();
    let nodes: Vec<(String, Vec<String>)> = model
        .graph
        .nodes
        .iter()
        .map(|n| (n.op_type.clone(), n.inputs.clone()))
        .collect();
    println!("initializers: {names:?}");
    println!("nodes: {nodes:?}");
    let w1 = &model.graph.initializers["w1"];
    println!(
        "w1 dims {:?} floats {} dtype {}",
        w1.dims,
        w1.floats.len(),
        w1.data_type
    );
    assert_eq!(names, ["b1", "b2", "w1", "w2"]);
}

#[test]
fn gemm_export_becomes_a_dense_relu_dense_softmax_chain() {
    let imported = import(&mlp_gemm(), None).expect("Gemm export should import");
    let spec = &imported.spec;

    assert_eq!(spec.features, FEATURES);
    assert_eq!(spec.labels.len(), CLASSES);
    let ops: Vec<&str> = spec.layers.iter().map(|l| l.op.as_str()).collect();
    assert_eq!(ops, ["dense", "relu", "dense", "softmax"]);

    let first = &spec.layers[0];
    assert_eq!((first.input, first.output), (FEATURES, HIDDEN));
    assert_eq!(first.weights.len(), FEATURES * HIDDEN);
    assert_eq!(first.bias.len(), HIDDEN);
    // Same row-major order as the export, quantised to Q16.16.
    assert_eq!(first.weights[0], (0.5f32 * 65_536.0).round() as i32);
    assert_eq!(first.weights[1], (-0.25f32 * 65_536.0).round() as i32);
    assert_eq!(first.bias[0], (0.05f32 * 65_536.0).round() as i32);

    let second = &spec.layers[2];
    assert_eq!((second.input, second.output), (HIDDEN, CLASSES));
    assert_eq!(second.bias, vec![655, -655]);
}

#[test]
fn matmul_with_separate_bias_produces_the_same_layers_as_gemm() {
    let gemm = import(&mlp_gemm(), None).expect("gemm import");
    let matmul = import(&mlp_matmul(), None).expect("matmul import");
    let a = serde_json::to_value(&gemm.spec.layers).unwrap();
    let b = serde_json::to_value(&matmul.spec.layers).unwrap();
    assert_eq!(a, b, "both exports describe the same network");
}

#[test]
fn quantised_execution_keeps_the_float_prediction_on_every_sample() {
    let imported = import(&mlp_gemm(), None).expect("gemm import");
    let samples = sample_features(imported.spec.features, 40);
    let parity = compare(&imported.spec, &imported.float, &samples).expect("parity runs");
    assert_eq!(parity.samples, 40);
    assert!(
        parity.is_exact_enough(),
        "integer execution changed the class: {:?}",
        parity.mismatches
    );
    assert_eq!(parity.argmax_agree, parity.samples);
    // Q16.16 rounding of small weights is far below a full label's margin.
    assert!(
        parity.max_logit_diff < 1e-3,
        "unexpected logit drift: {}",
        parity.max_logit_diff
    );
}

#[test]
fn unsupported_ops_are_refused_with_the_node_named() {
    let err = import(&unsupported_op(), None).expect_err("Conv must be refused");
    assert!(err.contains("Conv"), "{err}");
    assert!(err.contains("conv1"), "{err}");
}

#[test]
fn label_count_must_match_the_graph_output() {
    let model = parse_model(&mlp_gemm()).unwrap();
    let opts = ImportOptions::new(
        "unit-mlp",
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
    );
    let err = convert(&model, &opts).expect_err("3 labels for 2 outputs must fail");
    assert!(err.contains("2 outputs"), "{err}");
}

#[test]
fn dynamic_input_width_requires_explicit_features() {
    let err = import(&mlp_dynamic_input(), None).expect_err("dynamic input needs --features");
    assert!(err.contains("--features"), "{err}");

    let imported =
        import(&mlp_dynamic_input(), Some(FEATURES)).expect("with --features it imports");
    assert_eq!(imported.spec.features, FEATURES);
    assert_eq!(imported.spec.layers[0].input, FEATURES);
}

#[test]
fn declaring_the_wrong_feature_width_is_refused() {
    let err = import(&mlp_gemm(), Some(8)).expect_err("mismatched width must fail");
    assert!(err.contains("says 8"), "{err}");
}
