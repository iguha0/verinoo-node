//! Hand-encoded ONNX messages for the importer tests.
//!
//! Test fixtures are built with the same protobuf wire format the importer
//! reads, so a test failure points at decoding or conversion rather than at a
//! binary fixture that cannot be reviewed in a diff.

#![allow(dead_code)]

pub fn varint(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            return out;
        }
    }
}

fn key(number: u32, wire: u8) -> Vec<u8> {
    varint(((number as u64) << 3) | wire as u64)
}

pub fn f_varint(number: u32, value: u64) -> Vec<u8> {
    let mut out = key(number, 0);
    out.extend(varint(value));
    out
}

pub fn f_bytes(number: u32, bytes: &[u8]) -> Vec<u8> {
    let mut out = key(number, 2);
    out.extend(varint(bytes.len() as u64));
    out.extend_from_slice(bytes);
    out
}

pub fn f_str(number: u32, text: &str) -> Vec<u8> {
    f_bytes(number, text.as_bytes())
}

pub fn f_f32(number: u32, value: f32) -> Vec<u8> {
    let mut out = key(number, 5);
    out.extend(value.to_le_bytes());
    out
}

pub fn cat(parts: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for p in parts {
        out.extend_from_slice(p);
    }
    out
}

/// TensorProto with packed `float_data`.
pub fn tensor(name: &str, dims: &[usize], data: &[f32]) -> Vec<u8> {
    let mut packed_dims = Vec::new();
    for d in dims {
        packed_dims.extend(varint(*d as u64));
    }
    let mut packed_data = Vec::new();
    for v in data {
        packed_data.extend(v.to_le_bytes());
    }
    cat(&[
        f_bytes(1, &packed_dims),
        f_varint(2, 1), // DT_FLOAT
        f_bytes(4, &packed_data),
        f_str(8, name),
    ])
}

pub fn node(op: &str, name: &str, inputs: &[&str], outputs: &[&str], attrs: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for i in inputs {
        out.extend(f_str(1, i));
    }
    for o in outputs {
        out.extend(f_str(2, o));
    }
    out.extend(f_str(3, name));
    out.extend(f_str(4, op));
    for a in attrs {
        out.extend(f_bytes(5, a));
    }
    out
}

pub fn attr_int(name: &str, value: i64) -> Vec<u8> {
    cat(&[f_str(1, name), f_varint(3, value as u64)])
}

pub fn value_info(name: &str, dims: &[Option<usize>]) -> Vec<u8> {
    let mut shape = Vec::new();
    for d in dims {
        let mut dim = Vec::new();
        if let Some(v) = d {
            dim.extend(f_varint(1, *v as u64));
        }
        shape.extend(f_bytes(1, &dim));
    }
    let mut tensor_type = f_varint(1, 1); // elem_type = FLOAT
    tensor_type.extend(f_bytes(2, &shape));
    cat(&[f_str(1, name), f_bytes(2, &f_bytes(1, &tensor_type))])
}

fn graph(nodes: &[Vec<u8>], inits: &[Vec<u8>], ins: &[Vec<u8>], outs: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for n in nodes {
        out.extend(f_bytes(1, n));
    }
    for i in inits {
        out.extend(f_bytes(5, i));
    }
    for i in ins {
        out.extend(f_bytes(11, i));
    }
    for o in outs {
        out.extend(f_bytes(12, o));
    }
    out
}

fn model(graph: Vec<u8>) -> Vec<u8> {
    // OperatorSetIdProto { domain = "", version = 13 }
    let opset = f_varint(2, 13);
    cat(&[
        f_varint(1, 7),            // ir_version
        f_str(2, "verinoo-test"),  // producer_name
        f_bytes(6, PACKED_MARKER), // doc_string: ignored by the importer
        f_bytes(7, &graph),
        f_bytes(8, &opset),
    ])
}

/// Placeholder bytes for a field the parser ignores, so the fixture also
/// exercises unknown-field skipping.
const PACKED_MARKER: &[u8] = b"unused-doc-string";

pub const FEATURES: usize = 3;
pub const HIDDEN: usize = 4;
pub const CLASSES: usize = 2;
pub const LABELS: [&str; CLASSES] = ["negative", "positive"];

/// Hidden layer weights, row-major `HIDDEN × FEATURES`.
pub const W1: [f32; HIDDEN * FEATURES] = [
    0.5, -0.25, 0.125, -0.5, 0.25, 0.75, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3,
];
pub const B1: [f32; HIDDEN] = [0.05, -0.05, 0.02, 0.0];
/// Output weights, row-major `CLASSES × HIDDEN`.
pub const W2: [f32; CLASSES * HIDDEN] = [0.3, -0.2, 0.5, 0.1, -0.3, 0.2, -0.5, -0.1];
pub const B2: [f32; CLASSES] = [0.01, -0.01];

fn transpose(rows: usize, cols: usize, data: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// A two-layer MLP exported the way frameworks do it: `Gemm` with `transB = 1`
/// and the bias in the third input.
pub fn mlp_gemm() -> Vec<u8> {
    let nodes = [
        node(
            "Gemm",
            "fc1",
            &["X", "w1", "b1"],
            &["h1"],
            &[attr_int("transB", 1)],
        ),
        node("Relu", "act", &["h1"], &["h2"], &[]),
        node(
            "Gemm",
            "fc2",
            &["h2", "w2", "b2"],
            &["logits"],
            &[attr_int("transB", 1)],
        ),
        node("Softmax", "prob", &["logits"], &["probs"], &[]),
    ];
    let inits = [
        tensor("w1", &[HIDDEN, FEATURES], &W1),
        tensor("b1", &[HIDDEN], &B1),
        tensor("w2", &[CLASSES, HIDDEN], &W2),
        tensor("b2", &[CLASSES], &B2),
    ];
    let graph = graph(
        &nodes,
        &inits,
        &[value_info("X", &[Some(1), Some(FEATURES)])],
        &[value_info("probs", &[Some(1), Some(CLASSES)])],
    );
    model(graph)
}

/// The same network exported as `MatMul` + `Add` (activation × weight), which
/// exercises the transpose path and separate bias folding.
pub fn mlp_matmul() -> Vec<u8> {
    let w1t = transpose(HIDDEN, FEATURES, &W1);
    let w2t = transpose(CLASSES, HIDDEN, &W2);
    let nodes = [
        node("MatMul", "mm1", &["X", "w1t"], &["h1"], &[]),
        node("Add", "bias1", &["h1", "b1"], &["h1b"], &[]),
        node("Relu", "act", &["h1b"], &["h2"], &[]),
        node("MatMul", "mm2", &["h2", "w2t"], &["l1"], &[]),
        node("Add", "bias2", &["l1", "b2"], &["logits"], &[]),
        node("Softmax", "prob", &["logits"], &["probs"], &[]),
    ];
    let inits = [
        tensor("w1t", &[FEATURES, HIDDEN], &w1t),
        tensor("b1", &[HIDDEN], &B1),
        tensor("w2t", &[HIDDEN, CLASSES], &w2t),
        tensor("b2", &[CLASSES], &B2),
    ];
    let graph = graph(
        &nodes,
        &inits,
        &[value_info("X", &[Some(1), Some(FEATURES)])],
        &[value_info("probs", &[Some(1), Some(CLASSES)])],
    );
    model(graph)
}

/// A graph with a dynamic batch and feature dimension.
pub fn mlp_dynamic_input() -> Vec<u8> {
    let nodes = [node(
        "Gemm",
        "fc1",
        &["X", "w1", "b1"],
        &["logits"],
        &[attr_int("transB", 1)],
    )];
    let inits = [
        tensor("w1", &[CLASSES, FEATURES], &W2[..CLASSES * FEATURES]),
        tensor("b1", &[CLASSES], &B2),
    ];
    let graph = graph(
        &nodes,
        &inits,
        &[value_info("X", &[None, None])],
        &[value_info("logits", &[None, Some(CLASSES)])],
    );
    model(graph)
}

/// A graph using an op the runtime cannot execute.
pub fn unsupported_op() -> Vec<u8> {
    let nodes = [node("Conv", "conv1", &["X", "kernel"], &["y"], &[])];
    let inits = [tensor("kernel", &[1, 1, 3], &[1.0, 0.0, -1.0])];
    let graph = graph(
        &nodes,
        &inits,
        &[value_info("X", &[Some(1), Some(FEATURES)])],
        &[value_info("y", &[Some(1), Some(CLASSES)])],
    );
    model(graph)
}
