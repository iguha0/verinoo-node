//! Hand-encoded GGUF fixtures for the importer tests.
//!
//! The container is written here byte by byte so a failing test points at the
//! parser or the converter, not at an opaque binary blob checked into the tree.

#![allow(dead_code)]

pub struct FixtureTensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub kind: u32,
    pub data: Vec<u8>,
}

fn string(text: &str) -> Vec<u8> {
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(text.as_bytes());
    out
}

pub fn kv_string(key: &str, value: &str) -> Vec<u8> {
    let mut out = string(key);
    out.extend(8u32.to_le_bytes()); // GGUF_TYPE_STRING
    out.extend(string(value));
    out
}

pub fn kv_u32(key: &str, value: u32) -> Vec<u8> {
    let mut out = string(key);
    out.extend(4u32.to_le_bytes()); // GGUF_TYPE_UINT32
    out.extend(value.to_le_bytes());
    out
}

pub fn gguf(tensors: &[FixtureTensor], kv: &[Vec<u8>], alignment: usize) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend(b"GGUF");
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((kv.len() as u64).to_le_bytes());
    for entry in kv {
        out.extend(entry);
    }

    let mut offset = 0usize;
    for t in tensors {
        out.extend(string(&t.name));
        out.extend((t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            out.extend((*d as u64).to_le_bytes());
        }
        out.extend(t.kind.to_le_bytes());
        out.extend((offset as u64).to_le_bytes());
        offset += t.data.len().div_ceil(alignment) * alignment;
    }

    while out.len() % alignment != 0 {
        out.push(0);
    }
    for t in tensors {
        out.extend(&t.data);
        while out.len() % alignment != 0 {
            out.push(0);
        }
    }
    out
}

/// IEEE binary16 encoder, exact for the small values the fixtures use.
pub fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    if value == 0.0 {
        return sign << 15;
    }
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exp >= 31 {
        return (sign << 15) | 0x7c00;
    }
    if exp <= 0 {
        let mantissa = ((bits >> 23) & 0x7f | 0x80) >> (1 - exp);
        return (sign << 15) | (mantissa as u16 & 0x3ff);
    }
    let mantissa = ((bits >> 13) & 0x3ff) as u16;
    (sign << 15) | ((exp as u16) << 10) | mantissa
}

pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub const FEATURES: usize = 8;
pub const HIDDEN: usize = 4;
pub const CLASSES: usize = 2;

pub fn w1() -> Vec<f32> {
    (0..HIDDEN * FEATURES)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) / 16.0)
        .collect()
}

pub fn b1() -> Vec<f32> {
    vec![0.05, -0.05, 0.02, 0.0]
}

pub fn w2() -> Vec<f32> {
    (0..CLASSES * HIDDEN)
        .map(|i| ((i * 5 % 7) as f32 - 3.0) / 8.0)
        .collect()
}

pub fn b2() -> Vec<f32> {
    vec![0.01, -0.01]
}

fn tensor(name: &str, dims: &[usize], kind: u32, data: Vec<u8>) -> FixtureTensor {
    FixtureTensor {
        name: name.to_string(),
        dims: dims.to_vec(),
        kind,
        data,
    }
}

/// A two-layer classifier as F32 tensors, the layout a converter would emit.
pub fn f32_mlp(with_bias: bool) -> Vec<u8> {
    let mut tensors = vec![
        tensor("fc1.weight", &[FEATURES, HIDDEN], 0, f32_bytes(&w1())),
        tensor("fc2.weight", &[HIDDEN, CLASSES], 0, f32_bytes(&w2())),
    ];
    if with_bias {
        tensors.push(tensor("fc1.bias", &[HIDDEN], 0, f32_bytes(&b1())));
        tensors.push(tensor("fc2.bias", &[CLASSES], 0, f32_bytes(&b2())));
    }
    gguf(
        &tensors,
        &[
            kv_string("general.architecture", "acme"),
            kv_u32("general.alignment", 32),
        ],
        32,
    )
}

pub const Q_FEATURES: usize = 32;
pub const Q_CLASSES: usize = 2;
pub const Q_SCALE: f32 = 0.125;

/// Value the reference Q8_0 block layout produces for element `i` of row `row`.
pub fn q8_value(row: usize, i: usize) -> f32 {
    let q = ((i % 16) as i32 - 8) as f32;
    Q_SCALE * if row == 0 { q } else { -q }
}

/// A single Q8_0 layer: 2 rows × 32 columns, one 34-byte block per row.
pub fn q8_mlp() -> Vec<u8> {
    let mut weights = Vec::new();
    for row in 0..Q_CLASSES {
        weights.extend(f16_bits(Q_SCALE).to_le_bytes());
        for i in 0..Q_FEATURES {
            let q = ((i % 16) as i32 - 8) * if row == 0 { 1 } else { -1 };
            weights.push(q as i8 as u8);
        }
    }
    let tensors = vec![
        tensor("head.weight", &[Q_FEATURES, Q_CLASSES], 8, weights),
        tensor("head.bias", &[Q_CLASSES], 0, f32_bytes(&[0.0, 0.0])),
    ];
    gguf(&tensors, &[kv_string("general.architecture", "q8")], 32)
}

/// A file whose weight tensor uses a K-quant the importer refuses.
pub fn k_quant() -> Vec<u8> {
    // Two outputs so the label-count check passes and the refusal is the
    // quantisation itself, not a shape mismatch.
    let tensors = vec![tensor("quant.weight", &[256, 2], 12, vec![0u8; 2 * 144])];
    gguf(&tensors, &[], 32)
}
