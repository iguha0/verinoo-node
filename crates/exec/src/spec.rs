//! Deterministic layer specifications per architecture.
//! Port of reference `src/zk/index.ts` getLayerSpec + seeded PRNG.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerSpec {
    pub index: u64,
    pub name: String,
    #[serde(rename = "opType")]
    pub op_type: String,
    #[serde(rename = "inputShape")]
    pub input_shape: Vec<u64>,
    #[serde(rename = "outputShape")]
    pub output_shape: Vec<u64>,
    pub tolerance: f64,
}

/// mulberry32 — bit-exact port of the reference PRNG.
fn mulberry32(mut seed: u32) -> impl FnMut() -> f64 {
    move || {
        seed = seed.wrapping_add(0x6D2B79F5);
        let mut t = seed;
        t = t.wrapping_mul(t ^ (t >> 15));
        t ^= t.wrapping_add(t | 1);
        t ^= t.wrapping_add(t ^ (t >> 7));
        let r = (t ^ t.wrapping_shr(14)) as f64;
        r / 4294967296.0
    }
}

fn to_hash_seed(s: &str) -> u32 {
    let digest = sha256_bytes(s);
    u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]])
}

pub fn sha256_bytes(input: &str) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(input.as_bytes());
    h.finalize().into()
}

pub fn get_layer_spec(architecture: &str) -> Vec<LayerSpec> {
    let seed = to_hash_seed(&format!("{architecture}-spec"));
    let mut rng = mulberry32(seed);

    let known: Vec<(&str, Vec<LayerSpec>)> = vec![
        (
            "Gemma-2B-IT",
            vec![
                layer(0, "embedding", "embedding", 256, 2048, 0.0),
                layer(1, "attn_0", "attention", 2048, 2048, 0.001),
                layer(2, "ffn_0", "ffn", 2048, 2048, 0.001),
                layer(3, "head_norm", "layernorm", 2048, 2048, 0.0001),
                layer(4, "head_lm", "head", 2048, 256000, 0.0),
            ],
        ),
        (
            "Phi-2-Medical-v1",
            vec![
                layer(0, "emb", "embedding", 51200, 2560, 0.0),
                layer(1, "attn_0", "attention", 2560, 2560, 0.001),
                layer(2, "ffn_0", "ffn", 2560, 2560, 0.001),
                layer(3, "head", "head", 2560, 51200, 0.0),
            ],
        ),
        (
            "LiveNet-Model",
            vec![
                layer(0, "embedding", "embedding", 2048, 2048, 0.0),
                layer(1, "attn_0", "attention", 2048, 2048, 0.001),
                layer(2, "ffn_0", "ffn", 2048, 2048, 0.001),
                layer(3, "head_norm", "layernorm", 2048, 2048, 0.0001),
                layer(4, "head", "head", 2048, 32000, 0.0),
            ],
        ),
        ("Relu-Test-Net", vec![layer(0, "relu_0", "relu", 8, 8, 0.0)]),
        (
            "Tiny-Test-Net",
            vec![
                layer(0, "attn_0", "attention", 4, 4, 0.001),
                layer(1, "ffn_0", "ffn", 4, 4, 0.001),
                layer(2, "head", "head", 4, 4, 0.0),
            ],
        ),
    ];

    if let Some((_, specs)) = known.iter().find(|(name, _)| *name == architecture) {
        return specs.clone();
    }

    // Deterministic fallback for unknown architectures.
    let layer_count = 3 + (rng() * 5.0) as usize;
    let ops = ["embedding", "attention", "ffn", "layernorm", "head"];
    let mut out: Vec<LayerSpec> = Vec::new();
    for i in 0..layer_count {
        let op = ops[(rng() * ops.len() as f64) as usize];
        let hidden = 512 + (rng() * 4096.0) as u64;
        let out_shape = if op == "head" { 32000 } else { hidden };
        out.push(layer(
            i as u64,
            &format!("layer_{i}"),
            op,
            hidden,
            out_shape,
            0.001,
        ));
    }
    if out.first().map(|l| l.op_type.as_str()) != Some("embedding") {
        let hidden = 512 + (rng() * 4096.0) as u64;
        out.insert(0, layer(0, "embedding", "embedding", 256, hidden, 0.0));
        for (i, l) in out.iter_mut().enumerate().skip(1) {
            l.index = i as u64;
        }
    }
    if out.last().map(|l| l.op_type.as_str()) != Some("head") {
        let hidden = 512 + (rng() * 4096.0) as u64;
        let idx = out.len() as u64;
        out.push(layer(idx, "head", "head", hidden, 32000, 0.0));
    }
    out
}

fn layer(
    index: u64,
    name: &str,
    op_type: &str,
    input: u64,
    output: u64,
    tolerance: f64,
) -> LayerSpec {
    LayerSpec {
        index,
        name: name.into(),
        op_type: op_type.into(),
        input_shape: vec![1, input],
        output_shape: vec![1, output],
        tolerance,
    }
}
