//! Verinoo reference model — deterministic intent classification.
//!
//! This is a real, if small, model: multinomial logistic regression over
//! hashed word uni-/bi-grams. The weights in `weights/intent-v1.json` are
//! produced by `cargo run -p verinoo-model --bin train` from the committed
//! corpus in `data/intents.jsonl`; nothing is generated at runtime.
//!
//! Inference is integer-only (Q16.16). Given the same input every node
//! computes byte-identical scores, which is what lets the chain treat the
//! output as part of consensus state.

use sha2::{Digest, Sha256};
use std::sync::OnceLock;

pub const MODEL_ID: &str = "intent-classifier-v1";
pub const ARCHITECTURE: &str = "hash-unigram-logreg-2048x4";
pub const FEATURES: usize = 2048;
pub const CLASSES: usize = 4;
pub const LABELS: [&str; CLASSES] = ["refund", "shipping", "billing", "cancel"];
pub const DESCRIPTION: &str =
    "Support-ticket intent classifier over refund / shipping / billing / cancel; integer-only deterministic inference";

/// Q16.16 one.
pub const ONE: i32 = 65_536;

const WEIGHTS_JSON: &str = include_str!("../weights/intent-v1.json");

#[derive(serde::Deserialize)]
struct Weights {
    #[serde(rename = "modelId")]
    model_id: String,
    features: usize,
    classes: usize,
    labels: Vec<String>,
    #[serde(default)]
    w: Vec<i32>,
    #[serde(default)]
    b: Vec<i32>,
}

impl Weights {
    fn normalised(mut self) -> Self {
        let need = self.features * self.classes;
        if self.w.len() < need {
            self.w.resize(need, 0);
        }
        if self.b.len() < self.classes {
            self.b.resize(self.classes, 0);
        }
        self
    }
}

fn weights() -> &'static Weights {
    static W: OnceLock<Weights> = OnceLock::new();
    W.get_or_init(|| {
        let parsed: Weights =
            serde_json::from_str(WEIGHTS_JSON).expect("embedded weights are valid JSON");
        parsed.normalised()
    })
}

pub fn sha256_hex(input: &str) -> String {
    let mut h = Sha256::new();
    h.update(input.as_bytes());
    hex::encode(h.finalize())
}

/// Identity of the exact weight bytes the node runs. Committed in receipts so
/// a verifier knows which parameters produced an output.
pub fn weights_hash() -> String {
    sha256_hex(WEIGHTS_JSON)
}

pub fn architecture() -> String {
    weights().model_id.clone()
}

pub fn parameter_count() -> u64 {
    let w = weights();
    (w.w.len() + w.b.len()) as u64
}

pub fn labels() -> Vec<String> {
    weights().labels.clone()
}

/// Deterministic feature bucket for a token (or `a|b` bigram string).
pub fn bucket(token: &str) -> usize {
    let digest = Sha256::digest(token.as_bytes());
    (u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]) as usize) % FEATURES
}

pub fn tokens(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

/// Integer bag-of-unigrams, expressed as Q16.16 shares of the total so the
/// feature vector is scale-free and platform-independent.
///
/// Unigrams only: with a corpus this small, bigrams memorise the training
/// sentences and generalise worse than plain word features.
pub fn featurize(text: &str) -> Vec<i32> {
    let toks = tokens(text);
    let mut counts = vec![0i64; FEATURES];
    for t in &toks {
        counts[bucket(t)] += 1;
    }
    let total: i64 = counts.iter().sum();
    if total == 0 {
        return vec![0; FEATURES];
    }
    counts
        .iter()
        .map(|&c| ((c * ONE as i64) / total) as i32)
        .collect()
}

/// Raw class logits in Q16.16.
pub fn logits(features: &[i32]) -> Vec<i32> {
    let w = weights();
    let mut out = vec![0i32; w.classes];
    for (c, slot) in out.iter_mut().enumerate() {
        let mut acc = w.b[c] as i64;
        let row = c * w.features;
        for (f, &feature) in features.iter().enumerate().take(w.features) {
            acc += ((feature as i64) * (w.w[row + f] as i64)) >> 16;
        }
        *slot = acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
    out
}

/// Integer softmax (same shape as the Q16.16 op witnesses).
pub fn softmax_q16(x: &[i32]) -> Vec<i32> {
    let f = ONE as i64;
    let max_v = *x.iter().max().unwrap_or(&0) as i64;
    let e: Vec<i64> = x
        .iter()
        .map(|&v| {
            let d = v as i64 - max_v;
            if d < 0 {
                0
            } else {
                d + f
            }
        })
        .collect();
    let denom = e.iter().sum::<i64>().max(1);
    e.iter().map(|&ei| ((ei * f) / denom) as i32).collect()
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Prediction {
    pub label: String,
    /// Class probabilities in Q16.16, index-aligned with [`LABELS`].
    pub scores: Vec<i32>,
    pub top: i32,
}

pub fn infer(input: &str) -> Prediction {
    let features = featurize(input);
    let scores = softmax_q16(&logits(&features));
    let mut best = 0usize;
    for i in 1..scores.len() {
        if scores[i] > scores[best] {
            best = i;
        }
    }
    Prediction {
        label: weights().labels[best].clone(),
        top: scores[best],
        scores,
    }
}

/// The exact bytes a task result commits to. Every field is derived from the
/// input and the model identity, so two honest nodes agree byte-for-byte.
pub fn result_json(input: &str) -> String {
    let p = infer(input);
    serde_json::json!({
        "modelId": MODEL_ID,
        "weightsHash": weights_hash(),
        "label": p.label,
        "scores": p.scores,
    })
    .to_string()
}

pub fn output_hash(input: &str) -> String {
    sha256_hex(&result_json(input))
}

/// The compiled-in classifier as a runtime graph, so it can be executed by
/// `verinoo-runtime` exactly like an operator-supplied model. Used by the
/// exporter binary and by the parity test.
pub fn export_graph(model_id: &str) -> serde_json::Value {
    let w = weights();
    serde_json::json!({
        "modelId": model_id,
        "architecture": ARCHITECTURE,
        "description": DESCRIPTION,
        "features": w.features,
        "labels": w.labels,
        "layers": [
            {
                "op": "dense",
                "input": w.features,
                "output": w.classes,
                "weights": w.w,
                "bias": w.b,
            },
            { "op": "softmax" },
        ],
    })
}

/// Canonical description served by the model registry endpoint.
pub fn descriptor() -> serde_json::Value {
    serde_json::json!({
        "modelId": MODEL_ID,
        "architecture": ARCHITECTURE,
        "description": DESCRIPTION,
        "labels": labels(),
        "features": FEATURES,
        "parameterCount": parameter_count(),
        "weightsHash": weights_hash(),
        "runtime": "q16.16-integer",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_is_deterministic() {
        let a = result_json("where is my package");
        let b = result_json("where is my package");
        assert_eq!(a, b);
        assert_eq!(
            output_hash("where is my package"),
            output_hash("where is my package")
        );
    }

    #[test]
    fn distinguishes_intents() {
        assert_eq!(
            infer("I want my money back for this broken item").label,
            "refund"
        );
        assert_eq!(infer("my parcel was never delivered").label, "shipping");
        assert_eq!(
            infer("why is there an extra charge on my invoice").label,
            "billing"
        );
        assert_eq!(infer("please cancel my subscription").label, "cancel");
    }

    #[test]
    fn scores_are_probabilities() {
        let p = infer("my delivery is late");
        assert_eq!(p.scores.len(), CLASSES);
        assert!(p.scores.iter().all(|s| *s >= 0 && *s <= ONE));
    }

    #[test]
    fn weights_identity_is_stable() {
        assert_eq!(weights_hash().len(), 64);
        assert_eq!(
            labels(),
            LABELS.iter().map(|s| s.to_string()).collect::<Vec<_>>()
        );
    }
}
