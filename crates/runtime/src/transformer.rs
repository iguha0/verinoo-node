//! Deterministic transformer-encoder inference.
//!
//! This is the second runtime in the node: a small pre-norm transformer encoder
//! over a word-level vocabulary, followed by a classifier head. It is not a
//! generative LLM runtime: there is no tokenizer beyond whitespace/ASCII words,
//! no KV cache, no sampling and no causal mask.
//!
//! Every arithmetic operation is integer or fixed point:
//!
//! * activations and weights are Q16.16 (`ONE = 65_536`)
//! * dense layers accumulate in `i64` and shift each product by 16
//! * layer normalisation uses integer sums, a biased variance, integer square
//!   root and a Q16.16 reciprocal
//! * attention reuses the deterministic Q16.16 kernel in [`crate::attention`]
//! * softmax is the same integer approximation the classifier uses
//!
//! The result is a model that every node can execute byte-identically. The
//! committed `models/intent-transformer-v1.json` is trained by
//! `scripts/train_transformer.py`; the fixture next to it holds float-reference
//! outputs for parity tests.

use crate::{attention, sha256_hex, softmax, ONE};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

const MAX_HIDDEN: usize = 1024;
const MAX_SEQ_LEN: usize = 4096;
const MAX_VOCAB: usize = 1_000_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TransformerLayer {
    pub ln1_gamma: Vec<i32>,
    pub ln1_beta: Vec<i32>,
    pub q_weights: Vec<i32>,
    pub q_bias: Vec<i32>,
    pub k_weights: Vec<i32>,
    pub k_bias: Vec<i32>,
    pub v_weights: Vec<i32>,
    pub v_bias: Vec<i32>,
    pub o_weights: Vec<i32>,
    pub o_bias: Vec<i32>,
    pub ln2_gamma: Vec<i32>,
    pub ln2_beta: Vec<i32>,
    pub ff1_weights: Vec<i32>,
    pub ff1_bias: Vec<i32>,
    pub ff2_weights: Vec<i32>,
    pub ff2_bias: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TransformerSpec {
    #[serde(default)]
    pub kind: String,
    pub model_id: String,
    pub architecture: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_pooling")]
    pub pooling: String,
    pub vocab: Vec<String>,
    pub max_seq_len: usize,
    pub hidden: usize,
    pub heads: usize,
    pub ff: usize,
    pub layers: usize,
    pub labels: Vec<String>,
    pub embedding: Vec<i32>,
    pub positional: Vec<i32>,
    pub encoder: Vec<TransformerLayer>,
    pub lnf_gamma: Vec<i32>,
    pub lnf_beta: Vec<i32>,
    pub classifier_weights: Vec<i32>,
    pub classifier_bias: Vec<i32>,
}

fn default_pooling() -> String {
    "cls".into()
}

#[derive(Debug, Clone)]
pub struct Prediction {
    pub label: String,
    pub scores: Vec<i32>,
    pub logits: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct TransformerModel {
    spec: TransformerSpec,
    weights_hash: String,
    vocab: BTreeMap<String, usize>,
    cls_id: usize,
    unk_id: usize,
    head_dim: usize,
}

impl TransformerModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let value: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| format!("invalid JSON: {e}"))?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &serde_json::Value) -> Result<Self, String> {
        let spec: TransformerSpec = serde_json::from_value(value.clone())
            .map_err(|e| format!("invalid transformer: {e}"))?;
        validate_spec(&spec)?;
        let mut vocab = BTreeMap::new();
        for (i, token) in spec.vocab.iter().enumerate() {
            vocab.insert(token.clone(), i);
        }
        let cls_id = *vocab
            .get("<cls>")
            .ok_or_else(|| "vocabulary is missing <cls>".to_string())?;
        let unk_id = *vocab
            .get("<unk>")
            .ok_or_else(|| "vocabulary is missing <unk>".to_string())?;
        let weights_hash = sha256_hex(
            &serde_json::to_string(&spec).map_err(|e| format!("cannot hash model: {e}"))?,
        );
        Ok(Self {
            head_dim: spec.hidden / spec.heads,
            spec,
            weights_hash,
            vocab,
            cls_id,
            unk_id,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.spec.model_id
    }

    pub fn architecture(&self) -> &str {
        &self.spec.architecture
    }

    pub fn description(&self) -> &str {
        &self.spec.description
    }

    pub fn labels(&self) -> &[String] {
        &self.spec.labels
    }

    pub fn vocab_size(&self) -> usize {
        self.spec.vocab.len()
    }

    pub fn max_seq_len(&self) -> usize {
        self.spec.max_seq_len
    }

    pub fn weights_hash(&self) -> &str {
        &self.weights_hash
    }

    pub fn parameter_count(&self) -> u64 {
        parameter_count(&self.spec)
    }

    /// ASCII word tokenizer, identical to `scripts/train_transformer.py`:
    /// lowercase, split on anything outside `[a-z0-9]`, prepend `<cls>`,
    /// map unknown words to `<unk>`, truncate to `maxSeqLen`.
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::with_capacity(self.spec.max_seq_len.min(32));
        ids.push(self.cls_id);
        for token in ascii_words(text) {
            if ids.len() >= self.spec.max_seq_len {
                break;
            }
            ids.push(*self.vocab.get(&token).unwrap_or(&self.unk_id));
        }
        ids
    }

    pub fn forward_logits(&self, token_ids: &[usize]) -> Result<Vec<i32>, String> {
        if token_ids.is_empty() {
            return Err("token sequence must not be empty".into());
        }
        if token_ids.len() > self.spec.max_seq_len {
            return Err(format!(
                "sequence length {} exceeds maxSeqLen {}",
                token_ids.len(),
                self.spec.max_seq_len
            ));
        }
        if token_ids.iter().any(|id| *id >= self.spec.vocab.len()) {
            return Err("token id outside the vocabulary".into());
        }
        let hidden = self.spec.hidden;
        let mut x = vec![0i32; token_ids.len() * hidden];
        for (pos, token) in token_ids.iter().enumerate() {
            let emb = &self.spec.embedding[token * hidden..(token + 1) * hidden];
            let posv = &self.spec.positional[pos * hidden..(pos + 1) * hidden];
            for j in 0..hidden {
                x[pos * hidden + j] = emb[j].saturating_add(posv[j]);
            }
        }

        for layer in &self.spec.encoder {
            let norm = layer_norm(
                &x,
                &layer.ln1_gamma,
                &layer.ln1_beta,
                token_ids.len(),
                hidden,
            );
            let q = dense_rows(
                &norm,
                &layer.q_weights,
                &layer.q_bias,
                hidden,
                hidden,
                token_ids.len(),
            )?;
            let k = dense_rows(
                &norm,
                &layer.k_weights,
                &layer.k_bias,
                hidden,
                hidden,
                token_ids.len(),
            )?;
            let v = dense_rows(
                &norm,
                &layer.v_weights,
                &layer.v_bias,
                hidden,
                hidden,
                token_ids.len(),
            )?;
            let attn = attention::multi_head_attention(&q, &k, &v, self.spec.heads, self.head_dim)?;
            let projected = dense_rows(
                &attn,
                &layer.o_weights,
                &layer.o_bias,
                hidden,
                hidden,
                token_ids.len(),
            )?;
            add_in_place(&mut x, &projected);

            let norm = layer_norm(
                &x,
                &layer.ln2_gamma,
                &layer.ln2_beta,
                token_ids.len(),
                hidden,
            );
            let mut ff = dense_rows(
                &norm,
                &layer.ff1_weights,
                &layer.ff1_bias,
                hidden,
                self.spec.ff,
                token_ids.len(),
            )?;
            for value in ff.iter_mut() {
                *value = (*value).max(0);
            }
            let ff = dense_rows(
                &ff,
                &layer.ff2_weights,
                &layer.ff2_bias,
                self.spec.ff,
                hidden,
                token_ids.len(),
            )?;
            add_in_place(&mut x, &ff);
        }

        let x = layer_norm(
            &x,
            &self.spec.lnf_gamma,
            &self.spec.lnf_beta,
            token_ids.len(),
            hidden,
        );
        let pooled = match self.spec.pooling.as_str() {
            "cls" => x[..hidden].to_vec(),
            "mean" => {
                let mut out = vec![0i64; hidden];
                let count = token_ids.len() as i64;
                for pos in 0..token_ids.len() {
                    for j in 0..hidden {
                        out[j] += x[pos * hidden + j] as i64;
                    }
                }
                out.iter()
                    .map(|v| (v / count).clamp(i32::MIN as i64, i32::MAX as i64) as i32)
                    .collect()
            }
            other => return Err(format!("unsupported pooling `{other}`")),
        };
        dense(
            &pooled,
            &self.spec.classifier_weights,
            &self.spec.classifier_bias,
            hidden,
            self.spec.labels.len(),
        )
    }

    pub fn infer(&self, text: &str) -> Result<Prediction, String> {
        let logits = self.forward_logits(&self.tokenize(text))?;
        let scores = softmax(&logits);
        let label = argmax_index(&scores)
            .and_then(|i| self.spec.labels.get(i).cloned())
            .unwrap_or_default();
        Ok(Prediction {
            label,
            scores,
            logits,
        })
    }

    pub fn result_json(&self, text: &str) -> Result<String, String> {
        let prediction = self.infer(text)?;
        Ok(serde_json::json!({
            "label": prediction.label,
            "modelId": self.spec.model_id,
            "scores": prediction.scores,
            "weightsHash": self.weights_hash,
        })
        .to_string())
    }

    pub fn descriptor(&self) -> serde_json::Value {
        serde_json::json!({
            "modelId": self.spec.model_id,
            "architecture": self.spec.architecture,
            "description": self.spec.description,
            "labels": self.spec.labels,
            "vocabSize": self.spec.vocab.len(),
            "maxSeqLen": self.spec.max_seq_len,
            "hidden": self.spec.hidden,
            "heads": self.spec.heads,
            "layers": self.spec.layers,
            "parameterCount": self.parameter_count(),
            "runtime": "q16.16-transformer-encoder",
            "weightsHash": self.weights_hash,
        })
    }
}

fn validate_spec(spec: &TransformerSpec) -> Result<(), String> {
    if spec.kind != "transformer-classifier" {
        return Err(format!(
            "kind must be `transformer-classifier`, got `{}`",
            spec.kind
        ));
    }
    if spec.model_id.trim().is_empty() {
        return Err("modelId must not be empty".into());
    }
    if spec.vocab.is_empty() || spec.vocab.len() > MAX_VOCAB {
        return Err("vocab size is invalid".into());
    }
    if spec.labels.is_empty() {
        return Err("at least one label is required".into());
    }
    if spec.hidden == 0 || spec.hidden > MAX_HIDDEN {
        return Err(format!("hidden must be between 1 and {MAX_HIDDEN}"));
    }
    if spec.heads == 0 || spec.heads > spec.hidden || !spec.hidden.is_multiple_of(spec.heads) {
        return Err("heads must divide hidden".into());
    }
    if spec.max_seq_len == 0 || spec.max_seq_len > MAX_SEQ_LEN {
        return Err(format!("maxSeqLen must be between 1 and {MAX_SEQ_LEN}"));
    }
    if spec.ff == 0 || spec.ff > MAX_HIDDEN * 8 {
        return Err("ff is out of range".into());
    }
    if spec.layers == 0 || spec.layers != spec.encoder.len() {
        return Err("layers must equal the number of encoder blocks".into());
    }
    let h = spec.hidden;
    let ff = spec.ff;
    expect_len("embedding", spec.embedding.len(), spec.vocab.len() * h)?;
    expect_len("positional", spec.positional.len(), spec.max_seq_len * h)?;
    expect_len("lnfGamma", spec.lnf_gamma.len(), h)?;
    expect_len("lnfBeta", spec.lnf_beta.len(), h)?;
    expect_len(
        "classifierWeights",
        spec.classifier_weights.len(),
        spec.labels.len() * h,
    )?;
    expect_len(
        "classifierBias",
        spec.classifier_bias.len(),
        spec.labels.len(),
    )?;
    let matrix = h * h;
    for (i, layer) in spec.encoder.iter().enumerate() {
        for (name, len) in [
            ("ln1Gamma", layer.ln1_gamma.len()),
            ("ln1Beta", layer.ln1_beta.len()),
            ("ln2Gamma", layer.ln2_gamma.len()),
            ("ln2Beta", layer.ln2_beta.len()),
            ("qBias", layer.q_bias.len()),
            ("kBias", layer.k_bias.len()),
            ("vBias", layer.v_bias.len()),
            ("oBias", layer.o_bias.len()),
        ] {
            expect_len(&format!("layer {i} {name}"), len, h)?;
        }
        for (name, len) in [
            ("qWeights", layer.q_weights.len()),
            ("kWeights", layer.k_weights.len()),
            ("vWeights", layer.v_weights.len()),
            ("oWeights", layer.o_weights.len()),
        ] {
            expect_len(&format!("layer {i} {name}"), len, matrix)?;
        }
        expect_len(
            &format!("layer {i} ff1Weights"),
            layer.ff1_weights.len(),
            ff * h,
        )?;
        expect_len(&format!("layer {i} ff1Bias"), layer.ff1_bias.len(), ff)?;
        expect_len(
            &format!("layer {i} ff2Weights"),
            layer.ff2_weights.len(),
            h * ff,
        )?;
        expect_len(&format!("layer {i} ff2Bias"), layer.ff2_bias.len(), h)?;
    }
    Ok(())
}

fn expect_len(name: &str, got: usize, want: usize) -> Result<(), String> {
    if got == want {
        Ok(())
    } else {
        Err(format!("{name}: expected {want} values, got {got}"))
    }
}

fn parameter_count(spec: &TransformerSpec) -> u64 {
    let mut total = (spec.embedding.len()
        + spec.positional.len()
        + spec.lnf_gamma.len()
        + spec.lnf_beta.len()
        + spec.classifier_weights.len()
        + spec.classifier_bias.len()) as u64;
    for layer in &spec.encoder {
        total += (layer.ln1_gamma.len()
            + layer.ln1_beta.len()
            + layer.q_weights.len()
            + layer.q_bias.len()
            + layer.k_weights.len()
            + layer.k_bias.len()
            + layer.v_weights.len()
            + layer.v_bias.len()
            + layer.o_weights.len()
            + layer.o_bias.len()
            + layer.ln2_gamma.len()
            + layer.ln2_beta.len()
            + layer.ff1_weights.len()
            + layer.ff1_bias.len()
            + layer.ff2_weights.len()
            + layer.ff2_bias.len()) as u64;
    }
    total
}

pub(crate) fn dense(
    x: &[i32],
    weights: &[i32],
    bias: &[i32],
    input: usize,
    output: usize,
) -> Result<Vec<i32>, String> {
    if x.len() != input {
        return Err(format!("dense input: expected {input}, got {}", x.len()));
    }
    if weights.len() != output * input {
        return Err(format!(
            "dense weights: expected {}, got {}",
            output * input,
            weights.len()
        ));
    }
    if bias.len() != output {
        return Err(format!("dense bias: expected {output}, got {}", bias.len()));
    }
    let mut out = vec![0i32; output];
    for (j, slot) in out.iter_mut().enumerate() {
        let mut acc = bias[j] as i64;
        let row = j * input;
        for (f, &xv) in x.iter().enumerate() {
            acc += ((xv as i64) * (weights[row + f] as i64)) >> 16;
        }
        *slot = acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
    Ok(out)
}

pub(crate) fn dense_rows(
    x: &[i32],
    weights: &[i32],
    bias: &[i32],
    input: usize,
    output: usize,
    rows: usize,
) -> Result<Vec<i32>, String> {
    if x.len() != rows * input {
        return Err(format!(
            "dense rows input: expected {}, got {}",
            rows * input,
            x.len()
        ));
    }
    let mut out = Vec::with_capacity(rows * output);
    for row in 0..rows {
        let values = dense(
            &x[row * input..(row + 1) * input],
            weights,
            bias,
            input,
            output,
        )?;
        out.extend_from_slice(&values);
    }
    Ok(out)
}

pub(crate) fn add_in_place(x: &mut [i32], delta: &[i32]) {
    for (a, b) in x.iter_mut().zip(delta) {
        *a = (*a as i64 + *b as i64).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
}

pub(crate) fn layer_norm(
    x: &[i32],
    gamma: &[i32],
    beta: &[i32],
    rows: usize,
    hidden: usize,
) -> Vec<i32> {
    let mut out = vec![0i32; rows * hidden];
    for row in 0..rows {
        let values = &x[row * hidden..(row + 1) * hidden];
        let sum: i64 = values.iter().map(|v| *v as i64).sum();
        let mean = div_round(sum, hidden as i64);
        let mut variance = 0i64;
        for value in values {
            let d = *value as i64 - mean;
            variance += d * d;
        }
        variance = div_round(variance, hidden as i64);
        let mut std = isqrt(variance.max(0) as u64) as i64;
        if std < 1 {
            std = 1;
        }
        let inv = ((ONE as i64 * ONE as i64) / std).clamp(0, 8 * ONE as i64);
        for j in 0..hidden {
            let d = values[j] as i64 - mean;
            let norm = (d * inv) >> 16;
            let y = ((norm * gamma[j] as i64) >> 16) + beta[j] as i64;
            out[row * hidden + j] = y.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }
    out
}

fn div_round(value: i64, divisor: i64) -> i64 {
    if divisor == 0 {
        return 0;
    }
    if value >= 0 {
        (value + divisor / 2) / divisor
    } else {
        (value - divisor / 2) / divisor
    }
}

pub(crate) fn isqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

pub(crate) fn argmax_index(values: &[i32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by_key(|(_, value)| *value)
        .map(|(index, _)| index)
}

fn ascii_words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_ascii_alphanumeric() {
            current.push(c.to_ascii_lowercase());
        } else if !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct TransformerRegistry {
    models: BTreeMap<String, Arc<TransformerModel>>,
}

impl TransformerRegistry {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Load every `*.json` file with `kind: transformer-classifier`. Graph
    /// model files are ignored; the graph registry loads those.
    pub fn load_dir<P: AsRef<Path>>(dir: P) -> (Self, Vec<String>) {
        let dir = dir.as_ref();
        let mut reg = Self::empty();
        let mut problems = Vec::new();
        let Ok(entries) = std::fs::read_dir(dir) else {
            return (reg, problems);
        };
        let mut paths: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|e| e == "json").unwrap_or(false))
            .collect();
        paths.sort();
        for path in paths {
            let bytes = match std::fs::read(&path) {
                Ok(bytes) => bytes,
                Err(e) => {
                    problems.push(format!("{}: {e}", path.display()));
                    continue;
                }
            };
            let is_transformer = serde_json::from_slice::<serde_json::Value>(&bytes)
                .ok()
                .and_then(|v| v.get("kind").and_then(|k| k.as_str()).map(String::from))
                .map(|kind| kind == "transformer-classifier")
                .unwrap_or(false);
            if !is_transformer {
                continue;
            }
            match TransformerModel::from_bytes(&bytes) {
                Ok(model) => {
                    if let Err(e) = reg.insert(model) {
                        problems.push(format!("{}: {e}", path.display()));
                    }
                }
                Err(e) => problems.push(format!("{}: {e}", path.display())),
            }
        }
        (reg, problems)
    }

    pub fn insert(&mut self, model: TransformerModel) -> Result<(), String> {
        let id = model.model_id().to_string();
        if self.models.contains_key(&id) {
            return Err(format!("duplicate model id `{id}`"));
        }
        self.models.insert(id, Arc::new(model));
        Ok(())
    }

    pub fn get(&self, model_id: &str) -> Option<&Arc<TransformerModel>> {
        self.models.get(model_id)
    }

    pub fn ids(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    pub fn descriptors(&self) -> Vec<serde_json::Value> {
        self.models.values().map(|m| m.descriptor()).collect()
    }

    pub fn hash(&self) -> String {
        let mut lines: Vec<String> = self
            .models
            .values()
            .map(|m| format!("{}:{}", m.model_id(), m.weights_hash()))
            .collect();
        lines.sort();
        sha256_hex(&lines.join("\n"))
    }
}

static TRANSFORMERS: OnceLock<TransformerRegistry> = OnceLock::new();

pub fn transformer_global() -> &'static TransformerRegistry {
    TRANSFORMERS.get_or_init(TransformerRegistry::empty)
}

pub fn install_transformers_once(registry: TransformerRegistry) {
    let _ = TRANSFORMERS.set(registry);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/intent-transformer-v1.json")
    }

    fn fixture_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/intent-transformer-v1.test.json")
    }

    fn load() -> TransformerModel {
        let bytes = std::fs::read(model_path()).expect("committed model");
        TransformerModel::from_bytes(&bytes).expect("valid transformer")
    }

    #[test]
    fn tokenizer_matches_the_training_script() {
        let model = load();
        let ids = model.tokenize("I want a REFUND for my last order!");
        assert_eq!(ids[0], 2, "CLS is first");
        assert_eq!(ids.len(), 9, "CLS + eight words");
        assert!(model.tokenize("").len() == 1);
        let long = model.tokenize("a b c d e f g h i j k l m n o p");
        assert_eq!(long.len(), model.max_seq_len());
    }

    #[test]
    fn deterministic_forward_pass() {
        let model = load();
        let text = "my parcel was never delivered";
        let a = model.infer(text).unwrap();
        let b = model.infer(text).unwrap();
        assert_eq!(a.scores, b.scores);
        assert_eq!(a.label, b.label);
        assert!(!a.label.is_empty());
    }

    #[test]
    fn labels_match_the_float_reference() {
        let model = load();
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(fixture_path()).expect("fixture"))
                .expect("fixture JSON");
        let cases = fixture["cases"].as_array().expect("cases");
        let mut checked = 0usize;
        let mut mismatches = Vec::new();
        let mut float_ok = 0usize;
        let mut int_ok = 0usize;
        for case in cases {
            let text = case["text"].as_str().unwrap();
            let truth = case["label"].as_str().unwrap();
            let expected = case["predicted"].as_str().unwrap();
            let got = model.infer(text).unwrap();
            checked += 1;
            float_ok += usize::from(expected == truth);
            int_ok += usize::from(got.label == truth);
            if got.label != expected {
                mismatches.push(format!(
                    "{text}: int {}, float {expected}, truth {truth}",
                    got.label
                ));
            }
        }
        println!(
            "parity {}/{}  float accuracy {:.3}  integer accuracy {:.3}",
            checked - mismatches.len(),
            checked,
            float_ok as f64 / checked as f64,
            int_ok as f64 / checked as f64
        );
        assert_eq!(
            mismatches.len(),
            0,
            "label parity failed on {} of {checked}: {:?}",
            mismatches.len(),
            mismatches
        );
    }

    #[test]
    fn model_descriptor_is_stable() {
        let model = load();
        assert_eq!(model.model_id(), "intent-transformer-v1");
        assert_eq!(model.labels().len(), 4);
        assert_eq!(model.weights_hash().len(), 64);
        assert!(model.parameter_count() > 5000);
        let d = model.descriptor();
        assert_eq!(d["runtime"], "q16.16-transformer-encoder");
        assert_eq!(d["heads"], 2);
    }

    #[test]
    fn malformed_models_are_refused() {
        let bytes = std::fs::read(model_path()).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        value["kind"] = serde_json::json!("graph");
        assert!(TransformerModel::from_value(&value).is_err());
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        value["classifierWeights"] = serde_json::json!([]);
        assert!(TransformerModel::from_value(&value).is_err());
    }

    #[test]
    fn registry_hashes_and_ignores_graph_files() {
        let model = load();
        let mut reg = TransformerRegistry::empty();
        reg.insert(model.clone()).unwrap();
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.hash().len(), 64);
        assert_eq!(
            reg.get("intent-transformer-v1").unwrap().labels(),
            model.labels()
        );
    }
}
