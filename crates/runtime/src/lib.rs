//! Deterministic inference runtime for operator-supplied models.
//!
//! The chain can only treat an output as consensus state if every node derives
//! byte-identical bytes from the same weights. This runtime is therefore
//! integer-only: Q16.16 fixed point, `i64` accumulators, no floating point, no
//! threads, and no dependency on a framework runtime. A model is a small graph
//! committed as JSON next to the parameters it describes.
//!
//! Supported ops:
//!
//! * `dense` — `y[j] = bias[j] + Σ_f (x[f] * W[j][f]) >> 16`, row-major `W`
//! * `relu`  — `x = max(x, 0)`
//! * `softmax` — integer approximation (see [`softmax`])
//!
//! Anything else is rejected at load time rather than silently mis-executed,
//! because a divergent node would fork the chain.
//!
//! ```json
//! {
//!   "modelId": "acme-support-v1",
//!   "architecture": "dense-2048-4",
//!   "features": 2048,
//!   "labels": ["refund", "shipping", "billing", "cancel"],
//!   "layers": [
//!     {"op": "dense", "input": 2048, "output": 4, "weights": [0, 1, 2], "bias": [0]},
//!     {"op": "softmax"}
//!   ]
//! }
//! ```

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use sha2::{Digest, Sha256};

/// Q16.16 one.
pub const ONE: i32 = 65_536;

const MAX_LAYERS: usize = 64;
const MAX_PARAMS_PER_LAYER: usize = 16_000_000;

pub fn sha256_hex(input: &str) -> String {
    let mut h = Sha256::new();
    h.update(input.as_bytes());
    hex::encode(h.finalize())
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Layer {
    pub op: String,
    #[serde(default)]
    pub input: usize,
    #[serde(default)]
    pub output: usize,
    #[serde(default)]
    pub weights: Vec<i32>,
    #[serde(default)]
    pub bias: Vec<i32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelSpec {
    #[serde(rename = "modelId")]
    pub model_id: String,
    #[serde(default)]
    pub architecture: String,
    #[serde(default)]
    pub description: String,
    /// Width of the hashed-unigram feature vector.
    pub features: usize,
    pub labels: Vec<String>,
    pub layers: Vec<Layer>,
}

/// A loaded, validated model. Holding one means it is safe to execute: the
/// graph shape was checked against the parameter arrays at load time.
#[derive(Debug, Clone)]
pub struct Model {
    spec: ModelSpec,
    source_hash: String,
    params: u64,
}

impl Model {
    /// Parse and validate a model file.
    ///
    /// The identity hash is taken over the *canonical* definition, not the raw
    /// bytes, so reformatting the JSON (or reordering keys) does not change a
    /// model's `weightsHash` and invalidate old receipts. Any semantic edit —
    /// a parameter, a label, the featuriser width — does.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let spec: ModelSpec =
            serde_json::from_slice(bytes).map_err(|e| format!("invalid model file: {e}"))?;
        let params = validate(&spec)?;
        let canonical = serde_json::to_string(&spec).map_err(|e| e.to_string())?;
        Ok(Self {
            spec,
            source_hash: sha256_hex(&canonical),
            params,
        })
    }

    pub fn from_value(value: &serde_json::Value) -> Result<Self, String> {
        let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
        Self::from_bytes(&bytes)
    }

    pub fn model_id(&self) -> &str {
        &self.spec.model_id
    }

    pub fn architecture(&self) -> &str {
        &self.spec.architecture
    }

    pub fn features(&self) -> usize {
        self.spec.features
    }

    pub fn labels(&self) -> &[String] {
        &self.spec.labels
    }

    pub fn parameter_count(&self) -> u64 {
        self.params
    }

    /// sha256 of the committed model file — changes with any parameter or
    /// label edit.
    pub fn weights_hash(&self) -> &str {
        &self.source_hash
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    pub fn featurize(&self, text: &str) -> Vec<i32> {
        featurize(text, self.spec.features)
    }

    /// Run the graph. The last layer's activations are returned as Q16.16
    /// scores; a trailing softmax is applied implicitly if the graph does not
    /// end with one.
    pub fn forward(&self, features: &[i32]) -> Vec<i32> {
        let mut x = self.run_layers(features, false);
        let has_softmax = self.spec.layers.iter().any(|l| l.op == "softmax");
        if !has_softmax {
            x = softmax(&x);
        }
        x
    }

    /// Apply a `dense` layer to `x`: `y[j] = bias[j] + Σ_f (x[f] * W[j][f]) >> 16`.
    fn dense_into(&self, layer: &Layer, x: &[i32]) -> Vec<i32> {
        let mut out = vec![0i32; layer.output];
        for (j, slot) in out.iter_mut().enumerate() {
            let mut acc = layer.bias[j] as i64;
            let row = j * layer.input;
            for (f, &xv) in x.iter().enumerate().take(layer.input) {
                acc += ((xv as i64) * (layer.weights[row + f] as i64)) >> 16;
            }
            *slot = acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
        out
    }

    fn run_layers(&self, features: &[i32], stop_before_softmax: bool) -> Vec<i32> {
        let mut x = features.to_vec();
        x.resize(self.spec.features, 0);
        for layer in &self.spec.layers {
            match layer.op.as_str() {
                "dense" => x = self.dense_into(layer, &x),
                "relu" => {
                    for v in x.iter_mut() {
                        if *v < 0 {
                            *v = 0;
                        }
                    }
                }
                "softmax" => {
                    if stop_before_softmax {
                        return x;
                    }
                    x = softmax(&x);
                }
                _ => {}
            }
        }
        x
    }

    /// Raw logits: the graph's scores before softmax. A converter uses these to
    /// compare integer execution against a float reference implementation.
    pub fn forward_logits(&self, features: &[i32]) -> Vec<i32> {
        self.run_layers(features, true)
    }

    pub fn infer(&self, text: &str) -> Prediction {
        let scores = self.forward(&self.featurize(text));
        let mut best = 0usize;
        for i in 1..scores.len() {
            if scores[i] > scores[best] {
                best = i;
            }
        }
        Prediction {
            label: self.spec.labels.get(best).cloned().unwrap_or_default(),
            top: scores.get(best).copied().unwrap_or(0),
            scores,
        }
    }

    /// The exact result bytes committed on-chain. Field order is serde_json's
    /// (alphabetical), which keeps the hash stable across versions.
    pub fn result_json(&self, text: &str) -> String {
        let p = self.infer(text);
        serde_json::json!({
            "modelId": self.spec.model_id,
            "weightsHash": self.source_hash,
            "label": p.label,
            "scores": p.scores,
        })
        .to_string()
    }

    pub fn descriptor(&self) -> serde_json::Value {
        serde_json::json!({
            "modelId": self.spec.model_id,
            "architecture": self.spec.architecture,
            "description": self.spec.description,
            "labels": self.spec.labels,
            "features": self.spec.features,
            "layers": self.spec.layers.len(),
            "parameterCount": self.params,
            "weightsHash": self.source_hash,
            "runtime": "q16.16-integer",
        })
    }
}

fn validate(spec: &ModelSpec) -> Result<u64, String> {
    if spec.model_id.trim().is_empty() {
        return Err("modelId is required".into());
    }
    if spec.features == 0 {
        return Err("features must be greater than zero".into());
    }
    if spec.labels.is_empty() {
        return Err("labels must not be empty".into());
    }
    if spec.layers.is_empty() {
        return Err("graph has no layers".into());
    }
    if spec.layers.len() > MAX_LAYERS {
        return Err(format!("graph has more than {MAX_LAYERS} layers"));
    }

    let mut width = spec.features;
    let mut params = 0u64;
    for (i, layer) in spec.layers.iter().enumerate() {
        match layer.op.as_str() {
            "dense" => {
                if layer.input != width {
                    return Err(format!(
                        "layer {i}: input is {}, previous layer produces {width}",
                        layer.input
                    ));
                }
                if layer.output == 0 {
                    return Err(format!("layer {i}: output must be greater than zero"));
                }
                let need = layer
                    .input
                    .checked_mul(layer.output)
                    .ok_or_else(|| format!("layer {i}: dimensions overflow"))?;
                if need > MAX_PARAMS_PER_LAYER {
                    return Err(format!("layer {i}: {need} parameters is too large"));
                }
                if layer.weights.len() != need {
                    return Err(format!(
                        "layer {i}: expected {need} weights for {}x{}, found {}",
                        layer.input,
                        layer.output,
                        layer.weights.len()
                    ));
                }
                if layer.bias.len() != layer.output {
                    return Err(format!(
                        "layer {i}: expected {} bias terms, found {}",
                        layer.output,
                        layer.bias.len()
                    ));
                }
                params += (layer.weights.len() + layer.bias.len()) as u64;
                width = layer.output;
            }
            "relu" | "softmax" => {}
            other => {
                return Err(format!(
                    "layer {i}: unsupported op `{other}` (supported: dense, relu, softmax)"
                ))
            }
        }
    }
    if width != spec.labels.len() {
        return Err(format!(
            "graph produces {width} outputs but {} labels are declared",
            spec.labels.len()
        ));
    }
    Ok(params)
}

/// Integer softmax approximation, identical to the reference classifier's, so
/// a graph model and the built-in model agree byte-for-byte on the same
/// parameters.
pub fn softmax(x: &[i32]) -> Vec<i32> {
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

pub fn bucket(token: &str, features: usize) -> usize {
    let digest = Sha256::digest(token.as_bytes());
    (u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]) as usize) % features
}

pub fn tokens(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

/// Integer bag-of-unigrams as Q16.16 shares of the token total. Same
/// featurisation as the reference model, so exported weights keep working.
pub fn featurize(text: &str, features: usize) -> Vec<i32> {
    if features == 0 {
        return Vec::new();
    }
    let mut counts = vec![0i64; features];
    for t in tokens(text) {
        counts[bucket(&t, features)] += 1;
    }
    let total: i64 = counts.iter().sum();
    if total == 0 {
        return vec![0; features];
    }
    counts
        .iter()
        .map(|&c| ((c * ONE as i64) / total) as i32)
        .collect()
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Prediction {
    pub label: String,
    pub scores: Vec<i32>,
    pub top: i32,
}

/// The set of models an operator has installed. Every node in a network must
/// run the same set: execution is part of consensus, so a missing or different
/// model changes the resulting state root.
#[derive(Debug, Clone, Default)]
pub struct Registry {
    models: BTreeMap<String, Arc<Model>>,
}

impl Registry {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Load every `*.json` model file in `dir`. Problems are returned rather
    /// than panicking so a node can report them and keep running.
    pub fn load_dir<P: AsRef<Path>>(dir: P) -> (Self, Vec<String>) {
        let dir = dir.as_ref();
        let mut reg = Registry::empty();
        let mut problems = Vec::new();
        let Ok(entries) = std::fs::read_dir(dir) else {
            return (reg, problems);
        };
        let mut paths: Vec<_> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|e| e == "json").unwrap_or(false))
            .collect();
        paths.sort();
        for path in paths {
            match std::fs::read(&path) {
                Ok(bytes) => match Model::from_bytes(&bytes) {
                    Ok(model) => {
                        if let Err(e) = reg.insert(model) {
                            problems.push(format!("{}: {e}", path.display()));
                        }
                    }
                    Err(e) => problems.push(format!("{}: {e}", path.display())),
                },
                Err(e) => problems.push(format!("{}: {e}", path.display())),
            }
        }
        (reg, problems)
    }

    pub fn insert(&mut self, model: Model) -> Result<(), String> {
        let id = model.model_id().to_string();
        if self.models.contains_key(&id) {
            return Err(format!("duplicate model id `{id}`"));
        }
        self.models.insert(id, Arc::new(model));
        Ok(())
    }

    pub fn get(&self, model_id: &str) -> Option<&Arc<Model>> {
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

    /// Fingerprint of the whole model set. Two nodes must agree on this to
    /// stay on the same chain.
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

static REGISTRY: OnceLock<Registry> = OnceLock::new();

/// The process-wide model set. Empty until [`install`] runs, so tests and
/// non-validating tools work without models installed.
pub fn global() -> &'static Registry {
    REGISTRY.get_or_init(Registry::empty)
}

pub fn install(registry: Registry) -> Result<(), Registry> {
    REGISTRY.set(registry)
}

/// Install once; later calls are ignored (a node installs exactly one set).
pub fn install_once(registry: Registry) {
    let _ = REGISTRY.set(registry);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_model() -> Model {
        // 2 features -> 3 hidden (relu) -> 2 labels
        let spec = serde_json::json!({
            "modelId": "unit-graph",
            "architecture": "dense-relu-dense",
            "features": 2,
            "labels": ["no", "yes"],
            "layers": [
                {"op": "dense", "input": 2, "output": 3,
                 "weights": [ONE, 0, 0, ONE, ONE / 2, ONE / 2], "bias": [0, 0, 0]},
                {"op": "relu"},
                {"op": "dense", "input": 3, "output": 2,
                 "weights": [ONE, 0, 0, 0, ONE, 0], "bias": [0, 0]},
                {"op": "softmax"}
            ]
        });
        Model::from_value(&spec).unwrap()
    }

    #[test]
    fn dense_layer_uses_q16_16_accumulation() {
        let model = dense_model();
        let out = model.forward(&[ONE, ONE]);
        // hidden = [1.0, 1.0, 0.5]
        // logits = [1.0, 1.0] -> equal scores, no label wins by a margin
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], out[1]);
        assert!(out.iter().all(|s| *s >= 0 && *s <= ONE));
    }

    #[test]
    fn relu_clamps_negative_activations() {
        let spec = serde_json::json!({
            "modelId": "relu-only",
            "features": 2,
            "labels": ["a", "b"],
            "layers": [
                {"op": "dense", "input": 2, "output": 2,
                 "weights": [ONE, 0, 0, ONE], "bias": [-2 * ONE, 0]},
                {"op": "relu"},
                {"op": "softmax"}
            ]
        });
        let model = Model::from_value(&spec).unwrap();
        let out = model.forward(&[ONE, 0]);
        // first unit is clamped to 0, second stays 0 -> both equal after softmax
        assert_eq!(out[0], out[1]);
    }

    #[test]
    fn graph_is_validated_against_its_parameters() {
        let bad = serde_json::json!({
            "modelId": "broken",
            "features": 4,
            "labels": ["a", "b"],
            "layers": [
                {"op": "dense", "input": 4, "output": 2, "weights": [1, 2, 3], "bias": [0, 0]}
            ]
        });
        let err = Model::from_value(&bad).unwrap_err();
        assert!(err.contains("weights"), "{err}");

        let wrong_width = serde_json::json!({
            "modelId": "broken",
            "features": 4,
            "labels": ["a", "b"],
            "layers": [{"op": "dense", "input": 2, "output": 2,
                        "weights": [0, 0, 0, 0], "bias": [0, 0]}]
        });
        assert!(Model::from_value(&wrong_width).is_err());

        let unknown_op = serde_json::json!({
            "modelId": "broken",
            "features": 1,
            "labels": ["a"],
            "layers": [{"op": "matmul", "input": 1, "output": 1}]
        });
        assert!(Model::from_value(&unknown_op)
            .unwrap_err()
            .contains("unsupported"));
    }

    #[test]
    fn inference_is_deterministic_and_hash_stable() {
        let model = dense_model();
        assert_eq!(
            model.result_json("hello world"),
            model.result_json("hello world")
        );
        assert_eq!(model.weights_hash().len(), 64);
        let clone = Model::from_bytes(&serde_json::to_vec(model.spec()).unwrap()).unwrap();
        assert_eq!(model.weights_hash(), clone.weights_hash());
    }

    #[test]
    fn registry_rejects_duplicate_ids_and_hashes_the_set() {
        let mut reg = Registry::empty();
        reg.insert(dense_model()).unwrap();
        assert!(reg.insert(dense_model()).is_err());
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.hash().len(), 64);
        // 2*3 + 3 biases + 3*2 + 2 biases
        assert_eq!(
            reg.get("unit-graph").unwrap().parameter_count(),
            6 + 3 + 6 + 2
        );
    }
}
