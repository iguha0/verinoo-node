//! Deterministic GPT-style decoder with a BPE tokenizer and KV cache.
//!
//! This is a small demonstration decoder, not a production LLM. It executes a
//! two-layer causal transformer over a learned char-BPE vocabulary, generates
//! with greedy argmax, and caches keys/values so each new token is one forward
//! step instead of a full recomputation.
//!
//! Every op is integer or Q16.16. The tokenizer, causal mask, KV cache and
//! greedy loop are all deterministic, so two nodes derive the same generated
//! token ids from the same prompt and weights.

use crate::attention::softmax_exp;
use crate::sha256_hex;
use crate::transformer::{add_in_place, argmax_index, dense, dense_rows, isqrt, layer_norm};
use crate::ONE;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

const MAX_HIDDEN: usize = 2048;
const MAX_CONTEXT: usize = 8192;
const MAX_VOCAB: usize = 1_000_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GptLayer {
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
pub struct GptSpec {
    #[serde(default)]
    pub kind: String,
    pub model_id: String,
    pub architecture: String,
    #[serde(default)]
    pub description: String,
    pub vocab: Vec<String>,
    pub merges: Vec<[String; 2]>,
    pub max_context: usize,
    pub max_new_tokens: usize,
    pub hidden: usize,
    pub heads: usize,
    pub ff: usize,
    pub layers: usize,
    pub bos_id: usize,
    pub eos_id: usize,
    pub pad_id: usize,
    pub unk_id: usize,
    pub embedding: Vec<i32>,
    pub positional: Vec<i32>,
    pub decoder: Vec<GptLayer>,
    pub lnf_gamma: Vec<i32>,
    pub lnf_beta: Vec<i32>,
    pub head_weights: Vec<i32>,
    pub head_bias: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct Generated {
    pub prompt: String,
    pub prompt_ids: Vec<usize>,
    pub generated_ids: Vec<usize>,
    pub text: String,
}

#[derive(Debug, Clone)]
struct BpeTokenizer {
    vocab: Vec<String>,
    token_to_id: BTreeMap<String, usize>,
    merges: Vec<(String, String)>,
    bos_id: usize,
    eos_id: usize,
    pad_id: usize,
    unk_id: usize,
}

impl BpeTokenizer {
    fn new(spec: &GptSpec) -> Result<Self, String> {
        let mut token_to_id = BTreeMap::new();
        for (i, token) in spec.vocab.iter().enumerate() {
            if token_to_id.insert(token.clone(), i).is_some() {
                return Err(format!("duplicate vocabulary token `{token}`"));
            }
        }
        if spec.vocab.get(spec.bos_id).map(String::as_str) != Some("<bos>") {
            return Err("bosId must point at <bos>".into());
        }
        if spec.vocab.get(spec.eos_id).map(String::as_str) != Some("<eos>") {
            return Err("eosId must point at <eos>".into());
        }
        if spec.vocab.get(spec.pad_id).map(String::as_str) != Some("<pad>") {
            return Err("padId must point at <pad>".into());
        }
        if spec.vocab.get(spec.unk_id).map(String::as_str) != Some("<unk>") {
            return Err("unkId must point at <unk>".into());
        }
        let mut merges = Vec::with_capacity(spec.merges.len());
        for pair in &spec.merges {
            if pair[0].is_empty() || pair[1].is_empty() {
                return Err("BPE merge contains an empty token".into());
            }
            merges.push((pair[0].clone(), pair[1].clone()));
        }
        Ok(Self {
            vocab: spec.vocab.clone(),
            token_to_id,
            merges,
            bos_id: spec.bos_id,
            eos_id: spec.eos_id,
            pad_id: spec.pad_id,
            unk_id: spec.unk_id,
        })
    }

    /// Ordered BPE merge application, identical to scripts/train_gpt.py.
    fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        for (a, b) in &self.merges {
            let new = format!("{a}{b}");
            let mut out = Vec::with_capacity(tokens.len());
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && &tokens[i] == a && &tokens[i + 1] == b {
                    out.push(new.clone());
                    i += 2;
                } else {
                    out.push(tokens[i].clone());
                    i += 1;
                }
            }
            tokens = out;
        }
        tokens
            .iter()
            .map(|token| *self.token_to_id.get(token).unwrap_or(&self.unk_id))
            .collect()
    }

    fn decode(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        for id in ids {
            if *id == self.pad_id || *id == self.bos_id || *id == self.eos_id || *id == self.unk_id
            {
                continue;
            }
            if let Some(token) = self.vocab.get(*id) {
                out.push_str(token);
            }
        }
        out
    }
}

#[derive(Debug, Clone, Default)]
struct LayerCache {
    k: Vec<i32>,
    v: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct KVCache {
    layers: Vec<LayerCache>,
    hidden: usize,
}

impl KVCache {
    pub fn new(layers: usize, hidden: usize) -> Self {
        Self {
            layers: vec![LayerCache::default(); layers],
            hidden,
        }
    }

    fn len(&self) -> usize {
        self.layers
            .first()
            .map(|layer| layer.k.len() / self.hidden)
            .unwrap_or(0)
    }

    fn append(&mut self, layer: usize, k: &[i32], v: &[i32]) -> Result<(), String> {
        let cache = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| format!("cache layer {layer} does not exist"))?;
        if k.len() != v.len() || !k.len().is_multiple_of(self.hidden) {
            return Err("cache append shape mismatch".into());
        }
        cache.k.extend_from_slice(k);
        cache.v.extend_from_slice(v);
        Ok(())
    }

    fn k(&self, layer: usize) -> &[i32] {
        &self.layers[layer].k
    }

    fn v(&self, layer: usize) -> &[i32] {
        &self.layers[layer].v
    }
}

#[derive(Debug, Clone)]
pub struct GptModel {
    spec: GptSpec,
    weights_hash: String,
    tokenizer: BpeTokenizer,
    head_dim: usize,
}

impl GptModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let value: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| format!("invalid JSON: {e}"))?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &serde_json::Value) -> Result<Self, String> {
        let spec: GptSpec =
            serde_json::from_value(value.clone()).map_err(|e| format!("invalid decoder: {e}"))?;
        validate_spec(&spec)?;
        let tokenizer = BpeTokenizer::new(&spec)?;
        let weights_hash = sha256_hex(
            &serde_json::to_string(&spec).map_err(|e| format!("cannot hash model: {e}"))?,
        );
        Ok(Self {
            head_dim: spec.hidden / spec.heads,
            spec,
            weights_hash,
            tokenizer,
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

    pub fn weights_hash(&self) -> &str {
        &self.weights_hash
    }

    pub fn parameter_count(&self) -> u64 {
        parameter_count(&self.spec)
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.tokenizer.encode(text)
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        self.tokenizer.decode(ids)
    }

    /// Full-sequence forward pass, used by tests and verification. Generation
    /// uses the KV cache path instead.
    pub fn forward_logits(&self, ids: &[usize]) -> Result<Vec<i32>, String> {
        let mut cache = KVCache::new(self.spec.layers, self.spec.hidden);
        self.forward_sequence(ids, &mut cache)
    }

    fn forward_sequence(&self, ids: &[usize], cache: &mut KVCache) -> Result<Vec<i32>, String> {
        if ids.is_empty() {
            return Err("empty token sequence".into());
        }
        if ids.iter().any(|id| *id >= self.spec.vocab.len()) {
            return Err("token id outside the vocabulary".into());
        }
        let past = cache.len();
        if ids.len() + past > self.spec.max_context {
            return Err(format!(
                "sequence length {} exceeds maxContext {} (past {past})",
                ids.len(),
                self.spec.max_context
            ));
        }
        let hidden = self.spec.hidden;
        let rows = ids.len();
        let mut x = vec![0i32; rows * hidden];
        for (offset, token) in ids.iter().enumerate() {
            let position = past + offset;
            let emb = &self.spec.embedding[token * hidden..(token + 1) * hidden];
            let pos = &self.spec.positional[position * hidden..(position + 1) * hidden];
            for j in 0..hidden {
                x[offset * hidden + j] = emb[j].saturating_add(pos[j]);
            }
        }

        for (layer_index, layer) in self.spec.decoder.iter().enumerate() {
            let norm = layer_norm(&x, &layer.ln1_gamma, &layer.ln1_beta, rows, hidden);
            let q = dense_rows(&norm, &layer.q_weights, &layer.q_bias, hidden, hidden, rows)?;
            let k = dense_rows(&norm, &layer.k_weights, &layer.k_bias, hidden, hidden, rows)?;
            let v = dense_rows(&norm, &layer.v_weights, &layer.v_bias, hidden, hidden, rows)?;
            cache.append(layer_index, &k, &v)?;
            let attn = causal_attention(
                &q,
                cache.k(layer_index),
                cache.v(layer_index),
                rows,
                past,
                self.spec.heads,
                self.head_dim,
            )?;
            let projected =
                dense_rows(&attn, &layer.o_weights, &layer.o_bias, hidden, hidden, rows)?;
            add_in_place(&mut x, &projected);

            let norm = layer_norm(&x, &layer.ln2_gamma, &layer.ln2_beta, rows, hidden);
            let mut ff = dense_rows(
                &norm,
                &layer.ff1_weights,
                &layer.ff1_bias,
                hidden,
                self.spec.ff,
                rows,
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
                rows,
            )?;
            add_in_place(&mut x, &ff);
        }

        let x = layer_norm(&x, &self.spec.lnf_gamma, &self.spec.lnf_beta, rows, hidden);
        let last = &x[(rows - 1) * hidden..rows * hidden];
        dense(
            last,
            &self.spec.head_weights,
            &self.spec.head_bias,
            hidden,
            self.spec.vocab.len(),
        )
    }

    pub fn generate(&self, prompt: &str) -> Result<Generated, String> {
        let mut prompt_ids = self.tokenizer.encode(prompt);
        let max_prompt = self.spec.max_context.saturating_sub(1);
        if prompt_ids.len() > max_prompt {
            prompt_ids = prompt_ids[prompt_ids.len() - max_prompt..].to_vec();
        }
        let mut ids = Vec::with_capacity(self.spec.max_context);
        ids.push(self.spec.bos_id);
        ids.extend_from_slice(&prompt_ids);

        let mut cache = KVCache::new(self.spec.layers, self.spec.hidden);
        let mut logits = self.forward_sequence(&ids, &mut cache)?;
        let mut generated = Vec::new();
        for _ in 0..self.spec.max_new_tokens {
            if ids.len() >= self.spec.max_context {
                break;
            }
            let next = argmax_index(&logits).ok_or_else(|| "empty logits".to_string())?;
            ids.push(next);
            generated.push(next);
            if next == self.spec.eos_id {
                break;
            }
            logits = self.forward_sequence(&[next], &mut cache)?;
        }
        let text = self.tokenizer.decode(&generated);
        Ok(Generated {
            prompt: prompt.to_string(),
            prompt_ids,
            generated_ids: generated,
            text,
        })
    }

    pub fn result_json(&self, prompt: &str) -> Result<String, String> {
        let generated = self.generate(prompt)?;
        Ok(serde_json::json!({
            "generatedText": generated.text,
            "prompt": generated.prompt,
            "promptTokens": generated.prompt_ids,
            "tokenIds": generated.generated_ids,
            "modelId": self.spec.model_id,
            "weightsHash": self.weights_hash,
        })
        .to_string())
    }

    pub fn descriptor(&self) -> serde_json::Value {
        serde_json::json!({
            "modelId": self.spec.model_id,
            "architecture": self.spec.architecture,
            "description": self.spec.description,
            "vocabSize": self.spec.vocab.len(),
            "maxContext": self.spec.max_context,
            "maxNewTokens": self.spec.max_new_tokens,
            "hidden": self.spec.hidden,
            "heads": self.spec.heads,
            "layers": self.spec.layers,
            "parameterCount": self.parameter_count(),
            "runtime": "q16.16-gpt-decoder",
            "weightsHash": self.weights_hash,
        })
    }
}

fn causal_attention(
    q: &[i32],
    k: &[i32],
    v: &[i32],
    q_len: usize,
    past: usize,
    heads: usize,
    head_dim: usize,
) -> Result<Vec<i32>, String> {
    let hidden = heads
        .checked_mul(head_dim)
        .ok_or_else(|| "heads * head_dim overflows".to_string())?;
    let kv_len = past
        .checked_add(q_len)
        .ok_or_else(|| "kv length overflows".to_string())?;
    if q.len() != q_len * hidden {
        return Err(format!(
            "q length: expected {}, got {}",
            q_len * hidden,
            q.len()
        ));
    }
    if k.len() != kv_len * hidden || v.len() != kv_len * hidden {
        return Err(format!(
            "k/v length: expected {}, got {} and {}",
            kv_len * hidden,
            k.len(),
            v.len()
        ));
    }
    let scale = isqrt((ONE as u64 * ONE as u64) / head_dim as u64) as i64;
    let mut out = vec![0i32; q_len * hidden];
    for h in 0..heads {
        let offset = h * head_dim;
        for i in 0..q_len {
            let absolute = past + i;
            let mut scores = Vec::with_capacity(absolute + 1);
            for j in 0..=absolute {
                let mut acc = 0i64;
                for d in 0..head_dim {
                    acc = acc.saturating_add(
                        q[i * hidden + offset + d] as i64 * k[j * hidden + offset + d] as i64,
                    );
                }
                let dot = acc >> 16;
                let scaled = ((dot as i128 * scale as i128) >> 16)
                    .clamp(i32::MIN as i128, i32::MAX as i128) as i32;
                scores.push(scaled);
            }
            let weights = softmax_exp(&scores);
            for d in 0..head_dim {
                let mut acc = 0i64;
                for (j, weight) in weights.iter().enumerate() {
                    acc = acc.saturating_add(*weight as i64 * v[j * hidden + offset + d] as i64);
                }
                out[i * hidden + offset + d] = (acc >> 16) as i32;
            }
        }
    }
    Ok(out)
}

fn validate_spec(spec: &GptSpec) -> Result<(), String> {
    if spec.kind != "gpt-decoder" {
        return Err(format!("kind must be `gpt-decoder`, got `{}`", spec.kind));
    }
    if spec.model_id.trim().is_empty() {
        return Err("modelId must not be empty".into());
    }
    if spec.vocab.is_empty() || spec.vocab.len() > MAX_VOCAB {
        return Err("vocab size is invalid".into());
    }
    if spec.hidden == 0 || spec.hidden > MAX_HIDDEN {
        return Err(format!("hidden must be between 1 and {MAX_HIDDEN}"));
    }
    if spec.heads == 0 || spec.heads > spec.hidden || !spec.hidden.is_multiple_of(spec.heads) {
        return Err("heads must divide hidden".into());
    }
    if spec.max_context == 0 || spec.max_context > MAX_CONTEXT {
        return Err(format!("maxContext must be between 1 and {MAX_CONTEXT}"));
    }
    if spec.max_new_tokens == 0 || spec.max_new_tokens > spec.max_context {
        return Err("maxNewTokens must be between 1 and maxContext".into());
    }
    if spec.ff == 0 || spec.ff > MAX_HIDDEN * 8 {
        return Err("ff is out of range".into());
    }
    if spec.layers == 0 || spec.layers != spec.decoder.len() {
        return Err("layers must equal the number of decoder blocks".into());
    }
    let h = spec.hidden;
    let ff = spec.ff;
    expect_len("embedding", spec.embedding.len(), spec.vocab.len() * h)?;
    expect_len("positional", spec.positional.len(), spec.max_context * h)?;
    expect_len("lnfGamma", spec.lnf_gamma.len(), h)?;
    expect_len("lnfBeta", spec.lnf_beta.len(), h)?;
    expect_len("headWeights", spec.head_weights.len(), spec.vocab.len() * h)?;
    expect_len("headBias", spec.head_bias.len(), spec.vocab.len())?;
    let matrix = h * h;
    for (i, layer) in spec.decoder.iter().enumerate() {
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

fn parameter_count(spec: &GptSpec) -> u64 {
    let mut total = (spec.embedding.len()
        + spec.positional.len()
        + spec.lnf_gamma.len()
        + spec.lnf_beta.len()
        + spec.head_weights.len()
        + spec.head_bias.len()) as u64;
    for layer in &spec.decoder {
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

#[derive(Debug, Default)]
pub struct GptRegistry {
    models: BTreeMap<String, Arc<GptModel>>,
}

impl GptRegistry {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Load every `*.json` file with `kind: gpt-decoder`.
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
            let is_decoder = serde_json::from_slice::<serde_json::Value>(&bytes)
                .ok()
                .and_then(|v| v.get("kind").and_then(|k| k.as_str()).map(String::from))
                .map(|kind| kind == "gpt-decoder")
                .unwrap_or(false);
            if !is_decoder {
                continue;
            }
            match GptModel::from_bytes(&bytes) {
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

    pub fn insert(&mut self, model: GptModel) -> Result<(), String> {
        let id = model.model_id().to_string();
        if self.models.contains_key(&id) {
            return Err(format!("duplicate model id `{id}`"));
        }
        self.models.insert(id, Arc::new(model));
        Ok(())
    }

    pub fn get(&self, model_id: &str) -> Option<&Arc<GptModel>> {
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

static DECODERS: OnceLock<GptRegistry> = OnceLock::new();

pub fn gpt_global() -> &'static GptRegistry {
    DECODERS.get_or_init(GptRegistry::empty)
}

pub fn install_gpt_once(registry: GptRegistry) {
    let _ = DECODERS.set(registry);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt-v1.json")
    }

    fn fixture_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt-v1.test.json")
    }

    fn load() -> GptModel {
        let bytes = std::fs::read(model_path()).expect("committed decoder model");
        GptModel::from_bytes(&bytes).expect("valid decoder")
    }

    #[test]
    fn bpe_matches_the_training_script() {
        let model = load();
        let text = "please cancel order 2064";
        let ids = model.encode(text);
        assert_eq!(model.decode(&ids), text);
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(fixture_path()).expect("fixture"))
                .expect("fixture JSON");
        let case = &fixture["cases"][0];
        let prompt = case["prompt"].as_str().unwrap();
        let expected: Vec<usize> = case["promptTokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(model.encode(prompt), expected);
    }

    #[test]
    fn kv_cache_matches_full_forward() {
        let model = load();
        let ids = model.encode("please cancel order 2064");
        let full = model.forward_logits(&ids).unwrap();
        let mut cache = KVCache::new(model.spec.layers, model.spec.hidden);
        let prefill = model.forward_sequence(&ids, &mut cache).unwrap();
        assert_eq!(prefill, full);
        let next = argmax_index(&full).unwrap();
        let mut extended = ids.clone();
        extended.push(next);
        let full_next = model.forward_logits(&extended).unwrap();
        let step = model.forward_sequence(&[next], &mut cache).unwrap();
        assert_eq!(step, full_next);
    }

    #[test]
    fn greedy_generation_matches_the_float_reference() {
        let model = load();
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(fixture_path()).expect("fixture"))
                .expect("fixture JSON");
        let cases = fixture["cases"].as_array().expect("cases");
        let mut mismatches = Vec::new();
        for case in cases {
            let prompt = case["prompt"].as_str().unwrap();
            let expected_ids: Vec<usize> = case["tokenIds"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let generated = model.generate(prompt).unwrap();
            if generated.generated_ids != expected_ids {
                mismatches.push(format!(
                    "{prompt}: got {:?}, expected {expected_ids:?}",
                    generated.generated_ids
                ));
            }
        }
        assert!(
            mismatches.is_empty(),
            "token parity failed on {} cases: {:?}",
            mismatches.len(),
            mismatches
        );
    }

    #[test]
    fn generation_is_deterministic() {
        let model = load();
        let a = model.generate("please cancel").unwrap();
        let b = model.generate("please cancel").unwrap();
        assert_eq!(a.generated_ids, b.generated_ids);
        assert_eq!(a.text, b.text);
        assert!(!a.generated_ids.is_empty());
    }

    #[test]
    fn malformed_decoders_are_refused() {
        let bytes = std::fs::read(model_path()).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        value["kind"] = serde_json::json!("transformer-classifier");
        assert!(GptModel::from_value(&value).is_err());
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        value["headWeights"] = serde_json::json!([]);
        assert!(GptModel::from_value(&value).is_err());
    }

    #[test]
    fn descriptor_and_registry_are_stable() {
        let model = load();
        let descriptor = model.descriptor();
        assert_eq!(descriptor["runtime"], "q16.16-gpt-decoder");
        assert!(model.parameter_count() > 50_000);
        let mut registry = GptRegistry::empty();
        registry.insert(model).unwrap();
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.hash().len(), 64);
    }
}
