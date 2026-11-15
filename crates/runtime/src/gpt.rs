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

use crate::attention::{exp_neg, softmax_exp};
use crate::sha256_hex;
use crate::transformer::{
    add_in_place, argmax_index, dense_accurate, dense_rows_accurate, isqrt, layer_norm_eps,
};
use crate::weights::WeightsData;
use crate::ONE;
use fancy_regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, OnceLock};

const MAX_HIDDEN: usize = 2048;
const MAX_CONTEXT: usize = 8192;
const MAX_VOCAB: usize = 1_000_000;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GptLayer {
    #[serde(default)]
    pub ln1_gamma: Vec<i32>,
    #[serde(default)]
    pub ln1_beta: Vec<i32>,
    #[serde(default)]
    pub q_weights: Vec<i32>,
    #[serde(default)]
    pub q_bias: Vec<i32>,
    #[serde(default)]
    pub k_weights: Vec<i32>,
    #[serde(default)]
    pub k_bias: Vec<i32>,
    #[serde(default)]
    pub v_weights: Vec<i32>,
    #[serde(default)]
    pub v_bias: Vec<i32>,
    #[serde(default)]
    pub o_weights: Vec<i32>,
    #[serde(default)]
    pub o_bias: Vec<i32>,
    #[serde(default)]
    pub ln2_gamma: Vec<i32>,
    #[serde(default)]
    pub ln2_beta: Vec<i32>,
    #[serde(default)]
    pub ff1_weights: Vec<i32>,
    #[serde(default)]
    pub ff1_bias: Vec<i32>,
    #[serde(default)]
    pub ff2_weights: Vec<i32>,
    #[serde(default)]
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
    #[serde(default = "default_tokenizer_kind")]
    pub tokenizer_kind: String,
    #[serde(default = "default_activation")]
    pub activation: String,
    #[serde(default = "default_attention_scale")]
    pub attention_scale: String,
    /// Q32.32 epsilon added to the layer-norm variance.
    #[serde(default)]
    pub layer_norm_epsilon: i64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Binary weight file, relative to this model file. When set, the weight
    /// arrays below must be empty; the loader fills them from the binary.
    #[serde(default)]
    pub weights_file: String,
    /// SHA-256 of the binary weight payload. Required when weightsFile is set.
    #[serde(default)]
    pub weights_sha256: String,
    #[serde(default = "default_prepend_bos")]
    pub prepend_bos: bool,
    #[serde(default)]
    pub sampling: Sampling,
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
    #[serde(default)]
    pub embedding: Vec<i32>,
    #[serde(default)]
    pub positional: Vec<i32>,
    #[serde(default)]
    pub decoder: Vec<GptLayer>,
    #[serde(default)]
    pub lnf_gamma: Vec<i32>,
    #[serde(default)]
    pub lnf_beta: Vec<i32>,
    #[serde(default)]
    pub head_weights: Vec<i32>,
    #[serde(default)]
    pub head_bias: Vec<i32>,
}

fn default_tokenizer_kind() -> String {
    "char-bpe".into()
}

fn default_activation() -> String {
    "relu".into()
}

fn default_attention_scale() -> String {
    "sqrt".into()
}

fn default_prepend_bos() -> bool {
    true
}

/// Deterministic sampling parameters. `temperatureMilli: 0` means greedy.
/// `topK: 0` and `topPMilli: 0` disable those filters. `seed` makes the
/// sequence reproducible across nodes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Sampling {
    #[serde(default)]
    pub temperature_milli: u32,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub top_p_milli: u32,
    #[serde(default)]
    pub seed: u64,
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

#[derive(Debug, Clone)]
enum Tokenizer {
    Char(Box<BpeTokenizer>),
    Byte(Box<ByteBpeTokenizer>),
}

impl Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
        match self {
            Tokenizer::Char(tokenizer) => Ok(tokenizer.encode(text)),
            Tokenizer::Byte(tokenizer) => tokenizer.encode(text),
        }
    }

    fn decode(&self, ids: &[usize]) -> String {
        match self {
            Tokenizer::Char(tokenizer) => tokenizer.decode(ids),
            Tokenizer::Byte(tokenizer) => tokenizer.decode(ids),
        }
    }
}

/// GPT-2 byte-level BPE. The vocabulary and merge ranks come from the
/// checkpoint's `vocab.json` and `merges.txt`; the pre-tokenizer is the
/// standard GPT-2 ByteLevel regex.
#[derive(Debug, Clone)]
struct ByteBpeTokenizer {
    vocab: Vec<String>,
    token_to_id: BTreeMap<String, usize>,
    ranks: BTreeMap<(String, String), usize>,
    byte_to_unicode: [String; 256],
    unicode_to_byte: BTreeMap<char, u8>,
    bos_id: usize,
    eos_id: usize,
    pad_id: usize,
    unk_id: usize,
}

static GPT2_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        .expect("GPT-2 tokenizer regex")
});

impl ByteBpeTokenizer {
    fn new(spec: &GptSpec) -> Result<Self, String> {
        let mut token_to_id = BTreeMap::new();
        for (i, token) in spec.vocab.iter().enumerate() {
            if token_to_id.insert(token.clone(), i).is_some() {
                return Err(format!("duplicate vocabulary token `{token}`"));
            }
        }
        for (name, id) in [
            ("bosId", spec.bos_id),
            ("eosId", spec.eos_id),
            ("padId", spec.pad_id),
            ("unkId", spec.unk_id),
        ] {
            if id >= spec.vocab.len() {
                return Err(format!("{name} is outside the vocabulary"));
            }
        }
        let mut ranks = BTreeMap::new();
        for (i, pair) in spec.merges.iter().enumerate() {
            if pair[0].is_empty() || pair[1].is_empty() {
                return Err("BPE merge contains an empty token".into());
            }
            if ranks
                .insert((pair[0].clone(), pair[1].clone()), i)
                .is_some()
            {
                return Err("duplicate BPE merge".into());
            }
        }
        let (byte_to_unicode, unicode_to_byte) = bytes_to_unicode();
        Ok(Self {
            vocab: spec.vocab.clone(),
            token_to_id,
            ranks,
            byte_to_unicode,
            unicode_to_byte,
            bos_id: spec.bos_id,
            eos_id: spec.eos_id,
            pad_id: spec.pad_id,
            unk_id: spec.unk_id,
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
        let mut out = Vec::new();
        for found in GPT2_RE.find_iter(text) {
            let matched = found.map_err(|e| format!("tokenizer regex: {e}"))?;
            out.extend(self.encode_piece(matched.as_str())?);
        }
        Ok(out)
    }

    fn encode_piece(&self, piece: &str) -> Result<Vec<usize>, String> {
        let mapped: String = piece
            .bytes()
            .map(|byte| self.byte_to_unicode[byte as usize].as_str())
            .collect();
        let mut parts: Vec<String> = mapped.chars().map(|ch| ch.to_string()).collect();
        loop {
            let mut best: Option<(usize, (String, String))> = None;
            for i in 0..parts.len().saturating_sub(1) {
                let pair = (parts[i].clone(), parts[i + 1].clone());
                if let Some(rank) = self.ranks.get(&pair) {
                    if best.as_ref().is_none_or(|(best_rank, _)| rank < best_rank) {
                        best = Some((*rank, pair));
                    }
                }
            }
            let Some((_, (a, b))) = best else {
                break;
            };
            let mut merged = Vec::with_capacity(parts.len());
            let mut i = 0;
            while i < parts.len() {
                if i + 1 < parts.len() && parts[i] == a && parts[i + 1] == b {
                    merged.push(format!("{a}{b}"));
                    i += 2;
                } else {
                    merged.push(parts[i].clone());
                    i += 1;
                }
            }
            parts = merged;
        }
        parts
            .iter()
            .map(|part| {
                self.token_to_id
                    .get(part)
                    .copied()
                    .ok_or_else(|| format!("BPE token `{part}` is missing from the vocabulary"))
            })
            .collect()
    }

    fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            if *id == self.pad_id || *id == self.bos_id || *id == self.eos_id || *id == self.unk_id
            {
                continue;
            }
            if let Some(token) = self.vocab.get(*id) {
                for ch in token.chars() {
                    if let Some(byte) = self.unicode_to_byte.get(&ch) {
                        bytes.push(*byte);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn bytes_to_unicode() -> ([String; 256], BTreeMap<char, u8>) {
    let mut bs: Vec<u8> = Vec::new();
    bs.extend(b'!'..=b'~');
    bs.extend(0xA1..=0xAC);
    bs.extend(0xAE..=0xFF);
    let mut cs: Vec<u32> = bs.iter().map(|b| *b as u32).collect();
    let mut n = 0u32;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut table: [String; 256] = std::array::from_fn(|_| String::new());
    let mut inverse = BTreeMap::new();
    for (byte, code) in bs.iter().zip(cs) {
        let ch = char::from_u32(code).expect("valid byte-level code point");
        table[*byte as usize] = ch.to_string();
        inverse.insert(ch, *byte);
    }
    (table, inverse)
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

#[derive(Debug, Clone, Copy)]
struct TensorRange {
    offset: usize,
    len: usize,
}

impl TensorRange {
    fn index(self) -> std::ops::Range<usize> {
        self.offset..self.offset + self.len
    }
}

#[derive(Debug, Clone)]
struct LayerLayout {
    ln1_gamma: TensorRange,
    ln1_beta: TensorRange,
    q_weights: TensorRange,
    q_bias: TensorRange,
    k_weights: TensorRange,
    k_bias: TensorRange,
    v_weights: TensorRange,
    v_bias: TensorRange,
    o_weights: TensorRange,
    o_bias: TensorRange,
    ln2_gamma: TensorRange,
    ln2_beta: TensorRange,
    ff1_weights: TensorRange,
    ff1_bias: TensorRange,
    ff2_weights: TensorRange,
    ff2_bias: TensorRange,
}

#[derive(Debug, Clone)]
struct WeightLayout {
    embedding: TensorRange,
    positional: TensorRange,
    decoder: Vec<LayerLayout>,
    lnf_gamma: TensorRange,
    lnf_beta: TensorRange,
    head_weights: TensorRange,
    head_bias: TensorRange,
    total: usize,
}

#[derive(Debug, Clone)]
pub struct GptModel {
    spec: GptSpec,
    weights: WeightsData,
    layout: WeightLayout,
    weights_hash: String,
    tokenizer: Tokenizer,
    head_dim: usize,
}

impl GptModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let value: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| format!("invalid JSON: {e}"))?;
        Self::from_value(&value)
    }

    /// Load a decoder from its JSON metadata. When `weightsFile` is set, the
    /// binary file is resolved relative to the JSON file, mapped, and its
    /// payload is verified against `weightsSha256` before the model is
    /// constructed.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("invalid JSON: {e}"))?;
        let mut spec: GptSpec =
            serde_json::from_value(value).map_err(|e| format!("invalid decoder: {e}"))?;
        if spec.weights_file.is_empty() {
            return Self::from_owned_spec(spec);
        }
        if !spec.embedding.is_empty()
            || !spec.positional.is_empty()
            || spec.decoder.iter().any(|layer| !layer.q_weights.is_empty())
        {
            return Err("weightsFile is set but the JSON also contains weight arrays".into());
        }
        let metadata =
            serde_json::to_string(&spec).map_err(|e| format!("cannot hash model metadata: {e}"))?;
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        let weights_path = dir.join(&spec.weights_file);
        let loaded = crate::weights::load(&weights_path)?;
        if !spec.weights_sha256.is_empty() && spec.weights_sha256 != loaded.sha256 {
            return Err(format!(
                "{}: weightsSha256 does not match the binary payload",
                path.display()
            ));
        }
        if spec.decoder.is_empty() && spec.layers > 0 {
            spec.decoder = vec![GptLayer::default(); spec.layers];
        }
        validate_metadata(&spec)?;
        let layout = WeightLayout::new(&spec)?;
        if layout.total > loaded.weights.len() {
            return Err(format!(
                "binary is missing {} weight(s) from the canonical tensor order",
                layout.total - loaded.weights.len()
            ));
        }
        if layout.total < loaded.weights.len() {
            return Err(format!(
                "binary has {} trailing weight(s) after the canonical tensor order",
                loaded.weights.len() - layout.total
            ));
        }
        let identity = format!("{metadata}:{}", loaded.sha256);
        Self::build(spec, loaded.weights, layout, sha256_hex(&identity))
    }

    pub fn from_value(value: &serde_json::Value) -> Result<Self, String> {
        let spec: GptSpec =
            serde_json::from_value(value.clone()).map_err(|e| format!("invalid decoder: {e}"))?;
        if !spec.weights_file.is_empty() {
            return Err("model has weightsFile; load it with from_file".into());
        }
        Self::from_owned_spec(spec)
    }

    fn from_owned_spec(mut spec: GptSpec) -> Result<Self, String> {
        validate_spec(&spec)?;
        let layout = WeightLayout::new(&spec)?;
        let weights_hash = sha256_hex(
            &serde_json::to_string(&spec).map_err(|e| format!("cannot hash model: {e}"))?,
        );
        let weights = WeightsData::owned(flatten_inline_weights(&spec, &layout));
        clear_spec_weights(&mut spec);
        Self::build(spec, weights, layout, weights_hash)
    }

    fn build(
        spec: GptSpec,
        weights: WeightsData,
        layout: WeightLayout,
        weights_hash: String,
    ) -> Result<Self, String> {
        let tokenizer = match spec.tokenizer_kind.as_str() {
            "char-bpe" => Tokenizer::Char(Box::new(BpeTokenizer::new(&spec)?)),
            "byte-bpe" => Tokenizer::Byte(Box::new(ByteBpeTokenizer::new(&spec)?)),
            other => return Err(format!("unknown tokenizerKind `{other}`")),
        };
        Ok(Self {
            head_dim: spec.hidden / spec.heads,
            spec,
            weights,
            layout,
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

    /// True when the parameters are a read-only mapping of a `.vnnw` payload.
    pub fn is_mapped(&self) -> bool {
        self.weights.is_mapped()
    }

    pub fn parameter_count(&self) -> u64 {
        self.layout.total as u64
    }

    fn tensor(&self, range: TensorRange) -> &[i32] {
        self.weights.tensor(range.offset, range.len)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
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
        let embedding = self.tensor(self.layout.embedding);
        let positional = self.tensor(self.layout.positional);
        for (offset, token) in ids.iter().enumerate() {
            let position = past + offset;
            let emb = &embedding[token * hidden..(token + 1) * hidden];
            let pos = &positional[position * hidden..(position + 1) * hidden];
            for j in 0..hidden {
                x[offset * hidden + j] = emb[j].saturating_add(pos[j]);
            }
        }

        for (layer_index, layer) in self.layout.decoder.iter().enumerate() {
            let norm = layer_norm_eps(
                &x,
                self.tensor(layer.ln1_gamma),
                self.tensor(layer.ln1_beta),
                rows,
                hidden,
                self.spec.layer_norm_epsilon,
            );
            let q = dense_rows_accurate(
                &norm,
                self.tensor(layer.q_weights),
                self.tensor(layer.q_bias),
                hidden,
                hidden,
                rows,
            )?;
            let k = dense_rows_accurate(
                &norm,
                self.tensor(layer.k_weights),
                self.tensor(layer.k_bias),
                hidden,
                hidden,
                rows,
            )?;
            let v = dense_rows_accurate(
                &norm,
                self.tensor(layer.v_weights),
                self.tensor(layer.v_bias),
                hidden,
                hidden,
                rows,
            )?;
            cache.append(layer_index, &k, &v)?;
            let attn = causal_attention(
                &q,
                cache.k(layer_index),
                cache.v(layer_index),
                rows,
                past,
                self.spec.heads,
                self.head_dim,
                self.spec.attention_scale == "sqrt",
            )?;
            let projected = dense_rows_accurate(
                &attn,
                self.tensor(layer.o_weights),
                self.tensor(layer.o_bias),
                hidden,
                hidden,
                rows,
            )?;
            add_in_place(&mut x, &projected);

            let norm = layer_norm_eps(
                &x,
                self.tensor(layer.ln2_gamma),
                self.tensor(layer.ln2_beta),
                rows,
                hidden,
                self.spec.layer_norm_epsilon,
            );
            let mut ff = dense_rows_accurate(
                &norm,
                self.tensor(layer.ff1_weights),
                self.tensor(layer.ff1_bias),
                hidden,
                self.spec.ff,
                rows,
            )?;
            for value in ff.iter_mut() {
                *value = self.activate(*value);
            }
            let ff = dense_rows_accurate(
                &ff,
                self.tensor(layer.ff2_weights),
                self.tensor(layer.ff2_bias),
                self.spec.ff,
                hidden,
                rows,
            )?;
            add_in_place(&mut x, &ff);
        }

        let x = layer_norm_eps(
            &x,
            self.tensor(self.layout.lnf_gamma),
            self.tensor(self.layout.lnf_beta),
            rows,
            hidden,
            self.spec.layer_norm_epsilon,
        );
        let last = &x[(rows - 1) * hidden..rows * hidden];
        let (head_weights, head_bias): (&[i32], &[i32]) = if self.spec.tie_word_embeddings {
            (self.tensor(self.layout.embedding), &[])
        } else {
            (
                self.tensor(self.layout.head_weights),
                self.tensor(self.layout.head_bias),
            )
        };
        dense_accurate(last, head_weights, head_bias, hidden, self.spec.vocab.len())
    }

    fn activate(&self, value: i32) -> i32 {
        match self.spec.activation.as_str() {
            "gelu_new" => gelu_new(value as i64).clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            _ => value.max(0),
        }
    }

    pub fn generate(&self, prompt: &str) -> Result<Generated, String> {
        self.generate_with_sampling(prompt, &Sampling::default())
    }

    /// Generate with explicit sampling. `temperature_milli: 0` is greedy.
    /// The RNG is seeded by `sampling.seed`, so the same prompt, weights and
    /// seed produce the same token ids on every node.
    pub fn generate_with_sampling(
        &self,
        prompt: &str,
        sampling: &Sampling,
    ) -> Result<Generated, String> {
        let mut prompt_ids = self.tokenizer.encode(prompt)?;
        let reserve = if self.spec.prepend_bos { 1 } else { 0 };
        let max_prompt = self.spec.max_context.saturating_sub(reserve);
        if prompt_ids.len() > max_prompt {
            prompt_ids = prompt_ids[prompt_ids.len() - max_prompt..].to_vec();
        }
        let mut ids = Vec::with_capacity(self.spec.max_context);
        if self.spec.prepend_bos {
            ids.push(self.spec.bos_id);
        }
        ids.extend_from_slice(&prompt_ids);
        if ids.is_empty() {
            ids.push(self.spec.bos_id);
        }

        let mut cache = KVCache::new(self.spec.layers, self.spec.hidden);
        let mut logits = self.forward_sequence(&ids, &mut cache)?;
        let mut rng = SplitMix64::new(sampling.seed);
        let mut generated = Vec::new();
        for _ in 0..self.spec.max_new_tokens {
            if ids.len() >= self.spec.max_context {
                break;
            }
            let next = self
                .sample_token(&logits, sampling, &mut rng)
                .ok_or_else(|| "empty logits".to_string())?;
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

    fn sample_token(
        &self,
        logits: &[i32],
        sampling: &Sampling,
        rng: &mut SplitMix64,
    ) -> Option<usize> {
        if logits.is_empty() {
            return None;
        }
        if sampling.temperature_milli == 0 {
            return argmax_index(logits);
        }
        let temperature = sampling.temperature_milli.max(1) as i64;
        let scale = ((ONE as i64 * 1000) / temperature).max(1);
        let scaled: Vec<i32> = logits
            .iter()
            .map(|value| {
                ((*value as i64 * scale) >> 16).clamp(i32::MIN as i64, i32::MAX as i64) as i32
            })
            .collect();

        let mut indices: Vec<usize> = (0..scaled.len()).collect();
        if sampling.top_k > 0 && sampling.top_k < indices.len() {
            indices.sort_by_key(|index| std::cmp::Reverse(scaled[*index]));
            indices.truncate(sampling.top_k);
        }
        if sampling.top_p_milli > 0
            && sampling.top_p_milli < 1000
            && (sampling.top_k == 0 || sampling.top_k >= indices.len())
        {
            indices.sort_by_key(|index| std::cmp::Reverse(scaled[*index]));
        }
        let scores: Vec<i32> = indices.iter().map(|index| scaled[*index]).collect();
        let mut probs = softmax_exp(&scores);
        if sampling.top_p_milli > 0 && sampling.top_p_milli < 1000 {
            let threshold = (sampling.top_p_milli as i64 * ONE as i64) / 1000;
            let mut cumulative = 0i64;
            let mut keep = 0usize;
            for (i, probability) in probs.iter().enumerate() {
                cumulative += *probability as i64;
                keep = i + 1;
                if cumulative >= threshold {
                    break;
                }
            }
            indices.truncate(keep);
            probs.truncate(keep);
        }
        let total: u64 = probs.iter().map(|p| *p as u64).sum();
        if total == 0 {
            return indices.first().copied();
        }
        let mut pick = rng.next_u64() % total;
        for (index, probability) in indices.iter().zip(probs.iter()) {
            if pick < *probability as u64 {
                return Some(*index);
            }
            pick -= *probability as u64;
        }
        indices.last().copied()
    }

    pub fn result_json(&self, prompt: &str) -> Result<String, String> {
        let generated = self.generate_with_sampling(prompt, &self.spec.sampling)?;
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
            "sampling": self.spec.sampling,
            "parameterCount": self.parameter_count(),
            "runtime": "q16.16-gpt-decoder",
            "weightsHash": self.weights_hash,
        })
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// NewGELU, the tanh approximation used by GPT-2/GPT-Neo:
/// `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`.
fn gelu_new(x: i64) -> i64 {
    let x2 = qmul(x, x);
    let x3 = qmul(x2, x);
    let inner = qmul(52_293, x.saturating_add(qmul(2_931, x3)));
    let t = tanh_q(inner);
    qmul(x, (ONE as i64 + t) / 2)
}

fn tanh_q(x: i64) -> i64 {
    let x = x.clamp(-8 * ONE as i64, 8 * ONE as i64);
    let e = exp_q(x.saturating_mul(2));
    ((e - ONE as i64) * ONE as i64) / (e + ONE as i64)
}

fn exp_q(x: i64) -> i64 {
    if x >= 0 {
        let x = x.min(10 * ONE as i64);
        (ONE as i64 * ONE as i64) / exp_neg(x).max(1)
    } else {
        exp_neg(-x)
    }
}

fn qmul(a: i64, b: i64) -> i64 {
    ((a as i128 * b as i128) >> 16).clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

#[allow(clippy::too_many_arguments)]
fn causal_attention(
    q: &[i32],
    k: &[i32],
    v: &[i32],
    q_len: usize,
    past: usize,
    heads: usize,
    head_dim: usize,
    scale_attention: bool,
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
    let scale = if scale_attention {
        isqrt((ONE as u64 * ONE as u64) / head_dim as u64) as i64
    } else {
        ONE as i64
    };
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

impl WeightLayout {
    fn new(spec: &GptSpec) -> Result<Self, String> {
        let h = spec.hidden;
        let ff = spec.ff;
        let vocab = spec.vocab.len();
        let mut pos = 0usize;
        let mut take = |len: usize, name: &str| -> Result<TensorRange, String> {
            let end = pos
                .checked_add(len)
                .ok_or_else(|| format!("{name}: length overflow"))?;
            let range = TensorRange { offset: pos, len };
            pos = end;
            Ok(range)
        };
        let embedding = take(vocab * h, "embedding")?;
        let positional = take(spec.max_context * h, "positional")?;
        let mut decoder = Vec::with_capacity(spec.decoder.len());
        for i in 0..spec.decoder.len() {
            decoder.push(LayerLayout {
                ln1_gamma: take(h, &format!("layer {i} ln1Gamma"))?,
                ln1_beta: take(h, &format!("layer {i} ln1Beta"))?,
                q_weights: take(h * h, &format!("layer {i} qWeights"))?,
                q_bias: take(h, &format!("layer {i} qBias"))?,
                k_weights: take(h * h, &format!("layer {i} kWeights"))?,
                k_bias: take(h, &format!("layer {i} kBias"))?,
                v_weights: take(h * h, &format!("layer {i} vWeights"))?,
                v_bias: take(h, &format!("layer {i} vBias"))?,
                o_weights: take(h * h, &format!("layer {i} oWeights"))?,
                o_bias: take(h, &format!("layer {i} oBias"))?,
                ln2_gamma: take(h, &format!("layer {i} ln2Gamma"))?,
                ln2_beta: take(h, &format!("layer {i} ln2Beta"))?,
                ff1_weights: take(ff * h, &format!("layer {i} ff1Weights"))?,
                ff1_bias: take(ff, &format!("layer {i} ff1Bias"))?,
                ff2_weights: take(h * ff, &format!("layer {i} ff2Weights"))?,
                ff2_bias: take(h, &format!("layer {i} ff2Bias"))?,
            });
        }
        let lnf_gamma = take(h, "lnfGamma")?;
        let lnf_beta = take(h, "lnfBeta")?;
        let (head_weights, head_bias) = if spec.tie_word_embeddings {
            (
                TensorRange {
                    offset: pos,
                    len: 0,
                },
                TensorRange {
                    offset: pos,
                    len: 0,
                },
            )
        } else {
            (take(vocab * h, "headWeights")?, take(vocab, "headBias")?)
        };
        Ok(Self {
            embedding,
            positional,
            decoder,
            lnf_gamma,
            lnf_beta,
            head_weights,
            head_bias,
            total: pos,
        })
    }
}

fn flatten_inline_weights(spec: &GptSpec, layout: &WeightLayout) -> Vec<i32> {
    let mut out = vec![0i32; layout.total];
    out[layout.embedding.index()].copy_from_slice(&spec.embedding);
    out[layout.positional.index()].copy_from_slice(&spec.positional);
    for (layer, ranges) in spec.decoder.iter().zip(&layout.decoder) {
        out[ranges.ln1_gamma.index()].copy_from_slice(&layer.ln1_gamma);
        out[ranges.ln1_beta.index()].copy_from_slice(&layer.ln1_beta);
        out[ranges.q_weights.index()].copy_from_slice(&layer.q_weights);
        out[ranges.q_bias.index()].copy_from_slice(&layer.q_bias);
        out[ranges.k_weights.index()].copy_from_slice(&layer.k_weights);
        out[ranges.k_bias.index()].copy_from_slice(&layer.k_bias);
        out[ranges.v_weights.index()].copy_from_slice(&layer.v_weights);
        out[ranges.v_bias.index()].copy_from_slice(&layer.v_bias);
        out[ranges.o_weights.index()].copy_from_slice(&layer.o_weights);
        out[ranges.o_bias.index()].copy_from_slice(&layer.o_bias);
        out[ranges.ln2_gamma.index()].copy_from_slice(&layer.ln2_gamma);
        out[ranges.ln2_beta.index()].copy_from_slice(&layer.ln2_beta);
        out[ranges.ff1_weights.index()].copy_from_slice(&layer.ff1_weights);
        out[ranges.ff1_bias.index()].copy_from_slice(&layer.ff1_bias);
        out[ranges.ff2_weights.index()].copy_from_slice(&layer.ff2_weights);
        out[ranges.ff2_bias.index()].copy_from_slice(&layer.ff2_bias);
    }
    out[layout.lnf_gamma.index()].copy_from_slice(&spec.lnf_gamma);
    out[layout.lnf_beta.index()].copy_from_slice(&spec.lnf_beta);
    if !spec.tie_word_embeddings {
        out[layout.head_weights.index()].copy_from_slice(&spec.head_weights);
        out[layout.head_bias.index()].copy_from_slice(&spec.head_bias);
    }
    out
}

fn clear_spec_weights(spec: &mut GptSpec) {
    spec.embedding = Vec::new();
    spec.positional = Vec::new();
    for layer in &mut spec.decoder {
        layer.ln1_gamma = Vec::new();
        layer.ln1_beta = Vec::new();
        layer.q_weights = Vec::new();
        layer.q_bias = Vec::new();
        layer.k_weights = Vec::new();
        layer.k_bias = Vec::new();
        layer.v_weights = Vec::new();
        layer.v_bias = Vec::new();
        layer.o_weights = Vec::new();
        layer.o_bias = Vec::new();
        layer.ln2_gamma = Vec::new();
        layer.ln2_beta = Vec::new();
        layer.ff1_weights = Vec::new();
        layer.ff1_bias = Vec::new();
        layer.ff2_weights = Vec::new();
        layer.ff2_bias = Vec::new();
    }
    spec.lnf_gamma = Vec::new();
    spec.lnf_beta = Vec::new();
    spec.head_weights = Vec::new();
    spec.head_bias = Vec::new();
}

fn validate_spec(spec: &GptSpec) -> Result<(), String> {
    validate_metadata(spec)?;
    let h = spec.hidden;
    let ff = spec.ff;
    expect_len("embedding", spec.embedding.len(), spec.vocab.len() * h)?;
    expect_len("positional", spec.positional.len(), spec.max_context * h)?;
    expect_len("lnfGamma", spec.lnf_gamma.len(), h)?;
    expect_len("lnfBeta", spec.lnf_beta.len(), h)?;
    if spec.tie_word_embeddings {
        if !spec.head_weights.is_empty() || !spec.head_bias.is_empty() {
            return Err("tied embeddings must not include headWeights or headBias".into());
        }
    } else {
        expect_len("headWeights", spec.head_weights.len(), spec.vocab.len() * h)?;
        expect_len("headBias", spec.head_bias.len(), spec.vocab.len())?;
    }
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

fn validate_metadata(spec: &GptSpec) -> Result<(), String> {
    if spec.kind != "gpt-decoder" {
        return Err(format!("kind must be `gpt-decoder`, got `{}`", spec.kind));
    }
    if spec.model_id.trim().is_empty() {
        return Err("modelId must not be empty".into());
    }
    if !matches!(spec.tokenizer_kind.as_str(), "char-bpe" | "byte-bpe") {
        return Err(format!(
            "tokenizerKind must be `char-bpe` or `byte-bpe`, got `{}`",
            spec.tokenizer_kind
        ));
    }
    if !matches!(spec.activation.as_str(), "relu" | "gelu_new") {
        return Err(format!(
            "activation must be `relu` or `gelu_new`, got `{}`",
            spec.activation
        ));
    }
    if !matches!(spec.attention_scale.as_str(), "sqrt" | "none") {
        return Err(format!(
            "attentionScale must be `sqrt` or `none`, got `{}`",
            spec.attention_scale
        ));
    }
    if spec.layer_norm_epsilon < 0 {
        return Err("layerNormEpsilon must not be negative".into());
    }
    if spec.sampling.temperature_milli > 100_000 {
        return Err("sampling.temperatureMilli is out of range".into());
    }
    if spec.sampling.top_p_milli > 1000 {
        return Err("sampling.topPMilli must be at most 1000".into());
    }
    if spec.sampling.top_k > spec.vocab.len() {
        return Err("sampling.topK exceeds the vocabulary".into());
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
    Ok(())
}

fn expect_len(name: &str, got: usize, want: usize) -> Result<(), String> {
    if got == want {
        Ok(())
    } else {
        Err(format!("{name}: expected {want} values, got {got}"))
    }
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
            match GptModel::from_file(&path) {
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
        let ids = model.encode(text).unwrap();
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
        assert_eq!(model.encode(prompt).unwrap(), expected);
    }

    #[test]
    fn kv_cache_matches_full_forward() {
        let model = load();
        let ids = model.encode("please cancel order 2064").unwrap();
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

    static STORIES: OnceLock<GptModel> = OnceLock::new();

    fn stories_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-stories-1m.json")
    }

    fn stories_fixture_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-stories-1m.test.json")
    }

    fn stories_available() -> bool {
        stories_path().exists() && stories_fixture_path().exists()
    }

    fn skip_without_stories() -> bool {
        if stories_available() {
            false
        } else {
            eprintln!("skip: run scripts/import_tiny_stories.py to create the TinyStories model");
            true
        }
    }

    fn stories() -> &'static GptModel {
        STORIES.get_or_init(|| {
            let bytes = std::fs::read(stories_path()).expect("committed TinyStories model");
            GptModel::from_bytes(&bytes).expect("valid TinyStories model")
        })
    }

    fn stories_fixture() -> serde_json::Value {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-stories-1m.test.json");
        serde_json::from_slice(&std::fs::read(path).expect("TinyStories fixture"))
            .expect("fixture JSON")
    }

    #[test]
    fn gpt2_tokenizer_matches_the_reference() {
        if skip_without_stories() {
            return;
        }
        let model = stories();
        let fixture = stories_fixture();
        for case in fixture["cases"].as_array().unwrap() {
            let prompt = case["prompt"].as_str().unwrap();
            let expected: Vec<usize> = case["promptTokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(model.encode(prompt).unwrap(), expected, "{prompt}");
        }
    }

    #[test]
    fn tiny_stories_greedy_tokens_match_the_reference() {
        if skip_without_stories() {
            return;
        }
        let model = stories();
        let fixture = stories_fixture();
        let mut mismatches = Vec::new();
        for case in fixture["cases"].as_array().unwrap() {
            let prompt = case["prompt"].as_str().unwrap();
            let expected: Vec<usize> = case["tokenIds"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let got = model.generate(prompt).unwrap();
            if got.generated_ids != expected {
                let first_diff = got
                    .generated_ids
                    .iter()
                    .zip(&expected)
                    .position(|(a, b)| a != b);
                mismatches.push(format!(
                    "{prompt}: first diff {:?}, got len {}, expected len {}",
                    first_diff,
                    got.generated_ids.len(),
                    expected.len()
                ));
            }
        }
        assert!(
            mismatches.is_empty(),
            "token parity failed on {} prompts: {:?}",
            mismatches.len(),
            mismatches
        );
    }

    #[test]
    fn tiny_stories_kv_cache_matches_full_forward() {
        if skip_without_stories() {
            return;
        }
        let model = stories();
        let ids = model.encode("Once upon a time").unwrap();
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
    fn gelu_new_matches_the_reference() {
        for (x, want) in [
            (-3.0f64, -0.003637392),
            (-1.0, -0.158808009),
            (0.0, 0.0),
            (1.0, 0.841191991),
            (3.0, 2.996362608),
        ] {
            let got = gelu_new((x * ONE as f64) as i64) as f64 / ONE as f64;
            assert!(
                (got - want).abs() < 0.02,
                "gelu_new({x}) = {got}, want {want}"
            );
        }
    }

    struct BinaryFixture {
        dir: PathBuf,
        metadata: PathBuf,
        weights: PathBuf,
        original: GptModel,
    }

    impl Drop for BinaryFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn binary_fixture(label: &str) -> BinaryFixture {
        let original = load();
        let dir = std::env::temp_dir().join(format!("verinoo-gpt-{label}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let weights_path = dir.join("tiny.vnnw");
        let sha = crate::weights::write(&weights_path, original.weights.as_slice()).unwrap();
        let metadata_path = dir.join("tiny.json");
        let mut metadata = original.spec.clone();
        metadata.weights_file = "tiny.vnnw".into();
        metadata.weights_sha256 = sha;
        std::fs::write(&metadata_path, serde_json::to_string(&metadata).unwrap()).unwrap();
        BinaryFixture {
            dir,
            metadata: metadata_path,
            weights: weights_path,
            original,
        }
    }

    #[test]
    fn inline_json_weights_are_owned() {
        let model = load();
        assert!(!model.is_mapped());
        let generated = model.generate("please cancel").unwrap();
        assert!(!generated.generated_ids.is_empty());
    }

    #[test]
    fn binary_weights_are_mapped() {
        let fixture = binary_fixture("mapped");
        let loaded = GptModel::from_file(&fixture.metadata).unwrap();
        assert!(loaded.is_mapped());
        assert_eq!(loaded.parameter_count(), fixture.original.parameter_count());
        assert_eq!(loaded.weights_hash().len(), 64);
    }

    #[test]
    fn binary_weights_roundtrip_and_tamper_detection() {
        let fixture = binary_fixture("roundtrip");
        let loaded = GptModel::from_file(&fixture.metadata).unwrap();
        assert!(loaded.is_mapped());
        assert_eq!(loaded.parameter_count(), fixture.original.parameter_count());
        assert_eq!(loaded.weights_hash().len(), 64);
        let a = fixture.original.generate("please cancel").unwrap();
        let b = loaded.generate("please cancel").unwrap();
        assert_eq!(a.generated_ids, b.generated_ids);
        assert_eq!(a.text, b.text);

        let mut bytes = std::fs::read(&fixture.weights).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&fixture.weights, &bytes).unwrap();
        let err = GptModel::from_file(&fixture.metadata).unwrap_err();
        assert!(err.contains("hash mismatch"), "{err}");
    }

    #[test]
    fn sampling_is_deterministic_and_greedy_when_cold() {
        let model = load();
        let greedy = model.generate("please cancel").unwrap();
        let zero = model
            .generate_with_sampling(
                "please cancel",
                &Sampling {
                    temperature_milli: 0,
                    top_k: 0,
                    top_p_milli: 0,
                    seed: 99,
                },
            )
            .unwrap();
        assert_eq!(greedy.generated_ids, zero.generated_ids);

        let sampling = Sampling {
            temperature_milli: 800,
            top_k: 40,
            top_p_milli: 900,
            seed: 7,
        };
        let a = model
            .generate_with_sampling("please cancel", &sampling)
            .unwrap();
        let b = model
            .generate_with_sampling("please cancel", &sampling)
            .unwrap();
        assert_eq!(a.generated_ids, b.generated_ids);
        assert_eq!(a.text, b.text);
        assert!(!a.generated_ids.is_empty());

        let top_one = model
            .generate_with_sampling(
                "please cancel",
                &Sampling {
                    temperature_milli: 1000,
                    top_k: 1,
                    top_p_milli: 0,
                    seed: 3,
                },
            )
            .unwrap();
        assert_eq!(greedy.generated_ids, top_one.generated_ids);
    }

    fn distilgpt2_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/distilgpt2.json")
    }

    fn distilgpt2_fixture_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/distilgpt2.test.json")
    }

    fn distilgpt2_enabled() -> bool {
        std::env::var("VNN_TEST_DISTILGPT2")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
            && distilgpt2_path().exists()
            && distilgpt2_fixture_path().exists()
    }

    #[test]
    fn distilgpt2_binary_model_matches_float_greedy_if_enabled() {
        if !distilgpt2_enabled() {
            eprintln!("skip: run scripts/import_distilgpt2.py and set VNN_TEST_DISTILGPT2=1");
            return;
        }
        let model = GptModel::from_file(&distilgpt2_path()).expect("valid DistilGPT-2");
        assert_eq!(model.spec.max_context, 1024);
        assert_eq!(model.spec.hidden, 768);
        assert_eq!(model.spec.heads, 12);
        let fixture: serde_json::Value = serde_json::from_slice(
            &std::fs::read(distilgpt2_fixture_path()).expect("DistilGPT-2 fixture"),
        )
        .expect("fixture JSON");
        for case in fixture["cases"].as_array().expect("cases") {
            let prompt = case["prompt"].as_str().unwrap();
            let prompt_ids: Vec<usize> = case["promptTokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let expected: Vec<usize> = case["tokenIds"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(model.encode(prompt).unwrap(), prompt_ids, "{prompt}");
            let generated = model.generate(prompt).unwrap();
            assert_eq!(generated.generated_ids, expected, "{prompt}");
        }
    }
}
