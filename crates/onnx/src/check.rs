//! Parity between the imported integer graph and a float evaluation of the
//! same layers.
//!
//! A conversion is only worth anything if the customer's answers survive it.
//! The importer therefore keeps the original f32 parameters, evaluates both
//! paths on the same feature vectors, and reports the largest logit divergence
//! and whether the predicted class ever changes. Argmax disagreement is a hard
//! failure: quantisation is allowed to move a score, not to change the answer.

use verinoo_runtime::{featurize, Model, ModelSpec};

use crate::convert::FloatLayer;

#[derive(Debug, Clone)]
pub struct Parity {
    pub samples: usize,
    pub max_logit_diff: f32,
    /// Samples where float and integer execution pick the same label.
    pub argmax_agree: usize,
    /// Human-readable description of every disagreement.
    pub mismatches: Vec<String>,
}

impl Parity {
    /// A conversion is acceptable only if no sample changes class.
    pub fn is_exact_enough(&self) -> bool {
        self.mismatches.is_empty()
    }
}

pub fn float_logits(layers: &[FloatLayer], input: &[i32], width: usize) -> Vec<f32> {
    let mut x: Vec<f32> = (0..width)
        .map(|i| input.get(i).copied().unwrap_or(0) as f32 / 65_536.0)
        .collect();
    for layer in layers {
        match layer {
            FloatLayer::Dense {
                weights,
                bias,
                input,
                output,
            } => {
                let mut y = vec![0.0f32; *output];
                for j in 0..*output {
                    let mut acc = bias[j];
                    let row = j * *input;
                    for f in 0..*input {
                        acc += x[f] * weights[row + f];
                    }
                    y[j] = acc;
                }
                x = y;
            }
            FloatLayer::Relu => {
                for v in x.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            // Softmax is monotone, so it cannot change the argmax; logits are
            // compared directly.
            FloatLayer::Softmax => {}
        }
    }
    x
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

/// Deterministic sample inputs: real sentences through the production
/// featuriser, then pseudo-random token bags that reach feature buckets those
/// sentences do not.
pub fn sample_features(width: usize, count: usize) -> Vec<(String, Vec<i32>)> {
    const SENTENCES: [&str; 10] = [
        "my parcel was never delivered",
        "i was charged twice for the same order",
        "please cancel my subscription",
        "i want a refund for the broken item",
        "where is my invoice for last month",
        "the delivery is three days late",
        "stop billing me, i already cancelled",
        "returning the shoes, they do not fit",
        "can you resend the receipt",
        "refund approved, thanks for the help",
    ];

    let mut out = Vec::new();
    for sentence in SENTENCES {
        if out.len() >= count {
            return out;
        }
        out.push((sentence.to_string(), featurize(sentence, width)));
    }

    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    while out.len() < count {
        let tokens = 3 + (next(&mut state) % 10) as usize;
        let mut text = String::new();
        for _ in 0..tokens {
            text.push_str(&format!("tok{} ", next(&mut state) % 97));
        }
        out.push((text.clone(), featurize(&text, width)));
    }
    out
}

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

pub fn compare(
    spec: &ModelSpec,
    float: &[FloatLayer],
    samples: &[(String, Vec<i32>)],
) -> Result<Parity, String> {
    let json = serde_json::to_value(spec).map_err(|e| e.to_string())?;
    let model = Model::from_value(&json).map_err(|e| format!("model does not load: {e}"))?;

    let mut parity = Parity {
        samples: samples.len(),
        max_logit_diff: 0.0,
        argmax_agree: 0,
        mismatches: Vec::new(),
    };

    for (text, features) in samples {
        let integer = model.forward_logits(features);
        let reference = float_logits(float, features, spec.features);
        if integer.len() != reference.len() {
            return Err(format!(
                "graph produces {} logits but the float reference produces {}",
                integer.len(),
                reference.len()
            ));
        }
        let quantised: Vec<f32> = integer.iter().map(|v| *v as f32 / 65_536.0).collect();
        for (a, b) in quantised.iter().zip(reference.iter()) {
            let diff = (a - b).abs();
            if diff > parity.max_logit_diff {
                parity.max_logit_diff = diff;
            }
        }
        let int_best = argmax(&quantised);
        let ref_best = argmax(&reference);
        if int_best == ref_best {
            parity.argmax_agree += 1;
        } else {
            let name = |i: usize| {
                spec.labels
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("#{i}"))
            };
            parity.mismatches.push(format!(
                "`{text}`: float picks `{}`, imported graph picks `{}`",
                name(ref_best),
                name(int_best)
            ));
        }
    }

    Ok(parity)
}
