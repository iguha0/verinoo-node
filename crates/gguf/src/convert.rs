//! Turn a GGUF tensor table into a runtime graph.
//!
//! GGUF records no graph: it lists named tensors and their shapes. So the caller
//! supplies a *plan* — which tensors are dense layers, in what order — either
//! explicitly (`--layers fc1,fc2`) or by convention (every 2-D tensor whose name
//! ends in `.weight`, sorted by name). Widths are then checked against each
//! other and against the declared labels; a plan that does not describe a chain
//! is refused rather than guessed at.

use verinoo_onnx::convert::FloatLayer;
use verinoo_onnx::quant::to_q16;
use verinoo_runtime::{Layer, ModelSpec};

use crate::gguf::{type_name, Gguf};

#[derive(Debug, Clone)]
pub struct GgufOptions {
    pub model_id: String,
    pub labels: Vec<String>,
    /// Explicit layer prefixes; `None` uses the `.weight` convention.
    pub layers: Option<Vec<String>>,
    /// `relu` (default) or `none`.
    pub activation: String,
    /// Featuriser width. Checked against the first layer's input width.
    pub features: Option<usize>,
    pub description: String,
    pub architecture: Option<String>,
}

impl GgufOptions {
    pub fn new(model_id: impl Into<String>, labels: Vec<String>) -> Self {
        Self {
            model_id: model_id.into(),
            labels,
            layers: None,
            activation: "relu".into(),
            features: None,
            description: String::new(),
            architecture: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerPlan {
    pub name: String,
    pub weight: String,
    pub bias: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Imported {
    pub spec: ModelSpec,
    pub float: Vec<FloatLayer>,
    pub plan: Vec<LayerPlan>,
    pub notes: Vec<String>,
}

fn resolve_weight(file: &Gguf, name: &str) -> Option<String> {
    for candidate in [
        name.to_string(),
        format!("{name}.weight"),
        format!("{name}/weight"),
        format!("{name}.w"),
    ] {
        if let Some(t) = file.tensor(&candidate) {
            if t.is_matrix() {
                return Some(candidate);
            }
        }
    }
    None
}

fn resolve_bias(file: &Gguf, name: &str) -> Option<String> {
    for candidate in [
        format!("{name}.bias"),
        format!("{name}/bias"),
        format!("{name}.b"),
    ] {
        if let Some(t) = file.tensor(&candidate) {
            if t.dims.len() == 1 {
                return Some(candidate);
            }
        }
    }
    None
}

/// Work out which tensors are layers, without reading their data.
pub fn plan_layers(file: &Gguf, opts: &GgufOptions) -> Result<Vec<LayerPlan>, String> {
    let names: Vec<String> = match &opts.layers {
        Some(requested) => {
            if requested.is_empty() {
                return Err("--layers was given but names no layers".into());
            }
            requested.clone()
        }
        None => {
            let mut auto: Vec<String> = file
                .tensors
                .iter()
                .filter(|t| t.is_matrix())
                .filter_map(|t| {
                    t.name
                        .strip_suffix(".weight")
                        .or_else(|| t.name.strip_suffix("/weight"))
                        .map(|s| s.to_string())
                })
                .collect();
            auto.sort();
            auto.dedup();
            if auto.is_empty() {
                let available = file.names();
                return Err(format!(
                    "no 2-D tensors named `*.weight` were found, so there is no plan to infer; \
                     pass --layers. Tensors in the file: {}",
                    if available.is_empty() {
                        "none".to_string()
                    } else {
                        available.join(", ")
                    }
                ));
            }
            auto
        }
    };

    let mut plan = Vec::with_capacity(names.len());
    for name in names {
        let weight = resolve_weight(file, &name).ok_or_else(|| {
            format!(
                "layer `{name}`: no 2-D weight tensor (looked for `{name}.weight`, \
                 `{name}/weight`, `{name}.w`)"
            )
        })?;
        let bias = resolve_bias(file, &name);
        plan.push(LayerPlan { name, weight, bias });
    }
    Ok(plan)
}

pub fn convert(file: &mut Gguf, opts: &GgufOptions) -> Result<Imported, String> {
    if opts.labels.is_empty() {
        return Err("at least one label is required (GGUF does not store class names)".into());
    }
    let activation = match opts.activation.as_str() {
        "relu" => Some("relu"),
        "none" => None,
        other => {
            return Err(format!(
                "unsupported --activation `{other}` (the runtime has relu and softmax)"
            ))
        }
    };

    let plan = plan_layers(file, opts)?;
    let mut notes = Vec::new();

    // Shapes first: refuse a plan that is not a chain before reading any data.
    let mut widths: Vec<usize> = Vec::new();
    for (i, layer) in plan.iter().enumerate() {
        let info = file
            .tensor(&layer.weight)
            .ok_or_else(|| format!("layer `{}`: weight tensor vanished", layer.name))?
            .clone();
        let (input, output) = info.matrix_shape().ok_or_else(|| {
            format!(
                "layer `{}`: weight `{}` is not a matrix",
                layer.name, layer.weight
            )
        })?;
        if i == 0 {
            widths.push(input);
        } else if *widths.last().unwrap_or(&0) != input {
            return Err(format!(
                "layer `{}`: weight `{}` expects {input} inputs but the previous layer \
                 produces {}; the plan is not a chain",
                layer.name,
                layer.weight,
                widths.last().copied().unwrap_or(0)
            ));
        }
        widths.push(output);
    }
    let features = *widths.first().ok_or_else(|| "plan is empty".to_string())?;
    if features == 0 {
        return Err("first layer has a zero-width input".into());
    }
    if let Some(given) = opts.features {
        if given != features {
            return Err(format!(
                "first layer expects {features} input features but --features says {given}"
            ));
        }
    }
    if *widths.last().unwrap_or(&0) != opts.labels.len() {
        return Err(format!(
            "graph produces {} outputs but {} labels were supplied",
            widths.last().copied().unwrap_or(0),
            opts.labels.len()
        ));
    }

    let mut spec_layers: Vec<Layer> = Vec::new();
    let mut float: Vec<FloatLayer> = Vec::new();
    for (i, layer) in plan.iter().enumerate() {
        let info = file
            .tensor(&layer.weight)
            .ok_or_else(|| format!("layer `{}`: weight tensor vanished", layer.name))?
            .clone();
        let (input, output) = info.matrix_shape().unwrap();
        let data = file.read_tensor(&layer.weight)?;
        if data.len() != input * output {
            return Err(format!(
                "layer `{}`: weight `{}` produced {} values for a {output}×{input} matrix",
                layer.name,
                layer.weight,
                data.len()
            ));
        }
        if info.kind != 0 {
            notes.push(format!(
                "{}: {} dequantised to f32, then quantised to Q16.16",
                layer.weight,
                type_name(info.kind)
            ));
        }

        let bias_data: Vec<f32> = match &layer.bias {
            Some(name) => {
                let b = file.read_tensor(name)?;
                if b.len() != output {
                    return Err(format!(
                        "layer `{}`: bias `{name}` has {} values but the layer has {output} outputs",
                        layer.name,
                        b.len()
                    ));
                }
                b
            }
            None => {
                notes.push(format!("{}: no bias tensor, using zeros", layer.name));
                vec![0.0; output]
            }
        };

        let mut qw = Vec::with_capacity(data.len());
        for (k, v) in data.iter().enumerate() {
            qw.push(to_q16(*v, &format!("{}: weight[{k}]", layer.weight))?);
        }
        let mut qb = Vec::with_capacity(bias_data.len());
        for (k, v) in bias_data.iter().enumerate() {
            let what = layer.bias.clone().unwrap_or_else(|| layer.name.clone());
            qb.push(to_q16(*v, &format!("{what}: bias[{k}]"))?);
        }
        spec_layers.push(Layer {
            op: "dense".into(),
            input,
            output,
            weights: qw,
            bias: qb,
        });
        float.push(FloatLayer::Dense {
            weights: data,
            bias: bias_data,
            input,
            output,
        });

        if i + 1 < plan.len() {
            if let Some(op) = activation {
                spec_layers.push(Layer {
                    op: op.into(),
                    input: 0,
                    output: 0,
                    weights: Vec::new(),
                    bias: Vec::new(),
                });
                float.push(FloatLayer::Relu);
            }
        }
    }

    spec_layers.push(Layer {
        op: "softmax".into(),
        input: 0,
        output: 0,
        weights: Vec::new(),
        bias: Vec::new(),
    });
    float.push(FloatLayer::Softmax);

    let arch_tag = opts
        .architecture
        .clone()
        .or_else(|| file.architecture().map(|a| format!("gguf-{a}")))
        .unwrap_or_else(|| "gguf".into());
    let architecture = format!(
        "{arch_tag}-{}",
        widths
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join("-")
    );
    let description = if opts.description.is_empty() {
        format!(
            "imported from GGUF ({}): {}",
            file.architecture().unwrap_or("unlabelled"),
            plan.iter()
                .map(|l| l.name.clone())
                .collect::<Vec<_>>()
                .join(" → ")
        )
    } else {
        opts.description.clone()
    };

    let spec = ModelSpec {
        model_id: opts.model_id.clone(),
        architecture,
        description,
        features,
        labels: opts.labels.clone(),
        layers: spec_layers,
    };

    let json = serde_json::to_value(&spec).map_err(|e| e.to_string())?;
    verinoo_runtime::Model::from_value(&json)
        .map_err(|e| format!("converted graph failed runtime validation: {e}"))?;

    Ok(Imported {
        spec,
        float,
        plan,
        notes,
    })
}
