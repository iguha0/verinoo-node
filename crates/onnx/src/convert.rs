//! Fold an ONNX graph into the linear chain the runtime can execute.
//!
//! Supported shapes:
//!
//! * `Gemm` (`alpha = 1`, bias as the third input) — the linear layer most
//!   exporters emit for a dense model
//! * `MatMul` (weight × activation, or activation × weight, which is
//!   transposed) optionally followed by `Add` for the bias
//! * `Relu`, `Softmax`
//! * `Identity`, `Dropout`, `Flatten`, `Reshape`, `Squeeze`, `Unsqueeze` as
//!   pass-through so a "shape hygiene" prefix does not block an import
//!
//! Anything else is a hard error naming the node. A converter that guessed at
//! an op would produce a model whose answers differ from the customer's, and
//! the difference would only surface after the model was committed on-chain.

use std::collections::HashMap;

use verinoo_runtime::{Layer, ModelSpec};

use crate::onnx::{Graph, Model, Tensor};
use crate::quant;

#[derive(Debug, Clone)]
pub enum FloatLayer {
    Dense {
        weights: Vec<f32>,
        bias: Vec<f32>,
        input: usize,
        output: usize,
    },
    Relu,
    Softmax,
}

#[derive(Debug, Clone)]
pub struct ImportOptions {
    pub model_id: String,
    pub labels: Vec<String>,
    /// Featuriser width. Required only when the graph input is dynamic.
    pub features: Option<usize>,
    pub description: String,
    pub architecture: Option<String>,
}

impl ImportOptions {
    pub fn new(model_id: impl Into<String>, labels: Vec<String>) -> Self {
        Self {
            model_id: model_id.into(),
            labels,
            features: None,
            description: String::new(),
            architecture: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Imported {
    pub spec: ModelSpec,
    /// The same graph in f32, kept so the importer can measure how far the
    /// integer execution lands from the customer's reference numerics.
    pub float: Vec<FloatLayer>,
    pub input_name: String,
    pub notes: Vec<String>,
}

enum Val<'a> {
    Act { width: usize },
    Weight(&'a Tensor),
    Shape(Vec<usize>),
}

struct Chain {
    layers: Vec<Layer>,
    float: Vec<FloatLayer>,
    width: usize,
    widths: Vec<usize>,
}

impl Chain {
    fn dense(
        &mut self,
        input: usize,
        output: usize,
        w: &[f32],
        bias: &[f32],
        what: &str,
    ) -> Result<(), String> {
        if w.len() != input * output {
            return Err(format!(
                "{what}: weight matrix has {} values but {output}×{input} are required",
                w.len()
            ));
        }
        if bias.len() != output {
            return Err(format!(
                "{what}: bias has {} values but {output} are required",
                bias.len()
            ));
        }
        let mut qw = Vec::with_capacity(w.len());
        for (i, v) in w.iter().enumerate() {
            qw.push(quant::to_q16(*v, &format!("{what}: weight[{i}]"))?);
        }
        let mut qb = Vec::with_capacity(bias.len());
        for (i, v) in bias.iter().enumerate() {
            qb.push(quant::to_q16(*v, &format!("{what}: bias[{i}]"))?);
        }
        self.layers.push(Layer {
            op: "dense".into(),
            input,
            output,
            weights: qw,
            bias: qb,
        });
        self.float.push(FloatLayer::Dense {
            weights: w.to_vec(),
            bias: bias.to_vec(),
            input,
            output,
        });
        self.width = output;
        self.widths.push(output);
        Ok(())
    }

    /// Fold a bias that arrived as a separate `Add` node into the dense layer
    /// that produced the current activation.
    fn add_bias(&mut self, bias: &[f32], what: &str) -> Result<(), String> {
        let Some(layer) = self.float.last_mut() else {
            return Err(format!("{what}: bias add without a preceding dense layer"));
        };
        match layer {
            FloatLayer::Dense { bias: fb, .. } => {
                for (i, v) in bias.iter().enumerate() {
                    fb[i] += *v;
                }
            }
            _ => return Err(format!("{what}: bias add must follow a dense layer")),
        }
        let qlayer = self
            .layers
            .last_mut()
            .ok_or_else(|| format!("{what}: bias add without a preceding layer"))?;
        if qlayer.op != "dense" {
            return Err(format!("{what}: bias add must follow a dense layer"));
        }
        for (i, v) in bias.iter().enumerate() {
            let q = quant::to_q16(*v, &format!("{what}: bias[{i}]"))?;
            qlayer.bias[i] = qlayer.bias[i].saturating_add(q);
        }
        Ok(())
    }

    fn relu(&mut self) {
        self.layers.push(Layer {
            op: "relu".into(),
            input: 0,
            output: 0,
            weights: Vec::new(),
            bias: Vec::new(),
        });
        self.float.push(FloatLayer::Relu);
    }

    fn softmax(&mut self) {
        self.layers.push(Layer {
            op: "softmax".into(),
            input: 0,
            output: 0,
            weights: Vec::new(),
            bias: Vec::new(),
        });
        self.float.push(FloatLayer::Softmax);
    }

    fn last_is_dense(&self) -> bool {
        self.layers.last().map(|l| l.op == "dense").unwrap_or(false)
    }
}

pub fn convert(model: &Model, opts: &ImportOptions) -> Result<Imported, String> {
    let graph = &model.graph;
    if opts.labels.is_empty() {
        return Err("at least one label is required (ONNX does not store them)".into());
    }
    let mut notes = Vec::new();

    // The data input is the only graph input that is not also an initializer.
    let mut data_inputs = graph
        .inputs
        .iter()
        .filter(|i| !graph.initializers.contains_key(&i.name));
    let input = data_inputs
        .next()
        .ok_or_else(|| "graph has no data input".to_string())?;
    if data_inputs.next().is_some() {
        return Err(
            "graph has more than one data input; only single-input classifiers import".into(),
        );
    }
    let declared_width = match input.shape.len() {
        0 => None,
        1 => input.shape[0],
        _ => *input.shape.last().unwrap(),
    };
    let features = match (declared_width, opts.features) {
        (Some(declared), Some(given)) if declared != given => {
            return Err(format!(
                "graph input `{}` expects {declared} features but --features says {given}",
                input.name
            ))
        }
        (Some(declared), _) => declared,
        (None, Some(given)) => given,
        (None, None) => {
            return Err(format!(
                "graph input `{}` has a dynamic width; pass --features",
                input.name
            ))
        }
    };
    if features == 0 {
        return Err("feature width must be greater than zero".into());
    }

    let mut values: HashMap<String, Val> = HashMap::new();
    for tensor in graph.initializers.values() {
        if tensor.is_integer() {
            values.insert(
                tensor.name.clone(),
                Val::Shape(tensor.ints.iter().map(|&v| v.max(0) as usize).collect()),
            );
        } else if tensor.is_float() {
            values.insert(tensor.name.clone(), Val::Weight(tensor));
        }
    }
    values.insert(input.name.clone(), Val::Act { width: features });

    let mut chain = Chain {
        layers: Vec::new(),
        float: Vec::new(),
        width: features,
        widths: vec![features],
    };

    for node in &graph.nodes {
        let out_name = node
            .outputs
            .first()
            .cloned()
            .ok_or_else(|| format!("{}: node has no output", node.label()))?;
        let width = match node.op_type.as_str() {
            "MatMul" => matmul(node, &values, &mut chain)?,
            "Gemm" => gemm(node, &values, &mut chain)?,
            "Add" => {
                add(node, &values, &mut chain)?;
                chain.width
            }
            "Relu" => {
                require_activation(node, &values)?;
                chain.relu();
                chain.width
            }
            "Softmax" => {
                require_activation(node, &values)?;
                let axis = node.int("axis", -1);
                if axis != -1 && axis != 1 {
                    return Err(format!(
                        "{}: softmax over axis {axis} is not supported (only the class axis)",
                        node.label()
                    ));
                }
                chain.softmax();
                chain.width
            }
            "Identity" | "Dropout" | "Flatten" | "Reshape" | "Squeeze" | "Unsqueeze" => {
                pass_through(node, &values, &mut notes)?
            }
            other => {
                return Err(format!(
                    "unsupported op `{other}` in node {}; the runtime executes only \
                     dense/relu/softmax chains, so this model cannot be imported",
                    node.label()
                ))
            }
        };
        if width != chain.width {
            return Err(format!(
                "{}: internal width bookkeeping mismatch",
                node.label()
            ));
        }
        values.insert(out_name, Val::Act { width });
    }

    if chain.layers.is_empty() {
        return Err("graph contains no executable layer".into());
    }
    let softmax_tail = chain
        .layers
        .last()
        .map(|l| l.op == "softmax")
        .unwrap_or(false);
    if !softmax_tail {
        chain.softmax();
        notes.push("graph did not end in Softmax; one was appended".into());
    }
    if chain.width != opts.labels.len() {
        return Err(format!(
            "graph produces {} outputs but {} labels were supplied",
            chain.width,
            opts.labels.len()
        ));
    }

    let architecture = opts.architecture.clone().unwrap_or_else(|| {
        format!(
            "onnx-{}",
            chain
                .widths
                .iter()
                .map(|w| w.to_string())
                .collect::<Vec<_>>()
                .join("-")
        )
    });

    let spec = ModelSpec {
        model_id: opts.model_id.clone(),
        architecture,
        description: opts.description.clone(),
        features,
        labels: opts.labels.clone(),
        layers: chain.layers,
    };

    // Round-trip through the runtime so an importer bug cannot emit a file the
    // node would refuse to load.
    let json = serde_json::to_value(&spec).map_err(|e| e.to_string())?;
    verinoo_runtime::Model::from_value(&json)
        .map_err(|e| format!("converted graph failed runtime validation: {e}"))?;

    if model.opset > 0 && model.opset < 13 {
        notes.push(format!(
            "model declares opset {}; checked against the default op semantics",
            model.opset
        ));
    }

    Ok(Imported {
        spec,
        float: chain.float,
        input_name: input.name.clone(),
        notes,
    })
}

fn activation_width<'a>(values: &HashMap<String, Val<'a>>, name: &str) -> Option<usize> {
    match values.get(name) {
        Some(Val::Act { width }) => Some(*width),
        _ => None,
    }
}

fn require_activation(
    node: &crate::onnx::Node,
    values: &HashMap<String, Val<'_>>,
) -> Result<usize, String> {
    let name = node
        .inputs
        .first()
        .ok_or_else(|| format!("{}: node has no input", node.label()))?;
    activation_width(values, name)
        .ok_or_else(|| format!("{}: input `{name}` is not an activation", node.label()))
}

fn matmul(
    node: &crate::onnx::Node,
    values: &HashMap<String, Val<'_>>,
    chain: &mut Chain,
) -> Result<usize, String> {
    if node.inputs.len() != 2 {
        return Err(format!("{}: MatMul needs exactly two inputs", node.label()));
    }
    let (a, b) = (&node.inputs[0], &node.inputs[1]);
    match (values.get(a), values.get(b)) {
        (Some(Val::Weight(w)), Some(Val::Act { width })) => {
            if w.rank() != 2 || w.dims[1] != *width {
                return Err(format!(
                    "{}: weight shape {:?} does not match {width} input features",
                    node.label(),
                    w.dims
                ));
            }
            chain.dense(
                *width,
                w.dims[0],
                &w.floats,
                &vec![0.0; w.dims[0]],
                &node.label(),
            )?;
            Ok(w.dims[0])
        }
        (Some(Val::Act { width }), Some(Val::Weight(w))) => {
            if w.rank() != 2 || w.dims[0] != *width {
                return Err(format!(
                    "{}: weight shape {:?} is not ({width}, outputs)",
                    node.label(),
                    w.dims
                ));
            }
            let out = w.dims[1];
            let mut t = vec![0.0f32; out * *width];
            for j in 0..out {
                for f in 0..*width {
                    t[j * *width + f] = w.floats[f * out + j];
                }
            }
            chain.dense(*width, out, &t, &vec![0.0; out], &node.label())?;
            Ok(out)
        }
        _ => Err(format!(
            "{}: MatMul needs one float weight initializer and one activation",
            node.label()
        )),
    }
}

fn gemm(
    node: &crate::onnx::Node,
    values: &HashMap<String, Val<'_>>,
    chain: &mut Chain,
) -> Result<usize, String> {
    if node.inputs.len() < 2 {
        return Err(format!("{}: Gemm needs at least two inputs", node.label()));
    }
    let alpha = node.float("alpha", 1.0);
    let beta = node.float("beta", 1.0);
    let trans_a = node.int("transA", 0) != 0;
    let trans_b = node.int("transB", 0) != 0;
    if (alpha - 1.0).abs() > 1e-6 {
        return Err(format!(
            "{}: alpha = {alpha} is not supported (only the default 1.0)",
            node.label()
        ));
    }
    if (beta - 1.0).abs() > 1e-6 {
        return Err(format!(
            "{}: beta = {beta} is not supported (only the default 1.0)",
            node.label()
        ));
    }

    // ONNX Gemm is `Y = alpha·A'·B' + beta·C`. A framework exporting a dense
    // layer puts the *activation* in A and the weights in B with `transB = 1`
    // (B stored as `out × in`); hand-written graphs sometimes put the weights
    // in A instead. Both orders are handled, and the matrix is normalised to
    // the runtime's row-major `out × in`.
    let (w, width, out, inner, stored_transposed) =
        match (values.get(&node.inputs[0]), values.get(&node.inputs[1])) {
            (Some(Val::Act { width }), Some(Val::Weight(w))) => {
                let (out, inner) = if trans_b {
                    (
                        w.dims.first().copied().unwrap_or(0),
                        w.dims.get(1).copied().unwrap_or(0),
                    )
                } else {
                    (
                        w.dims.get(1).copied().unwrap_or(0),
                        w.dims.first().copied().unwrap_or(0),
                    )
                };
                (w, *width, out, inner, !trans_b)
            }
            (Some(Val::Weight(w)), Some(Val::Act { width })) => {
                let (out, inner) = if trans_a {
                    (
                        w.dims.get(1).copied().unwrap_or(0),
                        w.dims.first().copied().unwrap_or(0),
                    )
                } else {
                    (
                        w.dims.first().copied().unwrap_or(0),
                        w.dims.get(1).copied().unwrap_or(0),
                    )
                };
                (w, *width, out, inner, trans_a)
            }
            _ => {
                return Err(format!(
                    "{}: Gemm needs one float weight initializer and one activation",
                    node.label()
                ))
            }
        };
    if w.rank() != 2 || w.dims[0].checked_mul(w.dims[1]) != Some(w.floats.len()) {
        return Err(format!(
            "{}: Gemm weights must be a 2-D matrix, found dims {:?} with {} values",
            node.label(),
            w.dims,
            w.floats.len()
        ));
    }
    if inner != width {
        return Err(format!(
            "{}: weights are {out}×{inner} but the activation has {width} features",
            node.label()
        ));
    }
    if out == 0 || inner == 0 {
        return Err(format!("{}: Gemm weight matrix is empty", node.label()));
    }
    let mut m = vec![0.0f32; out * inner];
    for r in 0..out {
        for c in 0..inner {
            let src = if stored_transposed {
                c * out + r
            } else {
                r * inner + c
            };
            m[r * inner + c] = w.floats[src];
        }
    }

    let mut bias = vec![0.0f32; out];
    if let Some(c_name) = node.inputs.get(2).filter(|n| !n.is_empty()) {
        match values.get(c_name) {
            Some(Val::Weight(c)) if c.element_count() == out => bias.copy_from_slice(&c.floats),
            Some(Val::Weight(c)) => {
                return Err(format!(
                    "{}: bias `{c_name}` has {} values but the layer has {out} outputs",
                    node.label(),
                    c.element_count()
                ))
            }
            _ => {
                return Err(format!(
                    "{}: bias `{c_name}` is not a float initializer",
                    node.label()
                ))
            }
        }
    }

    chain.dense(width, out, &m, &bias, &node.label())?;
    Ok(out)
}

fn add(
    node: &crate::onnx::Node,
    values: &HashMap<String, Val<'_>>,
    chain: &mut Chain,
) -> Result<(), String> {
    if node.inputs.len() != 2 {
        return Err(format!("{}: Add needs exactly two inputs", node.label()));
    }
    let (act_name, bias) = match (values.get(&node.inputs[0]), values.get(&node.inputs[1])) {
        (Some(Val::Act { .. }), Some(Val::Weight(w))) => (&node.inputs[0], *w),
        (Some(Val::Weight(w)), Some(Val::Act { .. })) => (&node.inputs[1], *w),
        _ => {
            return Err(format!(
                "{}: Add is supported only for bias addition (activation + initializer)",
                node.label()
            ))
        }
    };
    let width = activation_width(values, act_name)
        .ok_or_else(|| format!("{}: no activation to add the bias to", node.label()))?;
    if bias.dims.len() != 1 || bias.dims[0] != width {
        return Err(format!(
            "{}: bias shape {:?} does not match the {width}-wide activation",
            node.label(),
            bias.dims
        ));
    }
    if !chain.last_is_dense() {
        return Err(format!(
            "{}: bias add must come directly after the dense layer it belongs to",
            node.label()
        ));
    }
    chain.add_bias(&bias.floats, &node.label())
}

fn pass_through(
    node: &crate::onnx::Node,
    values: &HashMap<String, Val<'_>>,
    notes: &mut Vec<String>,
) -> Result<usize, String> {
    let width = require_activation(node, values)?;
    if node.op_type == "Flatten" {
        let axis = node.int("axis", 1);
        if axis != 1 {
            return Err(format!(
                "{}: Flatten with axis {axis} would merge the batch dimension",
                node.label()
            ));
        }
    }
    if node.op_type == "Reshape" || node.op_type == "Squeeze" || node.op_type == "Unsqueeze" {
        if let Some(shape_name) = node.inputs.get(1).filter(|n| !n.is_empty()) {
            if let Some(Val::Shape(shape)) = values.get(shape_name) {
                let known: Vec<usize> = shape.iter().copied().filter(|d| *d > 0).collect();
                let product: usize = known.iter().product();
                if !known.is_empty() && product != width && !shape.contains(&0) {
                    return Err(format!(
                        "{}: reshape to {shape:?} does not preserve the {width}-wide feature vector",
                        node.label()
                    ));
                }
            }
        }
    }
    if node.op_type == "Reshape" {
        notes.push(format!(
            "{} treated as a no-op on the feature vector",
            node.label()
        ));
    }
    Ok(width)
}

/// Every node's op type, in execution order — useful in the CLI summary.
pub fn op_chain(graph: &Graph) -> Vec<String> {
    graph.nodes.iter().map(|n| n.op_type.clone()).collect()
}
