//! The subset of the ONNX protobuf schema an importer needs.
//!
//! Field numbers come from `onnx/onnx.proto` (ModelProto, GraphProto,
//! NodeProto, AttributeProto, TensorProto, ValueInfoProto). Everything else in
//! the file is skipped rather than rejected, so a model exported by a newer
//! toolchain still imports as long as it uses the ops we support.

use std::collections::BTreeMap;

use crate::wire::{fields, push_packed_floats, push_packed_ints};

pub const DT_FLOAT: i64 = 1;
pub const DT_INT32: i64 = 6;
pub const DT_INT64: i64 = 7;
pub const DT_DOUBLE: i64 = 11;

#[derive(Debug, Clone, Default)]
pub struct Tensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub data_type: i64,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
}

impl Tensor {
    pub fn element_count(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn is_float(&self) -> bool {
        matches!(self.data_type, DT_FLOAT | DT_DOUBLE)
    }

    pub fn is_integer(&self) -> bool {
        matches!(self.data_type, DT_INT64 | DT_INT32)
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

#[derive(Debug, Clone)]
pub enum Attr {
    Float(f32),
    Int(i64),
    /// String attributes ("auto_pad", "mode", …).
    Text(String),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Tensor(Tensor),
}

#[derive(Debug, Clone)]
pub struct Node {
    pub op_type: String,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: BTreeMap<String, Attr>,
}

impl Node {
    pub fn int(&self, key: &str, default: i64) -> i64 {
        match self.attrs.get(key) {
            Some(Attr::Int(v)) => *v,
            _ => default,
        }
    }

    pub fn float(&self, key: &str, default: f32) -> f32 {
        match self.attrs.get(key) {
            Some(Attr::Float(v)) => *v,
            _ => default,
        }
    }

    /// Description used in error messages: `Gemm "linear_1"`.
    pub fn label(&self) -> String {
        if self.name.is_empty() {
            self.op_type.clone()
        } else {
            format!("{} \"{}\"", self.op_type, self.name)
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub elem_type: i64,
    /// `None` for a symbolic or dynamic dimension.
    pub shape: Vec<Option<usize>>,
}

#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub initializers: BTreeMap<String, Tensor>,
    pub inputs: Vec<TensorInfo>,
    pub outputs: Vec<TensorInfo>,
}

#[derive(Debug, Clone, Default)]
pub struct Model {
    pub ir_version: i64,
    pub producer: String,
    pub opset: i64,
    pub graph: Graph,
}

pub fn parse_model(bytes: &[u8]) -> Result<Model, String> {
    let mut model = Model::default();
    for field in fields(bytes)? {
        match (field.number, field.wire) {
            (1, 0) => model.ir_version = field.varint as i64,
            (2, 2) => model.producer = field.string(),
            (7, 2) => model.graph = parse_graph(field.bytes)?,
            (8, 2) => {
                // OperatorSetIdProto { domain = 1, version = 2 }
                for f in fields(field.bytes)? {
                    if f.number == 2 && f.wire == 0 {
                        model.opset = f.varint as i64;
                    }
                }
            }
            _ => {}
        }
    }
    if model.graph.nodes.is_empty() && model.graph.initializers.is_empty() {
        return Err("file contains no ONNX graph (is it a checkpoint rather than a model?)".into());
    }
    Ok(model)
}

fn parse_graph(bytes: &[u8]) -> Result<Graph, String> {
    let mut graph = Graph::default();
    for field in fields(bytes)? {
        match (field.number, field.wire) {
            (1, 2) => graph.nodes.push(parse_node(field.bytes)?),
            (5, 2) => {
                let tensor = parse_tensor(field.bytes)?;
                if tensor.name.is_empty() {
                    return Err("initializer without a name".into());
                }
                if graph
                    .initializers
                    .insert(tensor.name.clone(), tensor)
                    .is_some()
                {
                    return Err("duplicate initializer name".into());
                }
            }
            (11, 2) => graph.inputs.push(parse_value_info(field.bytes)?),
            (12, 2) => graph.outputs.push(parse_value_info(field.bytes)?),
            _ => {}
        }
    }
    Ok(graph)
}

fn parse_node(bytes: &[u8]) -> Result<Node, String> {
    let mut node = Node {
        op_type: String::new(),
        name: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        attrs: BTreeMap::new(),
    };
    for field in fields(bytes)? {
        match (field.number, field.wire) {
            (1, 2) => node.inputs.push(field.string()),
            (2, 2) => node.outputs.push(field.string()),
            (3, 2) => node.name = field.string(),
            (4, 2) => node.op_type = field.string(),
            (5, 2) => {
                let (name, attr) = parse_attribute(field.bytes)?;
                node.attrs.insert(name, attr);
            }
            _ => {}
        }
    }
    if node.op_type.is_empty() {
        return Err("node without an op_type".into());
    }
    Ok(node)
}

fn parse_attribute(bytes: &[u8]) -> Result<(String, Attr), String> {
    let mut name = String::new();
    let mut value = Attr::Int(0);
    let mut floats = Vec::new();
    let mut ints = Vec::new();
    for field in fields(bytes)? {
        match (field.number, field.wire) {
            (1, 2) => name = field.string(),
            (2, 5) => value = Attr::Float(field.f32()),
            (3, 0) => value = Attr::Int(field.varint as i64),
            (4, 2) => value = Attr::Text(field.string()),
            (5, 2) => value = Attr::Tensor(parse_tensor(field.bytes)?),
            (7, 2) => push_packed_floats(field.bytes, &mut floats)?,
            (7, 5) => floats.push(field.f32()),
            (8, 2) => push_packed_ints(field.bytes, &mut ints)?,
            (8, 0) => ints.push(field.varint as i64),
            _ => {}
        }
    }
    if !floats.is_empty() {
        value = Attr::Floats(floats);
    }
    if !ints.is_empty() {
        value = Attr::Ints(ints);
    }
    Ok((name, value))
}

fn parse_value_info(bytes: &[u8]) -> Result<TensorInfo, String> {
    let mut info = TensorInfo {
        name: String::new(),
        elem_type: 0,
        shape: Vec::new(),
    };
    for field in fields(bytes)? {
        match (field.number, field.wire) {
            (1, 2) => info.name = field.string(),
            // TypeProto { tensor_type = 1 }
            (2, 2) => {
                for t in fields(field.bytes)? {
                    if t.number == 1 && t.wire == 2 {
                        // TypeProto.Tensor { elem_type = 1, shape = 2 }
                        for tf in fields(t.bytes)? {
                            match (tf.number, tf.wire) {
                                (1, 0) => info.elem_type = tf.varint as i64,
                                (2, 2) => {
                                    // TensorShapeProto { dim = 1 }
                                    for dim in fields(tf.bytes)? {
                                        if dim.number != 1 || dim.wire != 2 {
                                            continue;
                                        }
                                        let mut value = None;
                                        for d in fields(dim.bytes)? {
                                            // Dimension { dim_value = 1, dim_param = 2 }
                                            if d.number == 1 && d.wire == 0 {
                                                value = Some(d.varint as usize);
                                            }
                                        }
                                        info.shape.push(value);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(info)
}

fn parse_tensor(bytes: &[u8]) -> Result<Tensor, String> {
    let mut tensor = Tensor::default();
    let mut raw: Option<Vec<u8>> = None;
    let mut dims: Vec<i64> = Vec::new();
    for field in fields(bytes)? {
        match (field.number, field.wire) {
            (1, 2) => push_packed_ints(field.bytes, &mut dims)?,
            (1, 0) => dims.push(field.varint as i64),
            (2, 0) => tensor.data_type = field.varint as i64,
            (4, 2) => push_packed_floats(field.bytes, &mut tensor.floats)?,
            (4, 5) => tensor.floats.push(field.f32()),
            (5, 2) => push_packed_ints(field.bytes, &mut tensor.ints)?,
            (5, 0) => tensor.ints.push(field.varint as i32 as i64),
            (7, 2) => push_packed_ints(field.bytes, &mut tensor.ints)?,
            (7, 0) => tensor.ints.push(field.varint as i64),
            (8, 2) => tensor.name = field.string(),
            (9, 2) => raw = Some(field.bytes.to_vec()),
            _ => {}
        }
    }
    tensor.dims = dims
        .into_iter()
        .map(|d| if d < 0 { 0 } else { d as usize })
        .collect();
    if let Some(raw) = raw {
        match tensor.data_type {
            DT_FLOAT => push_packed_floats(&raw, &mut tensor.floats)?,
            DT_DOUBLE => {
                if !raw.len().is_multiple_of(8) {
                    return Err(format!(
                        "tensor `{}`: raw double data is truncated",
                        tensor.name
                    ));
                }
                for chunk in raw.as_chunks::<8>().0 {
                    tensor.floats.push(f64::from_le_bytes(*chunk) as f32);
                }
            }
            DT_INT64 => {
                if !raw.len().is_multiple_of(8) {
                    return Err(format!(
                        "tensor `{}`: raw int64 data is truncated",
                        tensor.name
                    ));
                }
                for chunk in raw.as_chunks::<8>().0 {
                    tensor.ints.push(i64::from_le_bytes(*chunk));
                }
            }
            DT_INT32 => {
                if !raw.len().is_multiple_of(4) {
                    return Err(format!(
                        "tensor `{}`: raw int32 data is truncated",
                        tensor.name
                    ));
                }
                for chunk in raw.as_chunks::<4>().0 {
                    tensor.ints.push(i32::from_le_bytes(*chunk) as i64);
                }
            }
            other => {
                return Err(format!(
                    "tensor `{}`: unsupported data type {other} (float and int tensors only)",
                    tensor.name
                ))
            }
        }
    }
    if !tensor.dims.is_empty() || tensor.data_type != 0 {
        let want = tensor.element_count();
        let have = if tensor.is_float() {
            tensor.floats.len()
        } else {
            tensor.ints.len()
        };
        if want != have {
            return Err(format!(
                "tensor `{}`: dims {:?} need {want} values but {} are present",
                tensor.name, tensor.dims, have
            ));
        }
    }
    Ok(tensor)
}
