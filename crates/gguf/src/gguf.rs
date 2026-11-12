//! Reader for the GGUF container and the ggml tensor formats inside it.
//!
//! GGUF is what llama.cpp writes: a header, a metadata key/value block, a table
//! of named tensors, then the tensor data (32-byte aligned by default). It
//! carries **no graph** — only tensor names and shapes — so an importer has to
//! be told which tensors form the network. That plan lives in `convert.rs`.
//!
//! Two properties matter for consensus: a file is parsed with bounds checks (a
//! truncated or hostile file is an error, never a panic), and quantised data is
//! dequantised with the reference block layout so the integers committed on-chain
//! do not depend on which node did the conversion.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub const MAGIC: &[u8; 4] = b"GGUF";
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// ggml tensor types. Names follow `ggml.h`.
pub fn type_name(kind: u32) -> &'static str {
    match kind {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        30 => "BF16",
        _ => "unknown",
    }
}

/// Bytes per block (or per element for unquantised types).
pub fn type_size(kind: u32) -> Option<usize> {
    match kind {
        0 => Some(4),    // F32
        1 => Some(2),    // F16
        2 => Some(18),   // Q4_0
        3 => Some(20),   // Q4_1
        6 => Some(22),   // Q5_0
        7 => Some(24),   // Q5_1
        8 => Some(34),   // Q8_0
        10 => Some(84),  // Q2_K
        11 => Some(110), // Q3_K
        12 => Some(144), // Q4_K
        13 => Some(176), // Q5_K
        14 => Some(210), // Q6_K
        15 => Some(292), // Q8_K
        30 => Some(2),   // BF16
        _ => None,
    }
}

/// Elements per block; 1 for unquantised types.
pub fn block_elements(kind: u32) -> usize {
    match kind {
        2 | 3 | 6 | 7 | 8 => 32,
        10..=15 => 256,
        _ => 1,
    }
}

/// QK_K: the super-block width used by the K-quants.
const QK_K: usize = 256;

fn read_f16(bytes: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]))
}

fn read_i8(bytes: &[u8], off: usize) -> i8 {
    bytes[off] as i8
}

/// Reference layout for the 6-bit scale/min pairs in Q4_K and Q5_K.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Value::U32(v) => Some(*v),
            Value::U64(v) => u32::try_from(*v).ok(),
            Value::I32(v) => u32::try_from(*v).ok(),
            Value::I64(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U32(v) => Some(*v as u64),
            Value::U64(v) => Some(*v),
            Value::I64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Value::U8(_) => "u8",
            Value::I8(_) => "i8",
            Value::U16(_) => "u16",
            Value::I16(_) => "i16",
            Value::U32(_) => "u32",
            Value::I32(_) => "i32",
            Value::F32(_) => "f32",
            Value::Bool(_) => "bool",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::U64(_) => "u64",
            Value::I64(_) => "i64",
            Value::F64(_) => "f64",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml dimensions: `dims[0]` varies fastest, so a matrix is stored
    /// `dims[0] = columns (input width)`, `dims[1] = rows (output width)`.
    pub dims: Vec<usize>,
    pub kind: u32,
    pub offset: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn is_matrix(&self) -> bool {
        self.dims.len() == 2
    }

    /// `(input_width, output_width)` for a 2-D weight tensor.
    pub fn matrix_shape(&self) -> Option<(usize, usize)> {
        if self.dims.len() == 2 {
            Some((self.dims[0], self.dims[1]))
        } else {
            None
        }
    }

    pub fn shape_string(&self) -> String {
        let dims: Vec<String> = self.dims.iter().map(|d| d.to_string()).collect();
        format!("[{}] {}", dims.join(" × "), type_name(self.kind))
    }
}

#[derive(Debug)]
pub struct Gguf {
    path: PathBuf,
    file: File,
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    pub data_start: u64,
}

fn truncated(what: &str) -> String {
    format!("file ends while reading {what}")
}

impl Gguf {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path).map_err(|e| format!("opening {}: {e}", path.display()))?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|_| "file is shorter than the GGUF magic".to_string())?;
        if &magic != MAGIC {
            return Err(format!(
                "not a GGUF file: magic is {:?}, expected \"GGUF\" \
                 (an ONNX file should be imported with onnx-import)",
                String::from_utf8_lossy(&magic)
            ));
        }
        let version = read_u32(&mut file)?;
        if !(2..=3).contains(&version) {
            return Err(format!(
                "unsupported GGUF version {version} (this importer reads versions 2 and 3)"
            ));
        }
        let tensor_count = read_u64(&mut file)?;
        let kv_count = read_u64(&mut file)?;
        // A hostile header must not make us allocate unbounded memory.
        if tensor_count > 1_000_000 || kv_count > 1_000_000 {
            return Err(format!(
                "implausible header: {tensor_count} tensors, {kv_count} metadata entries"
            ));
        }

        let mut metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = read_string(&mut file)?;
            let value = read_value(&mut file, 0)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_string(&mut file)?;
            let n_dims = read_u32(&mut file)?;
            if n_dims == 0 || n_dims > 4 {
                return Err(format!(
                    "tensor `{name}` declares {n_dims} dimensions (1–4 supported)"
                ));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let d = read_u64(&mut file)?;
                if d > 1 << 40 {
                    return Err(format!("tensor `{name}` has an implausible dimension {d}"));
                }
                dims.push(d as usize);
            }
            let kind = read_u32(&mut file)?;
            let offset = read_u64(&mut file)?;
            tensors.push(TensorInfo {
                name,
                dims,
                kind,
                offset,
            });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .map(|v| v as u64)
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(format!("invalid general.alignment {alignment}"));
        }

        let header_end = file
            .stream_position()
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        let data_start = header_end.div_ceil(alignment) * alignment;

        let gguf = Gguf {
            path,
            file,
            version,
            metadata,
            tensors,
            alignment,
            data_start,
        };
        gguf.check_bounds()?;
        Ok(gguf)
    }

    /// Every tensor must fit inside the file. Checked once at open time so a
    /// truncated download is rejected before any data is read.
    fn check_bounds(&self) -> Result<(), String> {
        let len = self
            .file
            .metadata()
            .map_err(|e| format!("stat {}: {e}", self.path.display()))?
            .len();
        for t in &self.tensors {
            let Some(size) = type_size(t.kind) else {
                continue; // refused later, with the tensor name
            };
            let blocks = t.elements().div_ceil(block_elements(t.kind));
            let bytes = (blocks as u64)
                .checked_mul(size as u64)
                .ok_or_else(|| format!("tensor `{}` size overflows", t.name))?;
            let end = self
                .data_start
                .checked_add(t.offset)
                .and_then(|s| s.checked_add(bytes))
                .ok_or_else(|| format!("tensor `{}` offset overflows", t.name))?;
            if end > len {
                return Err(format!(
                    "tensor `{}` needs bytes {}..{} but the file is {len} bytes (truncated?)",
                    t.name,
                    self.data_start + t.offset,
                    end
                ));
            }
        }
        Ok(())
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
    }

    pub fn metadata_string(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Tensor names in file order.
    pub fn names(&self) -> Vec<String> {
        self.tensors.iter().map(|t| t.name.clone()).collect()
    }

    /// Read one tensor and dequantise it to f32.
    pub fn read_tensor(&mut self, name: &str) -> Result<Vec<f32>, String> {
        let info = self
            .tensor(name)
            .ok_or_else(|| format!("tensor `{name}` is not in the file"))?
            .clone();
        let Some(size) = type_size(info.kind) else {
            return Err(format!(
                "tensor `{name}` uses ggml type {kind} ({kind_name}), which this importer \
                 cannot dequantise; re-export it as F32, F16, BF16, Q8_0 or a legacy Q4/Q5 format",
                kind = info.kind,
                kind_name = type_name(info.kind)
            ));
        };
        let elements = info.elements();
        let blocks = elements.div_ceil(block_elements(info.kind));
        let bytes = blocks * size;
        let mut buf = vec![0u8; bytes];
        self.file
            .seek(SeekFrom::Start(self.data_start + info.offset))
            .map_err(|e| format!("seeking to tensor `{name}`: {e}"))?;
        self.file
            .read_exact(&mut buf)
            .map_err(|_| format!("reading tensor `{name}`: file is truncated"))?;
        dequantize(info.kind, &buf, elements)
    }
}

/// Convert one tensor's raw bytes to f32 values, using the llama.cpp block
/// layouts.
pub fn dequantize(kind: u32, bytes: &[u8], elements: usize) -> Result<Vec<f32>, String> {
    let per_block = block_elements(kind);
    let size = type_size(kind).ok_or_else(|| format!("unsupported ggml type {kind}"))?;
    let blocks = elements.div_ceil(per_block);
    if bytes.len() < blocks * size {
        return Err(format!(
            "dequantise: need {} bytes for {elements} elements, got {}",
            blocks * size,
            bytes.len()
        ));
    }

    let mut out = Vec::with_capacity(blocks * per_block);
    match kind {
        0 => {
            for chunk in bytes[..elements * 4].as_chunks::<4>().0 {
                out.push(f32::from_le_bytes(*chunk));
            }
        }
        1 => {
            for chunk in bytes[..elements * 2].as_chunks::<2>().0 {
                out.push(f16_to_f32(u16::from_le_bytes(*chunk)));
            }
        }
        30 => {
            for chunk in bytes[..elements * 2].as_chunks::<2>().0 {
                let bits = u16::from_le_bytes(*chunk) as u32;
                out.push(f32::from_bits(bits << 16));
            }
        }
        2 | 3 | 6 | 7 | 8 => {
            for block in bytes.chunks_exact(size) {
                match kind {
                    2 => {
                        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                        let qs = &block[2..18];
                        for &q in qs {
                            out.push(d * ((q & 0x0f) as f32 - 8.0));
                        }
                        for &q in qs {
                            out.push(d * ((q >> 4) as f32 - 8.0));
                        }
                    }
                    3 => {
                        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                        let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                        let qs = &block[4..20];
                        for &q in qs {
                            out.push(d * (q & 0x0f) as f32 + m);
                        }
                        for &q in qs {
                            out.push(d * (q >> 4) as f32 + m);
                        }
                    }
                    6 => {
                        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                        let qh = u32::from_le_bytes(block[2..6].try_into().unwrap());
                        let qs = &block[6..22];
                        for (i, &q) in qs.iter().enumerate() {
                            let v = ((q & 0x0f) as u32) | (((qh >> i) & 1) << 4);
                            out.push(d * (v as f32 - 16.0));
                        }
                        for (i, &q) in qs.iter().enumerate() {
                            let v = ((q >> 4) as u32) | (((qh >> (i + 16)) & 1) << 4);
                            out.push(d * (v as f32 - 16.0));
                        }
                    }
                    7 => {
                        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                        let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                        let qh = u32::from_le_bytes(block[4..8].try_into().unwrap());
                        let qs = &block[8..24];
                        for (i, &q) in qs.iter().enumerate() {
                            let v = ((q & 0x0f) as u32) | (((qh >> i) & 1) << 4);
                            out.push(d * v as f32 + m);
                        }
                        for (i, &q) in qs.iter().enumerate() {
                            let v = ((q >> 4) as u32) | (((qh >> (i + 16)) & 1) << 4);
                            out.push(d * v as f32 + m);
                        }
                    }
                    8 => {
                        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                        for &q in &block[2..34] {
                            out.push(d * (q as i8) as f32);
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
        10 => {
            // Q2_K: scales[16] | qs[64] | d(f16) | dmin(f16)
            for block in bytes.chunks_exact(size) {
                let d = read_f16(block, 80);
                let min = read_f16(block, 82);
                let scales = &block[0..16];
                let qs = &block[16..80];
                let mut is = 0usize;
                for n in 0..2 {
                    let mut shift = 0u32;
                    for _ in 0..4 {
                        let sc = scales[is];
                        is += 1;
                        let dl = d * (sc & 0x0f) as f32;
                        let ml = min * (sc >> 4) as f32;
                        for l in 0..16 {
                            out.push(dl * ((qs[n * 32 + l] >> shift) & 3) as f32 - ml);
                        }
                        let sc = scales[is];
                        is += 1;
                        let dl = d * (sc & 0x0f) as f32;
                        let ml = min * (sc >> 4) as f32;
                        for l in 0..16 {
                            out.push(dl * ((qs[n * 32 + l + 16] >> shift) & 3) as f32 - ml);
                        }
                        shift += 2;
                    }
                }
            }
        }
        11 => {
            // Q3_K: hmask[32] | qs[64] | scales[12] | d(f16)
            for block in bytes.chunks_exact(size) {
                let d_all = read_f16(block, 108);
                let hm = &block[0..32];
                let qs = &block[32..96];
                let raw = &block[96..108];
                let mut aux = [0u32; 4];
                for (i, slot) in aux.iter_mut().take(3).enumerate() {
                    *slot = u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
                }
                let tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & 0x0f0f_0f0f) | (((tmp >> 4) & 0x0303_0303) << 4);
                aux[3] = ((aux[1] >> 4) & 0x0f0f_0f0f) | (((tmp >> 6) & 0x0303_0303) << 4);
                aux[0] = (aux[0] & 0x0f0f_0f0f) | ((tmp & 0x0303_0303) << 4);
                aux[1] = (aux[1] & 0x0f0f_0f0f) | (((tmp >> 2) & 0x0303_0303) << 4);
                let mut scales = [0i8; 16];
                for (i, word) in aux.iter().enumerate() {
                    for (b, byte) in word.to_le_bytes().iter().enumerate() {
                        scales[i * 4 + b] = *byte as i8;
                    }
                }
                let mut is = 0usize;
                let mut m: u8 = 1;
                for n in 0..2 {
                    let mut shift = 0u32;
                    for _ in 0..4 {
                        let dl = d_all * (scales[is] as f32 - 32.0);
                        is += 1;
                        for l in 0..16 {
                            let q = ((qs[n * 32 + l] >> shift) & 3) as i8;
                            let h = if hm[l] & m != 0 { 0 } else { 4 };
                            out.push(dl * (q - h) as f32);
                        }
                        let dl = d_all * (scales[is] as f32 - 32.0);
                        is += 1;
                        for l in 0..16 {
                            let q = ((qs[n * 32 + l + 16] >> shift) & 3) as i8;
                            let h = if hm[l + 16] & m != 0 { 0 } else { 4 };
                            out.push(dl * (q - h) as f32);
                        }
                        shift += 2;
                        m <<= 1;
                    }
                }
            }
        }
        12 => {
            // Q4_K: d(f16) | dmin(f16) | scales[12] | qs[128]
            for block in bytes.chunks_exact(size) {
                let d = read_f16(block, 0);
                let min = read_f16(block, 2);
                let scales = &block[4..16];
                let qs = &block[16..144];
                let mut is = 0usize;
                for j in 0..4 {
                    let (sc, m) = get_scale_min_k4(is, scales);
                    let d1 = d * sc as f32;
                    let m1 = min * m as f32;
                    let (sc, m) = get_scale_min_k4(is + 1, scales);
                    let d2 = d * sc as f32;
                    let m2 = min * m as f32;
                    let q = &qs[j * 32..(j + 1) * 32];
                    for &byte in q {
                        out.push(d1 * (byte & 0x0f) as f32 - m1);
                    }
                    for &byte in q {
                        out.push(d2 * (byte >> 4) as f32 - m2);
                    }
                    is += 2;
                }
            }
        }
        13 => {
            // Q5_K: d(f16) | dmin(f16) | scales[12] | qh[32] | qs[128]
            for block in bytes.chunks_exact(size) {
                let d = read_f16(block, 0);
                let min = read_f16(block, 2);
                let scales = &block[4..16];
                let qh = &block[16..48];
                let ql = &block[48..176];
                let mut is = 0usize;
                let mut u1: u8 = 1;
                let mut u2: u8 = 2;
                for j in 0..4 {
                    let (sc, m) = get_scale_min_k4(is, scales);
                    let d1 = d * sc as f32;
                    let m1 = min * m as f32;
                    let (sc, m) = get_scale_min_k4(is + 1, scales);
                    let d2 = d * sc as f32;
                    let m2 = min * m as f32;
                    for l in 0..32 {
                        let v = (ql[j * 32 + l] & 0x0f) + if qh[l] & u1 != 0 { 16 } else { 0 };
                        out.push(d1 * v as f32 - m1);
                    }
                    for l in 0..32 {
                        let v = (ql[j * 32 + l] >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 };
                        out.push(d2 * v as f32 - m2);
                    }
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        14 => {
            // Q6_K: ql[128] | qh[64] | scales[16] | d(f16)
            for block in bytes.chunks_exact(size) {
                let ql = &block[0..128];
                let qh = &block[128..192];
                let sc = &block[192..208];
                let d = read_f16(block, 208);
                let mut chunk = [0f32; QK_K];
                for n in 0..2 {
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[n * 64 + l] & 0x0f) | ((qh[n * 32 + l] & 3) << 4)) as i8 - 32;
                        let q2 = ((ql[n * 64 + l + 32] & 0x0f) | (((qh[n * 32 + l] >> 2) & 3) << 4))
                            as i8
                            - 32;
                        let q3 =
                            ((ql[n * 64 + l] >> 4) | (((qh[n * 32 + l] >> 4) & 3) << 4)) as i8 - 32;
                        let q4 = ((ql[n * 64 + l + 32] >> 4) | (((qh[n * 32 + l] >> 6) & 3) << 4))
                            as i8
                            - 32;
                        chunk[n * 128 + l] = d * read_i8(sc, n * 8 + is) as f32 * q1 as f32;
                        chunk[n * 128 + l + 32] =
                            d * read_i8(sc, n * 8 + is + 2) as f32 * q2 as f32;
                        chunk[n * 128 + l + 64] =
                            d * read_i8(sc, n * 8 + is + 4) as f32 * q3 as f32;
                        chunk[n * 128 + l + 96] =
                            d * read_i8(sc, n * 8 + is + 6) as f32 * q4 as f32;
                    }
                }
                out.extend_from_slice(&chunk);
            }
        }
        15 => {
            // Q8_K: d(f32) | qs[256] | bsums[16] (bsums are not needed for weights)
            for block in bytes.chunks_exact(size) {
                let d = f32::from_le_bytes(block[0..4].try_into().unwrap());
                for &q in &block[4..260] {
                    out.push(d * (q as i8) as f32);
                }
            }
        }
        other => return Err(format!("unsupported ggml type {other}")),
    }

    if out.len() < elements {
        return Err(format!(
            "dequantise: produced {} values, needed {elements}",
            out.len()
        ));
    }
    out.truncate(elements);
    Ok(out)
}

/// IEEE 754 binary16 → f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let magnitude = if exp == 0 {
        (mant as f32) * 2f32.powi(-24)
    } else if exp == 31 {
        if mant == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + mant as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 {
        -magnitude
    } else {
        magnitude
    }
}

fn read_u32(file: &mut File) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).map_err(|_| truncated("u32"))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(file: &mut File) -> Result<u64, String> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf).map_err(|_| truncated("u64"))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string(file: &mut File) -> Result<String, String> {
    let len = read_u64(file)?;
    if len > 1 << 28 {
        return Err(format!("implausible string length {len} in GGUF header"));
    }
    let mut buf = vec![0u8; len as usize];
    file.read_exact(&mut buf).map_err(|_| truncated("string"))?;
    String::from_utf8(buf).map_err(|_| "GGUF string is not valid UTF-8".to_string())
}

/// `depth` guards against absurdly nested arrays in a hostile file.
fn read_value(file: &mut File, depth: u32) -> Result<Value, String> {
    if depth > 4 {
        return Err("GGUF metadata arrays are nested too deeply".into());
    }
    let kind = read_u32(file)?;
    Ok(match kind {
        0 => Value::U8(read_u8(file)?),
        1 => Value::I8(read_u8(file)? as i8),
        2 => Value::U16(read_u16(file)?),
        3 => Value::I16(read_u16(file)? as i16),
        4 => Value::U32(read_u32(file)?),
        5 => Value::I32(read_u32(file)? as i32),
        6 => Value::F32(f32::from_bits(read_u32(file)?)),
        7 => Value::Bool(read_u8(file)? != 0),
        8 => Value::String(read_string(file)?),
        9 => {
            let elem = read_u32(file)?;
            let count = read_u64(file)?;
            if count > 1_000_000 {
                return Err(format!("implausible metadata array length {count}"));
            }
            let mut items = Vec::with_capacity(count.min(1024) as usize);
            for _ in 0..count {
                items.push(read_array_element(file, elem, depth + 1)?);
            }
            Value::Array(items)
        }
        10 => Value::U64(read_u64(file)?),
        11 => Value::I64(read_u64(file)? as i64),
        12 => Value::F64(f64::from_bits(read_u64(file)?)),
        other => return Err(format!("unknown GGUF metadata value type {other}")),
    })
}

fn read_array_element(file: &mut File, kind: u32, _depth: u32) -> Result<Value, String> {
    Ok(match kind {
        0 => Value::U8(read_u8(file)?),
        1 => Value::I8(read_u8(file)? as i8),
        2 => Value::U16(read_u16(file)?),
        3 => Value::I16(read_u16(file)? as i16),
        4 => Value::U32(read_u32(file)?),
        5 => Value::I32(read_u32(file)? as i32),
        6 => Value::F32(f32::from_bits(read_u32(file)?)),
        7 => Value::Bool(read_u8(file)? != 0),
        8 => Value::String(read_string(file)?),
        9 => return Err("nested arrays in GGUF metadata are not supported".into()),
        10 => Value::U64(read_u64(file)?),
        11 => Value::I64(read_u64(file)? as i64),
        12 => Value::F64(f64::from_bits(read_u64(file)?)),
        other => return Err(format!("unknown GGUF array element type {other}")),
    })
}

fn read_u8(file: &mut File) -> Result<u8, String> {
    let mut buf = [0u8; 1];
    file.read_exact(&mut buf).map_err(|_| truncated("u8"))?;
    Ok(buf[0])
}

fn read_u16(file: &mut File) -> Result<u16, String> {
    let mut buf = [0u8; 2];
    file.read_exact(&mut buf).map_err(|_| truncated("u16"))?;
    Ok(u16::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion_matches_known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x3555) - 0.333_251_95).abs() < 1e-6);
        assert!(f16_to_f32(0x7c00).is_infinite());
    }

    #[test]
    fn q8_0_dequantises_a_block() {
        // d = 0.5, then 32 int8 values 0..31 offset by -16.
        let mut bytes = Vec::new();
        bytes.extend(0x3800u16.to_le_bytes()); // f16 0.5
        for i in 0..32i8 {
            bytes.push(i.wrapping_sub(16) as u8);
        }
        let values = dequantize(8, &bytes, 32).unwrap();
        assert_eq!(values.len(), 32);
        assert_eq!(values[0], 0.5 * -16.0);
        assert_eq!(values[16], 0.0);
        assert_eq!(values[31], 0.5 * 15.0);
    }

    #[test]
    fn q4_0_dequantises_a_block() {
        let mut bytes = Vec::new();
        bytes.extend(0x3c00u16.to_le_bytes()); // f16 1.0
        for i in 0..16u8 {
            bytes.push((i & 0x0f) | ((i & 0x0f) << 4));
        }
        let values = dequantize(2, &bytes, 32).unwrap();
        assert_eq!(values[0], 1.0 * (0.0 - 8.0));
        assert_eq!(values[1], 1.0 * (1.0 - 8.0));
        assert_eq!(values[16], 1.0 * (0.0 - 8.0));
        assert_eq!(values[17], 1.0 * (1.0 - 8.0));
    }

    #[test]
    fn unsupported_types_are_named() {
        assert_eq!(type_name(9), "Q8_1");
        assert!(type_size(9).is_none());
        let err = dequantize(9, &[0; 36], 32).unwrap_err();
        assert!(err.contains("unsupported"), "{err}");
    }
}
