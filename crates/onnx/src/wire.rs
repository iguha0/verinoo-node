//! Minimal protobuf wire-format reader.
//!
//! An ONNX file is a protobuf `ModelProto`, but we need only a handful of its
//! message types. Decoding the wire format directly keeps the importer free of
//! a build-time `protoc` dependency and of a serialization framework, and makes
//! it obvious what is read and what is ignored. Every read is bounds-checked:
//! a truncated or hostile file becomes an error, never a panic or an over-read.

/// One decoded field. Exactly one of `varint` / `bytes` carries the value,
/// depending on the wire type.
#[derive(Debug, Clone, Copy)]
pub struct Field<'a> {
    pub number: u32,
    /// 0 = varint, 1 = 64-bit, 2 = length-delimited, 5 = 32-bit.
    pub wire: u8,
    pub varint: u64,
    pub bytes: &'a [u8],
}

impl<'a> Field<'a> {
    pub fn f32(&self) -> f32 {
        f32::from_le_bytes([self.bytes[0], self.bytes[1], self.bytes[2], self.bytes[3]])
    }

    pub fn f64(&self) -> f64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(self.bytes);
        f64::from_le_bytes(b)
    }

    pub fn string(&self) -> String {
        String::from_utf8_lossy(self.bytes).into_owned()
    }
}

pub fn varint_at(buf: &[u8], mut pos: usize) -> Result<(u64, usize), String> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *buf.get(pos).ok_or_else(|| "truncated varint".to_string())?;
        pos += 1;
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((result, pos));
        }
        shift += 7;
        if shift >= 64 {
            return Err("varint is longer than 64 bits".into());
        }
    }
}

/// Split a message into its fields. Unknown field numbers are returned too;
/// callers ignore the ones they do not care about, which is what protobuf
/// forward compatibility expects.
pub fn fields(buf: &[u8]) -> Result<Vec<Field<'_>>, String> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < buf.len() {
        let (key, next) = varint_at(buf, pos)?;
        pos = next;
        let number = (key >> 3) as u32;
        let wire = (key & 7) as u8;
        if number == 0 || number > 536_870_911 {
            return Err(format!("invalid field number {number}"));
        }
        match wire {
            0 => {
                let (value, next) = varint_at(buf, pos)?;
                pos = next;
                out.push(Field {
                    number,
                    wire,
                    varint: value,
                    bytes: &[],
                });
            }
            1 => {
                let end = pos + 8;
                if end > buf.len() {
                    return Err("truncated 64-bit field".into());
                }
                out.push(Field {
                    number,
                    wire,
                    varint: u64::from_le_bytes(buf[pos..end].try_into().unwrap()),
                    bytes: &buf[pos..end],
                });
                pos = end;
            }
            2 => {
                let (len, next) = varint_at(buf, pos)?;
                pos = next;
                let end = pos
                    .checked_add(len as usize)
                    .ok_or_else(|| "length overflows address space".to_string())?;
                if end > buf.len() {
                    return Err("truncated length-delimited field".into());
                }
                out.push(Field {
                    number,
                    wire,
                    varint: len,
                    bytes: &buf[pos..end],
                });
                pos = end;
            }
            5 => {
                let end = pos + 4;
                if end > buf.len() {
                    return Err("truncated 32-bit field".into());
                }
                out.push(Field {
                    number,
                    wire,
                    varint: u32::from_le_bytes(buf[pos..end].try_into().unwrap()) as u64,
                    bytes: &buf[pos..end],
                });
                pos = end;
            }
            other => {
                return Err(format!(
                    "unsupported protobuf wire type {other} in field {number}"
                ))
            }
        }
    }
    Ok(out)
}

/// Repeated integer fields may be packed (wire type 2) or repeated one by one
/// (wire type 0). Both are valid protobuf and both appear in ONNX in the wild.
pub fn push_packed_ints(bytes: &[u8], out: &mut Vec<i64>) -> Result<(), String> {
    let mut pos = 0usize;
    while pos < bytes.len() {
        let (value, next) = varint_at(bytes, pos)?;
        pos = next;
        out.push(value as i64);
    }
    Ok(())
}

pub fn push_packed_floats(bytes: &[u8], out: &mut Vec<f32>) -> Result<(), String> {
    if !bytes.len().is_multiple_of(4) {
        return Err("packed float array is not a multiple of 4 bytes".into());
    }
    for chunk in bytes.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*chunk));
    }
    Ok(())
}
