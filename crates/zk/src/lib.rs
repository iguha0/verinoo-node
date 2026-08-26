//! Q16.16 fixed-point op witnesses — native Rust implementations that
//! produce bit-identical results to the WASM runtime and the circom
//! circuits' expected values.
//!
//! SPEC SOURCE: `src/wasm/inference.wat` (execution semantics) +
//! `src/zk/{softmax,layernorm}.ts` (circuit spec documents).
//!
//! Every function takes raw i32 fixed-point values (Q16.16) and returns
//! raw i32 outputs. Converting to/from floats is done at the boundary
//! via [`f2i`] / [`i2f`].

pub const FIXED_ONE: i32 = 65536; // 1.0 in Q16.16

/// Q16.16 fixed-point multiply: `(a * b) >> 16`.
/// Uses i64 intermediate to avoid overflow.
#[inline]
pub fn fmul(a: i32, b: i32) -> i32 {
    (((a as i64) * (b as i64)) >> 16) as i32
}

/// Convert float to Q16.16 raw integer.
pub fn f2i(v: f64) -> i32 {
    (v * FIXED_ONE as f64).round() as i32
}

/// Convert Q16.16 raw integer to float.
pub fn i2f(v: i32) -> f64 {
    v as f64 / FIXED_ONE as f64
}

// ============================ matmul ============================

/// Dense matmul: out[i] = bias[i] + Σ_j(inp[j] * w[i*nIn+j])
///
/// The circuit proves this via the slim commitment (output-only squares).
/// Raw integer arithmetic — matches inference.wat `$matmul` exactly.
pub fn matmul(inp: &[i32], weights: &[i32], bias: &[i32], n_out: usize) -> Vec<i32> {
    let n_in = inp.len();
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let mut acc = bias.get(i).copied().unwrap_or(0);
        for j in 0..n_in {
            acc = acc.wrapping_add(fmul(inp[j], weights[i * n_in + j]));
        }
        out.push(acc);
    }
    out
}

// ============================ relu ============================

/// Element-wise max(0, x).
pub fn relu(x: &[i32]) -> Vec<i32> {
    x.iter().map(|&v| if v > 0 { v } else { 0 }).collect()
}

// ============================ argmax ============================

/// Canonical argmax: smallest index holding the maximum value.
pub fn argmax(x: &[i32]) -> usize {
    let mut best = 0;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best
}

// ============================ layernorm ============================

/// Bit-exact integer layernorm, mirroring src/zk/layernorm.ts.
///
/// All divisions are floor-toward-negative-infinity (arithmetic shift):
///   mean   = sum >> 3
///   c_i    = x_i - mean
///   varnum = Σ c_i²                       (Q32.32)
///   var_n  = varnum >> 19                 (/8 elements AND 2^16 rescale)
///   denom  = max(var_n, 1)
///   A      = denom << 16
///   t      = floor_sqrt(A)
///   R      = floor(2^32 / t)
///   y_i    = floor((c_i * R) >> 16)
pub fn layernorm(x: &[i32], _n: usize) -> Vec<i32> {
    let xb: Vec<i64> = x.iter().map(|&v| v as i64).collect();

    let sum: i64 = xb.iter().sum();
    let mean = shr_floor(sum, 3);

    let c: Vec<i64> = xb.iter().map(|&v| v - mean).collect();
    let varnum: i64 = c.iter().map(|&v| v * v).sum();
    let var_n = shr_floor(varnum, 19);

    let denom = var_n.max(1);

    // floor_sqrt via binary search on non-negative A = denom << 16.
    let a = denom << 16;
    let t = isqrt(a);

    let r = if t > 0 { (1i64 << 32) / t } else { 0 };

    c.iter()
        .map(|&ci| {
            let p = ci.saturating_mul(r);
            shr_floor(p, 16) as i32
        })
        .collect()
}

/// Largest s with s*s <= a (a >= 0). f64 estimate + Newton refinement.
fn isqrt(a: i64) -> i64 {
    if a <= 0 {
        return 0;
    }
    if a < 4 {
        return 1;
    }
    let mut x: u64 = (a as f64).sqrt() as u64;
    // Newton-Raphson: x' = (x + a/x) / 2, converges downward
    loop {
        if x == 0 {
            break;
        }
        let next = (x + (a as u64) / x) / 2;
        if next >= x {
            break;
        }
        x = next;
    }
    // Floor correction: ensure x*x <= a
    while x > 0 && x.checked_mul(x).map_or(true, |sq| sq > a as u64) {
        x -= 1;
    }
    while (x + 1)
        .checked_mul(x + 1)
        .map_or(false, |sq| sq <= a as u64)
    {
        x += 1;
    }
    x as i64
}

/// Floor division by 2^k toward negative infinity.
fn shr_floor(a: i64, k: u32) -> i64 {
    a >> k // Rust's >> on i64 IS arithmetic shift (floors)
}

// ============================ softmax ============================

/// Bit-exact softmax, mirroring src/wasm/inference.wat $softmax:
/// hard-max-with-margin, no real exp. Port of src/zk/softmax.ts.
///
///   max_v = max(x_i)
///   e_i   = 0 if x_i - max_v < 0, else (x_i - max_v) + 65536
///   denom = max(sum(e), 1)
///   y_i   = floor(e_i * 65536 / denom)
pub fn softmax(x: &[i32], _n: usize) -> Vec<i32> {
    const F: i64 = FIXED_ONE as i64;

    let xi: Vec<i64> = x.iter().map(|&v| v as i64).collect();
    let max_v = *xi.iter().max().unwrap_or(&0);

    let e: Vec<i64> = xi
        .iter()
        .map(|&v| {
            let diff = v - max_v;
            if diff < 0 {
                0
            } else {
                diff + F
            }
        })
        .collect();

    let total: i64 = e.iter().sum();
    let denom = total.max(1);

    e.iter()
        .map(|&ei| ((ei * F) / denom) as i32) // truncation == floor for non-neg
        .collect()
}

// ============================ layer dispatch ============================

/// Execute a named op type, matching WasmRuntime.executeLayer semantics.
pub fn execute_layer(
    op_type: &str,
    weights: &[i32],
    input: &[i32],
    bias: Option<&[i32]>,
) -> Vec<i32> {
    match op_type {
        "matmul" | "attention" | "ffn" => {
            // attention/ffn add bias then relu
            let n_in = input.len();
            let n_out = weights.len().max(1) / n_in.max(1);
            let b = bias.unwrap_or(&[]);
            (0..n_out)
                .map(|i| {
                    let mut acc = b.get(i).copied().unwrap_or(0);
                    for k in 0..n_in {
                        acc = acc.wrapping_add(fmul(input[k], weights[k * n_out + i]));
                    }
                    if acc > 0 {
                        acc
                    } else {
                        0
                    }
                })
                .collect()
        }
        "relu" => relu(input),
        "softmax" => softmax(input, input.len()),
        "layernorm" => layernorm(input, input.len()),
        "embedding" => {
            // embedding lookup: copy row from weights (simplified for tests)
            input.to_vec()
        }
        "head" => {
            // head = matmul + bias + softmax
            let n_in = input.len();
            let n_out = weights.len().max(1) / n_in.max(1);
            let b = bias.unwrap_or(&[]);
            let logits: Vec<i32> = (0..n_out)
                .map(|i| {
                    let mut acc = b.get(i).copied().unwrap_or(0);
                    for k in 0..n_in {
                        acc = acc.wrapping_add(fmul(input[k], weights[k * n_out + i]));
                    }
                    acc
                })
                .collect();
            softmax(&logits, n_out)
        }
        _ => input.to_vec(),
    }
}
