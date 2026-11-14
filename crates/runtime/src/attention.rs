//! Deterministic Q16.16 attention kernels.
//!
//! This module implements one scaled dot-product attention head and a
//! multi-head wrapper. It is a prototype kernel: it is not wired into
//! [`crate::Model::forward`], the graph loader, the node, or the dispute game.
//!
//! # Layout
//!
//! For [`scaled_dot_product_attention`] and
//! [`scaled_dot_product_attention_shaped`], `q`, `k` and `v` are row-major
//! `[seq_len, head_dim]` matrices of Q16.16 values:
//!
//! ```text
//! q = [ q[0][0] q[0][1] ... q[0][head_dim-1]
//!       q[1][0] q[1][1] ... q[1][head_dim-1]
//!       ...
//!       q[seq_len-1][...] ]
//! ```
//!
//! [`multi_head_attention`] uses `[seq_len, heads, head_dim]` (token-major,
//! then head, then feature inside the head), so the feature width is
//! `heads * head_dim`. The output has the same layout as the input.
//!
//! # Arithmetic
//!
//! * `scores[i][j] = (Σ_d q[i][d] * k[j][d]) >> 16`, using a saturating `i64`
//!   accumulator.
//! * The score is multiplied by a Q16.16 approximation of `1 / sqrt(head_dim)`:
//!   `scale = floor(sqrt(2^32 / head_dim))`, which is within one Q16.16 unit of
//!   the real value. The product uses an `i128` intermediate and is clamped to
//!   `i32`, because [`softmax`](crate::softmax) consumes `i32`.
//! * Each score row is passed through the runtime's integer softmax.
//! * `out[i][d] = (Σ_j p[i][j] * v[j][d]) >> 16`, again with an `i64`
//!   accumulator and an `i32` clamp.
//!
//! No floating point, threads, hash maps or parallelism are used. Loop order is
//! fixed (`i`, then `j`, then `d` for scores; `i`, then `d`, then `j` for the
//! output), so two runs on the same input produce identical bytes. Any future
//! optimization must preserve that summation order and the single final shift.
//!
//! The runtime softmax is not `exp`: it maps a negative offset from the row
//! maximum to `0` and a non-negative offset `d` to `d + ONE`. Attention
//! inherits that approximation. Its rows sum to at most `ONE` because each
//! probability is floor-divided, so the output is a slightly biased-downward
//! weighted average.

use crate::{softmax, ONE};

/// One attention head over row-major `[seq_len, head_dim]` Q16.16 matrices.
///
/// A flat slice carries no row length, so this convenience entry point infers
/// `head_dim` as the largest divisor of the total length that does not exceed
/// `floor(sqrt(total))`, i.e. it prefers `seq_len >= head_dim`. For example, 6
/// values are read as `3 x 2` and 8 values as `4 x 2`. When the row length is
/// known and may be larger than the sequence length, call
/// [`scaled_dot_product_attention_shaped`] directly.
///
/// Mismatched lengths or empty input return an empty vector; use the shaped or
/// multi-head entry point when shape errors must be reported.
pub fn scaled_dot_product_attention(q: &[i32], k: &[i32], v: &[i32]) -> Vec<i32> {
    match infer_single_head_dims(q.len(), k.len(), v.len()) {
        Some((seq_len, head_dim)) => {
            scaled_dot_product_attention_shaped(q, k, v, seq_len, head_dim).unwrap_or_default()
        }
        None => Vec::new(),
    }
}

/// One attention head with an explicit shape.
///
/// All three inputs must contain exactly `seq_len * head_dim` Q16.16 values.
/// A zero `seq_len` produces an empty output; a zero `head_dim`, an overflow,
/// or a length mismatch is an error.
pub fn scaled_dot_product_attention_shaped(
    q: &[i32],
    k: &[i32],
    v: &[i32],
    seq_len: usize,
    head_dim: usize,
) -> Result<Vec<i32>, String> {
    if head_dim == 0 {
        return Err("head_dim must be greater than zero".into());
    }
    let expected = seq_len
        .checked_mul(head_dim)
        .ok_or_else(|| "seq_len * head_dim overflows".to_string())?;
    if q.len() != expected || k.len() != expected || v.len() != expected {
        return Err(format!(
            "one head needs {expected} values in q, k and v (seq_len {seq_len} x head_dim {head_dim}), got {}, {} and {}",
            q.len(),
            k.len(),
            v.len()
        ));
    }
    if seq_len == 0 {
        return Ok(Vec::new());
    }

    let scale = isqrt((ONE as u64 * ONE as u64) / head_dim as u64) as i64;
    let mut scores = vec![0i32; seq_len];
    let mut out = vec![0i32; expected];

    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        for j in 0..seq_len {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            let mut acc = 0i64;
            for (qv, kv) in q_row.iter().zip(k_row) {
                acc = acc.saturating_add((*qv as i64).saturating_mul(*kv as i64));
            }
            let dot = acc >> 16;
            let scaled = ((dot as i128 * scale as i128) >> 16)
                .clamp(i32::MIN as i128, i32::MAX as i128) as i32;
            scores[j] = scaled;
        }

        let weights = softmax(&scores);
        let out_row = &mut out[i * head_dim..(i + 1) * head_dim];
        for (d, slot) in out_row.iter_mut().enumerate() {
            let mut acc = 0i64;
            for (j, p) in weights.iter().enumerate() {
                acc = acc.saturating_add((*p as i64).saturating_mul(v[j * head_dim + d] as i64));
            }
            *slot = (acc >> 16).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }

    Ok(out)
}

/// Multi-head attention over row-major `[seq_len, heads, head_dim]` Q16.16
/// matrices.
///
/// Every head sees the same sequence and its own `head_dim`-wide slice of each
/// token. Heads are evaluated one after another and their outputs are
/// concatenated back into the `[seq_len, heads, head_dim]` layout. `heads` and
/// `head_dim` must be non-zero, all three inputs must have equal length, and
/// that length must be a multiple of `heads * head_dim`.
pub fn multi_head_attention(
    q: &[i32],
    k: &[i32],
    v: &[i32],
    heads: usize,
    head_dim: usize,
) -> Result<Vec<i32>, String> {
    if heads == 0 {
        return Err("heads must be greater than zero".into());
    }
    if head_dim == 0 {
        return Err("head_dim must be greater than zero".into());
    }
    let width = heads
        .checked_mul(head_dim)
        .ok_or_else(|| "heads * head_dim overflows".to_string())?;
    if q.len() != k.len() || q.len() != v.len() {
        return Err(format!(
            "q, k and v must all have the same length, got {}, {} and {}",
            q.len(),
            k.len(),
            v.len()
        ));
    }
    if !q.len().is_multiple_of(width) {
        return Err(format!(
            "input length {} is not a multiple of heads * head_dim ({heads} * {head_dim} = {width})",
            q.len()
        ));
    }
    let seq_len = q.len() / width;
    if seq_len == 0 {
        return Err("sequence length must be greater than zero".into());
    }

    let per_head = seq_len
        .checked_mul(head_dim)
        .ok_or_else(|| "seq_len * head_dim overflows".to_string())?;
    let mut q_head = vec![0i32; per_head];
    let mut k_head = vec![0i32; per_head];
    let mut v_head = vec![0i32; per_head];
    let mut out = vec![0i32; q.len()];

    for h in 0..heads {
        let offset = h * head_dim;
        for i in 0..seq_len {
            let src = i * width + offset;
            let dst = i * head_dim;
            q_head[dst..dst + head_dim].copy_from_slice(&q[src..src + head_dim]);
            k_head[dst..dst + head_dim].copy_from_slice(&k[src..src + head_dim]);
            v_head[dst..dst + head_dim].copy_from_slice(&v[src..src + head_dim]);
        }

        let head_out =
            scaled_dot_product_attention_shaped(&q_head, &k_head, &v_head, seq_len, head_dim)?;

        for i in 0..seq_len {
            let dst = i * width + offset;
            let src = i * head_dim;
            out[dst..dst + head_dim].copy_from_slice(&head_out[src..src + head_dim]);
        }
    }

    Ok(out)
}

fn infer_single_head_dims(q_len: usize, k_len: usize, v_len: usize) -> Option<(usize, usize)> {
    if q_len == 0 || q_len != k_len || q_len != v_len {
        return None;
    }
    let mut head_dim = isqrt(q_len as u64) as usize;
    while head_dim > 1 && !q_len.is_multiple_of(head_dim) {
        head_dim -= 1;
    }
    Some((q_len / head_dim, head_dim))
}

/// Largest `s` with `s * s <= n`. Integer-only, so it is safe on-chain.
fn isqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let mut lo = 1u64;
    let mut hi = 4_294_967_296u64;
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if mid <= n / mid {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_attention_returns_mean_of_v() {
        let q = [0i32; 4];
        let k = [0i32; 4];
        let v = [2 * ONE, 2 * ONE, 4 * ONE, 4 * ONE];

        assert_eq!(
            scaled_dot_product_attention(&q, &k, &v),
            vec![3 * ONE, 3 * ONE, 3 * ONE, 3 * ONE]
        );
    }

    #[test]
    fn one_hot_key_selects_matching_v_row() {
        let q = [ONE, 0, ONE, 0, ONE, 0];
        let k = [ONE, 0, 0, ONE, 0, 0];
        let v = [ONE, 2 * ONE, 3 * ONE, 4 * ONE, 5 * ONE, 6 * ONE];

        let out = scaled_dot_product_attention(&q, &k, &v);
        assert_eq!(out, vec![ONE, 2 * ONE, ONE, 2 * ONE, ONE, 2 * ONE]);
    }

    #[test]
    fn multi_head_zero_weights_averages_each_head_independently() {
        let q = [0i32; 8];
        let k = [0i32; 8];
        // [seq_len=2][heads=2][head_dim=2]
        let v = [
            2 * ONE,
            2 * ONE,
            8 * ONE,
            8 * ONE,
            4 * ONE,
            4 * ONE,
            12 * ONE,
            12 * ONE,
        ];

        let out = multi_head_attention(&q, &k, &v, 2, 2).unwrap();
        assert_eq!(
            out,
            vec![
                3 * ONE,
                3 * ONE,
                10 * ONE,
                10 * ONE,
                3 * ONE,
                3 * ONE,
                10 * ONE,
                10 * ONE,
            ]
        );
    }

    #[test]
    fn multi_head_identity_v_returns_the_attention_weights() {
        let q = [0i32; 8];
        let k = [0i32; 8];
        // head 0 sees the identity, head 1 sees twice the identity
        let v = [ONE, 0, 2 * ONE, 0, 0, ONE, 0, 2 * ONE];

        let out = multi_head_attention(&q, &k, &v, 2, 2).unwrap();
        assert_eq!(
            out,
            vec![ONE / 2, ONE / 2, ONE, ONE, ONE / 2, ONE / 2, ONE, ONE,]
        );
    }

    #[test]
    fn shape_errors_are_rejected() {
        let q = [0i32; 4];
        let k = [0i32; 4];
        let v = [0i32; 3];
        assert!(multi_head_attention(&q, &k, &v, 1, 2)
            .unwrap_err()
            .contains("same length"));

        let q = [0i32; 6];
        let k = [0i32; 6];
        let v = [0i32; 6];
        assert!(multi_head_attention(&q, &k, &v, 2, 2)
            .unwrap_err()
            .contains("multiple"));
        assert!(multi_head_attention(&q, &k, &v, 0, 2)
            .unwrap_err()
            .contains("heads"));
        assert!(multi_head_attention(&q, &k, &v, 2, 0)
            .unwrap_err()
            .contains("head_dim"));
        assert!(multi_head_attention(&[], &[], &[], 1, 2)
            .unwrap_err()
            .contains("sequence length"));

        assert!(scaled_dot_product_attention_shaped(&q, &k, &v, 4, 2).is_err());
        assert!(scaled_dot_product_attention(&q, &k, &v[..2]).is_empty());
    }

    #[test]
    fn attention_is_deterministic() {
        let q: Vec<i32> = (0..8).map(|i| (i - 3) * (ONE / 5)).collect();
        let k: Vec<i32> = (0..8).map(|i| (2 - (i % 4)) * (ONE / 7)).collect();
        let v: Vec<i32> = (0..8).map(|i| (i * 3 - 5) * (ONE / 11)).collect();

        let first = multi_head_attention(&q, &k, &v, 2, 2).unwrap();
        let second = multi_head_attention(&q, &k, &v, 2, 2).unwrap();
        assert_eq!(first, second);

        // 8 values infer as [seq_len=4, head_dim=2].
        assert_eq!(
            scaled_dot_product_attention(&q, &k, &v),
            scaled_dot_product_attention_shaped(&q, &k, &v, 4, 2).unwrap()
        );
    }
}
