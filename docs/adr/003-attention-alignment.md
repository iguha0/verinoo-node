# ADR 003: Attention/Matmul Q16.16 Pipeline Alignment

## Status: Investigated — deferred pending deeper WASM analysis

## Problem
Rust native matmul produces different results than WASM $matmul for the
same inputs. Root cause is NOT $fdiv (which was fixed) — the divergence
is in accumulation order or intermediate precision.

## Investigation findings

### Confirmed working
- Rust fmul matches BigInt reference exactly ✓
- Column-major weight indexing confirmed correct ✓  
- Individual Q16.16 multiplies verified ✓

### Divergence point
WASM attention output differs from BOTH:
- Rust column-native computation
- Manual BigInt step-by-step calculation

This suggests the issue is inside $matmul's accumulation loop, not in
individual multiplications.

### Hypothesis
$matmul may have an additional bug beyond $fdiv, similar to how $fdiv
was shifting by divisor instead of dividing. The accumulation loop might
have an indexing error or intermediate precision loss.

## Next steps
1. Export individual $fmul results from WASM for specific inputs
2. Compare against Rust fmul outputs step-by-step
3. If they match, the bug is in accumulation logic
4. Fix WAT source, rebuild, regenerate fixtures

## Workaround
Attention/matmul disputes resolve via the hash-based fallback path
(claimed-output equality against deterministic recompute) rather than
SNARK verification. This is functionally correct but slower.
