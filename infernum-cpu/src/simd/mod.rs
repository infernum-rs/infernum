//! SIMD dispatch layer.
//!
//! Provides architecture-specific SIMD kernels: AVX2+FMA on x86-64,
//! NEON on AArch64. No scalar fallback — unsupported platforms are a
//! compile error.
//!
//! Quantized integer dot products (Q8×Q8, Q4×Q8) have two tiers:
//! - **AVX2**: `vpmaddubsw` + `vpmaddwd` (baseline x86-64)
//! - **AVX-512 VNNI**: `vpdpbusd` (Cascade Lake+, Zen 4+) — single-instruction replacement
//!
//! The tier is selected at runtime via `is_x86_feature_detected!`.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("infernum-cpu requires x86-64 (AVX2+FMA) or AArch64 (NEON)");

/// Whether the CPU supports AVX-512 VNNI (runtime-detected, cached).
#[cfg(target_arch = "x86_64")]
fn has_vnni() -> bool {
    use std::sync::OnceLock;
    static HAS_VNNI: OnceLock<bool> = OnceLock::new();
    *HAS_VNNI.get_or_init(|| {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512bw")
    })
}

/// Whether the CPU supports AVX-512F (runtime-detected, cached).
#[cfg(target_arch = "x86_64")]
fn has_avx512f() -> bool {
    use std::sync::OnceLock;
    static HAS_AVX512F: OnceLock<bool> = OnceLock::new();
    *HAS_AVX512F.get_or_init(|| is_x86_feature_detected!("avx512f"))
}

/// Check that the current CPU supports the required SIMD features.
///
/// On AArch64 this always succeeds (NEON is baseline).
/// On x86-64 this checks for AVX2 + FMA at runtime.
///
/// # Errors
/// Returns an error if the CPU lacks required SIMD support.
pub fn check_cpu_support() -> infernum::Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return Err(infernum::Error::Other(
                "CPU backend requires AVX2 + FMA support".into(),
            ));
        }
    }
    Ok(())
}

// ---- Dispatch functions ----

/// Dot product of two f32 slices.
#[inline]
#[must_use]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        avx2::dot_f32(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::dot_f32(a, b)
    }
}

/// Fused multiply-add: `dst[i] += src[i] * scalar`.
#[inline]
pub fn vec_fmadd(dst: &mut [f32], src: &[f32], scalar: f32) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    {
        avx2::vec_axpy(dst, scalar, src);
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::vec_axpy(dst, scalar, src);
    }
}

/// Element-wise addition: `out[i] = a[i] + b[i]`.
#[inline]
pub fn vec_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    {
        avx2::vec_add(a, b, out);
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::vec_add(a, b, out);
    }
}

/// Element-wise in-place addition: `a[i] += b[i]`.
#[inline]
pub fn vec_add_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        avx2::vec_add_inplace(a, b);
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::vec_add_inplace(a, b);
    }
}

/// Scaled accumulate (AXPY): `out[i] += scale * src[i]`.
#[inline]
pub fn vec_axpy(out: &mut [f32], scale: f32, src: &[f32]) {
    debug_assert_eq!(out.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    {
        avx2::vec_axpy(out, scale, src);
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Fallback: scalar
        for (o, s) in out.iter_mut().zip(src.iter()) {
            *o = scale.mul_add(*s, *o);
        }
    }
}

/// Element-wise multiplication: `out[i] = a[i] * b[i]`.
#[inline]
pub fn vec_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    {
        avx2::vec_mul(a, b, out);
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::vec_mul(a, b, out);
    }
}

/// In-place scalar scaling: `a[i] *= scale`.
#[inline]
pub fn vec_scale(a: &mut [f32], scale: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        avx2::vec_scale(a, scale);
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::vec_scale(a, scale);
    }
}

/// Sum of squares: `sum(a[i] * a[i])`.
#[inline]
#[must_use]
pub fn sum_of_squares(a: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        avx2::sum_of_squares(a)
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::sum_of_squares(a)
    }
}

/// SiLU (Swish) activation fused with element-wise multiply: `out[i] = silu(gate[i]) * up[i]`.
///
/// Dispatches to AVX-512F vectorized exp (degree-4 polynomial, 16 floats/iter)
/// when available, otherwise falls back to scalar.
#[inline]
pub fn vec_silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512f() {
            avx512::vec_silu_mul(gate, up, out);
            return;
        }
        avx2::vec_silu_mul(gate, up, out);
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::vec_silu_mul(gate, up, out);
    }
}

/// In-place softmax: subtract max, exp, normalize by 1/sum.
///
/// Dispatches to AVX-512 vectorized exp when available.
#[inline]
pub fn vec_softmax_inplace(data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512f() {
            avx512::vec_softmax_inplace(data);
            return;
        }
    }
    // Scalar fallback.
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in data.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for x in data.iter_mut() {
            *x *= inv;
        }
    }
}

/// Dot product of f32 input with one Q8_0 block (32 int8 values, 1 scale).
///
/// Returns `sum(input[i] * int8[i]) * scale` for a single 32-element block.
/// `quants` contains 32 raw int8 bytes; `scale` is the pre-decoded f16→f32 scale.
#[inline]
#[must_use]
pub fn dot_q8_block(input: &[f32], quants: &[u8], scale: f32) -> f32 {
    debug_assert_eq!(input.len(), 32);
    debug_assert_eq!(quants.len(), 32);
    #[cfg(target_arch = "x86_64")]
    {
        avx2::dot_q8_block(input, quants, scale)
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::dot_q8_block(input, quants, scale)
    }
}

/// Dot product of f32 input with one Q4_0 block (16 packed bytes = 32 int4 values, 1 scale).
///
/// Returns `sum(input[i] * (nibble[i] - 8)) * scale` for a single 32-element block.
/// `packed` contains 16 bytes, each holding two 4-bit unsigned values (low nibble first).
/// Values are bias-corrected by subtracting 8 to get signed range [-8, 7].
#[inline]
#[must_use]
pub fn dot_q4_block(input: &[f32], packed: &[u8], scale: f32) -> f32 {
    debug_assert_eq!(input.len(), 32);
    debug_assert_eq!(packed.len(), 16);
    #[cfg(target_arch = "x86_64")]
    {
        avx2::dot_q4_block(input, packed, scale)
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::dot_q4_block(input, packed, scale)
    }
}

/// Dot product of f32 input with one Q4_1 block (16 packed bytes = 32 int4 values, 1 scale, 1 min).
///
/// Returns `sum(input[i] * nibble[i]) * scale + sum(input[i]) * min` for a single 32-element block.
/// `packed` contains 16 bytes, each holding two 4-bit unsigned values (low nibble first).
/// Unlike Q4_0, values are unsigned [0,15] — no bias subtraction.
#[inline]
#[must_use]
pub fn dot_q4_1_block(input: &[f32], packed: &[u8], scale: f32, min: f32) -> f32 {
    debug_assert_eq!(input.len(), 32);
    debug_assert_eq!(packed.len(), 16);
    #[cfg(target_arch = "x86_64")]
    {
        avx2::dot_q4_1_block(input, packed, scale, min)
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::dot_q4_1_block(input, packed, scale, min)
    }
}

/// Q8_0 row dot product: process all blocks for one neuron in a single call.
///
/// `input` is the full f32 input row (K elements), `quants` is the quantized
/// row (K bytes), `scales` is the per-block scale array (K/32 elements).
#[inline]
#[must_use]
pub fn dot_q8_row(input: &[f32], quants: &[u8], scales: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        avx2::dot_q8_row(input, quants, scales)
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Fallback: loop over blocks using per-block kernel
        let mut acc = 0.0f32;
        for (blk, &scale) in scales.iter().enumerate() {
            let start = blk * 32;
            acc += neon::dot_q8_block(&input[start..start + 32], &quants[start..start + 32], scale);
        }
        acc
    }
}

/// Q4_0 row dot product: process all blocks for one neuron in a single call.
///
/// `input` is the full f32 input row (K elements), `packed` is the packed Q4
/// row (K/2 bytes), `scales` is the per-block scale array (K/32 elements).
#[inline]
#[must_use]
pub fn dot_q4_row(input: &[f32], packed: &[u8], scales: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        avx2::dot_q4_row(input, packed, scales)
    }
    #[cfg(target_arch = "aarch64")]
    {
        let mut acc = 0.0f32;
        for (blk, &scale) in scales.iter().enumerate() {
            let inp_start = blk * 32;
            let p_start = blk * 16;
            acc += neon::dot_q4_block(
                &input[inp_start..inp_start + 32],
                &packed[p_start..p_start + 16],
                scale,
            );
        }
        acc
    }
}

/// Q4_1 row dot product: process all blocks for one neuron in a single call.
///
/// `input` is the full f32 input row (K elements), `packed` is the packed Q4_1
/// row (K/2 bytes), `scales` and `mins` are per-block arrays (K/32 elements).
#[inline]
#[must_use]
pub fn dot_q4_1_row(input: &[f32], packed: &[u8], scales: &[f32], mins: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        avx2::dot_q4_1_row(input, packed, scales, mins)
    }
    #[cfg(target_arch = "aarch64")]
    {
        let mut acc = 0.0f32;
        for (blk, (&scale, &min)) in scales.iter().zip(mins.iter()).enumerate() {
            let inp_start = blk * 32;
            let p_start = blk * 16;
            acc += neon::dot_q4_1_block(
                &input[inp_start..inp_start + 32],
                &packed[p_start..p_start + 16],
                scale,
                min,
            );
        }
        acc
    }
}

/// RMS norm: `out[i] = input[i] * weight[i] * rms_scale`
/// where `rms_scale = 1.0 / sqrt(mean_of_squares + eps)`.
#[inline]
pub fn vec_rmsnorm(input: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let ss = sum_of_squares(input);
    #[allow(clippy::cast_precision_loss)]
    let rms = 1.0 / (ss / input.len() as f32 + eps).sqrt();
    for i in 0..input.len() {
        out[i] = input[i] * rms * weight[i];
    }
}

/// Tiled F32 GEMM: `C[m,n] = sum_k A[m,k] * Bᵀ[n,k]`.
///
/// - `a` is row-major `(M, K)`.
/// - `bt` is row-major `(N, K)` — B transposed.
/// - `c` is row-major `(M, N)`, must be zero-initialized by the caller.
///
/// Dispatches to AVX-512F tiled micro-kernel when available, otherwise
/// falls back to row-by-row dot products.
#[allow(clippy::many_single_char_names)]
pub fn gemm_f32_tiled(a: &[f32], bt: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512f() {
            avx512::gemm_f32_tiled(a, bt, c, m, k, n);
            return;
        }
    }

    // Scalar fallback: row-by-row dot products.
    for row in 0..m {
        let a_row = &a[row * k..(row + 1) * k];
        for col in 0..n {
            let bt_row = &bt[col * k..(col + 1) * k];
            c[row * n + col] = dot_f32(a_row, bt_row);
        }
    }
}

/// Tiled Q8×Q8 GEMM: `output[m, n] = inp_quants[m, k] · wt_quants[n, k]`
/// with per-block scales.
///
/// Both operands are in Q8_0 format (blocks of 32 quant bytes + 1 f32 scale).
/// Dispatches to AVX-512 VNNI tiled micro-kernel when available, otherwise
/// falls back to row-by-row `dot_q8_q8_row`.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
pub fn gemm_q8_tiled(
    output: &mut [f32],
    inp_quants: &[u8],
    inp_scales: &[f32],
    wt_quants: &[u8],
    wt_scales: &[f32],
    m: usize,
    n: usize,
    num_blocks: usize,
    bytes_per_row: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_vnni() {
            avx512::gemm_q8_tiled(
                output,
                inp_quants,
                inp_scales,
                wt_quants,
                wt_scales,
                m,
                n,
                num_blocks,
                bytes_per_row,
            );
            return;
        }
    }

    // Scalar fallback: row-by-row Q8×Q8 dot products.
    for row in 0..m {
        let iq = &inp_quants[row * bytes_per_row..(row + 1) * bytes_per_row];
        let is = &inp_scales[row * num_blocks..(row + 1) * num_blocks];
        for col in 0..n {
            let wq = &wt_quants[col * bytes_per_row..(col + 1) * bytes_per_row];
            let ws = &wt_scales[col * num_blocks..(col + 1) * num_blocks];
            output[row * n + col] = dot_q8_q8_row(iq, is, wq, ws);
        }
    }
}

// ---- Integer dot product dispatch (Q8×Q8, Q4×Q8) ----

/// Quantize a row of f32 values to Q8_0 format (on-the-fly, for integer GEMV).
///
/// For each 32-element block: finds max_abs, computes `scale = max_abs / 127`,
/// stores `round(input / scale)` as int8 and the scale as f32.
///
/// `out_quants` must have length K, `out_scales` must have length K/32.
#[inline]
pub fn quantize_row_q8(input: &[f32], out_quants: &mut [u8], out_scales: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        avx2::quantize_row_q8(input, out_quants, out_scales);
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::quantize_row_q8(input, out_quants, out_scales);
    }
}

/// Q8_0 integer row dot product: weight_q8 × input_q8.
///
/// Both operands are pre-quantized Q8. Dispatches to VNNI or AVX2 integer kernels.
#[inline]
#[must_use]
pub fn dot_q8_q8_row(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_quants: &[u8],
    weight_scales: &[f32],
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_vnni() {
            avx512::dot_q8_q8_row(input_quants, input_scales, weight_quants, weight_scales)
        } else {
            avx2::dot_q8_q8_row(input_quants, input_scales, weight_quants, weight_scales)
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::dot_q8_q8_row(input_quants, input_scales, weight_quants, weight_scales)
    }
}

/// Q4_0 integer row dot product: weight_q4 × input_q8.
///
/// Weight nibbles are unpacked, then integer dot with pre-quantized Q8 input.
#[inline]
#[must_use]
pub fn dot_q4_q8_row(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_packed: &[u8],
    weight_scales: &[f32],
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_vnni() {
            avx512::dot_q4_q8_row(input_quants, input_scales, weight_packed, weight_scales)
        } else {
            avx2::dot_q4_q8_row(input_quants, input_scales, weight_packed, weight_scales)
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::dot_q4_q8_row(input_quants, input_scales, weight_packed, weight_scales)
    }
}

/// Q4_1 integer row dot product: weight_q4_1 × input_q8.
///
/// Uses integer dot for nibble×quant part, f32 for the min correction term.
#[inline]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn dot_q4_1_q8_row(
    input_quants: &[u8],
    input_scales: &[f32],
    input_row: &[f32],
    weight_packed: &[u8],
    weight_scales: &[f32],
    weight_mins: &[f32],
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_vnni() {
            avx512::dot_q4_1_q8_row(
                input_quants,
                input_scales,
                input_row,
                weight_packed,
                weight_scales,
                weight_mins,
            )
        } else {
            avx2::dot_q4_1_q8_row(
                input_quants,
                input_scales,
                input_row,
                weight_packed,
                weight_scales,
                weight_mins,
            )
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::dot_q4_1_q8_row(
            input_quants,
            input_scales,
            input_row,
            weight_packed,
            weight_scales,
            weight_mins,
        )
    }
}

// ---- Multi-row GEMV dispatchers ----

/// 2-row Q8×Q8 GEMV: computes dot products for two adjacent weight rows
/// against the same input vector. Returns `(dot0, dot1)`.
#[inline]
#[must_use]
pub fn dot_q8_q8_2row(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_quants_0: &[u8],
    weight_scales_0: &[f32],
    weight_quants_1: &[u8],
    weight_scales_1: &[f32],
) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_vnni() {
            avx512::dot_q8_q8_2row(
                input_quants,
                input_scales,
                weight_quants_0,
                weight_scales_0,
                weight_quants_1,
                weight_scales_1,
            )
        } else {
            // Fallback: two sequential single-row calls
            let d0 =
                avx2::dot_q8_q8_row(input_quants, input_scales, weight_quants_0, weight_scales_0);
            let d1 =
                avx2::dot_q8_q8_row(input_quants, input_scales, weight_quants_1, weight_scales_1);
            (d0, d1)
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let d0 = neon::dot_q8_q8_row(input_quants, input_scales, weight_quants_0, weight_scales_0);
        let d1 = neon::dot_q8_q8_row(input_quants, input_scales, weight_quants_1, weight_scales_1);
        (d0, d1)
    }
}

/// 2-row Q4×Q8 GEMV: computes dot products for two adjacent Q4_0 weight rows
/// against the same Q8 input vector. Returns `(dot0, dot1)`.
#[inline]
#[must_use]
pub fn dot_q4_q8_2row(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_packed_0: &[u8],
    weight_scales_0: &[f32],
    weight_packed_1: &[u8],
    weight_scales_1: &[f32],
) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_vnni() {
            avx512::dot_q4_q8_2row(
                input_quants,
                input_scales,
                weight_packed_0,
                weight_scales_0,
                weight_packed_1,
                weight_scales_1,
            )
        } else {
            let d0 =
                avx2::dot_q4_q8_row(input_quants, input_scales, weight_packed_0, weight_scales_0);
            let d1 =
                avx2::dot_q4_q8_row(input_quants, input_scales, weight_packed_1, weight_scales_1);
            (d0, d1)
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let d0 = neon::dot_q4_q8_row(input_quants, input_scales, weight_packed_0, weight_scales_0);
        let d1 = neon::dot_q4_q8_row(input_quants, input_scales, weight_packed_1, weight_scales_1);
        (d0, d1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_cpu_support() {
        check_cpu_support().expect("SIMD support check failed");
    }

    #[test]
    fn test_dot_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let result = dot_f32(&a, &b);
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 1e-5, "{result} != {expected}");
    }

    #[test]
    fn test_dot_f32_large() {
        let n = 960; // SmolLM2 hidden_size
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n).map(|i| ((n - i) as f32) * 0.01).collect();
        let result = dot_f32(&a, &b);
        let expected: f64 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        assert!(
            (f64::from(result) - expected).abs() < 1.0,
            "{result} != {expected}"
        );
    }

    #[test]
    fn test_vec_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out = vec![0.0; 5];
        vec_add(&a, &b, &mut out);
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0, 55.0]);
    }

    #[test]
    fn test_vec_mul() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0; 5];
        vec_mul(&a, &b, &mut out);
        assert_eq!(out, vec![2.0, 6.0, 12.0, 20.0, 30.0]);
    }

    #[test]
    fn test_vec_scale() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        vec_scale(&mut a, 2.0);
        assert_eq!(a, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_sum_of_squares() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sum_of_squares(&a);
        assert!((result - 55.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec_silu_mul() {
        let gate = vec![0.0, 1.0, -1.0, 2.0, 3.0];
        let up = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 5];
        vec_silu_mul(&gate, &up, &mut out);
        // silu(x) = x * sigmoid(x)
        for i in 0..5 {
            let expected = gate[i] / (1.0 + (-gate[i]).exp()) * up[i];
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "index {i}: {} != {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn test_vec_rmsnorm() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];
        vec_rmsnorm(&input, &weight, 1e-6, &mut out);
        let ss: f32 = input.iter().map(|x| x * x).sum();
        let rms = 1.0 / (ss / 4.0 + 1e-6).sqrt();
        for i in 0..4 {
            let expected = input[i] * rms;
            assert!((out[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dot_q8_block() {
        // 32 f32 input values
        let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        // 32 int8 weights
        let quants: Vec<u8> = (0..32).map(|i| (i as i8 - 16) as u8).collect();
        let scale = 0.5f32;

        // Expected: sum(input[i] * int8[i]) * scale
        let expected: f32 = input
            .iter()
            .zip(quants.iter())
            .map(|(&a, &b)| a * (b as i8 as f32))
            .sum::<f32>()
            * scale;

        let result = dot_q8_block(&input, &quants, scale);
        assert!(
            (result - expected).abs() < 1e-3,
            "dot_q8_block: got {result}, expected {expected}"
        );
    }

    #[test]
    fn test_dot_q4_block() {
        // 32 f32 input values
        let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        // 16 packed bytes: low nibble = element[i], high nibble = element[i+16]
        let mut packed = vec![0u8; 16];
        for i in 0..16 {
            let lo: u8 = ((i as i8 - 3) + 8) as u8 & 0x0F; // signed val, biased
            let hi: u8 = ((i as i8 + 2) + 8) as u8 & 0x0F;
            packed[i] = lo | (hi << 4);
        }
        let scale = 0.25f32;

        // Expected: sum(input[i] * (nibble[i] - 8)) * scale
        let mut expected = 0.0f32;
        for i in 0..16 {
            let lo_val = (packed[i] & 0x0F) as i32 - 8;
            let hi_val = ((packed[i] >> 4) & 0x0F) as i32 - 8;
            expected += input[i] * lo_val as f32;
            expected += input[i + 16] * hi_val as f32;
        }
        expected *= scale;

        let result = dot_q4_block(&input, &packed, scale);
        assert!(
            (result - expected).abs() < 1e-3,
            "dot_q4_block: got {result}, expected {expected}"
        );
    }

    #[test]
    fn test_dot_q4_1_block() {
        let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let mut packed = vec![0u8; 16];
        for i in 0..16 {
            let lo: u8 = (i as u8) & 0x0F;
            let hi: u8 = ((15 - i) as u8) & 0x0F;
            packed[i] = lo | (hi << 4);
        }
        let scale = 0.5f32;
        let min = -2.0f32;

        // Expected: sum(input[i] * nibble[i]) * scale + sum(input[i]) * min
        let mut dot = 0.0f32;
        let mut input_sum = 0.0f32;
        for i in 0..16 {
            let lo_val = (packed[i] & 0x0F) as f32;
            let hi_val = ((packed[i] >> 4) & 0x0F) as f32;
            dot += input[i] * lo_val;
            dot += input[i + 16] * hi_val;
            input_sum += input[i] + input[i + 16];
        }
        let expected = dot * scale + input_sum * min;

        let result = dot_q4_1_block(&input, &packed, scale, min);
        assert!(
            (result - expected).abs() < 1e-2,
            "dot_q4_1_block: got {result}, expected {expected}"
        );
    }
}
