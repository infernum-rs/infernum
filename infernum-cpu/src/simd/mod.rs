//! SIMD dispatch layer.
//!
//! Provides architecture-specific SIMD kernels: AVX2+FMA on x86-64,
//! NEON on AArch64. No scalar fallback — unsupported platforms are a
//! compile error.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("infernum-cpu requires x86-64 (AVX2+FMA) or AArch64 (NEON)");

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
#[inline]
pub fn vec_silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    {
        avx2::vec_silu_mul(gate, up, out);
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::vec_silu_mul(gate, up, out);
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
}
