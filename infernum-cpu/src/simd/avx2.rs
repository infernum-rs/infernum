//! AVX2+FMA SIMD kernels for x86-64.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Horizontal sum of an __m256 register.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_256(v: __m256) -> f32 {
    // v = [a0 a1 a2 a3 | a4 a5 a6 a7]
    let hi128 = _mm256_extractf128_ps(v, 1); // [a4 a5 a6 a7]
    let lo128 = _mm256_castps256_ps128(v); // [a0 a1 a2 a3]
    let sum128 = _mm_add_ps(lo128, hi128); // [a0+a4 a1+a5 a2+a6 a3+a7]
    let shuf = _mm_movehdup_ps(sum128); // [a1+a5 a1+a5 a3+a7 a3+a7]
    let sum64 = _mm_add_ps(sum128, shuf); // [s01 _ s23 _]
    let hi32 = _mm_movehl_ps(sum64, sum64); // [s23 _ _ _]
    let sum32 = _mm_add_ss(sum64, hi32); // [s0123 _ _ _]
    _mm_cvtss_f32(sum32)
}

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_f32_inner(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = _mm256_setzero_ps();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a_ptr.add(i * 8));
        let vb = _mm256_loadu_ps(b_ptr.add(i * 8));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    let mut sum = hsum_256(acc);
    let tail_start = chunks * 8;
    for i in 0..remainder {
        sum = a[tail_start + i].mul_add(b[tail_start + i], sum);
    }
    sum
}

pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: check_cpu_support() verified AVX2+FMA at backend init.
    unsafe { dot_f32_inner(a, b) }
}

#[target_feature(enable = "avx2")]
unsafe fn vec_add_inner(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        _mm256_storeu_ps(out.as_mut_ptr().add(i * 8), _mm256_add_ps(va, vb));
    }

    let tail = chunks * 8;
    for i in 0..remainder {
        out[tail + i] = a[tail + i] + b[tail + i];
    }
}

pub fn vec_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    unsafe {
        vec_add_inner(a, b, out);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn vec_add_inplace_inner(a: &mut [f32], b: &[f32]) {
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        _mm256_storeu_ps(a.as_mut_ptr().add(i * 8), _mm256_add_ps(va, vb));
    }

    let tail = chunks * 8;
    for i in 0..remainder {
        a[tail + i] += b[tail + i];
    }
}

pub fn vec_add_inplace(a: &mut [f32], b: &[f32]) {
    unsafe {
        vec_add_inplace_inner(a, b);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn vec_mul_inner(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        _mm256_storeu_ps(out.as_mut_ptr().add(i * 8), _mm256_mul_ps(va, vb));
    }

    let tail = chunks * 8;
    for i in 0..remainder {
        out[tail + i] = a[tail + i] * b[tail + i];
    }
}

pub fn vec_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    unsafe {
        vec_mul_inner(a, b, out);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn vec_scale_inner(a: &mut [f32], scale: f32) {
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;
    let vs = _mm256_set1_ps(scale);

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        _mm256_storeu_ps(a.as_mut_ptr().add(i * 8), _mm256_mul_ps(va, vs));
    }

    let tail = chunks * 8;
    for i in 0..remainder {
        a[tail + i] *= scale;
    }
}

pub fn vec_scale(a: &mut [f32], scale: f32) {
    unsafe {
        vec_scale_inner(a, scale);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sum_of_squares_inner(a: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        acc = _mm256_fmadd_ps(va, va, acc);
    }

    let mut sum = hsum_256(acc);
    let tail = chunks * 8;
    for i in 0..remainder {
        sum = a[tail + i].mul_add(a[tail + i], sum);
    }
    sum
}

pub fn sum_of_squares(a: &[f32]) -> f32 {
    unsafe { sum_of_squares_inner(a) }
}

pub fn vec_silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    // SiLU is hard to vectorize efficiently (exp), so use scalar with FMA
    for i in 0..gate.len() {
        let sigmoid = 1.0 / (1.0 + (-gate[i]).exp());
        out[i] = gate[i] * sigmoid * up[i];
    }
}
