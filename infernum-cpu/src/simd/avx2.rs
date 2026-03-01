//! AVX2+FMA SIMD kernels for x86-64.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256, _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps,
    _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehdup_ps, _mm_movehl_ps,
};

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

/// Q8_0 block dot product: `sum(input[i] * int8[i]) * scale` for 32 elements.
pub fn dot_q8_block(input: &[f32], quants: &[u8], scale: f32) -> f32 {
    unsafe { dot_q8_block_inner(input, quants, scale) }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_mm256_cvtepi32_ps, _mm256_cvtepi8_epi32, _mm_loadl_epi64};

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q8_block_inner(input: &[f32], quants: &[u8], scale: f32) -> f32 {
    // Process 32 int8 values in 4 groups of 8:
    // load 8×i8 → cvtepi8_epi32 → cvtepi32_ps → FMA
    let inp = input.as_ptr();
    let q = quants.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    // Group 0: elements 0..7
    let q8 = _mm_loadl_epi64(q.cast());
    let q32 = _mm256_cvtepi8_epi32(q8);
    let qf = _mm256_cvtepi32_ps(q32);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(inp), qf, acc0);

    // Group 1: elements 8..15
    let q8 = _mm_loadl_epi64(q.add(8).cast());
    let q32 = _mm256_cvtepi8_epi32(q8);
    let qf = _mm256_cvtepi32_ps(q32);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(inp.add(8)), qf, acc1);

    // Group 2: elements 16..23
    let q8 = _mm_loadl_epi64(q.add(16).cast());
    let q32 = _mm256_cvtepi8_epi32(q8);
    let qf = _mm256_cvtepi32_ps(q32);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(inp.add(16)), qf, acc0);

    // Group 3: elements 24..31
    let q8 = _mm_loadl_epi64(q.add(24).cast());
    let q32 = _mm256_cvtepi8_epi32(q8);
    let qf = _mm256_cvtepi32_ps(q32);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(inp.add(24)), qf, acc1);

    hsum_256(_mm256_add_ps(acc0, acc1)) * scale
}

/// Q4_0 block dot product: `sum(input[i] * (nibble[i] - 8)) * scale` for 32 elements.
pub fn dot_q4_block(input: &[f32], packed: &[u8], scale: f32) -> f32 {
    unsafe { dot_q4_block_inner(input, packed, scale) }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_mm256_sub_ps, _mm_and_si128, _mm_set1_epi8, _mm_srli_epi16};

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q4_block_inner(input: &[f32], packed: &[u8], scale: f32) -> f32 {
    // Q4_0 layout: byte[i] has low nibble = element[i], high nibble = element[i+16].
    // So 16 bytes → low nibbles are elements 0..15, high nibbles are elements 16..31.
    let inp = input.as_ptr();
    let bias = _mm256_set1_ps(8.0);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let mask_0f = _mm_set1_epi8(0x0F);

    // Process low nibbles of all 16 bytes → elements 0..15
    for group in 0..2 {
        let raw = _mm_loadl_epi64(packed.as_ptr().add(group * 8).cast());
        let lo = _mm_and_si128(raw, mask_0f);
        let lo_i32 = _mm256_cvtepi8_epi32(lo);
        let lo_f32 = _mm256_sub_ps(_mm256_cvtepi32_ps(lo_i32), bias);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(inp.add(group * 8)), lo_f32, acc0);
    }

    // Process high nibbles of all 16 bytes → elements 16..31
    for group in 0..2 {
        let raw = _mm_loadl_epi64(packed.as_ptr().add(group * 8).cast());
        let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);
        let hi_i32 = _mm256_cvtepi8_epi32(hi);
        let hi_f32 = _mm256_sub_ps(_mm256_cvtepi32_ps(hi_i32), bias);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(inp.add(16 + group * 8)), hi_f32, acc1);
    }

    hsum_256(_mm256_add_ps(acc0, acc1)) * scale
}
