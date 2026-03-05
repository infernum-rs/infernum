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

/// Scaled accumulate: `out[i] += scale * src[i]` (AXPY).
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn vec_axpy_inner(out: &mut [f32], scale: f32, src: &[f32]) {
    let n = out.len();
    let chunks = n / 8;
    let remainder = n % 8;
    let vs = _mm256_set1_ps(scale);

    for i in 0..chunks {
        let vo = _mm256_loadu_ps(out.as_ptr().add(i * 8));
        let vsrc = _mm256_loadu_ps(src.as_ptr().add(i * 8));
        _mm256_storeu_ps(out.as_mut_ptr().add(i * 8), _mm256_fmadd_ps(vs, vsrc, vo));
    }

    let tail = chunks * 8;
    for i in 0..remainder {
        out[tail + i] = scale.mul_add(src[tail + i], out[tail + i]);
    }
}

pub fn vec_axpy(out: &mut [f32], scale: f32, src: &[f32]) {
    unsafe {
        vec_axpy_inner(out, scale, src);
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

/// Q8_0 row dot product: process all blocks for one neuron in a single call.
///
/// `input` is the full input row (K elements), `quants` is the quantized row
/// (K bytes), `scales` is the per-block scale array (K/32 elements).
/// Keeps accumulators in registers across blocks to avoid per-block function
/// call overhead.
pub fn dot_q8_row(input: &[f32], quants: &[u8], scales: &[f32]) -> f32 {
    unsafe { dot_q8_row_inner(input, quants, scales) }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q8_row_inner(input: &[f32], quants: &[u8], scales: &[f32]) -> f32 {
    let num_blocks = scales.len();
    let inp = input.as_ptr();
    let q = quants.as_ptr();

    let mut total = _mm256_setzero_ps();

    for blk in 0..num_blocks {
        let scale = _mm256_set1_ps(*scales.get_unchecked(blk));
        let blk_inp = inp.add(blk * 32);
        let blk_q = q.add(blk * 32);

        let mut acc = _mm256_setzero_ps();

        // 4 groups of 8 int8 values
        let q8 = _mm_loadl_epi64(blk_q.cast());
        let qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(blk_inp), qf, acc);

        let q8 = _mm_loadl_epi64(blk_q.add(8).cast());
        let qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(blk_inp.add(8)), qf, acc);

        let q8 = _mm_loadl_epi64(blk_q.add(16).cast());
        let qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(blk_inp.add(16)), qf, acc);

        let q8 = _mm_loadl_epi64(blk_q.add(24).cast());
        let qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(blk_inp.add(24)), qf, acc);

        total = _mm256_fmadd_ps(acc, scale, total);
    }

    hsum_256(total)
}

/// Q4_0 block dot product: `sum(input[i] * (nibble[i] - 8)) * scale` for 32 elements.
pub fn dot_q4_block(input: &[f32], packed: &[u8], scale: f32) -> f32 {
    unsafe { dot_q4_block_inner(input, packed, scale) }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_mm256_sub_ps, _mm_and_si128, _mm_set1_epi8, _mm_srli_epi16};

#[allow(clippy::similar_names)]
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

/// Q4_0 row dot product: process all blocks for one neuron in a single call.
///
/// `input` is the full input row (K elements), `packed` is the packed Q4
/// row (K/2 bytes), `scales` is the per-block scale array (K/32 elements).
pub fn dot_q4_row(input: &[f32], packed: &[u8], scales: &[f32]) -> f32 {
    unsafe { dot_q4_row_inner(input, packed, scales) }
}

#[allow(clippy::similar_names)]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q4_row_inner(input: &[f32], packed: &[u8], scales: &[f32]) -> f32 {
    let num_blocks = scales.len();
    let inp = input.as_ptr();
    let p = packed.as_ptr();
    let bias = _mm256_set1_ps(8.0);
    let mask_0f = _mm_set1_epi8(0x0F);

    let mut total = _mm256_setzero_ps();

    for blk in 0..num_blocks {
        let scale = _mm256_set1_ps(*scales.get_unchecked(blk));
        let blk_inp = inp.add(blk * 32);
        let blk_p = p.add(blk * 16);

        let mut acc = _mm256_setzero_ps();

        // Low nibbles → elements 0..15 (2 groups of 8)
        for group in 0..2 {
            let raw = _mm_loadl_epi64(blk_p.add(group * 8).cast());
            let lo = _mm_and_si128(raw, mask_0f);
            let lo_f32 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), bias);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(blk_inp.add(group * 8)), lo_f32, acc);
        }

        // High nibbles → elements 16..31 (2 groups of 8)
        for group in 0..2 {
            let raw = _mm_loadl_epi64(blk_p.add(group * 8).cast());
            let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);
            let hi_f32 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), bias);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(blk_inp.add(16 + group * 8)), hi_f32, acc);
        }

        total = _mm256_fmadd_ps(acc, scale, total);
    }

    hsum_256(total)
}

/// Q4_1 block dot product: `sum(input[i] * nibble[i]) * scale + sum(input[i]) * min`
/// for 32 elements. Unlike Q4_0, nibbles are unsigned [0,15] — no bias subtraction.
pub fn dot_q4_1_block(input: &[f32], packed: &[u8], scale: f32, min: f32) -> f32 {
    unsafe { dot_q4_1_block_inner(input, packed, scale, min) }
}

#[allow(clippy::similar_names)]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q4_1_block_inner(input: &[f32], packed: &[u8], scale: f32, min: f32) -> f32 {
    let inp = input.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();

    let mask_0f = _mm_set1_epi8(0x0F);

    // Process low nibbles of all 16 bytes → elements 0..15
    for group in 0..2 {
        let raw = _mm_loadl_epi64(packed.as_ptr().add(group * 8).cast());
        let lo = _mm_and_si128(raw, mask_0f);
        let lo_i32 = _mm256_cvtepi8_epi32(lo);
        let lo_f32 = _mm256_cvtepi32_ps(lo_i32);
        let in_vec = _mm256_loadu_ps(inp.add(group * 8));
        acc0 = _mm256_fmadd_ps(in_vec, lo_f32, acc0);
        sum0 = _mm256_add_ps(sum0, in_vec);
    }

    // Process high nibbles of all 16 bytes → elements 16..31
    for group in 0..2 {
        let raw = _mm_loadl_epi64(packed.as_ptr().add(group * 8).cast());
        let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);
        let hi_i32 = _mm256_cvtepi8_epi32(hi);
        let hi_f32 = _mm256_cvtepi32_ps(hi_i32);
        let in_vec = _mm256_loadu_ps(inp.add(16 + group * 8));
        acc1 = _mm256_fmadd_ps(in_vec, hi_f32, acc1);
        sum1 = _mm256_add_ps(sum1, in_vec);
    }

    let dot = hsum_256(_mm256_add_ps(acc0, acc1));
    let input_sum = hsum_256(_mm256_add_ps(sum0, sum1));
    dot * scale + input_sum * min
}

/// Q4_1 row dot product: process all blocks for one neuron in a single call.
///
/// `input` is the full input row (K elements), `packed` is the packed Q4_1
/// row (K/2 bytes), `scales` and `mins` are per-block arrays (K/32 elements).
pub fn dot_q4_1_row(input: &[f32], packed: &[u8], scales: &[f32], mins: &[f32]) -> f32 {
    unsafe { dot_q4_1_row_inner(input, packed, scales, mins) }
}

#[allow(clippy::similar_names)]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q4_1_row_inner(input: &[f32], packed: &[u8], scales: &[f32], mins: &[f32]) -> f32 {
    let num_blocks = scales.len();
    let inp = input.as_ptr();
    let p = packed.as_ptr();
    let mask_0f = _mm_set1_epi8(0x0F);

    let mut total_dot = _mm256_setzero_ps();
    let mut total_sum = _mm256_setzero_ps();

    for blk in 0..num_blocks {
        let scale = _mm256_set1_ps(*scales.get_unchecked(blk));
        let min = _mm256_set1_ps(*mins.get_unchecked(blk));
        let blk_inp = inp.add(blk * 32);
        let blk_p = p.add(blk * 16);

        let mut acc = _mm256_setzero_ps();
        let mut input_sum = _mm256_setzero_ps();

        // Low nibbles → elements 0..15
        for group in 0..2 {
            let raw = _mm_loadl_epi64(blk_p.add(group * 8).cast());
            let lo = _mm_and_si128(raw, mask_0f);
            let lo_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
            let in_vec = _mm256_loadu_ps(blk_inp.add(group * 8));
            acc = _mm256_fmadd_ps(in_vec, lo_f32, acc);
            input_sum = _mm256_add_ps(input_sum, in_vec);
        }

        // High nibbles → elements 16..31
        for group in 0..2 {
            let raw = _mm_loadl_epi64(blk_p.add(group * 8).cast());
            let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);
            let hi_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
            let in_vec = _mm256_loadu_ps(blk_inp.add(16 + group * 8));
            acc = _mm256_fmadd_ps(in_vec, hi_f32, acc);
            input_sum = _mm256_add_ps(input_sum, in_vec);
        }

        total_dot = _mm256_fmadd_ps(acc, scale, total_dot);
        total_sum = _mm256_fmadd_ps(input_sum, min, total_sum);
    }

    hsum_256(_mm256_add_ps(total_dot, total_sum))
}

// =====================================================================
// Integer dot product kernels (AVX2: vpmaddubsw + vpmaddwd)
//
// These keep computation in the integer domain until the final per-block
// scale multiply. For Q8×Q8: uses the vpsignb trick to create one
// unsigned + one signed operand for vpmaddubsw.
// =====================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi16,
    _mm256_sign_epi8,
};

/// Q8_0 integer row dot product using AVX2 vpmaddubsw + vpmaddwd.
///
/// Both weight quants and input quants are int8. Uses vpsignb trick:
/// `abs(a) × sign(a,b)` where a=weight, b=input_quants.
/// `input_quants` must be pre-quantized int8 (as u8-typed bytes).
///
/// Returns the full dot product (with scales applied).
pub fn dot_q8_q8_row(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_quants: &[u8],
    weight_scales: &[f32],
) -> f32 {
    unsafe { dot_q8_q8_row_inner(input_quants, input_scales, weight_quants, weight_scales) }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q8_q8_row_inner(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_quants: &[u8],
    weight_scales: &[f32],
) -> f32 {
    let num_blocks = weight_scales.len();
    let iq = input_quants.as_ptr();
    let wq = weight_quants.as_ptr();
    let ones_16 = _mm256_set1_epi16(1);

    let mut total = _mm256_setzero_ps();

    for blk in 0..num_blocks {
        let combined_scale =
            _mm256_set1_ps(*input_scales.get_unchecked(blk) * *weight_scales.get_unchecked(blk));
        let blk_offset = blk * 32;

        // Load 32 bytes from each operand
        let a = _mm256_loadu_si256(wq.add(blk_offset).cast());
        let b = _mm256_loadu_si256(iq.add(blk_offset).cast());

        // vpsignb trick: abs(a) is unsigned, sign(b,a) adjusts b's signs
        let a_abs = _mm256_sign_epi8(a, a); // abs(a) — unsigned
        let b_signed = _mm256_sign_epi8(b, a); // b with a's signs applied

        // vpmaddubsw: 32 uint8×int8 → 16 int16 (pairwise add)
        let prod_16 = _mm256_maddubs_epi16(a_abs, b_signed);
        // vpmaddwd: 16 int16 × 16 int16(ones) → 8 int32 (pairwise add)
        let prod_32 = _mm256_madd_epi16(prod_16, ones_16);

        // Convert to f32 and accumulate with combined scale
        let dot_f32 = _mm256_cvtepi32_ps(prod_32);
        total = _mm256_fmadd_ps(dot_f32, combined_scale, total);
    }

    hsum_256(total)
}

/// Quantize a row of f32 values to Q8_0 blocks in-place.
///
/// For each block of 32 floats, finds the max absolute value, computes
/// scale = max_abs / 127, and stores quantized int8 values + scale.
///
/// `out_quants` must have length K (one byte per element).
/// `out_scales` must have length K/32 (one scale per block).
pub fn quantize_row_q8(input: &[f32], out_quants: &mut [u8], out_scales: &mut [f32]) {
    unsafe { quantize_row_q8_inner(input, out_quants, out_scales) }
}

#[allow(clippy::items_after_statements)]
#[target_feature(enable = "avx2")]
unsafe fn quantize_row_q8_inner(input: &[f32], out_quants: &mut [u8], out_scales: &mut [f32]) {
    use std::arch::x86_64::{
        _mm256_and_ps, _mm256_castsi256_ps, _mm256_max_ps, _mm256_set1_epi32, _mm256_set1_ps,
    };

    let num_blocks = out_scales.len();
    let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFFu32.cast_signed())); // abs mask

    for blk in 0..num_blocks {
        let blk_start = blk * 32;
        let inp = input.as_ptr().add(blk_start);

        // Find max absolute value across 32 elements
        let mut max_abs = _mm256_setzero_ps();
        for g in 0..4 {
            let v = _mm256_loadu_ps(inp.add(g * 8));
            let abs_v = _mm256_and_ps(v, mask);
            max_abs = _mm256_max_ps(max_abs, abs_v);
        }

        // Horizontal max
        let max_scalar = {
            // Extract all 8 lanes and take scalar max
            let mut buf = [0.0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), max_abs);
            buf.iter().copied().fold(0.0f32, f32::max)
        };

        let scale = max_scalar / 127.0;
        *out_scales.get_unchecked_mut(blk) = scale;

        if scale == 0.0 {
            // All zeros
            for i in 0..32 {
                *out_quants.get_unchecked_mut(blk_start + i) = 0;
            }
            continue;
        }

        let inv_scale = _mm256_set1_ps(1.0 / scale);

        // Quantize: round(input / scale) clamped to [-127, 127]
        use std::arch::x86_64::{_mm256_cvtps_epi32, _mm256_packs_epi32, _mm256_permute4x64_epi64};

        // Process 32 elements in 4 groups of 8
        // Pack: 8xi32 → 8xi16 (×2 groups → 16xi16) → 16xi8 (×2 → 32xi8)
        let v0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(inp), inv_scale));
        let v1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(inp.add(8)), inv_scale));
        let v2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(inp.add(16)), inv_scale));
        let v3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(inp.add(24)), inv_scale));

        // packs_epi32 operates per 128-bit lane, interleaving 4 from each operand:
        //   Lane 0: v0[0:4], v1[0:4]   Lane 1: v0[4:8], v1[4:8]
        // Fix with permute4x64(0xD8) to restore sequential order:
        //   Lane 0: v0[0:4], v0[4:8]   Lane 1: v1[0:4], v1[4:8]
        let packed_16_01 = _mm256_packs_epi32(v0, v1);
        let packed_16_01 = _mm256_permute4x64_epi64(packed_16_01, 0xD8);
        let packed_16_23 = _mm256_packs_epi32(v2, v3);
        let packed_16_23 = _mm256_permute4x64_epi64(packed_16_23, 0xD8);

        // Same lane-crossing issue applies to packs_epi16 → fix again
        use std::arch::x86_64::_mm256_packs_epi16;
        let packed_8 = _mm256_packs_epi16(packed_16_01, packed_16_23);
        let fixed = _mm256_permute4x64_epi64(packed_8, 0xD8);

        // Store 32 bytes
        use std::arch::x86_64::_mm256_storeu_si256;
        _mm256_storeu_si256(out_quants.as_mut_ptr().add(blk_start).cast(), fixed);
    }
}

/// Q4_0 integer row dot product using AVX2.
///
/// Weight nibbles are unpacked to int8, then integer dot product with
/// pre-quantized Q8 input using vpmaddubsw + vpmaddwd.
///
/// Returns the full dot product (with scales applied).
pub fn dot_q4_q8_row(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_packed: &[u8],
    weight_scales: &[f32],
) -> f32 {
    unsafe { dot_q4_q8_row_inner(input_quants, input_scales, weight_packed, weight_scales) }
}

#[allow(clippy::similar_names, clippy::items_after_statements)]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q4_q8_row_inner(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_packed: &[u8],
    weight_scales: &[f32],
) -> f32 {
    use std::arch::x86_64::{
        _mm256_and_si256, _mm256_set1_epi8, _mm256_srli_epi16, _mm256_sub_epi8,
    };

    let num_blocks = weight_scales.len();
    let iq = input_quants.as_ptr();
    let wp = weight_packed.as_ptr();
    let ones_16 = _mm256_set1_epi16(1);
    let mask_0f = _mm256_set1_epi8(0x0F);
    let bias_8 = _mm256_set1_epi8(8);

    let mut total = _mm256_setzero_ps();

    for blk in 0..num_blocks {
        let combined_scale =
            _mm256_set1_ps(*input_scales.get_unchecked(blk) * *weight_scales.get_unchecked(blk));
        let inp_offset = blk * 32;
        let wp_offset = blk * 16;

        // Load 16 packed bytes
        // We need to unpack to 32 int8 values. Layout: byte[i] low=elem[i], high=elem[i+16]
        // Load 16 bytes, then expand to 32 bytes: low nibbles first, then high nibbles
        use std::arch::x86_64::_mm_loadu_si128;
        let packed_128 = _mm_loadu_si128(wp.add(wp_offset).cast());

        // Broadcast 128-bit to 256-bit (duplicate in both lanes)
        use std::arch::x86_64::_mm256_set_m128i;
        let packed = _mm256_set_m128i(packed_128, packed_128);

        // Low nibbles in low 128 bits, high nibbles in high 128 bits
        let lo = _mm256_and_si256(packed, mask_0f); // [0..15] in low lane, [0..15] in high lane
        let hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_0f); // shifted nibbles

        // Combine: lo lane keeps low 128 bits of lo, hi lane gets low 128 bits of hi
        // We want: [lo_nibble[0..15], hi_nibble[0..15]] as 32 bytes
        // Since we duplicated, lo has lo_nibbles in both lanes, hi has hi_nibbles in both lanes
        // Use blend or permute to combine:
        use std::arch::x86_64::_mm256_permute2x128_si256;
        let unpacked = _mm256_permute2x128_si256(lo, hi, 0x20); // lo low128 | hi low128

        // Subtract bias: Q4_0 uses signed range [-8, 7]
        let weight_i8 = _mm256_sub_epi8(unpacked, bias_8);

        // Load 32 input quants
        let input_i8 = _mm256_loadu_si256(iq.add(inp_offset).cast());

        // Integer dot product: vpsignb trick for two signed operands
        let w_abs = _mm256_sign_epi8(weight_i8, weight_i8);
        let i_signed = _mm256_sign_epi8(input_i8, weight_i8);

        let prod_16 = _mm256_maddubs_epi16(w_abs, i_signed);
        let prod_32 = _mm256_madd_epi16(prod_16, ones_16);

        let prod_f32 = _mm256_cvtepi32_ps(prod_32);
        total = _mm256_fmadd_ps(prod_f32, combined_scale, total);
    }

    hsum_256(total)
}

/// Q4_1 integer row dot product using AVX2.
///
/// Weight nibbles are unsigned [0,15]. Uses integer dot for the nibble×input_quant
/// part, plus a separate input sum × min correction term.
pub fn dot_q4_1_q8_row(
    input_quants: &[u8],
    input_scales: &[f32],
    input_row: &[f32],
    weight_packed: &[u8],
    weight_scales: &[f32],
    weight_mins: &[f32],
) -> f32 {
    unsafe {
        dot_q4_1_q8_row_inner(
            input_quants,
            input_scales,
            input_row,
            weight_packed,
            weight_scales,
            weight_mins,
        )
    }
}

#[allow(
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::items_after_statements
)]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q4_1_q8_row_inner(
    input_quants: &[u8],
    input_scales: &[f32],
    input_row: &[f32],
    weight_packed: &[u8],
    weight_scales: &[f32],
    weight_mins: &[f32],
) -> f32 {
    use std::arch::x86_64::{_mm256_and_si256, _mm256_set1_epi8, _mm256_srli_epi16};

    let num_blocks = weight_scales.len();
    let iq = input_quants.as_ptr();
    let wp = weight_packed.as_ptr();
    let inp_f32 = input_row.as_ptr();
    let ones_16 = _mm256_set1_epi16(1);
    let mask_0f = _mm256_set1_epi8(0x0F);

    let mut total_dot = _mm256_setzero_ps();
    let mut total_min = _mm256_setzero_ps();

    for blk in 0..num_blocks {
        let scale = *weight_scales.get_unchecked(blk);
        let min = *weight_mins.get_unchecked(blk);
        let input_scale = *input_scales.get_unchecked(blk);
        let inp_offset = blk * 32;
        let wp_offset = blk * 16;

        // Unpack Q4_1 nibbles (unsigned, no bias subtraction)
        use std::arch::x86_64::{_mm256_permute2x128_si256, _mm256_set_m128i, _mm_loadu_si128};
        let packed_128 = _mm_loadu_si128(wp.add(wp_offset).cast());
        let packed = _mm256_set_m128i(packed_128, packed_128);
        let lo = _mm256_and_si256(packed, mask_0f);
        let hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_0f);
        let unpacked = _mm256_permute2x128_si256(lo, hi, 0x20);

        // For Q4_1, nibbles are unsigned [0,15]. The input quants are signed int8.
        // vpmaddubsw needs (unsigned, signed), so unpacked_nibbles (unsigned) × input_quants (signed)
        let input_i8 = _mm256_loadu_si256(iq.add(inp_offset).cast());
        let prod_16 = _mm256_maddubs_epi16(unpacked, input_i8);
        let prod_32 = _mm256_madd_epi16(prod_16, ones_16);

        let combined_scale = _mm256_set1_ps(scale * input_scale);
        let prod_f32 = _mm256_cvtepi32_ps(prod_32);
        total_dot = _mm256_fmadd_ps(prod_f32, combined_scale, total_dot);

        // Compute sum(input_f32[blk]) for the min correction term
        // Use f32 for this since we need the original input values
        let s0 = _mm256_loadu_ps(inp_f32.add(inp_offset));
        let s1 = _mm256_loadu_ps(inp_f32.add(inp_offset + 8));
        let s2 = _mm256_loadu_ps(inp_f32.add(inp_offset + 16));
        let s3 = _mm256_loadu_ps(inp_f32.add(inp_offset + 24));
        let block_sum = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        let min_vec = _mm256_set1_ps(min);
        total_min = _mm256_fmadd_ps(block_sum, min_vec, total_min);
    }

    hsum_256(total_dot) + hsum_256(total_min)
}
