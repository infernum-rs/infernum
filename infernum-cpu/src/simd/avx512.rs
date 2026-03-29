//! AVX-512 VNNI SIMD kernels for x86-64.
//!
//! These kernels use `vpdpbusd` (256-bit VNNI) for integer dot products.
//! They require AVX-512F + AVX-512VNNI + AVX-512VL + AVX-512BW
//! (Cascade Lake+, Zen 4+).
//!
//! Since Rust stable doesn't expose `_mm256_dpbusd_epi32` as an intrinsic,
//! we use inline assembly for the `vpdpbusd` instruction.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256, __m256i, _mm256_add_ps, _mm256_castps256_ps128, _mm256_cvtepi32_ps,
    _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_setzero_si256, _mm256_sign_epi8, _mm_add_ps, _mm_add_ss,
    _mm_cvtss_f32, _mm_movehdup_ps, _mm_movehl_ps,
};

/// Horizontal sum of an __m256 register.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_256(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let hi32 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

/// `vpdpbusd ymm, ymm, ymm` via inline assembly.
///
/// Computes `acc += dot(a_uint8, b_int8)` where the dot product multiplies
/// groups of 4 bytes: `sum(a[4i+j] * b[4i+j] for j in 0..4)` accumulated
/// into 8 int32 lanes.
#[target_feature(enable = "avx512f", enable = "avx512vnni", enable = "avx512vl")]
#[inline]
unsafe fn dpbusd_256(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
    let result: __m256i;
    std::arch::asm!(
        "vpdpbusd {dst}, {src1}, {src2}",
        dst = inlateout(ymm_reg) acc => result,
        src1 = in(ymm_reg) a,
        src2 = in(ymm_reg) b,
        options(pure, nomem, nostack),
    );
    result
}

/// Q8×Q8 integer row dot product using AVX-512 VNNI `vpdpbusd`.
pub fn dot_q8_q8_row(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_quants: &[u8],
    weight_scales: &[f32],
) -> f32 {
    unsafe { dot_q8_q8_row_inner(input_quants, input_scales, weight_quants, weight_scales) }
}

#[target_feature(
    enable = "avx512f",
    enable = "avx512vnni",
    enable = "avx512vl",
    enable = "fma"
)]
unsafe fn dot_q8_q8_row_inner(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_quants: &[u8],
    weight_scales: &[f32],
) -> f32 {
    let num_blocks = weight_scales.len();
    let iq = input_quants.as_ptr();
    let wq = weight_quants.as_ptr();

    let mut total = _mm256_setzero_ps();

    for blk in 0..num_blocks {
        let combined_scale =
            _mm256_set1_ps(*input_scales.get_unchecked(blk) * *weight_scales.get_unchecked(blk));
        let blk_offset = blk * 32;

        let a = _mm256_loadu_si256(wq.add(blk_offset).cast());
        let b = _mm256_loadu_si256(iq.add(blk_offset).cast());

        // vpsignb trick: abs(a) is unsigned, sign(b,a) makes b signed-correct
        let a_abs = _mm256_sign_epi8(a, a);
        let b_signed = _mm256_sign_epi8(b, a);

        // vpdpbusd: 32 uint8×int8 → accumulate into 8 int32
        let prod_32 = dpbusd_256(_mm256_setzero_si256(), a_abs, b_signed);

        let dot_f32 = _mm256_cvtepi32_ps(prod_32);
        total = _mm256_fmadd_ps(dot_f32, combined_scale, total);
    }

    hsum_256(total)
}

/// Q4_0 integer row dot product using AVX-512 VNNI.
pub fn dot_q4_q8_row(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_packed: &[u8],
    weight_scales: &[f32],
) -> f32 {
    unsafe { dot_q4_q8_row_inner(input_quants, input_scales, weight_packed, weight_scales) }
}

#[allow(clippy::similar_names)]
#[target_feature(
    enable = "avx512f",
    enable = "avx512vnni",
    enable = "avx512vl",
    enable = "avx512bw",
    enable = "fma"
)]
unsafe fn dot_q4_q8_row_inner(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_packed: &[u8],
    weight_scales: &[f32],
) -> f32 {
    use std::arch::x86_64::{
        _mm256_and_si256, _mm256_permute2x128_si256, _mm256_set1_epi8, _mm256_set_m128i,
        _mm256_srli_epi16, _mm256_sub_epi8, _mm_loadu_si128,
    };

    let num_blocks = weight_scales.len();
    let iq = input_quants.as_ptr();
    let wp = weight_packed.as_ptr();
    let mask_0f = _mm256_set1_epi8(0x0F);
    let bias_8 = _mm256_set1_epi8(8);

    let mut total = _mm256_setzero_ps();

    for blk in 0..num_blocks {
        let combined_scale =
            _mm256_set1_ps(*input_scales.get_unchecked(blk) * *weight_scales.get_unchecked(blk));
        let inp_offset = blk * 32;
        let wp_offset = blk * 16;

        // Unpack 16 bytes → 32 int8
        let packed_128 = _mm_loadu_si128(wp.add(wp_offset).cast());
        let packed = _mm256_set_m128i(packed_128, packed_128);
        let lo = _mm256_and_si256(packed, mask_0f);
        let hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_0f);
        let unpacked = _mm256_permute2x128_si256(lo, hi, 0x20);
        let weight_i8 = _mm256_sub_epi8(unpacked, bias_8);

        let input_i8 = _mm256_loadu_si256(iq.add(inp_offset).cast());

        // vpsignb trick + vpdpbusd
        let w_abs = _mm256_sign_epi8(weight_i8, weight_i8);
        let i_signed = _mm256_sign_epi8(input_i8, weight_i8);

        let prod_32 = dpbusd_256(_mm256_setzero_si256(), w_abs, i_signed);

        let prod_f32 = _mm256_cvtepi32_ps(prod_32);
        total = _mm256_fmadd_ps(prod_f32, combined_scale, total);
    }

    hsum_256(total)
}

/// Q4_1 integer row dot product using AVX-512 VNNI.
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

#[allow(clippy::similar_names, clippy::too_many_arguments)]
#[target_feature(
    enable = "avx512f",
    enable = "avx512vnni",
    enable = "avx512vl",
    enable = "avx512bw",
    enable = "fma"
)]
unsafe fn dot_q4_1_q8_row_inner(
    input_quants: &[u8],
    input_scales: &[f32],
    input_row: &[f32],
    weight_packed: &[u8],
    weight_scales: &[f32],
    weight_mins: &[f32],
) -> f32 {
    use std::arch::x86_64::{
        _mm256_and_si256, _mm256_permute2x128_si256, _mm256_set1_epi8, _mm256_set_m128i,
        _mm256_srli_epi16, _mm_loadu_si128,
    };

    let num_blocks = weight_scales.len();
    let iq = input_quants.as_ptr();
    let wp = weight_packed.as_ptr();
    let inp_f32 = input_row.as_ptr();
    let mask_0f = _mm256_set1_epi8(0x0F);

    let mut total_dot = _mm256_setzero_ps();
    let mut total_min = _mm256_setzero_ps();

    for blk in 0..num_blocks {
        let scale = *weight_scales.get_unchecked(blk);
        let min = *weight_mins.get_unchecked(blk);
        let input_scale = *input_scales.get_unchecked(blk);
        let inp_offset = blk * 32;
        let wp_offset = blk * 16;

        // Unpack Q4_1 nibbles (unsigned)
        let packed_128 = _mm_loadu_si128(wp.add(wp_offset).cast());
        let packed = _mm256_set_m128i(packed_128, packed_128);
        let lo = _mm256_and_si256(packed, mask_0f);
        let hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_0f);
        let unpacked = _mm256_permute2x128_si256(lo, hi, 0x20);

        // nibbles are unsigned [0,15], input quants are signed int8
        // vpdpbusd needs (unsigned, signed) — nibbles are already unsigned
        let input_i8 = _mm256_loadu_si256(iq.add(inp_offset).cast());
        let prod_32 = dpbusd_256(_mm256_setzero_si256(), unpacked, input_i8);

        let combined_scale = _mm256_set1_ps(scale * input_scale);
        let prod_f32 = _mm256_cvtepi32_ps(prod_32);
        total_dot = _mm256_fmadd_ps(prod_f32, combined_scale, total_dot);

        // Min correction: sum(input_f32) * min
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

// ---- Multi-row GEMV kernels ----
//
// Process 2 output neurons per pass over the input vector. The input data
// (quants + scales) is loaded once and reused for both weight rows, halving
// input memory traffic. Weight data is streamed sequentially for both rows.

/// 2-row Q8×Q8 GEMV: computes dot products for two adjacent weight rows
/// against the same input vector. Returns `(dot0, dot1)`.
pub fn dot_q8_q8_2row(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_quants_0: &[u8],
    weight_scales_0: &[f32],
    weight_quants_1: &[u8],
    weight_scales_1: &[f32],
) -> (f32, f32) {
    unsafe {
        dot_q8_q8_2row_inner(
            input_quants,
            input_scales,
            weight_quants_0,
            weight_scales_0,
            weight_quants_1,
            weight_scales_1,
        )
    }
}

#[allow(clippy::similar_names)]
#[target_feature(
    enable = "avx512f",
    enable = "avx512vnni",
    enable = "avx512vl",
    enable = "fma"
)]
unsafe fn dot_q8_q8_2row_inner(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_quants_0: &[u8],
    weight_scales_0: &[f32],
    weight_quants_1: &[u8],
    weight_scales_1: &[f32],
) -> (f32, f32) {
    let num_blocks = input_scales.len();
    let iq = input_quants.as_ptr();
    let wq0 = weight_quants_0.as_ptr();
    let wq1 = weight_quants_1.as_ptr();

    let mut total0 = _mm256_setzero_ps();
    let mut total1 = _mm256_setzero_ps();

    for blk in 0..num_blocks {
        let blk_offset = blk * 32;
        let inp_scale = *input_scales.get_unchecked(blk);

        // Load input once, reuse for both rows
        let input_i8 = _mm256_loadu_si256(iq.add(blk_offset).cast());

        // Row 0
        let w0 = _mm256_loadu_si256(wq0.add(blk_offset).cast());
        let w0_abs = _mm256_sign_epi8(w0, w0);
        let i0_signed = _mm256_sign_epi8(input_i8, w0);
        let prod0 = dpbusd_256(_mm256_setzero_si256(), w0_abs, i0_signed);
        let scale0 = _mm256_set1_ps(inp_scale * *weight_scales_0.get_unchecked(blk));
        total0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod0), scale0, total0);

        // Row 1
        let w1 = _mm256_loadu_si256(wq1.add(blk_offset).cast());
        let w1_abs = _mm256_sign_epi8(w1, w1);
        let i1_signed = _mm256_sign_epi8(input_i8, w1);
        let prod1 = dpbusd_256(_mm256_setzero_si256(), w1_abs, i1_signed);
        let scale1 = _mm256_set1_ps(inp_scale * *weight_scales_1.get_unchecked(blk));
        total1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod1), scale1, total1);
    }

    (hsum_256(total0), hsum_256(total1))
}

/// 2-row Q4×Q8 GEMV: computes dot products for two adjacent Q4_0 weight rows
/// against the same Q8 input vector. Returns `(dot0, dot1)`.
pub fn dot_q4_q8_2row(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_packed_0: &[u8],
    weight_scales_0: &[f32],
    weight_packed_1: &[u8],
    weight_scales_1: &[f32],
) -> (f32, f32) {
    unsafe {
        dot_q4_q8_2row_inner(
            input_quants,
            input_scales,
            weight_packed_0,
            weight_scales_0,
            weight_packed_1,
            weight_scales_1,
        )
    }
}

#[allow(clippy::similar_names)]
#[target_feature(
    enable = "avx512f",
    enable = "avx512vnni",
    enable = "avx512vl",
    enable = "avx512bw",
    enable = "fma"
)]
unsafe fn dot_q4_q8_2row_inner(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_packed_0: &[u8],
    weight_scales_0: &[f32],
    weight_packed_1: &[u8],
    weight_scales_1: &[f32],
) -> (f32, f32) {
    use std::arch::x86_64::{
        _mm256_and_si256, _mm256_permute2x128_si256, _mm256_set1_epi8, _mm256_set_m128i,
        _mm256_srli_epi16, _mm256_sub_epi8, _mm_loadu_si128,
    };

    let num_blocks = input_scales.len();
    let iq = input_quants.as_ptr();
    let wp0 = weight_packed_0.as_ptr();
    let wp1 = weight_packed_1.as_ptr();
    let mask_0f = _mm256_set1_epi8(0x0F);
    let bias_8 = _mm256_set1_epi8(8);

    let mut total0 = _mm256_setzero_ps();
    let mut total1 = _mm256_setzero_ps();

    for blk in 0..num_blocks {
        let inp_offset = blk * 32;
        let wp_offset = blk * 16;
        let inp_scale = *input_scales.get_unchecked(blk);

        // Load input once, reuse for both rows
        let input_i8 = _mm256_loadu_si256(iq.add(inp_offset).cast());

        // Unpack row 0: 16 bytes → 32 int8
        let packed0_128 = _mm_loadu_si128(wp0.add(wp_offset).cast());
        let packed0 = _mm256_set_m128i(packed0_128, packed0_128);
        let lo0 = _mm256_and_si256(packed0, mask_0f);
        let hi0 = _mm256_and_si256(_mm256_srli_epi16(packed0, 4), mask_0f);
        let unpacked0 = _mm256_permute2x128_si256(lo0, hi0, 0x20);
        let weight0_i8 = _mm256_sub_epi8(unpacked0, bias_8);

        let w0_abs = _mm256_sign_epi8(weight0_i8, weight0_i8);
        let i0_signed = _mm256_sign_epi8(input_i8, weight0_i8);
        let prod0 = dpbusd_256(_mm256_setzero_si256(), w0_abs, i0_signed);
        let scale0 = _mm256_set1_ps(inp_scale * *weight_scales_0.get_unchecked(blk));
        total0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod0), scale0, total0);

        // Unpack row 1: 16 bytes → 32 int8
        let packed1_128 = _mm_loadu_si128(wp1.add(wp_offset).cast());
        let packed1 = _mm256_set_m128i(packed1_128, packed1_128);
        let lo1 = _mm256_and_si256(packed1, mask_0f);
        let hi1 = _mm256_and_si256(_mm256_srli_epi16(packed1, 4), mask_0f);
        let unpacked1 = _mm256_permute2x128_si256(lo1, hi1, 0x20);
        let weight1_i8 = _mm256_sub_epi8(unpacked1, bias_8);

        let w1_abs = _mm256_sign_epi8(weight1_i8, weight1_i8);
        let i1_signed = _mm256_sign_epi8(input_i8, weight1_i8);
        let prod1 = dpbusd_256(_mm256_setzero_si256(), w1_abs, i1_signed);
        let scale1 = _mm256_set1_ps(inp_scale * *weight_scales_1.get_unchecked(blk));
        total1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod1), scale1, total1);
    }

    (hsum_256(total0), hsum_256(total1))
}

// ---- Dot-product F32 GEMM (AVX-512F + FMA) ----
//
// Modeled after llama.cpp's tinyBLAS: each accumulator holds a partial
// dot-product sum as a 16-wide vector. The K loop loads contiguous 16-float
// vectors from both A rows and Bᵀ rows, doing element-wise FMA. At the end,
// each accumulator is reduced via hsum to a single output element.
//
// Micro-tile: RM=4 rows × RN=6 columns = 24 zmm accumulators.
// Per K step: 4 A loads + 6 B loads + 24 FMAs = 34 zmm instructions.
// No K-blocking needed (both A and B are accessed contiguously along K).

/// Micro-kernel row count (4 rows of A).
const RM: usize = 4;
/// Micro-kernel column count (6 columns of Bᵀ → 6 output columns).
const RN: usize = 6;
/// Elements per AVX-512 vector.
const KN: usize = 16;

/// Dot-product F32 GEMM: `C[m,n] = A[m,k] · Bᵀ[n,k]`.
///
/// - `a` is row-major `(M, K)`.
/// - `bt` is row-major `(N, K)` — i.e., B transposed so each "row" of `bt` is
///   one column of the original B.
/// - `c` is row-major `(M, N)`, must be zero-initialized by the caller.
///
/// Uses a 4×6 dot-product micro-kernel with AVX-512F FMA, following the
/// tinyBLAS approach: each zmm accumulator holds a partial K-sum that is
/// horizontally reduced at the end. 24 zmm accumulators + 4 A loads + 1 B
/// load ≈ 29 registers.
#[allow(clippy::many_single_char_names)]
pub fn gemm_f32_tiled(a: &[f32], bt: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    unsafe { gemm_f32_dotprod_inner(a, bt, c, m, k, n) }
}

#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
#[target_feature(enable = "avx512f")]
unsafe fn gemm_f32_dotprod_inner(
    a: &[f32],
    bt: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let m_full = m - m % RM;
    let n_full = n - n % RN;

    // Full RM×RN tiles.
    for i in (0..m_full).step_by(RM) {
        for j in (0..n_full).step_by(RN) {
            microkernel_4x6(a, bt, c, k, n, i, j);
        }
        // N remainder (< 6 columns): scalar fallback.
        for jj in n_full..n {
            for ii in i..i + RM {
                let a_row = &a[ii * k..(ii + 1) * k];
                let bt_row = &bt[jj * k..(jj + 1) * k];
                c[ii * n + jj] = super::dot_f32(a_row, bt_row);
            }
        }
    }

    // M remainder (< 4 rows): scalar fallback.
    for ii in m_full..m {
        for jj in 0..n {
            let a_row = &a[ii * k..(ii + 1) * k];
            let bt_row = &bt[jj * k..(jj + 1) * k];
            c[ii * n + jj] = super::dot_f32(a_row, bt_row);
        }
    }
}

/// 4×6 dot-product AVX-512F micro-kernel (inline assembly).
///
/// Computes 4 rows × 6 columns of C = A · Bᵀ by accumulating dot products
/// along the full K dimension. Uses inline assembly to guarantee register
/// allocation: 24 zmm accumulators in zmm8–zmm31, A vectors in zmm0–zmm3,
/// B vector in zmm4. No register spilling.
///
/// Per K step (stride 16):
/// - Load 4 A vectors (contiguous along K)
/// - Load 6 Bᵀ vectors (contiguous along K, one at a time)
/// - 24 FMA operations
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names
)]
#[inline]
unsafe fn microkernel_4x6(
    a: &[f32],
    bt: &[f32],
    c: &mut [f32],
    k: usize,
    n: usize,
    i: usize,
    j: usize,
) {
    let a_ptr = a.as_ptr();
    let bt_ptr = bt.as_ptr();

    // Pointers to A rows.
    let a0 = a_ptr.add(i * k);
    let a1 = a_ptr.add((i + 1) * k);
    let a2 = a_ptr.add((i + 2) * k);
    let a3 = a_ptr.add((i + 3) * k);

    // Pointers to Bᵀ rows.
    let b0 = bt_ptr.add(j * k);
    let b1 = bt_ptr.add((j + 1) * k);
    let b2 = bt_ptr.add((j + 2) * k);
    let b3 = bt_ptr.add((j + 3) * k);
    let b4 = bt_ptr.add((j + 4) * k);
    let b5 = bt_ptr.add((j + 5) * k);

    let k_iters = k / KN; // number of full 16-float iterations
    let stride = KN * 4; // byte stride per iteration = 64

    // 24 output scalars.
    let mut out: [f32; RM * RN] = [0.0; RM * RN];

    if k_iters > 0 {
        // Inline asm: 24 accumulators in zmm8-zmm31, A in zmm0-3, B temp in zmm4.
        //
        // Register map (accumulators — cv[col][row]):
        //   col0: zmm8(r0)  zmm9(r1)  zmm10(r2) zmm11(r3)
        //   col1: zmm12(r0) zmm13(r1) zmm14(r2) zmm15(r3)
        //   col2: zmm16(r0) zmm17(r1) zmm18(r2) zmm19(r3)
        //   col3: zmm20(r0) zmm21(r1) zmm22(r2) zmm23(r3)
        //   col4: zmm24(r0) zmm25(r1) zmm26(r2) zmm27(r3)
        //   col5: zmm28(r0) zmm29(r1) zmm30(r2) zmm31(r3)
        std::arch::asm!(
            // Zero all 24 accumulators.
            "vpxord zmm8, zmm8, zmm8",
            "vpxord zmm9, zmm9, zmm9",
            "vpxord zmm10, zmm10, zmm10",
            "vpxord zmm11, zmm11, zmm11",
            "vpxord zmm12, zmm12, zmm12",
            "vpxord zmm13, zmm13, zmm13",
            "vpxord zmm14, zmm14, zmm14",
            "vpxord zmm15, zmm15, zmm15",
            "vpxord zmm16, zmm16, zmm16",
            "vpxord zmm17, zmm17, zmm17",
            "vpxord zmm18, zmm18, zmm18",
            "vpxord zmm19, zmm19, zmm19",
            "vpxord zmm20, zmm20, zmm20",
            "vpxord zmm21, zmm21, zmm21",
            "vpxord zmm22, zmm22, zmm22",
            "vpxord zmm23, zmm23, zmm23",
            "vpxord zmm24, zmm24, zmm24",
            "vpxord zmm25, zmm25, zmm25",
            "vpxord zmm26, zmm26, zmm26",
            "vpxord zmm27, zmm27, zmm27",
            "vpxord zmm28, zmm28, zmm28",
            "vpxord zmm29, zmm29, zmm29",
            "vpxord zmm30, zmm30, zmm30",
            "vpxord zmm31, zmm31, zmm31",

            // Loop counter.
            "xor {off:e}, {off:e}",   // off = 0 (byte offset)

            "2:",
            // Load 4 A vectors.
            "vmovups zmm0, [{a0} + {off}]",
            "vmovups zmm1, [{a1} + {off}]",
            "vmovups zmm2, [{a2} + {off}]",
            "vmovups zmm3, [{a3} + {off}]",

            // Col 0: load B0, FMA into zmm8-11.
            "vmovups zmm4, [{b0} + {off}]",
            "vfmadd231ps zmm8,  zmm0, zmm4",
            "vfmadd231ps zmm9,  zmm1, zmm4",
            "vfmadd231ps zmm10, zmm2, zmm4",
            "vfmadd231ps zmm11, zmm3, zmm4",

            // Col 1: load B1, FMA into zmm12-15.
            "vmovups zmm4, [{b1} + {off}]",
            "vfmadd231ps zmm12, zmm0, zmm4",
            "vfmadd231ps zmm13, zmm1, zmm4",
            "vfmadd231ps zmm14, zmm2, zmm4",
            "vfmadd231ps zmm15, zmm3, zmm4",

            // Col 2: load B2, FMA into zmm16-19.
            "vmovups zmm4, [{b2} + {off}]",
            "vfmadd231ps zmm16, zmm0, zmm4",
            "vfmadd231ps zmm17, zmm1, zmm4",
            "vfmadd231ps zmm18, zmm2, zmm4",
            "vfmadd231ps zmm19, zmm3, zmm4",

            // Col 3: load B3, FMA into zmm20-23.
            "vmovups zmm4, [{b3} + {off}]",
            "vfmadd231ps zmm20, zmm0, zmm4",
            "vfmadd231ps zmm21, zmm1, zmm4",
            "vfmadd231ps zmm22, zmm2, zmm4",
            "vfmadd231ps zmm23, zmm3, zmm4",

            // Col 4: load B4, FMA into zmm24-27.
            "vmovups zmm4, [{b4} + {off}]",
            "vfmadd231ps zmm24, zmm0, zmm4",
            "vfmadd231ps zmm25, zmm1, zmm4",
            "vfmadd231ps zmm26, zmm2, zmm4",
            "vfmadd231ps zmm27, zmm3, zmm4",

            // Col 5: load B5, FMA into zmm28-31.
            "vmovups zmm4, [{b5} + {off}]",
            "vfmadd231ps zmm28, zmm0, zmm4",
            "vfmadd231ps zmm29, zmm1, zmm4",
            "vfmadd231ps zmm30, zmm2, zmm4",
            "vfmadd231ps zmm31, zmm3, zmm4",

            // Advance and loop.
            "add {off}, {stride}",
            "cmp {off}, {k_bytes}",
            "jb 2b",

            // ---- Horizontal reduction ----
            // Reduce each zmm accumulator to a scalar and store to `out` array.
            // Strategy: extract hi256, add to lo256, then 256→128→scalar.
            //
            // We reuse zmm0-zmm5 as scratch for reductions.

            // Macro-like pattern: reduce zmm{src} → store to [out + offset].
            // Step 1: vextractf32x8 ymm5 = hi256 of zmm{src}
            // Step 2: vaddps ymm5, ymm5, ymm_lo(zmm{src})  (lo256 is ymm{src})
            // Step 3: vextractf128 xmm4, ymm5, 1
            // Step 4: vaddps xmm4, xmm4, xmm5
            // Step 5: vmovshdup xmm3, xmm4   ; [1,1,3,3]
            // Step 6: vaddps xmm4, xmm4, xmm3
            // Step 7: vpermilps xmm3, xmm4, 0x4E  ; swap hi64/lo64
            // Step 8: vaddss xmm4, xmm4, xmm3
            // Step 9: vmovss [out + offset], xmm4

            // We generate this for all 24 accumulators (zmm8-zmm31).
            // out layout: [row0_col0, row0_col1, ..., row0_col5, row1_col0, ...]

            // --- Row 0 (zmm8, zmm12, zmm16, zmm20, zmm24, zmm28) ---
            "vextractf32x8 ymm5, zmm8, 1",
            "vaddps ymm5, ymm5, ymm8",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr}], xmm4",

            "vextractf32x8 ymm5, zmm12, 1",
            "vaddps ymm5, ymm5, ymm12",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 4], xmm4",

            "vextractf32x8 ymm5, zmm16, 1",
            "vaddps ymm5, ymm5, ymm16",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 8], xmm4",

            "vextractf32x8 ymm5, zmm20, 1",
            "vaddps ymm5, ymm5, ymm20",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 12], xmm4",

            "vextractf32x8 ymm5, zmm24, 1",
            "vaddps ymm5, ymm5, ymm24",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 16], xmm4",

            "vextractf32x8 ymm5, zmm28, 1",
            "vaddps ymm5, ymm5, ymm28",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 20], xmm4",

            // --- Row 1 (zmm9, zmm13, zmm17, zmm21, zmm25, zmm29) ---
            "vextractf32x8 ymm5, zmm9, 1",
            "vaddps ymm5, ymm5, ymm9",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 24], xmm4",

            "vextractf32x8 ymm5, zmm13, 1",
            "vaddps ymm5, ymm5, ymm13",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 28], xmm4",

            "vextractf32x8 ymm5, zmm17, 1",
            "vaddps ymm5, ymm5, ymm17",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 32], xmm4",

            "vextractf32x8 ymm5, zmm21, 1",
            "vaddps ymm5, ymm5, ymm21",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 36], xmm4",

            "vextractf32x8 ymm5, zmm25, 1",
            "vaddps ymm5, ymm5, ymm25",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 40], xmm4",

            "vextractf32x8 ymm5, zmm29, 1",
            "vaddps ymm5, ymm5, ymm29",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 44], xmm4",

            // --- Row 2 (zmm10, zmm14, zmm18, zmm22, zmm26, zmm30) ---
            "vextractf32x8 ymm5, zmm10, 1",
            "vaddps ymm5, ymm5, ymm10",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 48], xmm4",

            "vextractf32x8 ymm5, zmm14, 1",
            "vaddps ymm5, ymm5, ymm14",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 52], xmm4",

            "vextractf32x8 ymm5, zmm18, 1",
            "vaddps ymm5, ymm5, ymm18",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 56], xmm4",

            "vextractf32x8 ymm5, zmm22, 1",
            "vaddps ymm5, ymm5, ymm22",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 60], xmm4",

            "vextractf32x8 ymm5, zmm26, 1",
            "vaddps ymm5, ymm5, ymm26",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 64], xmm4",

            "vextractf32x8 ymm5, zmm30, 1",
            "vaddps ymm5, ymm5, ymm30",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 68], xmm4",

            // --- Row 3 (zmm11, zmm15, zmm19, zmm23, zmm27, zmm31) ---
            "vextractf32x8 ymm5, zmm11, 1",
            "vaddps ymm5, ymm5, ymm11",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 72], xmm4",

            "vextractf32x8 ymm5, zmm15, 1",
            "vaddps ymm5, ymm5, ymm15",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 76], xmm4",

            "vextractf32x8 ymm5, zmm19, 1",
            "vaddps ymm5, ymm5, ymm19",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 80], xmm4",

            "vextractf32x8 ymm5, zmm23, 1",
            "vaddps ymm5, ymm5, ymm23",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 84], xmm4",

            "vextractf32x8 ymm5, zmm27, 1",
            "vaddps ymm5, ymm5, ymm27",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 88], xmm4",

            "vextractf32x8 ymm5, zmm31, 1",
            "vaddps ymm5, ymm5, ymm31",
            "vextractf128 xmm4, ymm5, 1",
            "vaddps xmm4, xmm4, xmm5",
            "vmovshdup xmm3, xmm4",
            "vaddps xmm4, xmm4, xmm3",
            "vpermilps xmm3, xmm4, 0x4E",
            "vaddss xmm4, xmm4, xmm3",
            "vmovss [{out_ptr} + 92], xmm4",

            a0 = in(reg) a0,
            a1 = in(reg) a1,
            a2 = in(reg) a2,
            a3 = in(reg) a3,
            b0 = in(reg) b0,
            b1 = in(reg) b1,
            b2 = in(reg) b2,
            b3 = in(reg) b3,
            b4 = in(reg) b4,
            b5 = in(reg) b5,
            off = out(reg) _,
            stride = in(reg) stride,
            k_bytes = in(reg) k_iters * stride,
            out_ptr = in(reg) out.as_mut_ptr(),
            out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
            out("zmm4") _, out("zmm5") _,
            out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
            out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _,
            out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
            out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _,
            out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _,
            out("zmm28") _, out("zmm29") _, out("zmm30") _, out("zmm31") _,
            options(nostack),
        );
    }

    // K remainder: scalar accumulation for the last < 16 elements.
    let k_done = k_iters * KN;
    for p in k_done..k {
        let va = [
            *a_ptr.add(i * k + p),
            *a_ptr.add((i + 1) * k + p),
            *a_ptr.add((i + 2) * k + p),
            *a_ptr.add((i + 3) * k + p),
        ];
        for jj in 0..RN {
            let bv = *bt_ptr.add((j + jj) * k + p);
            for ii in 0..RM {
                out[ii * RN + jj] += va[ii] * bv;
            }
        }
    }

    // Write output to C.
    let c_ptr = c.as_mut_ptr();
    for ii in 0..RM {
        for jj in 0..RN {
            *c_ptr.add((i + ii) * n + (j + jj)) = out[ii * RN + jj];
        }
    }
}

// ---- Tiled Q8×Q8 GEMM (AVX-512 VNNI) ----
//
// Processes RM_Q8=4 input rows × RN_Q8=4 weight columns per micro-kernel
// invocation. Uses 16 ymm accumulators (ymm16-ymm31) that stay in registers
// across the full K (block) loop, with pre-computed combined scales to
// minimize GP register pressure in the inline asm.
//
// Per block, per weight column: load weight quants, vpsignb for abs, then
// for each input row: vpsignb + vpdpbusd + vcvtdq2ps + vfmadd231ps.

/// Micro-kernel row count for Q8 GEMM.
const RM_Q8: usize = 4;
/// Micro-kernel column count for Q8 GEMM.
const RN_Q8: usize = 6;

/// Tiled Q8×Q8 GEMM: `output[m, n] = inp_quants[m, k] · wt_quants[n, k]`
/// with per-block scales.
///
/// Both input and weight data are in Q8_0 format: K dimension divided into
/// blocks of 32 elements, each with a `[u8; 32]` quant array and one `f32`
/// scale.
///
/// Uses a 4×6 micro-kernel with AVX-512 VNNI for full tiles, falling back
/// to single-row `dot_q8_q8_row` for remainder rows/columns.
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
    unsafe {
        gemm_q8_tiled_inner(
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
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]
#[target_feature(
    enable = "avx512f",
    enable = "avx512vnni",
    enable = "avx512vl",
    enable = "avx512bw",
    enable = "fma"
)]
unsafe fn gemm_q8_tiled_inner(
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
    let m_full = m - m % RM_Q8;
    let n_full = n - n % RN_Q8;

    // Full RM_Q8 × RN_Q8 tiles.
    for i in (0..m_full).step_by(RM_Q8) {
        for j in (0..n_full).step_by(RN_Q8) {
            microkernel_q8_4x6(
                output,
                inp_quants,
                inp_scales,
                wt_quants,
                wt_scales,
                n,
                num_blocks,
                bytes_per_row,
                i,
                j,
            );
        }
        // N remainder: scalar fallback.
        for jj in n_full..n {
            for ii in i..i + RM_Q8 {
                output[ii * n + jj] = dot_q8_q8_row(
                    &inp_quants[ii * bytes_per_row..(ii + 1) * bytes_per_row],
                    &inp_scales[ii * num_blocks..(ii + 1) * num_blocks],
                    &wt_quants[jj * bytes_per_row..(jj + 1) * bytes_per_row],
                    &wt_scales[jj * num_blocks..(jj + 1) * num_blocks],
                );
            }
        }
    }

    // M remainder: scalar fallback.
    for ii in m_full..m {
        for jj in 0..n {
            output[ii * n + jj] = dot_q8_q8_row(
                &inp_quants[ii * bytes_per_row..(ii + 1) * bytes_per_row],
                &inp_scales[ii * num_blocks..(ii + 1) * num_blocks],
                &wt_quants[jj * bytes_per_row..(jj + 1) * bytes_per_row],
                &wt_scales[jj * num_blocks..(jj + 1) * num_blocks],
            );
        }
    }
}

/// Maximum number of quantization blocks supported per row.
///
/// Covers K up to 8192 (8192 / 32 = 256 blocks). Models with larger hidden
/// dimensions would need this increased.
const MAX_Q8_BLOCKS: usize = 256;

/// 4×6 Q8×Q8 micro-kernel using AVX-512 VNNI inline assembly.
///
/// Computes a 4-row × 6-column tile of the output matrix by accumulating
/// dot products over all K blocks. Uses 24 ymm accumulators (ymm8-ymm31)
/// that remain in registers across the entire block loop.
///
/// Register map:
/// - ymm0-ymm3: 4 input quant blocks (reloaded per K block)
/// - ymm4: weight quants (reloaded per column within block)
/// - ymm5: w_abs = vpsignb(w, w)
/// - ymm6, ymm7: scratch (inp_signed, zero+dpbusd, cvt, scale)
/// - ymm8-ymm11:  accumulators for col 0, rows 0-3
/// - ymm12-ymm15: accumulators for col 1, rows 0-3
/// - ymm16-ymm19: accumulators for col 2, rows 0-3
/// - ymm20-ymm23: accumulators for col 3, rows 0-3
/// - ymm24-ymm27: accumulators for col 4, rows 0-3
/// - ymm28-ymm31: accumulators for col 5, rows 0-3
///
/// Combined scales are pre-computed on the stack: `combined[blk][col][row]` =
/// `inp_scales[row][blk] * wt_scales[col][blk]`, laid out as
/// `[num_blocks * RN_Q8 * RM_Q8]` f32 values. Per block the asm advances
/// by 96 bytes (24 floats × 4 bytes).
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::large_stack_arrays
)]
#[inline]
unsafe fn microkernel_q8_4x6(
    output: &mut [f32],
    inp_quants: &[u8],
    inp_scales: &[f32],
    wt_quants: &[u8],
    wt_scales: &[f32],
    n: usize,
    num_blocks: usize,
    bytes_per_row: usize,
    i: usize,
    j: usize,
) {
    debug_assert!(
        num_blocks <= MAX_Q8_BLOCKS,
        "num_blocks ({num_blocks}) exceeds MAX_Q8_BLOCKS ({MAX_Q8_BLOCKS})"
    );

    // Pre-compute combined scales on the stack: layout [blk][col][row].
    let mut combined_scales = [0.0f32; MAX_Q8_BLOCKS * RN_Q8 * RM_Q8];
    for blk in 0..num_blocks {
        for col in 0..RN_Q8 {
            for row in 0..RM_Q8 {
                *combined_scales.get_unchecked_mut(blk * (RN_Q8 * RM_Q8) + col * RM_Q8 + row) =
                    *inp_scales.get_unchecked((i + row) * num_blocks + blk)
                        * *wt_scales.get_unchecked((j + col) * num_blocks + blk);
            }
        }
    }

    // Pointers to input quant rows.
    let iq0 = inp_quants.as_ptr().add(i * bytes_per_row);
    let iq1 = inp_quants.as_ptr().add((i + 1) * bytes_per_row);
    let iq2 = inp_quants.as_ptr().add((i + 2) * bytes_per_row);
    let iq3 = inp_quants.as_ptr().add((i + 3) * bytes_per_row);

    // Pointers to weight quant rows.
    let wq0 = wt_quants.as_ptr().add(j * bytes_per_row);
    let wq1 = wt_quants.as_ptr().add((j + 1) * bytes_per_row);
    let wq2 = wt_quants.as_ptr().add((j + 2) * bytes_per_row);
    let wq3 = wt_quants.as_ptr().add((j + 3) * bytes_per_row);
    let wq4 = wt_quants.as_ptr().add((j + 4) * bytes_per_row);
    let wq5 = wt_quants.as_ptr().add((j + 5) * bytes_per_row);

    let cs_ptr = combined_scales.as_ptr();
    let q_limit = num_blocks * 32; // loop limit in bytes

    // 24 output scalars: out[col * RM_Q8 + row].
    let mut out = [0.0f32; RM_Q8 * RN_Q8];

    if num_blocks > 0 {
        std::arch::asm!(
            // Zero 24 accumulators (ymm8-ymm31).
            "vpxord ymm8, ymm8, ymm8",
            "vpxord ymm9, ymm9, ymm9",
            "vpxord ymm10, ymm10, ymm10",
            "vpxord ymm11, ymm11, ymm11",
            "vpxord ymm12, ymm12, ymm12",
            "vpxord ymm13, ymm13, ymm13",
            "vpxord ymm14, ymm14, ymm14",
            "vpxord ymm15, ymm15, ymm15",
            "vpxord ymm16, ymm16, ymm16",
            "vpxord ymm17, ymm17, ymm17",
            "vpxord ymm18, ymm18, ymm18",
            "vpxord ymm19, ymm19, ymm19",
            "vpxord ymm20, ymm20, ymm20",
            "vpxord ymm21, ymm21, ymm21",
            "vpxord ymm22, ymm22, ymm22",
            "vpxord ymm23, ymm23, ymm23",
            "vpxord ymm24, ymm24, ymm24",
            "vpxord ymm25, ymm25, ymm25",
            "vpxord ymm26, ymm26, ymm26",
            "vpxord ymm27, ymm27, ymm27",
            "vpxord ymm28, ymm28, ymm28",
            "vpxord ymm29, ymm29, ymm29",
            "vpxord ymm30, ymm30, ymm30",
            "vpxord ymm31, ymm31, ymm31",

            // Loop counters.
            "xor {q_off:e}, {q_off:e}",
            "xor {cs_off:e}, {cs_off:e}",

            "2:",
            // Load 4 input quant blocks (32 bytes each) into ymm0-3.
            "vmovdqu ymm0, [{iq0} + {q_off}]",
            "vmovdqu ymm1, [{iq1} + {q_off}]",
            "vmovdqu ymm2, [{iq2} + {q_off}]",
            "vmovdqu ymm3, [{iq3} + {q_off}]",

            // ---- Col 0 (wq0, acc ymm8-11, cs_off + 0..12) ----
            "vmovdqu ymm4, [{wq0} + {q_off}]",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off}]",
            "vfmadd231ps ymm8, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 4]",
            "vfmadd231ps ymm9, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 8]",
            "vfmadd231ps ymm10, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 12]",
            "vfmadd231ps ymm11, ymm6, ymm7",

            // ---- Col 1 (wq1, acc ymm12-15, cs_off + 16..28) ----
            "vmovdqu ymm4, [{wq1} + {q_off}]",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 16]",
            "vfmadd231ps ymm12, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 20]",
            "vfmadd231ps ymm13, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 24]",
            "vfmadd231ps ymm14, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 28]",
            "vfmadd231ps ymm15, ymm6, ymm7",

            // ---- Col 2 (wq2, acc ymm16-19, cs_off + 32..44) ----
            "vmovdqu ymm4, [{wq2} + {q_off}]",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 32]",
            "vfmadd231ps ymm16, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 36]",
            "vfmadd231ps ymm17, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 40]",
            "vfmadd231ps ymm18, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 44]",
            "vfmadd231ps ymm19, ymm6, ymm7",

            // ---- Col 3 (wq3, acc ymm20-23, cs_off + 48..60) ----
            "vmovdqu ymm4, [{wq3} + {q_off}]",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 48]",
            "vfmadd231ps ymm20, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 52]",
            "vfmadd231ps ymm21, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 56]",
            "vfmadd231ps ymm22, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 60]",
            "vfmadd231ps ymm23, ymm6, ymm7",

            // ---- Col 4 (wq4, acc ymm24-27, cs_off + 64..76) ----
            "vmovdqu ymm4, [{wq4} + {q_off}]",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 64]",
            "vfmadd231ps ymm24, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 68]",
            "vfmadd231ps ymm25, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 72]",
            "vfmadd231ps ymm26, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 76]",
            "vfmadd231ps ymm27, ymm6, ymm7",

            // ---- Col 5 (wq5, acc ymm28-31, cs_off + 80..92) ----
            "vmovdqu ymm4, [{wq5} + {q_off}]",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 80]",
            "vfmadd231ps ymm28, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 84]",
            "vfmadd231ps ymm29, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 88]",
            "vfmadd231ps ymm30, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{cs_ptr} + {cs_off} + 92]",
            "vfmadd231ps ymm31, ymm6, ymm7",

            // Advance block.
            "add {q_off}, 32",
            "add {cs_off}, 96",
            "cmp {q_off}, {q_limit}",
            "jb 2b",

            // ---- Horizontal reduction ----
            // Reduce each ymm accumulator (8 f32) → scalar, store to out[].
            // For ymm16+, use EVEX vextractf32x4; for ymm8-15 use vextractf128.
            // Pattern: extract hi128 → xmm7, extract lo128 → xmm6, add, hsum, store.

            // Col 0: ymm8 → out[0], ymm9 → out[1], ymm10 → out[2], ymm11 → out[3]
            "vextractf128 xmm7, ymm8, 1",
            "vextractf128 xmm6, ymm8, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr}], xmm7",

            "vextractf128 xmm7, ymm9, 1",
            "vextractf128 xmm6, ymm9, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 4], xmm7",

            "vextractf128 xmm7, ymm10, 1",
            "vextractf128 xmm6, ymm10, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 8], xmm7",

            "vextractf128 xmm7, ymm11, 1",
            "vextractf128 xmm6, ymm11, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 12], xmm7",

            // Col 1: ymm12 → out[4], ymm13 → out[5], ymm14 → out[6], ymm15 → out[7]
            "vextractf128 xmm7, ymm12, 1",
            "vextractf128 xmm6, ymm12, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 16], xmm7",

            "vextractf128 xmm7, ymm13, 1",
            "vextractf128 xmm6, ymm13, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 20], xmm7",

            "vextractf128 xmm7, ymm14, 1",
            "vextractf128 xmm6, ymm14, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 24], xmm7",

            "vextractf128 xmm7, ymm15, 1",
            "vextractf128 xmm6, ymm15, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 28], xmm7",

            // Col 2: ymm16 → out[8], ymm17 → out[9], ymm18 → out[10], ymm19 → out[11]
            "vextractf32x4 xmm7, ymm16, 1",
            "vextractf32x4 xmm6, ymm16, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 32], xmm7",

            "vextractf32x4 xmm7, ymm17, 1",
            "vextractf32x4 xmm6, ymm17, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 36], xmm7",

            "vextractf32x4 xmm7, ymm18, 1",
            "vextractf32x4 xmm6, ymm18, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 40], xmm7",

            "vextractf32x4 xmm7, ymm19, 1",
            "vextractf32x4 xmm6, ymm19, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 44], xmm7",

            // Col 3: ymm20 → out[12], ymm21 → out[13], ymm22 → out[14], ymm23 → out[15]
            "vextractf32x4 xmm7, ymm20, 1",
            "vextractf32x4 xmm6, ymm20, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 48], xmm7",

            "vextractf32x4 xmm7, ymm21, 1",
            "vextractf32x4 xmm6, ymm21, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 52], xmm7",

            "vextractf32x4 xmm7, ymm22, 1",
            "vextractf32x4 xmm6, ymm22, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 56], xmm7",

            "vextractf32x4 xmm7, ymm23, 1",
            "vextractf32x4 xmm6, ymm23, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 60], xmm7",

            // Col 4: ymm24 → out[16], ymm25 → out[17], ymm26 → out[18], ymm27 → out[19]
            "vextractf32x4 xmm7, ymm24, 1",
            "vextractf32x4 xmm6, ymm24, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 64], xmm7",

            "vextractf32x4 xmm7, ymm25, 1",
            "vextractf32x4 xmm6, ymm25, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 68], xmm7",

            "vextractf32x4 xmm7, ymm26, 1",
            "vextractf32x4 xmm6, ymm26, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 72], xmm7",

            "vextractf32x4 xmm7, ymm27, 1",
            "vextractf32x4 xmm6, ymm27, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 76], xmm7",

            // Col 5: ymm28 → out[20], ymm29 → out[21], ymm30 → out[22], ymm31 → out[23]
            "vextractf32x4 xmm7, ymm28, 1",
            "vextractf32x4 xmm6, ymm28, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 80], xmm7",

            "vextractf32x4 xmm7, ymm29, 1",
            "vextractf32x4 xmm6, ymm29, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 84], xmm7",

            "vextractf32x4 xmm7, ymm30, 1",
            "vextractf32x4 xmm6, ymm30, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 88], xmm7",

            "vextractf32x4 xmm7, ymm31, 1",
            "vextractf32x4 xmm6, ymm31, 0",
            "vaddps xmm7, xmm7, xmm6",
            "vmovshdup xmm6, xmm7",
            "vaddps xmm7, xmm7, xmm6",
            "vpermilps xmm6, xmm7, 0x4E",
            "vaddss xmm7, xmm7, xmm6",
            "vmovss [{out_ptr} + 92], xmm7",

            iq0 = in(reg) iq0,
            iq1 = in(reg) iq1,
            iq2 = in(reg) iq2,
            iq3 = in(reg) iq3,
            wq0 = in(reg) wq0,
            wq1 = in(reg) wq1,
            wq2 = in(reg) wq2,
            wq3 = in(reg) wq3,
            wq4 = in(reg) wq4,
            wq5 = in(reg) wq5,
            cs_ptr = in(reg) cs_ptr,
            q_limit = in(reg) q_limit,
            out_ptr = in(reg) out.as_mut_ptr(),
            q_off = out(reg) _,
            cs_off = out(reg) _,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
            out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
            out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
            out("ymm16") _, out("ymm17") _, out("ymm18") _, out("ymm19") _,
            out("ymm20") _, out("ymm21") _, out("ymm22") _, out("ymm23") _,
            out("ymm24") _, out("ymm25") _, out("ymm26") _, out("ymm27") _,
            out("ymm28") _, out("ymm29") _, out("ymm30") _, out("ymm31") _,
            options(nostack),
        );
    }

    // Write results to output matrix.
    // out layout: [col * RM_Q8 + row], output layout: row-major [row * n + col].
    let out_ptr = output.as_mut_ptr();
    for row in 0..RM_Q8 {
        for col in 0..RN_Q8 {
            *out_ptr.add((i + row) * n + (j + col)) = out[col * RM_Q8 + row];
        }
    }
}

// ---- Vectorized SiLU-Mul ----

/// Vectorized SiLU-Mul: `out[i] = silu(gate[i]) * up[i]`
/// where `silu(x) = x / (1 + exp(-x))`.
///
/// Uses AVX-512F with a fast polynomial exp approximation (degree-4 minimax
/// on `[-87, 0]`, relative error < 2e-7). Processes 16 floats per iteration.
pub fn vec_silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    unsafe { vec_silu_mul_inner(gate, up, out) }
}

#[target_feature(enable = "avx512f")]
#[allow(clippy::many_single_char_names)]
unsafe fn vec_silu_mul_inner(gate: &[f32], up: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::{
        __m512, __m512i, _mm512_add_epi32, _mm512_add_ps, _mm512_castps_si512, _mm512_castsi512_ps,
        _mm512_cvtepi32_ps, _mm512_cvtps_epi32, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_max_ps,
        _mm512_min_ps, _mm512_mul_ps, _mm512_set1_ps, _mm512_slli_epi32, _mm512_storeu_ps,
        _mm512_sub_ps,
    };

    let n = gate.len();
    let chunks = n / 16;
    let remainder = n % 16;

    // Constants for exp(x) approximation.
    // Method: exp(x) = 2^(x * log2e) = 2^n * 2^f, where n=floor, f=fractional.
    // 2^n via integer shift, 2^f via degree-4 polynomial.
    let log2e: __m512 = _mm512_set1_ps(std::f32::consts::LOG2_E);
    let one: __m512 = _mm512_set1_ps(1.0);

    // Minimax polynomial coefficients for 2^f on [0, 1):
    // p(f) ≈ c0 + f*(c1 + f*(c2 + f*(c3 + f*c4)))
    let c0: __m512 = _mm512_set1_ps(1.0);
    #[allow(clippy::approx_constant)]
    let c1: __m512 = _mm512_set1_ps(0.693_147_2_f32);
    let c2: __m512 = _mm512_set1_ps(0.240_226_5_f32);
    let c3: __m512 = _mm512_set1_ps(5.550_357e-2_f32);
    let c4: __m512 = _mm512_set1_ps(9.675_54e-3_f32);

    // Clamp range to avoid overflow/underflow in exp.
    let exp_lo: __m512 = _mm512_set1_ps(-87.332_54_f32); // ln(FLT_MIN)
    let exp_hi: __m512 = _mm512_set1_ps(88.722_84_f32); // ln(FLT_MAX)

    let gate_ptr = gate.as_ptr();
    let up_ptr = up.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 16;
        let g: __m512 = _mm512_loadu_ps(gate_ptr.add(offset));
        let u: __m512 = _mm512_loadu_ps(up_ptr.add(offset));

        // neg_g = -gate (for exp(-gate))
        let neg_g: __m512 = _mm512_sub_ps(_mm512_set1_ps(0.0), g);

        // Clamp to valid exp range.
        let x: __m512 = _mm512_max_ps(_mm512_min_ps(neg_g, exp_hi), exp_lo);

        // t = x * log2(e)
        let t: __m512 = _mm512_mul_ps(x, log2e);

        // n = round(t) — round to nearest integer
        let n_f: __m512 = _mm512_cvtepi32_ps(_mm512_cvtps_epi32(t));

        // f = t - n (fractional part, in [-0.5, 0.5])
        // Shift to [0, 1) by adding 0.5 and adjusting n.
        let f: __m512 = _mm512_sub_ps(t, n_f);

        // Polynomial: 2^f ≈ c0 + f*(c1 + f*(c2 + f*(c3 + f*c4)))
        let poly: __m512 = _mm512_fmadd_ps(c4, f, c3);
        let poly: __m512 = _mm512_fmadd_ps(poly, f, c2);
        let poly: __m512 = _mm512_fmadd_ps(poly, f, c1);
        let poly: __m512 = _mm512_fmadd_ps(poly, f, c0);

        // 2^n: add n to the exponent field of poly (IEEE 754 trick).
        // poly * 2^n = poly with exponent += n
        let n_i: __m512i = _mm512_cvtps_epi32(n_f);
        let shift: __m512i = _mm512_slli_epi32(n_i, 23); // shift n into exponent position
        let exp_val: __m512 =
            _mm512_castsi512_ps(_mm512_add_epi32(_mm512_castps_si512(poly), shift));

        // sigmoid(-g) = exp(-g) / (1 + exp(-g))
        // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
        let denom: __m512 = _mm512_add_ps(one, exp_val);
        // out = gate * up / (1 + exp(-gate))
        let numer: __m512 = _mm512_mul_ps(g, u);
        // Use: numer / denom. AVX-512F has vrcp14ps for fast reciprocal,
        // but division is more accurate and still fast on modern CPUs.
        let result: __m512 = {
            use std::arch::x86_64::_mm512_div_ps;
            _mm512_div_ps(numer, denom)
        };

        _mm512_storeu_ps(out_ptr.add(offset), result);
    }

    // Scalar tail.
    let tail = chunks * 16;
    for i in 0..remainder {
        let g = *gate.get_unchecked(tail + i);
        let u = *up.get_unchecked(tail + i);
        let sigmoid = 1.0 / (1.0 + (-g).exp());
        *out.get_unchecked_mut(tail + i) = g * sigmoid * u;
    }
}

// ---- Vectorized Softmax ----

/// In-place softmax using AVX-512F fast polynomial exp.
pub fn vec_softmax_inplace(data: &mut [f32]) {
    unsafe { vec_softmax_inplace_inner(data) }
}

#[target_feature(enable = "avx512f")]
unsafe fn vec_softmax_inplace_inner(data: &mut [f32]) {
    use std::arch::x86_64::{
        __m512, _mm512_add_epi32, _mm512_add_ps, _mm512_castps_si512, _mm512_castsi512_ps,
        _mm512_cvtepi32_ps, _mm512_cvtps_epi32, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_max_ps,
        _mm512_min_ps, _mm512_mul_ps, _mm512_reduce_add_ps, _mm512_reduce_max_ps, _mm512_set1_ps,
        _mm512_slli_epi32, _mm512_storeu_ps, _mm512_sub_ps,
    };

    let n = data.len();
    let chunks = n / 16;
    let remainder = n % 16;
    let ptr = data.as_mut_ptr();

    // 1. Find max.
    let mut vmax: __m512 = _mm512_set1_ps(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = _mm512_loadu_ps(ptr.add(i * 16));
        vmax = _mm512_max_ps(vmax, v);
    }
    let mut max_val = _mm512_reduce_max_ps(vmax);
    let tail = chunks * 16;
    for i in 0..remainder {
        let v = *ptr.add(tail + i);
        if v > max_val {
            max_val = v;
        }
    }

    // Fast exp constants (same as vec_silu_mul).
    let log2e = _mm512_set1_ps(std::f32::consts::LOG2_E);
    let c0 = _mm512_set1_ps(1.0);
    #[allow(clippy::approx_constant)]
    let c1 = _mm512_set1_ps(0.693_147_2_f32);
    let c2 = _mm512_set1_ps(0.240_226_5_f32);
    let c3 = _mm512_set1_ps(5.550_357e-2_f32);
    let c4 = _mm512_set1_ps(9.675_54e-3_f32);
    let exp_lo = _mm512_set1_ps(-87.332_54_f32);
    let exp_hi = _mm512_set1_ps(88.722_84_f32);
    let vmax_broadcast = _mm512_set1_ps(max_val);

    // Inline fast exp(x) for a single zmm register.
    macro_rules! fast_exp_zmm {
        ($x:expr) => {{
            let x = _mm512_max_ps(_mm512_min_ps($x, exp_hi), exp_lo);
            let t = _mm512_mul_ps(x, log2e);
            let n_f = _mm512_cvtepi32_ps(_mm512_cvtps_epi32(t));
            let f = _mm512_sub_ps(t, n_f);
            let poly = _mm512_fmadd_ps(c4, f, c3);
            let poly = _mm512_fmadd_ps(poly, f, c2);
            let poly = _mm512_fmadd_ps(poly, f, c1);
            let poly = _mm512_fmadd_ps(poly, f, c0);
            let n_i = _mm512_cvtps_epi32(n_f);
            let shift = _mm512_slli_epi32(n_i, 23);
            _mm512_castsi512_ps(_mm512_add_epi32(_mm512_castps_si512(poly), shift))
        }};
    }

    // 2. Compute exp(x - max) in-place and sum.
    let mut vsum: __m512 = _mm512_set1_ps(0.0);
    for i in 0..chunks {
        let offset = i * 16;
        let v = _mm512_loadu_ps(ptr.add(offset));
        let x = _mm512_sub_ps(v, vmax_broadcast);
        let e = fast_exp_zmm!(x);
        _mm512_storeu_ps(ptr.add(offset), e);
        vsum = _mm512_add_ps(vsum, e);
    }
    let mut sum_val = _mm512_reduce_add_ps(vsum);
    for i in 0..remainder {
        let x = *ptr.add(tail + i) - max_val;
        let e = x.exp();
        *ptr.add(tail + i) = e;
        sum_val += e;
    }

    // 3. Normalize by 1/sum.
    if sum_val > 0.0 {
        let vinv = _mm512_set1_ps(1.0 / sum_val);
        for i in 0..chunks {
            let offset = i * 16;
            let v = _mm512_loadu_ps(ptr.add(offset));
            _mm512_storeu_ps(ptr.add(offset), _mm512_mul_ps(v, vinv));
        }
        let inv = 1.0 / sum_val;
        for i in 0..remainder {
            *ptr.add(tail + i) *= inv;
        }
    }
}
