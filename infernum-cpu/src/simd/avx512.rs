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
    _mm256_setzero_ps, _mm256_setzero_si256, _mm256_sign_epi8, _mm512_add_ps, _mm512_fmadd_ps,
    _mm512_loadu_ps, _mm512_set1_ps, _mm512_setzero_ps, _mm512_storeu_ps, _mm_add_ps, _mm_add_ss,
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

// ---- Tiled F32 GEMM (AVX-512F + FMA) ----

/// Cache-blocking tile sizes for the tiled GEMM.
///
/// - `KC` = 256 floats → 1 KB per row of A panel (fits comfortably in L1d=32 KB).
/// - `NC` = 128 columns of Bᵀ → Bᵀ panel is NC×KC = 128×256 = 128 KB (fits in L2=1 MB
///   with room for the A panel).
const KC: usize = 256;
const NC: usize = 128;

/// Micro-kernel tile sizes: 6 rows of A × 32 columns of Bᵀ.
const MR: usize = 6;
const NR: usize = 32;

/// Tiled F32 GEMM: `C[m,n] += A[m,k] * Bᵀ[n,k]`.
///
/// - `a` is row-major `(M, K)`.
/// - `bt` is row-major `(N, K)` — i.e., B transposed so each "row" of `bt` is
///   one column of the original B.
/// - `c` is row-major `(M, N)`, must be zero-initialized by the caller.
///
/// Uses a 6×32 micro-kernel with AVX-512F FMA, with cache-blocked tiling
/// (`KC=256`, `NC=128`) for L1/L2 locality. Edge tiles (M%6, N%32) fall back
/// to scalar dot products.
#[allow(clippy::many_single_char_names)]
pub fn gemm_f32_tiled(a: &[f32], bt: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    unsafe { gemm_f32_tiled_inner(a, bt, c, m, k, n) }
}

#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
#[target_feature(enable = "avx512f")]
unsafe fn gemm_f32_tiled_inner(a: &[f32], bt: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    let m_full = m - m % MR;

    for k_start in (0..k).step_by(KC) {
        let kc = KC.min(k - k_start);

        for n_start in (0..n).step_by(NC) {
            let nc = NC.min(n - n_start);
            let n_full = nc - nc % NR;

            // Process full MR×NR tiles.
            for i in (0..m_full).step_by(MR) {
                for j in (0..n_full).step_by(NR) {
                    microkernel_6x32(a, bt, c, k, n, i, n_start + j, k_start, kc);
                }
                // N remainder (< 32 columns): scalar fallback.
                if n_full < nc {
                    for ii in i..i + MR {
                        for jj in n_full..nc {
                            let col = n_start + jj;
                            let a_row = &a[ii * k + k_start..ii * k + k_start + kc];
                            let bt_row = &bt[col * k + k_start..col * k + k_start + kc];
                            c[ii * n + col] += super::dot_f32(a_row, bt_row);
                        }
                    }
                }
            }

            // M remainder (< 6 rows): scalar fallback.
            for ii in m_full..m {
                for jj in 0..nc {
                    let col = n_start + jj;
                    let a_row = &a[ii * k + k_start..ii * k + k_start + kc];
                    let bt_row = &bt[col * k + k_start..col * k + k_start + kc];
                    c[ii * n + col] += super::dot_f32(a_row, bt_row);
                }
            }
        }
    }
}

/// 6×32 AVX-512F micro-kernel.
///
/// Computes a 6-row × 32-column tile of `C += A * Bᵀ` for a `kc`-length
/// slice of the K dimension. Uses 12 zmm accumulators (6 rows × 2 zmm of
/// 16 f32 each = 32 columns).
///
/// `bt` is (N,K) row-major. For each K step, gathers 32 values from NR
/// rows of `bt` (stride = K) into a local buffer, then FMAs against 6
/// broadcast A values.
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::many_single_char_names
)]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn microkernel_6x32(
    a: &[f32],
    bt: &[f32],
    c: &mut [f32],
    k: usize,
    n: usize,
    i: usize,
    j: usize,
    k_start: usize,
    kc: usize,
) {
    let mut acc0a = _mm512_setzero_ps();
    let mut acc0b = _mm512_setzero_ps();
    let mut acc1a = _mm512_setzero_ps();
    let mut acc1b = _mm512_setzero_ps();
    let mut acc2a = _mm512_setzero_ps();
    let mut acc2b = _mm512_setzero_ps();
    let mut acc3a = _mm512_setzero_ps();
    let mut acc3b = _mm512_setzero_ps();
    let mut acc4a = _mm512_setzero_ps();
    let mut acc4b = _mm512_setzero_ps();
    let mut acc5a = _mm512_setzero_ps();
    let mut acc5b = _mm512_setzero_ps();

    let a_ptr = a.as_ptr();
    let bt_ptr = bt.as_ptr();

    // Row base offsets into A (row-major, stride = k).
    let a_base0 = i * k + k_start;
    let a_base1 = (i + 1) * k + k_start;
    let a_base2 = (i + 2) * k + k_start;
    let a_base3 = (i + 3) * k + k_start;
    let a_base4 = (i + 4) * k + k_start;
    let a_base5 = (i + 5) * k + k_start;

    for p in 0..kc {
        // Gather 32 Bᵀ values for K step p. bt is (N,K) row-major:
        // bt[col, k_idx] = bt[col*k + k_idx]. Adjacent columns are
        // K elements apart.
        let mut pack = [0.0f32; NR];
        for c_idx in 0..NR {
            *pack.get_unchecked_mut(c_idx) = *bt_ptr.add((j + c_idx) * k + k_start + p);
        }
        let b_lo = _mm512_loadu_ps(pack.as_ptr());
        let b_hi = _mm512_loadu_ps(pack.as_ptr().add(16));

        let a0 = _mm512_set1_ps(*a_ptr.add(a_base0 + p));
        acc0a = _mm512_fmadd_ps(a0, b_lo, acc0a);
        acc0b = _mm512_fmadd_ps(a0, b_hi, acc0b);

        let a1 = _mm512_set1_ps(*a_ptr.add(a_base1 + p));
        acc1a = _mm512_fmadd_ps(a1, b_lo, acc1a);
        acc1b = _mm512_fmadd_ps(a1, b_hi, acc1b);

        let a2 = _mm512_set1_ps(*a_ptr.add(a_base2 + p));
        acc2a = _mm512_fmadd_ps(a2, b_lo, acc2a);
        acc2b = _mm512_fmadd_ps(a2, b_hi, acc2b);

        let a3 = _mm512_set1_ps(*a_ptr.add(a_base3 + p));
        acc3a = _mm512_fmadd_ps(a3, b_lo, acc3a);
        acc3b = _mm512_fmadd_ps(a3, b_hi, acc3b);

        let a4 = _mm512_set1_ps(*a_ptr.add(a_base4 + p));
        acc4a = _mm512_fmadd_ps(a4, b_lo, acc4a);
        acc4b = _mm512_fmadd_ps(a4, b_hi, acc4b);

        let a5 = _mm512_set1_ps(*a_ptr.add(a_base5 + p));
        acc5a = _mm512_fmadd_ps(a5, b_lo, acc5a);
        acc5b = _mm512_fmadd_ps(a5, b_hi, acc5b);
    }

    // Store accumulators back to C (accumulate with existing partial sums).
    let c_ptr = c.as_mut_ptr();

    macro_rules! store_row {
        ($row:expr, $acc_a:expr, $acc_b:expr) => {{
            let c_off = $row * n + j;
            let existing_a = _mm512_loadu_ps(c_ptr.add(c_off));
            let existing_b = _mm512_loadu_ps(c_ptr.add(c_off + 16));
            _mm512_storeu_ps(c_ptr.add(c_off), _mm512_add_ps(existing_a, $acc_a));
            _mm512_storeu_ps(c_ptr.add(c_off + 16), _mm512_add_ps(existing_b, $acc_b));
        }};
    }

    store_row!(i, acc0a, acc0b);
    store_row!(i + 1, acc1a, acc1b);
    store_row!(i + 2, acc2a, acc2b);
    store_row!(i + 3, acc3a, acc3b);
    store_row!(i + 4, acc4a, acc4b);
    store_row!(i + 5, acc5a, acc5b);
}
