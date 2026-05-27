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
    __m128i, __m256, __m256i, _mm256_add_ps, _mm256_castps256_ps128, _mm256_cvtepi32_ps,
    _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_setzero_si256, _mm256_sign_epi8, _mm_add_ps, _mm_add_ss,
    _mm_cvtph_ps, _mm_cvtss_f32, _mm_loadl_epi64, _mm_movehdup_ps, _mm_movehl_ps, _mm_mul_ps,
    _mm_prefetch, _mm_set1_ps, _mm_shuffle_ps, _MM_HINT_T0,
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
        _mm256_set_m128i, _mm_and_si128, _mm_loadu_si128, _mm_set1_epi8, _mm_set_epi8,
        _mm_shuffle_epi8, _mm_srli_epi16,
    };

    let num_blocks = weight_scales.len();
    let iq = input_quants.as_ptr();
    let wp = weight_packed.as_ptr();
    // LUT: lut[i] = i - 8 maps unsigned Q4_0 nibbles (0..15) to signed (-8..7).
    // vpshufb within a 128-bit lane — no cross-lane permute needed.
    let lut = _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8);
    let mask_0f = _mm_set1_epi8(0x0F_u8 as i8);

    let mut total = _mm256_setzero_ps();

    for blk in 0..num_blocks {
        let combined_scale =
            _mm256_set1_ps(*input_scales.get_unchecked(blk) * *weight_scales.get_unchecked(blk));
        let inp_offset = blk * 32;
        let wp_offset = blk * 16;

        // Unpack 16 packed nibbles → 32 signed int8 using vpshufb LUT.
        // Eliminates the vperm2i128 cross-lane op present in the old set_m128i+permute approach.
        let packed = _mm_loadu_si128(wp.add(wp_offset).cast());
        let lo_128 = _mm_and_si128(packed, mask_0f);
        let hi_128 = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_0f);
        let weight_i8 =
            _mm256_set_m128i(_mm_shuffle_epi8(lut, hi_128), _mm_shuffle_epi8(lut, lo_128));

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
        _mm256_set_m128i, _mm_and_si128, _mm_loadu_si128, _mm_set1_epi8, _mm_set_epi8,
        _mm_shuffle_epi8, _mm_srli_epi16,
    };

    let num_blocks = input_scales.len();
    let iq = input_quants.as_ptr();
    let wp0 = weight_packed_0.as_ptr();
    let wp1 = weight_packed_1.as_ptr();
    // vpshufb LUT: maps nibble i (0..15) → i-8 (-8..7), within-lane (latency 1).
    let lut = _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8);
    let mask_0f = _mm_set1_epi8(0x0F_u8 as i8);

    let mut total0 = _mm256_setzero_ps();
    let mut total1 = _mm256_setzero_ps();

    for blk in 0..num_blocks {
        let inp_offset = blk * 32;
        let wp_offset = blk * 16;
        let inp_scale = *input_scales.get_unchecked(blk);

        // Load input once, reuse for both rows
        let input_i8 = _mm256_loadu_si256(iq.add(inp_offset).cast());

        // Unpack row 0: 16 packed nibbles → 32 signed int8 via vpshufb LUT.
        let packed0 = _mm_loadu_si128(wp0.add(wp_offset).cast());
        let lo0 = _mm_and_si128(packed0, mask_0f);
        let hi0 = _mm_and_si128(_mm_srli_epi16(packed0, 4), mask_0f);
        let weight0_i8 = _mm256_set_m128i(_mm_shuffle_epi8(lut, hi0), _mm_shuffle_epi8(lut, lo0));

        let w0_abs = _mm256_sign_epi8(weight0_i8, weight0_i8);
        let i0_signed = _mm256_sign_epi8(input_i8, weight0_i8);
        let prod0 = dpbusd_256(_mm256_setzero_si256(), w0_abs, i0_signed);
        let scale0 = _mm256_set1_ps(inp_scale * *weight_scales_0.get_unchecked(blk));
        total0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod0), scale0, total0);

        // Unpack row 1: 16 packed nibbles → 32 signed int8 via vpshufb LUT.
        let packed1 = _mm_loadu_si128(wp1.add(wp_offset).cast());
        let lo1 = _mm_and_si128(packed1, mask_0f);
        let hi1 = _mm_and_si128(_mm_srli_epi16(packed1, 4), mask_0f);
        let weight1_i8 = _mm256_set_m128i(_mm_shuffle_epi8(lut, hi1), _mm_shuffle_epi8(lut, lo1));

        let w1_abs = _mm256_sign_epi8(weight1_i8, weight1_i8);
        let i1_signed = _mm256_sign_epi8(input_i8, weight1_i8);
        let prod1 = dpbusd_256(_mm256_setzero_si256(), w1_abs, i1_signed);
        let scale1 = _mm256_set1_ps(inp_scale * *weight_scales_1.get_unchecked(blk));
        total1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod1), scale1, total1);
    }

    (hsum_256(total0), hsum_256(total1))
}

/// 4-row Q8×Q8 GEMV: computes dot products for four consecutive weight rows
/// against the same input vector. Returns `(dot0, dot1, dot2, dot3)`.
///
/// Weight data for all four rows is passed as a single contiguous slice
/// (4 × K bytes), reducing function-call argument count and allowing the
/// compiler to emit all four weight loads before any VNNI computation,
/// giving the CPU's OOO engine four concurrent DRAM requests per block.
pub fn dot_q8_q8_4row(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_quants_4rows: &[u8],
    weight_scales_4rows: &[f32],
) -> (f32, f32, f32, f32) {
    unsafe {
        dot_q8_q8_4row_inner(
            input_quants,
            input_scales,
            weight_quants_4rows,
            weight_scales_4rows,
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
unsafe fn dot_q8_q8_4row_inner(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_quants_4rows: &[u8],
    weight_scales_4rows: &[f32],
) -> (f32, f32, f32, f32) {
    let num_blocks = input_scales.len();
    let row_bytes = num_blocks * 32;

    let iq = input_quants.as_ptr();
    let wq0 = weight_quants_4rows.as_ptr();
    let wq1 = wq0.add(row_bytes);
    let wq2 = wq1.add(row_bytes);
    let wq3 = wq2.add(row_bytes);

    let ws0 = weight_scales_4rows.as_ptr();
    let ws1 = ws0.add(num_blocks);
    let ws2 = ws1.add(num_blocks);
    let ws3 = ws2.add(num_blocks);

    let mut total0 = _mm256_setzero_ps();
    let mut total1 = _mm256_setzero_ps();
    let mut total2 = _mm256_setzero_ps();
    let mut total3 = _mm256_setzero_ps();

    // Prefetch distance: DRAM latency (~150 ns) / block compute time (~17 ns) ≈ 9 blocks.
    // PF=12 is optimal: PF=8 leaves latency on the table; PF=16 over-saturates cache bandwidth.
    const PF: usize = 12;
    for blk in 0..num_blocks {
        let blk_offset = blk * 32;
        let inp_scale = *input_scales.get_unchecked(blk);

        // Prefetch next-block weight data for all 4 rows to hide DRAM latency.
        if blk + PF < num_blocks {
            let pf = (blk + PF) * 32;
            _mm_prefetch::<_MM_HINT_T0>(wq0.add(pf).cast());
            _mm_prefetch::<_MM_HINT_T0>(wq1.add(pf).cast());
            _mm_prefetch::<_MM_HINT_T0>(wq2.add(pf).cast());
            _mm_prefetch::<_MM_HINT_T0>(wq3.add(pf).cast());
        }

        let input_i8 = _mm256_loadu_si256(iq.add(blk_offset).cast());

        // Issue all 4 weight loads before any VNNI — four concurrent DRAM requests
        let w0 = _mm256_loadu_si256(wq0.add(blk_offset).cast());
        let w1 = _mm256_loadu_si256(wq1.add(blk_offset).cast());
        let w2 = _mm256_loadu_si256(wq2.add(blk_offset).cast());
        let w3 = _mm256_loadu_si256(wq3.add(blk_offset).cast());

        let w0_abs = _mm256_sign_epi8(w0, w0);
        let i0 = _mm256_sign_epi8(input_i8, w0);
        let prod0 = dpbusd_256(_mm256_setzero_si256(), w0_abs, i0);
        let scale0 = _mm256_set1_ps(inp_scale * *ws0.add(blk));
        total0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod0), scale0, total0);

        let w1_abs = _mm256_sign_epi8(w1, w1);
        let i1 = _mm256_sign_epi8(input_i8, w1);
        let prod1 = dpbusd_256(_mm256_setzero_si256(), w1_abs, i1);
        let scale1 = _mm256_set1_ps(inp_scale * *ws1.add(blk));
        total1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod1), scale1, total1);

        let w2_abs = _mm256_sign_epi8(w2, w2);
        let i2 = _mm256_sign_epi8(input_i8, w2);
        let prod2 = dpbusd_256(_mm256_setzero_si256(), w2_abs, i2);
        let scale2 = _mm256_set1_ps(inp_scale * *ws2.add(blk));
        total2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod2), scale2, total2);

        let w3_abs = _mm256_sign_epi8(w3, w3);
        let i3 = _mm256_sign_epi8(input_i8, w3);
        let prod3 = dpbusd_256(_mm256_setzero_si256(), w3_abs, i3);
        let scale3 = _mm256_set1_ps(inp_scale * *ws3.add(blk));
        total3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod3), scale3, total3);
    }

    (
        hsum_256(total0),
        hsum_256(total1),
        hsum_256(total2),
        hsum_256(total3),
    )
}

/// 4-row Q8×Q8 GEMV with 4-row-interleaved weight format.
///
/// Weight data layout (passed as `il_quants`, `il_scales_f16`):
/// - `il_quants[b * 136 + r * 32 .. +32]`       = quants for row r, block b (4 rows per block)
/// - `il_quants[b * 136 + 128 + r * 2 .. +2]`   = LE f16 scale for row r, block b
///
/// Unified layout: quants and scales are in one contiguous 136-byte block per 4-row group.
pub fn dot_q8_q8_4row_il(
    input_quants: &[u8],
    input_scales: &[f32],
    il_quants: &[u8],
) -> (f32, f32, f32, f32) {
    unsafe { dot_q8_q8_4row_il_inner(input_quants, input_scales, il_quants) }
}

#[allow(clippy::similar_names)]
#[target_feature(
    enable = "avx512f",
    enable = "avx512vnni",
    enable = "avx512vl",
    enable = "fma",
    enable = "f16c"
)]
unsafe fn dot_q8_q8_4row_il_inner(
    input_quants: &[u8],
    input_scales: &[f32],
    il_quants: &[u8],
) -> (f32, f32, f32, f32) {
    let nb = input_scales.len();

    let iq = input_quants.as_ptr();
    let wq = il_quants.as_ptr();

    let mut total0 = _mm256_setzero_ps();
    let mut total1 = _mm256_setzero_ps();
    let mut total2 = _mm256_setzero_ps();
    let mut total3 = _mm256_setzero_ps();

    // Prefetch 12 blocks ahead. Each block is 136 bytes = 3 partial cache lines;
    // Quant data: bytes 0..127 (CL0 = 0..63, CL1 = 64..127). Scale: bytes 128..135 (CL2).
    // All three cache lines must be prefetched; hardware prefetcher cannot bridge CL2.
    const PF: usize = 12;
    for blk in 0..nb {
        let inp_off = blk * 32;
        let wq_off = blk * 136;

        if blk + PF < nb {
            let pf = (blk + PF) * 136;
            _mm_prefetch::<_MM_HINT_T0>(wq.add(pf).cast());
            _mm_prefetch::<_MM_HINT_T0>(wq.add(pf + 64).cast());
            _mm_prefetch::<_MM_HINT_T0>(wq.add(pf + 128).cast()); // scale at CL2
        }

        let inp_scale = *input_scales.get_unchecked(blk);
        let input_i8 = _mm256_loadu_si256(iq.add(inp_off).cast());

        // All 4 weight loads are sequential within the unified block.
        let w0 = _mm256_loadu_si256(wq.add(wq_off).cast());
        let w1 = _mm256_loadu_si256(wq.add(wq_off + 32).cast());
        let w2 = _mm256_loadu_si256(wq.add(wq_off + 64).cast());
        let w3 = _mm256_loadu_si256(wq.add(wq_off + 96).cast());

        // Load 4 f16 scales (8 bytes at offset +128 within the unified block) → 4 f32.
        let scales_raw = _mm_loadl_epi64(wq.add(wq_off + 128).cast::<__m128i>());
        let scales_f32 = _mm_cvtph_ps(scales_raw);

        // Compute abs(input) once — reused for all 4 output rows (saves 3 vpsignb vs weight-side abs).
        let inp_abs = _mm256_sign_epi8(input_i8, input_i8);
        // Multiply all 4 weight scales by inp_scale in a single SIMD op.
        let combined = _mm_mul_ps(scales_f32, _mm_set1_ps(inp_scale));

        let i0 = _mm256_sign_epi8(w0, input_i8);
        let prod0 = dpbusd_256(_mm256_setzero_si256(), inp_abs, i0);
        let scale0 = _mm256_set1_ps(_mm_cvtss_f32(combined));
        total0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod0), scale0, total0);

        let i1 = _mm256_sign_epi8(w1, input_i8);
        let prod1 = dpbusd_256(_mm256_setzero_si256(), inp_abs, i1);
        let scale1 = _mm256_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(combined, combined, 0x55)));
        total1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod1), scale1, total1);

        let i2 = _mm256_sign_epi8(w2, input_i8);
        let prod2 = dpbusd_256(_mm256_setzero_si256(), inp_abs, i2);
        let scale2 = _mm256_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(combined, combined, 0xAA)));
        total2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod2), scale2, total2);

        let i3 = _mm256_sign_epi8(w3, input_i8);
        let prod3 = dpbusd_256(_mm256_setzero_si256(), inp_abs, i3);
        let scale3 = _mm256_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(combined, combined, 0xFF)));
        total3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod3), scale3, total3);
    }

    (
        hsum_256(total0),
        hsum_256(total1),
        hsum_256(total2),
        hsum_256(total3),
    )
}

/// 4-row Q4×Q8 GEMV: computes dot products for four consecutive Q4_0 weight rows
/// against the same Q8 input vector. Returns `(dot0, dot1, dot2, dot3)`.
pub fn dot_q4_q8_4row(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_packed_4rows: &[u8],
    weight_scales_4rows: &[f32],
) -> (f32, f32, f32, f32) {
    unsafe {
        dot_q4_q8_4row_inner(
            input_quants,
            input_scales,
            weight_packed_4rows,
            weight_scales_4rows,
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
unsafe fn dot_q4_q8_4row_inner(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_packed_4rows: &[u8],
    weight_scales_4rows: &[f32],
) -> (f32, f32, f32, f32) {
    use std::arch::x86_64::{
        _mm256_set_m128i, _mm_and_si128, _mm_loadu_si128, _mm_set1_epi8, _mm_set_epi8,
        _mm_shuffle_epi8, _mm_srli_epi16,
    };

    let num_blocks = input_scales.len();
    let packed_row_bytes = num_blocks * 16;

    let iq = input_quants.as_ptr();
    let wp0 = weight_packed_4rows.as_ptr();
    let wp1 = wp0.add(packed_row_bytes);
    let wp2 = wp1.add(packed_row_bytes);
    let wp3 = wp2.add(packed_row_bytes);

    let ws0 = weight_scales_4rows.as_ptr();
    let ws1 = ws0.add(num_blocks);
    let ws2 = ws1.add(num_blocks);
    let ws3 = ws2.add(num_blocks);

    // vpshufb LUT: maps nibble i (0..15) → i-8 (-8..7), within-lane (latency 1).
    let lut = _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8);
    let mask_0f = _mm_set1_epi8(0x0F_u8 as i8);

    let mut total0 = _mm256_setzero_ps();
    let mut total1 = _mm256_setzero_ps();
    let mut total2 = _mm256_setzero_ps();
    let mut total3 = _mm256_setzero_ps();

    const PF4: usize = 12; // prefetch distance in Q4 blocks (16 bytes each)
    for blk in 0..num_blocks {
        let inp_offset = blk * 32;
        let wp_offset = blk * 16;
        let inp_scale = *input_scales.get_unchecked(blk);

        // Prefetch next-block packed weight data for all 4 rows.
        if blk + PF4 < num_blocks {
            let pf = (blk + PF4) * 16;
            _mm_prefetch::<_MM_HINT_T0>(wp0.add(pf).cast());
            _mm_prefetch::<_MM_HINT_T0>(wp1.add(pf).cast());
            _mm_prefetch::<_MM_HINT_T0>(wp2.add(pf).cast());
            _mm_prefetch::<_MM_HINT_T0>(wp3.add(pf).cast());
        }

        let input_i8 = _mm256_loadu_si256(iq.add(inp_offset).cast());

        // Issue all 4 packed weight loads before unpacking
        let packed0 = _mm_loadu_si128(wp0.add(wp_offset).cast());
        let packed1 = _mm_loadu_si128(wp1.add(wp_offset).cast());
        let packed2 = _mm_loadu_si128(wp2.add(wp_offset).cast());
        let packed3 = _mm_loadu_si128(wp3.add(wp_offset).cast());

        // Unpack and process row 0: vpshufb LUT replaces set_m128i+permute2x128+sub.
        let lo0 = _mm_and_si128(packed0, mask_0f);
        let hi0 = _mm_and_si128(_mm_srli_epi16(packed0, 4), mask_0f);
        let w0_i8 = _mm256_set_m128i(_mm_shuffle_epi8(lut, hi0), _mm_shuffle_epi8(lut, lo0));
        let w0_abs = _mm256_sign_epi8(w0_i8, w0_i8);
        let i0 = _mm256_sign_epi8(input_i8, w0_i8);
        let prod0 = dpbusd_256(_mm256_setzero_si256(), w0_abs, i0);
        let scale0 = _mm256_set1_ps(inp_scale * *ws0.add(blk));
        total0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod0), scale0, total0);

        // Unpack and process row 1
        let lo1 = _mm_and_si128(packed1, mask_0f);
        let hi1 = _mm_and_si128(_mm_srli_epi16(packed1, 4), mask_0f);
        let w1_i8 = _mm256_set_m128i(_mm_shuffle_epi8(lut, hi1), _mm_shuffle_epi8(lut, lo1));
        let w1_abs = _mm256_sign_epi8(w1_i8, w1_i8);
        let i1 = _mm256_sign_epi8(input_i8, w1_i8);
        let prod1 = dpbusd_256(_mm256_setzero_si256(), w1_abs, i1);
        let scale1 = _mm256_set1_ps(inp_scale * *ws1.add(blk));
        total1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod1), scale1, total1);

        // Unpack and process row 2
        let lo2 = _mm_and_si128(packed2, mask_0f);
        let hi2 = _mm_and_si128(_mm_srli_epi16(packed2, 4), mask_0f);
        let w2_i8 = _mm256_set_m128i(_mm_shuffle_epi8(lut, hi2), _mm_shuffle_epi8(lut, lo2));
        let w2_abs = _mm256_sign_epi8(w2_i8, w2_i8);
        let i2 = _mm256_sign_epi8(input_i8, w2_i8);
        let prod2 = dpbusd_256(_mm256_setzero_si256(), w2_abs, i2);
        let scale2 = _mm256_set1_ps(inp_scale * *ws2.add(blk));
        total2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod2), scale2, total2);

        // Unpack and process row 3
        let lo3 = _mm_and_si128(packed3, mask_0f);
        let hi3 = _mm_and_si128(_mm_srli_epi16(packed3, 4), mask_0f);
        let w3_i8 = _mm256_set_m128i(_mm_shuffle_epi8(lut, hi3), _mm_shuffle_epi8(lut, lo3));
        let w3_abs = _mm256_sign_epi8(w3_i8, w3_i8);
        let i3 = _mm256_sign_epi8(input_i8, w3_i8);
        let prod3 = dpbusd_256(_mm256_setzero_si256(), w3_abs, i3);
        let scale3 = _mm256_set1_ps(inp_scale * *ws3.add(blk));
        total3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod3), scale3, total3);
    }

    (
        hsum_256(total0),
        hsum_256(total1),
        hsum_256(total2),
        hsum_256(total3),
    )
}
/// 4-row Q4_0 GEMV in unified 4-row-interleaved (IL) format.
///
/// IL layout: 72 bytes/block = `[64 bytes Q4_0 quants (4×16)][8 bytes f16 scales (4×u16 LE)]`.
/// - `il_quants_q4[blk * 72 + row * 16 .. +16]` = 16 packed Q4_0 bytes for row at block blk
/// - `il_quants_q4[blk * 72 + 64 .. +8]`        = 4 f16 scales (row 0..3) at block blk
///
/// Returns `(dot0, dot1, dot2, dot3)`.
pub fn dot_q4_q8_4row_il(
    input_quants: &[u8],
    input_scales: &[f32],
    il_quants_q4: &[u8],
) -> (f32, f32, f32, f32) {
    unsafe { dot_q4_q8_4row_il_inner(input_quants, input_scales, il_quants_q4) }
}

#[allow(clippy::similar_names)]
#[target_feature(
    enable = "avx512f",
    enable = "avx512vnni",
    enable = "avx512vl",
    enable = "avx512bw",
    enable = "fma",
    enable = "f16c"
)]
unsafe fn dot_q4_q8_4row_il_inner(
    input_quants: &[u8],
    input_scales: &[f32],
    il_quants_q4: &[u8],
) -> (f32, f32, f32, f32) {
    use std::arch::x86_64::{
        _mm256_set_m128i, _mm_and_si128, _mm_loadu_si128, _mm_set1_epi8, _mm_set_epi8,
        _mm_shuffle_epi8, _mm_srli_epi16,
    };

    let nb = input_scales.len();
    let iq = input_quants.as_ptr();
    let wq = il_quants_q4.as_ptr();

    // vpshufb LUT: nibble i → i-8, within-lane.
    let lut = _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8);
    let mask_0f = _mm_set1_epi8(0x0F_u8 as i8);

    let mut total0 = _mm256_setzero_ps();
    let mut total1 = _mm256_setzero_ps();
    let mut total2 = _mm256_setzero_ps();
    let mut total3 = _mm256_setzero_ps();

    // Each block occupies 72 bytes (64 bytes quants + 8 bytes f16 scales).
    // Bytes 0..63: quants (4 rows × 16 bytes, CL0); bytes 64..71: 4 f16 scales (CL1).
    // Two cache lines per block: prefetch both CL0 (quants) and CL1 (scales).
    const PF: usize = 12;
    for blk in 0..nb {
        let inp_off = blk * 32;
        let wq_off = blk * 72;

        if blk + PF < nb {
            let pf = (blk + PF) * 72;
            _mm_prefetch::<_MM_HINT_T0>(wq.add(pf).cast());
            _mm_prefetch::<_MM_HINT_T0>(wq.add(pf + 64).cast()); // scale CL1
        }

        let inp_scale = *input_scales.get_unchecked(blk);
        let input_i8 = _mm256_loadu_si256(iq.add(inp_off).cast());

        // Four sequential 16-byte reads — quants within the same 64-byte section.
        let p0 = _mm_loadu_si128(wq.add(wq_off).cast());
        let p1 = _mm_loadu_si128(wq.add(wq_off + 16).cast());
        let p2 = _mm_loadu_si128(wq.add(wq_off + 32).cast());
        let p3 = _mm_loadu_si128(wq.add(wq_off + 48).cast());

        // Load 4 f16 scales (8 bytes at offset +64 within the unified block) → 4 f32.
        let scales_raw = _mm_loadl_epi64(wq.add(wq_off + 64).cast::<__m128i>());
        let scales_f32 = _mm_cvtph_ps(scales_raw);
        let s0 = _mm_cvtss_f32(scales_f32);
        let s1 = _mm_cvtss_f32(_mm_shuffle_ps(scales_f32, scales_f32, 0x55));
        let s2 = _mm_cvtss_f32(_mm_shuffle_ps(scales_f32, scales_f32, 0xAA));
        let s3 = _mm_cvtss_f32(_mm_shuffle_ps(scales_f32, scales_f32, 0xFF));

        // Row 0: unpack nibbles → int8, sign-based VNNI dot.
        let lo0 = _mm_and_si128(p0, mask_0f);
        let hi0 = _mm_and_si128(_mm_srli_epi16(p0, 4), mask_0f);
        let w0_i8 = _mm256_set_m128i(_mm_shuffle_epi8(lut, hi0), _mm_shuffle_epi8(lut, lo0));
        let w0_abs = _mm256_sign_epi8(w0_i8, w0_i8);
        let i0 = _mm256_sign_epi8(input_i8, w0_i8);
        let prod0 = dpbusd_256(_mm256_setzero_si256(), w0_abs, i0);
        total0 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(prod0),
            _mm256_set1_ps(inp_scale * s0),
            total0,
        );

        // Row 1
        let lo1 = _mm_and_si128(p1, mask_0f);
        let hi1 = _mm_and_si128(_mm_srli_epi16(p1, 4), mask_0f);
        let w1_i8 = _mm256_set_m128i(_mm_shuffle_epi8(lut, hi1), _mm_shuffle_epi8(lut, lo1));
        let w1_abs = _mm256_sign_epi8(w1_i8, w1_i8);
        let i1 = _mm256_sign_epi8(input_i8, w1_i8);
        let prod1 = dpbusd_256(_mm256_setzero_si256(), w1_abs, i1);
        total1 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(prod1),
            _mm256_set1_ps(inp_scale * s1),
            total1,
        );

        // Row 2
        let lo2 = _mm_and_si128(p2, mask_0f);
        let hi2 = _mm_and_si128(_mm_srli_epi16(p2, 4), mask_0f);
        let w2_i8 = _mm256_set_m128i(_mm_shuffle_epi8(lut, hi2), _mm_shuffle_epi8(lut, lo2));
        let w2_abs = _mm256_sign_epi8(w2_i8, w2_i8);
        let i2 = _mm256_sign_epi8(input_i8, w2_i8);
        let prod2 = dpbusd_256(_mm256_setzero_si256(), w2_abs, i2);
        total2 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(prod2),
            _mm256_set1_ps(inp_scale * s2),
            total2,
        );

        // Row 3
        let lo3 = _mm_and_si128(p3, mask_0f);
        let hi3 = _mm_and_si128(_mm_srli_epi16(p3, 4), mask_0f);
        let w3_i8 = _mm256_set_m128i(_mm_shuffle_epi8(lut, hi3), _mm_shuffle_epi8(lut, lo3));
        let w3_abs = _mm256_sign_epi8(w3_i8, w3_i8);
        let i3 = _mm256_sign_epi8(input_i8, w3_i8);
        let prod3 = dpbusd_256(_mm256_setzero_si256(), w3_abs, i3);
        total3 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(prod3),
            _mm256_set1_ps(inp_scale * s3),
            total3,
        );
    }

    (
        hsum_256(total0),
        hsum_256(total1),
        hsum_256(total2),
        hsum_256(total3),
    )
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
const RN_Q8: usize = 4;

/// Tiled Q8×Q8 GEMM: `output[m, n_stride] = inp_quants[m, k] · wt_quants[n, k]`
/// with per-block scales.
///
/// Both input and weight data are in Q8_0 format: K dimension divided into
/// blocks of 32 elements, each with a `[u8; 32]` quant array and one `f32`
/// scale.
///
/// `n` is the number of weight columns to process (local chunk size).
/// `n_stride` is the row stride of `output` — use `n` for a contiguous
/// output buffer, or the global column count when writing directly into a
/// column-striped slice of the global output matrix.
///
/// Uses a 4×4 micro-kernel with AVX-512 VNNI for full tiles, falling back
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
    n_stride: usize,
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
            n_stride,
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
    n_stride: usize,
    num_blocks: usize,
    bytes_per_row: usize,
) {
    let m_full = m - m % RM_Q8;
    let n_full = n - n % RN_Q8;

    // Full RM_Q8 × RN_Q8 tiles.
    for i in (0..m_full).step_by(RM_Q8) {
        for j in (0..n_full).step_by(RN_Q8) {
            microkernel_q8_4x4(
                output,
                inp_quants,
                inp_scales,
                wt_quants,
                wt_scales,
                n_stride,
                num_blocks,
                bytes_per_row,
                i,
                j,
            );
        }
        // N remainder: scalar fallback.
        for jj in n_full..n {
            for ii in i..i + RM_Q8 {
                output[ii * n_stride + jj] = dot_q8_q8_row(
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
            output[ii * n_stride + jj] = dot_q8_q8_row(
                &inp_quants[ii * bytes_per_row..(ii + 1) * bytes_per_row],
                &inp_scales[ii * num_blocks..(ii + 1) * num_blocks],
                &wt_quants[jj * bytes_per_row..(jj + 1) * bytes_per_row],
                &wt_scales[jj * num_blocks..(jj + 1) * num_blocks],
            );
        }
    }
}

/// 4×6 Q8×Q8 micro-kernel using AVX-512 VNNI inline assembly.
///
/// Computes a 4-row × 6-column tile of the output matrix by accumulating
/// dot products over all K blocks. Uses 24 ymm accumulators (ymm8-ymm31)
/// that remain in registers across the entire block loop.
///
/// Direct inline scale approach: each cell broadcasts the input scale
/// directly from the register pointer, multiplies the int-dot result,
/// then loads the weight scale pointer from the `ptrs[]` array, broadcasts
/// the weight scale, and uses `vfmadd231ps` to accumulate. No combined-
/// scale buffer is used, avoiding store-forwarding penalties.
///
/// Register map:
/// - ymm0-ymm3: 4 input quant blocks (reloaded per K block)
/// - ymm4: weight quants (reloaded per column within block)
/// - ymm5: w_abs = vpsignb(w, w)
/// - ymm6, ymm7: scratch (dpbusd, cvt, scale broadcast)
/// - ymm8-ymm11:  accumulators for col 0, rows 0-3
/// - ymm12-ymm15: accumulators for col 1, rows 0-3
/// - ymm16-ymm19: accumulators for col 2, rows 0-3
/// - ymm20-ymm23: accumulators for col 3, rows 0-3
/// - ymm24-ymm27: accumulators for col 4, rows 0-3
/// - ymm28-ymm31: accumulators for col 5, rows 0-3
///
/// GPR map:
/// - iq0-iq3 (in): input quant row pointers
/// - is0-is3 (in): input scale row pointers (cast to byte ptrs)
/// - q_limit (in): loop bound = num_blocks × 32
/// - out_ptr (in): output array pointer
/// - arr_ptr (in): pointer to `ptrs[]` array (12 pointers: wq×6, ws×6)
/// - q_off (out): quant byte offset, incremented by 32 per block
/// - s_off (out): scale f32 offset = q_off >> 3, computed per block
/// - tmp (out): scratch GPR for pointer loads from `ptrs[]`
#[allow(
    dead_code,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names
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
    // Pointers to input quant rows (kept in registers for the hot path).
    let iq0 = inp_quants.as_ptr().add(i * bytes_per_row);
    let iq1 = inp_quants.as_ptr().add((i + 1) * bytes_per_row);
    let iq2 = inp_quants.as_ptr().add((i + 2) * bytes_per_row);
    let iq3 = inp_quants.as_ptr().add((i + 3) * bytes_per_row);

    // Input scale pointers — kept in dedicated registers (used 24× per block).
    let is0 = inp_scales.as_ptr().add(i * num_blocks).cast::<u8>();
    let is1 = inp_scales.as_ptr().add((i + 1) * num_blocks).cast::<u8>();
    let is2 = inp_scales.as_ptr().add((i + 2) * num_blocks).cast::<u8>();
    let is3 = inp_scales.as_ptr().add((i + 3) * num_blocks).cast::<u8>();

    // Stack-allocated pointer array accessed via two-level indirection
    // in the asm block. Layout (12 pointers, 96 bytes):
    //   [0..6)  = wq0..wq5  (weight quant row pointers)
    //   [6..12) = ws0..ws5  (weight scale col pointers)
    let ptrs: [*const u8; 12] = [
        wt_quants.as_ptr().add(j * bytes_per_row),
        wt_quants.as_ptr().add((j + 1) * bytes_per_row),
        wt_quants.as_ptr().add((j + 2) * bytes_per_row),
        wt_quants.as_ptr().add((j + 3) * bytes_per_row),
        wt_quants.as_ptr().add((j + 4) * bytes_per_row),
        wt_quants.as_ptr().add((j + 5) * bytes_per_row),
        wt_scales.as_ptr().add(j * num_blocks).cast(),
        wt_scales.as_ptr().add((j + 1) * num_blocks).cast(),
        wt_scales.as_ptr().add((j + 2) * num_blocks).cast(),
        wt_scales.as_ptr().add((j + 3) * num_blocks).cast(),
        wt_scales.as_ptr().add((j + 4) * num_blocks).cast(),
        wt_scales.as_ptr().add((j + 5) * num_blocks).cast(),
    ];

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

            // Loop counter.
            "xor {q_off:e}, {q_off:e}",

            "2:",
            // Compute scale offset: s_off = q_off >> 3
            // (quant blocks are 32 bytes, scales are 4 bytes → ratio 8:1).
            "mov {s_off}, {q_off}",
            "shr {s_off}, 3",

            // Load 4 input quant blocks (32 bytes each) into ymm0-3.
            "vmovdqu ymm0, [{iq0} + {q_off}]",
            "vmovdqu ymm1, [{iq1} + {q_off}]",
            "vmovdqu ymm2, [{iq2} + {q_off}]",
            "vmovdqu ymm3, [{iq3} + {q_off}]",

            // ---- Col 0 ----
            "mov {tmp}, [{arr_ptr} + 0*8]",
            "vmovdqu ymm4, [{tmp} + {q_off}]",
            "mov {tmp}, [{arr_ptr} + 6*8]",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is0} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm8, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is1} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm9, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is2} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm10, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is3} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm11, ymm6, ymm7",

            // ---- Col 1 ----
            "mov {tmp}, [{arr_ptr} + 1*8]",
            "vmovdqu ymm4, [{tmp} + {q_off}]",
            "mov {tmp}, [{arr_ptr} + 7*8]",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is0} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm12, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is1} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm13, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is2} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm14, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is3} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm15, ymm6, ymm7",

            // ---- Col 2 ----
            "mov {tmp}, [{arr_ptr} + 2*8]",
            "vmovdqu ymm4, [{tmp} + {q_off}]",
            "mov {tmp}, [{arr_ptr} + 8*8]",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is0} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm16, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is1} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm17, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is2} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm18, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is3} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm19, ymm6, ymm7",

            // ---- Col 3 ----
            "mov {tmp}, [{arr_ptr} + 3*8]",
            "vmovdqu ymm4, [{tmp} + {q_off}]",
            "mov {tmp}, [{arr_ptr} + 9*8]",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is0} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm20, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is1} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm21, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is2} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm22, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is3} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm23, ymm6, ymm7",

            // ---- Col 4 ----
            "mov {tmp}, [{arr_ptr} + 4*8]",
            "vmovdqu ymm4, [{tmp} + {q_off}]",
            "mov {tmp}, [{arr_ptr} + 10*8]",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is0} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm24, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is1} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm25, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is2} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm26, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is3} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm27, ymm6, ymm7",

            // ---- Col 5 ----
            "mov {tmp}, [{arr_ptr} + 5*8]",
            "vmovdqu ymm4, [{tmp} + {q_off}]",
            "mov {tmp}, [{arr_ptr} + 11*8]",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is0} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm28, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is1} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm29, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is2} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm30, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vbroadcastss ymm7, dword ptr [{is3} + {s_off}]",
            "vmulps ymm6, ymm6, ymm7",
            "vbroadcastss ymm7, dword ptr [{tmp} + {s_off}]",
            "vfmadd231ps ymm31, ymm6, ymm7",

            // Advance block.
            "add {q_off}, 32",
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
            is0 = in(reg) is0,
            is1 = in(reg) is1,
            is2 = in(reg) is2,
            is3 = in(reg) is3,
            q_limit = in(reg) q_limit,
            out_ptr = in(reg) out.as_mut_ptr(),
            arr_ptr = in(reg) ptrs.as_ptr(),
            q_off = out(reg) _,
            s_off = out(reg) _,
            tmp = out(reg) _,
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

/// 4×4 Q8×Q8 micro-kernel using AVX-512 VNNI inline assembly.
///
/// Computes a 4-row × 4-column tile of the output matrix by accumulating
/// dot products over all K blocks. Uses 16 ymm accumulators (ymm8-ymm23)
/// that remain in registers across the entire block loop.
///
/// Pre-broadcast scale approach: each block pre-loads all 4 input scales
/// into ymm24-27 and reuses them across all 4 weight columns. The weight
/// scale for the current column is broadcast into ymm28 once per column.
///
/// Register map:
/// - ymm0-ymm3: 4 input quant blocks (reloaded per K block)
/// - ymm4: weight quants (reloaded per column within block)
/// - ymm5: w_abs = vpsignb(w, w)
/// - ymm6, ymm7: scratch (dpbusd, cvt, mul)
/// - ymm8-ymm11:  accumulators for col 0, rows 0-3
/// - ymm12-ymm15: accumulators for col 1, rows 0-3
/// - ymm16-ymm19: accumulators for col 2, rows 0-3
/// - ymm20-ymm23: accumulators for col 3, rows 0-3
/// - ymm24-ymm27: pre-broadcast input scales (is0-is3, reloaded per block)
/// - ymm28: pre-broadcast weight scale (ws for current column)
/// - ymm29-ymm31: unused (clobbered for safety)
///
/// GPR map:
/// - iq0-iq3 (in): input quant row pointers
/// - is0-is3 (in): input scale row pointers (cast to byte ptrs)
/// - q_limit (in): loop bound = num_blocks × 32
/// - out_ptr (in): output array pointer
/// - arr_ptr (in): pointer to `ptrs[]` array (8 pointers: wq×4, ws×4)
/// - q_off (out): quant byte offset, incremented by 32 per block
/// - s_off (out): scale f32 offset = q_off >> 3, computed per block
/// - tmp (out): scratch GPR for pointer loads from `ptrs[]`
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names
)]
#[inline]
unsafe fn microkernel_q8_4x4(
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
    // Pointers to input quant rows (kept in registers for the hot path).
    let iq0 = inp_quants.as_ptr().add(i * bytes_per_row);
    let iq1 = inp_quants.as_ptr().add((i + 1) * bytes_per_row);
    let iq2 = inp_quants.as_ptr().add((i + 2) * bytes_per_row);
    let iq3 = inp_quants.as_ptr().add((i + 3) * bytes_per_row);

    // Input scale pointers — kept in dedicated registers (used 16× per block).
    let is0 = inp_scales.as_ptr().add(i * num_blocks).cast::<u8>();
    let is1 = inp_scales.as_ptr().add((i + 1) * num_blocks).cast::<u8>();
    let is2 = inp_scales.as_ptr().add((i + 2) * num_blocks).cast::<u8>();
    let is3 = inp_scales.as_ptr().add((i + 3) * num_blocks).cast::<u8>();

    // Stack-allocated pointer array accessed via two-level indirection
    // in the asm block. Layout (8 pointers, 64 bytes):
    //   [0..4)  = wq0..wq3  (weight quant row pointers)
    //   [4..8)  = ws0..ws3  (weight scale col pointers)
    let ptrs: [*const u8; 8] = [
        wt_quants.as_ptr().add(j * bytes_per_row),
        wt_quants.as_ptr().add((j + 1) * bytes_per_row),
        wt_quants.as_ptr().add((j + 2) * bytes_per_row),
        wt_quants.as_ptr().add((j + 3) * bytes_per_row),
        wt_scales.as_ptr().add(j * num_blocks).cast(),
        wt_scales.as_ptr().add((j + 1) * num_blocks).cast(),
        wt_scales.as_ptr().add((j + 2) * num_blocks).cast(),
        wt_scales.as_ptr().add((j + 3) * num_blocks).cast(),
    ];

    let q_limit = num_blocks * 32; // loop limit in bytes

    // 16 output scalars: out[col * RM_Q8 + row].
    let mut out = [0.0f32; 4 * 4];

    if num_blocks > 0 {
        std::arch::asm!(
            // Zero 16 accumulators (ymm8-ymm23).
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

            // Loop counter.
            "xor {q_off:e}, {q_off:e}",

            "2:",
            // Compute scale offset: s_off = q_off >> 3
            // (quant blocks are 32 bytes, scales are 4 bytes → ratio 8:1).
            "mov {s_off}, {q_off}",
            "shr {s_off}, 3",

            // Load 4 input quant blocks (32 bytes each) into ymm0-3.
            "vmovdqu ymm0, [{iq0} + {q_off}]",
            "vmovdqu ymm1, [{iq1} + {q_off}]",
            "vmovdqu ymm2, [{iq2} + {q_off}]",
            "vmovdqu ymm3, [{iq3} + {q_off}]",

            // Pre-broadcast 4 input scales for this block.
            "vbroadcastss ymm24, dword ptr [{is0} + {s_off}]",
            "vbroadcastss ymm25, dword ptr [{is1} + {s_off}]",
            "vbroadcastss ymm26, dword ptr [{is2} + {s_off}]",
            "vbroadcastss ymm27, dword ptr [{is3} + {s_off}]",

            // ---- Col 0 ----
            "mov {tmp}, [{arr_ptr} + 0*8]",
            "vmovdqu ymm4, [{tmp} + {q_off}]",
            "mov {tmp}, [{arr_ptr} + 4*8]",
            "vbroadcastss ymm28, dword ptr [{tmp} + {s_off}]",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm24, ymm28",
            "vfmadd231ps ymm8, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm25, ymm28",
            "vfmadd231ps ymm9, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm26, ymm28",
            "vfmadd231ps ymm10, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "mov {tmp}, [{arr_ptr} + 1*8]",   // early-load wq1 ptr during dpbusd latency
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm27, ymm28",
            "vfmadd231ps ymm11, ymm6, ymm7",

            // ---- Col 1 ---- ({tmp} already has wq1 pointer)
            "vmovdqu ymm4, [{tmp} + {q_off}]",
            "mov {tmp}, [{arr_ptr} + 5*8]",
            "vbroadcastss ymm28, dword ptr [{tmp} + {s_off}]",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm24, ymm28",
            "vfmadd231ps ymm12, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm25, ymm28",
            "vfmadd231ps ymm13, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm26, ymm28",
            "vfmadd231ps ymm14, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "mov {tmp}, [{arr_ptr} + 2*8]",   // early-load wq2 ptr during dpbusd latency
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm27, ymm28",
            "vfmadd231ps ymm15, ymm6, ymm7",

            // ---- Col 2 ---- ({tmp} already has wq2 pointer)
            "vmovdqu ymm4, [{tmp} + {q_off}]",
            "mov {tmp}, [{arr_ptr} + 6*8]",
            "vbroadcastss ymm28, dword ptr [{tmp} + {s_off}]",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm24, ymm28",
            "vfmadd231ps ymm16, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm25, ymm28",
            "vfmadd231ps ymm17, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm26, ymm28",
            "vfmadd231ps ymm18, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "mov {tmp}, [{arr_ptr} + 3*8]",   // early-load wq3 ptr during dpbusd latency
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm27, ymm28",
            "vfmadd231ps ymm19, ymm6, ymm7",

            // ---- Col 3 ---- ({tmp} already has wq3 pointer)
            "vmovdqu ymm4, [{tmp} + {q_off}]",
            "mov {tmp}, [{arr_ptr} + 7*8]",
            "vbroadcastss ymm28, dword ptr [{tmp} + {s_off}]",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm24, ymm28",
            "vfmadd231ps ymm20, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm25, ymm28",
            "vfmadd231ps ymm21, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm26, ymm28",
            "vfmadd231ps ymm22, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm27, ymm28",
            "vfmadd231ps ymm23, ymm6, ymm7",

            // Advance block.
            "add {q_off}, 32",
            "cmp {q_off}, {q_limit}",
            "jb 2b",

            // ---- Horizontal reduction ----
            // Reduce each ymm accumulator (8 f32) → scalar, store to out[].
            // For ymm16+, use EVEX vextractf32x4; for ymm8-15 use vextractf128.

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

            iq0 = in(reg) iq0,
            iq1 = in(reg) iq1,
            iq2 = in(reg) iq2,
            iq3 = in(reg) iq3,
            is0 = in(reg) is0,
            is1 = in(reg) is1,
            is2 = in(reg) is2,
            is3 = in(reg) is3,
            q_limit = in(reg) q_limit,
            out_ptr = in(reg) out.as_mut_ptr(),
            arr_ptr = in(reg) ptrs.as_ptr(),
            q_off = out(reg) _,
            s_off = out(reg) _,
            tmp = out(reg) _,
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
    // out layout: [col * 4 + row], output layout: row-major [row * n + col].
    let out_ptr = output.as_mut_ptr();
    for row in 0..4 {
        for col in 0..4 {
            *out_ptr.add((i + row) * n + (j + col)) = out[col * 4 + row];
        }
    }
}

// ---- Interleaved Q8×Q8 GEMM (4-row-interleaved weight format) ----
//
// Weight data is pre-packed into 4-row groups so each K-block's data for
// 4 consecutive weight rows is contiguous in memory (2 cache lines vs 4).
// This eliminates the ptrs[] indirection table and the `tmp` GPR, and
// replaces 8 pointer-load instructions per block with direct SIB addressing.
//
// Key addressing trick:
//   q_off advances by 32 per block (same as non-IL kernel, input unchanged).
//   Weight quant col k at block b: [{wt_base} + {q_off} * 4 + k*32]
//     (q_off*4 = b*128, k*32 = column offset within 128-byte group)
//   Weight scale col k at block b: [{ws_base} + {s_off} * 2 + k*2]
//     (s_off = q_off/8 = b*4; s_off*2 = b*8 bytes for 4 f16 IL scales)
//
// GPR count: 12 in (iq0-3, is0-3, wt_base, ws_base, q_limit, out_ptr)
//          +  2 out (q_off, s_off) = 14 total (fits x86-64 exactly).

/// Tiled Q8×Q8 GEMM using unified 4-row-interleaved weight format (136 bytes/block).
///
/// `wt_quants_il` contains quants and inline f16 scales. For group g = j/4, block b, row r:
///   `wt_quants_il[g * nb * 136 + b * 136 + r * 32 .. +32]` = quants
///   `wt_quants_il[g * nb * 136 + b * 136 + 128 + r * 2 .. +2]` = LE f16 scale
///
/// `col_start` must be a multiple of 4. `n` is the number of columns to process.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
pub fn gemm_q8_tiled_il(
    output: &mut [f32],
    inp_quants: &[u8],
    inp_scales: &[f32],
    wt_quants_il: &[u8],
    m: usize,
    n: usize,
    n_stride: usize,
    num_blocks: usize,
    bytes_per_row: usize,
    col_start: usize,
) {
    unsafe {
        gemm_q8_tiled_inner_il(
            output,
            inp_quants,
            inp_scales,
            wt_quants_il,
            m,
            n,
            n_stride,
            num_blocks,
            bytes_per_row,
            col_start,
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
    enable = "fma",
    enable = "f16c"
)]
unsafe fn gemm_q8_tiled_inner_il(
    output: &mut [f32],
    inp_quants: &[u8],
    inp_scales: &[f32],
    wt_quants_il: &[u8],
    m: usize,
    n: usize,
    n_stride: usize,
    num_blocks: usize,
    bytes_per_row: usize,
    col_start: usize,
) {
    let m_full = m - m % RM_Q8;
    let n_full = n - n % RN_Q8;

    for i in (0..m_full).step_by(RM_Q8) {
        for j in (0..n_full).step_by(RN_Q8) {
            let global_col = col_start + j;
            let group = global_col / 4;
            let wt_base = wt_quants_il.as_ptr().add(group * num_blocks * 136);
            microkernel_q8_4x4_il(
                output,
                inp_quants,
                inp_scales,
                wt_base,
                n_stride,
                num_blocks,
                bytes_per_row,
                i,
                j,
            );
        }
        // N remainder: scalar fallback.
        for jj in n_full..n {
            let global_col = col_start + jj;
            let group = global_col / 4;
            let row_in_group = global_col % 4;
            for ii in i..i + RM_Q8 {
                let mut acc = 0.0f32;
                for blk in 0..num_blocks {
                    let wq_off = group * num_blocks * 136 + blk * 136 + row_in_group * 32;
                    let scale_off = group * num_blocks * 136 + blk * 136 + 128 + row_in_group * 2;
                    let inp_off = ii * bytes_per_row + blk * 32;
                    let wq = &wt_quants_il[wq_off..wq_off + 32];
                    let iq = &inp_quants[inp_off..inp_off + 32];
                    let mut dot = 0i32;
                    for (w, a) in wq.iter().zip(iq.iter()) {
                        dot += (*w as i8 as i32) * (*a as i8 as i32);
                    }
                    let ws = half::f16::from_le_bytes([
                        wt_quants_il[scale_off],
                        wt_quants_il[scale_off + 1],
                    ])
                    .to_f32();
                    acc += (dot as f32) * inp_scales[ii * num_blocks + blk] * ws;
                }
                output[ii * n_stride + jj] = acc;
            }
        }
    }

    // M remainder: scalar fallback.
    for ii in m_full..m {
        for jj in 0..n {
            let global_col = col_start + jj;
            let group = global_col / 4;
            let row_in_group = global_col % 4;
            let mut acc = 0.0f32;
            for blk in 0..num_blocks {
                let wq_off = group * num_blocks * 136 + blk * 136 + row_in_group * 32;
                let scale_off = group * num_blocks * 136 + blk * 136 + 128 + row_in_group * 2;
                let inp_off = ii * bytes_per_row + blk * 32;
                let wq = &wt_quants_il[wq_off..wq_off + 32];
                let iq = &inp_quants[inp_off..inp_off + 32];
                let mut dot = 0i32;
                for (w, a) in wq.iter().zip(iq.iter()) {
                    dot += (*w as i8 as i32) * (*a as i8 as i32);
                }
                let ws = half::f16::from_le_bytes([
                    wt_quants_il[scale_off],
                    wt_quants_il[scale_off + 1],
                ])
                .to_f32();
                acc += (dot as f32) * inp_scales[ii * num_blocks + blk] * ws;
            }
            output[ii * n_stride + jj] = acc;
        }
    }
}

/// 4×4 Q8×Q8 microkernel using interleaved weight format (no ptrs[] table).
///
/// `wt_base`: pointer to the start of the 4-row group's interleaved quant data.
///   At block b: col k quants = `wt_base + b*128 + k*32`.
///   Addressed as: `[{wt_base} + {q_off} * 4 + k*32]` where q_off = b*32.
///
/// `ws_base`: byte pointer to the start of the 4-row group's interleaved f16 scales.
///   At block b: col k scale = `ws_base + b*8 + k*2` (f16, 2 bytes each).
///   Addressed as: `[{ws_base} + {s_off} * 2 + k*2]` where s_off = q_off/8.
///
/// GPR map (14 total):
///   iq0-3: input quant row ptrs (4)
///   is0-3: input scale row ptrs as bytes (4)
///   wt_base, ws_base: IL weight ptrs (2)
///   q_limit, out_ptr: loop bound + output (2)
///   q_off, s_off: loop counters (2, clobbered outputs)
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names
)]
#[inline]
unsafe fn microkernel_q8_4x4_il(
    output: &mut [f32],
    inp_quants: &[u8],
    inp_scales: &[f32],
    wt_base: *const u8,
    n: usize,
    num_blocks: usize,
    bytes_per_row: usize,
    i: usize,
    j: usize,
) {
    let iq0 = inp_quants.as_ptr().add(i * bytes_per_row);
    let iq1 = inp_quants.as_ptr().add((i + 1) * bytes_per_row);
    let iq2 = inp_quants.as_ptr().add((i + 2) * bytes_per_row);
    let iq3 = inp_quants.as_ptr().add((i + 3) * bytes_per_row);

    let is0 = inp_scales.as_ptr().add(i * num_blocks).cast::<u8>();
    let is1 = inp_scales.as_ptr().add((i + 1) * num_blocks).cast::<u8>();
    let is2 = inp_scales.as_ptr().add((i + 2) * num_blocks).cast::<u8>();
    let is3 = inp_scales.as_ptr().add((i + 3) * num_blocks).cast::<u8>();

    // iq_off steps by 32 (input byte stride); wt_off steps by 136 (unified block size).
    let q_limit = num_blocks * 32;

    let mut out = [0.0f32; 4 * 4];

    if num_blocks > 0 {
        std::arch::asm!(
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

            // wt_off: byte offset into weight data (steps by 136 per block).
            // iq_off: byte offset into input quant data (steps by 32 per block).
            "xor {wt_off:e}, {wt_off:e}",
            "xor {iq_off:e}, {iq_off:e}",

            "2:",
            // Prefetch weight data 12 IL-blocks ahead (12 * 136 = 1632 bytes).
            // Three cache lines per block: CL0=0..63, CL1=64..127, CL2=128..135 (scales).
            "prefetcht0 [{wt_base} + {wt_off} + 1632]",
            "prefetcht0 [{wt_base} + {wt_off} + 1696]",
            "prefetcht0 [{wt_base} + {wt_off} + 1760]",

            // Load 4 input quant blocks (32 bytes each, iq_off = blk * 32).
            "vmovdqu ymm0, [{iq0} + {iq_off}]",
            "vmovdqu ymm1, [{iq1} + {iq_off}]",
            "vmovdqu ymm2, [{iq2} + {iq_off}]",
            "vmovdqu ymm3, [{iq3} + {iq_off}]",

            // Input scale byte offset: blk * 4 bytes = iq_off / 8 (f32 is 4 bytes, blk = iq_off/32).
            "mov {s_off}, {iq_off}",
            "shr {s_off}, 3",

            // Pre-broadcast 4 input scales.
            "vbroadcastss ymm24, dword ptr [{is0} + {s_off}]",
            "vbroadcastss ymm25, dword ptr [{is1} + {s_off}]",
            "vbroadcastss ymm26, dword ptr [{is2} + {s_off}]",
            "vbroadcastss ymm27, dword ptr [{is3} + {s_off}]",

            // Load 4 f16 weight scales (8 bytes at offset +128 within the unified 136-byte block).
            "vcvtph2ps xmm29, qword ptr [{wt_base} + {wt_off} + 128]",

            // ---- Col 0 (row 0 of weight group) ----
            "vmovdqu ymm4, [{wt_base} + {wt_off}]",
            "vbroadcastss ymm28, xmm29",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm24, ymm28",
            "vfmadd231ps ymm8, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm25, ymm28",
            "vfmadd231ps ymm9, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm26, ymm28",
            "vfmadd231ps ymm10, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm27, ymm28",
            "vfmadd231ps ymm11, ymm6, ymm7",

            // ---- Col 1 ----
            "vmovdqu ymm4, [{wt_base} + {wt_off} + 32]",
            "vpermilps xmm30, xmm29, 0xe1",
            "vbroadcastss ymm28, xmm30",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm24, ymm28",
            "vfmadd231ps ymm12, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm25, ymm28",
            "vfmadd231ps ymm13, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm26, ymm28",
            "vfmadd231ps ymm14, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm27, ymm28",
            "vfmadd231ps ymm15, ymm6, ymm7",

            // ---- Col 2 ----
            "vmovdqu ymm4, [{wt_base} + {wt_off} + 64]",
            "vpermilps xmm30, xmm29, 0xe2",
            "vbroadcastss ymm28, xmm30",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm24, ymm28",
            "vfmadd231ps ymm16, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm25, ymm28",
            "vfmadd231ps ymm17, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm26, ymm28",
            "vfmadd231ps ymm18, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm27, ymm28",
            "vfmadd231ps ymm19, ymm6, ymm7",

            // ---- Col 3 ----
            "vmovdqu ymm4, [{wt_base} + {wt_off} + 96]",
            "vpermilps xmm30, xmm29, 0xe3",
            "vbroadcastss ymm28, xmm30",
            "vpsignb ymm5, ymm4, ymm4",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm24, ymm28",
            "vfmadd231ps ymm20, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm25, ymm28",
            "vfmadd231ps ymm21, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm26, ymm28",
            "vfmadd231ps ymm22, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm27, ymm28",
            "vfmadd231ps ymm23, ymm6, ymm7",

            "add {wt_off}, 136",
            "add {iq_off}, 32",
            "cmp {iq_off}, {q_limit}",
            "jb 2b",

            // ---- Horizontal reduction ----
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

            // Col 1: ymm12 → out[4..7]
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

            // Col 2 uses EVEX vextractf32x4 (ymm16+)
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

            // Col 3: ymm20 → out[12..15]
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

            iq0 = in(reg) iq0,
            iq1 = in(reg) iq1,
            iq2 = in(reg) iq2,
            iq3 = in(reg) iq3,
            is0 = in(reg) is0,
            is1 = in(reg) is1,
            is2 = in(reg) is2,
            is3 = in(reg) is3,
            wt_base = in(reg) wt_base,
            q_limit = in(reg) q_limit,
            out_ptr = in(reg) out.as_mut_ptr(),
            wt_off = out(reg) _,
            iq_off = out(reg) _,
            s_off = out(reg) _,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
            out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
            out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
            out("ymm16") _, out("ymm17") _, out("ymm18") _, out("ymm19") _,
            out("ymm20") _, out("ymm21") _, out("ymm22") _, out("ymm23") _,
            out("ymm24") _, out("ymm25") _, out("ymm26") _, out("ymm27") _,
            out("ymm28") _, out("ymm29") _, out("ymm30") _,
            options(nostack),
        );
    }

    let out_ptr = output.as_mut_ptr();
    for row in 0..4 {
        for col in 0..4 {
            *out_ptr.add((i + row) * n + (j + col)) = out[col * 4 + row];
        }
    }
}

// ---- Q4_0 IL expand (Q4_0 IL → Q8_0 IL) ----

/// Expand unified Q4_0 IL blocks → unified Q8_0 IL blocks using AVX-512 SIMD.
///
/// Each Q4_0 IL block (input): 72 bytes = 64 bytes Q4 quants + 8 bytes f16 scales.
/// Each Q8_0 IL block (output): 136 bytes = 128 bytes Q8 quants + 8 bytes f16 scales.
///
/// Input layout: `q4_il[g * nb * 72 + blk * 72 .. +72]`
/// Output layout: `q8[g * nb * 128 + blk * 128 .. + 128]`
#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub(super) unsafe fn expand_q4_il_to_q8_il_avx512(
    q4_il: &[u8],
    chunk_groups: usize,
    nb: usize,
    q8: &mut [u8],
) {
    use std::arch::x86_64::*;
    let mask0f = _mm512_set1_epi8(0x0F_u8 as i8);
    let minus8 = _mm512_set1_epi8(-8_i8);
    let total_blocks = chunk_groups * nb;
    // Prefetch distance: 20 blocks × 72 bytes = 1440 bytes ≈ 23 cache lines.
    // Hides ~100-cycle DRAM latency at ~5 cycles/block in the expand loop.
    const PF: usize = 20;
    for blk in 0..total_blocks {
        let src = q4_il.as_ptr().add(blk * 72) as *const __m512i;
        let dst = q8.as_mut_ptr().add(blk * 136);
        // Prefetch next Q4 blocks from DRAM/L3 into L1.
        _mm_prefetch(q4_il.as_ptr().add((blk + PF) * 72).cast(), _MM_HINT_T0);
        // Load 64 bytes of Q4 quants (first 64 bytes of the 72-byte unified Q4 block).
        let q4 = _mm512_loadu_si512(src);
        // Extract and center nibbles.
        let lo = _mm512_add_epi8(_mm512_and_si512(q4, mask0f), minus8);
        let hi = _mm512_add_epi8(_mm512_and_si512(_mm512_srli_epi16(q4, 4), mask0f), minus8);
        // Write 4 rows of Q8 quants into the unified Q8 block (128 bytes at offset 0).
        macro_rules! store_row {
            ($r:literal, $off:expr) => {
                let lo_r = _mm512_extracti32x4_epi32(lo, $r);
                let hi_r = _mm512_extracti32x4_epi32(hi, $r);
                let row = _mm256_inserti128_si256(_mm256_castsi128_si256(lo_r), hi_r, 1);
                _mm256_storeu_si256(dst.add($off) as *mut __m256i, row);
            };
        }
        store_row!(0, 0);
        store_row!(1, 32);
        store_row!(2, 64);
        store_row!(3, 96);
        // Copy 8 bytes of f16 scales from Q4 block offset +64 to Q8 block offset +128.
        let scales_src = q4_il.as_ptr().add(blk * 72 + 64) as *const u64;
        let scales_dst = dst.add(128) as *mut u64;
        scales_dst.write_unaligned(scales_src.read_unaligned());
    }
}

// ---- Q4_0 IL tiled GEMM ----

/// Low 128 bits = mask_0f (0x0F each byte), high 128 bits = minus8 (0xF8 = -8 each byte).
/// Loaded once into ymm31 at the top of `microkernel_q4_4x4_il`.
#[cfg(target_arch = "x86_64")]
static Q4_IL_UNPACK_CONSTS: [u8; 32] = [
    0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
    0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8,
];

/// Public entry point for Q4_0 IL tiled GEMM (compute output rows `col_start..col_start+n`).
/// Weight data is in compact Q4_0 IL format (`wt_quants_il`). No expand buffer needed.
#[allow(clippy::too_many_arguments)]
pub fn gemm_q4_tiled_il(
    output: &mut [f32],
    inp_quants: &[u8],
    inp_scales: &[f32],
    wt_quants_il: &[u8],
    m: usize,
    n: usize,
    n_stride: usize,
    num_blocks: usize,
    col_start: usize,
) {
    unsafe {
        gemm_q4_tiled_inner_il(
            output,
            inp_quants,
            inp_scales,
            wt_quants_il,
            m,
            n,
            n_stride,
            num_blocks,
            col_start,
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
    enable = "fma",
    enable = "f16c"
)]
unsafe fn gemm_q4_tiled_inner_il(
    output: &mut [f32],
    inp_quants: &[u8],
    inp_scales: &[f32],
    wt_quants_il: &[u8],
    m: usize,
    n: usize,
    n_stride: usize,
    num_blocks: usize,
    col_start: usize,
) {
    const RM: usize = 4;
    const RN: usize = 4;
    let bpr = num_blocks * 32;

    let m_full = m - m % RM;
    let n_full = n - n % RN;

    for i in (0..m_full).step_by(RM) {
        for j in (0..n_full).step_by(RN) {
            let global_col = col_start + j;
            let group = global_col / 4;
            let wt_base = wt_quants_il.as_ptr().add(group * num_blocks * 72);
            microkernel_q4_4x4_il(
                output, inp_quants, inp_scales, wt_base, n_stride, num_blocks, bpr, i, j,
            );
        }
        // N remainder: scalar fallback.
        for jj in n_full..n {
            let global_col = col_start + jj;
            let group = global_col / 4;
            let row_in_group = global_col % 4;
            for ii in i..i + RM {
                let mut acc = 0.0f32;
                for blk in 0..num_blocks {
                    let wq_off = group * num_blocks * 72 + blk * 72 + row_in_group * 16;
                    let scale_off = group * num_blocks * 72 + blk * 72 + 64 + row_in_group * 2;
                    let inp_off = ii * bpr + blk * 32;
                    let packed = &wt_quants_il[wq_off..wq_off + 16];
                    let iq = &inp_quants[inp_off..inp_off + 32];
                    let mut dot = 0i32;
                    for k in 0..16 {
                        dot += ((packed[k] & 0x0F).wrapping_sub(8) as i8 as i32)
                            * (iq[k] as i8 as i32);
                        dot += ((packed[k] >> 4).wrapping_sub(8) as i8 as i32)
                            * (iq[k + 16] as i8 as i32);
                    }
                    let ws = half::f16::from_le_bytes([
                        wt_quants_il[scale_off],
                        wt_quants_il[scale_off + 1],
                    ])
                    .to_f32();
                    acc += (dot as f32) * inp_scales[ii * num_blocks + blk] * ws;
                }
                output[ii * n_stride + jj] = acc;
            }
        }
    }

    // M remainder: scalar fallback.
    for ii in m_full..m {
        for jj in 0..n {
            let global_col = col_start + jj;
            let group = global_col / 4;
            let row_in_group = global_col % 4;
            let mut acc = 0.0f32;
            for blk in 0..num_blocks {
                let wq_off = group * num_blocks * 72 + blk * 72 + row_in_group * 16;
                let scale_off = group * num_blocks * 72 + blk * 72 + 64 + row_in_group * 2;
                let inp_off = ii * bpr + blk * 32;
                let packed = &wt_quants_il[wq_off..wq_off + 16];
                let iq = &inp_quants[inp_off..inp_off + 32];
                let mut dot = 0i32;
                for k in 0..16 {
                    dot += ((packed[k] & 0x0F).wrapping_sub(8) as i8 as i32) * (iq[k] as i8 as i32);
                    dot +=
                        ((packed[k] >> 4).wrapping_sub(8) as i8 as i32) * (iq[k + 16] as i8 as i32);
                }
                let ws = half::f16::from_le_bytes([
                    wt_quants_il[scale_off],
                    wt_quants_il[scale_off + 1],
                ])
                .to_f32();
                acc += (dot as f32) * inp_scales[ii * num_blocks + blk] * ws;
            }
            output[ii * n_stride + jj] = acc;
        }
    }
}

/// 4×4 Q4_0×Q8_0 microkernel using unified Q4_0 IL format (72 bytes/block: 64 quant + 8 f16).
///
/// `wt_base`: pointer to `il[group * nb * 72]`. At block b, col k (row k within group):
///   16 Q4_0 bytes at `wt_base + b*72 + k*16`; 4 f16 scales at `wt_base + b*72 + 64`.
///   Uses two counters: `wt_off` (steps by 72) and `iq_off` (steps by 32).
///
/// `ymm31`: packed constant (loaded from `Q4_IL_UNPACK_CONSTS`):
///   xmm31 (low 128 bits) = 0x0F mask (for nibble extraction).
///   high 128 bits = 0xF8 = -8 (for nibble centering, extracted via vextracti128).
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names
)]
#[inline]
unsafe fn microkernel_q4_4x4_il(
    output: &mut [f32],
    inp_quants: &[u8],
    inp_scales: &[f32],
    wt_base: *const u8,
    n: usize,
    num_blocks: usize,
    bytes_per_row: usize,
    i: usize,
    j: usize,
) {
    let iq0 = inp_quants.as_ptr().add(i * bytes_per_row);
    let iq1 = inp_quants.as_ptr().add((i + 1) * bytes_per_row);
    let iq2 = inp_quants.as_ptr().add((i + 2) * bytes_per_row);
    let iq3 = inp_quants.as_ptr().add((i + 3) * bytes_per_row);

    let is0 = inp_scales.as_ptr().add(i * num_blocks).cast::<u8>();
    let is1 = inp_scales.as_ptr().add((i + 1) * num_blocks).cast::<u8>();
    let is2 = inp_scales.as_ptr().add((i + 2) * num_blocks).cast::<u8>();
    let is3 = inp_scales.as_ptr().add((i + 3) * num_blocks).cast::<u8>();

    let q_limit = num_blocks * 32;

    let mut out = [0.0f32; 4 * 4];

    if num_blocks > 0 {
        std::arch::asm!(
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

            "xor {wt_off:e}, {wt_off:e}",
            "xor {iq_off:e}, {iq_off:e}",

            "2:",
            "mov {s_off}, {iq_off}",
            "shr {s_off}, 3",           // s_off = blk*4 (iq_off/8 = byte offset into f32 array)

            // Load 4 input quant blocks (32 bytes each = one Q8_0 block per row).
            "vmovdqu ymm0, [{iq0} + {iq_off}]",
            "vmovdqu ymm1, [{iq1} + {iq_off}]",
            "vmovdqu ymm2, [{iq2} + {iq_off}]",
            "vmovdqu ymm3, [{iq3} + {iq_off}]",

            // Input scales: s_off = blk*4 = byte offset into f32 array (4 bytes per f32).
            "vbroadcastss ymm24, dword ptr [{is0} + {s_off}]",
            "vbroadcastss ymm25, dword ptr [{is1} + {s_off}]",
            "vbroadcastss ymm26, dword ptr [{is2} + {s_off}]",
            "vbroadcastss ymm27, dword ptr [{is3} + {s_off}]",

            // Weight scales: 4 f16 inline at wt_off+64 in the unified 72-byte block.
            "vcvtph2ps xmm29, qword ptr [{wt_base} + {wt_off} + 64]",

            // ---- Col 0 (row 0 of weight group) ----
            // 16 Q4_0 bytes at wt_base + wt_off + 0*16.
            "vmovdqu xmm4, [{wt_base} + {wt_off}]",
            // lo nibbles: byte & mask_0f.
            "vpand xmm6, xmm4, xmmword ptr [{q4_consts}]",
            // Center lo nibbles: lo + (-8).
            "vpaddb xmm6, xmm6, xmmword ptr [{q4_consts} + 16]",
            // hi nibbles: (byte >> 4) & mask_0f (vpsrlw shifts 16-bit words, mask cleans overflow).
            "vpsrlw xmm7, xmm4, 4",
            "vpand xmm7, xmm7, xmmword ptr [{q4_consts}]",
            // Center hi nibbles.
            "vpaddb xmm7, xmm7, xmmword ptr [{q4_consts} + 16]",
            // Combine: ymm4 = [lo_centered(0..15) | hi_centered(0..15)] = Q8_0 expanded.
            "vinserti128 ymm4, ymm6, xmm7, 1",
            "vpsignb ymm5, ymm4, ymm4",            // ymm5 = abs(weight)
            // Broadcast scale[0].
            "vbroadcastss ymm28, xmm29",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm24, ymm28",
            "vfmadd231ps ymm8, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm25, ymm28",
            "vfmadd231ps ymm9, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm26, ymm28",
            "vfmadd231ps ymm10, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm27, ymm28",
            "vfmadd231ps ymm11, ymm6, ymm7",

            // ---- Col 1 (row 1 of weight group) ----
            "vmovdqu xmm4, [{wt_base} + {wt_off} + 16]",
            "vpermilps xmm30, xmm29, 0xe1",        // scale[1] → lane 0
            "vpand xmm6, xmm4, xmmword ptr [{q4_consts}]",
            "vpaddb xmm6, xmm6, xmmword ptr [{q4_consts} + 16]",
            "vpsrlw xmm7, xmm4, 4",
            "vpand xmm7, xmm7, xmmword ptr [{q4_consts}]",
            "vpaddb xmm7, xmm7, xmmword ptr [{q4_consts} + 16]",
            "vinserti128 ymm4, ymm6, xmm7, 1",
            "vpsignb ymm5, ymm4, ymm4",
            "vbroadcastss ymm28, xmm30",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm24, ymm28",
            "vfmadd231ps ymm12, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm25, ymm28",
            "vfmadd231ps ymm13, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm26, ymm28",
            "vfmadd231ps ymm14, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm27, ymm28",
            "vfmadd231ps ymm15, ymm6, ymm7",

            // ---- Col 2 (row 2 of weight group) ----
            "vmovdqu xmm4, [{wt_base} + {wt_off} + 32]",
            "vpermilps xmm30, xmm29, 0xe2",        // scale[2] → lane 0
            "vpand xmm6, xmm4, xmmword ptr [{q4_consts}]",
            "vpaddb xmm6, xmm6, xmmword ptr [{q4_consts} + 16]",
            "vpsrlw xmm7, xmm4, 4",
            "vpand xmm7, xmm7, xmmword ptr [{q4_consts}]",
            "vpaddb xmm7, xmm7, xmmword ptr [{q4_consts} + 16]",
            "vinserti128 ymm4, ymm6, xmm7, 1",
            "vpsignb ymm5, ymm4, ymm4",
            "vbroadcastss ymm28, xmm30",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm24, ymm28",
            "vfmadd231ps ymm16, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm25, ymm28",
            "vfmadd231ps ymm17, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm26, ymm28",
            "vfmadd231ps ymm18, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm27, ymm28",
            "vfmadd231ps ymm19, ymm6, ymm7",

            // ---- Col 3 (row 3 of weight group) ----
            "vmovdqu xmm4, [{wt_base} + {wt_off} + 48]",
            "vpermilps xmm30, xmm29, 0xe3",        // scale[3] → lane 0
            "vpand xmm6, xmm4, xmmword ptr [{q4_consts}]",
            "vpaddb xmm6, xmm6, xmmword ptr [{q4_consts} + 16]",
            "vpsrlw xmm7, xmm4, 4",
            "vpand xmm7, xmm7, xmmword ptr [{q4_consts}]",
            "vpaddb xmm7, xmm7, xmmword ptr [{q4_consts} + 16]",
            "vinserti128 ymm4, ymm6, xmm7, 1",
            "vpsignb ymm5, ymm4, ymm4",
            "vbroadcastss ymm28, xmm30",
            // row 0
            "vpsignb ymm7, ymm0, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm24, ymm28",
            "vfmadd231ps ymm20, ymm6, ymm7",
            // row 1
            "vpsignb ymm7, ymm1, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm25, ymm28",
            "vfmadd231ps ymm21, ymm6, ymm7",
            // row 2
            "vpsignb ymm7, ymm2, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm26, ymm28",
            "vfmadd231ps ymm22, ymm6, ymm7",
            // row 3
            "vpsignb ymm7, ymm3, ymm4",
            "vpxord  ymm6, ymm6, ymm6",
            "vpdpbusd ymm6, ymm5, ymm7",
            "vcvtdq2ps ymm6, ymm6",
            "vmulps ymm7, ymm27, ymm28",
            "vfmadd231ps ymm23, ymm6, ymm7",

            "add {wt_off}, 72",
            "add {iq_off}, 32",
            "cmp {iq_off}, {q_limit}",
            "jb 2b",

            // ---- Horizontal reduction (identical to microkernel_q8_4x4_il) ----
            // Col 0: ymm8→out[0], ymm9→out[1], ymm10→out[2], ymm11→out[3]
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

            // Col 1: ymm12→out[4..7]
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

            // Col 2: ymm16→out[8..11]  (vextractf32x4 = EVEX, supports ymm16+)
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

            // Col 3: ymm20→out[12..15]
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

            q4_consts = in(reg) Q4_IL_UNPACK_CONSTS.as_ptr(),
            iq0 = in(reg) iq0,
            iq1 = in(reg) iq1,
            iq2 = in(reg) iq2,
            iq3 = in(reg) iq3,
            is0 = in(reg) is0,
            is1 = in(reg) is1,
            is2 = in(reg) is2,
            is3 = in(reg) is3,
            wt_base = in(reg) wt_base,
            q_limit = in(reg) q_limit,
            out_ptr = in(reg) out.as_mut_ptr(),
            iq_off = out(reg) _,
            wt_off = out(reg) _,
            s_off = out(reg) _,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
            out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
            out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
            out("ymm16") _, out("ymm17") _, out("ymm18") _, out("ymm19") _,
            out("ymm20") _, out("ymm21") _, out("ymm22") _, out("ymm23") _,
            out("ymm24") _, out("ymm25") _, out("ymm26") _, out("ymm27") _,
            out("ymm28") _, out("ymm29") _, out("ymm30") _,
            options(nostack),
        );
    }

    let out_ptr = output.as_mut_ptr();
    for row in 0..4 {
        for col in 0..4 {
            *out_ptr.add((i + row) * n + (j + col)) = out[col * 4 + row];
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

// ---- Vectorized GeGLU ----

/// Fused GeGLU: `out[i] = gelu_approx(gate[i]) * up[i]`.
///
/// Uses the identity `gelu(x) = x * sigmoid(2·sqrt(2/π)·(x + 0.044715·x³))`,
/// computed with the same fast polynomial exp as `vec_silu_mul`.
pub fn vec_gelu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    unsafe { vec_gelu_mul_inner(gate, up, out) }
}

#[target_feature(enable = "avx512f")]
#[allow(clippy::many_single_char_names)]
unsafe fn vec_gelu_mul_inner(gate: &[f32], up: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::{
        __m512, __m512i, _mm512_add_epi32, _mm512_add_ps, _mm512_castps_si512, _mm512_castsi512_ps,
        _mm512_cvtepi32_ps, _mm512_cvtps_epi32, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_max_ps,
        _mm512_min_ps, _mm512_mul_ps, _mm512_set1_ps, _mm512_slli_epi32, _mm512_storeu_ps,
        _mm512_sub_ps,
    };

    let n = gate.len();
    let chunks = n / 16;
    let remainder = n % 16;

    // 2 * sqrt(2/π) — used to convert tanh form to sigmoid form.
    let two_c: __m512 = _mm512_set1_ps(1.595_769_1_f32);
    let cubic: __m512 = _mm512_set1_ps(0.044_715_f32);

    let log2e: __m512 = _mm512_set1_ps(std::f32::consts::LOG2_E);
    let one: __m512 = _mm512_set1_ps(1.0);
    let c0: __m512 = _mm512_set1_ps(1.0);
    #[allow(clippy::approx_constant)]
    let c1: __m512 = _mm512_set1_ps(0.693_147_2_f32);
    let c2: __m512 = _mm512_set1_ps(0.240_226_5_f32);
    let c3: __m512 = _mm512_set1_ps(5.550_357e-2_f32);
    let c4: __m512 = _mm512_set1_ps(9.675_54e-3_f32);
    let exp_lo: __m512 = _mm512_set1_ps(-87.332_54_f32);
    let exp_hi: __m512 = _mm512_set1_ps(88.722_84_f32);
    let zero: __m512 = _mm512_set1_ps(0.0);

    let gate_ptr = gate.as_ptr();
    let up_ptr = up.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 16;
        let g: __m512 = _mm512_loadu_ps(gate_ptr.add(offset));
        let u: __m512 = _mm512_loadu_ps(up_ptr.add(offset));

        // inner = 2·sqrt(2/π)·(g + 0.044715·g³)
        let g3: __m512 = _mm512_mul_ps(_mm512_mul_ps(g, g), g);
        let inner: __m512 = _mm512_mul_ps(two_c, _mm512_fmadd_ps(cubic, g3, g));

        // Fast exp(-inner) for sigmoid(inner) = 1/(1+exp(-inner)).
        let neg_inner: __m512 = _mm512_sub_ps(zero, inner);
        let x: __m512 = _mm512_max_ps(_mm512_min_ps(neg_inner, exp_hi), exp_lo);
        let t: __m512 = _mm512_mul_ps(x, log2e);
        let n_f: __m512 = _mm512_cvtepi32_ps(_mm512_cvtps_epi32(t));
        let f: __m512 = _mm512_sub_ps(t, n_f);
        let poly: __m512 = _mm512_fmadd_ps(c4, f, c3);
        let poly: __m512 = _mm512_fmadd_ps(poly, f, c2);
        let poly: __m512 = _mm512_fmadd_ps(poly, f, c1);
        let poly: __m512 = _mm512_fmadd_ps(poly, f, c0);
        let n_i: __m512i = _mm512_cvtps_epi32(n_f);
        let shift: __m512i = _mm512_slli_epi32(n_i, 23);
        let exp_val: __m512 =
            _mm512_castsi512_ps(_mm512_add_epi32(_mm512_castps_si512(poly), shift));

        // gelu(g) * u = g * u / (1 + exp(-inner))
        let denom: __m512 = _mm512_add_ps(one, exp_val);
        let result: __m512 = {
            use std::arch::x86_64::_mm512_div_ps;
            _mm512_div_ps(_mm512_mul_ps(g, u), denom)
        };

        _mm512_storeu_ps(out_ptr.add(offset), result);
    }

    // Scalar tail.
    let tail = chunks * 16;
    for i in 0..remainder {
        let g = *gate.get_unchecked(tail + i);
        let u = *up.get_unchecked(tail + i);
        let inner = 1.595_769_1_f32 * (g + 0.044_715 * g * g * g);
        let sigmoid = 1.0 / (1.0 + (-inner).exp());
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

// ---- Vectorized softcap ----

/// Applies `x → cap * tanh(x / cap)` in-place using AVX-512 polynomial exp.
///
/// Used by Gemma attention logit softcapping. Applied to all scores in a slice
/// at once, turning O(n) scalar tanh calls into O(n/16) vectorized iterations.
pub fn vec_softcap_inplace(data: &mut [f32], cap: f32) {
    unsafe { vec_softcap_inplace_inner(data, cap) }
}

#[target_feature(enable = "avx512f")]
#[allow(clippy::many_single_char_names)]
unsafe fn vec_softcap_inplace_inner(data: &mut [f32], cap: f32) {
    use std::arch::x86_64::{
        __m512, __m512i, _mm512_add_epi32, _mm512_add_ps, _mm512_castps_si512, _mm512_castsi512_ps,
        _mm512_cvtepi32_ps, _mm512_cvtps_epi32, _mm512_div_ps, _mm512_fmadd_ps, _mm512_loadu_ps,
        _mm512_max_ps, _mm512_min_ps, _mm512_mul_ps, _mm512_set1_ps, _mm512_slli_epi32,
        _mm512_storeu_ps, _mm512_sub_ps,
    };

    let n = data.len();
    let chunks = n / 16;
    let remainder = n % 16;

    // tanh(x / cap) = (exp(2x/cap) - 1) / (exp(2x/cap) + 1)
    let two_inv_cap: __m512 = _mm512_set1_ps(2.0 / cap);
    let cap_vec: __m512 = _mm512_set1_ps(cap);
    let one: __m512 = _mm512_set1_ps(1.0);

    // exp polynomial constants (same as vec_silu_mul_inner).
    let log2e: __m512 = _mm512_set1_ps(std::f32::consts::LOG2_E);
    let c0: __m512 = _mm512_set1_ps(1.0);
    #[allow(clippy::approx_constant)]
    let c1: __m512 = _mm512_set1_ps(0.693_147_2_f32);
    let c2: __m512 = _mm512_set1_ps(0.240_226_5_f32);
    let c3: __m512 = _mm512_set1_ps(5.550_357e-2_f32);
    let c4: __m512 = _mm512_set1_ps(9.675_54e-3_f32);
    let exp_lo: __m512 = _mm512_set1_ps(-87.332_54_f32);
    let exp_hi: __m512 = _mm512_set1_ps(88.722_84_f32);

    let ptr = data.as_mut_ptr();
    for i in 0..chunks {
        let offset = i * 16;
        let x: __m512 = _mm512_loadu_ps(ptr.add(offset));

        // Compute exp(2x/cap) via degree-4 polynomial: exp(v) = 2^(v * log2e).
        let v: __m512 = _mm512_mul_ps(x, two_inv_cap);
        let v_c: __m512 = _mm512_max_ps(_mm512_min_ps(v, exp_hi), exp_lo);
        let t: __m512 = _mm512_mul_ps(v_c, log2e);
        let n_f: __m512 = _mm512_cvtepi32_ps(_mm512_cvtps_epi32(t));
        let f: __m512 = _mm512_sub_ps(t, n_f);
        let poly: __m512 = _mm512_fmadd_ps(c4, f, c3);
        let poly: __m512 = _mm512_fmadd_ps(poly, f, c2);
        let poly: __m512 = _mm512_fmadd_ps(poly, f, c1);
        let poly: __m512 = _mm512_fmadd_ps(poly, f, c0);
        let n_i: __m512i = _mm512_cvtps_epi32(n_f);
        let shift: __m512i = _mm512_slli_epi32(n_i, 23);
        let exp_v: __m512 = _mm512_castsi512_ps(_mm512_add_epi32(_mm512_castps_si512(poly), shift));

        // tanh(x/cap) = (exp_v - 1) / (exp_v + 1); result = cap * tanh(x/cap).
        let numer: __m512 = _mm512_sub_ps(exp_v, one);
        let denom: __m512 = _mm512_add_ps(exp_v, one);
        let tanh_val: __m512 = _mm512_div_ps(numer, denom);
        let result: __m512 = _mm512_mul_ps(cap_vec, tanh_val);
        _mm512_storeu_ps(ptr.add(offset), result);
    }

    // Scalar tail.
    let tail = chunks * 16;
    let inv_cap = 1.0 / cap;
    for i in 0..remainder {
        let x = *data.get_unchecked(tail + i);
        *data.get_unchecked_mut(tail + i) = cap * (x * inv_cap).tanh();
    }
}
