//! NEON SIMD kernels for AArch64.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32, vst1q_f32,
};

pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: NEON is baseline on AArch64, always available.
    unsafe { dot_f32_inner(a, b) }
}

#[inline]
unsafe fn dot_f32_inner(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc = vdupq_n_f32(0.0);
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let va = vld1q_f32(a_ptr.add(i * 4));
        let vb = vld1q_f32(b_ptr.add(i * 4));
        acc = vfmaq_f32(acc, va, vb);
    }

    let mut sum = vaddvq_f32(acc);
    let tail = chunks * 4;
    for i in 0..remainder {
        sum = a[tail + i].mul_add(b[tail + i], sum);
    }
    sum
}

pub fn vec_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    unsafe {
        vec_add_inner(a, b, out);
    }
}

unsafe fn vec_add_inner(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        let vb = vld1q_f32(b.as_ptr().add(i * 4));
        vst1q_f32(out.as_mut_ptr().add(i * 4), vaddq_f32(va, vb));
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        out[tail + i] = a[tail + i] + b[tail + i];
    }
}

pub fn vec_add_inplace(a: &mut [f32], b: &[f32]) {
    unsafe {
        vec_add_inplace_inner(a, b);
    }
}

unsafe fn vec_add_inplace_inner(a: &mut [f32], b: &[f32]) {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        let vb = vld1q_f32(b.as_ptr().add(i * 4));
        vst1q_f32(a.as_mut_ptr().add(i * 4), vaddq_f32(va, vb));
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        a[tail + i] += b[tail + i];
    }
}

pub fn vec_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    unsafe {
        vec_mul_inner(a, b, out);
    }
}

unsafe fn vec_mul_inner(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        let vb = vld1q_f32(b.as_ptr().add(i * 4));
        vst1q_f32(out.as_mut_ptr().add(i * 4), vmulq_f32(va, vb));
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        out[tail + i] = a[tail + i] * b[tail + i];
    }
}

pub fn vec_scale(a: &mut [f32], scale: f32) {
    unsafe {
        vec_scale_inner(a, scale);
    }
}

unsafe fn vec_scale_inner(a: &mut [f32], scale: f32) {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;
    let vs = vdupq_n_f32(scale);

    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        vst1q_f32(a.as_mut_ptr().add(i * 4), vmulq_f32(va, vs));
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        a[tail + i] *= scale;
    }
}

pub fn sum_of_squares(a: &[f32]) -> f32 {
    unsafe { sum_of_squares_inner(a) }
}

unsafe fn sum_of_squares_inner(a: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        acc = vfmaq_f32(acc, va, va);
    }

    let mut sum = vaddvq_f32(acc);
    let tail = chunks * 4;
    for i in 0..remainder {
        sum = a[tail + i].mul_add(a[tail + i], sum);
    }
    sum
}

pub fn vec_silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for i in 0..gate.len() {
        let sigmoid = 1.0 / (1.0 + (-gate[i]).exp());
        out[i] = gate[i] * sigmoid * up[i];
    }
}

/// Q8_0 block dot product: `sum(input[i] * int8[i]) * scale` for 32 elements.
pub fn dot_q8_block(input: &[f32], quants: &[u8], scale: f32) -> f32 {
    unsafe { dot_q8_block_inner(input, quants, scale) }
}

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vcvtq_f32_s32, vmovl_high_s16, vmovl_s16, vmovl_s8};

#[inline]
unsafe fn dot_q8_block_inner(input: &[f32], quants: &[u8], scale: f32) -> f32 {
    // Process 32 int8 values in 4 groups of 8:
    // load 8×i8 → widen to 8×i16 → split to 2×4×i32 → cvt to 2×4×f32 → FMA
    let inp = input.as_ptr();
    let q = quants.as_ptr().cast::<i8>();

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    // Group 0: elements 0..7
    let q8 = std::arch::aarch64::vld1_s8(q);
    let q16 = vmovl_s8(q8); // 8×i16
    let q32_lo = vmovl_s16(std::arch::aarch64::vget_low_s16(q16)); // 4×i32
    let q32_hi = vmovl_high_s16(q16); // 4×i32
    acc0 = vfmaq_f32(acc0, vld1q_f32(inp), vcvtq_f32_s32(q32_lo));
    acc1 = vfmaq_f32(acc1, vld1q_f32(inp.add(4)), vcvtq_f32_s32(q32_hi));

    // Group 1: elements 8..15
    let q8 = std::arch::aarch64::vld1_s8(q.add(8));
    let q16 = vmovl_s8(q8);
    let q32_lo = vmovl_s16(std::arch::aarch64::vget_low_s16(q16));
    let q32_hi = vmovl_high_s16(q16);
    acc0 = vfmaq_f32(acc0, vld1q_f32(inp.add(8)), vcvtq_f32_s32(q32_lo));
    acc1 = vfmaq_f32(acc1, vld1q_f32(inp.add(12)), vcvtq_f32_s32(q32_hi));

    // Group 2: elements 16..23
    let q8 = std::arch::aarch64::vld1_s8(q.add(16));
    let q16 = vmovl_s8(q8);
    let q32_lo = vmovl_s16(std::arch::aarch64::vget_low_s16(q16));
    let q32_hi = vmovl_high_s16(q16);
    acc0 = vfmaq_f32(acc0, vld1q_f32(inp.add(16)), vcvtq_f32_s32(q32_lo));
    acc1 = vfmaq_f32(acc1, vld1q_f32(inp.add(20)), vcvtq_f32_s32(q32_hi));

    // Group 3: elements 24..31
    let q8 = std::arch::aarch64::vld1_s8(q.add(24));
    let q16 = vmovl_s8(q8);
    let q32_lo = vmovl_s16(std::arch::aarch64::vget_low_s16(q16));
    let q32_hi = vmovl_high_s16(q16);
    acc0 = vfmaq_f32(acc0, vld1q_f32(inp.add(24)), vcvtq_f32_s32(q32_lo));
    acc1 = vfmaq_f32(acc1, vld1q_f32(inp.add(28)), vcvtq_f32_s32(q32_hi));

    vaddvq_f32(vaddq_f32(acc0, acc1)) * scale
}

/// Q4_0 block dot product: `sum(input[i] * (nibble[i] - 8)) * scale` for 32 elements.
pub fn dot_q4_block(input: &[f32], packed: &[u8], scale: f32) -> f32 {
    unsafe { dot_q4_block_inner(input, packed, scale) }
}

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vld1_u8, vreinterpret_s8_u8, vshr_n_u8, vsubq_s32};

#[inline]
unsafe fn dot_q4_block_inner(input: &[f32], packed: &[u8], scale: f32) -> f32 {
    // Q4_0 layout: byte[i] has low nibble = element[i], high nibble = element[i+16].
    // So 16 bytes → low nibbles are elements 0..15, high nibbles are elements 16..31.
    let inp = input.as_ptr();
    let bias = vdupq_n_s32(8);
    let mask = std::arch::aarch64::vdup_n_u8(0x0F);

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    // Process low nibbles of all 16 bytes → elements 0..15
    for group in 0..2 {
        let raw = vld1_u8(packed.as_ptr().add(group * 8));
        let lo_u8 = std::arch::aarch64::vand_u8(raw, mask);
        let lo_signed = vreinterpret_s8_u8(lo_u8);
        let lo_s16 = vmovl_s8(lo_signed);
        let lo_s32_0 = vsubq_s32(vmovl_s16(std::arch::aarch64::vget_low_s16(lo_s16)), bias);
        let lo_s32_1 = vsubq_s32(vmovl_high_s16(lo_s16), bias);
        let base = group * 8;
        acc0 = vfmaq_f32(acc0, vld1q_f32(inp.add(base)), vcvtq_f32_s32(lo_s32_0));
        acc1 = vfmaq_f32(acc1, vld1q_f32(inp.add(base + 4)), vcvtq_f32_s32(lo_s32_1));
    }

    // Process high nibbles of all 16 bytes → elements 16..31
    for group in 0..2 {
        let raw = vld1_u8(packed.as_ptr().add(group * 8));
        let hi_u8 = vshr_n_u8(raw, 4);
        let hi_signed = vreinterpret_s8_u8(hi_u8);
        let hi_s16 = vmovl_s8(hi_signed);
        let hi_s32_0 = vsubq_s32(vmovl_s16(std::arch::aarch64::vget_low_s16(hi_s16)), bias);
        let hi_s32_1 = vsubq_s32(vmovl_high_s16(hi_s16), bias);
        let base = 16 + group * 8;
        acc0 = vfmaq_f32(acc0, vld1q_f32(inp.add(base)), vcvtq_f32_s32(hi_s32_0));
        acc1 = vfmaq_f32(acc1, vld1q_f32(inp.add(base + 4)), vcvtq_f32_s32(hi_s32_1));
    }

    vaddvq_f32(vaddq_f32(acc0, acc1)) * scale
}

/// Q4_1 block dot product: `sum(input[i] * nibble[i]) * scale + sum(input[i]) * min`
/// for 32 elements. Unlike Q4_0, nibbles are unsigned [0,15] — no bias subtraction.
pub fn dot_q4_1_block(input: &[f32], packed: &[u8], scale: f32, min: f32) -> f32 {
    unsafe { dot_q4_1_block_inner(input, packed, scale, min) }
}

#[inline]
unsafe fn dot_q4_1_block_inner(input: &[f32], packed: &[u8], scale: f32, min: f32) -> f32 {
    let inp = input.as_ptr();
    let mask = std::arch::aarch64::vdup_n_u8(0x0F);

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    // Accumulate sum(input[i]) for the min term
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);

    // Process low nibbles of all 16 bytes → elements 0..15
    for group in 0..2 {
        let raw = vld1_u8(packed.as_ptr().add(group * 8));
        let lo_u8 = std::arch::aarch64::vand_u8(raw, mask);
        let lo_signed = vreinterpret_s8_u8(lo_u8);
        let lo_s16 = vmovl_s8(lo_signed);
        let lo_s32_0 = vmovl_s16(std::arch::aarch64::vget_low_s16(lo_s16));
        let lo_s32_1 = vmovl_high_s16(lo_s16);
        let base = group * 8;
        let in0 = vld1q_f32(inp.add(base));
        let in1 = vld1q_f32(inp.add(base + 4));
        acc0 = vfmaq_f32(acc0, in0, vcvtq_f32_s32(lo_s32_0));
        acc1 = vfmaq_f32(acc1, in1, vcvtq_f32_s32(lo_s32_1));
        sum0 = vaddq_f32(sum0, in0);
        sum1 = vaddq_f32(sum1, in1);
    }

    // Process high nibbles of all 16 bytes → elements 16..31
    for group in 0..2 {
        let raw = vld1_u8(packed.as_ptr().add(group * 8));
        let hi_u8 = vshr_n_u8(raw, 4);
        let hi_signed = vreinterpret_s8_u8(hi_u8);
        let hi_s16 = vmovl_s8(hi_signed);
        let hi_s32_0 = vmovl_s16(std::arch::aarch64::vget_low_s16(hi_s16));
        let hi_s32_1 = vmovl_high_s16(hi_s16);
        let base = 16 + group * 8;
        let in0 = vld1q_f32(inp.add(base));
        let in1 = vld1q_f32(inp.add(base + 4));
        acc0 = vfmaq_f32(acc0, in0, vcvtq_f32_s32(hi_s32_0));
        acc1 = vfmaq_f32(acc1, in1, vcvtq_f32_s32(hi_s32_1));
        sum0 = vaddq_f32(sum0, in0);
        sum1 = vaddq_f32(sum1, in1);
    }

    let dot = vaddvq_f32(vaddq_f32(acc0, acc1));
    let input_sum = vaddvq_f32(vaddq_f32(sum0, sum1));
    dot * scale + input_sum * min
}

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vabsq_f32, vdupq_n_s32, vmaxq_f32, vmaxvq_f32, vst1q_s8};

// ---- Integer dot product kernels (Q8×Q8, Q4×Q8) ----
//
// These keep computation in the integer domain (i16 multiply, i32 accumulate)
// until the final per-block scale multiply. Uses the widening multiply approach:
// load 8×i8 pairs → vmull_s8 → 8×i16 → vmlal_s8/vpaddlq → 4×i32 → accumulate.

/// Inner helper: integer dot product of 32 int8 × int8 values, returning i32 sum.
///
/// Uses widening multiply: 8×i8 × 8×i8 → 8×i16, then pairwise-add to 4×i32.
#[inline]
unsafe fn dot_i8x32_inner(a: *const i8, b: *const i8) -> i32 {
    use std::arch::aarch64::{
        vaddq_s32, vaddvq_s32, vget_low_s8, vld1q_s8, vmull_high_s8, vmull_s8, vpaddlq_s16,
    };

    // Process 32 elements in 2 groups of 16
    // Group 0: elements 0..15
    let a0 = vld1q_s8(a);
    let b0 = vld1q_s8(b);
    let prod0_lo = vmull_s8(vget_low_s8(a0), vget_low_s8(b0)); // 8×i16
    let prod0_hi = vmull_high_s8(a0, b0); // 8×i16
    let sum0_lo = vpaddlq_s16(prod0_lo); // 4×i32 (pairwise add i16→i32)
    let sum0_hi = vpaddlq_s16(prod0_hi); // 4×i32

    // Group 1: elements 16..31
    let a1 = vld1q_s8(a.add(16));
    let b1 = vld1q_s8(b.add(16));
    let prod1_lo = vmull_s8(vget_low_s8(a1), vget_low_s8(b1));
    let prod1_hi = vmull_high_s8(a1, b1);
    let sum1_lo = vpaddlq_s16(prod1_lo);
    let sum1_hi = vpaddlq_s16(prod1_hi);

    // Sum all 4×i32 accumulators
    let total = vaddq_s32(vaddq_s32(sum0_lo, sum0_hi), vaddq_s32(sum1_lo, sum1_hi));
    vaddvq_s32(total)
}

/// Q8×Q8 integer row dot product using NEON widening multiply.
pub fn dot_q8_q8_row(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_quants: &[u8],
    weight_scales: &[f32],
) -> f32 {
    unsafe { dot_q8_q8_row_inner(input_quants, input_scales, weight_quants, weight_scales) }
}

#[allow(clippy::cast_precision_loss)]
#[inline]
unsafe fn dot_q8_q8_row_inner(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_quants: &[u8],
    weight_scales: &[f32],
) -> f32 {
    let num_blocks = weight_scales.len();
    let iq = input_quants.as_ptr().cast::<i8>();
    let wq = weight_quants.as_ptr().cast::<i8>();

    let mut total = 0.0f32;

    for blk in 0..num_blocks {
        let blk_offset = blk * 32;
        let combined_scale = *input_scales.get_unchecked(blk) * *weight_scales.get_unchecked(blk);

        let dot_i32 = dot_i8x32_inner(wq.add(blk_offset), iq.add(blk_offset));

        total += (dot_i32 as f32) * combined_scale;
    }

    total
}

/// Q4_0 integer row dot product using NEON: weight Q4 nibbles × input Q8.
///
/// Unpacks Q4_0 nibbles to int8, then integer dot with pre-quantized Q8 input.
pub fn dot_q4_q8_row(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_packed: &[u8],
    weight_scales: &[f32],
) -> f32 {
    unsafe { dot_q4_q8_row_inner(input_quants, input_scales, weight_packed, weight_scales) }
}

#[allow(clippy::cast_precision_loss, clippy::similar_names)]
#[inline]
unsafe fn dot_q4_q8_row_inner(
    input_quants: &[u8],
    input_scales: &[f32],
    weight_packed: &[u8],
    weight_scales: &[f32],
) -> f32 {
    use std::arch::aarch64::{
        vaddq_s32, vaddvq_s32, vand_u8, vdup_n_u8, vget_low_s8, vld1q_s8, vmull_high_s8, vmull_s8,
        vpaddlq_s16, vreinterpretq_s8_u8, vsubq_s8,
    };

    let num_blocks = weight_scales.len();
    let iq = input_quants.as_ptr().cast::<i8>();
    let wp = weight_packed.as_ptr();
    let mask = vdup_n_u8(0x0F);
    let bias = std::arch::aarch64::vdupq_n_s8(8);

    let mut total = 0.0f32;

    for blk in 0..num_blocks {
        let combined_scale = *input_scales.get_unchecked(blk) * *weight_scales.get_unchecked(blk);
        let inp_offset = blk * 32;
        let wp_offset = blk * 16;

        // Unpack 16 packed bytes → 32 int8 values
        // Low nibbles = elements 0..15, high nibbles = elements 16..31
        let raw_lo = vld1_u8(wp.add(wp_offset));
        let raw_hi = vld1_u8(wp.add(wp_offset + 8));

        // Extract low nibbles (elements 0..7 and 8..15)
        let lo_0 = vand_u8(raw_lo, mask);
        let lo_1 = vand_u8(raw_hi, mask);
        // Extract high nibbles (elements 16..23 and 24..31)
        let hi_0 = vshr_n_u8(raw_lo, 4);
        let hi_1 = vshr_n_u8(raw_hi, 4);

        // Combine into 16-byte vectors: [lo_0, lo_1] and [hi_0, hi_1]
        let lo_16 = std::arch::aarch64::vcombine_u8(lo_0, lo_1); // elements 0..15
        let hi_16 = std::arch::aarch64::vcombine_u8(hi_0, hi_1); // elements 16..31

        // Subtract bias 8 to get signed range [-8, 7]
        let lo_s8 = vsubq_s8(vreinterpretq_s8_u8(lo_16), bias);
        let hi_s8 = vsubq_s8(vreinterpretq_s8_u8(hi_16), bias);

        // Load input quants for this block
        let inp_lo = vld1q_s8(iq.add(inp_offset));
        let inp_hi = vld1q_s8(iq.add(inp_offset + 16));

        // Integer dot product: widening multiply + pairwise accumulate
        // Elements 0..15
        let prod_lo_lo = vmull_s8(vget_low_s8(lo_s8), vget_low_s8(inp_lo));
        let prod_lo_hi = vmull_high_s8(lo_s8, inp_lo);
        let sum_lo = vaddq_s32(vpaddlq_s16(prod_lo_lo), vpaddlq_s16(prod_lo_hi));

        // Elements 16..31
        let prod_hi_lo = vmull_s8(vget_low_s8(hi_s8), vget_low_s8(inp_hi));
        let prod_hi_hi = vmull_high_s8(hi_s8, inp_hi);
        let sum_hi = vaddq_s32(vpaddlq_s16(prod_hi_lo), vpaddlq_s16(prod_hi_hi));

        let dot_i32 = vaddvq_s32(vaddq_s32(sum_lo, sum_hi));
        total += (dot_i32 as f32) * combined_scale;
    }

    total
}

/// Q4_1 integer row dot product using NEON: weight Q4_1 nibbles × input Q8.
///
/// Nibbles are unsigned [0,15]. Uses integer dot for nibble×quant part,
/// f32 for the min correction term.
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
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_arguments
)]
#[inline]
unsafe fn dot_q4_1_q8_row_inner(
    input_quants: &[u8],
    input_scales: &[f32],
    input_row: &[f32],
    weight_packed: &[u8],
    weight_scales: &[f32],
    weight_mins: &[f32],
) -> f32 {
    use std::arch::aarch64::{
        vaddq_s32, vaddvq_s32, vand_u8, vcombine_u8, vdup_n_u8, vget_low_s8, vld1q_s8,
        vmull_high_s8, vmull_s8, vpaddlq_s16, vreinterpretq_s8_u8,
    };

    let num_blocks = weight_scales.len();
    let iq = input_quants.as_ptr().cast::<i8>();
    let wp = weight_packed.as_ptr();
    let inp_f32 = input_row.as_ptr();
    let mask = vdup_n_u8(0x0F);

    let mut total_dot = 0.0f32;
    let mut total_min = 0.0f32;

    for blk in 0..num_blocks {
        let scale = *weight_scales.get_unchecked(blk);
        let min = *weight_mins.get_unchecked(blk);
        let input_scale = *input_scales.get_unchecked(blk);
        let inp_offset = blk * 32;
        let wp_offset = blk * 16;

        // Unpack Q4_1 nibbles (unsigned, no bias)
        let raw_lo = vld1_u8(wp.add(wp_offset));
        let raw_hi = vld1_u8(wp.add(wp_offset + 8));
        let lo_0 = vand_u8(raw_lo, mask);
        let lo_1 = vand_u8(raw_hi, mask);
        let hi_0 = vshr_n_u8(raw_lo, 4);
        let hi_1 = vshr_n_u8(raw_hi, 4);
        let lo_16 = vcombine_u8(lo_0, lo_1);
        let hi_16 = vcombine_u8(hi_0, hi_1);

        // Nibbles are unsigned [0,15]. Input quants are signed int8.
        // Reinterpret nibbles as signed (values 0..15 fit in signed i8).
        let lo_s8 = vreinterpretq_s8_u8(lo_16);
        let hi_s8 = vreinterpretq_s8_u8(hi_16);

        let inp_lo = vld1q_s8(iq.add(inp_offset));
        let inp_hi = vld1q_s8(iq.add(inp_offset + 16));

        // Integer dot: nibble × input_quant
        let prod_lo_lo = vmull_s8(vget_low_s8(lo_s8), vget_low_s8(inp_lo));
        let prod_lo_hi = vmull_high_s8(lo_s8, inp_lo);
        let sum_lo = vaddq_s32(vpaddlq_s16(prod_lo_lo), vpaddlq_s16(prod_lo_hi));

        let prod_hi_lo = vmull_s8(vget_low_s8(hi_s8), vget_low_s8(inp_hi));
        let prod_hi_hi = vmull_high_s8(hi_s8, inp_hi);
        let sum_hi = vaddq_s32(vpaddlq_s16(prod_hi_lo), vpaddlq_s16(prod_hi_hi));

        let dot_i32 = vaddvq_s32(vaddq_s32(sum_lo, sum_hi));
        total_dot += (dot_i32 as f32) * scale * input_scale;

        // Min correction: sum(input_f32[blk]) * min
        let mut block_sum = vdupq_n_f32(0.0);
        for g in 0..4 {
            block_sum = vaddq_f32(block_sum, vld1q_f32(inp_f32.add(inp_offset + g * 4)));
        }
        // Use 4 more groups for remaining 16 elements
        for g in 0..4 {
            block_sum = vaddq_f32(block_sum, vld1q_f32(inp_f32.add(inp_offset + 16 + g * 4)));
        }
        total_min += vaddvq_f32(block_sum) * min;
    }

    total_dot + total_min
}

/// NEON-accelerated quantization of f32 row to Q8_0 format.
pub fn quantize_row_q8(input: &[f32], out_quants: &mut [u8], out_scales: &mut [f32]) {
    unsafe { quantize_row_q8_inner(input, out_quants, out_scales) }
}

#[inline]
unsafe fn quantize_row_q8_inner(input: &[f32], out_quants: &mut [u8], out_scales: &mut [f32]) {
    use std::arch::aarch64::{
        vcvtq_s32_f32, vmovn_high_s32, vmovn_s32, vqmovn_high_s16, vqmovn_s16,
    };

    let num_blocks = out_scales.len();

    for blk in 0..num_blocks {
        let blk_start = blk * 32;
        let inp = input.as_ptr().add(blk_start);

        // Find max absolute value across 32 elements using NEON
        let mut max_abs_v = vdupq_n_f32(0.0);
        for g in 0..8 {
            let v = vld1q_f32(inp.add(g * 4));
            max_abs_v = vmaxq_f32(max_abs_v, vabsq_f32(v));
        }
        let max_scalar = vmaxvq_f32(max_abs_v);

        let scale = max_scalar / 127.0;
        *out_scales.get_unchecked_mut(blk) = scale;

        if scale == 0.0 {
            for i in 0..32 {
                *out_quants.get_unchecked_mut(blk_start + i) = 0;
            }
            continue;
        }

        let inv_scale = vdupq_n_f32(1.0 / scale);

        // Quantize: round(input / scale), clamped to [-127, 127] by saturating narrow
        // Process 32 floats → 8×i32 → 4×i16 → ... → 32×i8
        let v0 = vcvtq_s32_f32(vmulq_f32(vld1q_f32(inp), inv_scale));
        let v1 = vcvtq_s32_f32(vmulq_f32(vld1q_f32(inp.add(4)), inv_scale));
        let v2 = vcvtq_s32_f32(vmulq_f32(vld1q_f32(inp.add(8)), inv_scale));
        let v3 = vcvtq_s32_f32(vmulq_f32(vld1q_f32(inp.add(12)), inv_scale));
        let v4 = vcvtq_s32_f32(vmulq_f32(vld1q_f32(inp.add(16)), inv_scale));
        let v5 = vcvtq_s32_f32(vmulq_f32(vld1q_f32(inp.add(20)), inv_scale));
        let v6 = vcvtq_s32_f32(vmulq_f32(vld1q_f32(inp.add(24)), inv_scale));
        let v7 = vcvtq_s32_f32(vmulq_f32(vld1q_f32(inp.add(28)), inv_scale));

        // Narrow: 4×i32 → 4×i16 (saturating), combine pairs → 8×i16
        let n01 = vmovn_high_s32(vmovn_s32(v0), v1); // 8×i16
        let n23 = vmovn_high_s32(vmovn_s32(v2), v3);
        let n45 = vmovn_high_s32(vmovn_s32(v4), v5);
        let n67 = vmovn_high_s32(vmovn_s32(v6), v7);

        // Narrow: 8×i16 → 8×i8 (saturating), combine pairs → 16×i8
        let b0 = vqmovn_high_s16(vqmovn_s16(n01), n23); // 16×i8
        let b1 = vqmovn_high_s16(vqmovn_s16(n45), n67); // 16×i8

        // Store 32 bytes
        vst1q_s8(out_quants.as_mut_ptr().add(blk_start).cast::<i8>(), b0);
        vst1q_s8(out_quants.as_mut_ptr().add(blk_start + 16).cast::<i8>(), b1);
    }
}
