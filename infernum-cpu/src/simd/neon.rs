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
use std::arch::aarch64::vdupq_n_s32;
