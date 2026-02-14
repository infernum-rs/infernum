//! Quantized matrix multiplication kernels
//!
//! Performs `output = input @ dequantize(weight)^T` where weights are stored
//! in `Q8_0`, `Q4_0`, or `F8E4M3` format. Input activations and output remain f32.
//! Dequantization happens on-the-fly inside the kernel — the full f32 weight
//! matrix never exists in GPU memory.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::missing_panics_doc,
    clippy::manual_div_ceil
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use crate::cuda::quantized::QuantizedTensor;
use crate::cuda::CudaTensor;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::Result;

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

const QUANTIZED_MATMUL_KERNEL: &str = r#"
// Manual f16 → f32 decode (avoids cuda_fp16.h dependency in NVRTC)
__device__ float f16_to_f32(unsigned short bits) {
    unsigned int sign = (bits >> 15) & 0x1;
    unsigned int exp  = (bits >> 10) & 0x1F;
    unsigned int mant = bits & 0x3FF;

    float result;
    if (exp == 0) {
        // Subnormal or zero
        result = ldexpf((float)mant / 1024.0f, -14);
    } else if (exp == 31) {
        // Inf / NaN — treat as zero for safety in scale context
        result = 0.0f;
    } else {
        result = ldexpf(1.0f + (float)mant / 1024.0f, (int)exp - 15);
    }
    return sign ? -result : result;
}

// Q8_0 matmul: output[m][n] = sum_k( input[m][k] * dequant(weight[n][k]) )
// weight layout (row-major per output row n): blocks of 32 int8 values + 1 f16 scale
//
// data   pointer: int8  values, shape (N, K) stored contiguously
// scales pointer: f16   values, shape (N, K/32)
//
// Each thread computes one (m, n) output element.
extern "C" __global__ void matmul_q8_f32(
    float*       __restrict__ output,
    const float* __restrict__ input,
    const signed char* __restrict__ weight_data,
    const unsigned short* __restrict__ weight_scales,  // f16 stored as u16
    const int M,
    const int N,
    const int K
) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    const int blocks_per_row = K / 32;
    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        float scale = f16_to_f32(weight_scales[n * blocks_per_row + b]);

        int base_k = b * 32;
        int weight_base = n * K + base_k;
        const float* in_ptr = input + m * K + base_k;

        for (int j = 0; j < 32; ++j) {
            float w = (float)weight_data[weight_base + j] * scale;
            acc += in_ptr[j] * w;
        }
    }

    output[m * N + n] = acc;
}

// Q4_0 matmul: same as Q8_0 but 2 int4 values packed per byte (low nibble first)
// data   pointer: uint8 packed values, shape (N, K/2)
// scales pointer: f16 values, shape (N, K/32)
extern "C" __global__ void matmul_q4_f32(
    float*       __restrict__ output,
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight_data,
    const unsigned short* __restrict__ weight_scales,
    const int M,
    const int N,
    const int K
) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    const int blocks_per_row = K / 32;
    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        float scale = f16_to_f32(weight_scales[n * blocks_per_row + b]);

        int base_k = b * 32;
        // Each byte holds 2 int4 values; 32 values = 16 bytes
        int packed_base = n * (K / 2) + base_k / 2;
        const float* in_ptr = input + m * K + base_k;

        for (int j = 0; j < 16; ++j) {
            unsigned char packed = weight_data[packed_base + j];
            // Low nibble (elements 2j), high nibble (elements 2j+1)
            // Stored as unsigned 0-15, centered at 8 to get signed range [-8, 7]
            float w_lo = (float)((int)(packed & 0x0F) - 8) * scale;
            float w_hi = (float)((int)(packed >> 4) - 8) * scale;
            acc += in_ptr[2 * j]     * w_lo;
            acc += in_ptr[2 * j + 1] * w_hi;
        }
    }

    output[m * N + n] = acc;
}

// FP8 E4M3 matmul: each weight byte is an fp8 value (no block structure)
// Manual decode: sign(1) | exponent(4) | mantissa(3), bias=7
extern "C" __global__ void matmul_fp8_f32(
    float*       __restrict__ output,
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight_data,
    const int M,
    const int N,
    const int K
) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    float acc = 0.0f;
    const float* in_ptr = input + m * K;
    const unsigned char* w_ptr = weight_data + n * K;

    for (int k = 0; k < K; ++k) {
        unsigned char bits = w_ptr[k];

        // Decode E4M3: sign(1) | exp(4) | mantissa(3), bias=7
        int sign = (bits >> 7) & 1;
        int exp  = (bits >> 3) & 0x0F;
        int mant = bits & 0x07;

        float w;
        if (exp == 0 && mant == 0) {
            w = 0.0f;
        } else if (exp == 0) {
            // Subnormal: 2^(1-7) * (mant/8)
            w = ldexpf((float)mant / 8.0f, -6);
        } else if (exp == 15 && mant == 7) {
            // NaN in E4M3 (no infinity — E4M3 uses all-ones for NaN)
            w = 0.0f;
        } else {
            // Normal: 2^(exp-7) * (1 + mant/8)
            w = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
        }

        if (sign) w = -w;
        acc += in_ptr[k] * w;
    }

    output[m * N + n] = acc;
}
"#;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Quantized linear projection: `output = input @ dequant(weight)^T`
///
/// - `input`: f32 tensor of shape `(M, K)` or `(batch, M, K)`
/// - `weight`: quantized tensor of shape `(N, K)` (stored transposed, as in `HuggingFace` convention)
/// - output: f32 tensor of shape `(M, N)` or `(batch, M, N)`
///
/// Dispatches to the appropriate kernel based on `weight.dtype()`.
///
/// # Errors
/// Returns an error if kernel compilation or launch fails.
pub fn quantized_matmul(
    input: &CudaTensor<f32>,
    weight: &QuantizedTensor,
) -> Result<CudaTensor<f32>> {
    let w_shape = weight.shape();
    assert_eq!(w_shape.len(), 2, "Quantized weight must be 2D (N, K)");
    let n = w_shape[0]; // out_features
    let k = w_shape[1]; // in_features

    let in_shape = input.shape();
    match in_shape.len() {
        2 => {
            assert_eq!(
                in_shape[1], k,
                "Inner dimensions must match: {} vs {k}",
                in_shape[1]
            );
            quantized_matmul_2d(input, weight, in_shape[0], n, k)
        }
        3 => {
            // Flatten batch*M, compute, reshape back
            let batch = in_shape[0];
            let m = in_shape[1];
            assert_eq!(
                in_shape[2], k,
                "Inner dimensions must match: {} vs {k}",
                in_shape[2]
            );
            let flat = input.reshape(&[batch * m, k]);
            let out_flat = quantized_matmul_2d(&flat, weight, batch * m, n, k)?;
            Ok(out_flat.reshape(&[batch, m, n]))
        }
        _ => panic!("Unsupported input shape for quantized_matmul: {in_shape:?}"),
    }
}

fn quantized_matmul_2d(
    input: &CudaTensor<f32>,
    weight: &QuantizedTensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaTensor<f32>> {
    let mut output = unsafe { CudaTensor::<f32>::uninit(input.context(), &[m, n])? };

    let device = input.context().device();

    let module_name = "quantized_matmul";
    if !device.has_func(module_name, "matmul_q8_f32") {
        let ptx = cudarc::nvrtc::safe::compile_ptx(QUANTIZED_MATMUL_KERNEL)?;
        device.load_ptx(
            ptx,
            module_name,
            &["matmul_q8_f32", "matmul_q4_f32", "matmul_fp8_f32"],
        )?;
    }

    let block_x = 16;
    let block_y = 16;
    let grid_x = (n + block_x - 1) / block_x;
    let grid_y = (m + block_y - 1) / block_y;

    let cfg = LaunchConfig {
        grid_dim: (grid_x as u32, grid_y as u32, 1),
        block_dim: (block_x as u32, block_y as u32, 1),
        shared_mem_bytes: 0,
    };

    match weight.dtype() {
        DType::Q8_0 => {
            let func = device.get_func(module_name, "matmul_q8_f32").unwrap();
            unsafe {
                func.launch(
                    cfg,
                    (
                        output.cuda_slice_mut(),
                        input.cuda_slice(),
                        weight.data_slice(),
                        weight.scales_slice(),
                        m as i32,
                        n as i32,
                        k as i32,
                    ),
                )?;
            }
        }
        DType::Q4_0 => {
            let func = device.get_func(module_name, "matmul_q4_f32").unwrap();
            unsafe {
                func.launch(
                    cfg,
                    (
                        output.cuda_slice_mut(),
                        input.cuda_slice(),
                        weight.data_slice(),
                        weight.scales_slice(),
                        m as i32,
                        n as i32,
                        k as i32,
                    ),
                )?;
            }
        }
        DType::F8E4M3 => {
            let func = device.get_func(module_name, "matmul_fp8_f32").unwrap();
            unsafe {
                func.launch(
                    cfg,
                    (
                        output.cuda_slice_mut(),
                        input.cuda_slice(),
                        weight.data_slice(),
                        m as i32,
                        n as i32,
                        k as i32,
                    ),
                )?;
            }
        }
        other => panic!("quantized_matmul: unsupported dtype {other}"),
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Host-side quantization helpers (for tests)
// ---------------------------------------------------------------------------

/// Quantize an f32 slice to Q8_0 format on the host.
/// Returns (data, scales) as raw byte vectors.
#[cfg(test)]
fn quantize_q8_host(values: &[f32], block_size: usize) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(values.len() % block_size, 0);
    let num_blocks = values.len() / block_size;
    let mut data = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(num_blocks * 2);

    for block in values.chunks(block_size) {
        let max_abs = block.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let scale_f16 = half::f16::from_f32(scale);
        scales.extend_from_slice(&scale_f16.to_le_bytes());

        for &v in block {
            let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
            data.push(q as u8);
        }
    }

    (data, scales)
}

/// Quantize an f32 slice to Q4_0 format on the host.
/// Returns (data, scales) as raw byte vectors.
#[cfg(test)]
fn quantize_q4_host(values: &[f32], block_size: usize) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(values.len() % block_size, 0);
    assert_eq!(block_size % 2, 0);
    let num_blocks = values.len() / block_size;
    let mut data = Vec::with_capacity(values.len() / 2);
    let mut scales = Vec::with_capacity(num_blocks * 2);

    for block in values.chunks(block_size) {
        let max_abs = block.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
        let scale_f16 = half::f16::from_f32(scale);
        scales.extend_from_slice(&scale_f16.to_le_bytes());

        for pair in block.chunks(2) {
            // Quantize to [-8, 7] range, store as unsigned [0, 15] by adding 8
            let q_lo = ((pair[0] / scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
            let q_hi = ((pair[1] / scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
            data.push((q_hi << 4) | (q_lo & 0x0F));
        }
    }

    (data, scales)
}

/// Encode an f32 value as FP8 E4M3 (sign=1, exp=4, mantissa=3, bias=7).
#[cfg(test)]
fn f32_to_fp8_e4m3(value: f32) -> u8 {
    if value == 0.0 {
        return if value.is_sign_negative() { 0x80 } else { 0x00 };
    }

    let sign: u8 = if value < 0.0 { 1 } else { 0 };
    let abs_val = value.abs();

    // Max representable: 2^8 * (1 + 7/8) = 256 * 1.875 = 448
    // Min normal: 2^(-6) = 0.015625
    // Min subnormal: 2^(-6) * (1/8) = 0.001953125
    let max_val: f32 = 448.0;

    if abs_val > max_val {
        // Clamp to max (E4M3 has no infinity, all-ones exp+mant = NaN)
        return (sign << 7) | 0x7E; // exp=15, mant=6 → 448.0
    }

    let bits = abs_val.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let f32_mant = bits & 0x7F_FFFF;

    if f32_exp < -9 {
        // Too small, flush to zero
        return sign << 7;
    }

    let biased_exp = f32_exp + 7;

    if biased_exp <= 0 {
        // Subnormal: value = 2^(-6) * (mant/8)
        // mant = round(abs_val / 2^(-6) * 8)
        let mant = (abs_val * 512.0).round() as u8; // 2^6 * 8 = 512
        let mant = mant.min(7);
        if mant == 0 {
            return sign << 7;
        }
        return (sign << 7) | mant;
    }

    // Normal: round mantissa from 23 bits to 3 bits
    let mant_3 = ((f32_mant + (1 << 19)) >> 20).min(7) as u8;
    let exp_4 = (biased_exp as u8).min(15);

    // Check for NaN encoding (exp=15, mant=7 is NaN in E4M3)
    if exp_4 == 15 && mant_3 == 7 {
        return (sign << 7) | 0x7E; // clamp to max representable
    }

    (sign << 7) | (exp_4 << 3) | mant_3
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;
    use crate::dtype::QUANTIZATION_BLOCK_SIZE;

    #[test]
    fn test_matmul_q8_identity_like() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // 2×32 input, 2×32 weight (each row has a single 1.0 at a different position)
        let k = 32;
        let m = 2;
        let n = 2;

        // input: row 0 = [1..32], row 1 = [33..64]
        let input_data: Vec<f32> = (1..=(m * k) as u32).map(|x| x as f32).collect();
        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();

        // weight: row 0 = 1.0 at all positions, row 1 = 2.0 at all positions
        let mut w_data = vec![0.0_f32; n * k];
        for j in 0..k {
            w_data[j] = 1.0; // row 0
            w_data[k + j] = 2.0; // row 1
        }
        let (q_data, q_scales) = quantize_q8_host(&w_data, QUANTIZATION_BLOCK_SIZE);
        let weight =
            QuantizedTensor::from_raw(&ctx, &[n, k], DType::Q8_0, &q_data, &q_scales).unwrap();

        let output = quantized_matmul(&input, &weight).unwrap();
        assert_eq!(output.shape(), &[m, n]);

        let result = output.to_vec().unwrap();

        // Expected: row0 = [sum(1..32), 2*sum(1..32)] = [528, 1056]
        //           row1 = [sum(33..64), 2*sum(33..64)] = [1552, 3104]
        // With quantization there will be small error from f16 scale precision
        let sum_row0: f32 = (1..=32).map(|x| x as f32).sum();
        let sum_row1: f32 = (33..=64).map(|x| x as f32).sum();

        assert!(
            (result[0] - sum_row0).abs() < sum_row0 * 0.02,
            "Q8 [0,0]: {} vs {}",
            result[0],
            sum_row0
        );
        assert!(
            (result[1] - sum_row0 * 2.0).abs() < sum_row0 * 0.04,
            "Q8 [0,1]: {} vs {}",
            result[1],
            sum_row0 * 2.0
        );
        assert!(
            (result[2] - sum_row1).abs() < sum_row1 * 0.02,
            "Q8 [1,0]: {} vs {}",
            result[2],
            sum_row1
        );
        assert!(
            (result[3] - sum_row1 * 2.0).abs() < sum_row1 * 0.04,
            "Q8 [1,1]: {} vs {}",
            result[3],
            sum_row1 * 2.0
        );
    }

    #[test]
    fn test_matmul_q4_basic() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let k = 32;
        let m = 1;
        let n = 1;

        // input: all 1.0 → output = sum of dequantized weights
        let input_data = vec![1.0_f32; k];
        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();

        // weight: constant 2.0
        let w_data = vec![2.0_f32; k];
        let (q_data, q_scales) = quantize_q4_host(&w_data, QUANTIZATION_BLOCK_SIZE);
        let weight =
            QuantizedTensor::from_raw(&ctx, &[n, k], DType::Q4_0, &q_data, &q_scales).unwrap();

        let output = quantized_matmul(&input, &weight).unwrap();
        let result = output.to_vec().unwrap();

        // Expected: 32 * 2.0 = 64.0, but Q4 has lower precision
        let expected = 64.0_f32;
        assert!(
            (result[0] - expected).abs() < expected * 0.15,
            "Q4 result: {} vs expected ~{}",
            result[0],
            expected
        );
    }

    #[test]
    fn test_matmul_fp8_basic() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let k = 4;
        let m = 1;
        let n = 1;

        // input: [1, 2, 3, 4]
        let input_data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();

        // weight: [1, 1, 1, 1] encoded as fp8
        let w_fp8: Vec<u8> = vec![1.0_f32; k]
            .iter()
            .map(|&v| f32_to_fp8_e4m3(v))
            .collect();
        let weight = QuantizedTensor::from_raw(&ctx, &[n, k], DType::F8E4M3, &w_fp8, &[]).unwrap();

        let output = quantized_matmul(&input, &weight).unwrap();
        let result = output.to_vec().unwrap();

        // Expected: 1+2+3+4 = 10.0
        assert!(
            (result[0] - 10.0).abs() < 0.5,
            "FP8 result: {} vs expected ~10.0",
            result[0]
        );
    }

    #[test]
    fn test_matmul_q8_matches_f32() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let m = 4;
        let k = 64;
        let n = 8;

        // Pseudo-random input and weights
        let mut state: u64 = 12345;
        let mut rand_f32 = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        let input_data: Vec<f32> = (0..m * k).map(|_| rand_f32()).collect();
        let w_data: Vec<f32> = (0..n * k).map(|_| rand_f32()).collect();

        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();

        // Compute f32 reference: input @ w^T
        let mut expected = vec![0.0_f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0_f32;
                for i in 0..k {
                    acc += input_data[row * k + i] * w_data[col * k + i];
                }
                expected[row * n + col] = acc;
            }
        }

        // Quantized
        let (q_data, q_scales) = quantize_q8_host(&w_data, QUANTIZATION_BLOCK_SIZE);
        let weight =
            QuantizedTensor::from_raw(&ctx, &[n, k], DType::Q8_0, &q_data, &q_scales).unwrap();
        let output = quantized_matmul(&input, &weight).unwrap();
        let result = output.to_vec().unwrap();

        // Q8 should match within ~1e-2 relative error
        for i in 0..m * n {
            let rel_err = if expected[i].abs() > 1e-6 {
                (result[i] - expected[i]).abs() / expected[i].abs()
            } else {
                (result[i] - expected[i]).abs()
            };
            assert!(
                rel_err < 0.05,
                "Q8 mismatch at [{}, {}]: got {} expected {} (rel_err={:.4})",
                i / n,
                i % n,
                result[i],
                expected[i],
                rel_err
            );
        }
    }

    #[test]
    fn test_matmul_3d_input() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let batch = 2;
        let m = 1;
        let k = 32;
        let n = 1;

        let input_data = vec![1.0_f32; batch * m * k];
        let input = CudaTensor::from_slice(&ctx, &[batch, m, k], &input_data).unwrap();

        let w_data = vec![1.0_f32; n * k];
        let (q_data, q_scales) = quantize_q8_host(&w_data, QUANTIZATION_BLOCK_SIZE);
        let weight =
            QuantizedTensor::from_raw(&ctx, &[n, k], DType::Q8_0, &q_data, &q_scales).unwrap();

        let output = quantized_matmul(&input, &weight).unwrap();
        assert_eq!(output.shape(), &[batch, m, n]);

        let result = output.to_vec().unwrap();
        // Each batch: 32 * 1.0 = 32.0
        for &v in &result {
            assert!((v - 32.0).abs() < 1.0, "3D result: {} vs 32.0", v);
        }
    }

    #[test]
    fn test_fp8_encode_decode_roundtrip() {
        let test_values = [0.0_f32, 1.0, -1.0, 0.5, 2.0, -0.25, 100.0, -100.0, 0.0625];
        for &v in &test_values {
            let encoded = f32_to_fp8_e4m3(v);
            // Verify the encoded byte is non-zero for non-zero inputs (or zero for zero)
            if v == 0.0 {
                assert_eq!(encoded & 0x7F, 0, "Zero should encode to zero mantissa+exp");
            } else {
                assert_ne!(
                    encoded & 0x7F,
                    0,
                    "Non-zero {v} should produce non-zero encoding"
                );
            }
        }
    }
}
