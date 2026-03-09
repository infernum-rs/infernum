//! MatmulOps implementation for Metal.
//!
//! Phase 1: CPU-side naive matmul via unified memory. This is correct
//! but slow — the hot path for Metal performance will be MPSMatrixMultiplication
//! or custom MSL kernels added later.
//!
//! Quantized weights (Q8_0, Q4_0, Q4_1, Q6_K) are stored in their original
//! `(out_features, in_features)` layout. The quantized linear kernel iterates
//! per output neuron (row), dequantizing blocks on the fly.

use infernum::backend::MatmulOps;
use infernum::dtype::{DType, Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_SIZE_BYTES, QUANTIZATION_BLOCK_SIZE};
use infernum::tensor::Tensor;
use infernum::weights::host::HostLinearWeight;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::weights::{decode_f16_scales, MetalLinearWeight, MetalQuantizedWeight};
use crate::MetalBackend;
use crate::MetalContext;

use metal::MTLSize;

/// Tile size for the tiled matmul kernel — must match TILE in matmul.metal.
const TILE: u64 = 16;

/// Packed params for the tiled matmul kernel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulParams {
    m: u32,
    n: u32,
    k: u32,
}

// ---------------------------------------------------------------------------
// Quantized linear (CPU-side, Phase 1)
// ---------------------------------------------------------------------------

/// Quantized linear: `input (M, K) × weight (N, K)_quantized → output (M, N)`.
///
/// Weight is stored as `(out_features=N, in_features=K)` in block-quantized
/// format. Scales are pre-decoded to f32 so no f16→f32 conversion happens
/// in the hot path.
///
/// NOTE: Stays on CPU because quantized data lives in `Vec<u8>`, not on
/// GPU buffers. Moving to GPU requires refactoring the weight storage.
#[allow(
    clippy::many_single_char_names,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]
fn quantized_linear(input: &MetalTensor, weight: &MetalQuantizedWeight) -> Result<MetalTensor> {
    let i_shape = input.shape();
    let m: usize = i_shape[..i_shape.len() - 1].iter().product();
    let k = *i_shape.last().unwrap();

    let n = weight.shape[0]; // out_features
    let wk = weight.shape[1]; // in_features
    assert_eq!(
        k, wk,
        "quantized_linear: input dim {k} != weight in_features {wk}"
    );

    let input_data = input.as_f32_slice();
    let mut output = vec![0.0f32; m * n];

    match weight.dtype {
        DType::Q8_0 => {
            let num_blocks_per_row = k / QUANTIZATION_BLOCK_SIZE;
            let quants = &weight.data;
            let scales = &weight.scales;
            let quant_bytes_per_row = num_blocks_per_row * QUANTIZATION_BLOCK_SIZE;

            for row in 0..m {
                let inp = &input_data[row * k..(row + 1) * k];
                for neuron in 0..n {
                    let q_start = neuron * quant_bytes_per_row;
                    let s_start = neuron * num_blocks_per_row;
                    let mut sum = 0.0f32;
                    for b in 0..num_blocks_per_row {
                        let scale = scales[s_start + b];
                        let block_start = q_start + b * QUANTIZATION_BLOCK_SIZE;
                        let inp_start = b * QUANTIZATION_BLOCK_SIZE;
                        for j in 0..QUANTIZATION_BLOCK_SIZE {
                            let q = quants[block_start + j] as i8;
                            sum += inp[inp_start + j] * f32::from(q) * scale;
                        }
                    }
                    output[row * n + neuron] = sum;
                }
            }
        }
        DType::Q4_0 => {
            let num_blocks_per_row = k / QUANTIZATION_BLOCK_SIZE;
            let packed = &weight.data;
            let scales = &weight.scales;
            let packed_bytes_per_row = num_blocks_per_row * (QUANTIZATION_BLOCK_SIZE / 2);

            for row in 0..m {
                let inp = &input_data[row * k..(row + 1) * k];
                for neuron in 0..n {
                    let p_start = neuron * packed_bytes_per_row;
                    let s_start = neuron * num_blocks_per_row;
                    let mut sum = 0.0f32;
                    for b in 0..num_blocks_per_row {
                        let scale = scales[s_start + b];
                        let block_start = p_start + b * (QUANTIZATION_BLOCK_SIZE / 2);
                        let inp_start = b * QUANTIZATION_BLOCK_SIZE;
                        for j in 0..QUANTIZATION_BLOCK_SIZE / 2 {
                            let byte = packed[block_start + j];
                            let lo = i32::from(byte & 0x0F) - 8;
                            let hi = i32::from(byte >> 4) - 8;
                            sum += inp[inp_start + j] * lo as f32 * scale;
                            sum += inp[inp_start + j + 16] * hi as f32 * scale;
                        }
                    }
                    output[row * n + neuron] = sum;
                }
            }
        }
        DType::Q4_1 => {
            let num_blocks_per_row = k / QUANTIZATION_BLOCK_SIZE;
            let packed = &weight.data;
            let scales = &weight.scales;
            let mins = weight
                .mins
                .as_ref()
                .expect("Q4_1 weight missing mins buffer");
            let packed_bytes_per_row = num_blocks_per_row * (QUANTIZATION_BLOCK_SIZE / 2);

            for row in 0..m {
                let inp = &input_data[row * k..(row + 1) * k];
                for neuron in 0..n {
                    let p_start = neuron * packed_bytes_per_row;
                    let s_start = neuron * num_blocks_per_row;
                    let mut sum = 0.0f32;
                    for b in 0..num_blocks_per_row {
                        let scale = scales[s_start + b];
                        let min = mins[s_start + b];
                        let block_start = p_start + b * (QUANTIZATION_BLOCK_SIZE / 2);
                        let inp_start = b * QUANTIZATION_BLOCK_SIZE;
                        for j in 0..QUANTIZATION_BLOCK_SIZE / 2 {
                            let byte = packed[block_start + j];
                            let lo = f32::from(byte & 0x0F);
                            let hi = f32::from(byte >> 4);
                            sum += inp[inp_start + j] * (lo * scale + min);
                            sum += inp[inp_start + j + 16] * (hi * scale + min);
                        }
                    }
                    output[row * n + neuron] = sum;
                }
            }
        }
        DType::Q6_K => {
            assert_eq!(
                k % Q6_K_BLOCK_ELEMENTS,
                0,
                "quantized_linear: in_features {k} not divisible by Q6_K block size {Q6_K_BLOCK_ELEMENTS}"
            );
            let num_superblocks_per_row = k / Q6_K_BLOCK_ELEMENTS;
            let raw = &weight.data;
            let superblock_bytes_per_row = num_superblocks_per_row * Q6_K_BLOCK_SIZE_BYTES;

            for row in 0..m {
                let inp = &input_data[row * k..(row + 1) * k];
                for neuron in 0..n {
                    let row_start = neuron * superblock_bytes_per_row;
                    let mut sum = 0.0f32;
                    for sb_idx in 0..num_superblocks_per_row {
                        let bs = row_start + sb_idx * Q6_K_BLOCK_SIZE_BYTES;
                        let ql = &raw[bs..bs + 128];
                        let qh = &raw[bs + 128..bs + 192];
                        let scales = &raw[bs + 192..bs + 208];
                        let d = half::f16::from_le_bytes([raw[bs + 208], raw[bs + 209]]).to_f32();

                        let inp_start = sb_idx * Q6_K_BLOCK_ELEMENTS;
                        for elem in 0..Q6_K_BLOCK_ELEMENTS {
                            let sub_block = elem / 16;
                            let flat_idx = elem;
                            let row8 = flat_idx / 32;
                            let col32 = flat_idx % 32;

                            let ql_half = row8 / 4;
                            let ql_nibble_sel = (row8 % 4) / 2;
                            let ql_offset = (row8 % 4) % 2;
                            let ql_byte_idx = ql_half * 64 + ql_offset * 32 + col32;
                            let ql_byte = ql[ql_byte_idx];
                            let ql_val = if ql_nibble_sel == 0 {
                                u32::from(ql_byte & 0x0F)
                            } else {
                                u32::from(ql_byte >> 4)
                            };

                            let qh_half = row8 / 4;
                            let qh_shift_sel = row8 % 4;
                            let qh_byte_idx = qh_half * 32 + col32;
                            let qh_byte = qh[qh_byte_idx];
                            let qh_shift = qh_shift_sel * 2;
                            let qh_val = u32::from((qh_byte >> qh_shift) & 0x03);

                            let q = (ql_val | (qh_val << 4)) as i32 - 32;
                            let sc = f32::from(scales[sub_block] as i8);
                            sum += inp[inp_start + elem] * d * sc * q as f32;
                        }
                    }
                    output[row * n + neuron] = sum;
                }
            }
        }
        other => {
            return Err(infernum::Error::UnsupportedDtype(format!(
                "quantized_linear: unsupported dtype {other}"
            )));
        }
    }

    let mut out_shape = i_shape[..i_shape.len() - 1].to_vec();
    out_shape.push(n);
    Ok(MetalTensor::from_f32(input.context(), &out_shape, &output))
}

// ---------------------------------------------------------------------------
// MatmulOps
// ---------------------------------------------------------------------------

impl MatmulOps for MetalBackend {
    type LinearWeight = MetalLinearWeight;

    #[allow(clippy::many_single_char_names, clippy::cast_possible_truncation)]
    fn matmul(a: &MetalTensor, b: &MetalTensor) -> Result<MetalTensor> {
        // a: (..., M, K), b: (K, N) → (..., M, N)
        let a_shape = a.shape();
        let b_shape = b.shape();
        let k = *a_shape.last().unwrap();
        let n = b_shape[1];
        let m = a.numel() / k;
        assert_eq!(k, b_shape[0], "matmul: K mismatch");

        let ctx = a.context();
        let mut out_shape = a_shape.to_vec();
        *out_shape.last_mut().unwrap() = n;
        let out = MetalTensor::zeros(ctx, &out_shape, DType::F32);

        let params = MatmulParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
        };

        let threadgroups = MTLSize::new((n as u64).div_ceil(TILE), (m as u64).div_ceil(TILE), 1);
        let threads_per_group = MTLSize::new(TILE, TILE, 1);

        ctx.dispatch_threadgroups(
            "matmul_f32",
            &[
                (a.metal_buffer(), a.buffer_offset()),
                (b.metal_buffer(), b.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            bytemuck::bytes_of(&params),
            threadgroups,
            threads_per_group,
            0,
        );

        Ok(out)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn linear(input: &MetalTensor, weight: &MetalLinearWeight) -> Result<MetalTensor> {
        match weight {
            MetalLinearWeight::Dense { weight_t, .. } => {
                // weight_t: (N, K) — pre-transposed
                let wt_shape = weight_t.shape();
                let n = wt_shape[0];
                let k = wt_shape[1];
                let m = input.numel() / k;

                let ctx = input.context();
                let mut out_shape = input.shape().to_vec();
                *out_shape.last_mut().unwrap() = n;
                let out = MetalTensor::zeros(ctx, &out_shape, DType::F32);

                let params = MatmulParams {
                    m: m as u32,
                    n: n as u32,
                    k: k as u32,
                };

                let threadgroups =
                    MTLSize::new((n as u64).div_ceil(TILE), (m as u64).div_ceil(TILE), 1);
                let threads_per_group = MTLSize::new(TILE, TILE, 1);

                ctx.dispatch_threadgroups(
                    "linear_dense_f32",
                    &[
                        (input.metal_buffer(), input.buffer_offset()),
                        (weight_t.metal_buffer(), weight_t.buffer_offset()),
                        (out.metal_buffer(), out.buffer_offset()),
                    ],
                    bytemuck::bytes_of(&params),
                    threadgroups,
                    threads_per_group,
                    0,
                );

                Ok(out)
            }
            MetalLinearWeight::Quantized(w) => quantized_linear(input, w),
        }
    }

    fn as_dense_weight(weight: &MetalLinearWeight) -> Option<&MetalTensor> {
        match weight {
            MetalLinearWeight::Dense { weight, .. } => Some(weight),
            MetalLinearWeight::Quantized(_) => None,
        }
    }

    fn dense_weight(tensor: MetalTensor) -> MetalLinearWeight {
        MetalLinearWeight::new_dense(tensor)
    }

    fn is_dense_weight(weight: &MetalLinearWeight) -> bool {
        matches!(weight, MetalLinearWeight::Dense { .. })
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn quantize_to_q8(
        _device: &MetalContext,
        shape: &[usize],
        data: &[f32],
    ) -> Result<MetalLinearWeight> {
        let out_features = shape[0];
        let in_features = shape[1];
        assert_eq!(
            in_features % QUANTIZATION_BLOCK_SIZE,
            0,
            "quantize_to_q8: in_features {in_features} not aligned to block size"
        );

        let num_blocks_per_row = in_features / QUANTIZATION_BLOCK_SIZE;
        let total_blocks = out_features * num_blocks_per_row;
        let mut qdata = Vec::with_capacity(total_blocks * QUANTIZATION_BLOCK_SIZE);
        let mut scales = Vec::with_capacity(total_blocks);

        for row in 0..out_features {
            let row_data = &data[row * in_features..(row + 1) * in_features];
            for b in 0..num_blocks_per_row {
                let block =
                    &row_data[b * QUANTIZATION_BLOCK_SIZE..(b + 1) * QUANTIZATION_BLOCK_SIZE];
                let max_abs = block.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
                let inv_scale = 1.0 / scale;
                scales.push(scale);
                for &v in block {
                    let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    qdata.push(q.cast_unsigned());
                }
            }
        }

        Ok(MetalLinearWeight::Quantized(MetalQuantizedWeight {
            shape: shape.to_vec(),
            dtype: DType::Q8_0,
            data: qdata,
            scales,
            mins: None,
        }))
    }

    fn upload_host_linear(
        device: &MetalContext,
        weight: &HostLinearWeight,
    ) -> Result<MetalLinearWeight> {
        match weight {
            HostLinearWeight::Dense(host) => {
                let tensor =
                    MetalTensor::from_raw_bytes(device, &host.shape, host.dtype, &host.data);
                // Cast to f32 if needed for the transposed copy
                let f32_tensor = if tensor.dtype() == DType::F32 {
                    tensor.clone()
                } else {
                    use infernum::backend::CastOps;
                    MetalBackend::cast_to_f32(&tensor)?
                };
                Ok(MetalLinearWeight::Dense {
                    weight: tensor,
                    weight_t: f32_tensor,
                })
            }
            HostLinearWeight::Quantized(hq) => match hq.dtype {
                DType::Q8_0 | DType::Q4_0 => {
                    Ok(MetalLinearWeight::Quantized(MetalQuantizedWeight {
                        shape: hq.shape.clone(),
                        dtype: hq.dtype,
                        data: hq.data.clone(),
                        scales: decode_f16_scales(&hq.scales),
                        mins: None,
                    }))
                }
                DType::Q4_1 => Ok(MetalLinearWeight::Quantized(MetalQuantizedWeight {
                    shape: hq.shape.clone(),
                    dtype: hq.dtype,
                    data: hq.data.clone(),
                    scales: decode_f16_scales(&hq.scales),
                    mins: hq.qzeros.as_deref().map(decode_f16_scales),
                })),
                DType::Q6_K => Ok(MetalLinearWeight::Quantized(MetalQuantizedWeight {
                    shape: hq.shape.clone(),
                    dtype: hq.dtype,
                    data: hq.data.clone(),
                    scales: Vec::new(),
                    mins: None,
                })),
                other => Err(infernum::Error::UnsupportedDtype(format!(
                    "Metal backend does not support {other} quantized weights"
                ))),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infernum::backend::TensorFactory;

    use crate::MetalContext;

    fn ctx() -> MetalContext {
        MetalContext::new()
    }

    #[test]
    fn test_matmul_2x3_times_3x2() {
        let c = ctx();
        // A: (2, 3), B: (3, 2) → (2, 2)
        let a = MetalBackend::from_f32_slice(&c, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b =
            MetalBackend::from_f32_slice(&c, &[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let out = MetalBackend::matmul(&a, &b).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        let result = out.as_f32_slice();
        // Row 0: 1*7+2*9+3*11 = 58, 1*8+2*10+3*12 = 64
        // Row 1: 4*7+5*9+6*11 = 139, 4*8+5*10+6*12 = 154
        assert_eq!(result, &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_linear_dense() {
        let c = ctx();
        // input: (1, 3), weight: (3, 2) → linear → (1, 2)
        let input = MetalBackend::from_f32_slice(&c, &[1, 3], &[1.0, 2.0, 3.0]).unwrap();
        let weight_tensor =
            MetalBackend::from_f32_slice(&c, &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let weight = MetalLinearWeight::new_dense(weight_tensor);
        let out = MetalBackend::linear(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        let result = out.as_f32_slice();
        // 1*1+2*3+3*5 = 22, 1*2+2*4+3*6 = 28
        assert_eq!(result, &[22.0, 28.0]);
    }

    #[test]
    fn test_linear_q8() {
        let c = ctx();
        // 2 output neurons × 32 input features (1 block each)
        // neuron 0: all quants = 1, scale = 2.0 → each element contributes input[i] * 1 * 2.0
        // neuron 1: all quants = 2, scale = 0.5 → each element contributes input[i] * 2 * 0.5
        let n = 2;
        let k = QUANTIZATION_BLOCK_SIZE; // 32

        // Build quantized weight data
        let mut qdata = vec![0u8; n * k];
        // neuron 0: quants = 1 (as i8 → u8)
        for j in 0..k {
            qdata[j] = 1_i8 as u8;
        }
        // neuron 1: quants = 2 (as i8 → u8)
        for j in 0..k {
            qdata[k + j] = 2_i8 as u8;
        }
        let scales = vec![2.0_f32, 0.5_f32];

        let weight = MetalLinearWeight::Quantized(MetalQuantizedWeight {
            shape: vec![n, k],
            dtype: DType::Q8_0,
            data: qdata,
            scales,
            mins: None,
        });

        // Input: all 1.0
        let input_data = vec![1.0f32; k];
        let input = MetalBackend::from_f32_slice(&c, &[1, k], &input_data).unwrap();

        let out = MetalBackend::linear(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[1, n]);
        let result = out.as_f32_slice();
        // neuron 0: sum(1.0 * 1 * 2.0) for 32 elements = 64.0
        // neuron 1: sum(1.0 * 2 * 0.5) for 32 elements = 32.0
        assert!(
            (result[0] - 64.0).abs() < 1e-3,
            "Q8 neuron 0: got {}, expected 64.0",
            result[0]
        );
        assert!(
            (result[1] - 32.0).abs() < 1e-3,
            "Q8 neuron 1: got {}, expected 32.0",
            result[1]
        );
    }

    #[test]
    fn test_linear_q4_0() {
        let c = ctx();
        // 1 output neuron × 32 input features (1 block)
        // Pack nibbles: each byte has lo=3, hi=5 → dequantized lo = 3-8 = -5, hi = 5-8 = -3
        let k = QUANTIZATION_BLOCK_SIZE;
        let packed: Vec<u8> = vec![0x53; k / 2]; // lo=3, hi=5
        let scale = 2.0_f32;

        let weight = MetalLinearWeight::Quantized(MetalQuantizedWeight {
            shape: vec![1, k],
            dtype: DType::Q4_0,
            data: packed,
            scales: vec![scale],
            mins: None,
        });

        // Input: all 1.0
        let input_data = vec![1.0f32; k];
        let input = MetalBackend::from_f32_slice(&c, &[1, k], &input_data).unwrap();

        let out = MetalBackend::linear(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[1, 1]);
        let result = out.as_f32_slice();
        // 16 lo elements: (-5) * 2.0 * 1.0 = -10.0 each → -160.0
        // 16 hi elements: (-3) * 2.0 * 1.0 = -6.0 each → -96.0
        // Total: -256.0
        assert!(
            (result[0] - (-256.0)).abs() < 1e-3,
            "Q4_0: got {}, expected -256.0",
            result[0]
        );
    }

    #[test]
    fn test_linear_q4_1() {
        let c = ctx();
        // 1 output neuron × 32 input features (1 block)
        // Pack nibbles: each byte has lo=3, hi=5
        // Q4_1: value = nibble * scale + min
        let k = QUANTIZATION_BLOCK_SIZE;
        let packed: Vec<u8> = vec![0x53; k / 2]; // lo=3, hi=5
        let scale = 0.5_f32;
        let min = -1.0_f32;

        let weight = MetalLinearWeight::Quantized(MetalQuantizedWeight {
            shape: vec![1, k],
            dtype: DType::Q4_1,
            data: packed,
            scales: vec![scale],
            mins: Some(vec![min]),
        });

        // Input: all 1.0
        let input_data = vec![1.0f32; k];
        let input = MetalBackend::from_f32_slice(&c, &[1, k], &input_data).unwrap();

        let out = MetalBackend::linear(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[1, 1]);
        let result = out.as_f32_slice();
        // 16 lo elements: (3 * 0.5 + (-1.0)) * 1.0 = 0.5 each → 8.0
        // 16 hi elements: (5 * 0.5 + (-1.0)) * 1.0 = 1.5 each → 24.0
        // Total: 32.0
        assert!(
            (result[0] - 32.0).abs() < 1e-3,
            "Q4_1: got {}, expected 32.0",
            result[0]
        );
    }

    #[test]
    fn test_quantize_to_q8_roundtrip() {
        let c = ctx();
        // Create a simple weight matrix and quantize it
        let shape = [2, QUANTIZATION_BLOCK_SIZE];
        let mut data = vec![0.0f32; 2 * QUANTIZATION_BLOCK_SIZE];
        for (i, v) in data.iter_mut().enumerate() {
            *v = (i as f32) * 0.1 - 3.2;
        }

        let weight = MetalBackend::quantize_to_q8(&c, &shape, &data).unwrap();
        assert!(!MetalBackend::is_dense_weight(&weight));

        // Run a linear operation to verify it produces reasonable output
        let input_data = vec![1.0f32; QUANTIZATION_BLOCK_SIZE];
        let input =
            MetalBackend::from_f32_slice(&c, &[1, QUANTIZATION_BLOCK_SIZE], &input_data).unwrap();
        let out = MetalBackend::linear(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        // Just verify no crash and output is finite
        for &v in out.as_f32_slice() {
            assert!(
                v.is_finite(),
                "quantize_to_q8 roundtrip produced non-finite value"
            );
        }
    }

    #[test]
    fn test_upload_host_quantized_q8() {
        use infernum::weights::host::{HostLinearWeight, HostQuantizedWeight};

        let c = ctx();
        let k = QUANTIZATION_BLOCK_SIZE;
        let n = 1;

        // Build a HostQuantizedWeight with Q8_0 data
        let scale_f16 = half::f16::from_f32(1.0);
        let mut scales_raw = Vec::new();
        scales_raw.extend_from_slice(&scale_f16.to_le_bytes());

        let qdata: Vec<u8> = (0..k as u8).collect();

        let hq = HostQuantizedWeight {
            shape: vec![n, k],
            dtype: DType::Q8_0,
            data: qdata.clone(),
            scales: scales_raw,
            qzeros: None,
            group_size: None,
            weight_scale: 1.0,
            channel_scales: None,
        };

        let weight =
            MetalBackend::upload_host_linear(&c, &HostLinearWeight::Quantized(hq)).unwrap();
        assert!(!MetalBackend::is_dense_weight(&weight));

        // Verify the data was loaded correctly
        if let MetalLinearWeight::Quantized(q) = &weight {
            assert_eq!(q.dtype, DType::Q8_0);
            assert_eq!(q.shape, vec![n, k]);
            assert_eq!(q.data, qdata);
            assert!((q.scales[0] - 1.0).abs() < 1e-3);
        } else {
            panic!("Expected Quantized variant");
        }
    }

    /// Build a Q6_K super-block (210 bytes) and dequantize all 256 elements to f32.
    ///
    /// Returns `(raw_bytes, dequantized_f32)`.
    fn make_q6k_superblock(d: f32, sub_scales: &[i8; 16]) -> (Vec<u8>, Vec<f32>) {
        let mut raw = vec![0u8; Q6_K_BLOCK_SIZE_BYTES]; // 210 bytes

        // ql: 128 bytes at [0..128]
        // qh: 64 bytes at [128..192]
        // scales: 16 bytes at [192..208]
        // d: 2 bytes at [208..210]

        // Set scales
        for (i, &sc) in sub_scales.iter().enumerate() {
            raw[192 + i] = sc as u8;
        }
        // Set d (f16)
        let d_f16 = half::f16::from_f32(d);
        let d_bytes = d_f16.to_le_bytes();
        raw[208] = d_bytes[0];
        raw[209] = d_bytes[1];

        // Set all ql nibbles to 5 (0x55 bytes → lo=5, hi=5)
        for i in 0..128 {
            raw[i] = 0x55;
        }
        // Set all qh to 0 (top 2 bits = 0)
        for i in 128..192 {
            raw[i] = 0x00;
        }

        // Dequantize to verify: for each element, q = (ql_val | (qh_val << 4)) - 32
        // With ql=5, qh=0: q = 5 - 32 = -27
        // value = d * sub_scales[sub_block] * q
        let mut dequant = Vec::with_capacity(Q6_K_BLOCK_ELEMENTS);
        let d_f32 = d_f16.to_f32(); // Use the actual f16→f32 value
        for elem in 0..Q6_K_BLOCK_ELEMENTS {
            let sub_block = elem / 16;
            let sc = f32::from(sub_scales[sub_block]);
            let q: f32 = -27.0; // ql=5, qh=0 → 5 - 32 = -27
            dequant.push(d_f32 * sc * q);
        }

        (raw, dequant)
    }

    #[test]
    fn test_linear_q6k() {
        let c = ctx();
        let k = Q6_K_BLOCK_ELEMENTS; // 256
        let n = 1; // 1 output neuron

        // All sub-block scales = 1
        let sub_scales: [i8; 16] = [1; 16];
        let (raw, dequant) = make_q6k_superblock(0.5, &sub_scales);

        let weight = MetalLinearWeight::Quantized(MetalQuantizedWeight {
            shape: vec![n, k],
            dtype: DType::Q6_K,
            data: raw,
            scales: Vec::new(),
            mins: None,
        });

        // Input: all 1.0 → output = sum of dequantized values
        let input_data = vec![1.0f32; k];
        let input = MetalBackend::from_f32_slice(&c, &[1, k], &input_data).unwrap();

        let out = MetalBackend::linear(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[1, n]);

        let expected: f32 = dequant.iter().sum();
        let result = out.as_f32_slice();
        assert!(
            (result[0] - expected).abs() < 1e-1,
            "Q6_K: got {}, expected {}",
            result[0],
            expected
        );
    }

    #[test]
    fn test_linear_q6k_varying_scales() {
        let c = ctx();
        let k = Q6_K_BLOCK_ELEMENTS;
        let n = 2; // 2 output neurons

        // Neuron 0: d=1.0, scales all 2
        let scales_0: [i8; 16] = [2; 16];
        let (raw_0, dequant_0) = make_q6k_superblock(1.0, &scales_0);

        // Neuron 1: d=0.25, scales all -1
        let scales_1: [i8; 16] = [-1; 16];
        let (raw_1, dequant_1) = make_q6k_superblock(0.25, &scales_1);

        let mut data = raw_0;
        data.extend_from_slice(&raw_1);

        let weight = MetalLinearWeight::Quantized(MetalQuantizedWeight {
            shape: vec![n, k],
            dtype: DType::Q6_K,
            data,
            scales: Vec::new(),
            mins: None,
        });

        // Input: all 1.0
        let input_data = vec![1.0f32; k];
        let input = MetalBackend::from_f32_slice(&c, &[1, k], &input_data).unwrap();

        let out = MetalBackend::linear(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[1, n]);

        let expected_0: f32 = dequant_0.iter().sum();
        let expected_1: f32 = dequant_1.iter().sum();
        let result = out.as_f32_slice();
        assert!(
            (result[0] - expected_0).abs() < 1.0,
            "Q6_K neuron 0: got {}, expected {}",
            result[0],
            expected_0
        );
        assert!(
            (result[1] - expected_1).abs() < 1.0,
            "Q6_K neuron 1: got {}, expected {}",
            result[1],
            expected_1
        );
    }

    #[test]
    fn test_upload_host_quantized_q6k() {
        use infernum::weights::host::{HostLinearWeight, HostQuantizedWeight};

        let c = ctx();
        let k = Q6_K_BLOCK_ELEMENTS;
        let n = 1;

        let sub_scales: [i8; 16] = [1; 16];
        let (raw, _) = make_q6k_superblock(1.0, &sub_scales);

        let hq = HostQuantizedWeight {
            shape: vec![n, k],
            dtype: DType::Q6_K,
            data: raw.clone(),
            scales: Vec::new(),
            qzeros: None,
            group_size: None,
            weight_scale: 1.0,
            channel_scales: None,
        };

        let weight =
            MetalBackend::upload_host_linear(&c, &HostLinearWeight::Quantized(hq)).unwrap();
        assert!(!MetalBackend::is_dense_weight(&weight));

        if let MetalLinearWeight::Quantized(q) = &weight {
            assert_eq!(q.dtype, DType::Q6_K);
            assert_eq!(q.shape, vec![n, k]);
            assert_eq!(q.data, raw);
            assert!(q.scales.is_empty());
            assert!(q.mins.is_none());
        } else {
            panic!("Expected Quantized variant");
        }
    }
}
