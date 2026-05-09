//! MatmulOps implementation for Metal.
//!
//! Dense matmul uses tiled GPU kernels. Quantized matmul uses GPU GEMV
//! kernels for M=1 (autoregressive decode) and falls back to CPU-side
//! dequant loops for M>1 (prefill).
//!
//! Quantized weights (Q8_0, Q4_0, Q4_1, Q6_K) are stored in Metal
//! buffers in their original `(out_features, in_features)` layout.

use std::sync::Arc;

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

/// Tile size for the SIMD-group matmul kernel (8×8 per SIMD-group).
const TILE: u64 = 8;

/// Threads per threadgroup for SIMD-group matmul: one SIMD-group = 32 threads.
const SIMD_GROUP_SIZE: u64 = 32;

/// Output rows per SIMD-group for quantized GEMV v2 (dot-product based).
const GEMV_V2_NR: u64 = 4;

/// SIMD-groups per threadgroup for GEMV v2.
const GEMV_V2_NSG: u64 = 2;

/// Total rows per threadgroup for v2.
const GEMV_V2_ROWS_PER_TG: u64 = GEMV_V2_NR * GEMV_V2_NSG;

/// Threads per threadgroup for v2.
const GEMV_V2_THREADS_PER_TG: u64 = GEMV_V2_NSG * SIMD_GROUP_SIZE;

/// Packed params for the tiled matmul kernel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulParams {
    m: u32,
    n: u32,
    k: u32,
}

/// Packed params for quantized GEMV kernels — must match MSL struct.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuantizedLinearParams {
    n: u32,
    k: u32,
}

// ---------------------------------------------------------------------------
// Quantized linear
// ---------------------------------------------------------------------------

/// Quantized linear: `input (M, K) × weight (N, K)_quantized → output (M, N)`.
///
/// For M=1: dispatches GPU GEMV kernels (one thread per output neuron).
/// For M>1: CPU fallback reading from Metal buffers via unified memory.
#[allow(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
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

    // --- GPU GEMV path (M=1) ---
    if m == 1 {
        let ctx = input.context();
        let mut out_shape = i_shape[..i_shape.len() - 1].to_vec();
        out_shape.push(n);
        let out_dtype = input.dtype(); // match input dtype (F16 or F32)
        let out = MetalTensor::zeros(ctx, &out_shape, out_dtype);
        let use_f16 = input.dtype() == DType::F16;

        let params = QuantizedLinearParams {
            n: n as u32,
            k: k as u32,
        };

        match weight.dtype {
            DType::Q8_0 => {
                let kernel = if use_f16 {
                    "gemv_q8_simd_v2_f16"
                } else {
                    "gemv_q8_simd_v2_f32"
                };
                let threadgroups = MTLSize::new((n as u64).div_ceil(GEMV_V2_ROWS_PER_TG), 1, 1);
                let threads_per_group = MTLSize::new(GEMV_V2_THREADS_PER_TG, 1, 1);
                ctx.dispatch_threadgroups(
                    kernel,
                    &[
                        (input.metal_buffer(), input.buffer_offset()),
                        (&weight.data, 0),
                        (&weight.scales, 0),
                        (out.metal_buffer(), out.buffer_offset()),
                    ],
                    bytemuck::bytes_of(&params),
                    threadgroups,
                    threads_per_group,
                    0,
                );
                return Ok(out);
            }
            DType::Q4_0 => {
                let kernel = if use_f16 {
                    "gemv_q4_simd_v2_f16"
                } else {
                    "gemv_q4_simd_v2_f32"
                };
                let threadgroups = MTLSize::new((n as u64).div_ceil(GEMV_V2_ROWS_PER_TG), 1, 1);
                let threads_per_group = MTLSize::new(GEMV_V2_THREADS_PER_TG, 1, 1);
                ctx.dispatch_threadgroups(
                    kernel,
                    &[
                        (input.metal_buffer(), input.buffer_offset()),
                        (&weight.data, 0),
                        (&weight.scales, 0),
                        (out.metal_buffer(), out.buffer_offset()),
                    ],
                    bytemuck::bytes_of(&params),
                    threadgroups,
                    threads_per_group,
                    0,
                );
                return Ok(out);
            }
            DType::Q4_1 => {
                let mins_buf = weight.mins.as_ref().expect("Q4_1 missing mins");
                ctx.dispatch_1d(
                    "gemv_q4_1_f32",
                    &[
                        (input.metal_buffer(), input.buffer_offset()),
                        (&weight.data, 0),
                        (&weight.scales, 0),
                        (mins_buf, 0),
                        (out.metal_buffer(), out.buffer_offset()),
                    ],
                    bytemuck::bytes_of(&params),
                    n,
                );
                return Ok(out);
            }
            DType::Q6_K => {
                ctx.dispatch_1d(
                    "gemv_q6k_f32",
                    &[
                        (input.metal_buffer(), input.buffer_offset()),
                        (&weight.data, 0),
                        (&weight.scales, 0),
                        (out.metal_buffer(), out.buffer_offset()),
                    ],
                    bytemuck::bytes_of(&params),
                    n,
                );
                return Ok(out);
            }
            _ => {} // fall through to CPU
        }
    }

    // --- CPU fallback (M>1 or unsupported dtype) ---
    let input_data = input.as_f32_slice();
    let mut output = vec![0.0f32; m * n];

    match weight.dtype {
        DType::Q8_0 => {
            let num_blocks_per_row = k / QUANTIZATION_BLOCK_SIZE;
            let quants = weight.data_bytes();
            let scales = weight.scales_f32();
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
            let packed = weight.data_bytes();
            let scales = weight.scales_f32();
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
            let packed = weight.data_bytes();
            let scales = weight.scales_f32();
            let mins = weight.mins_f32().expect("Q4_1 weight missing mins buffer");
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
            let raw = weight.data_bytes();
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
// Helper: create a Metal buffer from raw bytes
// ---------------------------------------------------------------------------

/// Create a Metal buffer from a byte slice. Returns a 4-byte placeholder
/// if the input is empty (Metal requires non-zero buffer sizes).
fn new_metal_buffer(device: &MetalContext, data: &[u8]) -> Arc<metal::Buffer> {
    if data.is_empty() {
        Arc::new(
            device
                .device()
                .new_buffer(4, metal::MTLResourceOptions::StorageModeShared),
        )
    } else {
        Arc::new(device.device().new_buffer_with_data(
            data.as_ptr().cast(),
            data.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ))
    }
}

/// Create a Metal buffer from an f32 slice.
fn new_metal_buffer_f32(device: &MetalContext, data: &[f32]) -> Arc<metal::Buffer> {
    new_metal_buffer(device, bytemuck::cast_slice(data))
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
        let threads_per_group = MTLSize::new(SIMD_GROUP_SIZE, 1, 1);

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
                let threads_per_group = MTLSize::new(SIMD_GROUP_SIZE, 1, 1);

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

    fn try_concat_linear_rows(
        a: &MetalLinearWeight,
        b: &MetalLinearWeight,
    ) -> Option<MetalLinearWeight> {
        match (a, b) {
            (MetalLinearWeight::Quantized(qa), MetalLinearWeight::Quantized(qb))
                if qa.dtype == qb.dtype
                    && qa.shape[1] == qb.shape[1]
                    && qa.mins.is_none()
                    && qb.mins.is_none() =>
            {
                Some(MetalLinearWeight::Quantized(
                    MetalQuantizedWeight::concat_rows(qa, qb),
                ))
            }
            _ => None,
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn quantize_to_q8(
        device: &MetalContext,
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
            ctx: device.clone(),
            data: new_metal_buffer(device, &qdata),
            scales: new_metal_buffer_f32(device, &scales),
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
                // Cast to f32 if needed, then use new_dense() which pre-computes
                // the transposed layout for efficient matmul
                let f32_tensor = if tensor.dtype() == DType::F32 {
                    tensor
                } else {
                    use infernum::backend::CastOps;
                    MetalBackend::cast_to_f32(&tensor)?
                };
                Ok(MetalLinearWeight::new_dense(f32_tensor))
            }
            HostLinearWeight::Quantized(hq) => match hq.dtype {
                DType::Q8_0 | DType::Q4_0 => {
                    let scales_f32 = decode_f16_scales(&hq.scales);
                    Ok(MetalLinearWeight::Quantized(MetalQuantizedWeight {
                        shape: hq.shape.clone(),
                        dtype: hq.dtype,
                        ctx: device.clone(),
                        data: new_metal_buffer(device, &hq.data),
                        scales: new_metal_buffer_f32(device, &scales_f32),
                        mins: None,
                    }))
                }
                DType::Q4_1 => {
                    let scales_f32 = decode_f16_scales(&hq.scales);
                    let mins_f32 = hq.qzeros.as_deref().map(decode_f16_scales);
                    Ok(MetalLinearWeight::Quantized(MetalQuantizedWeight {
                        shape: hq.shape.clone(),
                        dtype: hq.dtype,
                        ctx: device.clone(),
                        data: new_metal_buffer(device, &hq.data),
                        scales: new_metal_buffer_f32(device, &scales_f32),
                        mins: mins_f32.map(|m| new_metal_buffer_f32(device, &m)),
                    }))
                }
                DType::Q6_K => Ok(MetalLinearWeight::Quantized(MetalQuantizedWeight {
                    shape: hq.shape.clone(),
                    dtype: hq.dtype,
                    ctx: device.clone(),
                    data: new_metal_buffer(device, &hq.data),
                    scales: new_metal_buffer(device, &[]),
                    mins: None,
                })),
                other => Err(infernum::Error::UnsupportedDtype(format!(
                    "Metal backend does not support {other} quantized weights"
                ))),
            },
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn swiglu_linear(
        gate_up: &MetalTensor,
        intermediate_size: usize,
        down_proj: &MetalLinearWeight,
    ) -> Option<Result<MetalTensor>> {
        let MetalLinearWeight::Quantized(w) = down_proj else {
            return None;
        };
        // Only support Q8_0/Q4_0 with f16 input (quantized decode pipeline)
        if gate_up.dtype() != DType::F16 || !matches!(w.dtype, DType::Q8_0 | DType::Q4_0) {
            return None;
        }
        let m = gate_up.numel() / (2 * intermediate_size);
        if m != 1 {
            return None;
        }

        let n = w.shape[0]; // out_features (hidden_size)
        let ctx = gate_up.context();
        let out = MetalTensor::zeros(ctx, &[1, n], DType::F16);

        let params = QuantizedLinearParams {
            n: n as u32,
            k: intermediate_size as u32,
        };

        let kernel = match w.dtype {
            DType::Q8_0 => "gemv_swiglu_q8_simd_v2_f16",
            DType::Q4_0 => "gemv_swiglu_q4_simd_v2_f16",
            _ => unreachable!(),
        };

        let threadgroups = MTLSize::new((n as u64).div_ceil(GEMV_V2_ROWS_PER_TG), 1, 1);
        let threads_per_group = MTLSize::new(GEMV_V2_THREADS_PER_TG, 1, 1);
        ctx.dispatch_threadgroups(
            kernel,
            &[
                (gate_up.metal_buffer(), gate_up.buffer_offset()),
                (&w.data, 0),
                (&w.scales, 0),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            bytemuck::bytes_of(&params),
            threadgroups,
            threads_per_group,
            0,
        );

        Some(Ok(out))
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

    /// Create a Metal buffer from a byte slice (or a 4-byte placeholder if empty).
    fn make_metal_buffer(ctx: &MetalContext, data: &[u8]) -> Arc<metal::Buffer> {
        new_metal_buffer(ctx, data)
    }

    /// Create a Metal buffer from an f32 slice.
    fn make_metal_buffer_f32(ctx: &MetalContext, data: &[f32]) -> Arc<metal::Buffer> {
        new_metal_buffer_f32(ctx, data)
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

    // ---------------------------------------------------------------
    // Q8_0 tests
    // ---------------------------------------------------------------

    #[test]
    fn test_linear_q8() {
        let c = ctx();
        let n = 2;
        let k = QUANTIZATION_BLOCK_SIZE; // 32

        let mut qdata = vec![0u8; n * k];
        for j in 0..k {
            qdata[j] = 1_i8 as u8;
        }
        for j in 0..k {
            qdata[k + j] = 2_i8 as u8;
        }
        let scales = vec![2.0_f32, 0.5_f32];

        let weight = MetalLinearWeight::Quantized(MetalQuantizedWeight {
            shape: vec![n, k],
            dtype: DType::Q8_0,
            ctx: c.clone(),
            data: make_metal_buffer(&c, &qdata),
            scales: make_metal_buffer_f32(&c, &scales),
            mins: None,
        });

        let input_data = vec![1.0f32; k];
        let input = MetalBackend::from_f32_slice(&c, &[1, k], &input_data).unwrap();

        let out = MetalBackend::linear(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[1, n]);
        let result = out.as_f32_slice();
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
    fn test_gemv_q8_gpu_multi_neuron() {
        let c = ctx();
        let n = 4;
        let k = 64; // 2 blocks each
        let mut qdata = vec![0u8; n * k];
        // neuron 0: all quants = 1
        for j in 0..k {
            qdata[j] = 1_i8 as u8;
        }
        // neuron 1: all quants = -1
        for j in 0..k {
            qdata[k + j] = (-1_i8) as u8;
        }
        // neuron 2: all quants = 3
        for j in 0..k {
            qdata[2 * k + j] = 3_i8 as u8;
        }
        // neuron 3: all quants = 0
        for j in 0..k {
            qdata[3 * k + j] = 0;
        }
        let scales_vals: Vec<f32> = vec![1.0, 2.0, 0.5, 1.0, 0.25, 0.5, 3.0, 0.1];

        let weight = MetalLinearWeight::Quantized(MetalQuantizedWeight {
            shape: vec![n, k],
            dtype: DType::Q8_0,
            ctx: c.clone(),
            data: make_metal_buffer(&c, &qdata),
            scales: make_metal_buffer_f32(&c, &scales_vals),
            mins: None,
        });

        let input = MetalBackend::from_f32_slice(&c, &[1, k], &vec![1.0f32; k]).unwrap();
        let out = MetalBackend::linear(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[1, n]);
        let result = out.as_f32_slice();
        // neuron 0: 32*1*1.0 + 32*1*2.0 = 96
        assert!((result[0] - 96.0).abs() < 1e-3, "n0: {} vs 96", result[0]);
        // neuron 1: 32*(-1)*0.5 + 32*(-1)*1.0 = -48
        assert!(
            (result[1] - (-48.0)).abs() < 1e-3,
            "n1: {} vs -48",
            result[1]
        );
        // neuron 2: 32*3*0.25 + 32*3*0.5 = 72
        assert!((result[2] - 72.0).abs() < 1e-3, "n2: {} vs 72", result[2]);
        // neuron 3: all zeros
        assert!(result[3].abs() < 1e-3, "n3: {} vs 0", result[3]);
    }

    #[test]
    fn test_linear_q8_batched_cpu_fallback() {
        // M=2 forces CPU fallback
        let c = ctx();
        let n = 1;
        let k = QUANTIZATION_BLOCK_SIZE;
        let qdata: Vec<u8> = vec![1_i8 as u8; k];
        let scales = vec![2.0_f32];

        let weight = MetalLinearWeight::Quantized(MetalQuantizedWeight {
            shape: vec![n, k],
            dtype: DType::Q8_0,
            ctx: c.clone(),
            data: make_metal_buffer(&c, &qdata),
            scales: make_metal_buffer_f32(&c, &scales),
            mins: None,
        });

        let input = MetalBackend::from_f32_slice(&c, &[2, k], &vec![1.0f32; 2 * k]).unwrap();
        let out = MetalBackend::linear(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[2, n]);
        let result = out.as_f32_slice();
        // Both rows: 32 * 1 * 2.0 = 64.0
        assert!((result[0] - 64.0).abs() < 1e-3, "row 0: {}", result[0]);
        assert!((result[1] - 64.0).abs() < 1e-3, "row 1: {}", result[1]);
    }

    // ---------------------------------------------------------------
    // Q4_0 tests
    // ---------------------------------------------------------------

    #[test]
    fn test_linear_q4_0() {
        let c = ctx();
        let k = QUANTIZATION_BLOCK_SIZE;
        let packed: Vec<u8> = vec![0x53; k / 2]; // lo=3, hi=5
        let scale = 2.0_f32;

        let weight = MetalLinearWeight::Quantized(MetalQuantizedWeight {
            shape: vec![1, k],
            dtype: DType::Q4_0,
            ctx: c.clone(),
            data: make_metal_buffer(&c, &packed),
            scales: make_metal_buffer_f32(&c, &[scale]),
            mins: None,
        });

        let input_data = vec![1.0f32; k];
        let input = MetalBackend::from_f32_slice(&c, &[1, k], &input_data).unwrap();

        let out = MetalBackend::linear(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[1, 1]);
        let result = out.as_f32_slice();
        // 16 lo: (-5)*2.0 = -160, 16 hi: (-3)*2.0 = -96, total = -256
        assert!(
            (result[0] - (-256.0)).abs() < 1e-3,
            "Q4_0: got {}, expected -256.0",
            result[0]
        );
    }

    #[test]
    fn test_gemv_q4_gpu_multi_block() {
        let c = ctx();
        let k = 64; // 2 blocks
        let n = 2;
        let packed: Vec<u8> = vec![0x53; n * k / 2]; // lo=3, hi=5
        let scales = vec![2.0_f32, 1.0, 0.5, 3.0]; // 2 blocks per neuron

        let weight = MetalLinearWeight::Quantized(MetalQuantizedWeight {
            shape: vec![n, k],
            dtype: DType::Q4_0,
            ctx: c.clone(),
            data: make_metal_buffer(&c, &packed),
            scales: make_metal_buffer_f32(&c, &scales),
            mins: None,
        });

        let input = MetalBackend::from_f32_slice(&c, &[1, k], &vec![1.0f32; k]).unwrap();
        let out = MetalBackend::linear(&input, &weight).unwrap();
        let result = out.as_f32_slice();
        // per block with input=1: 16*(-5)*s + 16*(-3)*s = -128*s
        // n0: -128*2.0 + -128*1.0 = -384
        assert!(
            (result[0] - (-384.0)).abs() < 1e-3,
            "n0: {} vs -384",
            result[0]
        );
        // n1: -128*0.5 + -128*3.0 = -64 + -384 = -448
        assert!(
            (result[1] - (-448.0)).abs() < 1e-3,
            "n1: {} vs -448",
            result[1]
        );
    }

    // ---------------------------------------------------------------
    // Q4_1 tests
    // ---------------------------------------------------------------

    #[test]
    fn test_linear_q4_1() {
        let c = ctx();
        let k = QUANTIZATION_BLOCK_SIZE;
        let packed: Vec<u8> = vec![0x53; k / 2]; // lo=3, hi=5
        let scale = 0.5_f32;
        let min = -1.0_f32;

        let weight = MetalLinearWeight::Quantized(MetalQuantizedWeight {
            shape: vec![1, k],
            dtype: DType::Q4_1,
            ctx: c.clone(),
            data: make_metal_buffer(&c, &packed),
            scales: make_metal_buffer_f32(&c, &[scale]),
            mins: Some(make_metal_buffer_f32(&c, &[min])),
        });

        let input_data = vec![1.0f32; k];
        let input = MetalBackend::from_f32_slice(&c, &[1, k], &input_data).unwrap();

        let out = MetalBackend::linear(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[1, 1]);
        let result = out.as_f32_slice();
        // 16 lo: (3*0.5 + -1.0) = 0.5 → 8.0
        // 16 hi: (5*0.5 + -1.0) = 1.5 → 24.0
        // Total: 32.0
        assert!(
            (result[0] - 32.0).abs() < 1e-3,
            "Q4_1: got {}, expected 32.0",
            result[0]
        );
    }

    // ---------------------------------------------------------------
    // Q6_K tests
    // ---------------------------------------------------------------

    /// Build a Q6_K super-block (210 bytes) and dequantize all 256 elements to f32.
    ///
    /// Returns `(raw_bytes, dequantized_f32)`.
    fn make_q6k_superblock(d: f32, sub_scales: &[i8; 16]) -> (Vec<u8>, Vec<f32>) {
        let mut raw = vec![0u8; Q6_K_BLOCK_SIZE_BYTES]; // 210 bytes

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

        // Dequantize: q = (ql_val | (qh_val << 4)) - 32 = 5 - 32 = -27
        let mut dequant = Vec::with_capacity(Q6_K_BLOCK_ELEMENTS);
        let d_f32 = d_f16.to_f32();
        for elem in 0..Q6_K_BLOCK_ELEMENTS {
            let sub_block = elem / 16;
            let sc = f32::from(sub_scales[sub_block]);
            let q: f32 = -27.0;
            dequant.push(d_f32 * sc * q);
        }

        (raw, dequant)
    }

    #[test]
    fn test_linear_q6k() {
        let c = ctx();
        let k = Q6_K_BLOCK_ELEMENTS; // 256
        let n = 1;

        let sub_scales: [i8; 16] = [1; 16];
        let (raw, dequant) = make_q6k_superblock(0.5, &sub_scales);

        let weight = MetalLinearWeight::Quantized(MetalQuantizedWeight {
            shape: vec![n, k],
            dtype: DType::Q6_K,
            ctx: c.clone(),
            data: make_metal_buffer(&c, &raw),
            scales: make_metal_buffer(&c, &[]),
            mins: None,
        });

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
        let n = 2;

        let scales_0: [i8; 16] = [2; 16];
        let (raw_0, dequant_0) = make_q6k_superblock(1.0, &scales_0);

        let scales_1: [i8; 16] = [-1; 16];
        let (raw_1, dequant_1) = make_q6k_superblock(0.25, &scales_1);

        let mut data = raw_0;
        data.extend_from_slice(&raw_1);

        let weight = MetalLinearWeight::Quantized(MetalQuantizedWeight {
            shape: vec![n, k],
            dtype: DType::Q6_K,
            ctx: c.clone(),
            data: make_metal_buffer(&c, &data),
            scales: make_metal_buffer(&c, &[]),
            mins: None,
        });

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

    // ---------------------------------------------------------------
    // quantize_to_q8 and upload_host tests
    // ---------------------------------------------------------------

    #[test]
    fn test_quantize_to_q8_roundtrip() {
        let c = ctx();
        let shape = [2, QUANTIZATION_BLOCK_SIZE];
        let mut data = vec![0.0f32; 2 * QUANTIZATION_BLOCK_SIZE];
        for (i, v) in data.iter_mut().enumerate() {
            *v = (i as f32) * 0.1 - 3.2;
        }

        let weight = MetalBackend::quantize_to_q8(&c, &shape, &data).unwrap();
        assert!(!MetalBackend::is_dense_weight(&weight));

        let input_data = vec![1.0f32; QUANTIZATION_BLOCK_SIZE];
        let input =
            MetalBackend::from_f32_slice(&c, &[1, QUANTIZATION_BLOCK_SIZE], &input_data).unwrap();
        let out = MetalBackend::linear(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
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

        if let MetalLinearWeight::Quantized(q) = &weight {
            assert_eq!(q.dtype, DType::Q8_0);
            assert_eq!(q.shape, vec![n, k]);
            assert_eq!(q.data_bytes(), &qdata);
            assert!((q.scales_f32()[0] - 1.0).abs() < 1e-3);
        } else {
            panic!("Expected Quantized variant");
        }
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
            assert_eq!(q.data_bytes(), &raw);
            assert!(q.scales_f32().is_empty() || q.scales_f32().len() <= 1);
            assert!(q.mins.is_none());
        } else {
            panic!("Expected Quantized variant");
        }
    }
}
