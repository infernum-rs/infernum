//! Quantized matrix multiplication kernels
//!
//! Performs `output = input @ dequantize(weight)^T` where weights are stored
//! in `Q8_0`, `Q4_0`, `Q6_K`, `F8E4M3`, `GPTQ_INT4`, or `AWQ_INT4` format.
//! Input activations and output remain f32. Dequantization happens on-the-fly
//! inside the kernel — the full f32 weight matrix never exists in GPU memory.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::missing_panics_doc,
    clippy::manual_div_ceil
)]

use cudarc::cublaslt::MatmulShared;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceSlice, LaunchAsync, LaunchConfig};

use crate::cuda::buffer_pool::PooledSlice;
use crate::cuda::quantized::QuantizedTensor;
use crate::cuda::CudaContext;
use crate::cuda::CudaTensor;
use crate::dtype::{DType, Q6_K_BLOCK_ELEMENTS};
use crate::tensor::Tensor;
use crate::Result;

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

const QUANTIZED_MATMUL_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/kernels/quantized_matmul.ptx"));

const GPTQ_MATMUL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/gptq_matmul.ptx"));

const AWQ_MATMUL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/awq_matmul.ptx"));

// ---------------------------------------------------------------------------
// FP8 activation quantization kernel (compiled at build time)
// ---------------------------------------------------------------------------

const FP8_QUANTIZE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/fp8_quantize.ptx"));

const SCALE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/scale.ptx"));

// ---------------------------------------------------------------------------
// FP8 cuBLAS helpers
// ---------------------------------------------------------------------------

/// Quantize f32 activations to FP8 E4M3 on-the-fly, returning the
/// quantized buffer and a device-side inverse scale factor (for cuBLASLt).
///
/// Uses a single fused kernel that computes absmax and quantizes in one launch,
/// eliminating the overhead of a separate absmax kernel call.
///
/// Everything runs on the GPU — no CPU↔GPU synchronization.
fn quantize_activations_to_fp8(
    ctx: &CudaContext,
    input: &CudaTensor<f32>,
    numel: usize,
) -> Result<(CudaSlice<u8>, CudaSlice<f32>)> {
    let device = ctx.device();

    let module_name = "fp8_quantize";
    if !device.has_func(module_name, "fused_absmax_quantize_f32") {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(FP8_QUANTIZE_PTX),
            module_name,
            &[
                "absmax_f32",
                "quantize_f32_to_fp8",
                "fused_absmax_quantize_f32",
            ],
        )?;
    }

    let mut max_bits = device.alloc_zeros::<u32>(1)?;
    let mut block_counter = device.alloc_zeros::<u32>(1)?;
    let mut act_fp8 = unsafe { device.alloc::<u8>(numel)? };
    let mut d_inv_scale = unsafe { device.alloc::<f32>(1)? };

    let threads = 256;
    let blocks = ((numel + threads - 1) / threads).min(1024);
    let num_warps = threads / 32;
    let shared_mem = num_warps * std::mem::size_of::<f32>();

    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    let func = device
        .get_func(module_name, "fused_absmax_quantize_f32")
        .unwrap();

    unsafe {
        func.launch(
            cfg,
            (
                &input.cuda_slice(),
                &mut act_fp8,
                &mut max_bits,
                &mut d_inv_scale,
                &mut block_counter,
                numel as i32,
            ),
        )?;
    }

    Ok((act_fp8, d_inv_scale))
}

/// Execute the cuBLASLt FP8 GEMM: `output(M,N) = act(M,K) @ weight(N,K)^T`.
///
/// Both `act_fp8` and weight data are FP8 E4M3 bytes. Output is f32.
/// Uses the column-major trick (transpose A, swap operands) so that
/// row-major tensors are correctly multiplied.
///
/// Scale pointers are device-side — no CPU↔GPU synchronization.
#[allow(clippy::similar_names)]
fn execute_fp8_gemm(
    ctx: &CudaContext,
    act_fp8: &CudaSlice<u8>,
    weight: &QuantizedTensor,
    d_inv_scale_a: &CudaSlice<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaTensor<f32>> {
    // Resolve weight scale from the tensor
    let d_scale_b_owned;
    let d_scale_b = if let Some(s) = weight.d_weight_scale() {
        s
    } else {
        d_scale_b_owned = ctx.device().htod_sync_copy(&[weight.weight_scale()])?;
        &d_scale_b_owned
    };
    execute_fp8_gemm_with_scale(ctx, act_fp8, weight, d_inv_scale_a, d_scale_b, m, n, k)
}

/// Core cuBLASLt FP8 GEMM with an explicit weight scale buffer.
#[allow(clippy::similar_names, clippy::too_many_arguments)]
fn execute_fp8_gemm_with_scale(
    ctx: &CudaContext,
    act_fp8: &CudaSlice<u8>,
    weight: &QuantizedTensor,
    d_inv_scale_a: &CudaSlice<f32>,
    d_scale_b: &CudaSlice<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaTensor<f32>> {
    use cudarc::cublaslt::{result as lt_result, sys as lt_sys};

    let blas_lt = ctx.blas_lt();

    let fp8_type = lt_sys::cudaDataType_t::CUDA_R_8F_E4M3;
    let f32_type = lt_sys::cudaDataType_t::CUDA_R_32F;

    // Column-major trick: weight(N,K) row-major → (K,N) col-major; TRANSA=T → op(A)=(N,K)
    // input(M,K) row-major → (K,M) col-major; TRANSB=N → op(B)=(K,M)
    // D(N,M) col-major = op(A)(N,K) × op(B)(K,M), ld=N → row-major (M,N)
    let (m_gemm, n_gemm, k_gemm) = (n as u64, m as u64, k as u64);

    let a_layout = lt_result::create_matrix_layout(fp8_type, k_gemm, m_gemm, k_gemm as i64)?;
    let b_layout = lt_result::create_matrix_layout(fp8_type, k_gemm, n_gemm, k_gemm as i64)?;
    let d_layout = lt_result::create_matrix_layout(f32_type, m_gemm, n_gemm, m_gemm as i64)?;

    let matmul_desc =
        lt_result::create_matmul_desc(lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, f32_type)?;

    let transa = 1_i32; // CUBLAS_OP_T
    let a_scale_ptr = *d_scale_b.device_ptr();
    let b_scale_ptr = *d_inv_scale_a.device_ptr();

    unsafe {
        lt_result::set_matmul_desc_attribute(
            matmul_desc,
            lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            (&raw const transa).cast(),
            std::mem::size_of::<i32>(),
        )?;
        lt_result::set_matmul_desc_attribute(
            matmul_desc,
            lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            (&raw const a_scale_ptr).cast(),
            std::mem::size_of::<cudarc::driver::sys::CUdeviceptr>(),
        )?;
        lt_result::set_matmul_desc_attribute(
            matmul_desc,
            lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            (&raw const b_scale_ptr).cast(),
            std::mem::size_of::<cudarc::driver::sys::CUdeviceptr>(),
        )?;
    }

    // Workspace: use cached buffer from context
    let ws_guard = ctx.fp8_workspace()?;
    let ws_buf = ws_guard.as_ref().unwrap();
    let ws_size = ws_buf.len();

    let matmul_pref = lt_result::create_matmul_pref()?;
    unsafe {
        lt_result::set_matmul_pref_attribute(
            matmul_pref,
            lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&raw const ws_size).cast(),
            std::mem::size_of::<usize>(),
        )?;
    }

    let heuristic = unsafe {
        lt_result::get_matmul_algo_heuristic(
            *blas_lt.handle(),
            matmul_desc,
            a_layout,
            b_layout,
            d_layout,
            d_layout,
            matmul_pref,
        )?
    };

    let mut output = unsafe { CudaTensor::<f32>::uninit(ctx, &[m, n])? };
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    unsafe {
        lt_result::matmul(
            *blas_lt.handle(),
            matmul_desc,
            (&raw const alpha).cast(),
            (&raw const beta).cast(),
            *weight.data_slice().device_ptr() as *const _,
            a_layout,
            *act_fp8.device_ptr() as *const _,
            b_layout,
            *output.cuda_slice().device_ptr() as *const _,
            d_layout,
            *output.cuda_slice_mut().device_ptr_mut() as *mut _,
            d_layout,
            &raw const heuristic.algo,
            *ws_buf.device_ptr() as *mut std::ffi::c_void,
            ws_size,
            (*blas_lt.stream()).cast(),
        )?;
    }

    unsafe {
        lt_result::destroy_matrix_layout(a_layout)?;
        lt_result::destroy_matrix_layout(b_layout)?;
        lt_result::destroy_matrix_layout(d_layout)?;
        lt_result::destroy_matmul_desc(matmul_desc)?;
        lt_result::destroy_matmul_pref(matmul_pref)?;
    }

    Ok(output)
}

/// FP8 matmul using cuBLASLt: `output = input @ dequant(weight)^T`
///
/// Quantizes f32 activations to FP8 E4M3 on-the-fly, then invokes
/// cuBLASLt with FP8 tensor cores. Output is f32.
/// No CPU↔GPU synchronization on the hot path.
fn quantized_matmul_fp8_cublas(
    ctx: &CudaContext,
    input: &CudaTensor<f32>,
    weight: &QuantizedTensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaTensor<f32>> {
    let (act_fp8, d_inv_scale_a) = quantize_activations_to_fp8(ctx, input, m * k)?;

    if let Some(channel_scales) = weight.d_channel_scales() {
        // Per-channel scales: run GEMM with weight_scale=1.0, then post-multiply
        let d_one = ctx.device().htod_sync_copy(&[1.0_f32])?;
        let mut output =
            execute_fp8_gemm_with_scale(ctx, &act_fp8, weight, &d_inv_scale_a, &d_one, m, n, k)?;

        // Apply per-channel scales: output[m, n] *= channel_scales[n]
        let total = m * n;
        let device = ctx.device();
        let module_name = "scale";
        if !device.has_func(module_name, "scale_rows_f32") {
            device.load_ptx(
                cudarc::nvrtc::Ptx::from_src(SCALE_PTX),
                module_name,
                &["scale_f32", "scale_rows_f32"],
            )?;
        }
        let func = device.get_func(module_name, "scale_rows_f32").unwrap();
        let cfg = LaunchConfig::for_num_elems(total as u32);
        unsafe {
            func.launch(
                cfg,
                (
                    output.cuda_slice_mut(),
                    channel_scales,
                    n as i32,
                    total as i32,
                ),
            )?;
        }
        Ok(output)
    } else {
        execute_fp8_gemm(ctx, &act_fp8, weight, &d_inv_scale_a, m, n, k)
    }
}

// ---------------------------------------------------------------------------
// GEMV: M=1 decode fast path for quantized formats
// ---------------------------------------------------------------------------

/// Quantize f32 activations to `Q8_1` format on the GPU.
///
/// Returns `(data [K] as i8, scales [K/32] as f32, sums [K/32] as f32)`.
/// Each 32-element block gets: `scale = absmax / 127`, `qi = round(v / scale)`,
/// `sum = scale * Σqi`.
///
/// Scratch buffers are pool-backed when the context has a buffer pool enabled,
/// making this function safe to call during CUDA graph capture (after a warmup
/// step has populated the pool).
fn quantize_activations_to_q8_1(
    ctx: &CudaContext,
    input: &CudaTensor<f32>,
    k: usize,
) -> Result<(PooledSlice<i8>, PooledSlice<f32>, PooledSlice<f32>)> {
    assert_eq!(k % 32, 0, "K must be divisible by 32 for Q8_1 quantization");

    let device = ctx.device();
    let num_blocks = k / 32;

    let mut act_data = unsafe { ctx.pool_alloc::<i8>(k)? };
    let mut act_scales = unsafe { ctx.pool_alloc::<f32>(num_blocks)? };
    let mut act_sums = unsafe { ctx.pool_alloc::<f32>(num_blocks)? };

    let module_name = "quantized_matmul";
    if !device.has_func(module_name, "quantize_f32_to_q8_1") {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(QUANTIZED_MATMUL_PTX),
            module_name,
            &[
                "matmul_q8_f32",
                "matmul_q4_f32",
                "matmul_fp8_f32",
                "matmul_q6k_f32",
                "gemv_q8_f32",
                "gemv_q4_f32",
                "quantize_f32_to_q8_1",
                "gemv_q8_q8_dp4a",
                "gemv_q4_q8_dp4a",
            ],
        )?;
    }
    let func = device
        .get_func(module_name, "quantize_f32_to_q8_1")
        .unwrap();

    let cfg = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(
            cfg,
            (
                &mut *act_data,
                &mut *act_scales,
                &mut *act_sums,
                &input.cuda_slice(),
                k as i32,
            ),
        )?;
    }

    Ok((act_data, act_scales, act_sums))
}

/// GEMV kernel for M=1 decode: `output[1, N] = input[1, K] @ dequant(weight[N, K])^T`
///
/// Quantizes f32 activations to `Q8_1` on-the-fly, then uses `dp4a` integer dot
/// products for ~4× compute throughput and ~4× bandwidth reduction vs f32.
fn quantized_gemv(
    input: &CudaTensor<f32>,
    weight: &QuantizedTensor,
    n: usize,
    k: usize,
) -> Result<CudaTensor<f32>> {
    // Multi-warp GEMV: NWARPS warps per output row, K-split across all threads
    const NWARPS: u32 = 4;

    let ctx = input.context();
    let device = ctx.device();

    // Quantize activations: f32 [K] → Q8_1 (int8 data + f32 scales + f32 sums)
    let (act_data, act_scales, act_sums) = quantize_activations_to_q8_1(ctx, input, k)?;

    let mut output = unsafe { CudaTensor::<f32>::uninit(ctx, &[1, n])? };

    let module_name = "quantized_matmul";

    let cfg = LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        block_dim: (32, NWARPS, 1),
        shared_mem_bytes: NWARPS * 4, // NWARPS × sizeof(f32) for inter-warp reduction
    };

    match weight.dtype() {
        DType::Q8_0 => {
            let func = device.get_func(module_name, "gemv_q8_q8_dp4a").unwrap();
            unsafe {
                func.launch(
                    cfg,
                    (
                        output.cuda_slice_mut(),
                        &*act_data,
                        &*act_scales,
                        weight.data_slice(),
                        weight.scales_slice(),
                        n as i32,
                        k as i32,
                    ),
                )?;
            }
        }
        DType::Q4_0 => {
            let func = device.get_func(module_name, "gemv_q4_q8_dp4a").unwrap();
            unsafe {
                func.launch(
                    cfg,
                    (
                        output.cuda_slice_mut(),
                        &*act_data,
                        &*act_scales,
                        &*act_sums,
                        weight.data_slice(),
                        weight.scales_slice(),
                        n as i32,
                        k as i32,
                    ),
                )?;
            }
        }
        other => panic!("quantized_gemv: unsupported dtype {other}"),
    }

    Ok(output)
}

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

#[allow(clippy::too_many_lines)]
fn quantized_matmul_2d(
    input: &CudaTensor<f32>,
    weight: &QuantizedTensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaTensor<f32>> {
    let device = input.context().device();

    let module_name = "quantized_matmul";
    if !device.has_func(module_name, "matmul_q8_f32") {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(QUANTIZED_MATMUL_PTX),
            module_name,
            &[
                "matmul_q8_f32",
                "matmul_q4_f32",
                "matmul_fp8_f32",
                "matmul_q6k_f32",
                "gemv_q8_f32",
                "gemv_q4_f32",
                "quantize_f32_to_q8_1",
                "gemv_q8_q8_dp4a",
                "gemv_q4_q8_dp4a",
            ],
        )?;
    }

    // GEMV fast path for M=1 decode (Q8_0 and Q4_0 only)
    if m == 1 && matches!(weight.dtype(), DType::Q8_0 | DType::Q4_0) {
        return quantized_gemv(input, weight, n, k);
    }

    let mut output = unsafe { CudaTensor::<f32>::uninit(input.context(), &[m, n])? };

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
                        &input.cuda_slice(),
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
                        &input.cuda_slice(),
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
            // cuBLASLt handles padding for small M (including M=1 decode)
            if input.context().supports_fp8_tensor_cores() && n >= 16 && k >= 16 {
                return quantized_matmul_fp8_cublas(input.context(), input, weight, m, n, k);
            }

            // Per-channel scales: use stored buffer or broadcast scalar
            let scales_owned;
            let scales_slice = if let Some(cs) = weight.d_channel_scales() {
                cs
            } else {
                scales_owned = device.htod_sync_copy(&vec![weight.weight_scale(); n])?;
                &scales_owned
            };

            let func = device.get_func(module_name, "matmul_fp8_f32").unwrap();
            unsafe {
                func.launch(
                    cfg,
                    (
                        output.cuda_slice_mut(),
                        &input.cuda_slice(),
                        weight.data_slice(),
                        scales_slice,
                        m as i32,
                        n as i32,
                        k as i32,
                    ),
                )?;
            }
        }
        DType::Q6_K => {
            assert_eq!(
                k % Q6_K_BLOCK_ELEMENTS,
                0,
                "K ({k}) must be divisible by Q6_K block size ({Q6_K_BLOCK_ELEMENTS})"
            );
            let func = device.get_func(module_name, "matmul_q6k_f32").unwrap();
            unsafe {
                func.launch(
                    cfg,
                    (
                        output.cuda_slice_mut(),
                        &input.cuda_slice(),
                        weight.data_slice(),
                        m as i32,
                        n as i32,
                        k as i32,
                    ),
                )?;
            }
        }
        DType::GPTQ_INT4 => {
            let gptq_module = "gptq_matmul";
            if !device.has_func(gptq_module, "matmul_gptq_f32") {
                device.load_ptx(
                    cudarc::nvrtc::Ptx::from_src(GPTQ_MATMUL_PTX),
                    gptq_module,
                    &["matmul_gptq_f32"],
                )?;
            }
            let group_size = weight
                .group_size()
                .expect("GPTQ_INT4 weight must have group_size");
            let qzeros = weight
                .qzeros_slice()
                .expect("GPTQ_INT4 weight must have qzeros");
            let func = device.get_func(gptq_module, "matmul_gptq_f32").unwrap();
            unsafe {
                func.launch(
                    cfg,
                    (
                        output.cuda_slice_mut(),
                        &input.cuda_slice(),
                        weight.data_slice(),
                        weight.scales_slice(),
                        qzeros,
                        m as i32,
                        n as i32,
                        k as i32,
                        group_size as i32,
                    ),
                )?;
            }
        }
        DType::AWQ_INT4 => {
            let awq_module = "awq_matmul";
            if !device.has_func(awq_module, "matmul_awq_f32") {
                device.load_ptx(
                    cudarc::nvrtc::Ptx::from_src(AWQ_MATMUL_PTX),
                    awq_module,
                    &["matmul_awq_f32"],
                )?;
            }
            let group_size = weight
                .group_size()
                .expect("AWQ_INT4 weight must have group_size");
            let qzeros = weight
                .qzeros_slice()
                .expect("AWQ_INT4 weight must have qzeros");
            let func = device.get_func(awq_module, "matmul_awq_f32").unwrap();
            unsafe {
                func.launch(
                    cfg,
                    (
                        output.cuda_slice_mut(),
                        &input.cuda_slice(),
                        weight.data_slice(),
                        weight.scales_slice(),
                        qzeros,
                        m as i32,
                        n as i32,
                        k as i32,
                        group_size as i32,
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

/// Quantize an f32 slice to Q4_0 format on the host (GGML layout).
/// Returns (data, scales) as raw byte vectors.
/// GGML packing: byte[j] has element j in low nibble, element j+16 in high nibble.
#[cfg(test)]
fn quantize_q4_host(values: &[f32], block_size: usize) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(values.len() % block_size, 0);
    assert_eq!(block_size, 32, "Q4_0 requires block size of 32");
    let num_blocks = values.len() / block_size;
    let mut data = Vec::with_capacity(values.len() / 2);
    let mut scales = Vec::with_capacity(num_blocks * 2);

    for block in values.chunks(block_size) {
        let max_abs = block.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
        let scale_f16 = half::f16::from_f32(scale);
        scales.extend_from_slice(&scale_f16.to_le_bytes());

        // GGML packing: byte[j] has element j (low nibble) and element j+16 (high nibble)
        for j in 0..16 {
            let q_lo = ((block[j] / scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
            let q_hi = ((block[j + 16] / scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
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

/// Decode an FP8 E4M3 byte back to f32 (host-side, for test reference).
#[cfg(test)]
fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0_f32 } else { 1.0 };
    let exp = (byte >> 3) & 0x0F;
    let mant = byte & 0x07;

    if exp == 0 && mant == 0 {
        return 0.0;
    }

    let value = if exp == 0 {
        // Subnormal: 2^(-6) * (mant / 8)
        f32::from(mant) / 8.0 * (2.0_f32).powi(-6)
    } else {
        // Normal: 2^(exp - 7) * (1 + mant / 8)
        (2.0_f32).powi(i32::from(exp) - 7) * (1.0 + f32::from(mant) / 8.0)
    };

    sign * value
}

/// Quantize an f32 slice to Q6_K format on the host.
/// Returns packed super-block bytes (210 bytes per 256-element super-block).
/// Each element is quantized to 6 bits (range [-32, 31]) with per-16-element
/// sub-block i8 scales and a per-super-block f16 scale factor.
#[cfg(test)]
fn quantize_q6k_host(values: &[f32]) -> Vec<u8> {
    use crate::dtype::{Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_SIZE_BYTES};

    assert_eq!(values.len() % Q6_K_BLOCK_ELEMENTS, 0);
    let num_blocks = values.len() / Q6_K_BLOCK_ELEMENTS;
    let mut out = Vec::with_capacity(num_blocks * Q6_K_BLOCK_SIZE_BYTES);

    for block in values.chunks(Q6_K_BLOCK_ELEMENTS) {
        // Compute sub-block scales (16 sub-blocks of 16 elements each)
        // The sub-block scale is the max absolute value in the sub-block divided by 31
        // (since 6-bit centered values range from -32 to 31)
        let mut sub_scales = [0.0_f32; 16];
        for (sb, chunk) in block.chunks(16).enumerate() {
            let max_abs = chunk.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
            sub_scales[sb] = if max_abs == 0.0 { 0.0 } else { max_abs / 31.0 };
        }

        // Super-block scale d: chosen so that sc_i8 values fit in [-128, 127]
        // We want sc_i8 = round(sub_scale / d), so d = max(sub_scale) / 127
        let max_sub_scale = sub_scales.iter().fold(0.0_f32, |a, &b| a.max(b));
        let d = if max_sub_scale == 0.0 {
            1.0
        } else {
            max_sub_scale / 127.0
        };
        let inv_d = 1.0 / d;

        // Quantize sub-block scales to i8: sc_i8 = round(sub_scale / d)
        // Dequant formula: val = d * sc_i8 * q
        let mut sc_i8 = [0_i8; 16];
        for (i, &s) in sub_scales.iter().enumerate() {
            sc_i8[i] = (s * inv_d).round().clamp(-128.0, 127.0) as i8;
        }

        // Quantize each element to 6-bit [0, 63] (stored as unsigned, centered at 32)
        // Dequant: val = d * sc_i8 * (q - 32)
        // So: q = round(val / (d * sc_i8)) + 32
        let mut q6 = [0_u8; Q6_K_BLOCK_ELEMENTS];
        for (idx, &v) in block.iter().enumerate() {
            let sb = idx / 16;
            let effective_scale = d * (sc_i8[sb] as f32);
            let q = if effective_scale.abs() < 1e-10 {
                32 // zero value maps to center
            } else {
                ((v / effective_scale).round().clamp(-32.0, 31.0) as i32 + 32) as u8
            };
            q6[idx] = q;
        }

        // Pack into super-block: ql[128] | qh[64] | scales[16] | d(f16)
        // Uses ggml's interleaved layout (8x32 -> 16x16 reshape pattern)
        let mut ql = [0_u8; 128];
        let mut qh = [0_u8; 64];

        for elem in 0..Q6_K_BLOCK_ELEMENTS {
            let val = q6[elem];
            let lo4 = val & 0x0F;
            let hi2 = (val >> 4) & 0x03;

            // Map element index using ggml's interleaved layout
            // The data flows: (128,) -> (2,1,64) -> shift -> (2,2,64) -> (8,32) -> (16,16)
            let sb = elem / 16;
            let sb_elem = elem % 16;
            let flat_idx = sb * 16 + sb_elem;
            let row8 = flat_idx / 32; // 0-7
            let col32 = flat_idx % 32; // 0-31

            // ql layout after reshape to (8,32):
            // row 0: bytes 0-31 low nibbles    row 4: bytes 64-95 low nibbles
            // row 1: bytes 32-63 low nibbles   row 5: bytes 96-127 low nibbles
            // row 2: bytes 0-31 high nibbles   row 6: bytes 64-95 high nibbles
            // row 3: bytes 32-63 high nibbles  row 7: bytes 96-127 high nibbles
            let ql_half = row8 / 4; // 0 for rows 0-3, 1 for rows 4-7
            let ql_nibble_sel = (row8 % 4) / 2; // 0 for rows 0-1,4-5 (low), 1 for 2-3,6-7 (high)
            let ql_offset = (row8 % 4) % 2; // 0 for even rows in group, 1 for odd
            let ql_byte_idx = ql_half * 64 + ql_offset * 32 + col32;
            if ql_nibble_sel == 0 {
                ql[ql_byte_idx] |= lo4;
            } else {
                ql[ql_byte_idx] |= lo4 << 4;
            }

            // qh layout: (64,) -> (2,1,32) -> shift -> (2,4,32) -> (8,32)
            // row 0: bytes 0-31 bits 0-1   row 4: bytes 32-63 bits 0-1
            // row 1: bytes 0-31 bits 2-3   row 5: bytes 32-63 bits 2-3
            // row 2: bytes 0-31 bits 4-5   row 6: bytes 32-63 bits 4-5
            // row 3: bytes 0-31 bits 6-7   row 7: bytes 32-63 bits 6-7
            let qh_half = row8 / 4; // 0 or 1 (selects 32-byte half)
            let qh_shift_sel = row8 % 4; // 0,1,2,3 -> shift 0,2,4,6
            let qh_byte_idx = qh_half * 32 + col32;
            let qh_shift = qh_shift_sel * 2;
            qh[qh_byte_idx] |= hi2 << qh_shift;
        }

        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        for &s in &sc_i8 {
            out.push(s as u8);
        }
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
    }

    out
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

    #[test]
    fn test_matmul_q6k_basic() {
        use crate::dtype::Q6_K_BLOCK_ELEMENTS;

        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let k = Q6_K_BLOCK_ELEMENTS; // 256
        let m = 1;
        let n = 1;

        // input: all 1.0 → output = sum of dequantized weights
        let input_data = vec![1.0_f32; k];
        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();

        // weight: constant 2.0
        let w_data = vec![2.0_f32; k];
        let q_data = quantize_q6k_host(&w_data);
        let weight = QuantizedTensor::from_raw(&ctx, &[n, k], DType::Q6_K, &q_data, &[]).unwrap();

        let output = quantized_matmul(&input, &weight).unwrap();
        let result = output.to_vec().unwrap();

        // Expected: 256 * 2.0 = 512.0 (Q6_K has multi-level quantization so allow more error)
        let expected = 512.0_f32;
        assert!(
            (result[0] - expected).abs() < expected * 0.15,
            "Q6_K result: {} vs expected ~{}",
            result[0],
            expected
        );
    }

    #[test]
    fn test_matmul_q6k_matches_f32() {
        use crate::dtype::Q6_K_BLOCK_ELEMENTS;

        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let m = 2;
        let k = Q6_K_BLOCK_ELEMENTS; // 256
        let n = 4;

        // Pseudo-random input and weights
        let mut state: u64 = 54321;
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
        let q_data = quantize_q6k_host(&w_data);
        let weight = QuantizedTensor::from_raw(&ctx, &[n, k], DType::Q6_K, &q_data, &[]).unwrap();
        let output = quantized_matmul(&input, &weight).unwrap();
        let result = output.to_vec().unwrap();

        // Q6_K has multi-level quantization so allow more error than Q8
        for i in 0..m * n {
            let rel_err = if expected[i].abs() > 1e-6 {
                (result[i] - expected[i]).abs() / expected[i].abs()
            } else {
                (result[i] - expected[i]).abs()
            };
            assert!(
                rel_err < 0.30,
                "Q6_K mismatch at [{}, {}]: got {} expected {} (rel_err={:.4})",
                i / n,
                i % n,
                result[i],
                expected[i],
                rel_err
            );
        }
    }

    #[test]
    fn test_matmul_q4_matches_dequantized_reference() {
        // Compare quantized matmul against a reference computed with dequantized weights
        // This tests that the GPU kernel correctly implements dequantization.
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let m = 4;
        let k = 64;
        let n = 8;

        // Pseudo-random input and weights
        let mut state: u64 = 99999;
        let mut rand_f32 = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        let input_data: Vec<f32> = (0..m * k).map(|_| rand_f32()).collect();
        let w_data: Vec<f32> = (0..n * k).map(|_| rand_f32()).collect();

        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();

        // Quantize weights
        let (q_data, q_scales) = quantize_q4_host(&w_data, QUANTIZATION_BLOCK_SIZE);

        // Dequantize on CPU to get reference values
        let w_dequantized = dequantize_q4_host(&q_data, &q_scales, n * k);

        // Compute reference: input @ w_dequantized^T
        let mut expected = vec![0.0_f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0_f32;
                for i in 0..k {
                    acc += input_data[row * k + i] * w_dequantized[col * k + i];
                }
                expected[row * n + col] = acc;
            }
        }

        // Run quantized matmul
        let weight =
            QuantizedTensor::from_raw(&ctx, &[n, k], DType::Q4_0, &q_data, &q_scales).unwrap();
        let output = quantized_matmul(&input, &weight).unwrap();
        let result = output.to_vec().unwrap();

        // Should match closely (only f32 rounding differences)
        for i in 0..m * n {
            let diff = (result[i] - expected[i]).abs();
            assert!(
                diff < 1e-3,
                "Q4 mismatch at [{}, {}]: got {} expected {} (diff={:.6})",
                i / n,
                i % n,
                result[i],
                expected[i],
                diff
            );
        }
    }

    /// Dequantize Q4_0 data using the same logic as quantize_q4_host
    fn dequantize_q4_host(data: &[u8], scales: &[u8], numel: usize) -> Vec<f32> {
        let num_blocks = numel / 32;
        let mut out = vec![0.0; numel];

        for block_idx in 0..num_blocks {
            let scale =
                half::f16::from_le_bytes([scales[block_idx * 2], scales[block_idx * 2 + 1]])
                    .to_f32();

            let block_offset = block_idx * 32;
            let data_offset = block_idx * 16;

            // GGML Interleaved packing:
            // byte[j] has element j (low) and element j+16 (high)
            for j in 0..16 {
                let byte = data[data_offset + j];
                let q_lo = (byte & 0x0F) as i32 - 8;
                let q_hi = (byte >> 4) as i32 - 8;

                out[block_offset + j] = q_lo as f32 * scale;
                out[block_offset + j + 16] = q_hi as f32 * scale;
            }
        }
        out
    }

    #[test]
    fn test_q4_quantize_dequantize_roundtrip() {
        // Test that quantize -> dequantize produces reasonable values
        let k = 32;
        let w_data: Vec<f32> = (0..k).map(|i| (i as f32 - 16.0) / 16.0).collect();
        let (q_data, q_scales) = quantize_q4_host(&w_data, 32);

        let dequantized = dequantize_q4_host(&q_data, &q_scales, k);

        // Q4 has limited precision but should be close
        for i in 0..k {
            let rel_err = if w_data[i].abs() > 0.1 {
                (dequantized[i] - w_data[i]).abs() / w_data[i].abs()
            } else {
                (dequantized[i] - w_data[i]).abs()
            };
            assert!(
                rel_err < 0.3,
                "Q4 roundtrip mismatch at {}: original {} -> dequantized {} (rel_err={:.4})",
                i,
                w_data[i],
                dequantized[i],
                rel_err
            );
        }
    }

    #[test]
    fn test_q4_matmul_vs_dequantized_f32_matmul() {
        // Compare quantized matmul against matmul with CPU-dequantized weights
        use crate::cuda::ops::matmul::matmul;

        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let m = 2;
        let k = 64;
        let n = 4;

        // Pseudo-random input and weights
        let mut state: u64 = 77777;
        let mut rand_f32 = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        let input_data: Vec<f32> = (0..m * k).map(|_| rand_f32()).collect();
        let w_data: Vec<f32> = (0..n * k).map(|_| rand_f32()).collect();

        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();

        // Quantize weights
        let (q_data, q_scales) = quantize_q4_host(&w_data, 32);

        // Dequantize on CPU to get f32 reference
        let w_dequantized = dequantize_q4_host(&q_data, &q_scales, n * k);

        // Compute reference: input @ w_dequantized^T
        // The quantized matmul does input @ weight^T, so we need to transpose for f32 matmul
        // Standard matmul: (m, k) @ (k, n) = (m, n)
        // But weight is (n, k), so we need (m, k) @ (n, k)^T = (m, k) @ (k, n) = (m, n)
        // For GPU matmul, we need weight transposed: (k, n)
        let mut w_transposed = vec![0.0_f32; k * n];
        for row in 0..n {
            for col in 0..k {
                w_transposed[col * n + row] = w_dequantized[row * k + col];
            }
        }
        let w_gpu = CudaTensor::from_slice(&ctx, &[k, n], &w_transposed).unwrap();
        let expected_gpu = matmul(&input, &w_gpu).unwrap();
        let expected = expected_gpu.to_vec().unwrap();

        // Run quantized matmul
        let weight =
            QuantizedTensor::from_raw(&ctx, &[n, k], DType::Q4_0, &q_data, &q_scales).unwrap();
        let result_gpu = quantized_matmul(&input, &weight).unwrap();
        let result = result_gpu.to_vec().unwrap();

        // Should match exactly (both use same dequantized values)
        for i in 0..m * n {
            let diff = (result[i] - expected[i]).abs();
            assert!(
                diff < 1e-3,
                "Q4 quantized vs dequantized f32 mismatch at {}: {} vs {} (diff={:.6})",
                i,
                result[i],
                expected[i],
                diff
            );
        }
    }

    /// Compute reference FP8 matmul on CPU: `output(M,N) = input(M,K) @ dequant(weight(N,K))^T`
    fn fp8_matmul_reference(
        input: &[f32],
        weight_fp8: &[u8],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0_f32;
                for i in 0..k {
                    let w = fp8_e4m3_to_f32(weight_fp8[col * k + i]);
                    sum += input[row * k + i] * w;
                }
                output[row * n + col] = sum;
            }
        }
        output
    }

    #[test]
    fn test_fp8_quantize_activations_roundtrip() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let values = vec![1.0_f32, -2.0, 3.5, -0.25, 100.0, -100.0, 0.0, 448.0];
        let input = CudaTensor::from_slice(&ctx, &[1, values.len()], &values).unwrap();

        let (act_fp8, d_inv_scale) =
            quantize_activations_to_fp8(&ctx, &input, values.len()).unwrap();

        let fp8_host = ctx.device().dtoh_sync_copy(&act_fp8).unwrap();
        let inv_scale = ctx.device().dtoh_sync_copy(&d_inv_scale).unwrap()[0];

        // Dequantize and check roundtrip accuracy
        for (i, &original) in values.iter().enumerate() {
            let reconstructed = fp8_e4m3_to_f32(fp8_host[i]) * inv_scale;
            let tol = original.abs() * 0.15 + 0.01; // ~15% relative + small absolute
            assert!(
                (reconstructed - original).abs() < tol,
                "FP8 activation roundtrip[{}]: {} -> fp8={:#04x} -> {} (inv_scale={}, diff={})",
                i,
                original,
                fp8_host[i],
                reconstructed,
                inv_scale,
                (reconstructed - original).abs()
            );
        }
    }

    #[test]
    fn test_fp8_cublas_matmul_basic() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        if !ctx.supports_fp8_tensor_cores() {
            eprintln!("Skipping FP8 cuBLAS test: GPU does not support FP8 tensor cores");
            return;
        }

        let m = 16;
        let k = 32;
        let n = 16;

        let mut state: u64 = 777;
        let mut rand_f32 = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        let input_data: Vec<f32> = (0..m * k).map(|_| rand_f32()).collect();
        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();

        let w_f32: Vec<f32> = (0..n * k).map(|_| rand_f32()).collect();
        let w_fp8: Vec<u8> = w_f32.iter().map(|&v| f32_to_fp8_e4m3(v)).collect();
        let weight = QuantizedTensor::from_raw(&ctx, &[n, k], DType::F8E4M3, &w_fp8, &[]).unwrap();

        let output = quantized_matmul_fp8_cublas(&ctx, &input, &weight, m, n, k).unwrap();
        let result = output.to_vec().unwrap();

        let expected = fp8_matmul_reference(&input_data, &w_fp8, m, n, k);

        for i in 0..m * n {
            let tol = expected[i].abs() * 0.2 + 0.5;
            assert!(
                (result[i] - expected[i]).abs() < tol,
                "FP8 cuBLAS [{i}]: {} vs expected {} (diff={})",
                result[i],
                expected[i],
                (result[i] - expected[i]).abs()
            );
        }
    }

    #[test]
    fn test_fp8_cublas_matmul_with_weight_scale() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        if !ctx.supports_fp8_tensor_cores() {
            eprintln!("Skipping FP8 cuBLAS weight_scale test: no FP8 tensor cores");
            return;
        }

        let m = 16;
        let k = 32;
        let n = 16;
        let weight_scale = 0.5_f32;

        let mut state: u64 = 54321;
        let mut rand_f32 = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        let input_data: Vec<f32> = (0..m * k).map(|_| rand_f32()).collect();
        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();

        let w_f32: Vec<f32> = (0..n * k).map(|_| rand_f32()).collect();
        let w_fp8: Vec<u8> = w_f32.iter().map(|&v| f32_to_fp8_e4m3(v)).collect();
        let mut weight =
            QuantizedTensor::from_raw(&ctx, &[n, k], DType::F8E4M3, &w_fp8, &[]).unwrap();
        weight.set_weight_scale(&ctx, weight_scale).unwrap();

        let output = quantized_matmul_fp8_cublas(&ctx, &input, &weight, m, n, k).unwrap();
        let result = output.to_vec().unwrap();

        // Reference: weight_scale multiplies the dequantized weights
        let ref_unscaled = fp8_matmul_reference(&input_data, &w_fp8, m, n, k);
        for i in 0..m * n {
            let expected = ref_unscaled[i] * weight_scale;
            let tol = expected.abs() * 0.2 + 0.5;
            assert!(
                (result[i] - expected).abs() < tol,
                "FP8 cuBLAS weight_scale [{i}]: {} vs expected {} (diff={})",
                result[i],
                expected,
                (result[i] - expected).abs()
            );
        }
    }

    #[test]
    fn test_fp8_cublas_matmul_small_batch() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        if !ctx.supports_fp8_tensor_cores() {
            eprintln!("Skipping FP8 cuBLAS small batch test: no FP8 tensor cores");
            return;
        }

        let m = 16;
        let k = 64;
        let n = 32;

        let mut state: u64 = 42;
        let mut rand_f32 = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        let input_data: Vec<f32> = (0..m * k).map(|_| rand_f32()).collect();
        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();

        let w_f32: Vec<f32> = (0..n * k).map(|_| rand_f32()).collect();
        let w_fp8: Vec<u8> = w_f32.iter().map(|&v| f32_to_fp8_e4m3(v)).collect();
        let weight = QuantizedTensor::from_raw(&ctx, &[n, k], DType::F8E4M3, &w_fp8, &[]).unwrap();

        let output = quantized_matmul_fp8_cublas(&ctx, &input, &weight, m, n, k).unwrap();
        let result = output.to_vec().unwrap();

        let expected = fp8_matmul_reference(&input_data, &w_fp8, m, n, k);

        for i in 0..m * n {
            let tol = expected[i].abs() * 0.2 + 0.5;
            assert!(
                (result[i] - expected[i]).abs() < tol,
                "FP8 cuBLAS small batch [{i}]: {} vs expected {} (diff={})",
                result[i],
                expected[i],
                (result[i] - expected[i]).abs()
            );
        }
    }

    #[test]
    fn test_fp8_cublas_matmul_larger() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        if !ctx.supports_fp8_tensor_cores() {
            eprintln!("Skipping FP8 cuBLAS larger test: no FP8 tensor cores");
            return;
        }

        let m = 8;
        let k = 128;
        let n = 16;

        let mut state: u64 = 99999;
        let mut rand_f32 = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        let input_data: Vec<f32> = (0..m * k).map(|_| rand_f32()).collect();
        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();

        let w_f32: Vec<f32> = (0..n * k).map(|_| rand_f32()).collect();
        let w_fp8: Vec<u8> = w_f32.iter().map(|&v| f32_to_fp8_e4m3(v)).collect();
        let weight = QuantizedTensor::from_raw(&ctx, &[n, k], DType::F8E4M3, &w_fp8, &[]).unwrap();

        let output = quantized_matmul_fp8_cublas(&ctx, &input, &weight, m, n, k).unwrap();
        let result = output.to_vec().unwrap();

        let expected = fp8_matmul_reference(&input_data, &w_fp8, m, n, k);

        for i in 0..m * n {
            let tol = expected[i].abs() * 0.2 + 1.0;
            assert!(
                (result[i] - expected[i]).abs() < tol,
                "FP8 cuBLAS larger [{i}]: {} vs expected {} (diff={})",
                result[i],
                expected[i],
                (result[i] - expected[i]).abs()
            );
        }
    }

    #[test]
    fn test_gemv_q8_m1() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let m = 1;
        let k = 2048;
        let n = 2048;

        let mut state: u64 = 11111;
        let mut rand_f32 = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        let input_data: Vec<f32> = (0..m * k).map(|_| rand_f32()).collect();
        let w_data: Vec<f32> = (0..n * k).map(|_| rand_f32()).collect();

        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();
        let (q_data, q_scales) = quantize_q8_host(&w_data, QUANTIZATION_BLOCK_SIZE);
        let weight =
            QuantizedTensor::from_raw(&ctx, &[n, k], DType::Q8_0, &q_data, &q_scales).unwrap();

        // Q8_1 quantize activations on CPU (matching GPU: f32 scale = absmax/127)
        let blocks_per_row = k / QUANTIZATION_BLOCK_SIZE;
        let mut act_q8 = vec![0_i8; k];
        let mut act_f32_scales = vec![0.0_f32; blocks_per_row];
        for b in 0..blocks_per_row {
            let block = &input_data[b * QUANTIZATION_BLOCK_SIZE..(b + 1) * QUANTIZATION_BLOCK_SIZE];
            let max_abs = block.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
            let scale = max_abs / 127.0;
            act_f32_scales[b] = scale;
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for j in 0..QUANTIZATION_BLOCK_SIZE {
                act_q8[b * QUANTIZATION_BLOCK_SIZE + j] =
                    (block[j] * inv_scale).round().clamp(-128.0, 127.0) as i8;
            }
        }

        // Reference: dp4a-style dot product with both sides quantized
        let mut expected = vec![0.0_f32; m * n];
        for col in 0..n {
            let mut acc = 0.0_f64;
            for b in 0..blocks_per_row {
                let w_scale = half::f16::from_le_bytes([
                    q_scales[col * blocks_per_row * 2 + b * 2],
                    q_scales[col * blocks_per_row * 2 + b * 2 + 1],
                ])
                .to_f32();
                let a_scale = act_f32_scales[b];

                let mut sumi: i32 = 0;
                for j in 0..QUANTIZATION_BLOCK_SIZE {
                    let w_qi = q_data[col * k + b * QUANTIZATION_BLOCK_SIZE + j] as i8 as i32;
                    let a_qi = act_q8[b * QUANTIZATION_BLOCK_SIZE + j] as i32;
                    sumi += w_qi * a_qi;
                }
                acc += f64::from(w_scale * a_scale * sumi as f32);
            }
            expected[col] = acc as f32;
        }

        let output = quantized_matmul(&input, &weight).unwrap();
        let result = output.to_vec().unwrap();
        assert_eq!(result.len(), n);

        for i in 0..n {
            let diff = (result[i] - expected[i]).abs();
            let ok = if expected[i].abs() < 0.1 {
                diff < 0.05
            } else {
                diff / expected[i].abs() < 0.10
            };
            assert!(
                ok,
                "GEMV Q8 mismatch at [{}]: got {} expected {} (diff={:.6})",
                i, result[i], expected[i], diff
            );
        }
    }

    #[test]
    fn test_gemv_q4_m1() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let m = 1;
        let k = 2048;
        let n = 2048;

        let mut state: u64 = 22222;
        let mut rand_f32 = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        let input_data: Vec<f32> = (0..m * k).map(|_| rand_f32()).collect();
        let w_data: Vec<f32> = (0..n * k).map(|_| rand_f32()).collect();

        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();
        let (q4_data, q4_scales) = quantize_q4_host(&w_data, QUANTIZATION_BLOCK_SIZE);

        // Q8_1 quantize activations on CPU (matching GPU: f32 scales)
        let blocks_per_row = k / QUANTIZATION_BLOCK_SIZE;
        let mut act_q8 = vec![0_i8; k];
        let mut act_f32_scales = vec![0.0_f32; blocks_per_row];
        let mut act_f32_sums = vec![0.0_f32; blocks_per_row];
        for b in 0..blocks_per_row {
            let block = &input_data[b * QUANTIZATION_BLOCK_SIZE..(b + 1) * QUANTIZATION_BLOCK_SIZE];
            let max_abs = block.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
            let scale = max_abs / 127.0;
            act_f32_scales[b] = scale;
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            let mut qi_sum: i32 = 0;
            for j in 0..QUANTIZATION_BLOCK_SIZE {
                let qi = (block[j] * inv_scale).round().clamp(-128.0, 127.0) as i8;
                act_q8[b * QUANTIZATION_BLOCK_SIZE + j] = qi;
                qi_sum += qi as i32;
            }
            act_f32_sums[b] = scale * qi_sum as f32;
        }

        // Reference: simulate the dp4a Q4×Q8_1 computation with zero-point correction
        let mut expected = vec![0.0_f32; n];
        for col in 0..n {
            let mut acc = 0.0_f64;
            for b in 0..blocks_per_row {
                let d_w = half::f16::from_le_bytes([
                    q4_scales[col * blocks_per_row * 2 + b * 2],
                    q4_scales[col * blocks_per_row * 2 + b * 2 + 1],
                ])
                .to_f32();
                let d_a = act_f32_scales[b];
                let s_a = act_f32_sums[b];

                // dp4a-style: unsigned nibbles × int8 activations
                let w_offset = col * (k / 2) + b * 16;
                let a_offset = b * QUANTIZATION_BLOCK_SIZE;
                let mut sumi: i32 = 0;
                for j in 0..16 {
                    let byte = q4_data[w_offset + j];
                    let w_lo = (byte & 0x0F) as i32;
                    let w_hi = (byte >> 4) as i32;
                    let a_lo = act_q8[a_offset + j] as i32;
                    let a_hi = act_q8[a_offset + 16 + j] as i32;
                    sumi += w_lo * a_lo + w_hi * a_hi;
                }
                // Zero-point correction: subtract 8 * s_a
                acc += f64::from(d_w * (d_a * sumi as f32 - 8.0 * s_a));
            }
            expected[col] = acc as f32;
        }

        let weight =
            QuantizedTensor::from_raw(&ctx, &[n, k], DType::Q4_0, &q4_data, &q4_scales).unwrap();
        let output = quantized_matmul(&input, &weight).unwrap();
        let result = output.to_vec().unwrap();
        assert_eq!(result.len(), n);

        for i in 0..n {
            let diff = (result[i] - expected[i]).abs();
            let ok = if expected[i].abs() < 0.1 {
                diff < 0.05
            } else {
                diff / expected[i].abs() < 0.10
            };
            assert!(
                ok,
                "GEMV Q4 mismatch at [{}]: got {} expected {} (diff={:.6})",
                i, result[i], expected[i], diff
            );
        }
    }

    #[test]
    fn test_gemv_matches_gemm_q8() {
        // Verify GEMV (dp4a with Q8_1 activations) produces results close to
        // GEMM (f32 activations). The paths differ by activation quantization
        // error, so we allow ~10% relative tolerance.
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let k = 2048;
        let n = 512;

        let mut state: u64 = 33333;
        let mut rand_f32 = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        let input_data: Vec<f32> = (0..k).map(|_| rand_f32()).collect();
        let w_data: Vec<f32> = (0..n * k).map(|_| rand_f32()).collect();

        let (q_data, q_scales) = quantize_q8_host(&w_data, QUANTIZATION_BLOCK_SIZE);
        let weight =
            QuantizedTensor::from_raw(&ctx, &[n, k], DType::Q8_0, &q_data, &q_scales).unwrap();

        // M=1 → uses GEMV path (dp4a with Q8_1 quantized activations)
        let input_m1 = CudaTensor::from_slice(&ctx, &[1, k], &input_data).unwrap();
        let gemv_result = quantized_matmul(&input_m1, &weight)
            .unwrap()
            .to_vec()
            .unwrap();

        // M=2 with identical rows → uses GEMM path (f32 activations)
        let input_m2_data: Vec<f32> = input_data
            .iter()
            .chain(input_data.iter())
            .copied()
            .collect();
        let input_m2 = CudaTensor::from_slice(&ctx, &[2, k], &input_m2_data).unwrap();
        let gemm_result = quantized_matmul(&input_m2, &weight)
            .unwrap()
            .to_vec()
            .unwrap();

        for i in 0..n {
            let diff = (gemv_result[i] - gemm_result[i]).abs();
            let ok = if gemm_result[i].abs() < 0.5 {
                diff < 0.2
            } else {
                diff / gemm_result[i].abs() < 0.20
            };
            assert!(
                ok,
                "GEMV vs GEMM mismatch at [{}]: GEMV={} GEMM={} (diff={:.6})",
                i, gemv_result[i], gemm_result[i], diff
            );
        }
    }

    #[test]
    fn test_quantize_f32_to_q8_1() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let k = 256;

        let mut state: u64 = 44444;
        let mut rand_f32 = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        let input_data: Vec<f32> = (0..k).map(|_| rand_f32()).collect();
        let input = CudaTensor::from_slice(&ctx, &[1, k], &input_data).unwrap();

        let (act_data, act_scales, act_sums) =
            quantize_activations_to_q8_1(&ctx, &input, k).unwrap();

        let data_host: Vec<i8> = ctx.device().dtoh_sync_copy(&*act_data).unwrap();
        let scales_host: Vec<f32> = ctx.device().dtoh_sync_copy(&*act_scales).unwrap();
        let sums_host: Vec<f32> = ctx.device().dtoh_sync_copy(&*act_sums).unwrap();

        let num_blocks = k / 32;
        assert_eq!(scales_host.len(), num_blocks);
        assert_eq!(sums_host.len(), num_blocks);

        for b in 0..num_blocks {
            let block = &input_data[b * 32..(b + 1) * 32];
            let max_abs = block.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
            let expected_scale = max_abs / 127.0;

            // Scale should match closely
            assert!(
                (scales_host[b] - expected_scale).abs() < 1e-5,
                "Scale mismatch at block {}: got {} expected {}",
                b,
                scales_host[b],
                expected_scale
            );

            // Check quantized values and sum
            let inv_scale = if expected_scale > 0.0 {
                1.0 / expected_scale
            } else {
                0.0
            };
            let mut expected_qi_sum: i32 = 0;
            for j in 0..32 {
                let expected_qi = (block[j] * inv_scale).round().clamp(-128.0, 127.0) as i8 as i32;
                let actual_qi = data_host[b * 32 + j] as i32;
                assert!(
                    (actual_qi - expected_qi).abs() <= 1,
                    "Q8_1 data mismatch at block {} elem {}: got {} expected {}",
                    b,
                    j,
                    actual_qi,
                    expected_qi
                );
                expected_qi_sum += actual_qi;
            }

            let expected_sum = expected_scale * expected_qi_sum as f32;
            assert!(
                (sums_host[b] - expected_sum).abs() < 1e-3,
                "Sum mismatch at block {}: got {} expected {}",
                b,
                sums_host[b],
                expected_sum
            );
        }
    }

    fn pack_gptq_host(
        weights: &[f32],
        out_features: usize,
        in_features: usize,
        group_size: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        assert_eq!(weights.len(), out_features * in_features);
        assert_eq!(in_features % 8, 0);
        assert_eq!(in_features % group_size, 0);
        assert_eq!(out_features % 8, 0);

        let num_groups = in_features / group_size;
        let packed_rows = in_features / 8;

        // Compute per-group, per-output-feature scales
        // scale = max_abs / 7.0 (INT4 unsigned range is 0..15, centered at zero_point)
        // We use zero_point = 8 so effective range is [-8, 7] * scale
        let zero_point = 8_i32;

        let mut scales_f16 = vec![half::f16::from_f32(0.0); num_groups * out_features];
        let mut quantized = vec![0_i32; out_features * in_features];

        for n in 0..out_features {
            for g in 0..num_groups {
                let k_start = g * group_size;
                let k_end = k_start + group_size;
                let group_vals: Vec<f32> = (k_start..k_end)
                    .map(|k| weights[n * in_features + k])
                    .collect();
                let max_abs = group_vals.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
                scales_f16[g * out_features + n] = half::f16::from_f32(scale);

                for (j, &v) in group_vals.iter().enumerate() {
                    let q = ((v / scale).round() as i32 + zero_point).clamp(0, 15);
                    quantized[n * in_features + k_start + j] = q;
                }
            }
        }

        // Pack qweight: [in_features/8, out_features] as int32
        // qweight[pr * N + n] packs 8 values at k=pr*8..pr*8+7 for output n
        let mut qweight = vec![0_u8; packed_rows * out_features * 4];
        for pr in 0..packed_rows {
            for n in 0..out_features {
                let mut packed: u32 = 0;
                for j in 0..8 {
                    let k = pr * 8 + j;
                    let q = quantized[n * in_features + k] as u32;
                    packed |= (q & 0xF) << (j * 4);
                }
                let idx = (pr * out_features + n) * 4;
                qweight[idx..idx + 4].copy_from_slice(&packed.to_le_bytes());
            }
        }

        // Pack scales: [num_groups, out_features] as f16 (already computed)
        let mut scales_bytes = vec![0_u8; num_groups * out_features * 2];
        for i in 0..num_groups * out_features {
            let bytes = scales_f16[i].to_le_bytes();
            scales_bytes[i * 2] = bytes[0];
            scales_bytes[i * 2 + 1] = bytes[1];
        }

        // Pack qzeros: [num_groups, out_features/8] as int32
        // AutoGPTQ convention: stored = actual_zero_point - 1
        let stored_zp = (zero_point - 1) as u32;
        let qzeros_cols = out_features / 8;
        let mut qzeros = vec![0_u8; num_groups * qzeros_cols * 4];
        for g in 0..num_groups {
            for col in 0..qzeros_cols {
                let mut packed: u32 = 0;
                for j in 0..8 {
                    packed |= (stored_zp & 0xF) << (j * 4);
                }
                let idx = (g * qzeros_cols + col) * 4;
                qzeros[idx..idx + 4].copy_from_slice(&packed.to_le_bytes());
            }
        }

        (qweight, scales_bytes, qzeros)
    }

    /// CPU reference for GPTQ matmul: `output[m][n] = sum_k input[m][k] * dequant(weight)[k][n]`
    fn gptq_matmul_reference(
        input: &[f32],
        qweight: &[u8],
        scales: &[u8],
        qzeros: &[u8],
        m: usize,
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let packed_rows = k / 8;
        let qzeros_cols = n / 8;
        let mut output = vec![0.0_f32; m * n];

        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0_f32;
                for pr in 0..packed_rows {
                    let base_k = pr * 8;
                    let group_idx = base_k / group_size;

                    // Read scale (f16)
                    let s_idx = (group_idx * n + col) * 2;
                    let scale =
                        half::f16::from_le_bytes([scales[s_idx], scales[s_idx + 1]]).to_f32();

                    // Read zero-point (packed int4)
                    let qz_idx = (group_idx * qzeros_cols + col / 8) * 4;
                    let qz_packed = u32::from_le_bytes([
                        qzeros[qz_idx],
                        qzeros[qz_idx + 1],
                        qzeros[qz_idx + 2],
                        qzeros[qz_idx + 3],
                    ]);
                    let qz_shift = (col % 8) * 4;
                    // AutoGPTQ stores qzeros with -1 offset: actual = stored + 1
                    let qzero = (((qz_packed >> qz_shift) & 0xF) + 1) as i32;

                    // Read packed qweight
                    let qw_idx = (pr * n + col) * 4;
                    let packed = u32::from_le_bytes([
                        qweight[qw_idx],
                        qweight[qw_idx + 1],
                        qweight[qw_idx + 2],
                        qweight[qw_idx + 3],
                    ]);

                    for j in 0..8 {
                        let kk = base_k + j;
                        if kk >= k {
                            break;
                        }
                        let q = ((packed >> (j * 4)) & 0xF) as i32;
                        let w = (q - qzero) as f32 * scale;
                        acc += input[row * k + kk] * w;
                    }
                }
                output[row * n + col] = acc;
            }
        }
        output
    }

    #[test]
    fn test_matmul_gptq_basic() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let k = 128;
        let m = 1;
        let n = 8;
        let group_size = 128;

        // input: all 1.0 → output = sum of dequantized weight column
        let input_data = vec![1.0_f32; m * k];
        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();

        // weight: constant 2.0
        let w_data: Vec<f32> = vec![2.0_f32; n * k];
        let (qweight, scales, qzeros) = pack_gptq_host(&w_data, n, k, group_size);

        let weight = QuantizedTensor::from_gptq_raw(
            &ctx,
            &[n, k],
            DType::GPTQ_INT4,
            &qweight,
            &scales,
            &qzeros,
            group_size,
        )
        .unwrap();

        let output = quantized_matmul(&input, &weight).unwrap();
        assert_eq!(output.shape(), &[m, n]);

        let result = output.to_vec().unwrap();

        // Expected: 128 * 2.0 = 256.0 (INT4 quantization adds some error)
        let expected = 256.0_f32;
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected).abs() < expected * 0.15,
                "GPTQ basic [{i}]: {} vs expected ~{}",
                v,
                expected
            );
        }
    }

    #[test]
    fn test_matmul_gptq_matches_reference() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let m = 4;
        let k = 128;
        let n = 16;
        let group_size = 128;

        let mut state: u64 = 31337;
        let mut rand_f32 = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        let input_data: Vec<f32> = (0..m * k).map(|_| rand_f32()).collect();
        let w_data: Vec<f32> = (0..n * k).map(|_| rand_f32()).collect();

        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();

        let (qweight, scales, qzeros) = pack_gptq_host(&w_data, n, k, group_size);

        // CPU reference using the same packed data
        let expected =
            gptq_matmul_reference(&input_data, &qweight, &scales, &qzeros, m, n, k, group_size);

        let weight = QuantizedTensor::from_gptq_raw(
            &ctx,
            &[n, k],
            DType::GPTQ_INT4,
            &qweight,
            &scales,
            &qzeros,
            group_size,
        )
        .unwrap();

        let output = quantized_matmul(&input, &weight).unwrap();
        let result = output.to_vec().unwrap();

        for i in 0..m * n {
            let diff = (result[i] - expected[i]).abs();
            assert!(
                diff < 1e-2,
                "GPTQ mismatch at [{}, {}]: got {} expected {} (diff={:.6})",
                i / n,
                i % n,
                result[i],
                expected[i],
                diff
            );
        }
    }

    /// Pack f32 weights into AWQ INT4 format on the host.
    /// Returns (qweight, scales, qzeros) as raw byte vectors.
    ///
    /// - `weights`: `[out_features, in_features]` in row-major order
    /// - `qweight`: `[in_features, out_features/8]` as `int32` (packs 8 output channels per int32)
    /// - `scales`:  `[num_groups, out_features]` as `f16`
    /// - `qzeros`:  `[num_groups, out_features/8]` as `int32`
    fn pack_awq_host(
        weights: &[f32],
        out_features: usize,
        in_features: usize,
        group_size: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        assert_eq!(weights.len(), out_features * in_features);
        assert_eq!(out_features % 8, 0);
        assert_eq!(in_features % group_size, 0);

        let num_groups = in_features / group_size;
        let packed_cols = out_features / 8;
        let zero_point = 8_i32;

        // Compute scales and quantize (same grouping as GPTQ)
        let mut scales_f16 = vec![half::f16::from_f32(0.0); num_groups * out_features];
        let mut quantized = vec![0_i32; out_features * in_features];

        for n in 0..out_features {
            for g in 0..num_groups {
                let k_start = g * group_size;
                let k_end = k_start + group_size;
                let group_vals: Vec<f32> = (k_start..k_end)
                    .map(|k| weights[n * in_features + k])
                    .collect();
                let max_abs = group_vals.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
                scales_f16[g * out_features + n] = half::f16::from_f32(scale);

                for (j, &v) in group_vals.iter().enumerate() {
                    let q = ((v / scale).round() as i32 + zero_point).clamp(0, 15);
                    quantized[n * in_features + k_start + j] = q;
                }
            }
        }

        // Pack qweight: [in_features, out_features/8] as int32
        // qweight[k * packed_cols + col] packs 8 output channels col*8..col*8+7 for input k
        let mut qweight = vec![0_u8; in_features * packed_cols * 4];
        for k in 0..in_features {
            for col in 0..packed_cols {
                let mut packed: u32 = 0;
                for j in 0..8 {
                    let n = col * 8 + j;
                    let q = quantized[n * in_features + k] as u32;
                    packed |= (q & 0xF) << (j * 4);
                }
                let idx = (k * packed_cols + col) * 4;
                qweight[idx..idx + 4].copy_from_slice(&packed.to_le_bytes());
            }
        }

        // Pack scales: [num_groups, out_features] as f16
        let mut scales_bytes = vec![0_u8; num_groups * out_features * 2];
        for i in 0..num_groups * out_features {
            let bytes = scales_f16[i].to_le_bytes();
            scales_bytes[i * 2] = bytes[0];
            scales_bytes[i * 2 + 1] = bytes[1];
        }

        // Pack qzeros: [num_groups, out_features/8] as int32
        // AutoAWQ convention: stored = actual_zero_point - 1
        let stored_zp = (zero_point - 1) as u32;
        let mut qzeros = vec![0_u8; num_groups * packed_cols * 4];
        for g in 0..num_groups {
            for col in 0..packed_cols {
                let mut packed: u32 = 0;
                for j in 0..8 {
                    packed |= (stored_zp & 0xF) << (j * 4);
                }
                let idx = (g * packed_cols + col) * 4;
                qzeros[idx..idx + 4].copy_from_slice(&packed.to_le_bytes());
            }
        }

        (qweight, scales_bytes, qzeros)
    }

    /// CPU reference for AWQ matmul: `output[m][n] = sum_k input[m][k] * dequant(weight)[k][n]`
    fn awq_matmul_reference(
        input: &[f32],
        qweight: &[u8],
        scales: &[u8],
        qzeros: &[u8],
        m: usize,
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let packed_cols = n / 8;
        let mut output = vec![0.0_f32; m * n];

        for row in 0..m {
            for col in 0..n {
                let n_col = col / 8;
                let n_shift = (col % 8) * 4;
                let mut acc = 0.0_f32;

                for kk in 0..k {
                    let group_idx = kk / group_size;

                    // Scale (f16)
                    let s_idx = (group_idx * n + col) * 2;
                    let scale =
                        half::f16::from_le_bytes([scales[s_idx], scales[s_idx + 1]]).to_f32();

                    // Zero-point (packed int4)
                    let qz_idx = (group_idx * packed_cols + n_col) * 4;
                    let qz_packed = u32::from_le_bytes([
                        qzeros[qz_idx],
                        qzeros[qz_idx + 1],
                        qzeros[qz_idx + 2],
                        qzeros[qz_idx + 3],
                    ]);
                    // AutoAWQ stores qzeros with -1 offset: actual = stored + 1
                    let qzero = (((qz_packed >> n_shift) & 0xF) + 1) as i32;

                    // Quantized weight
                    let qw_idx = (kk * packed_cols + n_col) * 4;
                    let qw_packed = u32::from_le_bytes([
                        qweight[qw_idx],
                        qweight[qw_idx + 1],
                        qweight[qw_idx + 2],
                        qweight[qw_idx + 3],
                    ]);
                    let q = ((qw_packed >> n_shift) & 0xF) as i32;

                    let w = (q - qzero) as f32 * scale;
                    acc += input[row * k + kk] * w;
                }
                output[row * n + col] = acc;
            }
        }
        output
    }

    #[test]
    fn test_matmul_awq_basic() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let k = 128;
        let m = 1;
        let n = 8;
        let group_size = 128;

        let input_data = vec![1.0_f32; m * k];
        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();

        let w_data: Vec<f32> = vec![2.0_f32; n * k];
        let (qweight, scales, qzeros) = pack_awq_host(&w_data, n, k, group_size);

        let weight = QuantizedTensor::from_gptq_raw(
            &ctx,
            &[n, k],
            DType::AWQ_INT4,
            &qweight,
            &scales,
            &qzeros,
            group_size,
        )
        .unwrap();

        let output = quantized_matmul(&input, &weight).unwrap();
        assert_eq!(output.shape(), &[m, n]);

        let result = output.to_vec().unwrap();

        let expected = 256.0_f32;
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected).abs() < expected * 0.15,
                "AWQ basic [{i}]: {} vs expected ~{}",
                v,
                expected
            );
        }
    }

    #[test]
    fn test_matmul_awq_matches_reference() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let m = 4;
        let k = 128;
        let n = 16;
        let group_size = 128;

        let mut state: u64 = 42424;
        let mut rand_f32 = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        let input_data: Vec<f32> = (0..m * k).map(|_| rand_f32()).collect();
        let w_data: Vec<f32> = (0..n * k).map(|_| rand_f32()).collect();

        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();

        let (qweight, scales, qzeros) = pack_awq_host(&w_data, n, k, group_size);

        let expected =
            awq_matmul_reference(&input_data, &qweight, &scales, &qzeros, m, n, k, group_size);

        let weight = QuantizedTensor::from_gptq_raw(
            &ctx,
            &[n, k],
            DType::AWQ_INT4,
            &qweight,
            &scales,
            &qzeros,
            group_size,
        )
        .unwrap();

        let output = quantized_matmul(&input, &weight).unwrap();
        let result = output.to_vec().unwrap();

        for i in 0..m * n {
            let diff = (result[i] - expected[i]).abs();
            assert!(
                diff < 1e-2,
                "AWQ mismatch at [{}, {}]: got {} expected {} (diff={:.6})",
                i / n,
                i % n,
                result[i],
                expected[i],
                diff
            );
        }
    }

    #[test]
    fn test_fp8_channel_scales_custom_kernel() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let m = 2;
        let k = 32;
        let n = 4;

        let mut state: u64 = 99999;
        let mut rand_f32 = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        let input_data: Vec<f32> = (0..m * k).map(|_| rand_f32()).collect();
        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();

        let w_f32: Vec<f32> = (0..n * k).map(|_| rand_f32()).collect();
        let w_fp8: Vec<u8> = w_f32.iter().map(|&v| f32_to_fp8_e4m3(v)).collect();
        let channel_scales: Vec<f32> = (0..n).map(|_| rand_f32().abs() + 0.1).collect();

        let mut weight =
            QuantizedTensor::from_raw(&ctx, &[n, k], DType::F8E4M3, &w_fp8, &[]).unwrap();
        weight.set_channel_scales(&ctx, &channel_scales).unwrap();

        let output = quantized_matmul(&input, &weight).unwrap();
        let result = output.to_vec().unwrap();

        // Reference: matmul then per-channel scale
        let ref_unscaled = fp8_matmul_reference(&input_data, &w_fp8, m, n, k);
        for row in 0..m {
            for col in 0..n {
                let i = row * n + col;
                let expected = ref_unscaled[i] * channel_scales[col];
                let tol = expected.abs() * 0.15 + 0.5;
                assert!(
                    (result[i] - expected).abs() < tol,
                    "FP8 channel_scales custom [{row},{col}]: {} vs expected {} (diff={})",
                    result[i],
                    expected,
                    (result[i] - expected).abs()
                );
            }
        }
    }

    #[test]
    fn test_fp8_channel_scales_cublas() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        if !ctx.supports_fp8_tensor_cores() {
            eprintln!("Skipping FP8 cuBLAS channel_scales test: no FP8 tensor cores");
            return;
        }

        let m = 16;
        let k = 32;
        let n = 16;

        let mut state: u64 = 77777;
        let mut rand_f32 = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        let input_data: Vec<f32> = (0..m * k).map(|_| rand_f32()).collect();
        let input = CudaTensor::from_slice(&ctx, &[m, k], &input_data).unwrap();

        let w_f32: Vec<f32> = (0..n * k).map(|_| rand_f32()).collect();
        let w_fp8: Vec<u8> = w_f32.iter().map(|&v| f32_to_fp8_e4m3(v)).collect();
        let channel_scales: Vec<f32> = (0..n).map(|_| rand_f32().abs() + 0.1).collect();

        let mut weight =
            QuantizedTensor::from_raw(&ctx, &[n, k], DType::F8E4M3, &w_fp8, &[]).unwrap();
        weight.set_channel_scales(&ctx, &channel_scales).unwrap();

        let output = quantized_matmul_fp8_cublas(&ctx, &input, &weight, m, n, k).unwrap();
        let result = output.to_vec().unwrap();

        // Reference: matmul then per-channel scale
        let ref_unscaled = fp8_matmul_reference(&input_data, &w_fp8, m, n, k);
        for row in 0..m {
            for col in 0..n {
                let i = row * n + col;
                let expected = ref_unscaled[i] * channel_scales[col];
                let tol = expected.abs() * 0.2 + 0.5;
                assert!(
                    (result[i] - expected).abs() < tol,
                    "FP8 channel_scales cuBLAS [{row},{col}]: {} vs expected {} (diff={})",
                    result[i],
                    expected,
                    (result[i] - expected).abs()
                );
            }
        }
    }
}
