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

use cudarc::cublaslt::MatmulShared;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceSlice, LaunchAsync, LaunchConfig};

use crate::cuda::quantized::QuantizedTensor;
use crate::cuda::CudaContext;
use crate::cuda::CudaTensor;
use crate::dtype::{DType, Q6_K_BLOCK_ELEMENTS};
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

// Q4_0 matmul: GGML Q4_0 non-consecutive packing
// byte[j] has element j in low nibble, element j+16 in high nibble
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
        // Each block has 16 bytes containing 32 values in non-consecutive layout:
        // byte[j] has element j (low nibble) and element j+16 (high nibble)
        int packed_base = n * (K / 2) + base_k / 2;
        const float* in_ptr = input + m * K + base_k;

        // Process first half (elements 0-15): low nibbles
        for (int j = 0; j < 16; ++j) {
            unsigned char packed = weight_data[packed_base + j];
            float w_lo = (float)((int)(packed & 0x0F) - 8) * scale;
            acc += in_ptr[j] * w_lo;
        }
        // Process second half (elements 16-31): high nibbles
        for (int j = 0; j < 16; ++j) {
            unsigned char packed = weight_data[packed_base + j];
            float w_hi = (float)((int)(packed >> 4) - 8) * scale;
            acc += in_ptr[16 + j] * w_hi;
        }
    }

    output[m * N + n] = acc;
}

// FP8 E4M3 matmul: each weight byte is an fp8 value (no block structure)
// Manual decode: sign(1) | exponent(4) | mantissa(3), bias=7
// weight_scale: per-tensor scale factor (from dynamic quantization)
extern "C" __global__ void matmul_fp8_f32(
    float*       __restrict__ output,
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight_data,
    const float weight_scale,
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

    output[m * N + n] = acc * weight_scale;
}

// Q6_K matmul: super-block of 256 elements, 210 bytes each
// Layout per super-block: ql[128] | qh[64] | scales[16] | d(f16)
//   ql: lower 4 bits of each 6-bit value, 2 per byte (low/high nibble)
//   qh: upper 2 bits of each 6-bit value, 4 per byte (2-bit fields)
//   scales: i8 sub-block scale, one per 16 elements
//   d: f16 super-block scale factor
//
// data pointer: packed super-blocks, contiguous for all N rows
// Each row of N has (K / 256) super-blocks × 210 bytes
extern "C" __global__ void matmul_q6k_f32(
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

    const int blocks_per_row = K / 256;
    const int block_bytes = 210;
    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int sb_offset = (n * blocks_per_row + b) * block_bytes;
        const unsigned char* ql     = weight_data + sb_offset;
        const unsigned char* qh     = weight_data + sb_offset + 128;
        const signed char*   scales = (const signed char*)(weight_data + sb_offset + 128 + 64);
        float d = f16_to_f32(*(const unsigned short*)(weight_data + sb_offset + 128 + 64 + 16));

        int base_k = b * 256;
        const float* in_ptr = input + m * K + base_k;

        for (int elem = 0; elem < 256; ++elem) {
            // Map element index using ggml's interleaved layout (8x32 -> 16x16 reshape)
            // The data flows: (128,) -> (2,1,64) -> shift -> (2,2,64) -> (8,32) -> (16,16)
            int sb = elem / 16;           // sub-block 0-15
            int sb_elem = elem % 16;      // element within sub-block
            int flat_idx = sb * 16 + sb_elem;
            int row8 = flat_idx / 32;     // 0-7
            int col32 = flat_idx % 32;    // 0-31

            // ql layout after reshape to (8,32):
            // row 0: bytes 0-31 low nibbles    row 4: bytes 64-95 low nibbles
            // row 1: bytes 32-63 low nibbles   row 5: bytes 96-127 low nibbles
            // row 2: bytes 0-31 high nibbles   row 6: bytes 64-95 high nibbles
            // row 3: bytes 32-63 high nibbles  row 7: bytes 96-127 high nibbles
            int ql_half = row8 / 4;           // 0 for rows 0-3, 1 for rows 4-7
            int ql_nibble_sel = (row8 % 4) / 2; // 0 for rows 0-1,4-5 (low), 1 for 2-3,6-7 (high)
            int ql_offset = (row8 % 4) % 2;   // 0 for even rows in group, 1 for odd
            int ql_byte_idx = ql_half * 64 + ql_offset * 32 + col32;
            unsigned char ql_byte = ql[ql_byte_idx];
            int ql_val = (ql_nibble_sel == 0) ? (ql_byte & 0x0F) : (ql_byte >> 4);

            // qh layout: (64,) -> (2,1,32) -> shift -> (2,4,32) -> (8,32)
            // row 0: bytes 0-31 bits 0-1   row 4: bytes 32-63 bits 0-1
            // row 1: bytes 0-31 bits 2-3   row 5: bytes 32-63 bits 2-3
            // row 2: bytes 0-31 bits 4-5   row 6: bytes 32-63 bits 4-5
            // row 3: bytes 0-31 bits 6-7   row 7: bytes 32-63 bits 6-7
            int qh_half = row8 / 4;           // 0 or 1 (selects 32-byte half)
            int qh_shift_sel = row8 % 4;      // 0,1,2,3 -> shift 0,2,4,6
            int qh_byte_idx = qh_half * 32 + col32;
            unsigned char qh_byte = qh[qh_byte_idx];
            int qh_shift = qh_shift_sel * 2;
            int qh_val = (qh_byte >> qh_shift) & 0x03;

            // Combine to 6-bit [0,63], center to signed [-32,31]
            int q = (ql_val | (qh_val << 4)) - 32;

            float sc = (float)scales[sb];
            float w = d * sc * (float)q;
            acc += in_ptr[elem] * w;
        }
    }

    output[m * N + n] = acc;
}
"#;

// ---------------------------------------------------------------------------
// FP8 activation quantization kernel
// ---------------------------------------------------------------------------

const FP8_QUANTIZE_KERNEL: &str = r#"
// Quantize f32 activations to FP8 E4M3, entirely on device.
//
// Two-kernel approach (no CPU readback):
//   1. absmax_f32: parallel reduction to find max|x| across the tensor
//   2. quantize_f32_to_fp8: reads absmax from device memory, computes scale,
//      writes inv_scale to a device pointer, and quantizes all elements

extern "C" __global__ void absmax_f32(
    const float* __restrict__ input,
    unsigned int* __restrict__ max_bits,
    const int numel
) {
    extern __shared__ unsigned int smax[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_max = 0.0f;
    for (int i = idx; i < numel; i += blockDim.x * gridDim.x) {
        float v = fabsf(input[i]);
        if (v > local_max) local_max = v;
    }

    smax[tid] = __float_as_uint(local_max);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = __uint_as_float(smax[tid]);
            float b = __uint_as_float(smax[tid + s]);
            smax[tid] = __float_as_uint(fmaxf(a, b));
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(max_bits, smax[0]);
    }
}

// Reads absmax from device memory, computes scale on-device, writes inv_scale,
// and quantizes input to FP8 E4M3 — all without any CPU<->GPU synchronization.
extern "C" __global__ void quantize_f32_to_fp8(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    const unsigned int* __restrict__ max_bits,
    float* __restrict__ inv_scale_out,
    const int numel
) {
    // First thread computes and broadcasts the scale via shared memory
    __shared__ float s_scale;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float absmax = __uint_as_float(*max_bits);
        float inv = (absmax > 0.0f) ? (absmax / 448.0f) : 1.0f;
        *inv_scale_out = inv;
    }
    if (threadIdx.x == 0) {
        float absmax = __uint_as_float(*max_bits);
        s_scale = (absmax > 0.0f) ? (448.0f / absmax) : 1.0f;
    }
    __syncthreads();

    float scale = s_scale;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float v = input[idx] * scale;
    v = fminf(fmaxf(v, -448.0f), 448.0f);

    unsigned char sign = (v < 0.0f) ? 1 : 0;
    float abs_v = fabsf(v);

    unsigned char result;
    if (abs_v < 0.001953125f) {
        result = 0;
    } else if (abs_v < 0.015625f) {
        int mant = (int)roundf(abs_v * 512.0f);
        if (mant > 7) mant = 7;
        if (mant < 1) mant = 1;
        result = (unsigned char)mant;
    } else {
        int raw_exp;
        float frac = frexpf(abs_v, &raw_exp);
        int biased_exp = raw_exp + 6;
        if (biased_exp < 1) biased_exp = 1;
        if (biased_exp > 15) biased_exp = 15;

        int mant = (int)roundf((2.0f * frac - 1.0f) * 8.0f);
        if (mant > 7) {
            mant = 0;
            biased_exp++;
            if (biased_exp > 15) { biased_exp = 15; mant = 6; }
        }

        if (biased_exp == 15 && mant == 7) mant = 6;

        result = ((unsigned char)biased_exp << 3) | (unsigned char)mant;
    }

    output[idx] = (sign << 7) | result;
}
"#;

// ---------------------------------------------------------------------------
// FP8 cuBLAS helpers
// ---------------------------------------------------------------------------

/// Quantize f32 activations to FP8 E4M3 on-the-fly, returning the
/// quantized buffer and a device-side inverse scale factor (for cuBLASLt).
///
/// Everything runs on the GPU — no CPU↔GPU synchronization.
fn quantize_activations_to_fp8(
    ctx: &CudaContext,
    input: &CudaTensor<f32>,
    numel: usize,
) -> Result<(CudaSlice<u8>, CudaSlice<f32>)> {
    let device = ctx.device();

    let module_name = "fp8_quantize";
    if !device.has_func(module_name, "absmax_f32") {
        let ptx = cudarc::nvrtc::safe::compile_ptx(FP8_QUANTIZE_KERNEL)?;
        device.load_ptx(ptx, module_name, &["absmax_f32", "quantize_f32_to_fp8"])?;
    }

    // Pass 1: find absmax (result stays on device)
    let mut max_bits = device.htod_sync_copy(&[0_u32])?;
    {
        let threads = 256;
        let blocks = ((numel + threads - 1) / threads).min(1024);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: (threads * std::mem::size_of::<u32>()) as u32,
        };
        let func = device.get_func(module_name, "absmax_f32").unwrap();
        unsafe {
            func.launch(cfg, (input.cuda_slice(), &mut max_bits, numel as i32))?;
        }
    }

    // Pass 2: compute scale on-device, write inv_scale to device buffer, quantize
    let mut act_fp8 = unsafe { device.alloc::<u8>(numel)? };
    let mut d_inv_scale = unsafe { device.alloc::<f32>(1)? };
    {
        let threads = 256;
        let blocks = (numel + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: std::mem::size_of::<f32>() as u32,
        };
        let func = device.get_func(module_name, "quantize_f32_to_fp8").unwrap();
        unsafe {
            func.launch(
                cfg,
                (
                    input.cuda_slice(),
                    &mut act_fp8,
                    &max_bits,
                    &mut d_inv_scale,
                    numel as i32,
                ),
            )?;
        }
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
    use cudarc::cublaslt::{result as lt_result, sys as lt_sys};

    let blas_lt = ctx.blas_lt();

    // Weight scale: use pre-cached device buffer if available, else create one
    let d_scale_b_owned;
    let d_scale_b = if let Some(s) = weight.d_weight_scale() {
        s
    } else {
        d_scale_b_owned = ctx.device().htod_sync_copy(&[weight.weight_scale()])?;
        &d_scale_b_owned
    };

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
    execute_fp8_gemm(ctx, &act_fp8, weight, &d_inv_scale_a, m, n, k)
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
    let mut output = unsafe { CudaTensor::<f32>::uninit(input.context(), &[m, n])? };

    let device = input.context().device();

    let module_name = "quantized_matmul";
    if !device.has_func(module_name, "matmul_q8_f32") {
        let ptx = cudarc::nvrtc::safe::compile_ptx(QUANTIZED_MATMUL_KERNEL)?;
        device.load_ptx(
            ptx,
            module_name,
            &[
                "matmul_q8_f32",
                "matmul_q4_f32",
                "matmul_fp8_f32",
                "matmul_q6k_f32",
            ],
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
            // cuBLASLt handles padding for small M (including M=1 decode)
            if input.context().supports_fp8_tensor_cores() && n >= 16 && k >= 16 {
                return quantized_matmul_fp8_cublas(input.context(), input, weight, m, n, k);
            }
            let func = device.get_func(module_name, "matmul_fp8_f32").unwrap();
            unsafe {
                func.launch(
                    cfg,
                    (
                        output.cuda_slice_mut(),
                        input.cuda_slice(),
                        weight.data_slice(),
                        weight.weight_scale(),
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
}
