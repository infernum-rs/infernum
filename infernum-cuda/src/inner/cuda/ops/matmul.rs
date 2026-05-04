//! Matrix multiplication using cuBLAS and custom WMMA GEMV kernels.
//!
//! For M > 1 (prefill), all variants use `cublasGemmEx`.
//! For M = 1 (decode), BF16 and F32 dispatch to custom WMMA GEMV kernels
//! (`gemv_bf16` / `gemv_f32`) that exploit tensor cores even for M=1 by
//! replicating the input vector across WMMA_N virtual columns.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::missing_panics_doc,
    clippy::uninlined_format_args
)]

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::{result, sys};
use cudarc::driver::{DevicePtr, DevicePtrMut, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

use crate::cuda::CudaTensor;
use infernum::dtype::DType;
use infernum::tensor::Tensor;
use infernum::Result;

// PTX for the WMMA GEMV kernels (BF16 and F32), compiled from gemv_bf16.cu.
const GEMV_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/gemv_bf16.ptx"));
const GEMV_MODULE: &str = "gemv_bf16";

/// Launch a WMMA GEMV kernel for the M=1 decode path.
///
/// `weight`: shape `(N, K)` row-major (Dense pre-transposed convention).
/// `input`:  shape `(1, K)` or `(K,)`.
/// Output:   shape `(1, N)`, same dtype as input.
///
/// Dispatches to `gemv_bf16` for BF16 or `gemv_f32` for F32.
/// Falls back to `None` for other dtypes (caller uses cuBLAS).
fn gemv_wmma(
    input: &CudaTensor,
    weight: &CudaTensor,
    n: usize,
    k: usize,
) -> Option<Result<CudaTensor>> {
    let dtype = input.dtype();
    let kernel_name = match dtype {
        DType::BF16 => "gemv_bf16",
        DType::F32 => "gemv_f32",
        _ => return None,
    };

    let device = input.context().device();

    if !device.has_func(GEMV_MODULE, kernel_name) {
        device
            .load_ptx(
                Ptx::from_src(GEMV_PTX),
                GEMV_MODULE,
                &["gemv_bf16", "gemv_f32"],
            )
            .expect("failed to load gemv_bf16 PTX");
    }

    let func = device.get_func(GEMV_MODULE, kernel_name).unwrap();

    // Block: (COLS_PER_CTA=32, K_SPLITS=16) — 512 threads total.
    // cudarc LaunchConfig uses flat thread count; pass the 2D dims via cfg.
    const COLS_PER_CTA: u32 = 32;
    const K_SPLITS: u32 = 16;

    let grid = ((n as u32).div_ceil(COLS_PER_CTA), 1, 1);
    let block = (COLS_PER_CTA, K_SPLITS, 1);

    let mut output = match unsafe { CudaTensor::uninit(input.context(), &[1, n], dtype) } {
        Ok(t) => t,
        Err(e) => return Some(Err(e)),
    };

    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: 0,
    };

    // Pass CudaSlice/CudaView references so cudarc routes the launch to the
    // device's current stream (required for CUDA graph capture compatibility).
    let result = unsafe {
        func.launch(
            cfg,
            (
                &weight.cuda_slice(),
                &input.cuda_slice(),
                output.cuda_slice_mut(),
                n as i32,
                k as i32,
            ),
        )
    };

    match result {
        Ok(()) => Some(Ok(output)),
        Err(e) => Some(Err(e.into())),
    }
}

/// Map `DType` to the cuBLAS data type enum.
fn cublas_data_type(dtype: DType) -> sys::cudaDataType_t {
    match dtype {
        DType::F32 => sys::cudaDataType_t::CUDA_R_32F,
        DType::F16 => sys::cudaDataType_t::CUDA_R_16F,
        DType::BF16 => sys::cudaDataType_t::CUDA_R_16BF,
        _ => panic!("Unsupported dtype for matmul: {dtype:?}"),
    }
}

/// Perform matrix multiplication: C = A @ B
///
/// Supports 2D matrices and batched 3D tensors.
/// - 2D: (M, K) @ (K, N) -> (M, N)
/// - 3D: (B, M, K) @ (B, K, N) -> (B, M, N)
///
/// # Errors
/// Returns an error if shapes are incompatible or cuBLAS operation fails
pub fn matmul(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    match (a_shape.len(), b_shape.len()) {
        (2, 2) => matmul_2d(a, b),
        (3, 3) => matmul_batched(a, b),
        (3, 2) => {
            // Broadcast B across batch dimension
            let batch = a_shape[0];
            let m = a_shape[1];
            let k = a_shape[2];
            let n = b_shape[1];

            assert_eq!(b_shape[0], k, "Inner dimensions must match");

            let a_2d = a.reshape(&[batch * m, k]);
            let c_2d = matmul_2d(&a_2d, b)?;
            let c = c_2d.reshape(&[batch, m, n]);

            Ok(c)
        }
        _ => panic!("Unsupported matmul shapes: {:?} @ {:?}", a_shape, b_shape),
    }
}

/// 2D matrix multiplication: (M, K) @ (K, N) -> (M, N)
fn matmul_2d(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let dtype = a.dtype();

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    assert_eq!(
        k, b_shape[0],
        "Inner dimensions must match: {} vs {}",
        k, b_shape[0]
    );

    // M=1 decode fast path: use WMMA GEMV kernels (tensor-core path) for
    // BF16 and F32. Falls back to cuBLAS for F16 or other dtypes.
    if m == 1 {
        if let Some(result) = gemv_wmma(a, b, n, k) {
            return result;
        }
    }

    let c_shape = [m, n];
    let mut c = unsafe { CudaTensor::uninit(a.context(), &c_shape, dtype)? };

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let data_type = cublas_data_type(dtype);

    // cuBLAS uses column-major order, so we compute C^T = B^T @ A^T
    // which gives us C in row-major order
    unsafe {
        result::gemm_ex(
            *a.context().blas().handle(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n as i32,
            m as i32,
            k as i32,
            (&raw const alpha).cast(),
            *b.cuda_slice().device_ptr() as *const _,
            data_type,
            n as i32,
            *a.cuda_slice().device_ptr() as *const _,
            data_type,
            k as i32,
            (&raw const beta).cast(),
            *c.cuda_slice_mut().device_ptr_mut() as *mut _,
            data_type,
            n as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )?;
    }

    Ok(c)
}

/// Batched 3D matrix multiplication: (B, M, K) @ (B, K, N) -> (B, M, N)
fn matmul_batched(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let dtype = a.dtype();

    let batch = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[2];

    assert_eq!(batch, b_shape[0], "Batch dimensions must match");
    assert_eq!(k, b_shape[1], "Inner dimensions must match");

    let c_shape = [batch, m, n];
    let mut c = unsafe { CudaTensor::uninit(a.context(), &c_shape, dtype)? };

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let data_type = cublas_data_type(dtype);

    let stride_a = (k * n) as i64;
    let stride_b = (m * k) as i64;
    let stride_c = (m * n) as i64;

    // cuBLAS uses column-major, so we compute C^T = B^T @ A^T per batch
    unsafe {
        result::gemm_strided_batched_ex(
            *a.context().blas().handle(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n as i32,
            m as i32,
            k as i32,
            (&raw const alpha).cast(),
            *b.cuda_slice().device_ptr() as *const _,
            data_type,
            n as i32,
            stride_a,
            *a.cuda_slice().device_ptr() as *const _,
            data_type,
            k as i32,
            stride_b,
            (&raw const beta).cast(),
            *c.cuda_slice_mut().device_ptr_mut() as *mut _,
            data_type,
            n as i32,
            stride_c,
            batch as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )?;
    }

    Ok(c)
}

/// Mixed-precision matrix multiplication: bf16 inputs, f32 output.
///
/// Uses `cublasGemmEx` directly to avoid the output bf16→f32 cast.
/// Supports 2D and 3D×2D (broadcast B across batch dimension).
///
/// # Errors
/// Returns an error if shapes are incompatible or cuBLAS operation fails
pub fn matmul_bf16_f32(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    match (a_shape.len(), b_shape.len()) {
        (2, 2) => matmul_bf16_f32_2d(a, b),
        (3, 2) => {
            let batch = a_shape[0];
            let m = a_shape[1];
            let k = a_shape[2];
            let n = b_shape[1];
            assert_eq!(b_shape[0], k, "Inner dimensions must match");
            let a_2d = a.reshape(&[batch * m, k]);
            let c_2d = matmul_bf16_f32_2d(&a_2d, b)?;
            let c = c_2d.reshape(&[batch, m, n]);
            Ok(c)
        }
        _ => panic!(
            "Unsupported matmul_bf16_f32 shapes: {:?} @ {:?}",
            a_shape, b_shape
        ),
    }
}

/// 2D mixed-precision matmul: bf16 inputs, f32 output
fn matmul_bf16_f32_2d(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    assert_eq!(
        k, b_shape[0],
        "Inner dimensions must match: {} vs {}",
        k, b_shape[0]
    );

    let mut c = unsafe { CudaTensor::uninit(a.context(), &[m, n], DType::F32)? };

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    // cuBLAS uses column-major: compute C^T = B^T @ A^T
    unsafe {
        result::gemm_ex(
            *a.context().blas().handle(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n as i32,
            m as i32,
            k as i32,
            (&raw const alpha).cast(),
            *b.cuda_slice().device_ptr() as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            n as i32,
            *a.cuda_slice().device_ptr() as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            k as i32,
            (&raw const beta).cast(),
            *c.cuda_slice_mut().device_ptr_mut() as *mut _,
            sys::cudaDataType_t::CUDA_R_32F,
            n as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )?;
    }

    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;

    #[test]
    fn test_matmul_2d() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // A: 2x3, B: 3x4 -> C: 2x4
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];

        let a = CudaTensor::from_slice(&ctx, &[2, 3], &a_data).unwrap();
        let b = CudaTensor::from_slice(&ctx, &[3, 4], &b_data).unwrap();

        let c = matmul(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 4]);

        let result: Vec<f32> = c.to_vec::<f32>().unwrap();

        let expected: Vec<f32> = vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_batched_3d() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let a_data: Vec<f32> = vec![1.0, 2.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![3.0, 4.0, 7.0, 8.0];

        let a = CudaTensor::from_slice(&ctx, &[2, 1, 2], &a_data).unwrap();
        let b = CudaTensor::from_slice(&ctx, &[2, 2, 1], &b_data).unwrap();

        let c = matmul(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 1, 1]);

        let result: Vec<f32> = c.to_vec::<f32>().unwrap();
        assert!((result[0] - 11.0).abs() < 1e-4);
        assert!((result[1] - 83.0).abs() < 1e-4);
    }

    #[test]
    fn test_matmul_3d_times_2d_broadcast() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let a_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, // batch 0, row 0
            0.0, 1.0, 0.0, // batch 0, row 1
            0.0, 0.0, 1.0, // batch 1, row 0
            1.0, 1.0, 0.0, // batch 1, row 1
        ];
        let b_data: Vec<f32> = vec![
            1.0, 2.0, // col 0, col 1 of row 0
            3.0, 4.0, // col 0, col 1 of row 1
            5.0, 6.0, // col 0, col 1 of row 2
        ];

        let a = CudaTensor::from_slice(&ctx, &[2, 2, 3], &a_data).unwrap();
        let b = CudaTensor::from_slice(&ctx, &[3, 2], &b_data).unwrap();

        let c = matmul(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2, 2]);

        let result: Vec<f32> = c.to_vec::<f32>().unwrap();
        assert!((result[0] - 1.0).abs() < 1e-4);
        assert!((result[1] - 2.0).abs() < 1e-4);
        assert!((result[2] - 3.0).abs() < 1e-4);
        assert!((result[3] - 4.0).abs() < 1e-4);
        assert!((result[4] - 5.0).abs() < 1e-4);
        assert!((result[5] - 6.0).abs() < 1e-4);
        assert!((result[6] - 4.0).abs() < 1e-4);
        assert!((result[7] - 6.0).abs() < 1e-4);
    }

    #[test]
    fn test_matmul_identity() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let identity: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let a = CudaTensor::from_slice(&ctx, &[2, 3], &a_data).unwrap();
        let i = CudaTensor::from_slice(&ctx, &[3, 3], &identity).unwrap();

        let c = matmul(&a, &i).unwrap();

        assert_eq!(c.shape(), &[2, 3]);

        let result: Vec<f32> = c.to_vec::<f32>().unwrap();
        for (idx, (&got, &exp)) in result.iter().zip(a_data.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "Mismatch at {idx}: {got} vs {exp}"
            );
        }
    }

    #[test]
    fn test_matmul_bf16_f32_mixed() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let a_data: Vec<half::bf16> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .map(|&x| half::bf16::from_f32(x))
            .collect();
        let b_data: Vec<half::bf16> = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]
        .iter()
        .map(|&x| half::bf16::from_f32(x))
        .collect();

        let a = CudaTensor::from_slice(&ctx, &[2, 3], &a_data).unwrap();
        let b = CudaTensor::from_slice(&ctx, &[3, 4], &b_data).unwrap();

        let c = matmul_bf16_f32(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 4]);

        let result: Vec<f32> = c.to_vec::<f32>().unwrap();
        let expected: Vec<f32> = vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 0.5, "Mismatch at {i}: {got} vs {exp}");
        }
    }

    #[test]
    fn test_matmul_f16() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let a_data: Vec<half::f16> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(half::f16::from_f32)
            .collect();
        let b_data: Vec<half::f16> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]
        .into_iter()
        .map(half::f16::from_f32)
        .collect();

        let a = CudaTensor::from_slice(&ctx, &[2, 3], &a_data).unwrap();
        let b = CudaTensor::from_slice(&ctx, &[3, 4], &b_data).unwrap();

        let c = matmul(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 4]);

        let result: Vec<half::f16> = c.to_vec::<half::f16>().unwrap();
        let expected: Vec<half::f16> = vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]
            .into_iter()
            .map(half::f16::from_f32)
            .collect();

        for (idx, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got.to_f32() - exp.to_f32()).abs() < 0.5,
                "Mismatch at {idx}: {got} vs {exp}"
            );
        }
    }

    #[test]
    fn test_matmul_bf16() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let a_data: Vec<half::bf16> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(half::bf16::from_f32)
            .collect();
        let b_data: Vec<half::bf16> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]
        .into_iter()
        .map(half::bf16::from_f32)
        .collect();

        let a = CudaTensor::from_slice(&ctx, &[2, 3], &a_data).unwrap();
        let b = CudaTensor::from_slice(&ctx, &[3, 4], &b_data).unwrap();

        let c = matmul(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 4]);

        let result: Vec<half::bf16> = c.to_vec::<half::bf16>().unwrap();
        let expected: Vec<half::bf16> = vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]
            .into_iter()
            .map(half::bf16::from_f32)
            .collect();

        for (idx, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got.to_f32() - exp.to_f32()).abs() < 0.5,
                "Mismatch at {idx}: {got} vs {exp}"
            );
        }
    }
}
