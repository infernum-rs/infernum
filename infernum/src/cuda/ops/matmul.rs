//! Matrix multiplication using cuBLAS

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::missing_panics_doc,
    clippy::uninlined_format_args
)]

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{DevicePtr, DevicePtrMut, DeviceRepr};

use crate::cuda::CudaTensor;
use crate::dtype::TensorDType;
use crate::tensor::Tensor;
use crate::Result;

/// Perform matrix multiplication: C = A @ B
///
/// Supports 2D matrices and batched 3D tensors.
/// - 2D: (M, K) @ (K, N) -> (M, N)
/// - 3D: (B, M, K) @ (B, K, N) -> (B, M, N)
///
/// # Errors
/// Returns an error if shapes are incompatible or cuBLAS operation fails
pub fn matmul<T>(a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>>
where
    T: TensorDType + DeviceRepr + GemmScalar + Default,
    CudaBlas: Gemm<T>,
{
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

            // Perform batched matmul by treating A as batch*M x K and B as K x N
            // Then reshape result
            let a_2d = a.reshape(&[batch * m, k]);
            let c_2d = matmul_2d(&a_2d, b)?;
            let c = c_2d.reshape(&[batch, m, n]);

            Ok(c)
        }
        _ => panic!("Unsupported matmul shapes: {:?} @ {:?}", a_shape, b_shape),
    }
}

/// 2D matrix multiplication: (M, K) @ (K, N) -> (M, N)
fn matmul_2d<T>(a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>>
where
    T: TensorDType + DeviceRepr + GemmScalar + Default,
    CudaBlas: Gemm<T>,
{
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

    let c_shape = [m, n];
    let mut c = unsafe { CudaTensor::<T>::uninit(a.context(), &c_shape)? };

    // cuBLAS uses column-major order, so we compute C^T = B^T @ A^T
    // which gives us C in row-major order
    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: T::ONE,
        lda: n as i32,
        ldb: k as i32,
        beta: T::ZERO,
        ldc: n as i32,
    };

    unsafe {
        a.context()
            .blas()
            .gemm(cfg, &b.cuda_slice(), &a.cuda_slice(), c.cuda_slice_mut())?;
    }

    Ok(c)
}

/// Batched 3D matrix multiplication: (B, M, K) @ (B, K, N) -> (B, M, N)
///
/// Uses `gemm_strided_batched` for a single cuBLAS call across all batches.
fn matmul_batched<T>(a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>>
where
    T: TensorDType + DeviceRepr + GemmScalar + Default,
    CudaBlas: Gemm<T>,
{
    let a_shape = a.shape();
    let b_shape = b.shape();

    let batch = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[2];

    assert_eq!(batch, b_shape[0], "Batch dimensions must match");
    assert_eq!(k, b_shape[1], "Inner dimensions must match");

    let c_shape = [batch, m, n];
    let mut c = unsafe { CudaTensor::<T>::uninit(a.context(), &c_shape)? };

    // cuBLAS uses column-major, so we compute C^T = B^T @ A^T per batch
    let cfg = StridedBatchedConfig {
        gemm: GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: T::ONE,
            lda: n as i32,
            ldb: k as i32,
            beta: T::ZERO,
            ldc: n as i32,
        },
        batch_size: batch as i32,
        stride_a: (k * n) as i64,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
    };

    unsafe {
        a.context().blas().gemm_strided_batched(
            cfg,
            &b.cuda_slice(),
            &a.cuda_slice(),
            c.cuda_slice_mut(),
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
pub fn matmul_bf16_f32(
    a: &CudaTensor<half::bf16>,
    b: &CudaTensor<half::bf16>,
) -> Result<CudaTensor<f32>> {
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
fn matmul_bf16_f32_2d(
    a: &CudaTensor<half::bf16>,
    b: &CudaTensor<half::bf16>,
) -> Result<CudaTensor<f32>> {
    use cudarc::cublas::{result, sys};

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

    let mut c = unsafe { CudaTensor::<f32>::uninit(a.context(), &[m, n])? };

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
            (&alpha) as *const f32 as *const _,
            *b.cuda_slice().device_ptr() as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            n as i32,
            *a.cuda_slice().device_ptr() as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            k as i32,
            (&beta) as *const f32 as *const _,
            *c.cuda_slice_mut().device_ptr_mut() as *mut _,
            sys::cudaDataType_t::CUDA_R_32F,
            n as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )?;
    }

    Ok(c)
}

/// Trait for GEMM scalar values (alpha/beta coefficients)
pub trait GemmScalar {
    /// The multiplicative identity (1.0)
    const ONE: Self;
    /// The additive identity (0.0)
    const ZERO: Self;
}

impl GemmScalar for f32 {
    const ONE: Self = 1.0;
    const ZERO: Self = 0.0;
}

impl GemmScalar for half::f16 {
    const ONE: Self = half::f16::ONE;
    const ZERO: Self = half::f16::ZERO;
}

impl GemmScalar for half::bf16 {
    const ONE: Self = half::bf16::ONE;
    const ZERO: Self = half::bf16::ZERO;
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

        let result = c.to_vec().unwrap();

        // Expected: row 0 of A dot cols of B
        // C[0,0] = 1*1 + 2*5 + 3*9 = 1 + 10 + 27 = 38
        // C[0,1] = 1*2 + 2*6 + 3*10 = 2 + 12 + 30 = 44
        // C[0,2] = 1*3 + 2*7 + 3*11 = 3 + 14 + 33 = 50
        // C[0,3] = 1*4 + 2*8 + 3*12 = 4 + 16 + 36 = 56
        // C[1,0] = 4*1 + 5*5 + 6*9 = 4 + 25 + 54 = 83
        // C[1,1] = 4*2 + 5*6 + 6*10 = 8 + 30 + 60 = 98
        // C[1,2] = 4*3 + 5*7 + 6*11 = 12 + 35 + 66 = 113
        // C[1,3] = 4*4 + 5*8 + 6*12 = 16 + 40 + 72 = 128
        let expected: Vec<f32> = vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_batched_3d() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // Batch=2, A: (2,1,2), B: (2,2,1) -> C: (2,1,1)
        // Batch 0: [1,2] @ [[3],[4]] = [11]
        // Batch 1: [5,6] @ [[7],[8]] = [83]
        let a_data: Vec<f32> = vec![1.0, 2.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![3.0, 4.0, 7.0, 8.0];

        let a = CudaTensor::from_slice(&ctx, &[2, 1, 2], &a_data).unwrap();
        let b = CudaTensor::from_slice(&ctx, &[2, 2, 1], &b_data).unwrap();

        let c = matmul(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 1, 1]);

        let result = c.to_vec().unwrap();
        assert!((result[0] - 11.0).abs() < 1e-4);
        assert!((result[1] - 83.0).abs() < 1e-4);
    }

    #[test]
    fn test_matmul_3d_times_2d_broadcast() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // A: (2, 2, 3), B: (3, 2) -> C: (2, 2, 2)
        // B is broadcast across the batch dimension
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

        let result = c.to_vec().unwrap();
        // batch 0 row 0: [1,0,0]@B = [1,2]
        // batch 0 row 1: [0,1,0]@B = [3,4]
        // batch 1 row 0: [0,0,1]@B = [5,6]
        // batch 1 row 1: [1,1,0]@B = [4,6]
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

        // A @ I = A
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let identity: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let a = CudaTensor::from_slice(&ctx, &[2, 3], &a_data).unwrap();
        let i = CudaTensor::from_slice(&ctx, &[3, 3], &identity).unwrap();

        let c = matmul(&a, &i).unwrap();

        assert_eq!(c.shape(), &[2, 3]);

        let result = c.to_vec().unwrap();
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

        // A: 2x3 (bf16), B: 3x4 (bf16) -> C: 2x4 (f32)
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

        let result: Vec<f32> = c.to_vec().unwrap();
        let expected: Vec<f32> = vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 0.5, "Mismatch at {i}: {got} vs {exp}");
        }
    }
}
