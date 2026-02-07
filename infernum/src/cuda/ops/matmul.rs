//! Matrix multiplication using cuBLAS

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::missing_panics_doc,
    clippy::uninlined_format_args
)]

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::DeviceRepr;

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
            .gemm(cfg, b.cuda_slice(), a.cuda_slice(), c.cuda_slice_mut())?;
    }

    Ok(c)
}

/// Batched 3D matrix multiplication: (B, M, K) @ (B, K, N) -> (B, M, N)
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

    // For batched matmul, we iterate over batch dimension
    // (A more efficient implementation would use cublasGemmBatched or cublasGemmStridedBatched)
    let c_shape = [batch, m, n];
    let mut c = unsafe { CudaTensor::<T>::uninit(a.context(), &c_shape)? };

    // Flatten and process as a single large matmul
    // A: (batch*m, k), B: (k, n) broadcast -> C: (batch*m, n) -> reshape to (batch, m, n)
    // This is a simplification; proper batched GEMM would be more efficient

    for i in 0..batch {
        let a_offset = i * m * k;
        let b_offset = i * k * n;
        let c_offset = i * m * n;

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
            let a_slice = a.cuda_slice().slice(a_offset..a_offset + m * k);
            let b_slice = b.cuda_slice().slice(b_offset..b_offset + k * n);
            let mut c_slice = c.cuda_slice_mut().slice_mut(c_offset..c_offset + m * n);

            a.context()
                .blas()
                .gemm(cfg, &b_slice, &a_slice, &mut c_slice)?;
        }
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
}
