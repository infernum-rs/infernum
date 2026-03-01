//! MatmulOps and MatmulExtOps implementation for CpuBackend.
//!
//! All matmul is f32 row-major, using standard `A (M,K) × B (K,N) → C (M,N)`.
//! Weight matrices are stored as `(in_features, out_features)` after the
//! host transpose applied by `WeightLoader`.
//!
//! Performance: B is transposed to `Bᵀ(N,K)` so that `C[m,n] = dot(A[m,:], Bᵀ[n,:])`
//! becomes a contiguous SIMD dot product. Output rows are parallelized with Rayon.

use infernum::backend::{MatmulExtOps, MatmulOps};
use infernum::dtype::DType;
use infernum::tensor::Tensor;
use infernum::Result;
use rayon::prelude::*;

use crate::simd;
use crate::tensor::CpuTensor;
use crate::CpuBackend;

/// Transpose `B (K,N)` → `Bᵀ (N,K)` in row-major order.
#[allow(clippy::many_single_char_names)]
fn transpose(b: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut bt = vec![0.0f32; n * k];
    for row in 0..k {
        for col in 0..n {
            bt[col * k + row] = b[row * n + col];
        }
    }
    bt
}

/// Compute one row of C: `C[m,:] = A[m,:] × Bᵀ` using SIMD dot products.
#[allow(clippy::many_single_char_names)]
fn gemm_row(a_row: &[f32], bt: &[f32], c_row: &mut [f32], k: usize, n: usize) {
    for col in 0..n {
        let bt_row = &bt[col * k..(col + 1) * k];
        c_row[col] = simd::dot_f32(a_row, bt_row);
    }
}

/// Standard gemm: `A (M,K) × B (K,N) → C (M,N)`.
///
/// Transposes B once, then computes each `C[m,n] = dot(A[m,:], Bᵀ[n,:])`
/// using SIMD. Output rows are parallelized with Rayon when M > 1.
#[allow(clippy::many_single_char_names)]
fn gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let bt = transpose(b, k, n);
    let mut c = vec![0.0f32; m * n];

    if m == 1 {
        // GEMV (decode phase): single row, skip thread-pool overhead
        gemm_row(&a[..k], &bt, &mut c[..n], k, n);
    } else {
        // Parallel over output rows
        c.par_chunks_mut(n).enumerate().for_each(|(row, c_row)| {
            let a_row = &a[row * k..(row + 1) * k];
            gemm_row(a_row, &bt, c_row, k, n);
        });
    }

    c
}

impl MatmulOps for CpuBackend {
    type LinearWeight = CpuTensor;

    fn matmul(input: &CpuTensor, weight: &CpuTensor) -> Result<CpuTensor> {
        // Standard: input (..., K) × weight (K, N) → (..., N)
        let w_shape = weight.shape();
        assert!(
            w_shape.len() == 2,
            "matmul: weight must be 2D, got {w_shape:?}"
        );
        let k = w_shape[0];
        let n = w_shape[1];

        let i_shape = input.shape();
        let m: usize = i_shape[..i_shape.len() - 1].iter().product();

        assert_eq!(
            *i_shape.last().unwrap(),
            k,
            "matmul: input last dim {} != weight rows {}, input shape {:?}, weight shape {:?}",
            i_shape.last().unwrap(),
            k,
            i_shape,
            w_shape,
        );

        let output = gemm(input.as_f32_slice(), weight.as_f32_slice(), m, k, n);

        let mut out_shape = i_shape[..i_shape.len() - 1].to_vec();
        out_shape.push(n);
        Ok(CpuTensor::from_f32(&out_shape, &output))
    }

    fn linear(input: &CpuTensor, weight: &CpuTensor) -> Result<CpuTensor> {
        Self::matmul(input, weight)
    }

    fn as_dense_weight(weight: &CpuTensor) -> Option<&CpuTensor> {
        Some(weight)
    }

    fn dense_weight(tensor: CpuTensor) -> CpuTensor {
        tensor
    }

    fn is_dense_weight(_weight: &CpuTensor) -> bool {
        true
    }

    fn quantize_to_q8(_device: &(), shape: &[usize], data: &[f32]) -> Result<CpuTensor> {
        // CPU backend: store as f32 (no quantization acceleration)
        Ok(CpuTensor::from_f32(shape, data))
    }

    fn upload_host_linear(
        _device: &(),
        weight: &infernum::weights::host::HostLinearWeight,
    ) -> Result<CpuTensor> {
        use infernum::weights::host::HostLinearWeight;

        match weight {
            HostLinearWeight::Dense(host_tensor) => {
                // Weight is already in (out_features, in_features) layout,
                // same as PyTorch nn.Linear.weight and our gemm convention.
                let f32_data = match host_tensor.dtype {
                    DType::F32 => bytemuck::cast_slice::<u8, f32>(&host_tensor.data).to_vec(),
                    DType::BF16 => {
                        let bf16s: &[half::bf16] = bytemuck::cast_slice(&host_tensor.data);
                        bf16s.iter().map(|v| v.to_f32()).collect()
                    }
                    DType::F16 => {
                        let f16s: &[half::f16] = bytemuck::cast_slice(&host_tensor.data);
                        f16s.iter().map(|v| v.to_f32()).collect()
                    }
                    other => {
                        return Err(infernum::Error::UnsupportedDtype(format!(
                            "upload_host_linear: unsupported dense dtype {other}"
                        )));
                    }
                };
                Ok(CpuTensor::from_f32(&host_tensor.shape, &f32_data))
            }
            HostLinearWeight::Quantized(_) => Err(infernum::Error::UnsupportedDtype(
                "CPU backend only supports dense weights".into(),
            )),
        }
    }
}

#[allow(clippy::many_single_char_names)]
impl MatmulExtOps for CpuBackend {
    fn matmul_bf16_f32(a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        // Cast both inputs to f32 then standard matmul
        let a_f32 = a.to_f32_vec();
        let b_f32 = b.to_f32_vec();

        let a_shape = a.shape();
        let b_shape = b.shape();
        let k = b_shape[0];
        let n = b_shape[1];
        let m: usize = a_shape[..a_shape.len() - 1].iter().product();

        let output = gemm(&a_f32, &b_f32, m, k, n);

        let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
        out_shape.push(n);
        Ok(CpuTensor::from_f32(&out_shape, &output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2x3_times_3x4() {
        #[rustfmt::skip]
        let input = CpuTensor::from_f32(&[2, 3], &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);
        #[rustfmt::skip]
        let weight = CpuTensor::from_f32(&[3, 4], &[
            1.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
        ]);
        let out = CpuBackend::matmul(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
        assert_eq!(
            out.as_f32_slice(),
            &[1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0]
        );
    }

    #[test]
    fn test_gemv_single_row() {
        let input = CpuTensor::from_f32(&[1, 3], &[1.0, 2.0, 3.0]);
        #[rustfmt::skip]
        let weight = CpuTensor::from_f32(&[3, 2], &[
            1.0, 2.0,
            1.0, 2.0,
            1.0, 2.0,
        ]);
        let out = CpuBackend::matmul(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(out.as_f32_slice(), &[6.0, 12.0]);
    }

    #[test]
    fn test_identity_matrix() {
        // A × I = A
        let a_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let input = CpuTensor::from_f32(&[3, 4], &a_data);
        #[rustfmt::skip]
        let identity = CpuTensor::from_f32(&[4, 4], &[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]);
        let out = CpuBackend::matmul(&input, &identity).unwrap();
        assert_eq!(out.shape(), &[3, 4]);
        assert_eq!(out.as_f32_slice(), &a_data);
    }

    #[test]
    fn test_column_vector_output() {
        // (4, 3) × (3, 1) → (4, 1)
        let input = CpuTensor::from_f32(&[4, 3], &[1.0; 12]);
        let weight = CpuTensor::from_f32(&[3, 1], &[1.0, 2.0, 3.0]);
        let out = CpuBackend::matmul(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[4, 1]);
        assert_eq!(out.as_f32_slice(), &[6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_non_simd_aligned_k() {
        // K=7 is not aligned to NEON (4) or AVX2 (8) width
        let k = 7;
        let m = 3;
        let n = 5;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let input = CpuTensor::from_f32(&[m, k], &a_data);
        let weight = CpuTensor::from_f32(&[k, n], &b_data);
        let out = CpuBackend::matmul(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[m, n]);

        // Verify against scalar reference
        let result = out.as_f32_slice();
        for row in 0..m {
            for col in 0..n {
                let mut expected = 0.0f64;
                for ki in 0..k {
                    expected += f64::from(a_data[row * k + ki]) * f64::from(b_data[ki * n + col]);
                }
                let diff = (f64::from(result[row * n + col]) - expected).abs();
                assert!(diff < 1e-4, "mismatch at [{row},{col}]: {diff}");
            }
        }
    }

    #[test]
    fn test_large_matrix_parallel() {
        // 64×960 × 960×960 — exercises Rayon parallel path and SIMD
        let m = 64;
        let k = 960;
        let n = 960;
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32) * 0.001).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 113) as f32) * 0.001).collect();
        let input = CpuTensor::from_f32(&[m, k], &a_data);
        let weight = CpuTensor::from_f32(&[k, n], &b_data);
        let out = CpuBackend::matmul(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[m, n]);

        // Spot-check a few elements against scalar reference
        let result = out.as_f32_slice();
        for &(row, col) in &[
            (0, 0),
            (0, n - 1),
            (m - 1, 0),
            (m - 1, n - 1),
            (m / 2, n / 2),
        ] {
            let mut expected = 0.0f64;
            for ki in 0..k {
                expected += f64::from(a_data[row * k + ki]) * f64::from(b_data[ki * n + col]);
            }
            let diff = (f64::from(result[row * n + col]) - expected).abs();
            assert!(
                diff < 0.5,
                "mismatch at [{row},{col}]: got={}, expected={expected}, diff={diff}",
                result[row * n + col]
            );
        }
    }
}
