//! MatmulOps and MatmulExtOps implementation for CpuBackend.
//!
//! All matmul is f32 row-major, using standard `A (M,K) × B (K,N) → C (M,N)`.
//! Weight matrices are stored as `(in_features, out_features)` after the
//! host transpose applied by `WeightLoader`.

use infernum::backend::{MatmulExtOps, MatmulOps};
use infernum::dtype::DType;
use infernum::tensor::Tensor;
use infernum::Result;

use crate::tensor::CpuTensor;
use crate::CpuBackend;

/// Standard gemm: `A (M,K) × B (K,N) → C (M,N)`.
///
/// Each output element `C[m,n] = dot(A[m,:], B[:,n])`.  Since B is row-major,
/// `B[:,n]` is strided — we iterate over K explicitly and accumulate.
#[allow(clippy::many_single_char_names)]
fn gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        let a_row = &a[row * k..(row + 1) * k];
        let c_row = &mut c[row * n..(row + 1) * n];
        for ki in 0..k {
            let a_val = a_row[ki];
            let b_row = &b[ki * n..(ki + 1) * n];
            for col in 0..n {
                c_row[col] = a_val.mul_add(b_row[col], c_row[col]);
            }
        }
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
        // input: (2, 3), weight: (3, 4) → output: (2, 4)
        // Standard matrix multiply: C = A @ B
        #[rustfmt::skip]
        let input = CpuTensor::from_f32(&[2, 3], &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);
        // Weight is (K=3, N=4) — each column is a "neuron"
        #[rustfmt::skip]
        let weight = CpuTensor::from_f32(&[3, 4], &[
            1.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
        ]);
        let out = CpuBackend::matmul(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
        let data = out.as_f32_slice();
        // Row 0: [1*1+2*0+3*0, 1*0+2*1+3*0, 1*0+2*0+3*1, 1*1+2*1+3*1] = [1, 2, 3, 6]
        // Row 1: [4*1+5*0+6*0, 4*0+5*1+6*0, 4*0+5*0+6*1, 4*1+5*1+6*1] = [4, 5, 6, 15]
        assert_eq!(data, &[1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0]);
    }

    #[test]
    fn test_gemv_single_row() {
        // input: (1, 3), weight: (3, 2) → output: (1, 2)
        let input = CpuTensor::from_f32(&[1, 3], &[1.0, 2.0, 3.0]);
        #[rustfmt::skip]
        let weight = CpuTensor::from_f32(&[3, 2], &[
            1.0, 2.0,
            1.0, 2.0,
            1.0, 2.0,
        ]);
        let out = CpuBackend::matmul(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        // [1+2+3, 2+4+6] = [6, 12]
        assert_eq!(out.as_f32_slice(), &[6.0, 12.0]);
    }
}
