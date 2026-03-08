//! MatmulOps implementation for Metal.
//!
//! Phase 1: CPU-side naive matmul via unified memory. This is correct
//! but slow — the hot path for Metal performance will be MPSMatrixMultiplication
//! or custom MSL kernels added later.

use infernum::backend::MatmulOps;
use infernum::tensor::Tensor;
use infernum::weights::host::HostLinearWeight;
use infernum::DType;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::weights::MetalLinearWeight;
use crate::MetalBackend;
use crate::MetalContext;

impl MatmulOps for MetalBackend {
    type LinearWeight = MetalLinearWeight;

    #[allow(clippy::many_single_char_names)]
    fn matmul(a: &MetalTensor, b: &MetalTensor) -> Result<MetalTensor> {
        // a: (..., M, K), b: (K, N) → (..., M, N)
        let a_shape = a.shape();
        let b_shape = b.shape();
        let k = *a_shape.last().unwrap();
        let n = b_shape[1];
        let m = a.numel() / k;
        assert_eq!(k, b_shape[0], "matmul: K mismatch");

        let a_data = a.as_f32_slice();
        let b_data = b.as_f32_slice();
        let mut out = vec![0.0f32; m * n];

        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for i in 0..k {
                    sum += a_data[row * k + i] * b_data[i * n + col];
                }
                out[row * n + col] = sum;
            }
        }

        let device = metal::Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
        let mut out_shape = a_shape.to_vec();
        *out_shape.last_mut().unwrap() = n;
        Ok(MetalTensor::from_f32(&device, &out_shape, &out))
    }

    fn linear(input: &MetalTensor, weight: &MetalLinearWeight) -> Result<MetalTensor> {
        match weight {
            MetalLinearWeight::Dense { weight_t, .. } => {
                // weight_t: (N, K) — pre-transposed
                let wt_shape = weight_t.shape();
                let n = wt_shape[0];
                let k = wt_shape[1];
                let m = input.numel() / k;

                let in_data = input.as_f32_slice();
                let wt_data = weight_t.as_f32_slice();
                let mut out = vec![0.0f32; m * n];

                // dot(input_row, weight_t_row) for each (m, n) pair
                for row in 0..m {
                    for col in 0..n {
                        let mut sum = 0.0f32;
                        for i in 0..k {
                            sum += in_data[row * k + i] * wt_data[col * k + i];
                        }
                        out[row * n + col] = sum;
                    }
                }

                let device = metal::Device::system_default()
                    .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
                let mut out_shape = input.shape().to_vec();
                *out_shape.last_mut().unwrap() = n;
                Ok(MetalTensor::from_f32(&device, &out_shape, &out))
            }
            MetalLinearWeight::Quantized { .. } => Err(infernum::Error::Other(
                "Metal quantized linear not yet implemented".into(),
            )),
        }
    }

    fn as_dense_weight(weight: &MetalLinearWeight) -> Option<&MetalTensor> {
        match weight {
            MetalLinearWeight::Dense { weight, .. } => Some(weight),
            MetalLinearWeight::Quantized { .. } => None,
        }
    }

    fn dense_weight(tensor: MetalTensor) -> MetalLinearWeight {
        MetalLinearWeight::new_dense(tensor)
    }

    fn is_dense_weight(weight: &MetalLinearWeight) -> bool {
        matches!(weight, MetalLinearWeight::Dense { .. })
    }

    fn quantize_to_q8(
        device: &MetalContext,
        shape: &[usize],
        data: &[f32],
    ) -> Result<MetalLinearWeight> {
        // Phase 1: store as dense f32
        let out_features = shape[0];
        let in_features = shape[1];

        // Transpose: (out, in) → (in, out) for matmul-ready layout
        let mut transposed = vec![0.0f32; data.len()];
        for r in 0..out_features {
            for c in 0..in_features {
                transposed[c * out_features + r] = data[r * in_features + c];
            }
        }

        let weight =
            MetalTensor::from_f32(device.device(), &[in_features, out_features], &transposed);
        Ok(MetalLinearWeight::new_dense(weight))
    }

    fn upload_host_linear(
        device: &MetalContext,
        weight: &HostLinearWeight,
    ) -> Result<MetalLinearWeight> {
        match weight {
            HostLinearWeight::Dense(host) => {
                let tensor = MetalTensor::from_raw_bytes(
                    device.device(),
                    &host.shape,
                    host.dtype,
                    &host.data,
                );
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
            HostLinearWeight::Quantized(_q) => Err(infernum::Error::Other(
                "Metal quantized weight upload not yet implemented".into(),
            )),
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
}
