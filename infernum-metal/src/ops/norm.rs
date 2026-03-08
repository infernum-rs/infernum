//! NormOps implementation for Metal — RMS normalization.
//!
//! Phase 1: CPU-side via unified memory.

use infernum::backend::NormOps;
use infernum::tensor::Tensor;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;

impl NormOps for MetalBackend {
    #[allow(clippy::cast_precision_loss)]
    fn rms_norm(input: &MetalTensor, weight: &MetalTensor, eps: f32) -> Result<MetalTensor> {
        let shape = input.shape().to_vec();
        let hidden = *shape.last().unwrap();
        let rows = input.numel() / hidden;

        let data = input.as_f32_slice();
        let w = weight.as_f32_slice();

        let mut out = vec![0.0f32; data.len()];
        for r in 0..rows {
            let row = &data[r * hidden..(r + 1) * hidden];
            let ms: f32 = row.iter().map(|x| x * x).sum::<f32>() / hidden as f32;
            let scale = 1.0 / (ms + eps).sqrt();
            for (i, x) in row.iter().enumerate() {
                out[r * hidden + i] = x * scale * w[i];
            }
        }

        let device = metal::Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
        Ok(MetalTensor::from_f32(&device, &shape, &out))
    }

    fn rms_norm_inplace(input: &mut MetalTensor, weight: &MetalTensor, eps: f32) -> Result<()> {
        let result = Self::rms_norm(input, weight, eps)?;
        *input = result;
        Ok(())
    }

    #[allow(clippy::cast_precision_loss)]
    fn add_rmsnorm(
        residual: &MetalTensor,
        input: &MetalTensor,
        weight: &MetalTensor,
        eps: f32,
    ) -> Result<(MetalTensor, MetalTensor)> {
        let shape = input.shape().to_vec();
        let hidden = *shape.last().unwrap();
        let rows = input.numel() / hidden;

        let res_data = residual.as_f32_slice();
        let inp_data = input.as_f32_slice();
        let w = weight.as_f32_slice();

        let mut updated = vec![0.0f32; inp_data.len()];
        let mut normed = vec![0.0f32; inp_data.len()];

        for r in 0..rows {
            let off = r * hidden;
            // updated_residual = residual + input
            for i in 0..hidden {
                updated[off + i] = res_data[off + i] + inp_data[off + i];
            }
            // rmsnorm(updated_residual)
            let row = &updated[off..off + hidden];
            let ms: f32 = row.iter().map(|x| x * x).sum::<f32>() / hidden as f32;
            let scale = 1.0 / (ms + eps).sqrt();
            for i in 0..hidden {
                normed[off + i] = row[i] * scale * w[i];
            }
        }

        let device = metal::Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
        Ok((
            MetalTensor::from_f32(&device, &shape, &updated),
            MetalTensor::from_f32(&device, &shape, &normed),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalContext;
    use infernum::backend::{TensorDataOps, TensorFactory};

    fn ctx() -> MetalContext {
        MetalContext::new()
    }

    #[test]
    fn test_rms_norm_identity_weights() {
        let c = ctx();
        let input = MetalBackend::from_f32_slice(&c, &[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let weight = MetalBackend::from_f32_slice(&c, &[4], &[1.0, 1.0, 1.0, 1.0]).unwrap();
        let out = MetalBackend::rms_norm(&input, &weight, 1e-6).unwrap();
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let rms = (7.5f32).sqrt();
        for (a, x) in result.iter().zip([1.0, 2.0, 3.0, 4.0].iter()) {
            let expected = x / rms;
            assert!((a - expected).abs() < 1e-5, "{a} vs {expected}");
        }
    }

    #[test]
    fn test_rms_norm_2d() {
        let c = ctx();
        let input =
            MetalBackend::from_f32_slice(&c, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let weight = MetalBackend::from_f32_slice(&c, &[3], &[1.0, 1.0, 1.0]).unwrap();
        let out = MetalBackend::rms_norm(&input, &weight, 1e-6).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        // Each row is independently normalized
        assert!(!result.iter().any(|x| x.is_nan()));
    }

    #[test]
    fn test_add_rmsnorm() {
        let c = ctx();
        let res = MetalBackend::from_f32_slice(&c, &[3], &[1.0, 0.0, -1.0]).unwrap();
        let inp = MetalBackend::from_f32_slice(&c, &[3], &[0.5, 0.5, 0.5]).unwrap();
        let w = MetalBackend::from_f32_slice(&c, &[3], &[1.0, 1.0, 1.0]).unwrap();
        let (updated, normed) = MetalBackend::add_rmsnorm(&res, &inp, &w, 1e-6).unwrap();
        let u = MetalBackend::to_f32_vec(&updated).unwrap();
        assert_eq!(u, [1.5, 0.5, -0.5]);
        let n = MetalBackend::to_f32_vec(&normed).unwrap();
        assert!(!n.iter().any(|x| x.is_nan()));
    }
}
