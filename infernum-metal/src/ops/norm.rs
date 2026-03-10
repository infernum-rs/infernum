//! NormOps implementation for Metal — RMS normalization via GPU kernels.

use infernum::backend::NormOps;
use infernum::tensor::Tensor;
use infernum::DType;
use infernum::Result;
use metal::MTLSize;

use crate::context::reduction_threadgroup_size;
use crate::tensor::MetalTensor;
use crate::MetalBackend;

impl NormOps for MetalBackend {
    #[allow(clippy::cast_possible_truncation)]
    fn rms_norm(input: &MetalTensor, weight: &MetalTensor, eps: f32) -> Result<MetalTensor> {
        let shape = input.shape().to_vec();
        let hidden = *shape.last().unwrap();
        let rows = input.numel() / hidden;

        let ctx = input.context();
        let out = MetalTensor::zeros(ctx, &shape, input.dtype());

        let tg = reduction_threadgroup_size(hidden);
        let hidden_u32 = hidden as u32;

        // Pack params: hidden (u32) + eps (f32)
        let mut params = Vec::with_capacity(8);
        params.extend_from_slice(bytemuck::bytes_of(&hidden_u32));
        params.extend_from_slice(bytemuck::bytes_of(&eps));

        let kernel = if input.dtype() == DType::F16 {
            "rms_norm_f16"
        } else {
            "rms_norm_f32"
        };
        ctx.dispatch_threadgroups(
            kernel,
            &[
                (input.metal_buffer(), input.buffer_offset()),
                (weight.metal_buffer(), weight.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            &params,
            MTLSize::new(rows as u64, 1, 1),
            MTLSize::new(tg as u64, 1, 1),
            tg * std::mem::size_of::<f32>(),
        );

        Ok(out)
    }

    fn rms_norm_inplace(input: &mut MetalTensor, weight: &MetalTensor, eps: f32) -> Result<()> {
        let result = Self::rms_norm(input, weight, eps)?;
        *input = result;
        Ok(())
    }

    #[allow(clippy::cast_possible_truncation)]
    fn add_rmsnorm(
        residual: &MetalTensor,
        input: &MetalTensor,
        weight: &MetalTensor,
        eps: f32,
    ) -> Result<(MetalTensor, MetalTensor)> {
        let shape = input.shape().to_vec();
        let hidden = *shape.last().unwrap();
        let rows = input.numel() / hidden;

        let ctx = input.context();
        let updated = MetalTensor::zeros(ctx, &shape, input.dtype());
        let normed = MetalTensor::zeros(ctx, &shape, input.dtype());

        let tg = reduction_threadgroup_size(hidden);
        let hidden_u32 = hidden as u32;

        let mut params = Vec::with_capacity(8);
        params.extend_from_slice(bytemuck::bytes_of(&hidden_u32));
        params.extend_from_slice(bytemuck::bytes_of(&eps));

        let kernel = if input.dtype() == DType::F16 {
            "add_rmsnorm_f16"
        } else {
            "add_rmsnorm_f32"
        };
        ctx.dispatch_threadgroups(
            kernel,
            &[
                (residual.metal_buffer(), residual.buffer_offset()),
                (input.metal_buffer(), input.buffer_offset()),
                (weight.metal_buffer(), weight.buffer_offset()),
                (updated.metal_buffer(), updated.buffer_offset()),
                (normed.metal_buffer(), normed.buffer_offset()),
            ],
            &params,
            MTLSize::new(rows as u64, 1, 1),
            MTLSize::new(tg as u64, 1, 1),
            tg * std::mem::size_of::<f32>(),
        );

        Ok((updated, normed))
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

    #[test]
    fn test_rms_norm_large_hidden() {
        let c = ctx();
        let hidden = 256;
        let data: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let weights: Vec<f32> = vec![1.0; hidden];
        let input = MetalBackend::from_f32_slice(&c, &[hidden], &data).unwrap();
        let weight = MetalBackend::from_f32_slice(&c, &[hidden], &weights).unwrap();
        let out = MetalBackend::rms_norm(&input, &weight, 1e-6).unwrap();
        let result = MetalBackend::to_f32_vec(&out).unwrap();

        // CPU reference
        let ms: f32 = data.iter().map(|x| x * x).sum::<f32>() / hidden as f32;
        let scale = 1.0 / (ms + 1e-6).sqrt();
        for (i, (&gpu, &x)) in result.iter().zip(data.iter()).enumerate() {
            let expected = x * scale;
            assert!(
                (gpu - expected).abs() < 1e-4,
                "element {i}: gpu={gpu} vs cpu={expected}"
            );
        }
    }
}
