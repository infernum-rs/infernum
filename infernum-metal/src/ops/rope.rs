//! RopeOps and RopeInterleavedOps implementations for Metal — GPU kernel dispatch.

use infernum::backend::{RopeInterleavedOps, RopeOps};
use infernum::tensor::Tensor;
use infernum::DType;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;

/// Packed params struct matching MSL `RopeParams`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeParams {
    n_heads: u32,
    head_dim: u32,
    half_dim: u32,
    pos_offset: u32,
}

/// Packed params struct matching MSL `RopeInterleavedParams`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeInterleavedParams {
    n_heads: u32,
    head_dim: u32,
    n_pairs: u32,
    pos_offset: u32,
}

impl RopeOps for MetalBackend {
    #[allow(clippy::cast_possible_truncation)]
    fn apply_rope(
        input: &MetalTensor,
        cos_cache: &MetalTensor,
        sin_cache: &MetalTensor,
        position_offset: usize,
    ) -> Result<MetalTensor> {
        let shape = input.shape();
        let seq_len = shape[0];
        let n_heads = shape[1];
        let head_dim = shape[2];
        let half_dim = head_dim / 2;

        let ctx = input.context();
        let out = MetalTensor::zeros(ctx, shape, input.dtype());

        let n = seq_len * n_heads * half_dim;
        let params = RopeParams {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            half_dim: half_dim as u32,
            pos_offset: position_offset as u32,
        };

        let kernel = if input.dtype() == DType::F16 {
            "apply_rope_f16"
        } else {
            "apply_rope_f32"
        };
        ctx.dispatch_1d(
            kernel,
            &[
                (input.metal_buffer(), input.buffer_offset()),
                (cos_cache.metal_buffer(), cos_cache.buffer_offset()),
                (sin_cache.metal_buffer(), sin_cache.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            bytemuck::bytes_of(&params),
            n,
        );

        Ok(out)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn apply_rope_batched(
        input: &MetalTensor,
        cos_cache: &MetalTensor,
        sin_cache: &MetalTensor,
        positions: &MetalTensor,
        batch_size: usize,
    ) -> Result<MetalTensor> {
        let shape = input.shape();
        let n_heads = shape[1];
        let head_dim = shape[2];
        let half_dim = head_dim / 2;

        let ctx = input.context();
        let out = MetalTensor::zeros(ctx, shape, DType::F32);

        let n = batch_size * n_heads * half_dim;
        let params = RopeParams {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            half_dim: half_dim as u32,
            pos_offset: 0, // not used in batched variant
        };

        ctx.dispatch_1d(
            "apply_rope_batched_f32",
            &[
                (input.metal_buffer(), input.buffer_offset()),
                (cos_cache.metal_buffer(), cos_cache.buffer_offset()),
                (sin_cache.metal_buffer(), sin_cache.buffer_offset()),
                (positions.metal_buffer(), positions.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            bytemuck::bytes_of(&params),
            n,
        );

        Ok(out)
    }
}

impl RopeInterleavedOps for MetalBackend {
    #[allow(clippy::cast_possible_truncation)]
    fn apply_rope_interleaved(
        input: &MetalTensor,
        cos_cache: &MetalTensor,
        sin_cache: &MetalTensor,
        position_offset: usize,
    ) -> Result<MetalTensor> {
        let shape = input.shape();
        let seq_len = shape[0];
        let n_heads = shape[1];
        let head_dim = shape[2];
        let n_pairs = head_dim / 2;

        let ctx = input.context();
        let out = MetalTensor::zeros(ctx, shape, DType::F32);

        let n = seq_len * n_heads * n_pairs;
        let params = RopeInterleavedParams {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            n_pairs: n_pairs as u32,
            pos_offset: position_offset as u32,
        };

        ctx.dispatch_1d(
            "apply_rope_interleaved_f32",
            &[
                (input.metal_buffer(), input.buffer_offset()),
                (cos_cache.metal_buffer(), cos_cache.buffer_offset()),
                (sin_cache.metal_buffer(), sin_cache.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            bytemuck::bytes_of(&params),
            n,
        );

        Ok(out)
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
    fn test_rope_identity_at_zero() {
        let c = ctx();
        // At position 0 with cos=1, sin=0, output should equal input
        let input = MetalBackend::from_f32_slice(&c, &[1, 1, 4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let cos = MetalBackend::from_f32_slice(&c, &[1, 2], &[1.0, 1.0]).unwrap();
        let sin = MetalBackend::from_f32_slice(&c, &[1, 2], &[0.0, 0.0]).unwrap();
        let out = MetalBackend::apply_rope(&input, &cos, &sin, 0).unwrap();
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        assert_eq!(result, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_rope_90_degree_rotation() {
        let c = ctx();
        // cos=0, sin=1 → 90° rotation
        let input = MetalBackend::from_f32_slice(&c, &[1, 1, 4], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let cos = MetalBackend::from_f32_slice(&c, &[1, 2], &[0.0, 0.0]).unwrap();
        let sin = MetalBackend::from_f32_slice(&c, &[1, 2], &[1.0, 1.0]).unwrap();
        let out = MetalBackend::apply_rope(&input, &cos, &sin, 0).unwrap();
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        // x0'= x0*cos - x1*sin = 1*0 - 0*1 = 0
        // x1'= x1*cos + x0*sin = 0*0 + 1*1 = 1
        // x2'= x2*cos - x3*sin = 0*0 - 1*1 = -1
        // x3'= x3*cos + x2*sin = 1*0 + 0*1 = 0
        assert_eq!(result, [0.0, -1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_rope_interleaved() {
        let c = ctx();
        let input = MetalBackend::from_f32_slice(&c, &[1, 1, 4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let cos = MetalBackend::from_f32_slice(&c, &[1, 2], &[1.0, 1.0]).unwrap();
        let sin = MetalBackend::from_f32_slice(&c, &[1, 2], &[0.0, 0.0]).unwrap();
        let out = MetalBackend::apply_rope_interleaved(&input, &cos, &sin, 0).unwrap();
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        assert_eq!(result, [1.0, 2.0, 3.0, 4.0]);
    }
}
