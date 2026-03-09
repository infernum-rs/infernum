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

/// Packed params struct matching MSL `RopeQkParams`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeQkParams {
    q_heads: u32,
    k_heads: u32,
    head_dim: u32,
    half_dim: u32,
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
        let out = MetalTensor::zeros(ctx, shape, DType::F32);

        let n = seq_len * n_heads * half_dim;
        let params = RopeParams {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            half_dim: half_dim as u32,
            pos_offset: position_offset as u32,
        };

        ctx.dispatch_1d(
            "apply_rope_f32",
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

    #[allow(clippy::cast_possible_truncation)]
    fn apply_rope_qk_batched(
        q: &MetalTensor,
        k: &MetalTensor,
        cos_cache: &MetalTensor,
        sin_cache: &MetalTensor,
        positions: &MetalTensor,
        batch_size: usize,
    ) -> Result<(MetalTensor, MetalTensor)> {
        let q_shape = q.shape();
        let k_shape = k.shape();
        let q_heads = q_shape[1];
        let k_heads = k_shape[1];
        let head_dim = q_shape[2];
        let half_dim = head_dim / 2;

        let ctx = q.context();
        let q_out = MetalTensor::zeros(ctx, q_shape, DType::F32);
        let k_out = MetalTensor::zeros(ctx, k_shape, DType::F32);

        let n = batch_size * (q_heads + k_heads) * half_dim;
        let params = RopeQkParams {
            q_heads: q_heads as u32,
            k_heads: k_heads as u32,
            head_dim: head_dim as u32,
            half_dim: half_dim as u32,
        };

        ctx.dispatch_1d(
            "apply_rope_qk_batched_f32",
            &[
                (q.metal_buffer(), q.buffer_offset()),
                (k.metal_buffer(), k.buffer_offset()),
                (cos_cache.metal_buffer(), cos_cache.buffer_offset()),
                (sin_cache.metal_buffer(), sin_cache.buffer_offset()),
                (positions.metal_buffer(), positions.buffer_offset()),
                (q_out.metal_buffer(), q_out.buffer_offset()),
                (k_out.metal_buffer(), k_out.buffer_offset()),
            ],
            bytemuck::bytes_of(&params),
            n,
        );

        Ok((q_out, k_out))
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

    #[test]
    fn test_rope_qk_fused_matches_separate() {
        let c = ctx();
        // GQA config: batch=2, q_heads=4, k_heads=2, head_dim=4
        let q_data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect(); // 2*4*4=32
        let k_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.2 + 1.0).collect(); // 2*2*4=16
        let q = MetalBackend::from_f32_slice(&c, &[2, 4, 4], &q_data).unwrap();
        let k = MetalBackend::from_f32_slice(&c, &[2, 2, 4], &k_data).unwrap();
        // cos/sin at positions 3,7 — 2 positions, half_dim=2
        let cos = MetalBackend::from_f32_slice(
            &c,
            &[8, 2],
            &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // pos 0-2 unused
                0.5, 0.6, // pos 3
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // pos 4-6 unused
                0.7, 0.8, // pos 7
            ],
        )
        .unwrap();
        let sin = MetalBackend::from_f32_slice(
            &c,
            &[8, 2],
            &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // pos 0-2 unused
                0.3, 0.4, // pos 3
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // pos 4-6 unused
                0.1, 0.2, // pos 7
            ],
        )
        .unwrap();
        let positions =
            MetalTensor::from_raw_bytes(&c, &[2], DType::U32, bytemuck::cast_slice(&[3i32, 7]));

        // Separate path
        let q_sep = MetalBackend::apply_rope_batched(&q, &cos, &sin, &positions, 2).unwrap();
        let k_sep = MetalBackend::apply_rope_batched(&k, &cos, &sin, &positions, 2).unwrap();
        let q_sep_data = MetalBackend::to_f32_vec(&q_sep).unwrap();
        let k_sep_data = MetalBackend::to_f32_vec(&k_sep).unwrap();

        // Fused path
        let (q_fused, k_fused) =
            MetalBackend::apply_rope_qk_batched(&q, &k, &cos, &sin, &positions, 2).unwrap();
        let q_fused_data = MetalBackend::to_f32_vec(&q_fused).unwrap();
        let k_fused_data = MetalBackend::to_f32_vec(&k_fused).unwrap();

        // Compare
        for (i, (a, b)) in q_sep_data.iter().zip(&q_fused_data).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "Q mismatch at {i}: sep={a}, fused={b}"
            );
        }
        for (i, (a, b)) in k_sep_data.iter().zip(&k_fused_data).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "K mismatch at {i}: sep={a}, fused={b}"
            );
        }
    }
}
