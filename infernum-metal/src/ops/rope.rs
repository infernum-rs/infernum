//! RopeOps and RopeInterleavedOps implementations for Metal.
//!
//! Phase 1: CPU-side via unified memory.

use infernum::backend::{RopeInterleavedOps, RopeOps};
use infernum::tensor::Tensor;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;

impl RopeOps for MetalBackend {
    #[allow(clippy::cast_possible_truncation)]
    fn apply_rope(
        input: &MetalTensor,
        cos_cache: &MetalTensor,
        sin_cache: &MetalTensor,
        position_offset: usize,
    ) -> Result<MetalTensor> {
        // input: (seq, heads, head_dim)
        let shape = input.shape();
        let seq_len = shape[0];
        let n_heads = shape[1];
        let head_dim = shape[2];
        let half_dim = head_dim / 2;

        let data = input.as_f32_slice();
        let cos = cos_cache.as_f32_slice();
        let sin = sin_cache.as_f32_slice();

        let mut out = data.to_vec();

        for s in 0..seq_len {
            let pos = s + position_offset;
            for h in 0..n_heads {
                let base = (s * n_heads + h) * head_dim;
                for d in 0..half_dim {
                    let cos_val = cos[pos * half_dim + d];
                    let sin_val = sin[pos * half_dim + d];
                    let x0 = data[base + d];
                    let x1 = data[base + half_dim + d];
                    out[base + d] = x0 * cos_val - x1 * sin_val;
                    out[base + half_dim + d] = x1 * cos_val + x0 * sin_val;
                }
            }
        }

        Ok(MetalTensor::from_f32(input.context(), shape, &out))
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::needless_range_loop
    )]
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

        let data = input.as_f32_slice();
        let cos = cos_cache.as_f32_slice();
        let sin = sin_cache.as_f32_slice();
        let pos_data: &[i32] = bytemuck::cast_slice(positions.as_bytes());

        let mut out = data.to_vec();

        for b in 0..batch_size {
            let pos = pos_data[b] as usize;
            for h in 0..n_heads {
                let base = (b * n_heads + h) * head_dim;
                for d in 0..half_dim {
                    let cos_val = cos[pos * half_dim + d];
                    let sin_val = sin[pos * half_dim + d];
                    let x0 = data[base + d];
                    let x1 = data[base + half_dim + d];
                    out[base + d] = x0 * cos_val - x1 * sin_val;
                    out[base + half_dim + d] = x1 * cos_val + x0 * sin_val;
                }
            }
        }

        Ok(MetalTensor::from_f32(input.context(), shape, &out))
    }
}

impl RopeInterleavedOps for MetalBackend {
    fn apply_rope_interleaved(
        input: &MetalTensor,
        cos_cache: &MetalTensor,
        sin_cache: &MetalTensor,
        position_offset: usize,
    ) -> Result<MetalTensor> {
        // Interleaved: pairs are (x[0], x[1]), (x[2], x[3]), ...
        let shape = input.shape();
        let seq_len = shape[0];
        let n_heads = shape[1];
        let head_dim = shape[2];
        let n_pairs = head_dim / 2;

        let data = input.as_f32_slice();
        let cos = cos_cache.as_f32_slice();
        let sin = sin_cache.as_f32_slice();

        let mut out = data.to_vec();

        for s in 0..seq_len {
            let pos = s + position_offset;
            for h in 0..n_heads {
                let base = (s * n_heads + h) * head_dim;
                for p in 0..n_pairs {
                    let cos_val = cos[pos * n_pairs + p];
                    let sin_val = sin[pos * n_pairs + p];
                    let x0 = data[base + 2 * p];
                    let x1 = data[base + 2 * p + 1];
                    out[base + 2 * p] = x0 * cos_val - x1 * sin_val;
                    out[base + 2 * p + 1] = x1 * cos_val + x0 * sin_val;
                }
            }
        }

        Ok(MetalTensor::from_f32(input.context(), shape, &out))
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
