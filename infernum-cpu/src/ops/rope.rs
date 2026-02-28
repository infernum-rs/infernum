//! RopeOps implementation for CpuBackend (half-rotation layout).

use infernum::backend::RopeOps;
use infernum::tensor::Tensor;
use infernum::Result;

use crate::tensor::CpuTensor;
use crate::CpuBackend;

impl RopeOps for CpuBackend {
    fn apply_rope(
        input: &CpuTensor,
        cos_cache: &CpuTensor,
        sin_cache: &CpuTensor,
        position_offset: usize,
    ) -> Result<CpuTensor> {
        // input: (seq_len, num_heads, head_dim)
        let shape = input.shape();
        let seq_len = shape[0];
        let num_heads = shape[1];
        let head_dim = shape[2];
        let half_dim = head_dim / 2;

        let input_data = input.as_f32_slice();
        let cos_data = cos_cache.to_f32_vec();
        let sin_data = sin_cache.to_f32_vec();

        let mut out = vec![0.0f32; input_data.len()];

        for s in 0..seq_len {
            let pos = position_offset + s;
            let cos_row = &cos_data[pos * half_dim..(pos + 1) * half_dim];
            let sin_row = &sin_data[pos * half_dim..(pos + 1) * half_dim];

            for h in 0..num_heads {
                let base = (s * num_heads + h) * head_dim;
                for d in 0..half_dim {
                    let x0 = input_data[base + d];
                    let x1 = input_data[base + half_dim + d];
                    out[base + d] = x0 * cos_row[d] - x1 * sin_row[d];
                    out[base + half_dim + d] = x1 * cos_row[d] + x0 * sin_row[d];
                }
            }
        }

        Ok(CpuTensor::from_f32(shape, &out))
    }

    fn apply_rope_batched(
        input: &CpuTensor,
        cos_cache: &CpuTensor,
        sin_cache: &CpuTensor,
        positions: &CpuTensor,
        batch_size: usize,
    ) -> Result<CpuTensor> {
        // input: (batch_size, num_heads, head_dim)
        // positions: (batch_size,) i32
        let shape = input.shape();
        let num_heads = shape[1];
        let head_dim = shape[2];
        let half_dim = head_dim / 2;

        let input_data = input.as_f32_slice();
        let cos_data = cos_cache.to_f32_vec();
        let sin_data = sin_cache.to_f32_vec();
        let pos_data = positions.as_i32_slice();

        let mut out = vec![0.0f32; input_data.len()];

        #[allow(clippy::needless_range_loop)]
        for b in 0..batch_size {
            #[allow(clippy::cast_sign_loss)]
            let pos = pos_data[b] as usize;
            let cos_row = &cos_data[pos * half_dim..(pos + 1) * half_dim];
            let sin_row = &sin_data[pos * half_dim..(pos + 1) * half_dim];

            for h in 0..num_heads {
                let base = (b * num_heads + h) * head_dim;
                for d in 0..half_dim {
                    let x0 = input_data[base + d];
                    let x1 = input_data[base + half_dim + d];
                    out[base + d] = x0 * cos_row[d] - x1 * sin_row[d];
                    out[base + half_dim + d] = x1 * cos_row[d] + x0 * sin_row[d];
                }
            }
        }

        Ok(CpuTensor::from_f32(shape, &out))
    }
}
