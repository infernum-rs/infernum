//! EmbedOps implementation for Metal — embedding lookup via GPU kernel.

use infernum::backend::EmbedOps;
use infernum::tensor::Tensor;
use infernum::DType;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;

impl EmbedOps for MetalBackend {
    #[allow(clippy::cast_possible_truncation)]
    fn embedding_gather(table: &MetalTensor, indices: &[u32]) -> Result<MetalTensor> {
        let hidden = table.shape()[1];
        let n_tokens = indices.len();
        let ctx = table.context();

        // Upload indices to a temporary Metal buffer
        let idx_tensor = MetalTensor::from_raw_bytes(
            ctx,
            &[n_tokens],
            DType::U32,
            bytemuck::cast_slice(indices),
        );

        let out = MetalTensor::zeros(ctx, &[n_tokens, hidden], DType::F32);
        let hidden_u32 = hidden as u32;

        ctx.dispatch_1d(
            "embedding_gather_f32",
            &[
                (table.metal_buffer(), table.buffer_offset()),
                (idx_tensor.metal_buffer(), idx_tensor.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            bytemuck::bytes_of(&hidden_u32),
            n_tokens * hidden,
        );

        Ok(out)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn embedding_gather_tensor(
        table: &MetalTensor,
        indices: &MetalTensor,
        seq_len: usize,
    ) -> Result<MetalTensor> {
        let hidden = table.shape()[1];
        let ctx = table.context();
        let out = MetalTensor::zeros(ctx, &[seq_len, hidden], DType::F32);
        let hidden_u32 = hidden as u32;

        ctx.dispatch_1d(
            "embedding_gather_f32",
            &[
                (table.metal_buffer(), table.buffer_offset()),
                (indices.metal_buffer(), indices.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            bytemuck::bytes_of(&hidden_u32),
            seq_len * hidden,
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
    fn test_embedding_gather() {
        let c = ctx();
        // vocab=3, hidden=2
        let table =
            MetalBackend::from_f32_slice(&c, &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let out = MetalBackend::embedding_gather(&table, &[2, 0]).unwrap();
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        assert_eq!(result, [5.0, 6.0, 1.0, 2.0]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_embedding_gather_tensor() {
        let c = ctx();
        let table =
            MetalBackend::from_f32_slice(&c, &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let indices = MetalBackend::from_u32_slice(&c, &[2], &[1, 0]).unwrap();
        let out = MetalBackend::embedding_gather_tensor(&table, &indices, 2).unwrap();
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        assert_eq!(result, [3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_embedding_gather_duplicate_indices() {
        let c = ctx();
        let table =
            MetalBackend::from_f32_slice(&c, &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let out = MetalBackend::embedding_gather(&table, &[1, 1, 2]).unwrap();
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        assert_eq!(result, [3.0, 4.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(out.shape(), &[3, 2]);
    }
}
