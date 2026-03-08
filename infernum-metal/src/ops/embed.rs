//! EmbedOps implementation for Metal — embedding lookup.

use infernum::backend::EmbedOps;
use infernum::tensor::Tensor;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;

impl EmbedOps for MetalBackend {
    #[allow(clippy::cast_possible_truncation)]
    fn embedding_gather(table: &MetalTensor, indices: &[u32]) -> Result<MetalTensor> {
        let shape = table.shape();
        let hidden = shape[1];
        let data = table.as_f32_slice();

        let mut out = vec![0.0f32; indices.len() * hidden];
        for (i, &idx) in indices.iter().enumerate() {
            let src = &data[idx as usize * hidden..(idx as usize + 1) * hidden];
            out[i * hidden..(i + 1) * hidden].copy_from_slice(src);
        }

        Ok(MetalTensor::from_f32(
            table.context(),
            &[indices.len(), hidden],
            &out,
        ))
    }

    #[allow(clippy::cast_possible_truncation)]
    fn embedding_gather_tensor(
        table: &MetalTensor,
        indices: &MetalTensor,
        seq_len: usize,
    ) -> Result<MetalTensor> {
        // Read indices from shared memory
        let idx_bytes = indices.as_bytes();
        let idx_u32: &[u32] = bytemuck::cast_slice(idx_bytes);
        Self::embedding_gather(table, &idx_u32[..seq_len])
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
}
