//! EmbedOps implementation for CpuBackend.

use infernum::backend::EmbedOps;
use infernum::tensor::Tensor;
use infernum::Result;

use crate::tensor::CpuTensor;
use crate::CpuBackend;

impl EmbedOps for CpuBackend {
    fn embedding_gather(table: &CpuTensor, indices: &[u32]) -> Result<CpuTensor> {
        let table_data = table.as_f32_slice();
        let hidden_size = table.shape()[1];

        let mut out = Vec::with_capacity(indices.len() * hidden_size);
        for &idx in indices {
            let start = idx as usize * hidden_size;
            out.extend_from_slice(&table_data[start..start + hidden_size]);
        }
        Ok(CpuTensor::from_f32(&[indices.len(), hidden_size], &out))
    }

    fn embedding_gather_tensor(
        table: &CpuTensor,
        indices: &CpuTensor,
        seq_len: usize,
    ) -> Result<CpuTensor> {
        let idx_slice = indices.as_u32_slice();
        Self::embedding_gather(table, &idx_slice[..seq_len])
    }
}
