//! TensorDataOps implementation for Metal — download tensor data to host.
//!
//! With unified memory (StorageModeShared), "downloading" is just reading
//! the shared buffer pointer directly — no DMA transfer needed.

use infernum::backend::TensorDataOps;
use infernum::tensor::Tensor;
use infernum::DType;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;

impl TensorDataOps for MetalBackend {
    fn to_f32_vec(tensor: &MetalTensor) -> Result<Vec<f32>> {
        let bytes = tensor.as_bytes();
        match tensor.dtype() {
            DType::F32 => Ok(bytemuck::cast_slice(bytes).to_vec()),
            DType::BF16 => {
                let bf16s: &[half::bf16] = bytemuck::cast_slice(bytes);
                Ok(bf16s.iter().map(|v| v.to_f32()).collect())
            }
            DType::F16 => {
                let f16s: &[half::f16] = bytemuck::cast_slice(bytes);
                Ok(f16s.iter().map(|v| v.to_f32()).collect())
            }
            other => Err(infernum::Error::UnsupportedDtype(format!(
                "to_f32_vec: unsupported dtype {other}"
            ))),
        }
    }

    fn to_raw_bytes(tensor: &MetalTensor) -> Result<Vec<u8>> {
        Ok(tensor.as_bytes().to_vec())
    }
}
