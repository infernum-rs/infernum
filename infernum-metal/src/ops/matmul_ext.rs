//! MatmulExtOps implementation for Metal.
//!
//! Phase 1: CPU-side via unified memory.

use infernum::backend::MatmulExtOps;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;

impl MatmulExtOps for MetalBackend {
    fn matmul_bf16_f32(a: &MetalTensor, b: &MetalTensor) -> Result<MetalTensor> {
        // In Phase 1, everything is f32 internally. Just delegate to matmul.
        use infernum::backend::MatmulOps;
        Self::matmul(a, b)
    }
}
