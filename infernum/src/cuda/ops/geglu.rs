//! GeGLU activation: gelu(gate) * up
//!
//! The fused kernel (`gelu_mul`) computes `gelu(gate) * up` in a single pass,
//! halving global memory traffic. Analogous to SwiGLU but with GELU activation.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::missing_panics_doc
)]

use crate::cuda::CudaTensor;
use crate::dtype::DType;
use crate::Result;

infernum_macros::define_block! {
    /// GeGLU activation: `gelu(gate) * up`
    ///
    /// Uses the `gelu_mul` fused kernel directly for all dtypes,
    /// computing `gelu(gate) * up` in a single pass.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn geglu(
        gate: &CudaTensor,
        up: &CudaTensor,
    ) -> Result<CudaTensor> {
        super::gelu_mul(gate, up)
    }
}

infernum_macros::define_fusion! {
    name: "geglu",
    fn geglu_fused_f32(gate: &CudaTensor, up: &CudaTensor) -> Result<CudaTensor> {
        super::gelu_mul(gate, up)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;

    fn gelu_ref(x: f32) -> f32 {
        let inner = 0.797_884_6_f32 * (x + 0.044_715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    #[test]
    fn test_geglu_decomposed() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let gate_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let up_data: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0];

        let gate = CudaTensor::from_slice(&ctx, &[4], &gate_data).unwrap();
        let up = CudaTensor::from_slice(&ctx, &[4], &up_data).unwrap();

        let output = geglu_decomposed(&gate, &up).unwrap();
        let result = output.to_vec().unwrap();

        for i in 0..4 {
            let expected = gelu_ref(gate_data[i]) * up_data[i];
            assert!(
                (result[i] - expected).abs() < 1e-5,
                "Mismatch at {i}: {} vs {expected}",
                result[i]
            );
        }
    }

    #[test]
    fn test_fused_matches_decomposed() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let gate_data: Vec<f32> = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0];
        let up_data: Vec<f32> = vec![1.0, 0.5, 2.0, 1.5, 3.0, 0.1, 0.7, 1.2];

        let gate = CudaTensor::from_slice(&ctx, &[8], &gate_data).unwrap();
        let up = CudaTensor::from_slice(&ctx, &[8], &up_data).unwrap();

        let decomposed = geglu_decomposed(&gate, &up).unwrap().to_vec().unwrap();
        let fused = geglu_fused_f32(&gate, &up).unwrap().to_vec().unwrap();

        for i in 0..8 {
            assert!(
                (decomposed[i] - fused[i]).abs() < 1e-5,
                "Mismatch at {i}: decomposed={} vs fused={}",
                decomposed[i],
                fused[i]
            );
        }
    }
}
