//! SwiGLU activation: silu(gate) * up
//!
//! The decomposed form applies SiLU to the gate tensor, then multiplies
//! element-wise with the up tensor. The fused kernel (`silu_mul`) computes
//! both in a single pass, halving global memory traffic.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::missing_panics_doc
)]

use crate::cuda::CudaTensor;
use crate::Result;

/// SwiGLU activation: `silu(gate) * up`
///
/// The decomposed version calls `silu` then `mul`. When fusion is active
/// (release builds by default), the existing `silu_mul` fused kernel
/// handles both in a single pass.
///
/// # Errors
/// Returns an error if the operation fails.
infernum_macros::define_block! {
    pub fn swiglu(gate: &CudaTensor<f32>, up: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
        let activated = super::silu(gate)?;
        super::mul(&activated, up)
    }
}

infernum_macros::define_fusion! {
    block: SWIGLU_FUSED,
    fn swiglu_fused(gate: &CudaTensor<f32>, up: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
        super::silu_mul(gate, up)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;

    #[test]
    fn test_swiglu_decomposed() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let gate_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let up_data: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0];

        let gate = CudaTensor::from_slice(&ctx, &[4], &gate_data).unwrap();
        let up = CudaTensor::from_slice(&ctx, &[4], &up_data).unwrap();

        let output = swiglu_decomposed(&gate, &up).unwrap();
        let result = output.to_vec().unwrap();

        for i in 0..4 {
            let silu_gate = gate_data[i] / (1.0 + (-gate_data[i]).exp());
            let expected = silu_gate * up_data[i];
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

        let decomposed = swiglu_decomposed(&gate, &up).unwrap().to_vec().unwrap();
        let fused = swiglu_fused(&gate, &up).unwrap().to_vec().unwrap();

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
