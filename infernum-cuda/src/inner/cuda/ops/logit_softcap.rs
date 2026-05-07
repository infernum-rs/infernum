//! Logit soft-cap: `tanh(x / cap) * cap`.
//!
//! Applied as a final logit soft-capping step (Gemma 2 style).
//! Implemented via a CPU round-trip: cast to F32, download, apply, upload.
//! Logits are small (`[seq_len, vocab_size]`) so this is not on the hot path.

use crate::cuda::{ops, CudaTensor};
use infernum::tensor::Tensor;
use infernum::Result;

/// Apply element-wise logit soft-cap: `tanh(x / cap) * cap`.
///
/// The input may be any float dtype; it is cast to `F32` for the computation
/// and the result is returned as `F32`.
///
/// # Errors
/// Returns an error if the CUDA operation fails.
pub fn logit_softcap(input: &CudaTensor, cap: f32) -> Result<CudaTensor> {
    let shape = input.shape().to_vec();
    let ctx = input.context().clone();

    let input_f32;
    let input_f32_ref = if input.dtype() == infernum::dtype::DType::F32 {
        input
    } else {
        input_f32 = ops::cast_to_f32(input)?;
        &input_f32
    };

    let host: Vec<f32> = input_f32_ref.to_vec::<f32>()?;
    let softcapped: Vec<f32> = host.iter().map(|&x| (x / cap).tanh() * cap).collect();
    CudaTensor::from_slice::<f32>(&ctx, &shape, &softcapped)
}
