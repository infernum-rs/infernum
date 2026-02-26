//! CUDA implementation of the `Logits` trait.
//!
//! Wraps a `CudaTensor` of shape `(batch_size, vocab_size)` and implements
//! per-sequence argmax (via GPU kernel) and top-p sampling (via partial
//! D2H copy).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::missing_panics_doc
)]

use crate::cuda::ops::{argmax_last_scalar, sample_top_p};
use crate::cuda::CudaTensor;
use infernum::logits::Logits;
use infernum::tensor::Tensor;
use infernum::Result;

/// CUDA-backed logits: a 2D tensor of shape `(batch_size, vocab_size)`.
///
/// - `argmax` runs the GPU reduction kernel, transferring only 4 bytes.
/// - `sample_top_p` copies one vocab-row to CPU and does sort/cumsum there.
pub struct CudaLogits {
    tensor: CudaTensor,
}

impl CudaLogits {
    /// Wrap a raw logits tensor.
    ///
    /// # Panics
    /// Panics if `tensor` is not 2D.
    #[must_use]
    pub fn new(tensor: CudaTensor) -> Self {
        assert!(
            tensor.shape().len() == 2,
            "CudaLogits expects 2D tensor, got shape {:?}",
            tensor.shape()
        );
        Self { tensor }
    }

    /// Access the underlying `CudaTensor`.
    #[must_use]
    pub fn tensor(&self) -> &CudaTensor {
        &self.tensor
    }
}

impl Logits for CudaLogits {
    fn vocab_size(&self) -> usize {
        self.tensor.shape()[1]
    }

    fn batch_size(&self) -> usize {
        self.tensor.shape()[0]
    }

    fn argmax(&self, batch_index: usize) -> Result<u32> {
        let vocab = self.vocab_size();
        let seq_logits = self.tensor.slice_view(batch_index * vocab, &[1, vocab]);
        argmax_last_scalar(&seq_logits)
    }

    fn sample_top_p(
        &self,
        batch_index: usize,
        temperature: f32,
        top_p: f32,
        rng_seed: u64,
        repetition_penalty: f32,
        recent_tokens: &[u32],
    ) -> Result<u32> {
        let vocab = self.vocab_size();
        let seq_logits = self.tensor.slice_view(batch_index * vocab, &[1, vocab]);
        sample_top_p(
            &seq_logits,
            temperature,
            top_p,
            rng_seed,
            repetition_penalty,
            recent_tokens,
        )
    }
}
