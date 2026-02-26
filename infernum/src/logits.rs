//! Opaque logits trait for backend-agnostic sampling.
//!
//! The Engine calls `argmax` or `sample_top_p` per sequence — it decides
//! *what* to do (greedy vs sampling), the backend decides *how* (GPU
//! kernel, CPU scan, etc.).

use crate::Result;

/// Opaque logits from a forward pass.
///
/// Each backend returns its own logits type (e.g., a GPU tensor or a host
/// `Vec<f32>`). The Engine samples from logits via this trait without
/// knowing where the data lives or how sampling is implemented.
pub trait Logits: Send {
    /// Vocabulary size (number of logit values per sequence).
    fn vocab_size(&self) -> usize;

    /// Number of sequences in this batch.
    fn batch_size(&self) -> usize;

    /// Greedy: return the token index with the highest logit for a
    /// specific sequence in the batch.
    ///
    /// # Errors
    /// Returns an error if the underlying operation fails.
    fn argmax(&self, batch_index: usize) -> Result<u32>;

    /// Nucleus (top-p) sampling with temperature and repetition penalty
    /// for a specific sequence in the batch.
    ///
    /// # Errors
    /// Returns an error if the underlying operation fails.
    fn sample_top_p(
        &self,
        batch_index: usize,
        temperature: f32,
        top_p: f32,
        rng_seed: u64,
        repetition_penalty: f32,
        recent_tokens: &[u32],
    ) -> Result<u32>;
}
