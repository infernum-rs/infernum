//! Model trait for LLM inference
//!
//! Defines the interface that all model implementations (Llama, Qwen, etc.)
//! must satisfy to be used with the Engine and Runtime.

#[cfg(feature = "cuda")]
use crate::cuda::{CudaTensor, KvCache};
#[cfg(feature = "cuda")]
use crate::Result;

/// Configuration needed by the Engine to allocate resources (e.g., KV cache).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Maximum sequence length the model supports
    pub max_seq_len: usize,
    /// Number of key-value heads (GQA-aware)
    pub num_kv_heads: usize,
    /// Dimension of each attention head
    pub head_dim: usize,
    /// End-of-sequence token ID
    pub eos_token_id: u32,
}

/// Trait for LLM models that can be used with the Engine.
///
/// A model takes token IDs, runs a forward pass, and returns logits.
/// It supports both full-recompute and KV-cached inference.
#[cfg(feature = "cuda")]
pub trait Model {
    /// Get the model configuration needed for resource allocation.
    fn config(&self) -> ModelConfig;

    /// Full forward pass (no KV cache, recomputes everything).
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape (seq_len,)
    ///
    /// # Returns
    /// Logits tensor of shape (seq_len, vocab_size)
    fn forward(&self, input_ids: &[u32]) -> Result<CudaTensor<f32>>;

    /// Forward pass with KV cache (prefill phase).
    ///
    /// Processes all input tokens, populating the KV cache, and returns
    /// logits for the **last** token only: shape (1, vocab_size).
    fn forward_with_kv_cache(
        &self,
        input_ids: &[u32],
        kv_cache: &mut KvCache,
    ) -> Result<CudaTensor<f32>>;

    /// Forward pass for a single token with KV cache (decode phase).
    ///
    /// Appends the token's KV to the cache and returns logits of shape (1, vocab_size).
    fn forward_next_token(&self, token_id: u32, kv_cache: &mut KvCache) -> Result<CudaTensor<f32>>;
}
