//! Model configuration types
//!
//! The `ModelConfig` struct describes a model's resource requirements.
//! The `Model` and `ShardedLoadable` traits live in `infernum-cuda`
//! since they are CUDA-specific.

use crate::dtype::DType;

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
    /// Data type for KV cache entries
    pub cache_dtype: DType,
}
