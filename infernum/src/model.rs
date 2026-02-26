//! Model configuration and trait
//!
//! The `ModelConfig` struct describes a model's resource requirements.
//! The `Model` trait defines the backend-generic interface for LLM
//! inference — any model implementing it can be used with the Engine
//! and Runtime without CUDA-specific code.

use crate::backend::Backend;
use crate::block_allocator::{BlockConfig, BlockTable};
use crate::dtype::DType;
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
    /// Data type for KV cache entries
    pub cache_dtype: DType,
}

/// Trait for LLM models that can be used with the Engine.
///
/// The model has two pieces of mutable state, both owned by the engine
/// and passed in as `&mut` on every forward call:
///
/// - `KvCache` (associated type on `Model`) — the model architecture
///   decides what cache shape its attention needs. `DeepSeek` uses both
///   paged and contiguous caches; Llama uses only paged.
/// - `RuntimeState` (associated type on `Backend`) — the backend
///   decides what runtime optimisations it needs. CUDA uses this for
///   graph capture/replay; CPU uses `()`.
///
/// Multi-device (tensor-parallel) models like `ShardedModel` handle
/// per-device state internally. The engine always sees a single
/// `KvCache` and `RuntimeState` regardless of how many GPUs are involved.
pub trait Model: Send + 'static {
    /// The backend this model runs on.
    type B: Backend;

    /// Model-specific KV cache (shared pool across all requests).
    ///
    /// The model architecture decides what cache composition to use,
    /// built from the backend's cache types:
    /// - Llama: `type KvCache = B::PagedKvCache`
    /// - `DeepSeek`: `type KvCache = DeepSeekKvCache<B>` (wraps both
    ///   `B::PagedKvCache` and `B::KvCache`)
    type KvCache: Send;

    /// Model configuration (`max_seq_len`, `vocab_size`, etc.).
    ///
    /// The engine uses this for scheduler setup and block allocation
    /// sizing. The model knows its own architecture parameters.
    fn config(&self) -> ModelConfig;

    /// Allocate the KV cache for this model.
    ///
    /// Called once by the Engine at startup. `block_config` provides
    /// block size and count for backends that use paged allocation.
    /// Multi-device models allocate one cache per device internally
    /// and wrap them in a single `Self::KvCache`.
    ///
    /// # Errors
    /// Returns an error if cache allocation fails.
    fn allocate_kv_cache(&self, block_config: &BlockConfig) -> Result<Self::KvCache>;

    /// Full forward pass (no KV cache, recomputes everything).
    ///
    /// Returns logits of shape `(seq_len, vocab_size)`.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    fn forward(&self, input_ids: &[u32]) -> Result<<Self::B as Backend>::Logits>;

    /// Single-sequence prefill with KV cache.
    ///
    /// Processes prompt tokens for one sequence, writing K/V into the
    /// cache. Returns logits for the **last** token only (`batch_size=1`).
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    fn forward_prefill(
        &self,
        input_ids: &[u32],
        kv_cache: &mut Self::KvCache,
        runtime_state: &mut <Self::B as Backend>::RuntimeState,
        block_table: &BlockTable,
        start_pos: usize,
    ) -> Result<<Self::B as Backend>::Logits>;

    /// Batched decode with KV cache.
    ///
    /// Processes one new token per sequence for `batch_size` sequences.
    /// Returns logits with `batch_size` rows. The backend may
    /// internally use graph capture/replay or other optimisations
    /// via `runtime_state` — the engine is unaware.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    fn forward_batch_decode(
        &self,
        token_ids: &[u32],
        kv_cache: &mut Self::KvCache,
        runtime_state: &mut <Self::B as Backend>::RuntimeState,
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<<Self::B as Backend>::Logits>;
}
