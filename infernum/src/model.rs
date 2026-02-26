//! Model trait for LLM inference
//!
//! Defines the interface that all model implementations (Llama, Qwen, etc.)
//! must satisfy to be used with the Engine and Runtime.

#[cfg(feature = "nccl")]
use std::path::Path;

#[cfg(feature = "cuda")]
use crate::cuda::block_allocator::BlockTable;
#[cfg(feature = "cuda")]
use crate::cuda::BatchedGraphInputs;
#[cfg(feature = "nccl")]
use crate::cuda::{nccl::NcclCommunicator, ShardConfig};
#[cfg(feature = "cuda")]
use crate::cuda::{CudaContext, CudaTensor, PagedKvCache};
#[cfg(feature = "cuda")]
use crate::dtype::DType;
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
    /// Data type for KV cache entries
    #[cfg(feature = "cuda")]
    pub cache_dtype: DType,
}

/// Trait for LLM models that can be used with the Engine.
///
/// A model takes token IDs, runs a forward pass, and returns logits (always f32).
/// It supports both full-recompute and KV-cached inference.
#[cfg(feature = "cuda")]
pub trait Model {
    /// Get the model configuration needed for resource allocation.
    fn config(&self) -> ModelConfig;

    /// Devices this model spans.
    ///
    /// Single-GPU models return a single element; sharded models return one
    /// `CudaContext` per GPU. The Engine uses this to allocate one KV cache
    /// per device.
    fn devices(&self) -> Vec<&CudaContext>;

    /// Full forward pass (no KV cache, recomputes everything).
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape (`seq_len`,)
    ///
    /// # Returns
    /// Logits tensor of shape (`seq_len`, `vocab_size`)
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    fn forward(&self, input_ids: &[u32]) -> Result<CudaTensor>;

    /// Single-sequence prefill forward pass with paged KV cache.
    ///
    /// Processes all prompt tokens for one sequence, writing K/V into the
    /// paged cache. Returns logits for the **last** token only: shape
    /// `(1, vocab_size)`.
    ///
    /// The slice contains one `PagedKvCache` per device (see [`Self::devices`]).
    /// Single-GPU models receive a one-element slice.
    ///
    /// `start_pos` is the sequence position of the first token in `input_ids`
    /// (i.e., `block_table.seq_len()` before this call). The model does
    /// **not** call `block_table.advance()` — the caller is responsible.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    fn forward_prefill_paged(
        &self,
        input_ids: &[u32],
        paged_kvs: &mut [PagedKvCache],
        block_table: &BlockTable,
        start_pos: usize,
    ) -> Result<CudaTensor>;

    /// Batched decode forward pass with paged KV cache.
    ///
    /// Processes one new token per sequence for `batch_size` sequences.
    /// Each sequence has its own block table into the shared KV pool.
    ///
    /// The slice contains one `PagedKvCache` per device (see [`Self::devices`]).
    /// Single-GPU models receive a one-element slice.
    ///
    /// Returns logits of shape `(batch_size, vocab_size)`.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    fn forward_batch_decode(
        &self,
        token_ids: &[u32],
        paged_kvs: &mut [PagedKvCache],
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<CudaTensor>;

    /// Batched decode using indirect kernels for CUDA graph capture.
    ///
    /// Uses `_indirect` kernel variants that read token IDs, positions,
    /// block tables, and sequence lengths from GPU-resident buffers in
    /// [`BatchedGraphInputs`] instead of host slices. This allows a CUDA
    /// graph to be captured once and replayed on every decode step.
    ///
    /// `max_seq_len` is needed for shared memory sizing in attention and
    /// must be at least as large as the longest sequence in the batch.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    fn forward_batch_decode_indirect(
        &self,
        _graph_inputs: &BatchedGraphInputs,
        _paged_kvs: &mut [PagedKvCache],
        _max_seq_len: usize,
    ) -> Result<CudaTensor> {
        unimplemented!("forward_batch_decode_indirect not implemented for this model")
    }
}

/// Trait for models that can load a single tensor-parallel shard.
///
/// Implementing this trait allows a model to be used with
/// [`ShardedModel`](crate::cuda::ShardedModel), which orchestrates loading N
/// shards across N GPUs and wraps them behind the standard [`Model`] trait.
#[cfg(feature = "nccl")]
pub trait ShardedLoadable: Model + Send + Sync
where
    Self: Sized,
{
    /// Load a single shard of this model for tensor-parallel inference.
    ///
    /// # Arguments
    /// * `ctx` — CUDA context for the target GPU
    /// * `model_path` — Path to the model directory (`SafeTensors`, GGUF, etc.)
    /// * `shard` — This GPU's rank and world size
    /// * `comm` — NCCL communicator for all-reduce during inference
    ///
    /// # Errors
    /// Returns an error if weight loading or initialization fails.
    fn load_shard(
        ctx: &CudaContext,
        model_path: &Path,
        shard: ShardConfig,
        comm: NcclCommunicator,
    ) -> Result<Self>;
}
