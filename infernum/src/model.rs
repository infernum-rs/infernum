//! Model trait for LLM inference
//!
//! Defines the interface that all model implementations (Llama, Qwen, etc.)
//! must satisfy to be used with the Engine and Runtime.

#[cfg(feature = "nccl")]
use std::path::Path;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};

#[cfg(feature = "nccl")]
use crate::cuda::{nccl::NcclCommunicator, ShardConfig};
#[cfg(feature = "cuda")]
use crate::cuda::{CudaContext, CudaTensor, KvCache};
#[cfg(feature = "cuda")]
use crate::dtype::TensorDType;
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
/// A model takes token IDs, runs a forward pass, and returns logits (always f32).
/// It supports both full-recompute and KV-cached inference.
///
/// The associated type `CacheDtype` determines the KV cache element type,
/// allowing models to compute in f16/bf16 while the Engine allocates the
/// correct cache type.
#[cfg(feature = "cuda")]
pub trait Model {
    /// Element type for KV cache tensors (f32, f16, or bf16).
    type CacheDtype: TensorDType + DeviceRepr + ValidAsZeroBits;

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
    fn forward(&self, input_ids: &[u32]) -> Result<CudaTensor<f32>>;

    /// Forward pass with KV cache (prefill phase).
    ///
    /// Processes all input tokens, populating the KV cache, and returns
    /// logits for the **last** token only: shape (1, `vocab_size`).
    ///
    /// The slice contains one KV cache per device (see [`Self::devices`]).
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    fn forward_with_kv_cache(
        &self,
        input_ids: &[u32],
        kv_caches: &mut [KvCache<Self::CacheDtype>],
    ) -> Result<CudaTensor<f32>>;

    /// Forward pass for a single token with KV cache (decode phase).
    ///
    /// Appends the token's KV to the cache and returns logits of shape (1, `vocab_size`).
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    fn forward_next_token(
        &self,
        token_id: u32,
        kv_caches: &mut [KvCache<Self::CacheDtype>],
    ) -> Result<CudaTensor<f32>>;

    /// Forward pass for a single token with KV cache, reading the token ID
    /// from a GPU buffer instead of the host.
    ///
    /// This avoids the `htod_sync_copy` in [`Self::forward_next_token`], making
    /// the entire forward pass capturable by a CUDA graph.
    ///
    /// The default implementation copies the token to host and delegates to
    /// [`Self::forward_next_token`]. Models should override this to use
    /// [`embedding_gather_from_device`](crate::cuda::ops::embedding_gather_from_device)
    /// for a fully GPU-resident decode path.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    fn forward_next_token_device(
        &self,
        token_id_gpu: &CudaSlice<u32>,
        kv_caches: &mut [KvCache<Self::CacheDtype>],
    ) -> Result<CudaTensor<f32>> {
        let host = token_id_gpu.device().dtoh_sync_copy(token_id_gpu)?;
        self.forward_next_token(host[0], kv_caches)
    }

    /// Decode-phase forward pass using indirect kernels for CUDA graph capture.
    ///
    /// Uses `_indirect` kernel variants that read position/length from stable
    /// device pointers (via [`KvCache::current_position`]) instead of host
    /// scalars. This allows a CUDA graph to be captured once and replayed
    /// on every subsequent decode step without re-capture.
    ///
    /// **Does not call `kv_cache.advance()`** — the caller must do that
    /// outside the captured region after each graph launch.
    ///
    /// The default implementation falls back to [`Self::forward_next_token_device`].
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    fn forward_next_token_indirect(
        &self,
        token_id_gpu: &CudaSlice<u32>,
        kv_caches: &mut [KvCache<Self::CacheDtype>],
    ) -> Result<CudaTensor<f32>> {
        self.forward_next_token_device(token_id_gpu, kv_caches)
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
