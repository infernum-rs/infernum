//! Model support traits for the CUDA backend.
//!
//! The primary model trait is [`infernum::Model`] (defined in the core crate).
//! This module provides [`ShardedLoadable`] for tensor-parallel shard loading.

#[cfg(feature = "nccl")]
use std::path::Path;

#[cfg(feature = "nccl")]
use crate::cuda::{nccl::NcclCommunicator, CudaContext, ShardConfig};
#[cfg(feature = "nccl")]
use infernum::Result;

// Re-export ModelConfig from core so downstream crates can use it from either place.
pub use infernum::ModelConfig;

/// Trait for models that can load a single tensor-parallel shard.
///
/// Implementing this trait allows a model to be used with
/// [`ShardedModel`](crate::cuda::ShardedModel), which orchestrates loading N
/// shards across N GPUs and wraps them behind [`infernum::Model`].
#[cfg(feature = "nccl")]
pub trait ShardedLoadable: Send + Sync
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
