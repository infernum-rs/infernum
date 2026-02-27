//! CUDA-specific `ShardedLoadable` impl.
//!
//! Single-GPU loading, multi-GPU loading, and all forward methods are
//! generic over any backend and live in `model.rs`. This file only
//! provides the `ShardedLoadable` bridge for the old CUDA `ShardedModel`.

#[cfg(feature = "nccl")]
use std::path::Path;

#[cfg(feature = "nccl")]
use infernum_cuda::cuda::CudaContext;
#[cfg(feature = "nccl")]
use infernum_cuda::cuda::NcclCommunicator;
#[cfg(feature = "nccl")]
use infernum_cuda::CudaBackend;

#[cfg(feature = "nccl")]
use infernum::shard::GpuConfig;
#[cfg(feature = "nccl")]
use infernum::Result;

#[cfg(feature = "nccl")]
use crate::model::LlamaModel;

// ---- ShardedLoadable (old CUDA ShardedModel bridge) ----

#[cfg(feature = "nccl")]
impl infernum_cuda::ShardedLoadable for LlamaModel<CudaBackend> {
    fn load_shard(
        ctx: &CudaContext,
        model_path: &Path,
        shard: infernum::ShardConfig,
        comm: NcclCommunicator,
    ) -> Result<Self> {
        Self::from_pretrained_sharded(ctx, model_path, GpuConfig::Sharded(shard), Some(comm))
    }
}
