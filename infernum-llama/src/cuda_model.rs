//! CUDA-specific multi-GPU loading and `ShardedLoadable` impl.
//!
//! Single-GPU `from_pretrained`, `from_gguf`, and `forward_batch_decode`
//! are generic over any backend and live in `model.rs`.

#[cfg(feature = "nccl")]
use std::path::Path;

#[cfg(feature = "nccl")]
use infernum::shard::GpuConfig;
#[cfg(feature = "nccl")]
use infernum::Result;

#[cfg(feature = "nccl")]
use infernum_cuda::cuda::CudaContext;
#[cfg(feature = "nccl")]
use infernum_cuda::cuda::NcclCommunicator;
#[cfg(feature = "nccl")]
use infernum_cuda::weights::{CudaWeightLoader, SafeTensorsLoader};
#[cfg(feature = "nccl")]
use infernum_cuda::CudaBackend;

#[cfg(feature = "nccl")]
use crate::model::LlamaModel;
#[cfg(feature = "nccl")]
use crate::LlamaConfig;

// ---- Multi-GPU loading ----

#[cfg(feature = "nccl")]
impl LlamaModel<CudaBackend> {
    /// Load a Llama model with tensor-parallel sharding for multi-GPU.
    ///
    /// # Errors
    /// Returns an error if loading fails or head counts are not divisible.
    pub fn from_pretrained_sharded(
        ctx: &CudaContext,
        model_path: impl AsRef<Path>,
        gpu_config: GpuConfig,
        comm: NcclCommunicator,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        let config_path = model_path.join("config.json");
        let config = LlamaConfig::from_file(&config_path)?;
        let format_loader = SafeTensorsLoader::from_directory(model_path)?;
        let loader = CudaWeightLoader::new(ctx.clone(), format_loader);
        Self::load_weights_sharded(ctx.clone(), config, &loader, gpu_config, Some(comm))
    }
}

// ---- ShardedLoadable ----

#[cfg(feature = "nccl")]
impl infernum_cuda::ShardedLoadable for LlamaModel<CudaBackend> {
    fn load_shard(
        ctx: &CudaContext,
        model_path: &Path,
        shard: infernum::ShardConfig,
        comm: NcclCommunicator,
    ) -> Result<Self> {
        Self::from_pretrained_sharded(ctx, model_path, GpuConfig::Sharded(shard), comm)
    }
}

// Model trait impl is in model.rs (generic over any LlamaOps backend).
// CUDA unit tests are in tests/cuda_unit_tests.rs.
