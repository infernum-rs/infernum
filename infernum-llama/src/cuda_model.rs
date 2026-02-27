//! CUDA-specific factory methods for `LlamaModel`, convenience wrappers,
//! and `ShardedLoadable` impl.

use std::path::Path;

use infernum::block_allocator::BlockTable;
use infernum::dtype::DType;
#[cfg(feature = "nccl")]
use infernum::shard::GpuConfig;
use infernum::Result;

#[cfg(feature = "nccl")]
use infernum_cuda::cuda::NcclCommunicator;
use infernum_cuda::cuda::{CudaContext, CudaTensor, PagedKvCache};
use infernum_cuda::weights::{CudaWeightLoader, SafeTensorsLoader};
use infernum_cuda::CudaBackend;

#[cfg(feature = "nccl")]
use crate::model::AllReduceFn;
use crate::model::LlamaModel;
use crate::LlamaConfig;

// ---- SafeTensors loading ----

impl LlamaModel<CudaBackend> {
    /// Load a Llama model from a directory containing `SafeTensors` and `config.json`
    ///
    /// # Errors
    /// Returns an error if loading fails
    pub fn from_pretrained(ctx: &CudaContext, model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();
        let config_path = model_path.join("config.json");
        let config = LlamaConfig::from_file(&config_path)?;
        let format_loader = SafeTensorsLoader::from_directory(model_path)?;
        let loader = CudaWeightLoader::new(ctx.clone(), format_loader);
        Self::load_weights(ctx.clone(), config, &loader)
    }

    /// Load a Llama model with tensor-parallel sharding for multi-GPU.
    ///
    /// # Errors
    /// Returns an error if loading fails or head counts are not divisible.
    #[cfg(feature = "nccl")]
    pub fn from_pretrained_sharded(
        ctx: &CudaContext,
        model_path: impl AsRef<Path>,
        gpu_config: GpuConfig,
        nccl_comm: NcclCommunicator,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        let config_path = model_path.join("config.json");
        let config = LlamaConfig::from_file(&config_path)?;
        let format_loader = SafeTensorsLoader::from_directory(model_path)?;
        let loader = CudaWeightLoader::new(ctx.clone(), format_loader);
        Self::load_weights_sharded(
            ctx.clone(),
            config,
            &loader,
            gpu_config,
            make_all_reduce_fn(nccl_comm),
        )
    }
}

/// Create an all-reduce closure from an `NcclCommunicator`.
#[cfg(feature = "nccl")]
fn make_all_reduce_fn(comm: NcclCommunicator) -> AllReduceFn<CudaBackend> {
    use std::sync::Arc;
    let comm = Arc::new(comm);
    Box::new(move |tensor: &mut CudaTensor| comm.all_reduce_sum_inplace(tensor))
}

// ---- GGUF loading ----

impl LlamaModel<CudaBackend> {
    /// Load a Llama model from a GGUF file containing quantized weights.
    ///
    /// Parses the GGUF file using the core parser (CPU-only), then uploads
    /// weights to the GPU via the generic `load_weights_gguf` method.
    ///
    /// # Errors
    /// Returns an error if the file cannot be parsed or weights fail to load.
    pub fn from_gguf(ctx: &CudaContext, gguf_path: impl AsRef<Path>) -> Result<Self> {
        let loader = infernum::weights::gguf::GgufLoader::from_file(gguf_path)?;
        let config = LlamaConfig::from_gguf_metadata(loader.metadata())?;
        Self::load_weights_gguf(ctx.clone(), config, &loader)
    }
}

// ---- Convenience method (backwards compat) ----

impl LlamaModel<CudaBackend> {
    /// Run forward pass and return logits (no KV cache).
    ///
    /// This is a convenience wrapper around [`forward_full`](Self::forward_full).
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward(&self, input_ids: &[u32]) -> Result<CudaTensor> {
        self.forward_full(input_ids)
    }

    /// Batched decode with host-side inputs (convenience wrapper).
    ///
    /// Converts host arrays to device tensors and calls the generic
    /// [`forward_batch_decode_tensors`](Self::forward_batch_decode_tensors).
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn forward_batch_decode(
        &self,
        token_ids: &[u32],
        paged_kvs: &mut [PagedKvCache],
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<CudaTensor> {
        let batch_size = token_ids.len();
        let max_blocks_per_seq = block_tables
            .iter()
            .map(|bt| bt.blocks().len())
            .max()
            .unwrap_or(0);
        let mut bt_flat = vec![0i32; batch_size * max_blocks_per_seq];
        for (i, bt) in block_tables.iter().enumerate() {
            for (j, &block_id) in bt.blocks().iter().enumerate() {
                bt_flat[i * max_blocks_per_seq + j] = block_id as i32;
            }
        }
        let seq_lens: Vec<i32> = positions.iter().map(|&p| (p + 1) as i32).collect();
        let positions_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let max_seq_len = seq_lens.iter().copied().max().unwrap_or(0) as usize;

        let token_ids_t = CudaTensor::from_slice(&self.device, &[batch_size], token_ids)?;
        let bt_t = CudaTensor::from_raw_bytes(
            &self.device,
            &[batch_size * max_blocks_per_seq],
            DType::U32,
            unsafe { std::slice::from_raw_parts(bt_flat.as_ptr().cast::<u8>(), bt_flat.len() * 4) },
        )?;
        let sl_t = CudaTensor::from_raw_bytes(&self.device, &[batch_size], DType::U32, unsafe {
            std::slice::from_raw_parts(seq_lens.as_ptr().cast::<u8>(), seq_lens.len() * 4)
        })?;
        let pos_t = CudaTensor::from_raw_bytes(&self.device, &[batch_size], DType::U32, unsafe {
            std::slice::from_raw_parts(positions_i32.as_ptr().cast::<u8>(), positions_i32.len() * 4)
        })?;

        self.forward_batch_decode_tensors(
            &token_ids_t,
            paged_kvs,
            &bt_t,
            &sl_t,
            &pos_t,
            batch_size,
            max_blocks_per_seq,
            max_seq_len,
        )
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
