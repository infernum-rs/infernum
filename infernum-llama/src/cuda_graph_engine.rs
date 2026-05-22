//! CUDA graph-mode inference engine for the Llama model family.
//!
//! Provides [`LlamaCudaGraphEngine`] (a type alias for
//! [`infernum_cuda::CudaGraphEngine<LlamaConfig>`]) by implementing
//! [`infernum_cuda::CudaGraphEngineConfig`] for [`LlamaConfig`].

use std::path::Path;

use infernum::graph::{Graph, WeightStore};
use infernum::shard::ShardConfig;
use infernum::weights::QuantizationConfig;
use infernum::{DType, Result};
use infernum_cuda::{
    load_graph_weights_cuda, load_graph_weights_gguf_cuda, CudaContext, CudaGraphEngineConfig,
    CudaTensor, LinearWeight,
};

use crate::config::LlamaConfig;
use crate::graph_builder::{
    build_decode_graph, build_paged_decode_graph, build_prefill_graph, needs_unpermute,
    safetensors_to_gguf_name,
};

// ---------------------------------------------------------------------------
// CudaGraphEngineConfig impl
// ---------------------------------------------------------------------------

macro_rules! impl_common_config_getters {
    () => {
        fn num_hidden_layers(&self) -> usize {
            self.num_hidden_layers
        }
        fn max_position_embeddings(&self) -> usize {
            self.max_position_embeddings
        }
        fn rope_theta(&self) -> f32 {
            self.rope_theta
        }
        fn vocab_size(&self) -> usize {
            self.vocab_size
        }
        fn eos_token_id(&self) -> u32 {
            self.eos_token_id
        }
    };
}

impl CudaGraphEngineConfig for LlamaConfig {
    impl_common_config_getters!();

    fn num_kv_heads(&self) -> usize {
        LlamaConfig::num_kv_heads(self)
    }

    fn head_dim(&self) -> usize {
        LlamaConfig::head_dim(self)
    }

    fn quantization_config(&self) -> Option<&QuantizationConfig> {
        self.quantization_config.as_ref()
    }

    fn build_prefill_graph_cuda(
        &self,
        seq_len: usize,
        shard: Option<&ShardConfig>,
    ) -> Graph<infernum_cuda::CudaBackend> {
        let (graph, _) =
            build_prefill_graph::<infernum_cuda::CudaBackend>(self, seq_len, DType::BF16, shard);
        graph
    }

    fn build_decode_graph_cuda(
        &self,
        kv_len: usize,
        shard: Option<&ShardConfig>,
    ) -> Graph<infernum_cuda::CudaBackend> {
        let (graph, _) =
            build_decode_graph::<infernum_cuda::CudaBackend>(self, kv_len, DType::BF16, shard);
        graph
    }

    fn build_paged_decode_graph_cuda(
        &self,
        batch_size: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
        shard: Option<&ShardConfig>,
    ) -> Graph<infernum_cuda::CudaBackend> {
        build_paged_decode_graph::<infernum_cuda::CudaBackend>(
            self,
            batch_size,
            block_size,
            max_blocks_per_seq,
            DType::BF16,
            shard,
        )
    }

    fn load_weights_cuda_safetensors(
        &self,
        dummy_graph: &Graph<infernum_cuda::CudaBackend>,
        ctx: &CudaContext,
        model_dir: &Path,
        shard: Option<&ShardConfig>,
    ) -> Result<WeightStore<CudaTensor, LinearWeight>> {
        // SmolLM2 and some other Llama-family models use tied embeddings:
        // `lm_head.weight` is absent and shares `model.embed_tokens.weight`.
        load_graph_weights_cuda(
            dummy_graph,
            ctx,
            model_dir,
            /* lm_head_fallback */ true,
            self.quantization_config.as_ref(),
            shard,
        )
    }

    fn load_weights_cuda_gguf(
        &self,
        dummy_graph: &Graph<infernum_cuda::CudaBackend>,
        ctx: &CudaContext,
        gguf_path: &Path,
    ) -> Result<WeightStore<CudaTensor, LinearWeight>> {
        load_graph_weights_gguf_cuda(
            dummy_graph,
            ctx,
            gguf_path,
            safetensors_to_gguf_name,
            needs_unpermute,
            self.num_attention_heads,
            LlamaConfig::num_kv_heads(self),
            /* lm_head_fallback */ true,
        )
    }
}

// ---------------------------------------------------------------------------
// Type alias + convenience constructors
// ---------------------------------------------------------------------------

/// CUDA graph-mode engine for Llama-family models.
///
/// A type alias for [`infernum_cuda::CudaGraphEngine<LlamaConfig>`].
pub type LlamaCudaGraphEngine = infernum_cuda::CudaGraphEngine<LlamaConfig>;

/// Tensor-parallel CUDA graph engine for Llama-family models.
///
/// A type alias for [`infernum_cuda::ShardedGraphEngine<LlamaConfig>`].
#[cfg(feature = "nccl")]
pub type LlamaShardedGraphEngine = infernum_cuda::ShardedGraphEngine<LlamaConfig>;

/// Extension trait providing Llama-specific CUDA constructors.
pub trait LlamaCudaGraphEngineExt: Sized {
    /// Load a Llama-family model from a `SafeTensors` directory onto a CUDA device.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or `config.json` cannot be parsed.
    fn from_pretrained(ctx: CudaContext, model_dir: &Path) -> Result<Self>;

    /// Load a Llama-family model from a GGUF file onto a CUDA device.
    ///
    /// Weights are dequantized to BF16 on the host and uploaded. The config
    /// is read from the GGUF metadata, so no separate `config.json` is needed.
    ///
    /// # Errors
    ///
    /// Returns an error if the GGUF file is missing, the metadata cannot be
    /// parsed, or weights cannot be uploaded.
    fn from_gguf(ctx: CudaContext, gguf_path: &Path) -> Result<Self>;
}

impl LlamaCudaGraphEngineExt for LlamaCudaGraphEngine {
    fn from_pretrained(ctx: CudaContext, model_dir: &Path) -> Result<Self> {
        let config = LlamaConfig::from_file(model_dir.join("config.json"))?;
        infernum_cuda::CudaGraphEngine::from_config_and_dir(config, ctx, model_dir)
    }

    fn from_gguf(ctx: CudaContext, gguf_path: &Path) -> Result<Self> {
        let loader = infernum::weights::gguf::GgufLoader::from_file(
            infernum::path_to_utf8(gguf_path)?,
        )?;
        let config = LlamaConfig::from_gguf_metadata(loader.metadata())?;
        infernum_cuda::CudaGraphEngine::from_config_gguf(config, ctx, gguf_path)
    }
}

/// Extension trait providing Llama-specific tensor-parallel constructors.
#[cfg(feature = "nccl")]
pub trait LlamaShardedGraphEngineExt: Sized {
    /// Load a Llama-family model across multiple GPUs using tensor parallelism.
    ///
    /// # Errors
    ///
    /// Returns an error if device creation, NCCL setup, or weight loading fails.
    fn from_pretrained(num_devices: usize, model_dir: &Path) -> Result<Self>;
}

#[cfg(feature = "nccl")]
impl LlamaShardedGraphEngineExt for LlamaShardedGraphEngine {
    fn from_pretrained(num_devices: usize, model_dir: &Path) -> Result<Self> {
        let config = LlamaConfig::from_file(model_dir.join("config.json"))?;
        infernum_cuda::ShardedGraphEngine::from_config(config, num_devices, model_dir)
    }
}
