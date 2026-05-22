//! CUDA graph-mode inference engine for the Gemma model family.
//!
//! Provides [`GemmaCudaGraphEngine`] (a type alias for
//! [`infernum_cuda::CudaGraphEngine<GemmaConfig>`]) by implementing
//! [`infernum_cuda::CudaGraphEngineConfig`] for [`GemmaConfig`].

use std::path::Path;

use infernum::graph::{Graph, WeightStore};
use infernum::shard::ShardConfig;
use infernum::weights::QuantizationConfig;
use infernum::{DType, Result};
use infernum_cuda::{
    load_graph_weights_cuda, load_graph_weights_gguf_cuda, CudaContext, CudaGraphEngineConfig,
    CudaTensor, LinearWeight,
};

use crate::config::GemmaConfig;
use crate::graph_builder::{
    build_decode_graph, build_paged_decode_graph, build_prefill_graph, safetensors_to_gguf_name,
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

impl CudaGraphEngineConfig for GemmaConfig {
    impl_common_config_getters!();

    fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn quantization_config(&self) -> Option<&QuantizationConfig> {
        self.quantization_config.as_ref()
    }

    fn build_prefill_graph_cuda(
        &self,
        seq_len: usize,
        shard: Option<&ShardConfig>,
    ) -> Graph<infernum_cuda::CudaBackend> {
        build_prefill_graph::<infernum_cuda::CudaBackend>(self, seq_len, DType::BF16, shard)
    }

    fn build_decode_graph_cuda(
        &self,
        kv_len: usize,
        shard: Option<&ShardConfig>,
    ) -> Graph<infernum_cuda::CudaBackend> {
        build_decode_graph::<infernum_cuda::CudaBackend>(self, kv_len, DType::BF16, shard)
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
        // Gemma checkpoints use tied embeddings; lm_head.weight is absent and
        // the SafeTensors loader must fall back to model.embed_tokens.weight.
        load_graph_weights_cuda(
            dummy_graph,
            ctx,
            model_dir,
            /* lm_head_fallback */ true,
            /* quant_config */ None,
            shard,
        )
    }

    fn load_weights_cuda_gguf(
        &self,
        dummy_graph: &Graph<infernum_cuda::CudaBackend>,
        ctx: &CudaContext,
        gguf_path: &Path,
    ) -> Result<WeightStore<CudaTensor, LinearWeight>> {
        // Gemma does not use the Llama-style interleaved RoPE layout in GGUF,
        // so no Q/K row-unpermutation is needed.
        load_graph_weights_gguf_cuda(
            dummy_graph,
            ctx,
            gguf_path,
            safetensors_to_gguf_name,
            |_| false,
            self.num_attention_heads,
            self.num_key_value_heads,
            /* lm_head_fallback */ true,
        )
    }
}

// ---------------------------------------------------------------------------
// Type alias + convenience constructors
// ---------------------------------------------------------------------------

/// CUDA graph-mode engine for Gemma-family models.
///
/// A type alias for [`infernum_cuda::CudaGraphEngine<GemmaConfig>`].
pub type GemmaCudaGraphEngine = infernum_cuda::CudaGraphEngine<GemmaConfig>;

/// Tensor-parallel CUDA graph engine for Gemma-family models.
///
/// A type alias for [`infernum_cuda::ShardedGraphEngine<GemmaConfig>`].
#[cfg(feature = "nccl")]
pub type GemmaShardedGraphEngine = infernum_cuda::ShardedGraphEngine<GemmaConfig>;

/// Extension trait providing Gemma-specific CUDA constructors.
pub trait GemmaCudaGraphEngineExt: Sized {
    /// Load a Gemma-family model from a SafeTensors directory onto a CUDA device.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or `config.json` cannot be parsed.
    fn from_pretrained(ctx: CudaContext, model_dir: &Path) -> Result<Self>;

    /// Load a Gemma-family model from a GGUF file onto a CUDA device.
    ///
    /// # Errors
    ///
    /// Returns an error if the GGUF file is missing, metadata cannot be
    /// parsed, or weights cannot be uploaded.
    fn from_gguf(ctx: CudaContext, gguf_path: &Path) -> Result<Self>;
}

impl GemmaCudaGraphEngineExt for GemmaCudaGraphEngine {
    fn from_pretrained(ctx: CudaContext, model_dir: &Path) -> Result<Self> {
        let config = GemmaConfig::from_file(model_dir.join("config.json"))?;
        infernum_cuda::CudaGraphEngine::from_config_and_dir(config, ctx, model_dir)
    }

    fn from_gguf(ctx: CudaContext, gguf_path: &Path) -> Result<Self> {
        let loader = infernum::weights::gguf::GgufLoader::from_file(
            infernum::path_to_utf8(gguf_path)?,
        )?;
        let config = GemmaConfig::from_gguf_metadata(loader.metadata())?;
        infernum_cuda::CudaGraphEngine::from_config_gguf(config, ctx, gguf_path)
    }
}

/// Extension trait providing Gemma-specific tensor-parallel constructors.
#[cfg(feature = "nccl")]
pub trait GemmaShardedGraphEngineExt: Sized {
    /// Load a Gemma-family model across multiple GPUs using tensor parallelism.
    ///
    /// # Errors
    ///
    /// Returns an error if device creation, NCCL setup, or weight loading fails.
    fn from_pretrained(num_devices: usize, model_dir: &Path) -> Result<Self>;
}

#[cfg(feature = "nccl")]
impl GemmaShardedGraphEngineExt for GemmaShardedGraphEngine {
    fn from_pretrained(num_devices: usize, model_dir: &Path) -> Result<Self> {
        let config = GemmaConfig::from_file(model_dir.join("config.json"))?;
        infernum_cuda::ShardedGraphEngine::from_config(config, num_devices, model_dir)
    }
}
