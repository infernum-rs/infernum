//! CUDA graph-mode inference engine for the Qwen model family.
//!
//! Provides [`QwenCudaGraphEngine`] (a type alias for
//! [`infernum_cuda::CudaGraphEngine<QwenConfig>`]) by implementing
//! [`infernum_cuda::CudaGraphEngineConfig`] for [`QwenConfig`].

use std::path::Path;

use infernum::graph::{Graph, WeightStore};
use infernum::weights::QuantizationConfig;
use infernum::{DType, Result};
use infernum_cuda::{
    load_graph_weights_cuda, CudaContext, CudaGraphEngineConfig, CudaTensor, LinearWeight,
};

use crate::config::QwenConfig;
use crate::graph_builder::{build_decode_graph, build_paged_decode_graph, build_prefill_graph};

// ---------------------------------------------------------------------------
// CudaGraphEngineConfig impl
// ---------------------------------------------------------------------------

impl CudaGraphEngineConfig for QwenConfig {
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads()
    }

    fn head_dim(&self) -> usize {
        QwenConfig::head_dim(self)
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

    fn quantization_config(&self) -> Option<&QuantizationConfig> {
        self.quantization_config.as_ref()
    }

    fn build_prefill_graph_cuda(&self, seq_len: usize) -> Graph<infernum_cuda::CudaBackend> {
        let (graph, _) =
            build_prefill_graph::<infernum_cuda::CudaBackend>(self, seq_len, DType::BF16);
        graph
    }

    fn build_decode_graph_cuda(&self, kv_len: usize) -> Graph<infernum_cuda::CudaBackend> {
        let (graph, _) =
            build_decode_graph::<infernum_cuda::CudaBackend>(self, kv_len, DType::BF16);
        graph
    }

    fn build_paged_decode_graph_cuda(
        &self,
        batch_size: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Graph<infernum_cuda::CudaBackend> {
        build_paged_decode_graph::<infernum_cuda::CudaBackend>(
            self,
            batch_size,
            block_size,
            max_blocks_per_seq,
            DType::BF16,
        )
    }

    fn load_weights_cuda_safetensors(
        &self,
        dummy_graph: &Graph<infernum_cuda::CudaBackend>,
        ctx: &CudaContext,
        model_dir: &Path,
    ) -> Result<WeightStore<CudaTensor, LinearWeight>> {
        // Some Qwen models (Qwen2.5, Qwen3-MoE) use tied embeddings and do not
        // have a separate lm_head.weight; enable the fallback so the loader
        // reuses model.embed_tokens.weight in that case.
        load_graph_weights_cuda(
            dummy_graph,
            ctx,
            model_dir,
            /* lm_head_fallback */ true,
            self.quantization_config.as_ref(),
        )
    }
}

// ---------------------------------------------------------------------------
// Type alias + convenience constructors
// ---------------------------------------------------------------------------

/// CUDA graph-mode engine for Qwen-family models.
///
/// A type alias for [`infernum_cuda::CudaGraphEngine<QwenConfig>`].
pub type QwenCudaGraphEngine = infernum_cuda::CudaGraphEngine<QwenConfig>;

/// Extension trait providing Qwen-specific CUDA constructors.
pub trait QwenCudaGraphEngineExt: Sized {
    /// Load a Qwen-family model from a `SafeTensors` directory onto a CUDA device.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or `config.json` cannot be parsed.
    fn from_pretrained(ctx: CudaContext, model_dir: &Path) -> Result<Self>;
}

impl QwenCudaGraphEngineExt for QwenCudaGraphEngine {
    fn from_pretrained(ctx: CudaContext, model_dir: &Path) -> Result<Self> {
        let config = QwenConfig::from_file(model_dir.join("config.json"))?;
        infernum_cuda::CudaGraphEngine::from_config_and_dir(config, ctx, model_dir)
    }
}
