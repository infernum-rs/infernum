//! CUDA graph-mode inference engine for the Qwen model family.
//!
//! Provides [`QwenCudaGraphEngine`] (a type alias for
//! [`infernum_cuda::CudaGraphEngine<QwenConfig>`]) by implementing
//! [`infernum_cuda::CudaGraphEngineConfig`] for [`QwenConfig`].

use std::path::Path;

use infernum::graph::{Graph, WeightStore};
use infernum::{DType, Result};
use infernum_cuda::{
    load_graph_weights_cuda, CudaBackend, CudaContext, CudaGraphEngineConfig, CudaTensor,
    LinearWeight,
};

use crate::config::QwenConfig;
use crate::graph_builder::{build_decode_graph, build_prefill_graph};

// ---------------------------------------------------------------------------
// CudaGraphEngineConfig impl
// ---------------------------------------------------------------------------

impl CudaGraphEngineConfig for QwenConfig {
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads
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

    fn load_weights_cuda_safetensors(
        &self,
        dummy_graph: &Graph<infernum_cuda::CudaBackend>,
        ctx: &CudaContext,
        model_dir: &Path,
    ) -> Result<WeightStore<CudaTensor, LinearWeight>> {
        // Qwen models do not use tied embeddings; lm_head.weight is always present.
        load_graph_weights_cuda(
            dummy_graph,
            ctx,
            model_dir,
            /* lm_head_fallback */ false,
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
    /// Load a Qwen-family model from a SafeTensors directory onto a CUDA device.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or `config.json` cannot be parsed.
    fn from_pretrained(ctx: CudaContext, model_dir: &Path) -> Result<Self>;
}

impl QwenCudaGraphEngineExt for QwenCudaGraphEngine {
    fn from_pretrained(ctx: CudaContext, model_dir: &Path) -> Result<Self> {
        let config_text = std::fs::read_to_string(model_dir.join("config.json"))?;
        let config: QwenConfig = serde_json::from_str(&config_text)
            .map_err(|e| infernum::Error::InvalidConfig(e.to_string()))?;
        infernum_cuda::CudaGraphEngine::from_config_and_dir(config, ctx, model_dir)
    }
}
