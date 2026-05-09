//! Metal graph-mode inference engine for the Llama model family.
//!
//! Provides [`LlamaMetalGraphEngine`] (a type alias for
//! [`infernum_metal::MetalGraphEngine<LlamaConfig>`]) by implementing
//! [`infernum_metal::MetalGraphEngineConfig`] for [`LlamaConfig`].

use std::path::Path;

use infernum::graph::{Graph, WeightStore};
use infernum::{DType, Result};
use infernum_metal::{
    load_graph_weights_metal, MetalContext, MetalGraphEngineConfig, MetalLinearWeight, MetalTensor,
};

use crate::config::LlamaConfig;
use crate::graph_builder::{build_paged_decode_graph, build_prefill_graph};

// ---------------------------------------------------------------------------
// MetalGraphEngineConfig impl
// ---------------------------------------------------------------------------

impl MetalGraphEngineConfig for LlamaConfig {
    infernum_metal::impl_metal_config_getters!();

    fn num_kv_heads(&self) -> usize {
        LlamaConfig::num_kv_heads(self)
    }

    fn head_dim(&self) -> usize {
        LlamaConfig::head_dim(self)
    }

    fn build_prefill_graph_metal(&self, seq_len: usize) -> Graph<infernum_metal::MetalBackend> {
        let (graph, _) =
            build_prefill_graph::<infernum_metal::MetalBackend>(self, seq_len, DType::F32);
        graph
    }

    fn build_paged_decode_graph_metal(
        &self,
        batch_size: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Graph<infernum_metal::MetalBackend> {
        build_paged_decode_graph::<infernum_metal::MetalBackend>(
            self,
            batch_size,
            block_size,
            max_blocks_per_seq,
            DType::F32,
        )
    }

    fn load_weights_metal_safetensors(
        &self,
        dummy_graph: &Graph<infernum_metal::MetalBackend>,
        ctx: &MetalContext,
        model_dir: &Path,
    ) -> Result<WeightStore<MetalTensor, MetalLinearWeight>> {
        // SmolLM2 and some other Llama-family models use tied embeddings:
        // `lm_head.weight` is absent and shares `model.embed_tokens.weight`.
        load_graph_weights_metal(
            dummy_graph,
            ctx,
            model_dir,
            /* lm_head_fallback */ true,
        )
    }
}

// ---------------------------------------------------------------------------
// Type alias + convenience constructors
// ---------------------------------------------------------------------------

/// Metal graph-mode engine for Llama-family models.
///
/// A type alias for [`infernum_metal::MetalGraphEngine<LlamaConfig>`].
pub type LlamaMetalGraphEngine = infernum_metal::MetalGraphEngine<LlamaConfig>;

/// Extension trait providing Llama-specific Metal constructors.
pub trait LlamaMetalGraphEngineExt: Sized {
    /// Load a Llama-family model from a `SafeTensors` directory onto a Metal device.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or `config.json` cannot be parsed.
    fn from_pretrained(ctx: MetalContext, model_dir: &Path) -> Result<Self>;
}

impl LlamaMetalGraphEngineExt for LlamaMetalGraphEngine {
    fn from_pretrained(ctx: MetalContext, model_dir: &Path) -> Result<Self> {
        let config = LlamaConfig::from_file(model_dir.join("config.json"))?;
        infernum_metal::MetalGraphEngine::from_config_and_dir(config, ctx, model_dir)
    }
}
