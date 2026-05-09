//! Metal graph-mode inference engine for the Gemma model family.
//!
//! Provides [`GemmaMetalGraphEngine`] (a type alias for
//! [`infernum_metal::MetalGraphEngine<GemmaConfig>`]) by implementing
//! [`infernum_metal::MetalGraphEngineConfig`] for [`GemmaConfig`].

use std::path::Path;

use infernum::graph::{Graph, WeightStore};
use infernum::{DType, Result};
use infernum_metal::{
    load_graph_weights_metal, MetalContext, MetalGraphEngineConfig, MetalLinearWeight, MetalTensor,
};

use crate::config::GemmaConfig;
use crate::graph_builder::{build_paged_decode_graph, build_prefill_graph};

// ---------------------------------------------------------------------------
// MetalGraphEngineConfig impl
// ---------------------------------------------------------------------------

impl MetalGraphEngineConfig for GemmaConfig {
    infernum_metal::impl_metal_config_getters!();

    fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn build_prefill_graph_metal(&self, seq_len: usize) -> Graph<infernum_metal::MetalBackend> {
        build_prefill_graph::<infernum_metal::MetalBackend>(self, seq_len, DType::F32)
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
        // Gemma checkpoints use tied embeddings; lm_head.weight is absent and
        // the SafeTensors loader must fall back to model.embed_tokens.weight.
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

/// Metal graph-mode engine for Gemma-family models.
///
/// A type alias for [`infernum_metal::MetalGraphEngine<GemmaConfig>`].
pub type GemmaMetalGraphEngine = infernum_metal::MetalGraphEngine<GemmaConfig>;

/// Extension trait providing Gemma-specific Metal constructors.
pub trait GemmaMetalGraphEngineExt: Sized {
    /// Load a Gemma-family model from a SafeTensors directory onto a Metal device.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or `config.json` cannot be parsed.
    fn from_pretrained(ctx: MetalContext, model_dir: &Path) -> Result<Self>;
}

impl GemmaMetalGraphEngineExt for GemmaMetalGraphEngine {
    fn from_pretrained(ctx: MetalContext, model_dir: &Path) -> Result<Self> {
        let config = GemmaConfig::from_file(model_dir.join("config.json"))?;
        infernum_metal::MetalGraphEngine::from_config_and_dir(config, ctx, model_dir)
    }
}
