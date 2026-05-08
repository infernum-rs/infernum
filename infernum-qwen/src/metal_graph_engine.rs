//! Metal graph-mode inference engine for the Qwen model family.
//!
//! Provides [`QwenMetalGraphEngine`] (a type alias for
//! [`infernum_metal::MetalGraphEngine<QwenConfig>`]) by implementing
//! [`infernum_metal::MetalGraphEngineConfig`] for [`QwenConfig`].

use std::path::Path;

use infernum::graph::{Graph, WeightStore};
use infernum::{DType, Result};
use infernum_metal::{
    load_graph_weights_metal, MetalContext, MetalGraphEngineConfig, MetalLinearWeight, MetalTensor,
};

use crate::config::QwenConfig;
use crate::graph_builder::{build_decode_graph, build_paged_decode_graph, build_prefill_graph};

// ---------------------------------------------------------------------------
// MetalGraphEngineConfig impl
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

impl MetalGraphEngineConfig for QwenConfig {
    impl_common_config_getters!();

    fn num_kv_heads(&self) -> usize {
        QwenConfig::num_kv_heads(self)
    }

    fn head_dim(&self) -> usize {
        QwenConfig::head_dim(self)
    }

    fn build_prefill_graph_metal(&self, seq_len: usize) -> Graph<infernum_metal::MetalBackend> {
        let (graph, _) =
            build_prefill_graph::<infernum_metal::MetalBackend>(self, seq_len, DType::F32);
        graph
    }

    fn build_decode_graph_metal(&self, kv_len: usize) -> Graph<infernum_metal::MetalBackend> {
        let (graph, _) =
            build_decode_graph::<infernum_metal::MetalBackend>(self, kv_len, DType::F32);
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
        // Some Qwen models (Qwen2.5, Qwen3-MoE) use tied embeddings and do not
        // have a separate lm_head.weight; enable the fallback so the loader
        // reuses model.embed_tokens.weight in that case.
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

/// Metal graph-mode engine for Qwen-family models.
///
/// A type alias for [`infernum_metal::MetalGraphEngine<QwenConfig>`].
pub type QwenMetalGraphEngine = infernum_metal::MetalGraphEngine<QwenConfig>;

/// Extension trait providing Qwen-specific Metal constructors.
pub trait QwenMetalGraphEngineExt: Sized {
    /// Load a Qwen-family model from a `SafeTensors` directory onto a Metal device.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or `config.json` cannot be parsed.
    fn from_pretrained(ctx: MetalContext, model_dir: &Path) -> Result<Self>;
}

impl QwenMetalGraphEngineExt for QwenMetalGraphEngine {
    fn from_pretrained(ctx: MetalContext, model_dir: &Path) -> Result<Self> {
        let config = QwenConfig::from_file(model_dir.join("config.json"))?;
        infernum_metal::MetalGraphEngine::from_config_and_dir(config, ctx, model_dir)
    }
}
