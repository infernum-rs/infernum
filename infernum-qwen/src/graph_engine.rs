//! CPU graph-mode inference engine for the Qwen model family.
//!
//! Provides [`QwenGraphEngine`] (a type alias for
//! [`infernum_cpu::GraphEngine<QwenConfig>`]) by implementing
//! [`infernum_cpu::GraphEngineConfig`] for [`QwenConfig`].
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use infernum_qwen::graph_engine::{QwenGraphEngine, QwenGraphEngineExt as _};
//!
//! let engine = QwenGraphEngine::from_pretrained(Path::new("/path/to/model")).unwrap();
//! let tokens = engine.generate(&[1, 1234, 567], 128, 2).unwrap();
//! println!("{tokens:?}");
//! ```

use std::path::Path;

use infernum::graph::{Graph, WeightStore};
use infernum::{DType, Result};
use infernum_cpu::graph_engine::GraphEngineConfig;
use infernum_cpu::tensor::{CpuLinearWeight, CpuTensor};
use infernum_cpu::CpuBackend;

use crate::config::QwenConfig;
use crate::graph_builder::{
    build_decode_graph, build_prefill_graph, load_graph_weights_safetensors,
};

// ---------------------------------------------------------------------------
// GraphEngineConfig impl
// ---------------------------------------------------------------------------

impl GraphEngineConfig for QwenConfig {
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    fn num_kv_heads(&self) -> usize {
        QwenConfig::num_kv_heads(self)
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

    fn build_prefill_graph(&self, seq_len: usize) -> Graph<CpuBackend> {
        let (graph, _) = build_prefill_graph::<CpuBackend>(self, seq_len, DType::F32);
        graph
    }

    fn build_decode_graph(&self, kv_len: usize) -> Graph<CpuBackend> {
        let (graph, _) = build_decode_graph::<CpuBackend>(self, kv_len, DType::F32);
        graph
    }

    fn load_weights_safetensors(
        &self,
        dummy_graph: &Graph<CpuBackend>,
        model_dir: &Path,
    ) -> Result<WeightStore<CpuTensor, CpuLinearWeight>> {
        load_graph_weights_safetensors(dummy_graph, model_dir, self)
    }
    // No GGUF support for Qwen (no load_graph_weights_gguf in graph_builder.rs).
}

// ---------------------------------------------------------------------------
// Type alias + convenience constructors
// ---------------------------------------------------------------------------

/// CPU graph-mode engine for Qwen-family models (Qwen2, Qwen3, Qwen3-MoE).
///
/// A type alias for [`infernum_cpu::GraphEngine<QwenConfig>`].
/// See [`infernum_cpu::GraphEngine`] for the full API.
pub type QwenGraphEngine = infernum_cpu::GraphEngine<QwenConfig>;

/// Extension trait providing Qwen-specific constructors on [`QwenGraphEngine`].
pub trait QwenGraphEngineExt: Sized {
    /// Load a Qwen-family model from a `SafeTensors` directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or `config.json` cannot be parsed.
    fn from_pretrained(model_dir: &Path) -> Result<Self>;
}

impl QwenGraphEngineExt for QwenGraphEngine {
    fn from_pretrained(model_dir: &Path) -> Result<Self> {
        let config = QwenConfig::from_file(model_dir.join("config.json"))?;
        infernum_cpu::GraphEngine::from_config_and_dir(config, model_dir)
    }
}
