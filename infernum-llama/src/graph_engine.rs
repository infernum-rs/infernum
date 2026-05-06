//! CPU graph-mode inference engine for the Llama model family.
//!
//! Provides [`LlamaGraphEngine`] (a type alias for
//! [`infernum_cpu::GraphEngine<LlamaConfig>`]) by implementing
//! [`infernum_cpu::GraphEngineConfig`] for [`LlamaConfig`].
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use infernum_llama::graph_engine::{LlamaGraphEngine, LlamaGraphEngineExt as _};
//!
//! let engine = LlamaGraphEngine::from_pretrained(Path::new("/path/to/model")).unwrap();
//! let tokens = engine.generate(&[1, 1234, 567], 128, 2).unwrap();
//! println!("{tokens:?}");
//! ```

use std::path::Path;

use infernum::graph::{Graph, WeightStore};
use infernum::{DType, Result};
use infernum_cpu::graph_engine::GraphEngineConfig;
use infernum_cpu::tensor::{CpuLinearWeight, CpuTensor};
use infernum_cpu::CpuBackend;

use crate::config::LlamaConfig;
use crate::graph_builder::{build_decode_graph, build_prefill_graph, load_graph_weights_gguf};

// ---------------------------------------------------------------------------
// GraphEngineConfig impl
// ---------------------------------------------------------------------------

/// Generate trivial forwarding getters that map directly to struct fields.
///
/// Methods that require special handling (e.g. `num_kv_heads` and `head_dim`
/// on `LlamaConfig`, which shadow inherent methods) are left as explicit impls.
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

impl GraphEngineConfig for LlamaConfig {
    impl_common_config_getters!();

    fn num_kv_heads(&self) -> usize {
        LlamaConfig::num_kv_heads(self)
    }

    fn head_dim(&self) -> usize {
        LlamaConfig::head_dim(self)
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
        infernum_cpu::load_cpu_safetensors_weights(dummy_graph, model_dir, true)
    }

    fn load_weights_gguf(
        &self,
        dummy_graph: &Graph<CpuBackend>,
        gguf_path: &Path,
    ) -> Option<Result<WeightStore<CpuTensor, CpuLinearWeight>>> {
        Some(load_graph_weights_gguf(dummy_graph, self, gguf_path))
    }
}

// ---------------------------------------------------------------------------
// Type alias + convenience constructors
// ---------------------------------------------------------------------------

/// CPU graph-mode engine for Llama-family models.
///
/// A type alias for [`infernum_cpu::GraphEngine<LlamaConfig>`].
/// See [`infernum_cpu::GraphEngine`] for the full API.
pub type LlamaGraphEngine = infernum_cpu::GraphEngine<LlamaConfig>;

/// Extension trait providing Llama-specific constructors on [`LlamaGraphEngine`].
pub trait LlamaGraphEngineExt: Sized {
    /// Load a Llama-family model from a `SafeTensors` directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or `config.json` cannot be parsed.
    fn from_pretrained(model_dir: &Path) -> Result<Self>;

    /// Load a Llama-family model from a GGUF file.
    ///
    /// Weights are stored in their native quantization format (`Q8_0`, `Q4_0`,
    /// F32, etc.) and dequantized lazily inside each matmul kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if the GGUF file cannot be opened, cannot be parsed,
    /// or contains unsupported quantization types (e.g. `Q6_K`).
    fn from_gguf(gguf_path: &Path) -> Result<Self>;
}

impl LlamaGraphEngineExt for LlamaGraphEngine {
    fn from_pretrained(model_dir: &Path) -> Result<Self> {
        let config = LlamaConfig::from_file(model_dir.join("config.json"))?;
        infernum_cpu::GraphEngine::from_config_and_dir(config, model_dir)
    }

    fn from_gguf(gguf_path: &Path) -> Result<Self> {
        let loader =
            infernum::weights::gguf::GgufLoader::from_file(infernum::path_to_utf8(gguf_path)?)?;
        let config = LlamaConfig::from_gguf_metadata(loader.metadata())?;
        infernum_cpu::GraphEngine::from_gguf_with_config(config, gguf_path)
    }
}

// Re-export `GraphKvCache` so existing users of
// `infernum_llama::graph_engine::GraphKvCache` continue to work.
pub use infernum_cpu::GraphKvCache;
