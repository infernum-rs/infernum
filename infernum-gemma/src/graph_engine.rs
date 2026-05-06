//! CPU graph-mode inference engine for the Gemma model family.
//!
//! Provides [`GemmaGraphEngine`] (a type alias for
//! [`infernum_cpu::GraphEngine<GemmaConfig>`]) by implementing
//! [`infernum_cpu::GraphEngineConfig`] for [`GemmaConfig`].
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use infernum_gemma::graph_engine::{GemmaGraphEngine, GemmaGraphEngineExt as _};
//!
//! let engine = GemmaGraphEngine::from_pretrained(Path::new("/path/to/model")).unwrap();
//! let tokens = engine.generate(&[2, 1234, 567], 128, 1).unwrap();
//! println!("{tokens:?}");
//! ```

use std::path::Path;

use infernum::graph::{Graph, WeightStore};
use infernum::{DType, Result};
use infernum_cpu::graph_engine::GraphEngineConfig;
use infernum_cpu::tensor::{CpuLinearWeight, CpuTensor};
use infernum_cpu::CpuBackend;

use crate::config::GemmaConfig;
use crate::graph_builder::{build_decode_graph, build_prefill_graph, load_graph_weights_gguf};

// ---------------------------------------------------------------------------
// GraphEngineConfig impl
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

impl GraphEngineConfig for GemmaConfig {
    impl_common_config_getters!();

    fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn build_prefill_graph(&self, seq_len: usize) -> Graph<CpuBackend> {
        build_prefill_graph::<CpuBackend>(self, seq_len, DType::F32)
    }

    fn build_decode_graph(&self, kv_len: usize) -> Graph<CpuBackend> {
        build_decode_graph::<CpuBackend>(self, kv_len, DType::F32)
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

/// CPU graph-mode engine for Gemma-family models (Gemma 2 and Gemma 3 text).
///
/// A type alias for [`infernum_cpu::GraphEngine<GemmaConfig>`].
/// See [`infernum_cpu::GraphEngine`] for the full API.
pub type GemmaGraphEngine = infernum_cpu::GraphEngine<GemmaConfig>;

/// Extension trait providing Gemma-specific constructors on [`GemmaGraphEngine`].
pub trait GemmaGraphEngineExt: Sized {
    /// Load a Gemma-family model from a `SafeTensors` directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing or weights cannot be loaded.
    ///
    /// # Errors
    ///
    /// Returns an error if `config.json` is missing, cannot be parsed, or
    /// contains an unsupported `model_type`.
    fn from_pretrained(model_dir: &Path) -> Result<Self>;

    /// Load a Gemma-family model from a GGUF file.
    ///
    /// # Errors
    ///
    /// Returns an error if the GGUF file cannot be opened, cannot be parsed,
    /// or contains unsupported quantization types.
    fn from_gguf(gguf_path: &Path) -> Result<Self>;
}

impl GemmaGraphEngineExt for GemmaGraphEngine {
    fn from_pretrained(model_dir: &Path) -> Result<Self> {
        let config = GemmaConfig::from_file(model_dir.join("config.json"))?;
        infernum_cpu::GraphEngine::from_config_and_dir(config, model_dir)
    }

    fn from_gguf(gguf_path: &Path) -> Result<Self> {
        let loader =
            infernum::weights::gguf::GgufLoader::from_file(infernum::path_to_utf8(gguf_path)?)?;
        let config = GemmaConfig::from_gguf_metadata(loader.metadata())?;
        infernum_cpu::GraphEngine::from_gguf_with_config(config, gguf_path)
    }
}
