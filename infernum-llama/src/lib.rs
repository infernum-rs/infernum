//! Llama model implementation for Infernum
//!
//! Also supports Mistral and Mixtral models, which are architecturally
//! identical to Llama. [`MistralModel`] and [`MixtralModel`] are provided
//! as type aliases for API clarity.

mod chat_templates;
mod config;
#[cfg(feature = "cuda")]
pub mod cuda_graph_engine;
pub mod graph_builder;
#[cfg(feature = "cpu")]
pub mod graph_engine;
mod model;

pub use chat_templates::{Llama3Template, MistralTemplate};
pub use config::{LlamaConfig, QuantizationConfig};
#[cfg(feature = "cuda")]
pub use cuda_graph_engine::{LlamaCudaGraphEngine, LlamaCudaGraphEngineExt};
pub use graph_builder::{
    build_decode_graph, build_indirect_decode_graph, build_prefill_graph, IndirectDecodeExtraIds,
    LayerWeightIds, LlamaGraphOps, ModelWeightIds,
};
#[cfg(feature = "cpu")]
pub use graph_builder::{load_graph_weights_gguf, load_graph_weights_safetensors};
#[cfg(feature = "cpu")]
pub use graph_engine::{GraphKvCache, LlamaGraphEngine, LlamaGraphEngineExt};
pub use model::LlamaModel;

/// Mistral model (architecturally identical to Llama).
pub type MistralModel<B> = LlamaModel<B>;

/// Mixtral model (Llama + Mixture-of-Experts).
pub type MixtralModel<B> = LlamaModel<B>;
