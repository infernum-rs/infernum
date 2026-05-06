//! Llama model implementation for Infernum

mod chat_templates;
mod config;
#[cfg(feature = "cuda")]
pub mod cuda_graph_engine;
pub mod graph_builder;
#[cfg(feature = "cpu")]
pub mod graph_engine;

pub use chat_templates::{Llama3Template, MistralTemplate};
pub use config::{LlamaConfig, QuantizationConfig};
#[cfg(feature = "cuda")]
pub use cuda_graph_engine::{LlamaCudaGraphEngine, LlamaCudaGraphEngineExt};
pub use graph_builder::{
    build_decode_graph, build_indirect_decode_graph, build_paged_decode_graph, build_prefill_graph,
    IndirectDecodeExtraIds, LayerWeightIds, LlamaGraphOps, ModelWeightIds,
};
#[cfg(feature = "cpu")]
pub use graph_builder::{load_graph_weights_gguf, load_graph_weights_safetensors};
#[cfg(feature = "cpu")]
pub use graph_engine::{GraphKvCache, LlamaGraphEngine, LlamaGraphEngineExt};
