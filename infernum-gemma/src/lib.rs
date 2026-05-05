//! Gemma model family implementation for Infernum (Gemma 2 + Gemma 3)
//!
//! Supports Gemma 2 (`model_type: "gemma2"`) and Gemma 3 text
//! (`model_type: "gemma3_text"`). A single [`GemmaModel`] handles both
//! generations, with Gemma 3 differences (QK-norm, dual-theta `RoPE`,
//! no logit soft-capping) toggled by config fields.

#![allow(clippy::doc_markdown)]

mod chat_templates;
mod config;
#[cfg(feature = "cuda")]
pub mod cuda_graph_engine;
pub mod graph_builder;
#[cfg(feature = "cpu")]
pub mod graph_engine;

pub use chat_templates::GemmaTemplate;
pub use config::GemmaConfig;
#[cfg(feature = "cuda")]
pub use cuda_graph_engine::{GemmaCudaGraphEngine, GemmaCudaGraphEngineExt};
pub use graph_builder::{
    build_decode_graph, build_prefill_graph, GemmaGraphOps, LayerWeightIds, ModelWeightIds,
    QkNormIds,
};
#[cfg(feature = "cpu")]
pub use graph_builder::{load_graph_weights_gguf, load_graph_weights_safetensors};
#[cfg(feature = "cpu")]
pub use graph_engine::{GemmaGraphEngine, GemmaGraphEngineExt};
