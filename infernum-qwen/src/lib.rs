//! Qwen model family implementation for Infernum
//!
//! Supports Qwen2/2.5 (dense, Q/K/V bias), Qwen3/3.5 (dense, QK-norm),
//! and Qwen3-MoE (shared expert, Q/K/V bias).

mod chat_templates;
mod config;
pub mod graph_builder;
#[cfg(feature = "cpu")]
pub mod graph_engine;
mod model;

pub use chat_templates::ChatMLTemplate;
pub use config::QwenConfig;
#[cfg(feature = "cpu")]
pub use graph_builder::load_graph_weights_safetensors;
pub use graph_builder::{
    build_decode_graph, build_prefill_graph, DenseLayerWeightIds, LayerWeightIds, ModelWeightIds,
    MoeLayerWeightIds, QkNormIds, QkvBiasIds, QwenGraphOps,
};
#[cfg(feature = "cpu")]
pub use graph_engine::{QwenGraphEngine, QwenGraphEngineExt};
pub use model::{QwenModel, QwenOps};
