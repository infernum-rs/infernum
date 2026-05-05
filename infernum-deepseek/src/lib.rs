//! `DeepSeek` V3 / R1 model family implementation for Infernum
//!
//! Supports DeepSeek-V3 and DeepSeek-R1 (same architecture, `model_type: "deepseek_v3"`).

mod chat_templates;
mod config;
pub mod weights;

#[cfg(feature = "cpu")]
pub mod graph_builder;
// graph_engine is deferred: CpuBackend does not yet implement MlaAttentionOps.
// Once MlaAttentionOps is implemented for CpuBackend, re-enable this module.
// #[cfg(feature = "cpu")]
// pub mod graph_engine;

pub use chat_templates::DeepSeekTemplate;
pub use config::DeepSeekConfig;
#[cfg(feature = "cpu")]
pub use graph_builder::load_graph_weights_safetensors;
// #[cfg(feature = "cpu")]
// pub use graph_engine::DeepSeekGraphEngine;
pub use weights::split_kv_b_proj_dense;
