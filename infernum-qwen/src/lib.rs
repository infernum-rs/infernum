//! Qwen model family implementation for Infernum
//!
//! Supports Qwen2/2.5 (dense, Q/K/V bias), Qwen3/3.5 (dense, QK-norm),
//! and Qwen3-MoE (shared expert, Q/K/V bias).

mod chat_templates;
mod config;
#[cfg(feature = "cuda")]
mod model;

pub use chat_templates::ChatMLTemplate;
pub use config::QwenConfig;
#[cfg(feature = "cuda")]
pub use model::QwenModel;
