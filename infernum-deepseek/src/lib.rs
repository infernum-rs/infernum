//! `DeepSeek` V3 / R1 model family implementation for Infernum
//!
//! Supports DeepSeek-V3 and DeepSeek-R1 (same architecture, `model_type: "deepseek_v3"`).

mod chat_templates;
mod config;
mod model;

pub use chat_templates::DeepSeekTemplate;
pub use config::DeepSeekConfig;
pub use model::{split_kv_b_proj_dense, DeepSeekModel, DeepSeekOps};
