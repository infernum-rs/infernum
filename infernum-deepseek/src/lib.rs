//! `DeepSeek` V3 / R1 model family implementation for Infernum
//!
//! Supports DeepSeek-V3 and DeepSeek-R1 (same architecture, `model_type: "deepseek_v3"`).

mod chat_templates;
mod config;
#[cfg(feature = "cuda")]
mod model;

pub use chat_templates::DeepSeekTemplate;
pub use config::DeepSeekConfig;
#[cfg(feature = "cuda")]
pub use model::DeepSeekModel;
