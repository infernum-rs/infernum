//! Gemma model family implementation for Infernum (Gemma 2 + Gemma 3)
//!
//! Supports Gemma 2 (`model_type: "gemma2"`) and Gemma 3 text
//! (`model_type: "gemma3_text"`). A single [`GemmaModel`] handles both
//! generations, with Gemma 3 differences (QK-norm, dual-theta `RoPE`,
//! no logit soft-capping) toggled by config fields.

#![allow(clippy::doc_markdown)]

mod chat_templates;
mod config;
pub mod graph_builder;
mod model;

pub use chat_templates::GemmaTemplate;
pub use config::GemmaConfig;
pub use graph_builder::{
    build_decode_graph, build_prefill_graph, GemmaGraphOps, LayerWeightIds, ModelWeightIds,
    QkNormIds,
};
#[cfg(feature = "cpu")]
pub use graph_builder::{load_graph_weights_gguf, load_graph_weights_safetensors};
pub use model::GemmaModel;

/// Type alias for Gemma 2 models (`model_type: "gemma2"`)
pub type Gemma2Model<B> = GemmaModel<B>;

/// Type alias for Gemma 3 text models (`model_type: "gemma3_text"`)
pub type Gemma3Model<B> = GemmaModel<B>;
