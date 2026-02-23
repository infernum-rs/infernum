//! Gemma model family implementation for Infernum (Gemma 2 + Gemma 3)
//!
//! Supports Gemma 2 (`model_type: "gemma2"`) and Gemma 3 text
//! (`model_type: "gemma3_text"`). A single [`GemmaModel`] handles both
//! generations, with Gemma 3 differences (QK-norm, dual-theta `RoPE`,
//! no logit soft-capping) toggled by config fields.

#![allow(clippy::doc_markdown)]

mod config;
#[cfg(feature = "cuda")]
mod model;

pub use config::GemmaConfig;
#[cfg(feature = "cuda")]
pub use model::GemmaModel;

/// Type alias for Gemma 2 models (`model_type: "gemma2"`)
#[cfg(feature = "cuda")]
pub type Gemma2Model<T> = GemmaModel<T>;

/// Type alias for Gemma 3 text models (`model_type: "gemma3_text"`)
#[cfg(feature = "cuda")]
pub type Gemma3Model<T> = GemmaModel<T>;
