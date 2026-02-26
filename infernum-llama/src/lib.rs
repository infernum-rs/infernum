//! Llama model implementation for Infernum
//!
//! Also supports Mistral and Mixtral models, which are architecturally
//! identical to Llama. [`MistralModel`] and [`MixtralModel`] are provided
//! as type aliases for API clarity.

mod chat_templates;
mod config;
#[cfg(feature = "cuda")]
mod model;

pub use chat_templates::{Llama3Template, MistralTemplate};
pub use config::{LlamaConfig, QuantizationConfig};
#[cfg(feature = "cuda")]
pub use model::LlamaModel;

/// Mistral model (architecturally identical to Llama).
#[cfg(feature = "cuda")]
pub type MistralModel<B> = LlamaModel<B>;

/// Mixtral model (Llama + Mixture-of-Experts).
#[cfg(feature = "cuda")]
pub type MixtralModel<B> = LlamaModel<B>;
