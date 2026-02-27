//! Llama model implementation for Infernum
//!
//! Also supports Mistral and Mixtral models, which are architecturally
//! identical to Llama. [`MistralModel`] and [`MixtralModel`] are provided
//! as type aliases for API clarity.

mod chat_templates;
mod config;
#[cfg(feature = "nccl")]
mod cuda_model;
mod model;

pub use chat_templates::{Llama3Template, MistralTemplate};
pub use config::{LlamaConfig, QuantizationConfig};
pub use model::LlamaModel;

/// Mistral model (architecturally identical to Llama).
pub type MistralModel<B> = LlamaModel<B>;

/// Mixtral model (Llama + Mixture-of-Experts).
pub type MixtralModel<B> = LlamaModel<B>;
