//! Llama model implementation for Infernum
//!
//! Also supports Mistral and Mixtral models, which are architecturally
//! identical to Llama. [`MistralModel`] and [`MixtralModel`] are provided
//! as type aliases for API clarity.

mod config;
#[cfg(feature = "cuda")]
mod model;

pub use config::{LlamaConfig, QuantizationConfig};
#[cfg(feature = "cuda")]
pub use model::LlamaModel;

/// Mistral model (architecturally identical to Llama).
#[cfg(feature = "cuda")]
pub type MistralModel<T> = LlamaModel<T>;

/// Mixtral model (Llama + Mixture-of-Experts).
#[cfg(feature = "cuda")]
pub type MixtralModel<T> = LlamaModel<T>;
