//! Infernum Runtime: text-in, text-out inference
//!
//! This crate provides the [`Engine`] (token-level) and [`Runtime`] (text-level)
//! abstractions for running LLM inference.
//!
//! # Architecture
//!
//! ```text
//! Runtime<M, T>    ← text in, text out (owns Engine + Tokenizer)
//!   └── Engine<M>  ← tokens in, tokens out (owns Model + KvCache)
//!         └── M: Model  ← forward pass only
//! ```
//!
//! # Features
//!
//! - `cuda` - Enable CUDA GPU support (requires CUDA toolkit)

#[cfg(feature = "cuda")]
mod engine;
#[cfg(feature = "cuda")]
mod runtime;

#[cfg(feature = "cuda")]
pub use engine::Engine;
#[cfg(feature = "cuda")]
pub use engine::{FinishReason, GenerationEvent, TokenSender};
#[cfg(feature = "cuda")]
pub use runtime::Runtime;
