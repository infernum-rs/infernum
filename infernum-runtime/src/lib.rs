//! Infernum Runtime: text-in, text-out inference
//!
//! This crate provides the [`Engine`] (token-level) and [`Runtime`] (text-level)
//! abstractions for running LLM inference.
//!
//! # Architecture
//!
//! ```text
//! Runtime<M, T>    ← text in, text out (owns Engine + Tokenizer)
//!   └── Engine<M>  ← tokens in, tokens out (owns Model + PagedKvCache)
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
mod scheduler;

mod engine2;
mod runtime2;
mod scheduler2;

#[cfg(feature = "cuda")]
pub use engine::Engine;
#[cfg(feature = "cuda")]
pub use engine::{FinishReason, GenerationEvent, TokenSender};
#[cfg(feature = "cuda")]
pub use runtime::Runtime;
pub use scheduler::BatchConfig;
#[cfg(feature = "cuda")]
pub use scheduler::{
    DecodeTask, PrefillTask, Scheduler, SchedulerOutput, SequencePhase, SequenceState,
};

pub use engine2::Engine2;
pub use runtime2::Runtime2;
pub use scheduler2::BatchConfig as BatchConfig2;
pub use scheduler2::{
    FinishReason as FinishReason2, GenerationEvent as GenerationEvent2, Scheduler2,
    TokenSender as TokenSender2,
};
