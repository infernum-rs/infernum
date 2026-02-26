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

mod engine;
mod runtime;
mod scheduler;

pub use engine::Engine;
pub use runtime::Runtime;
pub use scheduler::{
    BatchConfig, DecodeTask, FinishReason, GenerationEvent, PrefillTask, Scheduler,
    SchedulerOutput, SequencePhase, SequenceState, TokenSender,
};
