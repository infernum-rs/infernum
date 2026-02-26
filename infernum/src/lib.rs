//! Infernum: A Rust-based LLM inference server
//!
//! This crate provides the core types and traits for LLM inference.
//! Backend-specific implementations (CUDA, CPU, etc.) are in separate crates.

// Allow proc macros to use `::infernum::` paths from within this crate.
extern crate self as infernum;

pub mod block_allocator;
pub mod chat_template;
pub mod dtype;
pub mod error;
pub mod fusion;
pub mod gguf_meta;
pub mod model;
pub mod sampling;
pub mod shard;
pub mod tensor;
pub mod tokenizer;

pub use block_allocator::{BlockAllocator, BlockConfig, BlockTable};
pub use dtype::{DType, GPTQ_GROUP_SIZE, QUANTIZATION_BLOCK_SIZE};
pub use error::{Error, Result};
pub use gguf_meta::GgufValue;
pub use model::ModelConfig;
pub use sampling::{GenerateOptions, SamplingParams};
pub use shard::{shard_strategy_for_weight, GpuConfig, ShardConfig, ShardStrategy};
pub use tensor::Tensor;
pub use tokenizer::GgufTokenizer;
pub use tokenizer::Tokenizer;

pub use chat_template::{ChatMessage, ChatTemplate, RawTemplate};
