//! Infernum: A Rust-based LLM inference server
//!
//! This crate provides the core functionality for running LLM inference on CUDA GPUs.
//!
//! # Features
//!
//! - `cuda` - Enable CUDA GPU support (requires CUDA toolkit)

// Allow proc macros to use `::infernum::` paths from within this crate.
extern crate self as infernum;

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod dtype;
pub mod error;
pub mod fusion;
pub mod model;
pub mod sampling;
pub mod tensor;
pub mod tokenizer;
pub mod weights;

#[cfg(feature = "cuda")]
pub use cuda::QuantizedTensor;
#[cfg(feature = "cuda")]
pub use cuda::{CudaTensor, KvCache};
pub use dtype::{DType, QUANTIZATION_BLOCK_SIZE};
pub use error::{Error, Result};
#[cfg(feature = "cuda")]
pub use model::Model;
pub use model::ModelConfig;
pub use sampling::{GenerateOptions, SamplingParams};
pub use tensor::Tensor;
#[cfg(feature = "cuda")]
pub use tokenizer::GgufTokenizer;
pub use tokenizer::Tokenizer;
#[cfg(feature = "cuda")]
pub use weights::GgufLoader;
