//! Infernum: A Rust-based LLM inference server
//!
//! This crate provides the core functionality for running LLM inference on CUDA GPUs.
//!
//! # Features
//!
//! - `cuda` - Enable CUDA GPU support (requires CUDA toolkit)

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod dtype;
pub mod error;
pub mod tensor;
pub mod tokenizer;
pub mod weights;

#[cfg(feature = "cuda")]
pub use cuda::CudaTensor;
pub use dtype::DType;
pub use error::{Error, Result};
pub use tensor::Tensor;
