//! CUDA backend for Infernum
//!
//! This crate contains all CUDA-specific code: tensors, ops, kernels,
//! KV caches, weight loaders, the `Model` trait, and NCCL sharding.
//!
//! Without the `cuda` feature the crate compiles as an empty shell, which
//! lets `cargo clippy --all` succeed on CI without a CUDA toolkit.

// All CUDA modules live inside `inner`. When adding new modules, add them
// there (not here) so the feature gate stays in one place.
#[cfg(feature = "cuda")]
mod inner;

#[cfg(feature = "cuda")]
pub use inner::*;

// Re-export infernum core types that are commonly used alongside CUDA types
pub use infernum::DType;
pub use infernum::Error;
pub use infernum::Result;
pub use infernum::Tensor;
