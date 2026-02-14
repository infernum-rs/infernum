//! Error types for Infernum

use thiserror::Error;

/// Result type alias using Infernum's Error
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for Infernum operations
#[derive(Error, Debug)]
pub enum Error {
    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),

    #[cfg(feature = "cuda")]
    #[error("cuBLAS error: {0}")]
    CuBlas(#[from] cudarc::cublas::result::CublasError),

    #[cfg(feature = "cuda")]
    #[error("NVRTC error: {0}")]
    Nvrtc(#[from] cudarc::nvrtc::result::NvrtcError),

    #[cfg(feature = "cuda")]
    #[error("NVRTC compile error: {0}")]
    NvrtcCompile(#[from] cudarc::nvrtc::CompileError),

    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("Dtype mismatch: expected {expected}, got {got}")]
    DtypeMismatch { expected: String, got: String },

    #[error("SafeTensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    #[error("Weight not found: {0}")]
    WeightNotFound(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Unsupported dtype: {0}")]
    UnsupportedDtype(String),
}
