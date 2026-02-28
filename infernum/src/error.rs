//! Error types for Infernum

use thiserror::Error;

/// Result type alias using Infernum's Error
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for Infernum operations
#[derive(Error, Debug)]
pub enum Error {
    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("cuBLAS error: {0}")]
    CuBlas(String),

    #[error("cuBLASLt error: {0}")]
    CuBlasLt(String),

    #[error("NVRTC error: {0}")]
    Nvrtc(String),

    #[error("NVRTC compile error: {0}")]
    NvrtcCompile(String),

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
    SafeTensors(String),

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

    #[error("Unsupported model: {0}")]
    UnsupportedModel(String),

    #[error("CUDA graph error: {0}")]
    CudaGraph(String),

    #[error("NCCL error: {0}")]
    Nccl(String),

    #[error("{0}")]
    Other(String),
}

// ---------------------------------------------------------------------------
// Optional From impls for cudarc error types (enabled by `cuda-errors` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda-errors")]
impl From<cudarc::driver::DriverError> for Error {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self::Cuda(e.to_string())
    }
}

#[cfg(feature = "cuda-errors")]
impl From<cudarc::cublas::result::CublasError> for Error {
    fn from(e: cudarc::cublas::result::CublasError) -> Self {
        Self::CuBlas(format!("{e:?}"))
    }
}

#[cfg(feature = "cuda-errors")]
impl From<cudarc::cublaslt::result::CublasError> for Error {
    fn from(e: cudarc::cublaslt::result::CublasError) -> Self {
        Self::CuBlasLt(format!("{e:?}"))
    }
}

#[cfg(feature = "cuda-errors")]
impl From<cudarc::nvrtc::result::NvrtcError> for Error {
    fn from(e: cudarc::nvrtc::result::NvrtcError) -> Self {
        Self::Nvrtc(format!("{e:?}"))
    }
}

#[cfg(feature = "cuda-errors")]
impl From<cudarc::nvrtc::CompileError> for Error {
    fn from(e: cudarc::nvrtc::CompileError) -> Self {
        Self::NvrtcCompile(e.to_string())
    }
}

#[cfg(feature = "cuda-errors")]
impl From<safetensors::SafeTensorError> for Error {
    fn from(e: safetensors::SafeTensorError) -> Self {
        Self::SafeTensors(e.to_string())
    }
}

#[cfg(feature = "nccl-errors")]
impl From<cudarc::nccl::result::NcclError> for Error {
    fn from(e: cudarc::nccl::result::NcclError) -> Self {
        use cudarc::nccl::sys::ncclResult_t;
        let name = match e.0 {
            ncclResult_t::ncclUnhandledCudaError => {
                "ncclUnhandledCudaError (an unhandled CUDA error)"
            }
            ncclResult_t::ncclSystemError => "ncclSystemError (a system call failed)",
            ncclResult_t::ncclInternalError => "ncclInternalError (an internal NCCL error)",
            ncclResult_t::ncclInvalidArgument => {
                "ncclInvalidArgument (an invalid argument was passed)"
            }
            ncclResult_t::ncclInvalidUsage => "ncclInvalidUsage (invalid API usage)",
            _ => "unknown NCCL error code",
        };
        Self::Nccl(name.to_string())
    }
}
