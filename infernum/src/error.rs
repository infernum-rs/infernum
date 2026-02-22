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
    #[error("cuBLASLt error: {0}")]
    CuBlasLt(#[from] cudarc::cublaslt::result::CublasError),

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

    #[cfg(feature = "cuda")]
    #[error("CUDA graph error: {0}")]
    CudaGraph(String),

    #[cfg(feature = "nccl")]
    #[error("NCCL error: {0}")]
    Nccl(#[from] NcclErrorWrapper),
}

/// Wrapper around `cudarc::nccl::result::NcclError` that provides useful
/// `Debug` and `Display` output. The upstream type's `Debug` impl discards
/// the `ncclResult_t` error code, printing only the opaque string "`NcclError`".
#[cfg(feature = "nccl")]
pub struct NcclErrorWrapper(pub cudarc::nccl::result::NcclError);

#[cfg(feature = "nccl")]
impl NcclErrorWrapper {
    fn error_name(&self) -> &'static str {
        use cudarc::nccl::sys::ncclResult_t;
        match self.0 .0 {
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
        }
    }
}

#[cfg(feature = "nccl")]
impl std::fmt::Debug for NcclErrorWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error_name())
    }
}

#[cfg(feature = "nccl")]
impl std::fmt::Display for NcclErrorWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error_name())
    }
}

#[cfg(feature = "nccl")]
impl std::error::Error for NcclErrorWrapper {}

#[cfg(feature = "nccl")]
impl From<cudarc::nccl::result::NcclError> for Error {
    fn from(e: cudarc::nccl::result::NcclError) -> Self {
        Self::Nccl(NcclErrorWrapper(e))
    }
}
