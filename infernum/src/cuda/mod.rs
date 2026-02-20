//! CUDA backend implementation

mod context;
pub mod kv_cache;
pub mod ops;
mod quantized;
mod tensor;

pub use context::CudaContext;
pub use kv_cache::KvCache;
pub use quantized::QuantizedTensor;
pub use tensor::CudaTensor;

// Re-export cudarc traits needed by downstream generic model code
pub use cudarc::cublas::{CudaBlas, Gemm};
pub use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
