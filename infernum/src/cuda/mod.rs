//! CUDA backend implementation

pub mod buffer_pool;
mod context;
mod graph;
pub mod kv_cache;
pub mod ops;
mod quantized;
pub mod seq_position;
mod tensor;

pub use buffer_pool::BufferPool;
pub use context::CudaContext;
pub use graph::CudaGraph;
pub use kv_cache::KvCache;
pub use quantized::QuantizedTensor;
pub use seq_position::SeqPosition;
pub use tensor::CudaTensor;

// Re-export cudarc types/traits needed by downstream generic model code
pub use cudarc::cublas::{CudaBlas, Gemm};
pub use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};
