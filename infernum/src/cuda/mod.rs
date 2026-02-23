//! CUDA backend implementation

pub mod block_allocator;
pub mod buffer_pool;
mod context;
mod graph;
pub mod kv_cache;
pub mod moe;
#[cfg(feature = "nccl")]
pub mod nccl;
pub mod ops;
mod quantized;
pub mod seq_position;
pub mod shard;
#[cfg(feature = "nccl")]
mod sharded;
mod tensor;

pub use block_allocator::{BlockAllocator, BlockConfig, BlockTable};
pub use buffer_pool::{BufferPool, PooledSlice};
pub use context::CudaContext;
#[cfg(feature = "nccl")]
pub use cudarc::nccl::safe::NcclType;
pub use graph::CudaGraph;
pub use kv_cache::KvCache;
#[cfg(feature = "nccl")]
pub use nccl::NcclCommunicator;
pub use quantized::QuantizedTensor;
pub use seq_position::SeqPosition;
pub use shard::{GpuConfig, ShardConfig, ShardStrategy};
#[cfg(feature = "nccl")]
pub use sharded::ShardedModel;
pub use tensor::CudaTensor;

// Re-export cudarc types/traits needed by downstream generic model code
pub use cudarc::cublas::{CudaBlas, Gemm};
pub use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};
