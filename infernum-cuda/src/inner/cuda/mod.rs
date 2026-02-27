//! CUDA backend implementation

#[allow(dead_code)]
mod batched_graph;
pub mod buffer_pool;
mod context;
mod graph;
pub mod kv_cache;
pub mod moe;
#[cfg(feature = "nccl")]
pub mod nccl;
pub mod ops;
pub mod paged_kv_cache;
mod quantized;
pub mod seq_position;
#[cfg(feature = "nccl")]
mod sharded;
mod tensor;

pub use buffer_pool::{BufferPool, PooledSlice};
pub use context::CudaContext;
pub use graph::CudaGraph;
pub use kv_cache::KvCache;
#[cfg(feature = "nccl")]
pub use nccl::NcclCommunicator;
pub use paged_kv_cache::PagedKvCache;
pub use quantized::QuantizedTensor;
pub use seq_position::SeqPosition;
#[cfg(feature = "nccl")]
pub use sharded::{ShardedKvCache, ShardedModel};
pub use tensor::CudaTensor;

// Re-export from infernum core (pure data types, no CUDA dependency)
pub use infernum::block_allocator::{BlockAllocator, BlockConfig, BlockTable};
pub use infernum::shard::{shard_strategy_for_weight, GpuConfig, ShardConfig, ShardStrategy};

// Re-export cudarc types/traits needed by downstream generic model code
pub use cudarc::cublas::{CudaBlas, Gemm};
pub use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};
