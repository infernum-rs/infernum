//! CUDA backend implementation

mod batched_graph;
pub mod block_allocator;
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
pub mod shard;
#[cfg(feature = "nccl")]
mod sharded;
mod tensor;

pub use batched_graph::BatchedGraphInputs;
pub use block_allocator::{BlockAllocator, BlockConfig, BlockTable};
pub use buffer_pool::{BufferPool, PooledSlice};
pub use context::CudaContext;
pub use graph::CudaGraph;
pub use kv_cache::KvCache;
#[cfg(feature = "nccl")]
pub use nccl::NcclCommunicator;
pub use paged_kv_cache::PagedKvCache;
pub use quantized::QuantizedTensor;
pub use seq_position::SeqPosition;
pub use shard::{shard_strategy_for_weight, GpuConfig, ShardConfig, ShardStrategy};
#[cfg(feature = "nccl")]
pub use sharded::ShardedModel;
pub use tensor::CudaTensor;

// Re-export cudarc types/traits needed by downstream generic model code
pub use cudarc::cublas::{CudaBlas, Gemm};
pub use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};
