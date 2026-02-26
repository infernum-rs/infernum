mod backend_impl;
pub mod cuda;
mod error;
pub mod model;
pub mod weights;

pub use backend_impl::CudaBackend;

pub use cuda::ops::LinearWeight;
pub use cuda::BatchedGraphInputs;
pub use cuda::BlockAllocator;
pub use cuda::BlockConfig;
pub use cuda::BlockTable;
pub use cuda::BufferPool;
pub use cuda::CudaContext;
pub use cuda::CudaGraph;
pub use cuda::CudaTensor;
pub use cuda::KvCache;
pub use cuda::PagedKvCache;
pub use cuda::QuantizedTensor;
pub use cuda::SeqPosition;
#[cfg(feature = "nccl")]
pub use cuda::{nccl::NcclId, NcclCommunicator, ShardedModel};
pub use cuda::{GpuConfig, ShardConfig, ShardStrategy};
pub use model::Model;
#[cfg(feature = "nccl")]
pub use model::ShardedLoadable;
pub use weights::{GgufLoader, GgufValue, SafeTensorsLoader, WeightLoader};

// Re-export cudarc types/traits needed by downstream generic model code
pub use cudarc::cublas::{CudaBlas, Gemm};
pub use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};
