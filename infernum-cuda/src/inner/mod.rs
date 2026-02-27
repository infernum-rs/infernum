mod backend_impl;
pub mod cuda;
pub mod cuda_logits;
pub mod cuda_runtime_state;
mod error;
pub mod model;
pub mod weights;

pub use backend_impl::CudaBackend;
pub use cuda_logits::CudaLogits;
pub use cuda_runtime_state::CudaRuntimeState;

pub use cuda::ops::LinearWeight;
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
pub use cuda::{GpuConfig, ShardConfig, ShardStrategy};
#[cfg(feature = "nccl")]
pub use cuda::{NcclCommunicator, NcclId};
pub use weights::{CudaWeightLoader, GgufLoader, GgufValue, SafeTensorsLoader, WeightLoader};

// Re-export cudarc types/traits needed by downstream generic model code
pub use cudarc::cublas::{CudaBlas, Gemm};
pub use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};
