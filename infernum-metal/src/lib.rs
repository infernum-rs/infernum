//! Metal backend for Infernum.
//!
//! Provides GPU-accelerated inference on Apple Silicon using the Metal API.
//! Uses unified memory (`StorageModeShared`) for near-zero-cost host↔device
//! data transfer.
//!
//! This backend is designed for Apple Silicon Macs (M1–M4) and requires
//! macOS with Metal 3 support.

#![allow(
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions
)]

pub mod context;
pub mod execute_context;
pub mod executor;
pub mod graph_engine;
pub mod logits;
pub mod ops;
pub mod tensor;
pub mod weights;

use infernum::backend::Backend;

pub use context::MetalContext;
pub use graph_engine::{load_graph_weights_metal, MetalGraphEngine, MetalGraphEngineConfig};
pub use logits::MetalLogits;
pub use tensor::MetalTensor;
pub use weights::{MetalLinearWeight, MetalQuantizedWeight, MetalSafeTensorsLoader};

/// Paged KV cache for Metal (block-based, used by most attention mechanisms).
///
/// Uses Metal buffers with `StorageModeShared` so the CPU can directly
/// read/write the KV pool data through unified memory.
pub struct MetalPagedKvCache {
    /// Per-layer K pool tensors: shape `(num_blocks * block_size, num_kv_heads, head_dim)`.
    pub(crate) k_pools: Vec<MetalTensor>,
    /// Per-layer V pool tensors: same shape.
    pub(crate) v_pools: Vec<MetalTensor>,
    pub(crate) block_size: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
}

// SAFETY: MetalTensor is Send+Sync; Metal shared-mode buffers are thread-safe.
unsafe impl Send for MetalPagedKvCache {}

/// Contiguous KV cache for Metal (used by DeepSeek MLA).
///
/// Stores per-layer K/V data that grows as tokens are appended.
/// Data is kept in Metal buffers via unified memory.
pub struct MetalKvCache {
    pub(crate) layers: Vec<MetalKvLayer>,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) ctx: MetalContext,
}

pub(crate) struct MetalKvLayer {
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) len: usize,
}

// SAFETY: MetalKvCache contains a MetalContext (Arc-wrapped, thread-safe) and Vec data.
unsafe impl Send for MetalKvCache {}

/// Executor state for graph-mode inference on Metal.
///
/// Mirrors the CUDA executor state: a 2D grid of optional tensors indexed
/// by `(node_id, output_idx)`. Each graph node writes its outputs here
/// during execution, and downstream nodes read them via `ctx_read`.
pub struct MetalExecutorState {
    /// Per-node output buffers: `buffers[node_id][output_idx]`.
    pub buffers: Vec<Vec<Option<MetalTensor>>>,
}

// SAFETY: `MetalExecutorState` is only constructed and used within a single
// `execute` call on one thread.
unsafe impl Send for MetalExecutorState {}

/// Marker type for the Metal backend.
///
/// All op trait impls are on this type. Models parameterised by
/// `B: Backend` can use `MetalBackend` to run on Apple Silicon GPUs.
pub struct MetalBackend;

impl Backend for MetalBackend {
    type Tensor = MetalTensor;
    type DeviceHandle = MetalContext;
    type PagedKvCache = MetalPagedKvCache;
    type KvCache = MetalKvCache;
    type RuntimeState = ();
    type ExecutorState = MetalExecutorState;
    type Logits = MetalLogits;
    type Comm = ();

    fn logits_from_tensor(tensor: MetalTensor) -> MetalLogits {
        MetalLogits::from_tensor(tensor)
    }
}
