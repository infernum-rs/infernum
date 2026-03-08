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
pub mod logits;
pub mod ops;
pub mod tensor;
pub mod weights;

use infernum::backend::Backend;

pub use context::MetalContext;
pub use logits::MetalLogits;
pub use tensor::MetalTensor;
pub use weights::{MetalLinearWeight, MetalSafeTensorsLoader};

/// Paged KV cache for Metal (block-based, used by most attention mechanisms).
pub struct MetalPagedKvCache {
    // Placeholder — implemented in Step 8
    _private: (),
}

// SAFETY: MetalPagedKvCache will contain Metal buffers which are thread-safe
// when accessed through command buffers.
unsafe impl Send for MetalPagedKvCache {}

/// Contiguous KV cache for Metal (used by DeepSeek MLA).
pub struct MetalKvCache {
    // Placeholder — implemented in Step 9
    _private: (),
}

// SAFETY: MetalKvCache will contain Metal buffers which are thread-safe
// when accessed through command buffers.
unsafe impl Send for MetalKvCache {}

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
    type Logits = MetalLogits;
    type Comm = ();

    fn logits_from_tensor(tensor: MetalTensor) -> MetalLogits {
        MetalLogits::from_tensor(tensor)
    }
}
