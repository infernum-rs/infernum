//! CPU backend for Infernum.
//!
//! Provides a pure-CPU inference path using AVX2+FMA (x86-64) or NEON
//! (AArch64) SIMD. All compute is done in f32; bf16/f16 weights are cast
//! on load.
//!
//! This backend is intended for debugging, testing, and running on
//! machines without a GPU. It is not optimised for throughput.

#![allow(
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions
)]

pub mod logits;
pub mod ops;
pub mod simd;
pub mod tensor;
pub mod weights;

use infernum::backend::Backend;

pub use logits::CpuLogits;
pub use ops::attention::{CpuKvCache, CpuPagedKvCache};
pub use tensor::CpuTensor;
pub use weights::CpuSafeTensorsLoader;

/// Marker type for the CPU backend.
///
/// All op trait impls are on this type. Models parameterised by
/// `B: Backend` can use `CpuBackend` to run on CPU.
pub struct CpuBackend;

impl Backend for CpuBackend {
    type Tensor = CpuTensor;
    type DeviceHandle = ();
    type PagedKvCache = CpuPagedKvCache;
    type KvCache = CpuKvCache;
    type RuntimeState = ();
    type Logits = CpuLogits;
    type Comm = ();

    fn logits_from_tensor(tensor: CpuTensor) -> CpuLogits {
        CpuLogits::from_tensor(tensor)
    }
}

/// Check that the current CPU supports the required SIMD features.
///
/// Call this once at startup before using the backend.
///
/// # Errors
/// Returns an error if AVX2+FMA is missing on x86-64.
pub fn check_cpu_support() -> infernum::Result<()> {
    simd::check_cpu_support()
}
