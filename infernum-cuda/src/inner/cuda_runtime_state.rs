//! CUDA backend runtime state.
//!
//! Holds CUDA graph capture/replay state. The backend internally decides
//! whether to use graphs based on kernel compatibility (e.g., BF16 kernels
//! are graph-safe, some quantised kernels are not).

use infernum::block_allocator::BlockConfig;
use infernum::runtime_state::{BatchConfig, RuntimeStateInit};
use infernum::Result;

/// Runtime state for the CUDA backend.
///
/// Currently a placeholder — graph capture/replay state will be moved
/// here from the Engine when the full migration is complete.
pub struct CudaRuntimeState {
    _private: (),
}

impl CudaRuntimeState {
    /// Create a placeholder runtime state (no graph capture).
    #[cfg(feature = "nccl")]
    pub(crate) fn new_placeholder() -> Self {
        Self { _private: () }
    }

    /// Create a placeholder for tests that call model methods directly.
    #[must_use]
    pub fn test_placeholder() -> Self {
        Self { _private: () }
    }
}

impl RuntimeStateInit for CudaRuntimeState {
    fn new(_batch_config: &BatchConfig, _block_config: &BlockConfig) -> Result<Self> {
        Ok(Self { _private: () })
    }
}
