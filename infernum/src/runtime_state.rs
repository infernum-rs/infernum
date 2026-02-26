//! Runtime state initialisation trait.
//!
//! Each backend provides a `RuntimeState` type that holds optimisation state
//! persisting across forward calls (e.g., CUDA graph capture/replay). The
//! engine allocates it via `RuntimeStateInit::new()` at startup and passes
//! `&mut` into every forward call without inspecting the contents.

use crate::block_allocator::BlockConfig;
use crate::Result;

/// Configuration for the batched engine / scheduler.
///
/// Re-exported from `infernum_runtime::BatchConfig` — duplicated here so
/// `RuntimeStateInit` can reference it without a circular dependency.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of sequences in the running batch at once.
    pub max_batch_size: usize,
    /// Maximum number of prompt tokens to process in a single prefill chunk.
    pub max_prefill_tokens: usize,
    /// Number of tokens per KV cache block.
    pub block_size: usize,
    /// Total number of KV cache blocks in the pool.
    pub num_blocks: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_prefill_tokens: 512,
            block_size: 16,
            num_blocks: 1024,
        }
    }
}

/// Trait for constructing backend runtime state.
///
/// The engine calls `RuntimeState::new(...)` at startup.
/// Backends that need no state implement this for `()`.
pub trait RuntimeStateInit: Send + Sized {
    /// Create a new runtime state for the given configuration.
    ///
    /// # Errors
    /// Returns an error if backend-specific initialisation fails.
    fn new(batch_config: &BatchConfig, block_config: &BlockConfig) -> Result<Self>;
}

impl RuntimeStateInit for () {
    fn new(_batch_config: &BatchConfig, _block_config: &BlockConfig) -> Result<Self> {
        Ok(())
    }
}
