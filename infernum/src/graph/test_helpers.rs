//! Shared test scaffolding for graph-builder unit tests across model crates.
//!
//! Provides the dummy value types used by the no-op `TestBackend` that each
//! model crate defines locally. Centralising these avoids repeating ~60 lines
//! of boilerplate per crate while still satisfying Rust's orphan rule (each
//! model crate defines its own `TestBackend` and implements the
//! family-specific traits for it).
//!
//! Enabled by the `test-helpers` crate feature (or automatically in `cfg(test)`
//! within this crate itself).

use crate::block_allocator::BlockConfig;
use crate::dtype::DType;
use crate::logits::Logits;
use crate::runtime_state::{BatchConfig, RuntimeStateInit};
use crate::tensor::Tensor;

/// Dummy tensor returned by all no-op ops in test backends.
#[derive(Clone)]
pub struct DummyTensor;

impl Tensor for DummyTensor {
    fn shape(&self) -> &[usize] {
        &[]
    }
    fn dtype(&self) -> DType {
        DType::F32
    }
    fn reshape(&self, _shape: &[usize]) -> Self {
        Self
    }
    fn slice_view(&self, _offset: usize, _shape: &[usize]) -> Self {
        Self
    }
}

/// Dummy logits returned by test backends.
pub struct DummyLogits;

impl Logits for DummyLogits {
    fn vocab_size(&self) -> usize {
        0
    }
    fn batch_size(&self) -> usize {
        0
    }
    fn argmax(&self, _batch_index: usize) -> crate::Result<u32> {
        Ok(0)
    }
    fn sample_top_p(
        &self,
        _batch_index: usize,
        _temperature: f32,
        _top_p: f32,
        _rng_seed: u64,
        _repetition_penalty: f32,
        _recent_tokens: &[u32],
    ) -> crate::Result<u32> {
        Ok(0)
    }
}

/// Dummy runtime state used by test backends.
pub struct DummyRuntimeState;

impl RuntimeStateInit for DummyRuntimeState {
    fn new(_batch_config: &BatchConfig, _block_config: &BlockConfig) -> crate::Result<Self> {
        Ok(Self)
    }

    fn new_placeholder() -> Self {
        Self
    }
}
