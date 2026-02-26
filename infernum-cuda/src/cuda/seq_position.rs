//! GPU-friendly sequence position for CUDA graph compatibility.
//!
//! [`SeqPosition`] holds a sequence position as both a host `usize` and a
//! single-element GPU `u32` buffer. Ops use the host value for launch
//! configuration (grid/block sizing) and pass the device pointer to kernels,
//! which read the position at execution time rather than having it baked into
//! the graph as a literal.
//!
//! This makes CUDA graph replay work without re-capture: the graph references
//! the fixed device address, and we update the value at that address between
//! launches.

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};

use infernum::Result;

/// Opaque sequence position, usable by both CPU logic and GPU kernels.
///
/// Model writers get this from [`KvCache::current_position()`] and pass it
/// to ops like `apply_rope`, `fused_attention_decode`, etc. They never need
/// to know about the GPU buffer.
pub struct SeqPosition {
    host: usize,
    device: CudaSlice<u32>,
}

impl SeqPosition {
    /// Create a new `SeqPosition` initialized to zero.
    ///
    /// # Errors
    /// Returns an error if GPU allocation fails.
    pub fn new(cuda_device: &Arc<CudaDevice>) -> Result<Self> {
        let device = cuda_device.alloc_zeros::<u32>(1)?;
        Ok(Self { host: 0, device })
    }

    /// Current position as a host-side `usize`.
    ///
    /// Used by Rust code for launch config sizing and assertions.
    #[must_use]
    pub fn host(&self) -> usize {
        self.host
    }

    /// Device pointer to the position value (single `u32` on GPU).
    ///
    /// Passed to CUDA kernels so they read the position at execution time.
    /// The device address is stable across graph replays — only the value
    /// at that address changes.
    #[must_use]
    pub fn device(&self) -> &CudaSlice<u32> {
        &self.device
    }

    /// Set the position to a new value, updating both host and device.
    ///
    /// # Errors
    /// Returns an error if the host-to-device copy fails.
    pub fn set(&mut self, value: usize, cuda_device: &Arc<CudaDevice>) -> Result<()> {
        self.host = value;
        cuda_device.htod_copy_into(vec![value as u32], &mut self.device)?;
        Ok(())
    }

    /// Advance the position by `n`, updating both host and device.
    ///
    /// # Errors
    /// Returns an error if the host-to-device copy fails.
    pub fn advance(&mut self, n: usize, cuda_device: &Arc<CudaDevice>) -> Result<()> {
        self.set(self.host + n, cuda_device)
    }

    /// Reset to zero, updating both host and device.
    ///
    /// # Errors
    /// Returns an error if the host-to-device copy fails.
    pub fn reset(&mut self, cuda_device: &Arc<CudaDevice>) -> Result<()> {
        self.set(0, cuda_device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;

    #[test]
    fn test_seq_position_new() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let pos = SeqPosition::new(ctx.device()).unwrap();
        assert_eq!(pos.host(), 0);

        let gpu_val = ctx.device().dtoh_sync_copy(pos.device()).unwrap();
        assert_eq!(gpu_val, vec![0]);
    }

    #[test]
    fn test_seq_position_set() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let mut pos = SeqPosition::new(ctx.device()).unwrap();

        pos.set(42, ctx.device()).unwrap();
        assert_eq!(pos.host(), 42);

        let gpu_val = ctx.device().dtoh_sync_copy(pos.device()).unwrap();
        assert_eq!(gpu_val, vec![42]);
    }

    #[test]
    fn test_seq_position_advance() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let mut pos = SeqPosition::new(ctx.device()).unwrap();

        pos.set(10, ctx.device()).unwrap();
        pos.advance(5, ctx.device()).unwrap();
        assert_eq!(pos.host(), 15);

        let gpu_val = ctx.device().dtoh_sync_copy(pos.device()).unwrap();
        assert_eq!(gpu_val, vec![15]);
    }

    #[test]
    fn test_seq_position_reset() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let mut pos = SeqPosition::new(ctx.device()).unwrap();

        pos.set(100, ctx.device()).unwrap();
        pos.reset(ctx.device()).unwrap();
        assert_eq!(pos.host(), 0);

        let gpu_val = ctx.device().dtoh_sync_copy(pos.device()).unwrap();
        assert_eq!(gpu_val, vec![0]);
    }
}
