//! CUDA context management

use cudarc::cublas::CudaBlas;
use cudarc::cublaslt::CudaBlasLT;
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::{Arc, Mutex};

use super::buffer_pool::BufferPool;
use crate::Result;

/// Manages CUDA device and associated resources (cuBLAS handle, etc.)
#[derive(Clone)]
pub struct CudaContext {
    device: Arc<CudaDevice>,
    blas: Arc<CudaBlas>,
    blas_lt: Arc<CudaBlasLT>,
    /// Cached cuBLASLt workspace buffer (lazily allocated)
    fp8_workspace: Arc<Mutex<Option<CudaSlice<u8>>>>,
    /// Cached compute capability (major, minor)
    compute_capability: (i32, i32),
    /// Optional buffer pool for reusing GPU allocations
    buffer_pool: Option<BufferPool>,
}

impl CudaContext {
    /// Create a new CUDA context for the specified device ordinal
    ///
    /// # Errors
    /// Returns an error if CUDA device initialization fails
    pub fn new(ordinal: usize) -> Result<Self> {
        // Use a non-default stream so that CUDA graph capture is possible.
        let device = CudaDevice::new_with_stream(ordinal)?;
        let blas = CudaBlas::new(device.clone())?;
        let blas_lt = CudaBlasLT::new(device.clone())?;

        let major = device
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )
            .unwrap_or(0);
        let minor = device
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )
            .unwrap_or(0);

        Ok(Self {
            device,
            blas: Arc::new(blas),
            blas_lt: Arc::new(blas_lt),
            fp8_workspace: Arc::new(Mutex::new(None)),
            compute_capability: (major, minor),
            buffer_pool: None,
        })
    }

    /// Get a reference to the underlying CUDA device
    #[must_use]
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get a reference to the cuBLAS handle
    #[must_use]
    pub fn blas(&self) -> &Arc<CudaBlas> {
        &self.blas
    }

    /// Get a reference to the cuBLASLt handle
    #[must_use]
    pub fn blas_lt(&self) -> &Arc<CudaBlasLT> {
        &self.blas_lt
    }

    /// Compute capability as (major, minor)
    #[must_use]
    pub fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    /// Whether the GPU supports FP8 tensor cores (Ada `sm_89`+ or Hopper `sm_90`+)
    #[must_use]
    pub fn supports_fp8_tensor_cores(&self) -> bool {
        let (major, minor) = self.compute_capability;
        major > 8 || (major == 8 && minor >= 9)
    }

    /// Get or allocate the cuBLASLt workspace buffer for FP8 operations.
    ///
    /// The workspace is allocated once and reused across all FP8 matmul calls.
    /// Size: 32 MiB for Hopper (`sm_90`+), 4 MiB for Ada (`sm_89`).
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    ///
    /// # Errors
    /// Returns an error if GPU allocation fails.
    pub fn fp8_workspace(&self) -> Result<std::sync::MutexGuard<'_, Option<CudaSlice<u8>>>> {
        let mut guard = self.fp8_workspace.lock().unwrap();
        if guard.is_none() {
            let (major, _) = self.compute_capability;
            let ws_size: usize = if major >= 9 { 33_554_432 } else { 4_194_304 };
            *guard = Some(unsafe { self.device.alloc::<u8>(ws_size)? });
        }
        Ok(guard)
    }

    /// Enable the GPU buffer pool for this context.
    ///
    /// Once enabled, `CudaTensor::uninit()` will attempt to reuse previously
    /// freed GPU allocations instead of calling `cuMemAlloc` each time.
    /// All clones of this context share the same pool.
    pub fn enable_buffer_pool(&mut self) {
        if self.buffer_pool.is_none() {
            self.buffer_pool = Some(BufferPool::new(&self.device));
        }
    }

    /// Get a reference to the buffer pool, if enabled.
    #[must_use]
    pub fn buffer_pool(&self) -> Option<&BufferPool> {
        self.buffer_pool.as_ref()
    }

    /// Allocate a pool-backed GPU buffer of `numel` elements of type `T`.
    ///
    /// When the buffer pool is enabled, tries to reuse a previously freed
    /// allocation of the same byte size.  Otherwise falls back to
    /// `cuMemAlloc`.  The returned [`PooledSlice`] returns the buffer to
    /// the pool on drop.
    ///
    /// # Safety
    /// The buffer contents are uninitialized.  The caller must write before
    /// reading.
    ///
    /// # Errors
    /// Returns an error if GPU memory allocation fails.
    pub unsafe fn pool_alloc<T: cudarc::driver::DeviceRepr>(
        &self,
        numel: usize,
    ) -> Result<super::buffer_pool::PooledSlice<T>> {
        use super::buffer_pool::PooledSlice;

        let byte_size = numel * std::mem::size_of::<T>();

        if let Some(pool) = self.buffer_pool() {
            if let Some(slice) = pool.acquire::<T>(byte_size, numel) {
                return Ok(PooledSlice::pooled(slice, pool.clone(), byte_size));
            }
            let slice = self.device.alloc::<T>(numel)?;
            Ok(PooledSlice::pooled(slice, pool.clone(), byte_size))
        } else {
            let slice = self.device.alloc::<T>(numel)?;
            Ok(PooledSlice::unpooled(slice))
        }
    }

    /// Synchronize the CUDA device (wait for all operations to complete)
    ///
    /// # Errors
    /// Returns an error if synchronization fails
    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        // Verify device and blas handles are accessible
        let _ = ctx.device();
        let _ = ctx.blas();
    }

    #[test]
    fn test_context_clone() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let ctx2 = ctx.clone();

        // Both should reference the same device
        assert!(std::sync::Arc::ptr_eq(ctx.device(), ctx2.device()));
        assert!(std::sync::Arc::ptr_eq(ctx.blas(), ctx2.blas()));
        assert!(std::sync::Arc::ptr_eq(ctx.blas_lt(), ctx2.blas_lt()));
        assert!(std::sync::Arc::ptr_eq(
            &ctx.fp8_workspace,
            &ctx2.fp8_workspace,
        ));
    }

    #[test]
    fn test_buffer_pool_shared_across_clones() {
        let mut ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        ctx.enable_buffer_pool();

        let ctx2 = ctx.clone();

        // Both should see the same pool
        assert!(ctx.buffer_pool().is_some());
        assert!(ctx2.buffer_pool().is_some());

        // Release a buffer through one context, acquire through the other
        let buf = ctx.device().alloc_zeros::<u8>(256).unwrap();
        ctx.buffer_pool().unwrap().release(buf, 256);
        assert!(unsafe { ctx2.buffer_pool().unwrap().acquire::<u8>(256, 256) }.is_some());
    }

    #[test]
    fn test_compute_capability() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let (major, _minor) = ctx.compute_capability();
        assert!(major > 0, "GPU should have a valid compute capability");
    }

    #[test]
    fn test_context_synchronize() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        ctx.synchronize().expect("Synchronize should succeed");
    }
}
