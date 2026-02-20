//! Pre-allocated GPU buffer pool
//!
//! Eliminates per-op `cuMemAlloc`/`cuMemFree` overhead by caching freed GPU
//! allocations and reusing them for subsequent allocations of the same size.
//!
//! Buffers are keyed by **byte size** for maximum reuse across different
//! element types and shapes.  The pool stores raw device pointers
//! (`CUdeviceptr`) — it is type-erased and relies on [`CudaSlice::leak`] /
//! [`CudaDevice::upgrade_device_ptr`] for safe conversion.

#![allow(clippy::missing_panics_doc)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr};

/// A pool of reusable GPU memory buffers.
///
/// When a buffer is released back to the pool, it is stored by byte size.
/// The next allocation request for the same size will reuse the buffer
/// instead of calling `cuMemAlloc`.
///
/// The pool is interior-mutable (wrapped in `Mutex`) so it can be shared
/// via `Arc` across cloned `CudaContext` handles.
#[derive(Clone)]
pub struct BufferPool {
    inner: Arc<Mutex<PoolInner>>,
}

struct PoolInner {
    /// Free buffers indexed by byte size. Each entry is a LIFO stack.
    free: HashMap<usize, Vec<RawBuffer>>,
    /// Total bytes currently held (free) in the pool.
    free_bytes: usize,
    /// Reference to the CUDA device (needed to reconstruct `CudaSlice`).
    device: Arc<CudaDevice>,
    /// Statistics
    hits: u64,
    misses: u64,
}

/// Type-erased GPU buffer stored in the pool.
struct RawBuffer {
    ptr: cudarc::driver::sys::CUdeviceptr,
    byte_size: usize,
}

impl Drop for PoolInner {
    fn drop(&mut self) {
        // Free all pooled buffers by reconstructing CudaSlice<u8> and letting
        // its Drop call cuMemFree.
        for (_, stack) in self.free.drain() {
            for buf in stack {
                let _slice: CudaSlice<u8> =
                    unsafe { self.device.upgrade_device_ptr(buf.ptr, buf.byte_size) };
                // _slice drops here → cuMemFree
            }
        }
    }
}

impl BufferPool {
    /// Create a new, empty buffer pool bound to the given device.
    #[must_use]
    pub fn new(device: &Arc<CudaDevice>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(PoolInner {
                free: HashMap::new(),
                free_bytes: 0,
                device: Arc::clone(device),
                hits: 0,
                misses: 0,
            })),
        }
    }

    /// Try to acquire a buffer of exactly `byte_size` bytes, returned as a
    /// typed `CudaSlice<T>` with `numel` elements.
    ///
    /// Returns `Some(CudaSlice<T>)` if a cached buffer is available,
    /// `None` if the caller must allocate fresh GPU memory.
    ///
    /// # Safety
    /// The returned buffer contents are uninitialized. The caller must write
    /// before reading.
    #[must_use]
    pub unsafe fn acquire<T: DeviceRepr>(
        &self,
        byte_size: usize,
        numel: usize,
    ) -> Option<CudaSlice<T>> {
        let mut pool = self.inner.lock().unwrap();
        if let Some(stack) = pool.free.get_mut(&byte_size) {
            if let Some(buf) = stack.pop() {
                pool.free_bytes -= buf.byte_size;
                pool.hits += 1;
                return Some(pool.device.upgrade_device_ptr(buf.ptr, numel));
            }
        }
        pool.misses += 1;
        None
    }

    /// Return a buffer to the pool for future reuse.
    ///
    /// The `CudaSlice<T>` is leaked (its Drop is suppressed) and the raw
    /// device pointer is stored for later reuse.
    pub fn release<T>(&self, slice: CudaSlice<T>, byte_size: usize) {
        let ptr = slice.leak();
        let mut pool = self.inner.lock().unwrap();
        pool.free_bytes += byte_size;
        pool.free
            .entry(byte_size)
            .or_default()
            .push(RawBuffer { ptr, byte_size });
    }

    /// Number of cache hits (buffer reused from pool).
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.inner.lock().unwrap().hits
    }

    /// Number of cache misses (fresh allocation required).
    #[must_use]
    pub fn misses(&self) -> u64 {
        self.inner.lock().unwrap().misses
    }

    /// Total bytes currently held (free) in the pool.
    #[must_use]
    pub fn free_bytes(&self) -> usize {
        self.inner.lock().unwrap().free_bytes
    }

    /// Number of distinct buffer sizes tracked.
    #[must_use]
    pub fn num_size_classes(&self) -> usize {
        self.inner.lock().unwrap().free.len()
    }

    /// Clear all cached buffers, freeing GPU memory.
    pub fn clear(&self) {
        let mut pool = self.inner.lock().unwrap();
        let device = Arc::clone(&pool.device);
        for (_, stack) in pool.free.drain() {
            for buf in stack {
                let _slice: CudaSlice<u8> =
                    unsafe { device.upgrade_device_ptr(buf.ptr, buf.byte_size) };
            }
        }
        pool.free_bytes = 0;
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device for default pool");
        Self::new(&device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_miss_then_hit() {
        let device = CudaDevice::new(0).unwrap();
        let pool = BufferPool::new(&device);

        // First acquire: miss
        assert!(unsafe { pool.acquire::<u8>(1024, 1024) }.is_none());
        assert_eq!(pool.misses(), 1);
        assert_eq!(pool.hits(), 0);

        // Release a buffer
        let buf = device.alloc_zeros::<u8>(1024).unwrap();
        pool.release(buf, 1024);
        assert_eq!(pool.free_bytes(), 1024);

        // Second acquire: hit
        let reused = unsafe { pool.acquire::<u8>(1024, 1024) };
        assert!(reused.is_some());
        assert_eq!(pool.hits(), 1);
        assert_eq!(pool.free_bytes(), 0);
    }

    #[test]
    fn test_pool_different_sizes_no_cross_reuse() {
        let device = CudaDevice::new(0).unwrap();
        let pool = BufferPool::new(&device);

        let buf = device.alloc_zeros::<u8>(1024).unwrap();
        pool.release(buf, 1024);

        // Different size: miss
        assert!(unsafe { pool.acquire::<u8>(2048, 2048) }.is_none());
        assert_eq!(pool.misses(), 1);

        // Same size: hit
        assert!(unsafe { pool.acquire::<u8>(1024, 1024) }.is_some());
        assert_eq!(pool.hits(), 1);
    }

    #[test]
    fn test_pool_lifo_order() {
        let device = CudaDevice::new(0).unwrap();
        let pool = BufferPool::new(&device);

        let buf1 = device.alloc_zeros::<u8>(512).unwrap();
        let buf2 = device.alloc_zeros::<u8>(512).unwrap();

        let ptr1 = *cudarc::driver::DevicePtr::device_ptr(&buf1);
        let ptr2 = *cudarc::driver::DevicePtr::device_ptr(&buf2);

        pool.release(buf1, 512);
        pool.release(buf2, 512);

        // LIFO: buf2 should come out first
        let out1 = unsafe { pool.acquire::<u8>(512, 512) }.unwrap();
        let out1_ptr = *cudarc::driver::DevicePtr::device_ptr(&out1);
        assert_eq!(out1_ptr, ptr2);

        let out2 = unsafe { pool.acquire::<u8>(512, 512) }.unwrap();
        let out2_ptr = *cudarc::driver::DevicePtr::device_ptr(&out2);
        assert_eq!(out2_ptr, ptr1);
    }

    #[test]
    fn test_pool_clear() {
        let device = CudaDevice::new(0).unwrap();
        let pool = BufferPool::new(&device);

        let buf = device.alloc_zeros::<u8>(4096).unwrap();
        pool.release(buf, 4096);
        assert_eq!(pool.free_bytes(), 4096);

        pool.clear();
        assert_eq!(pool.free_bytes(), 0);
        assert!(unsafe { pool.acquire::<u8>(4096, 4096) }.is_none());
    }

    #[test]
    fn test_pool_multiple_sizes() {
        let device = CudaDevice::new(0).unwrap();
        let pool = BufferPool::new(&device);

        let buf_a = device.alloc_zeros::<u8>(256).unwrap();
        let buf_b = device.alloc_zeros::<u8>(1024).unwrap();

        pool.release(buf_a, 256);
        pool.release(buf_b, 1024);

        assert_eq!(pool.free_bytes(), 256 + 1024);
        assert_eq!(pool.num_size_classes(), 2);

        assert!(unsafe { pool.acquire::<u8>(256, 256) }.is_some());
        assert!(unsafe { pool.acquire::<u8>(1024, 1024) }.is_some());
        assert_eq!(pool.free_bytes(), 0);
    }

    #[test]
    fn test_pool_type_reinterpret() {
        let device = CudaDevice::new(0).unwrap();
        let pool = BufferPool::new(&device);

        // Release as f32 (4 elements = 16 bytes)
        let buf_f32: CudaSlice<f32> = device.alloc_zeros::<f32>(4).unwrap();
        let ptr = *cudarc::driver::DevicePtr::device_ptr(&buf_f32);
        pool.release(buf_f32, 16);

        // Acquire as u8 (16 bytes)
        let buf_u8 = unsafe { pool.acquire::<u8>(16, 16) }.unwrap();
        let ptr_u8 = *cudarc::driver::DevicePtr::device_ptr(&buf_u8);
        assert_eq!(ptr, ptr_u8);
    }
}
