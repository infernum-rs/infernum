//! CUDA tensor implementation

#![allow(clippy::cast_possible_truncation, clippy::missing_panics_doc)]

use std::sync::Arc;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice};

use crate::cuda::buffer_pool::BufferPool;
use crate::cuda::CudaContext;
use crate::dtype::{DType, TensorDType};
use crate::tensor::Tensor;
use crate::Result;

/// GPU buffer wrapper that optionally returns memory to a [`BufferPool`] on drop.
///
/// Stores raw bytes (`CudaSlice<u8>`) — the tensor's `DType` determines
/// interpretation. When pool-backed, the buffer is leaked on drop and the
/// raw device pointer is returned to the pool for reuse.
struct PoolableBuffer {
    /// The raw GPU allocation (bytes). Wrapped in `Option` so we can `take()`
    /// it in `drop()` to call `leak()`.
    slice: Option<CudaSlice<u8>>,
    /// If set, the buffer will be returned to this pool on drop.
    pool: Option<BufferPool>,
    /// Original byte size of the allocation (for pool bookkeeping).
    byte_size: usize,
}

impl PoolableBuffer {
    /// Wrap a `CudaSlice<u8>` without pool backing (will free normally on drop).
    fn unpooled(slice: CudaSlice<u8>) -> Self {
        Self {
            slice: Some(slice),
            pool: None,
            byte_size: 0,
        }
    }

    /// Wrap a `CudaSlice<u8>` with pool backing (will return to pool on drop).
    fn pooled(slice: CudaSlice<u8>, pool: BufferPool, byte_size: usize) -> Self {
        Self {
            slice: Some(slice),
            pool: Some(pool),
            byte_size,
        }
    }
}

impl Drop for PoolableBuffer {
    fn drop(&mut self) {
        if let (Some(pool), Some(slice)) = (self.pool.take(), self.slice.take()) {
            // Leak the CudaSlice (suppresses cuMemFree) and return to pool.
            pool.release(slice, self.byte_size);
        }
        // If no pool, `slice` drops normally via Option's Drop → CudaSlice::Drop
    }
}

impl std::ops::Deref for PoolableBuffer {
    type Target = CudaSlice<u8>;

    fn deref(&self) -> &Self::Target {
        self.slice.as_ref().expect("buffer already dropped")
    }
}

impl std::ops::DerefMut for PoolableBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice.as_mut().expect("buffer already dropped")
    }
}

/// A tensor stored on a CUDA GPU.
///
/// The tensor owns its GPU memory via `Arc`, enabling zero-copy reshape and
/// sub-slice views. The data type is carried as a runtime [`DType`] field
/// rather than a type parameter — this avoids monomorphization overhead and
/// simplifies generic code.
///
/// Multiple `CudaTensor`s may share the same underlying GPU allocation
/// (e.g. after `reshape()`). Mutable access (`cuda_slice_mut()`) uses
/// copy-on-write: it clones the buffer only when it is shared.
///
/// When the context has an active [`BufferPool`], scratch tensors created
/// via [`uninit`](CudaTensor::uninit) will return their GPU memory to the
/// pool on drop instead of calling `cuMemFree`.
pub struct CudaTensor {
    data: Arc<PoolableBuffer>,
    /// Byte offset into `data` where this tensor's elements begin.
    offset_bytes: usize,
    shape: Vec<usize>,
    dtype: DType,
    ctx: CudaContext,
}

impl CudaTensor {
    /// Create a new tensor on the GPU from typed host data.
    ///
    /// This is generic over `T` only because the host→device copy needs the
    /// concrete element type. The resulting `CudaTensor` is untyped.
    ///
    /// # Errors
    /// Returns an error if GPU memory allocation or copy fails
    pub fn from_slice<T: TensorDType + DeviceRepr>(
        ctx: &CudaContext,
        shape: &[usize],
        data: &[T],
    ) -> Result<Self> {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "Data length {} doesn't match shape {:?} (numel={})",
            data.len(),
            shape,
            numel
        );

        let typed_slice = ctx.device().htod_sync_copy(data)?;
        let byte_size = numel * std::mem::size_of::<T>();
        // Reinterpret the typed CudaSlice<T> as CudaSlice<u8>
        let raw_ptr = typed_slice.leak();
        let raw_slice = unsafe { ctx.device().upgrade_device_ptr(raw_ptr, byte_size) };

        Ok(Self {
            data: Arc::new(PoolableBuffer::unpooled(raw_slice)),
            offset_bytes: 0,
            shape: shape.to_vec(),
            dtype: T::DTYPE,
            ctx: ctx.clone(),
        })
    }

    /// Create an uninitialized tensor on the GPU.
    ///
    /// When the context has an active buffer pool, this will try to reuse a
    /// previously freed allocation of the same byte size. Otherwise it falls
    /// back to `cuMemAlloc`.
    ///
    /// # Safety
    /// The tensor contents are uninitialized. Reading before writing is undefined behavior.
    ///
    /// # Errors
    /// Returns an error if GPU memory allocation fails
    pub unsafe fn uninit(ctx: &CudaContext, shape: &[usize], dtype: DType) -> Result<Self> {
        let numel: usize = shape.iter().product();
        let elem_size = dtype.size_in_bytes();
        let byte_size = numel * elem_size;

        let buffer = if let Some(pool) = ctx.buffer_pool() {
            if let Some(slice) = pool.acquire::<u8>(byte_size, byte_size) {
                PoolableBuffer::pooled(slice, pool.clone(), byte_size)
            } else {
                let data = ctx.device().alloc::<u8>(byte_size)?;
                PoolableBuffer::pooled(data, pool.clone(), byte_size)
            }
        } else {
            let data = ctx.device().alloc::<u8>(byte_size)?;
            PoolableBuffer::unpooled(data)
        };

        Ok(Self {
            data: Arc::new(buffer),
            offset_bytes: 0,
            shape: shape.to_vec(),
            dtype,
            ctx: ctx.clone(),
        })
    }

    /// Create a tensor filled with zeros.
    ///
    /// When the context has an active buffer pool, this reuses a pooled
    /// allocation and zeros it via `cuMemsetD8`, avoiding `cuMemAlloc`.
    ///
    /// # Errors
    /// Returns an error if GPU memory allocation fails
    pub fn zeros(ctx: &CudaContext, shape: &[usize], dtype: DType) -> Result<Self> {
        if ctx.buffer_pool().is_some() {
            let mut t = unsafe { Self::uninit(ctx, shape, dtype)? };
            let byte_count = t.size_in_bytes();
            let ptr = *t.cuda_slice_mut().device_ptr_mut();
            let result = unsafe { cudarc::driver::sys::lib().cuMemsetD8_v2(ptr, 0, byte_count) };
            assert_eq!(
                result,
                cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                "cuMemsetD8 failed"
            );
            Ok(t)
        } else {
            let numel: usize = shape.iter().product();
            let byte_size = numel * dtype.size_in_bytes();
            let data = ctx.device().alloc_zeros::<u8>(byte_size)?;
            Ok(Self {
                data: Arc::new(PoolableBuffer::unpooled(data)),
                offset_bytes: 0,
                shape: shape.to_vec(),
                dtype,
                ctx: ctx.clone(),
            })
        }
    }

    /// Copy tensor data back to the host as typed elements.
    ///
    /// # Panics
    /// Panics if `T::DTYPE` doesn't match this tensor's dtype.
    ///
    /// # Errors
    /// Returns an error if the device-to-host copy fails
    pub fn to_vec<T: TensorDType + DeviceRepr>(&self) -> Result<Vec<T>> {
        assert_eq!(
            self.dtype,
            T::DTYPE,
            "to_vec dtype mismatch: tensor is {:?}, requested {:?}",
            self.dtype,
            T::DTYPE,
        );
        let byte_size = self.size_in_bytes();
        let slice: &CudaSlice<u8> = &self.data;
        let view = slice.slice(self.offset_bytes..self.offset_bytes + byte_size);
        // Reinterpret u8 view as typed for dtoh copy
        let raw_ptr = *view.device_ptr();
        let typed_slice: cudarc::driver::CudaSlice<T> =
            unsafe { self.ctx.device().upgrade_device_ptr(raw_ptr, self.numel()) };
        let data = self.ctx.device().dtoh_sync_copy(&typed_slice)?;
        // Leak the upgraded slice — we don't own this memory
        std::mem::forget(typed_slice);
        Ok(data)
    }

    /// Copy tensor data back to the host as raw bytes, regardless of dtype.
    ///
    /// Each element occupies `dtype().size_in_bytes()` bytes.
    /// Useful for dtype-agnostic host-side operations (concat, transpose).
    ///
    /// # Errors
    /// Returns an error if the device-to-host copy fails
    pub fn to_raw_bytes(&self) -> Result<Vec<u8>> {
        let byte_size = self.size_in_bytes();
        let slice: &CudaSlice<u8> = &self.data;
        let view = slice.slice(self.offset_bytes..self.offset_bytes + byte_size);
        let data = self.ctx.device().dtoh_sync_copy(&view)?;
        Ok(data)
    }

    /// Create a tensor on the GPU from raw bytes with a specified dtype and shape.
    ///
    /// # Panics
    /// Panics if `data.len()` doesn't match `shape.product() * dtype.size_in_bytes()`.
    ///
    /// # Errors
    /// Returns an error if GPU memory allocation or copy fails
    pub fn from_raw_bytes(
        ctx: &CudaContext,
        shape: &[usize],
        dtype: DType,
        data: &[u8],
    ) -> Result<Self> {
        let numel: usize = shape.iter().product();
        let expected_bytes = numel * dtype.size_in_bytes();
        assert_eq!(
            data.len(),
            expected_bytes,
            "from_raw_bytes: data length {} doesn't match shape {:?} * dtype {:?} (expected {})",
            data.len(),
            shape,
            dtype,
            expected_bytes
        );

        let raw_slice = ctx.device().htod_sync_copy(data)?;
        Ok(Self {
            data: Arc::new(PoolableBuffer::unpooled(raw_slice)),
            offset_bytes: 0,
            shape: shape.to_vec(),
            dtype,
            ctx: ctx.clone(),
        })
    }

    /// Get the CUDA context this tensor belongs to
    #[must_use]
    pub fn context(&self) -> &CudaContext {
        &self.ctx
    }

    /// Get the size of the tensor data in bytes.
    #[must_use]
    pub fn size_in_bytes(&self) -> usize {
        self.numel() * self.dtype.size_in_bytes()
    }

    /// Get a raw device pointer to the tensor data (as `u64`).
    ///
    /// This is the untyped equivalent of the old `as_ptr()`. Use for
    /// passing to CUDA kernel launches that accept `void*`.
    #[must_use]
    pub fn device_ptr(&self) -> u64 {
        let slice: &CudaSlice<u8> = &self.data;
        let view = slice.slice(self.offset_bytes..self.offset_bytes + self.size_in_bytes());
        *view.device_ptr()
    }

    /// Get a mutable raw device pointer to the tensor data (as `u64`).
    ///
    /// If the underlying buffer is shared, this will copy-on-write first.
    #[must_use]
    pub fn device_ptr_mut(&mut self) -> u64 {
        self.ensure_exclusive_ownership();
        let data =
            Arc::get_mut(&mut self.data).expect("ensure_exclusive_ownership guarantees unique Arc");
        let slice = data.slice.as_mut().expect("buffer already dropped");
        *slice.device_ptr_mut()
    }

    /// Get the underlying CUDA slice as raw bytes for this tensor's region.
    #[must_use]
    pub fn cuda_slice(&self) -> cudarc::driver::CudaView<'_, u8> {
        let slice: &CudaSlice<u8> = &self.data;
        slice.slice(self.offset_bytes..self.offset_bytes + self.size_in_bytes())
    }

    /// Get a mutable reference to the underlying CUDA slice (raw bytes).
    ///
    /// If the underlying buffer is shared or is a sub-slice view, this will
    /// copy-on-write first (compact into a fresh, exclusively-owned allocation).
    /// After this call, `offset_bytes` is guaranteed to be 0 and the `Arc` is unique.
    pub fn cuda_slice_mut(&mut self) -> &mut CudaSlice<u8> {
        self.ensure_exclusive_ownership();
        let data =
            Arc::get_mut(&mut self.data).expect("ensure_exclusive_ownership guarantees unique Arc");
        data.slice.as_mut().expect("buffer already dropped")
    }

    /// Reshape the tensor to a new shape with the same number of elements.
    ///
    /// This is a zero-copy operation — the returned tensor shares the same
    /// GPU memory via `Arc`.
    ///
    /// # Panics
    /// Panics if the new shape has a different number of elements
    #[must_use]
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            new_numel,
            "Cannot reshape tensor of {} elements to shape {:?} ({} elements)",
            self.numel(),
            new_shape,
            new_numel
        );

        Self {
            data: Arc::clone(&self.data),
            offset_bytes: self.offset_bytes,
            shape: new_shape.to_vec(),
            dtype: self.dtype,
            ctx: self.ctx.clone(),
        }
    }

    /// Create a zero-copy sub-slice view of this tensor.
    ///
    /// The returned tensor shares the same GPU allocation and starts at
    /// element `offset_elems` with the given `shape`.
    ///
    /// # Panics
    /// Panics if `offset_elems + numel(shape)` exceeds the backing allocation.
    #[must_use]
    pub fn slice_view(&self, offset_elems: usize, shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        let elem_size = self.dtype.size_in_bytes();
        let new_offset_bytes = self.offset_bytes + offset_elems * elem_size;
        let backing_bytes: usize = self.data.slice.as_ref().map_or(0, DeviceSlice::len);
        let needed_bytes = new_offset_bytes + numel * elem_size;
        assert!(
            needed_bytes <= backing_bytes,
            "slice_view out of bounds: offset_bytes {new_offset_bytes} + data_bytes {} > allocation {backing_bytes}",
            numel * elem_size,
        );

        Self {
            data: Arc::clone(&self.data),
            offset_bytes: new_offset_bytes,
            shape: shape.to_vec(),
            dtype: self.dtype,
            ctx: self.ctx.clone(),
        }
    }

    /// If the Arc is shared or we have a non-zero offset, compact into a
    /// fresh, exclusively-owned allocation containing only our bytes.
    fn ensure_exclusive_ownership(&mut self) {
        let is_shared = Arc::strong_count(&self.data) > 1;
        let has_offset = self.offset_bytes != 0;
        let backing_bytes: usize = self.data.slice.as_ref().map_or(0, DeviceSlice::len);
        let my_bytes = self.size_in_bytes();
        let is_partial = my_bytes < backing_bytes;

        if is_shared || has_offset || is_partial {
            let slice: &CudaSlice<u8> = &self.data;
            let view = slice.slice(self.offset_bytes..self.offset_bytes + my_bytes);
            let mut new_data = unsafe { self.ctx.device().alloc::<u8>(my_bytes) }
                .expect("GPU allocation failed during copy-on-write");
            self.ctx
                .device()
                .dtod_copy(&view, &mut new_data)
                .expect("dtod_copy failed during copy-on-write");
            self.data = Arc::new(PoolableBuffer::unpooled(new_data));
            self.offset_bytes = 0;
        }
    }
}

impl Clone for CudaTensor {
    fn clone(&self) -> Self {
        // Clone creates a new allocation with copied data (real GPU copy)
        let my_bytes = self.size_in_bytes();
        let slice: &CudaSlice<u8> = &self.data;
        let view = slice.slice(self.offset_bytes..self.offset_bytes + my_bytes);
        let mut new_data = unsafe { self.ctx.device().alloc::<u8>(my_bytes) }
            .expect("GPU allocation failed during clone");
        self.ctx
            .device()
            .dtod_copy(&view, &mut new_data)
            .expect("dtod_copy failed during clone");
        Self {
            data: Arc::new(PoolableBuffer::unpooled(new_data)),
            offset_bytes: 0,
            shape: self.shape.clone(),
            dtype: self.dtype,
            ctx: self.ctx.clone(),
        }
    }
}

impl Tensor for CudaTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation_and_roundtrip() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2, 3];

        let tensor = CudaTensor::from_slice(&ctx, &shape, &data).expect("Failed to create tensor");

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.ndim(), 2);

        let result: Vec<f32> = tensor.to_vec::<f32>().expect("Failed to copy back to host");
        assert_eq!(result, data);
    }

    #[test]
    fn test_tensor_zeros() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let tensor =
            CudaTensor::zeros(&ctx, &[3, 4], DType::F32).expect("Failed to create zeros tensor");

        assert_eq!(tensor.shape(), &[3, 4]);
        assert_eq!(tensor.numel(), 12);

        let result: Vec<f32> = tensor.to_vec::<f32>().expect("Failed to copy back");
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor_reshape() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let tensor =
            CudaTensor::from_slice(&ctx, &[2, 3, 4], &data).expect("Failed to create tensor");

        let reshaped = tensor.reshape(&[6, 4]);
        assert_eq!(reshaped.shape(), &[6, 4]);
        assert_eq!(reshaped.numel(), 24);

        let result: Vec<f32> = reshaped.to_vec::<f32>().expect("Failed to copy back");
        assert_eq!(result, data);
    }

    #[test]
    fn test_tensor_strides() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let tensor =
            CudaTensor::zeros(&ctx, &[2, 3, 4], DType::F32).expect("Failed to create tensor");

        let strides = tensor.strides();
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_uninit_pool_enabled_by_default() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        assert!(ctx.buffer_pool().is_some());

        let tensor = unsafe { CudaTensor::uninit(&ctx, &[4, 8], DType::F32).unwrap() };
        assert_eq!(tensor.shape(), &[4, 8]);
        assert_eq!(tensor.numel(), 32);
    }

    #[test]
    fn test_uninit_with_pool_miss_then_hit() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let pool = ctx.buffer_pool().unwrap();
        assert_eq!(pool.hits(), 0);
        assert_eq!(pool.misses(), 0);

        // First allocation: miss (no cached buffers)
        let t1 = unsafe { CudaTensor::uninit(&ctx, &[2, 4], DType::F32).unwrap() };
        assert_eq!(pool.misses(), 1);
        assert_eq!(pool.hits(), 0);

        // Write data so we can verify pointer reuse
        let ptr1 = t1.device_ptr();

        // Drop t1 → returns buffer to pool
        drop(t1);
        assert_eq!(pool.free_bytes(), 2 * 4 * 4); // 32 bytes

        // Second allocation of same shape: hit
        let t2 = unsafe { CudaTensor::uninit(&ctx, &[2, 4], DType::F32).unwrap() };
        assert_eq!(pool.hits(), 1);
        assert_eq!(pool.free_bytes(), 0);

        // Should reuse the same GPU pointer
        assert_eq!(t2.device_ptr(), ptr1);
    }

    #[test]
    fn test_pool_roundtrip_data_integrity() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // Allocate, write, drop (returns to pool)
        {
            let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let _t = CudaTensor::from_slice(&ctx, &[4], &data).unwrap();
            // _t is unpooled (from_slice), so drop won't return to pool
        }

        // Allocate via uninit (pooled), write, read back
        let mut t = unsafe { CudaTensor::uninit(&ctx, &[4], DType::F32).unwrap() };

        // Write data through the CUDA slice — need to use typed copy
        let src: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let src_bytes: &[u8] = bytemuck::cast_slice(&src);
        ctx.device()
            .htod_sync_copy_into(src_bytes, t.cuda_slice_mut())
            .unwrap();

        let result: Vec<f32> = t.to_vec::<f32>().unwrap();
        assert_eq!(result, src);
    }

    #[test]
    fn test_pool_different_sizes_separate() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let pool = ctx.buffer_pool().unwrap().clone();

        // Allocate two different sizes
        let t1 = unsafe { CudaTensor::uninit(&ctx, &[4], DType::F32).unwrap() }; // 16 bytes
        let t2 = unsafe { CudaTensor::uninit(&ctx, &[8], DType::F32).unwrap() }; // 32 bytes
        assert_eq!(pool.misses(), 2);

        drop(t1);
        drop(t2);
        assert_eq!(pool.num_size_classes(), 2);

        // Requesting the smaller size should not give the larger buffer
        let t3 = unsafe { CudaTensor::uninit(&ctx, &[4], DType::F32).unwrap() };
        assert_eq!(pool.hits(), 1);
        assert_eq!(t3.numel(), 4);
    }

    #[test]
    fn test_pool_shared_tensor_no_early_return() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let pool = ctx.buffer_pool().unwrap().clone();

        let t1 = unsafe { CudaTensor::uninit(&ctx, &[8], DType::F32).unwrap() };
        let t2 = t1.reshape(&[2, 4]); // shares the same Arc

        // Drop one — Arc refcount > 1, buffer should NOT return to pool
        drop(t1);
        assert_eq!(pool.free_bytes(), 0);

        // Drop the last reference — buffer returns to pool
        drop(t2);
        assert_eq!(pool.free_bytes(), 8 * 4);
    }

    #[test]
    fn test_zeros_pool_aware() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let pool = ctx.buffer_pool().unwrap().clone();
        pool.reset_stats();

        // First zeros: pool miss (allocates fresh, then zeros)
        let t1 = CudaTensor::zeros(&ctx, &[2, 4], DType::F32).unwrap();
        assert_eq!(pool.misses(), 1);
        let data: Vec<f32> = t1.to_vec::<f32>().unwrap();
        assert!(data.iter().all(|&x| x == 0.0));

        let ptr1 = t1.device_ptr();
        drop(t1);

        // Second zeros of same shape: pool hit + correctly zeroed
        let t2 = CudaTensor::zeros(&ctx, &[2, 4], DType::F32).unwrap();
        assert_eq!(pool.hits(), 1);
        assert_eq!(t2.device_ptr(), ptr1);
        let data: Vec<f32> = t2.to_vec::<f32>().unwrap();
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_pool_report() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let pool = ctx.buffer_pool().unwrap().clone();
        pool.reset_stats();

        let _ = unsafe { CudaTensor::uninit(&ctx, &[4], DType::F32).unwrap() };
        let report = pool.report();
        assert!(report.contains("0 hits"));
        assert!(report.contains("1 misses"));
    }
}
