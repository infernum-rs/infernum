//! CUDA tensor implementation

#![allow(clippy::cast_possible_truncation, clippy::missing_panics_doc)]

use std::sync::Arc;

use cudarc::driver::{
    CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice, ValidAsZeroBits,
};

use crate::cuda::buffer_pool::BufferPool;
use crate::cuda::CudaContext;
use crate::dtype::{DType, TensorDType};
use crate::tensor::Tensor;
use crate::Result;

/// GPU buffer wrapper that optionally returns memory to a [`BufferPool`] on drop.
///
/// When pool-backed, the `CudaSlice<T>` is leaked (via [`CudaSlice::leak`])
/// on drop and the raw device pointer is returned to the pool for reuse,
/// rather than calling `cuMemFree`.
struct PoolableBuffer<T: TensorDType> {
    /// The typed GPU allocation. Wrapped in `Option` so we can `take()` it
    /// in `drop()` to call `leak()`.
    slice: Option<CudaSlice<T>>,
    /// If set, the buffer will be returned to this pool on drop.
    pool: Option<BufferPool>,
    /// Original byte size of the allocation (for pool bookkeeping).
    byte_size: usize,
}

impl<T: TensorDType> PoolableBuffer<T> {
    /// Wrap a `CudaSlice` without pool backing (will free normally on drop).
    fn unpooled(slice: CudaSlice<T>) -> Self {
        Self {
            slice: Some(slice),
            pool: None,
            byte_size: 0,
        }
    }

    /// Wrap a `CudaSlice` with pool backing (will return to pool on drop).
    fn pooled(slice: CudaSlice<T>, pool: BufferPool, byte_size: usize) -> Self {
        Self {
            slice: Some(slice),
            pool: Some(pool),
            byte_size,
        }
    }
}

impl<T: TensorDType> Drop for PoolableBuffer<T> {
    fn drop(&mut self) {
        if let (Some(pool), Some(slice)) = (self.pool.take(), self.slice.take()) {
            // Leak the CudaSlice (suppresses cuMemFree) and return to pool.
            pool.release(slice, self.byte_size);
        }
        // If no pool, `slice` drops normally via Option's Drop → CudaSlice::Drop
    }
}

impl<T: TensorDType> std::ops::Deref for PoolableBuffer<T> {
    type Target = CudaSlice<T>;

    fn deref(&self) -> &Self::Target {
        self.slice.as_ref().expect("buffer already dropped")
    }
}

impl<T: TensorDType> std::ops::DerefMut for PoolableBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice.as_mut().expect("buffer already dropped")
    }
}

/// A tensor stored on a CUDA GPU
///
/// The tensor owns its GPU memory via `Arc`, enabling zero-copy reshape and
/// sub-slice views. The element type is encoded in the type parameter,
/// preventing accidental mixing of f32 and f16 tensors.
///
/// Multiple `CudaTensor`s may share the same underlying GPU allocation
/// (e.g. after `reshape()`). Mutable access (`cuda_slice_mut()`) uses
/// copy-on-write: it clones the buffer only when it is shared.
///
/// When the context has an active [`BufferPool`], scratch tensors created
/// via [`uninit`](CudaTensor::uninit) will return their GPU memory to the
/// pool on drop instead of calling `cuMemFree`.
pub struct CudaTensor<T: TensorDType> {
    data: Arc<PoolableBuffer<T>>,
    /// Byte offset into `data` where this tensor's elements begin.
    /// Measured in number of `T` elements, not bytes.
    offset: usize,
    shape: Vec<usize>,
    ctx: CudaContext,
}

impl<T: TensorDType + DeviceRepr> CudaTensor<T> {
    /// Create a new tensor on the GPU from host data
    ///
    /// # Errors
    /// Returns an error if GPU memory allocation or copy fails
    pub fn from_slice(ctx: &CudaContext, shape: &[usize], data: &[T]) -> Result<Self> {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "Data length {} doesn't match shape {:?} (numel={})",
            data.len(),
            shape,
            numel
        );

        let data = ctx.device().htod_sync_copy(data)?;
        Ok(Self {
            data: Arc::new(PoolableBuffer::unpooled(data)),
            offset: 0,
            shape: shape.to_vec(),
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
    pub unsafe fn uninit(ctx: &CudaContext, shape: &[usize]) -> Result<Self> {
        let numel: usize = shape.iter().product();
        let byte_size = numel * std::mem::size_of::<T>();

        let buffer = if let Some(pool) = ctx.buffer_pool() {
            if let Some(slice) = pool.acquire::<T>(byte_size, numel) {
                PoolableBuffer::pooled(slice, pool.clone(), byte_size)
            } else {
                let data = ctx.device().alloc::<T>(numel)?;
                PoolableBuffer::pooled(data, pool.clone(), byte_size)
            }
        } else {
            let data = ctx.device().alloc::<T>(numel)?;
            PoolableBuffer::unpooled(data)
        };

        Ok(Self {
            data: Arc::new(buffer),
            offset: 0,
            shape: shape.to_vec(),
            ctx: ctx.clone(),
        })
    }

    /// Create a tensor filled with zeros
    ///
    /// # Errors
    /// Returns an error if GPU memory allocation fails
    pub fn zeros(ctx: &CudaContext, shape: &[usize]) -> Result<Self>
    where
        T: ValidAsZeroBits,
    {
        let numel: usize = shape.iter().product();
        let data = ctx.device().alloc_zeros::<T>(numel)?;
        Ok(Self {
            data: Arc::new(PoolableBuffer::unpooled(data)),
            offset: 0,
            shape: shape.to_vec(),
            ctx: ctx.clone(),
        })
    }

    /// Copy tensor data back to the host
    ///
    /// # Errors
    /// Returns an error if the device-to-host copy fails
    pub fn to_vec(&self) -> Result<Vec<T>> {
        let slice: &CudaSlice<T> = &self.data;
        let view = slice.slice(self.offset..self.offset + self.numel());
        let data = self.ctx.device().dtoh_sync_copy(&view)?;
        Ok(data)
    }

    /// Get the CUDA context this tensor belongs to
    #[must_use]
    pub fn context(&self) -> &CudaContext {
        &self.ctx
    }

    /// Get a raw device pointer to the tensor data
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        let slice: &CudaSlice<T> = &self.data;
        let view = slice.slice(self.offset..self.offset + self.numel());
        *view.device_ptr() as *const T
    }

    /// Get a mutable raw device pointer to the tensor data
    ///
    /// If the underlying buffer is shared, this will copy-on-write first.
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ensure_exclusive_ownership();
        let data =
            Arc::get_mut(&mut self.data).expect("ensure_exclusive_ownership guarantees unique Arc");
        let slice = data.slice.as_mut().expect("buffer already dropped");
        *slice.device_ptr_mut() as *mut T
    }

    /// Get the underlying CUDA slice for this tensor's region.
    ///
    /// When the tensor has no offset and covers the full allocation, returns
    /// a reference to the underlying `CudaSlice` directly (zero-cost).
    /// When there is an offset (sub-slice view), returns a `CudaView` which
    /// is also zero-cost but borrows from the allocation.
    #[must_use]
    pub fn cuda_slice(&self) -> cudarc::driver::CudaView<'_, T> {
        let slice: &CudaSlice<T> = &self.data;
        slice.slice(self.offset..self.offset + self.numel())
    }

    /// Get a mutable reference to the underlying CUDA slice.
    ///
    /// If the underlying buffer is shared or is a sub-slice view, this will
    /// copy-on-write first (compact into a fresh, exclusively-owned allocation).
    /// After this call, `offset` is guaranteed to be 0 and the `Arc` is unique.
    pub fn cuda_slice_mut(&mut self) -> &mut CudaSlice<T> {
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
            offset: self.offset,
            shape: new_shape.to_vec(),
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
        let new_offset = self.offset + offset_elems;
        let backing_len: usize = self.data.slice.as_ref().map_or(0, DeviceSlice::len);
        assert!(
            new_offset + numel <= backing_len,
            "slice_view out of bounds: offset {new_offset} + numel {numel} > allocation {backing_len}",
        );

        Self {
            data: Arc::clone(&self.data),
            offset: new_offset,
            shape: shape.to_vec(),
            ctx: self.ctx.clone(),
        }
    }

    /// Reinterpret this tensor as a different element type with the same size.
    ///
    /// This is a zero-copy operation: no data is moved or converted. The GPU
    /// memory is shared via the same `Arc`. Use this when generic code needs
    /// to convert between concrete types that are known (at runtime) to be
    /// the same size — e.g. when `T::DTYPE == DType::F32` and you hold a
    /// `CudaTensor<f32>` that needs to become `CudaTensor<T>`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `T` and `U` have identical in-memory
    /// representations (same size, same alignment, same bit-pattern semantics).
    /// In practice this means they should be the same type (e.g. both `f32`),
    /// just opaque to the compiler due to generics.
    #[must_use]
    pub unsafe fn reinterpret<U: TensorDType + DeviceRepr>(self) -> CudaTensor<U> {
        assert_eq!(
            std::mem::size_of::<T>(),
            std::mem::size_of::<U>(),
            "reinterpret: size mismatch ({} bytes vs {} bytes)",
            std::mem::size_of::<T>(),
            std::mem::size_of::<U>(),
        );
        // All four fields (data, offset, shape, ctx) are transmuted.
        // `Arc<CudaSlice<T>>` and `Arc<CudaSlice<U>>` have the same layout
        // when T and U have the same size (CudaSlice is just a device pointer
        // + length, parameterised by PhantomData<T>).
        std::mem::transmute(self)
    }

    /// If the Arc is shared or we have a non-zero offset, compact into a
    /// fresh, exclusively-owned allocation containing only our elements.
    fn ensure_exclusive_ownership(&mut self) {
        let is_shared = Arc::strong_count(&self.data) > 1;
        let has_offset = self.offset != 0;
        let backing_len: usize = self.data.slice.as_ref().map_or(0, DeviceSlice::len);
        let is_partial = self.numel() < backing_len;

        if is_shared || has_offset || is_partial {
            let numel = self.numel();
            let slice: &CudaSlice<T> = &self.data;
            let view = slice.slice(self.offset..self.offset + numel);
            let mut new_data = unsafe { self.ctx.device().alloc::<T>(numel) }
                .expect("GPU allocation failed during copy-on-write");
            self.ctx
                .device()
                .dtod_copy(&view, &mut new_data)
                .expect("dtod_copy failed during copy-on-write");
            self.data = Arc::new(PoolableBuffer::unpooled(new_data));
            self.offset = 0;
        }
    }
}

impl<T: TensorDType + DeviceRepr> Clone for CudaTensor<T> {
    fn clone(&self) -> Self {
        // Clone creates a new allocation with copied data (real GPU copy)
        let numel = self.numel();
        let slice: &CudaSlice<T> = &self.data;
        let view = slice.slice(self.offset..self.offset + numel);
        let mut new_data = unsafe { self.ctx.device().alloc::<T>(numel) }
            .expect("GPU allocation failed during clone");
        self.ctx
            .device()
            .dtod_copy(&view, &mut new_data)
            .expect("dtod_copy failed during clone");
        Self {
            data: Arc::new(PoolableBuffer::unpooled(new_data)),
            offset: 0,
            shape: self.shape.clone(),
            ctx: self.ctx.clone(),
        }
    }
}

impl<T: TensorDType> Tensor for CudaTensor<T> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        T::DTYPE
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

        let result = tensor.to_vec().expect("Failed to copy back to host");
        assert_eq!(result, data);
    }

    #[test]
    fn test_tensor_zeros() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let tensor: CudaTensor<f32> =
            CudaTensor::zeros(&ctx, &[3, 4]).expect("Failed to create zeros tensor");

        assert_eq!(tensor.shape(), &[3, 4]);
        assert_eq!(tensor.numel(), 12);

        let result = tensor.to_vec().expect("Failed to copy back");
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

        let result = reshaped.to_vec().expect("Failed to copy back");
        assert_eq!(result, data);
    }

    #[test]
    fn test_tensor_strides() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let tensor: CudaTensor<f32> =
            CudaTensor::zeros(&ctx, &[2, 3, 4]).expect("Failed to create tensor");

        let strides = tensor.strides();
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_uninit_pool_enabled_by_default() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        assert!(ctx.buffer_pool().is_some());

        let tensor = unsafe { CudaTensor::<f32>::uninit(&ctx, &[4, 8]).unwrap() };
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
        let t1 = unsafe { CudaTensor::<f32>::uninit(&ctx, &[2, 4]).unwrap() };
        assert_eq!(pool.misses(), 1);
        assert_eq!(pool.hits(), 0);

        // Write data so we can verify pointer reuse
        let ptr1 = t1.as_ptr();

        // Drop t1 → returns buffer to pool
        drop(t1);
        assert_eq!(pool.free_bytes(), 2 * 4 * 4); // 32 bytes

        // Second allocation of same shape: hit
        let t2 = unsafe { CudaTensor::<f32>::uninit(&ctx, &[2, 4]).unwrap() };
        assert_eq!(pool.hits(), 1);
        assert_eq!(pool.free_bytes(), 0);

        // Should reuse the same GPU pointer
        assert_eq!(t2.as_ptr(), ptr1);
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
        let mut t = unsafe { CudaTensor::<f32>::uninit(&ctx, &[4]).unwrap() };

        // Write data through the CUDA slice
        let src: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        ctx.device()
            .htod_sync_copy_into(&src, t.cuda_slice_mut())
            .unwrap();

        let result = t.to_vec().unwrap();
        assert_eq!(result, src);
    }

    #[test]
    fn test_pool_different_sizes_separate() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let pool = ctx.buffer_pool().unwrap().clone();

        // Allocate two different sizes
        let t1 = unsafe { CudaTensor::<f32>::uninit(&ctx, &[4]).unwrap() }; // 16 bytes
        let t2 = unsafe { CudaTensor::<f32>::uninit(&ctx, &[8]).unwrap() }; // 32 bytes
        assert_eq!(pool.misses(), 2);

        drop(t1);
        drop(t2);
        assert_eq!(pool.num_size_classes(), 2);

        // Requesting the smaller size should not give the larger buffer
        let t3 = unsafe { CudaTensor::<f32>::uninit(&ctx, &[4]).unwrap() };
        assert_eq!(pool.hits(), 1);
        assert_eq!(t3.numel(), 4);
    }

    #[test]
    fn test_pool_shared_tensor_no_early_return() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let pool = ctx.buffer_pool().unwrap().clone();

        let t1 = unsafe { CudaTensor::<f32>::uninit(&ctx, &[8]).unwrap() };
        let t2 = t1.reshape(&[2, 4]); // shares the same Arc

        // Drop one — Arc refcount > 1, buffer should NOT return to pool
        drop(t1);
        assert_eq!(pool.free_bytes(), 0);

        // Drop the last reference — buffer returns to pool
        drop(t2);
        assert_eq!(pool.free_bytes(), 8 * 4);
    }
}
