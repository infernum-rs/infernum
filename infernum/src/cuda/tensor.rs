//! CUDA tensor implementation

#![allow(clippy::cast_possible_truncation, clippy::missing_panics_doc)]

use std::sync::Arc;

use cudarc::driver::{
    CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice, ValidAsZeroBits,
};

use crate::cuda::CudaContext;
use crate::dtype::{DType, TensorDType};
use crate::tensor::Tensor;
use crate::Result;

/// A tensor stored on a CUDA GPU
///
/// The tensor owns its GPU memory via `Arc`, enabling zero-copy reshape and
/// sub-slice views. The element type is encoded in the type parameter,
/// preventing accidental mixing of f32 and f16 tensors.
///
/// Multiple `CudaTensor`s may share the same underlying GPU allocation
/// (e.g. after `reshape()`). Mutable access (`cuda_slice_mut()`) uses
/// copy-on-write: it clones the buffer only when it is shared.
pub struct CudaTensor<T: TensorDType> {
    data: Arc<CudaSlice<T>>,
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
            data: Arc::new(data),
            offset: 0,
            shape: shape.to_vec(),
            ctx: ctx.clone(),
        })
    }

    /// Create an uninitialized tensor on the GPU
    ///
    /// # Safety
    /// The tensor contents are uninitialized. Reading before writing is undefined behavior.
    ///
    /// # Errors
    /// Returns an error if GPU memory allocation fails
    pub unsafe fn uninit(ctx: &CudaContext, shape: &[usize]) -> Result<Self> {
        let numel: usize = shape.iter().product();
        let data = ctx.device().alloc::<T>(numel)?;
        Ok(Self {
            data: Arc::new(data),
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
            data: Arc::new(data),
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
        let view = self.data.slice(self.offset..self.offset + self.numel());
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
        let view = self.data.slice(self.offset..self.offset + self.numel());
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
        *data.device_ptr_mut() as *mut T
    }

    /// Get the underlying CUDA slice for this tensor's region.
    ///
    /// When the tensor has no offset and covers the full allocation, returns
    /// a reference to the underlying `CudaSlice` directly (zero-cost).
    /// When there is an offset (sub-slice view), returns a `CudaView` which
    /// is also zero-cost but borrows from the allocation.
    #[must_use]
    pub fn cuda_slice(&self) -> cudarc::driver::CudaView<'_, T> {
        self.data.slice(self.offset..self.offset + self.numel())
    }

    /// Get a mutable reference to the underlying CUDA slice.
    ///
    /// If the underlying buffer is shared or is a sub-slice view, this will
    /// copy-on-write first (compact into a fresh, exclusively-owned allocation).
    /// After this call, `offset` is guaranteed to be 0 and the `Arc` is unique.
    pub fn cuda_slice_mut(&mut self) -> &mut CudaSlice<T> {
        self.ensure_exclusive_ownership();
        Arc::get_mut(&mut self.data).expect("ensure_exclusive_ownership guarantees unique Arc")
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
        assert!(
            new_offset + numel <= self.data.len(),
            "slice_view out of bounds: offset {} + numel {} > allocation {}",
            new_offset,
            numel,
            self.data.len(),
        );

        Self {
            data: Arc::clone(&self.data),
            offset: new_offset,
            shape: shape.to_vec(),
            ctx: self.ctx.clone(),
        }
    }

    /// If the Arc is shared or we have a non-zero offset, compact into a
    /// fresh, exclusively-owned allocation containing only our elements.
    fn ensure_exclusive_ownership(&mut self) {
        let is_shared = Arc::strong_count(&self.data) > 1;
        let has_offset = self.offset != 0;
        let is_partial = self.numel() < self.data.len();

        if is_shared || has_offset || is_partial {
            let numel = self.numel();
            let view = self.data.slice(self.offset..self.offset + numel);
            let mut new_data = unsafe { self.ctx.device().alloc::<T>(numel) }
                .expect("GPU allocation failed during copy-on-write");
            self.ctx
                .device()
                .dtod_copy(&view, &mut new_data)
                .expect("dtod_copy failed during copy-on-write");
            self.data = Arc::new(new_data);
            self.offset = 0;
        }
    }
}

impl<T: TensorDType + DeviceRepr> Clone for CudaTensor<T> {
    fn clone(&self) -> Self {
        // Clone creates a new allocation with copied data (real GPU copy)
        let numel = self.numel();
        let view = self.data.slice(self.offset..self.offset + numel);
        let mut new_data = unsafe { self.ctx.device().alloc::<T>(numel) }
            .expect("GPU allocation failed during clone");
        self.ctx
            .device()
            .dtod_copy(&view, &mut new_data)
            .expect("dtod_copy failed during clone");
        Self {
            data: Arc::new(new_data),
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
}
