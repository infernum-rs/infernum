//! CUDA tensor implementation

#![allow(clippy::cast_possible_truncation, clippy::missing_panics_doc)]

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr, ValidAsZeroBits};

use crate::cuda::CudaContext;
use crate::dtype::{DType, TensorDType};
use crate::tensor::Tensor;
use crate::Result;

/// A tensor stored on a CUDA GPU
///
/// The tensor owns its GPU memory and is parameterized by the element type.
/// This ensures type safety: you cannot accidentally mix f32 and f16 tensors.
pub struct CudaTensor<T: TensorDType> {
    data: CudaSlice<T>,
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
            data,
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
            data,
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
            data,
            shape: shape.to_vec(),
            ctx: ctx.clone(),
        })
    }

    /// Copy tensor data back to the host
    ///
    /// # Errors
    /// Returns an error if the device-to-host copy fails
    pub fn to_vec(&self) -> Result<Vec<T>> {
        let data = self.ctx.device().dtoh_sync_copy(&self.data)?;
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
        *self.data.device_ptr() as *const T
    }

    /// Get a mutable raw device pointer to the tensor data
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        *self.data.device_ptr_mut() as *mut T
    }

    /// Get the underlying CUDA slice
    #[must_use]
    pub fn cuda_slice(&self) -> &CudaSlice<T> {
        &self.data
    }

    /// Get a mutable reference to the underlying CUDA slice
    #[must_use]
    pub fn cuda_slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.data
    }

    /// Reshape the tensor (returns a new tensor with the same data but different shape)
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
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            ctx: self.ctx.clone(),
        }
    }
}

impl<T: TensorDType + DeviceRepr> Clone for CudaTensor<T> {
    fn clone(&self) -> Self {
        // Clone creates a new allocation with copied data
        let data = self.data.clone();
        Self {
            data,
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
