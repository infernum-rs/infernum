//! Weight loader trait and utilities

#![allow(clippy::missing_errors_doc)]

#[cfg(feature = "cuda")]
use crate::cuda::{CudaContext, CudaTensor, QuantizedTensor};
#[cfg(feature = "cuda")]
use crate::dtype::DType;
#[cfg(feature = "cuda")]
use crate::{Error, Result};

/// Trait for loading model weights from various formats
#[cfg(feature = "cuda")]
pub trait WeightLoader {
    /// Load a tensor by name, converting to f32
    ///
    /// # Arguments
    /// * `name` - The name/key of the tensor in the weight file
    ///
    /// # Errors
    /// Returns an error if the tensor is not found or loading fails
    fn load_f32(&self, ctx: &CudaContext, name: &str) -> Result<CudaTensor<f32>>;

    /// Load a tensor as bf16, preserving half-precision on the GPU
    ///
    /// # Errors
    /// Returns an error if the tensor is not found or loading fails.
    /// The default implementation returns `UnsupportedDtype`.
    fn load_bf16(&self, _ctx: &CudaContext, name: &str) -> Result<CudaTensor<half::bf16>> {
        let dtype = self.get_dtype(name)?;
        Err(Error::UnsupportedDtype(format!(
            "load_bf16 not supported for dtype {dtype}"
        )))
    }

    /// Load a tensor as a quantized tensor (FP8, `Q8_0`, etc.)
    ///
    /// # Errors
    /// Returns an error if the tensor is not found, not quantized, or loading fails.
    /// The default implementation returns `UnsupportedDtype`.
    fn load_quantized(&self, _ctx: &CudaContext, name: &str) -> Result<QuantizedTensor> {
        let dtype = self.get_dtype(name)?;
        Err(Error::UnsupportedDtype(format!(
            "load_quantized not supported for dtype {dtype}"
        )))
    }

    /// Get the shape of a tensor without loading it
    fn get_shape(&self, name: &str) -> Result<Vec<usize>>;

    /// Get the dtype of a tensor
    fn get_dtype(&self, name: &str) -> Result<DType>;

    /// List all tensor names in the file
    fn tensor_names(&self) -> Vec<String>;

    /// Check if a tensor exists
    fn contains(&self, name: &str) -> bool;
}
