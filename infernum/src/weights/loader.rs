//! Weight loader trait and utilities

#![allow(clippy::missing_errors_doc)]

#[cfg(feature = "cuda")]
use crate::cuda::shard::{ShardConfig, ShardStrategy};
#[cfg(feature = "cuda")]
use crate::cuda::{CudaContext, CudaTensor, QuantizedTensor};
#[cfg(feature = "cuda")]
use crate::dtype::DType;
#[cfg(feature = "cuda")]
use crate::{Error, Result};
#[cfg(feature = "cuda")]
use half::{bf16, f16};

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

    /// Load a tensor by name as f16 (no conversion if already f16)
    ///
    /// # Errors
    /// Returns an error if the tensor is not found or loading fails.
    /// The default implementation returns `UnsupportedDtype`.
    fn load_f16(&self, _ctx: &CudaContext, name: &str) -> Result<CudaTensor<f16>> {
        let dtype = self.get_dtype(name)?;
        Err(Error::UnsupportedDtype(format!(
            "load_f16 not supported for dtype {dtype}"
        )))
    }

    /// Load a tensor as bf16, preserving half-precision on the GPU
    ///
    /// # Errors
    /// Returns an error if the tensor is not found or loading fails.
    /// The default implementation returns `UnsupportedDtype`.
    fn load_bf16(&self, _ctx: &CudaContext, name: &str) -> Result<CudaTensor<bf16>> {
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

    /// Load an f32 tensor, slicing according to the shard strategy.
    ///
    /// For `Column`: splits along dim 0 (output features / rows).
    /// For `Row`: splits along dim 1 (input features / columns).
    /// For `Replicate`: loads the full tensor.
    ///
    /// The default implementation loads the full tensor on the host and
    /// slices before uploading to the GPU. Format-specific loaders can
    /// override this with zero-copy slicing from memory-mapped files.
    fn load_f32_sharded(
        &self,
        ctx: &CudaContext,
        name: &str,
        shard: &ShardConfig,
        strategy: ShardStrategy,
    ) -> Result<CudaTensor<f32>> {
        let full = self.load_f32(ctx, name)?;
        shard_tensor_on_host(&full, shard, strategy)
    }

    /// Load an f16 tensor, slicing according to the shard strategy.
    fn load_f16_sharded(
        &self,
        ctx: &CudaContext,
        name: &str,
        shard: &ShardConfig,
        strategy: ShardStrategy,
    ) -> Result<CudaTensor<f16>> {
        let full = self.load_f16(ctx, name)?;
        shard_tensor_on_host(&full, shard, strategy)
    }

    /// Load a bf16 tensor, slicing according to the shard strategy.
    fn load_bf16_sharded(
        &self,
        ctx: &CudaContext,
        name: &str,
        shard: &ShardConfig,
        strategy: ShardStrategy,
    ) -> Result<CudaTensor<bf16>> {
        let full = self.load_bf16(ctx, name)?;
        shard_tensor_on_host(&full, shard, strategy)
    }

    /// Load a quantized tensor, slicing according to the shard strategy.
    ///
    /// Default implementation returns an error for non-replicate strategies
    /// because slicing quantized blocks is non-trivial.
    fn load_quantized_sharded(
        &self,
        ctx: &CudaContext,
        name: &str,
        _shard: &ShardConfig,
        strategy: ShardStrategy,
    ) -> Result<QuantizedTensor> {
        if strategy != ShardStrategy::Replicate {
            return Err(Error::UnsupportedDtype(
                "Sharded loading of quantized tensors is not yet supported; \
                 use SafeTensors (FP8/BF16) for multi-GPU"
                    .to_string(),
            ));
        }
        self.load_quantized(ctx, name)
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

/// Shard a 2D tensor on the host by downloading, slicing, and re-uploading.
///
/// This is the generic fallback used by the default `load_*_sharded` methods.
#[cfg(feature = "cuda")]
fn shard_tensor_on_host<T>(
    tensor: &CudaTensor<T>,
    shard: &ShardConfig,
    strategy: ShardStrategy,
) -> Result<CudaTensor<T>>
where
    T: crate::dtype::TensorDType + cudarc::driver::DeviceRepr + Default + Copy,
{
    use crate::tensor::Tensor;

    match strategy {
        ShardStrategy::Replicate => {
            // Return as-is (already on GPU)
            Ok(tensor.reshape(tensor.shape()))
        }
        ShardStrategy::Column => {
            // Split along dim 0 (rows)
            let shape = tensor.shape();
            assert_eq!(shape.len(), 2, "Column shard requires a 2D tensor");
            let (rows, cols) = (shape[0], shape[1]);
            let (start_row, shard_rows) = shard.shard_range(rows);

            let data = tensor.to_vec()?;
            let shard_data: Vec<T> =
                data[start_row * cols..(start_row + shard_rows) * cols].to_vec();
            CudaTensor::from_slice(tensor.context(), &[shard_rows, cols], &shard_data)
        }
        ShardStrategy::Row => {
            // Split along dim 1 (columns)
            let shape = tensor.shape();
            assert_eq!(shape.len(), 2, "Row shard requires a 2D tensor");
            let (rows, cols) = (shape[0], shape[1]);
            let (start_col, shard_cols) = shard.shard_range(cols);

            let data = tensor.to_vec()?;
            let mut shard_data = vec![T::default(); rows * shard_cols];
            for r in 0..rows {
                shard_data[r * shard_cols..(r + 1) * shard_cols].copy_from_slice(
                    &data[r * cols + start_col..r * cols + start_col + shard_cols],
                );
            }
            CudaTensor::from_slice(tensor.context(), &[rows, shard_cols], &shard_data)
        }
    }
}
