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
    fn load_f32(&self, ctx: &CudaContext, name: &str) -> Result<CudaTensor>;

    /// Load a tensor by name as f16 (no conversion if already f16)
    ///
    /// # Errors
    /// Returns an error if the tensor is not found or loading fails.
    /// The default implementation returns `UnsupportedDtype`.
    fn load_f16(&self, _ctx: &CudaContext, name: &str) -> Result<CudaTensor> {
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
    fn load_bf16(&self, _ctx: &CudaContext, name: &str) -> Result<CudaTensor> {
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

    /// Load a GPTQ INT4 quantized linear layer from `{prefix}.qweight`,
    /// `{prefix}.scales`, and `{prefix}.qzeros` tensors.
    ///
    /// # Errors
    /// Returns `UnsupportedDtype` by default. Overridden by `SafeTensorsLoader`.
    fn load_gptq_linear(
        &self,
        _ctx: &CudaContext,
        _prefix: &str,
        _group_size: usize,
    ) -> Result<QuantizedTensor> {
        Err(Error::UnsupportedDtype(
            "load_gptq_linear not supported by this loader".into(),
        ))
    }

    /// Load an AWQ INT4 quantized linear layer from `{prefix}.qweight`,
    /// `{prefix}.scales`, and `{prefix}.qzeros` tensors.
    ///
    /// # Errors
    /// Returns `UnsupportedDtype` by default. Overridden by `SafeTensorsLoader`.
    fn load_awq_linear(
        &self,
        _ctx: &CudaContext,
        _prefix: &str,
        _group_size: usize,
    ) -> Result<QuantizedTensor> {
        Err(Error::UnsupportedDtype(
            "load_awq_linear not supported by this loader".into(),
        ))
    }

    /// Load a GPTQ INT4 quantized linear layer, slicing the packed weights
    /// according to the shard strategy before uploading to the GPU.
    ///
    /// - `Column`: splits along the output dimension (N)
    /// - `Row`: splits along the input dimension (K)
    /// - `Replicate`: loads the full tensor
    ///
    /// The default implementation ignores the shard config and loads the full
    /// tensor (equivalent to `Replicate`).
    fn load_gptq_linear_sharded(
        &self,
        ctx: &CudaContext,
        prefix: &str,
        group_size: usize,
        _shard: &ShardConfig,
        _strategy: ShardStrategy,
    ) -> Result<QuantizedTensor> {
        self.load_gptq_linear(ctx, prefix, group_size)
    }

    /// Load an AWQ INT4 quantized linear layer, slicing the packed weights
    /// according to the shard strategy before uploading to the GPU.
    ///
    /// - `Column`: splits along the output dimension (N)
    /// - `Row`: splits along the input dimension (K)
    /// - `Replicate`: loads the full tensor
    ///
    /// The default implementation ignores the shard config and loads the full
    /// tensor (equivalent to `Replicate`).
    fn load_awq_linear_sharded(
        &self,
        ctx: &CudaContext,
        prefix: &str,
        group_size: usize,
        _shard: &ShardConfig,
        _strategy: ShardStrategy,
    ) -> Result<QuantizedTensor> {
        self.load_awq_linear(ctx, prefix, group_size)
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
    ) -> Result<CudaTensor> {
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
    ) -> Result<CudaTensor> {
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
    ) -> Result<CudaTensor> {
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
/// This is the untyped fallback used by the default `load_*_sharded` methods.
/// It operates on raw bytes, using `dtype.size_in_bytes()` for element addressing.
#[cfg(feature = "cuda")]
fn shard_tensor_on_host(
    tensor: &CudaTensor,
    shard: &ShardConfig,
    strategy: ShardStrategy,
) -> Result<CudaTensor> {
    use crate::tensor::Tensor;

    let dtype = tensor.dtype();
    let elem = dtype.size_in_bytes();

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

            let data = tensor.to_raw_bytes()?;
            let row_bytes = cols * elem;
            let start = start_row * row_bytes;
            let end = start + shard_rows * row_bytes;
            CudaTensor::from_raw_bytes(
                tensor.context(),
                &[shard_rows, cols],
                dtype,
                &data[start..end],
            )
        }
        ShardStrategy::Row => {
            // Split along dim 1 (columns)
            let shape = tensor.shape();
            assert_eq!(shape.len(), 2, "Row shard requires a 2D tensor");
            let (rows, cols) = (shape[0], shape[1]);
            let (start_col, shard_cols) = shard.shard_range(cols);

            let data = tensor.to_raw_bytes()?;
            let mut shard_data = vec![0u8; rows * shard_cols * elem];
            for r in 0..rows {
                let src_start = (r * cols + start_col) * elem;
                let dst_start = r * shard_cols * elem;
                let chunk = shard_cols * elem;
                shard_data[dst_start..dst_start + chunk]
                    .copy_from_slice(&data[src_start..src_start + chunk]);
            }
            CudaTensor::from_raw_bytes(tensor.context(), &[rows, shard_cols], dtype, &shard_data)
        }
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    fn shard(rank: usize, world_size: usize) -> ShardConfig {
        ShardConfig { rank, world_size }
    }

    #[test]
    fn test_replicate_returns_same_data() {
        let ctx = CudaContext::new(0).unwrap();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = CudaTensor::from_slice(&ctx, &[2, 3], &data).unwrap();

        let result = shard_tensor_on_host(&tensor, &shard(0, 2), ShardStrategy::Replicate).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.to_vec::<f32>().unwrap(), data);
    }

    #[test]
    fn test_column_shard_2_gpus() {
        let ctx = CudaContext::new(0).unwrap();
        // 4x3 matrix
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
            10.0, 11.0, 12.0, // row 3
        ];
        let tensor = CudaTensor::from_slice(&ctx, &[4, 3], &data).unwrap();

        let r0 = shard_tensor_on_host(&tensor, &shard(0, 2), ShardStrategy::Column).unwrap();
        assert_eq!(r0.shape(), &[2, 3]);
        assert_eq!(
            r0.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        let r1 = shard_tensor_on_host(&tensor, &shard(1, 2), ShardStrategy::Column).unwrap();
        assert_eq!(r1.shape(), &[2, 3]);
        assert_eq!(
            r1.to_vec::<f32>().unwrap(),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
    }

    #[test]
    fn test_row_shard_2_gpus() {
        let ctx = CudaContext::new(0).unwrap();
        // 3x4 matrix
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // row 0
            5.0, 6.0, 7.0, 8.0, // row 1
            9.0, 10.0, 11.0, 12.0, // row 2
        ];
        let tensor = CudaTensor::from_slice(&ctx, &[3, 4], &data).unwrap();

        let r0 = shard_tensor_on_host(&tensor, &shard(0, 2), ShardStrategy::Row).unwrap();
        assert_eq!(r0.shape(), &[3, 2]);
        assert_eq!(
            r0.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0]
        );

        let r1 = shard_tensor_on_host(&tensor, &shard(1, 2), ShardStrategy::Row).unwrap();
        assert_eq!(r1.shape(), &[3, 2]);
        assert_eq!(
            r1.to_vec::<f32>().unwrap(),
            vec![3.0, 4.0, 7.0, 8.0, 11.0, 12.0]
        );
    }

    #[test]
    fn test_column_shard_single_gpu() {
        let ctx = CudaContext::new(0).unwrap();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = CudaTensor::from_slice(&ctx, &[2, 3], &data).unwrap();

        let result = shard_tensor_on_host(&tensor, &shard(0, 1), ShardStrategy::Column).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.to_vec::<f32>().unwrap(), data);
    }

    #[test]
    fn test_row_shard_single_gpu() {
        let ctx = CudaContext::new(0).unwrap();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = CudaTensor::from_slice(&ctx, &[2, 3], &data).unwrap();

        let result = shard_tensor_on_host(&tensor, &shard(0, 1), ShardStrategy::Row).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.to_vec::<f32>().unwrap(), data);
    }

    #[test]
    fn test_column_shard_4_gpus() {
        let ctx = CudaContext::new(0).unwrap();
        // 8x2 matrix — 8 rows split among 4 GPUs → 2 rows each
        let data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let tensor = CudaTensor::from_slice(&ctx, &[8, 2], &data).unwrap();

        for rank in 0..4 {
            let result =
                shard_tensor_on_host(&tensor, &shard(rank, 4), ShardStrategy::Column).unwrap();
            assert_eq!(result.shape(), &[2, 2]);
            let start = rank * 4;
            assert_eq!(
                result.to_vec::<f32>().unwrap(),
                data[start..start + 4].to_vec()
            );
        }
    }

    #[test]
    fn test_row_shard_4_gpus() {
        let ctx = CudaContext::new(0).unwrap();
        // 2x8 matrix — 8 cols split among 4 GPUs → 2 cols each
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // row 0
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, // row 1
        ];
        let tensor = CudaTensor::from_slice(&ctx, &[2, 8], &data).unwrap();

        let r0 = shard_tensor_on_host(&tensor, &shard(0, 4), ShardStrategy::Row).unwrap();
        assert_eq!(r0.shape(), &[2, 2]);
        assert_eq!(r0.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 9.0, 10.0]);

        let r3 = shard_tensor_on_host(&tensor, &shard(3, 4), ShardStrategy::Row).unwrap();
        assert_eq!(r3.shape(), &[2, 2]);
        assert_eq!(r3.to_vec::<f32>().unwrap(), vec![7.0, 8.0, 15.0, 16.0]);
    }

    #[test]
    #[should_panic(expected = "Column shard requires a 2D tensor")]
    fn test_column_shard_rejects_1d() {
        let ctx = CudaContext::new(0).unwrap();
        let tensor = CudaTensor::from_slice(&ctx, &[6], &[1.0f32; 6]).unwrap();
        let _ = shard_tensor_on_host(&tensor, &shard(0, 2), ShardStrategy::Column);
    }

    #[test]
    #[should_panic(expected = "Row shard requires a 2D tensor")]
    fn test_row_shard_rejects_1d() {
        let ctx = CudaContext::new(0).unwrap();
        let tensor = CudaTensor::from_slice(&ctx, &[6], &[1.0f32; 6]).unwrap();
        let _ = shard_tensor_on_host(&tensor, &shard(0, 2), ShardStrategy::Row);
    }
}
