//! Implements the generic `infernum::WeightLoader<CudaBackend>` trait
//! for any format loader (`SafeTensorsLoader`, `GgufLoader`, etc.) via
//! a blanket impl over `FormatLoader`.
//!
//! `CudaWeightLoader` wraps a format loader + `CudaContext` and provides
//! the high-level `load_linear` logic (GPTQ/AWQ, FP8 scales, transpose).

#![allow(
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::implicit_clone
)]

use crate::cuda::ops::{transpose_2d, LinearWeight};
use crate::cuda::{CudaContext, CudaTensor};
use crate::inner::backend_impl::CudaBackend;
use crate::inner::weights::loader::WeightLoader as FormatLoader;

use infernum::dtype::DType;
use infernum::shard::{ShardConfig, ShardStrategy};
use infernum::tensor::Tensor;
use infernum::weights::{QuantizationConfig, WeightLoader};
use infernum::Result;

/// Wraps a format-specific loader (SafeTensors or GGUF) and a `CudaContext`
/// to implement `infernum::WeightLoader<CudaBackend>`.
///
/// This type encapsulates the complex logic for loading linear weights:
/// GPTQ/AWQ quantized loading, FP8 scale handling, and host-side transpose
/// for non-f32 dtypes.
pub struct CudaWeightLoader<F: FormatLoader> {
    ctx: CudaContext,
    inner: F,
}

impl<F: FormatLoader> CudaWeightLoader<F> {
    /// Create a new `CudaWeightLoader` wrapping a format loader and context.
    pub fn new(ctx: CudaContext, inner: F) -> Self {
        Self { ctx, inner }
    }

    /// Get a reference to the underlying format loader.
    pub fn inner(&self) -> &F {
        &self.inner
    }

    /// Get the CUDA context.
    pub fn context(&self) -> &CudaContext {
        &self.ctx
    }

    /// Load a tensor dispatching by dtype (f32/f16/bf16).
    fn load_typed(&self, name: &str, dtype: DType) -> infernum::Result<CudaTensor> {
        match dtype {
            DType::F32 => self.inner.load_f32(&self.ctx, name),
            DType::F16 => self.inner.load_f16(&self.ctx, name),
            DType::BF16 => self.inner.load_bf16(&self.ctx, name),
            other => panic!("Unsupported dtype for load_typed: {other}"),
        }
    }

    /// Load a tensor with sharding, dispatching by dtype.
    fn load_typed_sharded(
        &self,
        name: &str,
        dtype: DType,
        shard: &ShardConfig,
        strategy: ShardStrategy,
    ) -> infernum::Result<CudaTensor> {
        match dtype {
            DType::F32 => self
                .inner
                .load_f32_sharded(&self.ctx, name, shard, strategy),
            DType::F16 => self
                .inner
                .load_f16_sharded(&self.ctx, name, shard, strategy),
            DType::BF16 => self
                .inner
                .load_bf16_sharded(&self.ctx, name, shard, strategy),
            other => panic!("Unsupported dtype for load_typed_sharded: {other}"),
        }
    }

    /// Transpose a 2D weight matrix on the host (for non-f32 dtypes where
    /// the GPU transpose kernel isn't available).
    fn host_transpose_to_linear(
        &self,
        tensor: &CudaTensor,
        model_dtype: DType,
    ) -> infernum::Result<LinearWeight> {
        let shape = tensor.shape().to_vec();
        let rows = shape[0];
        let cols = shape[1];
        let elem = model_dtype.size_in_bytes();
        let data = tensor.to_raw_bytes()?;
        let mut transposed = vec![0u8; data.len()];
        for r in 0..rows {
            for c in 0..cols {
                let src = (r * cols + c) * elem;
                let dst = (c * rows + r) * elem;
                transposed[dst..dst + elem].copy_from_slice(&data[src..src + elem]);
            }
        }
        Ok(LinearWeight::Dense(CudaTensor::from_raw_bytes(
            &self.ctx,
            &[cols, rows],
            model_dtype,
            &transposed,
        )?))
    }
}

impl<F: FormatLoader> WeightLoader<CudaBackend> for CudaWeightLoader<F> {
    fn load_tensor(&self, name: &str, dtype: DType) -> Result<CudaTensor> {
        self.load_typed(name, dtype)
    }

    fn load_linear(
        &self,
        name: &str,
        model_dtype: DType,
        quant_config: Option<&QuantizationConfig>,
    ) -> Result<LinearWeight> {
        // GPTQ/AWQ: load via dedicated loader (errors for formats that don't support it)
        if let Some(qc) = quant_config {
            let prefix = name
                .strip_suffix(".weight")
                .expect("GPTQ/AWQ weight name must end with .weight");
            match qc.quant_method.as_str() {
                "gptq" => {
                    let qt = self
                        .inner
                        .load_gptq_linear(&self.ctx, prefix, qc.group_size)?;
                    return Ok(LinearWeight::Quantized(qt));
                }
                "awq" => {
                    let qt = self
                        .inner
                        .load_awq_linear(&self.ctx, prefix, qc.group_size)?;
                    return Ok(LinearWeight::Quantized(qt));
                }
                _ => {
                    // Unknown quant method (e.g. "compressed-tensors") —
                    // fall through to standard weight loading
                }
            }
        }

        let file_dtype = self.inner.get_dtype(name)?;
        if file_dtype.is_quantized() {
            let mut qt = self.inner.load_quantized(&self.ctx, name)?;

            // FP8 models store a scale as a sibling tensor
            let scale_name = format!("{name}_scale");
            if self.inner.contains(&scale_name) {
                let scale_tensor = self.inner.load_f32(&self.ctx, &scale_name)?;
                let scale_val = scale_tensor.to_vec::<f32>()?;
                if scale_val.len() == 1 {
                    qt.set_weight_scale(&self.ctx, scale_val[0])?;
                } else {
                    qt.set_channel_scales(&self.ctx, &scale_val)?;
                }
            }

            Ok(LinearWeight::Quantized(qt))
        } else if model_dtype == DType::F32 {
            let f32_weight = self.inner.load_f32(&self.ctx, name)?;
            Ok(LinearWeight::Dense(transpose_2d(&f32_weight)?))
        } else {
            let native = self.load_typed(name, model_dtype)?;
            self.host_transpose_to_linear(&native, model_dtype)
        }
    }

    fn load_linear_sharded(
        &self,
        name: &str,
        model_dtype: DType,
        quant_config: Option<&QuantizationConfig>,
        shard: &ShardConfig,
        strategy: ShardStrategy,
    ) -> Result<LinearWeight> {
        // GPTQ/AWQ sharded
        if let Some(qc) = quant_config {
            let prefix = name
                .strip_suffix(".weight")
                .expect("GPTQ/AWQ weight name must end with .weight");
            match qc.quant_method.as_str() {
                "gptq" => {
                    let qt = self.inner.load_gptq_linear_sharded(
                        &self.ctx,
                        prefix,
                        qc.group_size,
                        shard,
                        strategy,
                    )?;
                    return Ok(LinearWeight::Quantized(qt));
                }
                "awq" => {
                    let qt = self.inner.load_awq_linear_sharded(
                        &self.ctx,
                        prefix,
                        qc.group_size,
                        shard,
                        strategy,
                    )?;
                    return Ok(LinearWeight::Quantized(qt));
                }
                _ => {}
            }
        }

        let file_dtype = self.inner.get_dtype(name)?;
        if file_dtype.is_quantized() {
            let mut qt = self.inner.load_quantized_sharded(
                &self.ctx,
                name,
                shard,
                ShardStrategy::Replicate,
            )?;
            let scale_name = format!("{name}_scale");
            if self.inner.contains(&scale_name) {
                let scale_tensor = self.inner.load_f32(&self.ctx, &scale_name)?;
                let scale_val = scale_tensor.to_vec::<f32>()?;
                if scale_val.len() == 1 {
                    qt.set_weight_scale(&self.ctx, scale_val[0])?;
                } else {
                    qt.set_channel_scales(&self.ctx, &scale_val)?;
                }
            }
            Ok(LinearWeight::Quantized(qt))
        } else if model_dtype == DType::F32 {
            let f32_weight = self
                .inner
                .load_f32_sharded(&self.ctx, name, shard, strategy)?;
            Ok(LinearWeight::Dense(transpose_2d(&f32_weight)?))
        } else {
            let native = self.load_typed_sharded(name, model_dtype, shard, strategy)?;
            self.host_transpose_to_linear(&native, model_dtype)
        }
    }

    fn load_tensor_sharded(
        &self,
        name: &str,
        dtype: DType,
        shard: &ShardConfig,
        strategy: ShardStrategy,
    ) -> Result<CudaTensor> {
        self.load_typed_sharded(name, dtype, shard, strategy)
    }

    fn get_shape(&self, name: &str) -> Result<Vec<usize>> {
        self.inner.get_shape(name)
    }

    fn get_dtype(&self, name: &str) -> Result<DType> {
        self.inner.get_dtype(name)
    }

    fn contains(&self, name: &str) -> bool {
        self.inner.contains(name)
    }

    fn tensor_names(&self) -> Vec<String> {
        self.inner.tensor_names()
    }
}
