//! Generic weight loading traits and types.
//!
//! [`WeightLoader<B>`] is the backend-agnostic interface for loading model
//! weights from disk (SafeTensors, GGUF, etc.). Model code calls the trait
//! methods without knowing the backend or file format.
//!
//! Backend-specific implementations (e.g., `SafeTensorsLoader` for CUDA)
//! live in the backend crates.
//!
//! The [`format`] module provides [`FormatLoader`](format::FormatLoader),
//! the backend-agnostic parser interface that returns host-side buffers.
//! Format-specific parsers (GGUF, SafeTensors) implement this trait.

#![allow(clippy::doc_markdown)]

pub mod format;
pub mod gguf;
pub mod host;

use crate::backend::MatmulOps;
use crate::dtype::DType;
use crate::shard::{ShardConfig, ShardStrategy};
use crate::Result;

use serde::Deserialize;

/// Quantization configuration parsed from `config.json`.
///
/// Present in GPTQ, AWQ, and compressed-tensors quantized models.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization method: `"gptq"`, `"awq"`, `"compressed-tensors"`, etc.
    pub quant_method: String,

    /// Number of bits per weight (typically 4).
    #[serde(default = "default_quant_bits")]
    pub bits: u32,

    /// Number of elements per quantization group (typically 128).
    /// A value of 0 means per-channel quantization (one group = full input dim),
    /// resolved to `in_features` at load time. JSON value `-1` is deserialized as 0.
    #[serde(
        default = "default_group_size",
        deserialize_with = "deserialize_group_size"
    )]
    pub group_size: usize,
}

fn default_quant_bits() -> u32 {
    4
}

fn default_group_size() -> usize {
    128
}

/// Deserialize `group_size`: -1 (per-channel) → 0 sentinel, positive values pass through.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn deserialize_group_size<'de, D>(deserializer: D) -> std::result::Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = i64::deserialize(deserializer)?;
    if v < 0 {
        Ok(0) // sentinel for per-channel
    } else {
        Ok(v as usize)
    }
}

/// Backend-agnostic weight loader.
///
/// Model code uses this trait to load tensors and linear weights without
/// knowing the backend. The loader holds backend-specific context internally
/// (e.g., `CudaContext` for GPU upload).
///
/// # Type parameters
/// - `B`: The backend type. Must implement `MatmulOps` so that `load_linear`
///   can return `B::LinearWeight`.
pub trait WeightLoader<B: MatmulOps> {
    /// Load a tensor by name in the specified dtype.
    ///
    /// The loader handles dtype conversion (e.g., loading bf16 from disk
    /// when the model requests bf16).
    ///
    /// # Errors
    /// Returns an error if the tensor is not found or dtype conversion fails.
    fn load_tensor(&self, name: &str, dtype: DType) -> Result<B::Tensor>;

    /// Load a linear weight (dense or quantized) by name.
    ///
    /// The loader handles:
    /// - GPTQ/AWQ quantized loading when `quant_config` is provided
    /// - FP8 weight scale loading (companion `{name}_scale` tensor)
    /// - Dense weight transposition (matmul-ready layout)
    /// - Host-side transpose for non-f32 dtypes
    ///
    /// # Errors
    /// Returns an error if loading or quantized packing fails.
    fn load_linear(
        &self,
        name: &str,
        model_dtype: DType,
        quant_config: Option<&QuantizationConfig>,
    ) -> Result<B::LinearWeight>;

    /// Load a linear weight with tensor-parallel sharding.
    ///
    /// The shard strategy determines how the weight is split:
    /// - `Column`: split output features (rows)
    /// - `Row`: split input features (columns)
    /// - `Replicate`: load full tensor
    ///
    /// # Errors
    /// Returns an error if loading or sharding fails.
    fn load_linear_sharded(
        &self,
        name: &str,
        model_dtype: DType,
        quant_config: Option<&QuantizationConfig>,
        shard: &ShardConfig,
        strategy: ShardStrategy,
    ) -> Result<B::LinearWeight>;

    /// Load a tensor with tensor-parallel sharding.
    ///
    /// # Errors
    /// Returns an error if loading or sharding fails.
    fn load_tensor_sharded(
        &self,
        name: &str,
        dtype: DType,
        shard: &ShardConfig,
        strategy: ShardStrategy,
    ) -> Result<B::Tensor>;

    /// Get the shape of a tensor without loading it.
    ///
    /// # Errors
    /// Returns an error if the tensor is not found.
    fn get_shape(&self, name: &str) -> Result<Vec<usize>>;

    /// Get the dtype of a tensor.
    ///
    /// # Errors
    /// Returns an error if the tensor is not found.
    fn get_dtype(&self, name: &str) -> Result<DType>;

    /// Check if a tensor exists.
    fn contains(&self, name: &str) -> bool;

    /// List all tensor names.
    fn tensor_names(&self) -> Vec<String>;
}
