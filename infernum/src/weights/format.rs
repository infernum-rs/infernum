//! Backend-agnostic format loader trait.
//!
//! [`FormatLoader`] is implemented by file-format parsers (GGUF, SafeTensors).
//! Methods return host-side buffers ([`HostTensor`], [`HostQuantizedWeight`])
//! that can be uploaded to any backend.

#![allow(clippy::doc_markdown)]

use crate::dtype::DType;
use crate::shard::{ShardConfig, ShardStrategy};
use crate::Result;

use super::host::{HostLinearWeight, HostQuantizedWeight, HostTensor};

/// A file-format parser that loads tensors to host memory.
///
/// Implementations handle format-specific parsing (GGUF blocks, SafeTensors
/// mmap, etc.) and return raw host buffers. The backend then uploads these
/// to device memory.
pub trait FormatLoader {
    /// Load a tensor by name, dequantizing to f32 if necessary.
    ///
    /// # Errors
    /// Returns an error if the tensor is not found or decoding fails.
    fn load_f32(&self, name: &str) -> Result<HostTensor>;

    /// Load a tensor as f16 (no conversion if already f16).
    ///
    /// # Errors
    /// Returns an error if the tensor is not found or dtype is unsupported.
    fn load_f16(&self, name: &str) -> Result<HostTensor> {
        let _ = name;
        Err(crate::Error::UnsupportedDtype(
            "load_f16 not supported by this format".into(),
        ))
    }

    /// Load a tensor as bf16, preserving half-precision.
    ///
    /// # Errors
    /// Returns an error if the tensor is not found or dtype is unsupported.
    fn load_bf16(&self, name: &str) -> Result<HostTensor> {
        let _ = name;
        Err(crate::Error::UnsupportedDtype(
            "load_bf16 not supported by this format".into(),
        ))
    }

    /// Load a tensor in its native dtype (no conversion).
    ///
    /// For dense formats (f32/f16/bf16), this returns the raw bytes.
    /// For quantized formats, use [`load_quantized`](Self::load_quantized).
    ///
    /// # Errors
    /// Returns an error if the tensor is not found.
    fn load_native(&self, name: &str) -> Result<HostTensor> {
        let dtype = self.get_dtype(name)?;
        match dtype {
            DType::F32 => self.load_f32(name),
            DType::F16 => self.load_f16(name),
            DType::BF16 => self.load_bf16(name),
            _ => Err(crate::Error::UnsupportedDtype(format!(
                "load_native: unsupported dtype {dtype} for tensor '{name}'"
            ))),
        }
    }

    /// Load a tensor as a quantized weight (Q8_0, Q4_0, FP8, etc.).
    ///
    /// # Errors
    /// Returns an error if the tensor is not found, not quantized, or
    /// the format doesn't support quantized loading.
    fn load_quantized(&self, name: &str) -> Result<HostQuantizedWeight> {
        let _ = name;
        Err(crate::Error::UnsupportedDtype(
            "load_quantized not supported by this format".into(),
        ))
    }

    /// Load a quantized tensor with head-dimension unpermuting (GGUF only).
    ///
    /// GGUF files interleave Q/K head rows. This loads and un-permutes
    /// the rows to restore the HuggingFace sequential-half layout.
    ///
    /// # Errors
    /// Returns an error if the tensor is not found or unpermute fails.
    fn load_quantized_unpermute(&self, name: &str, n_head: usize) -> Result<HostQuantizedWeight> {
        let _ = (name, n_head);
        Err(crate::Error::UnsupportedDtype(
            "load_quantized_unpermute not supported by this format".into(),
        ))
    }

    /// Load a GPTQ INT4 quantized linear layer.
    ///
    /// # Errors
    /// Returns an error if not supported by this format.
    fn load_gptq_linear(&self, prefix: &str, group_size: usize) -> Result<HostQuantizedWeight> {
        let _ = (prefix, group_size);
        Err(crate::Error::UnsupportedDtype(
            "load_gptq_linear not supported by this format".into(),
        ))
    }

    /// Load an AWQ INT4 quantized linear layer.
    ///
    /// # Errors
    /// Returns an error if not supported by this format.
    fn load_awq_linear(&self, prefix: &str, group_size: usize) -> Result<HostQuantizedWeight> {
        let _ = (prefix, group_size);
        Err(crate::Error::UnsupportedDtype(
            "load_awq_linear not supported by this format".into(),
        ))
    }

    /// Load a linear weight (dense or quantized) by name.
    ///
    /// Dispatches to `load_f32`/`load_quantized` based on the tensor's dtype.
    /// Dense weights are pre-transposed to matmul-ready layout.
    ///
    /// # Errors
    /// Returns an error if loading fails.
    fn load_linear(&self, name: &str) -> Result<HostLinearWeight> {
        let dtype = self.get_dtype(name)?;
        if dtype.is_quantized() {
            Ok(HostLinearWeight::Quantized(self.load_quantized(name)?))
        } else {
            let tensor = self.load_f32(name)?;
            Ok(HostLinearWeight::Dense(host_transpose_2d(&tensor)?))
        }
    }

    /// Load a linear weight with unpermuting (for GGUF Q/K weights).
    ///
    /// # Errors
    /// Returns an error if loading fails.
    fn load_linear_unpermute(&self, name: &str, n_head: usize) -> Result<HostLinearWeight> {
        let dtype = self.get_dtype(name)?;
        if dtype.is_quantized() {
            Ok(HostLinearWeight::Quantized(
                self.load_quantized_unpermute(name, n_head)?,
            ))
        } else {
            let tensor = self.load_f32(name)?;
            let unpermuted = host_unpermute_f32(&tensor, n_head)?;
            Ok(HostLinearWeight::Dense(host_transpose_2d(&unpermuted)?))
        }
    }

    /// Load a tensor with sharding (host-side slicing).
    ///
    /// # Errors
    /// Returns an error if loading fails.
    fn load_f32_sharded(
        &self,
        name: &str,
        shard: &ShardConfig,
        strategy: ShardStrategy,
    ) -> Result<HostTensor> {
        let full = self.load_f32(name)?;
        host_shard_tensor(&full, shard, strategy)
    }

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

// ---- Host-side tensor operations ----

/// Transpose a 2D host tensor (row-major → column-major for matmul).
///
/// # Errors
/// Returns an error if the tensor is not 2D.
pub fn host_transpose_2d(tensor: &HostTensor) -> Result<HostTensor> {
    let shape = &tensor.shape;
    if shape.len() != 2 {
        return Err(crate::Error::InvalidShape(format!(
            "host_transpose_2d: expected 2D tensor, got {shape:?}"
        )));
    }
    let (rows, cols) = (shape[0], shape[1]);
    let elem = tensor.dtype.size_in_bytes();
    let mut out = vec![0u8; tensor.data.len()];

    for r in 0..rows {
        for c in 0..cols {
            let src = (r * cols + c) * elem;
            let dst = (c * rows + r) * elem;
            out[dst..dst + elem].copy_from_slice(&tensor.data[src..src + elem]);
        }
    }

    Ok(HostTensor {
        shape: vec![cols, rows],
        dtype: tensor.dtype,
        data: out,
    })
}

/// Reverse the llama.cpp Q/K weight permutation for f32 tensors.
///
/// GGUF files interleave each head's rows: `[h0, h_half, h1, h_{half+1}, ...]`.
/// This restores the HuggingFace sequential-half layout.
///
/// # Errors
/// Returns an error if the tensor is not f32 or not 2D.
///
/// # Panics
/// Panics if the tensor dtype is not F32 or the tensor is not 2D.
pub fn host_unpermute_f32(tensor: &HostTensor, n_head: usize) -> Result<HostTensor> {
    assert_eq!(tensor.dtype, DType::F32, "host_unpermute_f32: expected F32");
    let shape = &tensor.shape;
    assert_eq!(shape.len(), 2, "host_unpermute_f32: expected 2D tensor");

    let n_rows = shape[0];
    let n_cols = shape[1];
    let head_dim = n_rows / n_head;
    let half_dim = head_dim / 2;

    let data: &[f32] = bytemuck::cast_slice(&tensor.data);
    let mut out = vec![0.0_f32; data.len()];

    for h in 0..n_head {
        for i in 0..half_dim {
            let src0 = (h * head_dim + 2 * i) * n_cols;
            let src1 = (h * head_dim + 2 * i + 1) * n_cols;
            let dst0 = (h * head_dim + i) * n_cols;
            let dst1 = (h * head_dim + i + half_dim) * n_cols;
            out[dst0..dst0 + n_cols].copy_from_slice(&data[src0..src0 + n_cols]);
            out[dst1..dst1 + n_cols].copy_from_slice(&data[src1..src1 + n_cols]);
        }
    }

    Ok(HostTensor {
        shape: shape.clone(),
        dtype: DType::F32,
        data: bytemuck::cast_slice(&out).to_vec(),
    })
}

/// Shard a 2D host tensor by slicing rows (Column) or columns (Row).
///
/// # Errors
/// Returns an error if the tensor is not 2D (for non-Replicate strategies).
///
/// # Panics
/// Panics if a Column or Row shard is applied to a non-2D tensor.
pub fn host_shard_tensor(
    tensor: &HostTensor,
    shard: &ShardConfig,
    strategy: ShardStrategy,
) -> Result<HostTensor> {
    match strategy {
        ShardStrategy::Replicate => Ok(tensor.clone()),
        ShardStrategy::Column => {
            let shape = &tensor.shape;
            assert_eq!(shape.len(), 2, "Column shard requires a 2D tensor");
            let (rows, cols) = (shape[0], shape[1]);
            let elem = tensor.dtype.size_in_bytes();
            let (start_row, shard_rows) = shard.shard_range(rows);
            let row_bytes = cols * elem;
            let start = start_row * row_bytes;
            let end = start + shard_rows * row_bytes;
            Ok(HostTensor {
                shape: vec![shard_rows, cols],
                dtype: tensor.dtype,
                data: tensor.data[start..end].to_vec(),
            })
        }
        ShardStrategy::Row => {
            let shape = &tensor.shape;
            assert_eq!(shape.len(), 2, "Row shard requires a 2D tensor");
            let (rows, cols) = (shape[0], shape[1]);
            let elem = tensor.dtype.size_in_bytes();
            let (start_col, shard_cols) = shard.shard_range(cols);
            let mut shard_data = vec![0u8; rows * shard_cols * elem];
            for r in 0..rows {
                let src_start = (r * cols + start_col) * elem;
                let dst_start = r * shard_cols * elem;
                let chunk = shard_cols * elem;
                shard_data[dst_start..dst_start + chunk]
                    .copy_from_slice(&tensor.data[src_start..src_start + chunk]);
            }
            Ok(HostTensor {
                shape: vec![rows, shard_cols],
                dtype: tensor.dtype,
                data: shard_data,
            })
        }
    }
}
