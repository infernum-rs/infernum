//! Host-side weight buffers returned by format parsers.
//!
//! These types hold raw tensor data on the CPU, ready for upload to any
//! backend. The parsing (GGUF dequant, SafeTensors mmap, etc.) is pure
//! CPU work; only the final upload step is backend-specific.

use crate::dtype::DType;

/// Raw tensor data on the host, ready for upload to a backend.
///
/// The `data` field holds the raw bytes in the tensor's `dtype` encoding.
/// For f32 tensors, this is `numel * 4` bytes of little-endian f32.
/// For quantized types, see [`HostQuantizedWeight`].
#[derive(Debug, Clone)]
pub struct HostTensor {
    /// Logical shape (e.g., `[out_features, in_features]` for a 2D weight).
    pub shape: Vec<usize>,
    /// Element dtype.
    pub dtype: DType,
    /// Raw bytes in `dtype` encoding.
    pub data: Vec<u8>,
}

impl HostTensor {
    /// Number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Interpret the data as a slice of `f32`.
    ///
    /// # Panics
    /// Panics if `dtype` is not `F32`.
    #[must_use]
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(self.dtype, DType::F32, "as_f32_slice: expected F32");
        bytemuck::cast_slice(&self.data)
    }
}

/// Host-side quantized weight data.
///
/// Holds the raw quantized bytes, scale factors, and optional zero-points
/// on the CPU. The layout matches what the backend's quantized tensor
/// constructor expects.
#[derive(Debug, Clone)]
pub struct HostQuantizedWeight {
    /// Logical shape (number of elements per dimension).
    pub shape: Vec<usize>,
    /// Quantization format (e.g., `Q8_0`, `Q4_0`, `F8E4M3`, `GPTQ_INT4`).
    pub dtype: DType,
    /// Raw quantized data bytes.
    pub data: Vec<u8>,
    /// Per-block or per-group scale factors (f16 as raw bytes).
    /// Empty for FP8 (scale is per-tensor or per-channel).
    pub scales: Vec<u8>,
    /// Per-group zero-points (packed int32 as raw bytes, GPTQ/AWQ only).
    pub qzeros: Option<Vec<u8>>,
    /// Number of elements per quantization group (GPTQ/AWQ only).
    pub group_size: Option<usize>,
    /// Per-tensor scale factor (FP8 dynamic quantization; 1.0 for block-quantized).
    pub weight_scale: f32,
    /// Per-channel (per-row) scale factors (compressed-tensors FP8).
    pub channel_scales: Option<Vec<f32>>,
}

/// A linear weight (dense or quantized) on the host, ready for upload.
#[derive(Debug, Clone)]
pub enum HostLinearWeight {
    /// Dense (unquantized) weight, pre-transposed to matmul-ready layout.
    Dense(HostTensor),
    /// Quantized weight with separate data/scales/qzeros.
    Quantized(HostQuantizedWeight),
}

/// Concatenate two 2D host tensors along the inner (last) dimension.
///
/// Given tensors `[rows, cols_a]` and `[rows, cols_b]`, produces
/// `[rows, cols_a + cols_b]`. Used for fusing K/V or gate/up weights
/// before upload.
///
/// # Panics
/// Panics if the tensors don't have matching shapes (same rows, same dtype, 2D).
#[must_use]
pub fn host_concat_inner_dim(a: &HostTensor, b: &HostTensor) -> HostTensor {
    assert_eq!(
        a.shape.len(),
        2,
        "host_concat_inner_dim: expected 2D tensors"
    );
    assert_eq!(
        b.shape.len(),
        2,
        "host_concat_inner_dim: expected 2D tensors"
    );
    assert_eq!(
        a.shape[0], b.shape[0],
        "host_concat_inner_dim: row count mismatch"
    );
    assert_eq!(a.dtype, b.dtype, "host_concat_inner_dim: dtype mismatch");

    let rows = a.shape[0];
    let cols_a = a.shape[1];
    let cols_b = b.shape[1];
    let elem = a.dtype.size_in_bytes();
    let row_a = cols_a * elem;
    let row_b = cols_b * elem;
    let total_cols = cols_a + cols_b;

    let mut data = vec![0u8; rows * total_cols * elem];
    for r in 0..rows {
        let dst_start = r * total_cols * elem;
        let src_a = r * row_a;
        data[dst_start..dst_start + row_a].copy_from_slice(&a.data[src_a..src_a + row_a]);
        let src_b = r * row_b;
        data[dst_start + row_a..dst_start + row_a + row_b]
            .copy_from_slice(&b.data[src_b..src_b + row_b]);
    }

    HostTensor {
        shape: vec![rows, total_cols],
        dtype: a.dtype,
        data,
    }
}
