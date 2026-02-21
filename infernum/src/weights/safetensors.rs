//! `SafeTensors` file loading with memory mapping

#![allow(
    clippy::doc_markdown,
    clippy::implicit_clone,
    clippy::redundant_closure_for_method_calls
)]

use memmap2::Mmap;
use safetensors::tensor::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::cuda::shard::{ShardConfig, ShardStrategy};
use crate::cuda::{CudaContext, CudaTensor, QuantizedTensor};
use crate::dtype::DType;
use crate::weights::WeightLoader;
use crate::{Error, Result};

/// Loads weights from SafeTensors files using memory mapping
///
/// This avoids loading the entire model into RAM, instead mapping
/// the file directly and loading tensors on-demand to the GPU.
pub struct SafeTensorsLoader {
    /// Memory-mapped files (kept alive for the lifetime of the loader)
    mmaps: Vec<Mmap>,
    /// Tensor metadata: name -> (file_index, shape, dtype, data_offset, data_len)
    tensors: HashMap<String, TensorMeta>,
}

#[derive(Clone)]
struct TensorMeta {
    file_idx: usize,
    shape: Vec<usize>,
    dtype: DType,
    data_start: usize,
    data_len: usize,
}

impl SafeTensorsLoader {
    /// Load from a single SafeTensors file
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened or parsed
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_files(&[path.as_ref().to_path_buf()])
    }

    /// Load from multiple SafeTensors files (sharded models)
    ///
    /// # Errors
    /// Returns an error if any file cannot be opened or parsed
    pub fn from_files(paths: &[PathBuf]) -> Result<Self> {
        let mut mmaps = Vec::with_capacity(paths.len());
        let mut tensors = HashMap::new();

        for (file_idx, path) in paths.iter().enumerate() {
            let file = File::open(path)?;
            let mmap = unsafe { Mmap::map(&file)? };

            // Parse the SafeTensors header to get tensor metadata
            let st = SafeTensors::deserialize(&mmap)?;

            for (name, tensor) in st.tensors() {
                let shape: Vec<usize> = tensor.shape().to_vec();
                let dtype = safetensors_dtype_to_dtype(tensor.dtype())?;

                // Get raw data location within the mmap
                let data = tensor.data();
                let data_start = data.as_ptr() as usize - mmap.as_ptr() as usize;
                let data_len = data.len();

                tensors.insert(
                    name.to_owned(),
                    TensorMeta {
                        file_idx,
                        shape,
                        dtype,
                        data_start,
                        data_len,
                    },
                );
            }

            mmaps.push(mmap);
        }

        Ok(Self { mmaps, tensors })
    }

    /// Load from a directory containing SafeTensors files
    ///
    /// Automatically finds and loads all `.safetensors` files in the directory.
    ///
    /// # Errors
    /// Returns an error if the directory cannot be read or files cannot be loaded
    pub fn from_directory(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();

        // Look for model.safetensors or model-*.safetensors
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();

        if paths.is_empty() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("No .safetensors files found in {}", dir.display()),
            )));
        }

        // Sort for deterministic loading order
        paths.sort();

        Self::from_files(&paths)
    }

    /// Get raw tensor data from the memory-mapped file
    fn get_tensor_data(&self, name: &str) -> Result<&[u8]> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;

        let mmap = &self.mmaps[meta.file_idx];
        Ok(&mmap[meta.data_start..meta.data_start + meta.data_len])
    }
}

impl SafeTensorsLoader {
    /// Load a GPTQ-quantized linear layer from SafeTensors.
    ///
    /// Expects tensors named `{prefix}.qweight`, `{prefix}.scales`, `{prefix}.qzeros`.
    ///
    /// GPTQ layout:
    /// - `qweight`: `[in_features/8, out_features]` as int32 (8 INT4 values per int32)
    /// - `scales`:  `[in_features/group_size, out_features]` as f16
    /// - `qzeros`:  `[in_features/group_size, out_features/8]` as int32
    ///
    /// # Errors
    /// Returns an error if any required tensor is missing or GPU allocation fails.
    ///
    /// # Panics
    /// Panics if the scales or qzeros shapes are inconsistent with the qweight shape
    /// and group size.
    pub fn load_gptq_linear(
        &self,
        ctx: &CudaContext,
        prefix: &str,
        group_size: usize,
    ) -> Result<QuantizedTensor> {
        let qweight_name = format!("{prefix}.qweight");
        let scales_name = format!("{prefix}.scales");
        let qzeros_name = format!("{prefix}.qzeros");

        let qweight_meta = self
            .tensors
            .get(&qweight_name)
            .ok_or_else(|| Error::WeightNotFound(qweight_name.clone()))?;
        let scales_meta = self
            .tensors
            .get(&scales_name)
            .ok_or_else(|| Error::WeightNotFound(scales_name.clone()))?;
        let qzeros_meta = self
            .tensors
            .get(&qzeros_name)
            .ok_or_else(|| Error::WeightNotFound(qzeros_name.clone()))?;

        let qweight_data = self.get_tensor_data(&qweight_name)?;
        let scales_data = self.get_tensor_data(&scales_name)?;
        let qzeros_data = self.get_tensor_data(&qzeros_name)?;

        // Derive logical shape: [out_features, in_features] (N, K convention)
        let in_features = qweight_meta.shape[0] * 8; // unpacked from int32
        let out_features = qweight_meta.shape[1];

        // Resolve per-channel sentinel: group_size=0 means one group = full input dim
        let group_size = if group_size == 0 {
            in_features
        } else {
            group_size
        };

        // Validate scales shape
        let expected_num_groups = in_features / group_size;
        assert_eq!(
            scales_meta.shape,
            vec![expected_num_groups, out_features],
            "GPTQ scales shape mismatch: expected [{expected_num_groups}, {out_features}], got {:?}",
            scales_meta.shape
        );
        assert_eq!(
            qzeros_meta.shape,
            vec![expected_num_groups, out_features / 8],
            "GPTQ qzeros shape mismatch: expected [{expected_num_groups}, {}], got {:?}",
            out_features / 8,
            qzeros_meta.shape
        );

        QuantizedTensor::from_gptq_raw(
            ctx,
            &[out_features, in_features],
            DType::GPTQ_INT4,
            qweight_data,
            scales_data,
            qzeros_data,
            group_size,
        )
    }

    /// Load an AWQ-quantized linear layer from SafeTensors.
    ///
    /// Expects tensors named `{prefix}.qweight`, `{prefix}.scales`, `{prefix}.qzeros`.
    ///
    /// AWQ layout (transposed packing axis vs GPTQ):
    /// - `qweight`: `[in_features, out_features/8]` as int32 (8 INT4 values per int32)
    /// - `scales`:  `[in_features/group_size, out_features]` as f16
    /// - `qzeros`:  `[in_features/group_size, out_features/8]` as int32
    ///
    /// # Errors
    /// Returns an error if any required tensor is missing or GPU allocation fails.
    ///
    /// # Panics
    /// Panics if the scales or qzeros shapes are inconsistent with the qweight shape
    /// and group size.
    pub fn load_awq_linear(
        &self,
        ctx: &CudaContext,
        prefix: &str,
        group_size: usize,
    ) -> Result<QuantizedTensor> {
        let qweight_name = format!("{prefix}.qweight");
        let scales_name = format!("{prefix}.scales");
        let qzeros_name = format!("{prefix}.qzeros");

        let qweight_meta = self
            .tensors
            .get(&qweight_name)
            .ok_or_else(|| Error::WeightNotFound(qweight_name.clone()))?;
        let scales_meta = self
            .tensors
            .get(&scales_name)
            .ok_or_else(|| Error::WeightNotFound(scales_name.clone()))?;
        let qzeros_meta = self
            .tensors
            .get(&qzeros_name)
            .ok_or_else(|| Error::WeightNotFound(qzeros_name.clone()))?;

        let qweight_data = self.get_tensor_data(&qweight_name)?;
        let scales_data = self.get_tensor_data(&scales_name)?;
        let qzeros_data = self.get_tensor_data(&qzeros_name)?;

        // Derive logical shape: [out_features, in_features] (N, K convention)
        let in_features = qweight_meta.shape[0];
        let out_features = qweight_meta.shape[1] * 8; // unpacked from int32

        // Resolve per-channel sentinel: group_size=0 means one group = full input dim
        let group_size = if group_size == 0 {
            in_features
        } else {
            group_size
        };

        // Validate scales shape
        let expected_num_groups = in_features / group_size;
        assert_eq!(
            scales_meta.shape,
            vec![expected_num_groups, out_features],
            "AWQ scales shape mismatch: expected [{expected_num_groups}, {out_features}], got {:?}",
            scales_meta.shape
        );
        assert_eq!(
            qzeros_meta.shape,
            vec![expected_num_groups, out_features / 8],
            "AWQ qzeros shape mismatch: expected [{expected_num_groups}, {}], got {:?}",
            out_features / 8,
            qzeros_meta.shape
        );

        QuantizedTensor::from_gptq_raw(
            ctx,
            &[out_features, in_features],
            DType::AWQ_INT4,
            qweight_data,
            scales_data,
            qzeros_data,
            group_size,
        )
    }

    /// Load a GPTQ-quantized linear layer with tensor-parallel sharding.
    ///
    /// - `Column`: splits along the output dimension (N). Each rank gets
    ///   `N/world_size` output channels.
    /// - `Row`: splits along the input dimension (K). Each rank gets
    ///   `K/world_size` input channels. Requires all-reduce after matmul.
    /// - `Replicate`: loads the full tensor.
    ///
    /// # Errors
    /// Returns an error if any required tensor is missing or GPU allocation fails.
    ///
    /// # Panics
    /// Panics if dimensions are not compatible with sharding constraints.
    #[allow(clippy::similar_names)]
    pub fn load_gptq_linear_sharded(
        &self,
        ctx: &CudaContext,
        prefix: &str,
        group_size: usize,
        shard: &ShardConfig,
        strategy: ShardStrategy,
    ) -> Result<QuantizedTensor> {
        if strategy == ShardStrategy::Replicate || shard.world_size == 1 {
            return self.load_gptq_linear(ctx, prefix, group_size);
        }

        let qweight_name = format!("{prefix}.qweight");
        let scales_name = format!("{prefix}.scales");
        let qzeros_name = format!("{prefix}.qzeros");

        let qweight_meta = self
            .tensors
            .get(&qweight_name)
            .ok_or_else(|| Error::WeightNotFound(qweight_name.clone()))?;

        let qweight_data = self.get_tensor_data(&qweight_name)?;
        let scales_data = self.get_tensor_data(&scales_name)?;
        let qzeros_data = self.get_tensor_data(&qzeros_name)?;

        // GPTQ packed layout: qweight [K/8, N] int32
        let in_features = qweight_meta.shape[0] * 8;
        let out_features = qweight_meta.shape[1];

        // Resolve per-channel sentinel: group_size=0 means one group = full input dim
        let group_size = if group_size == 0 {
            in_features
        } else {
            group_size
        };
        let num_groups = in_features / group_size;

        match strategy {
            ShardStrategy::Column => {
                // Split along N (output dimension)
                let (n_start, n_shard) = shard.shard_range(out_features);
                assert_eq!(
                    n_start % 8,
                    0,
                    "GPTQ column shard: N start ({n_start}) must be aligned to 8 for qzeros packing"
                );

                // qweight [K/8, N] int32 → slice columns [n_start..n_start+n_shard]
                let qw_rows = in_features / 8;
                let qw_sliced =
                    slice_2d_columns(qweight_data, qw_rows, out_features, n_start, n_shard, 4);
                // scales [num_groups, N] f16 → slice columns
                let sc_sliced =
                    slice_2d_columns(scales_data, num_groups, out_features, n_start, n_shard, 2);
                // qzeros [num_groups, N/8] int32 → slice columns
                let qz_cols = out_features / 8;
                let qz_start = n_start / 8;
                let qz_shard = n_shard / 8;
                let qz_sliced =
                    slice_2d_columns(qzeros_data, num_groups, qz_cols, qz_start, qz_shard, 4);

                QuantizedTensor::from_gptq_raw(
                    ctx,
                    &[n_shard, in_features],
                    DType::GPTQ_INT4,
                    &qw_sliced,
                    &sc_sliced,
                    &qz_sliced,
                    group_size,
                )
            }
            ShardStrategy::Row => {
                // Split along K (input dimension)
                let (k_start, k_shard) = shard.shard_range(in_features);
                assert_eq!(
                    k_start % group_size,
                    0,
                    "GPTQ row shard: K start ({k_start}) must be aligned to group_size ({group_size})"
                );

                // qweight [K/8, N] int32 → slice rows [k_start/8..(k_start+k_shard)/8]
                let qw_row_start = k_start / 8;
                let qw_row_shard = k_shard / 8;
                let qw_sliced =
                    slice_2d_rows(qweight_data, out_features, qw_row_start, qw_row_shard, 4);
                // scales [num_groups, N] f16 → slice rows [g_start..g_start+g_shard]
                let g_start = k_start / group_size;
                let g_shard = k_shard / group_size;
                let sc_sliced = slice_2d_rows(scales_data, out_features, g_start, g_shard, 2);
                // qzeros [num_groups, N/8] int32 → slice rows
                let qz_cols = out_features / 8;
                let qz_sliced = slice_2d_rows(qzeros_data, qz_cols, g_start, g_shard, 4);

                QuantizedTensor::from_gptq_raw(
                    ctx,
                    &[out_features, k_shard],
                    DType::GPTQ_INT4,
                    &qw_sliced,
                    &sc_sliced,
                    &qz_sliced,
                    group_size,
                )
            }
            ShardStrategy::Replicate => unreachable!(),
        }
    }

    /// Load an AWQ-quantized linear layer with tensor-parallel sharding.
    ///
    /// Same semantics as [`load_gptq_linear_sharded`](Self::load_gptq_linear_sharded)
    /// but for AWQ's transposed packing layout.
    ///
    /// # Errors
    /// Returns an error if any required tensor is missing or GPU allocation fails.
    ///
    /// # Panics
    /// Panics if dimensions are not compatible with sharding constraints.
    #[allow(clippy::similar_names)]
    pub fn load_awq_linear_sharded(
        &self,
        ctx: &CudaContext,
        prefix: &str,
        group_size: usize,
        shard: &ShardConfig,
        strategy: ShardStrategy,
    ) -> Result<QuantizedTensor> {
        if strategy == ShardStrategy::Replicate || shard.world_size == 1 {
            return self.load_awq_linear(ctx, prefix, group_size);
        }

        let qweight_name = format!("{prefix}.qweight");
        let scales_name = format!("{prefix}.scales");
        let qzeros_name = format!("{prefix}.qzeros");

        let qweight_meta = self
            .tensors
            .get(&qweight_name)
            .ok_or_else(|| Error::WeightNotFound(qweight_name.clone()))?;

        let qweight_data = self.get_tensor_data(&qweight_name)?;
        let scales_data = self.get_tensor_data(&scales_name)?;
        let qzeros_data = self.get_tensor_data(&qzeros_name)?;

        // AWQ packed layout: qweight [K, N/8] int32
        let in_features = qweight_meta.shape[0];
        let out_features = qweight_meta.shape[1] * 8;

        // Resolve per-channel sentinel: group_size=0 means one group = full input dim
        let group_size = if group_size == 0 {
            in_features
        } else {
            group_size
        };
        let num_groups = in_features / group_size;

        match strategy {
            ShardStrategy::Column => {
                // Split along N (output dimension)
                let (n_start, n_shard) = shard.shard_range(out_features);
                assert_eq!(
                    n_start % 8,
                    0,
                    "AWQ column shard: N start ({n_start}) must be aligned to 8 for packing"
                );

                // qweight [K, N/8] int32 → slice columns [n_start/8..(n_start+n_shard)/8]
                let qw_cols = out_features / 8;
                let qw_col_start = n_start / 8;
                let qw_col_shard = n_shard / 8;
                let qw_sliced = slice_2d_columns(
                    qweight_data,
                    in_features,
                    qw_cols,
                    qw_col_start,
                    qw_col_shard,
                    4,
                );
                // scales [num_groups, N] f16 → slice columns
                let sc_sliced =
                    slice_2d_columns(scales_data, num_groups, out_features, n_start, n_shard, 2);
                // qzeros [num_groups, N/8] int32 → slice columns
                let qz_cols = out_features / 8;
                let qz_start = n_start / 8;
                let qz_shard = n_shard / 8;
                let qz_sliced =
                    slice_2d_columns(qzeros_data, num_groups, qz_cols, qz_start, qz_shard, 4);

                QuantizedTensor::from_gptq_raw(
                    ctx,
                    &[n_shard, in_features],
                    DType::AWQ_INT4,
                    &qw_sliced,
                    &sc_sliced,
                    &qz_sliced,
                    group_size,
                )
            }
            ShardStrategy::Row => {
                // Split along K (input dimension)
                let (k_start, k_shard) = shard.shard_range(in_features);
                assert_eq!(
                    k_start % group_size,
                    0,
                    "AWQ row shard: K start ({k_start}) must be aligned to group_size ({group_size})"
                );

                // qweight [K, N/8] int32 → slice rows [k_start..k_start+k_shard]
                let qw_cols = out_features / 8;
                let qw_sliced = slice_2d_rows(qweight_data, qw_cols, k_start, k_shard, 4);
                // scales [num_groups, N] f16 → slice rows
                let g_start = k_start / group_size;
                let g_shard = k_shard / group_size;
                let sc_sliced = slice_2d_rows(scales_data, out_features, g_start, g_shard, 2);
                // qzeros [num_groups, N/8] int32 → slice rows
                let qz_cols = out_features / 8;
                let qz_sliced = slice_2d_rows(qzeros_data, qz_cols, g_start, g_shard, 4);

                QuantizedTensor::from_gptq_raw(
                    ctx,
                    &[out_features, k_shard],
                    DType::AWQ_INT4,
                    &qw_sliced,
                    &sc_sliced,
                    &qz_sliced,
                    group_size,
                )
            }
            ShardStrategy::Replicate => unreachable!(),
        }
    }
}

/// Slice contiguous rows from a 2D row-major byte buffer.
///
/// Each row has `cols` elements of `elem_bytes` bytes each.
/// Returns bytes for rows `[row_start .. row_start + row_count)`.
fn slice_2d_rows(
    data: &[u8],
    cols: usize,
    row_start: usize,
    row_count: usize,
    elem_bytes: usize,
) -> Vec<u8> {
    let row_bytes = cols * elem_bytes;
    let start = row_start * row_bytes;
    let end = start + row_count * row_bytes;
    data[start..end].to_vec()
}

/// Slice columns from a 2D row-major byte buffer.
///
/// The buffer has `rows` rows of `total_cols` elements, each `elem_bytes` bytes.
/// Returns bytes for columns `[col_start .. col_start + col_count)` from every row.
fn slice_2d_columns(
    data: &[u8],
    rows: usize,
    total_cols: usize,
    col_start: usize,
    col_count: usize,
    elem_bytes: usize,
) -> Vec<u8> {
    let row_bytes = total_cols * elem_bytes;
    let col_start_bytes = col_start * elem_bytes;
    let col_count_bytes = col_count * elem_bytes;
    let mut result = Vec::with_capacity(rows * col_count_bytes);
    for r in 0..rows {
        let row_offset = r * row_bytes + col_start_bytes;
        result.extend_from_slice(&data[row_offset..row_offset + col_count_bytes]);
    }
    result
}

impl WeightLoader for SafeTensorsLoader {
    fn load_f32(&self, ctx: &CudaContext, name: &str) -> Result<CudaTensor<f32>> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;

        let data = self.get_tensor_data(name)?;

        // Convert data based on dtype
        let f32_data: Vec<f32> = match meta.dtype {
            DType::F32 => {
                // Direct interpretation as f32
                let f32_slice: &[f32] = bytemuck::cast_slice(data);
                f32_slice.to_vec()
            }
            DType::F16 => {
                // Convert f16 to f32
                let f16_slice: &[half::f16] = bytemuck::cast_slice(data);
                f16_slice.iter().map(|x| x.to_f32()).collect()
            }
            DType::BF16 => {
                // Convert bf16 to f32
                let bf16_slice: &[half::bf16] = bytemuck::cast_slice(data);
                bf16_slice.iter().map(|x| x.to_f32()).collect()
            }
            DType::U32 => {
                return Err(Error::UnsupportedDtype(
                    "cannot convert U32 weights to f32".to_string(),
                ));
            }
            DType::Q8_0
            | DType::Q4_0
            | DType::Q6_K
            | DType::F8E4M3
            | DType::GPTQ_INT4
            | DType::AWQ_INT4 => {
                return Err(Error::UnsupportedDtype(format!(
                    "cannot load quantized dtype {} as f32",
                    meta.dtype
                )));
            }
        };

        CudaTensor::from_slice(ctx, &meta.shape, &f32_data)
    }

    fn load_f16(&self, ctx: &CudaContext, name: &str) -> Result<CudaTensor<half::f16>> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;

        let data = self.get_tensor_data(name)?;

        let f16_data: Vec<half::f16> = match meta.dtype {
            DType::F16 => {
                // Direct interpretation as f16
                bytemuck::cast_slice(data).to_vec()
            }
            DType::F32 => {
                // Convert f32 to f16
                let f32_slice: &[f32] = bytemuck::cast_slice(data);
                f32_slice.iter().map(|&x| half::f16::from_f32(x)).collect()
            }
            DType::BF16 => {
                // Convert bf16 to f16
                let bf16_slice: &[half::bf16] = bytemuck::cast_slice(data);
                bf16_slice
                    .iter()
                    .map(|x| half::f16::from_f32(x.to_f32()))
                    .collect()
            }
            other => {
                return Err(Error::UnsupportedDtype(format!(
                    "Cannot load '{name}' as f16: dtype is {other}"
                )));
            }
        };

        CudaTensor::from_slice(ctx, &meta.shape, &f16_data)
    }

    fn load_bf16(&self, ctx: &CudaContext, name: &str) -> Result<CudaTensor<half::bf16>> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;

        let data = self.get_tensor_data(name)?;

        let bf16_data: Vec<half::bf16> = match meta.dtype {
            DType::BF16 => {
                // Zero-conversion: reinterpret bytes as bf16 directly
                bytemuck::cast_slice(data).to_vec()
            }
            DType::F32 => {
                // Convert f32 to bf16
                let f32_slice: &[f32] = bytemuck::cast_slice(data);
                f32_slice.iter().map(|&x| half::bf16::from_f32(x)).collect()
            }
            DType::F16 => {
                // Convert f16 to bf16
                let f16_slice: &[half::f16] = bytemuck::cast_slice(data);
                f16_slice
                    .iter()
                    .map(|x| half::bf16::from_f32(x.to_f32()))
                    .collect()
            }
            other => {
                return Err(Error::UnsupportedDtype(format!(
                    "Cannot load '{name}' as bf16: dtype is {other}"
                )));
            }
        };

        CudaTensor::from_slice(ctx, &meta.shape, &bf16_data)
    }

    fn load_quantized(&self, ctx: &CudaContext, name: &str) -> Result<QuantizedTensor> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;

        let data = self.get_tensor_data(name)?;

        match meta.dtype {
            DType::F8E4M3 => QuantizedTensor::from_raw(ctx, &meta.shape, DType::F8E4M3, data, &[]),
            other => Err(Error::UnsupportedDtype(format!(
                "load_quantized not supported for dtype {other}"
            ))),
        }
    }

    fn load_gptq_linear(
        &self,
        ctx: &CudaContext,
        prefix: &str,
        group_size: usize,
    ) -> Result<QuantizedTensor> {
        Self::load_gptq_linear(self, ctx, prefix, group_size)
    }

    fn load_awq_linear(
        &self,
        ctx: &CudaContext,
        prefix: &str,
        group_size: usize,
    ) -> Result<QuantizedTensor> {
        Self::load_awq_linear(self, ctx, prefix, group_size)
    }

    fn load_gptq_linear_sharded(
        &self,
        ctx: &CudaContext,
        prefix: &str,
        group_size: usize,
        shard: &ShardConfig,
        strategy: ShardStrategy,
    ) -> Result<QuantizedTensor> {
        Self::load_gptq_linear_sharded(self, ctx, prefix, group_size, shard, strategy)
    }

    fn load_awq_linear_sharded(
        &self,
        ctx: &CudaContext,
        prefix: &str,
        group_size: usize,
        shard: &ShardConfig,
        strategy: ShardStrategy,
    ) -> Result<QuantizedTensor> {
        Self::load_awq_linear_sharded(self, ctx, prefix, group_size, shard, strategy)
    }

    fn get_shape(&self, name: &str) -> Result<Vec<usize>> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;
        Ok(meta.shape.clone())
    }

    fn get_dtype(&self, name: &str) -> Result<DType> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;
        Ok(meta.dtype)
    }

    fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

/// Convert SafeTensors dtype to our DType
fn safetensors_dtype_to_dtype(dtype: safetensors::Dtype) -> Result<DType> {
    match dtype {
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::F16 => Ok(DType::F16),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        safetensors::Dtype::F8_E4M3 => Ok(DType::F8E4M3),
        // GPTQ/AWQ store qweight and qzeros as int32
        safetensors::Dtype::I32 => Ok(DType::U32),
        other => Err(Error::UnsupportedDtype(format!("{other:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_conversion() {
        assert!(matches!(
            safetensors_dtype_to_dtype(safetensors::Dtype::F32),
            Ok(DType::F32)
        ));
        assert!(matches!(
            safetensors_dtype_to_dtype(safetensors::Dtype::F16),
            Ok(DType::F16)
        ));
        assert!(matches!(
            safetensors_dtype_to_dtype(safetensors::Dtype::BF16),
            Ok(DType::BF16)
        ));
        assert!(matches!(
            safetensors_dtype_to_dtype(safetensors::Dtype::I32),
            Ok(DType::U32)
        ));
    }

    #[test]
    fn test_slice_2d_rows() {
        // 3x4 matrix of u8, elem_bytes=1
        let data: Vec<u8> = vec![
            1, 2, 3, 4, // row 0
            5, 6, 7, 8, // row 1
            9, 10, 11, 12, // row 2
        ];
        // Slice rows 1..2
        let sliced = slice_2d_rows(&data, 4, 1, 2, 1);
        assert_eq!(sliced, vec![5, 6, 7, 8, 9, 10, 11, 12]);

        // Slice single row 0
        let sliced = slice_2d_rows(&data, 4, 0, 1, 1);
        assert_eq!(sliced, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_slice_2d_rows_int32() {
        // 2x2 matrix of int32 (elem_bytes=4)
        let data: Vec<u8> = vec![
            1, 0, 0, 0, // (0,0)
            2, 0, 0, 0, // (0,1)
            3, 0, 0, 0, // (1,0)
            4, 0, 0, 0, // (1,1)
        ];
        let sliced = slice_2d_rows(&data, 2, 1, 1, 4);
        assert_eq!(sliced, vec![3, 0, 0, 0, 4, 0, 0, 0]);
    }

    #[test]
    fn test_slice_2d_columns() {
        // 3x4 matrix of u8
        let data: Vec<u8> = vec![
            1, 2, 3, 4, // row 0
            5, 6, 7, 8, // row 1
            9, 10, 11, 12, // row 2
        ];
        // Slice columns 1..3
        let sliced = slice_2d_columns(&data, 3, 4, 1, 2, 1);
        assert_eq!(sliced, vec![2, 3, 6, 7, 10, 11]);
    }

    #[test]
    fn test_slice_2d_columns_int32() {
        // 2x3 matrix of int32 (elem_bytes=4)
        let data: Vec<u8> = vec![
            1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, // row 0: [1, 2, 3]
            4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0, // row 1: [4, 5, 6]
        ];
        // Slice column 2 only
        let sliced = slice_2d_columns(&data, 2, 3, 2, 1, 4);
        assert_eq!(sliced, vec![3, 0, 0, 0, 6, 0, 0, 0]);
    }

    #[test]
    fn test_slice_2d_columns_full_width() {
        // Slicing all columns should return the original data
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let sliced = slice_2d_columns(&data, 2, 3, 0, 3, 1);
        assert_eq!(sliced, data);
    }
}
