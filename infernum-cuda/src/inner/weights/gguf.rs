//! GGUF file format loader
//!
//! Parses the GGUF binary format used by llama.cpp and loads tensors
//! as either `CudaTensor` (for unquantized weights) or
//! `QuantizedTensor` (for Q8_0, Q4_0, etc.).
//!
//! Reference: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::similar_names
)]

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

use memmap2::Mmap;

use crate::cuda::{CudaContext, CudaTensor, QuantizedTensor};
use crate::cuda::{ShardConfig, ShardStrategy};
use super::loader::WeightLoader;
use infernum::dtype::{
    DType, Q4K_BLOCK_ELEMENTS, Q4K_BLOCK_SIZE_BYTES, Q5K_BLOCK_ELEMENTS, Q5K_BLOCK_SIZE_BYTES,
    Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_SIZE_BYTES, QUANTIZATION_BLOCK_SIZE,
};
use infernum::Error;
use infernum::Result;

// ---------------------------------------------------------------------------
// GGUF constants
// ---------------------------------------------------------------------------

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" in little-endian
const GGUF_VERSION_3: u32 = 3;

/// GGML tensor type IDs (subset we support)
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q5_0: u32 = 6;
const GGML_TYPE_Q4_K: u32 = 12;
const GGML_TYPE_Q5_K: u32 = 13;
const GGML_TYPE_Q6_K: u32 = 14;
const GGML_TYPE_BF16: u32 = 30;

// Re-export GgufValue from infernum core (pure data type, no CUDA dependency)
pub use infernum::GgufValue;

// ---------------------------------------------------------------------------
// Tensor descriptor (parsed from header)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct GgufTensorInfo {
    shape: Vec<usize>,
    ggml_type: u32,
    offset: u64, // relative to tensor data start of its shard
    shard_idx: usize, // which mmap/tensor_data_offsets entry this tensor lives in
}

// ---------------------------------------------------------------------------
// GGUF Loader
// ---------------------------------------------------------------------------

/// Loads model weights from a GGUF file (single-file or split-file format).
///
/// Supports F32, F16, Q8_0, Q4_0, and quantized types. Tensors are memory-mapped
/// and loaded to the GPU on demand. Use `from_file` for a single GGUF file or
/// `from_split_files` for a multi-shard split (e.g. `model-00001-of-00005.gguf`).
pub struct GgufLoader {
    mmaps: Vec<Mmap>,
    metadata: HashMap<String, GgufValue>,
    tensors: HashMap<String, GgufTensorInfo>,
    tensor_data_offsets: Vec<usize>,
}

impl GgufLoader {
    /// Open and parse a single GGUF file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened, is not a valid GGUF file,
    /// or uses an unsupported version.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_split_files(&[path])
    }

    /// Open a GGUF file, auto-detecting split-shard format.
    ///
    /// If the filename matches the pattern `*-NNNNN-of-MMMMM.gguf` (e.g.
    /// `model-00001-of-00005.gguf`), all shards are loaded from the same
    /// directory. Otherwise, the file is opened as a single-file GGUF.
    ///
    /// # Errors
    /// Returns an error if any shard cannot be opened or parsed.
    pub fn from_file_or_split(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // Detect pattern: something-NNNNN-of-MMMMM.gguf
        let shard_paths = detect_split_gguf_shards(path, name);
        if let Some(shards) = shard_paths {
            let refs: Vec<&Path> = shards.iter().map(|p| p.as_path()).collect();
            Self::from_split_files(&refs)
        } else {
            Self::from_split_files(&[path])
        }
    }

    /// Open and parse a split GGUF model (multiple shard files).
    ///
    /// Pass all shards in order (shard 1 first). Each shard is a self-contained
    /// GGUF file with a subset of the model's tensors. Metadata is taken from
    /// the first shard; tensors are merged across all shards.
    ///
    /// # Errors
    /// Returns an error if any file cannot be opened, is not a valid GGUF file,
    /// or uses an unsupported version.
    pub fn from_split_files<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        assert!(!paths.is_empty(), "from_split_files: paths must not be empty");

        let mut mmaps = Vec::with_capacity(paths.len());
        let mut tensor_data_offsets = Vec::with_capacity(paths.len());
        let mut all_tensors: HashMap<String, GgufTensorInfo> = HashMap::new();
        let mut metadata = HashMap::new();

        for (shard_idx, path) in paths.iter().enumerate() {
            let file = std::fs::File::open(path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            let mut cursor = Cursor::new(mmap.as_ref());

            let magic = read_u32(&mut cursor)?;
            if magic != GGUF_MAGIC {
                return Err(Error::InvalidShape(format!(
                    "Not a GGUF file (magic: 0x{magic:08x}, expected 0x{GGUF_MAGIC:08x})"
                )));
            }

            let version = read_u32(&mut cursor)?;
            if version != GGUF_VERSION_3 {
                return Err(Error::UnsupportedDtype(format!(
                    "Unsupported GGUF version {version} (only v3 supported)"
                )));
            }

            let tensor_count = read_u64(&mut cursor)? as usize;
            let metadata_kv_count = read_u64(&mut cursor)? as usize;

            let mut shard_metadata = HashMap::with_capacity(metadata_kv_count);
            for _ in 0..metadata_kv_count {
                let key = read_gguf_string(&mut cursor)?;
                let value = read_gguf_value(&mut cursor)?;
                shard_metadata.insert(key, value);
            }
            if shard_idx == 0 {
                metadata = shard_metadata;
            }

            for _ in 0..tensor_count {
                let name = read_gguf_string(&mut cursor)?;
                let n_dims = read_u32(&mut cursor)? as usize;
                let mut shape = Vec::with_capacity(n_dims);
                for _ in 0..n_dims {
                    shape.push(read_u64(&mut cursor)? as usize);
                }
                // GGUF stores dimensions in column-major order (ne[0] = innermost),
                // reverse to our row-major convention (first dim = outermost).
                shape.reverse();
                let ggml_type = read_u32(&mut cursor)?;
                let offset = read_u64(&mut cursor)?;

                all_tensors.insert(
                    name,
                    GgufTensorInfo { shape, ggml_type, offset, shard_idx },
                );
            }

            let pos = cursor.position() as usize;
            tensor_data_offsets.push((pos + 31) & !31);
            mmaps.push(mmap);
        }

        Ok(Self {
            mmaps,
            metadata,
            tensors: all_tensors,
            tensor_data_offsets,
        })
    }

    /// Return the raw byte slice for a tensor's data region.
    ///
    /// Picks the correct shard mmap and applies the shard's base offset.
    fn tensor_slice<'a>(&'a self, info: &GgufTensorInfo, byte_len: usize) -> &'a [u8] {
        let start = self.tensor_data_offsets[info.shard_idx] + info.offset as usize;
        &self.mmaps[info.shard_idx][start..start + byte_len]
    }

    fn tensor_slice_range<'a>(&'a self, info: &GgufTensorInfo, offset: usize, len: usize) -> &'a [u8] {
        let start = self.tensor_data_offsets[info.shard_idx] + info.offset as usize;
        &self.mmaps[info.shard_idx][start + offset..start + offset + len]
    }

    /// Access parsed metadata
    #[must_use]
    pub fn metadata(&self) -> &HashMap<String, GgufValue> {
        &self.metadata
    }

    /// Get a metadata value by key
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Dequantize any GGUF tensor to a host `Vec<f32>`.
    ///
    /// Handles F32, F16, BF16, Q8_0, Q4_0, and Q6_K. Returns the element
    /// data and the tensor shape (used by callers that need to post-process
    /// before uploading, e.g. `load_bf16_unpermute`).
    fn dequantize_to_f32_vec(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;

        let dtype = ggml_type_to_dtype(info.ggml_type)?;
        let numel: usize = info.shape.iter().product();

        let f32_data: Vec<f32> = match dtype {
            DType::F32 => {
                bytemuck::cast_slice(self.tensor_slice(info, numel * 4)).to_vec()
            }
            DType::F16 => {
                let raw = self.tensor_slice(info, numel * 2);
                bytemuck::cast_slice::<_, half::f16>(raw)
                    .iter()
                    .map(|x| x.to_f32())
                    .collect()
            }
            DType::BF16 => {
                let raw = self.tensor_slice(info, numel * 2);
                bytemuck::cast_slice::<_, half::bf16>(raw)
                    .iter()
                    .map(|x| x.to_f32())
                    .collect()
            }
            DType::Q8_0 => {
                let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
                let block_bytes = dtype.block_size_in_bytes();
                let raw = self.tensor_slice(info, num_blocks * block_bytes);
                let mut out = Vec::with_capacity(numel);
                for block_idx in 0..num_blocks {
                    let bs = block_idx * block_bytes;
                    let scale = half::f16::from_le_bytes([raw[bs], raw[bs + 1]]).to_f32();
                    for j in 0..QUANTIZATION_BLOCK_SIZE {
                        out.push(f32::from(raw[bs + 2 + j] as i8) * scale);
                    }
                }
                out
            }
            DType::Q4_0 => {
                let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
                let block_bytes = dtype.block_size_in_bytes();
                let raw = self.tensor_slice(info, num_blocks * block_bytes);
                let mut out = vec![0.0f32; numel];
                for block_idx in 0..num_blocks {
                    let bs = block_idx * block_bytes;
                    let scale = half::f16::from_le_bytes([raw[bs], raw[bs + 1]]).to_f32();
                    let base = block_idx * QUANTIZATION_BLOCK_SIZE;
                    #[allow(clippy::cast_precision_loss)]
                    for j in 0..QUANTIZATION_BLOCK_SIZE / 2 {
                        let byte = raw[bs + 2 + j];
                        out[base + j] = (i32::from(byte & 0x0F) - 8) as f32 * scale;
                        out[base + j + 16] = (i32::from(byte >> 4) - 8) as f32 * scale;
                    }
                }
                out
            }
            DType::Q5_0 => {
                // Q5_0: 32 elements per block, 22 bytes.
                // Layout: d[f16] | qh[4 bytes] | qs[16 bytes]
                // High bit of each 5-bit value is packed in qh (1 bit per element).
                // Lower 4 bits are packed 2 per byte in qs.
                const Q5_0_BLOCK_BYTES: usize = 22;
                let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
                let raw = self.tensor_slice(info, num_blocks * Q5_0_BLOCK_BYTES);
                let mut out = vec![0.0f32; numel];
                #[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
                for block_idx in 0..num_blocks {
                    let b = block_idx * Q5_0_BLOCK_BYTES;
                    let d = half::f16::from_le_bytes([raw[b], raw[b + 1]]).to_f32();
                    let qh = &raw[b + 2..b + 6];
                    let qs = &raw[b + 6..b + 22];
                    let base = block_idx * QUANTIZATION_BLOCK_SIZE;
                    for i in 0..QUANTIZATION_BLOCK_SIZE {
                        let high = i32::from((qh[i / 8] >> (i % 8)) & 1);
                        let low = if i % 2 == 0 {
                            i32::from(qs[i / 2] & 0x0F)
                        } else {
                            i32::from(qs[i / 2] >> 4)
                        };
                        out[base + i] = d * ((high << 4 | low) as f32 - 16.0);
                    }
                }
                out
            }
            DType::Q5_K => {
                // Q5_K super-block: 256 elements, 176 bytes.
                // Layout: d[f16] dmin[f16] scales[12] qh[32] qs[128]
                // 8 sub-blocks of 32 elements; scales packed 6-bit in scales[12] (same as Q4_K).
                // Each element uses lower 4 bits from qs and 1 high bit from qh.
                const Q5K_BLOCK_ELEMS: usize = 256;
                const Q5K_BLOCK_BYTES: usize = 176;
                let num_blocks = numel / Q5K_BLOCK_ELEMS;
                let raw = self.tensor_slice(info, num_blocks * Q5K_BLOCK_BYTES);
                let mut out = Vec::with_capacity(numel);
                #[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
                for block_idx in 0..num_blocks {
                    let b = block_idx * Q5K_BLOCK_BYTES;
                    let d = half::f16::from_le_bytes([raw[b], raw[b + 1]]).to_f32();
                    let dmin = half::f16::from_le_bytes([raw[b + 2], raw[b + 3]]).to_f32();
                    let scales = &raw[b + 4..b + 16];
                    let qh = &raw[b + 16..b + 48]; // 32 bytes: 256 high bits (1 per element)
                    let qs = &raw[b + 48..b + Q5K_BLOCK_BYTES]; // 128 bytes: lower 4 bits
                    let mut q_idx: usize = 0;
                    let mut qh_idx: usize = 0;
                    let mut is: usize = 0;
                    for _ in 0..4 {
                        let (sc0, m0) = get_scale_min_k4(is, scales);
                        let (sc1, m1) = get_scale_min_k4(is + 1, scales);
                        let d1 = d * sc0 as f32;
                        let m1v = dmin * m0 as f32;
                        let d2 = d * sc1 as f32;
                        let m2v = dmin * m1 as f32;
                        for l in 0..32 {
                            let high = i32::from((qh[qh_idx + l / 8] >> (l % 8)) & 1);
                            let low = i32::from(qs[q_idx + l] & 0xF);
                            out.push(d1 * ((high << 4 | low) as f32) - m1v);
                        }
                        for l in 0..32 {
                            let high = i32::from((qh[qh_idx + 4 + l / 8] >> (l % 8)) & 1);
                            let low = i32::from(qs[q_idx + l] >> 4);
                            out.push(d2 * ((high << 4 | low) as f32) - m2v);
                        }
                        q_idx += 32;
                        qh_idx += 8;
                        is += 2;
                    }
                }
                out
            }
            DType::Q6_K => {
                // Q6_K super-block: 256 elements, 210 bytes.
                // Layout: ql[128] | qh[64] | scales[16 × i8] | d(f16)
                // Element mapping matches dequant_q6k_f16 CUDA kernel exactly.
                let num_blocks = numel / Q6_K_BLOCK_ELEMENTS;
                let raw = self.tensor_slice(info, num_blocks * Q6_K_BLOCK_SIZE_BYTES);
                let mut out = vec![0.0f32; numel];
                #[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
                for block_idx in 0..num_blocks {
                    let b = block_idx * Q6_K_BLOCK_SIZE_BYTES;
                    let d = half::f16::from_le_bytes([raw[b + 208], raw[b + 209]]).to_f32();
                    let ql = &raw[b..b + 128];
                    let qh = &raw[b + 128..b + 192];
                    let sc: &[i8] = bytemuck::cast_slice(&raw[b + 192..b + 208]);
                    let out_base = block_idx * Q6_K_BLOCK_ELEMENTS;
                    for elem in 0..Q6_K_BLOCK_ELEMENTS {
                        let sb = elem / 16;
                        let flat_idx = sb * 16 + (elem % 16);
                        let row8 = flat_idx / 32;
                        let col32 = flat_idx % 32;
                        let ql_half = row8 / 4;
                        let ql_nibble_sel = (row8 % 4) / 2;
                        let ql_offset = (row8 % 4) % 2;
                        let ql_byte = ql[ql_half * 64 + ql_offset * 32 + col32];
                        let ql_val = if ql_nibble_sel == 0 { ql_byte & 0x0F } else { ql_byte >> 4 };
                        let qh_half = row8 / 4;
                        let qh_byte = qh[qh_half * 32 + col32];
                        let qh_val = (qh_byte >> ((row8 % 4) * 2)) & 0x03;
                        let q = (i32::from(ql_val) | (i32::from(qh_val) << 4)) - 32;
                        out[out_base + elem] = d * sc[sb] as f32 * q as f32;
                    }
                }
                out
            }
            DType::Q4_K => {
                // Q4_K (K-quant): 256 elements per super-block (144 bytes).
                // Block layout: d[f16] dmin[f16] scales[12] qs[128]
                // 8 sub-blocks of 32 elements; scales packed 6-bit in scales[12].
                const Q4K_BLOCK_ELEMS: usize = 256;
                const Q4K_BLOCK_BYTES: usize = 144;
                let num_blocks = numel / Q4K_BLOCK_ELEMS;
                let raw = self.tensor_slice(info, num_blocks * Q4K_BLOCK_BYTES);
                let mut out = Vec::with_capacity(numel);
                #[allow(clippy::cast_precision_loss)]
                for block_idx in 0..num_blocks {
                    let b = block_idx * Q4K_BLOCK_BYTES;
                    let d = half::f16::from_le_bytes([raw[b], raw[b + 1]]).to_f32();
                    let dmin = half::f16::from_le_bytes([raw[b + 2], raw[b + 3]]).to_f32();
                    let scales = &raw[b + 4..b + 16];
                    let qs = &raw[b + 16..b + Q4K_BLOCK_BYTES];
                    let mut q_idx: usize = 0;
                    let mut is: usize = 0;
                    for _ in 0..4 {
                        let (sc0, m0) = get_scale_min_k4(is, scales);
                        let (sc1, m1) = get_scale_min_k4(is + 1, scales);
                        let d1 = d * sc0 as f32;
                        let m1v = dmin * m0 as f32;
                        let d2 = d * sc1 as f32;
                        let m2v = dmin * m1 as f32;
                        for l in 0..32 {
                            out.push(d1 * (qs[q_idx + l] & 0xF) as f32 - m1v);
                        }
                        for l in 0..32 {
                            out.push(d2 * (qs[q_idx + l] >> 4) as f32 - m2v);
                        }
                        q_idx += 32;
                        is += 2;
                    }
                }
                out
            }
            other => {
                return Err(Error::UnsupportedDtype(format!(
                    "dequantize_to_f32_vec: unsupported dtype {other} for '{name}'"
                )));
            }
        };
        Ok((f32_data, info.shape.clone()))
    }

    /// Load a tensor as BF16, dequantizing quantized types on the host and
    /// applying the GGUF Q/K row-permutation reversal before upload.
    ///
    /// `n_head` is the number of attention heads for this projection (use
    /// `num_attention_heads` for Q, `num_key_value_heads` for K).
    ///
    /// # Errors
    /// Returns an error if the tensor is not found, the dtype is unsupported,
    /// or GPU allocation fails.
    pub fn load_bf16_unpermute(
        &self,
        ctx: &CudaContext,
        name: &str,
        n_head: usize,
    ) -> Result<CudaTensor> {
        let (data, shape) = self.dequantize_to_f32_vec(name)?;
        let n_rows = shape[0];
        let n_cols = shape[1];
        let head_dim = n_rows / n_head;
        let half_dim = head_dim / 2;

        let mut unpermuted = vec![0.0f32; data.len()];
        for h in 0..n_head {
            for i in 0..half_dim {
                let src0 = (h * head_dim + 2 * i) * n_cols;
                let src1 = (h * head_dim + 2 * i + 1) * n_cols;
                let dst0 = (h * head_dim + i) * n_cols;
                let dst1 = (h * head_dim + i + half_dim) * n_cols;
                unpermuted[dst0..dst0 + n_cols].copy_from_slice(&data[src0..src0 + n_cols]);
                unpermuted[dst1..dst1 + n_cols].copy_from_slice(&data[src1..src1 + n_cols]);
            }
        }

        let bf16_data: Vec<half::bf16> = unpermuted
            .iter()
            .map(|&x| half::bf16::from_f32(x))
            .collect();
        CudaTensor::from_slice(ctx, &shape, &bf16_data)
    }

    /// Load a tensor as raw BF16 bytes on the host (no GPU upload).
    ///
    /// Returns `(bytes, shape)`. Caller is responsible for uploading and transposing.
    /// Use this in the sharded load path to avoid redundant GPU round-trips.
    pub fn load_bf16_bytes(&self, name: &str) -> Result<(Vec<u8>, Vec<usize>)> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;
        let dtype = ggml_type_to_dtype(info.ggml_type)?;
        let numel: usize = info.shape.iter().product();
        let bf16_data: Vec<half::bf16> = match dtype {
            DType::BF16 => {
                bytemuck::cast_slice(self.tensor_slice(info, numel * 2)).to_vec()
            }
            DType::F32 => {
                bytemuck::cast_slice::<_, f32>(self.tensor_slice(info, numel * 4))
                    .iter()
                    .map(|&x| half::bf16::from_f32(x))
                    .collect()
            }
            DType::F16 => {
                bytemuck::cast_slice::<_, half::f16>(self.tensor_slice(info, numel * 2))
                    .iter()
                    .map(|x| half::bf16::from_f32(x.to_f32()))
                    .collect()
            }
            _ => {
                let (f32_data, _) = self.dequantize_to_f32_vec(name)?;
                f32_data.iter().map(|&x| half::bf16::from_f32(x)).collect()
            }
        };
        Ok((
            bytemuck::cast_slice(&bf16_data).to_vec(),
            info.shape.clone(),
        ))
    }

    /// Load a Q/K tensor as raw BF16 bytes on host, reversing the llama.cpp RoPE permutation.
    pub fn load_bf16_bytes_unpermute(
        &self,
        name: &str,
        n_head: usize,
    ) -> Result<(Vec<u8>, Vec<usize>)> {
        let (data, shape) = self.dequantize_to_f32_vec(name)?;
        let n_rows = shape[0];
        let n_cols = shape[1];
        let head_dim = n_rows / n_head;
        let half_dim = head_dim / 2;
        let mut unpermuted = vec![0.0f32; data.len()];
        for h in 0..n_head {
            for i in 0..half_dim {
                let src0 = (h * head_dim + 2 * i) * n_cols;
                let src1 = (h * head_dim + 2 * i + 1) * n_cols;
                let dst0 = (h * head_dim + i) * n_cols;
                let dst1 = (h * head_dim + i + half_dim) * n_cols;
                unpermuted[dst0..dst0 + n_cols].copy_from_slice(&data[src0..src0 + n_cols]);
                unpermuted[dst1..dst1 + n_cols].copy_from_slice(&data[src1..src1 + n_cols]);
            }
        }
        let bf16_data: Vec<half::bf16> =
            unpermuted.iter().map(|&x| half::bf16::from_f32(x)).collect();
        Ok((bytemuck::cast_slice(&bf16_data).to_vec(), shape))
    }

    /// Load a tensor as a `QuantizedTensor` (for Q8_0 / Q4_0 weights).
    ///
    /// # Errors
    /// Returns an error if the tensor is not found, has an unsupported type,
    /// or GPU allocation fails.
    pub fn load_quantized(&self, ctx: &CudaContext, name: &str) -> Result<QuantizedTensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;

        let dtype = ggml_type_to_dtype(info.ggml_type)?;

        if !dtype.is_quantized() {
            return Err(Error::UnsupportedDtype(format!(
                "Tensor '{name}' is {dtype}, not a quantized type"
            )));
        }

        let numel: usize = info.shape.iter().product();

        match dtype {
            DType::Q8_0 | DType::Q4_0 => {
                let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
                let block_bytes = dtype.block_size_in_bytes();
                let total_bytes = num_blocks * block_bytes;
                let raw = self.tensor_slice(info, total_bytes);

                // GGUF block layout: [scale_f16 (2 bytes) | quants (block_bytes - 2)]
                // We need to split into separate data and scales arrays
                let quant_bytes_per_block = block_bytes - 2;
                let mut data_buf = Vec::with_capacity(num_blocks * quant_bytes_per_block);
                let mut scales_buf = Vec::with_capacity(num_blocks * 2);

                for block_idx in 0..num_blocks {
                    let block_start = block_idx * block_bytes;
                    // First 2 bytes are the f16 scale
                    scales_buf.extend_from_slice(&raw[block_start..block_start + 2]);
                    // Remaining bytes are the quantized values
                    data_buf.extend_from_slice(&raw[block_start + 2..block_start + block_bytes]);
                }

                QuantizedTensor::from_raw(ctx, &info.shape, dtype, &data_buf, &scales_buf)
            }
            DType::F8E4M3 => {
                let raw = self.tensor_slice(info, numel);
                QuantizedTensor::from_raw(ctx, &info.shape, dtype, raw, &[])
            }
            DType::Q6_K => {
                let num_blocks = numel / Q6_K_BLOCK_ELEMENTS;
                let total_bytes = num_blocks * Q6_K_BLOCK_SIZE_BYTES;
                let raw = self.tensor_slice(info, total_bytes);
                QuantizedTensor::from_raw(ctx, &info.shape, dtype, raw, &[])
            }
            DType::Q4_K => {
                let num_blocks = numel / Q4K_BLOCK_ELEMENTS;
                let total_bytes = num_blocks * Q4K_BLOCK_SIZE_BYTES;
                let raw = self.tensor_slice(info, total_bytes);
                QuantizedTensor::from_raw(ctx, &info.shape, dtype, raw, &[])
            }
            DType::Q5_K => {
                let num_blocks = numel / Q5K_BLOCK_ELEMENTS;
                let total_bytes = num_blocks * Q5K_BLOCK_SIZE_BYTES;
                let raw = self.tensor_slice(info, total_bytes);
                QuantizedTensor::from_raw(ctx, &info.shape, dtype, raw, &[])
            }
            other => Err(Error::UnsupportedDtype(format!(
                "Tensor '{name}' is {other}, which is not supported by load_quantized \
                 (use load_f32 to dequantize instead)"
            ))),
        }
    }

    /// Load a quantized tensor and reverse the llama.cpp Q/K weight permutation.
    ///
    /// GGUF files produced by `convert-hf-to-gguf.py` permute Q and K projection
    /// weights so that the interleaved RoPE convention used by llama.cpp produces
    /// correct results. Infernum uses the sequential half-half RoPE convention
    /// (matching HuggingFace), so we must reverse the permutation on load.
    ///
    /// The permutation interleaves the first and second halves of each head's
    /// rows: `(n_head, 2, half_dim, in_features)` → swapaxes(1,2) →
    /// `(n_head, half_dim, 2, in_features)`. This method applies the inverse.
    ///
    /// # Errors
    /// Returns an error if the tensor is not found, has an unsupported type,
    /// or GPU allocation fails.
    pub fn load_quantized_unpermute(
        &self,
        ctx: &CudaContext,
        name: &str,
        n_head: usize,
    ) -> Result<QuantizedTensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;

        let dtype = ggml_type_to_dtype(info.ggml_type)?;

        if !dtype.is_quantized() {
            return Err(Error::UnsupportedDtype(format!(
                "Tensor '{name}' is {dtype}, not a quantized type"
            )));
        }

        let numel: usize = info.shape.iter().product();

        // shape is [out_features, in_features] after GGUF→row-major conversion
        let n_rows = info.shape[0];
        let n_cols = info.shape[1];
        let head_dim = n_rows / n_head;
        let half_dim = head_dim / 2;

        match dtype {
            DType::Q8_0 | DType::Q4_0 => {
                let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
                let block_bytes = dtype.block_size_in_bytes();
                let total_bytes = num_blocks * block_bytes;
                let raw = self.tensor_slice(info, total_bytes);

                // Compute per-row byte count in the raw (interleaved) format
                let blocks_per_row = n_cols / QUANTIZATION_BLOCK_SIZE;
                let row_bytes = blocks_per_row * block_bytes;

                // Un-permute rows: GGUF interleaved → HuggingFace sequential halves
                // GGUF row (within a head) at position `2*i` → HF row `i`
                // GGUF row (within a head) at position `2*i+1` → HF row `i + half_dim`
                let mut unpermuted = vec![0u8; total_bytes];
                for h in 0..n_head {
                    for i in 0..half_dim {
                        let src0 = (h * head_dim + 2 * i) * row_bytes;
                        let src1 = (h * head_dim + 2 * i + 1) * row_bytes;
                        let dst0 = (h * head_dim + i) * row_bytes;
                        let dst1 = (h * head_dim + i + half_dim) * row_bytes;
                        unpermuted[dst0..dst0 + row_bytes]
                            .copy_from_slice(&raw[src0..src0 + row_bytes]);
                        unpermuted[dst1..dst1 + row_bytes]
                            .copy_from_slice(&raw[src1..src1 + row_bytes]);
                    }
                }

                // Split into separate data and scales arrays (same as load_quantized)
                let quant_bytes_per_block = block_bytes - 2;
                let mut data_buf = Vec::with_capacity(num_blocks * quant_bytes_per_block);
                let mut scales_buf = Vec::with_capacity(num_blocks * 2);

                for block_idx in 0..num_blocks {
                    let block_start = block_idx * block_bytes;
                    scales_buf.extend_from_slice(&unpermuted[block_start..block_start + 2]);
                    data_buf
                        .extend_from_slice(&unpermuted[block_start + 2..block_start + block_bytes]);
                }

                QuantizedTensor::from_raw(ctx, &info.shape, dtype, &data_buf, &scales_buf)
            }
            DType::Q6_K => {
                let num_blocks = numel / Q6_K_BLOCK_ELEMENTS;
                let total_bytes = num_blocks * Q6_K_BLOCK_SIZE_BYTES;
                let raw = self.tensor_slice(info, total_bytes);

                // Un-permute rows for Q6_K super-blocks
                let blocks_per_row = n_cols / Q6_K_BLOCK_ELEMENTS;
                let row_bytes = blocks_per_row * Q6_K_BLOCK_SIZE_BYTES;

                let mut unpermuted = vec![0u8; total_bytes];
                for h in 0..n_head {
                    for i in 0..half_dim {
                        let src0 = (h * head_dim + 2 * i) * row_bytes;
                        let src1 = (h * head_dim + 2 * i + 1) * row_bytes;
                        let dst0 = (h * head_dim + i) * row_bytes;
                        let dst1 = (h * head_dim + i + half_dim) * row_bytes;
                        unpermuted[dst0..dst0 + row_bytes]
                            .copy_from_slice(&raw[src0..src0 + row_bytes]);
                        unpermuted[dst1..dst1 + row_bytes]
                            .copy_from_slice(&raw[src1..src1 + row_bytes]);
                    }
                }

                QuantizedTensor::from_raw(ctx, &info.shape, dtype, &unpermuted, &[])
            }
            other => Err(Error::UnsupportedDtype(format!(
                "Tensor '{name}' is {other}, which is not supported by load_quantized_unpermute"
            ))),
        }
    }

    /// Load a quantized tensor, sharding along row or column block boundaries.
    ///
    /// For `Column`: splits along output rows (dim 0). For `Row`: splits along
    /// input columns (dim 1, must be a multiple of `block_size * world_size`).
    /// For `Replicate` or `world_size == 1`: delegates to `load_quantized`.
    ///
    /// Works directly from the mmap — no GPU round-trip needed for the slice.
    ///
    /// # Errors
    /// Returns an error if the tensor is not found, has an unsupported type,
    /// or GPU allocation fails.
    pub fn load_quantized_sharded(
        &self,
        ctx: &CudaContext,
        name: &str,
        shard: &ShardConfig,
        strategy: ShardStrategy,
    ) -> Result<QuantizedTensor> {
        if strategy == ShardStrategy::Replicate || shard.world_size == 1 {
            return self.load_quantized(ctx, name);
        }

        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;
        let dtype = ggml_type_to_dtype(info.ggml_type)?;
        if !dtype.is_quantized() {
            return Err(Error::UnsupportedDtype(format!(
                "load_quantized_sharded: '{name}' is {dtype}, not a quantized type"
            )));
        }

        let n_rows = info.shape[0];
        let n_cols = info.shape[1];

        match dtype {
            DType::Q8_0 | DType::Q4_0 => {
                let block_bytes = dtype.block_size_in_bytes();
                let blocks_per_row = n_cols / QUANTIZATION_BLOCK_SIZE;
                let row_bytes = blocks_per_row * block_bytes;
                let total_bytes = n_rows * row_bytes;
                let raw = self.tensor_slice(info, total_bytes);

                let (shard_raw, shard_shape): (Vec<u8>, Vec<usize>) = match strategy {
                    ShardStrategy::Replicate => unreachable!(),
                    ShardStrategy::Column => {
                        let (start_row, shard_rows) = shard.shard_range(n_rows);
                        let start = start_row * row_bytes;
                        let end = start + shard_rows * row_bytes;
                        (raw[start..end].to_vec(), vec![shard_rows, n_cols])
                    }
                    ShardStrategy::Row => {
                        let blocks_per_shard = blocks_per_row / shard.world_size;
                        assert_eq!(
                            blocks_per_row % shard.world_size,
                            0,
                            "n_cols ({n_cols}) must be divisible by block_size*world_size \
                             ({} * {})",
                            QUANTIZATION_BLOCK_SIZE,
                            shard.world_size
                        );
                        let shard_cols = blocks_per_shard * QUANTIZATION_BLOCK_SIZE;
                        let start_block = shard.rank * blocks_per_shard;
                        let shard_block_bytes = blocks_per_shard * block_bytes;

                        let mut buf = Vec::with_capacity(n_rows * shard_block_bytes);
                        for row in 0..n_rows {
                            let row_start = row * row_bytes + start_block * block_bytes;
                            buf.extend_from_slice(
                                &raw[row_start..row_start + shard_block_bytes],
                            );
                        }
                        (buf, vec![n_rows, shard_cols])
                    }
                };

                // Split interleaved GGUF blocks into separate data and scales buffers.
                let quant_bytes_per_block = block_bytes - 2;
                let num_shard_blocks = shard_raw.len() / block_bytes;
                let mut data_buf = Vec::with_capacity(num_shard_blocks * quant_bytes_per_block);
                let mut scales_buf = Vec::with_capacity(num_shard_blocks * 2);
                for block_idx in 0..num_shard_blocks {
                    let bs = block_idx * block_bytes;
                    scales_buf.extend_from_slice(&shard_raw[bs..bs + 2]);
                    data_buf.extend_from_slice(&shard_raw[bs + 2..bs + block_bytes]);
                }

                QuantizedTensor::from_raw(ctx, &shard_shape, dtype, &data_buf, &scales_buf)
            }
            DType::Q4_K | DType::Q5_K => {
                let (block_elems, block_bytes) = if dtype == DType::Q4_K {
                    (Q4K_BLOCK_ELEMENTS, Q4K_BLOCK_SIZE_BYTES)
                } else {
                    (Q5K_BLOCK_ELEMENTS, Q5K_BLOCK_SIZE_BYTES)
                };
                let blocks_per_row = n_cols / block_elems;
                let row_bytes = blocks_per_row * block_bytes;
                let total_bytes = n_rows * row_bytes;
                let raw = self.tensor_slice(info, total_bytes);

                let (shard_raw, shard_shape): (Vec<u8>, Vec<usize>) = match strategy {
                    ShardStrategy::Replicate => unreachable!(),
                    ShardStrategy::Column => {
                        let (start_row, shard_rows) = shard.shard_range(n_rows);
                        let start = start_row * row_bytes;
                        let end = start + shard_rows * row_bytes;
                        (raw[start..end].to_vec(), vec![shard_rows, n_cols])
                    }
                    ShardStrategy::Row => {
                        let blocks_per_shard = blocks_per_row / shard.world_size;
                        assert_eq!(
                            blocks_per_row % shard.world_size, 0,
                            "n_cols/block_elems ({blocks_per_row}) must be divisible by world_size ({})",
                            shard.world_size
                        );
                        let shard_cols = blocks_per_shard * block_elems;
                        let start_block = shard.rank * blocks_per_shard;
                        let shard_block_bytes = blocks_per_shard * block_bytes;

                        let mut buf = Vec::with_capacity(n_rows * shard_block_bytes);
                        for row in 0..n_rows {
                            let row_start = row * row_bytes + start_block * block_bytes;
                            buf.extend_from_slice(&raw[row_start..row_start + shard_block_bytes]);
                        }
                        (buf, vec![n_rows, shard_cols])
                    }
                };

                QuantizedTensor::from_raw(ctx, &shard_shape, dtype, &shard_raw, &[])
            }
            other => Err(Error::UnsupportedDtype(format!(
                "load_quantized_sharded: {other} sharding not yet supported"
            ))),
        }
    }

    /// Load a single expert's weight slice from a stacked expert tensor.
    ///
    /// Stacked expert tensors (e.g. `ffn_gate_exps.weight`) have GGUF shape
    /// `[num_experts, expert_rows, k]` (row-major after reversal). Expert `e`'s
    /// data is contiguous at byte offset `e * expert_bytes` from the tensor start.
    ///
    /// Supports Q4_K and Q5_K formats. Applies column or row sharding if requested.
    pub fn load_quantized_expert_slice(
        &self,
        ctx: &CudaContext,
        name: &str,
        expert_idx: usize,
        shard: Option<&ShardConfig>,
        strategy: ShardStrategy,
    ) -> Result<QuantizedTensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;
        let dtype = ggml_type_to_dtype(info.ggml_type)?;

        assert_eq!(
            info.shape.len(), 3,
            "Expert tensor '{name}' must be 3D (num_experts, expert_rows, k), got {:?}",
            info.shape
        );

        // Shape after reversal: [num_experts, expert_rows, k]
        let expert_rows = info.shape[1];
        let k = info.shape[2];

        let (block_elems, block_bytes) = match dtype {
            DType::Q4_K => (Q4K_BLOCK_ELEMENTS, Q4K_BLOCK_SIZE_BYTES),
            DType::Q5_K => (Q5K_BLOCK_ELEMENTS, Q5K_BLOCK_SIZE_BYTES),
            other => {
                return Err(Error::UnsupportedDtype(format!(
                    "load_quantized_expert_slice: unsupported dtype {other}"
                )));
            }
        };

        let blocks_per_k = k / block_elems;
        let row_bytes = blocks_per_k * block_bytes;
        let expert_bytes = expert_rows * row_bytes;

        let raw_expert = self.tensor_slice_range(info, expert_idx * expert_bytes, expert_bytes);

        let do_shard = shard.map_or(false, |s| s.world_size > 1)
            && strategy != ShardStrategy::Replicate;

        if !do_shard {
            return QuantizedTensor::from_raw(ctx, &[expert_rows, k], dtype, raw_expert, &[]);
        }

        let s = shard.unwrap();
        match strategy {
            ShardStrategy::Column => {
                let (start_row, shard_rows) = s.shard_range(expert_rows);
                let start = start_row * row_bytes;
                QuantizedTensor::from_raw(
                    ctx,
                    &[shard_rows, k],
                    dtype,
                    &raw_expert[start..start + shard_rows * row_bytes],
                    &[],
                )
            }
            ShardStrategy::Row => {
                let blocks_per_shard = blocks_per_k / s.world_size;
                assert_eq!(
                    blocks_per_k % s.world_size, 0,
                    "k/block_elems ({blocks_per_k}) must be divisible by world_size ({})",
                    s.world_size
                );
                let shard_k = blocks_per_shard * block_elems;
                let start_block = s.rank * blocks_per_shard;
                let shard_row_bytes = blocks_per_shard * block_bytes;

                let mut buf = Vec::with_capacity(expert_rows * shard_row_bytes);
                for row in 0..expert_rows {
                    let row_start = row * row_bytes + start_block * block_bytes;
                    buf.extend_from_slice(&raw_expert[row_start..row_start + shard_row_bytes]);
                }
                QuantizedTensor::from_raw(ctx, &[expert_rows, shard_k], dtype, &buf, &[])
            }
            ShardStrategy::Replicate => unreachable!(),
        }
    }

    /// Load a quantized Q/K weight, reverse the llama.cpp row permutation,
    /// and shard along the output dimension (Column strategy only).
    ///
    /// This is the sharded version of `load_quantized_unpermute`.
    /// Row-sharding Q/K projections is not supported (they are always
    /// Column-parallel in standard tensor-parallel layouts).
    ///
    /// # Errors
    /// Returns an error if the tensor is not found, has an unsupported type,
    /// or GPU allocation fails.
    pub fn load_quantized_unpermute_sharded(
        &self,
        ctx: &CudaContext,
        name: &str,
        n_head: usize,
        shard: &ShardConfig,
        strategy: ShardStrategy,
    ) -> Result<QuantizedTensor> {
        if strategy != ShardStrategy::Column || shard.world_size == 1 {
            return self.load_quantized_unpermute(ctx, name, n_head);
        }

        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;
        let dtype = ggml_type_to_dtype(info.ggml_type)?;
        if !dtype.is_quantized() {
            return Err(Error::UnsupportedDtype(format!(
                "load_quantized_unpermute_sharded: '{name}' is {dtype}, not a quantized type"
            )));
        }

        let numel: usize = info.shape.iter().product();
        let n_rows = info.shape[0];
        let n_cols = info.shape[1];
        let head_dim = n_rows / n_head;
        let half_dim = head_dim / 2;

        match dtype {
            DType::Q8_0 | DType::Q4_0 => {
                let block_bytes = dtype.block_size_in_bytes();
                let blocks_per_row = n_cols / QUANTIZATION_BLOCK_SIZE;
                let row_bytes = blocks_per_row * block_bytes;
                let total_bytes = num_blocks_from_numel(numel, block_bytes);
                let raw = self.tensor_slice(info, total_bytes);

                // Un-permute all rows into a contiguous host buffer (same as
                // load_quantized_unpermute), then slice the shard.
                let mut unpermuted = vec![0u8; total_bytes];
                for h in 0..n_head {
                    for i in 0..half_dim {
                        let src0 = (h * head_dim + 2 * i) * row_bytes;
                        let src1 = (h * head_dim + 2 * i + 1) * row_bytes;
                        let dst0 = (h * head_dim + i) * row_bytes;
                        let dst1 = (h * head_dim + i + half_dim) * row_bytes;
                        unpermuted[dst0..dst0 + row_bytes]
                            .copy_from_slice(&raw[src0..src0 + row_bytes]);
                        unpermuted[dst1..dst1 + row_bytes]
                            .copy_from_slice(&raw[src1..src1 + row_bytes]);
                    }
                }

                // Column shard: take rows [start_row, start_row + shard_rows).
                let (start_row, shard_rows) = shard.shard_range(n_rows);
                let start = start_row * row_bytes;
                let end = start + shard_rows * row_bytes;
                let shard_raw = &unpermuted[start..end];

                let quant_bytes_per_block = block_bytes - 2;
                let num_shard_blocks = shard_rows * blocks_per_row;
                let mut data_buf = Vec::with_capacity(num_shard_blocks * quant_bytes_per_block);
                let mut scales_buf = Vec::with_capacity(num_shard_blocks * 2);
                for block_idx in 0..num_shard_blocks {
                    let bs = block_idx * block_bytes;
                    scales_buf.extend_from_slice(&shard_raw[bs..bs + 2]);
                    data_buf.extend_from_slice(&shard_raw[bs + 2..bs + block_bytes]);
                }

                QuantizedTensor::from_raw(ctx, &[shard_rows, n_cols], dtype, &data_buf, &scales_buf)
            }
            other => Err(Error::UnsupportedDtype(format!(
                "load_quantized_unpermute_sharded: {other} not yet supported"
            ))),
        }
    }

    /// Get the dtype of a tensor in the file
    fn tensor_dtype(&self, name: &str) -> Result<DType> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;
        ggml_type_to_dtype(info.ggml_type)
    }
}

fn num_blocks_from_numel(numel: usize, block_bytes: usize) -> usize {
    // block_bytes includes 2 bytes of scale; quant elements = QUANTIZATION_BLOCK_SIZE per block
    numel / QUANTIZATION_BLOCK_SIZE * block_bytes
}

impl WeightLoader for GgufLoader {
    fn load_quantized(&self, ctx: &CudaContext, name: &str) -> Result<QuantizedTensor> {
        GgufLoader::load_quantized(self, ctx, name)
    }

    fn load_quantized_sharded(
        &self,
        ctx: &CudaContext,
        name: &str,
        shard: &ShardConfig,
        strategy: ShardStrategy,
    ) -> Result<QuantizedTensor> {
        GgufLoader::load_quantized_sharded(self, ctx, name, shard, strategy)
    }

    fn load_f32(&self, ctx: &CudaContext, name: &str) -> Result<CudaTensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;

        let dtype = ggml_type_to_dtype(info.ggml_type)?;
        let numel: usize = info.shape.iter().product();

        let f32_data: Vec<f32> = match dtype {
            DType::F32 => {
                bytemuck::cast_slice(self.tensor_slice(info, numel * 4)).to_vec()
            }
            DType::F16 => {
                let f16_slice: &[half::f16] =
                    bytemuck::cast_slice(self.tensor_slice(info, numel * 2));
                f16_slice.iter().map(|x| x.to_f32()).collect()
            }
            DType::Q8_0 => {
                // Block layout: [f16 scale (2 bytes) | 32 × i8 quants (32 bytes)] = 34 bytes
                let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
                let block_bytes = dtype.block_size_in_bytes();
                let raw = self.tensor_slice(info, num_blocks * block_bytes);

                let mut out = Vec::with_capacity(numel);
                for block_idx in 0..num_blocks {
                    let bs = block_idx * block_bytes;
                    let scale = half::f16::from_le_bytes([raw[bs], raw[bs + 1]]).to_f32();
                    for j in 0..QUANTIZATION_BLOCK_SIZE {
                        let q = raw[bs + 2 + j] as i8;
                        out.push(f32::from(q) * scale);
                    }
                }
                out
            }
            DType::Q4_0 => {
                // Block layout: [f16 scale (2 bytes) | 16 packed bytes (32 values)] = 18 bytes
                // GGML Q4_0 packing (interleaved):
                //   byte[j] holds element j (low nibble) and element j+16 (high nibble)
                // Unsigned [0,15] centered at 8 → signed [-8, 7]
                let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
                let block_bytes = dtype.block_size_in_bytes();
                let raw = self.tensor_slice(info, num_blocks * block_bytes);

                let mut out = vec![0.0; numel];
                for block_idx in 0..num_blocks {
                    let bs = block_idx * block_bytes;
                    let scale = half::f16::from_le_bytes([raw[bs], raw[bs + 1]]).to_f32();

                    let block_out_start = block_idx * QUANTIZATION_BLOCK_SIZE;

                    #[allow(clippy::cast_precision_loss)] // values in [-8, 7]
                    for j in 0..QUANTIZATION_BLOCK_SIZE / 2 {
                        let byte = raw[bs + 2 + j];
                        let lo = i32::from(byte & 0x0F) - 8;
                        let hi = i32::from(byte >> 4) - 8;

                        out[block_out_start + j] = lo as f32 * scale;
                        out[block_out_start + j + 16] = hi as f32 * scale;
                    }
                }
                out
            }
            DType::Q6_K => {
                // Super-block of 256 elements, 210 bytes:
                //   ql[128]   — lower 4 bits of each value (2 per byte, interleaved)
                //   qh[64]    — upper 2 bits of each value (4 per byte, interleaved)
                //   scales[16] — i8 sub-block scale (one per 16 elements)
                //   d (f16)   — super-block scale factor
                //
                // The layout matches ggml's block_q6_K with interleaved nibble/bit packing:
                // Elements are organized into 16 sub-blocks of 16 elements each.
                // ql and qh are accessed using the ggml interleaved layout.
                let num_blocks = numel / Q6_K_BLOCK_ELEMENTS;
                let total_bytes = num_blocks * Q6_K_BLOCK_SIZE_BYTES;
                let raw = self.tensor_slice(info, total_bytes);

                let mut out = Vec::with_capacity(numel);
                for block_idx in 0..num_blocks {
                    let bs = block_idx * Q6_K_BLOCK_SIZE_BYTES;
                    let ql = &raw[bs..bs + 128];
                    let qh = &raw[bs + 128..bs + 128 + 64];
                    let scales = &raw[bs + 128 + 64..bs + 128 + 64 + 16];
                    let d = half::f16::from_le_bytes([
                        raw[bs + 128 + 64 + 16],
                        raw[bs + 128 + 64 + 17],
                    ])
                    .to_f32();

                    for elem in 0..Q6_K_BLOCK_ELEMENTS {
                        // Map element index to ql/qh byte positions using ggml's interleaved layout
                        // The data flows through: (128,) -> (2,1,64) -> shift -> (2,2,64) -> (8,32)
                        // Then: (8,32) combined with qh -> (16,16) for 16 sub-blocks of 16 elements
                        let sb = elem / 16; // sub-block 0-15
                        let sb_elem = elem % 16; // element within sub-block 0-15
                        let flat_idx = sb * 16 + sb_elem;
                        let row8 = flat_idx / 32; // 0-7
                        let col32 = flat_idx % 32; // 0-31

                        // ql layout after reshape to (8,32):
                        // row 0: bytes 0-31 low nibbles
                        // row 1: bytes 32-63 low nibbles
                        // row 2: bytes 0-31 high nibbles
                        // row 3: bytes 32-63 high nibbles
                        // row 4: bytes 64-95 low nibbles
                        // row 5: bytes 96-127 low nibbles
                        // row 6: bytes 64-95 high nibbles
                        // row 7: bytes 96-127 high nibbles
                        let ql_half = row8 / 4; // 0 for rows 0-3, 1 for rows 4-7
                        let ql_nibble_sel = (row8 % 4) / 2; // 0 for rows 0-1,4-5 (low), 1 for 2-3,6-7 (high)
                        let ql_offset = (row8 % 4) % 2; // 0 for even rows in group, 1 for odd
                        let ql_byte_idx = ql_half * 64 + ql_offset * 32 + col32;
                        let ql_byte = ql[ql_byte_idx];
                        let ql_val = if ql_nibble_sel == 0 {
                            u32::from(ql_byte & 0x0F)
                        } else {
                            u32::from(ql_byte >> 4)
                        };

                        // qh layout: (64,) -> (2,1,32) -> shift -> (2,4,32) -> (8,32)
                        // row 0: bytes 0-31 bits 0-1
                        // row 1: bytes 0-31 bits 2-3
                        // row 2: bytes 0-31 bits 4-5
                        // row 3: bytes 0-31 bits 6-7
                        // row 4: bytes 32-63 bits 0-1
                        // row 5: bytes 32-63 bits 2-3
                        // row 6: bytes 32-63 bits 4-5
                        // row 7: bytes 32-63 bits 6-7
                        let qh_half = row8 / 4; // 0 or 1 (selects 32-byte half)
                        let qh_shift_sel = row8 % 4; // 0,1,2,3 -> shift 0,2,4,6
                        let qh_byte_idx = qh_half * 32 + col32;
                        let qh_byte = qh[qh_byte_idx];
                        let qh_shift = qh_shift_sel * 2;
                        let qh_val = u32::from((qh_byte >> qh_shift) & 0x03);

                        // Combine to 6-bit value and center: [0, 63] → [-32, 31]
                        let q = (ql_val | (qh_val << 4)) as i32 - 32;

                        let sc = f32::from(scales[sb] as i8);
                        #[allow(clippy::cast_precision_loss)] // q ∈ [-32, 31], no precision loss
                        out.push(d * sc * q as f32);
                    }
                }
                out
            }
            other => {
                return Err(Error::UnsupportedDtype(format!(
                    "Cannot load '{name}' as f32: dtype is {other} (use load_quantized instead)"
                )));
            }
        };

        CudaTensor::from_slice(ctx, &info.shape, &f32_data)
    }

    fn load_f16(&self, ctx: &CudaContext, name: &str) -> Result<CudaTensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;

        let dtype = ggml_type_to_dtype(info.ggml_type)?;
        let numel: usize = info.shape.iter().product();

        let f16_data: Vec<half::f16> = match dtype {
            DType::F16 => {
                bytemuck::cast_slice(self.tensor_slice(info, numel * 2)).to_vec()
            }
            DType::F32 => {
                let f32_slice: &[f32] = bytemuck::cast_slice(self.tensor_slice(info, numel * 4));
                f32_slice.iter().map(|&x| half::f16::from_f32(x)).collect()
            }
            DType::BF16 => {
                let bf16_slice: &[half::bf16] =
                    bytemuck::cast_slice(self.tensor_slice(info, numel * 2));
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

        CudaTensor::from_slice(ctx, &info.shape, &f16_data)
    }

    fn load_bf16(&self, ctx: &CudaContext, name: &str) -> Result<CudaTensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;

        let dtype = ggml_type_to_dtype(info.ggml_type)?;
        let numel: usize = info.shape.iter().product();

        let bf16_data: Vec<half::bf16> = match dtype {
            DType::BF16 => {
                bytemuck::cast_slice(self.tensor_slice(info, numel * 2)).to_vec()
            }
            DType::F32 => {
                let f32_slice: &[f32] = bytemuck::cast_slice(self.tensor_slice(info, numel * 4));
                f32_slice.iter().map(|&x| half::bf16::from_f32(x)).collect()
            }
            DType::F16 => {
                let f16_slice: &[half::f16] =
                    bytemuck::cast_slice(self.tensor_slice(info, numel * 2));
                f16_slice
                    .iter()
                    .map(|x| half::bf16::from_f32(x.to_f32()))
                    .collect()
            }
            _ => {
                // Quantized or other type: dequantize to f32 first, then cast.
                let (f32_data, _) = self.dequantize_to_f32_vec(name)?;
                f32_data
                    .iter()
                    .map(|&x| half::bf16::from_f32(x))
                    .collect()
            }
        };

        CudaTensor::from_slice(ctx, &info.shape, &bf16_data)
    }

    fn get_shape(&self, name: &str) -> Result<Vec<usize>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;
        Ok(info.shape.clone())
    }

    fn get_dtype(&self, name: &str) -> Result<DType> {
        self.tensor_dtype(name)
    }

    fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

// ---------------------------------------------------------------------------
// Split GGUF detection
// ---------------------------------------------------------------------------

/// Detect the split-GGUF filename pattern `stem-NNNNN-of-MMMMM.gguf`.
///
/// Returns all shard paths in order if the pattern matches, or `None` for
/// single-file GGUFs.
fn detect_split_gguf_shards(path: &Path, name: &str) -> Option<Vec<std::path::PathBuf>> {
    // Strip .gguf suffix
    let stem = name.strip_suffix(".gguf")?;

    // Find the last two numeric segments separated by "-of-"
    // e.g. "model-00001-of-00005" → prefix="model", n=1, total=5
    let of_pos = stem.rfind("-of-")?;
    let after_of = &stem[of_pos + 4..];
    let total: usize = after_of.parse().ok()?;
    if total <= 1 {
        return None; // single "shard" — treat as regular file
    }

    // The segment before "-of-MMMMM" is the current shard number
    let before_of = &stem[..of_pos];
    let dash_pos = before_of.rfind('-')?;
    let _shard_n: usize = before_of[dash_pos + 1..].parse().ok()?;
    let prefix = &before_of[..dash_pos]; // e.g. "model"

    let dir = path.parent().unwrap_or(std::path::Path::new("."));
    let digit_width = after_of.len(); // width of the MMMMM field (reuse for NNNNN)

    // Reconstruct all shard paths in order
    let mut shards = Vec::with_capacity(total);
    for i in 1..=total {
        let shard_name = format!(
            "{prefix}-{i:0width$}-of-{total:0width$}.gguf",
            width = digit_width
        );
        shards.push(dir.join(shard_name));
    }

    // Verify all shards exist
    if shards.iter().all(|p| p.exists()) {
        Some(shards)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Binary reading helpers
// ---------------------------------------------------------------------------

fn read_u8(cursor: &mut Cursor<&[u8]>) -> Result<u8> {
    let mut buf = [0u8; 1];
    cursor.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8(cursor: &mut Cursor<&[u8]>) -> Result<i8> {
    Ok(read_u8(cursor)? as i8)
}

fn read_u16(cursor: &mut Cursor<&[u8]>) -> Result<u16> {
    let mut buf = [0u8; 2];
    cursor.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16(cursor: &mut Cursor<&[u8]>) -> Result<i16> {
    let mut buf = [0u8; 2];
    cursor.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(cursor: &mut Cursor<&[u8]>) -> Result<i32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64(cursor: &mut Cursor<&[u8]>) -> Result<u64> {
    let mut buf = [0u8; 8];
    cursor.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(cursor: &mut Cursor<&[u8]>) -> Result<i64> {
    let mut buf = [0u8; 8];
    cursor.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32(cursor: &mut Cursor<&[u8]>) -> Result<f32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(cursor: &mut Cursor<&[u8]>) -> Result<f64> {
    let mut buf = [0u8; 8];
    cursor.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_gguf_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
    let len = read_u64(cursor)? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    String::from_utf8(buf)
        .map_err(|e| Error::Tokenizer(format!("Invalid UTF-8 in GGUF string: {e}")))
}

/// GGUF metadata value type IDs
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

fn read_gguf_value(cursor: &mut Cursor<&[u8]>) -> Result<GgufValue> {
    let value_type = read_u32(cursor)?;
    read_gguf_typed_value(cursor, value_type)
}

fn read_gguf_typed_value(cursor: &mut Cursor<&[u8]>, value_type: u32) -> Result<GgufValue> {
    match value_type {
        GGUF_TYPE_UINT8 => Ok(GgufValue::U8(read_u8(cursor)?)),
        GGUF_TYPE_INT8 => Ok(GgufValue::I8(read_i8(cursor)?)),
        GGUF_TYPE_UINT16 => Ok(GgufValue::U16(read_u16(cursor)?)),
        GGUF_TYPE_INT16 => Ok(GgufValue::I16(read_i16(cursor)?)),
        GGUF_TYPE_UINT32 => Ok(GgufValue::U32(read_u32(cursor)?)),
        GGUF_TYPE_INT32 => Ok(GgufValue::I32(read_i32(cursor)?)),
        GGUF_TYPE_UINT64 => Ok(GgufValue::U64(read_u64(cursor)?)),
        GGUF_TYPE_INT64 => Ok(GgufValue::I64(read_i64(cursor)?)),
        GGUF_TYPE_FLOAT32 => Ok(GgufValue::F32(read_f32(cursor)?)),
        GGUF_TYPE_FLOAT64 => Ok(GgufValue::F64(read_f64(cursor)?)),
        GGUF_TYPE_BOOL => Ok(GgufValue::Bool(read_u8(cursor)? != 0)),
        GGUF_TYPE_STRING => Ok(GgufValue::String(read_gguf_string(cursor)?)),
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32(cursor)?;
            let len = read_u64(cursor)? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_gguf_typed_value(cursor, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        other => Err(Error::UnsupportedDtype(format!(
            "Unknown GGUF metadata type: {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Type conversion
// ---------------------------------------------------------------------------

/// Decode one (scale, min) pair from the Q4_K packed 6-bit scales array.
///
/// `j` is the sub-block index (0..8); `q` is the 12-byte scales buffer.
/// Matches llama.cpp's `get_scale_min_k4`.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn ggml_type_to_dtype(ggml_type: u32) -> Result<DType> {
    match ggml_type {
        GGML_TYPE_F32 => Ok(DType::F32),
        GGML_TYPE_F16 => Ok(DType::F16),
        GGML_TYPE_BF16 => Ok(DType::BF16),
        GGML_TYPE_Q8_0 => Ok(DType::Q8_0),
        GGML_TYPE_Q4_0 => Ok(DType::Q4_0),
        GGML_TYPE_Q5_0 => Ok(DType::Q5_0),
        GGML_TYPE_Q4_K => Ok(DType::Q4_K),
        GGML_TYPE_Q5_K => Ok(DType::Q5_K),
        GGML_TYPE_Q6_K => Ok(DType::Q6_K),
        other => Err(Error::UnsupportedDtype(format!(
            "Unsupported GGML tensor type: {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Test helpers: build a minimal GGUF file in memory
// ---------------------------------------------------------------------------

#[cfg(test)]
mod test_helpers {
    use super::*;

    /// Builder for creating minimal GGUF v3 files in memory for testing.
    pub struct GgufBuilder {
        metadata: Vec<(String, u32, Vec<u8>)>, // (key, type_id, raw_value_bytes)
        tensors: Vec<(String, Vec<usize>, u32, Vec<u8>)>, // (name, shape, ggml_type, data)
    }

    impl GgufBuilder {
        pub fn new() -> Self {
            Self {
                metadata: Vec::new(),
                tensors: Vec::new(),
            }
        }

        pub fn add_metadata_u32(&mut self, key: &str, value: u32) {
            self.metadata.push((
                key.to_string(),
                GGUF_TYPE_UINT32,
                value.to_le_bytes().to_vec(),
            ));
        }

        pub fn add_metadata_string(&mut self, key: &str, value: &str) {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
            self.metadata
                .push((key.to_string(), GGUF_TYPE_STRING, bytes));
        }

        pub fn add_metadata_f32(&mut self, key: &str, value: f32) {
            self.metadata.push((
                key.to_string(),
                GGUF_TYPE_FLOAT32,
                value.to_le_bytes().to_vec(),
            ));
        }

        pub fn add_tensor_f32(&mut self, name: &str, shape: &[usize], data: &[f32]) {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            self.tensors
                .push((name.to_string(), shape.to_vec(), GGML_TYPE_F32, bytes));
        }

        /// Add a Q8_0 tensor. `blocks` is a sequence of (scale_f16_bytes, quant_i8_bytes).
        pub fn add_tensor_q8(
            &mut self,
            name: &str,
            shape: &[usize],
            blocks: &[(half::f16, Vec<i8>)],
        ) {
            let mut bytes = Vec::new();
            for (scale, quants) in blocks {
                assert_eq!(quants.len(), QUANTIZATION_BLOCK_SIZE);
                bytes.extend_from_slice(&scale.to_le_bytes());
                for &q in quants {
                    bytes.push(q as u8);
                }
            }
            self.tensors
                .push((name.to_string(), shape.to_vec(), GGML_TYPE_Q8_0, bytes));
        }

        /// Add a Q4_0 tensor. `blocks` is a sequence of (scale_f16, packed_bytes).
        /// GGML packing: byte[j] has element j in low nibble, element j+16 in high nibble.
        pub fn add_tensor_q4(
            &mut self,
            name: &str,
            shape: &[usize],
            blocks: &[(half::f16, Vec<u8>)],
        ) {
            let mut bytes = Vec::new();
            for (scale, packed) in blocks {
                assert_eq!(packed.len(), QUANTIZATION_BLOCK_SIZE / 2);
                bytes.extend_from_slice(&scale.to_le_bytes());
                bytes.extend_from_slice(packed);
            }
            self.tensors
                .push((name.to_string(), shape.to_vec(), GGML_TYPE_Q4_0, bytes));
        }

        /// Add a Q6_K tensor from raw block bytes (210 bytes per 256-element super-block).
        pub fn add_tensor_q6k_raw(&mut self, name: &str, shape: &[usize], raw_blocks: &[u8]) {
            let numel: usize = shape.iter().product();
            let num_blocks = numel / Q6_K_BLOCK_ELEMENTS;
            assert_eq!(
                raw_blocks.len(),
                num_blocks * Q6_K_BLOCK_SIZE_BYTES,
                "Q6_K raw data size mismatch"
            );
            self.tensors.push((
                name.to_string(),
                shape.to_vec(),
                GGML_TYPE_Q6_K,
                raw_blocks.to_vec(),
            ));
        }

        /// Build the GGUF file into a byte vector and write to a temp file.
        pub fn build_to_file(&self, path: &std::path::Path) {
            let mut buf = Vec::new();

            // Header
            buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
            buf.extend_from_slice(&GGUF_VERSION_3.to_le_bytes());
            buf.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            buf.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

            // Metadata KV pairs
            for (key, type_id, value_bytes) in &self.metadata {
                // key string
                buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
                buf.extend_from_slice(key.as_bytes());
                // value type
                buf.extend_from_slice(&type_id.to_le_bytes());
                // value data
                buf.extend_from_slice(value_bytes);
            }

            // Tensor descriptors — compute offsets
            let header_end_pos = self.compute_header_size(&buf, self.tensors.len());
            let _data_start = (header_end_pos + 31) & !31; // align to 32

            let mut current_offset: u64 = 0;
            let mut tensor_data_blobs = Vec::new();

            for (name, shape, ggml_type, data) in &self.tensors {
                // Name string
                buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
                buf.extend_from_slice(name.as_bytes());
                // n_dims
                buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
                // dims — reverse to GGUF column-major order
                for &d in shape.iter().rev() {
                    buf.extend_from_slice(&(d as u64).to_le_bytes());
                }
                // type
                buf.extend_from_slice(&ggml_type.to_le_bytes());
                // offset
                buf.extend_from_slice(&current_offset.to_le_bytes());

                let aligned_len = (data.len() + 31) & !31;
                current_offset += aligned_len as u64;
                tensor_data_blobs.push(data.clone());
            }

            // Pad to data_start alignment
            let current_len = buf.len();
            let aligned_start = (current_len + 31) & !31;
            buf.resize(aligned_start, 0);

            // Write tensor data
            for data in &tensor_data_blobs {
                let start = buf.len();
                buf.extend_from_slice(data);
                // Pad each tensor's data to 32-byte alignment
                let padded = (buf.len() - start + 31) & !31;
                buf.resize(start + padded, 0);
            }

            std::fs::write(path, &buf).expect("Failed to write test GGUF file");
        }

        fn compute_header_size(&self, _buf: &[u8], _tensor_count: usize) -> usize {
            // Rough estimate — the real offset is computed after writing descriptors
            // This is only used for initial alignment estimation; the actual position
            // is determined by cursor position during parsing.
            4096 // generous overestimate; alignment padding fills the gap
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_helpers::GgufBuilder;
    use super::*;
    use crate::cuda::CudaContext;
    use infernum::tensor::Tensor;

    fn test_gguf_path() -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        dir.join(format!("infernum_test_{}.gguf", std::process::id()))
    }

    #[test]
    fn test_gguf_parse_metadata() {
        let path = test_gguf_path();
        let mut builder = GgufBuilder::new();
        builder.add_metadata_u32("general.architecture", 1);
        builder.add_metadata_string("general.name", "test-model");
        builder.add_metadata_f32("general.rope_theta", 10000.0);
        builder.add_tensor_f32("dummy", &[2], &[1.0, 2.0]);
        builder.build_to_file(&path);

        let loader = GgufLoader::from_file(&path).unwrap();

        assert_eq!(
            loader
                .get_metadata("general.architecture")
                .unwrap()
                .as_usize(),
            Some(1)
        );
        assert_eq!(
            loader.get_metadata("general.name").unwrap().as_str(),
            Some("test-model")
        );
        assert!(
            (loader
                .get_metadata("general.rope_theta")
                .unwrap()
                .as_f32()
                .unwrap()
                - 10000.0)
                .abs()
                < 1e-3
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_load_f32_tensor() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let path = test_gguf_path();

        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut builder = GgufBuilder::new();
        builder.add_tensor_f32("test_weight", &[2, 3], &data);
        builder.build_to_file(&path);

        let loader = GgufLoader::from_file(&path).unwrap();
        let tensor = loader.load_f32(&ctx, "test_weight").unwrap();

        assert_eq!(tensor.shape(), &[2, 3]);
        let result = tensor.to_vec::<f32>().unwrap();
        assert_eq!(result, data);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_load_q8_tensor() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let path = test_gguf_path();

        // One block of 32 values, all quantized to 1 with scale 0.5
        // Dequantized value = 1 * 0.5 = 0.5
        let scale = half::f16::from_f32(0.5);
        let quants = vec![1_i8; QUANTIZATION_BLOCK_SIZE];

        let mut builder = GgufBuilder::new();
        builder.add_tensor_q8("q8_weight", &[1, 32], &[(scale, quants)]);
        builder.build_to_file(&path);

        let loader = GgufLoader::from_file(&path).unwrap();

        assert_eq!(loader.get_dtype("q8_weight").unwrap(), DType::Q8_0);

        let qt = loader.load_quantized(&ctx, "q8_weight").unwrap();
        assert_eq!(qt.shape(), &[1, 32]);
        assert_eq!(qt.dtype(), DType::Q8_0);
        assert_eq!(qt.numel(), 32);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_tensor_names() {
        let path = test_gguf_path();

        let mut builder = GgufBuilder::new();
        builder.add_tensor_f32("weight_a", &[2], &[1.0, 2.0]);
        builder.add_tensor_f32("weight_b", &[3], &[3.0, 4.0, 5.0]);
        builder.build_to_file(&path);

        let loader = GgufLoader::from_file(&path).unwrap();

        let mut names = loader.tensor_names();
        names.sort();
        assert_eq!(names, vec!["weight_a", "weight_b"]);
        assert!(loader.contains("weight_a"));
        assert!(!loader.contains("nonexistent"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_weight_not_found() {
        let path = test_gguf_path();

        let mut builder = GgufBuilder::new();
        builder.add_tensor_f32("dummy", &[1], &[0.0]);
        builder.build_to_file(&path);

        let loader = GgufLoader::from_file(&path).unwrap();
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let result = loader.load_f32(&ctx, "nonexistent");
        assert!(result.is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_load_f32_from_q8() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let path = test_gguf_path();

        // Block: scale=2.0, quants=[1,2,3,...,32]
        // Expected dequantized: [2.0, 4.0, 6.0, ..., 64.0]
        let scale = half::f16::from_f32(2.0);
        let quants: Vec<i8> = (1..=QUANTIZATION_BLOCK_SIZE as i8).collect();
        let expected: Vec<f32> = quants.iter().map(|&q| f32::from(q) * 2.0).collect();

        let mut builder = GgufBuilder::new();
        builder.add_tensor_q8("embed", &[1, 32], &[(scale, quants)]);
        builder.build_to_file(&path);

        let loader = GgufLoader::from_file(&path).unwrap();
        let tensor = loader.load_f32(&ctx, "embed").unwrap();

        assert_eq!(tensor.shape(), &[1, 32]);
        let result = tensor.to_vec::<f32>().unwrap();
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 0.01, "got {got}, want {want}");
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_load_f32_from_q4() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let path = test_gguf_path();

        // Block: scale=1.0, all nibbles = 8+3 = 11 (lo) and 8-2 = 6 (hi)
        // Packed byte: (6 << 4) | 11 = 0x6B
        // GGML packing: byte[j] has element j in low nibble, element j+16 in high nibble
        // Dequantized: elements 0-15 → (11-8)*1.0 = 3.0, elements 16-31 → (6-8)*1.0 = -2.0
        let scale = half::f16::from_f32(1.0);
        let packed = vec![0x6Bu8; QUANTIZATION_BLOCK_SIZE / 2];
        let mut expected = Vec::with_capacity(QUANTIZATION_BLOCK_SIZE);
        // First half (elements 0-15): low nibbles → 3.0
        for _ in 0..QUANTIZATION_BLOCK_SIZE / 2 {
            expected.push(3.0_f32);
        }
        // Second half (elements 16-31): high nibbles → -2.0
        for _ in 0..QUANTIZATION_BLOCK_SIZE / 2 {
            expected.push(-2.0_f32);
        }

        let mut builder = GgufBuilder::new();
        builder.add_tensor_q4("embed", &[1, 32], &[(scale, packed)]);
        builder.build_to_file(&path);

        let loader = GgufLoader::from_file(&path).unwrap();
        let tensor = loader.load_f32(&ctx, "embed").unwrap();

        assert_eq!(tensor.shape(), &[1, 32]);
        let result = tensor.to_vec::<f32>().unwrap();
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 0.01, "got {got}, want {want}");
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_load_f32_from_q6k() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let path = test_gguf_path();

        // Build a single Q6_K super-block (256 elements, 210 bytes)
        // Layout: ql[128] | qh[64] | scales[16] | d(f16)
        //
        // We set all quantized values to the same 6-bit value and verify dequantization.
        // Choose: ql nibble = 5, qh bits = 1 → combined = 5 | (1 << 4) = 21
        // Signed: 21 - 32 = -11
        // Scale for all sub-blocks: 2
        // Super-block d: 0.5
        // Expected dequantized value: 0.5 * 2.0 * (-11.0) = -11.0

        let mut block = vec![0u8; Q6_K_BLOCK_SIZE_BYTES];

        // ql[128]: each byte packs two 4-bit values.
        // For even indices: low nibble = 5; for odd indices: high nibble = 5.
        // So each byte = (5 << 4) | 5 = 0x55
        for b in &mut block[..128] {
            *b = 0x55;
        }

        // qh[64]: each byte packs four 2-bit values.
        // We want qh bits = 1 for each element.
        // Four 2-bit values of 1 packed: 0b01_01_01_01 = 0x55
        for b in &mut block[128..128 + 64] {
            *b = 0x55;
        }

        // scales[16]: i8 value of 2 for each sub-block
        for b in &mut block[128 + 64..128 + 64 + 16] {
            *b = 2_i8 as u8;
        }

        // d: f16 value of 0.5
        let d_bytes = half::f16::from_f32(0.5).to_le_bytes();
        block[128 + 64 + 16] = d_bytes[0];
        block[128 + 64 + 17] = d_bytes[1];

        let mut builder = GgufBuilder::new();
        builder.add_tensor_q6k_raw("q6k_weight", &[1, Q6_K_BLOCK_ELEMENTS], &block);
        builder.build_to_file(&path);

        let loader = GgufLoader::from_file(&path).unwrap();

        assert_eq!(loader.get_dtype("q6k_weight").unwrap(), DType::Q6_K);

        let tensor = loader.load_f32(&ctx, "q6k_weight").unwrap();
        assert_eq!(tensor.shape(), &[1, Q6_K_BLOCK_ELEMENTS]);

        let result = tensor.to_vec::<f32>().unwrap();
        assert_eq!(result.len(), Q6_K_BLOCK_ELEMENTS);

        let expected = -11.0_f32; // d=0.5, scale=2, q=(21-32)=-11 → 0.5*2*(-11) = -11
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected).abs() < 0.1,
                "Element {i}: got {val}, expected {expected}"
            );
        }

        std::fs::remove_file(&path).ok();
    }

    /// Test that GGUF Q4_0 load_quantized + quantized_matmul matches load_f32 + regular matmul.
    #[test]
    fn test_gguf_q4_quantized_matmul_matches_dequantized() {
        use crate::cuda::ops::{matmul, quantized_matmul};

        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let path = test_gguf_path();

        // Create a Q4_0 weight with varying values using GGML packing
        // Weight shape: (2, 32) = 2 output features, 32 input features
        // That's 2 blocks total, 1 block per row
        let scale1 = half::f16::from_f32(1.0);
        let scale2 = half::f16::from_f32(0.5);

        // Block 1: elements 0-15 = 1.0, elements 16-31 = 2.0
        // Q values (unsigned): 1.0 → round(1.0/1.0) + 8 = 9, 2.0 → round(2.0/1.0) + 8 = 10
        // Packed: byte[j] = (10 << 4) | 9 = 0xA9
        let packed1 = vec![0xA9u8; 16];

        // Block 2: elements 0-15 = 0.5, elements 16-31 = -0.5
        // Q values: 0.5/0.5 + 8 = 9, -0.5/0.5 + 8 = 7
        // Packed: byte[j] = (7 << 4) | 9 = 0x79
        let packed2 = vec![0x79u8; 16];

        let mut builder = GgufBuilder::new();
        builder.add_tensor_q4("weight", &[2, 32], &[(scale1, packed1), (scale2, packed2)]);
        builder.build_to_file(&path);

        let loader = GgufLoader::from_file(&path).unwrap();

        // Load both ways
        let weight_f32 = loader.load_f32(&ctx, "weight").unwrap();
        let weight_q4 = loader.load_quantized(&ctx, "weight").unwrap();

        // Create input: all 1.0s
        let input_data = vec![1.0_f32; 32];
        let input = CudaTensor::from_slice(&ctx, &[1, 32], &input_data).unwrap();

        // Compute matmul both ways
        // For f32: need to transpose weight first
        let weight_f32_t = crate::cuda::ops::transpose_2d(&weight_f32).unwrap();
        let result_f32 = matmul(&input, &weight_f32_t).unwrap();
        let result_q4 = quantized_matmul(&input, &weight_q4).unwrap();

        let out_f32 = result_f32.to_vec::<f32>().unwrap();
        let out_q4 = result_q4.to_vec::<f32>().unwrap();

        // Check they match within quantization error
        for (i, (&f32_val, &q4_val)) in out_f32.iter().zip(out_q4.iter()).enumerate() {
            let rel_err = if f32_val.abs() > 1e-6 {
                (f32_val - q4_val).abs() / f32_val.abs()
            } else {
                (f32_val - q4_val).abs()
            };
            assert!(
                rel_err < 0.1,
                "Mismatch at output[{i}]: f32={f32_val}, q4={q4_val}, rel_err={rel_err}"
            );
        }

        std::fs::remove_file(&path).ok();
    }
}
