//! GGUF file format loader (CPU-only)
//!
//! Parses the GGUF binary format used by llama.cpp and returns tensor data
//! as host-side buffers ([`HostTensor`], [`HostQuantizedWeight`]).
//! Backend-specific code then uploads these to device memory.
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

use crate::dtype::{DType, Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_SIZE_BYTES, QUANTIZATION_BLOCK_SIZE};
use crate::Error;
use crate::Result;

use super::host::{HostQuantizedWeight, HostTensor};

// Re-export GgufValue from the core gguf_meta module
pub use crate::GgufValue;

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
const GGML_TYPE_Q6_K: u32 = 14;
const GGML_TYPE_BF16: u32 = 30;

// ---------------------------------------------------------------------------
// Tensor descriptor (parsed from header)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct GgufTensorInfo {
    shape: Vec<usize>,
    ggml_type: u32,
    offset: u64, // relative to tensor data start
}

// ---------------------------------------------------------------------------
// GGUF Loader
// ---------------------------------------------------------------------------

/// Loads model weights from a GGUF file.
///
/// Supports F32, F16, BF16, Q8_0, Q4_0, and Q6_K tensor types. Tensors are
/// memory-mapped and loaded to host memory on demand. The resulting host
/// buffers can then be uploaded to any backend.
pub struct GgufLoader {
    mmap: Mmap,
    metadata: HashMap<String, GgufValue>,
    tensors: HashMap<String, GgufTensorInfo>,
    tensor_data_offset: usize,
}

impl GgufLoader {
    /// Open and parse a GGUF file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened, is not a valid GGUF file,
    /// or uses an unsupported version.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = Cursor::new(mmap.as_ref());

        // Parse header
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

        // Parse metadata
        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(&mut cursor)?;
            let value = read_gguf_value(&mut cursor)?;
            metadata.insert(key, value);
        }

        // Parse tensor descriptors
        let mut tensors = HashMap::with_capacity(tensor_count);
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

            tensors.insert(
                name,
                GgufTensorInfo {
                    shape,
                    ggml_type,
                    offset,
                },
            );
        }

        // Tensor data starts at current position, aligned to 32 bytes
        let pos = cursor.position() as usize;
        let tensor_data_offset = (pos + 31) & !31;

        Ok(Self {
            mmap,
            metadata,
            tensors,
            tensor_data_offset,
        })
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

    /// Load a tensor, dequantizing to f32 if necessary.
    ///
    /// Returns a [`HostTensor`] with dtype `F32` and the tensor's logical shape.
    ///
    /// # Errors
    /// Returns an error if the tensor is not found or has an unsupported type.
    pub fn load_f32(&self, name: &str) -> Result<HostTensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;

        let dtype = ggml_type_to_dtype(info.ggml_type)?;
        let data_start = self.tensor_data_offset + info.offset as usize;
        let numel: usize = info.shape.iter().product();

        let f32_data: Vec<f32> = match dtype {
            DType::F32 => {
                let byte_len = numel * 4;
                let raw = &self.mmap[data_start..data_start + byte_len];
                bytemuck::cast_slice(raw).to_vec()
            }
            DType::F16 => {
                let byte_len = numel * 2;
                let raw = &self.mmap[data_start..data_start + byte_len];
                let f16_slice: &[half::f16] = bytemuck::cast_slice(raw);
                f16_slice.iter().map(|x| x.to_f32()).collect()
            }
            DType::Q8_0 => {
                let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
                let block_bytes = dtype.block_size_in_bytes();
                let total_bytes = num_blocks * block_bytes;
                let raw = &self.mmap[data_start..data_start + total_bytes];

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
                let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
                let block_bytes = dtype.block_size_in_bytes();
                let total_bytes = num_blocks * block_bytes;
                let raw = &self.mmap[data_start..data_start + total_bytes];

                let mut out = vec![0.0; numel];
                for block_idx in 0..num_blocks {
                    let bs = block_idx * block_bytes;
                    let scale = half::f16::from_le_bytes([raw[bs], raw[bs + 1]]).to_f32();

                    let block_out_start = block_idx * QUANTIZATION_BLOCK_SIZE;

                    #[allow(clippy::cast_precision_loss)]
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
                let num_blocks = numel / Q6_K_BLOCK_ELEMENTS;
                let total_bytes = num_blocks * Q6_K_BLOCK_SIZE_BYTES;
                let raw = &self.mmap[data_start..data_start + total_bytes];

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
                        let sb = elem / 16;
                        let sb_elem = elem % 16;
                        let flat_idx = sb * 16 + sb_elem;
                        let row8 = flat_idx / 32;
                        let col32 = flat_idx % 32;

                        let ql_half = row8 / 4;
                        let ql_nibble_sel = (row8 % 4) / 2;
                        let ql_offset = (row8 % 4) % 2;
                        let ql_byte_idx = ql_half * 64 + ql_offset * 32 + col32;
                        let ql_byte = ql[ql_byte_idx];
                        let ql_val = if ql_nibble_sel == 0 {
                            u32::from(ql_byte & 0x0F)
                        } else {
                            u32::from(ql_byte >> 4)
                        };

                        let qh_half = row8 / 4;
                        let qh_shift_sel = row8 % 4;
                        let qh_byte_idx = qh_half * 32 + col32;
                        let qh_byte = qh[qh_byte_idx];
                        let qh_shift = qh_shift_sel * 2;
                        let qh_val = u32::from((qh_byte >> qh_shift) & 0x03);

                        let q = (ql_val | (qh_val << 4)) as i32 - 32;
                        let sc = f32::from(scales[sb] as i8);
                        #[allow(clippy::cast_precision_loss)]
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

        Ok(HostTensor {
            shape: info.shape.clone(),
            dtype: DType::F32,
            data: bytemuck::cast_slice(&f32_data).to_vec(),
        })
    }

    /// Load a tensor as f16 on the host.
    ///
    /// # Errors
    /// Returns an error if the tensor is not found or dtype conversion is unsupported.
    pub fn load_f16(&self, name: &str) -> Result<HostTensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;

        let dtype = ggml_type_to_dtype(info.ggml_type)?;
        let data_start = self.tensor_data_offset + info.offset as usize;
        let numel: usize = info.shape.iter().product();

        let f16_data: Vec<half::f16> = match dtype {
            DType::F16 => {
                let byte_len = numel * 2;
                let raw = &self.mmap[data_start..data_start + byte_len];
                bytemuck::cast_slice(raw).to_vec()
            }
            DType::F32 => {
                let byte_len = numel * 4;
                let raw = &self.mmap[data_start..data_start + byte_len];
                let f32_slice: &[f32] = bytemuck::cast_slice(raw);
                f32_slice.iter().map(|&x| half::f16::from_f32(x)).collect()
            }
            DType::BF16 => {
                let byte_len = numel * 2;
                let raw = &self.mmap[data_start..data_start + byte_len];
                let bf16_slice: &[half::bf16] = bytemuck::cast_slice(raw);
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

        Ok(HostTensor {
            shape: info.shape.clone(),
            dtype: DType::F16,
            data: bytemuck::cast_slice(&f16_data).to_vec(),
        })
    }

    /// Load a tensor as bf16 on the host.
    ///
    /// # Errors
    /// Returns an error if the tensor is not found or dtype conversion is unsupported.
    pub fn load_bf16(&self, name: &str) -> Result<HostTensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;

        let dtype = ggml_type_to_dtype(info.ggml_type)?;
        let data_start = self.tensor_data_offset + info.offset as usize;
        let numel: usize = info.shape.iter().product();

        let bf16_data: Vec<half::bf16> = match dtype {
            DType::BF16 => {
                let byte_len = numel * 2;
                let raw = &self.mmap[data_start..data_start + byte_len];
                bytemuck::cast_slice(raw).to_vec()
            }
            DType::F32 => {
                let byte_len = numel * 4;
                let raw = &self.mmap[data_start..data_start + byte_len];
                let f32_slice: &[f32] = bytemuck::cast_slice(raw);
                f32_slice.iter().map(|&x| half::bf16::from_f32(x)).collect()
            }
            DType::F16 => {
                let byte_len = numel * 2;
                let raw = &self.mmap[data_start..data_start + byte_len];
                let f16_slice: &[half::f16] = bytemuck::cast_slice(raw);
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

        Ok(HostTensor {
            shape: info.shape.clone(),
            dtype: DType::BF16,
            data: bytemuck::cast_slice(&bf16_data).to_vec(),
        })
    }

    /// Load a tensor as a quantized weight (Q8_0, Q4_0, Q6_K, FP8).
    ///
    /// Returns a [`HostQuantizedWeight`] with separated data and scale buffers
    /// ready for upload to a backend's quantized tensor format.
    ///
    /// # Errors
    /// Returns an error if the tensor is not found, has an unsupported type,
    /// or is not quantized.
    pub fn load_quantized(&self, name: &str) -> Result<HostQuantizedWeight> {
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

        let data_start = self.tensor_data_offset + info.offset as usize;
        let numel: usize = info.shape.iter().product();

        match dtype {
            DType::Q8_0 | DType::Q4_0 => {
                let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
                let block_bytes = dtype.block_size_in_bytes();
                let total_bytes = num_blocks * block_bytes;
                let raw = &self.mmap[data_start..data_start + total_bytes];

                // GGUF block layout: [scale_f16 (2 bytes) | quants (block_bytes - 2)]
                let quant_bytes_per_block = block_bytes - 2;
                let mut data_buf = Vec::with_capacity(num_blocks * quant_bytes_per_block);
                let mut scales_buf = Vec::with_capacity(num_blocks * 2);

                for block_idx in 0..num_blocks {
                    let block_start = block_idx * block_bytes;
                    scales_buf.extend_from_slice(&raw[block_start..block_start + 2]);
                    data_buf.extend_from_slice(&raw[block_start + 2..block_start + block_bytes]);
                }

                Ok(HostQuantizedWeight {
                    shape: info.shape.clone(),
                    dtype,
                    data: data_buf,
                    scales: scales_buf,
                    qzeros: None,
                    group_size: None,
                    weight_scale: 1.0,
                    channel_scales: None,
                })
            }
            DType::F8E4M3 => {
                let raw = &self.mmap[data_start..data_start + numel];
                Ok(HostQuantizedWeight {
                    shape: info.shape.clone(),
                    dtype,
                    data: raw.to_vec(),
                    scales: Vec::new(),
                    qzeros: None,
                    group_size: None,
                    weight_scale: 1.0,
                    channel_scales: None,
                })
            }
            DType::Q6_K => {
                let num_blocks = numel / Q6_K_BLOCK_ELEMENTS;
                let total_bytes = num_blocks * Q6_K_BLOCK_SIZE_BYTES;
                let raw = &self.mmap[data_start..data_start + total_bytes];
                // Store packed super-blocks directly — kernel reads them as-is
                Ok(HostQuantizedWeight {
                    shape: info.shape.clone(),
                    dtype,
                    data: raw.to_vec(),
                    scales: Vec::new(),
                    qzeros: None,
                    group_size: None,
                    weight_scale: 1.0,
                    channel_scales: None,
                })
            }
            other => Err(Error::UnsupportedDtype(format!(
                "Tensor '{name}' is {other}, which is not supported by load_quantized 
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
    /// # Errors
    /// Returns an error if the tensor is not found or has an unsupported type.
    pub fn load_quantized_unpermute(
        &self,
        name: &str,
        n_head: usize,
    ) -> Result<HostQuantizedWeight> {
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

        let data_start = self.tensor_data_offset + info.offset as usize;
        let numel: usize = info.shape.iter().product();

        let n_rows = info.shape[0];
        let n_cols = info.shape[1];
        let head_dim = n_rows / n_head;
        let half_dim = head_dim / 2;

        match dtype {
            DType::Q8_0 | DType::Q4_0 => {
                let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
                let block_bytes = dtype.block_size_in_bytes();
                let total_bytes = num_blocks * block_bytes;
                let raw = &self.mmap[data_start..data_start + total_bytes];

                let blocks_per_row = n_cols / QUANTIZATION_BLOCK_SIZE;
                let row_bytes = blocks_per_row * block_bytes;

                // Un-permute rows
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

                // Split into separate data and scales arrays
                let quant_bytes_per_block = block_bytes - 2;
                let mut data_buf = Vec::with_capacity(num_blocks * quant_bytes_per_block);
                let mut scales_buf = Vec::with_capacity(num_blocks * 2);

                for block_idx in 0..num_blocks {
                    let block_start = block_idx * block_bytes;
                    scales_buf.extend_from_slice(&unpermuted[block_start..block_start + 2]);
                    data_buf
                        .extend_from_slice(&unpermuted[block_start + 2..block_start + block_bytes]);
                }

                Ok(HostQuantizedWeight {
                    shape: info.shape.clone(),
                    dtype,
                    data: data_buf,
                    scales: scales_buf,
                    qzeros: None,
                    group_size: None,
                    weight_scale: 1.0,
                    channel_scales: None,
                })
            }
            DType::Q6_K => {
                let num_blocks = numel / Q6_K_BLOCK_ELEMENTS;
                let total_bytes = num_blocks * Q6_K_BLOCK_SIZE_BYTES;
                let raw = &self.mmap[data_start..data_start + total_bytes];

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

                Ok(HostQuantizedWeight {
                    shape: info.shape.clone(),
                    dtype,
                    data: unpermuted,
                    scales: Vec::new(),
                    qzeros: None,
                    group_size: None,
                    weight_scale: 1.0,
                    channel_scales: None,
                })
            }
            other => Err(Error::UnsupportedDtype(format!(
                "Tensor '{name}' is {other}, which is not supported by load_quantized_unpermute"
            ))),
        }
    }

    /// Get the dtype of a tensor in the file.
    ///
    /// # Errors
    /// Returns an error if the tensor is not found or has an unsupported GGML type.
    pub fn get_dtype(&self, name: &str) -> Result<DType> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;
        ggml_type_to_dtype(info.ggml_type)
    }

    /// Get the shape of a tensor without loading it.
    ///
    /// # Errors
    /// Returns an error if the tensor is not found.
    pub fn get_shape(&self, name: &str) -> Result<Vec<usize>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::WeightNotFound(name.to_string()))?;
        Ok(info.shape.clone())
    }

    /// Check if a tensor exists.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// List all tensor names.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// FormatLoader implementation
// ---------------------------------------------------------------------------

impl super::format::FormatLoader for GgufLoader {
    fn load_f32(&self, name: &str) -> Result<HostTensor> {
        Self::load_f32(self, name)
    }

    fn load_f16(&self, name: &str) -> Result<HostTensor> {
        Self::load_f16(self, name)
    }

    fn load_bf16(&self, name: &str) -> Result<HostTensor> {
        Self::load_bf16(self, name)
    }

    fn load_quantized(&self, name: &str) -> Result<HostQuantizedWeight> {
        Self::load_quantized(self, name)
    }

    fn load_quantized_unpermute(&self, name: &str, n_head: usize) -> Result<HostQuantizedWeight> {
        Self::load_quantized_unpermute(self, name, n_head)
    }

    fn get_shape(&self, name: &str) -> Result<Vec<usize>> {
        Self::get_shape(self, name)
    }

    fn get_dtype(&self, name: &str) -> Result<DType> {
        Self::get_dtype(self, name)
    }

    fn contains(&self, name: &str) -> bool {
        Self::contains(self, name)
    }

    fn tensor_names(&self) -> Vec<String> {
        Self::tensor_names(self)
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

fn ggml_type_to_dtype(ggml_type: u32) -> Result<DType> {
    match ggml_type {
        GGML_TYPE_F32 => Ok(DType::F32),
        GGML_TYPE_F16 => Ok(DType::F16),
        GGML_TYPE_BF16 => Ok(DType::BF16),
        GGML_TYPE_Q8_0 => Ok(DType::Q8_0),
        GGML_TYPE_Q4_0 => Ok(DType::Q4_0),
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
pub(crate) mod test_helpers {
    use super::*;

    /// Builder for creating minimal GGUF v3 files in memory for testing.
    pub struct GgufBuilder {
        metadata: Vec<(String, u32, Vec<u8>)>,
        tensors: Vec<(String, Vec<usize>, u32, Vec<u8>)>,
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

        pub fn build_to_file(&self, path: &std::path::Path) {
            let mut buf = Vec::new();

            // Header
            buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
            buf.extend_from_slice(&GGUF_VERSION_3.to_le_bytes());
            buf.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            buf.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

            // Metadata KV pairs
            for (key, type_id, value_bytes) in &self.metadata {
                buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
                buf.extend_from_slice(key.as_bytes());
                buf.extend_from_slice(&type_id.to_le_bytes());
                buf.extend_from_slice(value_bytes);
            }

            // Tensor descriptors
            let mut current_offset: u64 = 0;
            let mut tensor_data_blobs = Vec::new();

            for (name, shape, ggml_type, data) in &self.tensors {
                buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
                buf.extend_from_slice(name.as_bytes());
                buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
                for &d in shape.iter().rev() {
                    buf.extend_from_slice(&(d as u64).to_le_bytes());
                }
                buf.extend_from_slice(&ggml_type.to_le_bytes());
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
                let padded = (buf.len() - start + 31) & !31;
                buf.resize(start + padded, 0);
            }

            std::fs::write(path, &buf).expect("Failed to write test GGUF file");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_helpers::GgufBuilder;
    use super::*;

    fn test_gguf_path() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        dir.join(format!("infernum_test_{}_{id}.gguf", std::process::id()))
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
        let path = test_gguf_path();

        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut builder = GgufBuilder::new();
        builder.add_tensor_f32("test_weight", &[2, 3], &data);
        builder.build_to_file(&path);

        let loader = GgufLoader::from_file(&path).unwrap();
        let tensor = loader.load_f32("test_weight").unwrap();

        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.as_f32_slice(), &data);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_load_q8_tensor() {
        let path = test_gguf_path();

        let scale = half::f16::from_f32(0.5);
        let quants = vec![1_i8; QUANTIZATION_BLOCK_SIZE];

        let mut builder = GgufBuilder::new();
        builder.add_tensor_q8("q8_weight", &[1, 32], &[(scale, quants)]);
        builder.build_to_file(&path);

        let loader = GgufLoader::from_file(&path).unwrap();

        assert_eq!(loader.get_dtype("q8_weight").unwrap(), DType::Q8_0);

        let qt = loader.load_quantized("q8_weight").unwrap();
        assert_eq!(qt.shape, vec![1, 32]);
        assert_eq!(qt.dtype, DType::Q8_0);

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

        let result = loader.load_f32("nonexistent");
        assert!(result.is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_load_f32_from_q8() {
        let path = test_gguf_path();

        let scale = half::f16::from_f32(2.0);
        let quants: Vec<i8> = (1..=QUANTIZATION_BLOCK_SIZE as i8).collect();
        let expected: Vec<f32> = quants.iter().map(|&q| f32::from(q) * 2.0).collect();

        let mut builder = GgufBuilder::new();
        builder.add_tensor_q8("embed", &[1, 32], &[(scale, quants)]);
        builder.build_to_file(&path);

        let loader = GgufLoader::from_file(&path).unwrap();
        let tensor = loader.load_f32("embed").unwrap();

        assert_eq!(tensor.shape, vec![1, 32]);
        let result = tensor.as_f32_slice();
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 0.01, "got {got}, want {want}");
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_load_f32_from_q4() {
        let path = test_gguf_path();

        let scale = half::f16::from_f32(1.0);
        let packed = vec![0x6Bu8; QUANTIZATION_BLOCK_SIZE / 2];
        let mut expected = Vec::with_capacity(QUANTIZATION_BLOCK_SIZE);
        for _ in 0..QUANTIZATION_BLOCK_SIZE / 2 {
            expected.push(3.0_f32);
        }
        for _ in 0..QUANTIZATION_BLOCK_SIZE / 2 {
            expected.push(-2.0_f32);
        }

        let mut builder = GgufBuilder::new();
        builder.add_tensor_q4("embed", &[1, 32], &[(scale, packed)]);
        builder.build_to_file(&path);

        let loader = GgufLoader::from_file(&path).unwrap();
        let tensor = loader.load_f32("embed").unwrap();

        assert_eq!(tensor.shape, vec![1, 32]);
        let result = tensor.as_f32_slice();
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 0.01, "got {got}, want {want}");
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_load_f32_from_q6k() {
        let path = test_gguf_path();

        let mut block = vec![0u8; Q6_K_BLOCK_SIZE_BYTES];

        for b in &mut block[..128] {
            *b = 0x55;
        }
        for b in &mut block[128..128 + 64] {
            *b = 0x55;
        }
        for b in &mut block[128 + 64..128 + 64 + 16] {
            *b = 2_i8 as u8;
        }
        let d_bytes = half::f16::from_f32(0.5).to_le_bytes();
        block[128 + 64 + 16] = d_bytes[0];
        block[128 + 64 + 17] = d_bytes[1];

        let mut builder = GgufBuilder::new();
        builder.add_tensor_q6k_raw("q6k_weight", &[1, Q6_K_BLOCK_ELEMENTS], &block);
        builder.build_to_file(&path);

        let loader = GgufLoader::from_file(&path).unwrap();

        assert_eq!(loader.get_dtype("q6k_weight").unwrap(), DType::Q6_K);

        let tensor = loader.load_f32("q6k_weight").unwrap();
        assert_eq!(tensor.shape, vec![1, Q6_K_BLOCK_ELEMENTS]);

        let result = tensor.as_f32_slice();
        assert_eq!(result.len(), Q6_K_BLOCK_ELEMENTS);

        let expected = -11.0_f32;
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected).abs() < 0.1,
                "Element {i}: got {val}, expected {expected}"
            );
        }

        std::fs::remove_file(&path).ok();
    }
}
