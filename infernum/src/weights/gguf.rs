//! GGUF file format loader
//!
//! Parses the GGUF binary format used by llama.cpp and loads tensors
//! as either `CudaTensor<f32>` (for unquantized weights) or
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
use crate::dtype::{DType, Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_SIZE_BYTES, QUANTIZATION_BLOCK_SIZE};
use crate::weights::WeightLoader;
use crate::{Error, Result};

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

// ---------------------------------------------------------------------------
// GGUF metadata value types
// ---------------------------------------------------------------------------

/// A parsed GGUF metadata value
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Try to interpret the value as a usize
    #[must_use]
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Self::U8(v) => Some(*v as usize),
            Self::I8(v) => usize::try_from(*v).ok(),
            Self::U16(v) => Some(*v as usize),
            Self::I16(v) => usize::try_from(*v).ok(),
            Self::U32(v) => Some(*v as usize),
            Self::I32(v) => usize::try_from(*v).ok(),
            Self::U64(v) => usize::try_from(*v).ok(),
            Self::I64(v) => usize::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Try to interpret the value as an f32
    #[must_use]
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Try to interpret the value as a string
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to interpret the value as a bool
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

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
/// Supports F32, F16, Q8_0, and Q4_0 tensor types. Tensors are memory-mapped
/// and loaded to the GPU on demand.
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

        let data_start = self.tensor_data_offset + info.offset as usize;
        let numel: usize = info.shape.iter().product();

        match dtype {
            DType::Q8_0 | DType::Q4_0 => {
                let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
                let block_bytes = dtype.block_size_in_bytes();
                let total_bytes = num_blocks * block_bytes;
                let raw = &self.mmap[data_start..data_start + total_bytes];

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
                let raw = &self.mmap[data_start..data_start + numel];
                QuantizedTensor::from_raw(ctx, &info.shape, dtype, raw, &[])
            }
            DType::Q6_K => {
                let num_blocks = numel / Q6_K_BLOCK_ELEMENTS;
                let total_bytes = num_blocks * Q6_K_BLOCK_SIZE_BYTES;
                let raw = &self.mmap[data_start..data_start + total_bytes];
                // Store packed super-blocks directly — kernel reads them as-is
                QuantizedTensor::from_raw(ctx, &info.shape, dtype, raw, &[])
            }
            other => Err(Error::UnsupportedDtype(format!(
                "Tensor '{name}' is {other}, which is not supported by load_quantized \
                 (use load_f32 to dequantize instead)"
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

impl WeightLoader for GgufLoader {
    fn load_f32(&self, ctx: &CudaContext, name: &str) -> Result<CudaTensor<f32>> {
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
                // Block layout: [f16 scale (2 bytes) | 32 × i8 quants (32 bytes)] = 34 bytes
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
                // Block layout: [f16 scale (2 bytes) | 16 packed bytes (32 values)] = 18 bytes
                // Each byte packs 2 values: low nibble = even index, high nibble = odd index
                // Unsigned [0,15] centered at 8 → signed [-8, 7]
                let num_blocks = numel / QUANTIZATION_BLOCK_SIZE;
                let block_bytes = dtype.block_size_in_bytes();
                let total_bytes = num_blocks * block_bytes;
                let raw = &self.mmap[data_start..data_start + total_bytes];

                let mut out = Vec::with_capacity(numel);
                for block_idx in 0..num_blocks {
                    let bs = block_idx * block_bytes;
                    let scale = half::f16::from_le_bytes([raw[bs], raw[bs + 1]]).to_f32();
                    for j in 0..QUANTIZATION_BLOCK_SIZE / 2 {
                        let byte = raw[bs + 2 + j];
                        let lo = i32::from(byte & 0x0F) - 8;
                        let hi = i32::from(byte >> 4) - 8;
                        #[allow(clippy::cast_precision_loss)] // lo/hi ∈ [-8, 7], no precision loss
                        {
                            out.push(lo as f32 * scale);
                            out.push(hi as f32 * scale);
                        }
                    }
                }
                out
            }
            DType::Q6_K => {
                // Super-block of 256 elements, 210 bytes:
                //   ql[128]   — lower 4 bits of each value (2 per byte)
                //   qh[64]    — upper 2 bits of each value (4 per byte)
                //   scales[16] — i8 sub-block scale (one per 16 elements)
                //   d (f16)   — super-block scale factor
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

                    for idx in 0..Q6_K_BLOCK_ELEMENTS {
                        // Extract lower 4 bits from ql
                        let ql_byte = ql[idx / 2];
                        let ql_val = if idx % 2 == 0 {
                            u32::from(ql_byte & 0x0F)
                        } else {
                            u32::from(ql_byte >> 4)
                        };

                        // Extract upper 2 bits from qh
                        let qh_byte = qh[idx / 4];
                        let qh_shift = (idx % 4) * 2;
                        let qh_val = u32::from((qh_byte >> qh_shift) & 0x03);

                        // Combine to 6-bit value and center: [0, 63] → [-32, 31]
                        let q = (ql_val | (qh_val << 4)) as i32 - 32;

                        let sc = f32::from(scales[idx / 16] as i8);
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
        /// Each packed byte holds two 4-bit unsigned values: low nibble = even, high = odd.
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
    use crate::tensor::Tensor;

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
        let result = tensor.to_vec().unwrap();
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
        let result = tensor.to_vec().unwrap();
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
        // Dequantized: lo → (11-8)*1.0 = 3.0, hi → (6-8)*1.0 = -2.0
        let scale = half::f16::from_f32(1.0);
        let packed = vec![0x6Bu8; QUANTIZATION_BLOCK_SIZE / 2];
        let mut expected = Vec::with_capacity(QUANTIZATION_BLOCK_SIZE);
        for _ in 0..QUANTIZATION_BLOCK_SIZE / 2 {
            expected.push(3.0_f32);
            expected.push(-2.0_f32);
        }

        let mut builder = GgufBuilder::new();
        builder.add_tensor_q4("embed", &[1, 32], &[(scale, packed)]);
        builder.build_to_file(&path);

        let loader = GgufLoader::from_file(&path).unwrap();
        let tensor = loader.load_f32(&ctx, "embed").unwrap();

        assert_eq!(tensor.shape(), &[1, 32]);
        let result = tensor.to_vec().unwrap();
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

        let result = tensor.to_vec().unwrap();
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
}
