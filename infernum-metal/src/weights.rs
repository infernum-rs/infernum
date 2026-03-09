//! Weight types and SafeTensors loader for Metal backend.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::SafeTensors;

use infernum::backend::SafeTensorsLoaderOps;
use infernum::dtype::DType;
use infernum::shard::{ShardConfig, ShardStrategy};
use infernum::weights::{QuantizationConfig, WeightLoader};
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;
use crate::MetalContext;

use std::sync::Arc;

// ---- Quantized weight type ----

/// Block-quantized weight for Metal inference.
///
/// Stores quantized data and per-block scales in Metal buffers for GPU
/// kernel dispatch. Layout is row-major: `out_features` rows of
/// `in_features` elements.
///
/// For `Q8_0`/`Q4_0`/`Q4_1`: `in_features / 32` blocks per row, scales
/// pre-decoded to f32 at load time.
///
/// For `Q6_K`: raw packed super-blocks (210 bytes per 256 elements) stored
/// directly in `data`. Scales and `d` are embedded in the super-block bytes;
/// the `scales` buffer is a placeholder.
#[derive(Clone)]
pub struct MetalQuantizedWeight {
    /// Logical shape: `[out_features, in_features]`
    pub shape: Vec<usize>,
    /// Quantization format (`Q8_0`, `Q4_0`, `Q4_1`, or `Q6_K`)
    pub dtype: DType,
    /// Metal context for GPU dispatch.
    pub ctx: MetalContext,
    /// Raw quantized data in a Metal buffer — int8 bytes (Q8_0), packed
    /// nibbles (Q4_0/Q4_1), or packed super-blocks (Q6_K).
    pub data: Arc<metal::Buffer>,
    /// Per-block scales as f32 in a Metal buffer. Placeholder for Q6_K.
    pub scales: Arc<metal::Buffer>,
    /// Per-block mins as f32 in a Metal buffer (Q4_1 only).
    pub mins: Option<Arc<metal::Buffer>>,
}

#[allow(clippy::cast_possible_truncation)]
impl MetalQuantizedWeight {
    /// Read data buffer contents as a byte slice (for CPU fallback).
    ///
    /// Safe because `StorageModeShared` buffers are CPU-accessible and
    /// we synchronise via command buffer completion before reads.
    #[must_use]
    pub fn data_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.contents().cast::<u8>(),
                self.data.length() as usize,
            )
        }
    }

    /// Read scales buffer contents as f32 slice (for CPU fallback).
    #[must_use]
    pub fn scales_f32(&self) -> &[f32] {
        let len = self.scales.length() as usize / std::mem::size_of::<f32>();
        if len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.scales.contents().cast::<f32>(), len) }
    }

    /// Read mins buffer contents as f32 slice (for CPU fallback).
    #[must_use]
    pub fn mins_f32(&self) -> Option<&[f32]> {
        self.mins.as_ref().map(|buf| {
            let len = buf.length() as usize / std::mem::size_of::<f32>();
            unsafe { std::slice::from_raw_parts(buf.contents().cast::<f32>(), len) }
        })
    }
}

/// Decode a buffer of f16 values stored as raw little-endian bytes into f32.
#[must_use]
pub fn decode_f16_scales(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

// ---- Linear weight type ----

/// Metal linear weight — dense or quantized.
pub enum MetalLinearWeight {
    /// Dense weight stored as f32.
    ///
    /// - `weight`: original layout `(in_features, out_features)`
    /// - `weight_t`: transposed `(out_features, in_features)` for NT matmul
    Dense {
        /// Original-layout tensor
        weight: MetalTensor,
        /// Transposed for dot-product matmul
        weight_t: MetalTensor,
    },
    /// Block-quantized weight in `(out_features, in_features)` layout.
    Quantized(MetalQuantizedWeight),
}

impl MetalLinearWeight {
    /// Create a dense weight from a `(in_features, out_features)` tensor.
    ///
    /// Pre-computes the transposed layout for efficient matmul.
    #[must_use]
    pub fn new_dense(weight: MetalTensor) -> Self {
        use infernum::tensor::Tensor;
        let shape = weight.shape();
        let (rows, cols) = (shape[0], shape[1]);
        let data = weight.as_f32_slice();

        // Transpose: (K, N) → (N, K)
        let mut t_data = vec![0.0f32; data.len()];
        for r in 0..rows {
            for c in 0..cols {
                t_data[c * rows + r] = data[r * cols + c];
            }
        }

        let weight_t = MetalTensor::from_f32(weight.context(), &[cols, rows], &t_data);

        Self::Dense { weight, weight_t }
    }
}

// ---- SafeTensors loader ----

struct TensorMeta {
    file_idx: usize,
    dtype: DType,
    shape: Vec<usize>,
    data_start: usize,
    data_end: usize,
}

/// Metal SafeTensors weight loader.
///
/// Memory-maps all `.safetensors` files in a directory and provides
/// tensor loading via the `WeightLoader` trait.
pub struct MetalSafeTensorsLoader {
    files: Vec<(Mmap, PathBuf)>,
    index: HashMap<String, TensorMeta>,
    context: MetalContext,
}

impl MetalSafeTensorsLoader {
    /// Create a new loader from a model directory.
    ///
    /// # Errors
    /// Returns an error if the directory doesn't contain safetensors files.
    pub fn new(context: MetalContext, model_dir: &Path) -> Result<Self> {
        let mut st_paths: Vec<PathBuf> = std::fs::read_dir(model_dir)?
            .filter_map(std::result::Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        st_paths.sort();

        if st_paths.is_empty() {
            return Err(infernum::Error::SafeTensors(format!(
                "No .safetensors files in {}",
                model_dir.display()
            )));
        }

        let mut files = Vec::with_capacity(st_paths.len());
        let mut index = HashMap::new();

        for (file_idx, path) in st_paths.iter().enumerate() {
            let file = std::fs::File::open(path)?;
            let mmap = unsafe { Mmap::map(&file)? };

            let st = SafeTensors::deserialize(&mmap).map_err(|e| {
                infernum::Error::SafeTensors(format!("Failed to parse {}: {e}", path.display()))
            })?;

            for (name, info) in st.tensors() {
                let dtype_str = format!("{:?}", info.dtype());
                let dtype = DType::from_safetensors(&dtype_str)
                    .unwrap_or_else(|| panic!("Unsupported safetensors dtype: {dtype_str}"));
                let shape: Vec<usize> = info.shape().to_vec();
                let data = info.data();
                let data_start = data.as_ptr() as usize - mmap.as_ptr() as usize;
                let data_end = data_start + data.len();
                index.insert(
                    name.clone(),
                    TensorMeta {
                        file_idx,
                        dtype,
                        shape,
                        data_start,
                        data_end,
                    },
                );
            }

            files.push((mmap, path.clone()));
        }

        Ok(Self {
            files,
            index,
            context,
        })
    }

    fn load_raw(&self, name: &str) -> Result<(&[u8], DType, &[usize])> {
        let meta = self
            .index
            .get(name)
            .ok_or_else(|| infernum::Error::WeightNotFound(name.to_string()))?;
        let mmap = &self.files[meta.file_idx].0;
        let data = &mmap[meta.data_start..meta.data_end];
        Ok((data, meta.dtype, &meta.shape))
    }
}

/// Convert an FP8 E4M3 byte to f32.
fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 3) & 0xF;
    let mantissa = bits & 0x7;

    if exp == 0 && mantissa == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }
    if exp == 0xF && mantissa == 0x7 {
        return f32::NAN;
    }

    let mantissa_f = 1.0 + f32::from(mantissa) / 8.0;
    let exp_val = if exp == 0 {
        2.0f32.powi(-6) * (f32::from(mantissa) / 8.0)
    } else {
        2.0f32.powi(i32::from(exp) - 7) * mantissa_f
    };

    if sign == 1 {
        -exp_val
    } else {
        exp_val
    }
}

impl WeightLoader<MetalBackend> for MetalSafeTensorsLoader {
    fn load_tensor(&self, name: &str, _dtype: DType) -> Result<MetalTensor> {
        let (data, src_dtype, shape) = self.load_raw(name)?;

        // Convert to f32 (Metal Phase 1 works in f32)
        let f32_data: Vec<f32> = match src_dtype {
            DType::F32 => bytemuck::cast_slice(data).to_vec(),
            DType::BF16 => {
                let bf16s: &[half::bf16] = bytemuck::cast_slice(data);
                bf16s.iter().map(|v| v.to_f32()).collect()
            }
            DType::F16 => {
                let f16s: &[half::f16] = bytemuck::cast_slice(data);
                f16s.iter().map(|v| v.to_f32()).collect()
            }
            DType::U32 => {
                return Ok(MetalTensor::from_raw_bytes(
                    &self.context,
                    shape,
                    DType::U32,
                    data,
                ));
            }
            other => {
                return Err(infernum::Error::UnsupportedDtype(format!(
                    "load_tensor: unsupported dtype {other}"
                )));
            }
        };

        Ok(MetalTensor::from_f32(&self.context, shape, &f32_data))
    }

    #[allow(clippy::cast_possible_truncation)]
    fn load_linear(
        &self,
        name: &str,
        _model_dtype: DType,
        _quant_config: Option<&QuantizationConfig>,
    ) -> Result<MetalLinearWeight> {
        let (data, src_dtype, shape) = self.load_raw(name)?;
        let out_features = shape[0];
        let in_features = shape[1];

        let f32_data: Vec<f32> = match src_dtype {
            DType::F32 => bytemuck::cast_slice(data).to_vec(),
            DType::BF16 => {
                let bf16s: &[half::bf16] = bytemuck::cast_slice(data);
                bf16s.iter().map(|v| v.to_f32()).collect()
            }
            DType::F16 => {
                let f16s: &[half::f16] = bytemuck::cast_slice(data);
                f16s.iter().map(|v| v.to_f32()).collect()
            }
            DType::F8E4M3 => {
                let scale_name = format!("{name}_scale");
                if let Ok((scale_data, _, _)) = self.load_raw(&scale_name) {
                    let scale: f32 = bytemuck::cast_slice::<u8, f32>(scale_data)[0];
                    data.iter().map(|&b| fp8_e4m3_to_f32(b) * scale).collect()
                } else {
                    data.iter().map(|&b| fp8_e4m3_to_f32(b)).collect()
                }
            }
            other => {
                return Err(infernum::Error::UnsupportedDtype(format!(
                    "load_linear: unsupported dtype {other}"
                )));
            }
        };

        // Transpose: (out_features, in_features) → (in_features, out_features)
        let mut transposed = vec![0.0f32; out_features * in_features];
        for r in 0..out_features {
            for c in 0..in_features {
                transposed[c * out_features + r] = f32_data[r * in_features + c];
            }
        }

        let weight =
            MetalTensor::from_f32(&self.context, &[in_features, out_features], &transposed);
        Ok(MetalLinearWeight::new_dense(weight))
    }

    fn load_linear_sharded(
        &self,
        name: &str,
        model_dtype: DType,
        quant_config: Option<&QuantizationConfig>,
        shard: &ShardConfig,
        _strategy: ShardStrategy,
    ) -> Result<MetalLinearWeight> {
        if shard.world_size == 1 {
            return self.load_linear(name, model_dtype, quant_config);
        }
        Err(infernum::Error::Other(
            "Metal backend does not support tensor parallelism".into(),
        ))
    }

    fn load_tensor_sharded(
        &self,
        name: &str,
        dtype: DType,
        shard: &ShardConfig,
        _strategy: ShardStrategy,
    ) -> Result<MetalTensor> {
        if shard.world_size == 1 {
            return self.load_tensor(name, dtype);
        }
        Err(infernum::Error::Other(
            "Metal backend does not support tensor parallelism".into(),
        ))
    }

    fn get_shape(&self, name: &str) -> Result<Vec<usize>> {
        let meta = self
            .index
            .get(name)
            .ok_or_else(|| infernum::Error::WeightNotFound(name.to_string()))?;
        Ok(meta.shape.clone())
    }

    fn get_dtype(&self, name: &str) -> Result<DType> {
        let meta = self
            .index
            .get(name)
            .ok_or_else(|| infernum::Error::WeightNotFound(name.to_string()))?;
        Ok(meta.dtype)
    }

    fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn tensor_names(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }
}

impl SafeTensorsLoaderOps for MetalBackend {
    type SafeTensorsLoader = MetalSafeTensorsLoader;

    fn safetensors_loader(
        device: &MetalContext,
        model_dir: &Path,
    ) -> Result<MetalSafeTensorsLoader> {
        MetalSafeTensorsLoader::new(device.clone(), model_dir)
    }
}
