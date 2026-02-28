//! SafeTensors weight loader for the CPU backend.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::SafeTensors;

use infernum::backend::{MatmulOps, SafeTensorsLoaderOps};
use infernum::dtype::DType;
use infernum::shard::{ShardConfig, ShardStrategy};
use infernum::weights::{QuantizationConfig, WeightLoader};
use infernum::Result;

use crate::tensor::CpuTensor;
use crate::CpuBackend;

/// CPU SafeTensors weight loader.
///
/// Memory-maps all `.safetensors` files in a directory and provides
/// tensor loading via the `WeightLoader` trait.
pub struct CpuSafeTensorsLoader {
    /// (mmap, SafeTensors) per file — mmap must outlive SafeTensors
    files: Vec<(Mmap, PathBuf)>,
    /// tensor_name → (file_index, dtype, shape)
    index: HashMap<String, TensorMeta>,
}

struct TensorMeta {
    file_idx: usize,
    dtype: DType,
    shape: Vec<usize>,
    data_start: usize,
    data_end: usize,
}

impl CpuSafeTensorsLoader {
    /// Create a new loader from a model directory.
    ///
    /// # Errors
    /// Returns an error if the directory doesn't contain safetensors files.
    pub fn new(model_dir: &Path) -> Result<Self> {
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

            // Parse the safetensors metadata to build index
            let st = SafeTensors::deserialize(&mmap).map_err(|e| {
                infernum::Error::SafeTensors(format!("Failed to parse {}: {e}", path.display()))
            })?;

            for (name, info) in st.tensors() {
                let dtype_str = format!("{:?}", info.dtype());
                let dtype = DType::from_safetensors(&dtype_str)
                    .unwrap_or_else(|| panic!("Unsupported safetensors dtype: {dtype_str}"));
                let shape: Vec<usize> = info.shape().to_vec();
                // Store byte range for the tensor data
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

        Ok(Self { files, index })
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

impl WeightLoader<CpuBackend> for CpuSafeTensorsLoader {
    fn load_tensor(&self, name: &str, dtype: DType) -> Result<CpuTensor> {
        let (data, src_dtype, shape) = self.load_raw(name)?;

        // Convert to target dtype (always f32 for CPU compute)
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
                // Return as-is for integer tensors
                return Ok(CpuTensor::from_raw(shape, DType::U32, data.to_vec()));
            }
            other => {
                return Err(infernum::Error::UnsupportedDtype(format!(
                    "load_tensor: unsupported dtype {other}"
                )));
            }
        };

        let _ = dtype; // CPU always works in f32
        Ok(CpuTensor::from_f32(shape, &f32_data))
    }

    fn load_linear(
        &self,
        name: &str,
        model_dtype: DType,
        quant_config: Option<&QuantizationConfig>,
    ) -> Result<<CpuBackend as MatmulOps>::LinearWeight> {
        if quant_config.is_some() {
            return Err(infernum::Error::UnsupportedDtype(
                "CPU backend does not support quantized weights".into(),
            ));
        }

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
                // FP8: load the scale tensor and dequantize
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

        let _ = model_dtype;
        // Safetensors stores weight as (out_features, in_features).
        // Transpose to (in_features, out_features) for standard matmul: A(M,K) × B(K,N).
        let mut transposed = vec![0.0f32; out_features * in_features];
        for r in 0..out_features {
            for c in 0..in_features {
                transposed[c * out_features + r] = f32_data[r * in_features + c];
            }
        }
        Ok(CpuTensor::from_f32(
            &[in_features, out_features],
            &transposed,
        ))
    }

    fn load_linear_sharded(
        &self,
        name: &str,
        model_dtype: DType,
        quant_config: Option<&QuantizationConfig>,
        shard: &ShardConfig,
        _strategy: ShardStrategy,
    ) -> Result<<CpuBackend as MatmulOps>::LinearWeight> {
        if shard.world_size == 1 {
            return self.load_linear(name, model_dtype, quant_config);
        }
        Err(infernum::Error::Other(
            "CPU backend does not support tensor parallelism".into(),
        ))
    }

    fn load_tensor_sharded(
        &self,
        name: &str,
        dtype: DType,
        shard: &ShardConfig,
        _strategy: ShardStrategy,
    ) -> Result<CpuTensor> {
        if shard.world_size == 1 {
            return self.load_tensor(name, dtype);
        }
        Err(infernum::Error::Other(
            "CPU backend does not support tensor parallelism".into(),
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

impl SafeTensorsLoaderOps for CpuBackend {
    type SafeTensorsLoader = CpuSafeTensorsLoader;

    fn safetensors_loader(_device: &(), model_dir: &Path) -> Result<CpuSafeTensorsLoader> {
        CpuSafeTensorsLoader::new(model_dir)
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

    let f_sign = if sign == 1 { -1.0f32 } else { 1.0 };
    if exp == 0 {
        // Denormalized
        let frac = f32::from(mantissa) / 8.0;
        f_sign * frac * 2.0f32.powi(-6)
    } else {
        let frac = 1.0 + f32::from(mantissa) / 8.0;
        f_sign * frac * 2.0f32.powi(i32::from(exp) - 7)
    }
}
