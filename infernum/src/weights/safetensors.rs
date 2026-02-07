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

use crate::cuda::{CudaContext, CudaTensor};
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
        };

        CudaTensor::from_slice(ctx, &meta.shape, &f32_data)
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
    }
}
