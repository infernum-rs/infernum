//! Weight loader trait and utilities

#![allow(clippy::missing_errors_doc)]

#[cfg(feature = "cuda")]
use crate::cuda::{CudaContext, CudaTensor};
#[cfg(feature = "cuda")]
use crate::dtype::DType;
#[cfg(feature = "cuda")]
use crate::Result;

/// Trait for loading model weights from various formats
#[cfg(feature = "cuda")]
pub trait WeightLoader {
    /// Load a tensor by name
    ///
    /// # Arguments
    /// * `name` - The name/key of the tensor in the weight file
    ///
    /// # Errors
    /// Returns an error if the tensor is not found or loading fails
    fn load_f32(&self, ctx: &CudaContext, name: &str) -> Result<CudaTensor<f32>>;

    /// Get the shape of a tensor without loading it
    fn get_shape(&self, name: &str) -> Result<Vec<usize>>;

    /// Get the dtype of a tensor
    fn get_dtype(&self, name: &str) -> Result<DType>;

    /// List all tensor names in the file
    fn tensor_names(&self) -> Vec<String>;

    /// Check if a tensor exists
    fn contains(&self, name: &str) -> bool;
}

/// Mapping from model weight names to Infernum layer names
pub struct WeightNameMapper {
    prefix: String,
}

impl WeightNameMapper {
    /// Create a new mapper with a given prefix
    #[must_use]
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
        }
    }

    /// Map a Llama weight name to the internal format
    ///
    /// Input format: `model.layers.0.self_attn.q_proj.weight`
    /// Output format: `layers.0.attention.q_proj.weight`
    #[must_use]
    pub fn map_llama_name(&self, hf_name: &str) -> String {
        let name = if let Some(stripped) = hf_name.strip_prefix("model.") {
            stripped
        } else {
            hf_name
        };

        // Map attention names
        let name = name.replace("self_attn.", "attention.");

        if self.prefix.is_empty() {
            name
        } else {
            format!("{}.{}", self.prefix, name)
        }
    }

    /// Get the original weight name for a given internal name
    #[must_use]
    pub fn unmap_llama_name(&self, internal_name: &str) -> String {
        let name = if self.prefix.is_empty() {
            internal_name
        } else {
            internal_name
                .strip_prefix(&format!("{}.", self.prefix))
                .unwrap_or(internal_name)
        };

        format!("model.{}", name.replace("attention.", "self_attn."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_name_mapping() {
        let mapper = WeightNameMapper::new("");

        assert_eq!(
            mapper.map_llama_name("model.layers.0.self_attn.q_proj.weight"),
            "layers.0.attention.q_proj.weight"
        );

        assert_eq!(
            mapper.map_llama_name("model.embed_tokens.weight"),
            "embed_tokens.weight"
        );

        assert_eq!(mapper.map_llama_name("model.norm.weight"), "norm.weight");
    }
}
