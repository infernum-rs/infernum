//! Llama model configuration

use serde::Deserialize;
use std::path::Path;

use infernum::Result;

/// Configuration for Llama models
///
/// Parsed from the model's `config.json` file
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    /// Vocabulary size
    pub vocab_size: usize,

    /// Hidden dimension size
    pub hidden_size: usize,

    /// Intermediate size for MLP (FFN)
    pub intermediate_size: usize,

    /// Number of transformer layers
    pub num_hidden_layers: usize,

    /// Number of attention heads
    pub num_attention_heads: usize,

    /// Number of key-value heads (for GQA, defaults to `num_attention_heads`)
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Maximum sequence length
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// RMS norm epsilon
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    /// Rotary position embedding base frequency
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    /// Tie word embeddings with `lm_head`
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Beginning of sequence token ID
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: u32,

    /// End of sequence token ID
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,
}

fn default_max_position_embeddings() -> usize {
    2048
}

fn default_rms_norm_eps() -> f32 {
    1e-5
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_bos_token_id() -> u32 {
    1
}

fn default_eos_token_id() -> u32 {
    2
}

impl LlamaConfig {
    /// Load configuration from a JSON file
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or parsed
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Get the number of key-value heads (for grouped-query attention)
    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get the head dimension
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get the number of heads per key-value head (for GQA)
    #[must_use]
    pub fn num_heads_per_kv(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 22,
            "num_attention_heads": 32
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 22);
        assert_eq!(config.num_kv_heads(), 32); // Defaults to num_attention_heads
        assert_eq!(config.head_dim(), 64);
        assert_eq!(config.rms_norm_eps, 1e-5);
    }

    #[test]
    fn test_config_with_gqa() {
        let json = r#"{
            "vocab_size": 128256,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.num_kv_heads(), 8);
        assert_eq!(config.num_heads_per_kv(), 4);
    }
}
