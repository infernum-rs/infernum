//! Llama model configuration

use std::path::Path;

use serde::Deserialize;

use infernum::Result;

/// Quantization configuration parsed from `config.json`
///
/// Present in GPTQ and AWQ quantized models under the `quantization_config` key.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization method: `"gptq"` or `"awq"`
    pub quant_method: String,

    /// Number of bits per weight (typically 4)
    #[serde(default = "default_quant_bits")]
    pub bits: u32,

    /// Number of elements per quantization group (typically 128)
    #[serde(default = "default_group_size")]
    pub group_size: usize,
}

fn default_quant_bits() -> u32 {
    4
}

fn default_group_size() -> usize {
    128
}

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

    /// End of sequence token ID (first value when the config specifies an array)
    #[serde(
        default = "default_eos_token_id",
        deserialize_with = "deserialize_single_or_first"
    )]
    pub eos_token_id: u32,

    /// Quantization configuration (present for GPTQ/AWQ models)
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,

    /// Number of experts per `MoE` layer (e.g. 8 for Mixtral). `None` = dense model.
    #[serde(default)]
    pub num_local_experts: Option<usize>,

    /// Number of experts activated per token (e.g. 2 for Mixtral)
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
}

/// Deserialize a field that may be a single `u32` or an array of `u32`.
/// When an array is provided, the first element is used.
fn deserialize_single_or_first<'de, D>(deserializer: D) -> std::result::Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum SingleOrVec {
        Single(u32),
        Vec(Vec<u32>),
    }

    match SingleOrVec::deserialize(deserializer)? {
        SingleOrVec::Single(v) => Ok(v),
        SingleOrVec::Vec(v) => v
            .first()
            .copied()
            .ok_or_else(|| serde::de::Error::custom("eos_token_id array is empty")),
    }
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

    /// Returns true if this model uses Mixture-of-Experts layers
    #[must_use]
    pub fn is_moe(&self) -> bool {
        self.num_local_experts.is_some_and(|n| n > 1)
    }

    /// Build a `LlamaConfig` from GGUF metadata key-value pairs.
    ///
    /// GGUF metadata uses keys like `llama.embedding_length`, `llama.block_count`, etc.
    ///
    /// # Errors
    /// Returns an error if required metadata keys are missing.
    #[cfg(feature = "cuda")]
    pub fn from_gguf_metadata(
        metadata: &std::collections::HashMap<String, infernum::weights::GgufValue>,
    ) -> Result<Self> {
        use infernum::weights::GgufValue;

        let get_usize = |key: &str| -> Result<usize> {
            metadata
                .get(key)
                .and_then(GgufValue::as_usize)
                .ok_or_else(|| {
                    infernum::Error::InvalidShape(format!("Missing GGUF metadata: {key}"))
                })
        };

        let get_f32 = |key: &str, default: f32| -> f32 {
            metadata
                .get(key)
                .and_then(GgufValue::as_f32)
                .unwrap_or(default)
        };

        let hidden_size = get_usize("llama.embedding_length")?;
        let num_attention_heads = get_usize("llama.attention.head_count")?;

        let num_key_value_heads = metadata
            .get("llama.attention.head_count_kv")
            .and_then(GgufValue::as_usize);

        Ok(Self {
            vocab_size: get_usize("llama.vocab_size")
                .or_else(|_| get_usize("tokenizer.ggml.tokens_count"))
                .unwrap_or(32000),
            hidden_size,
            intermediate_size: get_usize("llama.feed_forward_length")?,
            num_hidden_layers: get_usize("llama.block_count")?,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings: get_usize("llama.context_length").unwrap_or(2048),
            rms_norm_eps: get_f32("llama.attention.layer_norm_rms_epsilon", 1e-5),
            rope_theta: get_f32("llama.rope.freq_base", 10000.0),
            tie_word_embeddings: metadata
                .get("llama.tie_word_embeddings")
                .and_then(GgufValue::as_bool)
                .unwrap_or(false),
            #[allow(clippy::cast_possible_truncation)] // token IDs always fit in u32
            bos_token_id: metadata
                .get("tokenizer.ggml.bos_token_id")
                .and_then(GgufValue::as_usize)
                .unwrap_or(1) as u32,
            #[allow(clippy::cast_possible_truncation)] // token IDs always fit in u32
            eos_token_id: metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(GgufValue::as_usize)
                .unwrap_or(2) as u32,
            quantization_config: None,
            num_local_experts: metadata
                .get("llama.expert_count")
                .and_then(GgufValue::as_usize),
            num_experts_per_tok: metadata
                .get("llama.expert_used_count")
                .and_then(GgufValue::as_usize),
        })
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

    #[test]
    fn test_config_all_defaults() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.max_position_embeddings, 2048);
        assert_eq!(config.rms_norm_eps, 1e-5);
        assert_eq!(config.rope_theta, 10000.0);
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.bos_token_id, 1);
        assert_eq!(config.eos_token_id, 2);
    }

    #[test]
    fn test_config_tie_word_embeddings() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 22,
            "num_attention_heads": 32,
            "tie_word_embeddings": true
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn test_config_head_dim() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn test_config_from_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("infernum_test_config.json");

        let json = r#"{
            "vocab_size": 128256,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rope_theta": 500000.0
        }"#;

        std::fs::write(&path, json).unwrap();

        let config = LlamaConfig::from_file(&path).unwrap();
        assert_eq!(config.vocab_size, 128256);
        assert_eq!(config.rope_theta, 500000.0);
        assert_eq!(config.num_kv_heads(), 8);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_config_gptq_quantization() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "quantization_config": {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128
            }
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        let qc = config.quantization_config.unwrap();
        assert_eq!(qc.quant_method, "gptq");
        assert_eq!(qc.bits, 4);
        assert_eq!(qc.group_size, 128);
    }

    #[test]
    fn test_config_awq_quantization() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128
            }
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        let qc = config.quantization_config.unwrap();
        assert_eq!(qc.quant_method, "awq");
    }

    #[test]
    fn test_config_no_quantization() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert!(config.quantization_config.is_none());
    }

    #[test]
    fn test_config_quantization_defaults() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "quantization_config": {
                "quant_method": "gptq"
            }
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        let qc = config.quantization_config.unwrap();
        assert_eq!(qc.bits, 4);
        assert_eq!(qc.group_size, 128);
    }

    #[test]
    fn test_config_from_file_missing() {
        let result = LlamaConfig::from_file("/nonexistent/path/config.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_config_from_invalid_json() {
        let dir = std::env::temp_dir();
        let path = dir.join("infernum_test_bad_config.json");

        std::fs::write(&path, "not json at all").unwrap();

        let result = LlamaConfig::from_file(&path);
        assert!(result.is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_config_eos_token_id_as_array() {
        let json = r#"{
            "vocab_size": 128256,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "eos_token_id": [128001, 128008, 128009]
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.eos_token_id, 128001);
    }

    #[test]
    fn test_config_eos_token_id_as_scalar() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 22,
            "num_attention_heads": 32,
            "eos_token_id": 42
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.eos_token_id, 42);
    }

    #[test]
    fn test_config_mixtral_moe() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_local_experts": 8,
            "num_experts_per_tok": 2
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_local_experts, Some(8));
        assert_eq!(config.num_experts_per_tok, Some(2));
        assert!(config.is_moe());
    }

    #[test]
    fn test_config_dense_has_no_moe_fields() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_local_experts, None);
        assert_eq!(config.num_experts_per_tok, None);
        assert!(!config.is_moe());
    }

    #[test]
    fn test_config_is_moe_requires_multiple_experts() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_local_experts": 1,
            "num_experts_per_tok": 1
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert!(!config.is_moe());
    }
}
