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

    /// Number of elements per quantization group (typically 128).
    /// A value of 0 means per-channel quantization (one group = full input dim),
    /// resolved to `in_features` at load time. JSON value `-1` is deserialized as 0.
    #[serde(
        default = "default_group_size",
        deserialize_with = "deserialize_group_size"
    )]
    pub group_size: usize,
}

fn default_quant_bits() -> u32 {
    4
}

fn default_group_size() -> usize {
    128
}

/// Deserialize `group_size`: -1 (per-channel) → 0 sentinel, positive values pass through.
#[allow(clippy::cast_possible_truncation)]
fn deserialize_group_size<'de, D>(deserializer: D) -> std::result::Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = i64::deserialize(deserializer)?;
    if value <= 0 {
        Ok(0)
    } else {
        #[allow(clippy::cast_sign_loss)]
        Ok(value as usize)
    }
}

/// Configuration for Llama models
///
/// Parsed from the model's `config.json` file
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    /// Model architecture type (e.g., `"llama"`, `"mistral"`, `"mixtral"`)
    #[serde(default = "default_model_type")]
    pub model_type: String,

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

    /// Sliding window size for local attention. When set and `use_sliding_window`
    /// is true, attention is restricted to the most recent `sliding_window` positions.
    #[serde(default)]
    pub sliding_window: Option<usize>,

    /// Whether sliding window attention is enabled. Default: false.
    #[serde(default)]
    pub use_sliding_window: bool,

    /// SWA applies only to layers `[0, max_window_layers)`. Layers at or above
    /// this index use full causal attention. When absent, all layers use SWA.
    #[serde(default)]
    pub max_window_layers: Option<usize>,
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

fn default_model_type() -> String {
    "llama".to_string()
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

/// Supported `model_type` values for the Llama model family.
const SUPPORTED_MODEL_TYPES: &[&str] = &["llama", "mistral", "mixtral"];

impl LlamaConfig {
    /// Load configuration from a JSON file
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or parsed, or if the
    /// `model_type` is not supported.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        config.validate_model_type()?;
        Ok(config)
    }

    /// Validate that the `model_type` is one we support.
    ///
    /// # Errors
    /// Returns [`infernum::Error::UnsupportedModel`] if the model type is unknown.
    pub fn validate_model_type(&self) -> Result<()> {
        if SUPPORTED_MODEL_TYPES.contains(&self.model_type.as_str()) {
            Ok(())
        } else {
            Err(infernum::Error::UnsupportedModel(format!(
                "Unsupported model_type: \"{}\". Supported: {}",
                self.model_type,
                SUPPORTED_MODEL_TYPES.join(", "),
            )))
        }
    }

    /// Returns true if this model uses Mixture-of-Experts layers
    #[must_use]
    pub fn is_moe(&self) -> bool {
        self.num_local_experts.is_some_and(|n| n > 1)
    }

    /// Returns the effective sliding window size for a given layer, or `None`
    /// if full causal attention should be used.
    #[must_use]
    pub fn effective_sliding_window(&self, layer_idx: usize) -> Option<usize> {
        if !self.use_sliding_window {
            return None;
        }
        if let Some(max_layers) = self.max_window_layers {
            if layer_idx >= max_layers {
                return None;
            }
        }
        self.sliding_window
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

        // GGUF stores model architecture in `general.architecture` (e.g. "llama", "mistral")
        let model_type = metadata
            .get("general.architecture")
            .and_then(GgufValue::as_str)
            .unwrap_or("llama")
            .to_string();

        let hidden_size = get_usize("llama.embedding_length")?;
        let num_attention_heads = get_usize("llama.attention.head_count")?;

        let num_key_value_heads = metadata
            .get("llama.attention.head_count_kv")
            .and_then(GgufValue::as_usize);

        let config = Self {
            model_type,
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
            sliding_window: metadata
                .get("llama.attention.sliding_window")
                .and_then(GgufValue::as_usize)
                .filter(|&w| w > 0),
            use_sliding_window: metadata
                .get("llama.attention.sliding_window")
                .and_then(GgufValue::as_usize)
                .is_some_and(|w| w > 0),
            max_window_layers: None,
        };
        config.validate_model_type()?;
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
    fn test_config_gptq_per_channel() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "quantization_config": {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": -1
            }
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        let qc = config.quantization_config.unwrap();
        assert_eq!(
            qc.group_size, 0,
            "group_size=-1 should deserialize as 0 (per-channel sentinel)"
        );
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

    #[test]
    fn test_config_model_type_default() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 22,
            "num_attention_heads": 32
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_type, "llama");
        assert!(config.validate_model_type().is_ok());
    }

    #[test]
    fn test_config_model_type_mistral() {
        let json = r#"{
            "model_type": "mistral",
            "vocab_size": 32768,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_type, "mistral");
        assert!(config.validate_model_type().is_ok());
    }

    #[test]
    fn test_config_model_type_mixtral() {
        let json = r#"{
            "model_type": "mixtral",
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
        assert_eq!(config.model_type, "mixtral");
        assert!(config.validate_model_type().is_ok());
    }

    #[test]
    fn test_config_model_type_unsupported() {
        let json = r#"{
            "model_type": "gpt2",
            "vocab_size": 50257,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        let err = config.validate_model_type().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("gpt2"),
            "Error should mention the unsupported type: {msg}"
        );
    }

    #[test]
    fn test_config_from_file_rejects_unsupported_model_type() {
        let dir = std::env::temp_dir();
        let path = dir.join("infernum_test_unsupported_model.json");

        let json = r#"{
            "model_type": "phi",
            "vocab_size": 32000,
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 22,
            "num_attention_heads": 32
        }"#;

        std::fs::write(&path, json).unwrap();

        let result = LlamaConfig::from_file(&path);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("phi"),
            "Error should mention the unsupported type: {msg}"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_config_sliding_window_mistral() {
        let json = r#"{
            "model_type": "mistral",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "sliding_window": 4096,
            "use_sliding_window": true,
            "max_window_layers": 28
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.sliding_window, Some(4096));
        assert!(config.use_sliding_window);
        assert_eq!(config.max_window_layers, Some(28));

        // Layers below max_window_layers use SWA
        assert_eq!(config.effective_sliding_window(0), Some(4096));
        assert_eq!(config.effective_sliding_window(27), Some(4096));

        // Layers at or above max_window_layers use full attention
        assert_eq!(config.effective_sliding_window(28), None);
        assert_eq!(config.effective_sliding_window(31), None);
    }

    #[test]
    fn test_config_sliding_window_disabled() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "sliding_window": 32768,
            "use_sliding_window": false
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.sliding_window, Some(32768));
        assert!(!config.use_sliding_window);

        // SWA disabled: all layers use full attention
        assert_eq!(config.effective_sliding_window(0), None);
    }

    #[test]
    fn test_config_sliding_window_all_layers() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "sliding_window": 4096,
            "use_sliding_window": true
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();

        // No max_window_layers: all layers use SWA
        assert_eq!(config.effective_sliding_window(0), Some(4096));
        assert_eq!(config.effective_sliding_window(31), Some(4096));
    }

    #[test]
    fn test_config_no_sliding_window_fields() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32
        }"#;

        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.sliding_window, None);
        assert!(!config.use_sliding_window);
        assert_eq!(config.max_window_layers, None);
        assert_eq!(config.effective_sliding_window(0), None);
    }
}
