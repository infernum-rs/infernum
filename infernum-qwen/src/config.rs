//! Qwen model configuration

#![allow(clippy::doc_markdown)]

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

/// RoPE scaling configuration (YaRN for extended context)
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScalingConfig {
    /// Scaling type: `"yarn"`, `"linear"`, etc.
    pub rope_type: String,

    /// Extension factor (e.g. 4.0 means 4× the original context)
    pub factor: f32,

    /// Original context length before scaling
    pub original_max_position_embeddings: usize,
}

/// Configuration for Qwen models
///
/// Parsed from the model's `config.json` file. Covers Qwen2/2.5, Qwen3/3.5,
/// and Qwen3-MoE architectures.
#[derive(Debug, Clone, Deserialize)]
pub struct QwenConfig {
    /// Vocabulary size
    pub vocab_size: usize,

    /// Hidden dimension size
    pub hidden_size: usize,

    /// Intermediate size for dense MLP (and shared expert in MoE)
    pub intermediate_size: usize,

    /// Number of transformer layers
    pub num_hidden_layers: usize,

    /// Number of attention heads
    pub num_attention_heads: usize,

    /// Number of key-value heads (for GQA, defaults to `num_attention_heads`)
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Per-head dimension. When present, overrides `hidden_size / num_attention_heads`.
    /// Qwen3 models use this to decouple head_dim from hidden_size.
    #[serde(rename = "head_dim", default)]
    pub explicit_head_dim: Option<usize>,

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

    // --- RoPE scaling ---
    /// RoPE scaling (YaRN for extended context)
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,

    // --- MoE fields ---
    /// Number of MoE experts (Qwen3-MoE). `None` = dense model.
    #[serde(default)]
    pub num_experts: Option<usize>,

    /// Number of experts activated per token (Qwen3-MoE)
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,

    /// Intermediate size for MoE routed experts (Qwen3-MoE).
    /// When absent, routed experts use `intermediate_size`.
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,

    /// Intermediate size for the shared expert (Qwen3-MoE).
    /// When absent, no shared expert is used.
    #[serde(default)]
    pub shared_expert_intermediate_size: Option<usize>,

    /// Whether to renormalize top-K router probabilities (Qwen3-MoE).
    /// Default: `true`.
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,

    /// Controls which layers use MoE vs dense MLP (Qwen3-MoE).
    /// Layer `i` uses MoE if `i % decoder_sparse_step == 0`.
    /// Default: 1 (all layers are MoE when `num_experts` is set).
    #[serde(default = "default_decoder_sparse_step")]
    pub decoder_sparse_step: usize,
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

fn default_true() -> bool {
    true
}

fn default_decoder_sparse_step() -> usize {
    1
}

impl QwenConfig {
    /// Load configuration from a JSON file
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or parsed
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Returns `true` if this model uses Mixture-of-Experts layers
    #[must_use]
    pub fn is_moe(&self) -> bool {
        self.num_experts.is_some_and(|n| n > 1)
    }

    /// Returns `true` if the model has a shared expert (Qwen3-MoE)
    #[must_use]
    pub fn has_shared_expert(&self) -> bool {
        self.shared_expert_intermediate_size.is_some()
    }

    /// Intermediate size for MoE routed experts.
    ///
    /// Falls back to `intermediate_size` when `moe_intermediate_size` is absent.
    #[must_use]
    pub fn moe_expert_intermediate_size(&self) -> usize {
        self.moe_intermediate_size.unwrap_or(self.intermediate_size)
    }

    /// Returns `true` if `layer_idx` should be an MoE layer.
    ///
    /// Follows HuggingFace's convention: `(layer_idx + 1) % decoder_sparse_step == 0`.
    /// For `decoder_sparse_step = 1` (default), all layers are MoE.
    /// For `decoder_sparse_step = 2`, layers 1, 3, 5, ... (0-indexed) are MoE.
    #[must_use]
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.is_moe() && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
    }

    /// Get the number of key-value heads (for grouped-query attention)
    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get the head dimension.
    ///
    /// Prefers the explicit `head_dim` from config (Qwen3), falls back
    /// to `hidden_size / num_attention_heads`.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.explicit_head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
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
    fn test_qwen2_config() {
        let json = r#"{
            "model_type": "qwen2",
            "vocab_size": 151936,
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true,
            "bos_token_id": 151643,
            "eos_token_id": [151645, 151643]
        }"#;

        let config: QwenConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.hidden_size, 896);
        assert_eq!(config.num_kv_heads(), 2);
        assert_eq!(config.head_dim(), 64);
        assert_eq!(config.num_heads_per_kv(), 7);
        // No explicit head_dim → falls back to hidden_size / num_attention_heads
        assert_eq!(config.head_dim(), 64);
        assert!(config.tie_word_embeddings);
        assert_eq!(config.eos_token_id, 151645);
        assert!(!config.is_moe());
        assert!(config.rope_scaling.is_none());
    }

    #[test]
    fn test_qwen3_config() {
        let json = r#"{
            "model_type": "qwen3",
            "vocab_size": 151936,
            "hidden_size": 1024,
            "intermediate_size": 2816,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "max_position_embeddings": 40960,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true
        }"#;

        let config: QwenConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_kv_heads(), 8);
        // Qwen3 head_dim (128) overrides hidden_size/num_heads (64)
        assert_eq!(config.head_dim(), 128);
        assert!(!config.is_moe());
    }

    #[test]
    fn test_qwen3_moe_config() {
        let json = r#"{
            "model_type": "qwen3_moe",
            "vocab_size": 151936,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "num_experts": 64,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 4096,
            "norm_topk_prob": true,
            "decoder_sparse_step": 1
        }"#;

        let config: QwenConfig = serde_json::from_str(json).unwrap();
        assert!(config.is_moe());
        assert!(config.has_shared_expert());
        assert_eq!(config.num_experts, Some(64));
        assert_eq!(config.num_experts_per_tok, Some(8));
        assert_eq!(config.moe_expert_intermediate_size(), 512);
        assert_eq!(config.shared_expert_intermediate_size, Some(4096));
        assert!(config.norm_topk_prob);
        assert!(config.is_moe_layer(0));
        assert!(config.is_moe_layer(1));
    }

    #[test]
    fn test_qwen3_moe_sparse_step() {
        let json = r#"{
            "vocab_size": 151936,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_experts": 64,
            "num_experts_per_tok": 8,
            "decoder_sparse_step": 2
        }"#;

        let config: QwenConfig = serde_json::from_str(json).unwrap();
        // HF convention: (layer_idx + 1) % step == 0
        // step=2 → layers 1, 3 are MoE; layers 0, 2 are dense
        assert!(!config.is_moe_layer(0));
        assert!(config.is_moe_layer(1));
        assert!(!config.is_moe_layer(2));
        assert!(config.is_moe_layer(3));
    }

    #[test]
    fn test_moe_expert_intermediate_size_fallback() {
        let json = r#"{
            "vocab_size": 151936,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_experts": 8,
            "num_experts_per_tok": 2
        }"#;

        let config: QwenConfig = serde_json::from_str(json).unwrap();
        // No moe_intermediate_size → falls back to intermediate_size
        assert_eq!(config.moe_expert_intermediate_size(), 8192);
    }

    #[test]
    fn test_dense_is_not_moe() {
        let json = r#"{
            "vocab_size": 151936,
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14
        }"#;

        let config: QwenConfig = serde_json::from_str(json).unwrap();
        assert!(!config.is_moe());
        assert!(!config.has_shared_expert());
        assert!(!config.is_moe_layer(0));
    }

    #[test]
    fn test_rope_scaling_config() {
        let json = r#"{
            "vocab_size": 151936,
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768
            }
        }"#;

        let config: QwenConfig = serde_json::from_str(json).unwrap();
        let rs = config.rope_scaling.unwrap();
        assert_eq!(rs.rope_type, "yarn");
        assert!((rs.factor - 4.0).abs() < 1e-5);
        assert_eq!(rs.original_max_position_embeddings, 32768);
    }

    #[test]
    fn test_defaults() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32
        }"#;

        let config: QwenConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.max_position_embeddings, 2048);
        assert_eq!(config.rms_norm_eps, 1e-5);
        assert_eq!(config.rope_theta, 10000.0);
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.bos_token_id, 1);
        assert_eq!(config.eos_token_id, 2);
        assert!(config.norm_topk_prob);
        assert_eq!(config.decoder_sparse_step, 1);
    }

    #[test]
    fn test_from_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("infernum_test_qwen_config.json");

        let json = r#"{
            "vocab_size": 151936,
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "rope_theta": 1000000.0
        }"#;

        std::fs::write(&path, json).unwrap();
        let config = QwenConfig::from_file(&path).unwrap();
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.rope_theta, 1000000.0);
        assert_eq!(config.num_kv_heads(), 2);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_from_file_missing() {
        let result = QwenConfig::from_file("/nonexistent/path/config.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_quantization_config() {
        let json = r#"{
            "vocab_size": 151936,
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "quantization_config": {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128
            }
        }"#;

        let config: QwenConfig = serde_json::from_str(json).unwrap();
        let qc = config.quantization_config.unwrap();
        assert_eq!(qc.quant_method, "gptq");
        assert_eq!(qc.bits, 4);
        assert_eq!(qc.group_size, 128);
    }

    #[test]
    fn test_eos_token_id_as_array() {
        let json = r#"{
            "vocab_size": 151936,
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "eos_token_id": [151645, 151643]
        }"#;

        let config: QwenConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.eos_token_id, 151645);
    }

    #[test]
    fn test_norm_topk_prob_false() {
        let json = r#"{
            "vocab_size": 151936,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_experts": 64,
            "num_experts_per_tok": 8,
            "norm_topk_prob": false
        }"#;

        let config: QwenConfig = serde_json::from_str(json).unwrap();
        assert!(!config.norm_topk_prob);
    }
}
