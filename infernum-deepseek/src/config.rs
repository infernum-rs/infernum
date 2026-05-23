//! DeepSeek model configuration

#![allow(clippy::doc_markdown)]

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use infernum::{GgufValue, Result};

/// Re-exported from `infernum` core — the same struct is used by
/// `WeightLoader<B>` for generic weight loading.
pub use infernum::QuantizationConfig;

/// RoPE scaling configuration
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScalingConfig {
    /// Scaling type: `"yarn"`, `"linear"`, etc.
    #[serde(alias = "type")]
    pub rope_type: String,

    /// Extension factor
    pub factor: f32,

    /// Original context length before scaling
    pub original_max_position_embeddings: usize,

    /// YaRN mscale all-dim factor (DeepSeek V3 uses 1.0)
    #[serde(default)]
    pub mscale_all_dim: Option<f32>,
}

impl From<&RopeScalingConfig> for infernum::RopeScaling {
    fn from(rs: &RopeScalingConfig) -> Self {
        Self {
            rope_type: rs.rope_type.clone(),
            factor: rs.factor,
            original_max_position_embeddings: rs.original_max_position_embeddings,
        }
    }
}

/// Configuration for DeepSeek V3 / R1 models
///
/// Parsed from the model's `config.json` file.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekConfig {
    /// Model architecture type (`"deepseek_v3"`)
    #[serde(default = "default_model_type")]
    pub model_type: String,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Hidden dimension size
    pub hidden_size: usize,

    /// Intermediate size for dense MLP (first_k_dense_replace layers)
    pub intermediate_size: usize,

    /// Intermediate size for MoE routed experts
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,

    /// Number of transformer layers
    pub num_hidden_layers: usize,

    /// Number of attention heads
    pub num_attention_heads: usize,

    /// Number of key-value heads
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    // --- MLA dimensions ---
    /// Q compression rank (None = no Q compression, e.g., DeepSeek V2-Lite)
    #[serde(default)]
    pub q_lora_rank: Option<usize>,

    /// KV compression rank
    pub kv_lora_rank: usize,

    /// Non-RoPE portion of Q/K per head
    pub qk_nope_head_dim: usize,

    /// RoPE portion of Q/K per head
    pub qk_rope_head_dim: usize,

    /// Value head dimension
    pub v_head_dim: usize,

    // --- MoE ---
    /// Number of routed experts
    #[serde(default)]
    pub n_routed_experts: Option<usize>,

    /// Number of shared experts
    #[serde(default = "default_one")]
    pub n_shared_experts: usize,

    /// Number of experts activated per token
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,

    /// Number of expert groups for grouped top-k routing
    #[serde(default = "default_n_group")]
    pub n_group: usize,

    /// Number of top groups to select
    #[serde(default = "default_topk_group")]
    pub topk_group: usize,

    /// First N layers use dense MLP instead of MoE
    #[serde(default)]
    pub first_k_dense_replace: usize,

    /// Whether to normalize top-K routing probabilities
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,

    /// Scaling factor for routed expert weights
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,

    /// Scoring function: `"sigmoid"` (V3) or `"softmax"` (V2)
    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,

    // --- RoPE ---
    /// Base frequency for RoPE
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    /// RoPE scaling configuration
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,

    /// Maximum sequence length
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    // --- Misc ---
    /// RMS norm epsilon
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    /// Tie word embeddings with lm_head
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Beginning of sequence token ID
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: u32,

    /// End of sequence token ID
    #[serde(
        default = "default_eos_token_id",
        deserialize_with = "deserialize_single_or_first"
    )]
    pub eos_token_id: u32,

    /// Quantization configuration
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,
}

/// Deserialize a field that may be a single `u32` or an array of `u32`.
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
    "deepseek_v3".to_string()
}

fn default_one() -> usize {
    1
}

fn default_num_experts_per_tok() -> usize {
    8
}

fn default_n_group() -> usize {
    8
}

fn default_topk_group() -> usize {
    4
}

fn default_true() -> bool {
    true
}

fn default_routed_scaling_factor() -> f32 {
    2.5
}

fn default_scoring_func() -> String {
    "sigmoid".to_string()
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_max_position_embeddings() -> usize {
    4096
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_bos_token_id() -> u32 {
    0
}

fn default_eos_token_id() -> u32 {
    1
}

impl DeepSeekConfig {
    /// Load configuration from a JSON file
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or parsed
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Build a `DeepSeekConfig` from GGUF metadata key-value pairs.
    ///
    /// # Errors
    /// Returns an error if required metadata keys are missing.
    pub fn from_gguf_metadata(metadata: &HashMap<String, GgufValue>) -> Result<Self> {
        use infernum::gguf_meta::{get_f32, get_usize};

        let arch = "deepseek2";

        let num_attention_heads = get_usize(metadata, &format!("{arch}.attention.head_count"))?;
        let kv_lora_rank = get_usize(metadata, &format!("{arch}.attention.kv_lora_rank"))?;

        // q_lora_rank == 0 means no Q compression (DeepSeek V2-Lite)
        let q_lora_rank = get_usize(metadata, &format!("{arch}.attention.q_lora_rank"))
            .ok()
            .filter(|&r| r > 0);

        // rope.dimension_count is the full RoPE dim; half goes to Q/K RoPE portion
        let qk_rope_head_dim = get_usize(metadata, &format!("{arch}.rope.dimension_count"))
            .map(|d| d / 2)
            .unwrap_or(64);

        // scoring_func: 0=softmax, 1=sigmoid
        let scoring_func = metadata
            .get(&format!("{arch}.expert_gating_func"))
            .and_then(GgufValue::as_usize)
            .map(|f| if f == 1 { "sigmoid" } else { "softmax" }.to_string())
            .unwrap_or_else(default_scoring_func);

        // Reconstruct RoPE YaRN scaling from GGUF keys when present
        let rope_scaling = {
            let factor = metadata
                .get(&format!("{arch}.rope.scaling.factor"))
                .and_then(GgufValue::as_f32);
            let orig_ctx = metadata
                .get(&format!("{arch}.rope.scaling.original_context_length"))
                .and_then(GgufValue::as_usize);
            let mscale_all_dim = metadata
                .get(&format!("{arch}.rope.scaling.yarn_log_multiplier"))
                .and_then(GgufValue::as_f32);
            match (factor, orig_ctx) {
                (Some(f), Some(orig)) => Some(RopeScalingConfig {
                    rope_type: "yarn".to_string(),
                    factor: f,
                    original_max_position_embeddings: orig,
                    mscale_all_dim,
                }),
                _ => None,
            }
        };

        Ok(Self {
            model_type: "deepseek_v3".to_string(),
            vocab_size: get_usize(metadata, &format!("{arch}.vocab_size"))
                .or_else(|_| get_usize(metadata, "tokenizer.ggml.tokens_count"))
                .unwrap_or(129_280),
            hidden_size: get_usize(metadata, &format!("{arch}.embedding_length"))?,
            intermediate_size: get_usize(metadata, &format!("{arch}.feed_forward_length"))?,
            moe_intermediate_size: get_usize(
                metadata,
                &format!("{arch}.expert_feed_forward_length"),
            )
            .ok(),
            num_hidden_layers: get_usize(metadata, &format!("{arch}.block_count"))?,
            num_attention_heads,
            num_key_value_heads: metadata
                .get(&format!("{arch}.attention.head_count_kv"))
                .and_then(GgufValue::as_usize),
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim: get_usize(metadata, &format!("{arch}.attention.key_length_mla"))
                .unwrap_or(128),
            qk_rope_head_dim,
            v_head_dim: get_usize(metadata, &format!("{arch}.attention.value_length_mla"))
                .unwrap_or(128),
            n_routed_experts: get_usize(metadata, &format!("{arch}.expert_count")).ok(),
            n_shared_experts: get_usize(metadata, &format!("{arch}.expert_shared_count"))
                .unwrap_or_else(|_| default_one()),
            num_experts_per_tok: get_usize(metadata, &format!("{arch}.expert_used_count"))
                .unwrap_or_else(|_| default_num_experts_per_tok()),
            n_group: get_usize(metadata, &format!("{arch}.expert_group_count"))
                .unwrap_or_else(|_| default_n_group()),
            topk_group: get_usize(metadata, &format!("{arch}.expert_group_used_count"))
                .unwrap_or_else(|_| default_topk_group()),
            first_k_dense_replace: get_usize(
                metadata,
                &format!("{arch}.leading_dense_block_count"),
            )
            .unwrap_or(0),
            norm_topk_prob: true,
            routed_scaling_factor: get_f32(
                metadata,
                &format!("{arch}.expert_weights_scale"),
                default_routed_scaling_factor(),
            ),
            scoring_func,
            rope_theta: get_f32(metadata, &format!("{arch}.rope.freq_base"), 10000.0),
            rope_scaling,
            max_position_embeddings: get_usize(metadata, &format!("{arch}.context_length"))
                .unwrap_or_else(|_| default_max_position_embeddings()),
            rms_norm_eps: get_f32(
                metadata,
                &format!("{arch}.attention.layernorm_rms_eps"),
                default_rms_norm_eps(),
            ),
            tie_word_embeddings: false,
            #[allow(clippy::cast_possible_truncation)]
            bos_token_id: metadata
                .get("tokenizer.ggml.bos_token_id")
                .and_then(GgufValue::as_usize)
                .unwrap_or(0) as u32,
            #[allow(clippy::cast_possible_truncation)]
            eos_token_id: metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(GgufValue::as_usize)
                .unwrap_or(1) as u32,
            quantization_config: None,
        })
    }

    /// Total Q/K head dimension (nope + rope)
    #[must_use]
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Returns `true` if `layer_idx` should use MoE (layer >= first_k_dense_replace)
    #[must_use]
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.n_routed_experts.is_some_and(|n| n > 1) && layer_idx >= self.first_k_dense_replace
    }

    /// Number of key-value heads
    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Intermediate size for MoE routed experts
    #[must_use]
    pub fn moe_expert_intermediate_size(&self) -> usize {
        self.moe_intermediate_size.unwrap_or(self.intermediate_size)
    }

    /// Shared expert intermediate size: `moe_intermediate_size * n_shared_experts`
    #[must_use]
    pub fn shared_expert_intermediate_size(&self) -> usize {
        self.moe_expert_intermediate_size() * self.n_shared_experts
    }

    /// Compute the MLA attention scale, adjusted by YaRN mscale if configured.
    ///
    /// When `rope_scaling.mscale_all_dim > 0`:
    ///   `mscale = yarn_get_mscale(factor, mscale_all_dim)`
    ///   `scale = (1/sqrt(qk_head_dim)) * mscale * mscale`
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mla_attn_scale(&self) -> f32 {
        let base_scale = 1.0 / (self.qk_head_dim() as f32).sqrt();

        if let Some(ref rs) = self.rope_scaling {
            if let Some(mscale_all_dim) = rs.mscale_all_dim {
                if mscale_all_dim > 0.0 {
                    let mscale = yarn_get_mscale(rs.factor, mscale_all_dim);
                    return base_scale * mscale * mscale;
                }
            }
        }

        base_scale
    }
}

/// Compute the YaRN magnitude scaling factor.
///
/// `mscale = (0.1 * ln(factor) + 1.0) ^ mscale_all_dim`
fn yarn_get_mscale(factor: f32, mscale_all_dim: f32) -> f32 {
    if factor <= 1.0 {
        return 1.0;
    }
    (0.1 * factor.ln() + 1.0).powf(mscale_all_dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepseek_v3_config() {
        let json = r#"{
            "model_type": "deepseek_v3",
            "vocab_size": 129280,
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "n_group": 8,
            "topk_group": 4,
            "first_k_dense_replace": 3,
            "norm_topk_prob": true,
            "routed_scaling_factor": 2.5,
            "scoring_func": "sigmoid",
            "rope_theta": 10000.0,
            "max_position_embeddings": 163840,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "bos_token_id": 0,
            "eos_token_id": 1
        }"#;

        let config: DeepSeekConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.vocab_size, 129280);
        assert_eq!(config.hidden_size, 7168);
        assert_eq!(config.q_lora_rank, Some(1536));
        assert_eq!(config.kv_lora_rank, 512);
        assert_eq!(config.qk_head_dim(), 192);
        assert_eq!(config.v_head_dim, 128);
        assert_eq!(config.n_routed_experts, Some(256));
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.first_k_dense_replace, 3);
        assert!(!config.is_moe_layer(0));
        assert!(!config.is_moe_layer(2));
        assert!(config.is_moe_layer(3));
        assert!(config.is_moe_layer(60));
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.moe_expert_intermediate_size(), 2048);
        assert_eq!(config.shared_expert_intermediate_size(), 2048);
    }

    #[test]
    fn test_deepseek_v3_tiny_config() {
        // Config from yujiepan/deepseek-v3-tiny-random
        let json = r#"{
            "model_type": "deepseek_v3",
            "vocab_size": 129280,
            "hidden_size": 16,
            "intermediate_size": 32,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "q_lora_rank": 16,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 8,
            "v_head_dim": 8,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "n_group": 8,
            "topk_group": 4,
            "first_k_dense_replace": 1,
            "norm_topk_prob": true,
            "routed_scaling_factor": 2.5,
            "scoring_func": "sigmoid",
            "rope_theta": 10000.0,
            "max_position_embeddings": 163840,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "bos_token_id": 0,
            "eos_token_id": 1
        }"#;

        let config: DeepSeekConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 16);
        assert_eq!(config.q_lora_rank, Some(16));
        assert_eq!(config.kv_lora_rank, 16);
        assert_eq!(config.qk_head_dim(), 16);
        assert!(!config.is_moe_layer(0));
        assert!(config.is_moe_layer(1));
    }

    #[test]
    fn test_mla_attn_scale_no_scaling() {
        let json = r#"{
            "vocab_size": 129280,
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128
        }"#;

        let config: DeepSeekConfig = serde_json::from_str(json).unwrap();
        let expected = 1.0 / (192.0_f32).sqrt();
        assert!((config.mla_attn_scale() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_mla_attn_scale_with_yarn_mscale() {
        let json = r#"{
            "vocab_size": 129280,
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 40.0,
                "original_max_position_embeddings": 4096,
                "mscale_all_dim": 1.0
            }
        }"#;

        let config: DeepSeekConfig = serde_json::from_str(json).unwrap();
        let base = 1.0 / (192.0_f32).sqrt();
        let mscale = yarn_get_mscale(40.0, 1.0);
        let expected = base * mscale * mscale;
        assert!(
            (config.mla_attn_scale() - expected).abs() < 1e-6,
            "got {}, expected {}",
            config.mla_attn_scale(),
            expected
        );
    }

    #[test]
    fn test_yarn_get_mscale() {
        // factor <= 1.0 should return 1.0
        assert!((yarn_get_mscale(1.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((yarn_get_mscale(0.5, 1.0) - 1.0).abs() < 1e-6);

        // factor=40, mscale_all_dim=1.0
        let expected = 0.1 * 40.0_f32.ln() + 1.0;
        assert!((yarn_get_mscale(40.0, 1.0) - expected).abs() < 1e-5);
    }

    #[test]
    fn test_eos_token_id_as_array() {
        let json = r#"{
            "vocab_size": 129280,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 8,
            "v_head_dim": 8,
            "eos_token_id": [1, 2, 3]
        }"#;

        let config: DeepSeekConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.eos_token_id, 1);
    }

    #[test]
    fn test_from_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("infernum_test_deepseek_config.json");

        let json = r#"{
            "vocab_size": 129280,
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "rope_theta": 10000.0
        }"#;

        std::fs::write(&path, json).unwrap();
        let config = DeepSeekConfig::from_file(&path).unwrap();
        assert_eq!(config.vocab_size, 129280);
        assert_eq!(config.kv_lora_rank, 512);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_from_file_missing() {
        let result = DeepSeekConfig::from_file("/nonexistent/path/config.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_gguf_metadata_deepseek_v3() {
        use infernum::GgufValue;

        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("deepseek2".into()),
        );
        meta.insert("deepseek2.block_count".into(), GgufValue::U32(61));
        meta.insert("deepseek2.embedding_length".into(), GgufValue::U32(7168));
        meta.insert("deepseek2.feed_forward_length".into(), GgufValue::U32(18432));
        meta.insert("deepseek2.attention.head_count".into(), GgufValue::U32(128));
        meta.insert(
            "deepseek2.attention.head_count_kv".into(),
            GgufValue::U32(128),
        );
        meta.insert(
            "deepseek2.attention.q_lora_rank".into(),
            GgufValue::U32(1536),
        );
        meta.insert(
            "deepseek2.attention.kv_lora_rank".into(),
            GgufValue::U32(512),
        );
        meta.insert(
            "deepseek2.attention.key_length_mla".into(),
            GgufValue::U32(128),
        );
        meta.insert(
            "deepseek2.attention.value_length_mla".into(),
            GgufValue::U32(128),
        );
        meta.insert("deepseek2.rope.dimension_count".into(), GgufValue::U32(128));
        meta.insert("deepseek2.rope.freq_base".into(), GgufValue::F32(10000.0));
        meta.insert(
            "deepseek2.attention.layernorm_rms_eps".into(),
            GgufValue::F32(1e-6),
        );
        meta.insert("deepseek2.expert_count".into(), GgufValue::U32(256));
        meta.insert("deepseek2.expert_used_count".into(), GgufValue::U32(8));
        meta.insert("deepseek2.expert_shared_count".into(), GgufValue::U32(1));
        meta.insert(
            "deepseek2.expert_feed_forward_length".into(),
            GgufValue::U32(2048),
        );
        meta.insert(
            "deepseek2.leading_dense_block_count".into(),
            GgufValue::U32(3),
        );
        meta.insert("deepseek2.expert_weights_scale".into(), GgufValue::F32(2.5));
        meta.insert("deepseek2.expert_group_count".into(), GgufValue::U32(8));
        meta.insert("deepseek2.expert_group_used_count".into(), GgufValue::U32(4));
        meta.insert("deepseek2.expert_gating_func".into(), GgufValue::U32(1));
        meta.insert("deepseek2.context_length".into(), GgufValue::U32(163840));
        meta.insert("deepseek2.vocab_size".into(), GgufValue::U32(129280));
        meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::U32(0));
        meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(1));

        let config = DeepSeekConfig::from_gguf_metadata(&meta).unwrap();

        assert_eq!(config.num_hidden_layers, 61);
        assert_eq!(config.hidden_size, 7168);
        assert_eq!(config.intermediate_size, 18432);
        assert_eq!(config.num_attention_heads, 128);
        assert_eq!(config.num_key_value_heads, Some(128));
        assert_eq!(config.q_lora_rank, Some(1536));
        assert_eq!(config.kv_lora_rank, 512);
        assert_eq!(config.qk_nope_head_dim, 128);
        assert_eq!(config.qk_rope_head_dim, 64); // 128 / 2
        assert_eq!(config.v_head_dim, 128);
        assert_eq!(config.qk_head_dim(), 192);
        assert_eq!(config.n_routed_experts, Some(256));
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.n_shared_experts, 1);
        assert_eq!(config.moe_intermediate_size, Some(2048));
        assert_eq!(config.first_k_dense_replace, 3);
        assert!(!config.is_moe_layer(0));
        assert!(!config.is_moe_layer(2));
        assert!(config.is_moe_layer(3));
        assert_eq!(config.n_group, 8);
        assert_eq!(config.topk_group, 4);
        assert_eq!(config.scoring_func, "sigmoid");
        assert!((config.routed_scaling_factor - 2.5).abs() < 1e-6);
        assert_eq!(config.max_position_embeddings, 163840);
        assert!((config.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert_eq!(config.vocab_size, 129280);
        assert_eq!(config.bos_token_id, 0);
        assert_eq!(config.eos_token_id, 1);
        assert!(config.rope_scaling.is_none());
        assert!(config.norm_topk_prob);
    }

    #[test]
    fn test_from_gguf_metadata_deepseek_q_lora_rank_zero() {
        use infernum::GgufValue;

        let mut meta = HashMap::new();
        meta.insert("deepseek2.block_count".into(), GgufValue::U32(2));
        meta.insert("deepseek2.embedding_length".into(), GgufValue::U32(16));
        meta.insert("deepseek2.feed_forward_length".into(), GgufValue::U32(32));
        meta.insert("deepseek2.attention.head_count".into(), GgufValue::U32(2));
        meta.insert("deepseek2.attention.kv_lora_rank".into(), GgufValue::U32(16));
        // q_lora_rank == 0 → no Q compression
        meta.insert("deepseek2.attention.q_lora_rank".into(), GgufValue::U32(0));
        meta.insert("deepseek2.expert_gating_func".into(), GgufValue::U32(0));

        let config = DeepSeekConfig::from_gguf_metadata(&meta).unwrap();
        assert!(config.q_lora_rank.is_none());
        assert_eq!(config.scoring_func, "softmax");
    }

    #[test]
    fn test_from_gguf_metadata_yarn_scaling() {
        use infernum::GgufValue;

        let mut meta = HashMap::new();
        meta.insert("deepseek2.block_count".into(), GgufValue::U32(2));
        meta.insert("deepseek2.embedding_length".into(), GgufValue::U32(16));
        meta.insert("deepseek2.feed_forward_length".into(), GgufValue::U32(32));
        meta.insert("deepseek2.attention.head_count".into(), GgufValue::U32(2));
        meta.insert("deepseek2.attention.kv_lora_rank".into(), GgufValue::U32(16));
        meta.insert(
            "deepseek2.rope.scaling.factor".into(),
            GgufValue::F32(40.0),
        );
        meta.insert(
            "deepseek2.rope.scaling.original_context_length".into(),
            GgufValue::U32(4096),
        );
        meta.insert(
            "deepseek2.rope.scaling.yarn_log_multiplier".into(),
            GgufValue::F32(1.0),
        );

        let config = DeepSeekConfig::from_gguf_metadata(&meta).unwrap();
        let rs = config.rope_scaling.as_ref().unwrap();
        assert_eq!(rs.rope_type, "yarn");
        assert!((rs.factor - 40.0).abs() < 1e-5);
        assert_eq!(rs.original_max_position_embeddings, 4096);
        assert_eq!(rs.mscale_all_dim, Some(1.0));

        // mla_attn_scale uses mscale_all_dim
        let base = 1.0 / (config.qk_head_dim() as f32).sqrt();
        let mscale = yarn_get_mscale(40.0, 1.0);
        let expected = base * mscale * mscale;
        assert!((config.mla_attn_scale() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_defaults() {
        let json = r#"{
            "vocab_size": 129280,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 8,
            "v_head_dim": 8
        }"#;

        let config: DeepSeekConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_type, "deepseek_v3");
        assert_eq!(config.max_position_embeddings, 4096);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.rope_theta, 10000.0);
        assert!(!config.tie_word_embeddings);
        assert!(config.norm_topk_prob);
        assert!((config.routed_scaling_factor - 2.5).abs() < 1e-6);
        assert_eq!(config.scoring_func, "sigmoid");
        assert_eq!(config.n_shared_experts, 1);
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.n_group, 8);
        assert_eq!(config.topk_group, 4);
    }
}
