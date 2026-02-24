#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::must_use_candidate,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::should_implement_trait
)]

use serde::Deserialize;
use std::path::Path;

/// Quantization configuration for pre-quantized models.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub quant_method: String,
    #[serde(default = "default_quant_bits")]
    pub bits: u32,
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

/// Configuration for Gemma 2 and Gemma 3 text models.
#[derive(Debug, Clone)]
pub struct GemmaConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub query_pre_attn_scalar: f32,

    // Sliding window
    pub sliding_window: Option<usize>,
    pub layer_types: Vec<String>,
    pub sliding_window_pattern: Option<usize>,

    // Soft-capping (Gemma 2: Some, Gemma 3: None)
    pub attn_logit_softcapping: Option<f32>,
    pub final_logit_softcapping: Option<f32>,

    // RoPE
    pub rope_theta: f32,
    pub rope_local_base_freq: Option<f32>,

    // QK-norm (Gemma 3 only)
    pub has_qk_norm: bool,

    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub quantization_config: Option<QuantizationConfig>,
}

/// Raw JSON structure for deserialization.
#[derive(Deserialize)]
struct RawConfig {
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    #[serde(default = "default_query_pre_attn_scalar")]
    query_pre_attn_scalar: f64,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    sliding_window_pattern: Option<usize>,
    #[serde(default)]
    attn_logit_softcapping: Option<f64>,
    #[serde(default)]
    final_logit_softcapping: Option<f64>,
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default)]
    rope_local_base_freq: Option<f64>,
    #[serde(default = "default_bos")]
    bos_token_id: u32,
    #[serde(
        default = "default_eos",
        deserialize_with = "deserialize_single_or_first"
    )]
    eos_token_id: u32,
    #[serde(default)]
    quantization_config: Option<QuantizationConfig>,
}

fn default_max_pos() -> usize {
    8192
}
fn default_true() -> bool {
    true
}
fn default_query_pre_attn_scalar() -> f64 {
    256.0
}
fn default_rope_theta() -> f64 {
    10000.0
}
fn default_bos() -> u32 {
    2
}
fn default_eos() -> u32 {
    1
}

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

impl GemmaConfig {
    /// Parse a `config.json` file into a `GemmaConfig`.
    ///
    /// # Panics
    ///
    /// Panics if the file cannot be read or parsed, or if the `model_type` is
    /// not `gemma2` or `gemma3_text`.
    pub fn from_json(path: &Path) -> Self {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
        Self::from_str(&text)
    }

    /// Parse a JSON string into a `GemmaConfig`.
    ///
    /// # Panics
    ///
    /// Panics if parsing fails or `model_type` is unsupported.
    pub fn from_str(text: &str) -> Self {
        let raw: RawConfig =
            serde_json::from_str(text).unwrap_or_else(|e| panic!("Failed to parse config: {e}"));

        assert!(
            raw.model_type == "gemma2" || raw.model_type == "gemma3_text",
            "Unsupported model_type: {}. Expected `gemma2` or `gemma3_text`",
            raw.model_type
        );

        let is_gemma3 = raw.model_type == "gemma3_text";

        // Generate layer_types if not explicitly provided
        let layer_types = raw.layer_types.unwrap_or_else(|| {
            let pattern = raw
                .sliding_window_pattern
                .unwrap_or(if is_gemma3 { 6 } else { 2 });
            (0..raw.num_hidden_layers)
                .map(|i| {
                    if (i + 1) % pattern != 0 {
                        "sliding_attention".to_string()
                    } else {
                        "full_attention".to_string()
                    }
                })
                .collect()
        });

        Self {
            model_type: raw.model_type,
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            rms_norm_eps: raw.rms_norm_eps as f32,
            max_position_embeddings: raw.max_position_embeddings,
            tie_word_embeddings: raw.tie_word_embeddings,
            query_pre_attn_scalar: raw.query_pre_attn_scalar as f32,
            sliding_window: raw.sliding_window,
            layer_types,
            sliding_window_pattern: raw.sliding_window_pattern,
            attn_logit_softcapping: raw.attn_logit_softcapping.map(|v| v as f32),
            final_logit_softcapping: raw.final_logit_softcapping.map(|v| v as f32),
            rope_theta: raw.rope_theta as f32,
            rope_local_base_freq: raw.rope_local_base_freq.map(|v| v as f32),
            has_qk_norm: is_gemma3,
            bos_token_id: raw.bos_token_id,
            eos_token_id: raw.eos_token_id,
            quantization_config: raw.quantization_config,
        }
    }

    /// Returns the effective sliding window for a given layer.
    ///
    /// - For `"sliding_attention"` layers: returns `Some(sliding_window)` if configured
    /// - For `"full_attention"` layers: returns `None` (full causal attention)
    pub fn effective_sliding_window(&self, layer_idx: usize) -> Option<usize> {
        if self.layer_types.get(layer_idx).map(String::as_str) == Some("sliding_attention") {
            self.sliding_window
        } else {
            None
        }
    }

    /// Returns the attention scale: `1 / sqrt(query_pre_attn_scalar)`.
    pub fn attn_scale(&self) -> f32 {
        1.0 / self.query_pre_attn_scalar.sqrt()
    }

    /// Returns the RoPE theta for a given layer.
    ///
    /// - Gemma 3: `rope_local_base_freq` for sliding layers, `rope_theta` for full layers
    /// - Gemma 2: always `rope_theta`
    pub fn rope_theta_for_layer(&self, layer_idx: usize) -> f32 {
        if let Some(local_freq) = self.rope_local_base_freq {
            if self.layer_types.get(layer_idx).map(String::as_str) == Some("sliding_attention") {
                return local_freq;
            }
        }
        self.rope_theta
    }

    /// Returns `true` if this is a Gemma 3 model.
    pub fn is_gemma3(&self) -> bool {
        self.model_type == "gemma3_text"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gemma2_config() {
        let json = r#"{
            "model_type": "gemma2",
            "vocab_size": 256000,
            "hidden_size": 2304,
            "intermediate_size": 9216,
            "num_hidden_layers": 26,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 8192,
            "tie_word_embeddings": true,
            "query_pre_attn_scalar": 256,
            "sliding_window": 4096,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "rope_theta": 10000.0
        }"#;
        let cfg = GemmaConfig::from_str(json);

        assert_eq!(cfg.model_type, "gemma2");
        assert_eq!(cfg.hidden_size, 2304);
        assert_eq!(cfg.num_hidden_layers, 26);
        assert_eq!(cfg.head_dim, 256);
        assert!(!cfg.is_gemma3());
        assert!(!cfg.has_qk_norm);
        assert_eq!(cfg.attn_logit_softcapping, Some(50.0));
        assert_eq!(cfg.final_logit_softcapping, Some(30.0));
        assert_eq!(cfg.sliding_window, Some(4096));

        // Gemma 2 alternates every 2nd layer
        assert_eq!(cfg.layer_types.len(), 26);
        assert_eq!(cfg.layer_types[0], "sliding_attention");
        assert_eq!(cfg.layer_types[1], "full_attention");
        assert_eq!(cfg.layer_types[2], "sliding_attention");

        assert_eq!(cfg.effective_sliding_window(0), Some(4096));
        assert_eq!(cfg.effective_sliding_window(1), None);

        let expected_scale = 1.0 / 256.0_f32.sqrt();
        assert!((cfg.attn_scale() - expected_scale).abs() < 1e-6);
    }

    #[test]
    fn test_parse_gemma3_config() {
        let json = r#"{
            "model_type": "gemma3_text",
            "vocab_size": 262144,
            "hidden_size": 1152,
            "intermediate_size": 6912,
            "num_hidden_layers": 18,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 32768,
            "tie_word_embeddings": true,
            "query_pre_attn_scalar": 256,
            "sliding_window": 1024,
            "sliding_window_pattern": 6,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0
        }"#;
        let cfg = GemmaConfig::from_str(json);

        assert_eq!(cfg.model_type, "gemma3_text");
        assert!(cfg.is_gemma3());
        assert!(cfg.has_qk_norm);
        assert_eq!(cfg.attn_logit_softcapping, None);
        assert_eq!(cfg.final_logit_softcapping, None);

        // Gemma 3 pattern: every 6th layer is full
        assert_eq!(cfg.layer_types.len(), 18);
        for i in 0..18 {
            if (i + 1) % 6 == 0 {
                assert_eq!(cfg.layer_types[i], "full_attention", "layer {i}");
            } else {
                assert_eq!(cfg.layer_types[i], "sliding_attention", "layer {i}");
            }
        }

        // Dual-theta RoPE
        assert_eq!(cfg.rope_theta_for_layer(0), 10000.0); // sliding
        assert_eq!(cfg.rope_theta_for_layer(5), 1_000_000.0); // full (6th layer)
        assert_eq!(cfg.rope_theta_for_layer(11), 1_000_000.0); // full (12th layer)
        assert_eq!(cfg.rope_theta_for_layer(4), 10000.0); // sliding
    }

    #[test]
    fn test_explicit_layer_types() {
        let json = r#"{
            "model_type": "gemma2",
            "vocab_size": 256000,
            "hidden_size": 2304,
            "intermediate_size": 9216,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "rms_norm_eps": 1e-6,
            "query_pre_attn_scalar": 256,
            "sliding_window": 4096,
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
        }"#;
        let cfg = GemmaConfig::from_str(json);

        assert_eq!(cfg.layer_types.len(), 4);
        assert_eq!(cfg.layer_types[0], "sliding_attention");
        assert_eq!(cfg.layer_types[1], "full_attention");
    }
}
