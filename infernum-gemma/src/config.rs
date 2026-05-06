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

/// Re-exported from `infernum` core — the same struct is used by
/// `WeightLoader<B>` for generic weight loading.
pub use infernum::QuantizationConfig;

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
    /// Parse a `config.json` file into a `GemmaConfig`, returning an error on failure.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, the JSON is malformed, or
    /// the `model_type` is not `gemma2` or `gemma3_text`.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> infernum::Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())?;
        // Validate model_type before the panicking from_str path.
        let model_type: serde_json::Value = serde_json::from_str(&text)?;
        let mt = model_type
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if mt != "gemma2" && mt != "gemma3_text" {
            return Err(infernum::Error::UnsupportedModel(format!(
                "Unsupported model_type: `{mt}`. Expected `gemma2` or `gemma3_text`"
            )));
        }
        Ok(Self::from_str(&text))
    }

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

    /// Build a `GemmaConfig` from GGUF metadata key-value pairs.
    ///
    /// GGUF metadata uses keys like `gemma2.embedding_length` or
    /// `gemma3.embedding_length`. The architecture prefix is read from
    /// `general.architecture`. Gemma-specific fields like soft-capping and
    /// layer_types are derived from metadata or sensible defaults.
    ///
    /// # Errors
    /// Returns an error if required metadata keys are missing.
    pub fn from_gguf_metadata(
        metadata: &std::collections::HashMap<String, infernum::GgufValue>,
    ) -> infernum::Result<Self> {
        use infernum::GgufValue;

        let arch = metadata
            .get("general.architecture")
            .and_then(GgufValue::as_str)
            .unwrap_or("gemma2");

        let get_usize = |key: &str| -> infernum::Result<usize> {
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

        let is_gemma3 = arch == "gemma3";
        let model_type = if is_gemma3 {
            "gemma3_text".to_string()
        } else {
            "gemma2".to_string()
        };

        let hidden_size = get_usize(&format!("{arch}.embedding_length"))?;
        let num_hidden_layers = get_usize(&format!("{arch}.block_count"))?;
        let num_attention_heads = get_usize(&format!("{arch}.attention.head_count"))?;
        let num_key_value_heads = metadata
            .get(&format!("{arch}.attention.head_count_kv"))
            .and_then(GgufValue::as_usize)
            .unwrap_or(num_attention_heads);

        let head_dim = metadata
            .get(&format!("{arch}.attention.key_length"))
            .and_then(GgufValue::as_usize)
            .unwrap_or(hidden_size / num_attention_heads);

        let sliding_window = metadata
            .get(&format!("{arch}.attention.sliding_window"))
            .and_then(GgufValue::as_usize)
            .filter(|&w| w > 0);

        // Generate layer_types from pattern
        let default_pattern = if is_gemma3 { 6 } else { 2 };
        let sliding_window_pattern = Some(default_pattern);
        let layer_types = (0..num_hidden_layers)
            .map(|i| {
                if (i + 1) % default_pattern != 0 {
                    "sliding_attention".to_string()
                } else {
                    "full_attention".to_string()
                }
            })
            .collect();

        // Soft-capping: Gemma 2 has these in GGUF metadata, Gemma 3 does not
        let attn_logit_softcapping = metadata
            .get(&format!("{arch}.attn_logit_softcapping"))
            .and_then(GgufValue::as_f32)
            .filter(|&v| v > 0.0);
        let final_logit_softcapping = metadata
            .get(&format!("{arch}.final_logit_softcapping"))
            .and_then(GgufValue::as_f32)
            .filter(|&v| v > 0.0);

        // query_pre_attn_scalar: derived from head_dim (not stored in GGUF)
        let query_pre_attn_scalar = head_dim as f32;

        Ok(Self {
            model_type,
            vocab_size: get_usize(&format!("{arch}.vocab_size"))
                .or_else(|_| get_usize("tokenizer.ggml.tokens_count"))
                .unwrap_or(256_000),
            hidden_size,
            intermediate_size: get_usize(&format!("{arch}.feed_forward_length"))?,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps: get_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"), 1e-6),
            max_position_embeddings: get_usize(&format!("{arch}.context_length")).unwrap_or(8192),
            tie_word_embeddings: true, // Gemma always ties
            query_pre_attn_scalar,
            sliding_window,
            layer_types,
            sliding_window_pattern,
            attn_logit_softcapping,
            final_logit_softcapping,
            rope_theta: get_f32(&format!("{arch}.rope.freq_base"), 10000.0),
            rope_local_base_freq: None, // Not stored in GGUF; dual-theta from Gemma 3 JSON only
            has_qk_norm: is_gemma3,
            #[allow(clippy::cast_possible_truncation)]
            bos_token_id: metadata
                .get("tokenizer.ggml.bos_token_id")
                .and_then(GgufValue::as_usize)
                .unwrap_or(2) as u32,
            #[allow(clippy::cast_possible_truncation)]
            eos_token_id: metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(GgufValue::as_usize)
                .unwrap_or(1) as u32,
            quantization_config: None,
        })
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
        for i in 0_usize..18 {
            if (i + 1).is_multiple_of(6) {
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

    #[test]
    fn test_from_gguf_metadata_gemma2() {
        use infernum::GgufValue;
        use std::collections::HashMap;

        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("gemma2".into()),
        );
        meta.insert("gemma2.embedding_length".into(), GgufValue::U32(2304));
        meta.insert("gemma2.block_count".into(), GgufValue::U32(26));
        meta.insert("gemma2.attention.head_count".into(), GgufValue::U32(8));
        meta.insert("gemma2.attention.head_count_kv".into(), GgufValue::U32(4));
        meta.insert("gemma2.attention.key_length".into(), GgufValue::U32(256));
        meta.insert("gemma2.feed_forward_length".into(), GgufValue::U32(9216));
        meta.insert("gemma2.context_length".into(), GgufValue::U32(8192));
        meta.insert(
            "gemma2.attention.layer_norm_rms_epsilon".into(),
            GgufValue::F32(1e-6),
        );
        meta.insert("gemma2.rope.freq_base".into(), GgufValue::F32(10000.0));
        meta.insert("gemma2.attn_logit_softcapping".into(), GgufValue::F32(50.0));
        meta.insert(
            "gemma2.final_logit_softcapping".into(),
            GgufValue::F32(30.0),
        );
        meta.insert(
            "gemma2.attention.sliding_window".into(),
            GgufValue::U32(4096),
        );
        meta.insert("gemma2.vocab_size".into(), GgufValue::U32(256000));
        meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::U32(2));
        meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(1));

        let cfg = GemmaConfig::from_gguf_metadata(&meta).unwrap();

        assert_eq!(cfg.model_type, "gemma2");
        assert!(!cfg.is_gemma3());
        assert!(!cfg.has_qk_norm);
        assert_eq!(cfg.hidden_size, 2304);
        assert_eq!(cfg.num_hidden_layers, 26);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.intermediate_size, 9216);
        assert_eq!(cfg.attn_logit_softcapping, Some(50.0));
        assert_eq!(cfg.final_logit_softcapping, Some(30.0));
        assert_eq!(cfg.sliding_window, Some(4096));
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.bos_token_id, 2);
        assert_eq!(cfg.eos_token_id, 1);
        assert!((cfg.query_pre_attn_scalar - 256.0).abs() < 1e-6);

        // Gemma 2 pattern=2: layer 0=sliding, 1=full, 2=sliding, ...
        assert_eq!(cfg.layer_types.len(), 26);
        assert_eq!(cfg.layer_types[0], "sliding_attention");
        assert_eq!(cfg.layer_types[1], "full_attention");
        assert_eq!(cfg.effective_sliding_window(0), Some(4096));
        assert_eq!(cfg.effective_sliding_window(1), None);
    }

    #[test]
    fn test_from_gguf_metadata_gemma3() {
        use infernum::GgufValue;
        use std::collections::HashMap;

        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("gemma3".into()),
        );
        meta.insert("gemma3.embedding_length".into(), GgufValue::U32(1152));
        meta.insert("gemma3.block_count".into(), GgufValue::U32(18));
        meta.insert("gemma3.attention.head_count".into(), GgufValue::U32(4));
        meta.insert("gemma3.attention.head_count_kv".into(), GgufValue::U32(1));
        meta.insert("gemma3.attention.key_length".into(), GgufValue::U32(256));
        meta.insert("gemma3.feed_forward_length".into(), GgufValue::U32(6912));
        meta.insert("gemma3.context_length".into(), GgufValue::U32(32768));
        meta.insert(
            "gemma3.attention.sliding_window".into(),
            GgufValue::U32(1024),
        );
        meta.insert("gemma3.rope.freq_base".into(), GgufValue::F32(1_000_000.0));

        let cfg = GemmaConfig::from_gguf_metadata(&meta).unwrap();

        assert_eq!(cfg.model_type, "gemma3_text");
        assert!(cfg.is_gemma3());
        assert!(cfg.has_qk_norm);
        assert_eq!(cfg.attn_logit_softcapping, None);
        assert_eq!(cfg.final_logit_softcapping, None);
        assert_eq!(cfg.sliding_window, Some(1024));

        // Gemma 3 pattern=6: layers 0-4=sliding, 5=full, 6-10=sliding, 11=full, ...
        assert_eq!(cfg.layer_types.len(), 18);
        for i in 0_usize..18 {
            if (i + 1) % 6 == 0 {
                assert_eq!(cfg.layer_types[i], "full_attention", "layer {i}");
                assert_eq!(cfg.effective_sliding_window(i), None, "layer {i}");
            } else {
                assert_eq!(cfg.layer_types[i], "sliding_attention", "layer {i}");
                assert_eq!(cfg.effective_sliding_window(i), Some(1024), "layer {i}");
            }
        }
    }
}
