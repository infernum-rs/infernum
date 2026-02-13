//! Llama model implementation

#![allow(
    clippy::struct_field_names, // _proj suffix is conventional for Llama weights
    clippy::no_effect_underscore_binding,
    clippy::doc_markdown // tensor shape docs trigger false positives
)]

use std::path::Path;

use infernum::cuda::ops::{
    add, apply_rope, argmax_last, attention, embedding_gather, matmul, precompute_rope_cache,
    repeat_kv, rms_norm, silu_mul, transpose_2d,
};

/// Transpose a weight matrix once, for use in pre-transposed linear projections.
/// (out_features, in_features) -> (in_features, out_features)
fn pretranspose_weight(weight: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    transpose_2d(weight)
}
use infernum::cuda::{CudaContext, CudaTensor};
use infernum::tensor::Tensor;
use infernum::weights::{SafeTensorsLoader, WeightLoader};
use infernum::Result;

use crate::LlamaConfig;

/// Weights for a single Llama attention layer (stored pre-transposed for efficient matmul)
struct LlamaAttentionWeights {
    q_proj: CudaTensor<f32>,
    k_proj: CudaTensor<f32>,
    v_proj: CudaTensor<f32>,
    o_proj: CudaTensor<f32>,
}

/// Weights for a single Llama MLP layer (stored pre-transposed for efficient matmul)
struct LlamaMlpWeights {
    gate_proj: CudaTensor<f32>,
    up_proj: CudaTensor<f32>,
    down_proj: CudaTensor<f32>,
}

/// Weights for a single Llama decoder layer
struct LlamaLayerWeights {
    input_layernorm: CudaTensor<f32>,
    attention: LlamaAttentionWeights,
    post_attention_layernorm: CudaTensor<f32>,
    mlp: LlamaMlpWeights,
}

/// Complete Llama model
pub struct LlamaModel {
    config: LlamaConfig,
    ctx: CudaContext,

    // Embeddings
    embed_tokens: CudaTensor<f32>,

    // Transformer layers
    layers: Vec<LlamaLayerWeights>,

    // Final layer norm
    norm: CudaTensor<f32>,

    // Output projection (may be tied to embed_tokens)
    lm_head: CudaTensor<f32>,

    // RoPE caches
    cos_cache: CudaTensor<f32>,
    sin_cache: CudaTensor<f32>,
}

impl LlamaModel {
    /// Load a Llama model from a directory containing SafeTensors and config.json
    ///
    /// # Errors
    /// Returns an error if loading fails
    pub fn from_pretrained(ctx: &CudaContext, model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();

        // Load config
        let config_path = model_path.join("config.json");
        let config = LlamaConfig::from_file(&config_path)?;

        // Load weights
        let loader = SafeTensorsLoader::from_directory(model_path)?;

        Self::load_weights(ctx, config, &loader)
    }

    /// Load model weights from a weight loader
    fn load_weights(
        ctx: &CudaContext,
        config: LlamaConfig,
        loader: &impl WeightLoader,
    ) -> Result<Self> {
        // Load embeddings
        let embed_tokens = loader.load_f32(ctx, "model.embed_tokens.weight")?;

        // Load transformer layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            let layer = LlamaLayerWeights {
                input_layernorm: loader
                    .load_f32(ctx, &format!("{prefix}.input_layernorm.weight"))?,
                attention: LlamaAttentionWeights {
                    q_proj: pretranspose_weight(
                        &loader.load_f32(ctx, &format!("{prefix}.self_attn.q_proj.weight"))?,
                    )?,
                    k_proj: pretranspose_weight(
                        &loader.load_f32(ctx, &format!("{prefix}.self_attn.k_proj.weight"))?,
                    )?,
                    v_proj: pretranspose_weight(
                        &loader.load_f32(ctx, &format!("{prefix}.self_attn.v_proj.weight"))?,
                    )?,
                    o_proj: pretranspose_weight(
                        &loader.load_f32(ctx, &format!("{prefix}.self_attn.o_proj.weight"))?,
                    )?,
                },
                post_attention_layernorm: loader
                    .load_f32(ctx, &format!("{prefix}.post_attention_layernorm.weight"))?,
                mlp: LlamaMlpWeights {
                    gate_proj: pretranspose_weight(
                        &loader.load_f32(ctx, &format!("{prefix}.mlp.gate_proj.weight"))?,
                    )?,
                    up_proj: pretranspose_weight(
                        &loader.load_f32(ctx, &format!("{prefix}.mlp.up_proj.weight"))?,
                    )?,
                    down_proj: pretranspose_weight(
                        &loader.load_f32(ctx, &format!("{prefix}.mlp.down_proj.weight"))?,
                    )?,
                },
            };

            layers.push(layer);
        }

        // Load final norm
        let norm = loader.load_f32(ctx, "model.norm.weight")?;

        // Load or tie lm_head (pre-transposed for efficient linear projection)
        let lm_head = if config.tie_word_embeddings {
            pretranspose_weight(&embed_tokens)?
        } else {
            pretranspose_weight(&loader.load_f32(ctx, "lm_head.weight")?)?
        };

        // Precompute RoPE cache
        let (cos_cache, sin_cache) = precompute_rope_cache(
            ctx,
            config.max_position_embeddings,
            config.head_dim(),
            config.rope_theta,
        )?;

        Ok(Self {
            config,
            ctx: ctx.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
            cos_cache,
            sin_cache,
        })
    }

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// Run forward pass and return logits
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape (seq_len,)
    ///
    /// # Returns
    /// Logits tensor of shape (seq_len, vocab_size)
    ///
    /// # Errors
    /// Returns an error if forward pass fails
    pub fn forward(&self, input_ids: &[u32]) -> Result<CudaTensor<f32>> {
        let _seq_len = input_ids.len();

        // Embed tokens: (seq_len,) -> (seq_len, hidden_size)
        let mut hidden = self.embed(input_ids)?;

        // Run through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer(&hidden, layer, layer_idx)?;
        }

        // Final layer norm
        hidden = rms_norm(&hidden, &self.norm, self.config.rms_norm_eps)?;

        // Project to vocabulary: (seq_len, hidden_size) @ (vocab_size, hidden_size)^T -> (seq_len, vocab_size)
        let logits = self.lm_head_forward(&hidden)?;

        Ok(logits)
    }

    /// Greedy autoregressive generation
    ///
    /// Runs the model forward repeatedly, selecting the highest-probability
    /// token at each step via argmax. Stops when `max_new_tokens` are produced
    /// or the EOS token is emitted.
    ///
    /// # Arguments
    /// * `input_ids` - Prompt token IDs
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `eos_token_id` - Optional EOS token ID to stop generation early
    ///
    /// # Returns
    /// The full token sequence (prompt + generated tokens)
    ///
    /// # Errors
    /// Returns an error if a forward pass fails
    pub fn generate(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<Vec<u32>> {
        let mut tokens = input_ids.to_vec();

        for _ in 0..max_new_tokens {
            let logits = self.forward(&tokens)?;

            // logits: (seq_len, vocab_size) — extract last row via reshape
            let seq_len = logits.shape()[0];
            let vocab_size = logits.shape()[1];
            let last_logits = logits.reshape(&[seq_len, vocab_size]);

            // Argmax on GPU over the full (seq_len, vocab_size) matrix,
            // then take only the last row's result
            let all_argmax = argmax_last(&last_logits)?;
            let next_token = all_argmax[seq_len - 1];

            if eos_token_id == Some(next_token) {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Embed token IDs
    fn embed(&self, input_ids: &[u32]) -> Result<CudaTensor<f32>> {
        embedding_gather(&self.ctx, &self.embed_tokens, input_ids)
    }

    /// Forward pass through a single transformer layer
    fn forward_layer(
        &self,
        hidden: &CudaTensor<f32>,
        layer: &LlamaLayerWeights,
        _layer_idx: usize,
    ) -> Result<CudaTensor<f32>> {
        let _seq_len = hidden.shape()[0];
        let _hidden_size = self.config.hidden_size;

        // Pre-attention RMS norm
        let normed = rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

        // Self-attention
        let attn_output = self.forward_attention(&normed, &layer.attention)?;

        // Residual connection
        let hidden = add(hidden, &attn_output)?;

        // Pre-MLP RMS norm
        let normed = rms_norm(
            &hidden,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        // MLP
        let mlp_output = self.forward_mlp(&normed, &layer.mlp)?;

        // Residual connection
        add(&hidden, &mlp_output)
    }

    /// Forward pass through attention
    fn forward_attention(
        &self,
        hidden: &CudaTensor<f32>,
        weights: &LlamaAttentionWeights,
    ) -> Result<CudaTensor<f32>> {
        let seq_len = hidden.shape()[0];
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads();
        let head_dim = self.config.head_dim();

        // Project Q, K, V
        // hidden: (seq_len, hidden_size)
        // weights: (out_features, in_features) - need to transpose for matmul

        // Q: (seq_len, hidden_size) @ (hidden_size, num_heads * head_dim)
        let q = linear(hidden, &weights.q_proj)?;
        let k = linear(hidden, &weights.k_proj)?;
        let v = linear(hidden, &weights.v_proj)?;

        // Reshape for attention: (seq_len, num_heads, head_dim)
        let q = q.reshape(&[seq_len, num_heads, head_dim]);
        let k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
        let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

        // Apply RoPE to Q and K
        let q = apply_rope(&q, &self.cos_cache, &self.sin_cache, 0)?;
        let k = apply_rope(&k, &self.cos_cache, &self.sin_cache, 0)?;

        // Expand K, V for GQA if needed
        let (k, v) = if num_kv_heads < num_heads {
            let k = repeat_kv(&k, self.config.num_heads_per_kv())?;
            let v = repeat_kv(&v, self.config.num_heads_per_kv())?;
            (k, v)
        } else {
            (k, v)
        };

        // Compute attention
        let attn_output = attention(&q, &k, &v, true)?;

        // Reshape back: (seq_len, num_heads, head_dim) -> (seq_len, hidden_size)
        let attn_output = attn_output.reshape(&[seq_len, hidden_size]);

        // Output projection
        linear(&attn_output, &weights.o_proj)
    }

    /// Forward pass through MLP (SwiGLU)
    #[allow(clippy::unused_self)] // Will use self.config when adding intermediate_size check
    fn forward_mlp(
        &self,
        hidden: &CudaTensor<f32>,
        weights: &LlamaMlpWeights,
    ) -> Result<CudaTensor<f32>> {
        // SwiGLU: silu(gate(x)) * up(x)
        let gate = linear(hidden, &weights.gate_proj)?;
        let up = linear(hidden, &weights.up_proj)?;

        // Fused SiLU + multiply
        let intermediate = silu_mul(&gate, &up)?;

        // Down projection
        linear(&intermediate, &weights.down_proj)
    }

    /// Project hidden states to vocabulary logits
    fn lm_head_forward(&self, hidden: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
        // hidden: (seq_len, hidden_size)
        // lm_head: (vocab_size, hidden_size)
        // output: (seq_len, vocab_size)
        linear(hidden, &self.lm_head)
    }
}

/// Linear projection: output = input @ weight
/// Weights must be stored pre-transposed as (in_features, out_features).
/// input: (seq_len, in_features)
/// weight: (in_features, out_features)
/// output: (seq_len, out_features)
fn linear(input: &CudaTensor<f32>, weight: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    matmul(input, weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use infernum::dtype::DType;
    use std::collections::HashMap;

    #[test]
    fn test_linear() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // input: (2, 3), weight pre-transposed: (3, 4) -> output: (2, 4)
        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // Columns correspond to output features:
        // col 0: picks dim 0, col 1: picks dim 1, col 2: picks dim 2, col 3: sum
        let weight_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 1.0, // row 0
            0.0, 1.0, 0.0, 1.0, // row 1
            0.0, 0.0, 1.0, 1.0, // row 2
        ];

        let input = CudaTensor::from_slice(&ctx, &[2, 3], &input_data).unwrap();
        let weight = CudaTensor::from_slice(&ctx, &[3, 4], &weight_data).unwrap();

        let output = linear(&input, &weight).unwrap();

        assert_eq!(output.shape(), &[2, 4]);

        let result = output.to_vec().unwrap();
        // row 0: [1, 2, 3, 6], row 1: [4, 5, 6, 15]
        assert!((result[0] - 1.0).abs() < 1e-4);
        assert!((result[1] - 2.0).abs() < 1e-4);
        assert!((result[2] - 3.0).abs() < 1e-4);
        assert!((result[3] - 6.0).abs() < 1e-4);
        assert!((result[4] - 4.0).abs() < 1e-4);
        assert!((result[5] - 5.0).abs() < 1e-4);
        assert!((result[6] - 6.0).abs() < 1e-4);
        assert!((result[7] - 15.0).abs() < 1e-4);
    }

    // --- End-to-end tests with a tiny model ---

    /// Deterministic pseudo-random f32 in [-scale, scale] for reproducible test weights
    fn pseudo_random_weights(n: usize, scale: f32) -> Vec<f32> {
        let mut values = Vec::with_capacity(n);
        let mut state: u64 = 42;
        for _ in 0..n {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let f = (state as f32) / (u64::MAX as f32); // [0, 1)
            values.push((f * 2.0 - 1.0) * scale);
        }
        values
    }

    /// Mock weight loader that stores pre-built tensors by name
    struct MockWeightLoader {
        tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
    }

    impl MockWeightLoader {
        fn new() -> Self {
            Self {
                tensors: HashMap::new(),
            }
        }

        fn add(&mut self, name: &str, shape: &[usize], data: Vec<f32>) {
            self.tensors
                .insert(name.to_string(), (shape.to_vec(), data));
        }
    }

    impl WeightLoader for MockWeightLoader {
        fn load_f32(&self, ctx: &CudaContext, name: &str) -> Result<CudaTensor<f32>> {
            let (shape, data) = self
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("MockWeightLoader: tensor not found: {name}"));
            CudaTensor::from_slice(ctx, shape, data)
        }

        fn get_shape(&self, name: &str) -> Result<Vec<usize>> {
            Ok(self.tensors.get(name).unwrap().0.clone())
        }

        fn get_dtype(&self, _name: &str) -> Result<DType> {
            Ok(DType::F32)
        }

        fn tensor_names(&self) -> Vec<String> {
            self.tensors.keys().cloned().collect()
        }

        fn contains(&self, name: &str) -> bool {
            self.tensors.contains_key(name)
        }
    }

    /// Build a tiny LlamaConfig for testing
    fn tiny_config() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: Some(2),
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }

    /// Build a MockWeightLoader with random weights matching the tiny config
    fn tiny_weight_loader(config: &LlamaConfig) -> MockWeightLoader {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();
        let scale = 0.02;

        let mut loader = MockWeightLoader::new();
        let mut seed_offset: usize = 0;

        let mut rand = |n: usize| -> Vec<f32> {
            seed_offset += 1;
            let mut vals = pseudo_random_weights(n, scale);
            // Rotate by seed_offset to get different values per tensor
            vals.rotate_left(seed_offset % n.max(1));
            vals
        };

        // Embeddings
        loader.add("model.embed_tokens.weight", &[vocab, h], rand(vocab * h));

        // Layer 0
        let prefix = "model.layers.0";
        loader.add(
            &format!("{prefix}.input_layernorm.weight"),
            &[h],
            vec![1.0; h],
        );
        loader.add(
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[num_heads * head_dim, h],
            rand(num_heads * head_dim * h),
        );
        loader.add(
            &format!("{prefix}.self_attn.k_proj.weight"),
            &[num_kv_heads * head_dim, h],
            rand(num_kv_heads * head_dim * h),
        );
        loader.add(
            &format!("{prefix}.self_attn.v_proj.weight"),
            &[num_kv_heads * head_dim, h],
            rand(num_kv_heads * head_dim * h),
        );
        loader.add(
            &format!("{prefix}.self_attn.o_proj.weight"),
            &[h, num_heads * head_dim],
            rand(h * num_heads * head_dim),
        );
        loader.add(
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[h],
            vec![1.0; h],
        );
        loader.add(
            &format!("{prefix}.mlp.gate_proj.weight"),
            &[inter, h],
            rand(inter * h),
        );
        loader.add(
            &format!("{prefix}.mlp.up_proj.weight"),
            &[inter, h],
            rand(inter * h),
        );
        loader.add(
            &format!("{prefix}.mlp.down_proj.weight"),
            &[h, inter],
            rand(h * inter),
        );

        // Final norm
        loader.add("model.norm.weight", &[h], vec![1.0; h]);

        // lm_head
        loader.add("lm_head.weight", &[vocab, h], rand(vocab * h));

        loader
    }

    fn build_tiny_model(ctx: &CudaContext) -> LlamaModel {
        let config = tiny_config();
        let loader = tiny_weight_loader(&config);
        LlamaModel::load_weights(ctx, config, &loader).expect("Failed to build tiny model")
    }

    #[test]
    fn test_forward_output_shape() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_model(&ctx);

        let input_ids: Vec<u32> = vec![1, 5, 10];
        let logits = model.forward(&input_ids).expect("Forward pass failed");

        // logits should be (seq_len, vocab_size)
        assert_eq!(logits.shape(), &[3, 64]);
    }

    #[test]
    fn test_forward_single_token() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_model(&ctx);

        let logits = model.forward(&[0]).expect("Forward pass failed");
        assert_eq!(logits.shape(), &[1, 64]);

        // Logits should be finite
        let data = logits.to_vec().unwrap();
        assert!(
            data.iter().all(|x| x.is_finite()),
            "Logits contain non-finite values"
        );
    }

    #[test]
    fn test_forward_deterministic() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_model(&ctx);

        let input_ids: Vec<u32> = vec![1, 5, 10];
        let logits1 = model.forward(&input_ids).unwrap().to_vec().unwrap();
        let logits2 = model.forward(&input_ids).unwrap().to_vec().unwrap();

        for (i, (a, b)) in logits1.iter().zip(logits2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "Non-deterministic at index {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_generate_respects_max_tokens() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_model(&ctx);

        let prompt = vec![1_u32, 5, 10];
        let max_new = 4;
        let tokens = model.generate(&prompt, max_new, None).unwrap();

        // Should produce at most prompt_len + max_new_tokens
        assert!(tokens.len() <= prompt.len() + max_new);
        assert!(
            tokens.len() > prompt.len(),
            "Should generate at least 1 token"
        );
        // Prompt should be preserved
        assert_eq!(&tokens[..prompt.len()], &prompt);
    }

    #[test]
    fn test_generate_stops_on_eos() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_model(&ctx);

        // Run generate with a very large max_new_tokens but EOS set to a
        // token that the model is likely to produce with random weights
        let prompt = vec![1_u32];
        let result_no_eos = model.generate(&prompt, 5, None).unwrap();

        // The generated tokens should not include the EOS token if we set it
        // to one that was actually produced
        if result_no_eos.len() > 1 {
            let first_generated = result_no_eos[1];
            let result_with_eos = model.generate(&prompt, 100, Some(first_generated)).unwrap();
            // Should stop immediately since the first generated token == EOS
            assert_eq!(
                result_with_eos.len(),
                prompt.len(),
                "Should stop before appending the EOS token"
            );
        }
    }
}
