//! Llama model implementation

#![allow(
    clippy::struct_field_names, // _proj suffix is conventional for Llama weights
    clippy::no_effect_underscore_binding,
    clippy::doc_markdown // tensor shape docs trigger false positives
)]

use std::path::Path;

use infernum::cuda::ops::{
    apply_rope, attention, matmul, precompute_rope_cache, rms_norm, silu_mul,
};
use infernum::cuda::{CudaContext, CudaTensor};
use infernum::tensor::Tensor;
use infernum::weights::{SafeTensorsLoader, WeightLoader};
use infernum::Result;

use crate::LlamaConfig;

/// Weights for a single Llama attention layer
struct LlamaAttentionWeights {
    q_proj: CudaTensor<f32>,
    k_proj: CudaTensor<f32>,
    v_proj: CudaTensor<f32>,
    o_proj: CudaTensor<f32>,
}

/// Weights for a single Llama MLP layer
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
                    q_proj: loader.load_f32(ctx, &format!("{prefix}.self_attn.q_proj.weight"))?,
                    k_proj: loader.load_f32(ctx, &format!("{prefix}.self_attn.k_proj.weight"))?,
                    v_proj: loader.load_f32(ctx, &format!("{prefix}.self_attn.v_proj.weight"))?,
                    o_proj: loader.load_f32(ctx, &format!("{prefix}.self_attn.o_proj.weight"))?,
                },
                post_attention_layernorm: loader
                    .load_f32(ctx, &format!("{prefix}.post_attention_layernorm.weight"))?,
                mlp: LlamaMlpWeights {
                    gate_proj: loader.load_f32(ctx, &format!("{prefix}.mlp.gate_proj.weight"))?,
                    up_proj: loader.load_f32(ctx, &format!("{prefix}.mlp.up_proj.weight"))?,
                    down_proj: loader.load_f32(ctx, &format!("{prefix}.mlp.down_proj.weight"))?,
                },
            };

            layers.push(layer);
        }

        // Load final norm
        let norm = loader.load_f32(ctx, "model.norm.weight")?;

        // Load or tie lm_head
        let lm_head = if config.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            loader.load_f32(ctx, "lm_head.weight")?
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

    /// Embed token IDs
    fn embed(&self, input_ids: &[u32]) -> Result<CudaTensor<f32>> {
        let seq_len = input_ids.len();
        let hidden_size = self.config.hidden_size;

        // Gather embeddings (done on CPU for simplicity)
        let embed_data = self.embed_tokens.to_vec()?;
        let mut output_data = vec![0.0_f32; seq_len * hidden_size];

        for (pos, &token_id) in input_ids.iter().enumerate() {
            let src_start = (token_id as usize) * hidden_size;
            let dst_start = pos * hidden_size;
            output_data[dst_start..dst_start + hidden_size]
                .copy_from_slice(&embed_data[src_start..src_start + hidden_size]);
        }

        CudaTensor::from_slice(&self.ctx, &[seq_len, hidden_size], &output_data)
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
        let hidden = add_tensors(hidden, &attn_output)?;

        // Pre-MLP RMS norm
        let normed = rms_norm(
            &hidden,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        // MLP
        let mlp_output = self.forward_mlp(&normed, &layer.mlp)?;

        // Residual connection
        add_tensors(&hidden, &mlp_output)
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

/// Linear projection: output = input @ weight^T
/// input: (seq_len, in_features)
/// weight: (out_features, in_features)
/// output: (seq_len, out_features)
fn linear(input: &CudaTensor<f32>, weight: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    let input_shape = input.shape();
    let weight_shape = weight.shape();

    let _seq_len = input_shape[0];
    let in_features = input_shape[1];
    let _out_features = weight_shape[0];

    assert_eq!(
        weight_shape[1], in_features,
        "Weight in_features doesn't match input"
    );

    // Transpose weight: (out_features, in_features) -> (in_features, out_features)
    let weight_t = transpose_2d(weight)?;

    // Matmul: (seq_len, in_features) @ (in_features, out_features) -> (seq_len, out_features)
    matmul(input, &weight_t)
}

/// Transpose a 2D tensor
fn transpose_2d(tensor: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    let shape = tensor.shape();
    assert_eq!(shape.len(), 2);

    let rows = shape[0];
    let cols = shape[1];

    let data = tensor.to_vec()?;
    let mut output_data = vec![0.0_f32; rows * cols];

    for i in 0..rows {
        for j in 0..cols {
            output_data[j * rows + i] = data[i * cols + j];
        }
    }

    CudaTensor::from_slice(tensor.context(), &[cols, rows], &output_data)
}

/// Add two tensors element-wise
fn add_tensors(a: &CudaTensor<f32>, b: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    assert_eq!(a.shape(), b.shape(), "Shapes must match for addition");

    let a_data = a.to_vec()?;
    let b_data = b.to_vec()?;

    let output_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x + y)
        .collect();

    CudaTensor::from_slice(a.context(), a.shape(), &output_data)
}

/// Repeat KV heads for grouped-query attention
fn repeat_kv(tensor: &CudaTensor<f32>, num_repeats: usize) -> Result<CudaTensor<f32>> {
    if num_repeats == 1 {
        return Ok(tensor.clone());
    }

    let shape = tensor.shape();
    let seq_len = shape[0];
    let num_kv_heads = shape[1];
    let head_dim = shape[2];

    let new_num_heads = num_kv_heads * num_repeats;
    let output_shape = [seq_len, new_num_heads, head_dim];

    let data = tensor.to_vec()?;
    let mut output_data = vec![0.0_f32; seq_len * new_num_heads * head_dim];

    for s in 0..seq_len {
        for kv_head in 0..num_kv_heads {
            for repeat in 0..num_repeats {
                let new_head = kv_head * num_repeats + repeat;
                for d in 0..head_dim {
                    let src_idx = s * num_kv_heads * head_dim + kv_head * head_dim + d;
                    let dst_idx = s * new_num_heads * head_dim + new_head * head_dim + d;
                    output_data[dst_idx] = data[src_idx];
                }
            }
        }
    }

    CudaTensor::from_slice(tensor.context(), &output_shape, &output_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_2d() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = CudaTensor::from_slice(&ctx, &[2, 3], &data).unwrap();

        let transposed = transpose_2d(&tensor).unwrap();

        assert_eq!(transposed.shape(), &[3, 2]);

        let result = transposed.to_vec().unwrap();
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_repeat_kv() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // (seq=2, num_kv_heads=2, head_dim=3)
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, // seq=0, head=0
            4.0, 5.0, 6.0, // seq=0, head=1
            7.0, 8.0, 9.0, // seq=1, head=0
            10.0, 11.0, 12.0, // seq=1, head=1
        ];

        let tensor = CudaTensor::from_slice(&ctx, &[2, 2, 3], &data).unwrap();

        let repeated = repeat_kv(&tensor, 2).unwrap();

        assert_eq!(repeated.shape(), &[2, 4, 3]);

        let result = repeated.to_vec().unwrap();

        // Each KV head should be repeated twice
        // head 0 -> heads 0, 1
        // head 1 -> heads 2, 3
        assert_eq!(result[0..3], [1.0, 2.0, 3.0]); // seq=0, head=0
        assert_eq!(result[3..6], [1.0, 2.0, 3.0]); // seq=0, head=1 (repeat of head 0)
        assert_eq!(result[6..9], [4.0, 5.0, 6.0]); // seq=0, head=2 (original head 1)
        assert_eq!(result[9..12], [4.0, 5.0, 6.0]); // seq=0, head=3 (repeat of head 1)
    }
}
