//! Llama model implementation

#![allow(
    clippy::struct_field_names, // _proj suffix is conventional for Llama weights
    clippy::no_effect_underscore_binding,
    clippy::doc_markdown // tensor shape docs trigger false positives
)]

use std::path::Path;

use infernum::cuda::ops::{
    add_inplace, add_rmsnorm, apply_rope, attention, attention_kv, cast_bf16_to_f32,
    cast_f32_to_bf16, embedding_gather, matmul, precompute_rope_cache, quantized_matmul, repeat_kv,
    rms_norm, rms_norm_inplace, swiglu, transpose_2d,
};
use infernum::KvCache;

/// Transpose a weight matrix once, for use in pre-transposed linear projections.
/// (out_features, in_features) -> (in_features, out_features)
fn pretranspose_weight(weight: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    transpose_2d(weight)
}

/// Transpose a bf16 weight matrix on the CPU, then upload to GPU.
///
/// The CUDA transpose kernel only supports f32, so we do this on the host.
/// This is a one-time cost at model load, not on the hot path.
/// (out_features, in_features) -> (in_features, out_features)
fn pretranspose_weight_bf16(weight: &CudaTensor<half::bf16>) -> Result<CudaTensor<half::bf16>> {
    let shape = weight.shape();
    assert_eq!(shape.len(), 2, "Expected 2D tensor for pretranspose");
    let rows = shape[0];
    let cols = shape[1];

    let data = weight.to_vec()?;
    let mut transposed = vec![half::bf16::ZERO; data.len()];

    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = data[r * cols + c];
        }
    }

    CudaTensor::from_slice(weight.context(), &[cols, rows], &transposed)
}

/// Reverse the llama.cpp Q/K weight permutation for f32 tensors.
///
/// GGUF files interleave each head's rows: `[h0, h_half, h1, h_{half+1}, ...]`.
/// This restores the HuggingFace sequential-half layout: `[h0..h_{half-1}, h_half..h_{dim-1}]`.
fn unpermute_f32(weight: &CudaTensor<f32>, n_head: usize) -> Result<CudaTensor<f32>> {
    let shape = weight.shape();
    let n_rows = shape[0];
    let n_cols = shape[1];
    let head_dim = n_rows / n_head;
    let half_dim = head_dim / 2;

    let data = weight.to_vec()?;
    let mut out = vec![0.0_f32; data.len()];

    for h in 0..n_head {
        for i in 0..half_dim {
            let src0 = (h * head_dim + 2 * i) * n_cols;
            let src1 = (h * head_dim + 2 * i + 1) * n_cols;
            let dst0 = (h * head_dim + i) * n_cols;
            let dst1 = (h * head_dim + i + half_dim) * n_cols;
            out[dst0..dst0 + n_cols].copy_from_slice(&data[src0..src0 + n_cols]);
            out[dst1..dst1 + n_cols].copy_from_slice(&data[src1..src1 + n_cols]);
        }
    }

    CudaTensor::from_slice(weight.context(), &[n_rows, n_cols], &out)
}
use infernum::cuda::{CudaContext, CudaTensor, QuantizedTensor};
use infernum::dtype::DType;
use infernum::tensor::Tensor;
use infernum::weights::{GgufLoader, SafeTensorsLoader, WeightLoader};
use infernum::Result;

use crate::LlamaConfig;

/// A linear layer weight that is either an f32 matrix (pre-transposed for
/// standard matmul) or a quantized tensor (dequantized on-the-fly in the kernel).
enum LinearWeight {
    /// Pre-transposed f32 weight: shape (in_features, out_features)
    F32(CudaTensor<f32>),
    /// Pre-transposed bf16 weight: shape (in_features, out_features)
    /// Activations are cast f32→bf16 before GEMM and bf16→f32 after.
    BF16(CudaTensor<half::bf16>),
    /// Quantized weight: shape (out_features, in_features) — transposed inside kernel
    Quantized(QuantizedTensor),
}

/// Weights for a single Llama attention layer
struct LlamaAttentionWeights {
    q_proj: LinearWeight,
    k_proj: LinearWeight,
    v_proj: LinearWeight,
    o_proj: LinearWeight,
}

/// Weights for a single Llama MLP layer
struct LlamaMlpWeights {
    gate_proj: LinearWeight,
    up_proj: LinearWeight,
    down_proj: LinearWeight,
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
    lm_head: LinearWeight,

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

    /// Load a Llama model from a GGUF file containing quantized weights
    ///
    /// # Errors
    /// Returns an error if the file cannot be parsed or weights fail to load
    pub fn from_gguf(ctx: &CudaContext, gguf_path: impl AsRef<Path>) -> Result<Self> {
        let loader = GgufLoader::from_file(gguf_path)?;

        // Extract config from GGUF metadata
        let config = LlamaConfig::from_gguf_metadata(loader.metadata())?;

        Self::load_weights_gguf(ctx, config, &loader)
    }

    /// Load model weights from a GGUF loader, using quantized weights where available
    fn load_weights_gguf(
        ctx: &CudaContext,
        config: LlamaConfig,
        loader: &GgufLoader,
    ) -> Result<Self> {
        /// Load a linear weight — quantized if the tensor uses a quantized dtype,
        /// otherwise f32 (pre-transposed).
        fn load_linear(ctx: &CudaContext, loader: &GgufLoader, name: &str) -> Result<LinearWeight> {
            let dtype = loader.get_dtype(name)?;
            if dtype.is_quantized() {
                Ok(LinearWeight::Quantized(loader.load_quantized(ctx, name)?))
            } else {
                Ok(LinearWeight::F32(pretranspose_weight(
                    &loader.load_f32(ctx, name)?,
                )?))
            }
        }

        /// Load a Q/K projection weight, reversing the llama.cpp interleaved
        /// permutation so that infernum's sequential half-half RoPE is correct.
        fn load_linear_unpermute(
            ctx: &CudaContext,
            loader: &GgufLoader,
            name: &str,
            n_head: usize,
        ) -> Result<LinearWeight> {
            let dtype = loader.get_dtype(name)?;
            if dtype.is_quantized() {
                Ok(LinearWeight::Quantized(
                    loader.load_quantized_unpermute(ctx, name, n_head)?,
                ))
            } else {
                let tensor = loader.load_f32(ctx, name)?;
                let unpermuted = unpermute_f32(&tensor, n_head)?;
                Ok(LinearWeight::F32(pretranspose_weight(&unpermuted)?))
            }
        }

        // Embeddings (always f32/f16)
        let embed_tokens = loader.load_f32(ctx, "token_embd.weight")?;

        // Transformer layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("blk.{i}");

            let layer = LlamaLayerWeights {
                input_layernorm: loader.load_f32(ctx, &format!("{prefix}.attn_norm.weight"))?,
                attention: LlamaAttentionWeights {
                    q_proj: load_linear_unpermute(
                        ctx,
                        loader,
                        &format!("{prefix}.attn_q.weight"),
                        config.num_attention_heads,
                    )?,
                    k_proj: load_linear_unpermute(
                        ctx,
                        loader,
                        &format!("{prefix}.attn_k.weight"),
                        config.num_kv_heads(),
                    )?,
                    v_proj: load_linear(ctx, loader, &format!("{prefix}.attn_v.weight"))?,
                    o_proj: load_linear(ctx, loader, &format!("{prefix}.attn_output.weight"))?,
                },
                post_attention_layernorm: loader
                    .load_f32(ctx, &format!("{prefix}.ffn_norm.weight"))?,
                mlp: LlamaMlpWeights {
                    gate_proj: load_linear(ctx, loader, &format!("{prefix}.ffn_gate.weight"))?,
                    up_proj: load_linear(ctx, loader, &format!("{prefix}.ffn_up.weight"))?,
                    down_proj: load_linear(ctx, loader, &format!("{prefix}.ffn_down.weight"))?,
                },
            };

            layers.push(layer);
        }

        // Final norm
        let norm = loader.load_f32(ctx, "output_norm.weight")?;

        // Output head
        let lm_head = if config.tie_word_embeddings {
            LinearWeight::F32(pretranspose_weight(&embed_tokens)?)
        } else if loader.contains("output.weight") {
            let dtype = loader.get_dtype("output.weight")?;
            if dtype.is_quantized() {
                LinearWeight::Quantized(loader.load_quantized(ctx, "output.weight")?)
            } else {
                LinearWeight::F32(pretranspose_weight(
                    &loader.load_f32(ctx, "output.weight")?,
                )?)
            }
        } else {
            // Fallback: tie to embeddings
            LinearWeight::F32(pretranspose_weight(&embed_tokens)?)
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

    /// Load model weights from a weight loader
    fn load_weights(
        ctx: &CudaContext,
        config: LlamaConfig,
        loader: &impl WeightLoader,
    ) -> Result<Self> {
        /// Load a linear weight — quantized if the tensor uses a quantized dtype,
        /// bf16 if stored as BF16/F16 (keeping half-precision on GPU),
        /// otherwise f32 (pre-transposed). For FP8 weights, also loads the
        /// companion `weight_scale` tensor if present.
        fn load_linear(
            ctx: &CudaContext,
            loader: &impl WeightLoader,
            name: &str,
        ) -> Result<LinearWeight> {
            let dtype = loader.get_dtype(name)?;
            if dtype.is_quantized() {
                let mut qt = loader.load_quantized(ctx, name)?;

                // FP8 models store a per-tensor scale as a sibling tensor
                // e.g. "model.layers.0.self_attn.q_proj.weight" ->
                //      "model.layers.0.self_attn.q_proj.weight_scale"
                let scale_name = format!("{name}_scale");
                if loader.contains(&scale_name) {
                    let scale_tensor = loader.load_f32(ctx, &scale_name)?;
                    let scale_val = scale_tensor.to_vec()?;
                    qt.set_weight_scale(ctx, scale_val[0])?;
                }

                Ok(LinearWeight::Quantized(qt))
            } else if dtype == DType::BF16 || dtype == DType::F16 {
                Ok(LinearWeight::BF16(pretranspose_weight_bf16(
                    &loader.load_bf16(ctx, name)?,
                )?))
            } else {
                Ok(LinearWeight::F32(pretranspose_weight(
                    &loader.load_f32(ctx, name)?,
                )?))
            }
        }

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
                    q_proj: load_linear(ctx, loader, &format!("{prefix}.self_attn.q_proj.weight"))?,
                    k_proj: load_linear(ctx, loader, &format!("{prefix}.self_attn.k_proj.weight"))?,
                    v_proj: load_linear(ctx, loader, &format!("{prefix}.self_attn.v_proj.weight"))?,
                    o_proj: load_linear(ctx, loader, &format!("{prefix}.self_attn.o_proj.weight"))?,
                },
                post_attention_layernorm: loader
                    .load_f32(ctx, &format!("{prefix}.post_attention_layernorm.weight"))?,
                mlp: LlamaMlpWeights {
                    gate_proj: load_linear(ctx, loader, &format!("{prefix}.mlp.gate_proj.weight"))?,
                    up_proj: load_linear(ctx, loader, &format!("{prefix}.mlp.up_proj.weight"))?,
                    down_proj: load_linear(ctx, loader, &format!("{prefix}.mlp.down_proj.weight"))?,
                },
            };

            layers.push(layer);
        }

        // Load final norm
        let norm = loader.load_f32(ctx, "model.norm.weight")?;

        // Load or tie lm_head
        let lm_head = if config.tie_word_embeddings {
            LinearWeight::F32(pretranspose_weight(&embed_tokens)?)
        } else {
            load_linear(ctx, loader, "lm_head.weight")?
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
        for layer in &self.layers {
            hidden = self.forward_layer(&hidden, layer)?;
        }

        // Final layer norm (in-place: hidden is consumed)
        rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;

        // Project to vocabulary: (seq_len, hidden_size) @ (vocab_size, hidden_size)^T -> (seq_len, vocab_size)
        let logits = self.lm_head_forward(&hidden)?;

        Ok(logits)
    }

    /// Forward pass with KV cache (prefill phase)
    ///
    /// Processes the full prompt, populating the KV cache for each layer,
    /// and returns logits for the **last** token only: shape `(1, vocab_size)`.
    ///
    /// After this call, `kv_cache.current_len()` equals `input_ids.len()`.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward_with_kv_cache(
        &self,
        input_ids: &[u32],
        kv_cache: &mut KvCache,
    ) -> Result<CudaTensor<f32>> {
        let seq_len = input_ids.len();
        let position_offset = kv_cache.current_len();

        // Embed tokens: (seq_len,) -> (seq_len, hidden_size)
        let mut hidden = self.embed(input_ids)?;

        // Run through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer_kv(&hidden, layer, layer_idx, kv_cache, position_offset)?;
        }

        // Advance KV cache position (once after all layers)
        kv_cache.advance(seq_len);

        // Final layer norm (in-place: hidden is consumed)
        rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;

        // Extract last hidden state, then project to vocab logits
        let last_hidden = self.extract_last_row(&hidden, seq_len)?;
        self.lm_head_forward(&last_hidden)
    }

    /// Forward pass for a single token with KV cache (decode phase)
    ///
    /// Processes one new token, appending its KV to the cache, and returns
    /// logits of shape `(1, vocab_size)`.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward_next_token(
        &self,
        token_id: u32,
        kv_cache: &mut KvCache,
    ) -> Result<CudaTensor<f32>> {
        self.forward_with_kv_cache(&[token_id], kv_cache)
    }

    /// Forward pass through a single transformer layer using KV cache
    fn forward_layer_kv(
        &self,
        hidden: &CudaTensor<f32>,
        layer: &LlamaLayerWeights,
        layer_idx: usize,
        kv_cache: &mut KvCache,
        position_offset: usize,
    ) -> Result<CudaTensor<f32>> {
        // Pre-attention RMS norm
        let normed = rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

        // Self-attention with KV cache
        let attn_output = self.forward_attention_kv(
            &normed,
            &layer.attention,
            layer_idx,
            kv_cache,
            position_offset,
        )?;

        // Residual add + pre-MLP RMS norm (fused in release builds)
        let (mut hidden, normed) = add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        // MLP
        let mlp_output = self.forward_mlp(&normed, &layer.mlp)?;

        // Residual connection (in-place: hidden += mlp_output)
        add_inplace(&mut hidden, &mlp_output)?;
        Ok(hidden)
    }

    /// Forward pass through attention with KV cache
    fn forward_attention_kv(
        &self,
        hidden: &CudaTensor<f32>,
        weights: &LlamaAttentionWeights,
        layer_idx: usize,
        kv_cache: &mut KvCache,
        position_offset: usize,
    ) -> Result<CudaTensor<f32>> {
        let seq_len = hidden.shape()[0];
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads();
        let head_dim = self.config.head_dim();

        // Project Q, K, V
        let q = linear(hidden, &weights.q_proj)?;
        let k = linear(hidden, &weights.k_proj)?;
        let v = linear(hidden, &weights.v_proj)?;

        // Reshape: (seq_len, num_heads * head_dim) -> (seq_len, num_heads, head_dim)
        let q = q.reshape(&[seq_len, num_heads, head_dim]);
        let k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
        let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

        // Apply RoPE with position offset
        let q = apply_rope(&q, &self.cos_cache, &self.sin_cache, position_offset)?;
        let k = apply_rope(&k, &self.cos_cache, &self.sin_cache, position_offset)?;

        // Compute attention using KV cache (handles GQA repeat internally)
        let attn_output = attention_kv(&q, kv_cache, layer_idx, &k, &v)?;

        // Reshape back: (seq_len, num_heads, head_dim) -> (seq_len, hidden_size)
        let attn_output = attn_output.reshape(&[seq_len, hidden_size]);

        // Output projection
        linear(&attn_output, &weights.o_proj)
    }

    /// Extract the last row from a (seq_len, hidden_size) tensor
    fn extract_last_row(
        &self,
        hidden: &CudaTensor<f32>,
        seq_len: usize,
    ) -> Result<CudaTensor<f32>> {
        if seq_len == 1 {
            return Ok(hidden.reshape(&[1, self.config.hidden_size]));
        }
        // hidden is already (seq_len, hidden_size) after final norm
        // We need the last row as (1, hidden_size)
        let hidden_size = hidden.shape()[1];
        let flat = hidden.reshape(&[seq_len * hidden_size]);
        let mut out = unsafe { CudaTensor::<f32>::uninit(&self.ctx, &[1, hidden_size])? };
        // Copy last hidden_size elements
        // We can't easily sub-slice with dtod_copy. Use a simple copy kernel.
        // Actually, we have the hidden states before lm_head. Let me rethink.
        // The cleanest approach: don't call lm_head on full logits, instead
        // extract last hidden BEFORE lm_head.
        // But we already computed lm_head above... Let's use a different approach.
        // We'll use a CUDA offset copy.
        let device = self.ctx.device();
        let src = flat.cuda_slice();
        let last_offset = (seq_len - 1) * hidden_size;
        // cudarc CudaSlice supports slice() for sub-slicing
        let src_sub = src.slice(last_offset..seq_len * hidden_size);
        device.dtod_copy(&src_sub, out.cuda_slice_mut())?;
        Ok(out)
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
    ) -> Result<CudaTensor<f32>> {
        let _seq_len = hidden.shape()[0];
        let _hidden_size = self.config.hidden_size;

        // Pre-attention RMS norm
        let normed = rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

        // Self-attention
        let attn_output = self.forward_attention(&normed, &layer.attention)?;

        // Residual add + pre-MLP RMS norm (fused in release builds)
        let (mut hidden, normed) = add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        // MLP
        let mlp_output = self.forward_mlp(&normed, &layer.mlp)?;

        // Residual connection (in-place: hidden += mlp_output)
        add_inplace(&mut hidden, &mlp_output)?;
        Ok(hidden)
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

        // SwiGLU activation (fused in release builds)
        let intermediate = swiglu(&gate, &up)?;

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

impl infernum::Model for LlamaModel {
    fn config(&self) -> infernum::ModelConfig {
        let config = self.config();
        infernum::ModelConfig {
            num_layers: config.num_hidden_layers,
            max_seq_len: config.max_position_embeddings,
            num_kv_heads: config.num_kv_heads(),
            head_dim: config.head_dim(),
            eos_token_id: config.eos_token_id,
        }
    }

    fn forward(&self, input_ids: &[u32]) -> Result<CudaTensor<f32>> {
        self.forward(input_ids)
    }

    fn forward_with_kv_cache(
        &self,
        input_ids: &[u32],
        kv_cache: &mut KvCache,
    ) -> Result<CudaTensor<f32>> {
        self.forward_with_kv_cache(input_ids, kv_cache)
    }

    fn forward_next_token(&self, token_id: u32, kv_cache: &mut KvCache) -> Result<CudaTensor<f32>> {
        self.forward_next_token(token_id, kv_cache)
    }
}

/// Linear projection: output = input @ weight
///
/// For `F32` weights: pre-transposed as (in_features, out_features), uses standard matmul.
/// For `Quantized` weights: stored as (out_features, in_features), dequantized on-the-fly.
fn linear(input: &CudaTensor<f32>, weight: &LinearWeight) -> Result<CudaTensor<f32>> {
    match weight {
        LinearWeight::F32(w) => matmul(input, w),
        LinearWeight::BF16(w) => {
            let input_bf16 = cast_f32_to_bf16(input)?;
            let output_bf16 = matmul(&input_bf16, w)?;
            cast_bf16_to_f32(&output_bf16)
        }
        LinearWeight::Quantized(w) => quantized_matmul(input, w),
    }
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
        let weight =
            LinearWeight::F32(CudaTensor::from_slice(&ctx, &[3, 4], &weight_data).unwrap());

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

    #[test]
    fn test_linear_bf16() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // input: (2, 3), weight pre-transposed: (3, 4) -> output: (2, 4)
        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight_data: Vec<half::bf16> = vec![
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(0.0),
            half::bf16::from_f32(0.0),
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(0.0),
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(0.0),
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(0.0),
            half::bf16::from_f32(0.0),
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(1.0),
        ];

        let input = CudaTensor::from_slice(&ctx, &[2, 3], &input_data).unwrap();
        let weight =
            LinearWeight::BF16(CudaTensor::from_slice(&ctx, &[3, 4], &weight_data).unwrap());

        let output = linear(&input, &weight).unwrap();

        assert_eq!(output.shape(), &[2, 4]);

        let result = output.to_vec().unwrap();
        // Same expected results as f32 test: row 0: [1, 2, 3, 6], row 1: [4, 5, 6, 15]
        assert!((result[0] - 1.0).abs() < 0.1);
        assert!((result[1] - 2.0).abs() < 0.1);
        assert!((result[2] - 3.0).abs() < 0.1);
        assert!((result[3] - 6.0).abs() < 0.1);
        assert!((result[4] - 4.0).abs() < 0.1);
        assert!((result[5] - 5.0).abs() < 0.1);
        assert!((result[6] - 6.0).abs() < 0.1);
        assert!((result[7] - 15.0).abs() < 0.2);
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

    fn build_tiny_engine(ctx: &CudaContext) -> infernum_runtime::Engine<LlamaModel> {
        let model = build_tiny_model(ctx);
        infernum_runtime::Engine::new(ctx, model).expect("Failed to build engine")
    }

    #[test]
    fn test_generate_respects_max_tokens() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let mut engine = build_tiny_engine(&ctx);

        let prompt = vec![1_u32, 5, 10];
        let max_new = 4;
        let options = infernum::GenerateOptions {
            max_new_tokens: max_new,
            use_kv_cache: false,
            ..Default::default()
        };
        let tokens = engine.generate(&prompt, &options).unwrap();

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
        let mut engine = build_tiny_engine(&ctx);

        // Run generate with a very large max_new_tokens but EOS set to a
        // token that the model is likely to produce with random weights
        let prompt = vec![1_u32];
        let options_no_eos = infernum::GenerateOptions {
            max_new_tokens: 5,
            use_kv_cache: false,
            ..Default::default()
        };
        let result_no_eos = engine.generate(&prompt, &options_no_eos).unwrap();

        // The generated tokens should not include the EOS token if we set it
        // to one that was actually produced
        if result_no_eos.len() > 1 {
            let first_generated = result_no_eos[1];
            let options_with_eos = infernum::GenerateOptions {
                max_new_tokens: 100,
                eos_token_id: Some(first_generated),
                use_kv_cache: false,
                ..Default::default()
            };
            let result_with_eos = engine.generate(&prompt, &options_with_eos).unwrap();
            // Should stop immediately since the first generated token == EOS
            assert_eq!(
                result_with_eos.len(),
                prompt.len(),
                "Should stop before appending the EOS token"
            );
        }
    }

    #[test]
    fn test_forward_with_kv_cache_output_shape() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_model(&ctx);
        let config = tiny_config();

        let mut kv_cache = KvCache::new(
            &ctx,
            config.num_hidden_layers,
            config.max_position_embeddings,
            config.num_kv_heads(),
            config.head_dim(),
        )
        .unwrap();

        let input_ids: Vec<u32> = vec![1, 5, 10];
        let logits = model
            .forward_with_kv_cache(&input_ids, &mut kv_cache)
            .expect("Forward with KV cache failed");

        // Returns (1, vocab_size) — logits for last token only
        assert_eq!(logits.shape(), &[1, config.vocab_size]);
        assert_eq!(kv_cache.current_len(), input_ids.len());
    }

    #[test]
    fn test_forward_next_token_output_shape() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_model(&ctx);
        let config = tiny_config();

        let mut kv_cache = KvCache::new(
            &ctx,
            config.num_hidden_layers,
            config.max_position_embeddings,
            config.num_kv_heads(),
            config.head_dim(),
        )
        .unwrap();

        // Prefill
        let prompt: Vec<u32> = vec![1, 5, 10];
        model.forward_with_kv_cache(&prompt, &mut kv_cache).unwrap();

        // Decode one token
        let logits = model
            .forward_next_token(42, &mut kv_cache)
            .expect("Forward next token failed");

        assert_eq!(logits.shape(), &[1, config.vocab_size]);
        assert_eq!(kv_cache.current_len(), prompt.len() + 1);
    }

    #[test]
    fn test_kv_cache_reset_reuse() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_model(&ctx);
        let config = tiny_config();

        let mut kv_cache = KvCache::new(
            &ctx,
            config.num_hidden_layers,
            config.max_position_embeddings,
            config.num_kv_heads(),
            config.head_dim(),
        )
        .unwrap();

        // First sequence
        let prompt1: Vec<u32> = vec![1, 5, 10];
        model
            .forward_with_kv_cache(&prompt1, &mut kv_cache)
            .unwrap();
        assert_eq!(kv_cache.current_len(), 3);

        // Reset and process a different sequence
        kv_cache.reset();
        assert_eq!(kv_cache.current_len(), 0);

        let prompt2: Vec<u32> = vec![2, 3];
        let logits = model
            .forward_with_kv_cache(&prompt2, &mut kv_cache)
            .unwrap();
        assert_eq!(logits.shape(), &[1, config.vocab_size]);
        assert_eq!(kv_cache.current_len(), 2);
    }

    #[test]
    fn test_kv_cache_generate_matches_naive_generate() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let mut engine = build_tiny_engine(&ctx);

        let prompt: Vec<u32> = vec![1, 5, 10];
        let max_new = 5;

        // Greedy generate without KV cache (recomputes full sequence each step)
        let naive_options = infernum::GenerateOptions {
            max_new_tokens: max_new,
            use_kv_cache: false,
            ..Default::default()
        };
        let tokens_naive = engine.generate(&prompt, &naive_options).unwrap();

        // Greedy generate with KV cache
        let kv_options = infernum::GenerateOptions {
            max_new_tokens: max_new,
            use_kv_cache: true,
            ..Default::default()
        };
        let tokens_kv = engine.generate(&prompt, &kv_options).unwrap();

        // Both should produce the exact same token sequence
        assert_eq!(
            tokens_naive, tokens_kv,
            "KV-cache generate should match naive generate"
        );
    }
}
