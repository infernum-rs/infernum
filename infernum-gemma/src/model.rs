//! Gemma model implementation — fully generic over the compute backend `B`.
//!
//! Supports Gemma 2 (`model_type: "gemma2"`) and Gemma 3 text
//! (`model_type: "gemma3_text"`). A single [`GemmaModel`] handles both
//! generations, with Gemma 3 differences (QK-norm, dual-theta RoPE,
//! no logit soft-capping) toggled by config fields.

#![allow(
    clippy::struct_field_names,
    clippy::no_effect_underscore_binding,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::module_name_repetitions,
    clippy::manual_div_ceil,
    dead_code,
    unused_mut
)]

use std::marker::PhantomData;
use std::path::Path;

use infernum::backend::{
    ArithOps, AttentionOps, Backend, CastOps, EmbedOps, GegluOps, MatmulExtOps, MatmulOps, NormOps,
    PagedAttentionOps, PagedKvCacheOps, RopeOps, TensorDataOps, TensorFactory, TensorOps,
};
use infernum::block_allocator::BlockTable;
use infernum::dtype::DType;
use infernum::shard::GpuConfig;
use infernum::tensor::Tensor;
use infernum::transformer::{self, GateUpWeight, KvProjWeight};
use infernum::Result;

use crate::GemmaConfig;

// --- Weight structures ---

struct GemmaAttentionWeights<B: Backend + MatmulOps> {
    q_proj: <B as MatmulOps>::LinearWeight,
    kv_proj: KvProjWeight<B>,
    o_proj: <B as MatmulOps>::LinearWeight,
    q_norm: Option<B::Tensor>,
    k_norm: Option<B::Tensor>,
}

type GemmaMlpWeights<B> = transformer::MlpWeights<B>;

struct GemmaLayerWeights<B: Backend + MatmulOps> {
    input_layernorm: B::Tensor,
    post_attention_layernorm: B::Tensor,
    pre_feedforward_layernorm: B::Tensor,
    post_feedforward_layernorm: B::Tensor,
    attention: GemmaAttentionWeights<B>,
    mlp: GemmaMlpWeights<B>,
}

/// Gemma model supporting both Gemma 2 and Gemma 3 text architectures.
pub struct GemmaModel<B: Backend + MatmulOps> {
    config: GemmaConfig,
    device: B::DeviceHandle,
    #[allow(dead_code)]
    gpu_config: GpuConfig,

    /// Optional communicator for tensor-parallel all-reduce.
    comm: Option<B::Comm>,

    tp_num_heads: usize,
    tp_num_kv_heads: usize,
    dtype: DType,

    embed_tokens: B::Tensor,
    layers: Vec<GemmaLayerWeights<B>>,
    norm: B::Tensor,
    lm_head: <B as MatmulOps>::LinearWeight,

    /// Embedding scale factor: sqrt(hidden_size)
    embed_scale: f32,

    /// Attention scale: 1 / sqrt(query_pre_attn_scalar)
    attn_scale: f32,

    // RoPE caches — Gemma 2: single set, Gemma 3: two sets (local + global)
    cos_cache: B::Tensor,
    sin_cache: B::Tensor,
    // Gemma 3 dual-theta RoPE: separate cache for full-attention layers
    cos_cache_global: Option<B::Tensor>,
    sin_cache_global: Option<B::Tensor>,

    _backend: PhantomData<B>,
}

// ---- Trait alias ----

/// Convenience alias: all op traits required by `GemmaModel` forward methods.
pub trait GemmaOps:
    Backend
    + MatmulOps
    + MatmulExtOps
    + NormOps
    + ArithOps
    + GegluOps
    + CastOps
    + EmbedOps
    + TensorOps
    + TensorFactory
    + TensorDataOps
    + RopeOps
    + AttentionOps
    + PagedAttentionOps
    + PagedKvCacheOps
{
}

impl<B> GemmaOps for B where
    B: Backend
        + MatmulOps
        + MatmulExtOps
        + NormOps
        + ArithOps
        + GegluOps
        + CastOps
        + EmbedOps
        + TensorOps
        + TensorFactory
        + TensorDataOps
        + RopeOps
        + AttentionOps
        + PagedAttentionOps
        + PagedKvCacheOps
{
}

// ---- Gemma-specific norm loading ----

/// Load a Gemma RMSNorm weight and add 1.0 to every element.
///
/// Gemma's RMSNorm computes `x_normed * (1 + weight)` (weights initialized to
/// zeros), unlike Llama's `x_normed * weight` (weights initialized to ones).
/// By pre-adding 1.0 at load time, we can reuse the standard RMSNorm kernel.
fn load_gemma_norm<B: MatmulOps + CastOps + TensorDataOps + TensorFactory>(
    loader: &impl infernum::WeightLoader<B>,
    device: &B::DeviceHandle,
    name: &str,
    dtype: DType,
) -> Result<B::Tensor> {
    let gpu_tensor = loader.load_tensor(name, dtype)?;
    let f32_tensor = B::cast_to_f32(&gpu_tensor)?;
    let mut host: Vec<f32> = B::to_f32_vec(&f32_tensor)?;
    for v in &mut host {
        *v += 1.0;
    }
    let adjusted = B::from_f32_slice(device, gpu_tensor.shape(), &host)?;
    B::cast_from_f32(&adjusted, dtype)
}

// ---- Generic forward pass methods ----

impl<B: GemmaOps> GemmaModel<B> {
    /// Access the model configuration
    #[must_use]
    pub fn config(&self) -> &GemmaConfig {
        &self.config
    }

    /// Get the model's compute dtype
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Build the runtime-facing [`ModelConfig`](infernum::ModelConfig).
    #[must_use]
    pub fn model_config(&self) -> infernum::ModelConfig {
        infernum::ModelConfig {
            num_layers: self.config.num_hidden_layers,
            max_seq_len: self.config.max_position_embeddings,
            num_kv_heads: self.tp_num_kv_heads,
            head_dim: self.config.head_dim,
            eos_token_id: self.config.eos_token_id,
            cache_dtype: self.dtype,
        }
    }

    // ---- Embedding / LM head helpers ----

    fn embed(&self, input_ids: &[u32]) -> Result<B::Tensor> {
        transformer::embed::<B>(&self.embed_tokens, input_ids)
    }

    fn lm_head_forward(&self, hidden: &B::Tensor) -> Result<B::Tensor> {
        let mut logits = transformer::lm_head_forward::<B>(hidden, &self.lm_head, self.dtype)?;
        self.apply_final_softcap(&mut logits)?;
        Ok(logits)
    }

    /// Apply final logit soft-capping (Gemma 2 only): tanh(logits / cap) * cap
    fn apply_final_softcap(&self, logits: &mut B::Tensor) -> Result<()> {
        if let Some(cap) = self.config.final_logit_softcapping {
            let data = B::to_f32_vec(logits)?;
            let capped: Vec<f32> = data.iter().map(|&x| (x / cap).tanh() * cap).collect();
            *logits = B::from_f32_slice(&self.device, logits.shape(), &capped)?;
        }
        Ok(())
    }

    fn maybe_all_reduce(&self, tensor: &mut B::Tensor) -> Result<()> {
        transformer::maybe_all_reduce::<B>(self.comm.as_ref(), tensor)
    }

    fn rope_caches_for_layer(&self, layer_idx: usize) -> (&B::Tensor, &B::Tensor) {
        if let (Some(ref cos_g), Some(ref sin_g)) = (&self.cos_cache_global, &self.sin_cache_global)
        {
            if self.config.effective_sliding_window(layer_idx).is_none() {
                return (cos_g, sin_g);
            }
        }
        (&self.cos_cache, &self.sin_cache)
    }

    // ---- GeGLU MLP ----

    fn forward_mlp(&self, hidden: &B::Tensor, weights: &GemmaMlpWeights<B>) -> Result<B::Tensor> {
        let (gate, up) = transformer::compute_gate_up::<B>(hidden, &weights.gate_up)?;
        let intermediate = B::geglu(&gate, &up)?;
        let mut out = B::linear(&intermediate, &weights.down_proj)?;
        self.maybe_all_reduce(&mut out)?;
        Ok(out)
    }

    // ---- Full forward (no KV cache) ----

    /// Full forward pass without KV cache (recomputes everything).
    ///
    /// Returns raw logits as a tensor of shape `(seq_len, vocab_size)`.
    ///
    /// # Errors
    /// Returns an error if any operation fails.
    pub fn forward_full(&self, input_ids: &[u32]) -> Result<B::Tensor> {
        let seq_len = input_ids.len();
        let mut hidden = self.embed(input_ids)?;
        B::scale_inplace(&mut hidden, self.embed_scale)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let normed = B::rms_norm(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            let num_heads = self.tp_num_heads;
            let num_kv_heads = self.tp_num_kv_heads;
            let head_dim = self.config.head_dim;
            let sliding_window = self.config.effective_sliding_window(layer_idx);

            let q = B::linear(&normed, &layer.attention.q_proj)?;
            let (k, v) = transformer::compute_kv_proj::<B>(&normed, &layer.attention.kv_proj)?;

            let mut q = q.reshape(&[seq_len, num_heads, head_dim]);
            let mut k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
            let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

            transformer::apply_qk_norm::<B>(
                &mut q,
                &mut k,
                layer.attention.q_norm.as_ref(),
                layer.attention.k_norm.as_ref(),
                num_heads,
                num_kv_heads,
                head_dim,
                self.config.rms_norm_eps,
            )?;

            let (cos, sin) = self.rope_caches_for_layer(layer_idx);
            let q = B::apply_rope(&q, cos, sin, 0)?;
            let k = B::apply_rope(&k, cos, sin, 0)?;

            let attn_output = B::fused_attention_prefill(
                &q,
                &k,
                &v,
                0,
                Some(self.attn_scale),
                self.config.attn_logit_softcapping,
                sliding_window,
            )?;
            let attn_output = attn_output.reshape(&[seq_len, num_heads * head_dim]);
            let mut attn_output = B::linear(&attn_output, &layer.attention.o_proj)?;
            self.maybe_all_reduce(&mut attn_output)?;

            // Post-attention norm + residual
            let post_attn = B::rms_norm(
                &attn_output,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;
            B::add_inplace(&mut hidden, &post_attn)?;

            // Pre-feedforward norm
            let normed_ffn = B::rms_norm(
                &hidden,
                &layer.pre_feedforward_layernorm,
                self.config.rms_norm_eps,
            )?;

            let mlp_output = self.forward_mlp(&normed_ffn, &layer.mlp)?;

            // Post-feedforward norm + residual
            let post_ffn = B::rms_norm(
                &mlp_output,
                &layer.post_feedforward_layernorm,
                self.config.rms_norm_eps,
            )?;
            B::add_inplace(&mut hidden, &post_ffn)?;
        }

        B::rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;
        self.lm_head_forward(&hidden)
    }

    // ---- Paged KV cache forward passes ----

    /// Batched decode with host-side inputs.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward_batch_decode(
        &self,
        token_ids: &[u32],
        paged_kv: &mut B::PagedKvCache,
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<B::Tensor> {
        transformer::forward_batch_decode_host::<B, _>(
            &self.device,
            token_ids,
            block_tables,
            positions,
            |tid, bt, sl, pos, bs, mbps, msl| {
                self.forward_batch_decode_tensors(tid, paged_kv, bt, sl, pos, bs, mbps, msl)
            },
        )
    }

    /// Batched decode forward pass with paged KV cache (device tensors).
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batch_decode_tensors(
        &self,
        token_ids: &B::Tensor,
        paged_kv: &mut B::PagedKvCache,
        block_tables: &B::Tensor,
        seq_lens: &B::Tensor,
        positions: &B::Tensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
    ) -> Result<B::Tensor> {
        let mut hidden = B::embedding_gather_tensor(&self.embed_tokens, token_ids, batch_size)?;
        B::scale_inplace(&mut hidden, self.embed_scale)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer_paged_decode_batched(
                &hidden,
                layer,
                layer_idx,
                paged_kv,
                block_tables,
                seq_lens,
                positions,
                batch_size,
                max_blocks_per_seq,
                max_seq_len,
            )?;
        }

        B::rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;
        self.lm_head_forward(&hidden.reshape(&[batch_size, self.config.hidden_size]))
    }

    /// Single-sequence prefill with paged KV cache.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward_prefill_paged(
        &self,
        input_ids: &[u32],
        paged_kv: &mut B::PagedKvCache,
        block_table: &BlockTable,
        start_pos: usize,
    ) -> Result<B::Tensor> {
        let seq_len = input_ids.len();

        let mut hidden = self.embed(input_ids)?;
        B::scale_inplace(&mut hidden, self.embed_scale)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer_paged_prefill(
                &hidden,
                layer,
                layer_idx,
                paged_kv,
                block_table,
                start_pos,
                seq_len,
            )?;
        }

        B::rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;
        let last_hidden = transformer::extract_last_row::<B>(&hidden, seq_len);
        self.lm_head_forward(&last_hidden)
    }

    // ---- Per-layer forward passes ----

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_paged_decode_batched(
        &self,
        hidden: &B::Tensor,
        layer: &GemmaLayerWeights<B>,
        layer_idx: usize,
        paged_kv: &mut B::PagedKvCache,
        block_tables: &B::Tensor,
        seq_lens: &B::Tensor,
        positions: &B::Tensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
    ) -> Result<B::Tensor> {
        let normed = B::rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

        let attn_output = self.forward_attention_paged_decode_batched(
            &normed,
            &layer.attention,
            layer_idx,
            paged_kv,
            block_tables,
            seq_lens,
            positions,
            batch_size,
            max_blocks_per_seq,
            max_seq_len,
        )?;

        let post_attn = B::rms_norm(
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;
        let mut hidden = hidden.clone();
        B::add_inplace(&mut hidden, &post_attn)?;

        let normed_ffn = B::rms_norm(
            &hidden,
            &layer.pre_feedforward_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_mlp(&normed_ffn, &layer.mlp)?;

        let post_ffn = B::rms_norm(
            &mlp_output,
            &layer.post_feedforward_layernorm,
            self.config.rms_norm_eps,
        )?;
        B::add_inplace(&mut hidden, &post_ffn)?;

        Ok(hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attention_paged_decode_batched(
        &self,
        hidden: &B::Tensor,
        weights: &GemmaAttentionWeights<B>,
        layer_idx: usize,
        paged_kv: &mut B::PagedKvCache,
        block_tables: &B::Tensor,
        seq_lens: &B::Tensor,
        positions: &B::Tensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
    ) -> Result<B::Tensor> {
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim;

        let q = B::linear(hidden, &weights.q_proj)?;
        let (k, v) =
            transformer::compute_kv_proj_decode::<B>(hidden, &weights.kv_proj, batch_size)?;

        let mut q = q.reshape(&[batch_size, num_heads, head_dim]);
        let mut k = k.reshape(&[batch_size, num_kv_heads, head_dim]);
        let v = v.reshape(&[batch_size, num_kv_heads, head_dim]);

        transformer::apply_qk_norm::<B>(
            &mut q,
            &mut k,
            weights.q_norm.as_ref(),
            weights.k_norm.as_ref(),
            num_heads,
            num_kv_heads,
            head_dim,
            self.config.rms_norm_eps,
        )?;

        let (cos, sin) = self.rope_caches_for_layer(layer_idx);
        let q = B::apply_rope_batched(&q, cos, sin, positions, batch_size)?;
        let k = B::apply_rope_batched(&k, cos, sin, positions, batch_size)?;

        B::append_paged_batched(
            paged_kv,
            layer_idx,
            &k,
            &v,
            block_tables,
            positions,
            batch_size,
            max_blocks_per_seq,
        )?;

        let sliding_window = self.config.effective_sliding_window(layer_idx);
        let (k_pool, v_pool) = B::get_pools(paged_kv, layer_idx);
        let attn_output = B::paged_attention_decode(
            &q,
            k_pool,
            v_pool,
            block_tables,
            seq_lens,
            B::block_size(paged_kv),
            max_blocks_per_seq,
            max_seq_len,
            None,
            self.config.attn_logit_softcapping,
            sliding_window,
        )?;

        let attn_output = attn_output.reshape(&[batch_size, num_heads * head_dim]);
        let mut out = B::linear(&attn_output, &weights.o_proj)?;
        self.maybe_all_reduce(&mut out)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_paged_prefill(
        &self,
        hidden: &B::Tensor,
        layer: &GemmaLayerWeights<B>,
        layer_idx: usize,
        paged_kv: &mut B::PagedKvCache,
        block_table: &BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<B::Tensor> {
        let normed = B::rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

        let attn_output = self.forward_attention_paged_prefill(
            &normed,
            &layer.attention,
            layer_idx,
            paged_kv,
            block_table,
            start_pos,
            seq_len,
        )?;

        let post_attn = B::rms_norm(
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;
        let mut hidden = hidden.clone();
        B::add_inplace(&mut hidden, &post_attn)?;

        let normed_ffn = B::rms_norm(
            &hidden,
            &layer.pre_feedforward_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_mlp(&normed_ffn, &layer.mlp)?;

        let post_ffn = B::rms_norm(
            &mlp_output,
            &layer.post_feedforward_layernorm,
            self.config.rms_norm_eps,
        )?;
        B::add_inplace(&mut hidden, &post_ffn)?;

        Ok(hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attention_paged_prefill(
        &self,
        hidden: &B::Tensor,
        weights: &GemmaAttentionWeights<B>,
        layer_idx: usize,
        paged_kv: &mut B::PagedKvCache,
        block_table: &BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<B::Tensor> {
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim;

        let q = B::linear(hidden, &weights.q_proj)?;
        let (k, v) = transformer::compute_kv_proj::<B>(hidden, &weights.kv_proj)?;

        let mut q = q.reshape(&[seq_len, num_heads, head_dim]);
        let mut k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
        let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

        transformer::apply_qk_norm::<B>(
            &mut q,
            &mut k,
            weights.q_norm.as_ref(),
            weights.k_norm.as_ref(),
            num_heads,
            num_kv_heads,
            head_dim,
            self.config.rms_norm_eps,
        )?;

        let (cos, sin) = self.rope_caches_for_layer(layer_idx);
        let q = B::apply_rope(&q, cos, sin, start_pos)?;
        let k = B::apply_rope(&k, cos, sin, start_pos)?;

        B::append_paged(paged_kv, layer_idx, block_table, &k, &v, start_pos)?;

        let mut gather_table = block_table.clone();
        gather_table.advance(seq_len);
        let (k_contig, v_contig) = B::gather_paged_kv(paged_kv, layer_idx, &gather_table)?;

        let sliding_window = self.config.effective_sliding_window(layer_idx);
        let attn_output = B::fused_attention_prefill(
            &q,
            &k_contig,
            &v_contig,
            start_pos,
            Some(self.attn_scale),
            self.config.attn_logit_softcapping,
            sliding_window,
        )?;

        let attn_output = attn_output.reshape(&[seq_len, num_heads * head_dim]);
        let mut out = B::linear(&attn_output, &weights.o_proj)?;
        self.maybe_all_reduce(&mut out)?;
        Ok(out)
    }

    // ---- Generic weight loading ----

    /// Load model weights from a backend-agnostic weight loader.
    ///
    /// # Errors
    /// Returns an error if any weight fails to load.
    #[allow(clippy::too_many_lines)]
    pub fn load_weights(
        device: B::DeviceHandle,
        config: GemmaConfig,
        loader: &impl infernum::WeightLoader<B>,
    ) -> Result<Self> {
        let qc = config.quantization_config.as_ref();

        let embed_dtype = loader.get_dtype("model.embed_tokens.weight")?;
        let dtype = if embed_dtype.is_quantized() {
            DType::F32
        } else {
            embed_dtype
        };

        let embed_tokens = loader.load_tensor("model.embed_tokens.weight", dtype)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            // Load attention weights
            let k = loader.load_linear(&format!("{prefix}.self_attn.k_proj.weight"), dtype, qc)?;
            let v = loader.load_linear(&format!("{prefix}.self_attn.v_proj.weight"), dtype, qc)?;
            let kv_dim = config.num_key_value_heads * config.head_dim;
            let kv_proj = if B::is_dense_weight(&k) && B::is_dense_weight(&v) {
                let k_t = B::as_dense_weight(&k).expect("checked dense");
                let v_t = B::as_dense_weight(&v).expect("checked dense");
                KvProjWeight::<B>::Fused {
                    kv_dim,
                    weight: B::concat_inner_dim(k_t, v_t)?,
                }
            } else {
                KvProjWeight::<B>::Separate {
                    k_proj: Box::new(k),
                    v_proj: Box::new(v),
                }
            };

            // Load QK-norm weights (Gemma 3 only)
            let q_norm_name = format!("{prefix}.self_attn.q_norm.weight");
            let k_norm_name = format!("{prefix}.self_attn.k_norm.weight");
            let q_norm = if loader.contains(&q_norm_name) {
                Some(load_gemma_norm::<B>(loader, &device, &q_norm_name, dtype)?)
            } else {
                None
            };
            let k_norm = if loader.contains(&k_norm_name) {
                Some(load_gemma_norm::<B>(loader, &device, &k_norm_name, dtype)?)
            } else {
                None
            };

            // Load MLP weights (GeGLU: gate_proj, up_proj, down_proj)
            let gate = loader.load_linear(&format!("{prefix}.mlp.gate_proj.weight"), dtype, qc)?;
            let up = loader.load_linear(&format!("{prefix}.mlp.up_proj.weight"), dtype, qc)?;
            let gate_up = if B::is_dense_weight(&gate) && B::is_dense_weight(&up) {
                let g = B::as_dense_weight(&gate).expect("checked dense");
                let u = B::as_dense_weight(&up).expect("checked dense");
                GateUpWeight::<B>::Fused {
                    weight: B::concat_inner_dim(g, u)?,
                    intermediate_size: config.intermediate_size,
                }
            } else {
                GateUpWeight::<B>::Separate {
                    gate_proj: Box::new(gate),
                    up_proj: Box::new(up),
                }
            };

            let layer = GemmaLayerWeights {
                input_layernorm: load_gemma_norm::<B>(
                    loader,
                    &device,
                    &format!("{prefix}.input_layernorm.weight"),
                    dtype,
                )?,
                post_attention_layernorm: load_gemma_norm::<B>(
                    loader,
                    &device,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    dtype,
                )?,
                pre_feedforward_layernorm: load_gemma_norm::<B>(
                    loader,
                    &device,
                    &format!("{prefix}.pre_feedforward_layernorm.weight"),
                    dtype,
                )?,
                post_feedforward_layernorm: load_gemma_norm::<B>(
                    loader,
                    &device,
                    &format!("{prefix}.post_feedforward_layernorm.weight"),
                    dtype,
                )?,
                attention: GemmaAttentionWeights {
                    q_proj: loader.load_linear(
                        &format!("{prefix}.self_attn.q_proj.weight"),
                        dtype,
                        qc,
                    )?,
                    kv_proj,
                    o_proj: loader.load_linear(
                        &format!("{prefix}.self_attn.o_proj.weight"),
                        dtype,
                        qc,
                    )?,
                    q_norm,
                    k_norm,
                },
                mlp: GemmaMlpWeights {
                    gate_up,
                    down_proj: loader.load_linear(
                        &format!("{prefix}.mlp.down_proj.weight"),
                        dtype,
                        qc,
                    )?,
                },
            };

            layers.push(layer);
        }

        let norm = load_gemma_norm::<B>(loader, &device, "model.norm.weight", dtype)?;

        // Gemma always ties lm_head to embed_tokens
        let lm_head =
            transformer::load_lm_head::<B>(&device, loader, &embed_tokens, true, dtype, qc)?;

        // Precompute RoPE caches
        // For Gemma 2: single cache with rope_theta
        // For Gemma 3: two caches — local (rope_local_base_freq) and global (rope_theta)
        let local_theta = config.rope_local_base_freq.unwrap_or(config.rope_theta);
        let (cos_cache, sin_cache) = transformer::build_rope_cache::<B>(
            &device,
            config.head_dim,
            config.max_position_embeddings,
            local_theta,
            None,
            dtype,
        )?;

        let (cos_cache_global, sin_cache_global) = if config.rope_local_base_freq.is_some() {
            let (cos_g, sin_g) = transformer::build_rope_cache::<B>(
                &device,
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                None,
                dtype,
            )?;
            (Some(cos_g), Some(sin_g))
        } else {
            (None, None)
        };

        let embed_scale = (config.hidden_size as f32).sqrt();
        let attn_scale = config.attn_scale();

        Ok(Self {
            tp_num_heads: config.num_attention_heads,
            tp_num_kv_heads: config.num_key_value_heads,
            dtype,
            config,
            device,
            gpu_config: GpuConfig::Single,
            comm: None,
            embed_tokens,
            layers,
            norm,
            lm_head,
            embed_scale,
            attn_scale,
            cos_cache,
            sin_cache,
            cos_cache_global,
            sin_cache_global,
            _backend: PhantomData,
        })
    }

    /// Load a SafeTensors model from a directory.
    ///
    /// # Errors
    /// Returns an error if the config is missing or weights fail to load.
    pub fn from_pretrained(device: &B::DeviceHandle, model_path: impl AsRef<Path>) -> Result<Self>
    where
        B: infernum::SafeTensorsLoaderOps,
    {
        let model_path = model_path.as_ref();
        let config_path = model_path.join("config.json");
        let config = GemmaConfig::from_json(&config_path);
        let loader = B::safetensors_loader(device, model_path)?;
        Self::load_weights(device.clone(), config, &loader)
    }

    /// Load a Gemma model with tensor-parallel sharding.
    ///
    /// # Errors
    /// Returns an error if loading fails or head counts are not divisible.
    pub fn from_pretrained_sharded(
        device: &B::DeviceHandle,
        model_path: impl AsRef<Path>,
        gpu_config: GpuConfig,
        comm: Option<B::Comm>,
    ) -> Result<Self>
    where
        B: infernum::SafeTensorsLoaderOps,
    {
        let model_path = model_path.as_ref();
        let config_path = model_path.join("config.json");
        let config = GemmaConfig::from_json(&config_path);
        let loader = B::safetensors_loader(device, model_path)?;
        Self::load_weights_sharded(device.clone(), config, &loader, gpu_config, comm)
    }

    // ---- Sharded weight loading ----

    /// Load model weights with tensor-parallel sharding, generic over backend.
    ///
    /// # Errors
    /// Returns an error if weight loading fails.
    ///
    /// # Panics
    /// Panics if head counts are not divisible by `world_size`.
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    pub fn load_weights_sharded(
        device: B::DeviceHandle,
        config: GemmaConfig,
        loader: &impl infernum::WeightLoader<B>,
        gpu_config: GpuConfig,
        comm: Option<B::Comm>,
    ) -> Result<Self> {
        use infernum::shard::shard_strategy_for_weight;

        let shard = match &gpu_config {
            GpuConfig::Sharded(s) => *s,
            GpuConfig::Single => {
                return Self::load_weights(device, config, loader).map(|mut m| {
                    m.comm = comm;
                    m
                })
            }
        };
        let world_size = shard.world_size;

        assert!(
            config.num_attention_heads.is_multiple_of(world_size),
            "num_attention_heads ({}) must be divisible by world_size ({world_size})",
            config.num_attention_heads
        );
        assert!(
            config.num_key_value_heads.is_multiple_of(world_size),
            "num_key_value_heads ({}) must be divisible by world_size ({world_size})",
            config.num_key_value_heads
        );

        let qc = config.quantization_config.as_ref();

        let tp_num_heads = config.num_attention_heads / world_size;
        let tp_num_kv_heads = config.num_key_value_heads / world_size;

        let embed_dtype = loader.get_dtype("model.embed_tokens.weight")?;
        let dtype = if embed_dtype.is_quantized() {
            DType::F32
        } else {
            embed_dtype
        };

        let embed_tokens = loader.load_tensor("model.embed_tokens.weight", dtype)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            let q_name = format!("{prefix}.self_attn.q_proj.weight");
            let k_name = format!("{prefix}.self_attn.k_proj.weight");
            let v_name = format!("{prefix}.self_attn.v_proj.weight");
            let o_name = format!("{prefix}.self_attn.o_proj.weight");

            let k_proj = loader.load_linear_sharded(
                &k_name,
                dtype,
                qc,
                &shard,
                shard_strategy_for_weight(&k_name),
            )?;
            let v_proj = loader.load_linear_sharded(
                &v_name,
                dtype,
                qc,
                &shard,
                shard_strategy_for_weight(&v_name),
            )?;
            let kv_proj = KvProjWeight::<B>::Separate {
                k_proj: Box::new(k_proj),
                v_proj: Box::new(v_proj),
            };

            // QK-norm weights are per-head, not sharded
            let q_norm_name = format!("{prefix}.self_attn.q_norm.weight");
            let k_norm_name = format!("{prefix}.self_attn.k_norm.weight");
            let q_norm = if loader.contains(&q_norm_name) {
                Some(load_gemma_norm::<B>(loader, &device, &q_norm_name, dtype)?)
            } else {
                None
            };
            let k_norm = if loader.contains(&k_norm_name) {
                Some(load_gemma_norm::<B>(loader, &device, &k_norm_name, dtype)?)
            } else {
                None
            };

            let gate_name = format!("{prefix}.mlp.gate_proj.weight");
            let up_name = format!("{prefix}.mlp.up_proj.weight");
            let down_name = format!("{prefix}.mlp.down_proj.weight");

            let gate = loader.load_linear_sharded(
                &gate_name,
                dtype,
                qc,
                &shard,
                shard_strategy_for_weight(&gate_name),
            )?;
            let up = loader.load_linear_sharded(
                &up_name,
                dtype,
                qc,
                &shard,
                shard_strategy_for_weight(&up_name),
            )?;
            let gate_up = GateUpWeight::<B>::Separate {
                gate_proj: Box::new(gate),
                up_proj: Box::new(up),
            };

            let layer = GemmaLayerWeights {
                input_layernorm: load_gemma_norm::<B>(
                    loader,
                    &device,
                    &format!("{prefix}.input_layernorm.weight"),
                    dtype,
                )?,
                post_attention_layernorm: load_gemma_norm::<B>(
                    loader,
                    &device,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    dtype,
                )?,
                pre_feedforward_layernorm: load_gemma_norm::<B>(
                    loader,
                    &device,
                    &format!("{prefix}.pre_feedforward_layernorm.weight"),
                    dtype,
                )?,
                post_feedforward_layernorm: load_gemma_norm::<B>(
                    loader,
                    &device,
                    &format!("{prefix}.post_feedforward_layernorm.weight"),
                    dtype,
                )?,
                attention: GemmaAttentionWeights {
                    q_proj: loader.load_linear_sharded(
                        &q_name,
                        dtype,
                        qc,
                        &shard,
                        shard_strategy_for_weight(&q_name),
                    )?,
                    kv_proj,
                    o_proj: loader.load_linear_sharded(
                        &o_name,
                        dtype,
                        qc,
                        &shard,
                        shard_strategy_for_weight(&o_name),
                    )?,
                    q_norm,
                    k_norm,
                },
                mlp: GemmaMlpWeights {
                    gate_up,
                    down_proj: loader.load_linear_sharded(
                        &down_name,
                        dtype,
                        qc,
                        &shard,
                        shard_strategy_for_weight(&down_name),
                    )?,
                },
            };

            layers.push(layer);
        }

        let norm = load_gemma_norm::<B>(loader, &device, "model.norm.weight", dtype)?;

        // Tied embeddings — not sharded for lm_head
        let lm_head =
            transformer::load_lm_head::<B>(&device, loader, &embed_tokens, true, dtype, qc)?;

        let local_theta = config.rope_local_base_freq.unwrap_or(config.rope_theta);
        let (cos_cache, sin_cache) = transformer::build_rope_cache::<B>(
            &device,
            config.head_dim,
            config.max_position_embeddings,
            local_theta,
            None,
            dtype,
        )?;

        let (cos_cache_global, sin_cache_global) = if config.rope_local_base_freq.is_some() {
            let (cos_g, sin_g) = transformer::build_rope_cache::<B>(
                &device,
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                None,
                dtype,
            )?;
            (Some(cos_g), Some(sin_g))
        } else {
            (None, None)
        };

        let embed_scale = (config.hidden_size as f32).sqrt();
        let attn_scale = config.attn_scale();

        Ok(Self {
            tp_num_heads,
            tp_num_kv_heads,
            dtype,
            config,
            device,
            gpu_config: GpuConfig::Sharded(shard),
            comm,
            embed_tokens,
            layers,
            norm,
            lm_head,
            embed_scale,
            attn_scale,
            cos_cache,
            sin_cache,
            cos_cache_global,
            sin_cache_global,
            _backend: PhantomData,
        })
    }
}

// ---- Model trait impl (generic over any backend) ----

impl<B: GemmaOps + Send + 'static> infernum::Model for GemmaModel<B>
where
    <B as MatmulOps>::LinearWeight: Send + Sync,
{
    type B = B;
    type KvCache = B::PagedKvCache;

    fn config(&self) -> infernum::ModelConfig {
        self.model_config()
    }

    fn device(&self) -> &B::DeviceHandle {
        &self.device
    }

    fn allocate_kv_cache(&self, block_config: &infernum::BlockConfig) -> Result<Self::KvCache> {
        B::allocate_paged_kv_cache(
            &self.device,
            self.config.num_hidden_layers,
            block_config,
            self.tp_num_kv_heads,
            self.config.head_dim,
            self.dtype,
        )
    }

    fn forward(&self, input_ids: &[u32]) -> Result<B::Logits> {
        let tensor = self.forward_full(input_ids)?;
        Ok(B::logits_from_tensor(tensor))
    }

    fn forward_prefill(
        &self,
        input_ids: &[u32],
        kv_cache: &mut Self::KvCache,
        _runtime_state: &mut B::RuntimeState,
        block_table: &BlockTable,
        start_pos: usize,
    ) -> Result<B::Logits> {
        let tensor = self.forward_prefill_paged(input_ids, kv_cache, block_table, start_pos)?;
        Ok(B::logits_from_tensor(tensor))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_decode(
        &self,
        token_ids: &B::Tensor,
        kv_cache: &mut Self::KvCache,
        _runtime_state: &mut B::RuntimeState,
        block_tables: &B::Tensor,
        seq_lens: &B::Tensor,
        positions: &B::Tensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
    ) -> Result<B::Logits> {
        let tensor = self.forward_batch_decode_tensors(
            token_ids,
            kv_cache,
            block_tables,
            seq_lens,
            positions,
            batch_size,
            max_blocks_per_seq,
            max_seq_len,
        )?;
        Ok(B::logits_from_tensor(tensor))
    }
}
