//! Graph builder for the Gemma model family (Gemma 2 and Gemma 3 text).
//!
//! Constructs a computation graph ([`Graph<B>`]) representing the full
//! Gemma forward pass. Supports:
//! - Gemma 2: 4 norms/layer, GeGLU, soft-capping (attn + final logit),
//!   alternating sliding-window / full-attention layers
//! - Gemma 3: same as Gemma 2 plus per-head QK-norm, no soft-capping,
//!   dual-theta RoPE (handled via per-layer rope_theta_for_layer)

use infernum::backend::{
    ArithOps, AttentionOps, Backend, EmbedOps, GegluOps, MatmulOps, NormOps, RopeOps, TensorOps,
};
use infernum::dtype::DType;
use infernum::graph::{
    Graph, GraphArithOps, GraphAttentionOps, GraphEmbedOps, GraphGegluOps, GraphMatmulOps,
    GraphNormOps, GraphRopeOps, GraphSoftcapOps, GraphTensorOps, OutputRef, WeightId,
};

use crate::config::GemmaConfig;

// ---------------------------------------------------------------------------
// Weight ID structures
// ---------------------------------------------------------------------------

/// Optional QK-norm weight IDs for Gemma 3.
pub struct QkNormIds {
    /// Per-head RMSNorm weight applied to Q before RoPE.
    pub q_norm: WeightId,
    /// Per-head RMSNorm weight applied to K before RoPE.
    pub k_norm: WeightId,
}

/// Weight IDs for a single Gemma transformer layer.
pub struct LayerWeightIds {
    /// RMS norm applied to hidden state before attention.
    pub input_layernorm: WeightId,
    /// Q projection (linear weight).
    pub q_proj: WeightId,
    /// K projection (linear weight).
    pub k_proj: WeightId,
    /// V projection (linear weight).
    pub v_proj: WeightId,
    /// Output projection (linear weight).
    pub o_proj: WeightId,
    /// RMS norm applied to attention output before residual add.
    pub post_attention_layernorm: WeightId,
    /// RMS norm applied before the FFN block.
    pub pre_feedforward_layernorm: WeightId,
    /// RMS norm applied to FFN output before residual add.
    pub post_feedforward_layernorm: WeightId,
    /// Gate projection (GeGLU).
    pub gate_proj: WeightId,
    /// Up projection (GeGLU).
    pub up_proj: WeightId,
    /// Down projection (GeGLU output back to hidden).
    pub down_proj: WeightId,
    /// Optional per-head QK-norm weights (Gemma 3 only).
    pub qk_norm: Option<QkNormIds>,
}

/// All weight IDs for the full Gemma model.
pub struct ModelWeightIds {
    /// Embedding table.
    pub embed_tokens: WeightId,
    /// Per-layer weight IDs.
    pub layers: Vec<LayerWeightIds>,
    /// Final RMS norm weight.
    pub final_norm: WeightId,
    /// LM head (may alias embed_tokens when `tie_word_embeddings` is true).
    pub lm_head: WeightId,
}

// ---------------------------------------------------------------------------
// Backend trait bound
// ---------------------------------------------------------------------------

/// Backend trait bounds required by the Gemma graph builder.
pub trait GemmaGraphOps:
    Backend + MatmulOps + NormOps + ArithOps + EmbedOps + TensorOps + RopeOps + AttentionOps + GegluOps
{
}

impl<B> GemmaGraphOps for B where
    B: Backend
        + MatmulOps
        + NormOps
        + ArithOps
        + EmbedOps
        + TensorOps
        + RopeOps
        + AttentionOps
        + GegluOps
{
}

// ---------------------------------------------------------------------------
// Weight registration
// ---------------------------------------------------------------------------

/// Register all Gemma model weights into the graph and return their IDs.
///
/// Names match the `SafeTensors` convention used by the HuggingFace checkpoints.
#[allow(clippy::too_many_lines)]
pub fn register_weights<B: Backend + MatmulOps>(
    graph: &mut Graph<B>,
    config: &GemmaConfig,
    weight_dtype: DType,
) -> ModelWeightIds {
    let hidden = config.hidden_size;
    let head_dim = config.head_dim;

    let embed_tokens = graph.register_tensor_weight(
        "model.embed_tokens.weight",
        &[config.vocab_size, hidden],
        weight_dtype,
    );

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let pfx = format!("model.layers.{i}");

        let input_layernorm = graph.register_tensor_weight(
            format!("{pfx}.input_layernorm.weight"),
            &[hidden],
            weight_dtype,
        );
        let post_attention_layernorm = graph.register_tensor_weight(
            format!("{pfx}.post_attention_layernorm.weight"),
            &[hidden],
            weight_dtype,
        );
        let pre_feedforward_layernorm = graph.register_tensor_weight(
            format!("{pfx}.pre_feedforward_layernorm.weight"),
            &[hidden],
            weight_dtype,
        );
        let post_feedforward_layernorm = graph.register_tensor_weight(
            format!("{pfx}.post_feedforward_layernorm.weight"),
            &[hidden],
            weight_dtype,
        );

        let q_proj = graph.register_linear_weight(
            format!("{pfx}.self_attn.q_proj.weight"),
            &[config.num_attention_heads * head_dim, hidden],
            weight_dtype,
        );
        let k_proj = graph.register_linear_weight(
            format!("{pfx}.self_attn.k_proj.weight"),
            &[config.num_key_value_heads * head_dim, hidden],
            weight_dtype,
        );
        let v_proj = graph.register_linear_weight(
            format!("{pfx}.self_attn.v_proj.weight"),
            &[config.num_key_value_heads * head_dim, hidden],
            weight_dtype,
        );
        let o_proj = graph.register_linear_weight(
            format!("{pfx}.self_attn.o_proj.weight"),
            &[hidden, config.num_attention_heads * head_dim],
            weight_dtype,
        );

        let qk_norm = if config.has_qk_norm {
            Some(QkNormIds {
                q_norm: graph.register_tensor_weight(
                    format!("{pfx}.self_attn.q_norm.weight"),
                    &[head_dim],
                    weight_dtype,
                ),
                k_norm: graph.register_tensor_weight(
                    format!("{pfx}.self_attn.k_norm.weight"),
                    &[head_dim],
                    weight_dtype,
                ),
            })
        } else {
            None
        };

        let gate_proj = graph.register_linear_weight(
            format!("{pfx}.mlp.gate_proj.weight"),
            &[config.intermediate_size, hidden],
            weight_dtype,
        );
        let up_proj = graph.register_linear_weight(
            format!("{pfx}.mlp.up_proj.weight"),
            &[config.intermediate_size, hidden],
            weight_dtype,
        );
        let down_proj = graph.register_linear_weight(
            format!("{pfx}.mlp.down_proj.weight"),
            &[hidden, config.intermediate_size],
            weight_dtype,
        );

        layers.push(LayerWeightIds {
            input_layernorm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            gate_proj,
            up_proj,
            down_proj,
            qk_norm,
        });
    }

    let final_norm = graph.register_tensor_weight("model.norm.weight", &[hidden], weight_dtype);

    let lm_head = if config.tie_word_embeddings {
        embed_tokens
    } else {
        graph.register_linear_weight("lm_head.weight", &[config.vocab_size, hidden], weight_dtype)
    };

    ModelWeightIds {
        embed_tokens,
        layers,
        final_norm,
        lm_head,
    }
}

// ---------------------------------------------------------------------------
// Attention helper
// ---------------------------------------------------------------------------

fn build_attention_prefill<B>(
    graph: &mut Graph<B>,
    config: &GemmaConfig,
    layer_idx: usize,
    ids: &LayerWeightIds,
    h_normed: OutputRef,
    cos_input: OutputRef,
    sin_input: OutputRef,
) -> OutputRef
where
    B: GemmaGraphOps,
    Graph<B>: GraphMatmulOps
        + GraphNormOps
        + GraphRopeOps
        + GraphTensorOps
        + GraphAttentionOps
        + GraphArithOps
        + GraphGegluOps
        + GraphSoftcapOps,
{
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;

    let q = graph.add_linear(h_normed, ids.q_proj);
    let k = graph.add_linear(h_normed, ids.k_proj);
    let v = graph.add_linear(h_normed, ids.v_proj);

    // Reshape to 3D [seq_len, num_heads, head_dim] for QK-norm and RoPE.
    // 0 is the dynamic seq_len placeholder in the prefill graph.
    let q = graph.add_reshape(q, &[0, num_heads, head_dim]);
    let k = graph.add_reshape(k, &[0, num_kv_heads, head_dim]);
    let v = graph.add_reshape(v, &[0, num_kv_heads, head_dim]);

    // Optional QK-norm (Gemma 3)
    let (q, k) = if let Some(ref qkn) = ids.qk_norm {
        graph.add_qk_norm(q, k, qkn.q_norm, qkn.k_norm, config.rms_norm_eps)
    } else {
        (q, k)
    };

    let q = graph.add_rope(q, cos_input, sin_input, 0);
    let k = graph.add_rope(k, cos_input, sin_input, 0);

    let sliding_window = config.effective_sliding_window(layer_idx);
    let attn_scale = Some(config.attn_scale());
    let softcap = config.attn_logit_softcapping;
    let attn_out =
        graph.add_fused_attention_prefill(q, k, v, 0, attn_scale, softcap, sliding_window);

    // Reshape back to 2D [seq_len, num_heads * head_dim] before the output projection.
    let attn_flat = graph.add_reshape(attn_out, &[0, num_heads * head_dim]);
    graph.add_linear(attn_flat, ids.o_proj)
}

#[allow(clippy::too_many_arguments)]
fn build_attention_decode<B>(
    graph: &mut Graph<B>,
    config: &GemmaConfig,
    layer_idx: usize,
    ids: &LayerWeightIds,
    h_normed: OutputRef,
    cos_input: OutputRef,
    sin_input: OutputRef,
    k_cache_input: OutputRef,
    v_cache_input: OutputRef,
) -> (OutputRef, OutputRef, OutputRef)
where
    B: GemmaGraphOps,
    Graph<B>: GraphMatmulOps
        + GraphNormOps
        + GraphRopeOps
        + GraphTensorOps
        + GraphAttentionOps
        + GraphArithOps
        + GraphGegluOps
        + GraphSoftcapOps,
{
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;

    let q = graph.add_linear(h_normed, ids.q_proj);
    let k = graph.add_linear(h_normed, ids.k_proj);
    let v = graph.add_linear(h_normed, ids.v_proj);

    // Reshape to 3D [1, num_heads, head_dim] for QK-norm and RoPE.
    let q = graph.add_reshape(q, &[1, num_heads, head_dim]);
    let k = graph.add_reshape(k, &[1, num_kv_heads, head_dim]);
    let v = graph.add_reshape(v, &[1, num_kv_heads, head_dim]);

    // Optional QK-norm (Gemma 3)
    let (q, k) = if let Some(ref qkn) = ids.qk_norm {
        graph.add_qk_norm(q, k, qkn.q_norm, qkn.k_norm, config.rms_norm_eps)
    } else {
        (q, k)
    };

    let q = graph.add_rope(q, cos_input, sin_input, 0);
    let k = graph.add_rope(k, cos_input, sin_input, 0);

    // Accumulate K/V history via concat_seq.
    let full_k = graph.add_concat_seq(k_cache_input, k);
    let full_v = graph.add_concat_seq(v_cache_input, v);

    let softcap = config.attn_logit_softcapping;
    // Sliding window is tracked via KV cache metadata at decode time; not passed here.
    let _ = layer_idx;
    let attn_out = graph.add_fused_attention_decode(q, full_k, full_v, softcap);

    // Reshape back to 2D [1, num_heads * head_dim] before the output projection.
    let attn_flat = graph.add_reshape(attn_out, &[1, num_heads * head_dim]);
    (graph.add_linear(attn_flat, ids.o_proj), full_k, full_v)
}

// ---------------------------------------------------------------------------
// Public graph constructors
// ---------------------------------------------------------------------------

/// Build the Gemma prefill graph.
///
/// The graph has two inputs: token IDs and a `RoPE` cache tensor.
#[must_use]
pub fn build_prefill_graph<B>(config: &GemmaConfig, weight_dtype: DType) -> Graph<B>
where
    B: GemmaGraphOps,
    Graph<B>: GraphMatmulOps
        + GraphNormOps
        + GraphEmbedOps
        + GraphRopeOps
        + GraphTensorOps
        + GraphAttentionOps
        + GraphArithOps
        + GraphGegluOps
        + GraphSoftcapOps,
{
    let mut graph = Graph::<B>::new();

    let ids = register_weights(&mut graph, config, weight_dtype);

    let hidden = config.hidden_size;

    // Inputs: token IDs + RoPE cos/sin cache
    let token_input = graph.add_input(&[0], DType::U32);
    let cos_input = graph.add_input(&[0, config.head_dim], weight_dtype);
    let sin_input = graph.add_input(&[0, config.head_dim], weight_dtype);

    // Embedding lookup + scale by sqrt(hidden_size)
    #[allow(clippy::cast_precision_loss)]
    let embed_scale = (hidden as f32).sqrt();
    let mut h = graph.add_embedding_gather(ids.embed_tokens, token_input);
    h = graph.add_scale(h, embed_scale);

    for (layer_idx, layer_ids) in ids.layers.iter().enumerate() {
        // Pre-attention norm
        let normed = graph.add_rms_norm(h, layer_ids.input_layernorm, config.rms_norm_eps);

        // Attention
        let attn_out = build_attention_prefill(
            &mut graph, config, layer_idx, layer_ids, normed, cos_input, sin_input,
        );

        // Post-attention norm then residual
        let post_attn_normed = graph.add_rms_norm(
            attn_out,
            layer_ids.post_attention_layernorm,
            config.rms_norm_eps,
        );
        h = graph.add_add(h, post_attn_normed);

        // Pre-FFN norm
        let normed_ffn =
            graph.add_rms_norm(h, layer_ids.pre_feedforward_layernorm, config.rms_norm_eps);

        // GeGLU FFN
        let gate = graph.add_linear(normed_ffn, layer_ids.gate_proj);
        let up = graph.add_linear(normed_ffn, layer_ids.up_proj);
        let activated = graph.add_geglu(gate, up);
        let down = graph.add_linear(activated, layer_ids.down_proj);

        // Post-FFN norm then residual
        let post_ffn_normed = graph.add_rms_norm(
            down,
            layer_ids.post_feedforward_layernorm,
            config.rms_norm_eps,
        );
        h = graph.add_add(h, post_ffn_normed);
    }

    // Final norm and LM head
    let normed = graph.add_rms_norm(h, ids.final_norm, config.rms_norm_eps);
    let logits = graph.add_lm_head(normed, ids.lm_head, weight_dtype);

    // Optional final logit soft-capping (Gemma 2)
    let logits = if let Some(cap) = config.final_logit_softcapping {
        graph.add_logit_softcap(logits, cap)
    } else {
        logits
    };

    graph.set_output(logits.0);
    graph
}

/// Build the Gemma single-token decode graph.
///
/// ## Inputs
/// 0. `token_id` — shape `[1]`, U32
/// 1. `cos_cache` — shape `[1, head_dim]`
/// 2. `sin_cache` — shape `[1, head_dim]`
/// 3. `k_cache_i` — shape `[kv_len, num_kv_heads, head_dim]` per layer
///    (indices `3..3+L`)
/// 4. `v_cache_i` — shape `[kv_len, num_kv_heads, head_dim]` per layer
///    (indices `3+L..3+2L`)
///
/// ## Outputs
/// 0. `logits` — shape `[1, vocab_size]`
/// 1. `full_k_i` — updated K caches (shape `[kv_len+1, ...]`, indices `1..1+L`)
/// 2. `full_v_i` — updated V caches (indices `1+L..1+2L`)
#[must_use]
#[allow(clippy::similar_names)]
pub fn build_decode_graph<B>(config: &GemmaConfig, kv_len: usize, weight_dtype: DType) -> Graph<B>
where
    B: GemmaGraphOps,
    Graph<B>: GraphMatmulOps
        + GraphNormOps
        + GraphEmbedOps
        + GraphRopeOps
        + GraphTensorOps
        + GraphAttentionOps
        + GraphArithOps
        + GraphGegluOps
        + GraphSoftcapOps,
{
    let mut graph = Graph::<B>::new();

    let ids = register_weights(&mut graph, config, weight_dtype);

    let hidden = config.hidden_size;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let num_layers = config.num_hidden_layers;

    // Inputs: single token ID + RoPE cos/sin
    let token_input = graph.add_input(&[1], DType::U32);
    let cos_input = graph.add_input(&[1, head_dim], weight_dtype);
    let sin_input = graph.add_input(&[1, head_dim], weight_dtype);

    // KV cache input nodes: one K + one V per layer.
    let mut k_cache_inputs = Vec::with_capacity(num_layers);
    let mut v_cache_inputs = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        k_cache_inputs.push(graph.add_input(&[kv_len, num_kv_heads, head_dim], weight_dtype));
        v_cache_inputs.push(graph.add_input(&[kv_len, num_kv_heads, head_dim], weight_dtype));
    }

    // Embedding lookup + scale
    #[allow(clippy::cast_precision_loss)]
    let embed_scale = (hidden as f32).sqrt();
    let mut h = graph.add_embedding_gather(ids.embed_tokens, token_input);
    h = graph.add_scale(h, embed_scale);

    let mut full_k_outputs = Vec::with_capacity(num_layers);
    let mut full_v_outputs = Vec::with_capacity(num_layers);

    for (layer_idx, layer_ids) in ids.layers.iter().enumerate() {
        // Pre-attention norm
        let normed = graph.add_rms_norm(h, layer_ids.input_layernorm, config.rms_norm_eps);

        // Attention (decode with KV cache accumulation)
        let (attn_out, full_k, full_v) = build_attention_decode(
            &mut graph,
            config,
            layer_idx,
            layer_ids,
            normed,
            cos_input,
            sin_input,
            k_cache_inputs[layer_idx],
            v_cache_inputs[layer_idx],
        );
        full_k_outputs.push(full_k);
        full_v_outputs.push(full_v);

        // Post-attention norm then residual
        let post_attn_normed = graph.add_rms_norm(
            attn_out,
            layer_ids.post_attention_layernorm,
            config.rms_norm_eps,
        );
        h = graph.add_add(h, post_attn_normed);

        // Pre-FFN norm
        let normed_ffn =
            graph.add_rms_norm(h, layer_ids.pre_feedforward_layernorm, config.rms_norm_eps);

        // GeGLU FFN
        let gate = graph.add_linear(normed_ffn, layer_ids.gate_proj);
        let up = graph.add_linear(normed_ffn, layer_ids.up_proj);
        let activated = graph.add_geglu(gate, up);
        let down = graph.add_linear(activated, layer_ids.down_proj);

        // Post-FFN norm then residual
        let post_ffn_normed = graph.add_rms_norm(
            down,
            layer_ids.post_feedforward_layernorm,
            config.rms_norm_eps,
        );
        h = graph.add_add(h, post_ffn_normed);
    }

    // Final norm and LM head
    let normed = graph.add_rms_norm(h, ids.final_norm, config.rms_norm_eps);
    let logits = graph.add_lm_head(normed, ids.lm_head, weight_dtype);

    // Optional final logit soft-capping (Gemma 2)
    let logits = if let Some(cap) = config.final_logit_softcapping {
        graph.add_logit_softcap(logits, cap)
    } else {
        logits
    };

    // Output 0: logits; outputs 1..1+L: full_k; outputs 1+L..1+2L: full_v.
    graph.set_output(logits.0);
    for fk in &full_k_outputs {
        graph.set_output(fk.0);
    }
    for fv in &full_v_outputs {
        graph.set_output(fv.0);
    }

    graph
}

// ---------------------------------------------------------------------------
// GGUF name mapping
// ---------------------------------------------------------------------------

/// Map a SafeTensors weight name (HuggingFace convention) to its GGUF key.
///
/// Gemma 2 / Gemma 3 use `blk.N.*` block prefixes with the following suffixes:
/// - `attn_norm` / `post_attention_norm` / `ffn_norm` / `post_ffw_norm` for the 4 norms
/// - `attn_q` / `attn_k` / `attn_v` / `attn_output` for attention projections
/// - `ffn_gate` / `ffn_up` / `ffn_down` for GeGLU projections
/// - Optional `attn_q_norm` / `attn_k_norm` for Gemma 3 per-head QK-norm
#[cfg(feature = "cpu")]
fn safetensors_to_gguf_name(name: &str) -> String {
    match name {
        "model.embed_tokens.weight" => return "token_embd.weight".to_string(),
        "model.norm.weight" => return "output_norm.weight".to_string(),
        "lm_head.weight" => return "output.weight".to_string(),
        _ => {}
    }
    if let Some(rest) = name.strip_prefix("model.layers.") {
        let dot = rest.find('.').expect("malformed layer weight name");
        let layer_idx = &rest[..dot];
        let suffix = &rest[dot + 1..];
        let gguf_suffix = match suffix {
            "input_layernorm.weight" => "attn_norm.weight",
            "post_attention_layernorm.weight" => "post_attention_norm.weight",
            "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
            "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            // Gemma 3 per-head QK-norm
            "self_attn.q_norm.weight" => "attn_q_norm.weight",
            "self_attn.k_norm.weight" => "attn_k_norm.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            other => panic!("Unknown Gemma layer suffix: {other}"),
        };
        return format!("blk.{layer_idx}.{gguf_suffix}");
    }
    panic!("Unknown Gemma weight name: {name}");
}

// ---------------------------------------------------------------------------
// Weight loaders (CPU backend)
// ---------------------------------------------------------------------------

/// Load SafeTensors weights from `model_dir` into a CPU weight store,
/// using the layout encoded in `graph`.
///
/// Tied embeddings: if `lm_head.weight` is absent, `model.embed_tokens.weight`
/// is used as the fallback (Gemma models often tie them).
///
/// # Errors
///
/// Returns an error if the directory contains no `.safetensors` files, a
/// required weight is missing, or a weight cannot be converted to the target
/// dtype.
///
/// # Panics
///
/// Panics if the number of weights exceeds `u32::MAX` (practically impossible).
#[cfg(feature = "cpu")]
pub fn load_graph_weights_safetensors(
    graph: &infernum::graph::Graph<infernum_cpu::CpuBackend>,
    model_dir: &std::path::Path,
    _config: &GemmaConfig,
) -> infernum::Result<
    infernum::graph::WeightStore<
        infernum_cpu::tensor::CpuTensor,
        infernum_cpu::tensor::CpuLinearWeight,
    >,
> {
    use infernum::graph::WeightId;
    use infernum::WeightLoader as _;
    use infernum_cpu::CpuSafeTensorsLoader;

    let loader = CpuSafeTensorsLoader::new(model_dir)?;

    let tensor_count = graph.tensor_weight_count();
    let linear_count = graph.linear_weight_count();

    let mut store = infernum::graph::WeightStore::with_capacity(tensor_count, linear_count);

    for i in 0..tensor_count {
        let meta = graph.tensor_weight_meta(WeightId::from_index(
            u32::try_from(i).expect("weight index fits u32"),
        ));
        let tensor = loader.load_tensor(&meta.name, meta.dtype)?;
        store.push_tensor_weight(tensor);
    }

    for i in 0..linear_count {
        let meta = graph.linear_weight_meta(WeightId::from_index(
            u32::try_from(i).expect("weight index fits u32"),
        ));
        // Handle tied embeddings: if `lm_head.weight` is absent in the
        // checkpoint, use `model.embed_tokens.weight` as a fallback.
        let name = if meta.name == "lm_head.weight" && !loader.contains("lm_head.weight") {
            "model.embed_tokens.weight"
        } else {
            &meta.name
        };
        let weight = loader.load_linear(name, meta.dtype, None)?;
        store.push_linear_weight(weight);
    }

    Ok(store)
}

/// Load GGUF weights from a single `.gguf` file into a CPU weight store.
///
/// Uses the GGUF key-naming convention (e.g. `blk.0.attn_q.weight`).
/// Name mapping from SafeTensors convention is handled by
/// [`safetensors_to_gguf_name`].
///
/// # Errors
///
/// Returns an error if the GGUF file cannot be opened, a required weight is
/// missing, or a weight cannot be converted to the target dtype.
///
/// # Panics
///
/// Panics if the number of weights exceeds `u32::MAX` (practically impossible).
#[cfg(feature = "cpu")]
pub fn load_graph_weights_gguf(
    graph: &infernum::graph::Graph<infernum_cpu::CpuBackend>,
    gguf_path: &std::path::Path,
    _config: &GemmaConfig,
) -> infernum::Result<
    infernum::graph::WeightStore<
        infernum_cpu::tensor::CpuTensor,
        infernum_cpu::tensor::CpuLinearWeight,
    >,
> {
    use infernum::graph::WeightId;
    use infernum::weights::format::{host_transpose_2d, FormatLoader};
    use infernum::weights::host::HostLinearWeight;
    use infernum_cpu::tensor::CpuTensor;
    use infernum_cpu::CpuBackend;

    let loader =
        infernum::weights::gguf::GgufLoader::from_file(gguf_path.to_str().ok_or_else(|| {
            infernum::Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "GGUF path is not valid UTF-8",
            ))
        })?)?;

    let tensor_count = graph.tensor_weight_count();
    let linear_count = graph.linear_weight_count();
    let mut store = infernum::graph::WeightStore::with_capacity(tensor_count, linear_count);

    // Tensor weights (embeddings, layernorms) — loaded as F32.
    for i in 0..tensor_count {
        let meta = graph.tensor_weight_meta(WeightId::from_index(
            u32::try_from(i).expect("weight index fits u32"),
        ));
        let gguf_name = safetensors_to_gguf_name(&meta.name);
        let host = loader.load_f32(&gguf_name)?;
        store.push_tensor_weight(CpuTensor::from_f32(&host.shape, host.as_f32_slice()));
    }

    // Linear weights — loaded in native format (quantized or dense).
    for i in 0..linear_count {
        let meta = graph.linear_weight_meta(WeightId::from_index(
            u32::try_from(i).expect("weight index fits u32"),
        ));
        let gguf_name = safetensors_to_gguf_name(&meta.name);

        // Resolve actual name with tied-embedding fallback.
        let actual_name = if loader.contains(&gguf_name) {
            gguf_name.clone()
        } else if meta.name == "lm_head.weight" {
            "token_embd.weight".to_string()
        } else {
            return Err(infernum::Error::WeightNotFound(gguf_name));
        };

        let dtype = FormatLoader::get_dtype(&loader, &actual_name)?;
        let host_linear = if dtype.is_quantized() {
            // Preserve native quantization for fast kernels at inference time.
            HostLinearWeight::Quantized(FormatLoader::load_quantized(&loader, &actual_name)?)
        } else {
            // Dense path: load as F32, then transpose for matmul convention.
            let host = loader.load_f32(&actual_name)?;
            HostLinearWeight::Dense(host_transpose_2d(&host)?)
        };
        let linear = CpuBackend::upload_host_linear(&(), &host_linear)?;
        store.push_linear_weight(linear);
    }

    Ok(store)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use infernum::DType;

    /// Minimal no-op backend for graph construction tests.
    struct TestBackend;

    #[derive(Clone)]
    struct DummyTensor;

    impl infernum::tensor::Tensor for DummyTensor {
        fn shape(&self) -> &[usize] {
            &[]
        }
        fn dtype(&self) -> DType {
            DType::F32
        }
        fn reshape(&self, _shape: &[usize]) -> Self {
            Self
        }
        fn slice_view(&self, _offset: usize, _shape: &[usize]) -> Self {
            Self
        }
    }

    struct DummyLogits;

    impl infernum::logits::Logits for DummyLogits {
        fn vocab_size(&self) -> usize {
            0
        }
        fn batch_size(&self) -> usize {
            0
        }
        fn argmax(&self, _batch_index: usize) -> infernum::Result<u32> {
            Ok(0)
        }
        fn sample_top_p(
            &self,
            _batch_index: usize,
            _temperature: f32,
            _top_p: f32,
            _rng_seed: u64,
            _repetition_penalty: f32,
            _recent_tokens: &[u32],
        ) -> infernum::Result<u32> {
            Ok(0)
        }
    }

    struct DummyRuntimeState;

    impl infernum::runtime_state::RuntimeStateInit for DummyRuntimeState {
        fn new(
            _batch_config: &infernum::runtime_state::BatchConfig,
            _block_config: &infernum::block_allocator::BlockConfig,
        ) -> infernum::Result<Self> {
            Ok(Self)
        }
        fn new_placeholder() -> Self {
            Self
        }
    }

    impl infernum::backend::Backend for TestBackend {
        type Tensor = DummyTensor;
        type DeviceHandle = ();
        type PagedKvCache = ();
        type KvCache = ();
        type RuntimeState = DummyRuntimeState;
        type Logits = DummyLogits;
        type Comm = ();

        fn logits_from_tensor(_tensor: Self::Tensor) -> Self::Logits {
            DummyLogits
        }
    }

    impl infernum::backend::EmbedOps for TestBackend {
        fn embedding_gather(
            _table: &DummyTensor,
            _indices: &[u32],
        ) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn embedding_gather_tensor(
            _table: &DummyTensor,
            _indices: &DummyTensor,
            _seq_len: usize,
        ) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl infernum::backend::NormOps for TestBackend {
        fn rms_norm(
            _input: &DummyTensor,
            _weight: &DummyTensor,
            _eps: f32,
        ) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn rms_norm_inplace(
            _input: &mut DummyTensor,
            _weight: &DummyTensor,
            _eps: f32,
        ) -> infernum::Result<()> {
            Ok(())
        }
        fn add_rmsnorm(
            _residual: &DummyTensor,
            _input: &DummyTensor,
            _weight: &DummyTensor,
            _eps: f32,
        ) -> infernum::Result<(DummyTensor, DummyTensor)> {
            Ok((DummyTensor, DummyTensor))
        }
    }

    impl infernum::backend::MatmulOps for TestBackend {
        type LinearWeight = DummyTensor;

        fn matmul(_a: &DummyTensor, _b: &DummyTensor) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn linear(_input: &DummyTensor, _weight: &DummyTensor) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn dense_weight(tensor: DummyTensor) -> DummyTensor {
            tensor
        }
        fn is_dense_weight(_weight: &DummyTensor) -> bool {
            true
        }
        fn quantize_to_q8(
            _device: &(),
            _shape: &[usize],
            _data: &[f32],
        ) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn upload_host_linear(
            _device: &(),
            _weight: &infernum::weights::host::HostLinearWeight,
        ) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl infernum::backend::ArithOps for TestBackend {
        fn add(_a: &DummyTensor, _b: &DummyTensor) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn add_inplace(_a: &mut DummyTensor, _b: &DummyTensor) -> infernum::Result<()> {
            Ok(())
        }
        fn mul(_a: &DummyTensor, _b: &DummyTensor) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn scale_inplace(_a: &mut DummyTensor, _scale: f32) -> infernum::Result<()> {
            Ok(())
        }
    }

    impl infernum::backend::TensorOps for TestBackend {
        fn transpose_2d(_input: &DummyTensor) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn split_inner_dim(
            _tensor: &DummyTensor,
            _dim1: usize,
            _dim2: usize,
        ) -> infernum::Result<(DummyTensor, DummyTensor)> {
            Ok((DummyTensor, DummyTensor))
        }
        fn concat_inner_dim(_a: &DummyTensor, _b: &DummyTensor) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn pad_inner_dim(
            _tensor: &DummyTensor,
            _new_width: usize,
        ) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn broadcast_to_heads(
            _tensor: &DummyTensor,
            _num_heads: usize,
        ) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn repeat_kv(_tensor: &DummyTensor, _num_repeats: usize) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn concat_rows(_parts: &[DummyTensor]) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl infernum::backend::RopeOps for TestBackend {
        fn apply_rope(
            _input: &DummyTensor,
            _cos: &DummyTensor,
            _sin: &DummyTensor,
            _offset: usize,
        ) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn apply_rope_batched(
            _input: &DummyTensor,
            _cos: &DummyTensor,
            _sin: &DummyTensor,
            _positions: &DummyTensor,
            _batch_size: usize,
        ) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl infernum::backend::AttentionOps for TestBackend {
        fn fused_attention_prefill(
            _q: &DummyTensor,
            _k: &DummyTensor,
            _v: &DummyTensor,
            _offset: usize,
            _scale: Option<f32>,
            _softcap: Option<f32>,
            _sliding_window: Option<usize>,
        ) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn fused_attention_decode(
            _q: &DummyTensor,
            _k: &DummyTensor,
            _v: &DummyTensor,
            _scale: Option<f32>,
            _softcap: Option<f32>,
            _sliding_window: Option<usize>,
        ) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn fused_attention_prefill_with_lse(
            _q: &DummyTensor,
            _k: &DummyTensor,
            _v: &DummyTensor,
            _offset: usize,
            _scale: Option<f32>,
            _softcap: Option<f32>,
            _sliding_window: Option<usize>,
        ) -> infernum::Result<(DummyTensor, DummyTensor)> {
            Ok((DummyTensor, DummyTensor))
        }
        fn combine_attention_with_lse(
            _out1: &DummyTensor,
            _lse1: &DummyTensor,
            _out2: &DummyTensor,
            _lse2: &DummyTensor,
        ) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl infernum::backend::GegluOps for TestBackend {
        fn geglu(_gate: &DummyTensor, _up: &DummyTensor) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    fn gemma2_config() -> GemmaConfig {
        GemmaConfig::from_str(
            r#"{
                "model_type": "gemma2",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "rms_norm_eps": 1e-6,
                "query_pre_attn_scalar": 256.0,
                "sliding_window": 32,
                "attn_logit_softcapping": 50.0,
                "final_logit_softcapping": 30.0,
                "rope_theta": 10000.0,
                "tie_word_embeddings": true,
                "bos_token_id": 2,
                "eos_token_id": 1
            }"#,
        )
    }

    fn gemma3_config() -> GemmaConfig {
        GemmaConfig::from_str(
            r#"{
                "model_type": "gemma3_text",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "rms_norm_eps": 1e-6,
                "query_pre_attn_scalar": 256.0,
                "sliding_window": 32,
                "rope_theta": 10000.0,
                "rope_local_base_freq": 1000.0,
                "tie_word_embeddings": true,
                "bos_token_id": 2,
                "eos_token_id": 1
            }"#,
        )
    }

    #[test]
    fn test_build_prefill_graph_gemma2() {
        let config = gemma2_config();
        let graph: Graph<TestBackend> = build_prefill_graph(&config, DType::F32);
        assert_eq!(graph.output_ids().len(), 1);
    }

    #[test]
    fn test_build_prefill_graph_gemma3() {
        let config = gemma3_config();
        let graph: Graph<TestBackend> = build_prefill_graph(&config, DType::F32);
        assert_eq!(graph.output_ids().len(), 1);
    }

    #[test]
    fn test_build_decode_graph_gemma2() {
        let config = gemma2_config();
        let graph: Graph<TestBackend> = build_decode_graph(&config, 0, DType::F32);
        // Output 0: logits; outputs 1..1+L: full_k; outputs 1+L..1+2L: full_v.
        assert_eq!(graph.output_ids().len(), 1 + 2 * config.num_hidden_layers);
    }

    #[test]
    fn test_build_decode_graph_gemma3() {
        let config = gemma3_config();
        let graph: Graph<TestBackend> = build_decode_graph(&config, 0, DType::F32);
        // Output 0: logits; outputs 1..1+L: full_k; outputs 1+L..1+2L: full_v.
        assert_eq!(graph.output_ids().len(), 1 + 2 * config.num_hidden_layers);
    }

    #[test]
    fn test_gemma2_has_softcapping() {
        let config = gemma2_config();
        assert!(config.attn_logit_softcapping.is_some());
        assert!(config.final_logit_softcapping.is_some());
    }

    #[test]
    fn test_gemma3_no_softcapping() {
        let config = gemma3_config();
        assert!(config.attn_logit_softcapping.is_none());
        assert!(config.final_logit_softcapping.is_none());
    }

    #[test]
    fn test_gemma3_has_qk_norm() {
        let config = gemma3_config();
        assert!(config.has_qk_norm);
    }
}
