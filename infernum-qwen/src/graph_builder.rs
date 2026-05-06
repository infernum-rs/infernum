//! Graph builder for the Qwen model family.
//!
//! Constructs a computation graph ([`Graph<B>`]) representing the full
//! Qwen forward pass. Supports:
//!
//! - **Qwen2/2.5** — dense, Q/K/V biases, tied embeddings
//! - **Qwen3/3.5** — dense, per-head QK-norm, explicit `head_dim`
//! - **Qwen3-MoE** — softmax `MoE` routing with shared expert, `decoder_sparse_step`
//! - Per-layer sliding window attention (SWA) via `effective_sliding_window`
//! - Prefill and single-token decode graphs
//! - `SafeTensors` weight loading on the CPU backend

use infernum::backend::{
    ArithOps, AttentionOps, Backend, BiasOps, EmbedOps, MatmulOps, MoeOps, MoeSigmoidOps, NormOps,
    PagedAttentionOps, PagedKvCacheOps, RopeOps, SwigluOps, TensorOps,
};
use infernum::dtype::DType;
use infernum::graph::{
    Graph, GraphArithOps, GraphAttentionOps, GraphBiasOps, GraphEmbedOps, GraphMatmulOps,
    GraphMoeOps, GraphNormOps, GraphPagedAttentionOps, GraphPagedKvCacheOps, GraphRopeOps,
    GraphSiluOps, GraphTensorOps, MoeExpertIds, OutputRef, WeightId,
};

use crate::config::QwenConfig;

// ---------------------------------------------------------------------------
// Weight ID structures
// ---------------------------------------------------------------------------

/// Weight IDs for Q/K/V attention projection biases (Qwen2).
pub struct QkvBiasIds {
    pub q_bias: WeightId,
    pub k_bias: WeightId,
    pub v_bias: WeightId,
}

/// Weight IDs for per-head QK norms (Qwen3).
pub struct QkNormIds {
    pub q_norm: WeightId,
    pub k_norm: WeightId,
}

/// Weight IDs for a single transformer layer (dense path).
pub struct DenseLayerWeightIds {
    pub input_layernorm: WeightId,
    pub q_proj: WeightId,
    pub k_proj: WeightId,
    pub v_proj: WeightId,
    /// Present in Qwen2/2.5; absent in Qwen3.
    pub qkv_bias: Option<QkvBiasIds>,
    /// Present in Qwen3/3.5; absent in Qwen2.
    pub qk_norm: Option<QkNormIds>,
    pub o_proj: WeightId,
    pub post_attention_layernorm: WeightId,
    pub gate_proj: WeightId,
    pub up_proj: WeightId,
    pub down_proj: WeightId,
}

/// Weight IDs for a single `MoE` layer (`Qwen3-MoE`).
pub struct MoeLayerWeightIds {
    pub input_layernorm: WeightId,
    pub q_proj: WeightId,
    pub k_proj: WeightId,
    pub v_proj: WeightId,
    pub qkv_bias: Option<QkvBiasIds>,
    pub qk_norm: Option<QkNormIds>,
    pub o_proj: WeightId,
    pub post_attention_layernorm: WeightId,
    /// Gating projection (shape `[num_experts, hidden_size]`).
    pub moe_gate: WeightId,
    /// Per-expert MLP weight IDs.
    pub experts: Vec<MoeExpertIds>,
    /// Shared expert (if `shared_expert_intermediate_size` is set).
    pub shared_gate: Option<WeightId>,
    pub shared_up: Option<WeightId>,
    pub shared_down: Option<WeightId>,
}

/// Per-layer weight IDs — either dense or `MoE`.
pub enum LayerWeightIds {
    Dense(DenseLayerWeightIds),
    Moe(MoeLayerWeightIds),
}

/// Attention weight IDs shared by both dense and `MoE` layers.
pub struct AttnIds<'a> {
    pub input_layernorm: WeightId,
    pub q_proj: WeightId,
    pub k_proj: WeightId,
    pub v_proj: WeightId,
    pub qkv_bias: Option<&'a QkvBiasIds>,
    pub qk_norm: Option<&'a QkNormIds>,
    pub o_proj: WeightId,
    pub post_attention_layernorm: WeightId,
}

impl LayerWeightIds {
    /// Return the attention weight IDs regardless of whether the layer is
    /// dense or `MoE`.  Both variants carry identical attention fields.
    #[must_use]
    pub fn attn_ids(&self) -> AttnIds<'_> {
        match self {
            LayerWeightIds::Dense(d) => AttnIds {
                input_layernorm: d.input_layernorm,
                q_proj: d.q_proj,
                k_proj: d.k_proj,
                v_proj: d.v_proj,
                qkv_bias: d.qkv_bias.as_ref(),
                qk_norm: d.qk_norm.as_ref(),
                o_proj: d.o_proj,
                post_attention_layernorm: d.post_attention_layernorm,
            },
            LayerWeightIds::Moe(m) => AttnIds {
                input_layernorm: m.input_layernorm,
                q_proj: m.q_proj,
                k_proj: m.k_proj,
                v_proj: m.v_proj,
                qkv_bias: m.qkv_bias.as_ref(),
                qk_norm: m.qk_norm.as_ref(),
                o_proj: m.o_proj,
                post_attention_layernorm: m.post_attention_layernorm,
            },
        }
    }
}

/// All weight IDs for the full Qwen model.
pub struct ModelWeightIds {
    /// Embedding table (tensor weight).
    pub embed_tokens: WeightId,
    /// Per-layer weight IDs.
    pub layers: Vec<LayerWeightIds>,
    /// Final RMS norm weight (tensor weight).
    pub final_norm: WeightId,
    /// LM head projection (linear weight).
    /// `None` when `tie_word_embeddings` is true — the embedding table is
    /// reused at inference time.
    pub lm_head: Option<WeightId>,
}

// ---------------------------------------------------------------------------
// Graph builder trait bound
// ---------------------------------------------------------------------------

/// Backend trait bounds required by the Qwen graph builder.
pub trait QwenGraphOps:
    Backend
    + MatmulOps
    + NormOps
    + ArithOps
    + BiasOps
    + EmbedOps
    + TensorOps
    + RopeOps
    + AttentionOps
    + SwigluOps
    + MoeOps
    + MoeSigmoidOps
{
}

impl<B> QwenGraphOps for B where
    B: Backend
        + MatmulOps
        + NormOps
        + ArithOps
        + BiasOps
        + EmbedOps
        + TensorOps
        + RopeOps
        + AttentionOps
        + SwigluOps
        + MoeOps
        + MoeSigmoidOps
{
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Whether this layer uses `MoE` based on `decoder_sparse_step`.
fn is_moe_layer(config: &QwenConfig, layer_idx: usize) -> bool {
    config.is_moe() && config.is_moe_layer(layer_idx)
}

// ---------------------------------------------------------------------------
// Weight registration
// ---------------------------------------------------------------------------

/// Register all model weights and return their IDs.
///
/// Registration order must be identical across prefill and decode graphs so
/// that the same [`WeightStore`] can be shared between them.
#[allow(clippy::too_many_lines)]
fn register_weights<B: Backend + MatmulOps>(
    graph: &mut Graph<B>,
    config: &QwenConfig,
    weight_dtype: DType,
) -> ModelWeightIds {
    let hidden = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads();
    let head_dim = config.head_dim();
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let vocab_size = config.vocab_size;
    let intermediate = config.intermediate_size;
    let num_layers = config.num_hidden_layers;

    let has_qkv_bias = config.has_qkv_bias();
    let has_qk_norm = config.has_qk_norm();

    let embed_tokens = graph.register_tensor_weight(
        "model.embed_tokens.weight",
        &[vocab_size, hidden],
        weight_dtype,
    );

    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let p = format!("model.layers.{i}");

        let input_layernorm = graph.register_tensor_weight(
            format!("{p}.input_layernorm.weight"),
            &[hidden],
            weight_dtype,
        );
        let q_proj = graph.register_linear_weight(
            format!("{p}.self_attn.q_proj.weight"),
            &[q_dim, hidden],
            weight_dtype,
        );
        let k_proj = graph.register_linear_weight(
            format!("{p}.self_attn.k_proj.weight"),
            &[kv_dim, hidden],
            weight_dtype,
        );
        let v_proj = graph.register_linear_weight(
            format!("{p}.self_attn.v_proj.weight"),
            &[kv_dim, hidden],
            weight_dtype,
        );
        let qkv_bias = if has_qkv_bias {
            let qb = graph.register_tensor_weight(
                format!("{p}.self_attn.q_proj.bias"),
                &[q_dim],
                DType::F32,
            );
            let kb = graph.register_tensor_weight(
                format!("{p}.self_attn.k_proj.bias"),
                &[kv_dim],
                DType::F32,
            );
            let vb = graph.register_tensor_weight(
                format!("{p}.self_attn.v_proj.bias"),
                &[kv_dim],
                DType::F32,
            );
            Some(QkvBiasIds {
                q_bias: qb,
                k_bias: kb,
                v_bias: vb,
            })
        } else {
            None
        };
        let qk_norm = if has_qk_norm {
            let qn = graph.register_tensor_weight(
                format!("{p}.self_attn.q_norm.weight"),
                &[head_dim],
                DType::F32,
            );
            let kn = graph.register_tensor_weight(
                format!("{p}.self_attn.k_norm.weight"),
                &[head_dim],
                DType::F32,
            );
            Some(QkNormIds {
                q_norm: qn,
                k_norm: kn,
            })
        } else {
            None
        };
        let o_proj = graph.register_linear_weight(
            format!("{p}.self_attn.o_proj.weight"),
            &[hidden, q_dim],
            weight_dtype,
        );
        let post_attention_layernorm = graph.register_tensor_weight(
            format!("{p}.post_attention_layernorm.weight"),
            &[hidden],
            weight_dtype,
        );

        if is_moe_layer(config, i) {
            let num_experts = config.num_experts.unwrap_or(0);
            let expert_intermediate = config.moe_expert_intermediate_size();

            let moe_gate = graph.register_tensor_weight(
                format!("{p}.mlp.gate.weight"),
                &[num_experts, hidden],
                DType::F32,
            );
            let mut experts = Vec::with_capacity(num_experts);
            for e in 0..num_experts {
                let ep = format!("{p}.mlp.experts.{e}");
                let gate_proj = graph.register_linear_weight(
                    format!("{ep}.gate_proj.weight"),
                    &[expert_intermediate, hidden],
                    weight_dtype,
                );
                let up_proj = graph.register_linear_weight(
                    format!("{ep}.up_proj.weight"),
                    &[expert_intermediate, hidden],
                    weight_dtype,
                );
                let down_proj = graph.register_linear_weight(
                    format!("{ep}.down_proj.weight"),
                    &[hidden, expert_intermediate],
                    weight_dtype,
                );
                experts.push(MoeExpertIds {
                    gate_proj,
                    up_proj,
                    down_proj,
                });
            }

            let (shared_gate, shared_up, shared_down) =
                if let Some(shared_intermediate) = config.shared_expert_intermediate_size {
                    let sg = graph.register_linear_weight(
                        format!("{p}.mlp.shared_expert.gate_proj.weight"),
                        &[shared_intermediate, hidden],
                        weight_dtype,
                    );
                    let su = graph.register_linear_weight(
                        format!("{p}.mlp.shared_expert.up_proj.weight"),
                        &[shared_intermediate, hidden],
                        weight_dtype,
                    );
                    let sd = graph.register_linear_weight(
                        format!("{p}.mlp.shared_expert.down_proj.weight"),
                        &[hidden, shared_intermediate],
                        weight_dtype,
                    );
                    (Some(sg), Some(su), Some(sd))
                } else {
                    (None, None, None)
                };

            layers.push(LayerWeightIds::Moe(MoeLayerWeightIds {
                input_layernorm,
                q_proj,
                k_proj,
                v_proj,
                qkv_bias,
                qk_norm,
                o_proj,
                post_attention_layernorm,
                moe_gate,
                experts,
                shared_gate,
                shared_up,
                shared_down,
            }));
        } else {
            let gate_proj = graph.register_linear_weight(
                format!("{p}.mlp.gate_proj.weight"),
                &[intermediate, hidden],
                weight_dtype,
            );
            let up_proj = graph.register_linear_weight(
                format!("{p}.mlp.up_proj.weight"),
                &[intermediate, hidden],
                weight_dtype,
            );
            let down_proj = graph.register_linear_weight(
                format!("{p}.mlp.down_proj.weight"),
                &[hidden, intermediate],
                weight_dtype,
            );

            layers.push(LayerWeightIds::Dense(DenseLayerWeightIds {
                input_layernorm,
                q_proj,
                k_proj,
                v_proj,
                qkv_bias,
                qk_norm,
                o_proj,
                post_attention_layernorm,
                gate_proj,
                up_proj,
                down_proj,
            }));
        }
    }

    let final_norm = graph.register_tensor_weight("model.norm.weight", &[hidden], weight_dtype);

    // Tied embeddings: if `tie_word_embeddings` the embedding table is reused;
    // no separate lm_head weight is stored in the checkpoint.
    let lm_head = if config.tie_word_embeddings {
        None
    } else {
        Some(graph.register_linear_weight("lm_head.weight", &[vocab_size, hidden], weight_dtype))
    };

    ModelWeightIds {
        embed_tokens,
        layers,
        final_norm,
        lm_head,
    }
}

// ---------------------------------------------------------------------------
// Shared attention sub-graph helper
// ---------------------------------------------------------------------------

/// Build the attention block for a single layer into the graph.
///
/// Handles Q/K/V projection, optional biases, optional QK-norm, `RoPE`,
/// and fused attention (decode or prefill depending on the caller).
/// Returns the output of the attention projection `[seq_len, hidden]`.
#[allow(clippy::too_many_arguments)]
fn build_attention_prefill<B: QwenGraphOps>(
    graph: &mut Graph<B>,
    h_normed: OutputRef,
    lw_q: WeightId,
    lw_k: WeightId,
    lw_v: WeightId,
    lw_o: WeightId,
    qkv_bias: Option<&QkvBiasIds>,
    qk_norm: Option<&QkNormIds>,
    eps: f32,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cos_input: OutputRef,
    sin_input: OutputRef,
    sliding_window: Option<usize>,
) -> OutputRef {
    // Q/K/V projections
    let (q, k, v) = graph.add_linear_triple(h_normed, lw_q, lw_k, lw_v);

    // Optional Q/K/V biases (Qwen2)
    let (q, k, v) = if let Some(bias) = qkv_bias {
        (
            graph.add_bias_add(q, bias.q_bias),
            graph.add_bias_add(k, bias.k_bias),
            graph.add_bias_add(v, bias.v_bias),
        )
    } else {
        (q, k, v)
    };

    // Reshape to 3D
    let q_3d = graph.add_reshape(q, &[seq_len, num_heads, head_dim]);
    let k_3d = graph.add_reshape(k, &[seq_len, num_kv_heads, head_dim]);
    let v_3d = graph.add_reshape(v, &[seq_len, num_kv_heads, head_dim]);

    // Optional per-head QK RMSNorm (Qwen3)
    let (q_3d, k_3d) = if let Some(norms) = qk_norm {
        graph.add_qk_norm(q_3d, k_3d, norms.q_norm, norms.k_norm, eps)
    } else {
        (q_3d, k_3d)
    };

    // RoPE (offset=0 for prefill)
    let q_rope = graph.add_rope(q_3d, cos_input, sin_input, 0);
    let k_rope = graph.add_rope(k_3d, cos_input, sin_input, 0);

    // Fused causal attention
    let attn_out =
        graph.add_fused_attention_prefill(q_rope, k_rope, v_3d, 0, None, None, sliding_window);

    // Reshape and output projection
    let attn_flat = graph.add_reshape(attn_out, &[seq_len, num_heads * head_dim]);
    graph.add_linear(attn_flat, lw_o)
}

/// Build the decode attention block for a single layer.
#[allow(clippy::too_many_arguments)]
fn build_attention_decode<B: QwenGraphOps>(
    graph: &mut Graph<B>,
    h_normed: OutputRef,
    lw_q: WeightId,
    lw_k: WeightId,
    lw_v: WeightId,
    lw_o: WeightId,
    qkv_bias: Option<&QkvBiasIds>,
    qk_norm: Option<&QkNormIds>,
    eps: f32,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cos_input: OutputRef,
    sin_input: OutputRef,
    k_cache_input: OutputRef,
    v_cache_input: OutputRef,
) -> (OutputRef, OutputRef, OutputRef) {
    let (q, k, v) = graph.add_linear_triple(h_normed, lw_q, lw_k, lw_v);

    let (q, k, v) = if let Some(bias) = qkv_bias {
        (
            graph.add_bias_add(q, bias.q_bias),
            graph.add_bias_add(k, bias.k_bias),
            graph.add_bias_add(v, bias.v_bias),
        )
    } else {
        (q, k, v)
    };

    let q_3d = graph.add_reshape(q, &[1, num_heads, head_dim]);
    let k_3d = graph.add_reshape(k, &[1, num_kv_heads, head_dim]);
    let v_3d = graph.add_reshape(v, &[1, num_kv_heads, head_dim]);

    let (q_3d, k_3d) = if let Some(norms) = qk_norm {
        graph.add_qk_norm(q_3d, k_3d, norms.q_norm, norms.k_norm, eps)
    } else {
        (q_3d, k_3d)
    };

    let q_rope = graph.add_rope(q_3d, cos_input, sin_input, 0);
    let k_rope = graph.add_rope(k_3d, cos_input, sin_input, 0);

    let full_k = graph.add_concat_seq(k_cache_input, k_rope);
    let full_v = graph.add_concat_seq(v_cache_input, v_3d);

    // Decode attention: no softcap for Qwen
    let attn_out = graph.add_fused_attention_decode(q_rope, full_k, full_v, None);

    let attn_flat = graph.add_reshape(attn_out, &[1, num_heads * head_dim]);
    let attn_proj = graph.add_linear(attn_flat, lw_o);

    (attn_proj, full_k, full_v)
}

// ---------------------------------------------------------------------------
// Prefill graph
// ---------------------------------------------------------------------------

/// Build a computation graph for the full Qwen prefill pass.
///
/// The graph takes three inputs:
/// 1. `input_ids` — token IDs, shape `[seq_len]`
/// 2. `cos_cache` — `RoPE` cosine cache, shape `[seq_len, head_dim / 2]`
/// 3. `sin_cache` — `RoPE` sine cache, shape `[seq_len, head_dim / 2]`
///
/// And produces `logits` of shape `[seq_len, vocab_size]`.
///
/// # Panics
///
/// Panics if `seq_len == 0`.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn build_prefill_graph<B: QwenGraphOps>(
    config: &QwenConfig,
    seq_len: usize,
    weight_dtype: DType,
) -> (Graph<B>, ModelWeightIds) {
    assert!(seq_len > 0, "seq_len must be > 0");

    let mut graph = Graph::<B>::new();

    let hidden = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads();
    let head_dim = config.head_dim();
    let vocab_size = config.vocab_size;
    let eps = config.rms_norm_eps;
    let num_experts_per_tok = config.num_experts_per_tok.unwrap_or(1);
    let norm_topk = config.norm_topk_prob;

    let model_weights = register_weights(&mut graph, config, weight_dtype);

    // -- Graph inputs --
    let input_ids = graph.add_input(&[seq_len], DType::U32);
    let cos_input = graph.add_input(&[seq_len, head_dim / 2], DType::F32);
    let sin_input = graph.add_input(&[seq_len, head_dim / 2], DType::F32);

    // -- Embedding --
    let mut h = graph.add_embedding_gather(model_weights.embed_tokens, input_ids);

    // -- Transformer layers --
    for (layer_idx, lw) in model_weights.layers.iter().enumerate() {
        let sliding_window = config.effective_sliding_window(layer_idx);

        let AttnIds {
            input_layernorm,
            q_proj,
            k_proj,
            v_proj,
            qkv_bias,
            qk_norm,
            o_proj,
            post_attention_layernorm: post_attn_norm,
        } = lw.attn_ids();

        // 1. Pre-attention norm
        let normed = graph.add_rms_norm(h, input_layernorm, eps);

        // 2. Attention block
        let attn_proj = build_attention_prefill(
            &mut graph,
            normed,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            qkv_bias,
            qk_norm,
            eps,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            cos_input,
            sin_input,
            sliding_window,
        );

        // 3. Residual + post-attention norm
        let h_updated = graph.add_add(h, attn_proj);
        let normed_post = graph.add_rms_norm(h_updated, post_attn_norm, eps);

        // 4. FFN or MoE
        h = match lw {
            LayerWeightIds::Dense(d) => {
                let (gate, up) = graph.add_linear_pair(normed_post, d.gate_proj, d.up_proj);
                let gate_activated = graph.add_silu(gate);
                let activated = graph.add_mul(gate_activated, up);
                let down = graph.add_linear(activated, d.down_proj);
                graph.add_add_inplace(h_updated, down)
            }
            LayerWeightIds::Moe(m) => {
                let moe_out = graph.add_moe_dispatch_softmax(
                    normed_post,
                    m.moe_gate,
                    m.experts.clone(),
                    num_experts_per_tok,
                    norm_topk,
                );
                // The Qwen3-MoE shared expert output is added to the routed
                // expert output before the residual.  The executor's
                // `moe_dispatch_softmax` arm handles only the routed experts,
                // so we model the shared expert as a separate dense MLP.
                let ffn_out = if let (Some(sg), Some(su), Some(sd)) =
                    (m.shared_gate, m.shared_up, m.shared_down)
                {
                    let (sgate, sup) = graph.add_linear_pair(normed_post, sg, su);
                    let sgate_act = graph.add_silu(sgate);
                    let sact = graph.add_mul(sgate_act, sup);
                    let sdown = graph.add_linear(sact, sd);
                    graph.add_add(moe_out, sdown)
                } else {
                    moe_out
                };
                graph.add_add_inplace(h_updated, ffn_out)
            }
        };
    }

    // -- Final norm --
    let normed_final = graph.add_rms_norm(h, model_weights.final_norm, eps);

    // -- LM head --
    // Tied embeddings: `embed_tokens` (tensor weight) is used directly.
    // Non-tied: separate `lm_head` linear weight.
    let logits = if let Some(lm_head) = model_weights.lm_head {
        graph.add_lm_head(normed_final, lm_head, weight_dtype)
    } else {
        // Tied: matmul with embed_tokens (shape [vocab, hidden]) transposed.
        // We use add_lm_head_tied which broadcasts the embedding table.
        // Since the builder only has add_lm_head (linear), we register a
        // synthetic linear weight that reuses the embed_tokens slot.
        // For simplicity, re-register lm_head.weight as tied (the SafeTensors
        // loader will fall back to embed_tokens if lm_head is absent).
        let lm_w =
            graph.register_linear_weight("lm_head.weight", &[vocab_size, hidden], weight_dtype);
        graph.add_lm_head(normed_final, lm_w, weight_dtype)
    };

    graph.set_output(logits.0);

    // Optimize: fuse primitives into efficient compound ops.
    infernum::graph::optimizer::optimize(&mut graph);

    (graph, model_weights)
}

// ---------------------------------------------------------------------------
// Decode graph
// ---------------------------------------------------------------------------

/// Build a computation graph for single-token decode with KV cache.
///
/// Inputs (in order):
/// 1. `input_id` — single token ID, shape `[1]`
/// 2. `cos` — `RoPE` cosine for the current position, shape `[1, head_dim / 2]`
/// 3. `sin` — `RoPE` sine for the current position, shape `[1, head_dim / 2]`
/// 4. `k_cache_i` shape `[kv_len, num_kv_heads, head_dim]` for each layer
///    (indices `4..3+L`)
/// 5. `v_cache_i` shape `[kv_len, num_kv_heads, head_dim]` for each layer
///    (indices `4+L..3+2L`)
///
/// Outputs (in order):
/// 1. `logits` shape `[1, vocab_size]`
/// 2. `full_k_i` shape `[kv_len+1, num_kv_heads, head_dim]` (indices `2..1+L`)
/// 3. `full_v_i` shape `[kv_len+1, num_kv_heads, head_dim]` (indices `2+L..1+2L`)
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn build_decode_graph<B: QwenGraphOps>(
    config: &QwenConfig,
    kv_len: usize,
    weight_dtype: DType,
) -> (Graph<B>, ModelWeightIds) {
    let mut graph = Graph::<B>::new();

    let hidden = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads();
    let head_dim = config.head_dim();
    let vocab_size = config.vocab_size;
    let eps = config.rms_norm_eps;
    let num_layers = config.num_hidden_layers;
    let num_experts_per_tok = config.num_experts_per_tok.unwrap_or(1);
    let norm_topk = config.norm_topk_prob;

    let model_weights = register_weights(&mut graph, config, weight_dtype);

    // -- Inputs --
    let input_id = graph.add_input(&[1], DType::U32);
    let cos_input = graph.add_input(&[1, head_dim / 2], DType::F32);
    let sin_input = graph.add_input(&[1, head_dim / 2], DType::F32);

    let mut k_cache_inputs = Vec::with_capacity(num_layers);
    let mut v_cache_inputs = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        k_cache_inputs.push(graph.add_input(&[kv_len, num_kv_heads, head_dim], weight_dtype));
        v_cache_inputs.push(graph.add_input(&[kv_len, num_kv_heads, head_dim], weight_dtype));
    }

    // -- Embedding --
    let mut h = graph.add_embedding_gather(model_weights.embed_tokens, input_id);

    let mut k_outputs = Vec::with_capacity(num_layers);
    let mut v_outputs = Vec::with_capacity(num_layers);

    for (layer_idx, lw) in model_weights.layers.iter().enumerate() {
        let AttnIds {
            input_layernorm,
            q_proj,
            k_proj,
            v_proj,
            qkv_bias,
            qk_norm,
            o_proj,
            post_attention_layernorm: post_attn_norm,
        } = lw.attn_ids();

        let normed = graph.add_rms_norm(h, input_layernorm, eps);

        let (attn_proj, full_k, full_v) = build_attention_decode(
            &mut graph,
            normed,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            qkv_bias,
            qk_norm,
            eps,
            num_heads,
            num_kv_heads,
            head_dim,
            cos_input,
            sin_input,
            k_cache_inputs[layer_idx],
            v_cache_inputs[layer_idx],
        );

        k_outputs.push(full_k);
        v_outputs.push(full_v);

        let h_updated = graph.add_add(h, attn_proj);
        let normed_post = graph.add_rms_norm(h_updated, post_attn_norm, eps);

        h = match lw {
            LayerWeightIds::Dense(d) => {
                let (gate, up) = graph.add_linear_pair(normed_post, d.gate_proj, d.up_proj);
                let gate_activated = graph.add_silu(gate);
                let activated = graph.add_mul(gate_activated, up);
                let down = graph.add_linear(activated, d.down_proj);
                graph.add_add_inplace(h_updated, down)
            }
            LayerWeightIds::Moe(m) => {
                let moe_out = graph.add_moe_dispatch_softmax(
                    normed_post,
                    m.moe_gate,
                    m.experts.clone(),
                    num_experts_per_tok,
                    norm_topk,
                );
                let ffn_out = if let (Some(sg), Some(su), Some(sd)) =
                    (m.shared_gate, m.shared_up, m.shared_down)
                {
                    let (sgate, sup) = graph.add_linear_pair(normed_post, sg, su);
                    let sgate_act = graph.add_silu(sgate);
                    let sact = graph.add_mul(sgate_act, sup);
                    let sdown = graph.add_linear(sact, sd);
                    graph.add_add(moe_out, sdown)
                } else {
                    moe_out
                };
                graph.add_add_inplace(h_updated, ffn_out)
            }
        };
    }

    // -- Final norm + LM head --
    let normed_final = graph.add_rms_norm(h, model_weights.final_norm, eps);

    let logits = if let Some(lm_head) = model_weights.lm_head {
        graph.add_lm_head(normed_final, lm_head, weight_dtype)
    } else {
        let lm_w =
            graph.register_linear_weight("lm_head.weight", &[vocab_size, hidden], weight_dtype);
        graph.add_lm_head(normed_final, lm_w, weight_dtype)
    };

    // -- Outputs: logits, then K caches, then V caches --
    graph.set_output(logits.0);
    for &k_out in &k_outputs {
        graph.set_output(k_out.0);
    }
    for &v_out in &v_outputs {
        graph.set_output(v_out.0);
    }

    (graph, model_weights)
}

// ---------------------------------------------------------------------------
// Paged decode graph
// ---------------------------------------------------------------------------

/// Build a computation graph for batched paged-KV-cache decode for Qwen.
///
/// Mirrors [`build_decode_graph`] but uses paged KV cache ops instead of
/// `concat_seq`. Supports dense layers, `MoE` layers, Q/K/V biases (`Qwen2`),
/// per-head QK-norm (`Qwen3`), and per-layer sliding window attention.
///
/// # Inputs (in order)
///
/// 0. `input_ids`    — U32, shape `[batch_size]`
/// 1. `cos_input`    — F32, shape `[batch_size, head_dim / 2]`
/// 2. `sin_input`    — F32, shape `[batch_size, head_dim / 2]`
/// 3. `block_tables` — U32, shape `[batch_size, max_blocks_per_seq]`
/// 4. `positions`    — U32, shape `[batch_size]`
/// 5. `seq_lens`     — U32, shape `[batch_size]`
///
/// # Panics
///
/// Panics if `batch_size == 0`.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]
pub fn build_paged_decode_graph<B>(
    config: &QwenConfig,
    batch_size: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
    weight_dtype: DType,
) -> Graph<B>
where
    B: QwenGraphOps + PagedKvCacheOps + PagedAttentionOps,
    Graph<B>: GraphPagedKvCacheOps + GraphPagedAttentionOps,
{
    assert!(batch_size > 0, "batch_size must be > 0");

    let mut graph = Graph::<B>::new();

    let hidden = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads();
    let head_dim = config.head_dim();
    let vocab_size = config.vocab_size;
    let eps = config.rms_norm_eps;
    let num_experts_per_tok = config.num_experts_per_tok.unwrap_or(1);
    let norm_topk = config.norm_topk_prob;

    let model_weights = register_weights(&mut graph, config, weight_dtype);

    // -- Graph inputs --
    let input_ids = graph.add_input(&[batch_size], DType::U32);
    let cos_input = graph.add_input(&[batch_size, head_dim / 2], DType::F32);
    let sin_input = graph.add_input(&[batch_size, head_dim / 2], DType::F32);
    let block_tables = graph.add_input(&[batch_size, max_blocks_per_seq], DType::U32);
    let positions = graph.add_input(&[batch_size], DType::U32);
    let seq_lens = graph.add_input(&[batch_size], DType::U32);

    // -- Embedding --
    let mut h = graph.add_embedding_gather(model_weights.embed_tokens, input_ids);

    for (layer_idx, lw) in model_weights.layers.iter().enumerate() {
        let sliding_window = config.effective_sliding_window(layer_idx);

        let AttnIds {
            input_layernorm,
            q_proj,
            k_proj,
            v_proj,
            qkv_bias,
            qk_norm,
            o_proj,
            post_attention_layernorm: post_attn_norm,
        } = lw.attn_ids();

        // 1. Pre-attention norm
        let normed = graph.add_rms_norm(h, input_layernorm, eps);

        // 2. Q/K/V projections
        let (q, k, v) = graph.add_linear_triple(normed, q_proj, k_proj, v_proj);

        // 3. Optional Q/K/V biases (Qwen2)
        let (q, k, v) = if let Some(bias) = qkv_bias {
            (
                graph.add_bias_add(q, bias.q_bias),
                graph.add_bias_add(k, bias.k_bias),
                graph.add_bias_add(v, bias.v_bias),
            )
        } else {
            (q, k, v)
        };

        // 4. Reshape to 3D
        let q_3d = graph.add_reshape(q, &[batch_size, num_heads, head_dim]);
        let k_3d = graph.add_reshape(k, &[batch_size, num_kv_heads, head_dim]);
        let v_3d = graph.add_reshape(v, &[batch_size, num_kv_heads, head_dim]);

        // 5. Optional per-head QK RMSNorm (Qwen3)
        let (q_3d, k_3d) = if let Some(norms) = qk_norm {
            graph.add_qk_norm(q_3d, k_3d, norms.q_norm, norms.k_norm, eps)
        } else {
            (q_3d, k_3d)
        };

        // 6. RoPE
        let q_rope = graph.add_rope(q_3d, cos_input, sin_input, 0);
        let k_rope = graph.add_rope(k_3d, cos_input, sin_input, 0);

        // 7. Append K/V to paged cache (side-effect)
        let append =
            graph.add_append_paged_batched(k_rope, v_3d, block_tables, positions, layer_idx);

        // 8. Paged attention decode
        //    `append` is listed as a dummy input to force the planner to
        //    schedule the append before this attention node.
        let attn_out = graph.add_paged_attention_decode(
            q_rope,
            block_tables,
            seq_lens,
            positions,
            append,
            layer_idx,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            sliding_window,
            None, // no attention logit softcap
        );

        // 9. Reshape and output projection
        let attn_flat = graph.add_reshape(attn_out, &[batch_size, num_heads * head_dim]);
        let attn_proj = graph.add_linear(attn_flat, o_proj);

        // 10. Residual + post-attention norm
        let h_updated = graph.add_add(h, attn_proj);
        let normed_post = graph.add_rms_norm(h_updated, post_attn_norm, eps);

        // 11. FFN or MoE
        h = match lw {
            LayerWeightIds::Dense(d) => {
                let (gate, up) = graph.add_linear_pair(normed_post, d.gate_proj, d.up_proj);
                let gate_activated = graph.add_silu(gate);
                let activated = graph.add_mul(gate_activated, up);
                let down = graph.add_linear(activated, d.down_proj);
                graph.add_add_inplace(h_updated, down)
            }
            LayerWeightIds::Moe(m) => {
                let moe_out = graph.add_moe_dispatch_softmax(
                    normed_post,
                    m.moe_gate,
                    m.experts.clone(),
                    num_experts_per_tok,
                    norm_topk,
                );
                let ffn_out = if let (Some(sg), Some(su), Some(sd)) =
                    (m.shared_gate, m.shared_up, m.shared_down)
                {
                    let (sgate, sup) = graph.add_linear_pair(normed_post, sg, su);
                    let sgate_act = graph.add_silu(sgate);
                    let sact = graph.add_mul(sgate_act, sup);
                    let sdown = graph.add_linear(sact, sd);
                    graph.add_add(moe_out, sdown)
                } else {
                    moe_out
                };
                graph.add_add_inplace(h_updated, ffn_out)
            }
        };
    }

    // -- Final norm + LM head --
    let normed_final = graph.add_rms_norm(h, model_weights.final_norm, eps);

    let logits = if let Some(lm_head) = model_weights.lm_head {
        graph.add_lm_head(normed_final, lm_head, weight_dtype)
    } else {
        let lm_w =
            graph.register_linear_weight("lm_head.weight", &[vocab_size, hidden], weight_dtype);
        graph.add_lm_head(normed_final, lm_w, weight_dtype)
    };

    // -- Output: only logits --
    graph.set_output(logits.0);

    graph
}

// ---------------------------------------------------------------------------
// CPU weight loading (feature-gated)
// ---------------------------------------------------------------------------

/// Load model weights from a `SafeTensors` directory into a `WeightStore` for
/// graph execution on the CPU backend.
///
/// Handles tied embeddings (Qwen2.5-0.5B style: `lm_head.weight` absent,
/// falls back to `model.embed_tokens.weight`).
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
    _config: &QwenConfig,
) -> infernum::Result<
    infernum::graph::WeightStore<
        infernum_cpu::tensor::CpuTensor,
        infernum_cpu::tensor::CpuLinearWeight,
    >,
> {
    // Qwen models (Qwen2.5, Qwen3) tie `lm_head.weight` to `embed_tokens.weight`.
    infernum_cpu::load_cpu_safetensors_weights(graph, model_dir, true)
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

    impl infernum::backend::SwigluOps for TestBackend {
        fn swiglu(_gate: &DummyTensor, _up: &DummyTensor) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl infernum::backend::BiasOps for TestBackend {
        fn bias_add_inplace(_input: &mut DummyTensor, _bias: &DummyTensor) -> infernum::Result<()> {
            Ok(())
        }
    }

    impl infernum::backend::MoeOps for TestBackend {
        fn moe_forward_softmax<F>(
            _hidden: &DummyTensor,
            _gate_weight: &DummyTensor,
            _num_experts: usize,
            _num_experts_per_tok: usize,
            _norm_topk_prob: bool,
            _expert_fn: F,
        ) -> infernum::Result<DummyTensor>
        where
            F: Fn(usize, &DummyTensor) -> infernum::Result<DummyTensor>,
        {
            Ok(DummyTensor)
        }
    }

    impl infernum::backend::MoeSigmoidOps for TestBackend {
        fn moe_forward_sigmoid<F>(
            _hidden: &DummyTensor,
            _gate_weight: &DummyTensor,
            _e_score_correction_bias: &[f32],
            _num_experts: usize,
            _num_experts_per_tok: usize,
            _n_group: usize,
            _topk_group: usize,
            _norm_topk_prob: bool,
            _routed_scaling_factor: f32,
            _expert_fn: F,
        ) -> infernum::Result<DummyTensor>
        where
            F: Fn(usize, &DummyTensor) -> infernum::Result<DummyTensor>,
        {
            Ok(DummyTensor)
        }
    }

    fn dense_config() -> QwenConfig {
        serde_json::from_str(
            r#"{
                "model_type": "qwen2",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false
            }"#,
        )
        .unwrap()
    }

    fn dense_config_tied() -> QwenConfig {
        serde_json::from_str(
            r#"{
                "model_type": "qwen2",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "tie_word_embeddings": true
            }"#,
        )
        .unwrap()
    }

    fn qwen3_config() -> QwenConfig {
        serde_json::from_str(
            r#"{
                "model_type": "qwen3",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "tie_word_embeddings": true
            }"#,
        )
        .unwrap()
    }

    fn moe_config() -> QwenConfig {
        // Real Qwen3-MoE configs always have head_dim set (triggering QK-norm).
        serde_json::from_str(
            r#"{
                "model_type": "qwen3_moe",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "moe_intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "tie_word_embeddings": true,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn test_build_prefill_graph_dense() {
        let config = dense_config();
        let (_graph, weights): (Graph<TestBackend>, _) =
            build_prefill_graph(&config, 4, DType::F32);
        // Verify we get a non-empty weight store.
        assert!(weights.layers.len() == 2);
        assert!(weights.lm_head.is_some());
    }

    #[test]
    fn test_build_prefill_graph_tied_embeddings() {
        let config = dense_config_tied();
        let (_graph, weights): (Graph<TestBackend>, _) =
            build_prefill_graph(&config, 4, DType::F32);
        // Tied: lm_head slot from register_weights is None.
        assert!(weights.lm_head.is_none());
    }

    #[test]
    fn test_build_prefill_graph_qwen3() {
        let config = qwen3_config();
        let (_graph, weights): (Graph<TestBackend>, _) =
            build_prefill_graph(&config, 4, DType::F32);
        // QK norms present on all layers.
        for lw in &weights.layers {
            if let LayerWeightIds::Dense(d) = lw {
                assert!(d.qk_norm.is_some());
                assert!(d.qkv_bias.is_none());
            }
        }
    }

    #[test]
    fn test_build_prefill_graph_moe() {
        let config = moe_config();
        let (_graph, weights): (Graph<TestBackend>, _) =
            build_prefill_graph(&config, 4, DType::F32);
        // All layers are MoE with decoder_sparse_step=1.
        for lw in &weights.layers {
            assert!(matches!(lw, LayerWeightIds::Moe(_)));
        }
    }

    #[test]
    fn test_build_decode_graph_dense() {
        let config = dense_config();
        let (graph, _): (Graph<TestBackend>, _) = build_decode_graph(&config, 10, DType::F32);
        // Outputs: logits + K per layer + V per layer = 1 + 2 + 2 = 5.
        assert_eq!(graph.output_ids().len(), 5);
    }
}
