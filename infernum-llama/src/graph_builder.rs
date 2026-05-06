//! Graph builder for the Llama model family.
//!
//! Constructs a computation graph ([`Graph<B>`]) representing the full
//! Llama forward pass. The graph captures the complete DAG of operations
//! from token IDs to logits, enabling ahead-of-time memory planning
//! and execution scheduling.
//!
//! Supports:
//! - Dense and `MoE` models (Llama, Mistral, Mixtral)
//! - Prefill, single-token decode, paged-KV decode, and CUDA-graph-compatible
//!   indirect decode graphs
//! - Per-layer sliding window attention via mask (`effective_sliding_window`)
//! - Single-GPU only (no tensor parallelism / `AllReduce`)

use infernum::backend::{
    ArithOps, AttentionOps, Backend, ContextBackend, EmbedOps, MatmulOps, NormOps,
    PagedAttentionOps, PagedKvCacheOps, RopeOps, SwigluOps, TensorOps,
};
use infernum::dtype::DType;
use infernum::graph::{
    Graph, GraphArithOps, GraphAttentionOps, GraphEmbedOps, GraphIndirectDecodeOps, GraphMatmulOps,
    GraphNormOps, GraphPagedAttentionOps, GraphPagedKvCacheOps, GraphRopeOps, GraphSiluOps,
    GraphTensorOps, WeightId,
};

use crate::config::LlamaConfig;

// ---------------------------------------------------------------------------
// Weight ID structures
// ---------------------------------------------------------------------------

/// Weight IDs for a single transformer layer.
pub struct LayerWeightIds {
    /// RMS norm weight before attention.
    pub input_layernorm: WeightId,
    /// Q projection (linear weight).
    pub q_proj: WeightId,
    /// K projection (linear weight).
    pub k_proj: WeightId,
    /// V projection (linear weight).
    pub v_proj: WeightId,
    /// Output projection (linear weight).
    pub o_proj: WeightId,
    /// RMS norm weight before FFN.
    pub post_attention_layernorm: WeightId,
    /// Gate projection (linear weight).
    pub gate_proj: WeightId,
    /// Up projection (linear weight).
    pub up_proj: WeightId,
    /// Down projection (linear weight).
    pub down_proj: WeightId,
}

/// All weight IDs for the full Llama model.
pub struct ModelWeightIds {
    /// Embedding table (tensor weight).
    pub embed_tokens: WeightId,
    /// Per-layer weight IDs.
    pub layers: Vec<LayerWeightIds>,
    /// Final RMS norm weight (tensor weight).
    pub final_norm: WeightId,
    /// LM head projection (linear weight).
    pub lm_head: WeightId,
}

// ---------------------------------------------------------------------------
// Graph builder trait bound
// ---------------------------------------------------------------------------

/// Backend trait bounds required by the Llama graph builder.
///
/// This is the subset of [`LlamaOps`](crate::model::LlamaOps) needed for
/// the prefill graph — excludes paged KV cache, `MoE`, and cast ops.
pub trait LlamaGraphOps:
    Backend
    + MatmulOps
    + ContextBackend
    + NormOps
    + ArithOps
    + EmbedOps
    + TensorOps
    + RopeOps
    + AttentionOps
    + SwigluOps
{
}

impl<B> LlamaGraphOps for B where
    B: Backend
        + MatmulOps
        + ContextBackend
        + NormOps
        + ArithOps
        + EmbedOps
        + TensorOps
        + RopeOps
        + AttentionOps
        + SwigluOps
{
}

// ---------------------------------------------------------------------------
// Weight registration
// ---------------------------------------------------------------------------

/// Register all model weights in the graph and return their IDs.
///
/// Weight names match the `SafeTensors` naming convention used by
/// [`LlamaModel::load_weights`](crate::LlamaModel).
fn register_weights<B: Backend + MatmulOps + ContextBackend>(
    graph: &mut Graph<B>,
    config: &LlamaConfig,
    hidden: usize,
    kv_dim: usize,
    intermediate: usize,
    vocab_size: usize,
    weight_dtype: DType,
) -> ModelWeightIds {
    let q_dim = config.num_attention_heads * config.head_dim();

    let embed_tokens = graph.register_tensor_weight(
        "model.embed_tokens.weight",
        &[vocab_size, hidden],
        weight_dtype,
    );

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
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

        layers.push(LayerWeightIds {
            input_layernorm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            post_attention_layernorm,
            gate_proj,
            up_proj,
            down_proj,
        });
    }

    let final_norm = graph.register_tensor_weight("model.norm.weight", &[hidden], weight_dtype);
    let lm_head =
        graph.register_linear_weight("lm_head.weight", &[vocab_size, hidden], weight_dtype);

    ModelWeightIds {
        embed_tokens,
        layers,
        final_norm,
        lm_head,
    }
}

// ---------------------------------------------------------------------------
// Prefill graph construction
// ---------------------------------------------------------------------------

/// Build a computation graph for the full Llama prefill pass.
///
/// The graph takes three inputs:
/// 1. `input_ids` — token IDs, shape `[seq_len]`
/// 2. `cos_cache` — `RoPE` cosine cache, shape `[seq_len, head_dim / 2]`
/// 3. `sin_cache` — `RoPE` sine cache, shape `[seq_len, head_dim / 2]`
///
/// And produces `logits` of shape `[seq_len, vocab_size]`.
///
/// This graph does **not** use a KV cache — it computes full attention
/// over the entire sequence. Suitable for initial prompt processing
/// or for models running without caching.
///
/// # Panics
///
/// Panics if the config describes an `MoE` model (`num_local_experts > 1`).
#[must_use]
pub fn build_prefill_graph<B: LlamaGraphOps>(
    config: &LlamaConfig,
    seq_len: usize,
    weight_dtype: DType,
) -> (Graph<B>, ModelWeightIds) {
    assert!(
        !config.is_moe(),
        "build_prefill_graph does not support MoE models"
    );

    let mut graph = Graph::<B>::new();

    let hidden = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads();
    let head_dim = config.head_dim();
    let kv_dim = num_kv_heads * head_dim;
    let intermediate = config.intermediate_size;
    let vocab_size = config.vocab_size;
    let eps = config.rms_norm_eps;

    // -- Register weights --
    let model_weights = register_weights(
        &mut graph,
        config,
        hidden,
        kv_dim,
        intermediate,
        vocab_size,
        weight_dtype,
    );

    // -- Graph inputs --
    let input_ids = graph.add_input(&[seq_len], DType::U32);
    let cos_input = graph.add_input(&[seq_len, head_dim / 2], DType::F32);
    let sin_input = graph.add_input(&[seq_len, head_dim / 2], DType::F32);

    // -- Embedding --
    let mut h = graph.add_embedding_gather(model_weights.embed_tokens, input_ids);

    // -- Transformer layers --
    for lw in &model_weights.layers {
        // 1. Pre-attention RMS norm
        let normed = graph.add_rms_norm(h, lw.input_layernorm, eps);

        // 2. Q/K/V projections (fused triple — single dispatch, shared quantization)
        let (q, k, v) = graph.add_linear_triple(normed, lw.q_proj, lw.k_proj, lw.v_proj);

        // 3. Reshape to 3D: [seq_len, num_heads, head_dim]
        let q_3d = graph.add_reshape(q, &[seq_len, num_heads, head_dim]);
        let k_3d = graph.add_reshape(k, &[seq_len, num_kv_heads, head_dim]);
        let v_3d = graph.add_reshape(v, &[seq_len, num_kv_heads, head_dim]);

        // 4. RoPE (offset = 0 for prefill)
        let q_rope = graph.add_rope(q_3d, cos_input, sin_input, 0);
        let k_rope = graph.add_rope(k_3d, cos_input, sin_input, 0);

        // 5. Fused causal attention (full, no sliding window)
        let attn_out = graph.add_fused_attention_prefill(q_rope, k_rope, v_3d, 0, None, None, None);

        // 6. Reshape back to 2D and output projection
        let attn_flat = graph.add_reshape(attn_out, &[seq_len, num_heads * head_dim]);
        let attn_proj = graph.add_linear(attn_flat, lw.o_proj);

        // 7. Residual add + post-attention RMS norm (primitives — optimizer fuses)
        let h_updated = graph.add_add(h, attn_proj);
        let normed_post = graph.add_rms_norm(h_updated, lw.post_attention_layernorm, eps);

        // 8. FFN: SiLU + Mul MLP (primitives — optimizer fuses into Swiglu)
        let (gate, up) = graph.add_linear_pair(normed_post, lw.gate_proj, lw.up_proj);
        let gate_activated = graph.add_silu(gate);
        let activated = graph.add_mul(gate_activated, up);
        let down = graph.add_linear(activated, lw.down_proj);

        // 9. Residual add
        h = graph.add_add_inplace(h_updated, down);
    }

    // -- Final norm --
    let normed_final = graph.add_rms_norm(h, model_weights.final_norm, eps);

    // -- LM head --
    let logits = graph.add_lm_head(normed_final, model_weights.lm_head, weight_dtype);
    graph.set_output(logits.0);

    // -- Optimize: fuse primitives into efficient compound ops --
    infernum::graph::optimizer::optimize(&mut graph);

    (graph, model_weights)
}

// ---------------------------------------------------------------------------
// Decode graph construction
// ---------------------------------------------------------------------------

/// Build a computation graph for single-token decode with KV cache.
///
/// The graph takes these inputs (in order):
/// 1. `input_id` — single token ID, shape `[1]`
/// 2. `cos_cache` — `RoPE` cosine for the current position, shape `[1, head_dim / 2]`
/// 3. `sin_cache` — `RoPE` sine for the current position, shape `[1, head_dim / 2]`
/// 4. For each layer `i`: `k_cache_i` of shape `[kv_len, num_kv_heads, head_dim]`
/// 5. For each layer `i`: `v_cache_i` of shape `[kv_len, num_kv_heads, head_dim]`
///
/// And produces these outputs (in order):
/// 1. `logits` of shape `[1, vocab_size]`
/// 2. For each layer `i`: `full_k_i` of shape `[kv_len + 1, num_kv_heads, head_dim]`
/// 3. For each layer `i`: `full_v_i` of shape `[kv_len + 1, num_kv_heads, head_dim]`
///
/// The caller feeds the updated K/V outputs back as inputs on the next step.
///
/// # Panics
///
/// Panics if the config describes an `MoE` model (`num_local_experts > 1`).
#[must_use]
pub fn build_decode_graph<B: LlamaGraphOps>(
    config: &LlamaConfig,
    kv_len: usize,
    weight_dtype: DType,
) -> (Graph<B>, ModelWeightIds) {
    assert!(
        !config.is_moe(),
        "build_decode_graph does not support MoE models"
    );

    let mut graph = Graph::<B>::new();

    let hidden = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads();
    let head_dim = config.head_dim();
    let kv_dim = num_kv_heads * head_dim;
    let intermediate = config.intermediate_size;
    let vocab_size = config.vocab_size;
    let eps = config.rms_norm_eps;

    // -- Register weights --
    let model_weights = register_weights(
        &mut graph,
        config,
        hidden,
        kv_dim,
        intermediate,
        vocab_size,
        weight_dtype,
    );

    // -- Graph inputs --
    let input_id = graph.add_input(&[1], DType::U32);
    let cos_input = graph.add_input(&[1, head_dim / 2], DType::F32);
    let sin_input = graph.add_input(&[1, head_dim / 2], DType::F32);

    // KV cache inputs: per-layer k and v
    let num_layers = config.num_hidden_layers;
    let mut k_cache_inputs = Vec::with_capacity(num_layers);
    let mut v_cache_inputs = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        k_cache_inputs.push(graph.add_input(&[kv_len, num_kv_heads, head_dim], weight_dtype));
        v_cache_inputs.push(graph.add_input(&[kv_len, num_kv_heads, head_dim], weight_dtype));
    }

    // -- Embedding --
    let mut h = graph.add_embedding_gather(model_weights.embed_tokens, input_id);

    // -- Transformer layers --
    let mut k_outputs = Vec::with_capacity(num_layers);
    let mut v_outputs = Vec::with_capacity(num_layers);

    for (layer_idx, lw) in model_weights.layers.iter().enumerate() {
        // 1. Pre-attention RMS norm
        let normed = graph.add_rms_norm(h, lw.input_layernorm, eps);

        // 2. Q/K/V projections (fused triple)
        let (q, k, v) = graph.add_linear_triple(normed, lw.q_proj, lw.k_proj, lw.v_proj);

        // 3. Reshape to 3D: [1, num_heads, head_dim]
        let q_3d = graph.add_reshape(q, &[1, num_heads, head_dim]);
        let k_3d = graph.add_reshape(k, &[1, num_kv_heads, head_dim]);
        let v_3d = graph.add_reshape(v, &[1, num_kv_heads, head_dim]);

        // 4. RoPE — offset=0 because cos/sin inputs contain data for just the
        //    current position (the caller pre-indexes into the RoPE table).
        let q_rope = graph.add_rope(q_3d, cos_input, sin_input, 0);
        let k_rope = graph.add_rope(k_3d, cos_input, sin_input, 0);

        // 5. Concatenate new K/V with cached K/V
        let full_k = graph.add_concat_seq(k_cache_inputs[layer_idx], k_rope);
        let full_v = graph.add_concat_seq(v_cache_inputs[layer_idx], v_3d);

        k_outputs.push(full_k);
        v_outputs.push(full_v);

        // 6. Decode attention: Q [1, heads, dim] against full K/V [kv_len+1, kv_heads, dim]
        let attn_out = graph.add_fused_attention_decode(q_rope, full_k, full_v, None);

        // 7. Reshape back to 2D and output projection
        let attn_flat = graph.add_reshape(attn_out, &[1, num_heads * head_dim]);
        let attn_proj = graph.add_linear(attn_flat, lw.o_proj);

        // 8. Residual add + post-attention RMS norm (primitives — optimizer fuses)
        let h_updated = graph.add_add(h, attn_proj);
        let normed_post = graph.add_rms_norm(h_updated, lw.post_attention_layernorm, eps);

        // 9. FFN: SiLU + Mul MLP (primitives — optimizer fuses into Swiglu)
        let (gate, up) = graph.add_linear_pair(normed_post, lw.gate_proj, lw.up_proj);
        let gate_activated = graph.add_silu(gate);
        let activated = graph.add_mul(gate_activated, up);
        let down = graph.add_linear(activated, lw.down_proj);

        // 10. Residual add
        h = graph.add_add_inplace(h_updated, down);
    }

    // -- Final norm --
    let normed_final = graph.add_rms_norm(h, model_weights.final_norm, eps);

    // -- LM head --
    let logits = graph.add_lm_head(normed_final, model_weights.lm_head, weight_dtype);

    // -- Outputs: logits first, then per-layer K/V caches --
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
// Paged decode graph construction
// ---------------------------------------------------------------------------

/// Build a computation graph for batched paged-KV-cache decode.
///
/// Unlike [`build_decode_graph`], this graph:
/// - Accepts a batch of tokens (one per sequence) rather than a single token.
/// - Uses [`GraphPagedKvCacheOps::add_append_paged_batched`] to write new K/V
///   into the paged pool (side-effect, no tensor output).
/// - Uses [`GraphPagedAttentionOps::add_paged_attention_decode`] to look up K/V
///   from the paged pool and compute attention.
/// - Produces **only** `logits` — no per-layer K/V outputs.
///
/// # Inputs (in order)
///
/// 0. `input_ids` — U32, shape `[batch_size]`
/// 1. `cos_input`  — F32, shape `[batch_size, head_dim / 2]`
/// 2. `sin_input`  — F32, shape `[batch_size, head_dim / 2]`
/// 3. `block_tables` — U32, shape `[batch_size, max_blocks_per_seq]`
/// 4. `positions`  — U32, shape `[batch_size]` (write index for append)
/// 5. `seq_lens`   — U32, shape `[batch_size]` (= positions + 1, for attention)
///
/// # Panics
///
/// Panics if the config describes an `MoE` model.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn build_paged_decode_graph<B>(
    config: &LlamaConfig,
    batch_size: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
    weight_dtype: DType,
) -> Graph<B>
where
    B: LlamaGraphOps + PagedKvCacheOps + PagedAttentionOps,
    Graph<B>: GraphPagedKvCacheOps + GraphPagedAttentionOps,
{
    assert!(
        !config.is_moe(),
        "build_paged_decode_graph does not support MoE models"
    );

    let mut graph = Graph::<B>::new();

    let hidden = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads();
    let head_dim = config.head_dim();
    let kv_dim = num_kv_heads * head_dim;
    let intermediate = config.intermediate_size;
    let vocab_size = config.vocab_size;
    let eps = config.rms_norm_eps;

    // -- Register weights (same order as build_decode_graph / build_prefill_graph) --
    let model_weights = register_weights(
        &mut graph,
        config,
        hidden,
        kv_dim,
        intermediate,
        vocab_size,
        weight_dtype,
    );

    // -- Graph inputs --
    let input_ids = graph.add_input(&[batch_size], DType::U32);
    let cos_input = graph.add_input(&[batch_size, head_dim / 2], DType::F32);
    let sin_input = graph.add_input(&[batch_size, head_dim / 2], DType::F32);
    let block_tables = graph.add_input(&[batch_size, max_blocks_per_seq], DType::U32);
    let positions = graph.add_input(&[batch_size], DType::U32);
    let seq_lens = graph.add_input(&[batch_size], DType::U32);

    // -- Embedding --
    let mut h = graph.add_embedding_gather(model_weights.embed_tokens, input_ids);

    // -- Transformer layers --
    for (layer_idx, lw) in model_weights.layers.iter().enumerate() {
        // 1. Pre-attention RMS norm
        let normed = graph.add_rms_norm(h, lw.input_layernorm, eps);

        // 2. Q/K/V projections
        let (q, k, v) = graph.add_linear_triple(normed, lw.q_proj, lw.k_proj, lw.v_proj);

        // 3. Reshape to 3D: [batch_size, num_heads, head_dim]
        let q_3d = graph.add_reshape(q, &[batch_size, num_heads, head_dim]);
        let k_3d = graph.add_reshape(k, &[batch_size, num_kv_heads, head_dim]);
        let v_3d = graph.add_reshape(v, &[batch_size, num_kv_heads, head_dim]);

        // 4. RoPE — offset = 0; cos/sin inputs carry per-token position data.
        let q_rope = graph.add_rope(q_3d, cos_input, sin_input, 0);
        let k_rope = graph.add_rope(k_3d, cos_input, sin_input, 0);

        // 5. Append K/V to paged cache (side-effect, no output tensor used).
        let append =
            graph.add_append_paged_batched(k_rope, v_3d, block_tables, positions, layer_idx);

        // 6. Paged attention decode: reads K/V from paged pool.
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
            config.effective_sliding_window(layer_idx),
            None, // no attention logit softcap
        );

        // 7. Reshape back to 2D and output projection
        let attn_flat = graph.add_reshape(attn_out, &[batch_size, num_heads * head_dim]);
        let attn_proj = graph.add_linear(attn_flat, lw.o_proj);

        // 8. Residual add + post-attention RMS norm
        let h_updated = graph.add_add(h, attn_proj);
        let normed_post = graph.add_rms_norm(h_updated, lw.post_attention_layernorm, eps);

        // 9. FFN: SiLU + Mul MLP
        let (gate, up) = graph.add_linear_pair(normed_post, lw.gate_proj, lw.up_proj);
        let gate_activated = graph.add_silu(gate);
        let activated = graph.add_mul(gate_activated, up);
        let down = graph.add_linear(activated, lw.down_proj);

        // 10. Residual add
        h = graph.add_add_inplace(h_updated, down);
    }

    // -- Final norm --
    let normed_final = graph.add_rms_norm(h, model_weights.final_norm, eps);

    // -- LM head --
    let logits = graph.add_lm_head(normed_final, model_weights.lm_head, weight_dtype);

    // -- Output: only logits (no K/V outputs — cache lives in PagedKvCache) --
    graph.set_output(logits.0);

    graph
}

// ---------------------------------------------------------------------------
// Indirect decode graph construction (CUDA graph compatible)
// ---------------------------------------------------------------------------

/// IDs for the additional weight tensors registered by [`build_indirect_decode_graph`].
///
/// These cover the pre-allocated `KV` cache buffers and `RoPE` tables that are
/// treated as graph weights (stable GPU addresses) rather than inputs.
pub struct IndirectDecodeExtraIds {
    /// Number of tensor weights belonging to the model (`embed_tokens`, norms, etc.).
    /// Tensor weight IDs in the range `[model_tensor_weight_count, ∞)` are `RoPE`
    /// caches and `KV` buffers that the caller must allocate on the GPU — they are
    /// not stored in the model checkpoint files.
    pub model_tensor_weight_count: usize,
    /// Weight ID of the cosine `RoPE` cache `[max_seq_len, head_dim/2]`.
    pub cos_cache: WeightId,
    /// Weight ID of the sine `RoPE` cache `[max_seq_len, head_dim/2]`.
    pub sin_cache: WeightId,
    /// Per-layer K cache weight IDs, shape `[max_seq_len, num_kv_heads, head_dim]`.
    pub k_caches: Vec<WeightId>,
    /// Per-layer V cache weight IDs, shape `[max_seq_len, num_kv_heads, head_dim]`.
    pub v_caches: Vec<WeightId>,
}

/// Build a computation graph for CUDA-graph-compatible single-token decode.
///
/// Unlike [`build_decode_graph`], this graph:
///
/// - Takes **zero** graph inputs — the token ID and sequence position are
///   provided out-of-band via a `SeqPosition` GPU pointer.
/// - Registers the pre-allocated `KV` cache buffers and full `RoPE` tables as
///   **tensor weights** (stable GPU addresses), so the graph captures the same
///   device pointers across all decode steps.
/// - Uses indirect op variants (`embedding_gather_indirect`, `rope_indirect`,
///   `append_kv_indirect`, `fused_attention_decode_indirect`) that read
///   dynamically-changing values from stable device addresses at execution time.
///
/// This makes it possible to capture the entire forward pass into a single
/// `cudaGraphExec_t` and replay it with `cuGraphExecUpdate_v2` each step.
///
/// # Arguments
///
/// * `config` — Model configuration.
/// * `max_seq_len` — Maximum total sequence length (prompt + generated). Determines
///   the pre-allocated `KV` cache size.
/// * `weight_dtype` — Data type for model weights and `KV` cache buffers.
///
/// # Returns
///
/// A tuple of `(graph, model_weight_ids, extra_ids)` where `extra_ids` contains
/// the weight IDs for the `RoPE` tables and `KV` cache buffers that the caller must
/// populate in the `WeightStore` before execution.
///
/// # Panics
///
/// Panics if the config describes an `MoE` model (`num_local_experts > 1`).
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn build_indirect_decode_graph<B: LlamaGraphOps>(
    config: &LlamaConfig,
    max_seq_len: usize,
    weight_dtype: DType,
) -> (Graph<B>, ModelWeightIds, IndirectDecodeExtraIds) {
    assert!(
        !config.is_moe(),
        "build_indirect_decode_graph does not support MoE models"
    );

    let mut graph = Graph::<B>::new();

    let hidden = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads();
    let head_dim = config.head_dim();
    let kv_dim = num_kv_heads * head_dim;
    let intermediate = config.intermediate_size;
    let vocab_size = config.vocab_size;
    let half_dim = head_dim / 2;
    let eps = config.rms_norm_eps;
    let num_layers = config.num_hidden_layers;
    #[allow(clippy::cast_precision_loss)]
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // -- Register model weights (same ordering as prefill graph for weight reuse) --
    let model_weights = register_weights(
        &mut graph,
        config,
        hidden,
        kv_dim,
        intermediate,
        vocab_size,
        weight_dtype,
    );

    // Snapshot the tensor weight count after model weights are registered.
    // Weights registered after this point (RoPE caches, KV buffers) must be
    // allocated by the caller — they are not stored in checkpoint files.
    let model_tensor_weight_count = graph.tensor_weight_count();

    // -- Register RoPE caches as tensor weights (stable GPU addresses) --
    let cos_cache =
        graph.register_tensor_weight("rope.cos_cache", &[max_seq_len, half_dim], DType::F32);
    let sin_cache =
        graph.register_tensor_weight("rope.sin_cache", &[max_seq_len, half_dim], DType::F32);

    // -- Register per-layer KV cache buffers as tensor weights (stable GPU addresses) --
    // The executor fetches these at execution time via kv_cache.full_buffers(layer_idx).
    // They are recorded here so the caller can populate the WeightStore before execution,
    // but they are NOT injected as graph inputs — the executor reads them out-of-band.
    let mut k_caches = Vec::with_capacity(num_layers);
    let mut v_caches = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let k_id = graph.register_tensor_weight(
            format!("kv_cache.layer{i}.k"),
            &[max_seq_len, num_kv_heads, head_dim],
            weight_dtype,
        );
        let v_id = graph.register_tensor_weight(
            format!("kv_cache.layer{i}.v"),
            &[max_seq_len, num_kv_heads, head_dim],
            weight_dtype,
        );
        k_caches.push(k_id);
        v_caches.push(v_id);
    }

    // -- Embedding (reads token ID from SeqPosition device pointer) --
    let mut h =
        graph.add_embedding_gather_indirect(model_weights.embed_tokens, hidden, weight_dtype);

    // -- Transformer layers --
    let layers: Vec<_> = model_weights.layers.iter().enumerate().collect();
    let num_layers = layers.len();

    // Pre-compute normed for layer 0: no preceding residual add, so plain rms_norm.
    let mut normed = graph.add_rms_norm(h, layers[0].1.input_layernorm, eps);

    for &(layer_idx, lw) in &layers {
        // 1. Q/K/V projections (fused triple) — `normed` carried from previous iteration
        //    (or pre-computed above for layer 0).
        let (q, k, v) = graph.add_linear_triple(normed, lw.q_proj, lw.k_proj, lw.v_proj);

        // 2. Reshape to 3D: [1, num_heads, head_dim]
        let q_3d = graph.add_reshape(q, &[1, num_heads, head_dim]);
        let k_3d = graph.add_reshape(k, &[1, num_kv_heads, head_dim]);
        let v_3d = graph.add_reshape(v, &[1, num_kv_heads, head_dim]);

        // 3. Indirect RoPE (position read from SeqPosition at execution time)
        let q_rope =
            graph.add_rope_indirect(q_3d, cos_cache, sin_cache, false, head_dim, num_heads);
        let k_rope =
            graph.add_rope_indirect(k_3d, cos_cache, sin_cache, false, head_dim, num_kv_heads);

        // 4. Indirect KV append (write offset read from SeqPosition at execution time)
        let _k_append =
            graph.add_append_kv_indirect(k_rope, layer_idx, true, num_kv_heads, head_dim);
        let _v_append =
            graph.add_append_kv_indirect(v_3d, layer_idx, false, num_kv_heads, head_dim);

        // 5. Indirect decode attention: K/V buffers and total_len are read from the
        //    executor's out-of-band KvCache (indexed by layer_idx).
        let attn_out = graph.add_fused_attention_decode_indirect(
            q_rope,
            layer_idx,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            None,
            config.effective_sliding_window(layer_idx),
        );

        // 6. Reshape back to 2D and output projection
        let attn_flat = graph.add_reshape(attn_out, &[1, hidden]);
        let attn_proj = graph.add_linear(attn_flat, lw.o_proj);

        // 7. Fused: residual add(h, attn_proj) + rms_norm(post_attention_layernorm)
        let (h_updated, normed_post) =
            graph.add_add_rmsnorm(h, attn_proj, lw.post_attention_layernorm, eps);

        // 8. FFN: SiLU + Mul MLP
        let (gate, up) = graph.add_linear_pair(normed_post, lw.gate_proj, lw.up_proj);
        let gate_activated = graph.add_silu(gate);
        let activated = graph.add_mul(gate_activated, up);
        let down = graph.add_linear(activated, lw.down_proj);

        // 9. Fused: residual add(h_updated, down) + rms_norm(next layer's input_layernorm),
        //    or plain add on the last layer (no next norm to fuse).
        if layer_idx + 1 < num_layers {
            let next_lw = layers[layer_idx + 1].1;
            let (h_new, normed_next) =
                graph.add_add_rmsnorm(h_updated, down, next_lw.input_layernorm, eps);
            h = h_new;
            normed = normed_next;
        } else {
            h = graph.add_add_inplace(h_updated, down);
        }
    }

    // -- Final norm --
    let normed_final = graph.add_rms_norm(h, model_weights.final_norm, eps);

    // -- LM head --
    let logits = graph.add_lm_head(normed_final, model_weights.lm_head, weight_dtype);

    // -- Argmax: keep token selection on-device so the CUDA graph replays the
    //    full decode step without a D→H sync on every step. The caller reads
    //    one u32 (4 bytes) instead of the full logits tensor. --
    let token = graph.add_argmax_last(logits);

    // -- Output: U32 token tensor [1] --
    graph.set_output(token.0);

    let extra = IndirectDecodeExtraIds {
        model_tensor_weight_count,
        cos_cache,
        sin_cache,
        k_caches,
        v_caches,
    };

    (graph, model_weights, extra)
}

// ---------------------------------------------------------------------------
// GGUF weight loader helpers
// ---------------------------------------------------------------------------

/// Map a SafeTensors-convention weight name to its GGUF tensor name.
///
/// Handles the top-level tensors (`model.embed_tokens.weight`, `model.norm.weight`,
/// `lm_head.weight`) and all per-layer projections / norms.
///
/// # Panics
///
/// Panics on an unrecognised layer suffix (indicates a bug in weight registration).
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
            "post_attention_layernorm.weight" => "ffn_norm.weight",
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            other => panic!("Unknown layer suffix: {other}"),
        };
        return format!("blk.{layer_idx}.{gguf_suffix}");
    }
    panic!("Unknown weight name: {name}");
}

/// Returns `true` if this GGUF tensor name is a Q or K projection that needs
/// the GGUF row-permutation reversal before use.
#[cfg(feature = "cpu")]
fn needs_unpermute(gguf_name: &str) -> bool {
    gguf_name.ends_with(".attn_q.weight") || gguf_name.ends_with(".attn_k.weight")
}

/// Load model weights from a GGUF file into a [`WeightStore`] for graph
/// execution on the CPU backend.
///
/// Weights are loaded in their native quantization format (`Q8_0`, `Q4_0`, F32,
/// etc.) using the same quantized kernels as the eager path. Dequantization
/// happens lazily inside each matmul kernel, so no extra memory is needed for
/// a full-precision copy of the weights.
///
/// # Arguments
///
/// * `graph` — A graph built for this model (used only to read weight metadata
///   and determine slot order). A 1-token prefill graph is sufficient.
/// * `config` — Llama model configuration (used for `num_attention_heads` /
///   `num_key_value_heads` when un-permuting Q/K rows).
/// * `gguf_path` — Path to the `.gguf` file.
///
/// # Errors
///
/// Returns an error if the GGUF file cannot be opened, a required tensor is
/// missing, or a weight cannot be uploaded to the CPU backend (e.g. unsupported
/// quantization type such as `Q6_K`).
///
/// # Panics
///
/// Panics if the number of registered weights exceeds `u32::MAX`.
#[cfg(feature = "cpu")]
pub fn load_graph_weights_gguf(
    graph: &infernum::graph::Graph<infernum_cpu::CpuBackend>,
    config: &LlamaConfig,
    gguf_path: &std::path::Path,
) -> infernum::Result<
    infernum::graph::WeightStore<
        infernum_cpu::tensor::CpuTensor,
        infernum_cpu::tensor::CpuLinearWeight,
    >,
> {
    use infernum::graph::WeightId;
    use infernum::weights::format::{host_transpose_2d, host_unpermute_f32, FormatLoader};
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

    // ── Tensor weights (embeddings, layernorms) — always loaded as F32 ───────
    for i in 0..tensor_count {
        let meta = graph.tensor_weight_meta(WeightId::from_index(
            u32::try_from(i).expect("weight count exceeds u32"),
        ));
        let gguf_name = safetensors_to_gguf_name(&meta.name);
        let host = loader.load_f32(&gguf_name)?;
        store.push_tensor_weight(CpuTensor::from_f32(&host.shape, host.as_f32_slice()));
    }

    // ── Linear weights — loaded in native format (quantized or dense) ─────────
    for i in 0..linear_count {
        let meta = graph.linear_weight_meta(WeightId::from_index(
            u32::try_from(i).expect("weight count exceeds u32"),
        ));
        let gguf_name = safetensors_to_gguf_name(&meta.name);

        // Resolve the actual GGUF name, falling back to tied embeddings.
        let actual_name = if loader.contains(&gguf_name) {
            gguf_name.clone()
        } else if meta.name == "lm_head.weight" {
            "token_embd.weight".to_string()
        } else {
            return Err(infernum::Error::WeightNotFound(gguf_name));
        };

        let dtype = FormatLoader::get_dtype(&loader, &actual_name)?;

        let host_linear = if dtype.is_quantized() {
            // Preserve native quantization — fast Q8_0/Q4_0 kernels will be
            // used at inference time.  Q/K projections need GGUF un-permuting
            // so that the HuggingFace-convention RoPE is correct.
            if needs_unpermute(&gguf_name) {
                let n_head = if gguf_name.contains("attn_q") {
                    config.num_attention_heads
                } else {
                    config
                        .num_key_value_heads
                        .unwrap_or(config.num_attention_heads)
                };
                HostLinearWeight::Quantized(FormatLoader::load_quantized_unpermute(
                    &loader,
                    &actual_name,
                    n_head,
                )?)
            } else {
                HostLinearWeight::Quantized(FormatLoader::load_quantized(&loader, &actual_name)?)
            }
        } else {
            // Dense path: load as F32, un-permute Q/K rows if required, then transpose.
            let host = loader.load_f32(&actual_name)?;
            let host = if needs_unpermute(&gguf_name) {
                let n_head = if gguf_name.contains("attn_q") {
                    config.num_attention_heads
                } else {
                    config
                        .num_key_value_heads
                        .unwrap_or(config.num_attention_heads)
                };
                host_unpermute_f32(&host, n_head)?
            } else {
                host
            };
            HostLinearWeight::Dense(host_transpose_2d(&host)?)
        };

        let linear = CpuBackend::upload_host_linear(&(), &host_linear)?;
        store.push_linear_weight(linear);
    }

    Ok(store)
}

#[cfg(test)]
mod tests {
    use super::*;

    use infernum::graph::test_helpers::{DummyLogits, DummyRuntimeState, DummyTensor};

    /// Minimal backend for graph construction tests.
    struct TestBackend;

    impl infernum::backend::Backend for TestBackend {
        type Tensor = DummyTensor;
        type ExecutorState = ();
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
            _cos_cache: &DummyTensor,
            _sin_cache: &DummyTensor,
            _position_offset: usize,
        ) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn apply_rope_batched(
            _input: &DummyTensor,
            _cos_cache: &DummyTensor,
            _sin_cache: &DummyTensor,
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

    impl infernum::backend::ContextBackend for TestBackend {
        fn ctx_read(
            _ctx: &infernum::graph::execute_context::ExecuteContext<'_, Self>,
            _output_ref: infernum::graph::OutputRef,
        ) -> DummyTensor {
            DummyTensor
        }
        fn ctx_write(
            _ctx: &mut infernum::graph::execute_context::ExecuteContext<'_, Self>,
            _node_id: infernum::graph::NodeId,
            _idx: u32,
            _tensor: DummyTensor,
        ) {
        }
        fn ctx_next_input(
            _ctx: &mut infernum::graph::execute_context::ExecuteContext<'_, Self>,
        ) -> DummyTensor {
            DummyTensor
        }
    }

    /// Llama 3.2 1B-like config for testing.
    fn test_config() -> LlamaConfig {
        serde_json::from_str(
            r#"{
                "vocab_size": 32000,
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "num_hidden_layers": 2,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-5
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn prefill_graph_output_shape() {
        let config = test_config();
        let seq_len = 16;
        let (graph, _weights) = build_prefill_graph::<TestBackend>(&config, seq_len, DType::BF16);

        assert_eq!(graph.output_ids().len(), 1);
        let logits_id = graph.output_ids()[0];
        let logits_ref = (logits_id, 0);
        assert_eq!(graph.node_shape(logits_ref), &[seq_len, config.vocab_size]);
        assert_eq!(graph.node_dtype(logits_ref), DType::F32);
    }

    #[test]
    fn prefill_graph_has_correct_input_count() {
        let config = test_config();
        let (graph, _weights) = build_prefill_graph::<TestBackend>(&config, 8, DType::BF16);

        // Should have exactly 3 Input nodes: input_ids, cos_cache, sin_cache
        let input_count = graph
            .nodes()
            .iter()
            .filter(|n| n.op.name() == "input")
            .count();
        assert_eq!(input_count, 3);
    }

    #[test]
    fn prefill_graph_weight_names_match_convention() {
        let config = test_config();
        let (graph, weights) = build_prefill_graph::<TestBackend>(&config, 8, DType::BF16);

        // Verify embed weight name
        assert_eq!(
            graph.tensor_weight_meta(weights.embed_tokens).name,
            "model.embed_tokens.weight"
        );

        // Verify layer 0 weight names
        let lw = &weights.layers[0];
        assert_eq!(
            graph.tensor_weight_meta(lw.input_layernorm).name,
            "model.layers.0.input_layernorm.weight"
        );
        assert_eq!(
            graph.linear_weight_meta(lw.q_proj).name,
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            graph.linear_weight_meta(lw.k_proj).name,
            "model.layers.0.self_attn.k_proj.weight"
        );
        assert_eq!(
            graph.linear_weight_meta(lw.v_proj).name,
            "model.layers.0.self_attn.v_proj.weight"
        );
        assert_eq!(
            graph.linear_weight_meta(lw.o_proj).name,
            "model.layers.0.self_attn.o_proj.weight"
        );
        assert_eq!(
            graph.tensor_weight_meta(lw.post_attention_layernorm).name,
            "model.layers.0.post_attention_layernorm.weight"
        );
        assert_eq!(
            graph.linear_weight_meta(lw.gate_proj).name,
            "model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            graph.linear_weight_meta(lw.up_proj).name,
            "model.layers.0.mlp.up_proj.weight"
        );
        assert_eq!(
            graph.linear_weight_meta(lw.down_proj).name,
            "model.layers.0.mlp.down_proj.weight"
        );

        // Verify final norm and lm_head
        assert_eq!(
            graph.tensor_weight_meta(weights.final_norm).name,
            "model.norm.weight"
        );
        assert_eq!(
            graph.linear_weight_meta(weights.lm_head).name,
            "lm_head.weight"
        );
    }

    #[test]
    fn prefill_graph_layer_count_matches_config() {
        let config = test_config();
        let (_graph, weights) = build_prefill_graph::<TestBackend>(&config, 8, DType::BF16);

        assert_eq!(weights.layers.len(), config.num_hidden_layers);
    }

    #[test]
    #[should_panic(expected = "does not support MoE")]
    fn prefill_graph_panics_on_moe() {
        let config: LlamaConfig = serde_json::from_str(
            r#"{
                "model_type": "mixtral",
                "vocab_size": 32000,
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_hidden_layers": 2,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_local_experts": 8,
                "num_experts_per_tok": 2
            }"#,
        )
        .unwrap();

        let _ = build_prefill_graph::<TestBackend>(&config, 8, DType::BF16);
    }

    #[test]
    fn prefill_graph_mha_same_kv_heads() {
        // MHA config: num_kv_heads == num_attention_heads
        let config: LlamaConfig = serde_json::from_str(
            r#"{
                "vocab_size": 32000,
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 1,
                "num_attention_heads": 32
            }"#,
        )
        .unwrap();

        let seq_len = 4;
        let (graph, _weights) = build_prefill_graph::<TestBackend>(&config, seq_len, DType::BF16);

        let logits_id = graph.output_ids()[0];
        assert_eq!(graph.node_shape((logits_id, 0)), &[seq_len, 32000]);
    }
}
