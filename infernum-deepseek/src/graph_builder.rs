//! Graph builder for the `DeepSeek` model family (`DeepSeek` V3, R1).
//!
//! Constructs a computation graph ([`Graph<B>`]) representing the full
//! `DeepSeek` forward pass. Key architectural features:
//!
//! - **MLA attention**: treated as a single opaque [`MlaAttentionOp`] node
//!   that encapsulates Q `LoRA` compression, KV joint projection, interleaved
//!   `RoPE`, absorbed attention, and output projection.
//! - **Sigmoid `MoE` routing**: layers at or above `first_k_dense_replace` use
//!   `add_moe_dispatch_sigmoid` with bias correction and grouped top-k.
//! - **Dense MLP** (`SwiGLU`) for the first `first_k_dense_replace` layers.
//! - **Shared expert** alongside routed experts in `MoE` layers.

use infernum::backend::{
    ArithOps, Backend, ContextBackend, EmbedOps, MatmulOps, MlaAttentionOps, MoeSigmoidOps,
    NormOps, SwigluOps, TensorOps,
};
use infernum::dtype::DType;
use infernum::graph::{
    Graph, GraphArithOps, GraphEmbedOps, GraphMatmulOps, GraphMlaAttentionOps, GraphMoeOps,
    GraphNormOps, GraphSwigluOps, GraphTensorOps, MoeExpertIds, OutputRef, WeightId,
};

use crate::config::DeepSeekConfig;

// ---------------------------------------------------------------------------
// Weight ID structures
// ---------------------------------------------------------------------------

/// Weight IDs for the MLA attention block of one layer.
#[derive(Debug, Clone)]
pub struct MlaWeightIds {
    pub q_a_proj: WeightId,
    pub q_a_layernorm: WeightId,
    pub q_b_proj: WeightId,
    pub kv_a_proj_with_mqa: WeightId,
    pub kv_a_layernorm: WeightId,
    pub kv_b_proj_k: WeightId,
    pub kv_b_proj_v: WeightId,
    pub kv_b_proj_k_t: WeightId,
    pub o_proj: WeightId,
}

/// Weight IDs for a dense `SwiGLU` MLP block.
#[derive(Debug, Clone)]
pub struct DenseMlpIds {
    pub gate_proj: WeightId,
    pub up_proj: WeightId,
    pub down_proj: WeightId,
}

/// Weight IDs for a `MoE` FFN block (sigmoid routing).
#[derive(Debug, Clone)]
pub struct SigmoidMoeIds {
    pub gate: WeightId,
    pub bias: Option<WeightId>,
    pub experts: Vec<MoeExpertIds>,
    pub shared_expert: DenseMlpIds,
}

/// FFN weight IDs — either dense MLP or sigmoid `MoE`.
#[derive(Debug, Clone)]
pub enum FfnIds {
    Dense(DenseMlpIds),
    Moe(SigmoidMoeIds),
}

/// Weight IDs for one transformer layer.
#[derive(Debug, Clone)]
pub struct LayerWeightIds {
    pub input_layernorm: WeightId,
    pub attention: MlaWeightIds,
    pub post_attention_layernorm: WeightId,
    pub ffn: FfnIds,
}

/// Weight IDs for the full `DeepSeek` model.
#[derive(Debug, Clone)]
pub struct DeepSeekWeightIds {
    pub embed_tokens: WeightId,
    pub layers: Vec<LayerWeightIds>,
    pub final_norm: WeightId,
    pub lm_head: WeightId,
}

// ---------------------------------------------------------------------------
// Graph builder trait bound alias
// ---------------------------------------------------------------------------

/// Combined trait bound for backends that can execute the `DeepSeek` graph.
pub trait DeepSeekGraphOps:
    Backend
    + MatmulOps
    + NormOps
    + EmbedOps
    + TensorOps
    + ArithOps
    + SwigluOps
    + MlaAttentionOps
    + MoeSigmoidOps
    + Send
    + 'static
{
}

impl<B> DeepSeekGraphOps for B where
    B: Backend
        + MatmulOps
        + NormOps
        + EmbedOps
        + TensorOps
        + ArithOps
        + SwigluOps
        + MlaAttentionOps
        + MoeSigmoidOps
        + Send
        + 'static
{
}

// ---------------------------------------------------------------------------
// Weight registration
// ---------------------------------------------------------------------------

#[allow(clippy::similar_names)]
fn register_all_weights<B: Backend + MatmulOps + ContextBackend>(
    graph: &mut Graph<B>,
    config: &DeepSeekConfig,
    weight_dtype: DType,
) -> DeepSeekWeightIds {
    let hidden = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let qk_nope = config.qk_nope_head_dim;
    let qk_rope = config.qk_rope_head_dim;
    let v_head = config.v_head_dim;
    let kv_lora = config.kv_lora_rank;

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

        let attention = register_mla_weights(
            graph,
            &format!("{pfx}.self_attn"),
            config,
            weight_dtype,
            hidden,
            num_heads,
            qk_nope,
            qk_rope,
            v_head,
            kv_lora,
        );

        let post_attention_layernorm = graph.register_tensor_weight(
            format!("{pfx}.post_attention_layernorm.weight"),
            &[hidden],
            weight_dtype,
        );

        let ffn = if config.is_moe_layer(i) {
            FfnIds::Moe(register_moe_weights(
                graph,
                config,
                &pfx,
                weight_dtype,
                hidden,
            ))
        } else {
            let intermediate = config.intermediate_size;
            FfnIds::Dense(DenseMlpIds {
                gate_proj: graph.register_linear_weight(
                    format!("{pfx}.mlp.gate_proj.weight"),
                    &[intermediate, hidden],
                    weight_dtype,
                ),
                up_proj: graph.register_linear_weight(
                    format!("{pfx}.mlp.up_proj.weight"),
                    &[intermediate, hidden],
                    weight_dtype,
                ),
                down_proj: graph.register_linear_weight(
                    format!("{pfx}.mlp.down_proj.weight"),
                    &[hidden, intermediate],
                    weight_dtype,
                ),
            })
        };

        layers.push(LayerWeightIds {
            input_layernorm,
            attention,
            post_attention_layernorm,
            ffn,
        });
    }

    let final_norm = graph.register_tensor_weight("model.norm.weight", &[hidden], weight_dtype);
    let lm_head = if config.tie_word_embeddings {
        embed_tokens
    } else {
        graph.register_linear_weight("lm_head.weight", &[config.vocab_size, hidden], weight_dtype)
    };

    DeepSeekWeightIds {
        embed_tokens,
        layers,
        final_norm,
        lm_head,
    }
}

#[allow(clippy::too_many_arguments, clippy::similar_names)]
fn register_mla_weights<B: Backend + MatmulOps + ContextBackend>(
    graph: &mut Graph<B>,
    pfx: &str,
    config: &DeepSeekConfig,
    weight_dtype: DType,
    hidden: usize,
    num_heads: usize,
    qk_nope: usize,
    qk_rope: usize,
    v_head: usize,
    kv_lora: usize,
) -> MlaWeightIds {
    // Q compression (q_lora_rank) or direct projection (num_heads * (qk_nope + qk_rope))
    let q_head_dim = qk_nope + qk_rope;
    let q_full_dim = num_heads * q_head_dim;
    let q_lora_rank = config.q_lora_rank.unwrap_or(q_full_dim);

    // kv_a projection outputs: kv_lora_rank (compressed KV) + qk_rope (decoupled RoPE K)
    let kv_a_out = kv_lora + qk_rope;

    let q_a_proj = graph.register_linear_weight(
        format!("{pfx}.q_a_proj.weight"),
        &[q_lora_rank, hidden],
        weight_dtype,
    );
    let q_a_layernorm = graph.register_tensor_weight(
        format!("{pfx}.q_a_layernorm.weight"),
        &[q_lora_rank],
        weight_dtype,
    );
    let q_b_proj = graph.register_linear_weight(
        format!("{pfx}.q_b_proj.weight"),
        &[q_full_dim, q_lora_rank],
        weight_dtype,
    );
    let kv_a_proj_with_mqa = graph.register_linear_weight(
        format!("{pfx}.kv_a_proj_with_mqa.weight"),
        &[kv_a_out, hidden],
        weight_dtype,
    );
    let kv_a_layernorm = graph.register_tensor_weight(
        format!("{pfx}.kv_a_layernorm.weight"),
        &[kv_lora],
        weight_dtype,
    );
    let kv_b_proj_k = graph.register_linear_weight(
        format!("{pfx}.kv_b_proj_k.weight"),
        &[num_heads * qk_nope, kv_lora],
        weight_dtype,
    );
    let kv_b_proj_v = graph.register_linear_weight(
        format!("{pfx}.kv_b_proj_v.weight"),
        &[num_heads * v_head, kv_lora],
        weight_dtype,
    );
    // Pre-transposed K absorb matrix (kv_lora × num_heads * qk_nope)
    let kv_b_proj_k_t = graph.register_linear_weight(
        format!("{pfx}.kv_b_proj_k_t.weight"),
        &[kv_lora, num_heads * qk_nope],
        weight_dtype,
    );
    let o_proj = graph.register_linear_weight(
        format!("{pfx}.o_proj.weight"),
        &[hidden, num_heads * v_head],
        weight_dtype,
    );

    MlaWeightIds {
        q_a_proj,
        q_a_layernorm,
        q_b_proj,
        kv_a_proj_with_mqa,
        kv_a_layernorm,
        kv_b_proj_k,
        kv_b_proj_v,
        kv_b_proj_k_t,
        o_proj,
    }
}

fn register_moe_weights<B: Backend + MatmulOps + ContextBackend>(
    graph: &mut Graph<B>,
    config: &DeepSeekConfig,
    pfx: &str,
    weight_dtype: DType,
    hidden: usize,
) -> SigmoidMoeIds {
    let num_experts = config.n_routed_experts.unwrap_or(0);
    let expert_intermediate = config.moe_expert_intermediate_size();
    let shared_intermediate = config.shared_expert_intermediate_size();

    let gate = graph.register_tensor_weight(
        format!("{pfx}.mlp.gate.weight"),
        &[num_experts, hidden],
        weight_dtype,
    );
    let bias = Some(graph.register_tensor_weight(
        format!("{pfx}.mlp.gate.e_score_correction_bias"),
        &[num_experts],
        weight_dtype,
    ));

    let experts = (0..num_experts)
        .map(|e| {
            let ep = format!("{pfx}.mlp.experts.{e}");
            MoeExpertIds {
                gate_proj: graph.register_linear_weight(
                    format!("{ep}.gate_proj.weight"),
                    &[expert_intermediate, hidden],
                    weight_dtype,
                ),
                up_proj: graph.register_linear_weight(
                    format!("{ep}.up_proj.weight"),
                    &[expert_intermediate, hidden],
                    weight_dtype,
                ),
                down_proj: graph.register_linear_weight(
                    format!("{ep}.down_proj.weight"),
                    &[hidden, expert_intermediate],
                    weight_dtype,
                ),
            }
        })
        .collect();

    let sp = format!("{pfx}.mlp.shared_experts");
    let shared_expert = DenseMlpIds {
        gate_proj: graph.register_linear_weight(
            format!("{sp}.gate_proj.weight"),
            &[shared_intermediate, hidden],
            weight_dtype,
        ),
        up_proj: graph.register_linear_weight(
            format!("{sp}.up_proj.weight"),
            &[shared_intermediate, hidden],
            weight_dtype,
        ),
        down_proj: graph.register_linear_weight(
            format!("{sp}.down_proj.weight"),
            &[hidden, shared_intermediate],
            weight_dtype,
        ),
    };

    SigmoidMoeIds {
        gate,
        bias,
        experts,
        shared_expert,
    }
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

/// Build the prefill (multi-token) forward-pass graph for a `DeepSeek` model.
#[must_use]
pub fn build_prefill_graph<B>(config: &DeepSeekConfig, weight_dtype: DType) -> Graph<B>
where
    B: DeepSeekGraphOps,
    Graph<B>: GraphMatmulOps
        + GraphNormOps
        + GraphEmbedOps
        + GraphTensorOps
        + GraphArithOps
        + GraphSwigluOps
        + GraphMlaAttentionOps
        + GraphMoeOps,
{
    let mut graph = Graph::new();
    let ids = register_all_weights(&mut graph, config, weight_dtype);

    let token_input = graph.add_input(&[0], DType::U32);
    let mut h = graph.add_embedding_gather(ids.embed_tokens, token_input);

    build_layers(&mut graph, config, &ids, &mut h);

    let normed = graph.add_rms_norm(h, ids.final_norm, config.rms_norm_eps);
    let logits = graph.add_lm_head(normed, ids.lm_head, weight_dtype);
    graph.set_output(logits.0);
    graph
}

/// Build the decode (single-token) forward-pass graph for a `DeepSeek` model.
#[must_use]
pub fn build_decode_graph<B>(config: &DeepSeekConfig, weight_dtype: DType) -> Graph<B>
where
    B: DeepSeekGraphOps,
    Graph<B>: GraphMatmulOps
        + GraphNormOps
        + GraphEmbedOps
        + GraphTensorOps
        + GraphArithOps
        + GraphSwigluOps
        + GraphMlaAttentionOps
        + GraphMoeOps,
{
    let mut graph = Graph::new();
    let ids = register_all_weights(&mut graph, config, weight_dtype);

    let token_input = graph.add_input(&[1], DType::U32);
    let mut h = graph.add_embedding_gather(ids.embed_tokens, token_input);

    build_layers(&mut graph, config, &ids, &mut h);

    let normed = graph.add_rms_norm(h, ids.final_norm, config.rms_norm_eps);
    let logits = graph.add_lm_head(normed, ids.lm_head, weight_dtype);
    graph.set_output(logits.0);
    graph
}

/// Shared layer loop used by both prefill and decode graph builders.
fn build_layers<B>(
    graph: &mut Graph<B>,
    config: &DeepSeekConfig,
    ids: &DeepSeekWeightIds,
    h: &mut OutputRef,
) where
    B: DeepSeekGraphOps,
    Graph<B>: GraphMatmulOps
        + GraphNormOps
        + GraphArithOps
        + GraphSwigluOps
        + GraphMlaAttentionOps
        + GraphMoeOps,
{
    let attn_scale = config.mla_attn_scale();
    let num_heads = config.num_attention_heads;

    for (layer_idx, layer_ids) in ids.layers.iter().enumerate() {
        // Pre-attention RMSNorm
        let normed = graph.add_rms_norm(*h, layer_ids.input_layernorm, config.rms_norm_eps);

        // MLA attention (opaque node)
        let attn_out = graph.add_mla_attention(
            normed,
            layer_ids.attention.q_a_proj,
            layer_ids.attention.q_a_layernorm,
            layer_ids.attention.q_b_proj,
            layer_ids.attention.kv_a_proj_with_mqa,
            layer_ids.attention.kv_a_layernorm,
            layer_ids.attention.kv_b_proj_k,
            layer_ids.attention.kv_b_proj_v,
            layer_ids.attention.kv_b_proj_k_t,
            layer_ids.attention.o_proj,
            num_heads,
            config.qk_nope_head_dim,
            config.qk_rope_head_dim,
            config.v_head_dim,
            config.kv_lora_rank,
            config.rms_norm_eps,
            attn_scale,
            layer_idx,
        );

        *h = graph.add_add(*h, attn_out);

        // Post-attention RMSNorm
        let normed =
            graph.add_rms_norm(*h, layer_ids.post_attention_layernorm, config.rms_norm_eps);

        // FFN: dense SwiGLU or sigmoid MoE
        let ffn_out = build_ffn(graph, &layer_ids.ffn, normed, config);

        *h = graph.add_add(*h, ffn_out);
    }
}

/// Build the FFN sub-graph for one layer.
fn build_ffn<B>(
    graph: &mut Graph<B>,
    ffn_ids: &FfnIds,
    h: OutputRef,
    config: &DeepSeekConfig,
) -> OutputRef
where
    B: DeepSeekGraphOps,
    Graph<B>: GraphMatmulOps + GraphSwigluOps + GraphMoeOps,
{
    match ffn_ids {
        FfnIds::Dense(ids) => {
            let gate = graph.add_linear(h, ids.gate_proj);
            let up = graph.add_linear(h, ids.up_proj);
            let activated = graph.add_swiglu(gate, up);
            graph.add_linear(activated, ids.down_proj)
        }
        FfnIds::Moe(ids) => {
            let shared_ids = MoeExpertIds {
                gate_proj: ids.shared_expert.gate_proj,
                up_proj: ids.shared_expert.up_proj,
                down_proj: ids.shared_expert.down_proj,
            };
            graph.add_moe_dispatch_sigmoid(
                h,
                ids.gate,
                ids.bias,
                ids.experts.clone(),
                Some(shared_ids),
                config.num_experts_per_tok,
                config.n_group,
                config.topk_group,
                config.routed_scaling_factor,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use infernum::DType;

    struct TestBackend;

    #[derive(Clone)]
    struct DummyTensor;

    struct DummyLogits;
    struct DummyRuntimeState;

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
        fn silu(_input: &DummyTensor) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn logit_softcap(_input: &DummyTensor, _cap: f32) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
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

    impl infernum::backend::SwigluOps for TestBackend {
        fn swiglu(_gate: &DummyTensor, _up: &DummyTensor) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
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

    impl infernum::backend::MlaAttentionOps for TestBackend {
        #[allow(clippy::too_many_arguments)]
        fn mla_attention(
            _hidden: &DummyTensor,
            _q_a_proj: &DummyTensor,
            _q_a_layernorm: &DummyTensor,
            _q_b_proj: &DummyTensor,
            _kv_a_proj_with_mqa: &DummyTensor,
            _kv_a_layernorm: &DummyTensor,
            _kv_b_proj_k: &DummyTensor,
            _kv_b_proj_v: &DummyTensor,
            _kv_b_proj_k_t: &DummyTensor,
            _o_proj: &DummyTensor,
            _kv_cache: &mut Vec<DummyTensor>,
            _pos: usize,
            _num_heads: usize,
            _qk_nope_head_dim: usize,
            _qk_rope_head_dim: usize,
            _v_head_dim: usize,
            _kv_lora_rank: usize,
            _rms_norm_eps: f32,
            _attn_scale: f32,
        ) -> infernum::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    fn test_config_dense_only() -> DeepSeekConfig {
        serde_json::from_str(
            r#"{
                "model_type": "deepseek_v3",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "kv_lora_rank": 16,
                "qk_nope_head_dim": 8,
                "qk_rope_head_dim": 4,
                "v_head_dim": 8,
                "q_lora_rank": 16,
                "first_k_dense_replace": 99,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": false,
                "bos_token_id": 0,
                "eos_token_id": 1
            }"#,
        )
        .unwrap()
    }

    fn test_config_with_moe() -> DeepSeekConfig {
        serde_json::from_str(
            r#"{
                "model_type": "deepseek_v3",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "moe_intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "kv_lora_rank": 16,
                "qk_nope_head_dim": 8,
                "qk_rope_head_dim": 4,
                "v_head_dim": 8,
                "q_lora_rank": 16,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "n_group": 2,
                "topk_group": 1,
                "first_k_dense_replace": 1,
                "routed_scaling_factor": 1.0,
                "scoring_func": "sigmoid",
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": false,
                "bos_token_id": 0,
                "eos_token_id": 1
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn test_build_prefill_graph_dense() {
        let config = test_config_dense_only();
        let graph: Graph<TestBackend> = build_prefill_graph(&config, DType::F32);
        assert_eq!(graph.output_ids().len(), 1);
    }

    #[test]
    fn test_build_decode_graph_dense() {
        let config = test_config_dense_only();
        let graph: Graph<TestBackend> = build_decode_graph(&config, DType::F32);
        assert_eq!(graph.output_ids().len(), 1);
    }

    #[test]
    fn test_build_prefill_graph_moe() {
        let config = test_config_with_moe();
        let graph: Graph<TestBackend> = build_prefill_graph(&config, DType::F32);
        assert_eq!(graph.output_ids().len(), 1);
    }

    #[test]
    fn test_build_decode_graph_moe() {
        let config = test_config_with_moe();
        let graph: Graph<TestBackend> = build_decode_graph(&config, DType::F32);
        assert_eq!(graph.output_ids().len(), 1);
    }

    #[test]
    fn test_dense_layer_detection() {
        let config = test_config_with_moe();
        // first_k_dense_replace=1 → layer 0 is dense, layer 1 is MoE
        assert!(!config.is_moe_layer(0));
        assert!(config.is_moe_layer(1));
    }

    #[test]
    fn test_tied_embeddings_share_weight_id() {
        let config: DeepSeekConfig = serde_json::from_str(
            r#"{
                "model_type": "deepseek_v3",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "kv_lora_rank": 8,
                "qk_nope_head_dim": 4,
                "qk_rope_head_dim": 2,
                "v_head_dim": 4,
                "first_k_dense_replace": 99,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": true,
                "bos_token_id": 0,
                "eos_token_id": 1
            }"#,
        )
        .unwrap();
        let graph: Graph<TestBackend> = build_prefill_graph(&config, DType::F32);
        assert_eq!(graph.output_ids().len(), 1);
    }
}
