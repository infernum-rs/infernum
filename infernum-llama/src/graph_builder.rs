//! Graph builder for the Llama model family.
//!
//! Constructs a computation graph ([`Graph<B>`]) representing the full
//! Llama forward pass. The graph captures the complete DAG of operations
//! from token IDs to logits, enabling ahead-of-time memory planning
//! and execution scheduling.
//!
//! Currently supports:
//! - Dense models only (no `MoE`)
//! - Prefill pass only (no decode graph yet)
//! - Single-GPU only (no tensor parallelism / `AllReduce`)
//! - Full causal attention (no sliding window)

use infernum::backend::{
    ArithOps, AttentionOps, Backend, EmbedOps, MatmulOps, NormOps, RopeOps, SwigluOps, TensorOps,
};
use infernum::dtype::DType;
use infernum::graph::{
    Graph, GraphArithOps, GraphAttentionOps, GraphEmbedOps, GraphMatmulOps, GraphNormOps,
    GraphRopeOps, GraphSwigluOps, GraphTensorOps, WeightId,
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
    Backend + MatmulOps + NormOps + ArithOps + SwigluOps + EmbedOps + TensorOps + RopeOps + AttentionOps
{
}

impl<B> LlamaGraphOps for B where
    B: Backend
        + MatmulOps
        + NormOps
        + ArithOps
        + SwigluOps
        + EmbedOps
        + TensorOps
        + RopeOps
        + AttentionOps
{
}

// ---------------------------------------------------------------------------
// Weight registration
// ---------------------------------------------------------------------------

/// Register all model weights in the graph and return their IDs.
///
/// Weight names match the `SafeTensors` naming convention used by
/// [`LlamaModel::load_weights`](crate::LlamaModel).
fn register_weights<B: Backend>(
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

        // 7. Fused residual add + post-attention RMS norm
        let (h_updated, normed_post) =
            graph.add_add_rmsnorm(h, attn_proj, lw.post_attention_layernorm, eps);

        // 8. FFN: SwiGLU MLP (fused pair — single dispatch, shared quantization)
        let (gate, up) = graph.add_linear_pair(normed_post, lw.gate_proj, lw.up_proj);
        let activated = graph.add_swiglu(gate, up);
        let down = graph.add_linear(activated, lw.down_proj);

        // 9. Residual add
        h = graph.add_add_inplace(h_updated, down);
    }

    // -- Final norm --
    let normed_final = graph.add_rms_norm(h, model_weights.final_norm, eps);

    // -- LM head --
    let logits = graph.add_lm_head(normed_final, model_weights.lm_head, weight_dtype);
    graph.set_output(logits);

    (graph, model_weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    use infernum::graph::Op;

    /// Minimal backend for graph construction tests. Matches the pattern
    /// used in `infernum/src/graph/mod.rs` tests.
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

    impl infernum::backend::SwigluOps for TestBackend {
        fn swiglu(_gate: &DummyTensor, _up: &DummyTensor) -> infernum::Result<DummyTensor> {
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
        assert_eq!(graph.node_shape(logits_id), &[seq_len, config.vocab_size]);
        assert_eq!(graph.node_dtype(logits_id), DType::F32);
    }

    #[test]
    fn prefill_graph_has_correct_input_count() {
        let config = test_config();
        let (graph, _weights) = build_prefill_graph::<TestBackend>(&config, 8, DType::BF16);

        // Should have exactly 3 Input nodes: input_ids, cos_cache, sin_cache
        let input_count = graph
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Input))
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
        assert_eq!(graph.node_shape(logits_id), &[seq_len, 32000]);
    }
}
