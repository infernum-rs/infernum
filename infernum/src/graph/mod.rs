//! Computation graph representation for LLM inference.
//!
//! The graph is built by model code and consumed by backend executors.
//! It captures the full forward pass as a DAG of typed operations,
//! enabling fusion, scheduling, and memory planning.

mod arena;
mod builder;
mod builder_traits;
pub mod builtin_ops;
pub mod execute_context;
mod node;
mod op_node;
pub mod optimizer;
mod planner;
mod weight_store;

pub use arena::Arena;
pub use builder::Graph;
pub use builder_traits::{
    GraphArithOps, GraphAttentionOps, GraphBiasOps, GraphCastOps, GraphEmbedOps, GraphGegluOps,
    GraphIndirectDecodeOps, GraphMatmulExtOps, GraphMatmulOps, GraphMlaAttentionOps, GraphMoeOps,
    GraphNormOps, GraphPagedAttentionOps, GraphPagedKvCacheOps, GraphRopeInterleavedOps,
    GraphRopeOps, GraphSiluOps, GraphSoftcapOps, GraphSwigluOps, GraphTensorOps,
};
pub use builtin_ops::{MlaAttentionOp, MoeExpertIds};
pub use execute_context::{ExecuteContext, KvCacheAccess};
pub use node::{GraphNode, NodeId, WeightId, WeightMeta, WeightRef};
pub use op_node::{OpNode, OutputRef};
pub use planner::{plan, BufferSlot, ExecutionPlan, PlanCache};
pub use weight_store::WeightStore;

#[cfg(any(test, feature = "test-helpers"))]
pub mod test_helpers;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::dtype::DType;

    /// Minimal backend for testing. Not a real backend — just satisfies
    /// the trait bounds so we can instantiate `Graph<TestBackend>`.
    struct TestBackend;

    #[derive(Clone)]
    struct DummyTensor;

    impl crate::tensor::Tensor for DummyTensor {
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

    impl crate::logits::Logits for DummyLogits {
        fn vocab_size(&self) -> usize {
            0
        }
        fn batch_size(&self) -> usize {
            0
        }
        fn argmax(&self, _batch_index: usize) -> crate::Result<u32> {
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
        ) -> crate::Result<u32> {
            Ok(0)
        }
    }

    struct DummyRuntimeState;

    impl crate::runtime_state::RuntimeStateInit for DummyRuntimeState {
        fn new(
            _batch_config: &crate::runtime_state::BatchConfig,
            _block_config: &crate::block_allocator::BlockConfig,
        ) -> crate::Result<Self> {
            Ok(Self)
        }

        fn new_placeholder() -> Self {
            Self
        }
    }

    impl Backend for TestBackend {
        type Tensor = DummyTensor;
        type DeviceHandle = ();
        type PagedKvCache = ();
        type KvCache = ();
        type RuntimeState = DummyRuntimeState;
        type ExecutorState = ();
        type Logits = DummyLogits;
        type Comm = ();

        fn logits_from_tensor(_tensor: Self::Tensor) -> Self::Logits {
            DummyLogits
        }
    }

    #[test]
    fn empty_graph() {
        let graph = Graph::<TestBackend>::new();
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
    }

    #[test]
    fn input_reshape_output() {
        let mut graph = Graph::<TestBackend>::new();

        // Input: (4, 128)
        let input = graph.add_input(&[4, 128], DType::BF16);
        assert_eq!(graph.node_shape(input), &[4, 128]);
        assert_eq!(graph.node_dtype(input), DType::BF16);

        // Reshape to (2, 256)
        let reshaped = graph.add_reshape(input, &[2, 256]);

        graph.set_output(reshaped.0);

        assert_eq!(graph.len(), 2);
        assert!(!graph.is_empty());
        assert_eq!(graph.node_shape(reshaped), &[2, 256]);

        // Verify the output was recorded.
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.outputs[0], reshaped.0);
    }

    #[test]
    fn register_weights_and_retrieve_metadata() {
        let mut graph = Graph::<TestBackend>::new();

        let tw = graph.register_tensor_weight("ln.weight", &[4096], DType::BF16);
        let lw = graph.register_linear_weight("attn.q_proj", &[4096, 4096], DType::BF16);

        let tmeta = graph.tensor_weight_meta(tw);
        assert_eq!(tmeta.name, "ln.weight");
        assert_eq!(tmeta.shape, vec![4096]);
        assert_eq!(tmeta.dtype, DType::BF16);

        let lmeta = graph.linear_weight_meta(lw);
        assert_eq!(lmeta.name, "attn.q_proj");
        assert_eq!(lmeta.shape, vec![4096, 4096]);
        assert_eq!(lmeta.dtype, DType::BF16);
    }

    #[test]
    fn mini_transformer_connectivity() {
        let mut graph = Graph::<TestBackend>::new();

        // Weights
        let norm_w = graph.register_tensor_weight("ln.weight", &[512], DType::BF16);
        let proj_w = graph.register_linear_weight("proj.weight", &[512, 512], DType::BF16);

        // Input: (8, 512)
        let input = graph.add_input(&[8, 512], DType::BF16);

        // RmsNorm
        let normed = graph.add_rms_norm(input, norm_w, 1e-5);

        // Linear
        let projected = graph.add_linear(normed, proj_w);

        // Residual add
        let added = graph.add_add(input, projected);

        graph.set_output(added.0);

        assert_eq!(graph.len(), 4); // input, norm, linear, add
        assert_eq!(graph.node_shape(added), &[8, 512]);

        // Verify connectivity: the add node should have 2 inputs.
        let add_node = &graph.nodes[added.0 .0 as usize];
        assert_eq!(add_node.inputs.len(), 2);
        assert_eq!(add_node.inputs[0], input);
        assert_eq!(add_node.inputs[1], projected);
    }

    #[test]
    fn multi_output_add_rmsnorm() {
        let mut graph = Graph::<TestBackend>::new();

        let norm_w = graph.register_tensor_weight("ln.weight", &[256], DType::F32);

        let residual = graph.add_input(&[4, 256], DType::F32);
        let delta = graph.add_input(&[4, 256], DType::F32);

        let (updated_residual, normalized) = graph.add_add_rmsnorm(residual, delta, norm_w, 1e-6);

        // Single multi-output node: 2 inputs + 1 add_rmsnorm = 3 nodes
        assert_eq!(graph.len(), 3);
        assert_eq!(graph.node_shape(updated_residual), &[4, 256]);
        assert_eq!(graph.node_shape(normalized), &[4, 256]);
        assert_eq!(graph.node_dtype(updated_residual), DType::F32);
        assert_eq!(graph.node_dtype(normalized), DType::F32);

        // Both outputs should reference the same node.
        assert_eq!(updated_residual.0, normalized.0);
        assert_eq!(updated_residual.1, 0);
        assert_eq!(normalized.1, 1);

        // The node should be named "add_rms_norm".
        let node = &graph.nodes[updated_residual.0 .0 as usize];
        assert_eq!(node.op.name(), "add_rms_norm");
    }

    #[test]
    fn set_output_records_multiple() {
        let mut graph = Graph::<TestBackend>::new();

        let a = graph.add_input(&[1, 128], DType::F32);
        let b = graph.add_input(&[1, 128], DType::F32);

        graph.set_output(a.0);
        graph.set_output(b.0);

        assert_eq!(graph.outputs.len(), 2);
        assert_eq!(graph.outputs[0], a.0);
        assert_eq!(graph.outputs[1], b.0);
    }

    #[test]
    fn weight_ref_variants() {
        let tid = WeightId(0);
        let lid = WeightId(1);

        let tref = WeightRef::Tensor(tid);
        let lref = WeightRef::Linear(lid);

        // Verify equality and debug formatting work.
        assert_eq!(tref, WeightRef::Tensor(WeightId(0)));
        assert_ne!(tref, lref);
        assert_eq!(format!("{tref:?}"), "Tensor(WeightId(0))");
        assert_eq!(format!("{lref:?}"), "Linear(WeightId(1))");
    }

    // -----------------------------------------------------------------------
    // Op trait stubs on TestBackend (for builder_traits tests)
    // -----------------------------------------------------------------------

    impl crate::backend::EmbedOps for TestBackend {
        fn embedding_gather(_table: &DummyTensor, _indices: &[u32]) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn embedding_gather_tensor(
            _table: &DummyTensor,
            _indices: &DummyTensor,
            _seq_len: usize,
        ) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl crate::backend::NormOps for TestBackend {
        fn rms_norm(
            _input: &DummyTensor,
            _weight: &DummyTensor,
            _eps: f32,
        ) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn rms_norm_inplace(
            _input: &mut DummyTensor,
            _weight: &DummyTensor,
            _eps: f32,
        ) -> crate::Result<()> {
            Ok(())
        }
        fn add_rmsnorm(
            _residual: &DummyTensor,
            _input: &DummyTensor,
            _weight: &DummyTensor,
            _eps: f32,
        ) -> crate::Result<(DummyTensor, DummyTensor)> {
            Ok((DummyTensor, DummyTensor))
        }
    }

    impl crate::backend::MatmulOps for TestBackend {
        type LinearWeight = DummyTensor;

        fn matmul(_a: &DummyTensor, _b: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn linear(_input: &DummyTensor, _weight: &DummyTensor) -> crate::Result<DummyTensor> {
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
        ) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn upload_host_linear(
            _device: &(),
            _weight: &crate::weights::host::HostLinearWeight,
        ) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl crate::backend::ArithOps for TestBackend {
        fn add(_a: &DummyTensor, _b: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn add_inplace(_a: &mut DummyTensor, _b: &DummyTensor) -> crate::Result<()> {
            Ok(())
        }
        fn mul(_a: &DummyTensor, _b: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn scale_inplace(_a: &mut DummyTensor, _scale: f32) -> crate::Result<()> {
            Ok(())
        }
    }

    impl crate::backend::AttentionOps for TestBackend {
        fn fused_attention_prefill(
            _q: &DummyTensor,
            _k: &DummyTensor,
            _v: &DummyTensor,
            _offset: usize,
            _scale: Option<f32>,
            _softcap: Option<f32>,
            _sliding_window: Option<usize>,
        ) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn fused_attention_decode(
            _q: &DummyTensor,
            _k: &DummyTensor,
            _v: &DummyTensor,
            _scale: Option<f32>,
            _softcap: Option<f32>,
            _sliding_window: Option<usize>,
        ) -> crate::Result<DummyTensor> {
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
        ) -> crate::Result<(DummyTensor, DummyTensor)> {
            Ok((DummyTensor, DummyTensor))
        }
        fn combine_attention_with_lse(
            _out1: &DummyTensor,
            _lse1: &DummyTensor,
            _out2: &DummyTensor,
            _lse2: &DummyTensor,
        ) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl crate::backend::TensorOps for TestBackend {
        fn transpose_2d(_input: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn split_inner_dim(
            _tensor: &DummyTensor,
            _dim1: usize,
            _dim2: usize,
        ) -> crate::Result<(DummyTensor, DummyTensor)> {
            Ok((DummyTensor, DummyTensor))
        }
        fn concat_inner_dim(_a: &DummyTensor, _b: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn pad_inner_dim(_tensor: &DummyTensor, _new_width: usize) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn broadcast_to_heads(
            _tensor: &DummyTensor,
            _num_heads: usize,
        ) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn repeat_kv(_tensor: &DummyTensor, _num_repeats: usize) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn concat_rows(_parts: &[DummyTensor]) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl crate::backend::RopeOps for TestBackend {
        fn apply_rope(
            _input: &DummyTensor,
            _cos_cache: &DummyTensor,
            _sin_cache: &DummyTensor,
            _position_offset: usize,
        ) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn apply_rope_batched(
            _input: &DummyTensor,
            _cos_cache: &DummyTensor,
            _sin_cache: &DummyTensor,
            _positions: &DummyTensor,
            _batch_size: usize,
        ) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl crate::backend::CastOps for TestBackend {
        fn cast_to_f32(_input: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn cast_from_f32(_input: &DummyTensor, _target: DType) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl crate::backend::SwigluOps for TestBackend {
        fn swiglu(_gate: &DummyTensor, _up: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl crate::backend::ContextBackend for TestBackend {
        fn ctx_read(
            _ctx: &crate::graph::execute_context::ExecuteContext<'_, Self>,
            _output_ref: crate::graph::OutputRef,
        ) -> DummyTensor {
            DummyTensor
        }
        fn ctx_write(
            _ctx: &mut crate::graph::execute_context::ExecuteContext<'_, Self>,
            _node_id: crate::graph::NodeId,
            _idx: u32,
            _tensor: DummyTensor,
        ) {
        }
        fn ctx_next_input(
            _ctx: &mut crate::graph::execute_context::ExecuteContext<'_, Self>,
        ) -> DummyTensor {
            DummyTensor
        }
    }

    // -----------------------------------------------------------------------
    // Builder traits tests
    // -----------------------------------------------------------------------

    #[test]
    fn graph_linear_shape_inference() {
        let mut graph = Graph::<TestBackend>::new();

        // Weight: (2048, 4096) → projects from 4096 to 2048.
        let w = graph.register_linear_weight("proj", &[2048, 4096], DType::BF16);
        let input = graph.add_input(&[8, 4096], DType::BF16);

        let output = graph.add_linear(input, w);

        assert_eq!(graph.node_shape(output), &[8, 2048]);
        assert_eq!(graph.node_dtype(output), DType::BF16);
    }

    #[test]
    fn graph_split_inner_dim_shapes() {
        let mut graph = Graph::<TestBackend>::new();

        let input = graph.add_input(&[8, 768], DType::BF16);
        let (left, right) = graph.add_split_inner_dim(input, 512);

        assert_eq!(graph.node_shape(left), &[8, 512]);
        assert_eq!(graph.node_shape(right), &[8, 256]);
        assert_eq!(graph.node_dtype(left), DType::BF16);
        assert_eq!(graph.node_dtype(right), DType::BF16);
    }

    /// Build a mini attention block using builder traits:
    /// input → rms_norm → linear(Q) → reshape → rope →
    ///                     linear(K) → reshape → rope →
    ///                     linear(V) → reshape →
    ///         attention_prefill → reshape → linear(O) → output
    #[test]
    fn graph_mini_attention_block() {
        let mut graph = Graph::<TestBackend>::new();

        let hidden = 512;
        let num_heads = 8;
        let head_dim = hidden / num_heads; // 64
        let seq_len = 16;

        // Weights
        let norm_w = graph.register_tensor_weight("attn_norm.weight", &[hidden], DType::BF16);
        let q_proj = graph.register_linear_weight("q_proj", &[hidden, hidden], DType::BF16);
        let k_proj = graph.register_linear_weight("k_proj", &[hidden, hidden], DType::BF16);
        let v_proj = graph.register_linear_weight("v_proj", &[hidden, hidden], DType::BF16);
        let o_proj = graph.register_linear_weight("o_proj", &[hidden, hidden], DType::BF16);

        // RoPE caches (shape doesn't matter for graph shape inference,
        // but must be present as input nodes).
        let cos = graph.add_input(&[seq_len, head_dim], DType::BF16);
        let sin = graph.add_input(&[seq_len, head_dim], DType::BF16);

        // Input hidden states: (seq_len, hidden)
        let input = graph.add_input(&[seq_len, hidden], DType::BF16);

        // 1. RmsNorm
        let normed = graph.add_rms_norm(input, norm_w, 1e-5);
        assert_eq!(graph.node_shape(normed), &[seq_len, hidden]);

        // 2. Q/K/V projections
        let q = graph.add_linear(normed, q_proj);
        let k = graph.add_linear(normed, k_proj);
        let v = graph.add_linear(normed, v_proj);
        assert_eq!(graph.node_shape(q), &[seq_len, hidden]);

        // 3. Reshape to (seq_len, num_heads, head_dim)
        let q_3d = graph.add_reshape(q, &[seq_len, num_heads, head_dim]);
        let k_3d = graph.add_reshape(k, &[seq_len, num_heads, head_dim]);
        let v_3d = graph.add_reshape(v, &[seq_len, num_heads, head_dim]);
        assert_eq!(graph.node_shape(q_3d), &[seq_len, num_heads, head_dim]);

        // 4. RoPE
        let q_rope = graph.add_rope(q_3d, cos, sin, 0);
        let k_rope = graph.add_rope(k_3d, cos, sin, 0);
        assert_eq!(graph.node_shape(q_rope), &[seq_len, num_heads, head_dim]);

        // 5. Attention prefill
        let attn_out = graph.add_fused_attention_prefill(q_rope, k_rope, v_3d, 0, None, None, None);
        assert_eq!(graph.node_shape(attn_out), &[seq_len, num_heads, head_dim]);

        // 6. Reshape back to (seq_len, hidden)
        let flat = graph.add_reshape(attn_out, &[seq_len, hidden]);
        assert_eq!(graph.node_shape(flat), &[seq_len, hidden]);

        // 7. Output projection
        let output = graph.add_linear(flat, o_proj);
        assert_eq!(graph.node_shape(output), &[seq_len, hidden]);

        graph.set_output(output.0);

        // Total nodes: 3 inputs (cos, sin, hidden) + norm + 3 linears +
        //   3 reshapes + 2 ropes + attention + reshape + linear = 15
        assert_eq!(graph.len(), 15);
    }

    #[test]
    fn graph_embedding_gather_shape() {
        let mut graph = Graph::<TestBackend>::new();

        // Embedding table: (32000, 4096)
        let table = graph.register_tensor_weight("embed", &[32000, 4096], DType::BF16);
        let token_ids = graph.add_input(&[16], DType::F32);

        let hidden = graph.add_embedding_gather(table, token_ids);
        assert_eq!(graph.node_shape(hidden), &[16, 4096]);
        assert_eq!(graph.node_dtype(hidden), DType::BF16);
    }

    #[test]
    fn graph_cast_shape_and_dtype() {
        let mut graph = Graph::<TestBackend>::new();

        let input = graph.add_input(&[4, 128], DType::BF16);
        let f32_node = graph.add_cast_to_f32(input);
        assert_eq!(graph.node_shape(f32_node), &[4, 128]);
        assert_eq!(graph.node_dtype(f32_node), DType::F32);

        let bf16_node = graph.add_cast_from_f32(f32_node, DType::BF16);
        assert_eq!(graph.node_shape(bf16_node), &[4, 128]);
        assert_eq!(graph.node_dtype(bf16_node), DType::BF16);
    }

    #[test]
    fn graph_lm_head_shape() {
        let mut graph = Graph::<TestBackend>::new();

        let lm_w = graph.register_linear_weight("lm_head", &[32000, 4096], DType::BF16);
        let input = graph.add_input(&[8, 4096], DType::BF16);

        let logits = graph.add_lm_head(input, lm_w, DType::BF16);
        assert_eq!(graph.node_shape(logits), &[8, 32000]);
        assert_eq!(graph.node_dtype(logits), DType::F32);
    }

    #[test]
    fn graph_linear_triple_shape() {
        let mut graph = Graph::<TestBackend>::new();

        let w1 = graph.register_linear_weight("q", &[512, 4096], DType::BF16);
        let w2 = graph.register_linear_weight("k", &[128, 4096], DType::BF16);
        let w3 = graph.register_linear_weight("v", &[128, 4096], DType::BF16);

        let input = graph.add_input(&[8, 4096], DType::BF16);
        let (q, k, v) = graph.add_linear_triple(input, w1, w2, w3);

        assert_eq!(graph.node_shape(q), &[8, 512]);
        assert_eq!(graph.node_shape(k), &[8, 128]);
        assert_eq!(graph.node_shape(v), &[8, 128]);
    }

    #[test]
    fn graph_add_rmsnorm_via_trait() {
        let mut graph = Graph::<TestBackend>::new();

        let norm_w = graph.register_tensor_weight("ln.weight", &[256], DType::BF16);
        let residual = graph.add_input(&[4, 256], DType::BF16);
        let delta = graph.add_input(&[4, 256], DType::BF16);

        let (updated, normed) = graph.add_add_rmsnorm(residual, delta, norm_w, 1e-5);

        assert_eq!(graph.node_shape(updated), &[4, 256]);
        assert_eq!(graph.node_shape(normed), &[4, 256]);
        assert_eq!(graph.node_dtype(updated), DType::BF16);
        assert_eq!(graph.node_dtype(normed), DType::BF16);
    }

    #[test]
    fn graph_swiglu_shape() {
        let mut graph = Graph::<TestBackend>::new();

        let gate = graph.add_input(&[8, 2048], DType::BF16);
        let up = graph.add_input(&[8, 2048], DType::BF16);

        let out = graph.add_swiglu(gate, up);
        assert_eq!(graph.node_shape(out), &[8, 2048]);
        assert_eq!(graph.node_dtype(out), DType::BF16);
    }

    #[test]
    fn graph_repeat_kv_shape() {
        let mut graph = Graph::<TestBackend>::new();

        // (seq=16, kv_heads=4, head_dim=64) with repeats=8
        let input = graph.add_input(&[16, 4, 64], DType::BF16);
        let out = graph.add_repeat_kv(input, 8);
        assert_eq!(graph.node_shape(out), &[16, 32, 64]);
    }

    #[test]
    fn graph_extract_last_row_shape() {
        let mut graph = Graph::<TestBackend>::new();

        let input = graph.add_input(&[16, 512], DType::BF16);
        let out = graph.add_extract_last_row(input, 16);
        assert_eq!(graph.node_shape(out), &[1, 512]);
    }

    #[test]
    fn graph_transpose_2d_shape() {
        let mut graph = Graph::<TestBackend>::new();

        let input = graph.add_input(&[3, 7], DType::F32);
        let out = graph.add_transpose_2d(input);
        assert_eq!(graph.node_shape(out), &[7, 3]);
    }

    #[test]
    fn graph_concat_inner_dim_shape() {
        let mut graph = Graph::<TestBackend>::new();

        let a = graph.add_input(&[8, 256], DType::BF16);
        let b = graph.add_input(&[8, 128], DType::BF16);
        let out = graph.add_concat_inner_dim(a, b);
        assert_eq!(graph.node_shape(out), &[8, 384]);
    }

    #[test]
    fn graph_concat_seq_shape() {
        let mut graph = Graph::<TestBackend>::new();

        let a = graph.add_input(&[10, 4, 64], DType::F32);
        let b = graph.add_input(&[1, 4, 64], DType::F32);
        let out = graph.add_concat_seq(a, b);
        assert_eq!(graph.node_shape(out), &[11, 4, 64]);
    }

    // --- Custom op extensibility test ---

    /// Example custom op: element-wise ReLU.
    ///
    /// Demonstrates that downstream crates can define their own `OpNode`
    /// implementations and register them in a `Graph` via `add_node`.
    #[derive(Debug)]
    struct ReluOp;

    impl OpNode<TestBackend> for ReluOp {
        fn name(&self) -> &'static str {
            "relu"
        }

        fn num_inputs(&self) -> usize {
            1
        }

        fn num_outputs(&self) -> usize {
            1
        }

        fn output_shapes(&self, inputs: &[&[usize]]) -> Vec<Vec<usize>> {
            // Identity shape — ReLU is element-wise.
            vec![inputs[0].to_vec()]
        }

        fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
            vec![input_dtypes[0]]
        }

        fn execute(
            &self,
            _ctx: &mut ExecuteContext<'_, TestBackend>,
            _node_id: NodeId,
            _inputs: &[OutputRef],
        ) -> crate::Result<()> {
            Ok(()) // no-op test implementation
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn custom_op_can_be_registered() {
        let mut graph = Graph::<TestBackend>::new();

        let input = graph.add_input(&[4, 256], DType::BF16);
        let relu_id = graph.add_node(Box::new(ReluOp), &[input]);
        let relu_out: OutputRef = (relu_id, 0);

        assert_eq!(graph.node_shape(relu_out), &[4, 256]);
        assert_eq!(graph.node_dtype(relu_out), DType::BF16);

        let node = &graph.nodes()[relu_id.index() as usize];
        assert_eq!(node.op.name(), "relu");
    }
}
