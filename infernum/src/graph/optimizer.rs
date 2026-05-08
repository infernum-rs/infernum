//! Graph optimization pass.
//!
//! Rewrites primitive ops into fused ops where profitable. This pass
//! runs between graph construction and planning — the graph builder
//! emits only primitive ops, and the optimizer fuses them.
//!
//! Current fusion rules:
//! - `Silu(a) → Mul(silu_out, b)` → `Swiglu(a, b)` (when Silu has single consumer)
//! - `Add(a, b) → RmsNorm(sum)` → `AddRmsNorm(a, b)` (when Add has single consumer)

use crate::backend::{ArithOps, Backend, ContextBackend, MatmulOps, NormOps, RopeOps, SwigluOps};

use super::builder::Graph;
use super::builtin_ops::{AddRmsNormOp, FusedRopePairOp, RmsNormOp, RopeOp, SwigluOp};
use super::node::NodeId;

/// Apply all fusion rules to the graph in-place.
pub fn optimize<
    B: Backend + MatmulOps + ArithOps + NormOps + SwigluOps + RopeOps + ContextBackend,
>(
    graph: &mut Graph<B>,
) {
    fuse_swiglu(graph);
    fuse_add_rmsnorm(graph);
    fuse_rope_pairs(graph);
}

/// Count how many nodes use any output of `target` as an input.
fn consumer_count<B: Backend + MatmulOps + ContextBackend>(
    graph: &Graph<B>,
    target: NodeId,
) -> usize {
    graph
        .nodes
        .iter()
        .filter(|n| n.inputs.iter().any(|&(nid, _)| nid == target))
        .count()
}

/// Fuse `Silu(a) → Mul(silu_out, b)` into `Swiglu(a, b)`.
///
/// Preconditions for fusion:
/// - The `Mul` node has exactly two inputs, one of which is a `Silu` output.
/// - The `Silu` node has exactly one consumer (this `Mul`).
fn fuse_swiglu<B: Backend + MatmulOps + ArithOps + SwigluOps + ContextBackend>(
    graph: &mut Graph<B>,
) {
    let n = graph.nodes.len();
    for i in 0..n {
        if graph.nodes[i].op.name() != "mul" {
            continue;
        }

        let inputs = graph.nodes[i].inputs.clone();
        if inputs.len() != 2 {
            continue;
        }

        // Check if either input comes from a Silu node.
        let (silu_idx, other_idx) = if graph.nodes[inputs[0].0 .0 as usize].op.name() == "silu" {
            (0, 1)
        } else if graph.nodes[inputs[1].0 .0 as usize].op.name() == "silu" {
            (1, 0)
        } else {
            continue;
        };

        let silu_ref = inputs[silu_idx];
        let up_ref = inputs[other_idx];
        let silu_id = silu_ref.0;

        // Only fuse if the Silu output is used exclusively by this Mul.
        if consumer_count(graph, silu_id) != 1 {
            continue;
        }

        // Get the Silu's input (the gate).
        let gate_ref = graph.nodes[silu_id.0 as usize].inputs[0];

        // Replace Mul → Swiglu(gate, up).
        graph.nodes[i].op = Box::new(SwigluOp);
        graph.nodes[i].inputs.clear();
        graph.nodes[i].inputs.push(gate_ref);
        graph.nodes[i].inputs.push(up_ref);

        // The orphaned Silu node will be eliminated by dead-node removal
        // since no node consumes it anymore.
    }
}

/// Fuse `Add(a, b) → RmsNorm(sum, weight, eps)` into `AddRmsNorm`.
///
/// After fusion:
/// - The `Add` node becomes `AddRmsNorm`, producing two outputs:
///   output 0 = updated residual, output 1 = normalised.
/// - All downstream consumers of the old `RmsNorm` output `(NodeId(i), 0)`
///   are rewired to `(add_id, 1)`.
/// - The `RmsNorm` node becomes dead (no consumers) and will be eliminated
///   by the planner's dead-node pass.
///
/// Preconditions:
/// - The `RmsNorm`'s sole input is an `Add`/`AddInplace` output.
/// - The `Add` output has exactly one node-consumer (the `RmsNorm`).
#[allow(clippy::cast_possible_truncation)] // graph will never have 2^32 nodes
fn fuse_add_rmsnorm<B: Backend + MatmulOps + ArithOps + NormOps + ContextBackend>(
    graph: &mut Graph<B>,
) {
    let n = graph.nodes.len();
    for i in 0..n {
        if graph.nodes[i].op.name() != "rms_norm" {
            continue;
        }

        // Extract RmsNorm parameters via downcast.
        let (weight, eps) = {
            let Some(rms) = graph.nodes[i].op.as_any().downcast_ref::<RmsNormOp>() else {
                continue;
            };
            (rms.weight, rms.eps)
        };

        if graph.nodes[i].inputs.len() != 1 {
            continue;
        }

        let add_ref = graph.nodes[i].inputs[0];
        let add_id = add_ref.0;
        let add_idx = add_id.0 as usize;

        let add_name = graph.nodes[add_idx].op.name();
        if add_name != "add" && add_name != "add_inplace" {
            continue;
        }

        // Only fuse if the Add output feeds exactly one consumer.
        if consumer_count(graph, add_id) != 1 {
            continue;
        }

        let rms_node_id = NodeId(i as u32);

        // Replace Add → AddRmsNorm (2 outputs).
        graph.nodes[add_idx].op = Box::new(AddRmsNormOp { weight, eps });

        // Update add node's output_shapes/dtypes to have 2 entries.
        let shape = graph.nodes[add_idx].output_shapes[0].clone();
        let dtype = graph.nodes[add_idx].output_dtypes[0];
        graph.nodes[add_idx].output_shapes.push(shape);
        graph.nodes[add_idx].output_dtypes.push(dtype);

        // Rewire all downstream consumers of the old RmsNorm output
        // (rms_node_id, 0) → (add_id, 1).
        let old_ref = (rms_node_id, 0u32);
        let new_ref = (add_id, 1u32);
        for j in 0..graph.nodes.len() {
            for k in 0..graph.nodes[j].inputs.len() {
                if graph.nodes[j].inputs[k] == old_ref {
                    graph.nodes[j].inputs[k] = new_ref;
                }
            }
        }

        // Also rewire graph outputs.
        for out in &mut graph.outputs {
            if *out == rms_node_id {
                *out = add_id;
            }
        }

        // Clear the RmsNorm node's inputs so it becomes truly dead
        // (no node consumes it, and it consumes nothing).
        graph.nodes[i].inputs.clear();
    }
}

/// Fuse pairs of `RopeOp` nodes sharing the same cos/sin inputs.
///
/// When two `RopeOp` nodes in the graph both read from the same `cos` and
/// `sin` cache, they can be dispatched together as a single
/// [`FusedRopePairOp`].  This is the typical Q/K pattern in every transformer
/// layer.
///
/// After fusion:
/// - Node A becomes `FusedRopePairOp` with inputs
///   `[A_input, B_input, cos, sin]` and two outputs.
/// - All downstream consumers of node B's output `(B_id, 0)` are rewired to
///   `(A_id, 1)`.
/// - Node B's inputs are cleared, making it dead (the planner will drop it).
///
/// The pass runs over all node indices and is idempotent (a rope node is only
/// ever fused once because after fusion its name changes to
/// `"fused_rope_pair"`).
#[allow(clippy::cast_possible_truncation)] // graph will never have 2^32 nodes
fn fuse_rope_pairs<B: Backend + MatmulOps + RopeOps + ContextBackend>(graph: &mut Graph<B>) {
    let n = graph.nodes.len();
    // Collect all RopeOp node indices first to avoid borrow issues.
    let rope_nodes: Vec<usize> = (0..n)
        .filter(|&i| graph.nodes[i].op.name() == "rope")
        .collect();

    // Set of node indices already fused (as the second operand) — skip them.
    let mut fused_as_b: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for &i in &rope_nodes {
        if fused_as_b.contains(&i) {
            continue;
        }

        // Extract A's inputs: [A_input, cos_ref, sin_ref].
        if graph.nodes[i].inputs.len() != 3 {
            continue;
        }
        let a_input_ref = graph.nodes[i].inputs[0];
        let cos_ref = graph.nodes[i].inputs[1];
        let sin_ref = graph.nodes[i].inputs[2];

        // Extract A's position offset.
        let offset_a = {
            let Some(rope_a) = graph.nodes[i].op.as_any().downcast_ref::<RopeOp>() else {
                continue;
            };
            rope_a.offset
        };

        // Find another RopeOp that shares the same cos/sin.
        let Some(&j) = rope_nodes.iter().find(|&&j| {
            j != i
                && !fused_as_b.contains(&j)
                && graph.nodes[j].inputs.len() == 3
                && graph.nodes[j].inputs[1] == cos_ref
                && graph.nodes[j].inputs[2] == sin_ref
        }) else {
            continue;
        };

        let b_input_ref = graph.nodes[j].inputs[0];
        let offset_b = {
            let Some(rope_b) = graph.nodes[j].op.as_any().downcast_ref::<RopeOp>() else {
                continue;
            };
            rope_b.offset
        };

        let a_id = NodeId(i as u32);
        let b_id = NodeId(j as u32);

        // Replace node A with FusedRopePairOp.
        graph.nodes[i].op = Box::new(FusedRopePairOp { offset_a, offset_b });
        // New inputs: [A_input, B_input, cos, sin].
        graph.nodes[i].inputs = vec![a_input_ref, b_input_ref, cos_ref, sin_ref].into();
        // Add a second output for the K result.
        let shape_b = graph.nodes[j].output_shapes[0].clone();
        let dtype_b = graph.nodes[j].output_dtypes[0];
        graph.nodes[i].output_shapes.push(shape_b);
        graph.nodes[i].output_dtypes.push(dtype_b);

        // Rewire all consumers of B's output (B_id, 0) → (A_id, 1).
        let old_b_ref = (b_id, 0u32);
        let new_b_ref = (a_id, 1u32);
        for k in 0..graph.nodes.len() {
            for l in 0..graph.nodes[k].inputs.len() {
                if graph.nodes[k].inputs[l] == old_b_ref {
                    graph.nodes[k].inputs[l] = new_b_ref;
                }
            }
        }
        // Rewire graph outputs too.
        for out in &mut graph.outputs {
            if *out == b_id {
                *out = a_id;
            }
        }

        // Kill node B.
        graph.nodes[j].inputs.clear();

        fused_as_b.insert(j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::graph::builder_traits::{GraphArithOps, GraphNormOps, GraphSiluOps};
    use crate::graph::op_node::OutputRef;

    /// Minimal test backend (same pattern as graph/mod.rs tests).
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

    impl crate::backend::Backend for TestBackend {
        type Tensor = DummyTensor;
        type DeviceHandle = ();
        type PagedKvCache = ();
        type KvCache = ();
        type ExecutorState = ();
        type RuntimeState = DummyRuntimeState;
        type Logits = DummyLogits;
        type Comm = ();

        fn logits_from_tensor(_tensor: Self::Tensor) -> Self::Logits {
            DummyLogits
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
        fn silu(_input: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn logit_softcap(_input: &DummyTensor, _cap: f32) -> crate::Result<DummyTensor> {
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

    impl crate::backend::SwigluOps for TestBackend {
        fn swiglu(_gate: &DummyTensor, _up: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl crate::backend::MoeOps for TestBackend {
        fn moe_forward_softmax<F>(
            _hidden: &DummyTensor,
            _gate_weight: &DummyTensor,
            _num_experts: usize,
            _num_experts_per_tok: usize,
            _norm_topk_prob: bool,
            _expert_fn: F,
        ) -> crate::Result<DummyTensor>
        where
            F: Fn(usize, &DummyTensor) -> crate::Result<DummyTensor>,
        {
            unimplemented!()
        }
    }

    impl crate::backend::MoeSigmoidOps for TestBackend {
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
        ) -> crate::Result<DummyTensor>
        where
            F: Fn(usize, &DummyTensor) -> crate::Result<DummyTensor>,
        {
            unimplemented!()
        }
    }

    impl crate::backend::TensorDataOps for TestBackend {
        fn to_f32_vec(_tensor: &DummyTensor) -> crate::Result<Vec<f32>> {
            unimplemented!()
        }
        fn to_raw_bytes(_tensor: &DummyTensor) -> crate::Result<Vec<u8>> {
            unimplemented!()
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

    // NOTE: `ArgmaxLastOps` is required by `GraphArgmaxOps`. This impl is
    // local to this test module because `TestBackend` is defined here.
    // The canonical no-op body is identical across all test backends; the
    // orphan rule prevents moving it to `test_helpers` without also moving
    // the `TestBackend` type itself.
    impl crate::backend::ArgmaxLastOps for TestBackend {
        fn argmax_last_tensor(_input: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    #[test]
    fn fuse_silu_mul_into_swiglu() {
        let mut graph = Graph::<TestBackend>::new();

        let gate: OutputRef = graph.add_input(&[4, 128], DType::F32);
        let up: OutputRef = graph.add_input(&[4, 128], DType::F32);

        let silu_out = graph.add_silu(gate);
        let mul_out = graph.add_mul(silu_out, up);
        graph.set_output(mul_out.0);

        optimize(&mut graph);

        // The Mul node should have been replaced with Swiglu.
        let mul_node = &graph.nodes[mul_out.0 .0 as usize];
        assert_eq!(
            mul_node.op.name(),
            "swiglu",
            "Expected swiglu, got {}",
            mul_node.op.name()
        );
        // Swiglu inputs should be (gate, up) — the silu's original input
        // and the other mul operand.
        assert_eq!(mul_node.inputs[0], gate);
        assert_eq!(mul_node.inputs[1], up);
    }

    #[test]
    fn no_fuse_silu_with_multiple_consumers() {
        let mut graph = Graph::<TestBackend>::new();

        let gate: OutputRef = graph.add_input(&[4, 128], DType::F32);
        let up: OutputRef = graph.add_input(&[4, 128], DType::F32);

        let silu_out = graph.add_silu(gate);
        let mul_out = graph.add_mul(silu_out, up);
        // Second consumer of the silu output.
        let add_out = graph.add_add(silu_out, up);
        graph.set_output(mul_out.0);
        graph.set_output(add_out.0);

        optimize(&mut graph);

        // The Mul should NOT have been fused because silu has 2 consumers.
        let mul_node = &graph.nodes[mul_out.0 .0 as usize];
        assert_eq!(
            mul_node.op.name(),
            "mul",
            "Expected mul (unfused), got {}",
            mul_node.op.name()
        );
    }

    #[test]
    fn fuse_rope_pair_shared_cos_sin() {
        use crate::graph::builder_traits::GraphRopeOps as _;
        let mut graph = Graph::<TestBackend>::new();

        // seq=8, heads=4, head_dim=64
        let q = graph.add_input(&[8, 4, 64], DType::F32);
        let k = graph.add_input(&[8, 4, 64], DType::F32);
        let cos = graph.add_input(&[8, 32], DType::F32);
        let sin = graph.add_input(&[8, 32], DType::F32);

        let q_rope = graph.add_rope(q, cos, sin, 0);
        let k_rope = graph.add_rope(k, cos, sin, 0);
        graph.set_output(q_rope.0);
        graph.set_output(k_rope.0);

        optimize(&mut graph);

        // Q node should have become FusedRopePairOp.
        let q_node = &graph.nodes[q_rope.0 .0 as usize];
        assert_eq!(
            q_node.op.name(),
            "fused_rope_pair",
            "Expected fused_rope_pair, got {}",
            q_node.op.name()
        );
        // FusedRopePairOp has 4 inputs and 2 outputs.
        assert_eq!(q_node.inputs.len(), 4);
        assert_eq!(q_node.output_shapes.len(), 2);
        assert_eq!(q_node.output_dtypes.len(), 2);

        // K node should be dead (inputs cleared).
        let k_node = &graph.nodes[k_rope.0 .0 as usize];
        assert!(k_node.inputs.is_empty());

        // Graph outputs should both point at the Q (fused) node.
        assert_eq!(graph.outputs[0], q_rope.0);
        assert_eq!(graph.outputs[1], q_rope.0);
    }

    #[test]
    fn no_fuse_rope_different_cos_sin() {
        use crate::graph::builder_traits::GraphRopeOps as _;
        let mut graph = Graph::<TestBackend>::new();

        let q = graph.add_input(&[8, 4, 64], DType::F32);
        let k = graph.add_input(&[8, 4, 64], DType::F32);
        let cos1 = graph.add_input(&[8, 32], DType::F32);
        let sin1 = graph.add_input(&[8, 32], DType::F32);
        let cos2 = graph.add_input(&[8, 32], DType::F32);
        let sin2 = graph.add_input(&[8, 32], DType::F32);

        let q_rope = graph.add_rope(q, cos1, sin1, 0);
        let k_rope = graph.add_rope(k, cos2, sin2, 0);
        graph.set_output(q_rope.0);
        graph.set_output(k_rope.0);

        optimize(&mut graph);

        // Neither should be fused — different cos/sin.
        assert_eq!(graph.nodes[q_rope.0 .0 as usize].op.name(), "rope");
        assert_eq!(graph.nodes[k_rope.0 .0 as usize].op.name(), "rope");
    }

    #[test]
    fn fuse_add_rmsnorm() {
        let mut graph = Graph::<TestBackend>::new();

        let norm_w = graph.register_tensor_weight("ln.weight", &[128], DType::F32);
        let a: OutputRef = graph.add_input(&[4, 128], DType::F32);
        let b: OutputRef = graph.add_input(&[4, 128], DType::F32);

        let sum = graph.add_add(a, b);
        let normed = graph.add_rms_norm(sum, norm_w, 1e-5);
        graph.set_output(sum.0); // residual output
        graph.set_output(normed.0); // normed output

        // The add has 2 consumers (rms_norm + set_output)?
        // No: set_output doesn't count as a node consumer — it's in
        // graph.outputs, not in any node's inputs. So consumer_count
        // for `sum` is 1 (just the rms_norm).
        optimize(&mut graph);

        // The Add node should have become AddRmsNorm with 2 outputs.
        let add_node = &graph.nodes[sum.0 .0 as usize];
        assert_eq!(
            add_node.op.name(),
            "add_rms_norm",
            "Expected add_rms_norm, got {}",
            add_node.op.name()
        );
        assert_eq!(add_node.output_shapes.len(), 2);
        assert_eq!(add_node.output_dtypes.len(), 2);

        // The old RmsNorm node should be dead (inputs cleared).
        let norm_node = &graph.nodes[normed.0 .0 as usize];
        assert!(norm_node.inputs.is_empty());

        // Graph outputs should have been rewired: the normed output
        // now points at the add node (which produces both outputs).
        // Output 0 was sum.0 (already the add node).
        // Output 1 was normed.0 (the rms_norm node) — should be rewired to sum.0 (the add node).
        assert_eq!(graph.outputs[0], sum.0);
        assert_eq!(graph.outputs[1], sum.0);
    }
}
