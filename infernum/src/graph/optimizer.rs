//! Graph optimization pass.
//!
//! Rewrites primitive ops into fused ops where profitable. This pass
//! runs between graph construction and planning — the graph builder
//! emits only primitive ops, and the optimizer fuses them.
//!
//! Current fusion rules:
//! - `Silu(a) → Mul(silu_out, b)` → `Swiglu(a, b)` (when Silu has single consumer)
//! - `Add(a, b) → RmsNorm(sum)` → `AddRmsNorm(a, b)` (when Add has single consumer)

use crate::backend::{ArithOps, Backend, ContextBackend, MatmulOps, NormOps, SwigluOps};

use super::builder::Graph;
use super::builtin_ops::{AddRmsNormOp, RmsNormOp, SwigluOp};
use super::node::NodeId;

/// Apply all fusion rules to the graph in-place.
pub fn optimize<B: Backend + MatmulOps + ArithOps + NormOps + SwigluOps + ContextBackend>(
    graph: &mut Graph<B>,
) {
    fuse_swiglu(graph);
    fuse_add_rmsnorm(graph);
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
