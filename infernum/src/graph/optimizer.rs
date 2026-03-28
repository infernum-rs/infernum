//! Graph optimization pass.
//!
//! Rewrites primitive ops into fused ops where profitable. This pass
//! runs between graph construction and planning — the graph builder
//! emits only primitive ops, and the optimizer fuses them.
//!
//! Current fusion rules:
//! - `Silu(a) → Mul(silu_out, b)` → `Swiglu(a, b)` (when Silu has single consumer)
//! - `Add(a, b) → RmsNorm(sum)` → `AddRmsNorm(a, b)` (when Add has single consumer)

use crate::backend::Backend;

use super::builder::Graph;
use super::node::NodeId;
use super::ops::Op;

/// Apply all fusion rules to the graph in-place.
pub fn optimize<B: Backend>(graph: &mut Graph<B>) {
    fuse_swiglu(graph);
    fuse_add_rmsnorm(graph);
}

/// Count how many nodes use `target` as an input.
fn consumer_count<B: Backend>(graph: &Graph<B>, target: NodeId) -> usize {
    graph
        .nodes
        .iter()
        .filter(|n| n.inputs.contains(&target))
        .count()
}

/// Fuse `Silu(a) → Mul(silu_out, b)` into `Swiglu(a, b)`.
///
/// Preconditions for fusion:
/// - The `Mul` node has exactly two inputs, one of which is a `Silu` output.
/// - The `Silu` node has exactly one consumer (this `Mul`).
fn fuse_swiglu<B: Backend>(graph: &mut Graph<B>) {
    let n = graph.nodes.len();
    for i in 0..n {
        if !matches!(graph.nodes[i].op, Op::Mul) {
            continue;
        }

        let inputs = graph.nodes[i].inputs.clone();
        if inputs.len() != 2 {
            continue;
        }

        // Check if either input is a Silu node.
        let (silu_idx, other_idx) =
            if matches!(graph.nodes[inputs[0].index() as usize].op, Op::Silu) {
                (0, 1)
            } else if matches!(graph.nodes[inputs[1].index() as usize].op, Op::Silu) {
                (1, 0)
            } else {
                continue;
            };

        let silu_id = inputs[silu_idx];
        let up_id = inputs[other_idx];

        // Only fuse if the Silu output is used exclusively by this Mul.
        if consumer_count(graph, silu_id) != 1 {
            continue;
        }

        // Get the Silu's input (the gate).
        let gate_id = graph.nodes[silu_id.index() as usize].inputs[0];

        // Replace Mul → Swiglu(gate, up).
        graph.nodes[i].op = Op::Swiglu;
        graph.nodes[i].inputs.clear();
        graph.nodes[i].inputs.push(gate_id);
        graph.nodes[i].inputs.push(up_id);

        // The orphaned Silu node will still be executed but its output
        // buffer is freed immediately since no node consumes it.
    }
}

/// Fuse `Add(a, b) → RmsNorm(sum, weight, eps)` into `AddRmsNorm`.
///
/// After fusion:
/// - The `Add` node becomes `AddRmsNorm`, its primary output = updated residual.
/// - The `RmsNorm` node becomes `SecondOutput { source: add_id }`, output = normed.
///
/// Preconditions:
/// - The `RmsNorm`'s sole input is an `Add`/`AddInplace` output.
/// - The `Add` output must be consumed by exactly one node (the `RmsNorm`)
///   OR by multiple nodes. In either case, fusion is safe because the
///   `AddRmsNorm` primary output has the same value as the original `Add`.
///   However, we only fuse when the `Add` has exactly one consumer to keep
///   the first implementation simple and conservative.
fn fuse_add_rmsnorm<B: Backend>(graph: &mut Graph<B>) {
    let n = graph.nodes.len();
    for i in 0..n {
        let (weight, eps) = match &graph.nodes[i].op {
            Op::RmsNorm { weight, eps } => (*weight, *eps),
            _ => continue,
        };

        if graph.nodes[i].inputs.len() != 1 {
            continue;
        }

        let add_id = graph.nodes[i].inputs[0];
        let add_idx = add_id.index() as usize;

        if !matches!(graph.nodes[add_idx].op, Op::Add | Op::AddInplace) {
            continue;
        }

        // Only fuse if the Add output feeds exactly one consumer.
        // This is conservative — AddRmsNorm's primary output equals the
        // original Add output so multi-consumer fusion would be correct,
        // but we keep it simple for now.
        if consumer_count(graph, add_id) != 1 {
            continue;
        }

        // Replace Add → AddRmsNorm.
        graph.nodes[add_idx].op = Op::AddRmsNorm { weight, eps };

        // Replace RmsNorm → SecondOutput { source: add_id }.
        // The SecondOutput at index `i` now produces the normed output.
        // The AddRmsNorm at `add_idx` produces the updated residual.
        // All downstream NodeId references are unchanged.
        graph.nodes[i].op = Op::SecondOutput { source: add_id };
        graph.nodes[i].inputs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::graph::builder_traits::{GraphArithOps, GraphNormOps, GraphSiluOps};

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
        type RuntimeState = DummyRuntimeState;
        type Logits = DummyLogits;
        type Comm = ();

        fn logits_from_tensor(_tensor: Self::Tensor) -> Self::Logits {
            DummyLogits
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
    }

    #[test]
    fn fuse_silu_mul_into_swiglu() {
        let mut graph = Graph::<TestBackend>::new();

        let gate = graph.add_input(&[4, 128], DType::F32);
        let up = graph.add_input(&[4, 128], DType::F32);

        let silu_out = graph.add_silu(gate);
        let mul_out = graph.add_mul(silu_out, up);
        graph.set_output(mul_out);

        optimize(&mut graph);

        // The Mul node should have been replaced with Swiglu.
        let mul_node = &graph.nodes[mul_out.index() as usize];
        assert!(
            matches!(mul_node.op, Op::Swiglu),
            "Expected Swiglu, got {:?}",
            mul_node.op
        );
        assert_eq!(mul_node.inputs[0], gate);
        assert_eq!(mul_node.inputs[1], up);
    }

    #[test]
    fn no_fuse_silu_with_multiple_consumers() {
        let mut graph = Graph::<TestBackend>::new();

        let gate = graph.add_input(&[4, 128], DType::F32);
        let up = graph.add_input(&[4, 128], DType::F32);

        let silu_out = graph.add_silu(gate);
        let mul_out = graph.add_mul(silu_out, up);
        // Second consumer of the silu output.
        let add_out = graph.add_add(silu_out, up);
        graph.set_output(mul_out);
        graph.set_output(add_out);

        optimize(&mut graph);

        // The Mul should NOT have been fused because silu has 2 consumers.
        let mul_node = &graph.nodes[mul_out.index() as usize];
        assert!(
            matches!(mul_node.op, Op::Mul),
            "Expected Mul (unfused), got {:?}",
            mul_node.op
        );
    }

    #[test]
    fn fuse_add_rmsnorm() {
        let mut graph = Graph::<TestBackend>::new();

        let norm_w = graph.register_tensor_weight("ln.weight", &[128], DType::F32);
        let a = graph.add_input(&[4, 128], DType::F32);
        let b = graph.add_input(&[4, 128], DType::F32);

        let sum = graph.add_add(a, b);
        let normed = graph.add_rms_norm(sum, norm_w, 1e-5);
        graph.set_output(sum); // residual output
        graph.set_output(normed); // normed output

        // The add has 2 consumers (rms_norm + set_output).
        // But set_output doesn't count as a node consumer — it's in
        // graph.outputs, not in any node's inputs. So consumer_count
        // for `sum` is 1 (just the rms_norm).
        optimize(&mut graph);

        // The Add should have become AddRmsNorm.
        let add_node = &graph.nodes[sum.index() as usize];
        assert!(
            matches!(add_node.op, Op::AddRmsNorm { .. }),
            "Expected AddRmsNorm, got {:?}",
            add_node.op
        );

        // The RmsNorm should have become SecondOutput.
        let norm_node = &graph.nodes[normed.index() as usize];
        assert!(
            matches!(norm_node.op, Op::SecondOutput { source } if source == sum),
            "Expected SecondOutput pointing to add, got {:?}",
            norm_node.op
        );
    }
}
