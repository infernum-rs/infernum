//! Example: adding a custom op to a CPU graph without modifying infernum.
//!
//! Defines `ScaledSiluOp` — element-wise SiLU followed by a constant scale
//! factor — and wires it into a `Graph<CpuBackend>` alongside built-in ops.
//! The graph is then executed through the CPU executor to verify that the
//! custom op dispatches correctly via `OpNode::execute`.
//!
//! Run with:
//! ```
//! cargo run --example custom_graph_op
//! ```

use std::any::Any;

use infernum::backend::ContextBackend as _;
use infernum::dtype::DType;
use infernum::graph::execute_context::ExecuteContext;
use infernum::graph::{Arena, NodeId, OpNode, OutputRef, WeightStore};
use infernum::tensor::Tensor as _;
use infernum::{plan, Result};
use infernum_cpu::executor::execute;
use infernum_cpu::{CpuBackend, CpuLinearWeight, CpuTensor};

// ---------------------------------------------------------------------------
// Custom op definition
// ---------------------------------------------------------------------------

/// Element-wise SiLU (sigmoid linear unit) followed by a constant scale.
///
/// `output[i] = input[i] * sigmoid(input[i]) * scale`
///
/// This demonstrates a custom op that can be added to any `Graph<CpuBackend>`
/// from outside the `infernum` crate, without modifying any infernum source.
#[derive(Debug, Clone)]
struct ScaledSiluOp {
    scale: f32,
}

impl OpNode<CpuBackend> for ScaledSiluOp {
    fn name(&self) -> &'static str {
        "scaled_silu"
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        // Element-wise: output shape matches input shape.
        vec![input_shapes[0].to_vec()]
    }

    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }

    fn execute(
        &self,
        ctx: &mut ExecuteContext<'_, CpuBackend>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let input = CpuBackend::ctx_read(ctx, inputs[0]);
        let data = input.as_f32_slice();
        let out: Vec<f32> = data
            .iter()
            .map(|&x| {
                let silu = x / (1.0 + (-x).exp());
                silu * self.scale
            })
            .collect();
        let result = CpuTensor::from_f32(input.shape(), &out);
        CpuBackend::ctx_write(ctx, node_id, 0, result);
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Graph construction and execution
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    use infernum::graph::Graph;
    use infernum::GraphArithOps as _;

    // Build a small graph: input → add(input, input) → ScaledSiluOp(scale=2.0)
    // This mixes a built-in op (add) with our custom op in a single execution.
    let mut graph = Graph::<CpuBackend>::new();

    let input_ref = graph.add_input(&[4], DType::F32);

    // Built-in add op: x + x = 2x (verifies built-in ops still work alongside custom ones).
    let doubled = graph.add_add(input_ref, input_ref);

    // Custom ScaledSiluOp registered via the open extension point.
    let scaled_silu_id = graph.add_node(Box::new(ScaledSiluOp { scale: 2.0 }), &[doubled]);
    let output_ref: OutputRef = (scaled_silu_id, 0);
    graph.set_output(scaled_silu_id);

    // Plan and allocate the arena.
    let exec_plan = plan(&graph);
    let mut arena = Arena::new(exec_plan.arena_size);
    let weights = WeightStore::<CpuTensor, CpuLinearWeight>::new();

    // Input: [1.0, 0.0, -1.0, 2.0]
    let input = CpuTensor::from_f32(&[4], &[1.0_f32, 0.0, -1.0, 2.0]);

    let outputs = execute(
        &exec_plan,
        graph.nodes(),
        &mut arena,
        &weights,
        &[input],
        &[output_ref.0],
        None,
    )?;

    let result = outputs[0].as_f32_slice().to_vec();
    println!("custom_graph_op output: {result:?}");

    // Verify correctness against hand-computed expected values.
    // After add: doubled = [2.0, 0.0, -2.0, 4.0]
    // After scaled_silu(scale=2.0): silu(x)*2.0
    //   silu(2.0)  = 2.0 / (1 + e^-2)  ≈ 1.7616  → *2 ≈ 3.5232
    //   silu(0.0)  = 0.0 / (1 + e^0)   = 0.0     → *2 = 0.0
    //   silu(-2.0) = -2.0 / (1 + e^2)  ≈ -0.2384 → *2 ≈ -0.4769
    //   silu(4.0)  = 4.0 / (1 + e^-4)  ≈ 3.9281  → *2 ≈ 7.8562
    let expected = [3.523_2_f32, 0.0, -0.476_9, 7.856_2];
    for (got, exp) in result.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() < 1e-3,
            "output mismatch: got {got}, expected {exp}"
        );
    }

    println!("All assertions passed — custom op dispatched correctly.");
    Ok(())
}
