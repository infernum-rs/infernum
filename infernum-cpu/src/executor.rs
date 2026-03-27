//! Graph executor for the CPU backend.
//!
//! Walks an [`ExecutionPlan`] in topological order and dispatches each
//! [`Op`] to the existing CPU SIMD kernels. Intermediate tensors are
//! read from / written to a shared [`Arena`].

use infernum::backend::{
    ArithOps, AttentionOps, BiasOps, EmbedOps, MatmulExtOps, MatmulOps, NormOps, RopeOps,
    SwigluOps, TensorOps,
};
use infernum::graph::{Arena, WeightStore};
use infernum::tensor::Tensor;
use infernum::{ExecutionPlan, GraphNode, NodeId, Op, Result};

use crate::tensor::{CpuLinearWeight, CpuTensor};
use crate::CpuBackend;

/// Read a tensor from the arena using the plan's buffer slot for `node_id`.
fn read_tensor(
    arena: &Arena,
    plan: &ExecutionPlan,
    nodes: &[GraphNode],
    node_id: NodeId,
) -> CpuTensor {
    let node = &nodes[node_id.index() as usize];
    let slot = plan.slot(node_id).expect("node has no buffer slot");
    let num_elements: usize = node.shape.iter().product();
    let data = arena.f32_slice(slot.offset, num_elements);
    CpuTensor::from_f32(&node.shape, data)
}

/// Write a tensor into the arena at the plan's buffer slot for `node_id`.
fn write_tensor(arena: &mut Arena, plan: &ExecutionPlan, node_id: NodeId, tensor: &CpuTensor) {
    if let Some(slot) = plan.slot(node_id) {
        let src = tensor.as_f32_slice();
        let dst = arena.f32_slice_mut(slot.offset, src.len());
        dst.copy_from_slice(src);
    }
}

/// Execute a computation graph on the CPU backend.
///
/// Walks the `plan.schedule` in topological order, dispatching each [`Op`]
/// to the appropriate CPU SIMD kernel. Intermediate tensors are stored in
/// the shared `arena`.
///
/// # Arguments
///
/// * `plan` — Memory plan with schedule, buffer offsets, and arena size.
/// * `nodes` — Graph nodes (indexed by `NodeId`).
/// * `arena` — Pre-allocated byte arena for intermediate tensors.
/// * `weights` — Loaded model weights (embedding, norm, linear, bias, etc.).
/// * `inputs` — External input tensors, one per `Op::Input` in schedule order.
/// * `output_nodes` — `NodeId`s of the graph outputs to collect.
///
/// # Errors
///
/// Returns an error if any op kernel fails.
#[allow(clippy::too_many_lines)]
pub fn execute(
    plan: &ExecutionPlan,
    nodes: &[GraphNode],
    arena: &mut Arena,
    weights: &WeightStore<CpuTensor, CpuLinearWeight>,
    inputs: &[CpuTensor],
    output_nodes: &[NodeId],
) -> Result<Vec<CpuTensor>> {
    let mut input_idx: usize = 0;

    for &node_id in &plan.schedule {
        let node = &nodes[node_id.index() as usize];

        match &node.op {
            // --- Input ---
            Op::Input => {
                let tensor = &inputs[input_idx];
                input_idx += 1;
                write_tensor(arena, plan, node_id, tensor);
            }

            // --- Embedding ---
            Op::EmbeddingGather { table } => {
                let token_ids = read_tensor(arena, plan, nodes, node.inputs[0]);
                let table_w = weights.tensor_weight(*table);
                let seq_len = node.shape[0];
                let result = <CpuBackend as EmbedOps>::embedding_gather_tensor(
                    table_w, &token_ids, seq_len,
                )?;
                write_tensor(arena, plan, node_id, &result);
            }

            // --- Norm ---
            Op::RmsNorm { weight, eps } => {
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                let w = weights.tensor_weight(*weight);
                let result = <CpuBackend as NormOps>::rms_norm(&input, w, *eps)?;
                write_tensor(arena, plan, node_id, &result);
            }

            Op::AddRmsNorm { weight, eps } => {
                let residual = read_tensor(arena, plan, nodes, node.inputs[0]);
                let delta = read_tensor(arena, plan, nodes, node.inputs[1]);
                let w = weights.tensor_weight(*weight);
                let (updated, normed) =
                    <CpuBackend as NormOps>::add_rmsnorm(&residual, &delta, w, *eps)?;

                // Write primary output (updated residual).
                write_tensor(arena, plan, node_id, &updated);

                // Write secondary output (normed) — the next NodeId.
                let second_id = NodeId::from_index(node_id.index() + 1);
                write_tensor(arena, plan, second_id, &normed);
            }

            // --- Linear / Matmul ---
            Op::Linear { weight } | Op::LmHead { weight, .. } => {
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                let w = weights.linear_weight(*weight);
                let result = <CpuBackend as MatmulOps>::linear(&input, w)?;
                write_tensor(arena, plan, node_id, &result);
            }

            Op::LinearPair { w1, w2 } => {
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                let lw1 = weights.linear_weight(*w1);
                let lw2 = weights.linear_weight(*w2);
                let (out1, out2) = <CpuBackend as MatmulOps>::linear_pair(&input, lw1, lw2)?;
                write_tensor(arena, plan, node_id, &out1);
                let second_id = NodeId::from_index(node_id.index() + 1);
                write_tensor(arena, plan, second_id, &out2);
            }

            Op::LinearTriple { w1, w2, w3 } => {
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                let lw1 = weights.linear_weight(*w1);
                let lw2 = weights.linear_weight(*w2);
                let lw3 = weights.linear_weight(*w3);
                let (out1, out2, out3) =
                    <CpuBackend as MatmulOps>::linear_triple(&input, lw1, lw2, lw3)?;
                write_tensor(arena, plan, node_id, &out1);
                let second_id = NodeId::from_index(node_id.index() + 1);
                write_tensor(arena, plan, second_id, &out2);
                let third_id = NodeId::from_index(node_id.index() + 2);
                write_tensor(arena, plan, third_id, &out3);
            }

            Op::Matmul => {
                let a = read_tensor(arena, plan, nodes, node.inputs[0]);
                let b = read_tensor(arena, plan, nodes, node.inputs[1]);
                let result = <CpuBackend as MatmulOps>::matmul(&a, &b)?;
                write_tensor(arena, plan, node_id, &result);
            }

            Op::MatmulBf16F32 => {
                let a = read_tensor(arena, plan, nodes, node.inputs[0]);
                let b = read_tensor(arena, plan, nodes, node.inputs[1]);
                let result = <CpuBackend as MatmulExtOps>::matmul_bf16_f32(&a, &b)?;
                write_tensor(arena, plan, node_id, &result);
            }

            // --- Activations ---
            Op::Swiglu => {
                let gate = read_tensor(arena, plan, nodes, node.inputs[0]);
                let up = read_tensor(arena, plan, nodes, node.inputs[1]);
                let result = <CpuBackend as SwigluOps>::swiglu(&gate, &up)?;
                write_tensor(arena, plan, node_id, &result);
            }

            // --- Arithmetic ---
            Op::Add | Op::AddInplace => {
                let a = read_tensor(arena, plan, nodes, node.inputs[0]);
                let b = read_tensor(arena, plan, nodes, node.inputs[1]);
                let result = <CpuBackend as ArithOps>::add(&a, &b)?;
                write_tensor(arena, plan, node_id, &result);
            }

            Op::Scale { factor } => {
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                let mut result = input.clone();
                <CpuBackend as ArithOps>::scale_inplace(&mut result, *factor)?;
                write_tensor(arena, plan, node_id, &result);
            }

            Op::BiasAdd { bias } => {
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                let mut result = input.clone();
                let bias_w = weights.tensor_weight(*bias);
                <CpuBackend as BiasOps>::bias_add_inplace(&mut result, bias_w)?;
                write_tensor(arena, plan, node_id, &result);
            }

            // --- RoPE ---
            Op::Rope { offset } => {
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                let cos_cache = read_tensor(arena, plan, nodes, node.inputs[1]);
                let sin_cache = read_tensor(arena, plan, nodes, node.inputs[2]);
                let result =
                    <CpuBackend as RopeOps>::apply_rope(&input, &cos_cache, &sin_cache, *offset)?;
                write_tensor(arena, plan, node_id, &result);
            }

            // --- Attention ---
            Op::FusedAttentionPrefill {
                offset,
                scale,
                softcap,
                sliding_window,
            } => {
                let q = read_tensor(arena, plan, nodes, node.inputs[0]);
                let k = read_tensor(arena, plan, nodes, node.inputs[1]);
                let v = read_tensor(arena, plan, nodes, node.inputs[2]);
                let result = <CpuBackend as AttentionOps>::fused_attention_prefill(
                    &q,
                    &k,
                    &v,
                    *offset,
                    *scale,
                    *softcap,
                    *sliding_window,
                )?;
                write_tensor(arena, plan, node_id, &result);
            }

            Op::FusedAttentionDecode { softcap } => {
                let q = read_tensor(arena, plan, nodes, node.inputs[0]);
                let k = read_tensor(arena, plan, nodes, node.inputs[1]);
                let v = read_tensor(arena, plan, nodes, node.inputs[2]);
                let result = <CpuBackend as AttentionOps>::fused_attention_decode(
                    &q, &k, &v, None, *softcap, None,
                )?;
                write_tensor(arena, plan, node_id, &result);
            }

            // --- Shape / Data movement ---
            Op::Reshape { .. } => {
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                let reshaped = input.reshape(&node.shape);
                write_tensor(arena, plan, node_id, &reshaped);
            }

            Op::RepeatKv { num_repeats } => {
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                let result = <CpuBackend as TensorOps>::repeat_kv(&input, *num_repeats)?;
                write_tensor(arena, plan, node_id, &result);
            }

            Op::ExtractLastRow { seq_len } => {
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                let inner_size: usize = node.shape[1..].iter().product();
                let offset = (seq_len - 1) * inner_size;
                let result = input.slice_view(offset, &node.shape);
                write_tensor(arena, plan, node_id, &result);
            }

            Op::Transpose2d => {
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                let result = <CpuBackend as TensorOps>::transpose_2d(&input)?;
                write_tensor(arena, plan, node_id, &result);
            }

            Op::SplitInnerDim { left_size } => {
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                let inner = *input.shape().last().expect("non-empty shape");
                let right_size = inner - left_size;
                let (left, right) =
                    <CpuBackend as TensorOps>::split_inner_dim(&input, *left_size, right_size)?;
                write_tensor(arena, plan, node_id, &left);
                let second_id = NodeId::from_index(node_id.index() + 1);
                write_tensor(arena, plan, second_id, &right);
            }

            Op::ConcatInnerDim => {
                let a = read_tensor(arena, plan, nodes, node.inputs[0]);
                let b = read_tensor(arena, plan, nodes, node.inputs[1]);
                let result = <CpuBackend as TensorOps>::concat_inner_dim(&a, &b)?;
                write_tensor(arena, plan, node_id, &result);
            }

            Op::SliceView { offset, shape } => {
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                let result = input.slice_view(*offset, shape);
                write_tensor(arena, plan, node_id, &result);
            }

            // --- Cast (no-op on CPU: all data is f32) ---
            Op::CastToF32 | Op::CastFromF32 { .. } => {
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                write_tensor(arena, plan, node_id, &input);
            }

            // --- Multi-output secondary (already handled by parent op) ---
            Op::SecondOutput { .. } => {}

            // --- Unimplemented ---
            op => panic!("CPU executor: unimplemented op {op:?}"),
        }
    }

    // Collect output tensors.
    let outputs = output_nodes
        .iter()
        .map(|&id| read_tensor(arena, plan, nodes, id))
        .collect();

    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use infernum::dtype::DType;
    use infernum::graph::Graph;
    use infernum::{plan, GraphArithOps, GraphMatmulOps, GraphNormOps};

    /// Test 1: Simple add graph — two inputs added together.
    #[test]
    fn simple_add_graph() {
        let mut graph = Graph::<CpuBackend>::new();

        let a = graph.add_input(&[4], DType::F32);
        let b = graph.add_input(&[4], DType::F32);
        let c = graph.add_add(a, b);
        graph.set_output(c);

        let exec_plan = plan(&graph);
        let mut arena = Arena::new(exec_plan.arena_size);
        let weights = WeightStore::<CpuTensor, CpuLinearWeight>::new();

        let input_a = CpuTensor::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let input_b = CpuTensor::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]);

        let outputs = execute(
            &exec_plan,
            graph.nodes(),
            &mut arena,
            &weights,
            &[input_a, input_b],
            graph.output_ids(),
        )
        .unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_f32_slice(), &[11.0, 22.0, 33.0, 44.0]);
    }

    /// Test 2: Linear projection — input × weight.
    #[test]
    fn linear_projection() {
        let mut graph = Graph::<CpuBackend>::new();

        // Register a dense linear weight: (out_features=2, in_features=3).
        let w_id = graph.register_linear_weight("proj", &[2, 3], DType::F32);
        let input_node = graph.add_input(&[1, 3], DType::F32);
        let output_node = graph.add_linear(input_node, w_id);
        graph.set_output(output_node);

        let exec_plan = plan(&graph);
        let mut arena = Arena::new(exec_plan.arena_size);

        // Build the weight store with a dense linear weight.
        // new_dense takes (in_features, out_features) = (3, 2) and creates both layouts.
        let weight_data = CpuTensor::from_f32(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let linear_w = CpuLinearWeight::new_dense(weight_data);
        let mut weights = WeightStore::<CpuTensor, CpuLinearWeight>::new();
        weights.push_linear_weight(linear_w);

        let input = CpuTensor::from_f32(&[1, 3], &[3.0, 5.0, 7.0]);

        let outputs = execute(
            &exec_plan,
            graph.nodes(),
            &mut arena,
            &weights,
            &[input],
            graph.output_ids(),
        )
        .unwrap();

        assert_eq!(outputs.len(), 1);
        let result = outputs[0].as_f32_slice();
        // [3, 5, 7] × [[1, 0], [0, 1], [0, 0]] = [3, 5]
        assert_eq!(result, &[3.0, 5.0]);
    }

    /// Test 3: Norm + linear chain — rms_norm then linear projection.
    #[test]
    fn norm_then_linear() {
        let mut graph = Graph::<CpuBackend>::new();

        let norm_w_id = graph.register_tensor_weight("ln.weight", &[4], DType::F32);
        let proj_w_id = graph.register_linear_weight("proj", &[2, 4], DType::F32);

        let input_node = graph.add_input(&[1, 4], DType::F32);
        let normed = graph.add_rms_norm(input_node, norm_w_id, 1e-5);
        let output_node = graph.add_linear(normed, proj_w_id);
        graph.set_output(output_node);

        let exec_plan = plan(&graph);
        let mut arena = Arena::new(exec_plan.arena_size);

        // Norm weight: all ones (RMS norm with unit weight is just x / rms(x)).
        let norm_weight = CpuTensor::from_f32(&[4], &[1.0, 1.0, 1.0, 1.0]);

        // Linear weight: identity-ish (4, 2) — pick first two dims.
        let weight_data = CpuTensor::from_f32(&[4, 2], &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let linear_w = CpuLinearWeight::new_dense(weight_data);

        let mut weights = WeightStore::<CpuTensor, CpuLinearWeight>::new();
        weights.push_tensor_weight(norm_weight);
        weights.push_linear_weight(linear_w);

        let input = CpuTensor::from_f32(&[1, 4], &[1.0, 2.0, 3.0, 4.0]);

        let outputs = execute(
            &exec_plan,
            graph.nodes(),
            &mut arena,
            &weights,
            &[input],
            graph.output_ids(),
        )
        .unwrap();

        assert_eq!(outputs.len(), 1);
        let result = outputs[0].as_f32_slice();

        // RMS of [1, 2, 3, 4] = sqrt((1+4+9+16)/4) = sqrt(30/4) ≈ 2.7386
        // Normalized: [1/rms, 2/rms, 3/rms, 4/rms]
        // Linear picks first two: [1/rms, 2/rms]
        let rms = (30.0_f32 / 4.0).sqrt();
        let expected = [1.0 / rms, 2.0 / rms];
        assert!(
            (result[0] - expected[0]).abs() < 1e-5,
            "expected {}, got {}",
            expected[0],
            result[0]
        );
        assert!(
            (result[1] - expected[1]).abs() < 1e-5,
            "expected {}, got {}",
            expected[1],
            result[1]
        );
    }
}
