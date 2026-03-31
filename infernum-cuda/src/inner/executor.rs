//! Graph executor for the CUDA backend.
//!
//! Walks an [`ExecutionPlan`] in topological order and dispatches each
//! operation to the existing CUDA kernels. Intermediate tensors are stored
//! in a `Vec<Vec<Option<CudaTensor>>>` indexed by `(NodeId, output_index)`.
//!
//! Unlike the CPU executor which uses a flat byte arena, the GPU executor
//! stores tensors as individual `CudaTensor` values. The `BufferPool`
//! provides allocation reuse automatically.

use infernum::backend::{
    ArithOps, AttentionOps, BiasOps, CastOps, EmbedOps, GegluOps, MatmulExtOps, MatmulOps, NormOps,
    RopeInterleavedOps, RopeOps, SwigluOps, TensorOps,
};
use infernum::graph::builtin_ops::{
    AddRmsNormOp, BiasAddOp, CastFromF32Op, EmbeddingGatherOp, ExtractLastRowOp,
    FusedAttentionDecodeOp, FusedAttentionPrefillOp, GegluOp, LinearOp, LinearPairOp,
    LinearTripleOp, LmHeadOp, RepeatKvOp, ReshapeOp, RmsNormOp, RopeBatchedOp, RopeInterleavedOp,
    RopeOp, ScaleOp, SliceViewOp, SplitInnerDimOp,
};
use infernum::graph::{GraphNode, OutputRef, WeightStore};
use infernum::tensor::Tensor;
use infernum::{ExecutionPlan, NodeId, Result};

use crate::cuda::ops;
use crate::cuda::ops::LinearWeight;
use crate::cuda::CudaTensor;
use crate::CudaBackend;

/// Read a tensor from the node buffer by `OutputRef`.
///
/// # Panics
/// Panics if the node output has no stored tensor.
fn read(buffers: &[Vec<Option<CudaTensor>>], output_ref: OutputRef) -> &CudaTensor {
    let (node_id, output_idx) = output_ref;
    buffers[node_id.index() as usize][output_idx as usize]
        .as_ref()
        .unwrap_or_else(|| panic!("node {node_id:?} output {output_idx} has no stored tensor"))
}

/// Take (move) a tensor out of the node buffer. Used for the final outputs.
fn take(buffers: &mut [Vec<Option<CudaTensor>>], node_id: NodeId, output_idx: u32) -> CudaTensor {
    buffers[node_id.index() as usize][output_idx as usize]
        .take()
        .unwrap_or_else(|| panic!("node {node_id:?} output {output_idx} has no stored tensor"))
}

/// Store a tensor in the node buffer.
fn store(
    buffers: &mut [Vec<Option<CudaTensor>>],
    node_id: NodeId,
    output_idx: u32,
    tensor: CudaTensor,
) {
    buffers[node_id.index() as usize][output_idx as usize] = Some(tensor);
}

/// Execute a computation graph on the CUDA backend.
///
/// Walks the `plan.schedule` in topological order, dispatching each operation
/// to the appropriate CUDA kernel. Intermediate tensors are stored as
/// `CudaTensor` values (leveraging `BufferPool` for allocation reuse).
///
/// # Arguments
///
/// * `plan` — Memory plan with schedule and buffer offsets.
/// * `nodes` — Graph nodes (indexed by `NodeId`).
/// * `weights` — Loaded model weights.
/// * `inputs` — External input tensors, one per `Input` op in schedule order.
/// * `output_nodes` — `NodeId`s of the graph outputs to collect.
///
/// # Errors
///
/// Returns an error if any op kernel fails.
#[allow(clippy::too_many_lines, clippy::missing_panics_doc)]
pub fn execute(
    plan: &ExecutionPlan,
    nodes: &[GraphNode<CudaBackend>],
    weights: &WeightStore<CudaTensor, LinearWeight>,
    inputs: &[CudaTensor],
    output_nodes: &[NodeId],
) -> Result<Vec<CudaTensor>> {
    let mut buffers: Vec<Vec<Option<CudaTensor>>> = nodes
        .iter()
        .map(|node| {
            let num_outputs = node.output_shapes.len().max(1);
            vec![None; num_outputs]
        })
        .collect();
    let mut input_idx: usize = 0;

    for &node_id in &plan.schedule {
        let node = &nodes[node_id.index() as usize];
        let op_name = node.op.name();

        match op_name {
            // --- Input ---
            "input" => {
                let tensor = inputs[input_idx].clone();
                input_idx += 1;
                store(&mut buffers, node_id, 0, tensor);
            }

            // --- Embedding ---
            "embedding_gather" => {
                let op = node
                    .op
                    .as_any()
                    .downcast_ref::<EmbeddingGatherOp>()
                    .unwrap();
                let token_ids = read(&buffers, node.inputs[0]);
                let table_w = weights.tensor_weight(op.table);
                let seq_len = node.output_shapes[0][0];
                let result = <CudaBackend as EmbedOps>::embedding_gather_tensor(
                    table_w, token_ids, seq_len,
                )?;
                store(&mut buffers, node_id, 0, result);
            }

            // --- Norm ---
            "rms_norm" => {
                let op = node.op.as_any().downcast_ref::<RmsNormOp>().unwrap();
                let input = read(&buffers, node.inputs[0]);
                let w = weights.tensor_weight(op.weight);
                let result = <CudaBackend as NormOps>::rms_norm(input, w, op.eps)?;
                store(&mut buffers, node_id, 0, result);
            }

            "add_rms_norm" => {
                let op = node.op.as_any().downcast_ref::<AddRmsNormOp>().unwrap();
                let residual = read(&buffers, node.inputs[0]);
                let delta = read(&buffers, node.inputs[1]);
                let w = weights.tensor_weight(op.weight);
                let (updated, normed) =
                    <CudaBackend as NormOps>::add_rmsnorm(residual, delta, w, op.eps)?;
                store(&mut buffers, node_id, 0, updated);
                store(&mut buffers, node_id, 1, normed);
            }

            // --- Linear / Matmul ---
            "lm_head" | "linear" => {
                let weight = if op_name == "lm_head" {
                    node.op.as_any().downcast_ref::<LmHeadOp>().unwrap().weight
                } else {
                    node.op.as_any().downcast_ref::<LinearOp>().unwrap().weight
                };
                let input = read(&buffers, node.inputs[0]);
                let w = weights.linear_weight(weight);
                let result = <CudaBackend as MatmulOps>::linear(input, w)?;
                store(&mut buffers, node_id, 0, result);
            }

            "linear_pair" => {
                let op = node.op.as_any().downcast_ref::<LinearPairOp>().unwrap();
                let input = read(&buffers, node.inputs[0]);
                let lw1 = weights.linear_weight(op.w1);
                let lw2 = weights.linear_weight(op.w2);
                let (out1, out2) = <CudaBackend as MatmulOps>::linear_pair(input, lw1, lw2)?;
                store(&mut buffers, node_id, 0, out1);
                store(&mut buffers, node_id, 1, out2);
            }

            "linear_triple" => {
                let op = node.op.as_any().downcast_ref::<LinearTripleOp>().unwrap();
                let input = read(&buffers, node.inputs[0]);
                let lw1 = weights.linear_weight(op.w1);
                let lw2 = weights.linear_weight(op.w2);
                let lw3 = weights.linear_weight(op.w3);
                let (out1, out2, out3) =
                    <CudaBackend as MatmulOps>::linear_triple(input, lw1, lw2, lw3)?;
                store(&mut buffers, node_id, 0, out1);
                store(&mut buffers, node_id, 1, out2);
                store(&mut buffers, node_id, 2, out3);
            }

            "matmul" => {
                let a = read(&buffers, node.inputs[0]);
                let b = read(&buffers, node.inputs[1]);
                let result = <CudaBackend as MatmulOps>::matmul(a, b)?;
                store(&mut buffers, node_id, 0, result);
            }

            "matmul_bf16_f32" => {
                let a = read(&buffers, node.inputs[0]);
                let b = read(&buffers, node.inputs[1]);
                let result = <CudaBackend as MatmulExtOps>::matmul_bf16_f32(a, b)?;
                store(&mut buffers, node_id, 0, result);
            }

            // --- Activations ---
            "swiglu" => {
                let gate = read(&buffers, node.inputs[0]);
                let up = read(&buffers, node.inputs[1]);
                let result = <CudaBackend as SwigluOps>::swiglu(gate, up)?;
                store(&mut buffers, node_id, 0, result);
            }

            "geglu" => {
                let gate = read(&buffers, node.inputs[0]);
                let up = read(&buffers, node.inputs[1]);
                let result = <CudaBackend as GegluOps>::geglu(gate, up)?;
                store(&mut buffers, node_id, 0, result);
            }

            "silu" => {
                let input = read(&buffers, node.inputs[0]);
                let result = ops::silu(input)?;
                store(&mut buffers, node_id, 0, result);
            }

            "mul" => {
                let a = read(&buffers, node.inputs[0]);
                let b = read(&buffers, node.inputs[1]);
                let result = <CudaBackend as ArithOps>::mul(a, b)?;
                store(&mut buffers, node_id, 0, result);
            }

            // --- Arithmetic ---
            "add" => {
                let a = read(&buffers, node.inputs[0]);
                let b = read(&buffers, node.inputs[1]);
                let result = <CudaBackend as ArithOps>::add(a, b)?;
                store(&mut buffers, node_id, 0, result);
            }

            "add_inplace" => {
                let a = read(&buffers, node.inputs[0]);
                let b = read(&buffers, node.inputs[1]);
                // Clone to get a mutable copy, then add in-place.
                let mut result = a.clone();
                <CudaBackend as ArithOps>::add_inplace(&mut result, b)?;
                store(&mut buffers, node_id, 0, result);
            }

            "scale" => {
                let op = node.op.as_any().downcast_ref::<ScaleOp>().unwrap();
                let input = read(&buffers, node.inputs[0]);
                let mut result = input.clone();
                <CudaBackend as ArithOps>::scale_inplace(&mut result, op.factor)?;
                store(&mut buffers, node_id, 0, result);
            }

            "bias_add" => {
                let op = node.op.as_any().downcast_ref::<BiasAddOp>().unwrap();
                let input = read(&buffers, node.inputs[0]);
                let bias_w = weights.tensor_weight(op.bias);
                let mut result = input.clone();
                <CudaBackend as BiasOps>::bias_add_inplace(&mut result, bias_w)?;
                store(&mut buffers, node_id, 0, result);
            }

            // --- RoPE ---
            "rope" => {
                let op = node.op.as_any().downcast_ref::<RopeOp>().unwrap();
                let input = read(&buffers, node.inputs[0]);
                let cos_cache = read(&buffers, node.inputs[1]);
                let sin_cache = read(&buffers, node.inputs[2]);
                let result =
                    <CudaBackend as RopeOps>::apply_rope(input, cos_cache, sin_cache, op.offset)?;
                store(&mut buffers, node_id, 0, result);
            }

            "rope_batched" => {
                let op = node.op.as_any().downcast_ref::<RopeBatchedOp>().unwrap();
                let input = read(&buffers, node.inputs[0]);
                let cos_cache = read(&buffers, node.inputs[1]);
                let sin_cache = read(&buffers, node.inputs[2]);
                let positions = read(&buffers, node.inputs[3]);
                let result = <CudaBackend as RopeOps>::apply_rope_batched(
                    input,
                    cos_cache,
                    sin_cache,
                    positions,
                    op.batch_size,
                )?;
                store(&mut buffers, node_id, 0, result);
            }

            "rope_interleaved" => {
                let op = node
                    .op
                    .as_any()
                    .downcast_ref::<RopeInterleavedOp>()
                    .unwrap();
                let input = read(&buffers, node.inputs[0]);
                let cos_cache = read(&buffers, node.inputs[1]);
                let sin_cache = read(&buffers, node.inputs[2]);
                let result = <CudaBackend as RopeInterleavedOps>::apply_rope_interleaved(
                    input, cos_cache, sin_cache, op.offset,
                )?;
                store(&mut buffers, node_id, 0, result);
            }

            // --- Attention ---
            "fused_attention_prefill" => {
                let op = node
                    .op
                    .as_any()
                    .downcast_ref::<FusedAttentionPrefillOp>()
                    .unwrap();
                let q = read(&buffers, node.inputs[0]);
                let k = read(&buffers, node.inputs[1]);
                let v = read(&buffers, node.inputs[2]);
                let result = <CudaBackend as AttentionOps>::fused_attention_prefill(
                    q,
                    k,
                    v,
                    op.offset,
                    op.scale,
                    op.softcap,
                    op.sliding_window,
                )?;
                store(&mut buffers, node_id, 0, result);
            }

            "fused_attention_decode" => {
                let op = node
                    .op
                    .as_any()
                    .downcast_ref::<FusedAttentionDecodeOp>()
                    .unwrap();
                let q = read(&buffers, node.inputs[0]);
                let k = read(&buffers, node.inputs[1]);
                let v = read(&buffers, node.inputs[2]);
                let result = <CudaBackend as AttentionOps>::fused_attention_decode(
                    q, k, v, None, op.softcap, None,
                )?;
                store(&mut buffers, node_id, 0, result);
            }

            // --- Shape / Data movement ---
            "reshape" => {
                let op = node.op.as_any().downcast_ref::<ReshapeOp>().unwrap();
                let input = read(&buffers, node.inputs[0]);
                let result = input.reshape(&op.shape);
                store(&mut buffers, node_id, 0, result);
            }

            "slice_view" => {
                let op = node.op.as_any().downcast_ref::<SliceViewOp>().unwrap();
                let input = read(&buffers, node.inputs[0]);
                let result = input.slice_view(op.offset, &op.shape);
                store(&mut buffers, node_id, 0, result);
            }

            "transpose_2d" => {
                let input = read(&buffers, node.inputs[0]);
                let result = <CudaBackend as TensorOps>::transpose_2d(input)?;
                store(&mut buffers, node_id, 0, result);
            }

            "split_inner_dim" => {
                let op = node.op.as_any().downcast_ref::<SplitInnerDimOp>().unwrap();
                let input = read(&buffers, node.inputs[0]);
                let inner = *input.shape().last().expect("non-empty shape");
                let right_size = inner - op.left_size;
                let (left, right) =
                    <CudaBackend as TensorOps>::split_inner_dim(input, op.left_size, right_size)?;
                store(&mut buffers, node_id, 0, left);
                store(&mut buffers, node_id, 1, right);
            }

            "concat_inner_dim" => {
                let a = read(&buffers, node.inputs[0]);
                let b = read(&buffers, node.inputs[1]);
                let result = <CudaBackend as TensorOps>::concat_inner_dim(a, b)?;
                store(&mut buffers, node_id, 0, result);
            }

            "concat_seq" => {
                let a = read(&buffers, node.inputs[0]);
                let b = read(&buffers, node.inputs[1]);
                let result = <CudaBackend as TensorOps>::concat_rows(&[a.clone(), b.clone()])?;
                store(&mut buffers, node_id, 0, result);
            }

            "repeat_kv" => {
                let op = node.op.as_any().downcast_ref::<RepeatKvOp>().unwrap();
                let input = read(&buffers, node.inputs[0]);
                let result = <CudaBackend as TensorOps>::repeat_kv(input, op.num_repeats)?;
                store(&mut buffers, node_id, 0, result);
            }

            "extract_last_row" => {
                let op = node.op.as_any().downcast_ref::<ExtractLastRowOp>().unwrap();
                let input = read(&buffers, node.inputs[0]);
                let inner_size: usize = node.output_shapes[0][1..].iter().product();
                let offset = (op.seq_len - 1) * inner_size;
                let result = input.slice_view(offset, &node.output_shapes[0]);
                store(&mut buffers, node_id, 0, result);
            }

            // --- Cast ---
            "cast_to_f32" => {
                let input = read(&buffers, node.inputs[0]);
                let result = <CudaBackend as CastOps>::cast_to_f32(input)?;
                store(&mut buffers, node_id, 0, result);
            }

            "cast_from_f32" => {
                let op = node.op.as_any().downcast_ref::<CastFromF32Op>().unwrap();
                let input = read(&buffers, node.inputs[0]);
                let result = <CudaBackend as CastOps>::cast_from_f32(input, op.target)?;
                store(&mut buffers, node_id, 0, result);
            }

            // --- Unimplemented ---
            _ => panic!("CUDA executor: unimplemented op {op_name:?}"),
        }
    }

    // Collect output tensors.
    let outputs = output_nodes
        .iter()
        .map(|&id| take(&mut buffers, id, 0))
        .collect();

    Ok(outputs)
}
