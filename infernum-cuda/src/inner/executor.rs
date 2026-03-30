//! Graph executor for the CUDA backend.
//!
//! Walks an [`ExecutionPlan`] in topological order and dispatches each
//! [`Op`] to the existing CUDA kernels. Intermediate tensors are stored
//! in a `Vec<Option<CudaTensor>>` indexed by `NodeId`.
//!
//! Unlike the CPU executor which uses a flat byte arena, the GPU executor
//! stores tensors as individual `CudaTensor` values. The `BufferPool`
//! provides allocation reuse automatically.

use infernum::backend::{
    ArithOps, AttentionOps, BiasOps, CastOps, EmbedOps, GegluOps, MatmulExtOps, MatmulOps, NormOps,
    RopeInterleavedOps, RopeOps, SwigluOps, TensorOps,
};
use infernum::graph::WeightStore;
use infernum::tensor::Tensor;
use infernum::{ExecutionPlan, GraphNode, NodeId, Op, Result};

use crate::cuda::ops;
use crate::cuda::ops::LinearWeight;
use crate::cuda::CudaTensor;
use crate::CudaBackend;

/// Read a tensor from the node buffer.
///
/// # Panics
/// Panics if the node has no stored tensor.
fn read(buffers: &[Option<CudaTensor>], node_id: NodeId) -> &CudaTensor {
    buffers[node_id.index() as usize]
        .as_ref()
        .unwrap_or_else(|| panic!("node {node_id:?} has no stored tensor"))
}

/// Take (move) a tensor out of the node buffer. Used for the final outputs.
fn take(buffers: &mut [Option<CudaTensor>], node_id: NodeId) -> CudaTensor {
    buffers[node_id.index() as usize]
        .take()
        .unwrap_or_else(|| panic!("node {node_id:?} has no stored tensor"))
}

/// Store a tensor in the node buffer.
fn store(buffers: &mut [Option<CudaTensor>], node_id: NodeId, tensor: CudaTensor) {
    buffers[node_id.index() as usize] = Some(tensor);
}

/// Execute a computation graph on the CUDA backend.
///
/// Walks the `plan.schedule` in topological order, dispatching each [`Op`]
/// to the appropriate CUDA kernel. Intermediate tensors are stored as
/// `CudaTensor` values (leveraging `BufferPool` for allocation reuse).
///
/// # Arguments
///
/// * `plan` — Memory plan with schedule and buffer offsets.
/// * `nodes` — Graph nodes (indexed by `NodeId`).
/// * `weights` — Loaded model weights.
/// * `inputs` — External input tensors, one per `Op::Input` in schedule order.
/// * `output_nodes` — `NodeId`s of the graph outputs to collect.
///
/// # Errors
///
/// Returns an error if any op kernel fails.
#[allow(clippy::too_many_lines, clippy::missing_panics_doc)]
pub fn execute(
    plan: &ExecutionPlan,
    nodes: &[GraphNode],
    weights: &WeightStore<CudaTensor, LinearWeight>,
    inputs: &[CudaTensor],
    output_nodes: &[NodeId],
) -> Result<Vec<CudaTensor>> {
    let mut buffers: Vec<Option<CudaTensor>> = vec![None; nodes.len()];
    let mut input_idx: usize = 0;

    for &node_id in &plan.schedule {
        let node = &nodes[node_id.index() as usize];

        match &node.op {
            // --- Input ---
            Op::Input => {
                let tensor = inputs[input_idx].clone();
                input_idx += 1;
                store(&mut buffers, node_id, tensor);
            }

            // --- Embedding ---
            Op::EmbeddingGather { table } => {
                let token_ids = read(&buffers, node.inputs[0]);
                let table_w = weights.tensor_weight(*table);
                let seq_len = node.shape[0];
                let result = <CudaBackend as EmbedOps>::embedding_gather_tensor(
                    table_w, token_ids, seq_len,
                )?;
                store(&mut buffers, node_id, result);
            }

            // --- Norm ---
            Op::RmsNorm { weight, eps } => {
                let input = read(&buffers, node.inputs[0]);
                let w = weights.tensor_weight(*weight);
                let result = <CudaBackend as NormOps>::rms_norm(input, w, *eps)?;
                store(&mut buffers, node_id, result);
            }

            Op::AddRmsNorm { weight, eps } => {
                let residual = read(&buffers, node.inputs[0]);
                let delta = read(&buffers, node.inputs[1]);
                let w = weights.tensor_weight(*weight);
                let (updated, normed) =
                    <CudaBackend as NormOps>::add_rmsnorm(residual, delta, w, *eps)?;
                // Find the SecondOutput node.
                #[allow(clippy::cast_possible_truncation)]
                let second_id = nodes
                    .iter()
                    .enumerate()
                    .find(
                        |(_, n)| matches!(&n.op, Op::SecondOutput { source } if *source == node_id),
                    )
                    .map(|(i, _)| NodeId::from_index(i as u32))
                    .expect("AddRmsNorm must have a SecondOutput");
                store(&mut buffers, node_id, updated);
                store(&mut buffers, second_id, normed);
            }

            // --- Linear / Matmul ---
            Op::Linear { weight } | Op::LmHead { weight, .. } => {
                let input = read(&buffers, node.inputs[0]);
                let w = weights.linear_weight(*weight);
                let result = <CudaBackend as MatmulOps>::linear(input, w)?;
                store(&mut buffers, node_id, result);
            }

            Op::LinearPair { w1, w2 } => {
                let input = read(&buffers, node.inputs[0]);
                let lw1 = weights.linear_weight(*w1);
                let lw2 = weights.linear_weight(*w2);
                let (out1, out2) = <CudaBackend as MatmulOps>::linear_pair(input, lw1, lw2)?;
                store(&mut buffers, node_id, out1);
                let second_id = NodeId::from_index(node_id.index() + 1);
                store(&mut buffers, second_id, out2);
            }

            Op::LinearTriple { w1, w2, w3 } => {
                let input = read(&buffers, node.inputs[0]);
                let lw1 = weights.linear_weight(*w1);
                let lw2 = weights.linear_weight(*w2);
                let lw3 = weights.linear_weight(*w3);
                let (out1, out2, out3) =
                    <CudaBackend as MatmulOps>::linear_triple(input, lw1, lw2, lw3)?;
                store(&mut buffers, node_id, out1);
                let second_id = NodeId::from_index(node_id.index() + 1);
                store(&mut buffers, second_id, out2);
                let third_id = NodeId::from_index(node_id.index() + 2);
                store(&mut buffers, third_id, out3);
            }

            Op::Matmul => {
                let a = read(&buffers, node.inputs[0]);
                let b = read(&buffers, node.inputs[1]);
                let result = <CudaBackend as MatmulOps>::matmul(a, b)?;
                store(&mut buffers, node_id, result);
            }

            Op::MatmulBf16F32 => {
                let a = read(&buffers, node.inputs[0]);
                let b = read(&buffers, node.inputs[1]);
                let result = <CudaBackend as MatmulExtOps>::matmul_bf16_f32(a, b)?;
                store(&mut buffers, node_id, result);
            }

            // --- Activations ---
            Op::Swiglu => {
                let gate = read(&buffers, node.inputs[0]);
                let up = read(&buffers, node.inputs[1]);
                let result = <CudaBackend as SwigluOps>::swiglu(gate, up)?;
                store(&mut buffers, node_id, result);
            }

            Op::Geglu => {
                let gate = read(&buffers, node.inputs[0]);
                let up = read(&buffers, node.inputs[1]);
                let result = <CudaBackend as GegluOps>::geglu(gate, up)?;
                store(&mut buffers, node_id, result);
            }

            Op::Silu => {
                let input = read(&buffers, node.inputs[0]);
                let result = ops::silu(input)?;
                store(&mut buffers, node_id, result);
            }

            Op::Mul => {
                let a = read(&buffers, node.inputs[0]);
                let b = read(&buffers, node.inputs[1]);
                let result = <CudaBackend as ArithOps>::mul(a, b)?;
                store(&mut buffers, node_id, result);
            }

            // --- Arithmetic ---
            Op::Add => {
                let a = read(&buffers, node.inputs[0]);
                let b = read(&buffers, node.inputs[1]);
                let result = <CudaBackend as ArithOps>::add(a, b)?;
                store(&mut buffers, node_id, result);
            }

            Op::AddInplace => {
                let a = read(&buffers, node.inputs[0]);
                let b = read(&buffers, node.inputs[1]);
                // Clone to get a mutable copy, then add in-place.
                let mut result = a.clone();
                <CudaBackend as ArithOps>::add_inplace(&mut result, b)?;
                store(&mut buffers, node_id, result);
            }

            Op::Scale { factor } => {
                let input = read(&buffers, node.inputs[0]);
                let mut result = input.clone();
                <CudaBackend as ArithOps>::scale_inplace(&mut result, *factor)?;
                store(&mut buffers, node_id, result);
            }

            Op::BiasAdd { bias } => {
                let input = read(&buffers, node.inputs[0]);
                let bias_w = weights.tensor_weight(*bias);
                let mut result = input.clone();
                <CudaBackend as BiasOps>::bias_add_inplace(&mut result, bias_w)?;
                store(&mut buffers, node_id, result);
            }

            // --- RoPE ---
            Op::Rope { offset } => {
                let input = read(&buffers, node.inputs[0]);
                let cos_cache = read(&buffers, node.inputs[1]);
                let sin_cache = read(&buffers, node.inputs[2]);
                let result =
                    <CudaBackend as RopeOps>::apply_rope(input, cos_cache, sin_cache, *offset)?;
                store(&mut buffers, node_id, result);
            }

            Op::RopeBatched { batch_size } => {
                let input = read(&buffers, node.inputs[0]);
                let cos_cache = read(&buffers, node.inputs[1]);
                let sin_cache = read(&buffers, node.inputs[2]);
                let positions = read(&buffers, node.inputs[3]);
                let result = <CudaBackend as RopeOps>::apply_rope_batched(
                    input,
                    cos_cache,
                    sin_cache,
                    positions,
                    *batch_size,
                )?;
                store(&mut buffers, node_id, result);
            }

            Op::RopeInterleaved { offset } => {
                let input = read(&buffers, node.inputs[0]);
                let cos_cache = read(&buffers, node.inputs[1]);
                let sin_cache = read(&buffers, node.inputs[2]);
                let result = <CudaBackend as RopeInterleavedOps>::apply_rope_interleaved(
                    input, cos_cache, sin_cache, *offset,
                )?;
                store(&mut buffers, node_id, result);
            }

            // --- Attention ---
            Op::FusedAttentionPrefill {
                offset,
                scale,
                softcap,
                sliding_window,
            } => {
                let q = read(&buffers, node.inputs[0]);
                let k = read(&buffers, node.inputs[1]);
                let v = read(&buffers, node.inputs[2]);
                let result = <CudaBackend as AttentionOps>::fused_attention_prefill(
                    q,
                    k,
                    v,
                    *offset,
                    *scale,
                    *softcap,
                    *sliding_window,
                )?;
                store(&mut buffers, node_id, result);
            }

            Op::FusedAttentionDecode { softcap } => {
                let q = read(&buffers, node.inputs[0]);
                let k = read(&buffers, node.inputs[1]);
                let v = read(&buffers, node.inputs[2]);
                let result = <CudaBackend as AttentionOps>::fused_attention_decode(
                    q, k, v, None, *softcap, None,
                )?;
                store(&mut buffers, node_id, result);
            }

            // --- Shape / Data movement ---
            Op::Reshape { shape } => {
                let input = read(&buffers, node.inputs[0]);
                let result = input.reshape(shape);
                store(&mut buffers, node_id, result);
            }

            Op::SliceView { offset, shape } => {
                let input = read(&buffers, node.inputs[0]);
                let result = input.slice_view(*offset, shape);
                store(&mut buffers, node_id, result);
            }

            Op::Transpose2d => {
                let input = read(&buffers, node.inputs[0]);
                let result = <CudaBackend as TensorOps>::transpose_2d(input)?;
                store(&mut buffers, node_id, result);
            }

            Op::SplitInnerDim { left_size } => {
                let input = read(&buffers, node.inputs[0]);
                let inner = *input.shape().last().expect("non-empty shape");
                let right_size = inner - left_size;
                let (left, right) =
                    <CudaBackend as TensorOps>::split_inner_dim(input, *left_size, right_size)?;
                store(&mut buffers, node_id, left);
                let second_id = NodeId::from_index(node_id.index() + 1);
                store(&mut buffers, second_id, right);
            }

            Op::ConcatInnerDim => {
                let a = read(&buffers, node.inputs[0]);
                let b = read(&buffers, node.inputs[1]);
                let result = <CudaBackend as TensorOps>::concat_inner_dim(a, b)?;
                store(&mut buffers, node_id, result);
            }

            Op::ConcatSeq => {
                let a = read(&buffers, node.inputs[0]);
                let b = read(&buffers, node.inputs[1]);
                let result = <CudaBackend as TensorOps>::concat_rows(&[a.clone(), b.clone()])?;
                store(&mut buffers, node_id, result);
            }

            Op::RepeatKv { num_repeats } => {
                let input = read(&buffers, node.inputs[0]);
                let result = <CudaBackend as TensorOps>::repeat_kv(input, *num_repeats)?;
                store(&mut buffers, node_id, result);
            }

            Op::ExtractLastRow { seq_len } => {
                let input = read(&buffers, node.inputs[0]);
                let inner_size: usize = node.shape[1..].iter().product();
                let offset = (seq_len - 1) * inner_size;
                let result = input.slice_view(offset, &node.shape);
                store(&mut buffers, node_id, result);
            }

            // --- Cast ---
            Op::CastToF32 => {
                let input = read(&buffers, node.inputs[0]);
                let result = <CudaBackend as CastOps>::cast_to_f32(input)?;
                store(&mut buffers, node_id, result);
            }

            Op::CastFromF32 { target } => {
                let input = read(&buffers, node.inputs[0]);
                let result = <CudaBackend as CastOps>::cast_from_f32(input, *target)?;
                store(&mut buffers, node_id, result);
            }

            // --- Multi-output secondary (already handled by parent op) ---
            Op::SecondOutput { .. } => {}

            // --- Unimplemented ---
            op => panic!("CUDA executor: unimplemented op {op:?}"),
        }
    }

    // Collect output tensors.
    let outputs = output_nodes
        .iter()
        .map(|&id| take(&mut buffers, id))
        .collect();

    Ok(outputs)
}
