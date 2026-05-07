//! Graph executor for the CUDA backend.
//!
//! Walks an [`ExecutionPlan`] in topological order and dispatches each
//! operation to the appropriate CUDA kernel. Intermediate tensors are stored
//! in a `Vec<Vec<Option<CudaTensor>>>` indexed by `(NodeId, output_index)`.
//! Unlike the CPU executor, which uses a flat byte arena, the GPU executor
//! stores tensors as individual [`CudaTensor`] values. The [`BufferPool`]
//! provides allocation reuse automatically.
//!
//! ## Dispatch strategy
//!
//! Both `execute` and `execute_indirect` use a `match op_name { ... }` block
//! for all built-in ops. This keeps dispatch overhead to zero — no trait-object
//! vtable calls for the hot path. Rope-fusion look-ahead is handled inline in
//! that block.
//!
//! Unknown or custom ops fall through to the `_ =>` arm, which constructs an
//! [`ExecuteContext`] and calls `node.op.execute(ctx)`. This open-dispatch
//! fallback lets external crates add new ops without modifying infernum.

use infernum::backend::{
    ArithOps, AttentionOps, BiasOps, CastOps, EmbedOps, GegluOps, MatmulExtOps, MatmulOps,
    MlaAttentionOps, MoeOps, MoeSigmoidOps, NormOps, RopeInterleavedOps, RopeOps, SwigluOps,
    TensorOps,
};
use infernum::graph::builtin_ops::{
    AddRmsNormOp, AppendKvIndirectOp, AppendPagedBatchedOp, ArgmaxLastOp, BiasAddOp, CastFromF32Op,
    EmbeddingGatherIndirectOp, EmbeddingGatherOp, ExtractLastRowOp, FusedAttentionDecodeIndirectOp,
    FusedAttentionDecodeOp, FusedAttentionPrefillOp, LinearOp, LinearPairOp, LinearTripleOp,
    LmHeadOp, LogitSoftcapOp, MlaAttentionOp, MoeDispatchSigmoidOp, MoeDispatchSoftmaxOp,
    PagedAttentionDecodeOp, RepeatKvOp, ReshapeOp, RmsNormOp, RmsNormQkOp, RopeBatchedOp,
    RopeIndirectOp, RopeInterleavedOp, RopeOp, ScaleOp, SliceViewOp, SplitInnerDimOp,
};
use infernum::graph::{GraphNode, OutputRef, WeightStore};
use infernum::tensor::Tensor;
use infernum::{DType, ExecutionPlan, NodeId, Result};

use crate::cuda::ops;
use crate::cuda::ops::LinearWeight;
use crate::cuda::{CudaTensor, KvCache, SeqPosition};
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
/// Walks the `plan.schedule` in topological order. Built-in ops are dispatched
/// via a `match op_name { ... }` block (zero overhead, no trait-object call).
/// Rope-fusion look-ahead is handled inline in that block. Unknown or custom
/// ops fall through to the `_ =>` arm, which constructs an [`ExecuteContext`]
/// and calls `node.op.execute(ctx)` — the open-dispatch path for ops added by
/// external crates. Intermediate tensors are stored as `CudaTensor` values
/// (leveraging `BufferPool` for allocation reuse).
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
#[allow(
    clippy::too_many_lines,
    clippy::missing_panics_doc,
    clippy::too_many_arguments,
    clippy::similar_names
)]
pub fn execute(
    plan: &ExecutionPlan,
    nodes: &[GraphNode<CudaBackend>],
    weights: &WeightStore<CudaTensor, LinearWeight>,
    inputs: &[CudaTensor],
    output_nodes: &[NodeId],
    mut mla_kv_cache: Option<&mut Vec<Vec<CudaTensor>>>,
    paged_kv_cache: Option<&mut crate::cuda::PagedKvCache>,
    mla_seq_pos: usize,
) -> Result<Vec<CudaTensor>> {
    #[allow(clippy::needless_pass_by_ref_mut)]
    let mut paged_kv_cache = paged_kv_cache;
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
                // Use the runtime input length, not the statically declared
                // output shape. Graphs that use 0 as a dynamic seq_len
                // placeholder (e.g. Gemma's prefill graph) would produce
                // seq_len=0 here, causing CUDA_ERROR_INVALID_VALUE on launch.
                let seq_len = token_ids.shape()[0];
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
                // Cast bias to match input dtype when they differ (e.g. F32 bias
                // loaded from a Qwen2.5 checkpoint applied to a BF16 activation).
                // The CUDA kernel requires both operands to share the same dtype.
                let bias_cast;
                let bias_ref = if bias_w.dtype() == result.dtype() {
                    bias_w
                } else {
                    bias_cast = ops::cast_from_f32(bias_w, result.dtype())?;
                    &bias_cast
                };
                <CudaBackend as BiasOps>::bias_add_inplace(&mut result, bias_ref)?;
                store(&mut buffers, node_id, 0, result);
            }

            // --- RoPE ---
            "rope" => {
                let op = node.op.as_any().downcast_ref::<RopeOp>().unwrap();
                let input = read(&buffers, node.inputs[0]);
                let cos_cache = read(&buffers, node.inputs[1]);
                let sin_cache = read(&buffers, node.inputs[2]);
                // The rope kernel dispatches on input.dtype() and expects cos/sin
                // to share that dtype. Precomputed caches are always F32, so cast
                // them when the activation dtype differs (e.g. BF16 models).
                let cos_cast;
                let cos_ref = if cos_cache.dtype() == input.dtype() {
                    cos_cache
                } else {
                    cos_cast = ops::cast_from_f32(cos_cache, input.dtype())?;
                    &cos_cast
                };
                let sin_cast;
                let sin_ref = if sin_cache.dtype() == input.dtype() {
                    sin_cache
                } else {
                    sin_cast = ops::cast_from_f32(sin_cache, input.dtype())?;
                    &sin_cast
                };
                let result =
                    <CudaBackend as RopeOps>::apply_rope(input, cos_ref, sin_ref, op.offset)?;
                store(&mut buffers, node_id, 0, result);
            }

            "rope_batched" => {
                let op = node.op.as_any().downcast_ref::<RopeBatchedOp>().unwrap();
                let input = read(&buffers, node.inputs[0]);
                let cos_cache = read(&buffers, node.inputs[1]);
                let sin_cache = read(&buffers, node.inputs[2]);
                let positions = read(&buffers, node.inputs[3]);
                // Cast cos/sin to match activation dtype (same reason as "rope" arm).
                let cos_cast;
                let cos_ref = if cos_cache.dtype() == input.dtype() {
                    cos_cache
                } else {
                    cos_cast = ops::cast_from_f32(cos_cache, input.dtype())?;
                    &cos_cast
                };
                let sin_cast;
                let sin_ref = if sin_cache.dtype() == input.dtype() {
                    sin_cache
                } else {
                    sin_cast = ops::cast_from_f32(sin_cache, input.dtype())?;
                    &sin_cast
                };
                let result = <CudaBackend as RopeOps>::apply_rope_batched(
                    input,
                    cos_ref,
                    sin_ref,
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

            // --- Per-head QK RMSNorm (Qwen3, Gemma 3) ---
            "rms_norm_qk" => {
                let op = node.op.as_any().downcast_ref::<RmsNormQkOp>().unwrap();
                let q = read(&buffers, node.inputs[0]);
                let k = read(&buffers, node.inputs[1]);
                let q_w = weights.tensor_weight(op.q_weight);
                let k_w = weights.tensor_weight(op.k_weight);
                // The CUDA rms_norm kernel treats the outermost dims as a batch,
                // so [seq, num_heads, head_dim] works directly — each row of
                // head_dim is normalised independently.
                //
                // QK-norm weights are registered as F32 (checkpoint dtype), but
                // the activation `q`/`k` may be BF16. The rmsnorm_bf16 kernel
                // requires all tensors to share the same dtype, so cast if needed
                // (mirrors the dtype-cast pattern used in the `bias_add` arm).
                let q_w_cast;
                let q_w_ref = if q_w.dtype() == q.dtype() {
                    q_w
                } else {
                    q_w_cast = ops::cast_from_f32(q_w, q.dtype())?;
                    &q_w_cast
                };
                let k_w_cast;
                let k_w_ref = if k_w.dtype() == k.dtype() {
                    k_w
                } else {
                    k_w_cast = ops::cast_from_f32(k_w, k.dtype())?;
                    &k_w_cast
                };
                let q_normed = <CudaBackend as NormOps>::rms_norm(q, q_w_ref, op.eps)?;
                let k_normed = <CudaBackend as NormOps>::rms_norm(k, k_w_ref, op.eps)?;
                store(&mut buffers, node_id, 0, q_normed);
                store(&mut buffers, node_id, 1, k_normed);
            }

            // --- MoE softmax dispatch (Mixtral, Qwen3-MoE) ---
            "moe_dispatch_softmax" => {
                let op = node
                    .op
                    .as_any()
                    .downcast_ref::<MoeDispatchSoftmaxOp>()
                    .unwrap();
                let input = read(&buffers, node.inputs[0]).clone();
                let gate_t = weights.tensor_weight(op.gate);
                // The checkpoint stores gate as [num_experts, hidden]; moe_route
                // computes `hidden @ gate` so it needs [hidden, num_experts].
                let gate_t = ops::transpose_2d(gate_t)?;
                let num_experts = op.experts.len();
                let num_experts_per_tok = op.num_experts_per_tok;
                let norm_topk = op.norm_topk;
                let expert_ids = op.experts.clone();
                let result = <CudaBackend as MoeOps>::moe_forward_softmax(
                    &input,
                    &gate_t,
                    num_experts,
                    num_experts_per_tok,
                    norm_topk,
                    |expert_idx, expert_input| {
                        let eids = &expert_ids[expert_idx];
                        let gate_w = weights.linear_weight(eids.gate_proj);
                        let up_w = weights.linear_weight(eids.up_proj);
                        let down_w = weights.linear_weight(eids.down_proj);
                        let gate_out = <CudaBackend as MatmulOps>::linear(expert_input, gate_w)?;
                        let up_out = <CudaBackend as MatmulOps>::linear(expert_input, up_w)?;
                        let activated = <CudaBackend as SwigluOps>::swiglu(&gate_out, &up_out)?;
                        <CudaBackend as MatmulOps>::linear(&activated, down_w)
                    },
                )?;
                store(&mut buffers, node_id, 0, result);
            }

            // --- MoE sigmoid dispatch with bias correction (DeepSeek) ---
            "moe_dispatch_sigmoid" => {
                let op = node
                    .op
                    .as_any()
                    .downcast_ref::<MoeDispatchSigmoidOp>()
                    .unwrap();
                let input = read(&buffers, node.inputs[0]).clone();
                let gate_t = weights.tensor_weight(op.gate);
                // The checkpoint stores gate as [num_experts, hidden]; moe_route_sigmoid
                // computes `hidden @ gate` so it needs [hidden, num_experts].
                let gate_t = ops::transpose_2d(gate_t)?;
                let bias_data: Vec<f32> = if let Some(bias_id) = op.bias {
                    weights.tensor_weight(bias_id).to_vec::<f32>()?
                } else {
                    vec![0.0f32; op.experts.len()]
                };
                let num_experts = op.experts.len();
                let num_experts_per_tok = op.num_experts_per_tok;
                let n_group = op.n_group;
                let topk_group = op.topk_group;
                let routed_scaling_factor = op.routed_scaling_factor;
                let expert_ids = op.experts.clone();
                let shared_ids = op.shared_expert.clone();
                let mut result = <CudaBackend as MoeSigmoidOps>::moe_forward_sigmoid(
                    &input,
                    &gate_t,
                    &bias_data,
                    num_experts,
                    num_experts_per_tok,
                    n_group,
                    topk_group,
                    false, // DeepSeek normalises inside the kernel
                    routed_scaling_factor,
                    |expert_idx, expert_input| {
                        let eids = &expert_ids[expert_idx];
                        let gate_w = weights.linear_weight(eids.gate_proj);
                        let up_w = weights.linear_weight(eids.up_proj);
                        let down_w = weights.linear_weight(eids.down_proj);
                        let gate_out = <CudaBackend as MatmulOps>::linear(expert_input, gate_w)?;
                        let up_out = <CudaBackend as MatmulOps>::linear(expert_input, up_w)?;
                        let activated = <CudaBackend as SwigluOps>::swiglu(&gate_out, &up_out)?;
                        <CudaBackend as MatmulOps>::linear(&activated, down_w)
                    },
                )?;
                // Add shared expert output if present.
                if let Some(sids) = shared_ids {
                    let sg = weights.linear_weight(sids.gate_proj);
                    let su = weights.linear_weight(sids.up_proj);
                    let sd = weights.linear_weight(sids.down_proj);
                    let sgate = <CudaBackend as MatmulOps>::linear(&input, sg)?;
                    let sup_out = <CudaBackend as MatmulOps>::linear(&input, su)?;
                    let sact = <CudaBackend as SwigluOps>::swiglu(&sgate, &sup_out)?;
                    let shared_out = <CudaBackend as MatmulOps>::linear(&sact, sd)?;
                    <CudaBackend as ArithOps>::add_inplace(&mut result, &shared_out)?;
                }
                store(&mut buffers, node_id, 0, result);
            }

            // --- Logit soft-cap: tanh(x / cap) * cap (Gemma 2 final logit) ---
            "logit_softcap" => {
                // TODO: dedicated CUDA tanh kernel for better throughput.
                // For now: cast to F32 → download → CPU loop → upload. Logits
                // are small ([seq_len, vocab_size]) so this is not on the hot path.
                let op = node.op.as_any().downcast_ref::<LogitSoftcapOp>().unwrap();
                let input = read(&buffers, node.inputs[0]);
                let ctx = input.context().clone();
                // Use the runtime shape from the input, not the statically declared
                // output shape — graphs with dynamic seq_len (0 placeholder) would
                // produce the wrong shape here.
                let shape = input.shape().to_vec();
                // Cast to F32 if needed (input may be BF16/F16 for BF16 models).
                let f32_input;
                let input_f32 = if input.dtype() == DType::F32 {
                    input
                } else {
                    f32_input = ops::cast_to_f32(input)?;
                    &f32_input
                };
                let cap = op.cap;
                let host: Vec<f32> = input_f32.to_vec::<f32>()?;
                let softcapped: Vec<f32> = host.iter().map(|&x| (x / cap).tanh() * cap).collect();
                let result = CudaTensor::from_slice::<f32>(&ctx, &shape, &softcapped)?;
                store(&mut buffers, node_id, 0, result);
            }

            // --- Paged KV cache: batched append (side-effect, no output) ---
            "append_paged_batched" => {
                let op = node
                    .op
                    .as_any()
                    .downcast_ref::<AppendPagedBatchedOp>()
                    .unwrap();
                let k = read(&buffers, node.inputs[0]).clone();
                let v = read(&buffers, node.inputs[1]).clone();
                let block_tables = read(&buffers, node.inputs[2]).clone();
                let positions = read(&buffers, node.inputs[3]).clone();
                let batch_size = k.shape()[0];
                let max_blocks_per_seq = block_tables.shape()[1];
                let paged = paged_kv_cache
                    .as_deref_mut()
                    .expect("paged KV cache required for append_paged_batched");
                paged.append_paged_batched_tensor(
                    op.layer_idx,
                    &k,
                    &v,
                    &block_tables,
                    &positions,
                    batch_size,
                    max_blocks_per_seq,
                )?;
                // Side-effect op: no output tensor to store.
                // We store a dummy unit-value so the buffer slot is non-None
                // (the op has 0 declared outputs but add_node gives it 1 slot
                // via `.max(1)` in the buffer initialiser).
            }

            // --- Paged attention decode ---
            "paged_attention_decode" => {
                let op = node
                    .op
                    .as_any()
                    .downcast_ref::<PagedAttentionDecodeOp>()
                    .unwrap();
                let q = read(&buffers, node.inputs[0]).clone();
                let block_tables = read(&buffers, node.inputs[1]);
                let seq_lens = read(&buffers, node.inputs[2]);
                let paged = paged_kv_cache
                    .as_deref_mut()
                    .expect("paged KV cache required for paged_attention_decode");
                let (k_pool, v_pool) = paged.get_pools(op.layer_idx);
                let block_size = paged.block_size();
                let max_blocks_per_seq = block_tables.shape()[1];
                let seq_lens_vec = seq_lens.to_vec::<u32>()?;
                let max_seq_len = seq_lens_vec
                    .iter()
                    .copied()
                    .map(|x| x as usize)
                    .max()
                    .unwrap_or(1);
                let attn_out = ops::paged_attention_decode_from_tensor(
                    q.context(),
                    &q,
                    k_pool,
                    v_pool,
                    block_tables,
                    seq_lens,
                    block_size,
                    max_blocks_per_seq,
                    max_seq_len,
                    None, // scale: auto (1/sqrt(head_dim))
                    op.softcap,
                    op.sliding_window,
                )?;
                store(&mut buffers, node_id, 0, attn_out);
            }

            // --- MLA Attention (DeepSeek V3/R1) ---
            "mla_attention" => {
                let op = node.op.as_any().downcast_ref::<MlaAttentionOp>().unwrap();
                let hidden = read(&buffers, node.inputs[0]).clone();
                let q_a_proj = weights.linear_weight(op.q_a_proj);
                let q_a_layernorm = weights.tensor_weight(op.q_a_layernorm);
                let q_b_proj = weights.linear_weight(op.q_b_proj);
                let kv_a_proj_with_mqa = weights.linear_weight(op.kv_a_proj_with_mqa);
                let kv_a_layernorm = weights.tensor_weight(op.kv_a_layernorm);
                let kv_b_proj_k = weights.linear_weight(op.kv_b_proj_k);
                let kv_b_proj_v = weights.linear_weight(op.kv_b_proj_v);
                let kv_b_proj_k_t = weights.linear_weight(op.kv_b_proj_k_t);
                let o_proj = weights.linear_weight(op.o_proj);
                let layer_kv = mla_kv_cache
                    .as_deref_mut()
                    .unwrap_or_else(|| {
                        panic!(
                            "CUDA executor: mla_attention op requires mla_kv_cache to be provided"
                        )
                    })
                    .get_mut(op.layer_idx)
                    .unwrap_or_else(|| {
                        panic!(
                            "CUDA executor: mla_kv_cache has no entry for layer {}",
                            op.layer_idx
                        )
                    });
                let result = <CudaBackend as MlaAttentionOps>::mla_attention(
                    &hidden,
                    q_a_proj,
                    q_a_layernorm,
                    q_b_proj,
                    kv_a_proj_with_mqa,
                    kv_a_layernorm,
                    kv_b_proj_k,
                    kv_b_proj_v,
                    kv_b_proj_k_t,
                    o_proj,
                    layer_kv,
                    mla_seq_pos,
                    op.num_heads,
                    op.qk_nope_head_dim,
                    op.qk_rope_head_dim,
                    op.v_head_dim,
                    op.kv_lora_rank,
                    op.rms_norm_eps,
                    op.attn_scale,
                )?;
                store(&mut buffers, node_id, 0, result);
            }

            // --- Open dispatch: custom / unknown ops self-execute via OpNode::execute ---
            _ => {
                use crate::inner::execute_context::CudaExecutorState;
                use infernum::graph::execute_context::ExecuteContext;
                // Get the CUDA context from the first input tensor — every graph
                // has at least one input, so this is always valid.
                let device = inputs[0].context().clone();
                let mut state = CudaExecutorState { buffers };
                let mut input_idx_local = input_idx;
                {
                    let mut exec_ctx = ExecuteContext {
                        state: &mut state,
                        plan,
                        nodes,
                        weights,
                        device: &device,
                        kv_cache: None::<
                            &mut dyn infernum::graph::execute_context::KvCacheAccess<CudaBackend>,
                        >,
                        input_tensors: inputs,
                        input_idx: &mut input_idx_local,
                    };
                    node.op.execute(&mut exec_ctx, node_id, &node.inputs)?;
                }
                buffers = state.buffers;
                input_idx = input_idx_local;
            }
        }
    }

    // Collect output tensors.
    let outputs = output_nodes
        .iter()
        .map(|&id| take(&mut buffers, id, 0))
        .collect();

    Ok(outputs)
}

/// Execute an indirect decode graph on the CUDA backend.
///
/// Identical to [`execute`] for standard ops (matmul, norm, `RoPE`, etc.), but
/// handles the four CUDA-graph-compatible indirect op names by calling the
/// corresponding indirect kernel wrappers instead of their standard counterparts:
///
/// - `"embedding_gather_indirect"` → reads token ID from `seq_pos.device()`
/// - `"rope_indirect"` → reads position from `seq_pos.device()`
/// - `"append_kv_indirect"` → writes into `kv_cache` at the current position
/// - `"fused_attention_decode_indirect"` → reads total length from `kv_cache.current_total_len()`
///
/// The `kv_cache` full-capacity buffers are injected as the `inputs` slice at
/// positions corresponding to the `input` graph nodes (one K buffer and one V
/// buffer per layer, in layer order). Their stable GPU addresses mean the
/// captured `cudaGraphExec_t` can be replayed without re-capture.
///
/// Like [`execute`], the `_ =>` fallback arm uses open dispatch: it constructs
/// an [`ExecuteContext`] and calls `node.op.execute(ctx)`, allowing external
/// crates to add new indirect-compatible ops without modifying infernum.
///
/// # Errors
/// Returns an error if any kernel launch fails.
#[allow(
    clippy::too_many_lines,
    clippy::missing_panics_doc,
    clippy::too_many_arguments
)]
pub fn execute_indirect(
    plan: &ExecutionPlan,
    nodes: &[GraphNode<CudaBackend>],
    weights: &WeightStore<CudaTensor, LinearWeight>,
    inputs: &[CudaTensor],
    output_nodes: &[NodeId],
    ctx: &crate::cuda::CudaContext,
    seq_pos: &SeqPosition,
    kv_cache: &mut KvCache,
) -> Result<Vec<CudaTensor>> {
    use crate::cuda::ops::{
        apply_rope_indirect, apply_rope_interleaved_indirect, embedding_gather_from_device,
        fused_attention_decode_indirect,
    };

    let mut buffers: Vec<Vec<Option<CudaTensor>>> = nodes
        .iter()
        .map(|node| {
            let num_outputs = node.output_shapes.len().max(1);
            vec![None; num_outputs]
        })
        .collect();
    let mut input_idx: usize = 0;
    // Per-layer scratch for K/V appends.  The topo sort may schedule K-append
    // and V-append for the same layer in either order.  Whichever arrives first
    // stores its tensor here; when both are present, `append_indirect` is called.
    let num_layers = kv_cache.num_layers();
    let mut pending_k: Vec<Option<CudaTensor>> = vec![None; num_layers];
    let mut pending_v: Vec<Option<CudaTensor>> = vec![None; num_layers];

    for &node_id in &plan.schedule {
        let node = &nodes[node_id.index() as usize];
        let op_name = node.op.name();

        match op_name {
            // --- Indirect embedding: reads token ID from SeqPosition device pointer ---
            "embedding_gather_indirect" => {
                let op = node
                    .op
                    .as_any()
                    .downcast_ref::<EmbeddingGatherIndirectOp>()
                    .unwrap();
                let table = weights.tensor_weight(op.table);
                let ctx = table.context();
                let result = embedding_gather_from_device(ctx, table, seq_pos.device(), 1)?;
                store(&mut buffers, node_id, 0, result);
            }

            // --- Indirect RoPE: reads position from SeqPosition device pointer ---
            "rope_indirect" => {
                let op = node.op.as_any().downcast_ref::<RopeIndirectOp>().unwrap();
                let input = read(&buffers, node.inputs[0]);
                let cos_cache = weights.tensor_weight(op.cos_cache);
                let sin_cache = weights.tensor_weight(op.sin_cache);
                let result = if op.interleaved {
                    apply_rope_interleaved_indirect(input, cos_cache, sin_cache, seq_pos)?
                } else {
                    apply_rope_indirect(input, cos_cache, sin_cache, seq_pos)?
                };
                store(&mut buffers, node_id, 0, result);
            }

            // --- Indirect KV append: writes to KvCache at position from SeqPosition ---
            "append_kv_indirect" => {
                let op = node
                    .op
                    .as_any()
                    .downcast_ref::<AppendKvIndirectOp>()
                    .unwrap();
                let new_tensor = read(&buffers, node.inputs[0]).clone();
                store(&mut buffers, node_id, 0, new_tensor.clone());
                // K and V appends for a layer may be scheduled in either order by
                // the topological sort.  Stash whichever arrives first; call
                // `append_indirect` once both are present.
                let layer = op.layer_idx;
                if op.is_key {
                    pending_k[layer] = Some(new_tensor);
                } else {
                    pending_v[layer] = Some(new_tensor);
                }
                if let (Some(k_ten), Some(v_ten)) =
                    (pending_k[layer].as_ref(), pending_v[layer].as_ref())
                {
                    kv_cache.append_indirect(layer, k_ten, v_ten)?;
                    pending_k[layer] = None;
                    pending_v[layer] = None;
                }
            }

            // --- Indirect attention: K/V and total_len from out-of-band KvCache ---
            "fused_attention_decode_indirect" => {
                let op = node
                    .op
                    .as_any()
                    .downcast_ref::<FusedAttentionDecodeIndirectOp>()
                    .unwrap();
                let q = read(&buffers, node.inputs[0]).clone();
                // K/V full buffers have stable GPU addresses — fetched from the KvCache
                // by layer_idx (not from the graph input buffer table).
                let (k_full, v_full) = kv_cache.full_buffers(op.layer_idx);
                let total_len = kv_cache.current_total_len();
                let max_seq_len = kv_cache.max_seq_len();
                let result = fused_attention_decode_indirect(
                    &q,
                    k_full,
                    v_full,
                    total_len,
                    max_seq_len,
                    Some(op.scale),
                    op.softcap,
                    op.sliding_window,
                )?;
                store(&mut buffers, node_id, 0, result);
            }

            // --- LM head (panics in OpNode::execute, needs explicit dispatch) ---
            "lm_head" => {
                let op = node.op.as_any().downcast_ref::<LmHeadOp>().unwrap();
                let input = read(&buffers, node.inputs[0]).clone();
                let w = weights.linear_weight(op.weight);
                let result = <CudaBackend as MatmulOps>::linear(&input, w)?;
                store(&mut buffers, node_id, 0, result);
            }

            // --- Device-side argmax (avoids D→H sync inside the graph) ---
            "argmax_last" => {
                let _op = node.op.as_any().downcast_ref::<ArgmaxLastOp>().unwrap();
                let input = read(&buffers, node.inputs[0]).clone();
                let result = ops::argmax_last_tensor(&input)?;
                store(&mut buffers, node_id, 0, result);
            }

            // --- Zero-copy view ops (cannot use OpNode::execute) ---
            "reshape" => {
                let op = node.op.as_any().downcast_ref::<ReshapeOp>().unwrap();
                let input = read(&buffers, node.inputs[0]).clone();
                let result = input.reshape(&op.shape);
                store(&mut buffers, node_id, 0, result);
            }

            "slice_view" => {
                let op = node.op.as_any().downcast_ref::<SliceViewOp>().unwrap();
                let input = read(&buffers, node.inputs[0]).clone();
                let result = input.slice_view(op.offset, &op.shape);
                store(&mut buffers, node_id, 0, result);
            }

            // --- Standard input injection ---
            "input" => {
                let tensor = inputs[input_idx].clone();
                input_idx += 1;
                store(&mut buffers, node_id, 0, tensor);
            }

            // --- Open dispatch: custom / unknown ops self-execute via OpNode::execute ---
            _ => {
                use crate::inner::execute_context::CudaExecutorState;
                use infernum::graph::execute_context::ExecuteContext;
                let mut state = CudaExecutorState { buffers };
                let mut input_idx_local = input_idx;
                {
                    let mut exec_ctx = ExecuteContext {
                        state: &mut state,
                        plan,
                        nodes,
                        weights,
                        device: ctx,
                        kv_cache: None::<
                            &mut dyn infernum::graph::execute_context::KvCacheAccess<CudaBackend>,
                        >,
                        input_tensors: inputs,
                        input_idx: &mut input_idx_local,
                    };
                    node.op.execute(&mut exec_ctx, node_id, &node.inputs)?;
                }
                buffers = state.buffers;
                input_idx = input_idx_local;
            }
        }
    }

    let outputs = output_nodes
        .iter()
        .map(|&id| take(&mut buffers, id, 0))
        .collect();

    Ok(outputs)
}
