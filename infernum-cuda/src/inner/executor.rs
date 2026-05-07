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

use infernum::backend::MlaAttentionOps;
use infernum::graph::builtin_ops::{
    AppendKvIndirectOp, AppendPagedBatchedOp, ArgmaxLastOp, EmbeddingGatherIndirectOp,
    FusedAttentionDecodeIndirectOp, MlaAttentionOp, PagedAttentionDecodeOp, RopeIndirectOp,
};
use infernum::graph::{GraphNode, OutputRef, WeightStore};
use infernum::tensor::Tensor;
use infernum::{ExecutionPlan, NodeId, Result};

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
/// Unknown or custom
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

            // --- Device-side argmax (avoids D→H sync inside the graph) ---
            "argmax_last" => {
                let _op = node.op.as_any().downcast_ref::<ArgmaxLastOp>().unwrap();
                let input = read(&buffers, node.inputs[0]).clone();
                let result = ops::argmax_last_tensor(&input)?;
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
