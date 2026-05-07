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
//! `execute` is a pure open-dispatch loop: every op goes through
//! `node.op.execute(ctx)`. [`CudaExecutorState`] carries `mla_kv_cache_ptr`
//! and `mla_seq_pos` so the concrete [`OpNode<CudaBackend>`] impl for
//! [`MlaAttentionOp`] (in `execute_context.rs`) can access them without any
//! named arm in the executor. Paged KV ops are wired via [`CudaPagedKvCacheAccess`].

use infernum::graph::builtin_ops::ArgmaxLastOp;
use infernum::graph::execute_context::KvCacheAccess;
use infernum::graph::{GraphNode, OutputRef, WeightStore};
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

        // --- Open dispatch: all ops self-execute via OpNode::execute ---
        // MlaAttentionOp has a concrete impl<CudaBackend> in execute_context.rs
        // that reads mla_kv_cache_ptr and mla_seq_pos directly from ctx.state.
        {
            use crate::inner::execute_context::{CudaExecutorState, CudaPagedKvCacheAccess};
            use infernum::graph::execute_context::ExecuteContext;
            let device = inputs[0].context().clone();
            let mla_ptr = mla_kv_cache
                .as_deref_mut()
                .map_or(std::ptr::null_mut(), std::ptr::from_mut);
            let mut state = CudaExecutorState {
                buffers,
                mla_kv_cache_ptr: mla_ptr,
                mla_seq_pos,
                graph_inputs: None,
            };
            let mut input_idx_local = input_idx;
            {
                let mut paged_acc = paged_kv_cache.as_deref_mut().map(CudaPagedKvCacheAccess);
                let kv: Option<&mut dyn KvCacheAccess<CudaBackend>> = paged_acc
                    .as_mut()
                    .map(|a| a as &mut dyn KvCacheAccess<CudaBackend>);
                let mut exec_ctx = ExecuteContext {
                    state: &mut state,
                    plan,
                    nodes,
                    weights,
                    device: &device,
                    kv_cache: kv,
                    input_tensors: inputs,
                    input_idx: &mut input_idx_local,
                };
                node.op.execute(&mut exec_ctx, node_id, &node.inputs)?;
            }
            buffers = state.buffers;
            input_idx = input_idx_local;
        }
    }

    // Collect output tensors.
    let outputs = output_nodes
        .iter()
        .map(|&id| take(&mut buffers, id, 0))
        .collect();

    Ok(outputs)
}
