//! Graph executor for the Metal backend.
//!
//! Walks an [`ExecutionPlan`] in topological order and dispatches each
//! operation to the appropriate Metal kernel via `node.op.execute()`.
//! Intermediate tensors are stored in a `Vec<Vec<Option<MetalTensor>>>`
//! indexed by `(NodeId, output_index)`.

use infernum::graph::execute_context::{ExecuteContext, KvCacheAccess};
use infernum::graph::{GraphNode, NodeId, WeightStore};
use infernum::{ExecutionPlan, Result};

use crate::context::MetalContext;
use crate::tensor::MetalTensor;
use crate::weights::MetalLinearWeight;
use crate::{MetalBackend, MetalExecutorState};

/// Take (move) a tensor out of the node buffer. Used for the final outputs.
fn take(buffers: &mut [Vec<Option<MetalTensor>>], node_id: NodeId, output_idx: u32) -> MetalTensor {
    buffers[node_id.index() as usize][output_idx as usize]
        .take()
        .unwrap_or_else(|| panic!("node {node_id:?} output {output_idx} has no stored tensor"))
}

/// Execute a computation graph on the Metal backend.
///
/// Walks the `plan.schedule` in topological order and dispatches each op
/// via `node.op.execute(ctx)`. Intermediate tensors are stored as
/// `MetalTensor` values (Metal unified memory handles allocation reuse
/// naturally).
///
/// # Arguments
///
/// * `device` — Metal context (device, command queue, pipelines).
/// * `plan` — Execution plan with schedule.
/// * `nodes` — Graph nodes (indexed by `NodeId`).
/// * `weights` — Loaded model weights.
/// * `inputs` — External input tensors, one per `Input` op in schedule order.
/// * `output_nodes` — `NodeId`s of the graph outputs to collect.
/// * `paged_kv_cache` — Optional paged KV cache for decode-mode execution.
///
/// # Errors
///
/// Returns an error if any op kernel fails.
#[allow(
    clippy::missing_panics_doc,
    clippy::too_many_arguments,
    clippy::needless_pass_by_ref_mut
)]
pub fn execute(
    device: &MetalContext,
    plan: &ExecutionPlan,
    nodes: &[GraphNode<MetalBackend>],
    weights: &WeightStore<MetalTensor, MetalLinearWeight>,
    inputs: &[MetalTensor],
    output_nodes: &[NodeId],
    mut paged_kv_cache: Option<&mut crate::MetalPagedKvCache>,
) -> Result<Vec<MetalTensor>> {
    // Build per-node output buffers.
    let buffers: Vec<Vec<Option<MetalTensor>>> = nodes
        .iter()
        .map(|node| {
            let num_outputs = node.output_shapes.len().max(1);
            vec![None; num_outputs]
        })
        .collect();
    let mut state = MetalExecutorState { buffers };
    let mut input_idx: usize = 0;

    // Execute nodes in scheduled order.
    for &node_id in &plan.schedule {
        let node = &nodes[node_id.index() as usize];

        let mut paged_acc = paged_kv_cache
            .as_deref_mut()
            .map(|cache| crate::execute_context::MetalPagedKvCacheAccess { cache });
        let kv: Option<&mut dyn KvCacheAccess<MetalBackend>> = paged_acc
            .as_mut()
            .map(|a| a as &mut dyn KvCacheAccess<MetalBackend>);

        let mut ctx = ExecuteContext {
            state: &mut state,
            plan,
            nodes,
            weights,
            device,
            kv_cache: kv,
            input_tensors: inputs,
            input_idx: &mut input_idx,
        };
        node.op.execute(&mut ctx, node_id, &node.inputs)?;
    }

    // Collect output tensors.
    let outputs = output_nodes
        .iter()
        .map(|&id| take(&mut state.buffers, id, 0))
        .collect();

    Ok(outputs)
}
