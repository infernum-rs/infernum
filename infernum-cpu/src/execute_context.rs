//! CPU implementation of [`ContextBackend`].
//!
//! Implements [`ContextBackend`] for [`CpuBackend`] so that generic op bodies
//! in `infernum/src/graph/builtin_ops.rs` can call `B::ctx_read`,
//! `B::ctx_write`, and `B::ctx_next_input` without violating Rust's orphan
//! rules (which forbid inherent `impl` blocks on foreign types).

use infernum::backend::ContextBackend;
use infernum::graph::execute_context::ExecuteContext;
use infernum::graph::{NodeId, OutputRef};
use infernum::DType;

use crate::tensor::CpuTensor;
use crate::CpuBackend;

impl ContextBackend for CpuBackend {
    /// Read a tensor from the arena (or KV cache overrides) for the given
    /// `OutputRef`.
    ///
    /// Checks the KV cache's override map first. If the node's output is
    /// stored there (e.g., from a KV concat step), it is returned directly.
    /// Otherwise the tensor is read from the arena using the execution plan's
    /// buffer slot.
    ///
    /// # Panics
    ///
    /// Panics if the node output has no buffer slot assigned by the planner.
    fn ctx_read(ctx: &ExecuteContext<'_, CpuBackend>, output_ref: OutputRef) -> CpuTensor {
        let (node_id, output_idx) = output_ref;
        // Check KV cache overrides first.
        if let Some(kv) = &ctx.kv_cache {
            if let Some(tensor) = kv.read_cache(node_id) {
                return tensor.clone();
            }
        }
        // Fall back to arena.
        let node = &ctx.nodes[node_id.index() as usize];
        let slot = ctx
            .plan
            .slot(node_id, output_idx)
            .expect("node output has no buffer slot");
        let shape = &node.output_shapes[output_idx as usize];
        let dtype = node.output_dtypes[output_idx as usize];
        let num_elements: usize = shape.iter().product();
        if dtype == DType::U32 {
            let data = ctx.state.u32_slice(slot.offset, num_elements);
            CpuTensor::from_u32(shape, data)
        } else {
            let data = ctx.state.f32_slice(slot.offset, num_elements);
            CpuTensor::from_f32(shape, data)
        }
    }

    /// Write a tensor into the arena at the plan's buffer slot for the given
    /// node output.
    ///
    /// If the planner did not assign a slot for this output (e.g., a dummy
    /// output on a side-effect op), the write is silently skipped.
    fn ctx_write(
        ctx: &mut ExecuteContext<'_, CpuBackend>,
        node_id: NodeId,
        idx: u32,
        tensor: CpuTensor,
    ) {
        use infernum::tensor::Tensor as TensorTrait;
        if let Some(slot) = ctx.plan.slot(node_id, idx) {
            if tensor.dtype() == DType::U32 {
                let src = tensor.as_u32_slice();
                let dst = ctx.state.u32_slice_mut(slot.offset, src.len());
                dst.copy_from_slice(src);
            } else {
                let src = tensor.as_f32_slice();
                let dst = ctx.state.f32_slice_mut(slot.offset, src.len());
                dst.copy_from_slice(src);
            }
        }
    }

    /// Consume the next graph input tensor, advancing the input cursor.
    ///
    /// # Panics
    ///
    /// Panics if all input tensors have already been consumed.
    fn ctx_next_input(ctx: &mut ExecuteContext<'_, CpuBackend>) -> CpuTensor {
        let tensor = ctx.input_tensors[*ctx.input_idx].clone();
        *ctx.input_idx += 1;
        tensor
    }
}
