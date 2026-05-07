//! CUDA implementation of [`ContextBackend`].
//!
//! Implements [`ContextBackend`] for [`CudaBackend`] so that generic op bodies
//! in `infernum/src/graph/builtin_ops.rs` can call `B::ctx_read`,
//! `B::ctx_write`, and `B::ctx_next_input` without violating Rust's orphan
//! rules (which forbid inherent `impl` blocks on foreign types).

use infernum::backend::ContextBackend;
use infernum::graph::execute_context::ExecuteContext;
use infernum::graph::{NodeId, OutputRef};

use crate::cuda::CudaTensor;
use crate::CudaBackend;

/// Backend-specific executor state for [`CudaBackend`].
///
/// Wraps the intermediate tensor buffer used by the CUDA executor:
/// a `Vec<Vec<Option<CudaTensor>>>` indexed by `(NodeId, output_index)`.
/// Each node gets a slot per output; slots are filled as ops execute and
/// freed (taken) once all downstream consumers have read them.
pub struct CudaExecutorState {
    pub buffers: Vec<Vec<Option<CudaTensor>>>,
}

impl ContextBackend for CudaBackend {
    /// Read a tensor from the CUDA buffer for the given `OutputRef`.
    ///
    /// # Panics
    ///
    /// Panics if the node output has no stored tensor.
    fn ctx_read(ctx: &ExecuteContext<'_, CudaBackend>, output_ref: OutputRef) -> CudaTensor {
        let (node_id, output_idx) = output_ref;
        ctx.state.buffers[node_id.index() as usize][output_idx as usize]
            .as_ref()
            .unwrap_or_else(|| panic!("node {node_id:?} output {output_idx} has no stored tensor"))
            .clone()
    }

    /// Write an op's output tensor into the CUDA buffer.
    fn ctx_write(
        ctx: &mut ExecuteContext<'_, CudaBackend>,
        node_id: NodeId,
        idx: u32,
        tensor: CudaTensor,
    ) {
        ctx.state.buffers[node_id.index() as usize][idx as usize] = Some(tensor);
    }

    /// Consume the next graph input tensor, advancing the input cursor.
    ///
    /// # Panics
    ///
    /// Panics if all input tensors have already been consumed.
    fn ctx_next_input(ctx: &mut ExecuteContext<'_, CudaBackend>) -> CudaTensor {
        let tensor = ctx.input_tensors[*ctx.input_idx].clone();
        *ctx.input_idx += 1;
        tensor
    }
}
