//! Trait for executable operation nodes in the computation graph.
//!
//! An [`OpNode`] wraps a single graph operation with enough metadata
//! for shape/dtype inference and runtime execution. The graph builder
//! stores `Box<dyn OpNode<B>>` inside each node, decoupling the static
//! graph structure from the concrete op implementations.

use std::any::Any;

use crate::backend::{Backend, MatmulOps};
use crate::dtype::DType;
use crate::Result;

use super::execute_context::ExecuteContext;
use super::node::NodeId;

/// Reference to a specific output of a graph node: `(node, output_index)`.
pub type OutputRef = (NodeId, u32);

/// An executable operation node in the computation graph.
///
/// Each implementor describes one logical operation (matmul, `RMSNorm`,
/// `RoPE`, etc.) and knows how to infer output shapes/dtypes and execute
/// on a concrete backend.
pub trait OpNode<B: Backend + MatmulOps>: Send + Sync + std::fmt::Debug {
    /// Human-readable name for debugging and profiling (e.g., `"matmul"`).
    fn name(&self) -> &'static str;

    /// Number of input tensors this operation consumes.
    fn num_inputs(&self) -> usize;

    /// Number of output tensors this operation produces.
    fn num_outputs(&self) -> usize;

    /// Infer output shapes from the given input shapes.
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>>;

    /// Infer output dtypes from the given input dtypes.
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType>;

    /// Execute the operation, reading inputs and writing outputs via `ctx`.
    ///
    /// # Errors
    /// Returns an error if the underlying backend operation fails.
    fn execute(
        &self,
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()>;

    /// Whether this operation has side effects (e.g., KV cache writes).
    ///
    /// Side-effect nodes are never eliminated by dead-code removal.
    fn is_side_effect(&self) -> bool {
        false
    }

    /// Downcast to a concrete type for optimizer fusion passes.
    fn as_any(&self) -> &dyn Any;
}
