//! Context passed to each op's [`OpNode::execute`] method.
//!
//! [`ExecuteContext`] replaces the old `(inputs, weights, device)` triple with
//! a single struct that gives ops access to everything they might need: arena
//! (CPU) or tensor map (CUDA), the execution plan, node metadata, weights,
//! the device handle, KV caches, and graph inputs.
//!
//! [`KvCacheAccess`] is a backend-agnostic trait over the concrete KV cache
//! types so that ops like `AppendKv` can operate without knowing whether they
//! are running on CPU or CUDA.

use crate::backend::{Backend, MatmulOps};
use crate::DType;

use super::node::{GraphNode, NodeId};
use super::op_node::OutputRef;
use super::planner::ExecutionPlan;
use super::weight_store::WeightStore;

/// Trait for KV cache types that can be accessed generically by ops.
///
/// CPU ops use `KvCacheStore` which implements this trait for `CpuBackend`.
/// CUDA ops use the CUDA KV cache type which implements it for `CudaBackend`.
/// The executor stores a `dyn KvCacheAccess<B>` so ops can call these methods
/// without knowing the concrete type.
pub trait KvCacheAccess<B: Backend> {
    /// Returns `true` if `node_id` is a KV-cache Input node (i.e. its tensor
    /// is backed by the cache buffer rather than the arena / tensor map).
    fn is_cache_input(&self, node_id: NodeId) -> bool;

    /// Borrow the cached tensor for `node_id`, or `None` if not present.
    fn read_cache(&self, node_id: NodeId) -> Option<&B::Tensor>;

    /// Store a tensor into the cache for `node_id`.
    fn write_cache(&mut self, node_id: NodeId, tensor: B::Tensor);

    /// Returns `(layer_index, is_key)` if `node_id` is a KV-cache `ConcatSeq`
    /// node, or `None` if it is a regular concat.
    fn cache_concat_info(&self, node_id: NodeId) -> Option<(usize, bool)>;

    /// If `node_id` is a KV concat node, appends `new_row` to the persistent
    /// cache and returns the full updated cache tensor. Returns `None` if this
    /// is not a KV concat node.
    fn try_append_kv(&mut self, node_id: NodeId, new_row: &B::Tensor) -> Option<B::Tensor>;

    /// Called after all ops in one decode step have executed, to advance
    /// internal sequence-length counters.
    fn finalize_step(&mut self);
}

/// Context passed to each op's [`super::op_node::OpNode::execute`] method.
///
/// Ops call [`read_tensor`](ExecuteContext::read_tensor),
/// [`write_tensor`](ExecuteContext::write_tensor), and
/// [`next_input`](ExecuteContext::next_input) rather than touching backend
/// internals directly.  The concrete implementations live in the backend
/// crates (`infernum-cpu`, `infernum-cuda`).
pub struct ExecuteContext<'a, B: Backend + MatmulOps> {
    /// Backend-specific executor state.
    ///
    /// For `CpuBackend` this is `&mut Arena` (the flat intermediate buffer).
    /// For `CudaBackend` this is `&mut CudaExecutorState` (the tensor map).
    pub state: &'a mut B::ExecutorState,

    /// The execution plan (schedule + buffer slots).
    pub plan: &'a ExecutionPlan,

    /// All nodes in the graph, indexed by `NodeId`.
    pub nodes: &'a [GraphNode<B>],

    /// Loaded weights (tensors and linear weights).
    pub weights: &'a WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,

    /// Opaque device handle (e.g. `CudaContext`, `()` for CPU).
    pub device: &'a B::DeviceHandle,

    /// Optional KV cache for decode-mode execution.
    pub kv_cache: Option<&'a mut dyn KvCacheAccess<B>>,

    /// Graph input tensors supplied by the caller for this forward pass.
    pub input_tensors: &'a [B::Tensor],

    /// Cursor into `input_tensors` — advanced by [`next_input`](ExecuteContext::next_input).
    pub input_idx: &'a mut usize,
}

impl<B: Backend + MatmulOps> ExecuteContext<'_, B> {
    /// Returns `(byte_offset, element_count, dtype)` for an input tensor
    /// identified by its [`OutputRef`].
    ///
    /// Used by op `execute()` bodies to locate arena slots without importing
    /// planner internals directly.
    ///
    /// # Panics
    ///
    /// Panics if the node output has no buffer slot assigned by the planner.
    #[must_use]
    pub fn input_slot(&self, output_ref: OutputRef) -> (usize, usize, DType) {
        let (node_id, output_idx) = output_ref;
        let node = &self.nodes[node_id.index() as usize];
        let slot = self
            .plan
            .slot(node_id, output_idx)
            .expect("input node output has no buffer slot");
        let shape = &node.output_shapes[output_idx as usize];
        let dtype = node.output_dtypes[output_idx as usize];
        let num_elements: usize = shape.iter().product();
        (slot.offset, num_elements, dtype)
    }

    /// Returns `(byte_offset, element_count)` for this node's output slot, or
    /// `None` if the planner did not assign a slot (side-effect-only outputs).
    #[must_use]
    pub fn output_slot(&self, node_id: NodeId, idx: u32) -> Option<(usize, usize)> {
        let node = &self.nodes[node_id.index() as usize];
        self.plan.slot(node_id, idx).map(|slot| {
            let shape = &node.output_shapes[idx as usize];
            let num_elements: usize = shape.iter().product();
            (slot.offset, num_elements)
        })
    }

    /// Read a tensor produced by a prior node in the graph.
    ///
    /// Returns an owned tensor (cloned from the backend's storage).
    /// Implemented concretely for each backend in the backend crates.
    ///
    /// # Panics
    ///
    /// Panics with a backend-specific message if the tensor is not found.
    #[must_use]
    pub fn read_tensor(&self, _output_ref: OutputRef) -> B::Tensor {
        unimplemented!(
            "ExecuteContext::read_tensor — implement in the backend crate (infernum-cpu / infernum-cuda)"
        )
    }

    /// Write an op's output tensor into the context for future ops to read.
    ///
    /// Implemented concretely for each backend in the backend crates.
    ///
    /// # Panics
    ///
    /// Panics with a backend-specific message if the write fails.
    pub fn write_tensor(&mut self, _node_id: NodeId, _output_idx: u32, _tensor: B::Tensor) {
        unimplemented!(
            "ExecuteContext::write_tensor — implement in the backend crate (infernum-cpu / infernum-cuda)"
        )
    }

    /// Consume the next graph input tensor, advancing the input cursor.
    ///
    /// Returns an owned tensor. Implemented concretely for each backend in the
    /// backend crates.
    ///
    /// # Panics
    ///
    /// Panics if all input tensors have already been consumed.
    pub fn next_input(&mut self) -> B::Tensor {
        unimplemented!(
            "ExecuteContext::next_input — implement in the backend crate (infernum-cpu / infernum-cuda)"
        )
    }
}
