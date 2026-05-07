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

use super::node::{GraphNode, NodeId};
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

    /// Append K/V tensors to the paged cache (batched, device-side block tables).
    ///
    /// `block_tables`: shape `(batch_size, max_blocks_per_seq)` i32 tensor.
    /// `positions`: shape `(batch_size,)` i32 tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend has no paged KV cache or if the kernel fails.
    #[allow(clippy::too_many_arguments)]
    fn append_paged_batched(
        &mut self,
        layer_idx: usize,
        k: &B::Tensor,
        v: &B::Tensor,
        block_tables: &B::Tensor,
        positions: &B::Tensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
    ) -> crate::error::Result<()> {
        let _ = (
            layer_idx,
            k,
            v,
            block_tables,
            positions,
            batch_size,
            max_blocks_per_seq,
        );
        Err(crate::error::Error::Other(
            "append_paged_batched not supported by this KV cache".to_string(),
        ))
    }

    /// Paged attention decode: attends over the paged KV pool for `layer_idx`.
    ///
    /// Returns the attention output tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend has no paged KV cache or if the kernel fails.
    #[allow(clippy::too_many_arguments)]
    fn paged_attention_decode(
        &mut self,
        layer_idx: usize,
        q: &B::Tensor,
        block_tables: &B::Tensor,
        seq_lens: &B::Tensor,
        block_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> crate::error::Result<B::Tensor> {
        let _ = (
            layer_idx,
            q,
            block_tables,
            seq_lens,
            block_size,
            max_blocks_per_seq,
            max_seq_len,
            softcap,
            sliding_window,
        );
        Err(crate::error::Error::Other(
            "paged_attention_decode not supported by this KV cache".to_string(),
        ))
    }
}

/// Context passed to each op's [`super::op_node::OpNode::execute`] method.
///
/// Ops access tensors via the static-dispatch helpers on [`crate::backend::ContextBackend`]:
/// [`ctx_read`](crate::backend::ContextBackend::ctx_read),
/// [`ctx_write`](crate::backend::ContextBackend::ctx_write), and
/// [`ctx_next_input`](crate::backend::ContextBackend::ctx_next_input).
/// These are associated functions rather than methods on `ExecuteContext` so
/// that each backend can implement them without depending on the other backend.
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

    /// Cursor into `input_tensors` — advanced by [`ContextBackend::ctx_next_input`](crate::backend::ContextBackend::ctx_next_input).
    pub input_idx: &'a mut usize,
}
