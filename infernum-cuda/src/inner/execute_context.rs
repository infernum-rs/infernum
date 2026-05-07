//! CUDA implementation of [`ContextBackend`] and paged-KV [`KvCacheAccess`].
//!
//! Implements [`ContextBackend`] for [`CudaBackend`] so that generic op bodies
//! in `infernum/src/graph/builtin_ops.rs` can call `B::ctx_read`,
//! `B::ctx_write`, and `B::ctx_next_input` without violating Rust's orphan
//! rules (which forbid inherent `impl` blocks on foreign types).

use infernum::backend::ContextBackend;
use infernum::graph::execute_context::{ExecuteContext, KvCacheAccess};
use infernum::graph::{NodeId, OutputRef};
use infernum::Result;

use crate::cuda::{ops, CudaTensor, PagedKvCache};
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

/// Wraps `&mut PagedKvCache` and exposes it as a [`KvCacheAccess<CudaBackend>`].
///
/// The CUDA executor passes this wrapper into [`ExecuteContext::kv_cache`] so
/// that [`AppendPagedBatchedOp`] and [`PagedAttentionDecodeOp`] can call the
/// paged methods generically through the trait, without knowing the concrete
/// [`PagedKvCache`] type.
///
/// The dense-KV methods (`is_cache_input`, `read_cache`, `write_cache`,
/// `cache_concat_info`, `try_append_kv`, `finalize_step`) are all no-ops:
/// the paged cache has no concept of per-node dense KV slots and these methods
/// are never called on this type.
pub struct CudaPagedKvCacheAccess<'a>(pub &'a mut PagedKvCache);

impl KvCacheAccess<CudaBackend> for CudaPagedKvCacheAccess<'_> {
    fn is_cache_input(&self, _node_id: NodeId) -> bool {
        false
    }

    fn read_cache(&self, _node_id: NodeId) -> Option<&CudaTensor> {
        None
    }

    fn write_cache(&mut self, _node_id: NodeId, _tensor: CudaTensor) {}

    fn cache_concat_info(&self, _node_id: NodeId) -> Option<(usize, bool)> {
        None
    }

    fn try_append_kv(&mut self, _node_id: NodeId, _new_row: &CudaTensor) -> Option<CudaTensor> {
        None
    }

    fn finalize_step(&mut self) {}

    fn append_paged_batched(
        &mut self,
        layer_idx: usize,
        k: &CudaTensor,
        v: &CudaTensor,
        block_tables: &CudaTensor,
        positions: &CudaTensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        self.0.append_paged_batched_tensor(
            layer_idx,
            k,
            v,
            block_tables,
            positions,
            batch_size,
            max_blocks_per_seq,
        )
    }

    fn paged_attention_decode(
        &mut self,
        layer_idx: usize,
        q: &CudaTensor,
        block_tables: &CudaTensor,
        seq_lens: &CudaTensor,
        block_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<CudaTensor> {
        let (k_pool, v_pool) = self.0.get_pools(layer_idx);
        ops::paged_attention_decode_from_tensor(
            q.context(),
            q,
            k_pool,
            v_pool,
            block_tables,
            seq_lens,
            block_size,
            max_blocks_per_seq,
            max_seq_len,
            None, // scale: auto (1/sqrt(head_dim))
            softcap,
            sliding_window,
        )
    }
}
