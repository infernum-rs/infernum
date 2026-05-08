//! Metal implementation of [`ContextBackend`] and paged-KV [`KvCacheAccess`].
//!
//! Implements [`ContextBackend`] for [`MetalBackend`] so that generic op bodies
//! in `infernum/src/graph/builtin_ops.rs` can call `B::ctx_read`,
//! `B::ctx_write`, and `B::ctx_next_input` without violating Rust's orphan
//! rules.
//!
//! [`MetalPagedKvCacheAccess`] wraps `&mut MetalPagedKvCache` for graph
//! execution, delegating append and paged-attention calls to the existing
//! Metal ops.

use infernum::backend::{ContextBackend, PagedAttentionOps, PagedKvCacheOps};
use infernum::graph::execute_context::{ExecuteContext, KvCacheAccess};
use infernum::graph::{NodeId, OutputRef};
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::{MetalBackend, MetalPagedKvCache};

// ---------------------------------------------------------------------------
// ContextBackend
// ---------------------------------------------------------------------------

impl ContextBackend for MetalBackend {
    /// Read a tensor from the Metal buffer for the given `OutputRef`.
    ///
    /// # Panics
    ///
    /// Panics if the node output has no stored tensor.
    fn ctx_read(ctx: &ExecuteContext<'_, MetalBackend>, output_ref: OutputRef) -> MetalTensor {
        let (node_id, output_idx) = output_ref;
        ctx.state.buffers[node_id.index() as usize][output_idx as usize]
            .as_ref()
            .unwrap_or_else(|| panic!("node {node_id:?} output {output_idx} has no stored tensor"))
            .clone()
    }

    /// Write an op's output tensor into the Metal buffer.
    fn ctx_write(
        ctx: &mut ExecuteContext<'_, MetalBackend>,
        node_id: NodeId,
        idx: u32,
        tensor: MetalTensor,
    ) {
        ctx.state.buffers[node_id.index() as usize][idx as usize] = Some(tensor);
    }

    /// Consume the next graph input tensor, advancing the input cursor.
    ///
    /// # Panics
    ///
    /// Panics if the input index is out of range.
    fn ctx_next_input(ctx: &mut ExecuteContext<'_, MetalBackend>) -> MetalTensor {
        let idx = *ctx.input_idx;
        *ctx.input_idx += 1;
        ctx.input_tensors[idx].clone()
    }
}

// ---------------------------------------------------------------------------
// MetalPagedKvCacheAccess
// ---------------------------------------------------------------------------

/// Wraps `&mut MetalPagedKvCache` for graph execution.
///
/// The Metal executor passes this wrapper into [`ExecuteContext::kv_cache`] so
/// that [`AppendPagedBatchedOp`] and [`PagedAttentionDecodeOp`] can call the
/// paged methods generically through the trait, without knowing the concrete
/// [`MetalPagedKvCache`] type.
///
/// The dense-KV methods (`is_cache_input`, `read_cache`, `write_cache`,
/// `cache_concat_info`, `try_append_kv`, `finalize_step`) are all no-ops:
/// the paged cache has no concept of per-node dense KV slots and these methods
/// are never called on this type.
pub struct MetalPagedKvCacheAccess<'a> {
    pub cache: &'a mut MetalPagedKvCache,
}

impl KvCacheAccess<MetalBackend> for MetalPagedKvCacheAccess<'_> {
    fn is_cache_input(&self, _node_id: NodeId) -> bool {
        false
    }

    fn read_cache(&self, _node_id: NodeId) -> Option<&MetalTensor> {
        None
    }

    fn write_cache(&mut self, _node_id: NodeId, _tensor: MetalTensor) {}

    fn cache_concat_info(&self, _node_id: NodeId) -> Option<(usize, bool)> {
        None
    }

    fn try_append_kv(&mut self, _node_id: NodeId, _new_row: &MetalTensor) -> Option<MetalTensor> {
        None
    }

    fn finalize_step(&mut self) {}

    fn append_paged_batched(
        &mut self,
        layer_idx: usize,
        k: &MetalTensor,
        v: &MetalTensor,
        block_tables: &MetalTensor,
        positions: &MetalTensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        MetalBackend::append_paged_batched(
            self.cache,
            layer_idx,
            k,
            v,
            block_tables,
            positions,
            batch_size,
            max_blocks_per_seq,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn paged_attention_decode(
        &mut self,
        layer_idx: usize,
        q: &MetalTensor,
        block_tables: &MetalTensor,
        seq_lens: &MetalTensor,
        block_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<MetalTensor> {
        // Resolve max_seq_len: if 0, read from the GPU tensor (eager path).
        let resolved = if max_seq_len != 0 {
            max_seq_len
        } else {
            let seq_lens_bytes = seq_lens.as_bytes();
            let seq_lens_u32: &[u32] = bytemuck::cast_slice(seq_lens_bytes);
            seq_lens_u32
                .iter()
                .copied()
                .map(|x| x as usize)
                .max()
                .unwrap_or(1)
        };
        let (k_pool, v_pool) = MetalBackend::get_pools(self.cache, layer_idx);
        MetalBackend::paged_attention_decode(
            q,
            k_pool,
            v_pool,
            block_tables,
            seq_lens,
            block_size,
            max_blocks_per_seq,
            resolved,
            None, // scale: auto (1/sqrt(head_dim))
            softcap,
            sliding_window,
        )
    }
}
