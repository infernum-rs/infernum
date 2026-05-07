//! CUDA implementation of [`ContextBackend`] and paged-KV [`KvCacheAccess`].
//!
//! Implements [`ContextBackend`] for [`CudaBackend`] so that generic op bodies
//! in `infernum/src/graph/builtin_ops.rs` can call `B::ctx_read`,
//! `B::ctx_write`, and `B::ctx_next_input` without violating Rust's orphan
//! rules (which forbid inherent `impl` blocks on foreign types).

use infernum::backend::{ContextBackend, MlaAttentionOps};
use infernum::graph::builtin_ops::MlaAttentionOp;
use infernum::graph::execute_context::{ExecuteContext, KvCacheAccess};
use infernum::graph::{NodeId, OutputRef};
use infernum::Result;

use crate::cuda::{ops, CudaTensor, PagedKvCache};
use crate::CudaBackend;

/// Backend-specific executor state for [`CudaBackend`].
///
/// Holds the intermediate tensor buffer indexed by `(NodeId, output_index)`,
/// plus optional MLA KV cache state for `DeepSeek` V3/R1 attention.
///
/// `mla_kv_cache_ptr` is a raw pointer to the per-layer MLA KV cache owned by
/// the graph engine.  It is valid for the lifetime of the `execute` call that
/// constructs this state; the `get_mla_layer` accessor re-borrows it safely
/// inside that scope.  Using a raw pointer avoids propagating a lifetime
/// parameter through the `Backend::ExecutorState` associated type.
pub struct CudaExecutorState {
    pub buffers: Vec<Vec<Option<CudaTensor>>>,
    /// Raw pointer to `Vec<Vec<CudaTensor>>` (per-layer MLA KV tensors), or
    /// null if this execution has no MLA KV cache.
    pub mla_kv_cache_ptr: *mut Vec<Vec<CudaTensor>>,
    /// Current sequence position for MLA attention.
    pub mla_seq_pos: usize,
}

// SAFETY: `CudaExecutorState` is only constructed and used within a single
// `execute` call on one thread.  The raw pointer is never aliased across
// threads.
unsafe impl Send for CudaExecutorState {}

impl CudaExecutorState {
    /// Borrow the MLA KV tensors for `layer_idx`.
    ///
    /// # Panics
    ///
    /// Panics if no MLA KV cache was provided or if `layer_idx` is out of
    /// range.
    pub fn get_mla_layer(&mut self, layer_idx: usize) -> &mut Vec<CudaTensor> {
        assert!(
            !self.mla_kv_cache_ptr.is_null(),
            "CUDA executor: mla_attention op requires mla_kv_cache to be provided"
        );
        // SAFETY: pointer is non-null and valid for the duration of the
        // enclosing `execute` call (see field doc).
        let cache = unsafe { &mut *self.mla_kv_cache_ptr };
        cache.get_mut(layer_idx).unwrap_or_else(|| {
            panic!("CUDA executor: mla_kv_cache has no entry for layer {layer_idx}")
        })
    }
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

    // Weight names (q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj_k, …) are
    // dictated by the DeepSeek V3 checkpoint and cannot be renamed.
    #[allow(clippy::similar_names)]
    fn ctx_execute_mla(
        op: &MlaAttentionOp,
        ctx: &mut ExecuteContext<'_, CudaBackend>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let hidden = CudaBackend::ctx_read(ctx, inputs[0]);
        let q_a_proj = ctx.weights.linear_weight(op.q_a_proj);
        let q_a_layernorm = ctx.weights.tensor_weight(op.q_a_layernorm);
        let q_b_proj = ctx.weights.linear_weight(op.q_b_proj);
        let kv_a_proj_with_mqa = ctx.weights.linear_weight(op.kv_a_proj_with_mqa);
        let kv_a_layernorm = ctx.weights.tensor_weight(op.kv_a_layernorm);
        let kv_b_proj_k = ctx.weights.linear_weight(op.kv_b_proj_k);
        let kv_b_proj_v = ctx.weights.linear_weight(op.kv_b_proj_v);
        let kv_b_proj_k_t = ctx.weights.linear_weight(op.kv_b_proj_k_t);
        let o_proj = ctx.weights.linear_weight(op.o_proj);
        // Read seq_pos before get_mla_layer to avoid simultaneous borrows.
        let seq_pos = ctx.state.mla_seq_pos;
        let layer_kv = ctx.state.get_mla_layer(op.layer_idx);
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
            seq_pos,
            op.num_heads,
            op.qk_nope_head_dim,
            op.qk_rope_head_dim,
            op.v_head_dim,
            op.kv_lora_rank,
            op.rms_norm_eps,
            op.attn_scale,
        )?;
        CudaBackend::ctx_write(ctx, node_id, 0, result);
        Ok(())
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
