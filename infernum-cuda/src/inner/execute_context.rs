//! CUDA implementation of [`ContextBackend`] and paged-KV [`KvCacheAccess`].
//!
//! Implements [`ContextBackend`] for [`CudaBackend`] so that generic op bodies
//! in `infernum/src/graph/builtin_ops.rs` can call `B::ctx_read`,
//! `B::ctx_write`, and `B::ctx_next_input` without violating Rust's orphan
//! rules (which forbid inherent `impl` blocks on foreign types).

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};
use infernum::backend::{ContextBackend, MlaAttentionOps};
use infernum::graph::builtin_ops::MlaAttentionOp;
use infernum::graph::execute_context::{ExecuteContext, KvCacheAccess};
use infernum::graph::{NodeId, OutputRef};
use infernum::Result;

use crate::cuda::{ops, CudaTensor, PagedKvCache};
use crate::CudaBackend;

// ---------------------------------------------------------------------------
// GraphInputs
// ---------------------------------------------------------------------------

/// Pre-allocated GPU-side input buffers for CUDA graph capture decode steps.
///
/// Allocated once per decode engine instance with fixed device addresses.
/// Between graph replays the engine writes new values via `htod_copy_into`.
/// The CUDA graph records the fixed addresses — no re-capture needed when only
/// the values change.
///
/// The 6 buffers map 1-to-1 to the 6 graph input nodes of the paged decode
/// graph in this fixed order:
///
/// 0. `token_ids`   — `[batch_size]` U32
/// 1. `cos`         — `[batch_size, half_dim]` F32
/// 2. `sin`         — `[batch_size, half_dim]` F32
/// 3. `block_table` — `[batch_size, max_blocks_per_seq]` U32
/// 4. `positions`   — `[batch_size]` U32
/// 5. `seq_lens`    — `[batch_size]` U32
pub struct GraphInputs {
    pub token_ids: CudaSlice<u32>,
    pub cos: CudaSlice<f32>,
    pub sin: CudaSlice<f32>,
    pub block_table: CudaSlice<u32>,
    pub positions: CudaSlice<u32>,
    pub seq_lens: CudaSlice<u32>,
    pub batch_size: usize,
    pub half_dim: usize,
    pub max_blocks_per_seq: usize,
    /// The maximum sequence length for this decode step (`max(seq_lens)`).
    ///
    /// Pre-computed on the host from the `seq_lens` slice before the CUDA
    /// graph capture window begins, so that [`CudaPagedKvCacheAccess`] can
    /// pass a concrete value to the paged-attention kernel without performing
    /// a synchronous device-to-host transfer inside the capture.
    pub max_seq_len: usize,
}

impl GraphInputs {
    /// Allocate all input buffers on the given device, zeroed.
    ///
    /// `max_seq_len` is the pre-computed maximum sequence length for this
    /// decode step.  Pass `0` for dummy/placeholder `GraphInputs` instances
    /// that are swapped out before capture begins.
    ///
    /// # Errors
    ///
    /// Returns an error if any device allocation fails.
    pub fn new(
        device: &Arc<CudaDevice>,
        batch_size: usize,
        half_dim: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        Ok(Self {
            token_ids: device.alloc_zeros::<u32>(batch_size)?,
            cos: device.alloc_zeros::<f32>(batch_size * half_dim)?,
            sin: device.alloc_zeros::<f32>(batch_size * half_dim)?,
            block_table: device.alloc_zeros::<u32>(batch_size * max_blocks_per_seq)?,
            positions: device.alloc_zeros::<u32>(batch_size)?,
            seq_lens: device.alloc_zeros::<u32>(batch_size)?,
            batch_size,
            half_dim,
            max_blocks_per_seq,
            max_seq_len,
        })
    }
}

// ---------------------------------------------------------------------------
// CudaExecutorState
// ---------------------------------------------------------------------------

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
    /// Pre-allocated GPU-side input buffers for CUDA graph capture.
    ///
    /// `Some` during `CudaDecodeEngine::step()` (captured decode).
    /// `None` during eager prefill/decode (inputs come from `input_tensors`).
    pub graph_inputs: Option<GraphInputs>,
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
    /// During CUDA graph capture (`state.graph_inputs = Some(…)`), returns a
    /// non-owning [`CudaTensor`] view over the corresponding pre-allocated
    /// [`GraphInputs`] buffer.  The device address is stable across replays,
    /// allowing the captured graph to reference it without re-capture.
    ///
    /// During eager execution (`state.graph_inputs = None`), clones from the
    /// `input_tensors` slice as before.
    ///
    /// # Panics
    ///
    /// Panics if the input index is out of range.
    fn ctx_next_input(ctx: &mut ExecuteContext<'_, CudaBackend>) -> CudaTensor {
        let idx = *ctx.input_idx;
        *ctx.input_idx += 1;
        if let Some(ref inputs) = ctx.state.graph_inputs {
            // Build a non-owning CudaTensor view from the corresponding buffer.
            // The CudaContext is borrowed from the device handle already in ctx.
            let cuda_ctx: &crate::cuda::CudaContext = ctx.device;
            return match idx {
                0 => CudaTensor::from_cuda_slice_view(
                    cuda_ctx,
                    &[inputs.batch_size],
                    &inputs.token_ids,
                ),
                1 => CudaTensor::from_cuda_slice_view(
                    cuda_ctx,
                    &[inputs.batch_size, inputs.half_dim],
                    &inputs.cos,
                ),
                2 => CudaTensor::from_cuda_slice_view(
                    cuda_ctx,
                    &[inputs.batch_size, inputs.half_dim],
                    &inputs.sin,
                ),
                3 => CudaTensor::from_cuda_slice_view(
                    cuda_ctx,
                    &[inputs.batch_size, inputs.max_blocks_per_seq],
                    &inputs.block_table,
                ),
                4 => CudaTensor::from_cuda_slice_view(
                    cuda_ctx,
                    &[inputs.batch_size],
                    &inputs.positions,
                ),
                5 => CudaTensor::from_cuda_slice_view(
                    cuda_ctx,
                    &[inputs.batch_size],
                    &inputs.seq_lens,
                ),
                _ => panic!(
                    "ctx_next_input: index {idx} out of range for GraphInputs (expected 0–5)"
                ),
            };
        }
        ctx.input_tensors[idx].clone()
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
pub struct CudaPagedKvCacheAccess<'a> {
    pub cache: &'a mut PagedKvCache,
    /// Pre-computed `max(seq_lens)` for the current decode step.
    /// `0` means "compute from GPU tensor" (eager path only, never during capture).
    pub max_seq_len: usize,
}

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
        self.cache.append_paged_batched_tensor(
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
        // The generic `PagedAttentionDecodeOp` passes `max_seq_len = 0` as a
        // sentinel.  The CUDA kernel uses `max_seq_len` to size its
        // shared-memory allocation; a zero value causes under-allocation and
        // corrupted attention output.
        //
        // Resolution priority:
        // 1. Non-zero sentinel passed by the op (future callers that know it).
        // 2. `self.max_seq_len` — pre-computed on the host from `seq_lens_u32`
        //    *before* the CUDA graph capture window.  This is always set for the
        //    graph-capture path, avoiding a synchronous D→H copy inside capture
        //    which would invalidate the stream on all GPU generations.
        // 3. Fallback: read from the GPU tensor (eager / non-captured paths only).
        let resolved = if max_seq_len != 0 {
            max_seq_len
        } else if self.max_seq_len != 0 {
            self.max_seq_len
        } else {
            // Eager path only — safe because we are not inside a stream capture.
            let seq_lens_vec = seq_lens.to_vec::<u32>()?;
            seq_lens_vec
                .iter()
                .copied()
                .map(|x| x as usize)
                .max()
                .unwrap_or(1)
        };
        let (k_pool, v_pool) = self.cache.get_pools(layer_idx);
        ops::paged_attention_decode_from_tensor(
            q.context(),
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
