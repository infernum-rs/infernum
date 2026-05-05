//! Graph builder traits — conditionally implemented on `Graph<B>`
//! when the backend `B` implements the corresponding op trait.
//!
//! Each method constructs the appropriate op struct from `builtin_ops`,
//! boxes it, and calls `graph.add_node()`. Shape inference is handled
//! by the op's `output_shapes()` method.

use crate::backend::{
    ArithOps, AttentionOps, Backend, BiasOps, CastOps, EmbedOps, GegluOps, MatmulExtOps, MatmulOps,
    MlaAttentionOps, MoeOps, MoeSigmoidOps, NormOps, PagedAttentionOps, PagedKvCacheOps,
    RopeInterleavedOps, RopeOps, SwigluOps, TensorOps,
};
use crate::dtype::DType;

use super::builder::Graph;
use super::builtin_ops::{
    AddInplaceOp, AddOp, AddRmsNormOp, AppendPagedBatchedOp, AppendPagedOp, BiasAddOp,
    CastFromF32Op, CastToF32Op, ConcatInnerDimOp, ConcatSeqOp, EmbeddingGatherOp, ExtractLastRowOp,
    FusedAttentionDecodeOp, FusedAttentionPrefillOp, GatherPagedKvOp, GegluOp, LinearOp,
    LinearPairOp, LinearTripleOp, LmHeadOp, LogitSoftcapOp, MatmulBf16F32Op, MatmulOp,
    MlaAttentionOp, MoeDispatchSigmoidOp, MoeDispatchSoftmaxOp, MoeExpertIds, MulOp,
    PagedAttentionDecodeOp, RepeatKvOp, ReshapeOp, RmsNormOp, RmsNormQkOp, RopeBatchedOp,
    RopeInterleavedOp as RopeIntOp, RopeOp, ScaleOp, SiluOp, SplitInnerDimOp, SwigluOp,
    Transpose2dOp,
};
use super::node::WeightId;
use super::op_node::OutputRef;

// ---------------------------------------------------------------------------
// EmbedOps
// ---------------------------------------------------------------------------

/// Graph builder methods for embedding operations.
pub trait GraphEmbedOps {
    /// Embedding table lookup.
    ///
    /// `table` is a tensor weight with shape `(vocab_size, embed_dim)`.
    /// `token_ids` is an output reference with shape `(seq_len,)`.
    /// Output shape: `(seq_len, embed_dim)`.
    fn add_embedding_gather(&mut self, table: WeightId, token_ids: OutputRef) -> OutputRef;
}

impl<B: Backend + MatmulOps + EmbedOps> GraphEmbedOps for Graph<B> {
    fn add_embedding_gather(&mut self, table: WeightId, token_ids: OutputRef) -> OutputRef {
        let embed_dim = self.tensor_weight_meta(table).shape[1];
        let dtype = self.tensor_weight_meta(table).dtype;
        let node_id = self.add_node(
            Box::new(EmbeddingGatherOp {
                table,
                embed_dim,
                dtype,
            }),
            &[token_ids],
        );
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// NormOps
// ---------------------------------------------------------------------------

/// Graph builder methods for normalization operations.
pub trait GraphNormOps {
    /// RMS normalization. Output shape = input shape, same dtype.
    fn add_rms_norm(&mut self, input: OutputRef, weight: WeightId, eps: f32) -> OutputRef;

    /// Fused residual add + RMS norm.
    ///
    /// Returns `(updated_residual, normalized)`, both with the same
    /// shape and dtype as `residual`.
    fn add_add_rmsnorm(
        &mut self,
        residual: OutputRef,
        delta: OutputRef,
        weight: WeightId,
        eps: f32,
    ) -> (OutputRef, OutputRef);

    /// Per-head `RMSNorm` on `Q` and `K` before `RoPE` (`Qwen3`, `Gemma3`).
    ///
    /// `q` and `k` must have shape `[seq, num_heads, head_dim]`.
    /// Returns `(q_normed, k_normed)` with the same shapes as `q` and `k`.
    fn add_qk_norm(
        &mut self,
        q: OutputRef,
        k: OutputRef,
        q_weight: WeightId,
        k_weight: WeightId,
        eps: f32,
    ) -> (OutputRef, OutputRef);
}

impl<B: Backend + MatmulOps + NormOps> GraphNormOps for Graph<B> {
    fn add_rms_norm(&mut self, input: OutputRef, weight: WeightId, eps: f32) -> OutputRef {
        let node_id = self.add_node(Box::new(RmsNormOp { weight, eps }), &[input]);
        (node_id, 0)
    }

    fn add_add_rmsnorm(
        &mut self,
        residual: OutputRef,
        delta: OutputRef,
        weight: WeightId,
        eps: f32,
    ) -> (OutputRef, OutputRef) {
        let node_id = self.add_node(Box::new(AddRmsNormOp { weight, eps }), &[residual, delta]);
        ((node_id, 0), (node_id, 1))
    }

    fn add_qk_norm(
        &mut self,
        q: OutputRef,
        k: OutputRef,
        q_weight: WeightId,
        k_weight: WeightId,
        eps: f32,
    ) -> (OutputRef, OutputRef) {
        let node_id = self.add_node(
            Box::new(RmsNormQkOp {
                q_weight,
                k_weight,
                eps,
            }),
            &[q, k],
        );
        ((node_id, 0), (node_id, 1))
    }
}

// ---------------------------------------------------------------------------
// SoftcapOps
// ---------------------------------------------------------------------------

/// Graph builder methods for logit soft-capping.
pub trait GraphSoftcapOps {
    /// Element-wise logit soft-cap: `tanh(x / cap) * cap`.
    ///
    /// Output shape = input shape, same dtype (`F32`).
    fn add_logit_softcap(&mut self, input: OutputRef, cap: f32) -> OutputRef;
}

impl<B: Backend + MatmulOps> GraphSoftcapOps for Graph<B> {
    fn add_logit_softcap(&mut self, input: OutputRef, cap: f32) -> OutputRef {
        let node_id = self.add_node(Box::new(LogitSoftcapOp { cap }), &[input]);
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// MatmulOps
// ---------------------------------------------------------------------------

/// Graph builder methods for matrix multiplication and linear layers.
pub trait GraphMatmulOps {
    /// Linear projection. Input `[..., in_features]` → output `[..., out_features]`.
    ///
    /// `out_features` is inferred from `weight` metadata `shape[0]`.
    fn add_linear(&mut self, input: OutputRef, weight: WeightId) -> OutputRef;

    /// Paired linear projection from the same input.
    ///
    /// Returns `(out1, out2)` where each output has its own `out_features`
    /// inferred from the respective weight metadata.
    fn add_linear_pair(
        &mut self,
        input: OutputRef,
        w1: WeightId,
        w2: WeightId,
    ) -> (OutputRef, OutputRef);

    /// Triple linear projection from the same input (Q/K/V).
    ///
    /// Returns `(out1, out2, out3)`.
    fn add_linear_triple(
        &mut self,
        input: OutputRef,
        w1: WeightId,
        w2: WeightId,
        w3: WeightId,
    ) -> (OutputRef, OutputRef, OutputRef);

    /// Raw matrix multiplication. `[M, K] × [K, N] → [M, N]`.
    fn add_matmul(&mut self, a: OutputRef, b: OutputRef) -> OutputRef;
}

impl<B: Backend + MatmulOps> GraphMatmulOps for Graph<B> {
    fn add_linear(&mut self, input: OutputRef, weight: WeightId) -> OutputRef {
        let out_features = self.linear_weight_meta(weight).shape[0];
        let node_id = self.add_node(
            Box::new(LinearOp {
                weight,
                out_features,
            }),
            &[input],
        );
        (node_id, 0)
    }

    fn add_linear_pair(
        &mut self,
        input: OutputRef,
        w1: WeightId,
        w2: WeightId,
    ) -> (OutputRef, OutputRef) {
        let out1 = self.linear_weight_meta(w1).shape[0];
        let out2 = self.linear_weight_meta(w2).shape[0];
        let node_id = self.add_node(Box::new(LinearPairOp { w1, w2, out1, out2 }), &[input]);
        ((node_id, 0), (node_id, 1))
    }

    fn add_linear_triple(
        &mut self,
        input: OutputRef,
        w1: WeightId,
        w2: WeightId,
        w3: WeightId,
    ) -> (OutputRef, OutputRef, OutputRef) {
        let out1 = self.linear_weight_meta(w1).shape[0];
        let out2 = self.linear_weight_meta(w2).shape[0];
        let out3 = self.linear_weight_meta(w3).shape[0];
        let node_id = self.add_node(
            Box::new(LinearTripleOp {
                w1,
                w2,
                w3,
                out1,
                out2,
                out3,
            }),
            &[input],
        );
        ((node_id, 0), (node_id, 1), (node_id, 2))
    }

    fn add_matmul(&mut self, a: OutputRef, b: OutputRef) -> OutputRef {
        let node_id = self.add_node(Box::new(MatmulOp), &[a, b]);
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// MatmulExtOps
// ---------------------------------------------------------------------------

/// Graph builder methods for extended (mixed-precision) matmul.
pub trait GraphMatmulExtOps {
    /// Mixed-precision matmul: bf16 inputs → f32 output.
    ///
    /// `[M, K] × [K, N] → [M, N]` with dtype `F32`.
    fn add_matmul_bf16_f32(&mut self, a: OutputRef, b: OutputRef) -> OutputRef;
}

impl<B: Backend + MatmulOps + MatmulExtOps> GraphMatmulExtOps for Graph<B> {
    fn add_matmul_bf16_f32(&mut self, a: OutputRef, b: OutputRef) -> OutputRef {
        let node_id = self.add_node(Box::new(MatmulBf16F32Op), &[a, b]);
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// ArithOps
// ---------------------------------------------------------------------------

/// Graph builder methods for arithmetic operations.
pub trait GraphArithOps {
    /// Element-wise addition. Output shape = first input shape, same dtype.
    fn add_add(&mut self, a: OutputRef, b: OutputRef) -> OutputRef;

    /// In-place addition. Output shape = first input shape, same dtype.
    fn add_add_inplace(&mut self, a: OutputRef, b: OutputRef) -> OutputRef;

    /// Element-wise multiplication. Output shape = first input shape, same dtype.
    fn add_mul(&mut self, a: OutputRef, b: OutputRef) -> OutputRef;

    /// Uniform scalar scaling. Output shape = input shape, same dtype.
    fn add_scale(&mut self, input: OutputRef, factor: f32) -> OutputRef;
}

impl<B: Backend + MatmulOps + ArithOps> GraphArithOps for Graph<B> {
    fn add_add(&mut self, a: OutputRef, b: OutputRef) -> OutputRef {
        let node_id = self.add_node(Box::new(AddOp), &[a, b]);
        (node_id, 0)
    }

    fn add_add_inplace(&mut self, a: OutputRef, b: OutputRef) -> OutputRef {
        let node_id = self.add_node(Box::new(AddInplaceOp), &[a, b]);
        (node_id, 0)
    }

    fn add_mul(&mut self, a: OutputRef, b: OutputRef) -> OutputRef {
        let node_id = self.add_node(Box::new(MulOp), &[a, b]);
        (node_id, 0)
    }

    fn add_scale(&mut self, input: OutputRef, factor: f32) -> OutputRef {
        let node_id = self.add_node(Box::new(ScaleOp { factor }), &[input]);
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// BiasOps
// ---------------------------------------------------------------------------

/// Graph builder methods for bias addition.
pub trait GraphBiasOps {
    /// Bias addition. Output shape = input shape, same dtype.
    fn add_bias_add(&mut self, input: OutputRef, bias: WeightId) -> OutputRef;
}

impl<B: Backend + MatmulOps + BiasOps> GraphBiasOps for Graph<B> {
    fn add_bias_add(&mut self, input: OutputRef, bias: WeightId) -> OutputRef {
        let node_id = self.add_node(Box::new(BiasAddOp { bias }), &[input]);
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// SwigluOps
// ---------------------------------------------------------------------------

/// Graph builder methods for `SwiGLU` activation.
pub trait GraphSwigluOps {
    /// `silu(gate) * up`. Output shape = gate shape, same dtype.
    fn add_swiglu(&mut self, gate: OutputRef, up: OutputRef) -> OutputRef;
}

impl<B: Backend + MatmulOps + SwigluOps> GraphSwigluOps for Graph<B> {
    fn add_swiglu(&mut self, gate: OutputRef, up: OutputRef) -> OutputRef {
        let node_id = self.add_node(Box::new(SwigluOp), &[gate, up]);
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// GegluOps
// ---------------------------------------------------------------------------

/// Graph builder methods for `GeGLU` activation.
pub trait GraphGegluOps {
    /// `gelu(gate) * up`. Output shape = gate shape, same dtype.
    fn add_geglu(&mut self, gate: OutputRef, up: OutputRef) -> OutputRef;
}

impl<B: Backend + MatmulOps + GegluOps> GraphGegluOps for Graph<B> {
    fn add_geglu(&mut self, gate: OutputRef, up: OutputRef) -> OutputRef {
        let node_id = self.add_node(Box::new(GegluOp), &[gate, up]);
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// SiluOps (primitive — used for pre-fusion SiLU)
// ---------------------------------------------------------------------------

/// Graph builder methods for `SiLU` activation.
pub trait GraphSiluOps {
    /// `silu(input) = input * sigmoid(input)`. Output shape = input shape, same dtype.
    fn add_silu(&mut self, input: OutputRef) -> OutputRef;
}

impl<B: Backend + MatmulOps> GraphSiluOps for Graph<B> {
    fn add_silu(&mut self, input: OutputRef) -> OutputRef {
        let node_id = self.add_node(Box::new(SiluOp), &[input]);
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// CastOps
// ---------------------------------------------------------------------------

/// Graph builder methods for type-casting operations.
pub trait GraphCastOps {
    /// Cast to f32. Output shape = input shape, dtype = `F32`.
    fn add_cast_to_f32(&mut self, input: OutputRef) -> OutputRef;

    /// Cast from f32 to `target` dtype. Output shape = input shape.
    fn add_cast_from_f32(&mut self, input: OutputRef, target: DType) -> OutputRef;
}

impl<B: Backend + MatmulOps + CastOps> GraphCastOps for Graph<B> {
    fn add_cast_to_f32(&mut self, input: OutputRef) -> OutputRef {
        let node_id = self.add_node(Box::new(CastToF32Op), &[input]);
        (node_id, 0)
    }

    fn add_cast_from_f32(&mut self, input: OutputRef, target: DType) -> OutputRef {
        let node_id = self.add_node(Box::new(CastFromF32Op { target }), &[input]);
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// RopeOps
// ---------------------------------------------------------------------------

/// Graph builder methods for rotary positional embeddings (half-rotation).
pub trait GraphRopeOps {
    /// Apply `RoPE`. Output shape = input shape, same dtype.
    fn add_rope(
        &mut self,
        input: OutputRef,
        cos: OutputRef,
        sin: OutputRef,
        offset: usize,
    ) -> OutputRef;

    /// Batched `RoPE` with per-token positions.
    /// Output shape = input shape, same dtype.
    fn add_rope_batched(
        &mut self,
        input: OutputRef,
        cos: OutputRef,
        sin: OutputRef,
        positions: OutputRef,
        batch_size: usize,
    ) -> OutputRef;
}

impl<B: Backend + MatmulOps + RopeOps> GraphRopeOps for Graph<B> {
    fn add_rope(
        &mut self,
        input: OutputRef,
        cos: OutputRef,
        sin: OutputRef,
        offset: usize,
    ) -> OutputRef {
        let node_id = self.add_node(Box::new(RopeOp { offset }), &[input, cos, sin]);
        (node_id, 0)
    }

    fn add_rope_batched(
        &mut self,
        input: OutputRef,
        cos: OutputRef,
        sin: OutputRef,
        positions: OutputRef,
        batch_size: usize,
    ) -> OutputRef {
        let node_id = self.add_node(
            Box::new(RopeBatchedOp { batch_size }),
            &[input, cos, sin, positions],
        );
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// RopeInterleavedOps
// ---------------------------------------------------------------------------

/// Graph builder methods for interleaved `RoPE` (`DeepSeek`).
pub trait GraphRopeInterleavedOps {
    /// Apply interleaved `RoPE`. Output shape = input shape, same dtype.
    fn add_rope_interleaved(
        &mut self,
        input: OutputRef,
        cos: OutputRef,
        sin: OutputRef,
        offset: usize,
    ) -> OutputRef;
}

impl<B: Backend + MatmulOps + RopeInterleavedOps> GraphRopeInterleavedOps for Graph<B> {
    fn add_rope_interleaved(
        &mut self,
        input: OutputRef,
        cos: OutputRef,
        sin: OutputRef,
        offset: usize,
    ) -> OutputRef {
        let node_id = self.add_node(Box::new(RopeIntOp { offset }), &[input, cos, sin]);
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// AttentionOps
// ---------------------------------------------------------------------------

/// Graph builder methods for fused attention.
pub trait GraphAttentionOps {
    /// Fused causal attention for prefill. Output shape = Q shape, same dtype.
    #[allow(clippy::too_many_arguments)]
    fn add_fused_attention_prefill(
        &mut self,
        q: OutputRef,
        k: OutputRef,
        v: OutputRef,
        offset: usize,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> OutputRef;

    /// Fused attention for single-token decode. Output shape = Q shape, same dtype.
    fn add_fused_attention_decode(
        &mut self,
        q: OutputRef,
        k: OutputRef,
        v: OutputRef,
        softcap: Option<f32>,
    ) -> OutputRef;
}

impl<B: Backend + MatmulOps + AttentionOps> GraphAttentionOps for Graph<B> {
    fn add_fused_attention_prefill(
        &mut self,
        q: OutputRef,
        k: OutputRef,
        v: OutputRef,
        offset: usize,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> OutputRef {
        let node_id = self.add_node(
            Box::new(FusedAttentionPrefillOp {
                offset,
                scale,
                softcap,
                sliding_window,
            }),
            &[q, k, v],
        );
        (node_id, 0)
    }

    fn add_fused_attention_decode(
        &mut self,
        q: OutputRef,
        k: OutputRef,
        v: OutputRef,
        softcap: Option<f32>,
    ) -> OutputRef {
        let node_id = self.add_node(Box::new(FusedAttentionDecodeOp { softcap }), &[q, k, v]);
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// PagedAttentionOps
// ---------------------------------------------------------------------------

/// Graph builder methods for paged attention.
pub trait GraphPagedAttentionOps {
    /// Paged attention decode. Q shape is `(batch, num_heads, head_dim)`.
    /// Output shape = Q shape.
    #[allow(clippy::too_many_arguments)]
    fn add_paged_attention_decode(
        &mut self,
        q: OutputRef,
        block_tables: OutputRef,
        seq_lens: OutputRef,
        positions: OutputRef,
        layer_idx: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        sliding_window: Option<usize>,
    ) -> OutputRef;
}

impl<B: Backend + MatmulOps + PagedAttentionOps> GraphPagedAttentionOps for Graph<B> {
    fn add_paged_attention_decode(
        &mut self,
        q: OutputRef,
        block_tables: OutputRef,
        seq_lens: OutputRef,
        positions: OutputRef,
        layer_idx: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        sliding_window: Option<usize>,
    ) -> OutputRef {
        let node_id = self.add_node(
            Box::new(PagedAttentionDecodeOp {
                layer_idx,
                num_heads,
                num_kv_heads,
                head_dim,
                block_size,
                sliding_window,
            }),
            &[q, block_tables, seq_lens, positions],
        );
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// PagedKvCacheOps
// ---------------------------------------------------------------------------

/// Graph builder methods for paged KV cache management.
pub trait GraphPagedKvCacheOps {
    /// Append K/V to paged cache (single sequence, side-effect op).
    ///
    /// Returns a dummy `OutputRef` (the op produces no output tensors,
    /// but a node is created for scheduling).
    fn add_append_paged(
        &mut self,
        k: OutputRef,
        v: OutputRef,
        layer_idx: usize,
        start_pos: usize,
    ) -> OutputRef;

    /// Batched append K/V to paged cache (side-effect op).
    ///
    /// Returns a dummy `OutputRef`.
    fn add_append_paged_batched(
        &mut self,
        k: OutputRef,
        v: OutputRef,
        block_tables: OutputRef,
        seq_lens: OutputRef,
        layer_idx: usize,
    ) -> OutputRef;

    /// Gather K/V from paged cache.
    ///
    /// Returns `(k, v)` with shapes `(kv_len, num_kv_heads, head_dim)`.
    fn add_gather_paged_kv(
        &mut self,
        layer_idx: usize,
        kv_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> (OutputRef, OutputRef);
}

impl<B: Backend + MatmulOps + PagedKvCacheOps> GraphPagedKvCacheOps for Graph<B> {
    fn add_append_paged(
        &mut self,
        k: OutputRef,
        v: OutputRef,
        layer_idx: usize,
        start_pos: usize,
    ) -> OutputRef {
        let node_id = self.add_node(
            Box::new(AppendPagedOp {
                layer_idx,
                start_pos,
            }),
            &[k, v],
        );
        // Side-effect op with 0 outputs. Return (node_id, 0) as a
        // scheduling handle; there is no actual output tensor.
        (node_id, 0)
    }

    fn add_append_paged_batched(
        &mut self,
        k: OutputRef,
        v: OutputRef,
        block_tables: OutputRef,
        seq_lens: OutputRef,
        layer_idx: usize,
    ) -> OutputRef {
        let node_id = self.add_node(
            Box::new(AppendPagedBatchedOp { layer_idx }),
            &[k, v, block_tables, seq_lens],
        );
        (node_id, 0)
    }

    fn add_gather_paged_kv(
        &mut self,
        layer_idx: usize,
        kv_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> (OutputRef, OutputRef) {
        let node_id = self.add_node(
            Box::new(GatherPagedKvOp {
                layer_idx,
                kv_len,
                num_kv_heads,
                head_dim,
                dtype,
            }),
            &[],
        );
        ((node_id, 0), (node_id, 1))
    }
}

// ---------------------------------------------------------------------------
// TensorOps
// ---------------------------------------------------------------------------

/// Graph builder methods for tensor reshaping and manipulation.
pub trait GraphTensorOps {
    /// Reshape (zero-copy view). Output shape = provided shape.
    fn add_reshape(&mut self, input: OutputRef, shape: &[usize]) -> OutputRef;

    /// Split inner dimension. Input `[..., total]` →
    /// `([..., left_size], [..., total - left_size])`.
    fn add_split_inner_dim(&mut self, input: OutputRef, left_size: usize)
        -> (OutputRef, OutputRef);

    /// Concatenate along inner dimension.
    /// `[..., a_last] + [..., b_last]` → `[..., a_last + b_last]`.
    fn add_concat_inner_dim(&mut self, a: OutputRef, b: OutputRef) -> OutputRef;

    /// Concatenate along the first (sequence) dimension.
    /// `[a_seq, ...] + [b_seq, ...]` → `[a_seq + b_seq, ...]`.
    fn add_concat_seq(&mut self, a: OutputRef, b: OutputRef) -> OutputRef;

    /// Repeat KV heads. `(seq, heads, dim)` → `(seq, heads * num_repeats, dim)`.
    fn add_repeat_kv(&mut self, input: OutputRef, num_repeats: usize) -> OutputRef;

    /// Extract last row. `(seq_len, hidden)` → `(1, hidden)`.
    fn add_extract_last_row(&mut self, input: OutputRef, seq_len: usize) -> OutputRef;

    /// 2D transpose. `(M, N)` → `(N, M)`.
    fn add_transpose_2d(&mut self, input: OutputRef) -> OutputRef;
}

impl<B: Backend + MatmulOps + TensorOps> GraphTensorOps for Graph<B> {
    fn add_reshape(&mut self, input: OutputRef, shape: &[usize]) -> OutputRef {
        let node_id = self.add_node(
            Box::new(ReshapeOp {
                shape: shape.to_vec(),
            }),
            &[input],
        );
        (node_id, 0)
    }

    fn add_split_inner_dim(
        &mut self,
        input: OutputRef,
        left_size: usize,
    ) -> (OutputRef, OutputRef) {
        let node_id = self.add_node(Box::new(SplitInnerDimOp { left_size }), &[input]);
        ((node_id, 0), (node_id, 1))
    }

    fn add_concat_inner_dim(&mut self, a: OutputRef, b: OutputRef) -> OutputRef {
        let node_id = self.add_node(Box::new(ConcatInnerDimOp), &[a, b]);
        (node_id, 0)
    }

    fn add_concat_seq(&mut self, a: OutputRef, b: OutputRef) -> OutputRef {
        let node_id = self.add_node(Box::new(ConcatSeqOp), &[a, b]);
        (node_id, 0)
    }

    fn add_repeat_kv(&mut self, input: OutputRef, num_repeats: usize) -> OutputRef {
        let node_id = self.add_node(Box::new(RepeatKvOp { num_repeats }), &[input]);
        (node_id, 0)
    }

    fn add_extract_last_row(&mut self, input: OutputRef, seq_len: usize) -> OutputRef {
        let node_id = self.add_node(Box::new(ExtractLastRowOp { seq_len }), &[input]);
        (node_id, 0)
    }

    fn add_transpose_2d(&mut self, input: OutputRef) -> OutputRef {
        let node_id = self.add_node(Box::new(Transpose2dOp), &[input]);
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// Indirect decode ops (CUDA graph capture compatible)
// ---------------------------------------------------------------------------

/// Graph builder methods for CUDA-graph-compatible indirect decode operations.
///
/// Each op reads dynamically-changing values (token ID, sequence position)
/// from stable GPU device pointers (`SeqPosition`) rather than baking them
/// into the graph as literal constants. This allows the same captured
/// `cudaGraphExec_t` to be replayed across all decode steps with only the
/// values at those addresses changing between replays.
///
/// The ops are implemented only in the CUDA executor — calling
/// `OpNode::execute` on them panics. The KV cache buffers and cos/sin tables
/// are registered as tensor weights so their GPU addresses are stable.
pub trait GraphIndirectDecodeOps {
    /// Embedding table lookup that reads the token ID from a stable GPU pointer.
    ///
    /// Takes zero graph inputs — the token ID is provided out-of-band via the
    /// `SeqPosition` passed to the CUDA executor. The embedding table is
    /// referenced by `table` (a tensor weight). Output shape: `[1, embed_dim]`.
    fn add_embedding_gather_indirect(
        &mut self,
        table: WeightId,
        embed_dim: usize,
        dtype: DType,
    ) -> OutputRef;

    /// Apply `RoPE` reading the position from a stable GPU pointer.
    ///
    /// Takes one graph input (the Q or K tensor). The full cos/sin caches are
    /// registered as tensor weights (`cos_cache`, `sin_cache`). The current
    /// sequence position is read from the `SeqPosition` passed to the executor.
    /// Output shape = input shape.
    fn add_rope_indirect(
        &mut self,
        input: OutputRef,
        cos_cache: WeightId,
        sin_cache: WeightId,
        interleaved: bool,
        head_dim: usize,
        num_heads: usize,
    ) -> OutputRef;

    /// Append a new K or V token into a pre-allocated KV cache buffer.
    ///
    /// Takes one graph input (new K or V, shape `[1, kv_heads, head_dim]`).
    /// The write offset is derived from the `SeqPosition` pointer passed to the
    /// executor. This is a side-effect op (returns a dummy `OutputRef` — callers
    /// should not use the output as a graph input).
    fn add_append_kv_indirect(
        &mut self,
        input: OutputRef,
        layer_idx: usize,
        is_key: bool,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> OutputRef;

    /// Fused decode attention whose K/V caches and total sequence length are
    /// read from the executor's out-of-band `KvCache` (indexed by `layer_idx`).
    ///
    /// Takes one graph input: Q `[1, num_heads, head_dim]`. The K and V cache
    /// full buffers are fetched from `kv_cache.full_buffers(layer_idx)` at
    /// execution time — their GPU addresses are stable across all decode steps.
    /// Output shape: `[1, num_heads, head_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn add_fused_attention_decode_indirect(
        &mut self,
        q: OutputRef,
        layer_idx: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> OutputRef;

    /// Add a device-side argmax over the last dimension of a 2D logits tensor.
    ///
    /// Input: `[1, vocab_size]` logits (F32).  Output: `[1]` U32 token index.
    /// The result stays on the GPU — the caller reads it with a 4-byte D→H copy,
    /// which is far cheaper than transferring the full logits tensor.
    fn add_argmax_last(&mut self, logits: OutputRef) -> OutputRef;
}

impl<B: Backend + MatmulOps> GraphIndirectDecodeOps for Graph<B> {
    fn add_embedding_gather_indirect(
        &mut self,
        table: WeightId,
        embed_dim: usize,
        dtype: DType,
    ) -> OutputRef {
        use super::builtin_ops::EmbeddingGatherIndirectOp;
        let node_id = self.add_node(
            Box::new(EmbeddingGatherIndirectOp {
                table,
                embed_dim,
                dtype,
            }),
            &[],
        );
        (node_id, 0)
    }

    fn add_rope_indirect(
        &mut self,
        input: OutputRef,
        cos_cache: WeightId,
        sin_cache: WeightId,
        interleaved: bool,
        head_dim: usize,
        num_heads: usize,
    ) -> OutputRef {
        use super::builtin_ops::RopeIndirectOp;
        let node_id = self.add_node(
            Box::new(RopeIndirectOp {
                cos_cache,
                sin_cache,
                interleaved,
                head_dim,
                num_heads,
            }),
            &[input],
        );
        (node_id, 0)
    }

    fn add_append_kv_indirect(
        &mut self,
        input: OutputRef,
        layer_idx: usize,
        is_key: bool,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> OutputRef {
        use super::builtin_ops::AppendKvIndirectOp;
        let node_id = self.add_node(
            Box::new(AppendKvIndirectOp {
                layer_idx,
                is_key,
                num_kv_heads,
                head_dim,
            }),
            &[input],
        );
        (node_id, 0)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_fused_attention_decode_indirect(
        &mut self,
        q: OutputRef,
        layer_idx: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> OutputRef {
        use super::builtin_ops::FusedAttentionDecodeIndirectOp;
        let node_id = self.add_node(
            Box::new(FusedAttentionDecodeIndirectOp {
                layer_idx,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
                softcap,
                sliding_window,
            }),
            &[q],
        );
        (node_id, 0)
    }

    fn add_argmax_last(&mut self, logits: OutputRef) -> OutputRef {
        use super::builtin_ops::ArgmaxLastOp;
        let node_id = self.add_node(Box::new(ArgmaxLastOp), &[logits]);
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// MoeOps — softmax and sigmoid MoE dispatch
// ---------------------------------------------------------------------------

/// Graph builder methods for Mixture-of-Experts dispatch.
///
/// Both methods register the gate projection and all per-expert MLP weights,
/// then add a single `MoeDispatch*` node that the executor handles by calling
/// the backend's `moe_forward_*` closure-based dispatch.
pub trait GraphMoeOps {
    /// Softmax `MoE` routing (`Mixtral`, `Qwen3-MoE`).
    ///
    /// `input` must have shape `[seq_len, hidden_size]`.  Returns the
    /// weighted combination of expert outputs, same shape.
    ///
    /// # Arguments
    ///
    /// * `input` — hidden states before the `MoE` layer.
    /// * `gate` — Gate projection weight ID (shape `[num_experts, hidden_size]`).
    /// * `experts` — Per-expert MLP weight IDs.
    /// * `num_experts_per_tok` — Number of experts activated per token.
    /// * `norm_topk_prob` — Whether to renormalise top-k router probabilities.
    fn add_moe_dispatch_softmax(
        &mut self,
        input: OutputRef,
        gate: WeightId,
        experts: Vec<MoeExpertIds>,
        num_experts_per_tok: usize,
        norm_topk: bool,
    ) -> OutputRef;

    /// Sigmoid `MoE` routing with bias correction and grouped top-k (`DeepSeek`).
    ///
    /// `input` must have shape `[seq_len, hidden_size]`.  Returns the
    /// weighted combination of expert outputs, same shape.
    #[allow(clippy::too_many_arguments)]
    fn add_moe_dispatch_sigmoid(
        &mut self,
        input: OutputRef,
        gate: WeightId,
        bias: Option<WeightId>,
        experts: Vec<MoeExpertIds>,
        shared_expert: Option<MoeExpertIds>,
        num_experts_per_tok: usize,
        n_group: usize,
        topk_group: usize,
        routed_scaling_factor: f32,
    ) -> OutputRef;
}

impl<B: Backend + MatmulOps + MoeOps + MoeSigmoidOps> GraphMoeOps for Graph<B> {
    fn add_moe_dispatch_softmax(
        &mut self,
        input: OutputRef,
        gate: WeightId,
        experts: Vec<MoeExpertIds>,
        num_experts_per_tok: usize,
        norm_topk: bool,
    ) -> OutputRef {
        let node_id = self.add_node(
            Box::new(MoeDispatchSoftmaxOp {
                gate,
                experts,
                num_experts_per_tok,
                norm_topk,
            }),
            &[input],
        );
        (node_id, 0)
    }

    fn add_moe_dispatch_sigmoid(
        &mut self,
        input: OutputRef,
        gate: WeightId,
        bias: Option<WeightId>,
        experts: Vec<MoeExpertIds>,
        shared_expert: Option<MoeExpertIds>,
        num_experts_per_tok: usize,
        n_group: usize,
        topk_group: usize,
        routed_scaling_factor: f32,
    ) -> OutputRef {
        let node_id = self.add_node(
            Box::new(MoeDispatchSigmoidOp {
                gate,
                bias,
                experts,
                shared_expert,
                num_experts_per_tok,
                n_group,
                topk_group,
                routed_scaling_factor,
            }),
            &[input],
        );
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// MlaAttentionOps
// ---------------------------------------------------------------------------

/// Graph builder method for the MLA attention block (`DeepSeek` V3/R1).
pub trait GraphMlaAttentionOps {
    /// Add an opaque MLA attention node.
    ///
    /// `hidden` is the input tensor `[seq_len, hidden_size]`. The node
    /// encapsulates all MLA projections, interleaved `RoPE`, fused attention,
    /// and the output projection. The executor handles KV cache access.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::similar_names)]
    fn add_mla_attention(
        &mut self,
        hidden: OutputRef,
        q_a_proj: WeightId,
        q_a_layernorm: WeightId,
        q_b_proj: WeightId,
        kv_a_proj_with_mqa: WeightId,
        kv_a_layernorm: WeightId,
        kv_b_proj_k: WeightId,
        kv_b_proj_v: WeightId,
        kv_b_proj_k_t: WeightId,
        o_proj: WeightId,
        num_heads: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        kv_lora_rank: usize,
        rms_norm_eps: f32,
        attn_scale: f32,
        layer_idx: usize,
    ) -> OutputRef;
}

impl<B: Backend + MatmulOps + MlaAttentionOps> GraphMlaAttentionOps for Graph<B> {
    #[allow(clippy::similar_names)]
    fn add_mla_attention(
        &mut self,
        hidden: OutputRef,
        q_a_proj: WeightId,
        q_a_layernorm: WeightId,
        q_b_proj: WeightId,
        kv_a_proj_with_mqa: WeightId,
        kv_a_layernorm: WeightId,
        kv_b_proj_k: WeightId,
        kv_b_proj_v: WeightId,
        kv_b_proj_k_t: WeightId,
        o_proj: WeightId,
        num_heads: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        kv_lora_rank: usize,
        rms_norm_eps: f32,
        attn_scale: f32,
        layer_idx: usize,
    ) -> OutputRef {
        let node_id = self.add_node(
            Box::new(MlaAttentionOp {
                q_a_proj,
                q_a_layernorm,
                q_b_proj,
                kv_a_proj_with_mqa,
                kv_a_layernorm,
                kv_b_proj_k,
                kv_b_proj_v,
                kv_b_proj_k_t,
                o_proj,
                num_heads,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                kv_lora_rank,
                rms_norm_eps,
                attn_scale,
                layer_idx,
            }),
            &[hidden],
        );
        (node_id, 0)
    }
}

// ---------------------------------------------------------------------------
// LM Head — always available (unconditional on Graph<B>)
// ---------------------------------------------------------------------------

impl<B: Backend + MatmulOps> Graph<B> {
    /// Add an LM head projection: `hidden → logits`.
    ///
    /// Output shape: `(seq_len, vocab_size)` where `vocab_size` comes
    /// from the weight metadata `shape[0]`.
    pub fn add_lm_head(
        &mut self,
        input: OutputRef,
        weight: WeightId,
        weight_dtype: DType,
    ) -> OutputRef {
        let vocab_size = self.linear_weight_meta(weight).shape[0];
        let node_id = self.add_node(
            Box::new(LmHeadOp {
                weight,
                weight_dtype,
                vocab_size,
            }),
            &[input],
        );
        (node_id, 0)
    }
}
