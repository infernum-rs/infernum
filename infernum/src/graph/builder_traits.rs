//! Graph builder traits — conditionally implemented on `Graph<B>`
//! when the backend `B` implements the corresponding op trait.
//!
//! Each method performs shape inference from its inputs and pushes a
//! new node (or nodes) into the graph. This keeps model-building code
//! high-level while ensuring shapes are tracked at construction time.

use smallvec::SmallVec;

use crate::backend::{
    ArithOps, AttentionOps, Backend, BiasOps, CastOps, EmbedOps, GegluOps, MatmulExtOps, MatmulOps,
    NormOps, PagedAttentionOps, PagedKvCacheOps, RopeInterleavedOps, RopeOps, SwigluOps, TensorOps,
};
use crate::dtype::DType;

use super::builder::Graph;
use super::node::{NodeId, WeightId};
use super::ops::Op;

// ---------------------------------------------------------------------------
// EmbedOps
// ---------------------------------------------------------------------------

/// Graph builder methods for embedding operations.
pub trait GraphEmbedOps {
    /// Embedding table lookup.
    ///
    /// `table` is a tensor weight with shape `(vocab_size, embed_dim)`.
    /// `token_ids` is a node with shape `(seq_len,)`.
    /// Output shape: `(seq_len, embed_dim)`.
    fn add_embedding_gather(&mut self, table: WeightId, token_ids: NodeId) -> NodeId;
}

impl<B: Backend + EmbedOps> GraphEmbedOps for Graph<B> {
    fn add_embedding_gather(&mut self, table: WeightId, token_ids: NodeId) -> NodeId {
        let seq_len = self.node_shape(token_ids)[0];
        let embed_dim = self.tensor_weight_meta(table).shape[1];
        let dtype = self.tensor_weight_meta(table).dtype;
        self.push_node(
            Op::EmbeddingGather { table },
            &[token_ids],
            vec![seq_len, embed_dim],
            dtype,
        )
    }
}

// ---------------------------------------------------------------------------
// NormOps
// ---------------------------------------------------------------------------

/// Graph builder methods for normalization operations.
pub trait GraphNormOps {
    /// RMS normalization. Output shape = input shape, same dtype.
    fn add_rms_norm(&mut self, input: NodeId, weight: WeightId, eps: f32) -> NodeId;

    /// Fused residual add + RMS norm.
    ///
    /// Returns `(updated_residual, normalized)`, both with the same
    /// shape and dtype as `residual`.
    fn add_add_rmsnorm(
        &mut self,
        residual: NodeId,
        delta: NodeId,
        weight: WeightId,
        eps: f32,
    ) -> (NodeId, NodeId);
}

impl<B: Backend + NormOps> GraphNormOps for Graph<B> {
    fn add_rms_norm(&mut self, input: NodeId, weight: WeightId, eps: f32) -> NodeId {
        let shape = self.node_shape(input).to_vec();
        let dtype = self.node_dtype(input);
        self.push_node(Op::RmsNorm { weight, eps }, &[input], shape, dtype)
    }

    fn add_add_rmsnorm(
        &mut self,
        residual: NodeId,
        delta: NodeId,
        weight: WeightId,
        eps: f32,
    ) -> (NodeId, NodeId) {
        let shape = self.node_shape(residual).to_vec();
        let dtype = self.node_dtype(residual);
        self.push_node_pair(
            Op::AddRmsNorm { weight, eps },
            &[residual, delta],
            shape.clone(),
            dtype,
            shape,
            dtype,
        )
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
    fn add_linear(&mut self, input: NodeId, weight: WeightId) -> NodeId;

    /// Paired linear projection from the same input.
    ///
    /// Returns `(out1, out2)` where each output has its own `out_features`
    /// inferred from the respective weight metadata.
    fn add_linear_pair(&mut self, input: NodeId, w1: WeightId, w2: WeightId) -> (NodeId, NodeId);

    /// Triple linear projection from the same input (Q/K/V).
    ///
    /// Returns `(out1, out2, out3)`.
    fn add_linear_triple(
        &mut self,
        input: NodeId,
        w1: WeightId,
        w2: WeightId,
        w3: WeightId,
    ) -> (NodeId, NodeId, NodeId);

    /// Raw matrix multiplication. `[M, K] × [K, N] → [M, N]`.
    fn add_matmul(&mut self, a: NodeId, b: NodeId) -> NodeId;
}

impl<B: Backend + MatmulOps> GraphMatmulOps for Graph<B> {
    fn add_linear(&mut self, input: NodeId, weight: WeightId) -> NodeId {
        let in_shape = self.node_shape(input).to_vec();
        let dtype = self.node_dtype(input);
        let out_features = self.linear_weight_meta(weight).shape[0];
        let mut out_shape = in_shape;
        *out_shape
            .last_mut()
            .expect("input must have at least one dimension") = out_features;
        self.push_node(Op::Linear { weight }, &[input], out_shape, dtype)
    }

    fn add_linear_pair(&mut self, input: NodeId, w1: WeightId, w2: WeightId) -> (NodeId, NodeId) {
        let in_shape = self.node_shape(input).to_vec();
        let dtype = self.node_dtype(input);
        let out1 = self.linear_weight_meta(w1).shape[0];
        let out2 = self.linear_weight_meta(w2).shape[0];
        let mut shape1 = in_shape.clone();
        *shape1
            .last_mut()
            .expect("input must have at least one dimension") = out1;
        let mut shape2 = in_shape;
        *shape2
            .last_mut()
            .expect("input must have at least one dimension") = out2;
        self.push_node_pair(
            Op::LinearPair { w1, w2 },
            &[input],
            shape1,
            dtype,
            shape2,
            dtype,
        )
    }

    fn add_linear_triple(
        &mut self,
        input: NodeId,
        w1: WeightId,
        w2: WeightId,
        w3: WeightId,
    ) -> (NodeId, NodeId, NodeId) {
        let in_shape = self.node_shape(input).to_vec();
        let dtype = self.node_dtype(input);
        let o1 = self.linear_weight_meta(w1).shape[0];
        let o2 = self.linear_weight_meta(w2).shape[0];
        let o3 = self.linear_weight_meta(w3).shape[0];
        let mut s1 = in_shape.clone();
        *s1.last_mut()
            .expect("input must have at least one dimension") = o1;
        let mut s2 = in_shape.clone();
        *s2.last_mut()
            .expect("input must have at least one dimension") = o2;
        let mut s3 = in_shape;
        *s3.last_mut()
            .expect("input must have at least one dimension") = o3;
        self.push_node_triple(
            Op::LinearTriple { w1, w2, w3 },
            &[input],
            s1,
            dtype,
            s2,
            dtype,
            s3,
            dtype,
        )
    }

    fn add_matmul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let a_shape = self.node_shape(a);
        let b_shape = self.node_shape(b);
        let m = a_shape[0];
        let n = b_shape[1];
        let dtype = self.node_dtype(a);
        self.push_node(Op::Matmul, &[a, b], vec![m, n], dtype)
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
    fn add_matmul_bf16_f32(&mut self, a: NodeId, b: NodeId) -> NodeId;
}

impl<B: Backend + MatmulExtOps> GraphMatmulExtOps for Graph<B> {
    fn add_matmul_bf16_f32(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let a_shape = self.node_shape(a);
        let b_shape = self.node_shape(b);
        let m = a_shape[0];
        let n = b_shape[1];
        self.push_node(Op::MatmulBf16F32, &[a, b], vec![m, n], DType::F32)
    }
}

// ---------------------------------------------------------------------------
// ArithOps
// ---------------------------------------------------------------------------

/// Graph builder methods for arithmetic operations.
pub trait GraphArithOps {
    /// Element-wise addition. Output shape = first input shape, same dtype.
    fn add_add(&mut self, a: NodeId, b: NodeId) -> NodeId;

    /// In-place addition. Output shape = first input shape, same dtype.
    fn add_add_inplace(&mut self, a: NodeId, b: NodeId) -> NodeId;

    /// Element-wise multiplication. Output shape = first input shape, same dtype.
    fn add_mul(&mut self, a: NodeId, b: NodeId) -> NodeId;

    /// Uniform scalar scaling. Output shape = input shape, same dtype.
    fn add_scale(&mut self, input: NodeId, factor: f32) -> NodeId;
}

impl<B: Backend + ArithOps> GraphArithOps for Graph<B> {
    fn add_add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let shape = self.node_shape(a).to_vec();
        let dtype = self.node_dtype(a);
        self.push_node(Op::Add, &[a, b], shape, dtype)
    }

    fn add_add_inplace(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let shape = self.node_shape(a).to_vec();
        let dtype = self.node_dtype(a);
        self.push_node(Op::AddInplace, &[a, b], shape, dtype)
    }

    fn add_mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let shape = self.node_shape(a).to_vec();
        let dtype = self.node_dtype(a);
        self.push_node(Op::Mul, &[a, b], shape, dtype)
    }

    fn add_scale(&mut self, input: NodeId, factor: f32) -> NodeId {
        let shape = self.node_shape(input).to_vec();
        let dtype = self.node_dtype(input);
        self.push_node(Op::Scale { factor }, &[input], shape, dtype)
    }
}

// ---------------------------------------------------------------------------
// BiasOps
// ---------------------------------------------------------------------------

/// Graph builder methods for bias addition.
pub trait GraphBiasOps {
    /// Bias addition. Output shape = input shape, same dtype.
    fn add_bias_add(&mut self, input: NodeId, bias: WeightId) -> NodeId;
}

impl<B: Backend + BiasOps> GraphBiasOps for Graph<B> {
    fn add_bias_add(&mut self, input: NodeId, bias: WeightId) -> NodeId {
        let shape = self.node_shape(input).to_vec();
        let dtype = self.node_dtype(input);
        self.push_node(Op::BiasAdd { bias }, &[input], shape, dtype)
    }
}

// ---------------------------------------------------------------------------
// SwigluOps
// ---------------------------------------------------------------------------

/// Graph builder methods for `SwiGLU` activation.
pub trait GraphSwigluOps {
    /// `silu(gate) * up`. Output shape = gate shape, same dtype.
    fn add_swiglu(&mut self, gate: NodeId, up: NodeId) -> NodeId;
}

impl<B: Backend + SwigluOps> GraphSwigluOps for Graph<B> {
    fn add_swiglu(&mut self, gate: NodeId, up: NodeId) -> NodeId {
        let shape = self.node_shape(gate).to_vec();
        let dtype = self.node_dtype(gate);
        self.push_node(Op::Swiglu, &[gate, up], shape, dtype)
    }
}

// ---------------------------------------------------------------------------
// GegluOps
// ---------------------------------------------------------------------------

/// Graph builder methods for `GeGLU` activation.
pub trait GraphGegluOps {
    /// `gelu(gate) * up`. Output shape = gate shape, same dtype.
    fn add_geglu(&mut self, gate: NodeId, up: NodeId) -> NodeId;
}

impl<B: Backend + GegluOps> GraphGegluOps for Graph<B> {
    fn add_geglu(&mut self, gate: NodeId, up: NodeId) -> NodeId {
        let shape = self.node_shape(gate).to_vec();
        let dtype = self.node_dtype(gate);
        self.push_node(Op::Geglu, &[gate, up], shape, dtype)
    }
}

// ---------------------------------------------------------------------------
// SiluOps (primitive — no backend trait bound needed)
// ---------------------------------------------------------------------------

/// Graph builder methods for `SiLU` activation.
pub trait GraphSiluOps {
    /// `silu(input) = input * sigmoid(input)`. Output shape = input shape, same dtype.
    fn add_silu(&mut self, input: NodeId) -> NodeId;
}

impl<B: Backend> GraphSiluOps for Graph<B> {
    fn add_silu(&mut self, input: NodeId) -> NodeId {
        let shape = self.node_shape(input).to_vec();
        let dtype = self.node_dtype(input);
        self.push_node(Op::Silu, &[input], shape, dtype)
    }
}

// ---------------------------------------------------------------------------
// CastOps
// ---------------------------------------------------------------------------

/// Graph builder methods for type-casting operations.
pub trait GraphCastOps {
    /// Cast to f32. Output shape = input shape, dtype = `F32`.
    fn add_cast_to_f32(&mut self, input: NodeId) -> NodeId;

    /// Cast from f32 to `target` dtype. Output shape = input shape.
    fn add_cast_from_f32(&mut self, input: NodeId, target: DType) -> NodeId;
}

impl<B: Backend + CastOps> GraphCastOps for Graph<B> {
    fn add_cast_to_f32(&mut self, input: NodeId) -> NodeId {
        let shape = self.node_shape(input).to_vec();
        self.push_node(Op::CastToF32, &[input], shape, DType::F32)
    }

    fn add_cast_from_f32(&mut self, input: NodeId, target: DType) -> NodeId {
        let shape = self.node_shape(input).to_vec();
        self.push_node(Op::CastFromF32 { target }, &[input], shape, target)
    }
}

// ---------------------------------------------------------------------------
// RopeOps
// ---------------------------------------------------------------------------

/// Graph builder methods for rotary positional embeddings (half-rotation).
pub trait GraphRopeOps {
    /// Apply `RoPE`. Output shape = input shape, same dtype.
    fn add_rope(&mut self, input: NodeId, cos: NodeId, sin: NodeId, offset: usize) -> NodeId;

    /// Batched `RoPE` with per-token positions.
    /// Output shape = input shape, same dtype.
    fn add_rope_batched(
        &mut self,
        input: NodeId,
        cos: NodeId,
        sin: NodeId,
        positions: NodeId,
        batch_size: usize,
    ) -> NodeId;
}

impl<B: Backend + RopeOps> GraphRopeOps for Graph<B> {
    fn add_rope(&mut self, input: NodeId, cos: NodeId, sin: NodeId, offset: usize) -> NodeId {
        let shape = self.node_shape(input).to_vec();
        let dtype = self.node_dtype(input);
        self.push_node(Op::Rope { offset }, &[input, cos, sin], shape, dtype)
    }

    fn add_rope_batched(
        &mut self,
        input: NodeId,
        cos: NodeId,
        sin: NodeId,
        positions: NodeId,
        batch_size: usize,
    ) -> NodeId {
        let shape = self.node_shape(input).to_vec();
        let dtype = self.node_dtype(input);
        self.push_node(
            Op::RopeBatched { batch_size },
            &[input, cos, sin, positions],
            shape,
            dtype,
        )
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
        input: NodeId,
        cos: NodeId,
        sin: NodeId,
        offset: usize,
    ) -> NodeId;
}

impl<B: Backend + RopeInterleavedOps> GraphRopeInterleavedOps for Graph<B> {
    fn add_rope_interleaved(
        &mut self,
        input: NodeId,
        cos: NodeId,
        sin: NodeId,
        offset: usize,
    ) -> NodeId {
        let shape = self.node_shape(input).to_vec();
        let dtype = self.node_dtype(input);
        self.push_node(
            Op::RopeInterleaved { offset },
            &[input, cos, sin],
            shape,
            dtype,
        )
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
        q: NodeId,
        k: NodeId,
        v: NodeId,
        offset: usize,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> NodeId;

    /// Fused attention for single-token decode. Output shape = Q shape, same dtype.
    fn add_fused_attention_decode(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        softcap: Option<f32>,
    ) -> NodeId;
}

impl<B: Backend + AttentionOps> GraphAttentionOps for Graph<B> {
    fn add_fused_attention_prefill(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        offset: usize,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> NodeId {
        let shape = self.node_shape(q).to_vec();
        let dtype = self.node_dtype(q);
        self.push_node(
            Op::FusedAttentionPrefill {
                offset,
                scale,
                softcap,
                sliding_window,
            },
            &[q, k, v],
            shape,
            dtype,
        )
    }

    fn add_fused_attention_decode(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        softcap: Option<f32>,
    ) -> NodeId {
        let shape = self.node_shape(q).to_vec();
        let dtype = self.node_dtype(q);
        self.push_node(
            Op::FusedAttentionDecode { softcap },
            &[q, k, v],
            shape,
            dtype,
        )
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
        q: NodeId,
        block_tables: NodeId,
        seq_lens: NodeId,
        positions: NodeId,
        layer_idx: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        sliding_window: Option<usize>,
    ) -> NodeId;
}

impl<B: Backend + PagedAttentionOps> GraphPagedAttentionOps for Graph<B> {
    fn add_paged_attention_decode(
        &mut self,
        q: NodeId,
        block_tables: NodeId,
        seq_lens: NodeId,
        positions: NodeId,
        layer_idx: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        sliding_window: Option<usize>,
    ) -> NodeId {
        let shape = self.node_shape(q).to_vec();
        let dtype = self.node_dtype(q);
        self.push_node(
            Op::PagedAttentionDecode {
                layer_idx,
                num_heads,
                num_kv_heads,
                head_dim,
                block_size,
                sliding_window,
            },
            &[q, block_tables, seq_lens, positions],
            shape,
            dtype,
        )
    }
}

// ---------------------------------------------------------------------------
// PagedKvCacheOps
// ---------------------------------------------------------------------------

/// Graph builder methods for paged KV cache management.
pub trait GraphPagedKvCacheOps {
    /// Append K/V to paged cache (single sequence, side-effect op).
    ///
    /// Output is a placeholder node with shape `[0]`.
    fn add_append_paged(
        &mut self,
        k: NodeId,
        v: NodeId,
        layer_idx: usize,
        start_pos: usize,
    ) -> NodeId;

    /// Batched append K/V to paged cache (side-effect op).
    ///
    /// Output is a placeholder node with shape `[0]`.
    fn add_append_paged_batched(
        &mut self,
        k: NodeId,
        v: NodeId,
        block_tables: NodeId,
        seq_lens: NodeId,
        layer_idx: usize,
    ) -> NodeId;

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
    ) -> (NodeId, NodeId);
}

impl<B: Backend + PagedKvCacheOps> GraphPagedKvCacheOps for Graph<B> {
    fn add_append_paged(
        &mut self,
        k: NodeId,
        v: NodeId,
        layer_idx: usize,
        start_pos: usize,
    ) -> NodeId {
        let dtype = self.node_dtype(k);
        self.push_node(
            Op::AppendPaged {
                layer_idx,
                start_pos,
            },
            &[k, v],
            vec![0],
            dtype,
        )
    }

    fn add_append_paged_batched(
        &mut self,
        k: NodeId,
        v: NodeId,
        block_tables: NodeId,
        seq_lens: NodeId,
        layer_idx: usize,
    ) -> NodeId {
        let dtype = self.node_dtype(k);
        self.push_node(
            Op::AppendPagedBatched { layer_idx },
            &[k, v, block_tables, seq_lens],
            vec![0],
            dtype,
        )
    }

    fn add_gather_paged_kv(
        &mut self,
        layer_idx: usize,
        kv_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> (NodeId, NodeId) {
        let shape = vec![kv_len, num_kv_heads, head_dim];
        self.push_node_pair(
            Op::GatherPagedKv { layer_idx },
            &[],
            shape.clone(),
            dtype,
            shape,
            dtype,
        )
    }
}

// ---------------------------------------------------------------------------
// TensorOps
// ---------------------------------------------------------------------------

/// Graph builder methods for tensor reshaping and manipulation.
pub trait GraphTensorOps {
    /// Reshape (zero-copy view). Output shape = provided shape.
    fn add_reshape(&mut self, input: NodeId, shape: &[usize]) -> NodeId;

    /// Split inner dimension. Input `[..., total]` →
    /// `([..., left_size], [..., total - left_size])`.
    fn add_split_inner_dim(&mut self, input: NodeId, left_size: usize) -> (NodeId, NodeId);

    /// Concatenate along inner dimension.
    /// `[..., a_last] + [..., b_last]` → `[..., a_last + b_last]`.
    fn add_concat_inner_dim(&mut self, a: NodeId, b: NodeId) -> NodeId;

    /// Concatenate along the first (sequence) dimension.
    /// `[a_seq, ...] + [b_seq, ...]` → `[a_seq + b_seq, ...]`.
    fn add_concat_seq(&mut self, a: NodeId, b: NodeId) -> NodeId;

    /// Repeat KV heads. `(seq, heads, dim)` → `(seq, heads * num_repeats, dim)`.
    fn add_repeat_kv(&mut self, input: NodeId, num_repeats: usize) -> NodeId;

    /// Extract last row. `(seq_len, hidden)` → `(1, hidden)`.
    fn add_extract_last_row(&mut self, input: NodeId, seq_len: usize) -> NodeId;

    /// 2D transpose. `(M, N)` → `(N, M)`.
    fn add_transpose_2d(&mut self, input: NodeId) -> NodeId;
}

impl<B: Backend + TensorOps> GraphTensorOps for Graph<B> {
    fn add_reshape(&mut self, input: NodeId, shape: &[usize]) -> NodeId {
        let dtype = self.node_dtype(input);
        self.push_node(
            Op::Reshape {
                shape: SmallVec::from_slice(shape),
            },
            &[input],
            shape.to_vec(),
            dtype,
        )
    }

    fn add_split_inner_dim(&mut self, input: NodeId, left_size: usize) -> (NodeId, NodeId) {
        let in_shape = self.node_shape(input).to_vec();
        let dtype = self.node_dtype(input);
        let total = *in_shape
            .last()
            .expect("input must have at least one dimension");
        let right_size = total - left_size;

        let mut shape1 = in_shape.clone();
        *shape1.last_mut().unwrap() = left_size;
        let mut shape2 = in_shape;
        *shape2.last_mut().unwrap() = right_size;

        self.push_node_pair(
            Op::SplitInnerDim { left_size },
            &[input],
            shape1,
            dtype,
            shape2,
            dtype,
        )
    }

    fn add_concat_inner_dim(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let a_shape = self.node_shape(a).to_vec();
        let b_last = *self
            .node_shape(b)
            .last()
            .expect("b must have at least one dimension");
        let dtype = self.node_dtype(a);

        let mut out_shape = a_shape;
        *out_shape.last_mut().unwrap() += b_last;

        self.push_node(Op::ConcatInnerDim, &[a, b], out_shape, dtype)
    }

    fn add_concat_seq(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let a_shape = self.node_shape(a).to_vec();
        let b_first = self.node_shape(b)[0];
        let dtype = self.node_dtype(a);

        let mut out_shape = a_shape;
        out_shape[0] += b_first;

        self.push_node(Op::ConcatSeq, &[a, b], out_shape, dtype)
    }

    fn add_repeat_kv(&mut self, input: NodeId, num_repeats: usize) -> NodeId {
        let in_shape = self.node_shape(input);
        // Input: (seq, heads, dim)
        let seq = in_shape[0];
        let heads = in_shape[1];
        let dim = in_shape[2];
        let dtype = self.node_dtype(input);
        self.push_node(
            Op::RepeatKv { num_repeats },
            &[input],
            vec![seq, heads * num_repeats, dim],
            dtype,
        )
    }

    fn add_extract_last_row(&mut self, input: NodeId, seq_len: usize) -> NodeId {
        let in_shape = self.node_shape(input);
        let hidden = in_shape[1];
        let dtype = self.node_dtype(input);
        self.push_node(
            Op::ExtractLastRow { seq_len },
            &[input],
            vec![1, hidden],
            dtype,
        )
    }

    fn add_transpose_2d(&mut self, input: NodeId) -> NodeId {
        let in_shape = self.node_shape(input);
        let m = in_shape[0];
        let n = in_shape[1];
        let dtype = self.node_dtype(input);
        self.push_node(Op::Transpose2d, &[input], vec![n, m], dtype)
    }
}

// ---------------------------------------------------------------------------
// LM Head — always available (unconditional on Graph<B>)
// ---------------------------------------------------------------------------

impl<B: Backend> Graph<B> {
    /// Add an LM head projection: `hidden → logits`.
    ///
    /// Output shape: `(seq_len, vocab_size)` where `vocab_size` comes
    /// from the weight metadata `shape[0]`.
    pub fn add_lm_head(&mut self, input: NodeId, weight: WeightId, weight_dtype: DType) -> NodeId {
        let in_shape = self.node_shape(input).to_vec();
        let vocab_size = self.linear_weight_meta(weight).shape[0];
        let seq_len = in_shape[0];
        self.push_node(
            Op::LmHead {
                weight,
                dtype: weight_dtype,
            },
            &[input],
            vec![seq_len, vocab_size],
            DType::F32,
        )
    }
}
