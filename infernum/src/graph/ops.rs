//! Computation graph operations.
//!
//! Each `Op` variant describes a single operation. Op-specific parameters
//! (epsilon, head counts, etc.) are stored inline. Data flows through
//! `NodeId` edges in `GraphNode::inputs`.

use smallvec::SmallVec;

use crate::dtype::DType;

use super::node::{NodeId, WeightId};

/// Computation graph operation.
///
/// Each variant describes a single operation. Op-specific parameters
/// (epsilon, head counts, etc.) are stored inline. Data flows through
/// `NodeId` edges in `GraphNode::inputs`.
#[derive(Clone, Debug)]
pub enum Op {
    /// Graph input placeholder. Not a real computation.
    ///
    /// Inputs: none.
    /// Output: tensor with the shape/dtype declared at graph construction.
    Input,

    /// Embedding table lookup.
    ///
    /// Inputs: `(token_ids)`.
    /// Output: hidden states `(seq_len, hidden_size)`.
    EmbeddingGather {
        /// Weight ID of the embedding table.
        table: WeightId,
    },

    // --- Norm ---
    /// RMS normalization.
    ///
    /// Inputs: `(input)`.
    /// Output: normalized tensor with the same shape.
    RmsNorm {
        /// Weight ID of the normalization weight vector.
        weight: WeightId,
        /// Epsilon for numerical stability.
        eps: f32,
    },

    /// Fused residual add + RMS norm.
    ///
    /// Inputs: `(residual, delta)`.
    /// Outputs: `(updated_residual, normalized)` — produces TWO outputs.
    /// The graph allocates two consecutive `NodeId`s; the second is a
    /// `SecondOutput` pointing back to this node.
    AddRmsNorm {
        /// Weight ID of the normalization weight vector.
        weight: WeightId,
        /// Epsilon for numerical stability.
        eps: f32,
    },

    /// Second output of a multi-output op. Points back to the primary node.
    ///
    /// Inputs: none (shape/dtype copied from source).
    /// Output: the secondary output of the referenced multi-output node.
    SecondOutput {
        /// `NodeId` of the primary (first) output node.
        source: NodeId,
    },

    // --- Linear / Matmul ---
    /// Linear layer (may be quantized).
    ///
    /// Inputs: `(input)`.
    /// Output: projected tensor.
    Linear {
        /// Weight ID of the linear weight.
        weight: WeightId,
    },

    /// Paired linear projection from the same input.
    ///
    /// Inputs: `(input)`.
    /// Outputs: `(out1, out2)` — two consecutive `NodeId`s.
    LinearPair {
        /// Weight ID for the first projection.
        w1: WeightId,
        /// Weight ID for the second projection.
        w2: WeightId,
    },

    /// Triple linear projection from the same input (Q/K/V).
    ///
    /// Inputs: `(input)`.
    /// Outputs: `(out1, out2, out3)` — three consecutive `NodeId`s.
    LinearTriple {
        /// Weight ID for the first projection.
        w1: WeightId,
        /// Weight ID for the second projection.
        w2: WeightId,
        /// Weight ID for the third projection.
        w3: WeightId,
    },

    /// Raw matrix multiplication.
    ///
    /// Inputs: `(a, b)`.
    /// Output: `a @ b`.
    Matmul,

    /// Mixed-precision matmul: bf16 inputs → f32 output.
    ///
    /// Inputs: `(a_bf16, b_bf16)`.
    /// Output: f32 tensor.
    MatmulBf16F32,

    // --- Activations ---
    /// `SwiGLU` activation: `silu(gate) * up`.
    ///
    /// Inputs: `(gate, up)`.
    /// Output: activated tensor.
    Swiglu,

    /// `GeGLU` activation: `gelu(gate) * up`.
    ///
    /// Inputs: `(gate, up)`.
    /// Output: activated tensor.
    Geglu,

    // --- Arithmetic ---
    /// Element-wise addition.
    ///
    /// Inputs: `(a, b)`.
    /// Output: `a + b`.
    Add,

    /// In-place addition. Memory planner may alias input/output.
    ///
    /// Inputs: `(a, b)`.
    /// Output: `a += b` (logically returns `a`).
    AddInplace,

    /// Uniform scalar scale.
    ///
    /// Inputs: `(input)`.
    /// Output: `input * factor`.
    Scale {
        /// Scale factor.
        factor: f32,
    },

    /// Bias addition.
    ///
    /// Inputs: `(input)`.
    /// Output: `input + bias`.
    BiasAdd {
        /// Weight ID of the bias vector.
        bias: WeightId,
    },

    // --- RoPE ---
    /// Half-rotation `RoPE` (Llama/Qwen/Gemma).
    ///
    /// Inputs: `(input, cos_cache, sin_cache)`.
    /// Output: rotated tensor.
    Rope {
        /// Position offset within the sequence.
        offset: usize,
    },

    /// Batched `RoPE` with per-token positions.
    ///
    /// Inputs: `(input, cos_cache, sin_cache, positions)`.
    /// Output: rotated tensor.
    RopeBatched {
        /// Number of sequences in the batch.
        batch_size: usize,
    },

    /// Interleaved `RoPE` (`DeepSeek`).
    ///
    /// Inputs: `(input, cos_cache, sin_cache)`.
    /// Output: rotated tensor.
    RopeInterleaved {
        /// Position offset within the sequence.
        offset: usize,
    },

    // --- Attention ---
    /// Fused causal attention for prefill.
    ///
    /// Inputs: `(q, k, v)`.
    /// Output: attention output.
    FusedAttentionPrefill {
        /// Position offset for causal mask.
        offset: usize,
        /// Optional attention scale (default: `1/sqrt(head_dim)`).
        scale: Option<f32>,
        /// Optional logit soft-capping value.
        softcap: Option<f32>,
        /// Optional sliding window size.
        sliding_window: Option<usize>,
    },

    /// Fused attention for single-token decode.
    ///
    /// Inputs: `(q, k, v)`.
    /// Output: attention output.
    FusedAttentionDecode {
        /// Optional logit soft-capping value.
        softcap: Option<f32>,
    },

    /// Paged attention decode against block-structured KV cache.
    ///
    /// Inputs: `(q, block_tables, seq_lens, positions)`.
    /// Output: attention output. K/V are read from the external paged cache.
    PagedAttentionDecode {
        /// Layer index for KV cache lookup.
        layer_idx: usize,
        /// Number of query heads.
        num_heads: usize,
        /// Number of key/value heads (for GQA).
        num_kv_heads: usize,
        /// Head dimension.
        head_dim: usize,
        /// Block size of the paged KV cache.
        block_size: usize,
        /// Optional sliding window size.
        sliding_window: Option<usize>,
    },

    // --- KV Cache (side-effect: mutates external cache) ---
    /// Append K/V to paged cache for a single sequence.
    ///
    /// Inputs: `(k, v)`.
    /// Output: `()` (side-effect only).
    AppendPaged {
        /// Layer index for KV cache.
        layer_idx: usize,
        /// Start position in the sequence.
        start_pos: usize,
    },

    /// Batched append K/V to paged cache.
    ///
    /// Inputs: `(k, v, block_tables, seq_lens)`.
    /// Output: `()` (side-effect only).
    AppendPagedBatched {
        /// Layer index for KV cache.
        layer_idx: usize,
    },

    /// Gather K/V from paged cache.
    ///
    /// Inputs: none (reads from external cache).
    /// Outputs: `(k, v)` — two consecutive `NodeId`s.
    GatherPagedKv {
        /// Layer index for KV cache.
        layer_idx: usize,
    },

    // --- Shape / Data movement ---
    /// Reshape (zero-copy view).
    ///
    /// Inputs: `(input)`.
    /// Output: reshaped view with the given shape.
    Reshape {
        /// Target shape.
        shape: SmallVec<[usize; 4]>,
    },

    /// Sub-tensor view at a byte offset.
    ///
    /// Inputs: `(input)`.
    /// Output: view at the given offset with the given shape.
    SliceView {
        /// Byte offset into the source tensor.
        offset: usize,
        /// Shape of the resulting view.
        shape: SmallVec<[usize; 4]>,
    },

    /// 2D transpose.
    ///
    /// Inputs: `(input)`.
    /// Output: transposed tensor.
    Transpose2d,

    /// Split inner dimension into two parts.
    ///
    /// Inputs: `(input)`.
    /// Outputs: `(left, right)` — two consecutive `NodeId`s.
    SplitInnerDim {
        /// Size of the left (first) partition.
        left_size: usize,
    },

    /// Concatenate along inner dimension.
    ///
    /// Inputs: `(a, b)`.
    /// Output: concatenated tensor.
    ConcatInnerDim,

    /// Concatenate along the first (sequence) dimension.
    ///
    /// Inputs: `(a, b)`.
    /// Output: concatenated tensor. `[a_seq + b_seq, ...]`.
    ConcatSeq,

    /// Repeat KV heads for grouped-query attention.
    ///
    /// Inputs: `(input)`.
    /// Output: tensor with heads repeated `num_repeats` times.
    RepeatKv {
        /// Number of times to repeat each KV head.
        num_repeats: usize,
    },

    /// Extract the last row (prefill → decode transition).
    ///
    /// Inputs: `(input)`.
    /// Output: single-row tensor.
    ExtractLastRow {
        /// Sequence length (row count) of the input.
        seq_len: usize,
    },

    // --- Cast ---
    /// Cast to f32.
    ///
    /// Inputs: `(input)`.
    /// Output: f32 tensor.
    CastToF32,

    /// Cast from f32 to a target dtype.
    ///
    /// Inputs: `(input)`.
    /// Output: tensor in the target dtype.
    CastFromF32 {
        /// Target data type.
        target: DType,
    },

    // --- MoE ---
    /// Softmax `MoE` routing (Mixtral/Qwen).
    ///
    /// Inputs: `(hidden)`.
    /// Output: routed output. Routing is internal/opaque.
    MoeDispatchSoftmax {
        /// Weight ID of the gating projection.
        gate: WeightId,
        /// Per-expert weight IDs for the MLP.
        experts: Vec<MoeExpertIds>,
        /// Number of experts selected per token.
        num_experts_per_tok: usize,
        /// Whether to normalize top-k probabilities.
        norm_topk: bool,
    },

    /// Sigmoid `MoE` routing (`DeepSeek`).
    ///
    /// Inputs: `(hidden)`.
    /// Output: routed output. Routing is internal/opaque.
    MoeDispatchSigmoid {
        /// Weight ID of the gating projection.
        gate: WeightId,
        /// Optional bias weight ID for score correction.
        bias: Option<WeightId>,
        /// Per-expert weight IDs for the MLP.
        experts: Vec<MoeExpertIds>,
        /// Optional shared-expert weight IDs.
        shared_expert: Option<MoeExpertIds>,
        /// Number of experts selected per token.
        num_experts_per_tok: usize,
        /// Number of expert groups for grouped top-k.
        n_group: usize,
        /// Number of groups selected in top-k.
        topk_group: usize,
        /// Scaling factor for routed expert outputs.
        routed_scaling_factor: f32,
    },

    // --- Communication ---
    /// All-reduce sum across devices (tensor parallelism sync point).
    ///
    /// Inputs: `(input)`.
    /// Output: reduced tensor.
    AllReduceSum,

    // --- LM Head ---
    /// Final projection to vocabulary logits.
    ///
    /// Inputs: `(hidden)`.
    /// Output: logits tensor `(seq_len, vocab_size)`.
    /// Uses `matmul_bf16_f32` fast path when applicable.
    LmHead {
        /// Weight ID of the LM head projection.
        weight: WeightId,
        /// Data type of the weight (determines matmul path).
        dtype: DType,
    },
}

/// Weight IDs for a single `MoE` expert's MLP.
#[derive(Clone, Debug)]
pub struct MoeExpertIds {
    /// Gate projection weight ID.
    pub gate_proj: WeightId,
    /// Up projection weight ID.
    pub up_proj: WeightId,
    /// Down projection weight ID.
    pub down_proj: WeightId,
}
