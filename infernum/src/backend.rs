//! Backend trait and op trait definitions for hardware-agnostic inference.
//!
//! Models are generic over `B: Backend`, and use op traits like `ArithOps`,
//! `MatmulOps`, etc. to express their compute requirements. Each backend
//! (CUDA, CPU, Metal) implements these traits with its own tensor type.
//!
//! # Design notes
//!
//! - **Op traits extend `Backend`** — they use `Self::Tensor` from the
//!   supertrait, avoiding repetition.
//! - **No `Context` in op traits.** Backend-specific context (e.g.,
//!   `CudaContext` with cuBLAS handles) is held by models internally. Ops
//!   that require allocation context (like `embedding_gather`) are called
//!   through the concrete backend, not through these traits.
//! - **`LinearWeight` lives on `MatmulOps`** because only matmul-related
//!   code needs it.
//! - **Activations are split** into `SwigluOps` and `GegluOps`. Models
//!   specify exactly which they need via where-clauses.

use crate::block_allocator::BlockTable;
use crate::logits::Logits;
use crate::runtime_state::RuntimeStateInit;
use crate::tensor::Tensor;
use crate::DType;
use crate::Result;

// ---- Core backend trait ----

/// A compute backend (CUDA, CPU, Metal, etc.).
pub trait Backend: 'static {
    /// The tensor type for this backend (e.g., `CudaTensor`).
    type Tensor: Tensor + Clone;

    /// Paged KV cache — block-based cache used by most attention mechanisms.
    /// Models that use standard multi-head or grouped-query attention
    /// set their `Model::KvCache` to this type.
    type PagedKvCache: Send;

    /// Contiguous KV cache — non-paged cache used by specialised attention
    /// like DeepSeek's MLA. Models that need both paged and contiguous
    /// caches compose them in their `Model::KvCache` associated type.
    type KvCache: Send;

    /// Opaque runtime state managed by the backend.
    ///
    /// Holds backend-specific optimisation state that persists across
    /// forward calls (e.g., CUDA graph capture/replay state, buffer
    /// pools). The engine allocates it via `RuntimeStateInit::new()`
    /// and passes `&mut` into forward calls — it never inspects the
    /// contents.
    type RuntimeState: RuntimeStateInit;

    /// Backend-specific logits type returned by forward passes.
    /// Must implement the `Logits` trait so the engine can sample from it.
    type Logits: Logits;
}

// ---- Op traits ----

/// Core tensor arithmetic.
pub trait ArithOps: Backend {
    /// Element-wise addition, returning a new tensor.
    fn add(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;

    /// Element-wise in-place addition: `a += b`.
    fn add_inplace(a: &mut Self::Tensor, b: &Self::Tensor) -> Result<()>;

    /// Element-wise multiplication, returning a new tensor.
    fn mul(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;

    /// In-place scalar scaling: `a *= scale`.
    fn scale_inplace(a: &mut Self::Tensor, scale: f32) -> Result<()>;
}

/// Matrix multiplication and linear layers.
pub trait MatmulOps: Backend {
    /// Backend-specific linear weight type (dense, quantized, etc.).
    type LinearWeight;

    /// General matrix multiplication.
    fn matmul(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;

    /// Linear layer: `input @ weight` (handles dense and quantized).
    fn linear(input: &Self::Tensor, weight: &Self::LinearWeight) -> Result<Self::Tensor>;
}

/// Normalization operations.
pub trait NormOps: Backend {
    /// RMS normalization, returning a new tensor.
    fn rms_norm(input: &Self::Tensor, weight: &Self::Tensor, eps: f32) -> Result<Self::Tensor>;

    /// RMS normalization in-place.
    fn rms_norm_inplace(input: &mut Self::Tensor, weight: &Self::Tensor, eps: f32) -> Result<()>;

    /// Fused residual add + RMS norm. Returns `(updated_residual, normalized)`.
    fn add_rmsnorm(
        residual: &Self::Tensor,
        input: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
    ) -> Result<(Self::Tensor, Self::Tensor)>;
}

/// SwiGLU activation (Llama, Qwen, DeepSeek).
pub trait SwigluOps: Backend {
    /// `silu(gate) * up` — fused SwiGLU.
    fn swiglu(gate: &Self::Tensor, up: &Self::Tensor) -> Result<Self::Tensor>;
}

/// GeGLU activation (Gemma).
pub trait GegluOps: Backend {
    /// `gelu(gate) * up` — fused GeGLU.
    fn geglu(gate: &Self::Tensor, up: &Self::Tensor) -> Result<Self::Tensor>;
}

/// Type-casting operations.
pub trait CastOps: Backend {
    /// Cast tensor to f32.
    fn cast_to_f32(input: &Self::Tensor) -> Result<Self::Tensor>;

    /// Cast tensor from f32 to the target dtype.
    fn cast_from_f32(input: &Self::Tensor, target: DType) -> Result<Self::Tensor>;
}

/// Embedding lookup.
pub trait EmbedOps: Backend {
    /// Gather rows from an embedding table by token IDs.
    ///
    /// `table` has shape `(vocab_size, hidden_size)`.
    /// Returns a tensor of shape `(indices.len(), hidden_size)`.
    fn embedding_gather(table: &Self::Tensor, indices: &[u32]) -> Result<Self::Tensor>;
}

/// Bias addition.
pub trait BiasOps: Backend {
    /// In-place bias addition: `input[row, col] += bias[col]`.
    fn bias_add_inplace(input: &mut Self::Tensor, bias: &Self::Tensor) -> Result<()>;
}

/// Tensor reshaping and manipulation.
pub trait TensorOps: Backend {
    /// Transpose a 2D tensor.
    fn transpose_2d(input: &Self::Tensor) -> Result<Self::Tensor>;

    /// Split the inner dimension of a 2D tensor into two parts.
    ///
    /// `(outer, dim1+dim2)` → `((outer, dim1), (outer, dim2))`
    fn split_inner_dim(
        tensor: &Self::Tensor,
        dim1: usize,
        dim2: usize,
    ) -> Result<(Self::Tensor, Self::Tensor)>;

    /// Concatenate two 2D tensors along the inner (column) dimension.
    ///
    /// `(outer, d1)` + `(outer, d2)` → `(outer, d1+d2)`
    fn concat_inner_dim(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;

    /// Zero-pad the inner dimension of a 2D tensor.
    ///
    /// `(outer, width)` → `(outer, new_width)` with `new_width >= width`.
    fn pad_inner_dim(tensor: &Self::Tensor, new_width: usize) -> Result<Self::Tensor>;

    /// Broadcast a `(batch, 1, head_dim)` tensor to `(batch, num_heads, head_dim)`.
    fn broadcast_to_heads(tensor: &Self::Tensor, num_heads: usize) -> Result<Self::Tensor>;

    /// Repeat each KV head `num_repeats` times to match the number of query heads.
    ///
    /// `(seq, num_kv_heads, head_dim)` → `(seq, num_kv_heads * repeats, head_dim)`
    fn repeat_kv(tensor: &Self::Tensor, num_repeats: usize) -> Result<Self::Tensor>;

    /// Vertically stack a slice of 2D tensors (each `(1, cols)`) into `(n, cols)`.
    fn concat_rows(parts: &[Self::Tensor]) -> Result<Self::Tensor>;
}

// ---- RoPE ----

/// Rotary positional embedding (half-rotation layout, used by Llama/Qwen/Gemma).
pub trait RopeOps: Backend {
    /// Apply RoPE to a 3D tensor `(seq, heads, head_dim)`.
    fn apply_rope(
        input: &Self::Tensor,
        cos_cache: &Self::Tensor,
        sin_cache: &Self::Tensor,
        position_offset: usize,
    ) -> Result<Self::Tensor>;

    /// Apply RoPE with per-token positions (for batched decode).
    fn apply_rope_batched(
        input: &Self::Tensor,
        cos_cache: &Self::Tensor,
        sin_cache: &Self::Tensor,
        positions: &[usize],
    ) -> Result<Self::Tensor>;
}

/// Rotary positional embedding (interleaved layout, used by DeepSeek).
pub trait RopeInterleavedOps: Backend {
    /// Apply interleaved RoPE to a 3D tensor.
    fn apply_rope_interleaved(
        input: &Self::Tensor,
        cos_cache: &Self::Tensor,
        sin_cache: &Self::Tensor,
        position_offset: usize,
    ) -> Result<Self::Tensor>;
}

// ---- Attention ----

/// Fused attention kernels (prefill and single-sequence decode).
pub trait AttentionOps: Backend {
    /// Fused causal attention for prefill.
    ///
    /// Q: `(seq, heads, head_dim)`, K/V: `(kv_len, kv_heads, head_dim)`.
    fn fused_attention_prefill(
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        offset: usize,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<Self::Tensor>;

    /// Fused causal attention for single-token decode.
    fn fused_attention_decode(
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<Self::Tensor>;

    /// Fused prefill attention returning both output and log-sum-exp (for MLA).
    fn fused_attention_prefill_with_lse(
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        offset: usize,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<(Self::Tensor, Self::Tensor)>;

    /// Combine two attention outputs using their log-sum-exp values (for MLA).
    fn combine_attention_with_lse(
        out1: &Self::Tensor,
        lse1: &Self::Tensor,
        out2: &Self::Tensor,
        lse2: &Self::Tensor,
    ) -> Result<Self::Tensor>;
}

/// Paged KV cache attention.
pub trait PagedAttentionOps: Backend {
    /// Paged attention decode: batched decode against block-structured KV cache.
    fn paged_attention_decode(
        q: &Self::Tensor,
        k_pool: &Self::Tensor,
        v_pool: &Self::Tensor,
        block_tables: &[BlockTable],
        block_size: usize,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<Self::Tensor>;

    /// Gather K/V from paged cache into contiguous tensors for a single sequence.
    fn gather_paged_kv(
        paged_kv: &Self::PagedKvCache,
        layer_idx: usize,
        block_table: &BlockTable,
    ) -> Result<(Self::Tensor, Self::Tensor)>;
}

// ---- KV cache management ----

/// Paged KV cache operations (append, query pools).
pub trait PagedKvCacheOps: Backend {
    /// Append K/V tensors to the paged cache at the given position.
    fn append_paged(
        cache: &mut Self::PagedKvCache,
        layer_idx: usize,
        block_table: &BlockTable,
        k: &Self::Tensor,
        v: &Self::Tensor,
        start_pos: usize,
    ) -> Result<()>;

    /// Get the raw K/V pool tensors for a given layer.
    fn get_pools(cache: &Self::PagedKvCache, layer_idx: usize) -> (&Self::Tensor, &Self::Tensor);

    /// Get the block size of the paged cache.
    fn block_size(cache: &Self::PagedKvCache) -> usize;
}

/// Contiguous (non-paged) KV cache operations (DeepSeek MLA).
pub trait KvCacheOps: Backend {
    /// Append K/V to a contiguous cache.
    fn append_kv(
        cache: &mut Self::KvCache,
        layer_idx: usize,
        k: &Self::Tensor,
        v: &Self::Tensor,
    ) -> Result<()>;

    /// Get cached K/V for a layer (up to current length).
    fn get_kv(cache: &Self::KvCache, layer_idx: usize) -> (Self::Tensor, Self::Tensor);

    /// Get cached K/V up to a specific length (for decode where append
    /// has already been called but advance has not).
    fn get_kv_up_to(
        cache: &Self::KvCache,
        layer_idx: usize,
        len: usize,
    ) -> (Self::Tensor, Self::Tensor);
}

// ---- Mixture-of-Experts ----

/// MoE routing and dispatch.
pub trait MoeOps: Backend {
    /// Softmax-gated MoE forward pass (Mixtral, Qwen-MoE).
    ///
    /// Routes tokens through `num_experts_per_tok` experts selected by
    /// softmax over `gate_weight`. Calls `expert_fn(expert_idx, input)`
    /// for each expert and combines outputs with routing weights.
    fn moe_forward_softmax<F>(
        hidden: &Self::Tensor,
        gate_weight: &Self::Tensor,
        num_experts: usize,
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        expert_fn: F,
    ) -> Result<Self::Tensor>
    where
        F: Fn(usize, &Self::Tensor) -> Result<Self::Tensor>;
}

// ---- Extended matmul ----

/// Extended matrix multiplication (mixed precision).
pub trait MatmulExtOps: Backend {
    /// Matmul with bf16 inputs producing f32 output (for MLA).
    fn matmul_bf16_f32(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;
}
