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

/// Tensor reshaping and manipulation.
pub trait TensorOps: Backend {
    /// Transpose a 2D tensor.
    fn transpose_2d(input: &Self::Tensor) -> Result<Self::Tensor>;
}
