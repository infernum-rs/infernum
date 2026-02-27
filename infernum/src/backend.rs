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

// ---- Communicator trait ----

/// Communicator for multi-device all-reduce operations.
///
/// Implemented by backend-specific communicators (e.g., NCCL for CUDA).
/// Single-GPU backends use the blanket `impl Comm<T> for ()` which is
/// a no-op.
///
/// Models store `Option<B::Comm>` and call [`all_reduce_sum`](Comm::all_reduce_sum)
/// at tensor-parallel sync points (after `o_proj` and `down_proj` matmuls).
pub trait Comm<T>: Send + Sync {
    /// In-place sum all-reduce across all ranks.
    ///
    /// # Errors
    /// Returns an error if the collective operation fails.
    fn all_reduce_sum(&self, tensor: &mut T) -> Result<()>;
}

/// No-op communicator for single-GPU backends.
impl<T> Comm<T> for () {
    fn all_reduce_sum(&self, _tensor: &mut T) -> Result<()> {
        Ok(())
    }
}

// ---- Core backend trait ----

/// A compute backend (CUDA, CPU, Metal, etc.).
pub trait Backend: 'static {
    /// The tensor type for this backend (e.g., `CudaTensor`).
    type Tensor: Tensor + Clone + Send + Sync;

    /// Opaque device handle for this backend.
    ///
    /// Models store this and pass it to ops that need device context
    /// (KV cache allocation, tensor creation). For CUDA this is
    /// `CudaContext`; for CPU it might be `()`.
    type DeviceHandle: Clone + Send + Sync;

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

    /// Communicator for multi-device all-reduce.
    ///
    /// For CUDA this is `NcclCommunicator`. Single-GPU backends use `()`
    /// (blanket no-op impl). Models store `Option<Self::Comm>` — `None`
    /// for single-GPU, `Some(comm)` for tensor-parallel sharded models.
    type Comm: Comm<Self::Tensor>;

    /// Wrap a raw logits tensor into the backend's `Logits` type.
    ///
    /// Used by generic `Model` impls to return backend-specific logits
    /// from forward passes without knowing the concrete type.
    fn logits_from_tensor(tensor: Self::Tensor) -> Self::Logits;
}

// ---- Op traits ----

/// Creating tensors from host data.
///
/// Enables generic model code to create tensors (RoPE cache, constants, etc.)
/// without knowing the backend. The backend handles device upload internally.
pub trait TensorFactory: Backend {
    /// Create a tensor from an `f32` slice on the host.
    ///
    /// The backend uploads the data to its device (GPU, etc.).
    fn from_f32_slice(
        device: &Self::DeviceHandle,
        shape: &[usize],
        data: &[f32],
    ) -> Result<Self::Tensor>;

    /// Create a tensor from raw bytes on the host with a specified dtype.
    ///
    /// Used by weight loaders and format-specific loading code.
    fn from_raw_bytes(
        device: &Self::DeviceHandle,
        shape: &[usize],
        dtype: DType,
        data: &[u8],
    ) -> Result<Self::Tensor>;

    /// Create a tensor from a `u32` slice on the host.
    ///
    /// Used by the engine to upload token IDs to the device before
    /// calling model forward methods.
    fn from_u32_slice(
        device: &Self::DeviceHandle,
        shape: &[usize],
        data: &[u32],
    ) -> Result<Self::Tensor>;

    /// Create a tensor from an `i32` slice on the host.
    ///
    /// Used by the engine to upload positions, block tables, and
    /// sequence lengths to the device before calling model forward methods.
    fn from_i32_slice(
        device: &Self::DeviceHandle,
        shape: &[usize],
        data: &[i32],
    ) -> Result<Self::Tensor>;
}

/// Device tensors for a batched decode step, produced by
/// [`DecodeBufferOps::prepare_decode_tensors`].
pub struct DecodeTensors<T> {
    /// Token IDs: shape `(batch_size,)`.
    pub token_ids: T,
    /// Flattened block tables: shape `(batch_size * max_blocks_per_seq,)`.
    pub block_tables: T,
    /// Sequence lengths (pos + 1): shape `(batch_size,)`.
    pub seq_lens: T,
    /// Positions: shape `(batch_size,)`.
    pub positions: T,
    /// Number of active sequences.
    pub batch_size: usize,
    /// Maximum blocks per sequence in the flattened block table.
    pub max_blocks_per_seq: usize,
    /// Maximum sequence length across the batch.
    pub max_seq_len: usize,
}

/// Prepare device tensors for a batched decode step.
///
/// The default implementation allocates fresh tensors each call via
/// [`TensorFactory`]. Backends that use CUDA graphs override this to
/// write into pre-allocated fixed-address buffers so that graph replay
/// sees the updated values without re-recording.
pub trait DecodeBufferOps: TensorFactory {
    /// Convert host-side decode batch data into device tensors.
    ///
    /// # Arguments
    /// * `state` — mutable backend runtime state (graph-enabled backends
    ///   store pre-allocated buffers here)
    /// * `device` — device handle for tensor creation
    /// * `token_ids` — one token per sequence (`batch_size` elements)
    /// * `positions` — current position per sequence (i32)
    /// * `block_tables_flat` — flattened block table
    ///   (`batch_size * max_blocks_per_seq` elements, i32)
    /// * `seq_lens` — sequence length per sequence (i32)
    /// * `max_blocks_per_seq` — max blocks per sequence in `block_tables_flat`
    ///
    /// # Errors
    /// Returns an error if device upload fails.
    #[allow(clippy::too_many_arguments)]
    fn prepare_decode_tensors(
        state: &mut Self::RuntimeState,
        device: &Self::DeviceHandle,
        token_ids: &[u32],
        positions: &[i32],
        block_tables_flat: &[i32],
        seq_lens: &[i32],
        max_blocks_per_seq: usize,
    ) -> Result<DecodeTensors<Self::Tensor>> {
        let _ = state;
        let batch_size = token_ids.len();
        #[allow(clippy::cast_sign_loss)]
        let max_seq_len = seq_lens.iter().copied().max().unwrap_or(0) as usize;

        let token_ids_t = Self::from_u32_slice(device, &[batch_size], token_ids)?;
        let block_tables_t = Self::from_i32_slice(
            device,
            &[batch_size * max_blocks_per_seq],
            block_tables_flat,
        )?;
        let seq_lens_t = Self::from_i32_slice(device, &[batch_size], seq_lens)?;
        let positions_t = Self::from_i32_slice(device, &[batch_size], positions)?;

        Ok(DecodeTensors {
            token_ids: token_ids_t,
            block_tables: block_tables_t,
            seq_lens: seq_lens_t,
            positions: positions_t,
            batch_size,
            max_blocks_per_seq,
            max_seq_len,
        })
    }
}

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

    /// Try to extract a reference to the underlying dense tensor from a
    /// `LinearWeight`. Returns `None` if the weight is quantized.
    ///
    /// Used by models for mixed-precision matmul paths (e.g., bf16 → f32).
    fn as_dense_weight(_weight: &Self::LinearWeight) -> Option<&Self::Tensor> {
        None
    }

    /// Wrap a pre-transposed dense tensor into a `LinearWeight`.
    ///
    /// The tensor must already be in matmul-ready layout
    /// `(in_features, out_features)`.
    fn dense_weight(tensor: Self::Tensor) -> Self::LinearWeight;

    /// Check whether a `LinearWeight` is a dense (non-quantized) tensor.
    fn is_dense_weight(weight: &Self::LinearWeight) -> bool;

    /// Quantize f32 data to Q8 and wrap as a `LinearWeight`.
    ///
    /// Used to quantize the lm_head weight when the model uses quantized
    /// layers (GPTQ/AWQ) for consistent decode throughput.
    ///
    /// `shape` is row-major `(out_features, in_features)`.
    fn quantize_to_q8(
        device: &Self::DeviceHandle,
        shape: &[usize],
        data: &[f32],
    ) -> Result<Self::LinearWeight>;

    /// Upload a host-side linear weight (dense or quantized) to the device.
    ///
    /// For dense weights, the host data is already transposed to matmul-ready
    /// layout. For quantized weights, the backend constructs its quantized
    /// tensor type from the raw data/scales/qzeros buffers.
    ///
    /// # Errors
    /// Returns an error if device allocation or upload fails.
    fn upload_host_linear(
        device: &Self::DeviceHandle,
        weight: &crate::weights::host::HostLinearWeight,
    ) -> Result<Self::LinearWeight>;
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

/// Downloading tensor data to the host.
///
/// Used by generic loading code that needs to manipulate weight data
/// on the host (e.g., GGUF unpermute, Q8 quantization of lm_head).
pub trait TensorDataOps: Backend {
    /// Download tensor contents to the host as `Vec<f32>`.
    ///
    /// The backend casts to f32 if necessary.
    fn to_f32_vec(tensor: &Self::Tensor) -> Result<Vec<f32>>;
}

/// Construct a [`WeightLoader`](crate::WeightLoader) from a SafeTensors directory.
///
/// Each backend provides its own loader type (e.g., for CUDA: `SafeTensorsLoader`
/// wrapped in `CudaWeightLoader`). Models use this trait to implement generic
/// `from_pretrained` methods without knowing the concrete loader type.
pub trait SafeTensorsLoaderOps: MatmulOps + Sized {
    /// The concrete `WeightLoader<Self>` type returned by this backend.
    type SafeTensorsLoader: crate::WeightLoader<Self>;

    /// Create a weight loader for a SafeTensors model directory.
    ///
    /// `model_dir` contains `*.safetensors` files and usually `config.json`.
    /// The backend handles device context (e.g., GPU upload handles)
    /// internally.
    ///
    /// # Errors
    /// Returns an error if the directory is missing or files are malformed.
    fn safetensors_loader(
        device: &Self::DeviceHandle,
        model_dir: &std::path::Path,
    ) -> Result<Self::SafeTensorsLoader>;
}

/// Embedding lookup.
pub trait EmbedOps: Backend {
    /// Gather rows from an embedding table by token IDs (host slice).
    ///
    /// `table` has shape `(vocab_size, hidden_size)`.
    /// Returns a tensor of shape `(indices.len(), hidden_size)`.
    fn embedding_gather(table: &Self::Tensor, indices: &[u32]) -> Result<Self::Tensor>;

    /// Gather rows from an embedding table by device-side token IDs.
    ///
    /// `indices` is a 1D tensor of u32 token IDs already on the device.
    /// `seq_len` is the number of valid tokens to gather.
    /// Returns a tensor of shape `(seq_len, hidden_size)`.
    fn embedding_gather_tensor(
        table: &Self::Tensor,
        indices: &Self::Tensor,
        seq_len: usize,
    ) -> Result<Self::Tensor>;
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
    ///
    /// `positions` is a 1D i32 tensor on the device with `batch_size` entries.
    fn apply_rope_batched(
        input: &Self::Tensor,
        cos_cache: &Self::Tensor,
        sin_cache: &Self::Tensor,
        positions: &Self::Tensor,
        batch_size: usize,
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
#[allow(clippy::too_many_arguments)]
pub trait PagedAttentionOps: Backend {
    /// Paged attention decode: batched decode against block-structured KV cache.
    ///
    /// `block_tables` is a flattened i32 tensor of shape
    /// `(batch_size * max_blocks_per_seq,)` containing physical block indices.
    /// `seq_lens` is a 1D i32 tensor of shape `(batch_size,)`.
    fn paged_attention_decode(
        q: &Self::Tensor,
        k_pool: &Self::Tensor,
        v_pool: &Self::Tensor,
        block_tables: &Self::Tensor,
        seq_lens: &Self::Tensor,
        block_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
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
    /// Allocate a paged KV cache.
    ///
    /// `block_config` specifies block size and count. The backend handles
    /// device-specific allocation (e.g., GPU memory pools).
    fn allocate_paged_kv_cache(
        device: &Self::DeviceHandle,
        num_layers: usize,
        block_config: &crate::block_allocator::BlockConfig,
        num_kv_heads: usize,
        head_dim: usize,
        cache_dtype: DType,
    ) -> Result<Self::PagedKvCache>;

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

    /// Batched KV cache append using device-side block tables and positions.
    ///
    /// `block_tables` is a flattened i32 tensor of shape
    /// `(batch_size * max_blocks_per_seq,)`.
    /// `positions` is a 1D i32 tensor of shape `(batch_size,)`.
    #[allow(clippy::too_many_arguments)]
    fn append_paged_batched(
        cache: &mut Self::PagedKvCache,
        layer_idx: usize,
        k: &Self::Tensor,
        v: &Self::Tensor,
        block_tables: &Self::Tensor,
        positions: &Self::Tensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()>;
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

// ---- Multi-device ops ----

/// Factory methods for creating devices and communicators across multiple
/// GPUs/devices. Used by [`ShardedModel`](crate::ShardedModel) to set up
/// tensor-parallel inference without knowing the backend.
///
/// Single-GPU backends don't implement this — it's only required when
/// constructing a `ShardedModel`.
pub trait MultiDeviceOps: Backend {
    /// Opaque ID used to coordinate communicator creation across ranks.
    ///
    /// For NCCL this is `NcclId`. Rank 0 generates it and broadcasts
    /// the raw bytes to other ranks.
    type CommId: Send + Sync + Copy;

    /// Generate a unique communicator ID (called once, shared across ranks).
    ///
    /// # Errors
    /// Returns an error if ID generation fails.
    fn create_comm_id() -> Result<Self::CommId>;

    /// Create a device handle for the given rank.
    ///
    /// # Errors
    /// Returns an error if device creation fails.
    fn create_device(rank: usize) -> Result<Self::DeviceHandle>;

    /// Create a communicator for the given rank.
    ///
    /// All ranks must call this concurrently with the same `comm_id`.
    ///
    /// # Errors
    /// Returns an error if communicator creation fails.
    fn create_comm(
        device: &Self::DeviceHandle,
        rank: usize,
        world_size: usize,
        comm_id: Self::CommId,
    ) -> Result<Self::Comm>;
}
