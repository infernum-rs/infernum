//! Shared building blocks for transformer model implementations.
//!
//! This module provides common weight types and helper functions used by
//! multiple model families (Llama, Qwen, Gemma, DeepSeek). Models compose
//! these rather than duplicating the same logic.
//!
//! # Weight types
//!
//! - [`KvProjWeight`] — fused or separate K/V projections
//! - [`GateUpWeight`] — fused or separate gate/up projections
//! - [`MlpWeights`] — gate+up and down projections for a single MLP
//!
//! # Helpers
//!
//! Free functions that operate on these types, callable from any model's
//! forward pass. They take explicit arguments rather than `&self` to avoid
//! coupling to a specific model struct.

#![allow(
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]

use crate::backend::{
    Backend, CastOps, Comm, EmbedOps, MatmulExtOps, MatmulOps, SwigluOps, TensorFactory, TensorOps,
};
use crate::block_allocator::BlockTable;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::Result;

// ---- Weight types ----

/// K+V projection storage: fused for dense weights, separate for quantized.
pub enum KvProjWeight<B: Backend + MatmulOps> {
    /// K and V weights concatenated into a single `(hidden, 2*kv_dim)` dense
    /// tensor. After matmul the output columns split as `[k(kv_dim), v(kv_dim)]`.
    Fused { weight: B::Tensor, kv_dim: usize },
    /// Separate K and V projections (used for quantized weights).
    Separate {
        k_proj: Box<<B as MatmulOps>::LinearWeight>,
        v_proj: Box<<B as MatmulOps>::LinearWeight>,
    },
}

/// Gate+Up projection storage: fused for dense weights, separate for quantized.
pub enum GateUpWeight<B: Backend + MatmulOps> {
    /// Gate and up weights concatenated into a single `(hidden, 2*intermediate)`
    /// dense tensor. After matmul the output columns split as
    /// `[gate(intermediate), up(intermediate)]`.
    Fused {
        weight: B::Tensor,
        intermediate_size: usize,
    },
    /// Separate gate and up projections (used for quantized weights).
    Separate {
        gate_proj: Box<<B as MatmulOps>::LinearWeight>,
        up_proj: Box<<B as MatmulOps>::LinearWeight>,
    },
}

/// Weights for a single MLP layer (gate+up and down projections).
pub struct MlpWeights<B: Backend + MatmulOps> {
    pub gate_up: GateUpWeight<B>,
    pub down_proj: <B as MatmulOps>::LinearWeight,
}

// ---- Helpers ----

/// Compute K and V projections from a [`KvProjWeight`].
///
/// For fused weights, performs a single matmul and splits the output.
/// For separate weights, performs two independent linear ops.
pub fn compute_kv_proj<B: MatmulOps + TensorOps>(
    hidden: &B::Tensor,
    kv_proj: &KvProjWeight<B>,
) -> Result<(B::Tensor, B::Tensor)> {
    match kv_proj {
        KvProjWeight::<B>::Fused { weight, kv_dim } => {
            let kv = B::matmul(hidden, weight)?;
            B::split_inner_dim(&kv, *kv_dim, *kv_dim)
        }
        KvProjWeight::<B>::Separate { k_proj, v_proj } => {
            let k = B::linear(hidden, k_proj)?;
            let v = B::linear(hidden, v_proj)?;
            Ok((k, v))
        }
    }
}

/// Compute K and V projections for batched decode (batch_size=1 fast path).
///
/// When `batch_size == 1` and the weight is fused, uses `slice_view` instead
/// of `split_inner_dim` to avoid a kernel launch.
pub fn compute_kv_proj_decode<B: MatmulOps + TensorOps>(
    hidden: &B::Tensor,
    kv_proj: &KvProjWeight<B>,
    batch_size: usize,
) -> Result<(B::Tensor, B::Tensor)> {
    match kv_proj {
        KvProjWeight::<B>::Fused { weight, kv_dim } => {
            let kv = B::matmul(hidden, weight)?;
            if batch_size == 1 {
                let k = kv.slice_view(0, &[1, *kv_dim]);
                let v = kv.slice_view(*kv_dim, &[1, *kv_dim]);
                Ok((k, v))
            } else {
                B::split_inner_dim(&kv, *kv_dim, *kv_dim)
            }
        }
        KvProjWeight::<B>::Separate { k_proj, v_proj } => {
            let k = B::linear(hidden, k_proj)?;
            let v = B::linear(hidden, v_proj)?;
            Ok((k, v))
        }
    }
}

/// Compute gate and up projections from a [`GateUpWeight`].
///
/// For fused weights, performs a single matmul and splits the output.
/// When `seq_len == 1`, uses `slice_view` for the fast path.
/// For separate weights, performs two independent linear ops.
pub fn compute_gate_up<B: MatmulOps + TensorOps>(
    hidden: &B::Tensor,
    gate_up: &GateUpWeight<B>,
) -> Result<(B::Tensor, B::Tensor)> {
    match gate_up {
        GateUpWeight::<B>::Fused {
            weight,
            intermediate_size,
        } => {
            let seq_len = hidden.shape()[0];
            let gate_up = B::matmul(hidden, weight)?;
            if seq_len == 1 {
                let gate = gate_up.slice_view(0, &[1, *intermediate_size]);
                let up = gate_up.slice_view(*intermediate_size, &[1, *intermediate_size]);
                Ok((gate, up))
            } else {
                B::split_inner_dim(&gate_up, *intermediate_size, *intermediate_size)
            }
        }
        GateUpWeight::<B>::Separate { gate_proj, up_proj } => {
            let gate = B::linear(hidden, gate_proj)?;
            let up = B::linear(hidden, up_proj)?;
            Ok((gate, up))
        }
    }
}

/// SwiGLU MLP forward pass with optional all-reduce.
///
/// Computes `silu(gate) * up`, then projects down via `down_proj`.
/// Calls all-reduce if a communicator is provided (tensor parallelism).
pub fn forward_mlp<B: MatmulOps + SwigluOps + TensorOps>(
    hidden: &B::Tensor,
    weights: &MlpWeights<B>,
    comm: Option<&B::Comm>,
) -> Result<B::Tensor> {
    let (gate, up) = compute_gate_up::<B>(hidden, &weights.gate_up)?;
    let intermediate = B::swiglu(&gate, &up)?;
    let mut out = B::linear(&intermediate, &weights.down_proj)?;
    maybe_all_reduce::<B>(comm, &mut out)?;
    Ok(out)
}

/// SwiGLU MLP forward pass without all-reduce.
///
/// Used by MoE experts where all-reduce happens after expert combination.
pub fn forward_mlp_no_reduce<B: MatmulOps + SwigluOps + TensorOps>(
    hidden: &B::Tensor,
    weights: &MlpWeights<B>,
) -> Result<B::Tensor> {
    let (gate, up) = compute_gate_up::<B>(hidden, &weights.gate_up)?;
    let intermediate = B::swiglu(&gate, &up)?;
    B::linear(&intermediate, &weights.down_proj)
}

/// Project hidden states to vocabulary logits (always returns f32).
///
/// Uses `matmul_bf16_f32` fast path when dtype is BF16 and the weight
/// is dense (non-quantized). Otherwise falls back to linear + cast.
pub fn lm_head_forward<B: MatmulOps + MatmulExtOps + CastOps>(
    hidden: &B::Tensor,
    lm_head: &<B as MatmulOps>::LinearWeight,
    dtype: DType,
) -> Result<B::Tensor> {
    if dtype == DType::BF16 {
        if let Some(w) = B::as_dense_weight(lm_head) {
            return B::matmul_bf16_f32(hidden, w);
        }
    }
    let logits_t = B::linear(hidden, lm_head)?;
    if dtype == DType::F32 {
        return Ok(logits_t);
    }
    B::cast_to_f32(&logits_t)
}

/// Gather embedding vectors for a slice of token IDs.
pub fn embed<B: EmbedOps>(table: &B::Tensor, input_ids: &[u32]) -> Result<B::Tensor> {
    B::embedding_gather(table, input_ids)
}

/// Extract the last row from a `(seq_len, hidden_size)` tensor.
///
/// Returns a `(1, hidden_size)` tensor. When `seq_len == 1`, returns a
/// reshaped view without copying.
pub fn extract_last_row<B: Backend>(hidden: &B::Tensor, seq_len: usize) -> B::Tensor {
    let hidden_size = hidden.shape()[1];
    if seq_len == 1 {
        return hidden.reshape(&[1, hidden_size]);
    }
    hidden.slice_view((seq_len - 1) * hidden_size, &[1, hidden_size])
}

/// In-place all-reduce if a communicator is provided. No-op for single GPU.
pub fn maybe_all_reduce<B: Backend>(comm: Option<&B::Comm>, tensor: &mut B::Tensor) -> Result<()> {
    if let Some(comm) = comm {
        comm.all_reduce_sum(tensor)?;
    }
    Ok(())
}

/// Convert host-side decode batch data into device tensors and call
/// `forward_fn` with the prepared device tensors.
///
/// This handles the common boilerplate of flattening block tables,
/// casting positions to i32, and uploading everything to the device.
///
/// # Errors
/// Returns an error if tensor upload or the forward pass fails.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn forward_batch_decode_host<B, F>(
    device: &B::DeviceHandle,
    token_ids: &[u32],
    block_tables: &[BlockTable],
    positions: &[usize],
    forward_fn: F,
) -> Result<B::Tensor>
where
    B: TensorFactory,
    F: FnOnce(
        &B::Tensor, // token_ids
        &B::Tensor, // block_tables (flat)
        &B::Tensor, // seq_lens
        &B::Tensor, // positions
        usize,      // batch_size
        usize,      // max_blocks_per_seq
        usize,      // max_seq_len
    ) -> Result<B::Tensor>,
{
    let batch_size = token_ids.len();
    let max_blocks_per_seq = block_tables
        .iter()
        .map(|bt| bt.blocks().len())
        .max()
        .unwrap_or(0);
    let mut bt_flat = vec![0i32; batch_size * max_blocks_per_seq];
    for (i, bt) in block_tables.iter().enumerate() {
        for (j, &block_id) in bt.blocks().iter().enumerate() {
            bt_flat[i * max_blocks_per_seq + j] = block_id as i32;
        }
    }
    let seq_lens: Vec<i32> = positions.iter().map(|&p| (p + 1) as i32).collect();
    let positions_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
    let max_seq_len = seq_lens.iter().copied().max().unwrap_or(0) as usize;

    let token_ids_t = B::from_u32_slice(device, &[batch_size], token_ids)?;
    let bt_t = B::from_i32_slice(device, &[batch_size * max_blocks_per_seq], &bt_flat)?;
    let sl_t = B::from_i32_slice(device, &[batch_size], &seq_lens)?;
    let pos_t = B::from_i32_slice(device, &[batch_size], &positions_i32)?;

    forward_fn(
        &token_ids_t,
        &bt_t,
        &sl_t,
        &pos_t,
        batch_size,
        max_blocks_per_seq,
        max_seq_len,
    )
}

/// Load a dense MLP's weights (gate, up, down projections) from a weight loader.
///
/// Fuses gate+up projections into a single tensor when both are dense.
///
/// # Errors
/// Returns an error if any weight fails to load.
pub fn load_mlp_weights<B: MatmulOps + TensorOps>(
    loader: &impl crate::WeightLoader<B>,
    prefix: &str,
    intermediate_size: usize,
    dtype: DType,
    qc: Option<&crate::QuantizationConfig>,
) -> Result<MlpWeights<B>> {
    let gate = loader.load_linear(&format!("{prefix}.gate_proj.weight"), dtype, qc)?;
    let up = loader.load_linear(&format!("{prefix}.up_proj.weight"), dtype, qc)?;
    let gate_up = if B::is_dense_weight(&gate) && B::is_dense_weight(&up) {
        let g = B::as_dense_weight(&gate).expect("checked dense");
        let u = B::as_dense_weight(&up).expect("checked dense");
        GateUpWeight::<B>::Fused {
            weight: B::concat_inner_dim(g, u)?,
            intermediate_size,
        }
    } else {
        GateUpWeight::<B>::Separate {
            gate_proj: Box::new(gate),
            up_proj: Box::new(up),
        }
    };
    Ok(MlpWeights {
        gate_up,
        down_proj: loader.load_linear(&format!("{prefix}.down_proj.weight"), dtype, qc)?,
    })
}

/// Load the lm_head linear weight, handling tied embeddings and quantized models.
///
/// # Tied embeddings
/// When `tie_word_embeddings` is true, the embedding table is reused as lm_head:
/// - Quantized models: cast to f32 and re-quantize as Q8
/// - Dense models: transpose to matmul-ready layout
///
/// # Separate lm_head
/// When not tied, loads `lm_head.weight` directly. For quantized models,
/// re-quantizes dense lm_head weights as Q8 for consistent decode throughput.
///
/// # Errors
/// Returns an error if weight loading or quantization fails.
pub fn load_lm_head<B: MatmulOps + CastOps + crate::TensorDataOps + TensorOps>(
    device: &B::DeviceHandle,
    loader: &impl crate::WeightLoader<B>,
    embed_tokens: &B::Tensor,
    tie_word_embeddings: bool,
    dtype: DType,
    qc: Option<&crate::QuantizationConfig>,
) -> Result<<B as MatmulOps>::LinearWeight> {
    if tie_word_embeddings {
        if qc.is_some() {
            let embed_f32 = B::cast_to_f32(embed_tokens)?;
            let data = B::to_f32_vec(&embed_f32)?;
            B::quantize_to_q8(device, embed_f32.shape(), &data)
        } else {
            let embed_f32 = B::cast_to_f32(embed_tokens)?;
            let transposed = B::transpose_2d(&embed_f32)?;
            Ok(B::dense_weight(B::cast_from_f32(&transposed, dtype)?))
        }
    } else {
        let lw = loader.load_linear("lm_head.weight", dtype, None)?;
        if qc.is_some() {
            if let Some(w) = B::as_dense_weight(&lw) {
                let f32_w = B::cast_to_f32(w)?;
                let row_major = B::transpose_2d(&f32_w)?;
                let data = B::to_f32_vec(&row_major)?;
                B::quantize_to_q8(device, row_major.shape(), &data)
            } else {
                Ok(lw)
            }
        } else {
            Ok(lw)
        }
    }
}

/// Precompute RoPE cosine and sine caches and upload to the device.
///
/// Returns `(cos_cache, sin_cache)` tensors of shape `(max_pos, head_dim/2)`
/// cast to the specified dtype.
///
/// # Errors
/// Returns an error if tensor creation or casting fails.
pub fn build_rope_cache<B: TensorFactory + CastOps>(
    device: &B::DeviceHandle,
    head_dim: usize,
    max_pos: usize,
    rope_theta: f32,
    rope_scaling: Option<&crate::RopeScaling>,
    dtype: DType,
) -> Result<(B::Tensor, B::Tensor)> {
    let half_dim = head_dim / 2;
    let (cos_data, sin_data) = if let Some(scaling) = rope_scaling {
        crate::rope::precompute_rope_data_scaled(max_pos, head_dim, rope_theta, scaling)
    } else {
        crate::rope::precompute_rope_data(max_pos, head_dim, rope_theta)
    };
    let cos_f32 = B::from_f32_slice(device, &[max_pos, half_dim], &cos_data)?;
    let sin_f32 = B::from_f32_slice(device, &[max_pos, half_dim], &sin_data)?;
    let cos_cache = B::cast_from_f32(&cos_f32, dtype)?;
    let sin_cache = B::cast_from_f32(&sin_f32, dtype)?;
    Ok((cos_cache, sin_cache))
}
