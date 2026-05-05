//! Weight utilities for `DeepSeek` models.
//!
//! Contains helpers for manipulating raw weight tensors that are shared between
//! the graph builder and the CUDA unit tests.

use infernum::backend::{TensorDataOps, TensorFactory};
use infernum::tensor::Tensor;
use infernum::Result;

/// Split a pre-transposed dense `kv_b_proj` weight into K-nope and V portions.
///
/// Input shape: `(kv_lora_rank, num_heads * (qk_nope_dim + v_head_dim))` (pre-transposed).
/// Columns are interleaved per head: for head `h`, columns
/// `[h*stride .. h*stride+qk_nope_dim]` are K-nope, followed by `v_head_dim` V columns.
///
/// Returns `(kv_b_proj_k, kv_b_proj_v, kv_b_proj_k_t)`:
/// - `kv_b_proj_k`: `(kv_lora_rank, num_heads * qk_nope_dim)` — K-nope decompression
/// - `kv_b_proj_v`: `(num_heads, kv_lora_rank, v_head_dim)` — V decompression (batched matmul)
/// - `kv_b_proj_k_t`: `(num_heads, qk_nope_dim, kv_lora_rank)` — Q absorption (batched matmul)
///
/// # Panics
///
/// Panics if the weight's column count does not equal `num_heads * (qk_nope_dim + v_head_dim)`.
///
/// # Errors
///
/// Returns an error if tensor data extraction or construction fails.
pub fn split_kv_b_proj_dense<B: TensorFactory + TensorDataOps>(
    device: &B::DeviceHandle,
    weight: &B::Tensor,
    num_heads: usize,
    qk_nope_dim: usize,
    v_head_dim: usize,
) -> Result<(B::Tensor, B::Tensor, B::Tensor)> {
    let shape = weight.shape();
    let dtype = weight.dtype();
    let elem = dtype.size_in_bytes();
    let kv_lora_rank = shape[0];
    let total_cols = shape[1];
    let stride = qk_nope_dim + v_head_dim;
    assert_eq!(
        total_cols,
        num_heads * stride,
        "split_kv_b_proj_dense: expected {} columns, got {total_cols}",
        num_heads * stride
    );

    let data = B::to_raw_bytes(weight)?;

    // Extract K-nope columns: shape (kv_lora_rank, num_heads * qk_nope_dim)
    let k_cols = num_heads * qk_nope_dim;
    let mut k_data = vec![0u8; kv_lora_rank * k_cols * elem];
    for row in 0..kv_lora_rank {
        for h in 0..num_heads {
            let src_offset = (row * total_cols + h * stride) * elem;
            let dst_offset = (row * k_cols + h * qk_nope_dim) * elem;
            let len = qk_nope_dim * elem;
            k_data[dst_offset..dst_offset + len]
                .copy_from_slice(&data[src_offset..src_offset + len]);
        }
    }

    // Extract V columns: shape (num_heads, kv_lora_rank, v_head_dim) for batched matmul
    let mut v_data = vec![0u8; num_heads * kv_lora_rank * v_head_dim * elem];
    for h in 0..num_heads {
        for row in 0..kv_lora_rank {
            let src_offset = (row * total_cols + h * stride + qk_nope_dim) * elem;
            let dst_offset = (h * kv_lora_rank * v_head_dim + row * v_head_dim) * elem;
            let len = v_head_dim * elem;
            v_data[dst_offset..dst_offset + len]
                .copy_from_slice(&data[src_offset..src_offset + len]);
        }
    }

    // K transposed per-head: shape (num_heads, qk_nope_dim, kv_lora_rank) for batched matmul.
    let mut k_t_data = vec![0u8; num_heads * qk_nope_dim * kv_lora_rank * elem];
    for h in 0..num_heads {
        for row in 0..kv_lora_rank {
            for col in 0..qk_nope_dim {
                let src_offset = (row * k_cols + h * qk_nope_dim + col) * elem;
                let dst_offset = (h * qk_nope_dim * kv_lora_rank + col * kv_lora_rank + row) * elem;
                k_t_data[dst_offset..dst_offset + elem]
                    .copy_from_slice(&k_data[src_offset..src_offset + elem]);
            }
        }
    }

    let k_tensor = B::from_raw_bytes(device, &[kv_lora_rank, k_cols], dtype, &k_data)?;
    let v_tensor = B::from_raw_bytes(
        device,
        &[num_heads, kv_lora_rank, v_head_dim],
        dtype,
        &v_data,
    )?;
    let k_t_tensor = B::from_raw_bytes(
        device,
        &[num_heads, qk_nope_dim, kv_lora_rank],
        dtype,
        &k_t_data,
    )?;

    Ok((k_tensor, v_tensor, k_t_tensor))
}
