//! Weight utilities for `DeepSeek` models.
//!
//! Contains helpers for manipulating raw weight tensors that are shared between
//! the graph builder and the CUDA unit tests.

use infernum::backend::{TensorDataOps, TensorFactory};
use infernum::shard::ShardConfig;
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

/// Split and shard `kv_b_proj` for one tensor-parallel rank.
///
/// Calls [`split_kv_b_proj_dense`] to get the full per-head tensors, then
/// extracts the slice for rank `shard.rank` out of `shard.world_size`:
/// - `k_shard`:   `(kv_lora_rank, num_heads_local * qk_nope_dim)`
/// - `v_shard`:   `(num_heads_local, kv_lora_rank, v_head_dim)`
/// - `k_t_shard`: `(num_heads_local, qk_nope_dim, kv_lora_rank)`
///
/// # Errors
///
/// Returns an error if tensor data extraction or construction fails.
pub fn split_kv_b_proj_dense_sharded<B: TensorFactory + TensorDataOps>(
    device: &B::DeviceHandle,
    weight: &B::Tensor,
    num_heads: usize,
    qk_nope_dim: usize,
    v_head_dim: usize,
    shard: &ShardConfig,
) -> Result<(B::Tensor, B::Tensor, B::Tensor)> {
    let (k_full, v_full, k_t_full) =
        split_kv_b_proj_dense::<B>(device, weight, num_heads, qk_nope_dim, v_head_dim)?;

    let nh_local = num_heads / shard.world_size;
    let rank = shard.rank;
    let dtype = weight.dtype();
    let elem = dtype.size_in_bytes();
    let kv_lora = weight.shape()[0];

    // k_full: [kv_lora, num_heads * qk_nope] — heads are consecutive column groups.
    // Each row contributes non-contiguous columns, so extract row by row.
    let k_data = B::to_raw_bytes(&k_full)?;
    let k_cols_full = num_heads * qk_nope_dim;
    let k_cols_local = nh_local * qk_nope_dim;
    let mut k_shard_data = vec![0u8; kv_lora * k_cols_local * elem];
    for row in 0..kv_lora {
        let src = (row * k_cols_full + rank * k_cols_local) * elem;
        let dst = row * k_cols_local * elem;
        k_shard_data[dst..dst + k_cols_local * elem]
            .copy_from_slice(&k_data[src..src + k_cols_local * elem]);
    }

    // v_full: [num_heads, kv_lora, v_head] — contiguous head block.
    let v_data = B::to_raw_bytes(&v_full)?;
    let v_inner = kv_lora * v_head_dim;
    let v_shard_data =
        v_data[rank * nh_local * v_inner * elem..(rank + 1) * nh_local * v_inner * elem].to_vec();

    // k_t_full: [num_heads, qk_nope, kv_lora] — contiguous head block.
    let k_t_data = B::to_raw_bytes(&k_t_full)?;
    let k_t_inner = qk_nope_dim * kv_lora;
    let k_t_shard_data = k_t_data
        [rank * nh_local * k_t_inner * elem..(rank + 1) * nh_local * k_t_inner * elem]
        .to_vec();

    let k_shard = B::from_raw_bytes(device, &[kv_lora, k_cols_local], dtype, &k_shard_data)?;
    let v_shard = B::from_raw_bytes(
        device,
        &[nh_local, kv_lora, v_head_dim],
        dtype,
        &v_shard_data,
    )?;
    let k_t_shard = B::from_raw_bytes(
        device,
        &[nh_local, qk_nope_dim, kv_lora],
        dtype,
        &k_t_shard_data,
    )?;

    Ok((k_shard, v_shard, k_t_shard))
}
