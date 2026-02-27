//! Unit tests for DeepSeek MLA helper functions using CUDA tensors.
//!
//! Run with:
//!   cargo test -p infernum-cuda --features cuda -- --test-threads=1 deepseek_unit
#![cfg(feature = "cuda")]

use infernum::Tensor;
use infernum_cuda::cuda::{CudaContext, CudaTensor};
use infernum_cuda::CudaBackend;
use infernum_deepseek::split_kv_b_proj_dense;

#[test]
fn split_kv_b_proj_shapes() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let kv_lora_rank = 16;
    let num_heads = 4;
    let qk_nope_dim = 8;
    let v_head_dim = 8;
    let total_cols = num_heads * (qk_nope_dim + v_head_dim);

    let data: Vec<f32> = (0..kv_lora_rank * total_cols).map(|i| i as f32).collect();
    let weight = CudaTensor::from_slice(&ctx, &[kv_lora_rank, total_cols], &data).unwrap();

    let (k, v, k_t) =
        split_kv_b_proj_dense::<CudaBackend>(&ctx, &weight, num_heads, qk_nope_dim, v_head_dim)
            .unwrap();

    assert_eq!(k.shape(), &[kv_lora_rank, num_heads * qk_nope_dim]);
    assert_eq!(v.shape(), &[num_heads, kv_lora_rank, v_head_dim]);
    assert_eq!(k_t.shape(), &[num_heads, qk_nope_dim, kv_lora_rank]);
}

#[test]
fn split_kv_b_proj_roundtrip() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let kv_lora_rank = 4;
    let num_heads = 2;
    let qk_nope_dim = 3;
    let v_head_dim = 5;
    let stride = qk_nope_dim + v_head_dim;
    let total_cols = num_heads * stride;

    let data: Vec<f32> = (0..kv_lora_rank * total_cols)
        .map(|i| i as f32 * 0.1)
        .collect();
    let weight = CudaTensor::from_slice(&ctx, &[kv_lora_rank, total_cols], &data).unwrap();

    let (k, v, _) =
        split_kv_b_proj_dense::<CudaBackend>(&ctx, &weight, num_heads, qk_nope_dim, v_head_dim)
            .unwrap();

    let k_data = k.to_vec::<f32>().unwrap();
    let v_data = v.to_vec::<f32>().unwrap();

    // Reconstruct the original by interleaving K and V columns back
    let mut reconstructed = vec![0.0_f32; kv_lora_rank * total_cols];
    for row in 0..kv_lora_rank {
        for h in 0..num_heads {
            for d in 0..qk_nope_dim {
                reconstructed[row * total_cols + h * stride + d] =
                    k_data[row * (num_heads * qk_nope_dim) + h * qk_nope_dim + d];
            }
            for d in 0..v_head_dim {
                reconstructed[row * total_cols + h * stride + qk_nope_dim + d] =
                    v_data[h * kv_lora_rank * v_head_dim + row * v_head_dim + d];
            }
        }
    }

    for (i, (&orig, &recon)) in data.iter().zip(reconstructed.iter()).enumerate() {
        assert!(
            (orig - recon).abs() < 1e-6,
            "Mismatch at index {i}: orig={orig}, recon={recon}"
        );
    }
}

#[test]
fn split_kv_b_proj_k_transpose() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let kv_lora_rank = 4;
    let num_heads = 2;
    let qk_nope_dim = 3;
    let v_head_dim = 5;
    let total_cols = num_heads * (qk_nope_dim + v_head_dim);

    let data: Vec<f32> = (0..kv_lora_rank * total_cols)
        .map(|i| i as f32 * 0.1)
        .collect();
    let weight = CudaTensor::from_slice(&ctx, &[kv_lora_rank, total_cols], &data).unwrap();

    let (k, _, k_t) =
        split_kv_b_proj_dense::<CudaBackend>(&ctx, &weight, num_heads, qk_nope_dim, v_head_dim)
            .unwrap();

    let k_cols = num_heads * qk_nope_dim;
    assert_eq!(k_t.shape(), &[num_heads, qk_nope_dim, kv_lora_rank]);

    let k_data = k.to_vec::<f32>().unwrap();
    let k_t_data = k_t.to_vec::<f32>().unwrap();

    // Verify k_t[h][d][r] == k[r][h * qk_nope_dim + d] (per-head transpose)
    for h in 0..num_heads {
        for d in 0..qk_nope_dim {
            for r in 0..kv_lora_rank {
                let k_val = k_data[r * k_cols + h * qk_nope_dim + d];
                let k_t_val = k_t_data[h * qk_nope_dim * kv_lora_rank + d * kv_lora_rank + r];
                assert!(
                    (k_val - k_t_val).abs() < 1e-6,
                    "Transpose mismatch at h={h}, d={d}, r={r}: k={k_val}, k_t={k_t_val}"
                );
            }
        }
    }
}
