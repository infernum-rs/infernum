//! Fused attention kernels
//!
//! Single-kernel attention that eliminates transposes, repeat_kv, and
//! intermediate allocations from the hot path.
//!
//! - `fused_attention_decode`: for single-token decode (seq_q == 1)
//! - `fused_attention_prefill`: for multi-token prefill with causal mask

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::too_many_arguments,
    clippy::manual_div_ceil
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use crate::cuda::CudaTensor;
use crate::dtype::TensorDType;
use crate::tensor::Tensor;
use crate::Result;

/// Kernel name suffix for dtype
fn kernel_suffix<T: cudarc::driver::DeviceRepr>() -> &'static str {
    let type_name = std::any::type_name::<T>();
    if type_name.contains("f32") {
        "f32"
    } else if type_name.contains("f16") && !type_name.contains("bf16") {
        "f16"
    } else if type_name.contains("bf16") {
        "bf16"
    } else {
        panic!("Unsupported dtype for fused_attention: {type_name}")
    }
}

/// Fused decode attention kernel.
///
/// One block per query head. Each block:
///   1. Loads the Q vector for this head from shared memory.
///   2. Iterates over all `total_len` cached K positions, computing dot(Q, K)
///      with GQA head mapping, and tracking the online softmax max/sum.
///   3. Iterates over all cached V positions, accumulating the weighted output.
///
/// Uses a two-pass approach (scores then weighted V) with shared memory for
/// the full score vector. This is simple and correct; a single-pass online
/// approach would be more memory-efficient for very long contexts but adds
/// complexity.
///
/// Inputs (all in native `(seq, heads, dim)` layout, no transposes needed):
///   - Q: `(1, num_heads, head_dim)` — the single query token
///   - K: `(total_len, num_kv_heads, head_dim)` — full cached keys
///   - V: `(total_len, num_kv_heads, head_dim)` — full cached values
///
/// Output: `(1, num_heads, head_dim)`
const FUSED_DECODE_PTX: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "/kernels/fused_decode_attention.ptx"
));

/// Fused prefill attention kernel with causal masking.
///
/// One block per `(head, query_position)` pair. Each block computes the full
/// attention output for one query position against all valid key positions
/// `[0 .. offset + query_pos + 1)`.
///
/// Handles GQA natively via `kv_head = head * num_kv_heads / num_heads`.
///
/// Inputs (all in native `(seq, heads, dim)` layout):
///   - Q: `(seq_q, num_heads, head_dim)`
///   - K: `(total_len, num_kv_heads, head_dim)` — includes prefill tokens
///   - V: `(total_len, num_kv_heads, head_dim)`
///
/// Output: `(seq_q, num_heads, head_dim)`
const FUSED_PREFILL_PTX: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "/kernels/fused_prefill_attention.ptx"
));

const FUSED_DECODE_KERNEL_NAMES: &[&str] = &[
    "fused_decode_attention_f32",
    "fused_decode_attention_f16",
    "fused_decode_attention_bf16",
];

const FUSED_PREFILL_KERNEL_NAMES: &[&str] = &[
    "fused_prefill_attention_f32",
    "fused_prefill_attention_f16",
    "fused_prefill_attention_bf16",
];

fn ensure_fused_decode_kernel<T: cudarc::driver::DeviceRepr>(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<()> {
    let module_name = "fused_decode_attention";
    let kernel_name = format!("fused_decode_attention_{}", kernel_suffix::<T>());
    if !device.has_func(module_name, &kernel_name) {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(FUSED_DECODE_PTX),
            module_name,
            FUSED_DECODE_KERNEL_NAMES,
        )?;
    }
    Ok(())
}

fn ensure_fused_prefill_kernel<T: cudarc::driver::DeviceRepr>(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<()> {
    let module_name = "fused_prefill_attention";
    let kernel_name = format!("fused_prefill_attention_{}", kernel_suffix::<T>());
    if !device.has_func(module_name, &kernel_name) {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(FUSED_PREFILL_PTX),
            module_name,
            FUSED_PREFILL_KERNEL_NAMES,
        )?;
    }
    Ok(())
}

/// Fused attention for single-token decode.
///
/// Computes `softmax(Q @ K^T / sqrt(d)) @ V` in a single kernel, reading
/// Q/K/V in their native `(seq, heads, dim)` layout. Handles GQA natively —
/// no `repeat_kv` or transpose needed.
///
/// # Arguments
/// * `q` — query tensor of shape `(1, num_heads, head_dim)`
/// * `k` — full cached keys of shape `(total_len, num_kv_heads, head_dim)`
/// * `v` — full cached values of shape `(total_len, num_kv_heads, head_dim)`
///
/// # Returns
/// Output tensor of shape `(1, num_heads, head_dim)`
///
/// # Errors
/// Returns an error if the kernel launch fails
pub fn fused_attention_decode<T: TensorDType + cudarc::driver::DeviceRepr>(
    q: &CudaTensor<T>,
    k: &CudaTensor<T>,
    v: &CudaTensor<T>,
) -> Result<CudaTensor<T>> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    assert_eq!(q_shape.len(), 3, "Q must be 3D: (1, num_heads, head_dim)");
    assert_eq!(q_shape[0], 1, "Q seq_len must be 1 for decode");
    assert_eq!(
        k_shape.len(),
        3,
        "K must be 3D: (total_len, num_kv_heads, head_dim)"
    );
    assert_eq!(
        v_shape.len(),
        3,
        "V must be 3D: (total_len, num_kv_heads, head_dim)"
    );

    let num_heads = q_shape[1];
    let head_dim = q_shape[2];
    let total_len = k_shape[0];
    let num_kv_heads = k_shape[1];

    assert_eq!(k_shape[2], head_dim, "K head_dim must match Q");
    assert_eq!(v_shape, k_shape, "V shape must match K");
    assert!(
        num_heads.is_multiple_of(num_kv_heads),
        "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    );

    let scale = 1.0 / (head_dim as f32).sqrt();
    let output_shape = [1, num_heads, head_dim];
    let mut output = unsafe { CudaTensor::<T>::uninit(q.context(), &output_shape)? };

    let device = q.context().device();
    ensure_fused_decode_kernel::<T>(device)?;

    let kernel_name = format!("fused_decode_attention_{}", kernel_suffix::<T>());
    let func = device
        .get_func("fused_decode_attention", &kernel_name)
        .unwrap();

    let block_size = 256_usize.min(total_len.next_power_of_two());
    // Shared memory: Q (head_dim) + cached weights (total_len) + reduction scratch (block_size)
    let shared_mem = (head_dim + total_len + block_size) * std::mem::size_of::<f32>();

    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    unsafe {
        func.launch(
            cfg,
            (
                output.cuda_slice_mut(),
                &q.cuda_slice(),
                &k.cuda_slice(),
                &v.cuda_slice(),
                scale,
                total_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
            ),
        )?;
    }

    Ok(output)
}

/// Fused attention for multi-token prefill with causal masking.
///
/// Computes `softmax(Q @ K^T / sqrt(d), causal_mask) @ V` in a single kernel.
/// Handles GQA natively. Query position `i` attends to key positions
/// `[0 .. offset + i + 1)`.
///
/// # Arguments
/// * `q` — query tensor of shape `(seq_q, num_heads, head_dim)`
/// * `k` — full cached keys of shape `(total_len, num_kv_heads, head_dim)`
/// * `v` — full cached values of shape `(total_len, num_kv_heads, head_dim)`
/// * `offset` — position offset for causal mask (from KV cache)
///
/// # Returns
/// Output tensor of shape `(seq_q, num_heads, head_dim)`
///
/// # Errors
/// Returns an error if the kernel launch fails
pub fn fused_attention_prefill<T: TensorDType + cudarc::driver::DeviceRepr>(
    q: &CudaTensor<T>,
    k: &CudaTensor<T>,
    v: &CudaTensor<T>,
    offset: usize,
) -> Result<CudaTensor<T>> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    assert_eq!(
        q_shape.len(),
        3,
        "Q must be 3D: (seq_q, num_heads, head_dim)"
    );
    assert_eq!(
        k_shape.len(),
        3,
        "K must be 3D: (total_len, num_kv_heads, head_dim)"
    );
    assert_eq!(
        v_shape.len(),
        3,
        "V must be 3D: (total_len, num_kv_heads, head_dim)"
    );

    let seq_q = q_shape[0];
    let num_heads = q_shape[1];
    let head_dim = q_shape[2];
    let total_len = k_shape[0];
    let num_kv_heads = k_shape[1];

    assert_eq!(k_shape[2], head_dim, "K head_dim must match Q");
    assert_eq!(v_shape, k_shape, "V shape must match K");
    assert!(
        num_heads.is_multiple_of(num_kv_heads),
        "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    );

    let scale = 1.0 / (head_dim as f32).sqrt();
    let output_shape = [seq_q, num_heads, head_dim];
    let mut output = unsafe { CudaTensor::<T>::uninit(q.context(), &output_shape)? };

    let device = q.context().device();
    ensure_fused_prefill_kernel::<T>(device)?;

    let kernel_name = format!("fused_prefill_attention_{}", kernel_suffix::<T>());
    let func = device
        .get_func("fused_prefill_attention", &kernel_name)
        .unwrap();

    let block_size = 256_usize.min(total_len.next_power_of_two());
    let shared_mem = (head_dim + block_size) * std::mem::size_of::<f32>();

    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, seq_q as u32, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    unsafe {
        func.launch(
            cfg,
            (
                output.cuda_slice_mut(),
                &q.cuda_slice(),
                &k.cuda_slice(),
                &v.cuda_slice(),
                scale,
                seq_q as i32,
                total_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                offset as i32,
            ),
        )?;
    }

    Ok(output)
}

// Register the fused attention as a replacement for the decomposed attention_kv block.
infernum_macros::define_fusion! {
    name: "attention_kv",
    fn attention_kv_fused(
        q: &CudaTensor<f32>,
        kv_cache: &mut crate::cuda::KvCache,
        layer_idx: usize,
        k_new: &CudaTensor<f32>,
        v_new: &CudaTensor<f32>,
    ) -> crate::Result<CudaTensor<f32>> {
        let q_shape = q.shape();
        let new_seq_len = q_shape[0];

        // Append new K/V to cache (writes at current_len offset, does NOT advance)
        kv_cache.append(layer_idx, k_new, v_new)?;

        // Retrieve full cached K/V including the just-appended tokens
        let total_len = kv_cache.current_len() + new_seq_len;
        let (k_full, v_full) = kv_cache.get_up_to(layer_idx, total_len);

        if new_seq_len == 1 {
            fused_attention_decode(q, &k_full, &v_full)
        } else {
            fused_attention_prefill(q, &k_full, &v_full, kv_cache.current_len())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::ops::attention;
    use crate::cuda::{CudaContext, KvCache};

    /// Reference decode attention using manual transpose+matmul+softmax.
    ///
    /// Independent of the fused kernel so we can compare against it.
    fn reference_attention_kv(
        q: &CudaTensor<f32>,
        k_full: &CudaTensor<f32>,
        v_full: &CudaTensor<f32>,
        num_heads: usize,
    ) -> CudaTensor<f32> {
        use super::super::{matmul, repeat_kv, softmax, transpose_012_to_102, transpose_last_two};

        let num_kv_heads = k_full.shape()[1];
        let head_dim = q.shape()[2];
        let scale = 1.0 / (head_dim as f32).sqrt();

        let (k_exp, v_exp) = if num_kv_heads < num_heads {
            let repeats = num_heads / num_kv_heads;
            (
                repeat_kv(k_full, repeats).unwrap(),
                repeat_kv(v_full, repeats).unwrap(),
            )
        } else {
            (
                k_full.reshape(k_full.shape()),
                v_full.reshape(v_full.shape()),
            )
        };

        // Transpose to (heads, seq, dim)
        let q_t = transpose_012_to_102(q).unwrap();
        let k_t = transpose_012_to_102(&k_exp).unwrap();
        let v_t = transpose_012_to_102(&v_exp).unwrap();
        let k_tt = transpose_last_two(&k_t).unwrap();

        // scores: (heads, 1, total_len) — scale via CPU roundtrip
        let scores = matmul(&q_t, &k_tt).unwrap();
        let mut scores_cpu = scores.to_vec().unwrap();
        for val in &mut scores_cpu {
            *val *= scale;
        }
        let scores_scaled =
            CudaTensor::from_slice(q.context(), scores.shape(), &scores_cpu).unwrap();

        // Softmax over last dim
        let probs = softmax(&scores_scaled).unwrap();

        // Output: (heads, 1, dim)
        let output = matmul(&probs, &v_t).unwrap();
        transpose_012_to_102(&output).unwrap()
    }

    #[test]
    fn test_fused_decode_basic() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let head_dim = 4;
        let total_len = 3;

        let q_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|x| (x as f32) * 0.1)
            .collect();
        let k_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| (x as f32) * 0.05)
            .collect();
        let v_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| (x as f32) * 0.02)
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &v_data).unwrap();

        let fused = fused_attention_decode(&q, &k, &v).unwrap();
        let reference = reference_attention_kv(&q, &k, &v, num_heads);

        assert_eq!(fused.shape(), &[1, num_heads, head_dim]);

        let fused_data = fused.to_vec().unwrap();
        let ref_data = reference.to_vec().unwrap();

        for (i, (&f, &r)) in fused_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-3,
                "Mismatch at {i}: fused={f}, reference={r}"
            );
        }
    }

    #[test]
    fn test_fused_decode_gqa() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let total_len = 5;

        let q_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|x| (x as f32) * 0.1)
            .collect();
        let k_data: Vec<f32> = (0..total_len * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.05)
            .collect();
        let v_data: Vec<f32> = (0..total_len * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.02)
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();
        let k =
            CudaTensor::from_slice(&ctx, &[total_len, num_kv_heads, head_dim], &k_data).unwrap();
        let v =
            CudaTensor::from_slice(&ctx, &[total_len, num_kv_heads, head_dim], &v_data).unwrap();

        let fused = fused_attention_decode(&q, &k, &v).unwrap();
        let reference = reference_attention_kv(&q, &k, &v, num_heads);

        let fused_data = fused.to_vec().unwrap();
        let ref_data = reference.to_vec().unwrap();

        for (i, (&f, &r)) in fused_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-3,
                "GQA mismatch at {i}: fused={f}, reference={r}"
            );
        }
    }

    #[test]
    fn test_fused_decode_single_token() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let q = CudaTensor::from_slice(&ctx, &[1, 1, 4], &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[1, 1, 4], &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[1, 1, 4], &[0.5, 0.5, 0.5, 0.5]).unwrap();

        let output = fused_attention_decode(&q, &k, &v).unwrap();
        let result = output.to_vec().unwrap();

        // Only one key position, so output == V
        for (i, &val) in result.iter().enumerate() {
            assert!((val - 0.5).abs() < 1e-4, "Mismatch at {i}: {val} vs 0.5");
        }
    }

    #[test]
    fn test_fused_prefill_basic() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let head_dim = 4;
        let seq_q = 3;

        let q_data: Vec<f32> = (0..seq_q * num_heads * head_dim)
            .map(|x| (x as f32) * 0.1)
            .collect();
        let k_data = q_data.clone();
        let v_data: Vec<f32> = (0..seq_q * num_heads * head_dim)
            .map(|x| (x as f32) * 0.02)
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[seq_q, num_heads, head_dim], &q_data).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[seq_q, num_heads, head_dim], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[seq_q, num_heads, head_dim], &v_data).unwrap();

        let fused = fused_attention_prefill(&q, &k, &v, 0).unwrap();
        let reference = attention(&q, &k, &v, true).unwrap();

        assert_eq!(fused.shape(), &[seq_q, num_heads, head_dim]);

        let fused_data = fused.to_vec().unwrap();
        let ref_data = reference.to_vec().unwrap();

        for (i, (&f, &r)) in fused_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-3,
                "Prefill mismatch at {i}: fused={f}, reference={r}"
            );
        }
    }

    #[test]
    fn test_fused_prefill_gqa() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let seq_q = 4;

        let q_data: Vec<f32> = (0..seq_q * num_heads * head_dim)
            .map(|x| (x as f32) * 0.05)
            .collect();
        let k_data: Vec<f32> = (0..seq_q * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.05)
            .collect();
        let v_data: Vec<f32> = (0..seq_q * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.02)
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[seq_q, num_heads, head_dim], &q_data).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[seq_q, num_kv_heads, head_dim], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[seq_q, num_kv_heads, head_dim], &v_data).unwrap();

        // Reference: expand KV then use causal attention
        let k_exp = super::super::repeat_kv(&k, num_heads / num_kv_heads).unwrap();
        let v_exp = super::super::repeat_kv(&v, num_heads / num_kv_heads).unwrap();
        let reference = attention(&q, &k_exp, &v_exp, true).unwrap();

        let fused = fused_attention_prefill(&q, &k, &v, 0).unwrap();

        let fused_data = fused.to_vec().unwrap();
        let ref_data = reference.to_vec().unwrap();

        for (i, (&f, &r)) in fused_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-3,
                "Prefill GQA mismatch at {i}: fused={f}, reference={r}"
            );
        }
    }

    /// CPU reference for causal attention with offset.
    /// Query position i attends to key positions `[0..offset+i+1)`.
    fn cpu_causal_attention(
        q: &[f32], // (seq_q, num_heads, head_dim)
        k: &[f32], // (total_len, num_kv_heads, head_dim)
        v: &[f32], // (total_len, num_kv_heads, head_dim)
        seq_q: usize,
        total_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        offset: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; seq_q * num_heads * head_dim];

        for qpos in 0..seq_q {
            for h in 0..num_heads {
                let kv_h = h * num_kv_heads / num_heads;
                let max_valid = (offset + qpos + 1).min(total_len);

                // Compute scores
                let mut scores = vec![f32::NEG_INFINITY; total_len];
                for t in 0..max_valid {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        let qi = q[qpos * num_heads * head_dim + h * head_dim + d];
                        let ki = k[t * num_kv_heads * head_dim + kv_h * head_dim + d];
                        dot += qi * ki;
                    }
                    scores[t] = dot * scale;
                }

                // Softmax over valid positions
                let max_s = scores[..max_valid]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for s in &mut scores[..max_valid] {
                    *s = (*s - max_s).exp();
                    sum_exp += *s;
                }
                for s in &mut scores[..max_valid] {
                    *s /= sum_exp;
                }
                for s in &mut scores[max_valid..] {
                    *s = 0.0;
                }

                // Weighted sum of V
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for t in 0..max_valid {
                        let vi = v[t * num_kv_heads * head_dim + kv_h * head_dim + d];
                        acc += scores[t] * vi;
                    }
                    output[qpos * num_heads * head_dim + h * head_dim + d] = acc;
                }
            }
        }
        output
    }

    #[test]
    fn test_fused_prefill_with_offset() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // Simulate: 3 tokens already in cache, then prefill 2 more
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let cache_len = 3;
        let new_seq = 2;

        let mut kv_cache = KvCache::new(&ctx, 1, 32, num_kv_heads, head_dim).unwrap();

        // Populate cache with initial tokens
        let k_init: Vec<f32> = (0..cache_len * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.05)
            .collect();
        let v_init: Vec<f32> = (0..cache_len * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.02)
            .collect();
        let k_init_t =
            CudaTensor::from_slice(&ctx, &[cache_len, num_kv_heads, head_dim], &k_init).unwrap();
        let v_init_t =
            CudaTensor::from_slice(&ctx, &[cache_len, num_kv_heads, head_dim], &v_init).unwrap();
        kv_cache.append(0, &k_init_t, &v_init_t).unwrap();
        kv_cache.advance(cache_len);

        // New tokens
        let q_data: Vec<f32> = (0..new_seq * num_heads * head_dim)
            .map(|x| (x as f32) * 0.1)
            .collect();
        let k_new_data: Vec<f32> = (0..new_seq * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.03)
            .collect();
        let v_new_data: Vec<f32> = (0..new_seq * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.01)
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[new_seq, num_heads, head_dim], &q_data).unwrap();
        let k_new =
            CudaTensor::from_slice(&ctx, &[new_seq, num_kv_heads, head_dim], &k_new_data).unwrap();
        let v_new =
            CudaTensor::from_slice(&ctx, &[new_seq, num_kv_heads, head_dim], &v_new_data).unwrap();

        // Fused path: append to our cache, get full K/V, call fused_attention_prefill
        kv_cache.append(0, &k_new, &v_new).unwrap();
        let total_len = cache_len + new_seq;
        let (k_full, v_full) = kv_cache.get_up_to(0, total_len);
        let fused = fused_attention_prefill(&q, &k_full, &v_full, cache_len).unwrap();

        // CPU reference with offset-aware causal mask
        let k_full_cpu = k_full.to_vec().unwrap();
        let v_full_cpu = v_full.to_vec().unwrap();
        let ref_data = cpu_causal_attention(
            &q_data,
            &k_full_cpu,
            &v_full_cpu,
            new_seq,
            total_len,
            num_heads,
            num_kv_heads,
            head_dim,
            cache_len,
        );

        let fused_data = fused.to_vec().unwrap();

        for (i, (&f, &r)) in fused_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-3,
                "Prefill+offset mismatch at {i}: fused={f}, reference={r}"
            );
        }
    }

    #[test]
    fn test_fused_decode_via_kv_cache() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 8;
        let prefill_len = 4;

        let mut kv_cache = KvCache::new(&ctx, 1, 32, num_kv_heads, head_dim).unwrap();

        // Prefill KV cache
        let kv_data: Vec<f32> = (0..prefill_len * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.01)
            .collect();
        let k =
            CudaTensor::from_slice(&ctx, &[prefill_len, num_kv_heads, head_dim], &kv_data).unwrap();
        let v =
            CudaTensor::from_slice(&ctx, &[prefill_len, num_kv_heads, head_dim], &kv_data).unwrap();
        kv_cache.append(0, &k, &v).unwrap();
        kv_cache.advance(prefill_len);

        // Decode one token
        let q1_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|x| (x as f32) * 0.02)
            .collect();
        let kv1_data: Vec<f32> = (0..num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.02)
            .collect();
        let q1 = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q1_data).unwrap();
        let k1 = CudaTensor::from_slice(&ctx, &[1, num_kv_heads, head_dim], &kv1_data).unwrap();
        let v1 = CudaTensor::from_slice(&ctx, &[1, num_kv_heads, head_dim], &kv1_data).unwrap();

        // Append new KV and get full cache
        kv_cache.append(0, &k1, &v1).unwrap();
        let total_len = prefill_len + 1;
        let (k_full, v_full) = kv_cache.get_up_to(0, total_len);

        // Fused kernel
        let fused_output = fused_attention_decode(&q1, &k_full, &v_full).unwrap();

        // Independent reference (transpose + matmul + softmax)
        let ref_output = reference_attention_kv(&q1, &k_full, &v_full, num_heads);

        let fused_data = fused_output.to_vec().unwrap();
        let ref_data = ref_output.to_vec().unwrap();

        for (i, (&f, &r)) in fused_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-3,
                "Decode KV mismatch at {i}: fused={f}, reference={r}"
            );
        }
    }

    #[test]
    fn test_fused_decode_larger_context() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;
        let total_len = 64;

        let q_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|x| ((x as f32) * 0.01).sin())
            .collect();
        let k_data: Vec<f32> = (0..total_len * num_kv_heads * head_dim)
            .map(|x| ((x as f32) * 0.007).cos())
            .collect();
        let v_data: Vec<f32> = (0..total_len * num_kv_heads * head_dim)
            .map(|x| ((x as f32) * 0.003).sin())
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();
        let k =
            CudaTensor::from_slice(&ctx, &[total_len, num_kv_heads, head_dim], &k_data).unwrap();
        let v =
            CudaTensor::from_slice(&ctx, &[total_len, num_kv_heads, head_dim], &v_data).unwrap();

        let fused = fused_attention_decode(&q, &k, &v).unwrap();
        let reference = reference_attention_kv(&q, &k, &v, num_heads);

        let fused_data = fused.to_vec().unwrap();
        let ref_data = reference.to_vec().unwrap();

        for (i, (&f, &r)) in fused_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-2,
                "Large context mismatch at {i}: fused={f}, reference={r}"
            );
        }
    }
}
