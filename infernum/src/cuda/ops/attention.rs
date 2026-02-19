//! Attention operation

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::range_plus_one,
    clippy::no_effect_underscore_binding,
    clippy::manual_div_ceil
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use super::matmul;
use crate::cuda::CudaTensor;
use crate::tensor::Tensor;
use crate::Result;

const SCALE_KERNEL: &str = r#"
extern "C" __global__ void scale_f32(
    float* __restrict__ data,
    const float scale,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}
"#;

const CAUSAL_SOFTMAX_BATCHED_KERNEL: &str = r#"
#define INFINITY __int_as_float(0x7f800000)

// Causal softmax over (batch, seq_q, seq_k) attention scores.
// One block per (batch, query) row. Causal mask: allow k <= query_idx.
// Layout: input/output are (batch * seq * seq) contiguous, row-major.
extern "C" __global__ void causal_softmax_batched_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int batch,
    const int seq
) {
    const int row_idx = blockIdx.x;  // which (batch, query) pair
    const int batch_idx = row_idx / seq;
    const int query_idx = row_idx % seq;
    const int tid = threadIdx.x;

    extern __shared__ float shared[];

    const int row_offset = (batch_idx * seq + query_idx) * seq;
    const float* row_input = input + row_offset;
    float* row_output = output + row_offset;

    const int max_valid_k = query_idx + 1;

    // Step 1: Find max over valid positions
    float local_max = -INFINITY;
    for (int i = tid; i < max_valid_k; i += blockDim.x) {
        local_max = fmaxf(local_max, row_input[i]);
    }

    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = shared[0];
    __syncthreads();

    // Step 2: Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < seq; i += blockDim.x) {
        if (i < max_valid_k) {
            float exp_val = expf(row_input[i] - max_val);
            row_output[i] = exp_val;
            local_sum += exp_val;
        } else {
            row_output[i] = 0.0f;
        }
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float sum_val = shared[0];

    // Step 3: Normalize
    for (int i = tid; i < max_valid_k; i += blockDim.x) {
        row_output[i] /= sum_val;
    }
}
"#;

#[allow(dead_code)]
const CAUSAL_SOFTMAX_OFFSET_KERNEL: &str = r#"
#define INFINITY __int_as_float(0x7f800000)

// Causal softmax over (batch, seq_q, seq_k) attention scores with position offset.
// For prefill with KV cache: query position i can attend to key positions [0..offset + i + 1].
// One block per (batch, query) row.
// Layout: input/output are (batch * seq_q * seq_k) contiguous, row-major.
extern "C" __global__ void causal_softmax_offset_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int batch,
    const int seq_q,
    const int seq_k,
    const int offset
) {
    const int row_idx = blockIdx.x;  // which (batch, query) pair
    const int query_idx = row_idx % seq_q;
    const int tid = threadIdx.x;

    extern __shared__ float shared[];

    const int row_offset = row_idx * seq_k;
    const float* row_input = input + row_offset;
    float* row_output = output + row_offset;

    const int max_valid_k = offset + query_idx + 1;

    // Step 1: Find max over valid positions
    float local_max = -INFINITY;
    for (int i = tid; i < max_valid_k && i < seq_k; i += blockDim.x) {
        local_max = fmaxf(local_max, row_input[i]);
    }

    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = shared[0];
    __syncthreads();

    // Step 2: Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < seq_k; i += blockDim.x) {
        if (i < max_valid_k) {
            float exp_val = expf(row_input[i] - max_val);
            row_output[i] = exp_val;
            local_sum += exp_val;
        } else {
            row_output[i] = 0.0f;
        }
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float sum_val = shared[0];

    // Step 3: Normalize
    for (int i = tid; i < max_valid_k && i < seq_k; i += blockDim.x) {
        row_output[i] /= sum_val;
    }
}
"#;

/// Naive attention without KV cache
///
/// Computes: softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// # Arguments
/// * `q` - Query tensor of shape (seq_len, num_heads, head_dim)
/// * `k` - Key tensor of shape (seq_len, num_heads, head_dim)
/// * `v` - Value tensor of shape (seq_len, num_heads, head_dim)
/// * `causal` - Whether to apply causal masking
///
/// # Returns
/// Output tensor of shape (seq_len, num_heads, head_dim)
///
/// # Errors
/// Returns an error if the operation fails
pub fn attention(
    q: &CudaTensor<f32>,
    k: &CudaTensor<f32>,
    v: &CudaTensor<f32>,
    causal: bool,
) -> Result<CudaTensor<f32>> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    assert_eq!(
        q_shape.len(),
        3,
        "Q must be 3D: (seq_len, num_heads, head_dim)"
    );
    assert_eq!(
        k_shape.len(),
        3,
        "K must be 3D: (seq_len, num_heads, head_dim)"
    );
    assert_eq!(
        v_shape.len(),
        3,
        "V must be 3D: (seq_len, num_heads, head_dim)"
    );

    let seq_len = q_shape[0];
    let num_heads = q_shape[1];
    let head_dim = q_shape[2];

    assert_eq!(k_shape[0], seq_len, "K seq_len must match Q");
    assert_eq!(k_shape[1], num_heads, "K num_heads must match Q");
    assert_eq!(k_shape[2], head_dim, "K head_dim must match Q");
    assert_eq!(v_shape[0], seq_len, "V seq_len must match Q");
    assert_eq!(v_shape[1], num_heads, "V num_heads must match Q");
    assert_eq!(v_shape[2], head_dim, "V head_dim must match Q");

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Reshape for batched matmul: (num_heads, seq_len, head_dim)
    // Q/K/V: (seq, heads, dim) -> (heads, seq, dim)
    let q_transposed = super::transpose_012_to_102(q)?;
    let k_transposed = super::transpose_012_to_102(k)?;
    let v_transposed = super::transpose_012_to_102(v)?;

    // K^T for Q @ K^T: (heads, seq, dim) -> (heads, dim, seq)
    let k_t = super::transpose_last_two(&k_transposed)?;

    // Compute attention scores: (heads, seq, dim) @ (heads, dim, seq) -> (heads, seq, seq)
    let mut scores = matmul(&q_transposed, &k_t)?;

    // Scale
    scale_inplace(&mut scores, scale)?;

    // Apply causal mask and softmax
    let probs = if causal {
        causal_softmax(&scores)?
    } else {
        softmax_batched(&scores)?
    };

    // Compute output: (heads, seq, seq) @ (heads, seq, dim) -> (heads, seq, dim)
    let output_transposed = matmul(&probs, &v_transposed)?;

    // Transpose back: (heads, seq, dim) -> (seq, heads, dim)
    super::transpose_012_to_102(&output_transposed)
}

/// Attention with KV cache for incremental decoding
///
/// Appends `k_new` and `v_new` to the cache for `layer_idx`, then computes
/// attention of `q` against all cached keys and values (including the newly
/// appended ones).
///
/// Does **not** call `kv_cache.advance()` — the caller must do that once
/// after all layers have been processed.
///
/// For single-token decode (`new_seq_len == 1`), no causal mask is needed.
/// For multi-token prefill, a causal mask is applied with the correct offset.
///
/// # Arguments
/// * `q` - Query tensor of shape `(new_seq_len, num_heads, head_dim)`
/// * `kv_cache` - Mutable reference to the KV cache
/// * `layer_idx` - Which transformer layer
/// * `k_new` - New key tensor of shape `(new_seq_len, num_kv_heads, head_dim)`
/// * `v_new` - New value tensor of shape `(new_seq_len, num_kv_heads, head_dim)`
///
/// # Returns
/// Output tensor of shape `(new_seq_len, num_heads, head_dim)`
///
/// # Errors
/// Returns an error if the operation fails
pub fn attention_kv(
    q: &CudaTensor<f32>,
    kv_cache: &mut crate::cuda::KvCache,
    layer_idx: usize,
    k_new: &CudaTensor<f32>,
    v_new: &CudaTensor<f32>,
) -> Result<CudaTensor<f32>> {
    let q_shape = q.shape();
    assert_eq!(
        q_shape.len(),
        3,
        "Q must be 3D: (new_seq, num_heads, head_dim)"
    );

    let new_seq_len = q_shape[0];

    // Append new K/V to cache (writes at current_len offset, does NOT advance)
    kv_cache.append(layer_idx, k_new, v_new)?;

    // Retrieve full cached K/V including the just-appended tokens
    let total_len = kv_cache.current_len() + new_seq_len;
    let (k_full, v_full) = kv_cache.get_up_to(layer_idx, total_len);
    // k_full, v_full: (total_len, num_kv_heads, head_dim)

    // Fused kernels handle GQA, scaling, softmax, and output in a single
    // kernel launch — no transposes, repeat_kv, or intermediate allocations.
    if new_seq_len == 1 {
        super::fused_attention_decode(q, &k_full, &v_full)
    } else {
        super::fused_attention_prefill(q, &k_full, &v_full, kv_cache.current_len())
    }
}

/// Scale tensor in place
fn scale_inplace(tensor: &mut CudaTensor<f32>, scale: f32) -> Result<()> {
    let n = tensor.numel();
    let device = tensor.context().device();

    // Compile kernel
    let module_name = "scale";
    if !device.has_func(module_name, "scale_f32") {
        let ptx = cudarc::nvrtc::safe::compile_ptx(SCALE_KERNEL)?;
        device.load_ptx(ptx, module_name, &["scale_f32"])?;
    }

    let func = device.get_func(module_name, "scale_f32").unwrap();

    let block_size = 256;
    let grid_size = (n + block_size - 1) / block_size;

    let cfg = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(cfg, (tensor.cuda_slice_mut(), scale, n as i32))?;
    }

    Ok(())
}

/// Softmax over batched attention scores
/// Input: (batch, seq_q, seq_k), output: (batch, seq_q, seq_k)
fn softmax_batched(scores: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    let shape = scores.shape();
    let batch = shape[0];
    let seq_q = shape[1];
    let seq_k = shape[2];

    // Reshape to (batch * seq_q, seq_k) for softmax
    let flat = scores.reshape(&[batch * seq_q, seq_k]);
    let probs = super::softmax(&flat)?;
    Ok(probs.reshape(&[batch, seq_q, seq_k]))
}

/// Causal softmax for attention
/// Input: (batch, seq, seq), applies causal mask entirely on GPU
fn causal_softmax(scores: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    let shape = scores.shape();
    let batch = shape[0];
    let seq = shape[1];

    // One block per (batch, query) row
    let num_rows = batch * seq;

    let mut output = unsafe { CudaTensor::<f32>::uninit(scores.context(), shape)? };

    let device = scores.context().device();

    let module_name = "causal_softmax_batched";
    if !device.has_func(module_name, "causal_softmax_batched_f32") {
        let ptx = cudarc::nvrtc::safe::compile_ptx(CAUSAL_SOFTMAX_BATCHED_KERNEL)?;
        device.load_ptx(ptx, module_name, &["causal_softmax_batched_f32"])?;
    }

    let func = device
        .get_func(module_name, "causal_softmax_batched_f32")
        .unwrap();

    let block_size = 256_usize.min(seq.next_power_of_two());
    let shared_mem = block_size * std::mem::size_of::<f32>();

    let cfg = LaunchConfig {
        grid_dim: (num_rows as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    unsafe {
        func.launch(
            cfg,
            (
                output.cuda_slice_mut(),
                &scores.cuda_slice(),
                batch as i32,
                seq as i32,
            ),
        )?;
    }

    Ok(output)
}

/// Causal softmax with position offset for KV-cache prefill.
///
/// Input: `(batch, seq_q, seq_k)` — query position `i` attends to key `[0..offset + i + 1)`.
// TODO(optimizer): Once `define_block!` + fusion rules land, this becomes an internal
// op composed automatically within the decomposed attention block.
#[allow(dead_code)]
fn causal_softmax_with_offset(scores: &CudaTensor<f32>, offset: usize) -> Result<CudaTensor<f32>> {
    let shape = scores.shape();
    let batch = shape[0];
    let seq_q = shape[1];
    let seq_k = shape[2];

    let num_rows = batch * seq_q;
    let mut output = unsafe { CudaTensor::<f32>::uninit(scores.context(), shape)? };

    let device = scores.context().device();

    let module_name = "causal_softmax_offset";
    if !device.has_func(module_name, "causal_softmax_offset_f32") {
        let ptx = cudarc::nvrtc::safe::compile_ptx(CAUSAL_SOFTMAX_OFFSET_KERNEL)?;
        device.load_ptx(ptx, module_name, &["causal_softmax_offset_f32"])?;
    }

    let func = device
        .get_func(module_name, "causal_softmax_offset_f32")
        .unwrap();

    let block_size = 256_usize.min(seq_k.next_power_of_two());
    let shared_mem = block_size * std::mem::size_of::<f32>();

    let cfg = LaunchConfig {
        grid_dim: (num_rows as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    unsafe {
        func.launch(
            cfg,
            (
                output.cuda_slice_mut(),
                &scores.cuda_slice(),
                batch as i32,
                seq_q as i32,
                seq_k as i32,
                offset as i32,
            ),
        )?;
    }

    Ok(output)
}

/// Decomposed attention with KV cache — readable, composable implementation.
///
/// This is the explicit op-by-op version of `attention_kv`: transpose, repeat_kv,
/// matmul, scale, softmax, matmul, transpose. It produces identical results to the
/// fused kernels used in `attention_kv` but with more intermediate allocations and
/// kernel launches.
///
/// Kept as the readable "source of truth" for how attention works. When the
/// `define_block!` macro + graph optimizer lands (see `docs/initial-plan.md`,
/// Phase 3), this becomes the body of the attention block definition, and the fused
/// kernels in `fused_attention.rs` become fusion rules that the optimizer
/// auto-substitutes. Until then, `attention_kv` calls the fused kernels directly.
// TODO(optimizer): Convert to a `define_block!` definition. The optimizer should
// recognize the [Transpose, RepeatKV, MatMul, Scale, Softmax, MatMul, Transpose]
// pattern and replace it with `fused_attention_decode` / `fused_attention_prefill`.
#[allow(dead_code)]
fn attention_kv_decomposed(
    q: &CudaTensor<f32>,
    kv_cache: &mut crate::cuda::KvCache,
    layer_idx: usize,
    k_new: &CudaTensor<f32>,
    v_new: &CudaTensor<f32>,
) -> Result<CudaTensor<f32>> {
    let q_shape = q.shape();
    assert_eq!(
        q_shape.len(),
        3,
        "Q must be 3D: (new_seq, num_heads, head_dim)"
    );

    let new_seq_len = q_shape[0];
    let num_heads = q_shape[1];
    let head_dim = q_shape[2];

    // Append new K/V to cache (writes at current_len offset, does NOT advance)
    kv_cache.append(layer_idx, k_new, v_new)?;

    // Retrieve full cached K/V including the just-appended tokens
    let total_len = kv_cache.current_len() + new_seq_len;
    let (k_full, v_full) = kv_cache.get_up_to(layer_idx, total_len);
    // k_full, v_full: (total_len, num_kv_heads, head_dim)

    // GQA: expand KV heads to match Q heads if necessary
    let num_kv_heads = k_full.shape()[1];
    let (k_expanded, v_expanded) = if num_kv_heads < num_heads {
        let num_repeats = num_heads / num_kv_heads;
        (
            super::repeat_kv(&k_full, num_repeats)?,
            super::repeat_kv(&v_full, num_repeats)?,
        )
    } else {
        (k_full, v_full)
    };
    // k_expanded, v_expanded: (total_len, num_heads, head_dim)

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Transpose to (num_heads, seq, head_dim) for batched matmul
    let q_t = super::transpose_012_to_102(q)?; // (heads, new_seq, dim)
    let k_t = super::transpose_012_to_102(&k_expanded)?; // (heads, total, dim)
    let v_t = super::transpose_012_to_102(&v_expanded)?; // (heads, total, dim)

    // K^T: (heads, dim, total)
    let k_tt = super::transpose_last_two(&k_t)?;

    // Scores: (heads, new_seq, dim) @ (heads, dim, total) -> (heads, new_seq, total)
    let mut scores = matmul(&q_t, &k_tt)?;
    scale_inplace(&mut scores, scale)?;

    // Apply softmax (with causal mask for prefill, without for decode)
    let probs = if new_seq_len == 1 {
        // Single-token decode: attend to all past + current, no mask needed
        softmax_batched(&scores)?
    } else {
        // Prefill: need causal mask with offset so that query position i
        // can attend to key positions [0..cache_offset + i + 1]
        causal_softmax_with_offset(&scores, kv_cache.current_len())?
    };

    // Output: (heads, new_seq, total) @ (heads, total, dim) -> (heads, new_seq, dim)
    let output_transposed = matmul(&probs, &v_t)?;

    // Transpose back: (heads, new_seq, dim) -> (new_seq, heads, dim)
    super::transpose_012_to_102(&output_transposed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;

    #[test]
    fn test_attention_shapes() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 8;

        let data: Vec<f32> = (0..seq_len * num_heads * head_dim)
            .map(|x| (x as f32) * 0.01)
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[seq_len, num_heads, head_dim], &data).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[seq_len, num_heads, head_dim], &data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[seq_len, num_heads, head_dim], &data).unwrap();

        let output = attention(&q, &k, &v, true).unwrap();

        assert_eq!(output.shape(), &[seq_len, num_heads, head_dim]);
    }

    #[test]
    fn test_attention_causal_first_token() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // For first token with causal mask, it should only attend to itself
        let q = CudaTensor::from_slice(&ctx, &[1, 1, 4], &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[1, 1, 4], &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[1, 1, 4], &[0.5, 0.5, 0.5, 0.5]).unwrap();

        let output = attention(&q, &k, &v, true).unwrap();
        let result = output.to_vec().unwrap();

        // Should output exactly V since there's only one token
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - 0.5).abs() < 1e-4,
                "Mismatch at {}: {} vs 0.5",
                i,
                val
            );
        }
    }

    #[test]
    fn test_attention_kv_prefill_shapes() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 8;
        let seq_len = 4;

        let mut kv_cache = crate::cuda::KvCache::new(&ctx, 1, 32, num_kv_heads, head_dim).unwrap();

        let q_data: Vec<f32> = (0..seq_len * num_heads * head_dim)
            .map(|x| (x as f32) * 0.01)
            .collect();
        let kv_data: Vec<f32> = (0..seq_len * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.01)
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[seq_len, num_heads, head_dim], &q_data).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &kv_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &kv_data).unwrap();

        let output = attention_kv(&q, &mut kv_cache, 0, &k, &v).unwrap();
        kv_cache.advance(seq_len);

        assert_eq!(output.shape(), &[seq_len, num_heads, head_dim]);
        assert_eq!(kv_cache.current_len(), seq_len);
    }

    #[test]
    fn test_attention_kv_decode_shapes() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 8;
        let prefill_len = 3;

        let mut kv_cache = crate::cuda::KvCache::new(&ctx, 1, 32, num_kv_heads, head_dim).unwrap();

        // Prefill
        let q_data: Vec<f32> = (0..prefill_len * num_heads * head_dim)
            .map(|x| (x as f32) * 0.01)
            .collect();
        let kv_data: Vec<f32> = (0..prefill_len * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.01)
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[prefill_len, num_heads, head_dim], &q_data).unwrap();
        let k =
            CudaTensor::from_slice(&ctx, &[prefill_len, num_kv_heads, head_dim], &kv_data).unwrap();
        let v =
            CudaTensor::from_slice(&ctx, &[prefill_len, num_kv_heads, head_dim], &kv_data).unwrap();

        let _prefill_out = attention_kv(&q, &mut kv_cache, 0, &k, &v).unwrap();
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

        let decode_out = attention_kv(&q1, &mut kv_cache, 0, &k1, &v1).unwrap();
        kv_cache.advance(1);

        assert_eq!(decode_out.shape(), &[1, num_heads, head_dim]);
        assert_eq!(kv_cache.current_len(), prefill_len + 1);
    }

    #[test]
    fn test_attention_kv_single_token_matches_attention() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // With a single token, attention_kv should match regular attention
        let q = CudaTensor::from_slice(&ctx, &[1, 1, 4], &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[1, 1, 4], &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[1, 1, 4], &[0.5, 0.5, 0.5, 0.5]).unwrap();

        // Regular attention
        let expected = attention(&q, &k, &v, true).unwrap().to_vec().unwrap();

        // KV-cache attention
        let mut kv_cache = crate::cuda::KvCache::new(&ctx, 1, 32, 1, 4).unwrap();
        let actual = attention_kv(&q, &mut kv_cache, 0, &k, &v)
            .unwrap()
            .to_vec()
            .unwrap();

        for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (e - a).abs() < 1e-4,
                "Mismatch at {i}: expected {e}, got {a}"
            );
        }
    }
}
