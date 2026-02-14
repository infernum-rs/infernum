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
/// Input: (batch, seq, seq), output: (batch, seq, seq)
fn softmax_batched(scores: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    let shape = scores.shape();
    let batch = shape[0];
    let seq = shape[1];

    // Reshape to (batch * seq, seq) for softmax
    let flat = scores.reshape(&[batch * seq, seq]);
    let probs = super::softmax(&flat)?;
    Ok(probs.reshape(&[batch, seq, seq]))
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
                scores.cuda_slice(),
                batch as i32,
                seq as i32,
            ),
        )?;
    }

    Ok(output)
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
}
