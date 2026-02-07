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
    // Q: (seq, heads, dim) -> (heads, seq, dim)
    // K: (seq, heads, dim) -> (heads, seq, dim) -> transpose to (heads, dim, seq)
    // V: (seq, heads, dim) -> (heads, seq, dim)

    // For simplicity, we'll transpose in memory
    let q_transposed = transpose_012_to_102(q)?;
    let k_transposed = transpose_012_to_102(k)?;
    let v_transposed = transpose_012_to_102(v)?;

    // K^T for Q @ K^T
    let k_t = transpose_last_two(&k_transposed)?;

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
    transpose_012_to_102(&output_transposed)
}

/// Transpose tensor from (a, b, c) to (b, a, c)
fn transpose_012_to_102(input: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    let shape = input.shape();
    assert_eq!(shape.len(), 3);

    let a = shape[0];
    let b = shape[1];
    let c = shape[2];

    let output_shape = [b, a, c];

    // This is inefficient but correct - copy element by element
    // A proper implementation would use a CUDA kernel
    let input_data = input.to_vec()?;
    let mut output_data = vec![0.0_f32; a * b * c];

    for i in 0..a {
        for j in 0..b {
            for k in 0..c {
                let src_idx = i * b * c + j * c + k;
                let dst_idx = j * a * c + i * c + k;
                output_data[dst_idx] = input_data[src_idx];
            }
        }
    }

    let output = CudaTensor::from_slice(input.context(), &output_shape, &output_data)?;
    Ok(output)
}

/// Transpose last two dimensions: (a, b, c) -> (a, c, b)
fn transpose_last_two(input: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    let shape = input.shape();
    assert_eq!(shape.len(), 3);

    let a = shape[0];
    let b = shape[1];
    let c = shape[2];

    let output_shape = [a, c, b];

    let input_data = input.to_vec()?;
    let mut output_data = vec![0.0_f32; a * b * c];

    for i in 0..a {
        for j in 0..b {
            for k in 0..c {
                let src_idx = i * b * c + j * c + k;
                let dst_idx = i * c * b + k * b + j;
                output_data[dst_idx] = input_data[src_idx];
            }
        }
    }

    CudaTensor::from_slice(input.context(), &output_shape, &output_data)
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
/// Input: (batch, seq, seq), applies causal mask
fn causal_softmax(scores: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    let shape = scores.shape();
    let batch = shape[0];
    let seq = shape[1];

    // Apply mask and softmax row by row
    let scores_data = scores.to_vec()?;
    let mut probs_data = vec![0.0_f32; batch * seq * seq];

    for b in 0..batch {
        for q in 0..seq {
            let row_start = b * seq * seq + q * seq;
            let _row_end = row_start + seq;

            // Find max (for stability) over valid positions
            let max_val = scores_data[row_start..row_start + q + 1]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Compute exp and sum for valid positions
            let mut sum = 0.0_f32;
            for k in 0..=q {
                let exp_val = (scores_data[row_start + k] - max_val).exp();
                probs_data[row_start + k] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for k in 0..=q {
                probs_data[row_start + k] /= sum;
            }

            // Masked positions stay at 0
        }
    }

    CudaTensor::from_slice(scores.context(), shape, &probs_data)
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
