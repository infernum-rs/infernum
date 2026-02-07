//! Rotary Positional Embeddings (RoPE)

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::missing_panics_doc
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use crate::cuda::{CudaContext, CudaTensor};
use crate::tensor::Tensor;
use crate::Result;

const ROPE_KERNEL: &str = r#"
extern "C" __global__ void rope_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int position_offset
) {
    // Each block handles one (seq_pos, head) pair
    // Each thread handles one pair of dimensions
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int pair_idx = threadIdx.x;  // Which pair of dimensions (0..head_dim/2)
    
    if (pair_idx >= head_dim / 2) return;
    
    const int pos = position_offset + seq_idx;
    
    // Input layout: (seq_len, num_heads, head_dim)
    const int base_idx = (seq_idx * num_heads + head_idx) * head_dim;
    const int idx0 = base_idx + pair_idx * 2;
    const int idx1 = base_idx + pair_idx * 2 + 1;
    
    // Cache layout: (max_seq_len, head_dim/2)
    const int cache_idx = pos * (head_dim / 2) + pair_idx;
    
    float cos_val = cos_cache[cache_idx];
    float sin_val = sin_cache[cache_idx];
    
    float x0 = input[idx0];
    float x1 = input[idx1];
    
    output[idx0] = x0 * cos_val - x1 * sin_val;
    output[idx1] = x0 * sin_val + x1 * cos_val;
}
"#;

/// Precompute cosine and sine caches for RoPE
///
/// # Arguments
/// * `ctx` - CUDA context
/// * `max_seq_len` - Maximum sequence length
/// * `head_dim` - Dimension of each attention head
/// * `base` - Base frequency (default 10000.0)
///
/// # Returns
/// (cos_cache, sin_cache) tensors of shape (max_seq_len, head_dim/2)
///
/// # Errors
/// Returns an error if allocation fails
pub fn precompute_rope_cache(
    ctx: &CudaContext,
    max_seq_len: usize,
    head_dim: usize,
    base: f32,
) -> Result<(CudaTensor<f32>, CudaTensor<f32>)> {
    let half_dim = head_dim / 2;

    let mut cos_data = vec![0.0_f32; max_seq_len * half_dim];
    let mut sin_data = vec![0.0_f32; max_seq_len * half_dim];

    for pos in 0..max_seq_len {
        for i in 0..half_dim {
            let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            cos_data[pos * half_dim + i] = angle.cos();
            sin_data[pos * half_dim + i] = angle.sin();
        }
    }

    let cos_cache = CudaTensor::from_slice(ctx, &[max_seq_len, half_dim], &cos_data)?;
    let sin_cache = CudaTensor::from_slice(ctx, &[max_seq_len, half_dim], &sin_data)?;

    Ok((cos_cache, sin_cache))
}

/// Apply rotary positional embeddings to Q and K tensors
///
/// # Arguments
/// * `input` - Input tensor of shape (seq_len, num_heads, head_dim)
/// * `cos_cache` - Precomputed cos cache of shape (max_seq_len, head_dim/2)
/// * `sin_cache` - Precomputed sin cache of shape (max_seq_len, head_dim/2)
/// * `position_offset` - Starting position (for incremental decoding)
///
/// # Errors
/// Returns an error if the operation fails
pub fn apply_rope(
    input: &CudaTensor<f32>,
    cos_cache: &CudaTensor<f32>,
    sin_cache: &CudaTensor<f32>,
    position_offset: usize,
) -> Result<CudaTensor<f32>> {
    let shape = input.shape();
    assert_eq!(
        shape.len(),
        3,
        "Input must be 3D: (seq_len, num_heads, head_dim)"
    );

    let seq_len = shape[0];
    let num_heads = shape[1];
    let head_dim = shape[2];

    assert_eq!(head_dim % 2, 0, "head_dim must be even");

    let mut output = unsafe { CudaTensor::<f32>::uninit(input.context(), shape)? };

    let device = input.context().device();

    // Compile kernel
    let module_name = "rope";
    if !device.has_func(module_name, "rope_f32") {
        let ptx = cudarc::nvrtc::safe::compile_ptx(ROPE_KERNEL)?;
        device.load_ptx(ptx, module_name, &["rope_f32"])?;
    }

    let func = device.get_func(module_name, "rope_f32").unwrap();

    let cfg = LaunchConfig {
        grid_dim: (seq_len as u32, num_heads as u32, 1),
        block_dim: ((head_dim / 2) as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(
            cfg,
            (
                output.cuda_slice_mut(),
                input.cuda_slice(),
                cos_cache.cuda_slice(),
                sin_cache.cuda_slice(),
                seq_len as i32,
                num_heads as i32,
                head_dim as i32,
                position_offset as i32,
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
    fn test_rope_identity_at_zero() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let seq_len = 2;
        let num_heads = 2;
        let head_dim = 4;

        let (cos_cache, sin_cache) = precompute_rope_cache(&ctx, 128, head_dim, 10000.0).unwrap();

        // At position 0, angle = 0, cos = 1, sin = 0
        // So output should equal input
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // seq=0, head=0
            5.0, 6.0, 7.0, 8.0, // seq=0, head=1
            9.0, 10.0, 11.0, 12.0, // seq=1, head=0
            13.0, 14.0, 15.0, 16.0, // seq=1, head=1
        ];

        let input =
            CudaTensor::from_slice(&ctx, &[seq_len, num_heads, head_dim], &input_data).unwrap();

        let output = apply_rope(&input, &cos_cache, &sin_cache, 0).unwrap();
        let result = output.to_vec().unwrap();

        // Position 0: cos(0) = 1, sin(0) = 0, so x' = x
        // Position 1 will have rotation applied
        assert!((result[0] - 1.0).abs() < 1e-5); // seq=0, no rotation
        assert!((result[1] - 2.0).abs() < 1e-5);
    }
}
