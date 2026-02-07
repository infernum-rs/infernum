//! Softmax operation

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::missing_panics_doc
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use crate::cuda::CudaTensor;
use crate::tensor::Tensor;
use crate::Result;

const SOFTMAX_KERNEL: &str = r#"
// CUDA doesn't have INFINITY in NVRTC, use the IEEE 754 representation
#define INFINITY __int_as_float(0x7f800000)

extern "C" __global__ void softmax_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int row_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    extern __shared__ float shared[];
    
    const float* row_input = input + row * row_size;
    float* row_output = output + row * row_size;
    
    // Step 1: Find max (for numerical stability)
    float local_max = -INFINITY;
    for (int i = tid; i < row_size; i += blockDim.x) {
        local_max = fmaxf(local_max, row_input[i]);
    }
    
    shared[tid] = local_max;
    __syncthreads();
    
    // Reduce to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    
    float max_val = shared[0];
    __syncthreads();
    
    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float exp_val = expf(row_input[i] - max_val);
        row_output[i] = exp_val;  // Temporarily store exp values
        local_sum += exp_val;
    }
    
    shared[tid] = local_sum;
    __syncthreads();
    
    // Reduce to find sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    float sum_val = shared[0];
    
    // Step 3: Normalize
    for (int i = tid; i < row_size; i += blockDim.x) {
        row_output[i] /= sum_val;
    }
}

// Softmax with causal mask for attention
extern "C" __global__ void softmax_causal_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int row_size,
    const int query_idx,      // Which query position this is
    const int position_offset // For KV cache scenarios
) {
    const int row = blockIdx.x;  // Which head
    const int tid = threadIdx.x;
    
    extern __shared__ float shared[];
    
    const float* row_input = input + row * row_size;
    float* row_output = output + row * row_size;
    
    // The causal mask means: for query at position q, 
    // we can only attend to key positions k where k <= q
    const int max_valid_k = query_idx + position_offset + 1;
    
    // Step 1: Find max (only over valid positions)
    float local_max = -INFINITY;
    for (int i = tid; i < row_size && i < max_valid_k; i += blockDim.x) {
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
    
    // Step 2: Compute exp and sum (only valid positions)
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        if (i < max_valid_k) {
            float exp_val = expf(row_input[i] - max_val);
            row_output[i] = exp_val;
            local_sum += exp_val;
        } else {
            row_output[i] = 0.0f;  // Masked positions get zero probability
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
    for (int i = tid; i < max_valid_k && i < row_size; i += blockDim.x) {
        row_output[i] /= sum_val;
    }
}
"#;

/// Apply softmax along the last dimension
///
/// Input shape: (..., row_size)
/// Softmax is applied independently to each row.
///
/// # Errors
/// Returns an error if the operation fails
pub fn softmax(input: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    let shape = input.shape();
    let row_size = *shape
        .last()
        .expect("Input must have at least one dimension");
    let num_rows: usize = shape[..shape.len() - 1].iter().product();
    let num_rows = if num_rows == 0 { 1 } else { num_rows };

    let mut output = unsafe { CudaTensor::<f32>::uninit(input.context(), shape)? };

    let device = input.context().device();

    // Compile kernel
    let module_name = "softmax";
    if !device.has_func(module_name, "softmax_f32") {
        let ptx = cudarc::nvrtc::safe::compile_ptx(SOFTMAX_KERNEL)?;
        device.load_ptx(ptx, module_name, &["softmax_f32", "softmax_causal_f32"])?;
    }

    let func = device.get_func(module_name, "softmax_f32").unwrap();

    let block_size = 256.min(row_size.next_power_of_two());
    let shared_mem = block_size * std::mem::size_of::<f32>();

    let cfg = LaunchConfig {
        grid_dim: (num_rows as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    unsafe {
        func.launch(
            cfg,
            (output.cuda_slice_mut(), input.cuda_slice(), row_size as i32),
        )?;
    }

    Ok(output)
}

/// Apply softmax with causal masking for attention
///
/// For each query position q, only key positions k where k <= q are considered.
///
/// # Arguments
/// * `input` - Attention scores of shape (num_heads, seq_len) for a single query
/// * `query_idx` - The query position (0-indexed within the current sequence)
/// * `position_offset` - Offset for KV cache scenarios
///
/// # Errors
/// Returns an error if the operation fails
pub fn softmax_causal(
    input: &CudaTensor<f32>,
    query_idx: usize,
    position_offset: usize,
) -> Result<CudaTensor<f32>> {
    let shape = input.shape();
    assert_eq!(shape.len(), 2, "Expected (num_heads, seq_len)");

    let num_heads = shape[0];
    let row_size = shape[1];

    let mut output = unsafe { CudaTensor::<f32>::uninit(input.context(), shape)? };

    let device = input.context().device();

    // Compile kernel
    let module_name = "softmax";
    if !device.has_func(module_name, "softmax_causal_f32") {
        let ptx = cudarc::nvrtc::safe::compile_ptx(SOFTMAX_KERNEL)?;
        device.load_ptx(ptx, module_name, &["softmax_f32", "softmax_causal_f32"])?;
    }

    let func = device.get_func(module_name, "softmax_causal_f32").unwrap();

    let block_size = 256.min(row_size.next_power_of_two());
    let shared_mem = block_size * std::mem::size_of::<f32>();

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
                input.cuda_slice(),
                row_size as i32,
                query_idx as i32,
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
    fn test_softmax() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0];
        let input = CudaTensor::from_slice(&ctx, &[2, 4], &input_data).unwrap();

        let output = softmax(&input).unwrap();
        let result = output.to_vec().unwrap();

        // Row 0: softmax([1, 2, 3, 4])
        let max0 = 4.0_f32;
        let exp0: Vec<f32> = vec![
            (1.0 - max0).exp(),
            (2.0 - max0).exp(),
            (3.0 - max0).exp(),
            (4.0 - max0).exp(),
        ];
        let sum0: f32 = exp0.iter().sum();
        let expected0: Vec<f32> = exp0.iter().map(|x| x / sum0).collect();

        for i in 0..4 {
            assert!(
                (result[i] - expected0[i]).abs() < 1e-5,
                "Row 0 mismatch at {}: {} vs {}",
                i,
                result[i],
                expected0[i]
            );
        }

        // Row 1: softmax([1, 1, 1, 1]) = [0.25, 0.25, 0.25, 0.25]
        for i in 4..8 {
            assert!(
                (result[i] - 0.25).abs() < 1e-5,
                "Row 1 mismatch at {}: {} vs 0.25",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let input_data: Vec<f32> = (0..32).map(|x| x as f32 * 0.1).collect();
        let input = CudaTensor::from_slice(&ctx, &[4, 8], &input_data).unwrap();

        let output = softmax(&input).unwrap();
        let result = output.to_vec().unwrap();

        // Each row should sum to 1
        for row in 0..4 {
            let sum: f32 = result[row * 8..(row + 1) * 8].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Row {} sum is {} instead of 1.0",
                row,
                sum
            );
        }
    }
}
