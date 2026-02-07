//! RMS Normalization

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

const RMSNORM_KERNEL: &str = r#"
extern "C" __global__ void rmsnorm_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;
    
    // Compute sum of squares using block reduction
    extern __shared__ float shared[];
    
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = row_input[i];
        local_sum += val * val;
    }
    
    shared[tid] = local_sum;
    __syncthreads();
    
    // Block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    float rms = rsqrtf(shared[0] / (float)hidden_size + eps);
    
    // Apply normalization and weight
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        row_output[i] = row_input[i] * rms * weight[i];
    }
}
"#;

/// Apply RMS normalization: output = (x / rms(x)) * weight
///
/// where rms(x) = sqrt(mean(x^2) + eps)
///
/// # Arguments
/// * `input` - Input tensor of shape (batch, seq_len, hidden) or (seq_len, hidden)
/// * `weight` - Weight tensor of shape (hidden,)
/// * `eps` - Small epsilon for numerical stability
///
/// # Errors
/// Returns an error if the operation fails
pub fn rms_norm(
    input: &CudaTensor<f32>,
    weight: &CudaTensor<f32>,
    eps: f32,
) -> Result<CudaTensor<f32>> {
    let shape = input.shape();
    let hidden_size = *shape
        .last()
        .expect("Input must have at least one dimension");
    let num_rows: usize = shape[..shape.len() - 1].iter().product();

    assert_eq!(
        weight.shape(),
        &[hidden_size],
        "Weight shape must match hidden dimension"
    );

    let mut output = unsafe { CudaTensor::<f32>::uninit(input.context(), shape)? };

    let device = input.context().device();

    // Compile kernel
    let module_name = "rmsnorm";
    if !device.has_func(module_name, "rmsnorm_f32") {
        let ptx = cudarc::nvrtc::safe::compile_ptx(RMSNORM_KERNEL)?;
        device.load_ptx(ptx, module_name, &["rmsnorm_f32"])?;
    }

    let func = device.get_func(module_name, "rmsnorm_f32").unwrap();

    let block_size = 256.min(hidden_size);
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
                input.cuda_slice(),
                weight.cuda_slice(),
                hidden_size as i32,
                eps,
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
    fn test_rmsnorm() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // Simple test: 2 rows, hidden_size=4
        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.0, 1.5, 2.0];
        let weight_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

        let input = CudaTensor::from_slice(&ctx, &[2, 4], &input_data).unwrap();
        let weight = CudaTensor::from_slice(&ctx, &[4], &weight_data).unwrap();

        let output = rms_norm(&input, &weight, 1e-6).unwrap();

        assert_eq!(output.shape(), &[2, 4]);

        let result = output.to_vec().unwrap();

        // Row 0: rms = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
        // Row 0 normalized: [0.365, 0.730, 1.095, 1.461]
        let rms0 = (1.0_f32 + 4.0 + 9.0 + 16.0) / 4.0;
        let rms0 = rms0.sqrt();
        let expected0: Vec<f32> = vec![1.0 / rms0, 2.0 / rms0, 3.0 / rms0, 4.0 / rms0];

        for (i, &val) in result[0..4].iter().enumerate() {
            assert!(
                (val - expected0[i]).abs() < 1e-4,
                "Mismatch at index {}: {} vs {}",
                i,
                val,
                expected0[i]
            );
        }
    }
}
