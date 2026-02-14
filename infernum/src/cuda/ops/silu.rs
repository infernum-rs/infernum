//! SiLU (Swish) activation function

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::manual_div_ceil
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use crate::cuda::CudaTensor;
use crate::tensor::Tensor;
use crate::Result;

const SILU_KERNEL: &str = r#"
extern "C" __global__ void silu_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

extern "C" __global__ void silu_inplace_f32(
    float* __restrict__ data,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        data[idx] = x / (1.0f + expf(-x));
    }
}

// SiLU with elementwise multiplication: output = silu(a) * b
// Used in SwiGLU: silu(gate) * up
extern "C" __global__ void silu_mul_f32(
    float* __restrict__ output,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = gate[idx];
        float silu_x = x / (1.0f + expf(-x));
        output[idx] = silu_x * up[idx];
    }
}
"#;

/// Apply SiLU (Swish) activation: output = x * sigmoid(x)
///
/// # Errors
/// Returns an error if the operation fails
pub fn silu(input: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    let shape = input.shape();
    let n = input.numel();

    let mut output = unsafe { CudaTensor::<f32>::uninit(input.context(), shape)? };

    let device = input.context().device();

    // Compile kernel
    let module_name = "silu";
    if !device.has_func(module_name, "silu_f32") {
        let ptx = cudarc::nvrtc::safe::compile_ptx(SILU_KERNEL)?;
        device.load_ptx(
            ptx,
            module_name,
            &["silu_f32", "silu_inplace_f32", "silu_mul_f32"],
        )?;
    }

    let func = device.get_func(module_name, "silu_f32").unwrap();

    let block_size = 256;
    let grid_size = (n + block_size - 1) / block_size;

    let cfg = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(cfg, (output.cuda_slice_mut(), input.cuda_slice(), n as i32))?;
    }

    Ok(output)
}

/// Fused SiLU + multiply for SwiGLU: output = silu(gate) * up
///
/// # Errors
/// Returns an error if the operation fails
pub fn silu_mul(gate: &CudaTensor<f32>, up: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    assert_eq!(gate.shape(), up.shape(), "gate and up must have same shape");

    let shape = gate.shape();
    let n = gate.numel();

    let mut output = unsafe { CudaTensor::<f32>::uninit(gate.context(), shape)? };

    let device = gate.context().device();

    // Compile kernel
    let module_name = "silu";
    if !device.has_func(module_name, "silu_mul_f32") {
        let ptx = cudarc::nvrtc::safe::compile_ptx(SILU_KERNEL)?;
        device.load_ptx(
            ptx,
            module_name,
            &["silu_f32", "silu_inplace_f32", "silu_mul_f32"],
        )?;
    }

    let func = device.get_func(module_name, "silu_mul_f32").unwrap();

    let block_size = 256;
    let grid_size = (n + block_size - 1) / block_size;

    let cfg = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(
            cfg,
            (
                output.cuda_slice_mut(),
                gate.cuda_slice(),
                up.cuda_slice(),
                n as i32,
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
    fn test_silu() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let input_data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let input = CudaTensor::from_slice(&ctx, &[5], &input_data).unwrap();

        let output = silu(&input).unwrap();
        let result = output.to_vec().unwrap();

        // SiLU(x) = x * sigmoid(x)
        for (i, &x) in input_data.iter().enumerate() {
            let expected = x / (1.0 + (-x).exp());
            assert!(
                (result[i] - expected).abs() < 1e-5,
                "Mismatch at {}: {} vs {}",
                i,
                result[i],
                expected
            );
        }
    }

    #[test]
    fn test_silu_mul() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let gate_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let up_data: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0];

        let gate = CudaTensor::from_slice(&ctx, &[4], &gate_data).unwrap();
        let up = CudaTensor::from_slice(&ctx, &[4], &up_data).unwrap();

        let output = silu_mul(&gate, &up).unwrap();
        let result = output.to_vec().unwrap();

        for i in 0..4 {
            let silu_gate = gate_data[i] / (1.0 + (-gate_data[i]).exp());
            let expected = silu_gate * up_data[i];
            assert!(
                (result[i] - expected).abs() < 1e-5,
                "Mismatch at {}: {} vs {}",
                i,
                result[i],
                expected
            );
        }
    }
}
