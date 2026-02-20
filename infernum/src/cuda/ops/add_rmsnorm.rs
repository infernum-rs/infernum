//! Fused residual add + RMS normalization
//!
//! At transformer layer boundaries the pattern is:
//!   hidden = residual + x;  normed = rmsnorm(hidden)
//! The fused kernel does both in a single pass, saving one global memory
//! round-trip.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::missing_panics_doc
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use crate::cuda::CudaTensor;
use crate::dtype::TensorDType;
use crate::tensor::Tensor;
use crate::Result;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/add_rmsnorm.ptx"));

const KERNEL_NAMES: &[&str] = &["add_rmsnorm_f32", "add_rmsnorm_f16", "add_rmsnorm_bf16"];

fn kernel_suffix<T: TensorDType>() -> &'static str {
    match T::DTYPE {
        crate::dtype::DType::F32 => "f32",
        crate::dtype::DType::F16 => "f16",
        crate::dtype::DType::BF16 => "bf16",
        other => panic!("add_rmsnorm not supported for dtype: {other}"),
    }
}

/// Residual add followed by RMS normalization.
///
/// Computes `sum = residual + x` and `normed = rmsnorm(sum, weight, eps)`.
/// Returns `(sum, normed)`.
///
/// Uses a single fused CUDA kernel for all dtypes, saving one global memory
/// round-trip compared to separate add + rmsnorm.
///
/// # Errors
/// Returns an error if the operation fails.
pub fn add_rmsnorm<T: TensorDType + cudarc::driver::DeviceRepr>(
    residual: &CudaTensor<T>,
    x: &CudaTensor<T>,
    weight: &CudaTensor<T>,
    eps: f32,
) -> Result<(CudaTensor<T>, CudaTensor<T>)> {
    let shape = residual.shape();
    assert_eq!(shape, x.shape(), "residual and x must have the same shape");

    let hidden_size = *shape
        .last()
        .expect("Input must have at least one dimension");
    let num_rows: usize = shape[..shape.len() - 1].iter().product();

    assert_eq!(
        weight.shape(),
        &[hidden_size],
        "Weight shape must match hidden dimension"
    );

    let mut sum_out = unsafe { CudaTensor::<T>::uninit(residual.context(), shape)? };
    let mut norm_out = unsafe { CudaTensor::<T>::uninit(residual.context(), shape)? };

    let device = residual.context().device();
    let kernel_name = format!("add_rmsnorm_{}", kernel_suffix::<T>());

    let module_name = "add_rmsnorm";
    if !device.has_func(module_name, &kernel_name) {
        device.load_ptx(cudarc::nvrtc::Ptx::from_src(PTX), module_name, KERNEL_NAMES)?;
    }

    let func = device.get_func(module_name, &kernel_name).unwrap();

    let block_size = 256.min(hidden_size);
    // Shared memory for warp-level reduction: one float per warp
    let num_warps = block_size.div_ceil(32);
    let shared_mem = num_warps * std::mem::size_of::<f32>();

    let cfg = LaunchConfig {
        grid_dim: (num_rows as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    unsafe {
        func.launch(
            cfg,
            (
                sum_out.cuda_slice_mut(),
                norm_out.cuda_slice_mut(),
                &residual.cuda_slice(),
                &x.cuda_slice(),
                &weight.cuda_slice(),
                hidden_size as i32,
                eps,
            ),
        )?;
    }

    Ok((sum_out, norm_out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;

    #[test]
    fn test_add_rmsnorm_f32() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let residual_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.0, 1.5, 2.0];
        let x_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.05, 0.1, 0.15, 0.2];
        let weight_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

        let residual = CudaTensor::from_slice(&ctx, &[2, 4], &residual_data).unwrap();
        let x = CudaTensor::from_slice(&ctx, &[2, 4], &x_data).unwrap();
        let weight = CudaTensor::from_slice(&ctx, &[4], &weight_data).unwrap();

        let (sum, normed) = add_rmsnorm(&residual, &x, &weight, 1e-6).unwrap();

        let sum_result = sum.to_vec().unwrap();
        let normed_result = normed.to_vec().unwrap();

        for i in 0..8 {
            let expected = residual_data[i] + x_data[i];
            assert!(
                (sum_result[i] - expected).abs() < 1e-5,
                "Sum mismatch at {i}: {} vs {expected}",
                sum_result[i],
            );
        }

        let expected_sum: Vec<f32> = residual_data
            .iter()
            .zip(x_data.iter())
            .map(|(a, b)| a + b)
            .collect();
        let rms0: f32 = (expected_sum[..4].iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        for i in 0..4 {
            let expected = expected_sum[i] / rms0;
            assert!(
                (normed_result[i] - expected).abs() < 1e-4,
                "Normed mismatch at {i}: {} vs {expected}",
                normed_result[i],
            );
        }
    }

    #[test]
    fn test_add_rmsnorm_bf16() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let residual_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.0, 1.5, 2.0];
        let x_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.05, 0.1, 0.15, 0.2];
        let weight_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

        let residual_bf16: Vec<half::bf16> = residual_data
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x_bf16: Vec<half::bf16> = x_data.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let weight_bf16: Vec<half::bf16> = weight_data
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();

        let residual = CudaTensor::from_slice(&ctx, &[2, 4], &residual_bf16).unwrap();
        let x = CudaTensor::from_slice(&ctx, &[2, 4], &x_bf16).unwrap();
        let weight = CudaTensor::from_slice(&ctx, &[4], &weight_bf16).unwrap();

        let (sum, normed) = add_rmsnorm(&residual, &x, &weight, 1e-6).unwrap();

        let sum_result: Vec<f32> = sum.to_vec().unwrap().iter().map(|v| v.to_f32()).collect();
        let normed_result: Vec<f32> = normed
            .to_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect();

        for i in 0..8 {
            let expected = residual_data[i] + x_data[i];
            assert!(
                (sum_result[i] - expected).abs() < 0.02,
                "Sum mismatch at {i}: {} vs {expected}",
                sum_result[i],
            );
        }

        let expected_sum: Vec<f32> = residual_data
            .iter()
            .zip(x_data.iter())
            .map(|(a, b)| a + b)
            .collect();
        let rms0: f32 = (expected_sum[..4].iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        for i in 0..4 {
            let expected = expected_sum[i] / rms0;
            assert!(
                (normed_result[i] - expected).abs() < 0.02,
                "Normed mismatch at {i}: {} vs {expected}",
                normed_result[i],
            );
        }
    }
}
