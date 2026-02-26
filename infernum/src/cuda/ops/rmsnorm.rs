//! RMS Normalization

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::missing_panics_doc
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use crate::cuda::CudaTensor;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::Result;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/rmsnorm.ptx"));

const KERNEL_NAMES: &[&str] = &[
    "rmsnorm_f32",
    "rmsnorm_inplace_f32",
    "rmsnorm_f16",
    "rmsnorm_inplace_f16",
    "rmsnorm_bf16",
    "rmsnorm_inplace_bf16",
];

fn kernel_suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        other => panic!("RMS norm not supported for dtype: {other}"),
    }
}

fn load_rmsnorm_kernels(device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<()> {
    let module_name = "rmsnorm";
    if !device.has_func(module_name, "rmsnorm_f32") {
        device.load_ptx(cudarc::nvrtc::Ptx::from_src(PTX), module_name, KERNEL_NAMES)?;
    }
    Ok(())
}

/// Apply RMS normalization: output = (x / rms(x)) * weight
///
/// where rms(x) = sqrt(mean(x^2) + eps)
///
/// Supports F32, F16, and BF16 tensor types.
///
/// # Arguments
/// * `input` - Input tensor of shape (batch, seq_len, hidden) or (seq_len, hidden)
/// * `weight` - Weight tensor of shape (hidden,)
/// * `eps` - Small epsilon for numerical stability
///
/// # Errors
/// Returns an error if the operation fails
pub fn rms_norm(input: &CudaTensor, weight: &CudaTensor, eps: f32) -> Result<CudaTensor> {
    let dtype = input.dtype();
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

    let mut output = unsafe { CudaTensor::uninit(input.context(), shape, dtype)? };

    let device = input.context().device();
    load_rmsnorm_kernels(device)?;

    let kernel_name = format!("rmsnorm_{}", kernel_suffix(dtype));
    let func = device.get_func("rmsnorm", &kernel_name).unwrap();

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
                output.cuda_slice_mut(),
                &input.cuda_slice(),
                &weight.cuda_slice(),
                hidden_size as i32,
                eps,
            ),
        )?;
    }

    Ok(output)
}

/// Apply RMS normalization in place: data = (data / rms(data)) * weight
///
/// where rms(x) = sqrt(mean(x^2) + eps)
///
/// Supports F32, F16, and BF16 tensor types.
///
/// # Arguments
/// * `input` - Input tensor of shape (batch, seq_len, hidden) or (seq_len, hidden), modified in place
/// * `weight` - Weight tensor of shape (hidden,)
/// * `eps` - Small epsilon for numerical stability
///
/// # Errors
/// Returns an error if the operation fails
pub fn rms_norm_inplace(input: &mut CudaTensor, weight: &CudaTensor, eps: f32) -> Result<()> {
    let dtype = input.dtype();
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

    let device = input.context().device();
    load_rmsnorm_kernels(device)?;

    let kernel_name = format!("rmsnorm_inplace_{}", kernel_suffix(dtype));
    let func = device.get_func("rmsnorm", &kernel_name).unwrap();

    let block_size = 256.min(hidden_size);
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
                input.cuda_slice_mut(),
                &weight.cuda_slice(),
                hidden_size as i32,
                eps,
            ),
        )?;
    }

    Ok(())
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

        let result = output.to_vec::<f32>().unwrap();

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

    #[test]
    fn test_rmsnorm_multi_warp() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let hidden_size = 2048;
        let input_data: Vec<f32> = (0..hidden_size)
            .map(|i| ((i as f32) * 0.001 - 0.5).sin())
            .collect();
        let weight_data: Vec<f32> = vec![1.0; hidden_size];

        let input = CudaTensor::from_slice(&ctx, &[1, hidden_size], &input_data).unwrap();
        let weight = CudaTensor::from_slice(&ctx, &[hidden_size], &weight_data).unwrap();

        let output = rms_norm(&input, &weight, 1e-6).unwrap();
        let result = output.to_vec::<f32>().unwrap();

        let sum_sq: f32 = input_data.iter().map(|x| x * x).sum();
        let rms = (sum_sq / hidden_size as f32).sqrt();
        let expected: Vec<f32> = input_data.iter().map(|x| x / rms).collect();

        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "Mismatch at index {i}: {got} vs {exp} (rms={rms})"
            );
        }

        assert!(
            result.iter().all(|x| x.is_finite()),
            "Output contains NaN or Inf"
        );
    }

    #[test]
    fn test_rmsnorm_inplace() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.0, 1.5, 2.0];
        let weight_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

        // Compute expected result with out-of-place version
        let input_ref = CudaTensor::from_slice(&ctx, &[2, 4], &input_data).unwrap();
        let weight = CudaTensor::from_slice(&ctx, &[4], &weight_data).unwrap();
        let expected = rms_norm(&input_ref, &weight, 1e-6)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();

        // Compute with in-place version
        let mut input = CudaTensor::from_slice(&ctx, &[2, 4], &input_data).unwrap();
        rms_norm_inplace(&mut input, &weight, 1e-6).unwrap();

        let result = input.to_vec::<f32>().unwrap();

        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "Mismatch at index {i}: {got} vs {exp}"
            );
        }
    }
}
