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
use crate::tensor::Tensor;
use crate::Result;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/add_rmsnorm.ptx"));

infernum_macros::define_block! {
    /// Residual add followed by RMS normalization.
    ///
    /// Computes `sum = residual + x` and `normed = rmsnorm(sum, weight, eps)`.
    /// Returns `(sum, normed)`.
    ///
    /// The decomposed version calls `add` then `rms_norm`. When fusion is
    /// active (release builds by default), a single CUDA kernel handles both.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn add_rmsnorm(
        residual: &CudaTensor<f32>,
        x: &CudaTensor<f32>,
        weight: &CudaTensor<f32>,
        eps: f32,
    ) -> Result<(CudaTensor<f32>, CudaTensor<f32>)> {
        let sum = super::add(residual, x)?;
        let normed = super::rms_norm(&sum, weight, eps)?;
        Ok((sum, normed))
    }
}

infernum_macros::define_fusion! {
    block: ADD_RMSNORM_FUSED,
    fn add_rmsnorm_fused(
        residual: &CudaTensor<f32>,
        x: &CudaTensor<f32>,
        weight: &CudaTensor<f32>,
        eps: f32,
    ) -> Result<(CudaTensor<f32>, CudaTensor<f32>)> {
        let shape = residual.shape();
        assert_eq!(shape, x.shape(), "residual and x must have the same shape");

        let hidden_size = *shape.last().expect("Input must have at least one dimension");
        let num_rows: usize = shape[..shape.len() - 1].iter().product();

        assert_eq!(
            weight.shape(),
            &[hidden_size],
            "Weight shape must match hidden dimension"
        );

        let mut sum_out = unsafe { CudaTensor::<f32>::uninit(residual.context(), shape)? };
        let mut norm_out = unsafe { CudaTensor::<f32>::uninit(residual.context(), shape)? };

        let device = residual.context().device();

        let module_name = "add_rmsnorm";
        if !device.has_func(module_name, "add_rmsnorm_f32") {
            device.load_ptx(
                cudarc::nvrtc::Ptx::from_src(PTX),
                module_name,
                &["add_rmsnorm_f32"],
            )?;
        }

        let func = device.get_func(module_name, "add_rmsnorm_f32").unwrap();

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;

    #[test]
    fn test_add_rmsnorm_decomposed() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let residual_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.0, 1.5, 2.0];
        let x_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.05, 0.1, 0.15, 0.2];
        let weight_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

        let residual = CudaTensor::from_slice(&ctx, &[2, 4], &residual_data).unwrap();
        let x = CudaTensor::from_slice(&ctx, &[2, 4], &x_data).unwrap();
        let weight = CudaTensor::from_slice(&ctx, &[4], &weight_data).unwrap();

        let (sum, normed) = add_rmsnorm_decomposed(&residual, &x, &weight, 1e-6).unwrap();

        let sum_result = sum.to_vec().unwrap();
        let normed_result = normed.to_vec().unwrap();

        // Verify the sum is correct
        for i in 0..8 {
            let expected = residual_data[i] + x_data[i];
            assert!(
                (sum_result[i] - expected).abs() < 1e-5,
                "Sum mismatch at {i}: {} vs {expected}",
                sum_result[i]
            );
        }

        // Verify normed is rmsnorm of the sum
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
                normed_result[i]
            );
        }
    }

    #[test]
    fn test_add_rmsnorm_fused_matches_decomposed() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let residual_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.0, 1.5, 2.0];
        let x_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.05, 0.1, 0.15, 0.2];
        let weight_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

        let residual = CudaTensor::from_slice(&ctx, &[2, 4], &residual_data).unwrap();
        let x = CudaTensor::from_slice(&ctx, &[2, 4], &x_data).unwrap();
        let weight = CudaTensor::from_slice(&ctx, &[4], &weight_data).unwrap();

        let (sum_d, norm_d) = add_rmsnorm_decomposed(&residual, &x, &weight, 1e-6).unwrap();
        let (sum_f, norm_f) = add_rmsnorm_fused(&residual, &x, &weight, 1e-6).unwrap();

        let sum_d = sum_d.to_vec().unwrap();
        let sum_f = sum_f.to_vec().unwrap();
        let norm_d = norm_d.to_vec().unwrap();
        let norm_f = norm_f.to_vec().unwrap();

        for (i, (&d, &f)) in sum_d.iter().zip(sum_f.iter()).enumerate() {
            assert!(
                (d - f).abs() < 1e-5,
                "Sum mismatch at {i}: decomposed={d}, fused={f}"
            );
        }
        for (i, (&d, &f)) in norm_d.iter().zip(norm_f.iter()).enumerate() {
            assert!(
                (d - f).abs() < 1e-4,
                "Normed mismatch at {i}: decomposed={d}, fused={f}"
            );
        }
    }
}
