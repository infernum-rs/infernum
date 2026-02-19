//! Cast operations between f32 and bf16

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::missing_panics_doc
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use crate::cuda::CudaTensor;
use crate::tensor::Tensor;
use crate::Result;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/cast.ptx"));

fn ensure_cast_kernels(device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<()> {
    let module_name = "cast";
    if !device.has_func(module_name, "cast_f32_to_bf16") {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(PTX),
            module_name,
            &["cast_f32_to_bf16", "cast_bf16_to_f32"],
        )?;
    }
    Ok(())
}

/// Cast an f32 tensor to bf16 on the GPU
///
/// # Errors
/// Returns an error if the kernel launch or allocation fails
pub fn cast_f32_to_bf16(input: &CudaTensor<f32>) -> Result<CudaTensor<half::bf16>> {
    let n = input.numel();
    let mut output = unsafe { CudaTensor::<half::bf16>::uninit(input.context(), input.shape())? };

    let device = input.context().device();
    ensure_cast_kernels(device)?;

    let func = device.get_func("cast", "cast_f32_to_bf16").unwrap();

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
            (output.cuda_slice_mut(), &input.cuda_slice(), n as i32),
        )?;
    }

    Ok(output)
}

/// Cast a bf16 tensor to f32 on the GPU
///
/// # Errors
/// Returns an error if the kernel launch or allocation fails
pub fn cast_bf16_to_f32(input: &CudaTensor<half::bf16>) -> Result<CudaTensor<f32>> {
    let n = input.numel();
    let mut output = unsafe { CudaTensor::<f32>::uninit(input.context(), input.shape())? };

    let device = input.context().device();
    ensure_cast_kernels(device)?;

    let func = device.get_func("cast", "cast_bf16_to_f32").unwrap();

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
            (output.cuda_slice_mut(), &input.cuda_slice(), n as i32),
        )?;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;

    #[test]
    fn test_f32_to_bf16_roundtrip() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let data: Vec<f32> = vec![1.0, -2.5, 3.14, 0.0, 100.0, -0.001];
        let input = CudaTensor::from_slice(&ctx, &[2, 3], &data).unwrap();

        let bf16 = cast_f32_to_bf16(&input).unwrap();
        assert_eq!(bf16.shape(), &[2, 3]);

        let back = cast_bf16_to_f32(&bf16).unwrap();
        assert_eq!(back.shape(), &[2, 3]);

        let result = back.to_vec().unwrap();
        for (i, (&orig, &got)) in data.iter().zip(result.iter()).enumerate() {
            let tol = orig.abs() * 0.01 + 1e-3;
            assert!((orig - got).abs() < tol, "Mismatch at {i}: {orig} vs {got}");
        }
    }

    #[test]
    fn test_cast_preserves_shape() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let data: Vec<f32> = vec![1.0; 24];
        let input = CudaTensor::from_slice(&ctx, &[2, 3, 4], &data).unwrap();

        let bf16 = cast_f32_to_bf16(&input).unwrap();
        assert_eq!(bf16.shape(), &[2, 3, 4]);

        let f32_back = cast_bf16_to_f32(&bf16).unwrap();
        assert_eq!(f32_back.shape(), &[2, 3, 4]);
    }
}
