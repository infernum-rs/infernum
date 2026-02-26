//! Dtype casting operations

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::missing_panics_doc,
    clippy::manual_div_ceil
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use crate::cuda::CudaTensor;
use infernum::dtype::DType;
use infernum::tensor::Tensor;
use infernum::Result;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/cast.ptx"));
const KERNEL_NAMES: &[&str] = &[
    "cast_f32_to_f16",
    "cast_f16_to_f32",
    "cast_f32_to_bf16",
    "cast_bf16_to_f32",
];

fn ensure_cast_kernels(device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<()> {
    let module_name = "cast";
    if !device.has_func(module_name, "cast_f32_to_bf16") {
        device.load_ptx(cudarc::nvrtc::Ptx::from_src(PTX), module_name, KERNEL_NAMES)?;
    }
    Ok(())
}

/// Cast a tensor of type `T` to `f32`.
///
/// For `f16` and `bf16`, a CUDA kernel converts each element on device.
/// For `f32`, this is a host-roundtrip copy (callers should avoid this
/// path when `T` is known to be `f32`).
///
/// # Errors
/// Returns an error if the kernel launch or copy fails.
///
/// # Panics
/// Panics if `T` is not `f32`, `f16`, or `bf16`.
pub fn cast_to_f32(input: &CudaTensor) -> Result<CudaTensor> {
    let dtype = input.dtype();
    let n = input.numel();
    let shape = input.shape();

    match dtype {
        DType::F32 => Ok(input.clone()),
        DType::F16 => launch_cast_kernel(input, "cast_f16_to_f32", n, shape),
        DType::BF16 => launch_cast_kernel(input, "cast_bf16_to_f32", n, shape),
        other => panic!("Unsupported dtype for cast_to_f32: {other}"),
    }
}

/// Cast an f32 tensor to the specified target dtype on the GPU.
///
/// If `target` is `F32`, returns a clone. For `F16` and `BF16`, launches
/// the appropriate CUDA cast kernel.
///
/// # Errors
/// Returns an error if the kernel launch or allocation fails.
///
/// # Panics
/// Panics if `target` is not `F32`, `F16`, or `BF16`.
pub fn cast_from_f32(input: &CudaTensor, target: DType) -> Result<CudaTensor> {
    assert_eq!(
        input.dtype(),
        DType::F32,
        "cast_from_f32: input must be f32"
    );
    match target {
        DType::F32 => Ok(input.clone()),
        DType::F16 => cast_f32_to_f16(input),
        DType::BF16 => cast_f32_to_bf16(input),
        other => panic!("Unsupported target dtype for cast_from_f32: {other}"),
    }
}

/// Cast an f32 tensor to f16 on the GPU
///
/// # Errors
/// Returns an error if the kernel launch or allocation fails
pub fn cast_f32_to_f16(input: &CudaTensor) -> Result<CudaTensor> {
    let n = input.numel();
    let mut output = unsafe { CudaTensor::uninit(input.context(), input.shape(), DType::F16)? };

    let device = input.context().device();
    ensure_cast_kernels(device)?;

    let func = device.get_func("cast", "cast_f32_to_f16").unwrap();

    let block_size = 256;
    let grid_size = n.div_ceil(block_size);

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

/// Cast an f32 tensor to bf16 on the GPU
///
/// # Errors
/// Returns an error if the kernel launch or allocation fails
pub fn cast_f32_to_bf16(input: &CudaTensor) -> Result<CudaTensor> {
    let n = input.numel();
    let mut output = unsafe { CudaTensor::uninit(input.context(), input.shape(), DType::BF16)? };

    let device = input.context().device();
    ensure_cast_kernels(device)?;

    let func = device.get_func("cast", "cast_f32_to_bf16").unwrap();

    let block_size = 256;
    let grid_size = n.div_ceil(block_size);

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
pub fn cast_bf16_to_f32(input: &CudaTensor) -> Result<CudaTensor> {
    let n = input.numel();
    let mut output = unsafe { CudaTensor::uninit(input.context(), input.shape(), DType::F32)? };

    let device = input.context().device();
    ensure_cast_kernels(device)?;

    let func = device.get_func("cast", "cast_bf16_to_f32").unwrap();

    let block_size = 256;
    let grid_size = n.div_ceil(block_size);

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

fn launch_cast_kernel(
    input: &CudaTensor,
    kernel_name: &str,
    n: usize,
    shape: &[usize],
) -> Result<CudaTensor> {
    let mut output = unsafe { CudaTensor::uninit(input.context(), shape, DType::F32)? };
    let device = input.context().device();

    ensure_cast_kernels(device)?;

    let func = device.get_func("cast", kernel_name).unwrap();

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
    fn test_cast_f32_roundtrip() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let data: Vec<f32> = vec![1.0, 2.5, -3.0, 4.75];
        let input = CudaTensor::from_slice(&ctx, &[4], &data).unwrap();

        let output = cast_to_f32(&input).unwrap();
        let result = output.to_vec::<f32>().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_cast_bf16_to_f32() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let f32_data: Vec<f32> = vec![1.0, 2.5, -3.0, 4.75];
        let bf16_data: Vec<half::bf16> =
            f32_data.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let input = CudaTensor::from_slice(&ctx, &[4], &bf16_data).unwrap();

        let output = cast_to_f32(&input).unwrap();
        let result = output.to_vec::<f32>().unwrap();

        for (i, (&got, &expected)) in result.iter().zip(f32_data.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-2,
                "Mismatch at {i}: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn test_cast_f16_to_f32() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let f32_data: Vec<f32> = vec![1.0, 2.5, -3.0, 4.75];
        let f16_data: Vec<half::f16> = f32_data.iter().map(|&x| half::f16::from_f32(x)).collect();
        let input = CudaTensor::from_slice(&ctx, &[4], &f16_data).unwrap();

        let output = cast_to_f32(&input).unwrap();
        let result = output.to_vec::<f32>().unwrap();

        for (i, (&got, &expected)) in result.iter().zip(f32_data.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-2,
                "Mismatch at {i}: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn test_f32_to_bf16_roundtrip() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let data: Vec<f32> = vec![1.0, -2.5, 3.14, 0.0, 100.0, -0.001];
        let input = CudaTensor::from_slice(&ctx, &[2, 3], &data).unwrap();

        let bf16 = cast_f32_to_bf16(&input).unwrap();
        assert_eq!(bf16.shape(), &[2, 3]);

        let back = cast_bf16_to_f32(&bf16).unwrap();
        assert_eq!(back.shape(), &[2, 3]);

        let result = back.to_vec::<f32>().unwrap();
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
