//! Dtype casting operations

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::missing_panics_doc,
    clippy::manual_div_ceil
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use crate::cuda::CudaTensor;
use crate::dtype::{DType, TensorDType};
use crate::tensor::Tensor;
use crate::Result;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/cast.ptx"));
const KERNEL_NAMES: &[&str] = &[
    "cast_f32_to_f16",
    "cast_f16_to_f32",
    "cast_f32_to_bf16",
    "cast_bf16_to_f32",
];

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
pub fn cast_to_f32<T: TensorDType + cudarc::driver::DeviceRepr>(
    input: &CudaTensor<T>,
) -> Result<CudaTensor<f32>> {
    let n = input.numel();
    let shape = input.shape();

    match T::DTYPE {
        DType::F32 => {
            // T is f32 — copy via host roundtrip. This path is rarely taken;
            // callers should use the tensor directly when T = f32.
            let host = input.to_vec()?;
            assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<f32>());
            let mut f32_data = Vec::<f32>::with_capacity(host.len());
            // SAFETY: T::DTYPE is F32, so T and f32 have identical layout
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host.as_ptr().cast::<f32>(),
                    f32_data.as_mut_ptr(),
                    host.len(),
                );
                f32_data.set_len(host.len());
            }
            CudaTensor::from_slice(input.context(), shape, &f32_data)
        }
        DType::F16 => launch_cast_kernel(input, "cast_f16_to_f32", n, shape),
        DType::BF16 => launch_cast_kernel(input, "cast_bf16_to_f32", n, shape),
        other => panic!("Unsupported dtype for cast_to_f32: {other}"),
    }
}

fn launch_cast_kernel<T: TensorDType + cudarc::driver::DeviceRepr>(
    input: &CudaTensor<T>,
    kernel_name: &str,
    n: usize,
    shape: &[usize],
) -> Result<CudaTensor<f32>> {
    let mut output = unsafe { CudaTensor::<f32>::uninit(input.context(), shape)? };
    let device = input.context().device();

    let module_name = "cast";
    if !device.has_func(module_name, kernel_name) {
        device.load_ptx(cudarc::nvrtc::Ptx::from_src(PTX), module_name, KERNEL_NAMES)?;
    }

    let func = device.get_func(module_name, kernel_name).unwrap();

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
        let result = output.to_vec().unwrap();
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
        let result = output.to_vec().unwrap();

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
        let result = output.to_vec().unwrap();

        for (i, (&got, &expected)) in result.iter().zip(f32_data.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-2,
                "Mismatch at {i}: {got} vs {expected}"
            );
        }
    }
}
