//! Elementwise addition

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::manual_div_ceil
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use crate::cuda::CudaTensor;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::Result;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/add.ptx"));
const KERNEL_NAMES: &[&str] = &[
    "add_f32",
    "add_inplace_f32",
    "add_f16",
    "add_inplace_f16",
    "add_bf16",
    "add_inplace_bf16",
];

/// Kernel name suffix for dtype
fn kernel_suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => panic!("Unsupported dtype: {dtype:?}"),
    }
}

/// Add two tensors element-wise (generic version)
fn add_generic(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
    let dtype = a.dtype();
    assert_eq!(a.shape(), b.shape(), "Shapes must match for addition");

    let shape = a.shape();
    let n = a.numel();

    let mut output = unsafe { CudaTensor::uninit(a.context(), shape, dtype)? };

    let device = a.context().device();
    let kernel_name = format!("add_{}", kernel_suffix(dtype));

    let module_name = "add";
    if !device.has_func(module_name, &kernel_name) {
        device.load_ptx(cudarc::nvrtc::Ptx::from_src(PTX), module_name, KERNEL_NAMES)?;
    }

    let func = device.get_func(module_name, &kernel_name).unwrap();

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
                &a.cuda_slice(),
                &b.cuda_slice(),
                n as i32,
            ),
        )?;
    }

    Ok(output)
}

/// Add tensor `b` into `a` in place (generic version)
fn add_inplace_generic(a: &mut CudaTensor, b: &CudaTensor) -> Result<()> {
    let dtype = a.dtype();
    assert_eq!(a.shape(), b.shape(), "Shapes must match for addition");

    let n = a.numel();
    let device = a.context().device();
    let kernel_name = format!("add_inplace_{}", kernel_suffix(dtype));

    let module_name = "add";
    if !device.has_func(module_name, &kernel_name) {
        device.load_ptx(cudarc::nvrtc::Ptx::from_src(PTX), module_name, KERNEL_NAMES)?;
    }

    let func = device.get_func(module_name, &kernel_name).unwrap();

    let block_size = 256;
    let grid_size = (n + block_size - 1) / block_size;

    let cfg = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(cfg, (a.cuda_slice_mut(), &b.cuda_slice(), n as i32))?;
    }

    Ok(())
}

/// Add two tensors element-wise on GPU: output = a + b
///
/// Supports F32, F16, and BF16 tensor types.
/// Both tensors must have the same shape.
///
/// # Errors
/// Returns an error if the operation fails
pub fn add(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
    add_generic(a, b)
}

/// Add tensor `b` into `a` in place on GPU: a += b
///
/// Supports F32, F16, and BF16 tensor types.
/// Both tensors must have the same shape.
///
/// # Errors
/// Returns an error if the operation fails
pub fn add_inplace(a: &mut CudaTensor, b: &CudaTensor) -> Result<()> {
    add_inplace_generic(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;

    #[test]
    fn test_add() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

        let a = CudaTensor::from_slice(&ctx, &[2, 3], &a_data).unwrap();
        let b = CudaTensor::from_slice(&ctx, &[2, 3], &b_data).unwrap();

        let c = add(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 3]);

        let result = c.to_vec().unwrap();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
    }

    #[test]
    fn test_add_large() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let n = 10_000;
        let a_data: Vec<f32> = (0..n).map(|x| x as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|x| x as f32 * 2.0).collect();

        let a = CudaTensor::from_slice(&ctx, &[n], &a_data).unwrap();
        let b = CudaTensor::from_slice(&ctx, &[n], &b_data).unwrap();

        let c = add(&a, &b).unwrap();
        let result = c.to_vec().unwrap();

        for i in 0..n {
            assert!(
                (result[i] - (i as f32 * 3.0)).abs() < 1e-5,
                "Mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_add_inplace() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

        let mut a = CudaTensor::from_slice(&ctx, &[2, 3], &a_data).unwrap();
        let b = CudaTensor::from_slice(&ctx, &[2, 3], &b_data).unwrap();

        add_inplace(&mut a, &b).unwrap();

        assert_eq!(a.shape(), &[2, 3]);

        let result = a.to_vec().unwrap();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
    }
}
