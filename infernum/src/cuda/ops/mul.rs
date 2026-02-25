//! Elementwise multiplication

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

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/mul.ptx"));
const KERNEL_NAMES: &[&str] = &["mul_f32", "mul_f16", "mul_bf16"];

/// Kernel name suffix for dtype
fn kernel_suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => panic!("Unsupported dtype: {dtype:?}"),
    }
}

/// Multiply two tensors element-wise on GPU: output = a * b
///
/// Both tensors must have the same shape.
///
/// # Errors
/// Returns an error if the operation fails
pub fn mul(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
    let dtype = a.dtype();
    assert_eq!(a.shape(), b.shape(), "Shapes must match for multiplication");

    let shape = a.shape();
    let n = a.numel();

    let mut output = unsafe { CudaTensor::uninit(a.context(), shape, dtype)? };

    let device = a.context().device();
    let kernel_name = format!("mul_{}", kernel_suffix(dtype));

    let module_name = "mul";
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

const SCALE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/scale.ptx"));

/// Scale all elements of an f32 tensor in-place: `data[i] *= scale`.
///
/// # Errors
/// Returns an error if the kernel launch fails.
pub fn scale_f32_inplace(tensor: &mut CudaTensor, scale: f32) -> Result<()> {
    let n = tensor.numel();
    let device = tensor.context().device();

    let module_name = "scale";
    if !device.has_func(module_name, "scale_f32") {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(SCALE_PTX),
            module_name,
            &["scale_f32"],
        )?;
    }

    let func = device.get_func(module_name, "scale_f32").unwrap();

    let block_size = 256;
    let grid_size = (n + block_size - 1) / block_size;

    let cfg = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(cfg, (tensor.cuda_slice_mut(), scale, n as i32))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;

    #[test]
    fn test_mul() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

        let a = CudaTensor::from_slice(&ctx, &[2, 3], &a_data).unwrap();
        let b = CudaTensor::from_slice(&ctx, &[2, 3], &b_data).unwrap();

        let c = mul(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 3]);

        let result = c.to_vec::<f32>().unwrap();
        assert_eq!(result, vec![10.0, 40.0, 90.0, 160.0, 250.0, 360.0]);
    }

    #[test]
    fn test_mul_large() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let n = 10_000;
        let a_data: Vec<f32> = (0..n).map(|x| x as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|x| x as f32 * 2.0).collect();

        let a = CudaTensor::from_slice(&ctx, &[n], &a_data).unwrap();
        let b = CudaTensor::from_slice(&ctx, &[n], &b_data).unwrap();

        let c = mul(&a, &b).unwrap();
        let result = c.to_vec::<f32>().unwrap();

        for i in 0..n {
            let expected = (i as f32) * (i as f32 * 2.0);
            assert!(
                (result[i] - expected).abs() < 1e-2,
                "Mismatch at {i}: {} vs {expected}",
                result[i]
            );
        }
    }
}
