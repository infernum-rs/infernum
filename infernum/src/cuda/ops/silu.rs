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
use crate::dtype::TensorDType;
use crate::tensor::Tensor;
use crate::Result;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/silu.ptx"));
const KERNEL_NAMES: &[&str] = &[
    "silu_f32",
    "silu_inplace_f32",
    "silu_mul_f32",
    "silu_f16",
    "silu_inplace_f16",
    "silu_mul_f16",
    "silu_bf16",
    "silu_inplace_bf16",
    "silu_mul_bf16",
];

/// Kernel name suffix for dtype
fn kernel_suffix<T: cudarc::driver::DeviceRepr>() -> &'static str {
    let type_name = std::any::type_name::<T>();
    if type_name.contains("f32") {
        "f32"
    } else if type_name.contains("f16") && !type_name.contains("bf16") {
        "f16"
    } else if type_name.contains("bf16") {
        "bf16"
    } else {
        panic!("Unsupported dtype for silu: {type_name}")
    }
}

/// Apply SiLU activation (generic version)
fn silu_generic<T: TensorDType + cudarc::driver::DeviceRepr>(
    input: &CudaTensor<T>,
) -> Result<CudaTensor<T>> {
    let shape = input.shape();
    let n = input.numel();

    let mut output = unsafe { CudaTensor::<T>::uninit(input.context(), shape)? };

    let device = input.context().device();
    let kernel_name = format!("silu_{}", kernel_suffix::<T>());

    let module_name = "silu";
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
            (output.cuda_slice_mut(), &input.cuda_slice(), n as i32),
        )?;
    }

    Ok(output)
}

/// Apply SiLU inplace (generic version)
fn silu_inplace_generic<T: TensorDType + cudarc::driver::DeviceRepr>(
    input: &mut CudaTensor<T>,
) -> Result<()> {
    let n = input.numel();
    let device = input.context().device();
    let kernel_name = format!("silu_inplace_{}", kernel_suffix::<T>());

    let module_name = "silu";
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
        func.launch(cfg, (input.cuda_slice_mut(), n as i32))?;
    }

    Ok(())
}

/// Fused SiLU + multiply (generic version)
fn silu_mul_generic<T: TensorDType + cudarc::driver::DeviceRepr>(
    gate: &CudaTensor<T>,
    up: &CudaTensor<T>,
) -> Result<CudaTensor<T>> {
    assert_eq!(gate.shape(), up.shape(), "gate and up must have same shape");

    let shape = gate.shape();
    let n = gate.numel();

    let mut output = unsafe { CudaTensor::<T>::uninit(gate.context(), shape)? };

    let device = gate.context().device();
    let kernel_name = format!("silu_mul_{}", kernel_suffix::<T>());

    let module_name = "silu";
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
                &gate.cuda_slice(),
                &up.cuda_slice(),
                n as i32,
            ),
        )?;
    }

    Ok(output)
}

/// Apply SiLU (Swish) activation: output = x * sigmoid(x)
///
/// Supports F32, F16, and BF16 tensor types.
///
/// # Errors
/// Returns an error if the operation fails
pub fn silu<T: TensorDType + cudarc::driver::DeviceRepr>(
    input: &CudaTensor<T>,
) -> Result<CudaTensor<T>> {
    silu_generic(input)
}

/// Apply SiLU (Swish) activation in place: x = x * sigmoid(x)
///
/// Supports F32, F16, and BF16 tensor types.
///
/// # Errors
/// Returns an error if the operation fails
pub fn silu_inplace<T: TensorDType + cudarc::driver::DeviceRepr>(
    input: &mut CudaTensor<T>,
) -> Result<()> {
    silu_inplace_generic(input)
}

/// Fused SiLU + multiply for SwiGLU: output = silu(gate) * up
///
/// Supports F32, F16, and BF16 tensor types.
///
/// # Errors
/// Returns an error if the operation fails
pub fn silu_mul<T: TensorDType + cudarc::driver::DeviceRepr>(
    gate: &CudaTensor<T>,
    up: &CudaTensor<T>,
) -> Result<CudaTensor<T>> {
    silu_mul_generic(gate, up)
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
    fn test_silu_inplace() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let input_data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut input = CudaTensor::from_slice(&ctx, &[5], &input_data).unwrap();

        silu_inplace(&mut input).unwrap();
        let result = input.to_vec().unwrap();

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
