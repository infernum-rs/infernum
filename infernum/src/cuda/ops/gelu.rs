//! GELU activation function (tanh approximation)
//!
//! Implements `gelu_pytorch_tanh`:
//! `y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
//!
//! Used by Gemma models in their GeGLU FFN.

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

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/gelu.ptx"));
const KERNEL_NAMES: &[&str] = &[
    "gelu_f32",
    "gelu_inplace_f32",
    "gelu_mul_f32",
    "gelu_f16",
    "gelu_inplace_f16",
    "gelu_mul_f16",
    "gelu_bf16",
    "gelu_inplace_bf16",
    "gelu_mul_bf16",
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

fn ensure_module(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    kernel_name: &str,
) -> Result<()> {
    let module_name = "gelu";
    if !device.has_func(module_name, kernel_name) {
        device.load_ptx(cudarc::nvrtc::Ptx::from_src(PTX), module_name, KERNEL_NAMES)?;
    }
    Ok(())
}

/// Apply GELU activation (tanh approximation)
///
/// Supports F32, F16, and BF16 tensor types.
///
/// # Errors
/// Returns an error if the operation fails
pub fn gelu(input: &CudaTensor) -> Result<CudaTensor> {
    let dtype = input.dtype();
    let shape = input.shape();
    let n = input.numel();

    let mut output = unsafe { CudaTensor::uninit(input.context(), shape, dtype)? };

    let device = input.context().device();
    let kernel_name = format!("gelu_{}", kernel_suffix(dtype));
    ensure_module(device, &kernel_name)?;

    let func = device.get_func("gelu", &kernel_name).unwrap();

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

/// Apply GELU activation in place
///
/// Supports F32, F16, and BF16 tensor types.
///
/// # Errors
/// Returns an error if the operation fails
pub fn gelu_inplace(input: &mut CudaTensor) -> Result<()> {
    let dtype = input.dtype();
    let n = input.numel();
    let device = input.context().device();
    let kernel_name = format!("gelu_inplace_{}", kernel_suffix(dtype));
    ensure_module(device, &kernel_name)?;

    let func = device.get_func("gelu", &kernel_name).unwrap();

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

/// Fused GELU + multiply for GeGLU: output = gelu(gate) * up
///
/// Supports F32, F16, and BF16 tensor types.
///
/// # Errors
/// Returns an error if the operation fails
pub fn gelu_mul(gate: &CudaTensor, up: &CudaTensor) -> Result<CudaTensor> {
    let dtype = gate.dtype();
    assert_eq!(gate.shape(), up.shape(), "gate and up must have same shape");

    let shape = gate.shape();
    let n = gate.numel();

    let mut output = unsafe { CudaTensor::uninit(gate.context(), shape, dtype)? };

    let device = gate.context().device();
    let kernel_name = format!("gelu_mul_{}", kernel_suffix(dtype));
    ensure_module(device, &kernel_name)?;

    let func = device.get_func("gelu", &kernel_name).unwrap();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;

    /// Reference GELU (tanh approximation) for testing
    fn gelu_ref(x: f32) -> f32 {
        let inner = 0.797_884_6_f32 * (x + 0.044_715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    #[test]
    fn test_gelu() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let input_data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let input = CudaTensor::from_slice(&ctx, &[5], &input_data).unwrap();

        let output = gelu(&input).unwrap();
        let result = output.to_vec::<f32>().unwrap();

        for (i, &x) in input_data.iter().enumerate() {
            let expected = gelu_ref(x);
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
    fn test_gelu_known_values() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let input_data: Vec<f32> = vec![0.0, 1.0, -1.0];
        let input = CudaTensor::from_slice(&ctx, &[3], &input_data).unwrap();

        let output = gelu(&input).unwrap();
        let result = output.to_vec::<f32>().unwrap();

        assert!(
            result[0].abs() < 1e-6,
            "gelu(0.0) should be ~0.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 0.8412).abs() < 1e-3,
            "gelu(1.0) should be ~0.8412, got {}",
            result[1]
        );
        assert!(
            (result[2] - (-0.1588)).abs() < 1e-3,
            "gelu(-1.0) should be ~-0.1588, got {}",
            result[2]
        );
    }

    #[test]
    fn test_gelu_inplace() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let input_data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut input = CudaTensor::from_slice(&ctx, &[5], &input_data).unwrap();

        gelu_inplace(&mut input).unwrap();
        let result = input.to_vec::<f32>().unwrap();

        for (i, &x) in input_data.iter().enumerate() {
            let expected = gelu_ref(x);
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
    fn test_gelu_mul() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let gate_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let up_data: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0];

        let gate = CudaTensor::from_slice(&ctx, &[4], &gate_data).unwrap();
        let up = CudaTensor::from_slice(&ctx, &[4], &up_data).unwrap();

        let output = gelu_mul(&gate, &up).unwrap();
        let result = output.to_vec::<f32>().unwrap();

        for i in 0..4 {
            let expected = gelu_ref(gate_data[i]) * up_data[i];
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
    fn test_gelu_bf16() {
        use half::bf16;

        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let input_data: Vec<bf16> = vec![
            bf16::from_f32(-2.0),
            bf16::from_f32(-1.0),
            bf16::from_f32(0.0),
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
        ];
        let input = CudaTensor::from_slice(&ctx, &[5], &input_data).unwrap();

        let output = gelu(&input).unwrap();
        let result: Vec<f32> = output
            .to_vec::<bf16>()
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect();

        let ref_data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        for (i, &x) in ref_data.iter().enumerate() {
            let expected = gelu_ref(x);
            let actual = result[i];
            assert!(
                (actual - expected).abs() < 0.05,
                "BF16 mismatch at {}: {} vs {}",
                i,
                actual,
                expected
            );
        }
    }
}
