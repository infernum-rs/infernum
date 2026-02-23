//! Bias addition: broadcast-add a 1D bias to each row of a 2D tensor

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

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/bias_add.ptx"));
const KERNEL_NAMES: &[&str] = &[
    "bias_add_f32",
    "bias_add_inplace_f32",
    "bias_add_f16",
    "bias_add_inplace_f16",
    "bias_add_bf16",
    "bias_add_inplace_bf16",
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
        panic!("Unsupported dtype for bias_add: {type_name}")
    }
}

/// Add a 1D bias to each row of a 2D tensor: `output[i, j] = input[i, j] + bias[j]`
///
/// Input shape: `(rows, cols)`, Bias shape: `(cols,)` → Output shape: `(rows, cols)`
///
/// # Errors
/// Returns an error if the operation fails.
pub fn bias_add<T: TensorDType + cudarc::driver::DeviceRepr>(
    input: &CudaTensor<T>,
    bias: &CudaTensor<T>,
) -> Result<CudaTensor<T>> {
    let shape = input.shape();
    assert_eq!(shape.len(), 2, "bias_add: input must be 2D");
    let rows = shape[0];
    let cols = shape[1];

    assert_eq!(
        bias.numel(),
        cols,
        "bias_add: bias length ({}) must match input cols ({cols})",
        bias.numel()
    );

    let n = rows * cols;
    let mut output = unsafe { CudaTensor::<T>::uninit(input.context(), shape)? };

    let device = input.context().device();
    let kernel_name = format!("bias_add_{}", kernel_suffix::<T>());

    let module_name = "bias_add";
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
                &input.cuda_slice(),
                &bias.cuda_slice(),
                rows as i32,
                cols as i32,
            ),
        )?;
    }

    Ok(output)
}

/// In-place bias addition: `input[i, j] += bias[j]`
///
/// Input shape: `(rows, cols)`, Bias shape: `(cols,)`
///
/// # Errors
/// Returns an error if the operation fails.
pub fn bias_add_inplace<T: TensorDType + cudarc::driver::DeviceRepr>(
    input: &mut CudaTensor<T>,
    bias: &CudaTensor<T>,
) -> Result<()> {
    let shape = input.shape().to_vec();
    assert_eq!(shape.len(), 2, "bias_add_inplace: input must be 2D");
    let rows = shape[0];
    let cols = shape[1];

    assert_eq!(
        bias.numel(),
        cols,
        "bias_add_inplace: bias length ({}) must match input cols ({cols})",
        bias.numel()
    );

    let n = rows * cols;
    let device = input.context().device();
    let kernel_name = format!("bias_add_inplace_{}", kernel_suffix::<T>());

    let module_name = "bias_add";
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
                input.cuda_slice_mut(),
                &bias.cuda_slice(),
                rows as i32,
                cols as i32,
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
    fn test_bias_add() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias_data: Vec<f32> = vec![10.0, 20.0, 30.0];

        let input = CudaTensor::from_slice(&ctx, &[2, 3], &input_data).unwrap();
        let bias = CudaTensor::from_slice(&ctx, &[3], &bias_data).unwrap();

        let output = bias_add(&input, &bias).unwrap();

        assert_eq!(output.shape(), &[2, 3]);
        let result = output.to_vec().unwrap();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_bias_add_zero_bias() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias_data: Vec<f32> = vec![0.0, 0.0, 0.0];

        let input = CudaTensor::from_slice(&ctx, &[2, 3], &input_data).unwrap();
        let bias = CudaTensor::from_slice(&ctx, &[3], &bias_data).unwrap();

        let output = bias_add(&input, &bias).unwrap();
        let result = output.to_vec().unwrap();
        assert_eq!(result, input_data);
    }

    #[test]
    fn test_bias_add_single_row() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bias_data: Vec<f32> = vec![0.5, -0.5, 1.0, -1.0];

        let input = CudaTensor::from_slice(&ctx, &[1, 4], &input_data).unwrap();
        let bias = CudaTensor::from_slice(&ctx, &[4], &bias_data).unwrap();

        let output = bias_add(&input, &bias).unwrap();
        let result = output.to_vec().unwrap();
        assert_eq!(result, vec![1.5, 1.5, 4.0, 3.0]);
    }

    #[test]
    fn test_bias_add_inplace() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias_data: Vec<f32> = vec![10.0, 20.0, 30.0];

        let mut input = CudaTensor::from_slice(&ctx, &[2, 3], &input_data).unwrap();
        let bias = CudaTensor::from_slice(&ctx, &[3], &bias_data).unwrap();

        bias_add_inplace(&mut input, &bias).unwrap();

        assert_eq!(input.shape(), &[2, 3]);
        let result = input.to_vec().unwrap();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_bias_add_many_rows() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let cols = 128;
        let rows = 64;
        let input_data: Vec<f32> = (0..rows * cols).map(|x| x as f32).collect();
        let bias_data: Vec<f32> = (0..cols).map(|x| x as f32 * 0.1).collect();

        let input = CudaTensor::from_slice(&ctx, &[rows, cols], &input_data).unwrap();
        let bias = CudaTensor::from_slice(&ctx, &[cols], &bias_data).unwrap();

        let output = bias_add(&input, &bias).unwrap();
        let result = output.to_vec().unwrap();

        for r in 0..rows {
            for c in 0..cols {
                let expected = input_data[r * cols + c] + bias_data[c];
                let got = result[r * cols + c];
                assert!(
                    (got - expected).abs() < 1e-5,
                    "Mismatch at ({r}, {c}): got {got}, expected {expected}"
                );
            }
        }
    }
}
