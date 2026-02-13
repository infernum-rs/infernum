//! Transpose operations

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::manual_div_ceil
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use crate::cuda::CudaTensor;
use crate::tensor::Tensor;
use crate::Result;

const TRANSPOSE_KERNEL: &str = r#"
// Transpose 2D: (rows, cols) -> (cols, rows)
extern "C" __global__ void transpose_2d_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int rows,
    const int cols
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx < total) {
        const int r = idx / cols;
        const int c = idx % cols;
        output[c * rows + r] = input[r * cols + c];
    }
}

// Transpose 3D: (a, b, c) -> (b, a, c)
// Swaps the first two dimensions
extern "C" __global__ void transpose_012_to_102_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int dim_a,
    const int dim_b,
    const int dim_c
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = dim_a * dim_b * dim_c;
    if (idx < total) {
        const int i = idx / (dim_b * dim_c);
        const int remainder = idx % (dim_b * dim_c);
        const int j = remainder / dim_c;
        const int k = remainder % dim_c;

        // src: (i, j, k) -> dst: (j, i, k)
        const int dst_idx = j * dim_a * dim_c + i * dim_c + k;
        output[dst_idx] = input[idx];
    }
}

// Transpose last two dims of 3D: (a, b, c) -> (a, c, b)
extern "C" __global__ void transpose_last_two_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int dim_a,
    const int dim_b,
    const int dim_c
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = dim_a * dim_b * dim_c;
    if (idx < total) {
        const int i = idx / (dim_b * dim_c);
        const int remainder = idx % (dim_b * dim_c);
        const int j = remainder / dim_c;
        const int k = remainder % dim_c;

        // src: (i, j, k) -> dst: (i, k, j)
        const int dst_idx = i * dim_c * dim_b + k * dim_b + j;
        output[dst_idx] = input[idx];
    }
}
"#;

fn ensure_transpose_kernel(device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<()> {
    let module_name = "transpose";
    if !device.has_func(module_name, "transpose_2d_f32") {
        let ptx = cudarc::nvrtc::safe::compile_ptx(TRANSPOSE_KERNEL)?;
        device.load_ptx(
            ptx,
            module_name,
            &[
                "transpose_2d_f32",
                "transpose_012_to_102_f32",
                "transpose_last_two_f32",
            ],
        )?;
    }
    Ok(())
}

/// Transpose a 2D tensor: (rows, cols) -> (cols, rows)
///
/// # Errors
/// Returns an error if the operation fails
pub fn transpose_2d(tensor: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    let shape = tensor.shape();
    assert_eq!(shape.len(), 2, "Expected 2D tensor");

    let rows = shape[0];
    let cols = shape[1];
    let n = rows * cols;

    let output_shape = [cols, rows];
    let mut output = unsafe { CudaTensor::<f32>::uninit(tensor.context(), &output_shape)? };

    let device = tensor.context().device();
    ensure_transpose_kernel(device)?;

    let func = device.get_func("transpose", "transpose_2d_f32").unwrap();

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
                tensor.cuda_slice(),
                rows as i32,
                cols as i32,
            ),
        )?;
    }

    Ok(output)
}

/// Transpose 3D tensor: (a, b, c) -> (b, a, c)
///
/// # Errors
/// Returns an error if the operation fails
pub fn transpose_012_to_102(tensor: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    let shape = tensor.shape();
    assert_eq!(shape.len(), 3, "Expected 3D tensor");

    let a = shape[0];
    let b = shape[1];
    let c = shape[2];
    let n = a * b * c;

    let output_shape = [b, a, c];
    let mut output = unsafe { CudaTensor::<f32>::uninit(tensor.context(), &output_shape)? };

    let device = tensor.context().device();
    ensure_transpose_kernel(device)?;

    let func = device
        .get_func("transpose", "transpose_012_to_102_f32")
        .unwrap();

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
                tensor.cuda_slice(),
                a as i32,
                b as i32,
                c as i32,
            ),
        )?;
    }

    Ok(output)
}

/// Transpose last two dims of 3D tensor: (a, b, c) -> (a, c, b)
///
/// # Errors
/// Returns an error if the operation fails
pub fn transpose_last_two(tensor: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    let shape = tensor.shape();
    assert_eq!(shape.len(), 3, "Expected 3D tensor");

    let a = shape[0];
    let b = shape[1];
    let c = shape[2];
    let n = a * b * c;

    let output_shape = [a, c, b];
    let mut output = unsafe { CudaTensor::<f32>::uninit(tensor.context(), &output_shape)? };

    let device = tensor.context().device();
    ensure_transpose_kernel(device)?;

    let func = device
        .get_func("transpose", "transpose_last_two_f32")
        .unwrap();

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
                tensor.cuda_slice(),
                a as i32,
                b as i32,
                c as i32,
            ),
        )?;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;

    #[test]
    fn test_transpose_2d() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = CudaTensor::from_slice(&ctx, &[2, 3], &data).unwrap();

        let transposed = transpose_2d(&tensor).unwrap();

        assert_eq!(transposed.shape(), &[3, 2]);

        let result = transposed.to_vec().unwrap();
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_012_to_102() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // (2, 3, 2) -> (3, 2, 2)
        let data: Vec<f32> = vec![
            1.0, 2.0, // (0,0,*)
            3.0, 4.0, // (0,1,*)
            5.0, 6.0, // (0,2,*)
            7.0, 8.0, // (1,0,*)
            9.0, 10.0, // (1,1,*)
            11.0, 12.0, // (1,2,*)
        ];
        let tensor = CudaTensor::from_slice(&ctx, &[2, 3, 2], &data).unwrap();

        let transposed = transpose_012_to_102(&tensor).unwrap();

        assert_eq!(transposed.shape(), &[3, 2, 2]);

        let result = transposed.to_vec().unwrap();
        // (b, a, c): (0,0,*)=1,2  (0,1,*)=7,8  (1,0,*)=3,4  (1,1,*)=9,10  (2,0,*)=5,6  (2,1,*)=11,12
        assert_eq!(
            result,
            vec![1.0, 2.0, 7.0, 8.0, 3.0, 4.0, 9.0, 10.0, 5.0, 6.0, 11.0, 12.0]
        );
    }

    #[test]
    fn test_transpose_last_two() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // (2, 3, 4) -> (2, 4, 3)
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let tensor = CudaTensor::from_slice(&ctx, &[2, 3, 4], &data).unwrap();

        let transposed = transpose_last_two(&tensor).unwrap();

        assert_eq!(transposed.shape(), &[2, 4, 3]);

        let result = transposed.to_vec().unwrap();

        // Verify a few elements:
        // src(0, 0, 0) = 0.0 -> dst(0, 0, 0) = 0.0
        assert_eq!(result[0], 0.0);
        // src(0, 0, 1) = 1.0 -> dst(0, 1, 0) = result[3]
        assert_eq!(result[3], 1.0);
        // src(0, 1, 0) = 4.0 -> dst(0, 0, 1) = result[1]
        assert_eq!(result[1], 4.0);
    }
}
