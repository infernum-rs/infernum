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
use crate::tensor::Tensor;
use crate::Result;

const ADD_KERNEL: &str = r#"
extern "C" __global__ void add_f32(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}
"#;

/// Add two tensors element-wise on GPU: output = a + b
///
/// Both tensors must have the same shape.
///
/// # Errors
/// Returns an error if the operation fails
pub fn add(a: &CudaTensor<f32>, b: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    assert_eq!(a.shape(), b.shape(), "Shapes must match for addition");

    let shape = a.shape();
    let n = a.numel();

    let mut output = unsafe { CudaTensor::<f32>::uninit(a.context(), shape)? };

    let device = a.context().device();

    // Compile kernel
    let module_name = "add";
    if !device.has_func(module_name, "add_f32") {
        let ptx = cudarc::nvrtc::safe::compile_ptx(ADD_KERNEL)?;
        device.load_ptx(ptx, module_name, &["add_f32"])?;
    }

    let func = device.get_func(module_name, "add_f32").unwrap();

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
                a.cuda_slice(),
                b.cuda_slice(),
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
}
