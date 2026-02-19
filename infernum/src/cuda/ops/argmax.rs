//! Argmax over the last dimension

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::manual_div_ceil
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use crate::cuda::{CudaContext, CudaTensor};
use crate::tensor::Tensor;
use crate::Result;

const ARGMAX_KERNEL: &str = r#"
extern "C" __global__ void argmax_last_f32(
    unsigned int* __restrict__ output,
    const float* __restrict__ input,
    const int row_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ char smem[];
    float* shared_val = (float*)smem;
    unsigned int* shared_idx = (unsigned int*)(smem + blockDim.x * sizeof(float));

    const float* row_input = input + row * row_size;

    // Each thread finds local max across its strided elements
    float local_max = -1e38f;
    unsigned int local_idx = 0;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float val = row_input[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    shared_val[tid] = local_max;
    shared_idx[tid] = local_idx;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_val[tid + stride] > shared_val[tid]) {
                shared_val[tid] = shared_val[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = shared_idx[0];
    }
}
"#;

/// Compute argmax over the last dimension of a 2D tensor.
///
/// Input: (num_rows, row_size)
/// Output: Vec<u32> of length num_rows, where each element is the index of the maximum value.
///
/// # Errors
/// Returns an error if the kernel launch or data transfer fails
pub fn argmax_last(input: &CudaTensor<f32>) -> Result<Vec<u32>> {
    let shape = input.shape();
    assert!(
        shape.len() == 2,
        "argmax_last expects a 2D tensor, got shape {shape:?}"
    );

    let num_rows = shape[0];
    let row_size = shape[1];
    let ctx = input.context();

    let output = argmax_last_gpu(ctx, input, num_rows, row_size)?;
    output.to_vec()
}

/// Compute argmax of the **last row** of a 2D logits tensor, returning a
/// single token index.
///
/// During decode, logits are `(1, vocab_size)`, so this avoids the `Vec`
/// allocation and only transfers 4 bytes (one `u32`) from GPU to host.
/// For multi-row tensors (prefill), it runs the kernel on just the last row.
///
/// # Errors
/// Returns an error if the kernel launch or data transfer fails.
pub fn argmax_last_scalar(input: &CudaTensor<f32>) -> Result<u32> {
    let shape = input.shape();
    assert!(
        shape.len() == 2,
        "argmax_last_scalar expects a 2D tensor, got shape {shape:?}"
    );

    let num_rows = shape[0];
    let row_size = shape[1];
    let ctx = input.context();
    let device = ctx.device();

    ensure_kernel_loaded(ctx)?;
    let func = device.get_func(MODULE_NAME, "argmax_last_f32").unwrap();

    let block_size = 256.min(row_size.next_power_of_two());
    let shared_mem = block_size * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>());

    // Single-element output on GPU
    let mut out_device = device.alloc_zeros::<u32>(1)?;

    // Point the kernel at the last row
    let last_row_offset = (num_rows - 1) * row_size;
    let last_row = input
        .cuda_slice()
        .slice(last_row_offset..num_rows * row_size);

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    unsafe {
        func.launch(cfg, (&mut out_device, &last_row, row_size as i32))?;
    }

    let result = device.dtoh_sync_copy(&out_device)?;
    Ok(result[0])
}

/// Ensure the argmax PTX module is loaded on the device.
fn ensure_kernel_loaded(ctx: &CudaContext) -> Result<()> {
    let device = ctx.device();
    if !device.has_func(MODULE_NAME, "argmax_last_f32") {
        let ptx = cudarc::nvrtc::safe::compile_ptx(ARGMAX_KERNEL)?;
        device.load_ptx(ptx, MODULE_NAME, &["argmax_last_f32"])?;
    }
    Ok(())
}

const MODULE_NAME: &str = "argmax";

/// Compute argmax over the last dimension, returning a GPU tensor of u32 indices.
fn argmax_last_gpu(
    ctx: &CudaContext,
    input: &CudaTensor<f32>,
    num_rows: usize,
    row_size: usize,
) -> Result<CudaTensor<u32>> {
    let mut output = unsafe { CudaTensor::<u32>::uninit(ctx, &[num_rows])? };

    ensure_kernel_loaded(ctx)?;
    let func = ctx
        .device()
        .get_func(MODULE_NAME, "argmax_last_f32")
        .unwrap();

    let block_size = 256.min(row_size.next_power_of_two());
    let shared_mem = block_size * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>());

    let cfg = LaunchConfig {
        grid_dim: (num_rows as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    unsafe {
        func.launch(
            cfg,
            (
                output.cuda_slice_mut(),
                &input.cuda_slice(),
                row_size as i32,
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
    fn test_argmax_single_row() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0];
        let input = CudaTensor::from_slice(&ctx, &[1, 4], &data).unwrap();

        let result = argmax_last(&input).unwrap();
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_argmax_multiple_rows() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let data: Vec<f32> = vec![
            1.0, 5.0, 3.0, // row 0: max at index 1
            9.0, 2.0, 4.0, // row 1: max at index 0
            0.0, 0.0, 7.0, // row 2: max at index 2
        ];
        let input = CudaTensor::from_slice(&ctx, &[3, 3], &data).unwrap();

        let result = argmax_last(&input).unwrap();
        assert_eq!(result, vec![1, 0, 2]);
    }

    #[test]
    fn test_argmax_large_row() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let row_size = 32_000; // typical vocab size
        let mut data: Vec<f32> = (0..row_size).map(|i| -(i as f32)).collect();
        data[12_345] = 100.0; // spike at index 12345

        let input = CudaTensor::from_slice(&ctx, &[1, row_size], &data).unwrap();

        let result = argmax_last(&input).unwrap();
        assert_eq!(result, vec![12_345]);
    }

    #[test]
    fn test_argmax_negative_values() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let data: Vec<f32> = vec![-10.0, -3.0, -5.0, -1.0, -8.0];
        let input = CudaTensor::from_slice(&ctx, &[1, 5], &data).unwrap();

        let result = argmax_last(&input).unwrap();
        assert_eq!(result, vec![3]); // -1.0 is the largest
    }

    #[test]
    fn test_argmax_scalar_single_row() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0];
        let input = CudaTensor::from_slice(&ctx, &[1, 4], &data).unwrap();

        let result = argmax_last_scalar(&input).unwrap();
        assert_eq!(result, 1);
    }

    #[test]
    fn test_argmax_scalar_picks_last_row() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let data: Vec<f32> = vec![
            9.0, 0.0, 0.0, // row 0: max at index 0 (ignored)
            0.0, 0.0, 7.0, // row 1 (last): max at index 2
        ];
        let input = CudaTensor::from_slice(&ctx, &[2, 3], &data).unwrap();

        let result = argmax_last_scalar(&input).unwrap();
        assert_eq!(result, 2);
    }

    #[test]
    fn test_argmax_scalar_large_vocab() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let row_size = 32_000;
        let mut data: Vec<f32> = (0..row_size).map(|i| -(i as f32)).collect();
        data[12_345] = 100.0;

        let input = CudaTensor::from_slice(&ctx, &[1, row_size], &data).unwrap();

        let result = argmax_last_scalar(&input).unwrap();
        assert_eq!(result, 12_345);
    }
}
