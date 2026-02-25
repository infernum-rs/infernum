//! Softmax operation

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::missing_panics_doc
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use crate::cuda::CudaTensor;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::Result;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/softmax.ptx"));

/// Apply softmax along the last dimension
///
/// Input shape: (..., row_size)
/// Softmax is applied independently to each row.
///
/// # Errors
/// Returns an error if the operation fails
pub fn softmax(input: &CudaTensor) -> Result<CudaTensor> {
    let shape = input.shape();
    let row_size = *shape
        .last()
        .expect("Input must have at least one dimension");
    let num_rows: usize = shape[..shape.len() - 1].iter().product();
    let num_rows = if num_rows == 0 { 1 } else { num_rows };

    let mut output = unsafe { CudaTensor::uninit(input.context(), shape, DType::F32)? };

    let device = input.context().device();

    let module_name = "softmax";
    if !device.has_func(module_name, "softmax_f32") {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(PTX),
            module_name,
            &["softmax_f32", "softmax_causal_f32"],
        )?;
    }

    let func = device.get_func(module_name, "softmax_f32").unwrap();

    let block_size = 256.min(row_size.next_power_of_two());
    let shared_mem = block_size * std::mem::size_of::<f32>();

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

/// Apply softmax with causal masking for attention
///
/// For each query position q, only key positions k where k <= q are considered.
///
/// # Arguments
/// * `input` - Attention scores of shape (num_heads, seq_len) for a single query
/// * `query_idx` - The query position (0-indexed within the current sequence)
/// * `position_offset` - Offset for KV cache scenarios
///
/// # Errors
/// Returns an error if the operation fails
pub fn softmax_causal(
    input: &CudaTensor,
    query_idx: usize,
    position_offset: usize,
) -> Result<CudaTensor> {
    let shape = input.shape();
    assert_eq!(shape.len(), 2, "Expected (num_heads, seq_len)");

    let num_heads = shape[0];
    let row_size = shape[1];

    let mut output = unsafe { CudaTensor::uninit(input.context(), shape, DType::F32)? };

    let device = input.context().device();

    let module_name = "softmax";
    if !device.has_func(module_name, "softmax_causal_f32") {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(PTX),
            module_name,
            &["softmax_f32", "softmax_causal_f32"],
        )?;
    }

    let func = device.get_func(module_name, "softmax_causal_f32").unwrap();

    let block_size = 256.min(row_size.next_power_of_two());
    let shared_mem = block_size * std::mem::size_of::<f32>();

    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
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
                query_idx as i32,
                position_offset as i32,
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
    fn test_softmax() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0];
        let input = CudaTensor::from_slice(&ctx, &[2, 4], &input_data).unwrap();

        let output = softmax(&input).unwrap();
        let result = output.to_vec().unwrap();

        // Row 0: softmax([1, 2, 3, 4])
        let max0 = 4.0_f32;
        let exp0: Vec<f32> = vec![
            (1.0 - max0).exp(),
            (2.0 - max0).exp(),
            (3.0 - max0).exp(),
            (4.0 - max0).exp(),
        ];
        let sum0: f32 = exp0.iter().sum();
        let expected0: Vec<f32> = exp0.iter().map(|x| x / sum0).collect();

        for i in 0..4 {
            assert!(
                (result[i] - expected0[i]).abs() < 1e-5,
                "Row 0 mismatch at {}: {} vs {}",
                i,
                result[i],
                expected0[i]
            );
        }

        // Row 1: softmax([1, 1, 1, 1]) = [0.25, 0.25, 0.25, 0.25]
        for i in 4..8 {
            assert!(
                (result[i] - 0.25).abs() < 1e-5,
                "Row 1 mismatch at {}: {} vs 0.25",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let input_data: Vec<f32> = (0..32).map(|x| x as f32 * 0.1).collect();
        let input = CudaTensor::from_slice(&ctx, &[4, 8], &input_data).unwrap();

        let output = softmax(&input).unwrap();
        let result = output.to_vec().unwrap();

        // Each row should sum to 1
        for row in 0..4 {
            let sum: f32 = result[row * 8..(row + 1) * 8].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Row {} sum is {} instead of 1.0",
                row,
                sum
            );
        }
    }

    #[test]
    fn test_softmax_causal_masks_future() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // 2 heads, 4 key positions, query at position 1 (can attend to k=0,1)
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // head 0
            0.5, 0.5, 0.5, 0.5, // head 1
        ];
        let input = CudaTensor::from_slice(&ctx, &[2, 4], &input_data).unwrap();

        let output = softmax_causal(&input, 1, 0).unwrap();
        let result = output.to_vec().unwrap();

        // Positions 2 and 3 should be masked (zero)
        for head in 0..2 {
            assert!(
                result[head * 4 + 2].abs() < 1e-6,
                "Head {head} position 2 should be masked"
            );
            assert!(
                result[head * 4 + 3].abs() < 1e-6,
                "Head {head} position 3 should be masked"
            );

            // Valid positions should sum to 1
            let sum: f32 = result[head * 4..head * 4 + 2].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Head {head} valid positions sum to {sum} instead of 1.0"
            );
        }
    }

    #[test]
    fn test_softmax_causal_first_position() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // query_idx=0: can only attend to position 0
        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let input = CudaTensor::from_slice(&ctx, &[1, 3], &input_data).unwrap();

        let output = softmax_causal(&input, 0, 0).unwrap();
        let result = output.to_vec().unwrap();

        // Only position 0 is valid -> softmax of single element = 1.0
        assert!(
            (result[0] - 1.0).abs() < 1e-5,
            "Position 0 should be 1.0, got {}",
            result[0]
        );
        assert!(result[1].abs() < 1e-6);
        assert!(result[2].abs() < 1e-6);
    }
}
