//! Repeat KV heads for grouped-query attention

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::manual_div_ceil
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use crate::cuda::CudaTensor;
use infernum::dtype::DType;
use infernum::tensor::Tensor;
use infernum::Result;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/repeat_kv.ptx"));

/// Repeat KV heads for grouped-query attention, entirely on GPU
///
/// Input shape: (seq_len, num_kv_heads, head_dim)
/// Output shape: (seq_len, num_kv_heads * num_repeats, head_dim)
///
/// # Errors
/// Returns an error if the operation fails
pub fn repeat_kv(tensor: &CudaTensor, num_repeats: usize) -> Result<CudaTensor> {
    if num_repeats == 1 {
        return Ok(tensor.reshape(tensor.shape()));
    }

    let shape = tensor.shape();
    assert_eq!(
        shape.len(),
        3,
        "Expected 3D tensor (seq_len, num_kv_heads, head_dim)"
    );

    let seq_len = shape[0];
    let num_kv_heads = shape[1];
    let head_dim = shape[2];

    let new_num_heads = num_kv_heads * num_repeats;
    let output_shape = [seq_len, new_num_heads, head_dim];
    let n = seq_len * new_num_heads * head_dim;

    let mut output = unsafe { CudaTensor::uninit(tensor.context(), &output_shape, DType::F32)? };

    let device = tensor.context().device();

    let module_name = "repeat_kv";
    if !device.has_func(module_name, "repeat_kv_f32") {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(PTX),
            module_name,
            &["repeat_kv_f32"],
        )?;
    }

    let func = device.get_func(module_name, "repeat_kv_f32").unwrap();

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
                &tensor.cuda_slice(),
                seq_len as i32,
                num_kv_heads as i32,
                num_repeats as i32,
                head_dim as i32,
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
    fn test_repeat_kv_no_repeat() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = CudaTensor::from_slice(&ctx, &[1, 2, 3], &data).unwrap();

        let repeated = repeat_kv(&tensor, 1).unwrap();
        assert_eq!(repeated.shape(), &[1, 2, 3]);

        let result = repeated.to_vec::<f32>().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_repeat_kv() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // (seq=2, num_kv_heads=2, head_dim=3)
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, // seq=0, head=0
            4.0, 5.0, 6.0, // seq=0, head=1
            7.0, 8.0, 9.0, // seq=1, head=0
            10.0, 11.0, 12.0, // seq=1, head=1
        ];

        let tensor = CudaTensor::from_slice(&ctx, &[2, 2, 3], &data).unwrap();

        let repeated = repeat_kv(&tensor, 2).unwrap();

        assert_eq!(repeated.shape(), &[2, 4, 3]);

        let result = repeated.to_vec::<f32>().unwrap();

        // Each KV head repeated twice
        assert_eq!(result[0..3], [1.0, 2.0, 3.0]); // seq=0, head=0
        assert_eq!(result[3..6], [1.0, 2.0, 3.0]); // seq=0, head=1 (repeat of head 0)
        assert_eq!(result[6..9], [4.0, 5.0, 6.0]); // seq=0, head=2 (original head 1)
        assert_eq!(result[9..12], [4.0, 5.0, 6.0]); // seq=0, head=3 (repeat of head 1)
    }
}
