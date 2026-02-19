//! Embedding gather operation

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

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/embed.ptx"));

/// Gather embeddings from an embedding table using token IDs, entirely on GPU
///
/// # Arguments
/// * `ctx` - CUDA context
/// * `embed_table` - Embedding weight table of shape (vocab_size, hidden_size)
/// * `input_ids` - Token IDs as a host slice of length seq_len
///
/// # Returns
/// Output tensor of shape (seq_len, hidden_size)
///
/// # Errors
/// Returns an error if the operation fails
pub fn embedding_gather(
    ctx: &CudaContext,
    embed_table: &CudaTensor<f32>,
    input_ids: &[u32],
) -> Result<CudaTensor<f32>> {
    let hidden_size = embed_table.shape()[1];
    let seq_len = input_ids.len();

    let output_shape = [seq_len, hidden_size];
    let mut output = unsafe { CudaTensor::<f32>::uninit(ctx, &output_shape)? };

    // Copy input_ids to GPU
    let ids_gpu = ctx.device().htod_sync_copy(input_ids)?;

    let device = ctx.device();

    let module_name = "embed";
    if !device.has_func(module_name, "embedding_gather_f32") {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(PTX),
            module_name,
            &["embedding_gather_f32"],
        )?;
    }

    let func = device
        .get_func(module_name, "embedding_gather_f32")
        .unwrap();

    let block_size = 256;
    let grid_y = (hidden_size + block_size - 1) / block_size;

    let cfg = LaunchConfig {
        grid_dim: (seq_len as u32, grid_y as u32, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(
            cfg,
            (
                output.cuda_slice_mut(),
                &embed_table.cuda_slice(),
                &ids_gpu,
                seq_len as i32,
                hidden_size as i32,
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
    fn test_embedding_gather() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // vocab_size=4, hidden_size=3
        let embed_data: Vec<f32> = vec![
            0.1, 0.2, 0.3, // token 0
            1.1, 1.2, 1.3, // token 1
            2.1, 2.2, 2.3, // token 2
            3.1, 3.2, 3.3, // token 3
        ];
        let embed_table = CudaTensor::from_slice(&ctx, &[4, 3], &embed_data).unwrap();

        let input_ids: Vec<u32> = vec![2, 0, 3];
        let output = embedding_gather(&ctx, &embed_table, &input_ids).unwrap();

        assert_eq!(output.shape(), &[3, 3]);

        let result = output.to_vec().unwrap();
        // token 2, token 0, token 3
        let expected: Vec<f32> = vec![2.1, 2.2, 2.3, 0.1, 0.2, 0.3, 3.1, 3.2, 3.3];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-5, "Mismatch at {i}: {got} vs {exp}");
        }
    }
}
