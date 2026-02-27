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
use infernum::dtype::DType;
use infernum::tensor::Tensor;
use infernum::Result;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/embed.ptx"));
const KERNEL_NAMES: &[&str] = &[
    "embedding_gather_f32",
    "embedding_gather_f16",
    "embedding_gather_bf16",
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

/// Gather embeddings (generic version)
fn embedding_gather_generic(
    ctx: &CudaContext,
    embed_table: &CudaTensor,
    input_ids: &[u32],
) -> Result<CudaTensor> {
    let dtype = embed_table.dtype();
    let hidden_size = embed_table.shape()[1];
    let seq_len = input_ids.len();

    let output_shape = [seq_len, hidden_size];
    let mut output = unsafe { CudaTensor::uninit(ctx, &output_shape, dtype)? };

    // Copy input_ids to GPU
    let ids_gpu = ctx.device().htod_sync_copy(input_ids)?;

    let device = ctx.device();
    let kernel_name = format!("embedding_gather_{}", kernel_suffix(dtype));

    let module_name = "embed";
    if !device.has_func(module_name, &kernel_name) {
        device.load_ptx(cudarc::nvrtc::Ptx::from_src(PTX), module_name, KERNEL_NAMES)?;
    }

    let func = device.get_func(module_name, &kernel_name).unwrap();

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

/// Gather embeddings from an embedding table using token IDs, entirely on GPU
///
/// Supports F32, F16, and BF16 tensor types.
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
    embed_table: &CudaTensor,
    input_ids: &[u32],
) -> Result<CudaTensor> {
    embedding_gather_generic(ctx, embed_table, input_ids)
}

/// Gather embeddings using token IDs that are already on the GPU.
///
/// Identical to [`embedding_gather`] but takes a device-side `CudaSlice<u32>`
/// instead of a host `&[u32]`, avoiding the `htod_sync_copy` that would break
/// CUDA graph capture.
///
/// # Errors
/// Returns an error if the operation fails
pub fn embedding_gather_from_device(
    ctx: &CudaContext,
    embed_table: &CudaTensor,
    input_ids: &cudarc::driver::CudaSlice<u32>,
    seq_len: usize,
) -> Result<CudaTensor> {
    let dtype = embed_table.dtype();
    let hidden_size = embed_table.shape()[1];

    let output_shape = [seq_len, hidden_size];
    let mut output = unsafe { CudaTensor::uninit(ctx, &output_shape, dtype)? };

    let device = ctx.device();
    let kernel_name = format!("embedding_gather_{}", kernel_suffix(dtype));

    let module_name = "embed";
    if !device.has_func(module_name, &kernel_name) {
        device.load_ptx(cudarc::nvrtc::Ptx::from_src(PTX), module_name, KERNEL_NAMES)?;
    }

    let func = device.get_func(module_name, &kernel_name).unwrap();

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
                input_ids,
                seq_len as i32,
                hidden_size as i32,
            ),
        )?;
    }

    Ok(output)
}

/// Gather embedding rows using a device-side tensor of token IDs.
///
/// `input_ids` is a `CudaTensor` with dtype I32 (holding u32 token IDs).
pub fn embedding_gather_from_tensor(
    ctx: &CudaContext,
    embed_table: &CudaTensor,
    input_ids: &CudaTensor,
    seq_len: usize,
) -> Result<CudaTensor> {
    assert!(
        input_ids.dtype() == DType::U32,
        "embedding_gather_from_tensor: input_ids must be U32 (holding u32 token IDs), got {:?}",
        input_ids.dtype()
    );
    let dtype = embed_table.dtype();
    let hidden_size = embed_table.shape()[1];

    let output_shape = [seq_len, hidden_size];
    let mut output = unsafe { CudaTensor::uninit(ctx, &output_shape, dtype)? };

    let device = ctx.device();
    let kernel_name = format!("embedding_gather_{}", kernel_suffix(dtype));

    let module_name = "embed";
    if !device.has_func(module_name, &kernel_name) {
        device.load_ptx(cudarc::nvrtc::Ptx::from_src(PTX), module_name, KERNEL_NAMES)?;
    }

    let func = device.get_func(module_name, &kernel_name).unwrap();

    let block_size = 256;
    let grid_y = (hidden_size + block_size - 1) / block_size;

    let cfg = LaunchConfig {
        grid_dim: (seq_len as u32, grid_y as u32, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // I32 and u32 are bit-identical; the kernel treats the pointer as
    // unsigned indices into the embedding table.
    unsafe {
        func.launch(
            cfg,
            (
                output.cuda_slice_mut(),
                &embed_table.cuda_slice(),
                &input_ids.cuda_slice(),
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

        let result = output.to_vec::<f32>().unwrap();
        // token 2, token 0, token 3
        let expected: Vec<f32> = vec![2.1, 2.2, 2.3, 0.1, 0.2, 0.3, 3.1, 3.2, 3.3];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-5, "Mismatch at {i}: {got} vs {exp}");
        }
    }

    #[test]
    fn test_embedding_gather_from_device() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // vocab_size=4, hidden_size=3
        let embed_data: Vec<f32> = vec![
            0.1, 0.2, 0.3, // token 0
            1.1, 1.2, 1.3, // token 1
            2.1, 2.2, 2.3, // token 2
            3.1, 3.2, 3.3, // token 3
        ];
        let embed_table = CudaTensor::from_slice(&ctx, &[4, 3], &embed_data).unwrap();

        let input_ids_gpu = ctx.device().htod_sync_copy(&[2_u32, 0, 3]).unwrap();
        let output = embedding_gather_from_device(&ctx, &embed_table, &input_ids_gpu, 3).unwrap();

        assert_eq!(output.shape(), &[3, 3]);

        let result = output.to_vec::<f32>().unwrap();
        let expected: Vec<f32> = vec![2.1, 2.2, 2.3, 0.1, 0.2, 0.3, 3.1, 3.2, 3.3];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-5, "Mismatch at {i}: {got} vs {exp}");
        }
    }

    #[test]
    fn test_embedding_gather_from_device_single_token() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let embed_data: Vec<f32> = vec![
            0.1, 0.2, 0.3, // token 0
            1.1, 1.2, 1.3, // token 1
        ];
        let embed_table = CudaTensor::from_slice(&ctx, &[2, 3], &embed_data).unwrap();

        let input_ids_gpu = ctx.device().htod_sync_copy(&[1_u32]).unwrap();
        let output = embedding_gather_from_device(&ctx, &embed_table, &input_ids_gpu, 1).unwrap();

        assert_eq!(output.shape(), &[1, 3]);

        let result = output.to_vec::<f32>().unwrap();
        let expected: Vec<f32> = vec![1.1, 1.2, 1.3];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-5, "Mismatch at {i}: {got} vs {exp}");
        }
    }
}
