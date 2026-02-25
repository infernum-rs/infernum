//! GPU-side tensor manipulation ops for MLA (Multi-head Latent Attention).
//!
//! These replace CPU roundtrip helpers (`.to_vec()` + `from_slice()`) with
//! lightweight CUDA kernels that keep data on the GPU.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::manual_div_ceil
)]

use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits};

use crate::cuda::CudaTensor;
use crate::dtype::{DType, TensorDType};
use crate::tensor::Tensor;
use crate::Result;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/mla_tensor_ops.ptx"));

const MODULE: &str = "mla_tensor_ops";

fn ensure_mla_kernel(device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<()> {
    if !device.has_func(MODULE, "split_inner_dim_f32") {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(PTX),
            MODULE,
            &[
                "split_inner_dim_f32",
                "split_inner_dim_bf16",
                "concat_inner_dim_f32",
                "concat_inner_dim_bf16",
                "broadcast_to_heads_f32",
                "broadcast_to_heads_bf16",
                "pad_inner_dim_f32",
                "pad_inner_dim_bf16",
            ],
        )?;
    }
    Ok(())
}

fn kernel_name(base: &str, dtype: DType) -> String {
    let suffix = match dtype {
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        _ => panic!("mla_tensor_ops: unsupported dtype {dtype:?}"),
    };
    format!("{base}_{suffix}")
}

fn launch_cfg(n: usize) -> LaunchConfig {
    let block_size: usize = 256;
    let grid_size = (n + block_size - 1) / block_size;
    LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Split a 2D tensor `[outer, dim1+dim2]` into `[outer, dim1]` + `[outer, dim2]`.
///
/// # Errors
/// Returns an error if the kernel launch fails.
pub fn split_inner_dim<T: TensorDType + DeviceRepr + ValidAsZeroBits>(
    tensor: &CudaTensor<T>,
    dim1: usize,
    dim2: usize,
) -> Result<(CudaTensor<T>, CudaTensor<T>)> {
    let shape = tensor.shape();
    assert_eq!(shape.len(), 2, "split_inner_dim: expected 2D tensor");
    let outer = shape[0];
    let total = shape[1];
    assert_eq!(total, dim1 + dim2, "split_inner_dim: dim mismatch");

    let mut out_a = unsafe { CudaTensor::<T>::uninit(tensor.context(), &[outer, dim1])? };
    let mut out_b = unsafe { CudaTensor::<T>::uninit(tensor.context(), &[outer, dim2])? };

    let device = tensor.context().device();
    ensure_mla_kernel(device)?;

    let func = device
        .get_func(MODULE, &kernel_name("split_inner_dim", T::DTYPE))
        .unwrap();

    let n = outer * total;
    unsafe {
        func.launch(
            launch_cfg(n),
            (
                out_a.cuda_slice_mut(),
                out_b.cuda_slice_mut(),
                &tensor.cuda_slice(),
                outer as i32,
                dim1 as i32,
                dim2 as i32,
            ),
        )?;
    }

    Ok((out_a, out_b))
}

/// Concatenate two 2D tensors along the inner dim:
/// `[outer, dim1]` + `[outer, dim2]` → `[outer, dim1+dim2]`.
///
/// # Errors
/// Returns an error if the kernel launch fails.
pub fn concat_inner_dim<T: TensorDType + DeviceRepr + ValidAsZeroBits>(
    a: &CudaTensor<T>,
    b: &CudaTensor<T>,
) -> Result<CudaTensor<T>> {
    let shape_a = a.shape();
    let shape_b = b.shape();
    assert_eq!(shape_a.len(), 2, "concat_inner_dim: expected 2D tensor (a)");
    assert_eq!(shape_b.len(), 2, "concat_inner_dim: expected 2D tensor (b)");
    let outer = shape_a[0];
    assert_eq!(outer, shape_b[0], "concat_inner_dim: outer dim mismatch");
    let dim1 = shape_a[1];
    let dim2 = shape_b[1];
    let total = dim1 + dim2;

    let mut output = unsafe { CudaTensor::<T>::uninit(a.context(), &[outer, total])? };

    let device = a.context().device();
    ensure_mla_kernel(device)?;

    let func = device
        .get_func(MODULE, &kernel_name("concat_inner_dim", T::DTYPE))
        .unwrap();

    let n = outer * total;
    unsafe {
        func.launch(
            launch_cfg(n),
            (
                output.cuda_slice_mut(),
                &a.cuda_slice(),
                &b.cuda_slice(),
                outer as i32,
                dim1 as i32,
                dim2 as i32,
            ),
        )?;
    }

    Ok(output)
}

/// Broadcast `[seq, 1, dim]` → `[seq, num_heads, dim]` by repeating each
/// row of `dim` elements across all heads.
///
/// The input is treated as `[seq * 1 * dim]` flat data where each `dim`-element
/// block is copied `num_heads` times per sequence position.
///
/// # Errors
/// Returns an error if the kernel launch fails.
pub fn broadcast_to_heads<T: TensorDType + DeviceRepr + ValidAsZeroBits>(
    tensor: &CudaTensor<T>,
    num_heads: usize,
) -> Result<CudaTensor<T>> {
    let shape = tensor.shape();
    assert_eq!(shape.len(), 3, "broadcast_to_heads: expected 3D tensor");
    assert_eq!(shape[1], 1, "broadcast_to_heads: expected 1 head in input");
    let seq_len = shape[0];
    let dim = shape[2];

    let mut output =
        unsafe { CudaTensor::<T>::uninit(tensor.context(), &[seq_len, num_heads, dim])? };

    let device = tensor.context().device();
    ensure_mla_kernel(device)?;

    let func = device
        .get_func(MODULE, &kernel_name("broadcast_to_heads", T::DTYPE))
        .unwrap();

    // The kernel broadcasts [dim] → [num_heads * dim].
    // For multi-seq, we launch once per sequence position.
    // Actually the kernel signature is (output, input, num_heads, dim) with
    // total = num_heads * dim threads. For seq > 1 we need to handle each
    // seq position. Since the kernel works on flat [dim] → [num_heads * dim],
    // we can treat it as outer=seq with each position being independent.
    // Launch seq separate kernels or flatten. For simplicity, loop over seq.
    for s in 0..seq_len {
        let src_offset = s * dim;
        let dst_offset = s * num_heads * dim;
        let n = num_heads * dim;

        let src_slice = tensor.cuda_slice().slice(src_offset..src_offset + dim);
        let dst_slice = &mut output
            .cuda_slice_mut()
            .slice_mut(dst_offset..dst_offset + n);

        unsafe {
            func.clone().launch(
                launch_cfg(n),
                (dst_slice, &src_slice, num_heads as i32, dim as i32),
            )?;
        }
    }

    Ok(output)
}

/// Pad inner dimension with zeros: `[outer, src_dim]` → `[outer, dst_dim]`.
///
/// The first `src_dim` elements of each row are copied; the rest are zero-filled.
///
/// # Errors
/// Returns an error if the kernel launch fails.
pub fn pad_inner_dim<T: TensorDType + DeviceRepr + ValidAsZeroBits>(
    tensor: &CudaTensor<T>,
    dst_dim: usize,
) -> Result<CudaTensor<T>> {
    let shape = tensor.shape();
    assert_eq!(shape.len(), 2, "pad_inner_dim: expected 2D tensor");
    let outer = shape[0];
    let src_dim = shape[1];
    assert!(
        dst_dim >= src_dim,
        "pad_inner_dim: dst_dim ({dst_dim}) < src_dim ({src_dim})"
    );

    if dst_dim == src_dim {
        return Ok(tensor.slice_view(0, tensor.shape()));
    }

    let mut output = unsafe { CudaTensor::<T>::uninit(tensor.context(), &[outer, dst_dim])? };

    let device = tensor.context().device();
    ensure_mla_kernel(device)?;

    let func = device
        .get_func(MODULE, &kernel_name("pad_inner_dim", T::DTYPE))
        .unwrap();

    let n = outer * dst_dim;
    unsafe {
        func.launch(
            launch_cfg(n),
            (
                output.cuda_slice_mut(),
                &tensor.cuda_slice(),
                outer as i32,
                src_dim as i32,
                dst_dim as i32,
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
    fn test_split_inner_dim() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        // [2, 5] → [2, 3] + [2, 2]
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, // row 0
            6.0, 7.0, 8.0, 9.0, 10.0, // row 1
        ];
        let tensor = CudaTensor::from_slice(&ctx, &[2, 5], &data).unwrap();
        let (a, b) = split_inner_dim(&tensor, 3, 2).unwrap();

        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(b.shape(), &[2, 2]);
        assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 6.0, 7.0, 8.0]);
        assert_eq!(b.to_vec().unwrap(), vec![4.0, 5.0, 9.0, 10.0]);
    }

    #[test]
    fn test_concat_inner_dim() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 6.0, 7.0, 8.0];
        let b_data: Vec<f32> = vec![4.0, 5.0, 9.0, 10.0];
        let a = CudaTensor::from_slice(&ctx, &[2, 3], &a_data).unwrap();
        let b = CudaTensor::from_slice(&ctx, &[2, 2], &b_data).unwrap();
        let out = concat_inner_dim(&a, &b).unwrap();

        assert_eq!(out.shape(), &[2, 5]);
        assert_eq!(
            out.to_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        );
    }

    #[test]
    fn test_broadcast_to_heads() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        // [2, 1, 3] → [2, 4, 3]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = CudaTensor::from_slice(&ctx, &[2, 1, 3], &data).unwrap();
        let out = broadcast_to_heads(&tensor, 4).unwrap();

        assert_eq!(out.shape(), &[2, 4, 3]);
        let result = out.to_vec().unwrap();
        // seq=0: [1,2,3] repeated 4 times
        for h in 0..4 {
            assert_eq!(&result[h * 3..(h + 1) * 3], &[1.0, 2.0, 3.0]);
        }
        // seq=1: [4,5,6] repeated 4 times
        for h in 0..4 {
            let off = 12 + h * 3;
            assert_eq!(&result[off..off + 3], &[4.0, 5.0, 6.0]);
        }
    }

    #[test]
    fn test_pad_inner_dim() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        // [2, 3] → [2, 5] with zero padding
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = CudaTensor::from_slice(&ctx, &[2, 3], &data).unwrap();
        let out = pad_inner_dim(&tensor, 5).unwrap();

        assert_eq!(out.shape(), &[2, 5]);
        assert_eq!(
            out.to_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_pad_inner_dim_noop() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = CudaTensor::from_slice(&ctx, &[2, 3], &data).unwrap();
        let out = pad_inner_dim(&tensor, 3).unwrap();

        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.to_vec().unwrap(), data);
    }

    #[test]
    fn test_split_concat_roundtrip() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let data: Vec<f32> = (0..30).map(|i| i as f32).collect();
        let tensor = CudaTensor::from_slice(&ctx, &[5, 6], &data).unwrap();

        let (a, b) = split_inner_dim(&tensor, 4, 2).unwrap();
        let restored = concat_inner_dim(&a, &b).unwrap();

        assert_eq!(restored.shape(), &[5, 6]);
        assert_eq!(restored.to_vec().unwrap(), data);
    }
}
