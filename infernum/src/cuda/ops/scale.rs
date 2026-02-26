//! In-place scalar scaling: `data[i] *= scale`

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

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/scale.ptx"));
const KERNEL_NAMES: &[&str] = &["scale_f32", "scale_f16", "scale_bf16", "scale_rows_f32"];

/// Kernel name suffix for dtype
fn kernel_suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => panic!("Unsupported dtype: {dtype:?}"),
    }
}

/// Scale a tensor in place: `data[i] *= scale`
///
/// Supports F32, F16, and BF16 tensor types. The scale factor is always f32.
///
/// # Errors
/// Returns an error if the operation fails
pub fn scale_inplace(data: &mut CudaTensor, scale: f32) -> Result<()> {
    let dtype = data.dtype();
    let n = data.numel();
    let device = data.context().device();
    let kernel_name = format!("scale_{}", kernel_suffix(dtype));

    let module_name = "scale";
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
        func.launch(cfg, (data.cuda_slice_mut(), scale, n as i32))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;

    #[test]
    fn test_scale_f32() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut tensor = CudaTensor::from_slice(&ctx, &[4], &data).unwrap();

        scale_inplace(&mut tensor, 2.5).unwrap();

        let result = tensor.to_vec::<f32>().unwrap();
        assert_eq!(result, vec![2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_scale_bf16() {
        use half::bf16;

        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let data: Vec<bf16> = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
        ];
        let mut tensor = CudaTensor::from_slice(&ctx, &[4], &data).unwrap();

        scale_inplace(&mut tensor, 2.0).unwrap();

        let result: Vec<f32> = tensor
            .to_vec::<bf16>()
            .unwrap()
            .iter()
            .map(|x| x.to_f32())
            .collect();
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_scale_identity() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut tensor = CudaTensor::from_slice(&ctx, &[3], &data).unwrap();

        scale_inplace(&mut tensor, 1.0).unwrap();

        let result = tensor.to_vec::<f32>().unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }
}
