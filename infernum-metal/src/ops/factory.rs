//! TensorFactory and DecodeBufferOps implementations for Metal.

use infernum::backend::{DecodeBufferOps, TensorFactory};
use infernum::DType;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;
use crate::MetalContext;

impl TensorFactory for MetalBackend {
    fn from_f32_slice(
        device: &MetalContext,
        shape: &[usize],
        data: &[f32],
    ) -> Result<MetalTensor> {
        Ok(MetalTensor::from_f32(device.device(), shape, data))
    }

    fn from_raw_bytes(
        device: &MetalContext,
        shape: &[usize],
        dtype: DType,
        data: &[u8],
    ) -> Result<MetalTensor> {
        Ok(MetalTensor::from_raw_bytes(device.device(), shape, dtype, data))
    }

    fn from_u32_slice(
        device: &MetalContext,
        shape: &[usize],
        data: &[u32],
    ) -> Result<MetalTensor> {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        Ok(MetalTensor::from_raw_bytes(device.device(), shape, DType::U32, bytes))
    }

    fn from_i32_slice(
        device: &MetalContext,
        shape: &[usize],
        data: &[i32],
    ) -> Result<MetalTensor> {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        Ok(MetalTensor::from_raw_bytes(device.device(), shape, DType::U32, bytes))
    }
}

/// Use the default `DecodeBufferOps` implementation which allocates fresh
/// tensors each call via `TensorFactory`. No CUDA-graph-style pre-allocation
/// is needed on Metal.
impl DecodeBufferOps for MetalBackend {}
