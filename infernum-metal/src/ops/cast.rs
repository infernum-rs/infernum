//! CastOps implementation for Metal — dtype casting.
//!
//! Phase 1: CPU-side casting via the shared buffer pointer.
//! This is viable on Apple Silicon thanks to unified memory — the data
//! doesn't move, we just reinterpret it. A Metal kernel can be added
//! later for throughput.

use infernum::backend::CastOps;
use infernum::tensor::Tensor;
use infernum::DType;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;

impl CastOps for MetalBackend {
    fn cast_to_f32(input: &MetalTensor) -> Result<MetalTensor> {
        let bytes = input.as_bytes();
        let f32_data: Vec<f32> = match input.dtype() {
            DType::F32 => return Ok(input.clone()),
            DType::BF16 => {
                let bf16s: &[half::bf16] = bytemuck::cast_slice(bytes);
                bf16s.iter().map(|v| v.to_f32()).collect()
            }
            DType::F16 => {
                let f16s: &[half::f16] = bytemuck::cast_slice(bytes);
                f16s.iter().map(|v| v.to_f32()).collect()
            }
            other => {
                return Err(infernum::Error::UnsupportedDtype(format!(
                    "cast_to_f32: unsupported dtype {other}"
                )));
            }
        };

        let device = metal::Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
        Ok(MetalTensor::from_f32(&device, input.shape(), &f32_data))
    }

    fn cast_from_f32(input: &MetalTensor, _target: DType) -> Result<MetalTensor> {
        // Phase 1: all Metal ops work in F32, so we always keep tensors as F32.
        // This prevents BF16/F16 tensors (e.g. RoPE caches) from propagating
        // through the pipeline and hitting as_f32_slice() panics.
        Ok(input.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalContext;
    use infernum::backend::{TensorDataOps, TensorFactory};

    fn ctx() -> MetalContext {
        MetalContext::new()
    }

    #[test]
    fn test_from_f32_roundtrip() {
        let c = ctx();
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = MetalBackend::from_f32_slice(&c, &[2, 3], &data).unwrap();
        let out = MetalBackend::to_f32_vec(&t).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_from_raw_bytes_f32() {
        let c = ctx();
        let data = [1.5f32, -2.5, 0.0];
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        let t = MetalBackend::from_raw_bytes(&c, &[3], DType::F32, bytes).unwrap();
        let out = MetalBackend::to_f32_vec(&t).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_from_raw_bytes_bf16() {
        let c = ctx();
        let f32_data = [1.0f32, 2.0, 3.0];
        let bf16_data: Vec<half::bf16> =
            f32_data.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&bf16_data);
        let t = MetalBackend::from_raw_bytes(&c, &[3], DType::BF16, bytes).unwrap();
        let out = MetalBackend::to_f32_vec(&t).unwrap();
        for (a, b) in out.iter().zip(f32_data.iter()) {
            assert!((a - b).abs() < 0.01, "bf16 roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn test_from_u32_slice() {
        let c = ctx();
        let data = [10u32, 20, 30];
        let t = MetalBackend::from_u32_slice(&c, &[3], &data).unwrap();
        let bytes = MetalBackend::to_raw_bytes(&t).unwrap();
        let out: &[u32] = bytemuck::cast_slice(&bytes);
        assert_eq!(out, &data);
    }

    #[test]
    fn test_from_i32_slice() {
        let c = ctx();
        let data = [-1i32, 0, 42];
        let t = MetalBackend::from_i32_slice(&c, &[3], &data).unwrap();
        let bytes = MetalBackend::to_raw_bytes(&t).unwrap();
        let out: &[i32] = bytemuck::cast_slice(&bytes);
        assert_eq!(out, &data);
    }

    #[test]
    fn test_shape_and_dtype() {
        let c = ctx();
        let t = MetalBackend::from_f32_slice(&c, &[2, 3], &[0.0; 6]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.ndim(), 2);
    }

    #[test]
    fn test_reshape() {
        let c = ctx();
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = MetalBackend::from_f32_slice(&c, &[2, 3], &data).unwrap();
        let t2 = t.reshape(&[3, 2]);
        assert_eq!(t2.shape(), &[3, 2]);
        let out = MetalBackend::to_f32_vec(&t2).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_slice_view() {
        let c = ctx();
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = MetalBackend::from_f32_slice(&c, &[6], &data).unwrap();
        let view = t.slice_view(2, &[3]);
        assert_eq!(view.shape(), &[3]);
        let out = MetalBackend::to_f32_vec(&view).unwrap();
        assert_eq!(out, &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_cast_from_f32_stays_f32() {
        // Phase 1: cast_from_f32 always returns F32 regardless of target.
        let c = ctx();
        let data = [1.0f32, 2.5, -3.0, 0.0];
        let t = MetalBackend::from_f32_slice(&c, &[4], &data).unwrap();
        let result = MetalBackend::cast_from_f32(&t, DType::BF16).unwrap();
        assert_eq!(result.dtype(), DType::F32);
        let out = MetalBackend::to_f32_vec(&result).unwrap();
        assert_eq!(out, data);
    }
}
