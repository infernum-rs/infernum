//! ArithOps implementation for Metal — element-wise arithmetic.
//!
//! Phase 1: CPU-side via unified memory. Metal compute kernels can be
//! swapped in later for throughput.

use infernum::backend::ArithOps;
use infernum::tensor::Tensor;
use infernum::DType;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;

impl ArithOps for MetalBackend {
    fn add(a: &MetalTensor, b: &MetalTensor) -> Result<MetalTensor> {
        let a_f32 = read_f32(a)?;
        let b_f32 = read_f32(b)?;
        assert_eq!(a_f32.len(), b_f32.len(), "add: length mismatch");
        let out: Vec<f32> = a_f32.iter().zip(b_f32.iter()).map(|(x, y)| x + y).collect();
        let device = metal::Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
        Ok(MetalTensor::from_f32(&device, a.shape(), &out))
    }

    fn add_inplace(a: &mut MetalTensor, b: &MetalTensor) -> Result<()> {
        let result = Self::add(a, b)?;
        *a = result;
        Ok(())
    }

    fn mul(a: &MetalTensor, b: &MetalTensor) -> Result<MetalTensor> {
        let a_f32 = read_f32(a)?;
        let b_f32 = read_f32(b)?;
        assert_eq!(a_f32.len(), b_f32.len(), "mul: length mismatch");
        let out: Vec<f32> = a_f32.iter().zip(b_f32.iter()).map(|(x, y)| x * y).collect();
        let device = metal::Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
        Ok(MetalTensor::from_f32(&device, a.shape(), &out))
    }

    fn scale_inplace(a: &mut MetalTensor, scale: f32) -> Result<()> {
        let a_f32 = read_f32(a)?;
        let out: Vec<f32> = a_f32.iter().map(|x| x * scale).collect();
        let device = metal::Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
        *a = MetalTensor::from_f32(&device, a.shape(), &out);
        Ok(())
    }
}

/// Helper: read tensor data as f32 regardless of stored dtype.
fn read_f32(t: &MetalTensor) -> Result<Vec<f32>> {
    let bytes = t.as_bytes();
    match t.dtype() {
        DType::F32 => Ok(bytemuck::cast_slice(bytes).to_vec()),
        DType::BF16 => {
            let bf16s: &[half::bf16] = bytemuck::cast_slice(bytes);
            Ok(bf16s.iter().map(|v| v.to_f32()).collect())
        }
        DType::F16 => {
            let f16s: &[half::f16] = bytemuck::cast_slice(bytes);
            Ok(f16s.iter().map(|v| v.to_f32()).collect())
        }
        other => Err(infernum::Error::UnsupportedDtype(format!(
            "arith read_f32: unsupported dtype {other}"
        ))),
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
    fn test_add() {
        let c = ctx();
        let a = MetalBackend::from_f32_slice(&c, &[3], &[1.0, 2.0, 3.0]).unwrap();
        let b = MetalBackend::from_f32_slice(&c, &[3], &[4.0, 5.0, 6.0]).unwrap();
        let out = MetalBackend::add(&a, &b).unwrap();
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        assert_eq!(result, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_inplace() {
        let c = ctx();
        let mut a = MetalBackend::from_f32_slice(&c, &[3], &[1.0, 2.0, 3.0]).unwrap();
        let b = MetalBackend::from_f32_slice(&c, &[3], &[10.0, 20.0, 30.0]).unwrap();
        MetalBackend::add_inplace(&mut a, &b).unwrap();
        let result = MetalBackend::to_f32_vec(&a).unwrap();
        assert_eq!(result, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_mul() {
        let c = ctx();
        let a = MetalBackend::from_f32_slice(&c, &[3], &[2.0, 3.0, 4.0]).unwrap();
        let b = MetalBackend::from_f32_slice(&c, &[3], &[5.0, 6.0, 7.0]).unwrap();
        let out = MetalBackend::mul(&a, &b).unwrap();
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        assert_eq!(result, [10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_scale_inplace() {
        let c = ctx();
        let mut a = MetalBackend::from_f32_slice(&c, &[3], &[2.0, 4.0, 6.0]).unwrap();
        MetalBackend::scale_inplace(&mut a, 0.5).unwrap();
        let result = MetalBackend::to_f32_vec(&a).unwrap();
        assert_eq!(result, [1.0, 2.0, 3.0]);
    }
}
