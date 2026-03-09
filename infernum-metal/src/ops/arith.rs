//! ArithOps implementation for Metal — element-wise arithmetic via GPU kernels.

use infernum::backend::ArithOps;
use infernum::tensor::Tensor;
use infernum::DType;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;

impl ArithOps for MetalBackend {
    fn add(a: &MetalTensor, b: &MetalTensor) -> Result<MetalTensor> {
        let n = a.numel();
        assert_eq!(n, b.numel(), "add: length mismatch");
        let ctx = a.context();
        let out = MetalTensor::zeros(ctx, a.shape(), DType::F32);
        ctx.dispatch_1d(
            "add_f32",
            &[
                (a.metal_buffer(), a.buffer_offset()),
                (b.metal_buffer(), b.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            &[],
            n,
        );
        Ok(out)
    }

    fn add_inplace(a: &mut MetalTensor, b: &MetalTensor) -> Result<()> {
        let n = a.numel();
        assert_eq!(n, b.numel(), "add_inplace: length mismatch");
        let ctx = a.context().clone();
        ctx.dispatch_1d(
            "add_inplace_f32",
            &[
                (a.metal_buffer(), a.buffer_offset()),
                (b.metal_buffer(), b.buffer_offset()),
            ],
            &[],
            n,
        );
        Ok(())
    }

    fn mul(a: &MetalTensor, b: &MetalTensor) -> Result<MetalTensor> {
        let n = a.numel();
        assert_eq!(n, b.numel(), "mul: length mismatch");
        let ctx = a.context();
        let out = MetalTensor::zeros(ctx, a.shape(), DType::F32);
        ctx.dispatch_1d(
            "mul_f32",
            &[
                (a.metal_buffer(), a.buffer_offset()),
                (b.metal_buffer(), b.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            &[],
            n,
        );
        Ok(out)
    }

    fn scale_inplace(a: &mut MetalTensor, scale: f32) -> Result<()> {
        let n = a.numel();
        let ctx = a.context().clone();
        ctx.dispatch_1d(
            "scale_inplace_f32",
            &[(a.metal_buffer(), a.buffer_offset())],
            bytemuck::bytes_of(&scale),
            n,
        );
        Ok(())
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
