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
        let out = MetalTensor::zeros(ctx, a.shape(), a.dtype());
        let kernel = if a.dtype() == DType::F16 {
            "add_f16"
        } else {
            "add_f32"
        };
        ctx.dispatch_1d(
            kernel,
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
        let kernel = if a.dtype() == DType::F16 {
            "add_inplace_f16"
        } else {
            "add_inplace_f32"
        };
        ctx.dispatch_1d(
            kernel,
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
        let out = MetalTensor::zeros(ctx, a.shape(), a.dtype());
        let kernel = if a.dtype() == DType::F16 {
            "mul_f16"
        } else {
            "mul_f32"
        };
        ctx.dispatch_1d(
            kernel,
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
        let kernel = if a.dtype() == DType::F16 {
            "scale_inplace_f16"
        } else {
            "scale_inplace_f32"
        };
        ctx.dispatch_1d(
            kernel,
            &[(a.metal_buffer(), a.buffer_offset())],
            bytemuck::bytes_of(&scale),
            n,
        );
        Ok(())
    }

    fn silu(input: &MetalTensor) -> Result<MetalTensor> {
        let n = input.numel();
        let ctx = input.context();
        let out = MetalTensor::zeros(ctx, input.shape(), DType::F32);
        ctx.dispatch_1d(
            "silu_f32",
            &[
                (input.metal_buffer(), input.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            &[],
            n,
        );
        Ok(out)
    }

    fn logit_softcap(input: &MetalTensor, cap: f32) -> Result<MetalTensor> {
        let n = input.numel();
        let ctx = input.context();
        let out = MetalTensor::zeros(ctx, input.shape(), DType::F32);
        ctx.dispatch_1d(
            "logit_softcap_f32",
            &[
                (input.metal_buffer(), input.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            bytemuck::bytes_of(&cap),
            n,
        );
        Ok(out)
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

    #[test]
    fn test_silu() {
        let c = ctx();
        let input = MetalBackend::from_f32_slice(&c, &[3], &[0.0, 1.0, -1.0]).unwrap();
        let out = MetalBackend::silu(&input).unwrap();
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        // silu(0) = 0, silu(1) ≈ 0.7311, silu(-1) ≈ -0.2689
        assert!(result[0].abs() < 1e-6);
        assert!((result[1] - 0.7311).abs() < 1e-3);
        assert!((result[2] - (-0.2689)).abs() < 1e-3);
    }

    #[test]
    fn test_logit_softcap() {
        let c = ctx();
        let input = MetalBackend::from_f32_slice(&c, &[3], &[0.0, 50.0, -50.0]).unwrap();
        let out = MetalBackend::logit_softcap(&input, 10.0).unwrap();
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        // softcap(0, 10) = tanh(0) * 10 = 0
        assert!(result[0].abs() < 1e-6);
        // softcap(50, 10) = tanh(5) * 10 ≈ 10 (saturated)
        assert!((result[1] - 10.0).abs() < 0.01);
        // softcap(-50, 10) = tanh(-5) * 10 ≈ -10
        assert!((result[2] - (-10.0)).abs() < 0.01);
    }
}
