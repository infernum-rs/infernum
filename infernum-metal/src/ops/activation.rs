//! SwigluOps and GegluOps implementations for Metal — GPU kernel dispatch.

use infernum::backend::{GegluOps, SwigluOps};
use infernum::tensor::Tensor;
use infernum::DType;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;

impl SwigluOps for MetalBackend {
    fn swiglu(gate: &MetalTensor, up: &MetalTensor) -> Result<MetalTensor> {
        let n = gate.numel();
        assert_eq!(n, up.numel(), "swiglu: gate/up length mismatch");
        let ctx = gate.context();
        let out = MetalTensor::zeros(ctx, gate.shape(), gate.dtype());
        let kernel = if gate.dtype() == DType::F16 {
            "swiglu_f16"
        } else {
            "swiglu_f32"
        };
        ctx.dispatch_1d(
            kernel,
            &[
                (gate.metal_buffer(), gate.buffer_offset()),
                (up.metal_buffer(), up.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            &[],
            n,
        );
        Ok(out)
    }
}

impl GegluOps for MetalBackend {
    fn geglu(gate: &MetalTensor, up: &MetalTensor) -> Result<MetalTensor> {
        let n = gate.numel();
        assert_eq!(n, up.numel(), "geglu: gate/up length mismatch");
        let ctx = gate.context();
        let out = MetalTensor::zeros(ctx, gate.shape(), gate.dtype());
        let kernel = if gate.dtype() == DType::F16 {
            "geglu_f16"
        } else {
            "geglu_f32"
        };
        ctx.dispatch_1d(
            kernel,
            &[
                (gate.metal_buffer(), gate.buffer_offset()),
                (up.metal_buffer(), up.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            &[],
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
    fn test_swiglu_basic() {
        let c = ctx();
        let gate = MetalBackend::from_f32_slice(&c, &[3], &[0.0, 1.0, -1.0]).unwrap();
        let up = MetalBackend::from_f32_slice(&c, &[3], &[1.0, 1.0, 1.0]).unwrap();
        let out = MetalBackend::swiglu(&gate, &up).unwrap();
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        // silu(0) = 0, silu(1) ≈ 0.7311, silu(-1) ≈ -0.2689
        assert!((result[0]).abs() < 1e-6, "silu(0) = 0");
        assert!((result[1] - 0.7311).abs() < 1e-3, "silu(1)");
        assert!((result[2] - (-0.2689)).abs() < 1e-3, "silu(-1)");
    }

    #[test]
    fn test_geglu_basic() {
        let c = ctx();
        let gate = MetalBackend::from_f32_slice(&c, &[3], &[0.0, 1.0, -1.0]).unwrap();
        let up = MetalBackend::from_f32_slice(&c, &[3], &[2.0, 2.0, 2.0]).unwrap();
        let out = MetalBackend::geglu(&gate, &up).unwrap();
        let result = MetalBackend::to_f32_vec(&out).unwrap();
        // gelu(0) = 0, gelu(1) ≈ 0.8412, gelu(-1) ≈ -0.1588
        assert!((result[0]).abs() < 1e-6, "gelu(0)*2 = 0");
        assert!((result[1] - 2.0 * 0.8412).abs() < 1e-2, "gelu(1)*2");
        assert!((result[2] - 2.0 * (-0.1588)).abs() < 1e-2, "gelu(-1)*2");
    }
}
