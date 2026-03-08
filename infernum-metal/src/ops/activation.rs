//! SwigluOps and GegluOps implementations for Metal.
//!
//! Phase 1: CPU-side via unified memory.

use infernum::backend::{GegluOps, SwigluOps};
use infernum::tensor::Tensor;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;

/// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
fn gelu(x: f32) -> f32 {
    let c = std::f32::consts::FRAC_2_PI.sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
}

impl SwigluOps for MetalBackend {
    fn swiglu(gate: &MetalTensor, up: &MetalTensor) -> Result<MetalTensor> {
        let g = gate.as_f32_slice();
        let u = up.as_f32_slice();
        assert_eq!(g.len(), u.len(), "swiglu: gate/up length mismatch");
        let out: Vec<f32> = g.iter().zip(u.iter()).map(|(g, u)| silu(*g) * u).collect();
        Ok(MetalTensor::from_f32(gate.context(), gate.shape(), &out))
    }
}

impl GegluOps for MetalBackend {
    fn geglu(gate: &MetalTensor, up: &MetalTensor) -> Result<MetalTensor> {
        let g = gate.as_f32_slice();
        let u = up.as_f32_slice();
        assert_eq!(g.len(), u.len(), "geglu: gate/up length mismatch");
        let out: Vec<f32> = g.iter().zip(u.iter()).map(|(g, u)| gelu(*g) * u).collect();
        Ok(MetalTensor::from_f32(gate.context(), gate.shape(), &out))
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
