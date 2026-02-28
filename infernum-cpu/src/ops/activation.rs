//! SwigluOps and GegluOps implementation for CpuBackend.

use infernum::backend::{GegluOps, SwigluOps};
use infernum::tensor::Tensor;
use infernum::Result;

use crate::simd;
use crate::tensor::CpuTensor;
use crate::CpuBackend;

impl SwigluOps for CpuBackend {
    fn swiglu(gate: &CpuTensor, up: &CpuTensor) -> Result<CpuTensor> {
        let gate_data = gate.as_f32_slice();
        let up_data = up.as_f32_slice();
        assert_eq!(
            gate_data.len(),
            up_data.len(),
            "swiglu: gate and up sizes differ"
        );
        let mut out = vec![0.0f32; gate_data.len()];
        simd::vec_silu_mul(gate_data, up_data, &mut out);
        Ok(CpuTensor::from_f32(gate.shape(), &out))
    }
}

/// GELU activation (approximate): 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
fn gelu_approx(x: f32) -> f32 {
    let coeff = 0.797_884_6; // sqrt(2/π)
    let inner = coeff * x.mul_add(0.044_715 * x * x, x);
    0.5 * x * (1.0 + inner.tanh())
}

impl GegluOps for CpuBackend {
    fn geglu(gate: &CpuTensor, up: &CpuTensor) -> Result<CpuTensor> {
        let gate_data = gate.as_f32_slice();
        let up_data = up.as_f32_slice();
        assert_eq!(
            gate_data.len(),
            up_data.len(),
            "geglu: gate and up sizes differ"
        );
        let mut out = vec![0.0f32; gate_data.len()];
        for i in 0..gate_data.len() {
            out[i] = gelu_approx(gate_data[i]) * up_data[i];
        }
        Ok(CpuTensor::from_f32(gate.shape(), &out))
    }
}
