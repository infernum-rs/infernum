//! ArithOps and ArgmaxLastOps implementations for CpuBackend.

use infernum::backend::{ArgmaxLastOps, ArithOps};
use infernum::tensor::Tensor;
use infernum::Result;

use crate::simd;
use crate::tensor::CpuTensor;
use crate::CpuBackend;

impl ArithOps for CpuBackend {
    fn add(a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        let a_data = a.as_f32_slice();
        let b_data = b.as_f32_slice();
        assert_eq!(a_data.len(), b_data.len(), "add: length mismatch");
        let mut out = vec![0.0f32; a_data.len()];
        simd::vec_add(a_data, b_data, &mut out);
        Ok(CpuTensor::from_f32(a.shape(), &out))
    }

    fn add_inplace(a: &mut CpuTensor, b: &CpuTensor) -> Result<()> {
        let b_data = b.as_f32_slice().to_vec();
        let a_data = a.as_f32_slice_mut();
        assert_eq!(a_data.len(), b_data.len(), "add_inplace: length mismatch");
        simd::vec_add_inplace(a_data, &b_data);
        Ok(())
    }

    fn mul(a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        let a_data = a.as_f32_slice();
        let b_data = b.as_f32_slice();
        assert_eq!(a_data.len(), b_data.len(), "mul: length mismatch");
        let mut out = vec![0.0f32; a_data.len()];
        simd::vec_mul(a_data, b_data, &mut out);
        Ok(CpuTensor::from_f32(a.shape(), &out))
    }

    fn scale_inplace(a: &mut CpuTensor, scale: f32) -> Result<()> {
        let a_data = a.as_f32_slice_mut();
        simd::vec_scale(a_data, scale);
        Ok(())
    }

    fn silu(input: &CpuTensor) -> Result<CpuTensor> {
        let data = input.as_f32_slice();
        let out: Vec<f32> = data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
        Ok(CpuTensor::from_f32(input.shape(), &out))
    }

    fn logit_softcap(input: &CpuTensor, cap: f32) -> Result<CpuTensor> {
        let data = input.as_f32_slice();
        let out: Vec<f32> = data.iter().map(|&x| (x / cap).tanh() * cap).collect();
        Ok(CpuTensor::from_f32(input.shape(), &out))
    }
}

impl ArgmaxLastOps for CpuBackend {
    /// Argmax over the last dimension of a 2-D F32 tensor.
    ///
    /// Input shape `[rows, cols]`; output shape `[rows]` with dtype `U32`.
    fn argmax_last_tensor(input: &CpuTensor) -> Result<CpuTensor> {
        let shape = input.shape();
        assert_eq!(shape.len(), 2, "argmax_last_tensor: expected 2-D tensor");
        let rows = shape[0];
        let cols = shape[1];
        let data = input.as_f32_slice();
        let indices: Vec<u32> = (0..rows)
            .map(|r| {
                let row = &data[r * cols..(r + 1) * cols];
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            })
            .collect();
        Ok(CpuTensor::from_u32(&[rows], &indices))
    }
}
