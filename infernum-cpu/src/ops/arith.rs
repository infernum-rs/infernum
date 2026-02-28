//! ArithOps implementation for CpuBackend.

use infernum::backend::ArithOps;
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
}
