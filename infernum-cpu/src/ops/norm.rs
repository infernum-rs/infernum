//! NormOps implementation for CpuBackend.

use infernum::backend::NormOps;
use infernum::tensor::Tensor;
use infernum::Result;

use crate::simd;
use crate::tensor::CpuTensor;
use crate::CpuBackend;

impl NormOps for CpuBackend {
    fn rms_norm(input: &CpuTensor, weight: &CpuTensor, eps: f32) -> Result<CpuTensor> {
        let input_data = input.as_f32_slice();
        let weight_data = weight.as_f32_slice();
        let hidden_size = weight_data.len();
        let num_rows = input_data.len() / hidden_size;

        let mut out = vec![0.0f32; input_data.len()];
        for row in 0..num_rows {
            let start = row * hidden_size;
            let row_in = &input_data[start..start + hidden_size];
            let row_out = &mut out[start..start + hidden_size];
            simd::vec_rmsnorm(row_in, weight_data, eps, row_out);
        }
        Ok(CpuTensor::from_f32(input.shape(), &out))
    }

    fn rms_norm_inplace(input: &mut CpuTensor, weight: &CpuTensor, eps: f32) -> Result<()> {
        let weight_data = weight.as_f32_slice().to_vec();
        let hidden_size = weight_data.len();
        let data = input.as_f32_slice_mut();
        let num_rows = data.len() / hidden_size;

        // Need a temp buffer because we read and write the same slice
        let mut row_buf = vec![0.0f32; hidden_size];
        for row in 0..num_rows {
            let start = row * hidden_size;
            let row_data = &data[start..start + hidden_size];
            simd::vec_rmsnorm(row_data, &weight_data, eps, &mut row_buf);
            data[start..start + hidden_size].copy_from_slice(&row_buf);
        }
        Ok(())
    }

    fn add_rmsnorm(
        residual: &CpuTensor,
        input: &CpuTensor,
        weight: &CpuTensor,
        eps: f32,
    ) -> Result<(CpuTensor, CpuTensor)> {
        // updated_residual = residual + input
        let res_data = residual.as_f32_slice();
        let inp_data = input.as_f32_slice();
        let weight_data = weight.as_f32_slice();
        let hidden_size = weight_data.len();
        let n = res_data.len();
        let num_rows = n / hidden_size;

        let mut updated = vec![0.0f32; n];
        simd::vec_add(res_data, inp_data, &mut updated);

        // normalized = rms_norm(updated_residual)
        let mut normed = vec![0.0f32; n];
        for row in 0..num_rows {
            let start = row * hidden_size;
            let row_in = &updated[start..start + hidden_size];
            let row_out = &mut normed[start..start + hidden_size];
            simd::vec_rmsnorm(row_in, weight_data, eps, row_out);
        }

        Ok((
            CpuTensor::from_f32(residual.shape(), &updated),
            CpuTensor::from_f32(residual.shape(), &normed),
        ))
    }
}
