//! NormOps implementation for CpuBackend.

use infernum::backend::NormOps;
use infernum::tensor::Tensor;
use infernum::Result;

use crate::simd;
use crate::tensor::CpuTensor;
use crate::CpuBackend;

/// Zero-copy RMS normalisation on raw slices.
///
/// Normalises each row of `input` (length `hidden_size`) and writes to `output`.
/// For `num_rows > 1` the work is split across the global thread pool.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn rms_norm_slices(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
    hidden_size: usize,
) {
    let num_rows = input.len() / hidden_size;
    debug_assert_eq!(input.len(), num_rows * hidden_size);
    debug_assert_eq!(output.len(), num_rows * hidden_size);

    if num_rows <= 1 {
        // Decode path — single row, skip dispatch overhead.
        for row in 0..num_rows {
            let start = row * hidden_size;
            let row_in = &input[start..start + hidden_size];
            let row_out = &mut output[start..start + hidden_size];
            simd::vec_rmsnorm(row_in, weight, eps, row_out);
        }
        return;
    }

    // Prefill path — parallelize across rows.
    let pool = crate::thread_pool::global_pool();
    let num_threads = pool.num_threads();
    let rows_per_thread = num_rows.div_ceil(num_threads);
    let num_tasks = num_threads.min(num_rows);
    let out_ptr = output.as_mut_ptr() as usize;
    let inp_ptr = input.as_ptr() as usize;

    pool.dispatch(num_tasks, |task_id, _| {
        let row_start = task_id * rows_per_thread;
        let row_end = (row_start + rows_per_thread).min(num_rows);
        if row_start >= num_rows {
            return;
        }
        for row in row_start..row_end {
            let start = row * hidden_size;
            unsafe {
                let row_in =
                    std::slice::from_raw_parts((inp_ptr as *const f32).add(start), hidden_size);
                let row_out =
                    std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), hidden_size);
                simd::vec_rmsnorm(row_in, weight, eps, row_out);
            }
        }
    });
}

/// Zero-copy fused add + RMS normalisation on raw slices.
///
/// Computes `updated_out = residual + delta` and `normed_out = rmsnorm(updated_out)`.
/// For `num_rows > 1` the work is split across the global thread pool.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
pub(crate) fn add_rmsnorm_slices(
    residual: &[f32],
    delta: &[f32],
    weight: &[f32],
    eps: f32,
    updated_out: &mut [f32],
    normed_out: &mut [f32],
    hidden_size: usize,
) {
    let n = residual.len();
    let num_rows = n / hidden_size;
    debug_assert_eq!(residual.len(), num_rows * hidden_size);
    debug_assert_eq!(delta.len(), num_rows * hidden_size);
    debug_assert_eq!(updated_out.len(), num_rows * hidden_size);
    debug_assert_eq!(normed_out.len(), num_rows * hidden_size);

    if num_rows <= 1 {
        simd::vec_add(residual, delta, updated_out);
        for row in 0..num_rows {
            let start = row * hidden_size;
            let row_in = &updated_out[start..start + hidden_size];
            let row_out = &mut normed_out[start..start + hidden_size];
            simd::vec_rmsnorm(row_in, weight, eps, row_out);
        }
        return;
    }

    // Prefill path — each thread handles add + rmsnorm for its rows.
    let pool = crate::thread_pool::global_pool();
    let num_threads = pool.num_threads();
    let rows_per_thread = num_rows.div_ceil(num_threads);
    let num_tasks = num_threads.min(num_rows);
    let res_ptr = residual.as_ptr() as usize;
    let del_ptr = delta.as_ptr() as usize;
    let upd_ptr = updated_out.as_mut_ptr() as usize;
    let nrm_ptr = normed_out.as_mut_ptr() as usize;

    pool.dispatch(num_tasks, |task_id, _| {
        let row_start = task_id * rows_per_thread;
        let row_end = (row_start + rows_per_thread).min(num_rows);
        if row_start >= num_rows {
            return;
        }
        for row in row_start..row_end {
            let start = row * hidden_size;
            unsafe {
                let r = std::slice::from_raw_parts((res_ptr as *const f32).add(start), hidden_size);
                let d = std::slice::from_raw_parts((del_ptr as *const f32).add(start), hidden_size);
                let u =
                    std::slice::from_raw_parts_mut((upd_ptr as *mut f32).add(start), hidden_size);
                let o =
                    std::slice::from_raw_parts_mut((nrm_ptr as *mut f32).add(start), hidden_size);
                simd::vec_add(r, d, u);
                simd::vec_rmsnorm(u, weight, eps, o);
            }
        }
    });
}

impl NormOps for CpuBackend {
    fn rms_norm(input: &CpuTensor, weight: &CpuTensor, eps: f32) -> Result<CpuTensor> {
        let input_data = input.as_f32_slice();
        let weight_data = weight.as_f32_slice();
        let hidden_size = weight_data.len();

        let mut out = vec![0.0f32; input_data.len()];
        rms_norm_slices(input_data, weight_data, eps, &mut out, hidden_size);
        Ok(CpuTensor::from_f32_vec(input.shape(), out))
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
        let res_data = residual.as_f32_slice();
        let inp_data = input.as_f32_slice();
        let weight_data = weight.as_f32_slice();
        let hidden_size = weight_data.len();
        let n = res_data.len();

        let mut updated = vec![0.0f32; n];
        let mut normed = vec![0.0f32; n];
        add_rmsnorm_slices(
            res_data,
            inp_data,
            weight_data,
            eps,
            &mut updated,
            &mut normed,
            hidden_size,
        );

        Ok((
            CpuTensor::from_f32_vec(residual.shape(), updated),
            CpuTensor::from_f32_vec(residual.shape(), normed),
        ))
    }
}
