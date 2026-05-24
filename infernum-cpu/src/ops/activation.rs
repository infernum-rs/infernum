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
        let n = gate_data.len();
        // SAFETY: every element is written by vec_silu_mul before it is read.
        #[allow(clippy::uninit_vec)]
        let mut out: Vec<f32> = {
            let mut v = Vec::with_capacity(n);
            unsafe { v.set_len(n) };
            v
        };

        let pool = crate::thread_pool::global_pool();
        let num_threads = pool.num_threads();
        // For decode (n ≈ 2560) the dispatch overhead exceeds the compute; only
        // parallelize for prefill-sized tensors where n is large enough to amortize.
        const MIN_PARALLEL: usize = 32_768;
        if n < MIN_PARALLEL {
            simd::vec_silu_mul(gate_data, up_data, &mut out);
        } else {
            let chunk = n.div_ceil(num_threads);
            let out_addr = out.as_mut_ptr() as usize;
            pool.dispatch(num_threads, |task_id, _| {
                let start = task_id * chunk;
                if start < n {
                    let end = (start + chunk).min(n);
                    let out_slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            (out_addr as *mut f32).add(start),
                            end - start,
                        )
                    };
                    simd::vec_silu_mul(&gate_data[start..end], &up_data[start..end], out_slice);
                }
            });
        }

        Ok(CpuTensor::from_f32_vec(gate.shape(), out))
    }
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
        let n = gate_data.len();
        // SAFETY: every element is written by vec_gelu_mul before it is read.
        #[allow(clippy::uninit_vec)]
        let mut out: Vec<f32> = {
            let mut v = Vec::with_capacity(n);
            unsafe { v.set_len(n) };
            v
        };

        let pool = crate::thread_pool::global_pool();
        let num_threads = pool.num_threads();
        const MIN_PARALLEL: usize = 32_768;
        if n < MIN_PARALLEL {
            simd::vec_gelu_mul(gate_data, up_data, &mut out);
        } else {
            let chunk = n.div_ceil(num_threads);
            let out_addr = out.as_mut_ptr() as usize;
            pool.dispatch(num_threads, |task_id, _| {
                let start = task_id * chunk;
                if start < n {
                    let end = (start + chunk).min(n);
                    let out_slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            (out_addr as *mut f32).add(start),
                            end - start,
                        )
                    };
                    simd::vec_gelu_mul(&gate_data[start..end], &up_data[start..end], out_slice);
                }
            });
        }

        Ok(CpuTensor::from_f32_vec(gate.shape(), out))
    }
}
