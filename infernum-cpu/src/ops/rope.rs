//! RopeOps implementation for CpuBackend (half-rotation layout).

use infernum::backend::RopeOps;
use infernum::tensor::Tensor;
use infernum::Result;

use crate::tensor::CpuTensor;
use crate::CpuBackend;

/// Zero-copy RoPE on raw f32 slices.
///
/// `input`  — `[seq_len, num_heads, head_dim]` flat
/// `cos`    — `[max_seq, half_dim]` flat (only positions `[offset..offset+seq_len]` used)
/// `sin`    — same layout as `cos`
/// `output` — same size as `input`
///
/// For `seq_len * num_heads > 1` the work is split across the global thread pool.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_rope_slices(
    input: &[f32],
    cos: &[f32],
    sin: &[f32],
    output: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    offset: usize,
) {
    let half_dim = head_dim / 2;
    let total_units = seq_len * num_heads;

    if total_units <= 1 {
        // Decode path — single unit, skip dispatch overhead.
        for unit in 0..total_units {
            let s = unit / num_heads;
            let h = unit % num_heads;
            let pos = offset + s;
            let cos_row = &cos[pos * half_dim..(pos + 1) * half_dim];
            let sin_row = &sin[pos * half_dim..(pos + 1) * half_dim];
            let base = (s * num_heads + h) * head_dim;
            for d in 0..half_dim {
                let x0 = input[base + d];
                let x1 = input[base + half_dim + d];
                output[base + d] = x0 * cos_row[d] - x1 * sin_row[d];
                output[base + half_dim + d] = x1 * cos_row[d] + x0 * sin_row[d];
            }
        }
        return;
    }

    // Prefill path — parallelize across (seq_pos, head) units.
    let pool = crate::thread_pool::global_pool();
    let num_threads = pool.num_threads();
    let units_per_thread = total_units.div_ceil(num_threads);
    let num_tasks = num_threads.min(total_units);
    let out_ptr = output.as_mut_ptr() as usize;
    let inp_ptr = input.as_ptr() as usize;

    pool.dispatch(num_tasks, |task_id, _| {
        let unit_start = task_id * units_per_thread;
        let unit_end = (unit_start + units_per_thread).min(total_units);
        if unit_start >= total_units {
            return;
        }
        for unit in unit_start..unit_end {
            let s = unit / num_heads;
            let h = unit % num_heads;
            let pos = offset + s;
            let cos_row = &cos[pos * half_dim..(pos + 1) * half_dim];
            let sin_row = &sin[pos * half_dim..(pos + 1) * half_dim];
            let base = (s * num_heads + h) * head_dim;
            unsafe {
                let inp = inp_ptr as *const f32;
                let out = out_ptr as *mut f32;
                for d in 0..half_dim {
                    let x0 = *inp.add(base + d);
                    let x1 = *inp.add(base + half_dim + d);
                    *out.add(base + d) = x0 * cos_row[d] - x1 * sin_row[d];
                    *out.add(base + half_dim + d) = x1 * cos_row[d] + x0 * sin_row[d];
                }
            }
        }
    });
}

/// Descriptor for one RoPE operand (Q or K), passed to [`apply_rope_pair_slices`].
///
/// All pointer fields are cast from `*const f32` / `*mut f32` to `usize`
/// for `Send` compatibility with the thread pool dispatch closure.
pub(crate) struct RopeOperand {
    pub inp_ptr: usize,
    pub out_ptr: usize,
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub offset: usize,
}

/// Fused two-operand RoPE: apply RoPE to both Q and K in a single dispatch.
///
/// Both operands share the same `cos`/`sin` cache.  The combined work units
/// (`total_a + total_b`) are split across the thread pool so that each thread
/// runs RoPE for its range of units, branching to operand A or B as needed.
///
/// # Safety
/// The caller must ensure that the `inp_ptr`/`out_ptr` fields in each operand
/// point to valid, disjoint f32 slices of the correct length.
pub(crate) unsafe fn apply_rope_pair_slices(a: &RopeOperand, b: &RopeOperand, cos: &[f32], sin: &[f32]) {
    let total_a = a.seq_len * a.num_heads;
    let total_b = b.seq_len * b.num_heads;
    let total_units = total_a + total_b;

    if total_units == 0 {
        return;
    }

    let pool = crate::thread_pool::global_pool();
    let num_threads = pool.num_threads();
    let units_per_thread = total_units.div_ceil(num_threads);
    let num_tasks = num_threads.min(total_units);

    let half_dim_a = a.head_dim / 2;
    let half_dim_b = b.head_dim / 2;

    pool.dispatch(num_tasks, |task_id, _| {
        let unit_start = task_id * units_per_thread;
        let unit_end = (unit_start + units_per_thread).min(total_units);

        for unit in unit_start..unit_end {
            if unit < total_a {
                // Operand A (e.g., Q)
                let s = unit / a.num_heads;
                let h = unit % a.num_heads;
                let pos = a.offset + s;
                let cos_row = &cos[pos * half_dim_a..(pos + 1) * half_dim_a];
                let sin_row = &sin[pos * half_dim_a..(pos + 1) * half_dim_a];
                let base = (s * a.num_heads + h) * a.head_dim;
                let inp = a.inp_ptr as *const f32;
                let out = a.out_ptr as *mut f32;
                for d in 0..half_dim_a {
                    let x0 = *inp.add(base + d);
                    let x1 = *inp.add(base + half_dim_a + d);
                    *out.add(base + d) = x0 * cos_row[d] - x1 * sin_row[d];
                    *out.add(base + half_dim_a + d) = x1 * cos_row[d] + x0 * sin_row[d];
                }
            } else {
                // Operand B (e.g., K)
                let local = unit - total_a;
                let s = local / b.num_heads;
                let h = local % b.num_heads;
                let pos = b.offset + s;
                let cos_row = &cos[pos * half_dim_b..(pos + 1) * half_dim_b];
                let sin_row = &sin[pos * half_dim_b..(pos + 1) * half_dim_b];
                let base = (s * b.num_heads + h) * b.head_dim;
                let inp = b.inp_ptr as *const f32;
                let out = b.out_ptr as *mut f32;
                for d in 0..half_dim_b {
                    let x0 = *inp.add(base + d);
                    let x1 = *inp.add(base + half_dim_b + d);
                    *out.add(base + d) = x0 * cos_row[d] - x1 * sin_row[d];
                    *out.add(base + half_dim_b + d) = x1 * cos_row[d] + x0 * sin_row[d];
                }
            }
        }
    });
}

impl RopeOps for CpuBackend {
    fn apply_rope(
        input: &CpuTensor,
        cos_cache: &CpuTensor,
        sin_cache: &CpuTensor,
        position_offset: usize,
    ) -> Result<CpuTensor> {
        // input: (seq_len, num_heads, head_dim)
        let shape = input.shape();
        let seq_len = shape[0];
        let num_heads = shape[1];
        let head_dim = shape[2];

        let input_data = input.as_f32_slice();
        // RoPE caches may be stored in model dtype (e.g. BF16); convert if needed.
        let cos_data = cos_cache.to_f32_cow();
        let sin_data = sin_cache.to_f32_cow();

        let mut out = vec![0.0f32; input_data.len()];
        apply_rope_slices(
            input_data,
            &cos_data,
            &sin_data,
            &mut out,
            seq_len,
            num_heads,
            head_dim,
            position_offset,
        );

        Ok(CpuTensor::from_f32_vec(shape, out))
    }

    fn apply_rope_batched(
        input: &CpuTensor,
        cos_cache: &CpuTensor,
        sin_cache: &CpuTensor,
        positions: &CpuTensor,
        batch_size: usize,
    ) -> Result<CpuTensor> {
        // input: (batch_size, num_heads, head_dim)
        // positions: (batch_size,) i32
        let shape = input.shape();
        let num_heads = shape[1];
        let head_dim = shape[2];
        let half_dim = head_dim / 2;

        let input_data = input.as_f32_slice();
        // RoPE caches may be stored in model dtype (e.g. BF16); convert if needed.
        let cos_data = cos_cache.to_f32_cow();
        let sin_data = sin_cache.to_f32_cow();
        let pos_data = positions.as_i32_slice();

        let mut out = vec![0.0f32; input_data.len()];

        #[allow(clippy::needless_range_loop)]
        for b in 0..batch_size {
            #[allow(clippy::cast_sign_loss)]
            let pos = pos_data[b] as usize;
            let cos_row = &cos_data[pos * half_dim..(pos + 1) * half_dim];
            let sin_row = &sin_data[pos * half_dim..(pos + 1) * half_dim];

            for h in 0..num_heads {
                let base = (b * num_heads + h) * head_dim;
                for d in 0..half_dim {
                    let x0 = input_data[base + d];
                    let x1 = input_data[base + half_dim + d];
                    out[base + d] = x0 * cos_row[d] - x1 * sin_row[d];
                    out[base + half_dim + d] = x1 * cos_row[d] + x0 * sin_row[d];
                }
            }
        }

        Ok(CpuTensor::from_f32_vec(shape, out))
    }
}
