//! MatmulOps and MatmulExtOps implementation for CpuBackend.
//!
//! Dense matmul is f32 row-major: `A (M,K) × B (K,N) → C (M,N)`.
//! Dense weight matrices are stored as `(in_features, out_features)` after
//! the host transpose applied by `WeightLoader`.
//!
//! Performance: B is transposed to `Bᵀ(N,K)` so that `C[m,n] = dot(A[m,:], Bᵀ[n,:])`
//! becomes a contiguous SIMD dot product. M>1 uses a cache-blocked tiled GEMM
//! (AVX-512F 6×32 micro-kernel) parallelized across rows via the thread pool.
//!
//! Quantized weights (Q8_0, Q4_0) are stored in their original
//! `(out_features, in_features)` layout. The quantized linear kernel iterates
//! per output neuron (row), dequantizing blocks on the fly.

use infernum::backend::{MatmulExtOps, MatmulOps};
use infernum::dtype::{DType, QUANTIZATION_BLOCK_SIZE};
use infernum::tensor::Tensor;
use infernum::Result;

use crate::simd;
use crate::tensor::{CpuLinearWeight, CpuQuantizedWeight, CpuTensor};
use crate::CpuBackend;

/// Cast a `*mut T` to `usize` for safe capture in closures.
///
/// `usize` is `Send + Sync`, avoiding Rust 2021 closure field-capture issues
/// with raw pointers. The caller must ensure disjoint access when used across
/// threads, and that the pointer remains valid for the duration of use.
#[inline]
pub(crate) fn ptr_to_usize<T>(p: *mut T) -> usize {
    p as usize
}

/// Transpose `B (K,N)` → `Bᵀ (N,K)` in row-major order.
#[allow(clippy::many_single_char_names)]
fn transpose(b: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut bt = vec![0.0f32; n * k];
    for row in 0..k {
        for col in 0..n {
            bt[col * k + row] = b[row * n + col];
        }
    }
    bt
}

/// Standard gemm: `A (M,K) × B (K,N) → C (M,N)`.
///
/// Transposes B once, then delegates to `gemm_with_bt`.
#[allow(clippy::many_single_char_names)]
fn gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let bt = transpose(b, k, n);
    gemm_with_bt(a, &bt, m, k, n)
}

/// GEMM using a pre-transposed weight `Bᵀ (N,K)`.
///
/// `C[m,n] = dot(A[m,:], Bᵀ[n,:])` — each dot product is contiguous SIMD.
/// - M=1 (GEMV / decode): parallel over output columns (N independent dot products).
/// - M>1 (prefill): parallel over output rows.
#[allow(clippy::many_single_char_names)]
fn gemm_with_bt(a: &[f32], bt: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    if m == 1 {
        // GEMV: parallel over chunks of output columns.
        let a_row = &a[..k];
        let pool = crate::thread_pool::global_pool();
        let num_threads = pool.num_threads();
        let chunk_size = (n / num_threads).max(64).min(n);
        let c_addr = ptr_to_usize(c.as_mut_ptr());
        let num_tasks = num_threads.min(n.div_ceil(chunk_size));
        pool.dispatch(num_tasks, |task_id, _| {
            let col_start = task_id * chunk_size;
            let col_end = (col_start + chunk_size).min(n);
            if col_start >= n {
                return;
            }
            let c_chunk = unsafe {
                std::slice::from_raw_parts_mut(
                    (c_addr as *mut f32).add(col_start),
                    col_end - col_start,
                )
            };
            for (i, out) in c_chunk.iter_mut().enumerate() {
                let col = col_start + i;
                let bt_row = &bt[col * k..(col + 1) * k];
                *out = simd::dot_f32(a_row, bt_row);
            }
        });
    } else {
        // Prefill: split M into chunks across threads, each calls tiled GEMM
        // (which internally packs Bᵀ panels per thread for cache locality).
        let pool = crate::thread_pool::global_pool();
        let num_threads = pool.num_threads();
        let rows_per_thread = m.div_ceil(num_threads);
        let num_tasks = num_threads.min(m.div_ceil(rows_per_thread));
        let c_addr = ptr_to_usize(c.as_mut_ptr());

        pool.dispatch(num_tasks, |task_id, _| {
            let row_start = task_id * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(m);
            if row_start >= m {
                return;
            }
            let chunk_m = row_end - row_start;
            let a_chunk = &a[row_start * k..row_end * k];
            let c_chunk = unsafe {
                std::slice::from_raw_parts_mut((c_addr as *mut f32).add(row_start * n), chunk_m * n)
            };
            simd::gemm_f32_tiled(a_chunk, bt, c_chunk, chunk_m, k, n);
        });
    }

    c
}

/// Parallel GEMV for Q8_0 using integer dot product (Q8×Q8).
///
/// Input is pre-quantized to Q8 by the caller. Uses vpdpbusd (VNNI) or
/// vpmaddubsw+vpmaddwd (AVX2) depending on CPU, keeping computation in
/// the integer domain until the final per-block scale multiply.
///
/// Processes neurons in pairs using 2-row kernels to halve input memory reads.
#[allow(clippy::too_many_arguments)]
fn q8_gemv_parallel(
    out: &mut [f32],
    inp_quants: &[u8],
    inp_scales: &[f32],
    weight_quants: &[u8],
    weight_scales: &[f32],
    num_blocks_per_row: usize,
    quant_bytes_per_row: usize,
) {
    let n = out.len();

    let gemv_body = |chunk: &mut [f32], neuron_offset: usize| {
        let chunk_len = chunk.len();
        let pairs = chunk_len / 2;
        let remainder = chunk_len % 2;

        for p in 0..pairs {
            let n0 = neuron_offset + p * 2;
            let n1 = n0 + 1;
            let q0 = n0 * quant_bytes_per_row;
            let q1 = n1 * quant_bytes_per_row;
            let s0 = n0 * num_blocks_per_row;
            let s1 = n1 * num_blocks_per_row;
            let (d0, d1) = simd::dot_q8_q8_2row(
                inp_quants,
                inp_scales,
                &weight_quants[q0..q0 + quant_bytes_per_row],
                &weight_scales[s0..s0 + num_blocks_per_row],
                &weight_quants[q1..q1 + quant_bytes_per_row],
                &weight_scales[s1..s1 + num_blocks_per_row],
            );
            chunk[p * 2] = d0;
            chunk[p * 2 + 1] = d1;
        }

        if remainder > 0 {
            let neuron = neuron_offset + pairs * 2;
            let q_row_start = neuron * quant_bytes_per_row;
            let s_row_start = neuron * num_blocks_per_row;
            chunk[pairs * 2] = simd::dot_q8_q8_row(
                inp_quants,
                inp_scales,
                &weight_quants[q_row_start..q_row_start + quant_bytes_per_row],
                &weight_scales[s_row_start..s_row_start + num_blocks_per_row],
            );
        }
    };

    // Only parallelize when there's enough work to amortize dispatch overhead.
    let pool = crate::thread_pool::global_pool();
    let num_threads = pool.num_threads();
    let min_neurons_per_task = 64;
    if n < num_threads * min_neurons_per_task {
        gemv_body(out, 0);
    } else {
        // Round chunk_size to even for clean pair processing
        let raw_chunk = (n / num_threads).max(min_neurons_per_task).min(n);
        let chunk_size = (raw_chunk + 1) & !1; // round up to even
        let out_addr = ptr_to_usize(out.as_mut_ptr());
        // SAFETY: each task writes to a disjoint slice [start..end].
        // dispatch() blocks until all tasks complete, so out_addr is valid.
        pool.dispatch(num_threads.min(n.div_ceil(chunk_size)), |task_id, _| {
            let start = task_id * chunk_size;
            let end = (start + chunk_size).min(n);
            if start >= n {
                return;
            }
            let chunk = unsafe {
                std::slice::from_raw_parts_mut((out_addr as *mut f32).add(start), end - start)
            };
            gemv_body(chunk, start);
        });
    }
}

/// Parallel GEMV for Q4_0 using integer dot product (Q4×Q8).
///
/// Processes neurons in pairs using 2-row kernels to halve input memory reads.
#[allow(clippy::too_many_arguments)]
fn q4_gemv_parallel(
    out: &mut [f32],
    inp_quants: &[u8],
    inp_scales: &[f32],
    weight_packed: &[u8],
    weight_scales: &[f32],
    num_blocks_per_row: usize,
    packed_bytes_per_row: usize,
) {
    let n = out.len();

    let gemv_body = |chunk: &mut [f32], neuron_offset: usize| {
        let chunk_len = chunk.len();
        let pairs = chunk_len / 2;
        let remainder = chunk_len % 2;

        for p in 0..pairs {
            let n0 = neuron_offset + p * 2;
            let n1 = n0 + 1;
            let p0 = n0 * packed_bytes_per_row;
            let p1 = n1 * packed_bytes_per_row;
            let s0 = n0 * num_blocks_per_row;
            let s1 = n1 * num_blocks_per_row;
            let (d0, d1) = simd::dot_q4_q8_2row(
                inp_quants,
                inp_scales,
                &weight_packed[p0..p0 + packed_bytes_per_row],
                &weight_scales[s0..s0 + num_blocks_per_row],
                &weight_packed[p1..p1 + packed_bytes_per_row],
                &weight_scales[s1..s1 + num_blocks_per_row],
            );
            chunk[p * 2] = d0;
            chunk[p * 2 + 1] = d1;
        }

        if remainder > 0 {
            let neuron = neuron_offset + pairs * 2;
            let p_row_start = neuron * packed_bytes_per_row;
            let s_row_start = neuron * num_blocks_per_row;
            chunk[pairs * 2] = simd::dot_q4_q8_row(
                inp_quants,
                inp_scales,
                &weight_packed[p_row_start..p_row_start + packed_bytes_per_row],
                &weight_scales[s_row_start..s_row_start + num_blocks_per_row],
            );
        }
    };

    let pool = crate::thread_pool::global_pool();
    let num_threads = pool.num_threads();
    let min_neurons_per_task = 64;
    if n < num_threads * min_neurons_per_task {
        gemv_body(out, 0);
    } else {
        // Round chunk_size to even for clean pair processing
        let raw_chunk = (n / num_threads).max(min_neurons_per_task).min(n);
        let chunk_size = (raw_chunk + 1) & !1; // round up to even
        let out_addr = ptr_to_usize(out.as_mut_ptr());
        pool.dispatch(num_threads.min(n.div_ceil(chunk_size)), |task_id, _| {
            let start = task_id * chunk_size;
            let end = (start + chunk_size).min(n);
            if start >= n {
                return;
            }
            let chunk = unsafe {
                std::slice::from_raw_parts_mut((out_addr as *mut f32).add(start), end - start)
            };
            gemv_body(chunk, start);
        });
    }
}

/// Parallel GEMV for Q4_1 using integer dot product (Q4_1×Q8).
#[allow(clippy::too_many_arguments)]
fn q4_1_gemv_parallel(
    out: &mut [f32],
    inp_quants: &[u8],
    inp_scales: &[f32],
    inp_f32: &[f32],
    weight_packed: &[u8],
    weight_scales: &[f32],
    weight_mins: &[f32],
    num_blocks_per_row: usize,
    packed_bytes_per_row: usize,
) {
    let n = out.len();

    let gemv_body = |chunk: &mut [f32], neuron_offset: usize| {
        for (i, out_val) in chunk.iter_mut().enumerate() {
            let neuron = neuron_offset + i;
            let p_row_start = neuron * packed_bytes_per_row;
            let s_row_start = neuron * num_blocks_per_row;
            *out_val = simd::dot_q4_1_q8_row(
                inp_quants,
                inp_scales,
                inp_f32,
                &weight_packed[p_row_start..p_row_start + packed_bytes_per_row],
                &weight_scales[s_row_start..s_row_start + num_blocks_per_row],
                &weight_mins[s_row_start..s_row_start + num_blocks_per_row],
            );
        }
    };

    let pool = crate::thread_pool::global_pool();
    let num_threads = pool.num_threads();
    let min_neurons_per_task = 64;
    if n < num_threads * min_neurons_per_task {
        gemv_body(out, 0);
    } else {
        let chunk_size = (n / num_threads).max(min_neurons_per_task).min(n);
        let out_addr = ptr_to_usize(out.as_mut_ptr());
        pool.dispatch(num_threads.min(n.div_ceil(chunk_size)), |task_id, _| {
            let start = task_id * chunk_size;
            let end = (start + chunk_size).min(n);
            if start >= n {
                return;
            }
            let chunk = unsafe {
                std::slice::from_raw_parts_mut((out_addr as *mut f32).add(start), end - start)
            };
            gemv_body(chunk, start);
        });
    }
}

/// Quantized linear: `input (M, K) × weight (N, K)_quantized → output (M, N)`.
///
/// Weight is stored as `(out_features=N, in_features=K)` in block-quantized
/// format. Scales are pre-decoded to f32 in `CpuQuantizedWeight` so no
/// f16→f32 conversion happens in the hot path.
///
/// - **M=1 (decode):** single GEMV with thread-pool dispatch over N neurons.
/// - **M>1 (prefill):** pre-quantize all M input rows, then single dispatch
///   parallelized over N neurons. Each thread computes all M rows for its
///   chunk of neurons, reusing weight data in cache across rows.
#[allow(clippy::many_single_char_names, clippy::too_many_lines)]
fn quantized_linear(input: &CpuTensor, weight: &CpuQuantizedWeight) -> Result<CpuTensor> {
    let i_shape = input.shape();
    let m: usize = i_shape[..i_shape.len() - 1].iter().product();
    let k = *i_shape.last().unwrap();

    let n = weight.shape[0]; // out_features
    let wk = weight.shape[1]; // in_features
    assert_eq!(
        k, wk,
        "quantized_linear: input dim {k} != weight in_features {wk}"
    );
    assert_eq!(
        k % QUANTIZATION_BLOCK_SIZE,
        0,
        "quantized_linear: in_features {k} not divisible by block size {QUANTIZATION_BLOCK_SIZE}"
    );

    let num_blocks_per_row = k / QUANTIZATION_BLOCK_SIZE;
    let input_data = input.as_f32_slice();
    let mut output = vec![0.0f32; m * n];

    if m == 1 {
        // M=1 decode path: single-row GEMV (unchanged)
        let mut inp_quants = vec![0u8; k];
        let mut inp_scales = vec![0.0f32; num_blocks_per_row];
        let inp = &input_data[..k];
        simd::quantize_row_q8(inp, &mut inp_quants, &mut inp_scales);

        match weight.dtype {
            DType::Q8_0 => {
                let bpr = num_blocks_per_row * QUANTIZATION_BLOCK_SIZE;
                q8_gemv_parallel(
                    &mut output,
                    &inp_quants,
                    &inp_scales,
                    &weight.data,
                    &weight.scales,
                    num_blocks_per_row,
                    bpr,
                );
            }
            DType::Q4_0 => {
                let bpr = num_blocks_per_row * (QUANTIZATION_BLOCK_SIZE / 2);
                q4_gemv_parallel(
                    &mut output,
                    &inp_quants,
                    &inp_scales,
                    &weight.data,
                    &weight.scales,
                    num_blocks_per_row,
                    bpr,
                );
            }
            DType::Q4_1 => {
                let bpr = num_blocks_per_row * (QUANTIZATION_BLOCK_SIZE / 2);
                q4_1_gemv_parallel(
                    &mut output,
                    &inp_quants,
                    &inp_scales,
                    inp,
                    &weight.data,
                    &weight.scales,
                    weight
                        .mins
                        .as_ref()
                        .expect("Q4_1 weight missing mins buffer"),
                    num_blocks_per_row,
                    bpr,
                );
            }
            other => {
                return Err(infernum::Error::UnsupportedDtype(format!(
                    "quantized_linear: unsupported dtype {other}"
                )));
            }
        }
    } else {
        // M>1 prefill path: pre-quantize all rows, single dispatch over N neurons.
        let all_quants = quantize_all_rows(input_data, m, k, num_blocks_per_row);

        match weight.dtype {
            DType::Q8_0 => {
                let bpr = num_blocks_per_row * QUANTIZATION_BLOCK_SIZE;
                let pool = crate::thread_pool::global_pool();
                let num_threads = pool.num_threads();

                // Split across N columns: all threads share input rows (hot in L3),
                // each reads a subset of weight columns (fits in L2).
                let cols_per_thread = n.div_ceil(num_threads);
                let num_tasks = num_threads.min(n.div_ceil(cols_per_thread));
                let out_addr = ptr_to_usize(output.as_mut_ptr());
                let wt_data_addr = weight.data.as_ptr() as usize;
                let wt_scales_addr = weight.scales.as_ptr() as usize;

                pool.dispatch(num_tasks, |task_id, _| {
                    let col_start = task_id * cols_per_thread;
                    let col_end = (col_start + cols_per_thread).min(n);
                    if col_start >= n {
                        return;
                    }
                    let chunk_n = col_end - col_start;

                    // Create a temporary contiguous output buffer for this column chunk.
                    let mut local_out = vec![0.0f32; m * chunk_n];

                    // Weight slices for columns [col_start..col_end].
                    let wt_data = unsafe {
                        std::slice::from_raw_parts(
                            (wt_data_addr as *const u8).add(col_start * bpr),
                            chunk_n * bpr,
                        )
                    };
                    let wt_scales = unsafe {
                        std::slice::from_raw_parts(
                            (wt_scales_addr as *const f32).add(col_start * num_blocks_per_row),
                            chunk_n * num_blocks_per_row,
                        )
                    };

                    simd::gemm_q8_tiled(
                        &mut local_out,
                        &all_quants.quants,
                        &all_quants.scales,
                        wt_data,
                        wt_scales,
                        m,
                        chunk_n,
                        num_blocks_per_row,
                        bpr,
                    );

                    // Scatter local contiguous output [m × chunk_n] into strided global output.
                    let out_global =
                        unsafe { std::slice::from_raw_parts_mut(out_addr as *mut f32, m * n) };
                    for row in 0..m {
                        let src = &local_out[row * chunk_n..(row + 1) * chunk_n];
                        let dst = &mut out_global[row * n + col_start..row * n + col_end];
                        dst.copy_from_slice(src);
                    }
                });
            }
            DType::Q4_0 => {
                let bpr = num_blocks_per_row * (QUANTIZATION_BLOCK_SIZE / 2);
                quantized_gemm_parallel(
                    &mut output,
                    &all_quants,
                    m,
                    k,
                    n,
                    num_blocks_per_row,
                    &weight.data,
                    &weight.scales,
                    bpr,
                    q4_gemv_body_inline,
                );
            }
            DType::Q4_1 => {
                let packed = &weight.data;
                let scales = &weight.scales;
                let mins = weight
                    .mins
                    .as_ref()
                    .expect("Q4_1 weight missing mins buffer");
                let bpr = num_blocks_per_row * (QUANTIZATION_BLOCK_SIZE / 2);

                // Q4_1 needs original f32 input, so fall back to row-by-row
                // (Q4_1 is rare enough that this path is acceptable)
                let mut inp_quants = vec![0u8; k];
                let mut inp_scales = vec![0.0f32; num_blocks_per_row];
                for row in 0..m {
                    let inp = &input_data[row * k..(row + 1) * k];
                    simd::quantize_row_q8(inp, &mut inp_quants, &mut inp_scales);
                    q4_1_gemv_parallel(
                        &mut output[row * n..(row + 1) * n],
                        &inp_quants,
                        &inp_scales,
                        inp,
                        packed,
                        scales,
                        mins,
                        num_blocks_per_row,
                        bpr,
                    );
                }
            }
            other => {
                return Err(infernum::Error::UnsupportedDtype(format!(
                    "quantized_linear: unsupported dtype {other}"
                )));
            }
        }
    }

    let mut out_shape = i_shape[..i_shape.len() - 1].to_vec();
    out_shape.push(n);
    Ok(CpuTensor::from_f32_vec(&out_shape, output))
}

/// Pre-quantize all M input rows to Q8 format.
///
/// Returns a flat buffer of `(quants, scales)` pairs laid out as:
/// `[row0_quants(K bytes), row0_scales(num_blocks f32s), row1_quants, row1_scales, ...]`
/// packed into a single `QuantizedRows` struct for cache-friendly access.
struct QuantizedRows {
    /// All quantized bytes: `M * K` bytes total, row-contiguous.
    quants: Vec<u8>,
    /// All scales: `M * num_blocks` f32s total, row-contiguous.
    scales: Vec<f32>,
}

fn quantize_all_rows(input_data: &[f32], m: usize, k: usize, num_blocks: usize) -> QuantizedRows {
    let mut quants = vec![0u8; m * k];
    let mut scales = vec![0.0f32; m * num_blocks];

    let pool = crate::thread_pool::global_pool();
    let num_threads = pool.num_threads();
    let rows_per_thread = m.div_ceil(num_threads);
    let num_tasks = num_threads.min(m.div_ceil(rows_per_thread));
    let q_addr = quants.as_mut_ptr() as usize;
    let s_addr = scales.as_mut_ptr() as usize;

    pool.dispatch(num_tasks, |task_id, _| {
        let row_start = task_id * rows_per_thread;
        let row_end = (row_start + rows_per_thread).min(m);
        if row_start >= m {
            return;
        }
        for row in row_start..row_end {
            let inp = &input_data[row * k..(row + 1) * k];
            // Safety: each thread writes to disjoint row regions.
            let q_out =
                unsafe { std::slice::from_raw_parts_mut((q_addr as *mut u8).add(row * k), k) };
            let s_out = unsafe {
                std::slice::from_raw_parts_mut(
                    (s_addr as *mut f32).add(row * num_blocks),
                    num_blocks,
                )
            };
            simd::quantize_row_q8(inp, q_out, s_out);
        }
    });

    QuantizedRows { quants, scales }
}

/// Function pointer type for dtype-specific GEMV body kernels.
///
/// Parameters: `(output, neuron_offset, chunk_len, inp_quants, inp_scales,
///               weight_data, weight_scales, num_blocks_per_row, bytes_per_row)`
type GemvBodyFn = fn(&mut [f32], usize, usize, &[u8], &[f32], &[u8], &[f32], usize, usize);

/// Batched quantized GEMM: single thread-pool dispatch over N neurons.
///
/// Each thread processes a chunk of output neurons, computing all M rows
/// for each neuron before moving to the next. This maximizes weight data
/// reuse in L1/L2 cache (each weight row is read once per thread, used
/// M times).
///
/// The `gemv_body_fn` parameter selects the dtype-specific SIMD kernel
/// (Q8_0 or Q4_0).
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn quantized_gemm_parallel(
    output: &mut [f32],
    all_rows: &QuantizedRows,
    m: usize,
    k: usize,
    n: usize,
    num_blocks_per_row: usize,
    weight_data: &[u8],
    weight_scales: &[f32],
    bytes_per_row: usize,
    gemv_body_fn: GemvBodyFn,
) {
    let pool = crate::thread_pool::global_pool();
    let num_threads = pool.num_threads();
    let min_neurons_per_task = 64;

    if n < num_threads * min_neurons_per_task {
        // Too few neurons to parallelize — sequential over N, all M rows
        for row in 0..m {
            let q = &all_rows.quants[row * k..(row + 1) * k];
            let s = &all_rows.scales[row * num_blocks_per_row..(row + 1) * num_blocks_per_row];
            let out_row = &mut output[row * n..(row + 1) * n];
            gemv_body_fn(
                out_row,
                0,
                n,
                q,
                s,
                weight_data,
                weight_scales,
                num_blocks_per_row,
                bytes_per_row,
            );
        }
    } else {
        let raw_chunk = (n / num_threads).max(min_neurons_per_task).min(n);
        let chunk_size = (raw_chunk + 1) & !1; // round up to even for pair processing
        let num_tasks = num_threads.min(n.div_ceil(chunk_size));
        let out_addr = ptr_to_usize(output.as_mut_ptr());

        pool.dispatch(num_tasks, |task_id, _| {
            let col_start = task_id * chunk_size;
            let col_end = (col_start + chunk_size).min(n);
            if col_start >= n {
                return;
            }
            let chunk_len = col_end - col_start;

            // For each input row, compute this thread's chunk of output neurons.
            // Weight data for neurons [col_start..col_end] stays hot in cache
            // across all M rows.
            for row in 0..m {
                let q = &all_rows.quants[row * k..(row + 1) * k];
                let s = &all_rows.scales[row * num_blocks_per_row..(row + 1) * num_blocks_per_row];
                let out_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        (out_addr as *mut f32).add(row * n + col_start),
                        chunk_len,
                    )
                };
                gemv_body_fn(
                    out_slice,
                    col_start,
                    chunk_len,
                    q,
                    s,
                    weight_data,
                    weight_scales,
                    num_blocks_per_row,
                    bytes_per_row,
                );
            }
        });
    }
}

/// Batched pair: pre-quantize all M rows, single dispatch over max(N1,N2) neurons.
///
/// Each thread computes all M rows for its chunk of neurons across both weight
/// matrices in sequence, sharing input quantization and eliminating per-row
/// dispatch overhead.
#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
fn quantized_linear_pair_batched(
    input: &CpuTensor,
    w1: &CpuQuantizedWeight,
    w2: &CpuQuantizedWeight,
) -> Result<(CpuTensor, CpuTensor)> {
    let i_shape = input.shape();
    let m: usize = i_shape[..i_shape.len() - 1].iter().product();
    let k = *i_shape.last().unwrap();
    let n1 = w1.shape[0];
    let n2 = w2.shape[0];
    let num_blocks = k / QUANTIZATION_BLOCK_SIZE;

    let input_data = input.as_f32_slice();
    let all_rows = quantize_all_rows(input_data, m, k, num_blocks);

    let mut out1 = vec![0.0f32; m * n1];
    let mut out2 = vec![0.0f32; m * n2];

    match w1.dtype {
        DType::Q8_0 => {
            // Q8_0: use tiled GEMM for both weight matrices.
            let bpr = num_blocks * QUANTIZATION_BLOCK_SIZE;
            let pool = crate::thread_pool::global_pool();
            let num_threads = pool.num_threads();
            let cols_per_thread1 = n1.div_ceil(num_threads);
            let cols_per_thread2 = n2.div_ceil(num_threads);
            let num_tasks = num_threads
                .min(n1.div_ceil(cols_per_thread1))
                .max(num_threads.min(n2.div_ceil(cols_per_thread2)));
            let out1_addr = ptr_to_usize(out1.as_mut_ptr());
            let out2_addr = ptr_to_usize(out2.as_mut_ptr());
            let w1_data_addr = w1.data.as_ptr() as usize;
            let w1_scales_addr = w1.scales.as_ptr() as usize;
            let w2_data_addr = w2.data.as_ptr() as usize;
            let w2_scales_addr = w2.scales.as_ptr() as usize;

            pool.dispatch(num_tasks, |task_id, _| {
                // Weight 1
                let col_start1 = task_id * cols_per_thread1;
                if col_start1 < n1 {
                    let col_end1 = (col_start1 + cols_per_thread1).min(n1);
                    let chunk_n1 = col_end1 - col_start1;
                    let mut local_out1 = vec![0.0f32; m * chunk_n1];
                    let wt_data1 = unsafe {
                        std::slice::from_raw_parts(
                            (w1_data_addr as *const u8).add(col_start1 * bpr),
                            chunk_n1 * bpr,
                        )
                    };
                    let wt_scales1 = unsafe {
                        std::slice::from_raw_parts(
                            (w1_scales_addr as *const f32).add(col_start1 * num_blocks),
                            chunk_n1 * num_blocks,
                        )
                    };
                    simd::gemm_q8_tiled(
                        &mut local_out1,
                        &all_rows.quants,
                        &all_rows.scales,
                        wt_data1,
                        wt_scales1,
                        m,
                        chunk_n1,
                        num_blocks,
                        bpr,
                    );
                    let out_global1 =
                        unsafe { std::slice::from_raw_parts_mut(out1_addr as *mut f32, m * n1) };
                    for row in 0..m {
                        out_global1[row * n1 + col_start1..row * n1 + col_end1]
                            .copy_from_slice(&local_out1[row * chunk_n1..(row + 1) * chunk_n1]);
                    }
                }

                // Weight 2
                let col_start2 = task_id * cols_per_thread2;
                if col_start2 < n2 {
                    let col_end2 = (col_start2 + cols_per_thread2).min(n2);
                    let chunk_n2 = col_end2 - col_start2;
                    let mut local_out2 = vec![0.0f32; m * chunk_n2];
                    let wt_data2 = unsafe {
                        std::slice::from_raw_parts(
                            (w2_data_addr as *const u8).add(col_start2 * bpr),
                            chunk_n2 * bpr,
                        )
                    };
                    let wt_scales2 = unsafe {
                        std::slice::from_raw_parts(
                            (w2_scales_addr as *const f32).add(col_start2 * num_blocks),
                            chunk_n2 * num_blocks,
                        )
                    };
                    simd::gemm_q8_tiled(
                        &mut local_out2,
                        &all_rows.quants,
                        &all_rows.scales,
                        wt_data2,
                        wt_scales2,
                        m,
                        chunk_n2,
                        num_blocks,
                        bpr,
                    );
                    let out_global2 =
                        unsafe { std::slice::from_raw_parts_mut(out2_addr as *mut f32, m * n2) };
                    for row in 0..m {
                        out_global2[row * n2 + col_start2..row * n2 + col_end2]
                            .copy_from_slice(&local_out2[row * chunk_n2..(row + 1) * chunk_n2]);
                    }
                }
            });
        }
        DType::Q4_0 => {
            // Q4_0: no tiled GEMM kernel — use GEMV row-by-row.
            let bpr = num_blocks * (QUANTIZATION_BLOCK_SIZE / 2);
            let pool = crate::thread_pool::global_pool();
            let num_threads = pool.num_threads();
            let min_neurons = 64;
            let n_max = n1.max(n2);

            if n_max < num_threads * min_neurons {
                for row in 0..m {
                    let q = &all_rows.quants[row * k..(row + 1) * k];
                    let s = &all_rows.scales[row * num_blocks..(row + 1) * num_blocks];
                    q4_gemv_body_inline(
                        &mut out1[row * n1..(row + 1) * n1],
                        0,
                        n1,
                        q,
                        s,
                        &w1.data,
                        &w1.scales,
                        num_blocks,
                        bpr,
                    );
                    q4_gemv_body_inline(
                        &mut out2[row * n2..(row + 1) * n2],
                        0,
                        n2,
                        q,
                        s,
                        &w2.data,
                        &w2.scales,
                        num_blocks,
                        bpr,
                    );
                }
            } else {
                let chunk1 = ((n1 / num_threads).max(min_neurons).min(n1) + 1) & !1;
                let chunk2 = ((n2 / num_threads).max(min_neurons).min(n2) + 1) & !1;
                let out1_addr = ptr_to_usize(out1.as_mut_ptr());
                let out2_addr = ptr_to_usize(out2.as_mut_ptr());

                pool.dispatch(num_threads, |task_id, _| {
                    for row in 0..m {
                        let q = &all_rows.quants[row * k..(row + 1) * k];
                        let s = &all_rows.scales[row * num_blocks..(row + 1) * num_blocks];

                        let start1 = task_id * chunk1;
                        if start1 < n1 {
                            let end1 = (start1 + chunk1).min(n1);
                            let slice1 = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (out1_addr as *mut f32).add(row * n1 + start1),
                                    end1 - start1,
                                )
                            };
                            q4_gemv_body_inline(
                                slice1,
                                start1,
                                end1 - start1,
                                q,
                                s,
                                &w1.data,
                                &w1.scales,
                                num_blocks,
                                bpr,
                            );
                        }
                        let start2 = task_id * chunk2;
                        if start2 < n2 {
                            let end2 = (start2 + chunk2).min(n2);
                            let slice2 = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (out2_addr as *mut f32).add(row * n2 + start2),
                                    end2 - start2,
                                )
                            };
                            q4_gemv_body_inline(
                                slice2,
                                start2,
                                end2 - start2,
                                q,
                                s,
                                &w2.data,
                                &w2.scales,
                                num_blocks,
                                bpr,
                            );
                        }
                    }
                });
            }
        }
        _ => {
            let a = quantized_linear(input, w1)?;
            let b = quantized_linear(input, w2)?;
            return Ok((a, b));
        }
    }

    let out_shape1 = vec![m, n1];
    let out_shape2 = vec![m, n2];
    Ok((
        CpuTensor::from_f32_vec(&out_shape1, out1),
        CpuTensor::from_f32_vec(&out_shape2, out2),
    ))
}

/// Batched triple: pre-quantize all M rows, single dispatch for three weights.
///
/// Same strategy as pair but for Q/K/V attention projections.
#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
fn quantized_linear_triple_batched(
    input: &CpuTensor,
    w1: &CpuQuantizedWeight,
    w2: &CpuQuantizedWeight,
    w3: &CpuQuantizedWeight,
) -> Result<(CpuTensor, CpuTensor, CpuTensor)> {
    let i_shape = input.shape();
    let m: usize = i_shape[..i_shape.len() - 1].iter().product();
    let k = *i_shape.last().unwrap();
    let n1 = w1.shape[0];
    let n2 = w2.shape[0];
    let n3 = w3.shape[0];
    let num_blocks = k / QUANTIZATION_BLOCK_SIZE;

    let input_data = input.as_f32_slice();
    let all_rows = quantize_all_rows(input_data, m, k, num_blocks);

    let mut out1 = vec![0.0f32; m * n1];
    let mut out2 = vec![0.0f32; m * n2];
    let mut out3 = vec![0.0f32; m * n3];

    match w1.dtype {
        DType::Q8_0 => {
            // Q8_0: use tiled GEMM for all three weight matrices.
            let bpr = num_blocks * QUANTIZATION_BLOCK_SIZE;
            let pool = crate::thread_pool::global_pool();
            let num_threads = pool.num_threads();
            let cpt1 = n1.div_ceil(num_threads);
            let cpt2 = n2.div_ceil(num_threads);
            let cpt3 = n3.div_ceil(num_threads);
            let num_tasks = num_threads
                .min(n1.div_ceil(cpt1))
                .max(num_threads.min(n2.div_ceil(cpt2)))
                .max(num_threads.min(n3.div_ceil(cpt3)));
            let out1_addr = ptr_to_usize(out1.as_mut_ptr());
            let out2_addr = ptr_to_usize(out2.as_mut_ptr());
            let out3_addr = ptr_to_usize(out3.as_mut_ptr());
            let w1d = w1.data.as_ptr() as usize;
            let w1s = w1.scales.as_ptr() as usize;
            let w2d = w2.data.as_ptr() as usize;
            let w2s = w2.scales.as_ptr() as usize;
            let w3d = w3.data.as_ptr() as usize;
            let w3s = w3.scales.as_ptr() as usize;

            pool.dispatch(num_tasks, |task_id, _| {
                // Helper: run tiled GEMM for one weight matrix's column chunk.
                macro_rules! gemm_chunk {
                    ($col_start:expr, $n:expr, $cpt:expr, $wd:expr, $ws:expr,
                     $out_addr:expr) => {
                        let cs = $col_start;
                        if cs < $n {
                            let ce = (cs + $cpt).min($n);
                            let cn = ce - cs;
                            let mut local = vec![0.0f32; m * cn];
                            let wd = unsafe {
                                std::slice::from_raw_parts(
                                    ($wd as *const u8).add(cs * bpr),
                                    cn * bpr,
                                )
                            };
                            let ws = unsafe {
                                std::slice::from_raw_parts(
                                    ($ws as *const f32).add(cs * num_blocks),
                                    cn * num_blocks,
                                )
                            };
                            simd::gemm_q8_tiled(
                                &mut local,
                                &all_rows.quants,
                                &all_rows.scales,
                                wd,
                                ws,
                                m,
                                cn,
                                num_blocks,
                                bpr,
                            );
                            let out = unsafe {
                                std::slice::from_raw_parts_mut($out_addr as *mut f32, m * $n)
                            };
                            for row in 0..m {
                                out[row * $n + cs..row * $n + ce]
                                    .copy_from_slice(&local[row * cn..(row + 1) * cn]);
                            }
                        }
                    };
                }
                gemm_chunk!(task_id * cpt1, n1, cpt1, w1d, w1s, out1_addr);
                gemm_chunk!(task_id * cpt2, n2, cpt2, w2d, w2s, out2_addr);
                gemm_chunk!(task_id * cpt3, n3, cpt3, w3d, w3s, out3_addr);
            });
        }
        DType::Q4_0 => {
            // Q4_0: no tiled GEMM kernel — use GEMV row-by-row.
            let bpr = num_blocks * (QUANTIZATION_BLOCK_SIZE / 2);
            let pool = crate::thread_pool::global_pool();
            let num_threads = pool.num_threads();
            let min_neurons = 64;
            let n_max = n1.max(n2).max(n3);

            if n_max < num_threads * min_neurons {
                for row in 0..m {
                    let q = &all_rows.quants[row * k..(row + 1) * k];
                    let s = &all_rows.scales[row * num_blocks..(row + 1) * num_blocks];
                    q4_gemv_body_inline(
                        &mut out1[row * n1..(row + 1) * n1],
                        0,
                        n1,
                        q,
                        s,
                        &w1.data,
                        &w1.scales,
                        num_blocks,
                        bpr,
                    );
                    q4_gemv_body_inline(
                        &mut out2[row * n2..(row + 1) * n2],
                        0,
                        n2,
                        q,
                        s,
                        &w2.data,
                        &w2.scales,
                        num_blocks,
                        bpr,
                    );
                    q4_gemv_body_inline(
                        &mut out3[row * n3..(row + 1) * n3],
                        0,
                        n3,
                        q,
                        s,
                        &w3.data,
                        &w3.scales,
                        num_blocks,
                        bpr,
                    );
                }
            } else {
                let chunk1 = ((n1 / num_threads).max(min_neurons).min(n1) + 1) & !1;
                let chunk2 = ((n2 / num_threads).max(min_neurons).min(n2) + 1) & !1;
                let chunk3 = ((n3 / num_threads).max(min_neurons).min(n3) + 1) & !1;
                let out1_addr = ptr_to_usize(out1.as_mut_ptr());
                let out2_addr = ptr_to_usize(out2.as_mut_ptr());
                let out3_addr = ptr_to_usize(out3.as_mut_ptr());

                pool.dispatch(num_threads, |task_id, _| {
                    for row in 0..m {
                        let q = &all_rows.quants[row * k..(row + 1) * k];
                        let s = &all_rows.scales[row * num_blocks..(row + 1) * num_blocks];

                        let start1 = task_id * chunk1;
                        if start1 < n1 {
                            let end1 = (start1 + chunk1).min(n1);
                            let slice1 = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (out1_addr as *mut f32).add(row * n1 + start1),
                                    end1 - start1,
                                )
                            };
                            q4_gemv_body_inline(
                                slice1,
                                start1,
                                end1 - start1,
                                q,
                                s,
                                &w1.data,
                                &w1.scales,
                                num_blocks,
                                bpr,
                            );
                        }
                        let start2 = task_id * chunk2;
                        if start2 < n2 {
                            let end2 = (start2 + chunk2).min(n2);
                            let slice2 = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (out2_addr as *mut f32).add(row * n2 + start2),
                                    end2 - start2,
                                )
                            };
                            q4_gemv_body_inline(
                                slice2,
                                start2,
                                end2 - start2,
                                q,
                                s,
                                &w2.data,
                                &w2.scales,
                                num_blocks,
                                bpr,
                            );
                        }
                        let start3 = task_id * chunk3;
                        if start3 < n3 {
                            let end3 = (start3 + chunk3).min(n3);
                            let slice3 = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (out3_addr as *mut f32).add(row * n3 + start3),
                                    end3 - start3,
                                )
                            };
                            q4_gemv_body_inline(
                                slice3,
                                start3,
                                end3 - start3,
                                q,
                                s,
                                &w3.data,
                                &w3.scales,
                                num_blocks,
                                bpr,
                            );
                        }
                    }
                });
            }
        }
        _ => {
            let a = quantized_linear(input, w1)?;
            let b = quantized_linear(input, w2)?;
            let c = quantized_linear(input, w3)?;
            return Ok((a, b, c));
        }
    }

    let out_shape1 = vec![m, n1];
    let out_shape2 = vec![m, n2];
    let out_shape3 = vec![m, n3];
    Ok((
        CpuTensor::from_f32_vec(&out_shape1, out1),
        CpuTensor::from_f32_vec(&out_shape2, out2),
        CpuTensor::from_f32_vec(&out_shape3, out3),
    ))
}

/// Compute two independent quantized matmuls in a single thread pool dispatch.
///
/// Each thread processes its output rows for BOTH weight matrices before
/// signaling completion, eliminating the sync gap between them.
/// Only supports the decode path (M=1) with matching dtypes.
#[allow(clippy::too_many_lines)]
fn quantized_linear_pair(
    input: &CpuTensor,
    w1: &CpuQuantizedWeight,
    w2: &CpuQuantizedWeight,
) -> Result<(CpuTensor, CpuTensor)> {
    let i_shape = input.shape();
    let m: usize = i_shape[..i_shape.len() - 1].iter().product();
    let k = *i_shape.last().unwrap();

    // Validate shapes
    assert_eq!(k, w1.shape[1], "linear_pair: input K != w1 K");
    assert_eq!(k, w2.shape[1], "linear_pair: input K != w2 K");
    assert_eq!(
        k % QUANTIZATION_BLOCK_SIZE,
        0,
        "linear_pair: K not aligned to block size"
    );

    let n1 = w1.shape[0];
    let n2 = w2.shape[0];
    let num_blocks = k / QUANTIZATION_BLOCK_SIZE;

    // Only fuse matching dtypes (Q4_0 or Q8_0).
    if w1.dtype != w2.dtype {
        let a = quantized_linear(input, w1)?;
        let b = quantized_linear(input, w2)?;
        return Ok((a, b));
    }

    // M>1: pre-quantize all rows, dispatch over N for both weights
    if m > 1 {
        return quantized_linear_pair_batched(input, w1, w2);
    }

    let input_data = input.as_f32_slice();
    let inp = &input_data[..k];

    // Quantize input once, shared by both matmuls.
    let mut inp_quants = vec![0u8; k];
    let mut inp_scales = vec![0.0f32; num_blocks];
    simd::quantize_row_q8(inp, &mut inp_quants, &mut inp_scales);

    let mut out1 = vec![0.0f32; n1];
    let mut out2 = vec![0.0f32; n2];

    let pool = crate::thread_pool::global_pool();
    let num_threads = pool.num_threads();
    let min_neurons = 64; // Lower threshold since we amortize over 2 matmuls

    match w1.dtype {
        DType::Q4_0 => {
            let bpr = num_blocks * (QUANTIZATION_BLOCK_SIZE / 2); // packed bytes per row

            if n1 < num_threads * min_neurons && n2 < num_threads * min_neurons {
                // Both too small to parallelize — sequential
                q4_gemv_body_inline(
                    &mut out1,
                    0,
                    n1,
                    &inp_quants,
                    &inp_scales,
                    &w1.data,
                    &w1.scales,
                    num_blocks,
                    bpr,
                );
                q4_gemv_body_inline(
                    &mut out2,
                    0,
                    n2,
                    &inp_quants,
                    &inp_scales,
                    &w2.data,
                    &w2.scales,
                    num_blocks,
                    bpr,
                );
            } else {
                let chunk1 = ((n1 / num_threads).max(min_neurons).min(n1) + 1) & !1;
                let chunk2 = ((n2 / num_threads).max(min_neurons).min(n2) + 1) & !1;
                let out1_addr = ptr_to_usize(out1.as_mut_ptr());
                let out2_addr = ptr_to_usize(out2.as_mut_ptr());

                pool.dispatch(num_threads, |task_id, _| {
                    // Process w1 rows for this thread
                    let start1 = task_id * chunk1;
                    if start1 < n1 {
                        let end1 = (start1 + chunk1).min(n1);
                        let slice1 = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out1_addr as *mut f32).add(start1),
                                end1 - start1,
                            )
                        };
                        q4_gemv_body_inline(
                            slice1,
                            start1,
                            end1 - start1,
                            &inp_quants,
                            &inp_scales,
                            &w1.data,
                            &w1.scales,
                            num_blocks,
                            bpr,
                        );
                    }
                    // Process w2 rows for this thread (no sync in between!)
                    let start2 = task_id * chunk2;
                    if start2 < n2 {
                        let end2 = (start2 + chunk2).min(n2);
                        let slice2 = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out2_addr as *mut f32).add(start2),
                                end2 - start2,
                            )
                        };
                        q4_gemv_body_inline(
                            slice2,
                            start2,
                            end2 - start2,
                            &inp_quants,
                            &inp_scales,
                            &w2.data,
                            &w2.scales,
                            num_blocks,
                            bpr,
                        );
                    }
                });
            }
        }
        DType::Q8_0 => {
            let bpr = num_blocks * QUANTIZATION_BLOCK_SIZE; // quant bytes per row

            if n1 < num_threads * min_neurons && n2 < num_threads * min_neurons {
                q8_gemv_body_inline(
                    &mut out1,
                    0,
                    n1,
                    &inp_quants,
                    &inp_scales,
                    &w1.data,
                    &w1.scales,
                    num_blocks,
                    bpr,
                );
                q8_gemv_body_inline(
                    &mut out2,
                    0,
                    n2,
                    &inp_quants,
                    &inp_scales,
                    &w2.data,
                    &w2.scales,
                    num_blocks,
                    bpr,
                );
            } else {
                let chunk1 = ((n1 / num_threads).max(min_neurons).min(n1) + 1) & !1;
                let chunk2 = ((n2 / num_threads).max(min_neurons).min(n2) + 1) & !1;
                let out1_addr = ptr_to_usize(out1.as_mut_ptr());
                let out2_addr = ptr_to_usize(out2.as_mut_ptr());

                pool.dispatch(num_threads, |task_id, _| {
                    let start1 = task_id * chunk1;
                    if start1 < n1 {
                        let end1 = (start1 + chunk1).min(n1);
                        let slice1 = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out1_addr as *mut f32).add(start1),
                                end1 - start1,
                            )
                        };
                        q8_gemv_body_inline(
                            slice1,
                            start1,
                            end1 - start1,
                            &inp_quants,
                            &inp_scales,
                            &w1.data,
                            &w1.scales,
                            num_blocks,
                            bpr,
                        );
                    }
                    let start2 = task_id * chunk2;
                    if start2 < n2 {
                        let end2 = (start2 + chunk2).min(n2);
                        let slice2 = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out2_addr as *mut f32).add(start2),
                                end2 - start2,
                            )
                        };
                        q8_gemv_body_inline(
                            slice2,
                            start2,
                            end2 - start2,
                            &inp_quants,
                            &inp_scales,
                            &w2.data,
                            &w2.scales,
                            num_blocks,
                            bpr,
                        );
                    }
                });
            }
        }
        _ => {
            // Fall back to two separate calls for unsupported dtypes
            let a = quantized_linear(input, w1)?;
            let b = quantized_linear(input, w2)?;
            return Ok((a, b));
        }
    }

    let out_shape1 = vec![m, n1];
    let out_shape2 = vec![m, n2];
    Ok((
        CpuTensor::from_f32_vec(&out_shape1, out1),
        CpuTensor::from_f32_vec(&out_shape2, out2),
    ))
}

/// Fused triple quantized linear: single dispatch for three weight matrices.
///
/// Quantizes the input once (shared by all three matmuls), then each
/// thread processes its output rows for all three matrices before
/// signaling completion. Used for Q+K+V attention projections.
/// Only supports the decode path (M=1) with matching dtypes.
#[allow(clippy::too_many_lines)]
fn quantized_linear_triple(
    input: &CpuTensor,
    w1: &CpuQuantizedWeight,
    w2: &CpuQuantizedWeight,
    w3: &CpuQuantizedWeight,
) -> Result<(CpuTensor, CpuTensor, CpuTensor)> {
    let i_shape = input.shape();
    let m: usize = i_shape[..i_shape.len() - 1].iter().product();
    let k = *i_shape.last().unwrap();

    assert_eq!(k, w1.shape[1], "linear_triple: input K != w1 K");
    assert_eq!(k, w2.shape[1], "linear_triple: input K != w2 K");
    assert_eq!(k, w3.shape[1], "linear_triple: input K != w3 K");
    assert_eq!(
        k % QUANTIZATION_BLOCK_SIZE,
        0,
        "linear_triple: K not aligned to block size"
    );

    let n1 = w1.shape[0];
    let n2 = w2.shape[0];
    let n3 = w3.shape[0];
    let num_blocks = k / QUANTIZATION_BLOCK_SIZE;

    // Only fuse matching dtypes (Q4_0 or Q8_0).
    if w1.dtype != w2.dtype || w1.dtype != w3.dtype {
        let out1 = quantized_linear(input, w1)?;
        let out2 = quantized_linear(input, w2)?;
        let out3 = quantized_linear(input, w3)?;
        return Ok((out1, out2, out3));
    }

    // M>1: pre-quantize all rows, dispatch over N for all three weights
    if m > 1 {
        return quantized_linear_triple_batched(input, w1, w2, w3);
    }

    let input_data = input.as_f32_slice();
    let inp = &input_data[..k];

    // Quantize input once, shared by all three matmuls.
    let mut inp_quants = vec![0u8; k];
    let mut inp_scales = vec![0.0f32; num_blocks];
    simd::quantize_row_q8(inp, &mut inp_quants, &mut inp_scales);

    let mut out1 = vec![0.0f32; n1];
    let mut out2 = vec![0.0f32; n2];
    let mut out3 = vec![0.0f32; n3];

    let pool = crate::thread_pool::global_pool();
    let num_threads = pool.num_threads();
    let min_neurons = 64;

    match w1.dtype {
        DType::Q4_0 => {
            let bpr = num_blocks * (QUANTIZATION_BLOCK_SIZE / 2);

            if n1 < num_threads * min_neurons
                && n2 < num_threads * min_neurons
                && n3 < num_threads * min_neurons
            {
                q4_gemv_body_inline(
                    &mut out1,
                    0,
                    n1,
                    &inp_quants,
                    &inp_scales,
                    &w1.data,
                    &w1.scales,
                    num_blocks,
                    bpr,
                );
                q4_gemv_body_inline(
                    &mut out2,
                    0,
                    n2,
                    &inp_quants,
                    &inp_scales,
                    &w2.data,
                    &w2.scales,
                    num_blocks,
                    bpr,
                );
                q4_gemv_body_inline(
                    &mut out3,
                    0,
                    n3,
                    &inp_quants,
                    &inp_scales,
                    &w3.data,
                    &w3.scales,
                    num_blocks,
                    bpr,
                );
            } else {
                let chunk1 = ((n1 / num_threads).max(min_neurons).min(n1) + 1) & !1;
                let chunk2 = ((n2 / num_threads).max(min_neurons).min(n2) + 1) & !1;
                let chunk3 = ((n3 / num_threads).max(min_neurons).min(n3) + 1) & !1;
                let out1_addr = ptr_to_usize(out1.as_mut_ptr());
                let out2_addr = ptr_to_usize(out2.as_mut_ptr());
                let out3_addr = ptr_to_usize(out3.as_mut_ptr());

                pool.dispatch(num_threads, |task_id, _| {
                    let start1 = task_id * chunk1;
                    if start1 < n1 {
                        let end1 = (start1 + chunk1).min(n1);
                        let slice1 = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out1_addr as *mut f32).add(start1),
                                end1 - start1,
                            )
                        };
                        q4_gemv_body_inline(
                            slice1,
                            start1,
                            end1 - start1,
                            &inp_quants,
                            &inp_scales,
                            &w1.data,
                            &w1.scales,
                            num_blocks,
                            bpr,
                        );
                    }
                    let start2 = task_id * chunk2;
                    if start2 < n2 {
                        let end2 = (start2 + chunk2).min(n2);
                        let slice2 = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out2_addr as *mut f32).add(start2),
                                end2 - start2,
                            )
                        };
                        q4_gemv_body_inline(
                            slice2,
                            start2,
                            end2 - start2,
                            &inp_quants,
                            &inp_scales,
                            &w2.data,
                            &w2.scales,
                            num_blocks,
                            bpr,
                        );
                    }
                    let start3 = task_id * chunk3;
                    if start3 < n3 {
                        let end3 = (start3 + chunk3).min(n3);
                        let slice3 = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out3_addr as *mut f32).add(start3),
                                end3 - start3,
                            )
                        };
                        q4_gemv_body_inline(
                            slice3,
                            start3,
                            end3 - start3,
                            &inp_quants,
                            &inp_scales,
                            &w3.data,
                            &w3.scales,
                            num_blocks,
                            bpr,
                        );
                    }
                });
            }
        }
        DType::Q8_0 => {
            let bpr = num_blocks * QUANTIZATION_BLOCK_SIZE;

            if n1 < num_threads * min_neurons
                && n2 < num_threads * min_neurons
                && n3 < num_threads * min_neurons
            {
                q8_gemv_body_inline(
                    &mut out1,
                    0,
                    n1,
                    &inp_quants,
                    &inp_scales,
                    &w1.data,
                    &w1.scales,
                    num_blocks,
                    bpr,
                );
                q8_gemv_body_inline(
                    &mut out2,
                    0,
                    n2,
                    &inp_quants,
                    &inp_scales,
                    &w2.data,
                    &w2.scales,
                    num_blocks,
                    bpr,
                );
                q8_gemv_body_inline(
                    &mut out3,
                    0,
                    n3,
                    &inp_quants,
                    &inp_scales,
                    &w3.data,
                    &w3.scales,
                    num_blocks,
                    bpr,
                );
            } else {
                let chunk1 = ((n1 / num_threads).max(min_neurons).min(n1) + 1) & !1;
                let chunk2 = ((n2 / num_threads).max(min_neurons).min(n2) + 1) & !1;
                let chunk3 = ((n3 / num_threads).max(min_neurons).min(n3) + 1) & !1;
                let out1_addr = ptr_to_usize(out1.as_mut_ptr());
                let out2_addr = ptr_to_usize(out2.as_mut_ptr());
                let out3_addr = ptr_to_usize(out3.as_mut_ptr());

                pool.dispatch(num_threads, |task_id, _| {
                    let start1 = task_id * chunk1;
                    if start1 < n1 {
                        let end1 = (start1 + chunk1).min(n1);
                        let slice1 = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out1_addr as *mut f32).add(start1),
                                end1 - start1,
                            )
                        };
                        q8_gemv_body_inline(
                            slice1,
                            start1,
                            end1 - start1,
                            &inp_quants,
                            &inp_scales,
                            &w1.data,
                            &w1.scales,
                            num_blocks,
                            bpr,
                        );
                    }
                    let start2 = task_id * chunk2;
                    if start2 < n2 {
                        let end2 = (start2 + chunk2).min(n2);
                        let slice2 = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out2_addr as *mut f32).add(start2),
                                end2 - start2,
                            )
                        };
                        q8_gemv_body_inline(
                            slice2,
                            start2,
                            end2 - start2,
                            &inp_quants,
                            &inp_scales,
                            &w2.data,
                            &w2.scales,
                            num_blocks,
                            bpr,
                        );
                    }
                    let start3 = task_id * chunk3;
                    if start3 < n3 {
                        let end3 = (start3 + chunk3).min(n3);
                        let slice3 = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out3_addr as *mut f32).add(start3),
                                end3 - start3,
                            )
                        };
                        q8_gemv_body_inline(
                            slice3,
                            start3,
                            end3 - start3,
                            &inp_quants,
                            &inp_scales,
                            &w3.data,
                            &w3.scales,
                            num_blocks,
                            bpr,
                        );
                    }
                });
            }
        }
        _ => {
            let out1 = quantized_linear(input, w1)?;
            let out2 = quantized_linear(input, w2)?;
            let out3 = quantized_linear(input, w3)?;
            return Ok((out1, out2, out3));
        }
    }

    let out_shape1 = vec![m, n1];
    let out_shape2 = vec![m, n2];
    let out_shape3 = vec![m, n3];
    Ok((
        CpuTensor::from_f32_vec(&out_shape1, out1),
        CpuTensor::from_f32_vec(&out_shape2, out2),
        CpuTensor::from_f32_vec(&out_shape3, out3),
    ))
}

/// Q4_0 GEMV body: process `chunk_len` neurons starting at `neuron_offset`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn q4_gemv_body_inline(
    chunk: &mut [f32],
    neuron_offset: usize,
    chunk_len: usize,
    inp_quants: &[u8],
    inp_scales: &[f32],
    weight_packed: &[u8],
    weight_scales: &[f32],
    num_blocks_per_row: usize,
    packed_bytes_per_row: usize,
) {
    let pairs = chunk_len / 2;
    let remainder = chunk_len % 2;

    for p in 0..pairs {
        let n0 = neuron_offset + p * 2;
        let n1 = n0 + 1;
        let p0 = n0 * packed_bytes_per_row;
        let p1 = n1 * packed_bytes_per_row;
        let s0 = n0 * num_blocks_per_row;
        let s1 = n1 * num_blocks_per_row;
        let (d0, d1) = simd::dot_q4_q8_2row(
            inp_quants,
            inp_scales,
            &weight_packed[p0..p0 + packed_bytes_per_row],
            &weight_scales[s0..s0 + num_blocks_per_row],
            &weight_packed[p1..p1 + packed_bytes_per_row],
            &weight_scales[s1..s1 + num_blocks_per_row],
        );
        chunk[p * 2] = d0;
        chunk[p * 2 + 1] = d1;
    }

    if remainder > 0 {
        let neuron = neuron_offset + pairs * 2;
        let p_start = neuron * packed_bytes_per_row;
        let s_start = neuron * num_blocks_per_row;
        chunk[pairs * 2] = simd::dot_q4_q8_row(
            inp_quants,
            inp_scales,
            &weight_packed[p_start..p_start + packed_bytes_per_row],
            &weight_scales[s_start..s_start + num_blocks_per_row],
        );
    }
}

/// Q8_0 GEMV body: process `chunk_len` neurons starting at `neuron_offset`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn q8_gemv_body_inline(
    chunk: &mut [f32],
    neuron_offset: usize,
    chunk_len: usize,
    inp_quants: &[u8],
    inp_scales: &[f32],
    weight_quants: &[u8],
    weight_scales: &[f32],
    num_blocks_per_row: usize,
    quant_bytes_per_row: usize,
) {
    let pairs = chunk_len / 2;
    let remainder = chunk_len % 2;

    for p in 0..pairs {
        let n0 = neuron_offset + p * 2;
        let n1 = n0 + 1;
        let q0 = n0 * quant_bytes_per_row;
        let q1 = n1 * quant_bytes_per_row;
        let s0 = n0 * num_blocks_per_row;
        let s1 = n1 * num_blocks_per_row;
        let (d0, d1) = simd::dot_q8_q8_2row(
            inp_quants,
            inp_scales,
            &weight_quants[q0..q0 + quant_bytes_per_row],
            &weight_scales[s0..s0 + num_blocks_per_row],
            &weight_quants[q1..q1 + quant_bytes_per_row],
            &weight_scales[s1..s1 + num_blocks_per_row],
        );
        chunk[p * 2] = d0;
        chunk[p * 2 + 1] = d1;
    }

    if remainder > 0 {
        let neuron = neuron_offset + pairs * 2;
        let q_start = neuron * quant_bytes_per_row;
        let s_start = neuron * num_blocks_per_row;
        chunk[pairs * 2] = simd::dot_q8_q8_row(
            inp_quants,
            inp_scales,
            &weight_quants[q_start..q_start + quant_bytes_per_row],
            &weight_scales[s_start..s_start + num_blocks_per_row],
        );
    }
}

impl MatmulOps for CpuBackend {
    type LinearWeight = CpuLinearWeight;

    fn matmul(input: &CpuTensor, weight: &CpuTensor) -> Result<CpuTensor> {
        // Standard: input (..., K) × weight (K, N) → (..., N)
        let w_shape = weight.shape();
        assert!(
            w_shape.len() == 2,
            "matmul: weight must be 2D, got {w_shape:?}"
        );
        let k = w_shape[0];
        let n = w_shape[1];

        let i_shape = input.shape();
        let m: usize = i_shape[..i_shape.len() - 1].iter().product();

        assert_eq!(
            *i_shape.last().unwrap(),
            k,
            "matmul: input last dim {} != weight rows {}, input shape {:?}, weight shape {:?}",
            i_shape.last().unwrap(),
            k,
            i_shape,
            w_shape,
        );

        let output = gemm(input.as_f32_slice(), weight.as_f32_slice(), m, k, n);

        let mut out_shape = i_shape[..i_shape.len() - 1].to_vec();
        out_shape.push(n);
        Ok(CpuTensor::from_f32_vec(&out_shape, output))
    }

    fn linear(input: &CpuTensor, weight: &CpuLinearWeight) -> Result<CpuTensor> {
        match weight {
            CpuLinearWeight::Dense { weight_nt, .. } => {
                let nt_shape = weight_nt.shape();
                let n = nt_shape[0];
                let k = nt_shape[1];

                let i_shape = input.shape();
                let m: usize = i_shape[..i_shape.len() - 1].iter().product();

                assert_eq!(
                    *i_shape.last().unwrap(),
                    k,
                    "linear: input last dim {} != weight in_features {k}",
                    i_shape.last().unwrap(),
                );

                let output = gemm_with_bt(input.as_f32_slice(), weight_nt.as_f32_slice(), m, k, n);

                let mut out_shape = i_shape[..i_shape.len() - 1].to_vec();
                out_shape.push(n);
                Ok(CpuTensor::from_f32_vec(&out_shape, output))
            }
            CpuLinearWeight::Quantized(w) => quantized_linear(input, w),
        }
    }

    fn as_dense_weight(weight: &CpuLinearWeight) -> Option<&CpuTensor> {
        match weight {
            CpuLinearWeight::Dense { weight, .. } => Some(weight),
            CpuLinearWeight::Quantized(_) => None,
        }
    }

    fn dense_weight(tensor: CpuTensor) -> CpuLinearWeight {
        CpuLinearWeight::new_dense(tensor)
    }

    fn is_dense_weight(weight: &CpuLinearWeight) -> bool {
        matches!(weight, CpuLinearWeight::Dense { .. })
    }

    fn quantize_to_q8(_device: &(), shape: &[usize], data: &[f32]) -> Result<CpuLinearWeight> {
        assert!(shape.len() == 2, "quantize_to_q8: expected 2D shape");
        let n = shape[0]; // out_features
        let k = shape[1]; // in_features
        assert!(
            k.is_multiple_of(QUANTIZATION_BLOCK_SIZE),
            "quantize_to_q8: in_features {k} must be a multiple of {QUANTIZATION_BLOCK_SIZE}"
        );
        let num_blocks_per_row = k / QUANTIZATION_BLOCK_SIZE;

        let mut qdata = Vec::with_capacity(n * k);
        let mut scales = Vec::with_capacity(n * num_blocks_per_row);

        for neuron in 0..n {
            let row = &data[neuron * k..(neuron + 1) * k];
            for blk in 0..num_blocks_per_row {
                let block =
                    &row[blk * QUANTIZATION_BLOCK_SIZE..(blk + 1) * QUANTIZATION_BLOCK_SIZE];
                let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
                let inv_scale = 1.0 / scale;

                // Store as f32 directly — no f16 round-trip needed for runtime quantization
                scales.push(scale);

                #[allow(clippy::cast_possible_truncation)]
                for &v in block {
                    let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    qdata.push(q.cast_unsigned());
                }
            }
        }

        Ok(CpuLinearWeight::Quantized(CpuQuantizedWeight {
            shape: shape.to_vec(),
            dtype: DType::Q8_0,
            data: qdata,
            scales,
            mins: None,
        }))
    }

    fn upload_host_linear(
        _device: &(),
        weight: &infernum::weights::host::HostLinearWeight,
    ) -> Result<CpuLinearWeight> {
        use infernum::weights::host::HostLinearWeight;

        match weight {
            HostLinearWeight::Dense(host_tensor) => {
                let f32_data = match host_tensor.dtype {
                    DType::F32 => bytemuck::cast_slice::<u8, f32>(&host_tensor.data).to_vec(),
                    DType::BF16 => {
                        let bf16s: &[half::bf16] = bytemuck::cast_slice(&host_tensor.data);
                        bf16s.iter().map(|v| v.to_f32()).collect()
                    }
                    DType::F16 => {
                        let f16s: &[half::f16] = bytemuck::cast_slice(&host_tensor.data);
                        f16s.iter().map(|v| v.to_f32()).collect()
                    }
                    other => {
                        return Err(infernum::Error::UnsupportedDtype(format!(
                            "upload_host_linear: unsupported dense dtype {other}"
                        )));
                    }
                };
                Ok(CpuLinearWeight::new_dense(CpuTensor::from_f32(
                    &host_tensor.shape,
                    &f32_data,
                )))
            }
            HostLinearWeight::Quantized(hq) => match hq.dtype {
                DType::Q8_0 | DType::Q4_0 => Ok(CpuLinearWeight::Quantized(CpuQuantizedWeight {
                    shape: hq.shape.clone(),
                    dtype: hq.dtype,
                    data: hq.data.clone(),
                    scales: crate::tensor::decode_f16_scales(&hq.scales),
                    mins: None,
                })),
                DType::Q4_1 => Ok(CpuLinearWeight::Quantized(CpuQuantizedWeight {
                    shape: hq.shape.clone(),
                    dtype: hq.dtype,
                    data: hq.data.clone(),
                    scales: crate::tensor::decode_f16_scales(&hq.scales),
                    mins: hq.qzeros.as_deref().map(crate::tensor::decode_f16_scales),
                })),
                other => Err(infernum::Error::UnsupportedDtype(format!(
                    "CPU backend does not support {other} quantized weights"
                ))),
            },
        }
    }

    fn linear_pair(
        input: &CpuTensor,
        w1: &CpuLinearWeight,
        w2: &CpuLinearWeight,
    ) -> Result<(CpuTensor, CpuTensor)> {
        // Fuse when both weights are quantized (the common decode path).
        if let (CpuLinearWeight::Quantized(q1), CpuLinearWeight::Quantized(q2)) = (w1, w2) {
            return quantized_linear_pair(input, q1, q2);
        }
        // Fallback: two separate calls.
        let a = Self::linear(input, w1)?;
        let b = Self::linear(input, w2)?;
        Ok((a, b))
    }

    fn linear_triple(
        input: &CpuTensor,
        w1: &CpuLinearWeight,
        w2: &CpuLinearWeight,
        w3: &CpuLinearWeight,
    ) -> Result<(CpuTensor, CpuTensor, CpuTensor)> {
        if let (
            CpuLinearWeight::Quantized(q1),
            CpuLinearWeight::Quantized(q2),
            CpuLinearWeight::Quantized(q3),
        ) = (w1, w2, w3)
        {
            return quantized_linear_triple(input, q1, q2, q3);
        }
        let a = Self::linear(input, w1)?;
        let (b, c) = Self::linear_pair(input, w2, w3)?;
        Ok((a, b, c))
    }
}

#[allow(clippy::many_single_char_names)]
impl MatmulExtOps for CpuBackend {
    fn matmul_bf16_f32(a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor> {
        // Cast both inputs to f32 then standard matmul
        let a_f32 = a.to_f32_vec();
        let b_f32 = b.to_f32_vec();

        let a_shape = a.shape();
        let b_shape = b.shape();
        let k = b_shape[0];
        let n = b_shape[1];
        let m: usize = a_shape[..a_shape.len() - 1].iter().product();

        let output = gemm(&a_f32, &b_f32, m, k, n);

        let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
        out_shape.push(n);
        Ok(CpuTensor::from_f32_vec(&out_shape, output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2x3_times_3x4() {
        // input: (2, 3), weight: (3, 4) → output: (2, 4)
        // Standard matrix multiply: C = A @ B
        #[rustfmt::skip]
        let input = CpuTensor::from_f32(&[2, 3], &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);
        // Weight is (K=3, N=4) — each column is a "neuron"
        #[rustfmt::skip]
        let weight = CpuTensor::from_f32(&[3, 4], &[
            1.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
        ]);
        let out = CpuBackend::matmul(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
        let data = out.as_f32_slice();
        // Row 0: [1*1+2*0+3*0, 1*0+2*1+3*0, 1*0+2*0+3*1, 1*1+2*1+3*1] = [1, 2, 3, 6]
        // Row 1: [4*1+5*0+6*0, 4*0+5*1+6*0, 4*0+5*0+6*1, 4*1+5*1+6*1] = [4, 5, 6, 15]
        assert_eq!(data, &[1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0]);
    }

    #[test]
    fn test_gemv_single_row() {
        // input: (1, 3), weight: (3, 2) → output: (1, 2)
        let input = CpuTensor::from_f32(&[1, 3], &[1.0, 2.0, 3.0]);
        #[rustfmt::skip]
        let weight = CpuTensor::from_f32(&[3, 2], &[
            1.0, 2.0,
            1.0, 2.0,
            1.0, 2.0,
        ]);
        let out = CpuBackend::matmul(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        // [1+2+3, 2+4+6] = [6, 12]
        assert_eq!(out.as_f32_slice(), &[6.0, 12.0]);
    }

    #[test]
    fn test_identity_matrix() {
        // A × I = A
        let a_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let input = CpuTensor::from_f32(&[3, 4], &a_data);
        #[rustfmt::skip]
        let identity = CpuTensor::from_f32(&[4, 4], &[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]);
        let out = CpuBackend::matmul(&input, &identity).unwrap();
        assert_eq!(out.shape(), &[3, 4]);
        assert_eq!(out.as_f32_slice(), &a_data);
    }

    #[test]
    fn test_column_vector_output() {
        // (4, 3) × (3, 1) → (4, 1)
        let input = CpuTensor::from_f32(&[4, 3], &[1.0; 12]);
        let weight = CpuTensor::from_f32(&[3, 1], &[1.0, 2.0, 3.0]);
        let out = CpuBackend::matmul(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[4, 1]);
        assert_eq!(out.as_f32_slice(), &[6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_non_simd_aligned_k() {
        // K=7 is not aligned to NEON (4) or AVX2 (8) width
        let k = 7;
        let m = 3;
        let n = 5;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let input = CpuTensor::from_f32(&[m, k], &a_data);
        let weight = CpuTensor::from_f32(&[k, n], &b_data);
        let out = CpuBackend::matmul(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[m, n]);

        // Verify against scalar reference
        let result = out.as_f32_slice();
        for row in 0..m {
            for col in 0..n {
                let mut expected = 0.0f64;
                for ki in 0..k {
                    expected += f64::from(a_data[row * k + ki]) * f64::from(b_data[ki * n + col]);
                }
                let diff = (f64::from(result[row * n + col]) - expected).abs();
                assert!(diff < 1e-4, "mismatch at [{row},{col}]: {diff}");
            }
        }
    }

    #[test]
    fn test_large_matrix_parallel() {
        // 64×960 × 960×960 — exercises Rayon parallel path and SIMD
        let m = 64;
        let k = 960;
        let n = 960;
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32) * 0.001).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 113) as f32) * 0.001).collect();
        let input = CpuTensor::from_f32(&[m, k], &a_data);
        let weight = CpuTensor::from_f32(&[k, n], &b_data);
        let out = CpuBackend::matmul(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[m, n]);

        // Spot-check a few elements against scalar reference
        let result = out.as_f32_slice();
        for &(row, col) in &[
            (0, 0),
            (0, n - 1),
            (m - 1, 0),
            (m - 1, n - 1),
            (m / 2, n / 2),
        ] {
            let mut expected = 0.0f64;
            for ki in 0..k {
                expected += f64::from(a_data[row * k + ki]) * f64::from(b_data[ki * n + col]);
            }
            let diff = (f64::from(result[row * n + col]) - expected).abs();
            assert!(
                diff < 0.5,
                "mismatch at [{row},{col}]: got={}, expected={expected}, diff={diff}",
                result[row * n + col]
            );
        }
    }

    /// Helper: manually quantize f32 values to Q8_0 format.
    /// Returns (data, scales) where data is int8 bytes and scales is Vec<f32>.
    fn manual_quantize_q8(values: &[f32]) -> (Vec<u8>, Vec<f32>) {
        assert_eq!(values.len() % 32, 0);
        let num_blocks = values.len() / 32;
        let mut data = Vec::with_capacity(num_blocks * 32);
        let mut scales = Vec::with_capacity(num_blocks);

        for blk in 0..num_blocks {
            let block = &values[blk * 32..(blk + 1) * 32];
            let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            let inv_scale = 1.0 / scale;

            scales.push(scale);

            for &v in block {
                let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
                data.push(q as u8);
            }
        }
        (data, scales)
    }

    /// Helper: manually quantize f32 values to Q4_0 format.
    /// Returns (data, scales) where data is packed nibbles and scales is Vec<f32>.
    fn manual_quantize_q4(values: &[f32]) -> (Vec<u8>, Vec<f32>) {
        assert_eq!(values.len() % 32, 0);
        let num_blocks = values.len() / 32;
        let mut data = Vec::with_capacity(num_blocks * 16);
        let mut scales = Vec::with_capacity(num_blocks);

        for blk in 0..num_blocks {
            let block = &values[blk * 32..(blk + 1) * 32];
            let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
            let inv_scale = 1.0 / scale;

            scales.push(scale);

            // Pack 32 values into 16 bytes (low nibble first, high nibble second)
            for i in 0..16 {
                let lo_val = (block[i] * inv_scale).round().clamp(-8.0, 7.0) as i8;
                let hi_val = (block[i + 16] * inv_scale).round().clamp(-8.0, 7.0) as i8;
                let lo_nibble = (lo_val + 8) as u8 & 0x0F;
                let hi_nibble = (hi_val + 8) as u8 & 0x0F;
                data.push(lo_nibble | (hi_nibble << 4));
            }
        }
        (data, scales)
    }

    #[test]
    fn test_quantized_linear_q8_gemv() {
        // Weight: 2 neurons, 64 inputs (2 blocks of 32 each)
        let n = 2;
        let k = 64;
        let weight_f32: Vec<f32> = (0..n * k).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let input_f32: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();

        // Compute expected output in f32
        let mut expected = vec![0.0f32; n];
        for neuron in 0..n {
            for j in 0..k {
                expected[neuron] += input_f32[j] * weight_f32[neuron * k + j];
            }
        }

        // Quantize weight to Q8_0
        let (qdata, qscales) = manual_quantize_q8(&weight_f32);
        let qweight = CpuQuantizedWeight {
            shape: vec![n, k],
            dtype: DType::Q8_0,
            data: qdata,
            scales: qscales,
            mins: None,
        };

        let input = CpuTensor::from_f32(&[1, k], &input_f32);
        let out = quantized_linear(&input, &qweight).unwrap();
        assert_eq!(out.shape(), &[1, n]);

        let result = out.as_f32_slice();
        for i in 0..n {
            let rel_err = if expected[i].abs() > 1e-6 {
                (result[i] - expected[i]).abs() / expected[i].abs()
            } else {
                (result[i] - expected[i]).abs()
            };
            assert!(
                rel_err < 0.02,
                "Q8_0 neuron {i}: got {}, expected {}, rel_err {rel_err}",
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_quantized_linear_q4_gemv() {
        let n = 2;
        let k = 64;
        let weight_f32: Vec<f32> = (0..n * k).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let input_f32: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();

        // Compute expected output in f32
        let mut expected = vec![0.0f32; n];
        for neuron in 0..n {
            for j in 0..k {
                expected[neuron] += input_f32[j] * weight_f32[neuron * k + j];
            }
        }

        // Quantize weight to Q4_0
        let (qdata, qscales) = manual_quantize_q4(&weight_f32);
        let qweight = CpuQuantizedWeight {
            shape: vec![n, k],
            dtype: DType::Q4_0,
            data: qdata,
            scales: qscales,
            mins: None,
        };

        let input = CpuTensor::from_f32(&[1, k], &input_f32);
        let out = quantized_linear(&input, &qweight).unwrap();
        assert_eq!(out.shape(), &[1, n]);

        let result = out.as_f32_slice();
        for i in 0..n {
            let rel_err = if expected[i].abs() > 1e-6 {
                (result[i] - expected[i]).abs() / expected[i].abs()
            } else {
                (result[i] - expected[i]).abs()
            };
            assert!(
                rel_err < 0.15,
                "Q4_0 neuron {i}: got {}, expected {}, rel_err {rel_err}",
                result[i],
                expected[i]
            );
        }
    }

    /// Manually quantize f32 weights to Q4_1 format (asymmetric, unsigned nibbles).
    fn manual_quantize_q4_1(data: &[f32]) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let block_size = 32;
        let num_blocks = data.len() / block_size;
        let mut packed = Vec::with_capacity(num_blocks * 16);
        let mut scales = Vec::with_capacity(num_blocks);
        let mut mins = Vec::with_capacity(num_blocks);

        for b in 0..num_blocks {
            let block = &data[b * block_size..(b + 1) * block_size];
            let min_val = block.iter().copied().fold(f32::INFINITY, f32::min);
            let max_val = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = if (max_val - min_val).abs() < f32::EPSILON {
                0.0
            } else {
                (max_val - min_val) / 15.0
            };
            scales.push(scale);
            mins.push(min_val);

            // Quantize to 4-bit unsigned [0,15]
            let mut nibbles = [0u8; 32];
            for i in 0..32 {
                nibbles[i] = if scale.abs() < f32::EPSILON {
                    0
                } else {
                    ((block[i] - min_val) / scale).round().clamp(0.0, 15.0) as u8
                };
            }
            // Pack: byte[i] = nibbles[i] | (nibbles[i+16] << 4)
            for i in 0..16 {
                packed.push(nibbles[i] | (nibbles[i + 16] << 4));
            }
        }
        (packed, scales, mins)
    }

    #[test]
    fn test_quantized_linear_q4_1() {
        let n = 4;
        let k = 64;
        let weight_f32: Vec<f32> = (0..n * k).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let input_f32: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();

        let mut expected = vec![0.0f32; n];
        for neuron in 0..n {
            for j in 0..k {
                expected[neuron] += input_f32[j] * weight_f32[neuron * k + j];
            }
        }

        let (qdata, qscales, qmins) = manual_quantize_q4_1(&weight_f32);
        let qweight = CpuQuantizedWeight {
            shape: vec![n, k],
            dtype: DType::Q4_1,
            data: qdata,
            scales: qscales,
            mins: Some(qmins),
        };

        let input = CpuTensor::from_f32(&[1, k], &input_f32);
        let out = quantized_linear(&input, &qweight).unwrap();
        assert_eq!(out.shape(), &[1, n]);

        let result = out.as_f32_slice();
        for i in 0..n {
            let rel_err = if expected[i].abs() > 1e-6 {
                (result[i] - expected[i]).abs() / expected[i].abs()
            } else {
                (result[i] - expected[i]).abs()
            };
            assert!(
                rel_err < 0.15,
                "Q4_1 neuron {i}: got {}, expected {}, rel_err {rel_err}",
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_linear_dispatch_dense() {
        let input = CpuTensor::from_f32(&[1, 3], &[1.0, 2.0, 3.0]);
        #[rustfmt::skip]
        let weight = CpuLinearWeight::new_dense(CpuTensor::from_f32(&[3, 2], &[
            1.0, 2.0,
            1.0, 2.0,
            1.0, 2.0,
        ]));
        let out = CpuBackend::linear(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(out.as_f32_slice(), &[6.0, 12.0]);
    }

    #[test]
    fn test_linear_dispatch_quantized_q8() {
        let k = 32;
        let n = 1;
        let weight_f32: Vec<f32> = (0..k).map(|i| i as f32 * 0.1).collect();
        let input_f32: Vec<f32> = vec![1.0; k];

        let expected: f32 = weight_f32.iter().sum();

        let (qdata, qscales) = manual_quantize_q8(&weight_f32);
        let weight = CpuLinearWeight::Quantized(CpuQuantizedWeight {
            shape: vec![n, k],
            dtype: DType::Q8_0,
            data: qdata,
            scales: qscales,
            mins: None,
        });

        let input = CpuTensor::from_f32(&[1, k], &input_f32);
        let out = CpuBackend::linear(&input, &weight).unwrap();
        assert_eq!(out.shape(), &[1, n]);
        let result = out.as_f32_slice()[0];
        let rel_err = (result - expected).abs() / expected.abs();
        assert!(
            rel_err < 0.02,
            "Q8 linear dispatch: got {result}, expected {expected}, rel_err {rel_err}"
        );
    }

    #[test]
    fn test_quantized_linear_q8_gemm() {
        // Test with M=2 (batch of 2 rows)
        let n = 2;
        let k = 32;
        let weight_f32: Vec<f32> = (0..n * k).map(|i| (i as f32 - 32.0) * 0.02).collect();
        let input_f32: Vec<f32> = (0..2 * k).map(|i| (i as f32) * 0.05).collect();

        let mut expected = vec![0.0f32; 2 * n];
        for row in 0..2 {
            for neuron in 0..n {
                for j in 0..k {
                    expected[row * n + neuron] +=
                        input_f32[row * k + j] * weight_f32[neuron * k + j];
                }
            }
        }

        let (qdata, qscales) = manual_quantize_q8(&weight_f32);
        let qweight = CpuQuantizedWeight {
            shape: vec![n, k],
            dtype: DType::Q8_0,
            data: qdata,
            scales: qscales,
            mins: None,
        };

        let input = CpuTensor::from_f32(&[2, k], &input_f32);
        let out = quantized_linear(&input, &qweight).unwrap();
        assert_eq!(out.shape(), &[2, n]);

        let result = out.as_f32_slice();
        for i in 0..2 * n {
            let rel_err = if expected[i].abs() > 1e-6 {
                (result[i] - expected[i]).abs() / expected[i].abs()
            } else {
                (result[i] - expected[i]).abs()
            };
            assert!(
                rel_err < 0.04,
                "Q8 GEMM element {i}: got {}, expected {}, rel_err {rel_err}",
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_quantize_to_q8_roundtrip() {
        let n = 2;
        let k = 64;
        let weight_f32: Vec<f32> = (0..n * k).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let input_f32: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();

        let mut expected = vec![0.0f32; n];
        for neuron in 0..n {
            for j in 0..k {
                expected[neuron] += input_f32[j] * weight_f32[neuron * k + j];
            }
        }

        let q_weight = CpuBackend::quantize_to_q8(&(), &[n, k], &weight_f32).unwrap();
        assert!(matches!(q_weight, CpuLinearWeight::Quantized(_)));

        let input = CpuTensor::from_f32(&[1, k], &input_f32);
        let out = CpuBackend::linear(&input, &q_weight).unwrap();
        assert_eq!(out.shape(), &[1, n]);

        let result = out.as_f32_slice();
        for i in 0..n {
            let rel_err = if expected[i].abs() > 1e-6 {
                (result[i] - expected[i]).abs() / expected[i].abs()
            } else {
                (result[i] - expected[i]).abs()
            };
            assert!(
                rel_err < 0.02,
                "quantize_to_q8 roundtrip neuron {i}: got {}, expected {}, rel_err {rel_err}",
                result[i],
                expected[i]
            );
        }
    }
}
