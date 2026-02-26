//! Fused attention kernels
//!
//! Single-kernel attention that eliminates transposes, repeat_kv, and
//! intermediate allocations from the hot path.
//!
//! - `fused_attention_decode`: for single-token decode (seq_q == 1)
//! - `fused_attention_prefill`: for multi-token prefill with causal mask

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::too_many_arguments,
    clippy::manual_div_ceil
)]

use std::ffi::c_void;

use cudarc::driver::{DevicePtr, DevicePtrMut, LaunchAsync, LaunchConfig};

use crate::cuda::CudaTensor;
use infernum::dtype::DType;
use infernum::tensor::Tensor;
use infernum::Result;

/// Kernel name suffix for dtype
fn kernel_suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => panic!("Unsupported dtype: {dtype:?}"),
    }
}

/// Fused decode attention kernel.
///
/// One block per query head. Each block:
///   1. Loads the Q vector for this head from shared memory.
///   2. Iterates over all `total_len` cached K positions, computing dot(Q, K)
///      with GQA head mapping, and tracking the online softmax max/sum.
///   3. Iterates over all cached V positions, accumulating the weighted output.
///
/// Uses a two-pass approach (scores then weighted V) with shared memory for
/// the full score vector. This is simple and correct; a single-pass online
/// approach would be more memory-efficient for very long contexts but adds
/// complexity.
///
/// Inputs (all in native `(seq, heads, dim)` layout, no transposes needed):
///   - Q: `(1, num_heads, head_dim)` — the single query token
///   - K: `(total_len, num_kv_heads, head_dim)` — full cached keys
///   - V: `(total_len, num_kv_heads, head_dim)` — full cached values
///
/// Output: `(1, num_heads, head_dim)`
const FUSED_DECODE_PTX: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "/kernels/fused_decode_attention.ptx"
));

/// Fused prefill attention kernel with causal masking.
///
/// One block per `(head, query_position)` pair. Each block computes the full
/// attention output for one query position against all valid key positions
/// `[0 .. offset + query_pos + 1)`.
///
/// Handles GQA natively via `kv_head = head * num_kv_heads / num_heads`.
///
/// Inputs (all in native `(seq, heads, dim)` layout):
///   - Q: `(seq_q, num_heads, head_dim)`
///   - K: `(total_len, num_kv_heads, head_dim)` — includes prefill tokens
///   - V: `(total_len, num_kv_heads, head_dim)`
///
/// Output: `(seq_q, num_heads, head_dim)`
const FUSED_PREFILL_PTX: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "/kernels/fused_prefill_attention.ptx"
));

const FUSED_DECODE_KERNEL_NAMES: &[&str] = &[
    "fused_decode_attention_f32",
    "fused_decode_attention_f16",
    "fused_decode_attention_bf16",
];

const FUSED_PREFILL_KERNEL_NAMES: &[&str] = &[
    "fused_prefill_attention_f32",
    "fused_prefill_attention_f16",
    "fused_prefill_attention_bf16",
];

fn ensure_fused_decode_kernel(device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<()> {
    let module_name = "fused_decode_attention";
    if !device.has_func(module_name, "fused_decode_attention_f32") {
        let all_names: Vec<&str> = FUSED_DECODE_KERNEL_NAMES
            .iter()
            .chain(FUSED_DECODE_INDIRECT_KERNEL_NAMES.iter())
            .copied()
            .collect();
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(FUSED_DECODE_PTX),
            module_name,
            &all_names,
        )?;
    }
    Ok(())
}

fn ensure_fused_prefill_kernel(device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<()> {
    let module_name = "fused_prefill_attention";
    if !device.has_func(module_name, "fused_prefill_attention_f32") {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(FUSED_PREFILL_PTX),
            module_name,
            FUSED_PREFILL_KERNEL_NAMES,
        )?;
    }
    Ok(())
}

/// Fused attention for single-token decode.
///
/// Computes `softmax(Q @ K^T * scale) @ V` in a single kernel, reading
/// Q/K/V in their native `(seq, heads, dim)` layout. Handles GQA natively —
/// no `repeat_kv` or transpose needed.
///
/// # Arguments
/// * `q` — query tensor of shape `(1, num_heads, head_dim)`
/// * `k` — full cached keys of shape `(total_len, num_kv_heads, head_dim)`
/// * `v` — full cached values of shape `(total_len, num_kv_heads, head_dim)`
/// * `scale` — attention scale; `None` uses the default `1/sqrt(head_dim)`
/// * `softcap` — if `Some(cap)`, applies `tanh(score/cap)*cap` after scaling
/// * `sliding_window` — if `Some(w)`, restrict attention to the last `w` positions
///
/// # Returns
/// Output tensor of shape `(1, num_heads, head_dim)`
///
/// # Errors
/// Returns an error if the kernel launch fails
pub fn fused_attention_decode(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    softcap: Option<f32>,
    sliding_window: Option<usize>,
) -> Result<CudaTensor> {
    let dtype = q.dtype();
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    assert_eq!(q_shape.len(), 3, "Q must be 3D: (1, num_heads, head_dim)");
    assert_eq!(q_shape[0], 1, "Q seq_len must be 1 for decode");
    assert_eq!(
        k_shape.len(),
        3,
        "K must be 3D: (total_len, num_kv_heads, head_dim)"
    );
    assert_eq!(
        v_shape.len(),
        3,
        "V must be 3D: (total_len, num_kv_heads, head_dim)"
    );

    let num_heads = q_shape[1];
    let head_dim = q_shape[2];
    let total_len = k_shape[0];
    let num_kv_heads = k_shape[1];

    assert_eq!(k_shape[2], head_dim, "K head_dim must match Q");
    assert_eq!(v_shape, k_shape, "V shape must match K");
    assert!(
        num_heads.is_multiple_of(num_kv_heads),
        "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    );

    let scale = scale.unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
    let softcap_val = softcap.unwrap_or(0.0);
    let output_shape = [1, num_heads, head_dim];
    let mut output = unsafe { CudaTensor::uninit(q.context(), &output_shape, dtype)? };

    let device = q.context().device();
    ensure_fused_decode_kernel(device)?;

    let kernel_name = format!("fused_decode_attention_{}", kernel_suffix(dtype));
    let func = device
        .get_func("fused_decode_attention", &kernel_name)
        .unwrap();

    let block_size = 256_usize.min(total_len.next_power_of_two());
    // Shared memory: Q (head_dim) + cached weights (total_len) + reduction scratch (block_size)
    let shared_mem = (head_dim + total_len + block_size) * std::mem::size_of::<f32>();

    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    let window_size = sliding_window.map_or(-1, |w| w as i32);

    unsafe {
        func.launch(
            cfg,
            (
                output.cuda_slice_mut(),
                &q.cuda_slice(),
                &k.cuda_slice(),
                &v.cuda_slice(),
                scale,
                softcap_val,
                total_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                window_size,
            ),
        )?;
    }

    Ok(output)
}

/// Fused attention for multi-token prefill with causal masking.
///
/// Computes `softmax(Q @ K^T * scale, causal_mask) @ V` in a single kernel.
/// Handles GQA natively. Query position `i` attends to key positions
/// `[0 .. offset + i + 1)`, further restricted by `sliding_window`.
///
/// # Arguments
/// * `q` — query tensor of shape `(seq_q, num_heads, head_dim)`
/// * `k` — full cached keys of shape `(total_len, num_kv_heads, head_dim)`
/// * `v` — full cached values of shape `(total_len, num_kv_heads, head_dim)`
/// * `offset` — position offset for causal mask (from KV cache)
/// * `scale` — attention scale; `None` uses the default `1/sqrt(head_dim)`
/// * `softcap` — if `Some(cap)`, applies `tanh(score/cap)*cap` after scaling
/// * `sliding_window` — if `Some(w)`, restrict attention to the last `w` positions
///
/// # Returns
/// Output tensor of shape `(seq_q, num_heads, head_dim)`
///
/// # Errors
/// Returns an error if the kernel launch fails
pub fn fused_attention_prefill(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    offset: usize,
    scale: Option<f32>,
    softcap: Option<f32>,
    sliding_window: Option<usize>,
) -> Result<CudaTensor> {
    let dtype = q.dtype();
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    assert_eq!(
        q_shape.len(),
        3,
        "Q must be 3D: (seq_q, num_heads, head_dim)"
    );
    assert_eq!(
        k_shape.len(),
        3,
        "K must be 3D: (total_len, num_kv_heads, head_dim)"
    );
    assert_eq!(
        v_shape.len(),
        3,
        "V must be 3D: (total_len, num_kv_heads, head_dim)"
    );

    let seq_q = q_shape[0];
    let num_heads = q_shape[1];
    let head_dim = q_shape[2];
    let total_len = k_shape[0];
    let num_kv_heads = k_shape[1];

    assert_eq!(k_shape[2], head_dim, "K head_dim must match Q");
    assert_eq!(v_shape, k_shape, "V shape must match K");
    assert!(
        num_heads.is_multiple_of(num_kv_heads),
        "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    );

    let scale = scale.unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
    let softcap_val = softcap.unwrap_or(0.0);
    let output_shape = [seq_q, num_heads, head_dim];
    let mut output = unsafe { CudaTensor::uninit(q.context(), &output_shape, dtype)? };

    let device = q.context().device();
    ensure_fused_prefill_kernel(device)?;

    let kernel_name = format!("fused_prefill_attention_{}", kernel_suffix(dtype));
    let func = device
        .get_func("fused_prefill_attention", &kernel_name)
        .unwrap();

    let block_size = 256_usize.min(total_len.next_power_of_two());
    let shared_mem = (head_dim + block_size) * std::mem::size_of::<f32>();

    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, seq_q as u32, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    let window_size = sliding_window.map_or(-1, |w| w as i32);

    unsafe {
        func.launch(
            cfg,
            (
                output.cuda_slice_mut(),
                &q.cuda_slice(),
                &k.cuda_slice(),
                &v.cuda_slice(),
                scale,
                softcap_val,
                total_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                offset as i32,
                window_size,
            ),
        )?;
    }

    Ok(output)
}

const FUSED_DECODE_INDIRECT_KERNEL_NAMES: &[&str] = &[
    "fused_decode_attention_indirect_f32",
    "fused_decode_attention_indirect_f16",
    "fused_decode_attention_indirect_bf16",
];

/// Fused decode attention using a GPU-resident total length.
///
/// Identical to [`fused_attention_decode`] but reads `total_len` from the
/// [`SeqPosition`]'s device pointer instead of a host scalar. The grid and
/// shared-memory allocation use the host-side value for sizing, while the
/// kernel reads the actual loop bound from GPU memory at execution time.
///
/// This makes the kernel capturable by a CUDA graph: the graph references
/// fixed device addresses, and only the values at those addresses change
/// between replays.
///
/// # Arguments
/// * `q` — query tensor of shape `(1, num_heads, head_dim)`
/// * `k` — full cached keys of shape `(max_seq_len, num_kv_heads, head_dim)`
/// * `v` — full cached values of shape `(max_seq_len, num_kv_heads, head_dim)`
/// * `position` — GPU-resident total sequence length
/// * `max_seq_len` — maximum sequence length (for shared-memory sizing)
/// * `scale` — attention scale; `None` uses the default `1/sqrt(head_dim)`
/// * `softcap` — if `Some(cap)`, applies `tanh(score/cap)*cap` after scaling
/// * `sliding_window` — if `Some(w)`, restrict attention to the last `w` positions
///
/// # Errors
/// Returns an error if the kernel launch fails
pub fn fused_attention_decode_indirect(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    position: &crate::cuda::SeqPosition,
    max_seq_len: usize,
    scale: Option<f32>,
    softcap: Option<f32>,
    sliding_window: Option<usize>,
) -> Result<CudaTensor> {
    let dtype = q.dtype();
    let q_shape = q.shape();
    assert_eq!(q_shape.len(), 3, "Q must be 3D: (1, num_heads, head_dim)");
    assert_eq!(q_shape[0], 1, "Q seq_len must be 1 for decode");

    let num_heads = q_shape[1];
    let head_dim = q_shape[2];
    let k_shape = k.shape();
    let num_kv_heads = k_shape[1];

    assert!(
        num_heads.is_multiple_of(num_kv_heads),
        "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    );

    let scale = scale.unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
    let softcap_val = softcap.unwrap_or(0.0);
    let output_shape = [1, num_heads, head_dim];
    let mut output = unsafe { CudaTensor::uninit(q.context(), &output_shape, dtype)? };

    let device = q.context().device();

    // Ensure both standard and indirect kernels are loaded
    let module_name = "fused_decode_attention";
    let indirect_name = format!("fused_decode_attention_indirect_{}", kernel_suffix(dtype));
    if !device.has_func(module_name, &indirect_name) {
        let all_names: Vec<&str> = FUSED_DECODE_KERNEL_NAMES
            .iter()
            .chain(FUSED_DECODE_INDIRECT_KERNEL_NAMES.iter())
            .copied()
            .collect();
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(FUSED_DECODE_PTX),
            module_name,
            &all_names,
        )?;
    }

    let func = device.get_func(module_name, &indirect_name).unwrap();

    // Use max_seq_len for block sizing and shared memory so the graph is valid
    // for all possible total_len values up to max_seq_len.
    let block_size = 256_usize.min(max_seq_len.next_power_of_two());
    let shared_mem = (head_dim + max_seq_len + block_size) * std::mem::size_of::<f32>();

    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    let window_size = sliding_window.map_or(-1, |w| w as i32);

    unsafe {
        func.launch(
            cfg,
            (
                output.cuda_slice_mut(),
                &q.cuda_slice(),
                &k.cuda_slice(),
                &v.cuda_slice(),
                scale,
                softcap_val,
                position.device(),
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                window_size,
            ),
        )?;
    }

    Ok(output)
}

// Register the fused attention as a replacement for the decomposed attention_kv block.
infernum_macros::define_fusion! {
    name: "attention_kv",
    fn attention_kv_fused(
        q: &CudaTensor,
        kv_cache: &mut crate::cuda::KvCache,
        layer_idx: usize,
        k_new: &CudaTensor,
        v_new: &CudaTensor,
    ) -> crate::Result<CudaTensor> {
        let q_shape = q.shape();
        let new_seq_len = q_shape[0];

        // Append new K/V to cache (writes at current_len offset, does NOT advance)
        kv_cache.append(layer_idx, k_new, v_new)?;

        // Retrieve full cached K/V including the just-appended tokens
        let total_len = kv_cache.current_len() + new_seq_len;
        let (k_full, v_full) = kv_cache.get_up_to(layer_idx, total_len);

        if new_seq_len == 1 {
            fused_attention_decode(q, &k_full, &v_full, None, None, None)
        } else {
            fused_attention_prefill(q, &k_full, &v_full, kv_cache.current_len(), None, None, None)
        }
    }
}

// ---------------------------------------------------------------------------
// Prefill attention with log-sum-exp output (for multi-chunk prefill combining)
// ---------------------------------------------------------------------------

const FUSED_PREFILL_WITH_LSE_PTX: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "/kernels/fused_prefill_attention_with_lse.ptx"
));

const FUSED_PREFILL_WITH_LSE_KERNEL_NAMES: &[&str] = &[
    "fused_prefill_attention_with_lse_f32",
    "fused_prefill_attention_with_lse_f16",
    "fused_prefill_attention_with_lse_bf16",
];

fn ensure_fused_prefill_with_lse_kernel(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<()> {
    let module_name = "fused_prefill_attention_with_lse";
    if !device.has_func(module_name, "fused_prefill_attention_with_lse_f32") {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(FUSED_PREFILL_WITH_LSE_PTX),
            module_name,
            FUSED_PREFILL_WITH_LSE_KERNEL_NAMES,
        )?;
    }
    Ok(())
}

/// Fused attention for multi-token prefill with causal masking, returning
/// both the attention output and per-head log-sum-exp (LSE).
///
/// Identical to [`fused_attention_prefill`] but additionally outputs
/// `lse[qpos, head] = log(sum(exp(scores)))` for numerically-stable
/// combining of attention over disjoint key ranges (multi-chunk prefill).
///
/// # Arguments
/// * `q` — query tensor of shape `(seq_q, num_heads, head_dim)`
/// * `k` — full keys of shape `(total_len, num_kv_heads, head_dim)`
/// * `v` — full values of shape `(total_len, num_kv_heads, head_dim)`
/// * `offset` — position offset for causal mask
/// * `scale` — attention scale; `None` uses `1/sqrt(head_dim)`
/// * `softcap` — optional logit soft-capping
/// * `sliding_window` — optional sliding window size
///
/// # Returns
/// `(output, lse)` where output is `(seq_q, num_heads, head_dim)` and
/// lse is `(seq_q, num_heads)` as f32.
///
/// # Errors
/// Returns an error if the kernel launch fails
pub fn fused_attention_prefill_with_lse(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    offset: usize,
    scale: Option<f32>,
    softcap: Option<f32>,
    sliding_window: Option<usize>,
) -> Result<(CudaTensor, CudaTensor)> {
    let dtype = q.dtype();
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    assert_eq!(
        q_shape.len(),
        3,
        "Q must be 3D: (seq_q, num_heads, head_dim)"
    );
    assert_eq!(
        k_shape.len(),
        3,
        "K must be 3D: (total_len, num_kv_heads, head_dim)"
    );
    assert_eq!(
        v_shape.len(),
        3,
        "V must be 3D: (total_len, num_kv_heads, head_dim)"
    );

    let seq_q = q_shape[0];
    let num_heads = q_shape[1];
    let head_dim = q_shape[2];
    let total_len = k_shape[0];
    let num_kv_heads = k_shape[1];

    assert_eq!(k_shape[2], head_dim, "K head_dim must match Q");
    assert_eq!(v_shape, k_shape, "V shape must match K");
    assert!(
        num_heads.is_multiple_of(num_kv_heads),
        "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    );

    let mut scale = scale.unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
    let mut softcap_val = softcap.unwrap_or(0.0);
    let mut output =
        unsafe { CudaTensor::uninit(q.context(), &[seq_q, num_heads, head_dim], dtype)? };
    let mut lse = unsafe { CudaTensor::uninit(q.context(), &[seq_q, num_heads], DType::F32)? };

    let device = q.context().device();
    ensure_fused_prefill_with_lse_kernel(device)?;

    let kernel_name = format!("fused_prefill_attention_with_lse_{}", kernel_suffix(dtype));
    let func = device
        .get_func("fused_prefill_attention_with_lse", &kernel_name)
        .unwrap();

    let block_size = 256_usize.min(total_len.next_power_of_two());
    let shared_mem = (head_dim + block_size) * std::mem::size_of::<f32>();

    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, seq_q as u32, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    let mut window_size = sliding_window.map_or(-1, |w| w as i32);
    let mut total_len_i32 = total_len as i32;
    let mut num_heads_i32 = num_heads as i32;
    let mut num_kv_heads_i32 = num_kv_heads as i32;
    let mut head_dim_i32 = head_dim as i32;
    let mut offset_i32 = offset as i32;

    let out_slice = output.cuda_slice_mut();
    let lse_slice = lse.cuda_slice_mut();
    let q_slice = q.cuda_slice();
    let k_slice = k.cuda_slice();
    let v_slice = v.cuda_slice();

    let mut args: Vec<*mut c_void> = vec![
        std::ptr::from_mut(out_slice.device_ptr_mut()).cast::<c_void>(),
        std::ptr::from_mut(lse_slice.device_ptr_mut()).cast::<c_void>(),
        std::ptr::from_ref(q_slice.device_ptr())
            .cast_mut()
            .cast::<c_void>(),
        std::ptr::from_ref(k_slice.device_ptr())
            .cast_mut()
            .cast::<c_void>(),
        std::ptr::from_ref(v_slice.device_ptr())
            .cast_mut()
            .cast::<c_void>(),
        (&raw mut scale).cast::<c_void>(),
        (&raw mut softcap_val).cast::<c_void>(),
        (&raw mut total_len_i32).cast::<c_void>(),
        (&raw mut num_heads_i32).cast::<c_void>(),
        (&raw mut num_kv_heads_i32).cast::<c_void>(),
        (&raw mut head_dim_i32).cast::<c_void>(),
        (&raw mut offset_i32).cast::<c_void>(),
        (&raw mut window_size).cast::<c_void>(),
    ];

    unsafe {
        func.launch(cfg, &mut args)?;
    }

    Ok((output, lse))
}

// ---------------------------------------------------------------------------
// Combine two attention outputs using log-sum-exp correction
// ---------------------------------------------------------------------------

const COMBINE_LSE_PTX: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "/kernels/combine_attention_lse.ptx"
));

const COMBINE_LSE_KERNEL_NAMES: &[&str] = &[
    "combine_attention_lse_f32",
    "combine_attention_lse_f16",
    "combine_attention_lse_bf16",
];

fn ensure_combine_lse_kernel(device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<()> {
    let module_name = "combine_attention_lse";
    if !device.has_func(module_name, "combine_attention_lse_f32") {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(COMBINE_LSE_PTX),
            module_name,
            COMBINE_LSE_KERNEL_NAMES,
        )?;
    }
    Ok(())
}

/// Combine two attention outputs using numerically-stable log-sum-exp correction.
///
/// Given two `(output, lse)` pairs from attention over disjoint key ranges,
/// produces the combined result as if attention had been computed over the
/// union of both key ranges.
///
/// # Arguments
/// * `out1` — attention output from range 1: `(N, num_heads, head_dim)`
/// * `lse1` — per-head LSE from range 1: `(N, num_heads)`, f32
/// * `out2` — attention output from range 2: `(N, num_heads, head_dim)`
/// * `lse2` — per-head LSE from range 2: `(N, num_heads)`, f32
///
/// # Returns
/// Combined output: `(N, num_heads, head_dim)`
///
/// # Errors
/// Returns an error if the kernel launch fails
pub fn combine_attention_with_lse(
    out1: &CudaTensor,
    lse1: &CudaTensor,
    out2: &CudaTensor,
    lse2: &CudaTensor,
) -> Result<CudaTensor> {
    let dtype = out1.dtype();
    let shape1 = out1.shape();
    let shape2 = out2.shape();
    assert_eq!(shape1.len(), 3, "out1 must be 3D: (N, num_heads, head_dim)");
    assert_eq!(shape1, shape2, "out1 and out2 must have the same shape");
    assert_eq!(
        lse1.shape(),
        lse2.shape(),
        "lse1 and lse2 must have the same shape"
    );

    let n = shape1[0];
    let num_heads = shape1[1];
    let head_dim = shape1[2];

    assert_eq!(
        lse1.shape(),
        &[n, num_heads],
        "LSE shape must be (N, num_heads)"
    );

    let mut combined = unsafe { CudaTensor::uninit(out1.context(), shape1, dtype)? };

    let device = out1.context().device();
    ensure_combine_lse_kernel(device)?;

    let kernel_name = format!("combine_attention_lse_{}", kernel_suffix(dtype));
    let func = device
        .get_func("combine_attention_lse", &kernel_name)
        .unwrap();

    let block_size = 256_usize.min(head_dim.next_power_of_two());

    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, n as u32, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(
            cfg,
            (
                combined.cuda_slice_mut(),
                &out1.cuda_slice(),
                &lse1.cuda_slice(),
                &out2.cuda_slice(),
                &lse2.cuda_slice(),
                num_heads as i32,
                head_dim as i32,
            ),
        )?;
    }

    Ok(combined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::ops::attention;
    use crate::cuda::{CudaContext, KvCache};

    /// Reference decode attention using manual transpose+matmul+softmax.
    ///
    /// Independent of the fused kernel so we can compare against it.
    fn reference_attention_kv(
        q: &CudaTensor,
        k_full: &CudaTensor,
        v_full: &CudaTensor,
        num_heads: usize,
    ) -> CudaTensor {
        use super::super::{matmul, repeat_kv, softmax, transpose_012_to_102, transpose_last_two};

        let num_kv_heads = k_full.shape()[1];
        let head_dim = q.shape()[2];
        let scale = 1.0 / (head_dim as f32).sqrt();

        let (k_exp, v_exp) = if num_kv_heads < num_heads {
            let repeats = num_heads / num_kv_heads;
            (
                repeat_kv(k_full, repeats).unwrap(),
                repeat_kv(v_full, repeats).unwrap(),
            )
        } else {
            (
                k_full.reshape(k_full.shape()),
                v_full.reshape(v_full.shape()),
            )
        };

        // Transpose to (heads, seq, dim)
        let q_t = transpose_012_to_102(q).unwrap();
        let k_t = transpose_012_to_102(&k_exp).unwrap();
        let v_t = transpose_012_to_102(&v_exp).unwrap();
        let k_tt = transpose_last_two(&k_t).unwrap();

        // scores: (heads, 1, total_len) — scale via CPU roundtrip
        let scores = matmul(&q_t, &k_tt).unwrap();
        let mut scores_cpu = scores.to_vec::<f32>().unwrap();
        for val in &mut scores_cpu {
            *val *= scale;
        }
        let scores_scaled =
            CudaTensor::from_slice(q.context(), scores.shape(), &scores_cpu).unwrap();

        // Softmax over last dim
        let probs = softmax(&scores_scaled).unwrap();

        // Output: (heads, 1, dim)
        let output = matmul(&probs, &v_t).unwrap();
        transpose_012_to_102(&output).unwrap()
    }

    #[test]
    fn test_fused_decode_basic() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let head_dim = 4;
        let total_len = 3;

        let q_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|x| (x as f32) * 0.1)
            .collect();
        let k_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| (x as f32) * 0.05)
            .collect();
        let v_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| (x as f32) * 0.02)
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &v_data).unwrap();

        let fused = fused_attention_decode(&q, &k, &v, None, None, None).unwrap();
        let reference = reference_attention_kv(&q, &k, &v, num_heads);

        assert_eq!(fused.shape(), &[1, num_heads, head_dim]);

        let fused_data = fused.to_vec::<f32>().unwrap();
        let ref_data = reference.to_vec::<f32>().unwrap();

        for (i, (&f, &r)) in fused_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-3,
                "Mismatch at {i}: fused={f}, reference={r}"
            );
        }
    }

    #[test]
    fn test_fused_decode_gqa() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let total_len = 5;

        let q_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|x| (x as f32) * 0.1)
            .collect();
        let k_data: Vec<f32> = (0..total_len * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.05)
            .collect();
        let v_data: Vec<f32> = (0..total_len * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.02)
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();
        let k =
            CudaTensor::from_slice(&ctx, &[total_len, num_kv_heads, head_dim], &k_data).unwrap();
        let v =
            CudaTensor::from_slice(&ctx, &[total_len, num_kv_heads, head_dim], &v_data).unwrap();

        let fused = fused_attention_decode(&q, &k, &v, None, None, None).unwrap();
        let reference = reference_attention_kv(&q, &k, &v, num_heads);

        let fused_data = fused.to_vec::<f32>().unwrap();
        let ref_data = reference.to_vec::<f32>().unwrap();

        for (i, (&f, &r)) in fused_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-3,
                "GQA mismatch at {i}: fused={f}, reference={r}"
            );
        }
    }

    #[test]
    fn test_fused_decode_single_token() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let q = CudaTensor::from_slice(&ctx, &[1, 1, 4], &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[1, 1, 4], &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[1, 1, 4], &[0.5, 0.5, 0.5, 0.5]).unwrap();

        let output = fused_attention_decode(&q, &k, &v, None, None, None).unwrap();
        let result = output.to_vec::<f32>().unwrap();

        // Only one key position, so output == V
        for (i, &val) in result.iter().enumerate() {
            assert!((val - 0.5).abs() < 1e-4, "Mismatch at {i}: {val} vs 0.5");
        }
    }

    #[test]
    fn test_fused_prefill_basic() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let head_dim = 4;
        let seq_q = 3;

        let q_data: Vec<f32> = (0..seq_q * num_heads * head_dim)
            .map(|x| (x as f32) * 0.1)
            .collect();
        let k_data = q_data.clone();
        let v_data: Vec<f32> = (0..seq_q * num_heads * head_dim)
            .map(|x| (x as f32) * 0.02)
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[seq_q, num_heads, head_dim], &q_data).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[seq_q, num_heads, head_dim], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[seq_q, num_heads, head_dim], &v_data).unwrap();

        let fused = fused_attention_prefill(&q, &k, &v, 0, None, None, None).unwrap();
        let reference = attention(&q, &k, &v, true).unwrap();

        assert_eq!(fused.shape(), &[seq_q, num_heads, head_dim]);

        let fused_data = fused.to_vec::<f32>().unwrap();
        let ref_data = reference.to_vec::<f32>().unwrap();

        for (i, (&f, &r)) in fused_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-3,
                "Prefill mismatch at {i}: fused={f}, reference={r}"
            );
        }
    }

    #[test]
    fn test_fused_prefill_gqa() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let seq_q = 4;

        let q_data: Vec<f32> = (0..seq_q * num_heads * head_dim)
            .map(|x| (x as f32) * 0.05)
            .collect();
        let k_data: Vec<f32> = (0..seq_q * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.05)
            .collect();
        let v_data: Vec<f32> = (0..seq_q * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.02)
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[seq_q, num_heads, head_dim], &q_data).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[seq_q, num_kv_heads, head_dim], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[seq_q, num_kv_heads, head_dim], &v_data).unwrap();

        // Reference: expand KV then use causal attention
        let k_exp = super::super::repeat_kv(&k, num_heads / num_kv_heads).unwrap();
        let v_exp = super::super::repeat_kv(&v, num_heads / num_kv_heads).unwrap();
        let reference = attention(&q, &k_exp, &v_exp, true).unwrap();

        let fused = fused_attention_prefill(&q, &k, &v, 0, None, None, None).unwrap();

        let fused_data = fused.to_vec::<f32>().unwrap();
        let ref_data = reference.to_vec::<f32>().unwrap();

        for (i, (&f, &r)) in fused_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-3,
                "Prefill GQA mismatch at {i}: fused={f}, reference={r}"
            );
        }
    }

    /// CPU reference for causal attention with offset.
    /// Query position i attends to key positions `[0..offset+i+1)`.
    fn cpu_causal_attention(
        q: &[f32], // (seq_q, num_heads, head_dim)
        k: &[f32], // (total_len, num_kv_heads, head_dim)
        v: &[f32], // (total_len, num_kv_heads, head_dim)
        seq_q: usize,
        total_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        offset: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; seq_q * num_heads * head_dim];

        for qpos in 0..seq_q {
            for h in 0..num_heads {
                let kv_h = h * num_kv_heads / num_heads;
                let max_valid = (offset + qpos + 1).min(total_len);

                // Compute scores
                let mut scores = vec![f32::NEG_INFINITY; total_len];
                for t in 0..max_valid {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        let qi = q[qpos * num_heads * head_dim + h * head_dim + d];
                        let ki = k[t * num_kv_heads * head_dim + kv_h * head_dim + d];
                        dot += qi * ki;
                    }
                    scores[t] = dot * scale;
                }

                // Softmax over valid positions
                let max_s = scores[..max_valid]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for s in &mut scores[..max_valid] {
                    *s = (*s - max_s).exp();
                    sum_exp += *s;
                }
                for s in &mut scores[..max_valid] {
                    *s /= sum_exp;
                }
                for s in &mut scores[max_valid..] {
                    *s = 0.0;
                }

                // Weighted sum of V
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for t in 0..max_valid {
                        let vi = v[t * num_kv_heads * head_dim + kv_h * head_dim + d];
                        acc += scores[t] * vi;
                    }
                    output[qpos * num_heads * head_dim + h * head_dim + d] = acc;
                }
            }
        }
        output
    }

    #[test]
    fn test_fused_prefill_with_offset() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // Simulate: 3 tokens already in cache, then prefill 2 more
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let cache_len = 3;
        let new_seq = 2;

        let mut kv_cache = KvCache::new(&ctx, 1, 32, num_kv_heads, head_dim, DType::F32).unwrap();

        // Populate cache with initial tokens
        let k_init: Vec<f32> = (0..cache_len * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.05)
            .collect();
        let v_init: Vec<f32> = (0..cache_len * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.02)
            .collect();
        let k_init_t =
            CudaTensor::from_slice(&ctx, &[cache_len, num_kv_heads, head_dim], &k_init).unwrap();
        let v_init_t =
            CudaTensor::from_slice(&ctx, &[cache_len, num_kv_heads, head_dim], &v_init).unwrap();
        kv_cache.append(0, &k_init_t, &v_init_t).unwrap();
        kv_cache.advance(cache_len).unwrap();

        // New tokens
        let q_data: Vec<f32> = (0..new_seq * num_heads * head_dim)
            .map(|x| (x as f32) * 0.1)
            .collect();
        let k_new_data: Vec<f32> = (0..new_seq * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.03)
            .collect();
        let v_new_data: Vec<f32> = (0..new_seq * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.01)
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[new_seq, num_heads, head_dim], &q_data).unwrap();
        let k_new =
            CudaTensor::from_slice(&ctx, &[new_seq, num_kv_heads, head_dim], &k_new_data).unwrap();
        let v_new =
            CudaTensor::from_slice(&ctx, &[new_seq, num_kv_heads, head_dim], &v_new_data).unwrap();

        // Fused path: append to our cache, get full K/V, call fused_attention_prefill
        kv_cache.append(0, &k_new, &v_new).unwrap();
        let total_len = cache_len + new_seq;
        let (k_full, v_full) = kv_cache.get_up_to(0, total_len);
        let fused =
            fused_attention_prefill(&q, &k_full, &v_full, cache_len, None, None, None).unwrap();

        // CPU reference with offset-aware causal mask
        let k_full_cpu = k_full.to_vec::<f32>().unwrap();
        let v_full_cpu = v_full.to_vec::<f32>().unwrap();
        let ref_data = cpu_causal_attention(
            &q_data,
            &k_full_cpu,
            &v_full_cpu,
            new_seq,
            total_len,
            num_heads,
            num_kv_heads,
            head_dim,
            cache_len,
        );

        let fused_data = fused.to_vec::<f32>().unwrap();

        for (i, (&f, &r)) in fused_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-3,
                "Prefill+offset mismatch at {i}: fused={f}, reference={r}"
            );
        }
    }

    #[test]
    fn test_fused_decode_via_kv_cache() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 8;
        let prefill_len = 4;

        let mut kv_cache = KvCache::new(&ctx, 1, 32, num_kv_heads, head_dim, DType::F32).unwrap();

        // Prefill KV cache
        let kv_data: Vec<f32> = (0..prefill_len * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.01)
            .collect();
        let k =
            CudaTensor::from_slice(&ctx, &[prefill_len, num_kv_heads, head_dim], &kv_data).unwrap();
        let v =
            CudaTensor::from_slice(&ctx, &[prefill_len, num_kv_heads, head_dim], &kv_data).unwrap();
        kv_cache.append(0, &k, &v).unwrap();
        kv_cache.advance(prefill_len).unwrap();

        // Decode one token
        let q1_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|x| (x as f32) * 0.02)
            .collect();
        let kv1_data: Vec<f32> = (0..num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.02)
            .collect();
        let q1 = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q1_data).unwrap();
        let k1 = CudaTensor::from_slice(&ctx, &[1, num_kv_heads, head_dim], &kv1_data).unwrap();
        let v1 = CudaTensor::from_slice(&ctx, &[1, num_kv_heads, head_dim], &kv1_data).unwrap();

        // Append new KV and get full cache
        kv_cache.append(0, &k1, &v1).unwrap();
        let total_len = prefill_len + 1;
        let (k_full, v_full) = kv_cache.get_up_to(0, total_len);

        // Fused kernel
        let fused_output = fused_attention_decode(&q1, &k_full, &v_full, None, None, None).unwrap();

        // Independent reference (transpose + matmul + softmax)
        let ref_output = reference_attention_kv(&q1, &k_full, &v_full, num_heads);

        let fused_data = fused_output.to_vec::<f32>().unwrap();
        let ref_data = ref_output.to_vec::<f32>().unwrap();

        for (i, (&f, &r)) in fused_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-3,
                "Decode KV mismatch at {i}: fused={f}, reference={r}"
            );
        }
    }

    #[test]
    fn test_fused_decode_larger_context() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;
        let total_len = 64;

        let q_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|x| ((x as f32) * 0.01).sin())
            .collect();
        let k_data: Vec<f32> = (0..total_len * num_kv_heads * head_dim)
            .map(|x| ((x as f32) * 0.007).cos())
            .collect();
        let v_data: Vec<f32> = (0..total_len * num_kv_heads * head_dim)
            .map(|x| ((x as f32) * 0.003).sin())
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();
        let k =
            CudaTensor::from_slice(&ctx, &[total_len, num_kv_heads, head_dim], &k_data).unwrap();
        let v =
            CudaTensor::from_slice(&ctx, &[total_len, num_kv_heads, head_dim], &v_data).unwrap();

        let fused = fused_attention_decode(&q, &k, &v, None, None, None).unwrap();
        let reference = reference_attention_kv(&q, &k, &v, num_heads);

        let fused_data = fused.to_vec::<f32>().unwrap();
        let ref_data = reference.to_vec::<f32>().unwrap();

        for (i, (&f, &r)) in fused_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-2,
                "Large context mismatch at {i}: fused={f}, reference={r}"
            );
        }
    }

    /// Helper: build full-sized (max_seq_len) K/V buffers with real data only
    /// in the first `total_len` rows.
    fn make_full_kv(
        ctx: &CudaContext,
        max_seq_len: usize,
        total_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seed: f32,
    ) -> (CudaTensor, Vec<f32>) {
        let mut data = vec![0.0_f32; max_seq_len * num_kv_heads * head_dim];
        for i in 0..(total_len * num_kv_heads * head_dim) {
            data[i] = ((i as f32) * seed).sin();
        }
        let tensor =
            CudaTensor::from_slice(ctx, &[max_seq_len, num_kv_heads, head_dim], &data).unwrap();
        (tensor, data)
    }

    #[test]
    fn test_fused_decode_indirect_basic() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let head_dim = 4;
        let total_len = 3;
        let max_seq_len = 16;

        let q_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|x| (x as f32) * 0.1)
            .collect();
        let q = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();

        // Direct: K/V sized to total_len
        let k_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| (x as f32) * 0.05)
            .collect();
        let v_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| (x as f32) * 0.02)
            .collect();
        let k_direct =
            CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &k_data).unwrap();
        let v_direct =
            CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &v_data).unwrap();
        let direct = fused_attention_decode(&q, &k_direct, &v_direct, None, None, None).unwrap();

        // Indirect: K/V sized to max_seq_len, only first total_len rows filled
        let mut k_full_data = vec![0.0_f32; max_seq_len * num_heads * head_dim];
        let mut v_full_data = vec![0.0_f32; max_seq_len * num_heads * head_dim];
        k_full_data[..k_data.len()].copy_from_slice(&k_data);
        v_full_data[..v_data.len()].copy_from_slice(&v_data);
        let k_full =
            CudaTensor::from_slice(&ctx, &[max_seq_len, num_heads, head_dim], &k_full_data)
                .unwrap();
        let v_full =
            CudaTensor::from_slice(&ctx, &[max_seq_len, num_heads, head_dim], &v_full_data)
                .unwrap();

        let mut pos = crate::cuda::SeqPosition::new(ctx.device()).unwrap();
        pos.set(total_len, ctx.device()).unwrap();

        let indirect = fused_attention_decode_indirect(
            &q,
            &k_full,
            &v_full,
            &pos,
            max_seq_len,
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(indirect.shape(), &[1, num_heads, head_dim]);

        let direct_data = direct.to_vec::<f32>().unwrap();
        let indirect_data = indirect.to_vec::<f32>().unwrap();

        for (i, (&d, &ind)) in direct_data.iter().zip(indirect_data.iter()).enumerate() {
            assert!(
                (d - ind).abs() < 1e-3,
                "Mismatch at {i}: direct={d}, indirect={ind}"
            );
        }
    }

    #[test]
    fn test_fused_decode_indirect_gqa() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let total_len = 5;
        let max_seq_len = 32;

        let q_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|x| (x as f32) * 0.1)
            .collect();
        let q = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();

        // Direct
        let k_data: Vec<f32> = (0..total_len * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.05)
            .collect();
        let v_data: Vec<f32> = (0..total_len * num_kv_heads * head_dim)
            .map(|x| (x as f32) * 0.02)
            .collect();
        let k_direct =
            CudaTensor::from_slice(&ctx, &[total_len, num_kv_heads, head_dim], &k_data).unwrap();
        let v_direct =
            CudaTensor::from_slice(&ctx, &[total_len, num_kv_heads, head_dim], &v_data).unwrap();
        let direct = fused_attention_decode(&q, &k_direct, &v_direct, None, None, None).unwrap();

        // Indirect
        let mut k_full_data = vec![0.0_f32; max_seq_len * num_kv_heads * head_dim];
        let mut v_full_data = vec![0.0_f32; max_seq_len * num_kv_heads * head_dim];
        k_full_data[..k_data.len()].copy_from_slice(&k_data);
        v_full_data[..v_data.len()].copy_from_slice(&v_data);
        let k_full =
            CudaTensor::from_slice(&ctx, &[max_seq_len, num_kv_heads, head_dim], &k_full_data)
                .unwrap();
        let v_full =
            CudaTensor::from_slice(&ctx, &[max_seq_len, num_kv_heads, head_dim], &v_full_data)
                .unwrap();

        let mut pos = crate::cuda::SeqPosition::new(ctx.device()).unwrap();
        pos.set(total_len, ctx.device()).unwrap();

        let indirect = fused_attention_decode_indirect(
            &q,
            &k_full,
            &v_full,
            &pos,
            max_seq_len,
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(indirect.shape(), &[1, num_heads, head_dim]);

        let direct_data = direct.to_vec::<f32>().unwrap();
        let indirect_data = indirect.to_vec::<f32>().unwrap();

        for (i, (&d, &ind)) in direct_data.iter().zip(indirect_data.iter()).enumerate() {
            assert!(
                (d - ind).abs() < 1e-3,
                "GQA mismatch at {i}: direct={d}, indirect={ind}"
            );
        }
    }

    #[test]
    fn test_fused_decode_indirect_updates_with_position() {
        // Simulates graph replay: same buffers, only SeqPosition changes.
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let head_dim = 4;
        let max_seq_len = 64;

        let q_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|x| ((x as f32) * 0.1).sin())
            .collect();
        let q = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();

        let (k_full, _) = make_full_kv(&ctx, max_seq_len, max_seq_len, num_heads, head_dim, 0.05);
        let (v_full, _) = make_full_kv(&ctx, max_seq_len, max_seq_len, num_heads, head_dim, 0.02);

        let mut pos = crate::cuda::SeqPosition::new(ctx.device()).unwrap();

        // total_len = 3
        pos.set(3, ctx.device()).unwrap();
        let out3 = fused_attention_decode_indirect(
            &q,
            &k_full,
            &v_full,
            &pos,
            max_seq_len,
            None,
            None,
            None,
        )
        .unwrap();

        // total_len = 10
        pos.set(10, ctx.device()).unwrap();
        let out10 = fused_attention_decode_indirect(
            &q,
            &k_full,
            &v_full,
            &pos,
            max_seq_len,
            None,
            None,
            None,
        )
        .unwrap();

        let data3 = out3.to_vec::<f32>().unwrap();
        let data10 = out10.to_vec::<f32>().unwrap();

        // Attending to more tokens must produce different results
        assert_ne!(
            data3, data10,
            "Different total_len should produce different attention output"
        );

        // Each should match the direct version
        let k3 = CudaTensor::from_slice(
            &ctx,
            &[3, num_heads, head_dim],
            &k_full.to_vec::<f32>().unwrap()[..3 * num_heads * head_dim],
        )
        .unwrap();
        let v3 = CudaTensor::from_slice(
            &ctx,
            &[3, num_heads, head_dim],
            &v_full.to_vec::<f32>().unwrap()[..3 * num_heads * head_dim],
        )
        .unwrap();
        let direct3 = fused_attention_decode(&q, &k3, &v3, None, None, None).unwrap();

        let direct3_data = direct3.to_vec::<f32>().unwrap();
        for (i, (&a, &b)) in data3.iter().zip(direct3_data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "total_len=3 mismatch at {i}: indirect={a}, direct={b}"
            );
        }
    }

    /// CPU reference for decode attention with sliding window.
    fn cpu_decode_sliding_window(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        total_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        window: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let win_start = total_len.saturating_sub(window);
        let mut output = vec![0.0f32; num_heads * head_dim];

        for h in 0..num_heads {
            let kv_h = h * num_kv_heads / num_heads;
            let mut scores = Vec::new();
            for t in win_start..total_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let qi = q[h * head_dim + d];
                    let ki = k[t * num_kv_heads * head_dim + kv_h * head_dim + d];
                    dot += qi * ki;
                }
                scores.push(dot * scale);
            }
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum_exp += *s;
            }
            for s in &mut scores {
                *s /= sum_exp;
            }
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (idx, t) in (win_start..total_len).enumerate() {
                    let vi = v[t * num_kv_heads * head_dim + kv_h * head_dim + d];
                    acc += scores[idx] * vi;
                }
                output[h * head_dim + d] = acc;
            }
        }
        output
    }

    #[test]
    fn test_fused_decode_sliding_window() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let head_dim = 4;
        let total_len = 8;
        let window = 3;

        let q_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|x| ((x as f32) * 0.17).sin())
            .collect();
        let k_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| ((x as f32) * 0.11).cos())
            .collect();
        let v_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| ((x as f32) * 0.07).sin())
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &v_data).unwrap();

        let windowed = fused_attention_decode(&q, &k, &v, None, None, Some(window)).unwrap();
        let full = fused_attention_decode(&q, &k, &v, None, None, None).unwrap();

        let windowed_data = windowed.to_vec::<f32>().unwrap();
        let full_data = full.to_vec::<f32>().unwrap();

        // Windowed and full should differ (total_len > window)
        assert_ne!(
            windowed_data, full_data,
            "Windowed should differ from full attention"
        );

        // Verify against CPU reference
        let ref_data = cpu_decode_sliding_window(
            &q_data, &k_data, &v_data, total_len, num_heads, num_heads, head_dim, window,
        );

        for (i, (&w, &r)) in windowed_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (w - r).abs() < 1e-3,
                "SWA decode mismatch at {i}: gpu={w}, cpu={r}"
            );
        }
    }

    #[test]
    fn test_fused_decode_sliding_window_larger_than_seq() {
        // When window >= total_len, windowed should match full attention
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let head_dim = 4;
        let total_len = 5;
        let window = 10;

        let q_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|x| (x as f32) * 0.1)
            .collect();
        let k_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| (x as f32) * 0.05)
            .collect();
        let v_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| (x as f32) * 0.02)
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &v_data).unwrap();

        let windowed = fused_attention_decode(&q, &k, &v, None, None, Some(window)).unwrap();
        let full = fused_attention_decode(&q, &k, &v, None, None, None).unwrap();

        let windowed_data = windowed.to_vec::<f32>().unwrap();
        let full_data = full.to_vec::<f32>().unwrap();

        for (i, (&w, &f)) in windowed_data.iter().zip(full_data.iter()).enumerate() {
            assert!(
                (w - f).abs() < 1e-4,
                "Window >= seq should match full at {i}: windowed={w}, full={f}"
            );
        }
    }

    #[test]
    fn test_fused_prefill_sliding_window() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let head_dim = 4;
        let seq_q = 6;
        let window = 3;

        let q_data: Vec<f32> = (0..seq_q * num_heads * head_dim)
            .map(|x| ((x as f32) * 0.13).sin())
            .collect();
        let k_data = q_data.clone();
        let v_data: Vec<f32> = (0..seq_q * num_heads * head_dim)
            .map(|x| ((x as f32) * 0.07).cos())
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[seq_q, num_heads, head_dim], &q_data).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[seq_q, num_heads, head_dim], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[seq_q, num_heads, head_dim], &v_data).unwrap();

        let windowed = fused_attention_prefill(&q, &k, &v, 0, None, None, Some(window)).unwrap();
        let full = fused_attention_prefill(&q, &k, &v, 0, None, None, None).unwrap();

        let windowed_data = windowed.to_vec::<f32>().unwrap();
        let full_data = full.to_vec::<f32>().unwrap();

        // First `window` positions should be identical (window covers all valid keys)
        let first_window_elems = window * num_heads * head_dim;
        for (i, (&w, &f)) in windowed_data[..first_window_elems]
            .iter()
            .zip(full_data[..first_window_elems].iter())
            .enumerate()
        {
            assert!(
                (w - f).abs() < 1e-4,
                "First {window} rows should match: mismatch at {i}: windowed={w}, full={f}"
            );
        }

        // Later positions should differ (window restricts attention)
        let later = &windowed_data[first_window_elems..];
        let later_full = &full_data[first_window_elems..];
        assert_ne!(
            later, later_full,
            "Later positions should differ with sliding window"
        );
    }

    #[test]
    fn test_fused_decode_indirect_sliding_window() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let head_dim = 4;
        let total_len = 8;
        let max_seq_len = 16;
        let window = 3;

        let q_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|x| ((x as f32) * 0.17).sin())
            .collect();
        let q = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();

        // Direct
        let k_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| ((x as f32) * 0.11).cos())
            .collect();
        let v_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| ((x as f32) * 0.07).sin())
            .collect();
        let k_direct =
            CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &k_data).unwrap();
        let v_direct =
            CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &v_data).unwrap();
        let direct =
            fused_attention_decode(&q, &k_direct, &v_direct, None, None, Some(window)).unwrap();

        // Indirect
        let mut k_full_data = vec![0.0_f32; max_seq_len * num_heads * head_dim];
        let mut v_full_data = vec![0.0_f32; max_seq_len * num_heads * head_dim];
        k_full_data[..k_data.len()].copy_from_slice(&k_data);
        v_full_data[..v_data.len()].copy_from_slice(&v_data);
        let k_full =
            CudaTensor::from_slice(&ctx, &[max_seq_len, num_heads, head_dim], &k_full_data)
                .unwrap();
        let v_full =
            CudaTensor::from_slice(&ctx, &[max_seq_len, num_heads, head_dim], &v_full_data)
                .unwrap();

        let mut pos = crate::cuda::SeqPosition::new(ctx.device()).unwrap();
        pos.set(total_len, ctx.device()).unwrap();

        let indirect = fused_attention_decode_indirect(
            &q,
            &k_full,
            &v_full,
            &pos,
            max_seq_len,
            None,
            None,
            Some(window),
        )
        .unwrap();

        let direct_data = direct.to_vec::<f32>().unwrap();
        let indirect_data = indirect.to_vec::<f32>().unwrap();

        for (i, (&d, &ind)) in direct_data.iter().zip(indirect_data.iter()).enumerate() {
            assert!(
                (d - ind).abs() < 1e-3,
                "SWA indirect mismatch at {i}: direct={d}, indirect={ind}"
            );
        }
    }

    #[test]
    fn test_fused_decode_indirect_softcap() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let head_dim = 8;
        let total_len = 6;
        let max_seq_len = 16;
        let cap = 50.0_f32;

        let q_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|x| ((x as f32) * 0.3).sin())
            .collect();
        let q = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();

        let k_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| ((x as f32) * 0.2).cos())
            .collect();
        let v_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| ((x as f32) * 0.1).sin())
            .collect();

        // Direct
        let k_direct =
            CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &k_data).unwrap();
        let v_direct =
            CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &v_data).unwrap();
        let direct =
            fused_attention_decode(&q, &k_direct, &v_direct, None, Some(cap), None).unwrap();

        // Indirect
        let mut k_full_data = vec![0.0_f32; max_seq_len * num_heads * head_dim];
        let mut v_full_data = vec![0.0_f32; max_seq_len * num_heads * head_dim];
        k_full_data[..k_data.len()].copy_from_slice(&k_data);
        v_full_data[..v_data.len()].copy_from_slice(&v_data);
        let k_full =
            CudaTensor::from_slice(&ctx, &[max_seq_len, num_heads, head_dim], &k_full_data)
                .unwrap();
        let v_full =
            CudaTensor::from_slice(&ctx, &[max_seq_len, num_heads, head_dim], &v_full_data)
                .unwrap();

        let mut pos = crate::cuda::SeqPosition::new(ctx.device()).unwrap();
        pos.set(total_len, ctx.device()).unwrap();

        let indirect = fused_attention_decode_indirect(
            &q,
            &k_full,
            &v_full,
            &pos,
            max_seq_len,
            None,
            Some(cap),
            None,
        )
        .unwrap();

        let direct_data = direct.to_vec::<f32>().unwrap();
        let indirect_data = indirect.to_vec::<f32>().unwrap();

        for (i, (&d, &ind)) in direct_data.iter().zip(indirect_data.iter()).enumerate() {
            assert!(
                (d - ind).abs() < 1e-3,
                "Softcap indirect mismatch at {i}: direct={d}, indirect={ind}"
            );
        }
    }

    /// CPU reference for decode attention with optional soft-capping.
    fn cpu_decode_attention_softcap(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        total_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        softcap: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; num_heads * head_dim];
        for h in 0..num_heads {
            let kv_h = h * num_kv_heads / num_heads;
            let mut scores = vec![0.0f32; total_len];
            for t in 0..total_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot +=
                        q[h * head_dim + d] * k[t * num_kv_heads * head_dim + kv_h * head_dim + d];
                }
                dot *= scale;
                if softcap > 0.0 {
                    dot = (dot / softcap).tanh() * softcap;
                }
                scores[t] = dot;
            }
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum_exp += *s;
            }
            for s in &mut scores {
                *s /= sum_exp;
            }
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for t in 0..total_len {
                    acc += scores[t] * v[t * num_kv_heads * head_dim + kv_h * head_dim + d];
                }
                output[h * head_dim + d] = acc;
            }
        }
        output
    }

    #[test]
    fn test_fused_decode_softcap() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let head_dim = 8;
        let total_len = 6;
        let cap = 50.0_f32;

        let q_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|x| ((x as f32) * 0.3).sin())
            .collect();
        let k_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| ((x as f32) * 0.2).cos())
            .collect();
        let v_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| ((x as f32) * 0.1).sin())
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &v_data).unwrap();

        let scale = 1.0 / (head_dim as f32).sqrt();

        // With softcap
        let capped = fused_attention_decode(&q, &k, &v, None, Some(cap), None).unwrap();
        // Without softcap
        let uncapped = fused_attention_decode(&q, &k, &v, None, None, None).unwrap();

        let capped_data = capped.to_vec::<f32>().unwrap();
        let uncapped_data = uncapped.to_vec::<f32>().unwrap();

        // Softcap should change the output
        assert_ne!(
            capped_data, uncapped_data,
            "Softcap should produce different attention output"
        );

        // Verify against CPU reference
        let ref_data = cpu_decode_attention_softcap(
            &q_data, &k_data, &v_data, total_len, num_heads, num_heads, head_dim, scale, cap,
        );

        for (i, (&g, &r)) in capped_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (g - r).abs() < 1e-3,
                "Softcap decode mismatch at {i}: gpu={g}, cpu={r}"
            );
        }
    }

    #[test]
    fn test_fused_prefill_softcap() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let head_dim = 4;
        let seq_q = 4;
        let cap = 50.0_f32;

        let q_data: Vec<f32> = (0..seq_q * num_heads * head_dim)
            .map(|x| ((x as f32) * 0.15).sin())
            .collect();
        let k_data = q_data.clone();
        let v_data: Vec<f32> = (0..seq_q * num_heads * head_dim)
            .map(|x| ((x as f32) * 0.07).cos())
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[seq_q, num_heads, head_dim], &q_data).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[seq_q, num_heads, head_dim], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[seq_q, num_heads, head_dim], &v_data).unwrap();

        let capped = fused_attention_prefill(&q, &k, &v, 0, None, Some(cap), None).unwrap();
        let uncapped = fused_attention_prefill(&q, &k, &v, 0, None, None, None).unwrap();

        let capped_data = capped.to_vec::<f32>().unwrap();
        let uncapped_data = uncapped.to_vec::<f32>().unwrap();

        // Softcap should change output (at least for later positions with larger scores)
        assert_ne!(
            capped_data, uncapped_data,
            "Softcap should produce different prefill output"
        );

        // Position 0 has only one valid key, so softmax output is always 1.0
        // regardless of capping — the output should match.
        let elems_per_pos = num_heads * head_dim;
        for (i, (&c, &u)) in capped_data[..elems_per_pos]
            .iter()
            .zip(uncapped_data[..elems_per_pos].iter())
            .enumerate()
        {
            assert!(
                (c - u).abs() < 1e-4,
                "Position 0 should be identical: capped={c}, uncapped={u} at {i}"
            );
        }
    }

    #[test]
    fn test_fused_decode_custom_scale() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let num_heads = 2;
        let head_dim = 8;
        let total_len = 4;

        let q_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|x| ((x as f32) * 0.2).sin())
            .collect();
        let k_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| ((x as f32) * 0.15).cos())
            .collect();
        let v_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| ((x as f32) * 0.1).sin())
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &v_data).unwrap();

        // Gemma-style: scale = 1/sqrt(query_pre_attn_scalar) instead of 1/sqrt(head_dim)
        let custom_scale = 1.0 / 256.0_f32.sqrt();
        let default_scale = 1.0 / (head_dim as f32).sqrt();

        let custom = fused_attention_decode(&q, &k, &v, Some(custom_scale), None, None).unwrap();
        let default = fused_attention_decode(&q, &k, &v, None, None, None).unwrap();

        let custom_data = custom.to_vec::<f32>().unwrap();
        let default_data = default.to_vec::<f32>().unwrap();

        // Different scales should produce different outputs
        assert_ne!(
            custom_data, default_data,
            "Custom scale should differ from default"
        );

        // Verify custom scale against CPU reference
        let ref_data = cpu_decode_attention_softcap(
            &q_data,
            &k_data,
            &v_data,
            total_len,
            num_heads,
            num_heads,
            head_dim,
            custom_scale,
            0.0,
        );

        for (i, (&g, &r)) in custom_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (g - r).abs() < 1e-3,
                "Custom scale mismatch at {i}: gpu={g}, cpu={r}"
            );
        }

        // Verify default scale against CPU reference
        let ref_default = cpu_decode_attention_softcap(
            &q_data,
            &k_data,
            &v_data,
            total_len,
            num_heads,
            num_heads,
            head_dim,
            default_scale,
            0.0,
        );

        for (i, (&g, &r)) in default_data.iter().zip(ref_default.iter()).enumerate() {
            assert!(
                (g - r).abs() < 1e-3,
                "Default scale mismatch at {i}: gpu={g}, cpu={r}"
            );
        }
    }

    #[test]
    fn fused_prefill_with_lse_matches_without() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let num_heads = 2;
        let head_dim = 8;
        let seq_q = 4;
        let total_len = 4;

        let q_data: Vec<f32> = (0..seq_q * num_heads * head_dim)
            .map(|x| ((x as f32) * 0.3).sin())
            .collect();
        let k_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| ((x as f32) * 0.2).cos())
            .collect();
        let v_data: Vec<f32> = (0..total_len * num_heads * head_dim)
            .map(|x| ((x as f32) * 0.1).sin())
            .collect();

        let q = CudaTensor::from_slice(&ctx, &[seq_q, num_heads, head_dim], &q_data).unwrap();
        let k = CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[total_len, num_heads, head_dim], &v_data).unwrap();

        let without = fused_attention_prefill(&q, &k, &v, 0, None, None, None).unwrap();
        let (with_out, lse) =
            fused_attention_prefill_with_lse(&q, &k, &v, 0, None, None, None).unwrap();

        let without_data = without.to_vec::<f32>().unwrap();
        let with_data = with_out.to_vec::<f32>().unwrap();
        let lse_data = lse.to_vec::<f32>().unwrap();

        for (i, (&a, &b)) in without_data.iter().zip(with_data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "prefill_with_lse output mismatch at {i}: without={a}, with={b}"
            );
        }

        assert_eq!(lse_data.len(), seq_q * num_heads);
        for (i, &l) in lse_data.iter().enumerate() {
            assert!(l.is_finite(), "LSE is not finite at {i}: {l}");
        }
    }

    #[test]
    fn combine_lse_equal_weights() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let n = 2;
        let num_heads = 2;
        let head_dim = 4;

        let out1_data: Vec<f32> = (0..n * num_heads * head_dim)
            .map(|x| x as f32 * 0.1)
            .collect();
        let out2_data: Vec<f32> = (0..n * num_heads * head_dim)
            .map(|x| x as f32 * 0.2)
            .collect();
        let lse_data: Vec<f32> = vec![1.0; n * num_heads];

        let out1 = CudaTensor::from_slice(&ctx, &[n, num_heads, head_dim], &out1_data).unwrap();
        let out2 = CudaTensor::from_slice(&ctx, &[n, num_heads, head_dim], &out2_data).unwrap();
        let lse1 = CudaTensor::from_slice(&ctx, &[n, num_heads], &lse_data).unwrap();
        let lse2 = CudaTensor::from_slice(&ctx, &[n, num_heads], &lse_data).unwrap();

        let combined = combine_attention_with_lse(&out1, &lse1, &out2, &lse2).unwrap();
        let result = combined.to_vec::<f32>().unwrap();

        for (i, (&r, (&a, &b))) in result
            .iter()
            .zip(out1_data.iter().zip(out2_data.iter()))
            .enumerate()
        {
            let expected = (a + b) / 2.0;
            assert!(
                (r - expected).abs() < 1e-5,
                "combine_lse equal mismatch at {i}: got={r}, expected={expected}"
            );
        }
    }

    #[test]
    fn combine_lse_dominated_by_one_side() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let n = 1;
        let num_heads = 2;
        let head_dim = 4;

        let out1_data: Vec<f32> = vec![1.0; n * num_heads * head_dim];
        let out2_data: Vec<f32> = vec![2.0; n * num_heads * head_dim];
        let lse1_data: Vec<f32> = vec![100.0; n * num_heads];
        let lse2_data: Vec<f32> = vec![0.0; n * num_heads];

        let out1 = CudaTensor::from_slice(&ctx, &[n, num_heads, head_dim], &out1_data).unwrap();
        let out2 = CudaTensor::from_slice(&ctx, &[n, num_heads, head_dim], &out2_data).unwrap();
        let lse1 = CudaTensor::from_slice(&ctx, &[n, num_heads], &lse1_data).unwrap();
        let lse2 = CudaTensor::from_slice(&ctx, &[n, num_heads], &lse2_data).unwrap();

        let combined = combine_attention_with_lse(&out1, &lse1, &out2, &lse2).unwrap();
        let result = combined.to_vec::<f32>().unwrap();

        for (i, &r) in result.iter().enumerate() {
            assert!(
                (r - 1.0).abs() < 1e-5,
                "combine_lse dominated mismatch at {i}: got={r}, expected=1.0"
            );
        }
    }
}
