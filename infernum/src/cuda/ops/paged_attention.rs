//! Paged attention kernels for batched inference
//!
//! Decode attention with K/V stored in a paged (block-table-indexed) pool.
//! For prefill, a gather step copies paged K/V into a contiguous buffer,
//! then the existing fused prefill kernel is reused.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::missing_panics_doc,
    clippy::too_many_arguments
)]

use cudarc::driver::{DevicePtr, DevicePtrMut, LaunchAsync, LaunchConfig, ValidAsZeroBits};
use std::ffi::c_void;

use crate::cuda::block_allocator::BlockTable;
use crate::cuda::paged_kv_cache::PagedKvCache;
use crate::cuda::CudaContext;
use crate::cuda::CudaTensor;
use crate::dtype::TensorDType;
use crate::tensor::Tensor;
use crate::Result;

const PAGED_DECODE_PTX: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "/kernels/paged_decode_attention.ptx"
));

const GATHER_PAGED_KV_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/kernels/gather_paged_kv.ptx"));

const GATHER_PAGED_KV_KERNEL_NAMES: &[&str] = &[
    "gather_paged_kv_f32",
    "gather_paged_kv_f16",
    "gather_paged_kv_bf16",
];

const PAGED_DECODE_KERNEL_NAMES: &[&str] = &[
    "paged_decode_attention_f32",
    "paged_decode_attention_f16",
    "paged_decode_attention_bf16",
];

fn kernel_suffix<T: cudarc::driver::DeviceRepr>() -> &'static str {
    let type_name = std::any::type_name::<T>();
    if type_name.contains("f32") {
        "f32"
    } else if type_name.contains("f16") && !type_name.contains("bf16") {
        "f16"
    } else if type_name.contains("bf16") {
        "bf16"
    } else {
        panic!("Unsupported dtype for paged_attention: {type_name}")
    }
}

fn ensure_paged_decode_kernel<T: cudarc::driver::DeviceRepr>(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<()> {
    let module_name = "paged_decode_attention";
    let kernel_name = format!("paged_decode_attention_{}", kernel_suffix::<T>());
    if !device.has_func(module_name, &kernel_name) {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(PAGED_DECODE_PTX),
            module_name,
            PAGED_DECODE_KERNEL_NAMES,
        )?;
    }
    Ok(())
}

fn ensure_gather_kernel<T: cudarc::driver::DeviceRepr>(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<()> {
    let module_name = "gather_paged_kv";
    let kernel_name = format!("gather_paged_kv_{}", kernel_suffix::<T>());
    if !device.has_func(module_name, &kernel_name) {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(GATHER_PAGED_KV_PTX),
            module_name,
            GATHER_PAGED_KV_KERNEL_NAMES,
        )?;
    }
    Ok(())
}

/// Batched paged decode attention.
///
/// Each request has one query token. K/V data is in a paged pool, accessed
/// via per-request block tables.
///
/// # Arguments
/// * `ctx` — CUDA context
/// * `q` — query tensor of shape `(batch_size, num_heads, head_dim)`
/// * `k_pool` — key pool of shape `(num_blocks, block_size, num_kv_heads, head_dim)`
/// * `v_pool` — value pool, same shape as `k_pool`
/// * `block_tables` — per-request block tables
/// * `block_size` — tokens per block
/// * `scale` — if `Some(s)`, use `s` as the attention scale; otherwise `1/sqrt(head_dim)`
/// * `softcap` — if `Some(cap)`, applies `tanh(score/cap)*cap` after scaling
/// * `sliding_window` — if `Some(w)`, restrict attention to the last `w` positions
///
/// # Returns
/// Output tensor of shape `(batch_size, num_heads, head_dim)`
///
/// # Errors
/// Returns an error if the kernel launch fails.
pub fn paged_attention_decode<T: TensorDType + cudarc::driver::DeviceRepr + ValidAsZeroBits>(
    ctx: &CudaContext,
    q: &CudaTensor<T>,
    k_pool: &CudaTensor<T>,
    v_pool: &CudaTensor<T>,
    block_tables: &[BlockTable],
    block_size: usize,
    scale: Option<f32>,
    softcap: Option<f32>,
    sliding_window: Option<usize>,
) -> Result<CudaTensor<T>> {
    let q_shape = q.shape();
    assert_eq!(
        q_shape.len(),
        3,
        "Q must be 3D: (batch_size, num_heads, head_dim)"
    );

    let batch_size = q_shape[0];
    let num_heads = q_shape[1];
    let head_dim = q_shape[2];

    let k_pool_shape = k_pool.shape();
    assert_eq!(k_pool_shape.len(), 4, "K pool must be 4D");
    let num_kv_heads = k_pool_shape[2];

    assert_eq!(
        block_tables.len(),
        batch_size,
        "block_tables length must match batch_size"
    );
    assert!(
        num_heads.is_multiple_of(num_kv_heads),
        "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    );

    let mut scale = scale.unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
    let mut softcap_val = softcap.unwrap_or(0.0);
    let window_size = sliding_window.map_or(0, |w| w as i32);

    // Find max blocks per sequence and max seq_len for shared memory sizing
    let max_blocks_per_seq = block_tables
        .iter()
        .map(BlockTable::num_blocks)
        .max()
        .unwrap_or(0);
    let max_seq_len = block_tables
        .iter()
        .map(BlockTable::seq_len)
        .max()
        .unwrap_or(0);

    // Active length accounts for sliding window
    #[allow(clippy::cast_sign_loss)]
    let max_active_len = if window_size > 0 {
        max_seq_len.min(window_size as usize)
    } else {
        max_seq_len
    };

    // Flatten block tables into a (batch_size, max_blocks_per_seq) matrix
    let mut flat_block_tables = vec![0i32; batch_size * max_blocks_per_seq];
    let mut seq_lens = vec![0i32; batch_size];
    for (i, table) in block_tables.iter().enumerate() {
        seq_lens[i] = table.seq_len() as i32;
        for (j, &block_idx) in table.blocks().iter().enumerate() {
            flat_block_tables[i * max_blocks_per_seq + j] = block_idx as i32;
        }
    }

    let device = ctx.device();
    let block_tables_gpu = device.htod_sync_copy(&flat_block_tables)?;
    let seq_lens_gpu = device.htod_sync_copy(&seq_lens)?;

    let output_shape = [batch_size, num_heads, head_dim];
    let mut output = unsafe { CudaTensor::<T>::uninit(ctx, &output_shape)? };

    launch_paged_decode::<T>(
        device,
        &mut output,
        q,
        k_pool,
        v_pool,
        &block_tables_gpu,
        &seq_lens_gpu,
        &mut scale,
        &mut softcap_val,
        block_size,
        num_heads,
        num_kv_heads,
        head_dim,
        max_blocks_per_seq,
        window_size,
        max_active_len,
        batch_size,
    )?;

    Ok(output)
}

/// Shared kernel launch logic for both eager and indirect paged decode.
#[allow(clippy::too_many_arguments)]
fn launch_paged_decode<T: TensorDType + cudarc::driver::DeviceRepr + ValidAsZeroBits>(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    output: &mut CudaTensor<T>,
    q: &CudaTensor<T>,
    k_pool: &CudaTensor<T>,
    v_pool: &CudaTensor<T>,
    block_tables_gpu: &cudarc::driver::CudaSlice<i32>,
    seq_lens_gpu: &cudarc::driver::CudaSlice<i32>,
    scale: &mut f32,
    softcap_val: &mut f32,
    block_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_blocks_per_seq: usize,
    window_size: i32,
    max_active_len: usize,
    batch_size: usize,
) -> Result<()> {
    ensure_paged_decode_kernel::<T>(device)?;

    let kernel_name = format!("paged_decode_attention_{}", kernel_suffix::<T>());
    let func = device
        .get_func("paged_decode_attention", &kernel_name)
        .unwrap();

    let threads = 256_usize.min(max_active_len.next_power_of_two()).max(1);

    // Shared memory: s_q (head_dim) + s_weights (max_active_len) + s_scratch (threads)
    let shared_mem = (head_dim + max_active_len + threads) * std::mem::size_of::<f32>();

    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, batch_size as u32, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    let out_slice = output.cuda_slice_mut();
    let q_slice = q.cuda_slice();
    let k_slice = k_pool.cuda_slice();
    let v_slice = v_pool.cuda_slice();
    let mut bs = block_size as i32;
    let mut nh = num_heads as i32;
    let mut nkv = num_kv_heads as i32;
    let mut hd = head_dim as i32;
    let mut mbps = max_blocks_per_seq as i32;
    let mut ws = window_size;

    let mut args: Vec<*mut c_void> = vec![
        std::ptr::from_mut(out_slice.device_ptr_mut()).cast::<c_void>(),
        std::ptr::from_ref(q_slice.device_ptr())
            .cast_mut()
            .cast::<c_void>(),
        std::ptr::from_ref(k_slice.device_ptr())
            .cast_mut()
            .cast::<c_void>(),
        std::ptr::from_ref(v_slice.device_ptr())
            .cast_mut()
            .cast::<c_void>(),
        std::ptr::from_ref(block_tables_gpu.device_ptr())
            .cast_mut()
            .cast::<c_void>(),
        std::ptr::from_ref(seq_lens_gpu.device_ptr())
            .cast_mut()
            .cast::<c_void>(),
        (&raw mut *scale).cast::<c_void>(),
        (&raw mut *softcap_val).cast::<c_void>(),
        (&raw mut bs).cast::<c_void>(),
        (&raw mut nh).cast::<c_void>(),
        (&raw mut nkv).cast::<c_void>(),
        (&raw mut hd).cast::<c_void>(),
        (&raw mut mbps).cast::<c_void>(),
        (&raw mut ws).cast::<c_void>(),
    ];

    unsafe {
        func.launch(cfg, &mut args)?;
    }

    Ok(())
}

/// Batched paged decode attention with GPU-resident block tables and seq lens.
///
/// Identical to [`paged_attention_decode`] but reads block tables and sequence
/// lengths from pre-allocated GPU buffers, making this safe for CUDA graph
/// capture.
///
/// # Arguments
/// * `ctx` — CUDA context
/// * `q` — query tensor of shape `(batch_size, num_heads, head_dim)`
/// * `k_pool` / `v_pool` — paged K/V pool tensors
/// * `block_tables_gpu` — flattened block tables `(batch_size * max_blocks_per_seq,)`
/// * `seq_lens_gpu` — sequence lengths `(batch_size,)`
/// * `block_size` — tokens per block
/// * `max_blocks_per_seq` — max blocks per sequence (stride for block table rows)
/// * `max_seq_len` — max sequence length (for shared memory sizing)
/// * `scale` — if `Some(s)`, use `s` as the attention scale; otherwise `1/sqrt(head_dim)`
/// * `softcap` — if `Some(cap)`, applies `tanh(score/cap)*cap` after scaling
/// * `sliding_window` — if `Some(w)`, restrict attention to the last `w` positions
///
/// # Errors
/// Returns an error if the kernel launch fails.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_decode_indirect<
    T: TensorDType + cudarc::driver::DeviceRepr + ValidAsZeroBits,
>(
    ctx: &CudaContext,
    q: &CudaTensor<T>,
    k_pool: &CudaTensor<T>,
    v_pool: &CudaTensor<T>,
    block_tables_gpu: &cudarc::driver::CudaSlice<i32>,
    seq_lens_gpu: &cudarc::driver::CudaSlice<i32>,
    block_size: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
    scale: Option<f32>,
    softcap: Option<f32>,
    sliding_window: Option<usize>,
) -> Result<CudaTensor<T>> {
    let q_shape = q.shape();
    assert_eq!(
        q_shape.len(),
        3,
        "Q must be 3D: (batch_size, num_heads, head_dim)"
    );

    let batch_size = q_shape[0];
    let num_heads = q_shape[1];
    let head_dim = q_shape[2];

    let k_pool_shape = k_pool.shape();
    assert_eq!(k_pool_shape.len(), 4, "K pool must be 4D");
    let num_kv_heads = k_pool_shape[2];

    assert!(
        num_heads.is_multiple_of(num_kv_heads),
        "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    );

    let mut scale = scale.unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
    let mut softcap_val = softcap.unwrap_or(0.0);
    let window_size = sliding_window.map_or(0, |w| w as i32);

    // Active length accounts for sliding window
    #[allow(clippy::cast_sign_loss)]
    let max_active_len = if window_size > 0 {
        max_seq_len.min(window_size as usize)
    } else {
        max_seq_len
    };

    let output_shape = [batch_size, num_heads, head_dim];
    let mut output = unsafe { CudaTensor::<T>::uninit(ctx, &output_shape)? };

    launch_paged_decode::<T>(
        ctx.device(),
        &mut output,
        q,
        k_pool,
        v_pool,
        block_tables_gpu,
        seq_lens_gpu,
        &mut scale,
        &mut softcap_val,
        block_size,
        num_heads,
        num_kv_heads,
        head_dim,
        max_blocks_per_seq,
        window_size,
        max_active_len,
        batch_size,
    )?;

    Ok(output)
}

/// Gather paged K/V into contiguous buffers for a single request.
///
/// Copies K/V data from the paged pool into contiguous tensors of shape
/// `(seq_len, num_kv_heads, head_dim)`, suitable for use with existing
/// non-paged attention kernels (e.g., fused prefill attention).
///
/// # Errors
/// Returns an error if GPU memory allocation or copy fails.
pub fn gather_paged_kv<T: TensorDType + cudarc::driver::DeviceRepr + ValidAsZeroBits>(
    paged_kv: &PagedKvCache<T>,
    layer_idx: usize,
    block_table: &BlockTable,
) -> Result<(CudaTensor<T>, CudaTensor<T>)> {
    let seq_len = block_table.seq_len();
    let num_kv_heads = paged_kv.num_kv_heads();
    let head_dim = paged_kv.head_dim();
    let block_size = paged_kv.block_size();
    let ctx = paged_kv.context();
    let device = ctx.device();

    ensure_gather_kernel::<T>(device)?;

    let shape = [seq_len, num_kv_heads, head_dim];
    let (k_pool, v_pool) = paged_kv.get_pools(layer_idx);

    // Upload block table to GPU
    let block_table_i32: Vec<i32> = block_table.blocks().iter().map(|&b| b as i32).collect();
    let block_table_gpu = device.htod_sync_copy(&block_table_i32)?;

    let total = seq_len * num_kv_heads * head_dim;
    let threads = 256;
    let blocks = total.div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let kernel_name = format!("gather_paged_kv_{}", kernel_suffix::<T>());

    let mut k_out = unsafe { CudaTensor::<T>::uninit(ctx, &shape)? };
    let func = device
        .get_func("gather_paged_kv", &kernel_name)
        .expect("gather kernel not loaded");
    unsafe {
        func.launch(
            cfg,
            (
                k_out.cuda_slice_mut(),
                &k_pool.cuda_slice(),
                &block_table_gpu,
                seq_len as i32,
                block_size as i32,
                num_kv_heads as i32,
                head_dim as i32,
            ),
        )?;
    }

    let mut v_out = unsafe { CudaTensor::<T>::uninit(ctx, &shape)? };
    let func = device
        .get_func("gather_paged_kv", &kernel_name)
        .expect("gather kernel not loaded");
    unsafe {
        func.launch(
            cfg,
            (
                v_out.cuda_slice_mut(),
                &v_pool.cuda_slice(),
                &block_table_gpu,
                seq_len as i32,
                block_size as i32,
                num_kv_heads as i32,
                head_dim as i32,
            ),
        )?;
    }

    Ok((k_out, v_out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::ops::fused_attention::fused_attention_decode;

    fn make_ctx() -> CudaContext {
        CudaContext::new(0).expect("Failed to create CUDA context")
    }

    /// Compare paged decode attention vs contiguous decode attention.
    ///
    /// Sets up identical Q/K/V data in both contiguous and paged layouts,
    /// runs both kernels, and verifies the outputs match within tolerance.
    #[test]
    fn paged_decode_matches_contiguous() {
        let ctx = make_ctx();
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let block_size = 4;
        let seq_len = 7; // 2 blocks (4 + 3 tokens)

        // Generate deterministic Q/K/V data
        let q_data: Vec<f32> = (0..(num_heads * head_dim))
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();
        let kv_len = seq_len * num_kv_heads * head_dim;
        let k_data: Vec<f32> = (0..kv_len).map(|i| ((i as f32) * 0.07).cos()).collect();
        let v_data: Vec<f32> = (0..kv_len).map(|i| ((i as f32) * 0.03).sin()).collect();

        // --- Contiguous path ---
        let q_contig = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();
        let k_contig =
            CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &k_data).unwrap();
        let v_contig =
            CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &v_data).unwrap();

        let expected =
            fused_attention_decode(&q_contig, &k_contig, &v_contig, None, None, None).unwrap();
        let expected_vals = expected.to_vec().unwrap();

        // --- Paged path ---
        let num_blocks = 8; // more than needed
        let config = crate::cuda::block_allocator::BlockConfig {
            block_size,
            num_blocks,
        };
        let mut paged_kv = PagedKvCache::new(&ctx, 1, &config, num_kv_heads, head_dim).unwrap();
        let mut allocator = crate::cuda::block_allocator::BlockAllocator::new(&config);

        // Allocate blocks and build block table
        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        let mut table = BlockTable::new(block_size);
        table.append_block(b0);
        table.append_block(b1);

        // Write K/V data into paged cache
        let k_gpu =
            CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &k_data).unwrap();
        let v_gpu =
            CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &v_data).unwrap();
        paged_kv.append_paged(0, &table, &k_gpu, &v_gpu, 0).unwrap();
        table.advance(seq_len);

        // Run paged decode attention
        let q_paged = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();
        let (k_pool, v_pool) = paged_kv.get_pools(0);
        let actual = paged_attention_decode(
            &ctx,
            &q_paged,
            k_pool,
            v_pool,
            &[table],
            block_size,
            None,
            None,
            None,
        )
        .unwrap();
        let actual_vals = actual.to_vec().unwrap();

        // Compare
        assert_eq!(expected_vals.len(), actual_vals.len());
        for (i, (e, a)) in expected_vals.iter().zip(actual_vals.iter()).enumerate() {
            assert!(
                (e - a).abs() < 1e-4,
                "Mismatch at index {i}: expected {e}, got {a}"
            );
        }
    }

    /// Test batched paged decode with multiple requests having different seq_lens.
    #[test]
    fn paged_decode_batched_different_seq_lens() {
        let ctx = make_ctx();
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let block_size = 4;
        let num_blocks = 16;

        let config = crate::cuda::block_allocator::BlockConfig {
            block_size,
            num_blocks,
        };
        let mut paged_kv = PagedKvCache::new(&ctx, 1, &config, num_kv_heads, head_dim).unwrap();
        let mut allocator = crate::cuda::block_allocator::BlockAllocator::new(&config);

        // Request 0: 3 tokens (1 block)
        let b0 = allocator.allocate().unwrap();
        let mut table0 = BlockTable::new(block_size);
        table0.append_block(b0);

        let seq0 = 3;
        let k0_data: Vec<f32> = (0..(seq0 * num_kv_heads * head_dim))
            .map(|i| (i as f32) * 0.1)
            .collect();
        let v0_data: Vec<f32> = (0..(seq0 * num_kv_heads * head_dim))
            .map(|i| (i as f32) * 0.05)
            .collect();
        let k0 = CudaTensor::from_slice(&ctx, &[seq0, num_kv_heads, head_dim], &k0_data).unwrap();
        let v0 = CudaTensor::from_slice(&ctx, &[seq0, num_kv_heads, head_dim], &v0_data).unwrap();
        paged_kv.append_paged(0, &table0, &k0, &v0, 0).unwrap();
        table0.advance(seq0);

        // Request 1: 6 tokens (2 blocks)
        let b1 = allocator.allocate().unwrap();
        let b2 = allocator.allocate().unwrap();
        let mut table1 = BlockTable::new(block_size);
        table1.append_block(b1);
        table1.append_block(b2);

        let seq1 = 6;
        let k1_data: Vec<f32> = (0..(seq1 * num_kv_heads * head_dim))
            .map(|i| (i as f32) * 0.2)
            .collect();
        let v1_data: Vec<f32> = (0..(seq1 * num_kv_heads * head_dim))
            .map(|i| (i as f32) * 0.15)
            .collect();
        let k1 = CudaTensor::from_slice(&ctx, &[seq1, num_kv_heads, head_dim], &k1_data).unwrap();
        let v1 = CudaTensor::from_slice(&ctx, &[seq1, num_kv_heads, head_dim], &v1_data).unwrap();
        paged_kv.append_paged(0, &table1, &k1, &v1, 0).unwrap();
        table1.advance(seq1);

        // Create batched query: (2, num_heads, head_dim)
        let q0_data: Vec<f32> = (0..(num_heads * head_dim))
            .map(|i| ((i as f32) * 0.3).sin())
            .collect();
        let q1_data: Vec<f32> = (0..(num_heads * head_dim))
            .map(|i| ((i as f32) * 0.4).cos())
            .collect();
        let mut q_data = q0_data.clone();
        q_data.extend_from_slice(&q1_data);

        let q = CudaTensor::from_slice(&ctx, &[2, num_heads, head_dim], &q_data).unwrap();

        let (k_pool, v_pool) = paged_kv.get_pools(0);
        let output = paged_attention_decode(
            &ctx,
            &q,
            k_pool,
            v_pool,
            &[table0.clone(), table1.clone()],
            block_size,
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(output.shape(), &[2, num_heads, head_dim]);

        // Verify each request independently against contiguous attention
        let out_vals = output.to_vec().unwrap();
        let req_size = num_heads * head_dim;

        // Request 0
        let q0_gpu = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q0_data).unwrap();
        let k0_contig =
            CudaTensor::from_slice(&ctx, &[seq0, num_kv_heads, head_dim], &k0_data).unwrap();
        let v0_contig =
            CudaTensor::from_slice(&ctx, &[seq0, num_kv_heads, head_dim], &v0_data).unwrap();
        let expected0 =
            fused_attention_decode(&q0_gpu, &k0_contig, &v0_contig, None, None, None).unwrap();
        let exp0 = expected0.to_vec().unwrap();
        for i in 0..req_size {
            assert!(
                (out_vals[i] - exp0[i]).abs() < 1e-4,
                "Request 0 mismatch at {i}: {} vs {}",
                out_vals[i],
                exp0[i]
            );
        }

        // Request 1
        let q1_gpu = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q1_data).unwrap();
        let k1_contig =
            CudaTensor::from_slice(&ctx, &[seq1, num_kv_heads, head_dim], &k1_data).unwrap();
        let v1_contig =
            CudaTensor::from_slice(&ctx, &[seq1, num_kv_heads, head_dim], &v1_data).unwrap();
        let expected1 =
            fused_attention_decode(&q1_gpu, &k1_contig, &v1_contig, None, None, None).unwrap();
        let exp1 = expected1.to_vec().unwrap();
        for i in 0..req_size {
            assert!(
                (out_vals[req_size + i] - exp1[i]).abs() < 1e-4,
                "Request 1 mismatch at {i}: {} vs {}",
                out_vals[req_size + i],
                exp1[i]
            );
        }
    }

    /// Test gather_paged_kv matches the original data.
    #[test]
    fn gather_paged_kv_matches_original() {
        let ctx = make_ctx();
        let num_kv_heads = 2;
        let head_dim = 4;
        let block_size = 4;
        let seq_len = 6;

        let config = crate::cuda::block_allocator::BlockConfig {
            block_size,
            num_blocks: 8,
        };
        let mut paged_kv = PagedKvCache::new(&ctx, 1, &config, num_kv_heads, head_dim).unwrap();
        let mut allocator = crate::cuda::block_allocator::BlockAllocator::new(&config);

        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        let mut table = BlockTable::new(block_size);
        table.append_block(b0);
        table.append_block(b1);

        let kv_len = seq_len * num_kv_heads * head_dim;
        let k_data: Vec<f32> = (0..kv_len).map(|i| i as f32).collect();
        let v_data: Vec<f32> = (0..kv_len).map(|i| (i as f32) + 100.0).collect();

        let k = CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &v_data).unwrap();
        paged_kv.append_paged(0, &table, &k, &v, 0).unwrap();
        table.advance(seq_len);

        let (k_gathered, v_gathered) = gather_paged_kv(&paged_kv, 0, &table).unwrap();

        assert_eq!(k_gathered.shape(), &[seq_len, num_kv_heads, head_dim]);
        assert_eq!(k_gathered.to_vec().unwrap(), k_data);
        assert_eq!(v_gathered.to_vec().unwrap(), v_data);
    }

    #[test]
    fn paged_decode_indirect_matches_eager() {
        let ctx = make_ctx();
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let block_size = 4;

        let config = crate::cuda::block_allocator::BlockConfig {
            block_size,
            num_blocks: 16,
        };
        let mut paged_kv = PagedKvCache::new(&ctx, 1, &config, num_kv_heads, head_dim).unwrap();
        let mut allocator = crate::cuda::block_allocator::BlockAllocator::new(&config);

        let b0 = allocator.allocate().unwrap();
        let mut table0 = BlockTable::new(block_size);
        table0.append_block(b0);
        let seq0 = 3;
        let k0_data: Vec<f32> = (0..(seq0 * num_kv_heads * head_dim))
            .map(|i| (i as f32) * 0.1)
            .collect();
        let v0_data: Vec<f32> = (0..(seq0 * num_kv_heads * head_dim))
            .map(|i| (i as f32) * 0.05)
            .collect();
        let k0 = CudaTensor::from_slice(&ctx, &[seq0, num_kv_heads, head_dim], &k0_data).unwrap();
        let v0 = CudaTensor::from_slice(&ctx, &[seq0, num_kv_heads, head_dim], &v0_data).unwrap();
        paged_kv.append_paged(0, &table0, &k0, &v0, 0).unwrap();
        table0.advance(seq0);

        let q_data: Vec<f32> = (0..(num_heads * head_dim))
            .map(|i| ((i as f32) * 0.3).sin())
            .collect();
        let q = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();

        // Eager
        let (k_pool, v_pool) = paged_kv.get_pools(0);
        let eager = paged_attention_decode(
            &ctx,
            &q,
            k_pool,
            v_pool,
            &[table0.clone()],
            block_size,
            None,
            None,
            None,
        )
        .unwrap();
        let eager_data = eager.to_vec().unwrap();

        // Indirect
        let max_blocks_per_seq = table0.num_blocks();
        let flat_tables: Vec<i32> = table0.blocks().iter().map(|&b| b as i32).collect();
        let seq_lens = [table0.seq_len() as i32];
        let tables_gpu = ctx.device().htod_sync_copy(&flat_tables).unwrap();
        let seq_lens_gpu = ctx.device().htod_sync_copy(&seq_lens).unwrap();

        let (k_pool, v_pool) = paged_kv.get_pools(0);
        let indirect = paged_attention_decode_indirect(
            &ctx,
            &q,
            k_pool,
            v_pool,
            &tables_gpu,
            &seq_lens_gpu,
            block_size,
            max_blocks_per_seq,
            table0.seq_len(),
            None,
            None,
            None,
        )
        .unwrap();
        let indirect_data = indirect.to_vec().unwrap();

        for (i, (&e, &ind)) in eager_data.iter().zip(indirect_data.iter()).enumerate() {
            assert!(
                (e - ind).abs() < 1e-6,
                "Mismatch at {i}: eager={e}, indirect={ind}"
            );
        }
    }

    /// Paged decode with softcap matches contiguous fused decode with softcap.
    #[test]
    fn paged_decode_with_softcap() {
        let ctx = make_ctx();
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let block_size = 4;
        let seq_len = 7;
        let cap = 50.0_f32;

        let q_data: Vec<f32> = (0..(num_heads * head_dim))
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();
        let kv_len = seq_len * num_kv_heads * head_dim;
        let k_data: Vec<f32> = (0..kv_len).map(|i| ((i as f32) * 0.07).cos()).collect();
        let v_data: Vec<f32> = (0..kv_len).map(|i| ((i as f32) * 0.03).sin()).collect();

        // Contiguous reference with softcap
        let q_contig = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();
        let k_contig =
            CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &k_data).unwrap();
        let v_contig =
            CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &v_data).unwrap();
        let expected =
            fused_attention_decode(&q_contig, &k_contig, &v_contig, None, Some(cap), None).unwrap();
        let expected_vals = expected.to_vec().unwrap();

        // Also compute without softcap to verify softcap changes the result
        let no_cap =
            fused_attention_decode(&q_contig, &k_contig, &v_contig, None, None, None).unwrap();
        assert_ne!(
            expected_vals,
            no_cap.to_vec().unwrap(),
            "Softcap should produce different output"
        );

        // Paged path with softcap
        let config = crate::cuda::block_allocator::BlockConfig {
            block_size,
            num_blocks: 8,
        };
        let mut paged_kv = PagedKvCache::new(&ctx, 1, &config, num_kv_heads, head_dim).unwrap();
        let mut allocator = crate::cuda::block_allocator::BlockAllocator::new(&config);

        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        let mut table = BlockTable::new(block_size);
        table.append_block(b0);
        table.append_block(b1);

        let k_gpu =
            CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &k_data).unwrap();
        let v_gpu =
            CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &v_data).unwrap();
        paged_kv.append_paged(0, &table, &k_gpu, &v_gpu, 0).unwrap();
        table.advance(seq_len);

        let q_paged = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();
        let (k_pool, v_pool) = paged_kv.get_pools(0);
        let actual = paged_attention_decode(
            &ctx,
            &q_paged,
            k_pool,
            v_pool,
            &[table],
            block_size,
            None,
            Some(cap),
            None,
        )
        .unwrap();
        let actual_vals = actual.to_vec().unwrap();

        assert_eq!(expected_vals.len(), actual_vals.len());
        for (i, (e, a)) in expected_vals.iter().zip(actual_vals.iter()).enumerate() {
            assert!(
                (e - a).abs() < 1e-4,
                "Softcap mismatch at index {i}: expected {e}, got {a}"
            );
        }
    }

    /// Paged decode with sliding window matches contiguous fused decode with sliding window.
    #[test]
    fn paged_decode_with_sliding_window() {
        let ctx = make_ctx();
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let block_size = 4;
        let seq_len = 10; // 3 blocks (4 + 4 + 2 tokens)
        let window = 4;

        let q_data: Vec<f32> = (0..(num_heads * head_dim))
            .map(|i| ((i as f32) * 0.13).sin())
            .collect();
        let kv_len = seq_len * num_kv_heads * head_dim;
        let k_data: Vec<f32> = (0..kv_len).map(|i| ((i as f32) * 0.09).cos()).collect();
        let v_data: Vec<f32> = (0..kv_len).map(|i| ((i as f32) * 0.05).sin()).collect();

        // Contiguous reference with sliding window
        let q_contig = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();
        let k_contig =
            CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &k_data).unwrap();
        let v_contig =
            CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &v_data).unwrap();
        let expected =
            fused_attention_decode(&q_contig, &k_contig, &v_contig, None, None, Some(window))
                .unwrap();
        let expected_vals = expected.to_vec().unwrap();

        // Also compute without window to verify it changes the result
        let no_window =
            fused_attention_decode(&q_contig, &k_contig, &v_contig, None, None, None).unwrap();
        assert_ne!(
            expected_vals,
            no_window.to_vec().unwrap(),
            "Sliding window should produce different output"
        );

        // Paged path with sliding window
        let config = crate::cuda::block_allocator::BlockConfig {
            block_size,
            num_blocks: 8,
        };
        let mut paged_kv = PagedKvCache::new(&ctx, 1, &config, num_kv_heads, head_dim).unwrap();
        let mut allocator = crate::cuda::block_allocator::BlockAllocator::new(&config);

        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        let b2 = allocator.allocate().unwrap();
        let mut table = BlockTable::new(block_size);
        table.append_block(b0);
        table.append_block(b1);
        table.append_block(b2);

        let k_gpu =
            CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &k_data).unwrap();
        let v_gpu =
            CudaTensor::from_slice(&ctx, &[seq_len, num_kv_heads, head_dim], &v_data).unwrap();
        paged_kv.append_paged(0, &table, &k_gpu, &v_gpu, 0).unwrap();
        table.advance(seq_len);

        let q_paged = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();
        let (k_pool, v_pool) = paged_kv.get_pools(0);
        let actual = paged_attention_decode(
            &ctx,
            &q_paged,
            k_pool,
            v_pool,
            &[table],
            block_size,
            None,
            None,
            Some(window),
        )
        .unwrap();
        let actual_vals = actual.to_vec().unwrap();

        assert_eq!(expected_vals.len(), actual_vals.len());
        for (i, (e, a)) in expected_vals.iter().zip(actual_vals.iter()).enumerate() {
            assert!(
                (e - a).abs() < 1e-4,
                "Sliding window mismatch at index {i}: expected {e}, got {a}"
            );
        }
    }

    /// Indirect paged decode with softcap matches eager paged decode with softcap.
    #[test]
    fn paged_decode_indirect_with_softcap() {
        let ctx = make_ctx();
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let block_size = 4;
        let cap = 50.0_f32;

        let config = crate::cuda::block_allocator::BlockConfig {
            block_size,
            num_blocks: 16,
        };
        let mut paged_kv = PagedKvCache::new(&ctx, 1, &config, num_kv_heads, head_dim).unwrap();
        let mut allocator = crate::cuda::block_allocator::BlockAllocator::new(&config);

        let b0 = allocator.allocate().unwrap();
        let mut table0 = BlockTable::new(block_size);
        table0.append_block(b0);
        let seq0 = 3;
        let k0_data: Vec<f32> = (0..(seq0 * num_kv_heads * head_dim))
            .map(|i| (i as f32) * 0.1)
            .collect();
        let v0_data: Vec<f32> = (0..(seq0 * num_kv_heads * head_dim))
            .map(|i| (i as f32) * 0.05)
            .collect();
        let k0 = CudaTensor::from_slice(&ctx, &[seq0, num_kv_heads, head_dim], &k0_data).unwrap();
        let v0 = CudaTensor::from_slice(&ctx, &[seq0, num_kv_heads, head_dim], &v0_data).unwrap();
        paged_kv.append_paged(0, &table0, &k0, &v0, 0).unwrap();
        table0.advance(seq0);

        let q_data: Vec<f32> = (0..(num_heads * head_dim))
            .map(|i| ((i as f32) * 0.3).sin())
            .collect();
        let q = CudaTensor::from_slice(&ctx, &[1, num_heads, head_dim], &q_data).unwrap();

        // Eager with softcap
        let (k_pool, v_pool) = paged_kv.get_pools(0);
        let eager = paged_attention_decode(
            &ctx,
            &q,
            k_pool,
            v_pool,
            &[table0.clone()],
            block_size,
            None,
            Some(cap),
            None,
        )
        .unwrap();
        let eager_data = eager.to_vec().unwrap();

        // Indirect with softcap
        let max_blocks_per_seq = table0.num_blocks();
        let flat_tables: Vec<i32> = table0.blocks().iter().map(|&b| b as i32).collect();
        let seq_lens = [table0.seq_len() as i32];
        let tables_gpu = ctx.device().htod_sync_copy(&flat_tables).unwrap();
        let seq_lens_gpu = ctx.device().htod_sync_copy(&seq_lens).unwrap();

        let (k_pool, v_pool) = paged_kv.get_pools(0);
        let indirect = paged_attention_decode_indirect(
            &ctx,
            &q,
            k_pool,
            v_pool,
            &tables_gpu,
            &seq_lens_gpu,
            block_size,
            max_blocks_per_seq,
            table0.seq_len(),
            None,
            Some(cap),
            None,
        )
        .unwrap();
        let indirect_data = indirect.to_vec().unwrap();

        for (i, (&e, &ind)) in eager_data.iter().zip(indirect_data.iter()).enumerate() {
            assert!(
                (e - ind).abs() < 1e-6,
                "Softcap indirect mismatch at {i}: eager={e}, indirect={ind}"
            );
        }
    }
}
