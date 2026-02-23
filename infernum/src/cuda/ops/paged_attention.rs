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

use cudarc::driver::{LaunchAsync, LaunchConfig, ValidAsZeroBits};

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

    let scale = 1.0 / (head_dim as f32).sqrt();

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

    ensure_paged_decode_kernel::<T>(device)?;

    let kernel_name = format!("paged_decode_attention_{}", kernel_suffix::<T>());
    let func = device
        .get_func("paged_decode_attention", &kernel_name)
        .unwrap();

    let threads = 256_usize.min(max_seq_len.next_power_of_two());
    let threads = threads.max(1); // at least 1 thread

    // Shared memory: s_q (head_dim) + s_weights (max_seq_len) + s_scratch (threads)
    let shared_mem = (head_dim + max_seq_len + threads) * std::mem::size_of::<f32>();

    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, batch_size as u32, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    unsafe {
        func.launch(
            cfg,
            (
                output.cuda_slice_mut(),
                &q.cuda_slice(),
                &k_pool.cuda_slice(),
                &v_pool.cuda_slice(),
                &block_tables_gpu,
                &seq_lens_gpu,
                scale,
                block_size as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                max_blocks_per_seq as i32,
            ),
        )?;
    }

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

    let shape = [seq_len, num_kv_heads, head_dim];
    let (k_pool, v_pool) = paged_kv.get_pools(layer_idx);

    // Read entire pool to host, gather the relevant blocks, upload contiguous result.
    // This is simple and correct. A GPU gather kernel would be faster for large sequences
    // but this path is only used for prefill (Step 10 profiles whether to optimize).
    let k_all = k_pool.to_vec()?;
    let v_all = v_pool.to_vec()?;

    let token_stride = num_kv_heads * head_dim;
    let block_stride = block_size * token_stride;

    let mut k_gathered = vec![T::default(); seq_len * token_stride];
    let mut v_gathered = vec![T::default(); seq_len * token_stride];

    for t in 0..seq_len {
        let logical_block = t / block_size;
        let offset_in_block = t % block_size;
        let physical_block = block_table.blocks()[logical_block];

        let src_offset = physical_block * block_stride + offset_in_block * token_stride;
        let dst_offset = t * token_stride;

        k_gathered[dst_offset..dst_offset + token_stride]
            .copy_from_slice(&k_all[src_offset..src_offset + token_stride]);
        v_gathered[dst_offset..dst_offset + token_stride]
            .copy_from_slice(&v_all[src_offset..src_offset + token_stride]);
    }

    let k_contig = CudaTensor::from_slice(ctx, &shape, &k_gathered)?;
    let v_contig = CudaTensor::from_slice(ctx, &shape, &v_gathered)?;

    Ok((k_contig, v_contig))
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

        let expected = fused_attention_decode(&q_contig, &k_contig, &v_contig, None).unwrap();
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
        let actual =
            paged_attention_decode(&ctx, &q_paged, k_pool, v_pool, &[table], block_size).unwrap();
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
        let expected0 = fused_attention_decode(&q0_gpu, &k0_contig, &v0_contig, None).unwrap();
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
        let expected1 = fused_attention_decode(&q1_gpu, &k1_contig, &v1_contig, None).unwrap();
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
}
