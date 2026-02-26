//! Paged KV cache for batched inference
//!
//! Stores K/V data in a pool of fixed-size blocks. Each request gets a chain
//! of blocks tracked by a [`BlockTable`]. The attention kernel uses the block
//! table to index into the shared pool.
//!
//! Pool layout per layer: `(num_blocks, block_size, num_kv_heads, head_dim)`,
//! row-major. This layout keeps a block's tokens contiguous in memory for
//! efficient sequential access during attention.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::missing_panics_doc
)]

use cudarc::driver::{LaunchAsync, LaunchConfig};

use super::CudaContext;
use super::CudaTensor;
use super::{BlockConfig, BlockTable};
use infernum::dtype::DType;
use infernum::tensor::Tensor;
use infernum::Result;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/append_kv_paged.ptx"));
const KERNEL_NAMES: &[&str] = &[
    "append_kv_paged_f32",
    "append_kv_paged_f16",
    "append_kv_paged_bf16",
    "append_kv_paged_batched_f32",
    "append_kv_paged_batched_f16",
    "append_kv_paged_batched_bf16",
];

fn kernel_suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => panic!("Unsupported dtype for append_kv_paged: {dtype}"),
    }
}

/// Per-layer K/V pool buffers.
struct LayerPool {
    /// Shape: `(num_blocks, block_size, num_kv_heads, head_dim)`
    k: CudaTensor,
    /// Shape: `(num_blocks, block_size, num_kv_heads, head_dim)`
    v: CudaTensor,
}

/// Paged KV cache for all transformer layers.
///
/// GPU memory is pre-allocated as a pool of blocks. Each request gets a
/// chain of blocks tracked by a [`BlockTable`]. The paged attention kernel
/// uses block tables to index into the shared pool.
pub struct PagedKvCache {
    layers: Vec<LayerPool>,
    ctx: CudaContext,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype: DType,
}

impl PagedKvCache {
    /// Allocate a new paged KV cache.
    ///
    /// # Arguments
    /// * `ctx` — CUDA context
    /// * `num_layers` — number of transformer layers
    /// * `block_config` — block size and total number of blocks
    /// * `num_kv_heads` — number of key-value heads (GQA-aware)
    /// * `head_dim` — dimension of each attention head
    /// * `dtype` — element type for the cache buffers
    ///
    /// # Errors
    /// Returns an error if GPU memory allocation fails.
    pub fn new(
        ctx: &CudaContext,
        num_layers: usize,
        block_config: &BlockConfig,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> Result<Self> {
        let shape = [
            block_config.num_blocks,
            block_config.block_size,
            num_kv_heads,
            head_dim,
        ];
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            // SAFETY: Pool memory is written before being read (via append_paged).
            // Block allocator ensures only allocated blocks are accessed.
            layers.push(LayerPool {
                k: unsafe { CudaTensor::uninit(ctx, &shape, dtype)? },
                v: unsafe { CudaTensor::uninit(ctx, &shape, dtype)? },
            });
        }

        Ok(Self {
            layers,
            ctx: ctx.clone(),
            block_size: block_config.block_size,
            num_kv_heads,
            head_dim,
            dtype,
        })
    }

    /// Append new K/V data for a layer using a block table.
    ///
    /// `k_new` and `v_new` must have shape `(new_seq_len, num_kv_heads, head_dim)`.
    /// The `block_table` determines which physical blocks to write into.
    /// `start_pos` is the sequence position of the first new token (i.e.,
    /// `block_table.seq_len()` before this append).
    ///
    /// The caller must ensure enough blocks are allocated in the block table
    /// to cover `start_pos + new_seq_len` tokens.
    ///
    /// # Errors
    /// Returns an error if the GPU kernel launch fails.
    pub fn append_paged(
        &mut self,
        layer_idx: usize,
        block_table: &BlockTable,
        k_new: &CudaTensor,
        v_new: &CudaTensor,
        start_pos: usize,
    ) -> Result<()> {
        let k_shape = k_new.shape();
        let v_shape = v_new.shape();
        assert_eq!(k_shape.len(), 3, "k_new must be 3D");
        assert_eq!(v_shape.len(), 3, "v_new must be 3D");

        let new_seq_len = k_shape[0];
        assert_eq!(k_shape[1], self.num_kv_heads, "k_new num_kv_heads mismatch");
        assert_eq!(k_shape[2], self.head_dim, "k_new head_dim mismatch");
        assert_eq!(v_shape, k_shape, "v_new shape must match k_new");

        // Verify the block table has enough blocks
        let end_pos = start_pos + new_seq_len;
        let blocks_needed = end_pos.div_ceil(self.block_size);
        assert!(
            block_table.num_blocks() >= blocks_needed,
            "block table has {} blocks but {} needed for {} tokens (start_pos={start_pos})",
            block_table.num_blocks(),
            blocks_needed,
            end_pos,
        );

        let pool = &mut self.layers[layer_idx];

        // Upload block table to GPU
        let block_table_i32: Vec<i32> = block_table.blocks().iter().map(|&b| b as i32).collect();
        let block_table_gpu = self.ctx.device().htod_sync_copy(&block_table_i32)?;

        launch_append_paged(
            &self.ctx,
            &mut pool.k,
            k_new,
            &block_table_gpu,
            start_pos,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            new_seq_len,
            self.dtype,
        )?;
        launch_append_paged(
            &self.ctx,
            &mut pool.v,
            v_new,
            &block_table_gpu,
            start_pos,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            new_seq_len,
            self.dtype,
        )?;

        Ok(())
    }

    /// Append one K/V token per sequence across a batch, using GPU-resident
    /// positions and block tables.
    ///
    /// `k_new` and `v_new` must have shape `(batch_size, num_kv_heads, head_dim)`.
    /// `positions_gpu` and `block_tables_gpu` are pre-allocated device buffers
    /// (e.g., from [`BatchedGraphInputs`](super::BatchedGraphInputs)).
    ///
    /// # Errors
    /// Returns an error if the kernel launch fails.
    #[allow(clippy::too_many_arguments)]
    pub fn append_paged_batched(
        &mut self,
        layer_idx: usize,
        k_new: &CudaTensor,
        v_new: &CudaTensor,
        block_tables_gpu: &cudarc::driver::CudaSlice<i32>,
        positions_gpu: &cudarc::driver::CudaSlice<i32>,
        batch_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        let k_shape = k_new.shape();
        assert_eq!(k_shape.len(), 3, "k_new must be 3D");
        assert_eq!(k_shape[0], batch_size);
        assert_eq!(k_shape[1], self.num_kv_heads);
        assert_eq!(k_shape[2], self.head_dim);

        let pool = &mut self.layers[layer_idx];

        launch_append_paged_batched(
            &self.ctx,
            &mut pool.k,
            k_new,
            block_tables_gpu,
            positions_gpu,
            batch_size,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            max_blocks_per_seq,
            self.dtype,
        )?;
        launch_append_paged_batched(
            &self.ctx,
            &mut pool.v,
            v_new,
            block_tables_gpu,
            positions_gpu,
            batch_size,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            max_blocks_per_seq,
            self.dtype,
        )?;

        Ok(())
    }

    /// Get the raw K/V pool tensors for a given layer.
    ///
    /// The attention kernel uses the block table to index into these pools.
    /// Shape: `(num_blocks, block_size, num_kv_heads, head_dim)`.
    #[must_use]
    pub fn get_pools(&self, layer_idx: usize) -> (&CudaTensor, &CudaTensor) {
        let pool = &self.layers[layer_idx];
        (&pool.k, &pool.v)
    }

    /// Number of tokens per block.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of KV heads.
    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Head dimension.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Element type of the cache buffers.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Number of transformer layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// The CUDA context this cache lives on.
    #[must_use]
    pub fn context(&self) -> &CudaContext {
        &self.ctx
    }
}

/// Launch the paged append kernel for one K or V pool.
#[allow(clippy::too_many_arguments)]
fn launch_append_paged(
    ctx: &CudaContext,
    pool: &mut CudaTensor,
    new_data: &CudaTensor,
    block_table_gpu: &cudarc::driver::CudaSlice<i32>,
    start_pos: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    new_seq_len: usize,
    dtype: DType,
) -> Result<()> {
    let total = new_seq_len * num_kv_heads * head_dim;
    let device = ctx.device();

    let kernel_name = format!("append_kv_paged_{}", kernel_suffix(dtype));
    let module_name = "append_kv_paged";
    if !device.has_func(module_name, &kernel_name) {
        device.load_ptx(cudarc::nvrtc::Ptx::from_src(PTX), module_name, KERNEL_NAMES)?;
    }

    let func = device.get_func(module_name, &kernel_name).unwrap();

    let threads = 256;
    let blocks = total.div_ceil(threads);

    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(
            cfg,
            (
                pool.cuda_slice_mut(),
                &new_data.cuda_slice(),
                block_table_gpu,
                start_pos as i32,
                block_size as i32,
                num_kv_heads as i32,
                head_dim as i32,
                new_seq_len as i32,
            ),
        )?;
    }

    Ok(())
}

/// Launch the batched paged append kernel for one K or V pool.
///
/// Appends one token per sequence across `batch_size` sequences in a single
/// kernel launch. Block tables and positions are read from pre-allocated GPU
/// buffers, making this safe for CUDA graph capture.
#[allow(clippy::too_many_arguments)]
fn launch_append_paged_batched(
    ctx: &CudaContext,
    pool: &mut CudaTensor,
    new_data: &CudaTensor,
    block_tables_gpu: &cudarc::driver::CudaSlice<i32>,
    positions_gpu: &cudarc::driver::CudaSlice<i32>,
    batch_size: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_blocks_per_seq: usize,
    dtype: DType,
) -> Result<()> {
    let device = ctx.device();

    let kernel_name = format!("append_kv_paged_batched_{}", kernel_suffix(dtype));
    let module_name = "append_kv_paged";
    if !device.has_func(module_name, &kernel_name) {
        device.load_ptx(cudarc::nvrtc::Ptx::from_src(PTX), module_name, KERNEL_NAMES)?;
    }

    let func = device.get_func(module_name, &kernel_name).unwrap();

    let total_per_token = num_kv_heads * head_dim;
    let threads = 256;
    let blocks_x = total_per_token.div_ceil(threads);

    let cfg = LaunchConfig {
        grid_dim: (blocks_x as u32, batch_size as u32, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(
            cfg,
            (
                pool.cuda_slice_mut(),
                &new_data.cuda_slice(),
                block_tables_gpu,
                positions_gpu,
                block_size as i32,
                num_kv_heads as i32,
                head_dim as i32,
                max_blocks_per_seq as i32,
            ),
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::BlockAllocator;
    use crate::cuda::CudaContext;

    fn make_ctx() -> CudaContext {
        CudaContext::new(0).expect("Failed to create CUDA context")
    }

    #[test]
    fn test_paged_kv_cache_new() {
        let ctx = make_ctx();
        let config = BlockConfig {
            block_size: 4,
            num_blocks: 10,
        };
        let cache = PagedKvCache::new(&ctx, 2, &config, 4, 16, DType::F32)
            .expect("Failed to create paged KV cache");

        assert_eq!(cache.num_layers(), 2);
        assert_eq!(cache.block_size(), 4);
        assert_eq!(cache.num_kv_heads(), 4);
        assert_eq!(cache.head_dim(), 16);
    }

    #[test]
    fn test_append_single_token() {
        let ctx = make_ctx();
        let num_kv_heads = 1;
        let head_dim = 2;
        let block_size = 4;

        let config = BlockConfig {
            block_size,
            num_blocks: 4,
        };
        let mut cache =
            PagedKvCache::new(&ctx, 1, &config, num_kv_heads, head_dim, DType::F32).unwrap();
        let mut allocator = BlockAllocator::new(&config);

        // Allocate one block and create a block table
        let block_idx = allocator.allocate().unwrap();
        let mut table = BlockTable::new(block_size);
        table.append_block(block_idx);

        let k_data = vec![1.0_f32, 2.0];
        let v_data = vec![10.0_f32, 20.0];
        let k = CudaTensor::from_slice(&ctx, &[1, num_kv_heads, head_dim], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[1, num_kv_heads, head_dim], &v_data).unwrap();

        cache.append_paged(0, &table, &k, &v, 0).unwrap();
        table.advance(1);

        // Read back the block from the pool and verify
        let (k_pool, v_pool) = cache.get_pools(0);
        let k_all: Vec<f32> = k_pool.to_vec::<f32>().unwrap();
        let v_all: Vec<f32> = v_pool.to_vec::<f32>().unwrap();

        // The data should be at physical block `block_idx`, offset 0
        let base = block_idx * block_size * num_kv_heads * head_dim;
        assert_eq!(k_all[base], 1.0);
        assert_eq!(k_all[base + 1], 2.0);
        assert_eq!(v_all[base], 10.0);
        assert_eq!(v_all[base + 1], 20.0);
    }

    #[test]
    fn test_append_fills_block_then_new_block() {
        let ctx = make_ctx();
        let num_kv_heads = 1;
        let head_dim = 2;
        let block_size = 2; // Small blocks for easy testing

        let config = BlockConfig {
            block_size,
            num_blocks: 4,
        };
        let mut cache =
            PagedKvCache::new(&ctx, 1, &config, num_kv_heads, head_dim, DType::F32).unwrap();
        let mut allocator = BlockAllocator::new(&config);

        let b0 = allocator.allocate().unwrap();
        let mut table = BlockTable::new(block_size);
        table.append_block(b0);

        // Token 0
        let k0 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[1.0_f32, 2.0]).unwrap();
        let v0 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[10.0_f32, 20.0]).unwrap();
        cache.append_paged(0, &table, &k0, &v0, 0).unwrap();
        table.advance(1);

        // Token 1 — fills block 0
        let k1 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[3.0_f32, 4.0]).unwrap();
        let v1 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[30.0_f32, 40.0]).unwrap();
        cache.append_paged(0, &table, &k1, &v1, 1).unwrap();
        table.advance(1);

        // Token 2 — needs new block
        assert!(table.needs_new_block());
        let b1 = allocator.allocate().unwrap();
        table.append_block(b1);

        let k2 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[5.0_f32, 6.0]).unwrap();
        let v2 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[50.0_f32, 60.0]).unwrap();
        cache.append_paged(0, &table, &k2, &v2, 2).unwrap();
        table.advance(1);

        // Verify data in both blocks
        let (k_pool, _) = cache.get_pools(0);
        let k_all: Vec<f32> = k_pool.to_vec::<f32>().unwrap();

        let stride = block_size * num_kv_heads * head_dim; // elements per block
                                                           // Block b0: tokens 0 and 1
        assert_eq!(k_all[b0 * stride], 1.0);
        assert_eq!(k_all[b0 * stride + 1], 2.0);
        assert_eq!(k_all[b0 * stride + 2], 3.0);
        assert_eq!(k_all[b0 * stride + 3], 4.0);
        // Block b1: token 2
        assert_eq!(k_all[b1 * stride], 5.0);
        assert_eq!(k_all[b1 * stride + 1], 6.0);
    }

    #[test]
    fn test_prefill_multiple_tokens() {
        let ctx = make_ctx();
        let num_kv_heads = 1;
        let head_dim = 2;
        let block_size = 4;

        let config = BlockConfig {
            block_size,
            num_blocks: 4,
        };
        let mut cache =
            PagedKvCache::new(&ctx, 1, &config, num_kv_heads, head_dim, DType::F32).unwrap();
        let mut allocator = BlockAllocator::new(&config);

        // Prefill 6 tokens → needs ceil(6/4) = 2 blocks
        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        let mut table = BlockTable::new(block_size);
        table.append_block(b0);
        table.append_block(b1);

        let k_data: Vec<f32> = (0..12).map(|i| i as f32).collect(); // 6 * 1 * 2
        let v_data: Vec<f32> = (0..12).map(|i| (i as f32) + 100.0).collect();
        let k = CudaTensor::from_slice(&ctx, &[6, 1, 2], &k_data).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[6, 1, 2], &v_data).unwrap();

        cache.append_paged(0, &table, &k, &v, 0).unwrap();
        table.advance(6);

        assert_eq!(table.seq_len(), 6);

        // Verify: block b0 has tokens 0-3, block b1 has tokens 4-5
        let (k_pool, v_pool) = cache.get_pools(0);
        let k_all: Vec<f32> = k_pool.to_vec::<f32>().unwrap();
        let v_all: Vec<f32> = v_pool.to_vec::<f32>().unwrap();

        let stride = block_size * num_kv_heads * head_dim;
        // Block b0: tokens 0-3 → elements 0..8
        for i in 0..8 {
            assert_eq!(k_all[b0 * stride + i], i as f32);
            assert_eq!(v_all[b0 * stride + i], (i as f32) + 100.0);
        }
        // Block b1: tokens 4-5 → elements 8..12
        for i in 0..4 {
            assert_eq!(k_all[b1 * stride + i], (i + 8) as f32);
            assert_eq!(v_all[b1 * stride + i], ((i + 8) as f32) + 100.0);
        }
    }

    #[test]
    fn test_multi_layer() {
        let ctx = make_ctx();
        let num_kv_heads = 1;
        let head_dim = 2;
        let block_size = 4;

        let config = BlockConfig {
            block_size,
            num_blocks: 4,
        };
        let mut cache =
            PagedKvCache::new(&ctx, 2, &config, num_kv_heads, head_dim, DType::F32).unwrap();
        let mut allocator = BlockAllocator::new(&config);

        let b0 = allocator.allocate().unwrap();
        let mut table = BlockTable::new(block_size);
        table.append_block(b0);

        let k0 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[1.0_f32, 2.0]).unwrap();
        let v0 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[10.0_f32, 20.0]).unwrap();
        let k1 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[5.0_f32, 6.0]).unwrap();
        let v1 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[50.0_f32, 60.0]).unwrap();

        // Same block table for both layers (they share block allocation)
        cache.append_paged(0, &table, &k0, &v0, 0).unwrap();
        cache.append_paged(1, &table, &k1, &v1, 0).unwrap();
        table.advance(1);

        let stride = block_size * num_kv_heads * head_dim;

        let (k_pool_0, _) = cache.get_pools(0);
        let k_all_0: Vec<f32> = k_pool_0.to_vec::<f32>().unwrap();
        assert_eq!(k_all_0[b0 * stride], 1.0);
        assert_eq!(k_all_0[b0 * stride + 1], 2.0);

        let (k_pool_1, _) = cache.get_pools(1);
        let k_all_1: Vec<f32> = k_pool_1.to_vec::<f32>().unwrap();
        assert_eq!(k_all_1[b0 * stride], 5.0);
        assert_eq!(k_all_1[b0 * stride + 1], 6.0);
    }

    #[test]
    fn test_block_reuse_after_free() {
        let ctx = make_ctx();
        let num_kv_heads = 1;
        let head_dim = 2;
        let block_size = 2;

        let config = BlockConfig {
            block_size,
            num_blocks: 2,
        };
        let mut cache =
            PagedKvCache::new(&ctx, 1, &config, num_kv_heads, head_dim, DType::F32).unwrap();
        let mut allocator = BlockAllocator::new(&config);

        // Request 1: use both blocks
        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        assert!(allocator.allocate().is_none()); // exhausted

        // Free request 1's blocks
        allocator.free_all(&[b0, b1]);
        assert_eq!(allocator.num_free(), 2);

        // Request 2: allocate and write
        let b2 = allocator.allocate().unwrap();
        let mut table = BlockTable::new(block_size);
        table.append_block(b2);

        let k = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[99.0_f32, 88.0]).unwrap();
        let v = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[77.0_f32, 66.0]).unwrap();
        cache.append_paged(0, &table, &k, &v, 0).unwrap();

        let (k_pool, v_pool) = cache.get_pools(0);
        let k_all: Vec<f32> = k_pool.to_vec::<f32>().unwrap();
        let v_all: Vec<f32> = v_pool.to_vec::<f32>().unwrap();
        let stride = block_size * num_kv_heads * head_dim;
        assert_eq!(k_all[b2 * stride], 99.0);
        assert_eq!(v_all[b2 * stride], 77.0);
    }

    #[test]
    fn test_batched_append_matches_per_sequence() {
        let ctx = make_ctx();
        let num_kv_heads = 1;
        let head_dim = 2;
        let block_size = 4;
        let batch_size = 2;

        let config = BlockConfig {
            block_size,
            num_blocks: 8,
        };
        let mut allocator = BlockAllocator::new(&config);

        // Sequence 0: already has 2 tokens, appending at position 2
        let b0 = allocator.allocate().unwrap();
        let mut table0 = BlockTable::new(block_size);
        table0.append_block(b0);

        // Sequence 1: already has 5 tokens (2 blocks), appending at position 5
        let b1 = allocator.allocate().unwrap();
        let b2 = allocator.allocate().unwrap();
        let mut table1 = BlockTable::new(block_size);
        table1.append_block(b1);
        table1.append_block(b2);

        // --- Per-sequence reference ---
        let mut cache_ref =
            PagedKvCache::new(&ctx, 1, &config, num_kv_heads, head_dim, DType::F32).unwrap();

        let k0 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[1.0_f32, 2.0]).unwrap();
        let v0 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[10.0_f32, 20.0]).unwrap();
        cache_ref.append_paged(0, &table0, &k0, &v0, 2).unwrap();

        let k1 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[3.0_f32, 4.0]).unwrap();
        let v1 = CudaTensor::from_slice(&ctx, &[1, 1, 2], &[30.0_f32, 40.0]).unwrap();
        cache_ref.append_paged(0, &table1, &k1, &v1, 5).unwrap();

        let (k_ref, v_ref) = cache_ref.get_pools(0);
        let k_ref_data: Vec<f32> = k_ref.to_vec::<f32>().unwrap();
        let v_ref_data: Vec<f32> = v_ref.to_vec::<f32>().unwrap();

        // --- Batched ---
        let mut cache_bat =
            PagedKvCache::new(&ctx, 1, &config, num_kv_heads, head_dim, DType::F32).unwrap();

        // Build batched K/V: (2, 1, 2)
        let k_bat = CudaTensor::from_slice(&ctx, &[2, 1, 2], &[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let v_bat =
            CudaTensor::from_slice(&ctx, &[2, 1, 2], &[10.0_f32, 20.0, 30.0, 40.0]).unwrap();

        // Build GPU buffers
        let max_blocks_per_seq = 2;
        let positions = [2_i32, 5];
        let mut flat_tables = vec![0_i32; batch_size * max_blocks_per_seq];
        flat_tables[0] = b0 as i32;
        flat_tables[1] = 0; // padding
        flat_tables[2] = b1 as i32;
        flat_tables[3] = b2 as i32;

        let positions_gpu = ctx.device().htod_sync_copy(&positions).unwrap();
        let tables_gpu = ctx.device().htod_sync_copy(&flat_tables).unwrap();

        cache_bat
            .append_paged_batched(
                0,
                &k_bat,
                &v_bat,
                &tables_gpu,
                &positions_gpu,
                batch_size,
                max_blocks_per_seq,
            )
            .unwrap();

        let (k_bat_pool, v_bat_pool) = cache_bat.get_pools(0);
        let k_bat_data: Vec<f32> = k_bat_pool.to_vec::<f32>().unwrap();
        let v_bat_data: Vec<f32> = v_bat_pool.to_vec::<f32>().unwrap();

        // Verify the blocks written match
        let stride = block_size * num_kv_heads * head_dim;
        // Seq 0: position 2 in block b0
        let offset0 = b0 * stride + 2 * num_kv_heads * head_dim;
        assert_eq!(k_bat_data[offset0], k_ref_data[offset0]);
        assert_eq!(k_bat_data[offset0 + 1], k_ref_data[offset0 + 1]);
        assert_eq!(v_bat_data[offset0], v_ref_data[offset0]);

        // Seq 1: position 5 in block b2 (5 / 4 = block 1, offset 1)
        let offset1 = b2 * stride + 1 * num_kv_heads * head_dim;
        assert_eq!(k_bat_data[offset1], k_ref_data[offset1]);
        assert_eq!(k_bat_data[offset1 + 1], k_ref_data[offset1 + 1]);
        assert_eq!(v_bat_data[offset1], v_ref_data[offset1]);
    }
}
