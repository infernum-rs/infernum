//! AttentionOps, PagedAttentionOps, PagedKvCacheOps, KvCacheOps implementations.
//!
//! Fused prefill/decode attention dispatched to Metal GPU kernels.
//! Paged attention, KV cache, and combine_attention_with_lse remain CPU-side.

use bytemuck::{Pod, Zeroable};
use infernum::backend::{
    AttentionOps, FusedDecodeOps, KvCacheOps, PagedAttentionOps, PagedKvCacheOps,
};
use infernum::block_allocator::{BlockConfig, BlockTable};
use infernum::tensor::Tensor;
use infernum::DType;
use infernum::Result;
use metal::MTLSize;

use crate::context::reduction_threadgroup_size;
use crate::tensor::MetalTensor;
use crate::{MetalBackend, MetalContext, MetalKvCache, MetalPagedKvCache};

// ---- GPU param structs ----

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttentionPrefillParams {
    seq_len: u32,
    kv_len: u32,
    n_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    offset: u32,
    scale: f32,
    softcap: f32,
    sliding_window: i32,
    compute_lse: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttentionDecodeParams {
    kv_len: u32,
    n_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    scale: f32,
    softcap: f32,
    sliding_window: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PagedAttentionDecodeParams {
    batch_size: u32,
    n_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    max_blocks_per_seq: u32,
    scale: f32,
    softcap: f32,
    sliding_window: i32,
}

// ---- Fused Attention (GPU) ----

impl AttentionOps for MetalBackend {
    #[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
    fn fused_attention_prefill(
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        offset: usize,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<MetalTensor> {
        let (out, _) =
            attention_prefill_gpu(q, k, v, offset, scale, softcap, sliding_window, false);
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn fused_attention_decode(
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<MetalTensor> {
        Ok(attention_decode_gpu(
            q,
            k,
            v,
            scale,
            softcap,
            sliding_window,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn fused_attention_prefill_with_lse(
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        offset: usize,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<(MetalTensor, MetalTensor)> {
        Ok(attention_prefill_gpu(
            q,
            k,
            v,
            offset,
            scale,
            softcap,
            sliding_window,
            true,
        ))
    }

    fn combine_attention_with_lse(
        out1: &MetalTensor,
        lse1: &MetalTensor,
        out2: &MetalTensor,
        lse2: &MetalTensor,
    ) -> Result<MetalTensor> {
        // CPU-side: simple per-head rescaling, not compute-bound.
        let shape = out1.shape();
        let total = out1.numel();
        let head_dim = *shape.last().unwrap();
        let num_heads = total / head_dim;

        let o1 = out1.as_f32_slice();
        let l1 = lse1.as_f32_slice();
        let o2 = out2.as_f32_slice();
        let l2 = lse2.as_f32_slice();

        let mut out = vec![0.0f32; total];

        for h in 0..num_heads {
            let m = l1[h].max(l2[h]);
            let e1 = (l1[h] - m).exp();
            let e2 = (l2[h] - m).exp();
            let denom = e1 + e2;
            for d in 0..head_dim {
                out[h * head_dim + d] =
                    (e1 * o1[h * head_dim + d] + e2 * o2[h * head_dim + d]) / denom;
            }
        }

        Ok(MetalTensor::from_f32(out1.context(), shape, &out))
    }
}

/// Dispatch fused prefill attention on the GPU.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_arguments
)]
fn attention_prefill_gpu(
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    offset: usize,
    scale: Option<f32>,
    softcap: Option<f32>,
    sliding_window: Option<usize>,
    compute_lse: bool,
) -> (MetalTensor, MetalTensor) {
    let q_shape = q.shape();
    let seq_len = q_shape[0];
    let n_heads = q_shape[1];
    let head_dim = q_shape[2];
    let kv_len = k.shape()[0];
    let kv_heads = k.shape()[1];

    let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

    let ctx = q.context();
    let out = MetalTensor::zeros(ctx, q_shape, DType::F32);
    let lse = MetalTensor::zeros(ctx, &[seq_len, n_heads], DType::F32);

    let params = AttentionPrefillParams {
        seq_len: seq_len as u32,
        kv_len: kv_len as u32,
        n_heads: n_heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        offset: offset as u32,
        scale,
        softcap: softcap.unwrap_or(0.0),
        sliding_window: sliding_window.map_or(-1, |w| w as i32),
        compute_lse: u32::from(compute_lse),
    };

    let tg_size = reduction_threadgroup_size(kv_len.max(head_dim));
    let num_threadgroups = seq_len * n_heads;

    ctx.dispatch_threadgroups(
        "fused_attention_prefill_f32",
        &[
            (q.metal_buffer(), q.buffer_offset()),
            (k.metal_buffer(), k.buffer_offset()),
            (v.metal_buffer(), v.buffer_offset()),
            (out.metal_buffer(), out.buffer_offset()),
            (lse.metal_buffer(), lse.buffer_offset()),
        ],
        bytemuck::bytes_of(&params),
        MTLSize::new(num_threadgroups as u64, 1, 1),
        MTLSize::new(tg_size as u64, 1, 1),
        tg_size * std::mem::size_of::<f32>(),
    );

    (out, lse)
}

/// Dispatch fused decode attention on the GPU (single query token).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
fn attention_decode_gpu(
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    scale: Option<f32>,
    softcap: Option<f32>,
    sliding_window: Option<usize>,
) -> MetalTensor {
    let q_shape = q.shape();
    let n_heads = q_shape[1];
    let head_dim = q_shape[2];
    let kv_len = k.shape()[0];
    let kv_heads = k.shape()[1];

    let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

    let ctx = q.context();
    let out = MetalTensor::zeros(ctx, q_shape, DType::F32);

    let params = AttentionDecodeParams {
        kv_len: kv_len as u32,
        n_heads: n_heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        scale,
        softcap: softcap.unwrap_or(0.0),
        sliding_window: sliding_window.map_or(-1, |w| w as i32),
    };

    let tg_size = reduction_threadgroup_size(kv_len.max(head_dim));

    ctx.dispatch_threadgroups(
        "fused_attention_decode_f32",
        &[
            (q.metal_buffer(), q.buffer_offset()),
            (k.metal_buffer(), k.buffer_offset()),
            (v.metal_buffer(), v.buffer_offset()),
            (out.metal_buffer(), out.buffer_offset()),
        ],
        bytemuck::bytes_of(&params),
        MTLSize::new(n_heads as u64, 1, 1),
        MTLSize::new(tg_size as u64, 1, 1),
        tg_size * std::mem::size_of::<f32>(),
    );

    out
}

// ---- Paged KV Cache ----

impl PagedKvCacheOps for MetalBackend {
    fn allocate_paged_kv_cache(
        device: &MetalContext,
        num_layers: usize,
        block_config: &BlockConfig,
        num_kv_heads: usize,
        head_dim: usize,
        cache_dtype: DType,
    ) -> Result<MetalPagedKvCache> {
        let block_size = block_config.block_size;
        let num_blocks = block_config.num_blocks;
        let pool_shape = [num_blocks * block_size, num_kv_heads, head_dim];

        // Use F16 for KV cache when model weights are quantized or F16,
        // otherwise default to F32.
        let pool_dtype = if cache_dtype.is_quantized() || cache_dtype == DType::F16 {
            DType::F16
        } else {
            DType::F32
        };

        let mut k_pools = Vec::with_capacity(num_layers);
        let mut v_pools = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            k_pools.push(MetalTensor::zeros(device, &pool_shape, pool_dtype));
            v_pools.push(MetalTensor::zeros(device, &pool_shape, pool_dtype));
        }

        Ok(MetalPagedKvCache {
            k_pools,
            v_pools,
            block_size,
            num_kv_heads,
            head_dim,
        })
    }

    fn append_paged(
        cache: &mut MetalPagedKvCache,
        layer_idx: usize,
        block_table: &BlockTable,
        k: &MetalTensor,
        v: &MetalTensor,
        start_pos: usize,
    ) -> Result<()> {
        let head_stride = cache.num_kv_heads * cache.head_dim;
        let seq_len = k.shape()[0];
        let elem_size = k.dtype().size_in_bytes();
        let byte_stride = head_stride * elem_size;

        k.context().flush();

        let k_bytes = k.as_bytes();
        let v_bytes = v.as_bytes();
        let k_pool_bytes = cache.k_pools[layer_idx].as_bytes_mut();
        let v_pool_bytes = cache.v_pools[layer_idx].as_bytes_mut();

        for t in 0..seq_len {
            let pos = start_pos + t;
            let block_idx = pos / cache.block_size;
            let block_offset = pos % cache.block_size;
            let physical_block = block_table.blocks()[block_idx];
            let dst_byte = (physical_block * cache.block_size + block_offset) * byte_stride;
            let src_byte = t * byte_stride;

            k_pool_bytes[dst_byte..dst_byte + byte_stride]
                .copy_from_slice(&k_bytes[src_byte..src_byte + byte_stride]);
            v_pool_bytes[dst_byte..dst_byte + byte_stride]
                .copy_from_slice(&v_bytes[src_byte..src_byte + byte_stride]);
        }

        Ok(())
    }

    fn get_pools(cache: &MetalPagedKvCache, layer_idx: usize) -> (&MetalTensor, &MetalTensor) {
        (&cache.k_pools[layer_idx], &cache.v_pools[layer_idx])
    }

    fn block_size(cache: &MetalPagedKvCache) -> usize {
        cache.block_size
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::cast_possible_truncation,
        clippy::items_after_statements
    )]
    fn append_paged_batched(
        cache: &mut MetalPagedKvCache,
        layer_idx: usize,
        k: &MetalTensor,
        v: &MetalTensor,
        block_tables: &MetalTensor,
        positions: &MetalTensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        let ctx = k.context();
        let total_per_token = cache.num_kv_heads * cache.head_dim;

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct AppendKvPagedParams {
            batch_size: u32,
            block_size: u32,
            num_kv_heads: u32,
            head_dim: u32,
            max_blocks_per_seq: u32,
        }

        let params = AppendKvPagedParams {
            batch_size: batch_size as u32,
            block_size: cache.block_size as u32,
            num_kv_heads: cache.num_kv_heads as u32,
            head_dim: cache.head_dim as u32,
            max_blocks_per_seq: max_blocks_per_seq as u32,
        };

        // Fused K+V append: single dispatch with 3D grid (elems, batch, 2)
        let kernel = if k.dtype() == DType::F16 {
            "append_kv_paged_batched_fused_f16"
        } else {
            "append_kv_paged_batched_fused_f32"
        };
        ctx.dispatch_3d(
            kernel,
            &[
                (
                    cache.k_pools[layer_idx].metal_buffer(),
                    cache.k_pools[layer_idx].buffer_offset(),
                ),
                (
                    cache.v_pools[layer_idx].metal_buffer(),
                    cache.v_pools[layer_idx].buffer_offset(),
                ),
                (k.metal_buffer(), k.buffer_offset()),
                (v.metal_buffer(), v.buffer_offset()),
                (block_tables.metal_buffer(), block_tables.buffer_offset()),
                (positions.metal_buffer(), positions.buffer_offset()),
            ],
            bytemuck::bytes_of(&params),
            total_per_token,
            batch_size,
            2,
        );

        Ok(())
    }
}

// ---- Paged Attention (Decode) ----

#[allow(clippy::needless_range_loop)]
impl PagedAttentionOps for MetalBackend {
    #[allow(
        clippy::too_many_arguments,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn paged_attention_decode(
        q: &MetalTensor,
        k_pool: &MetalTensor,
        v_pool: &MetalTensor,
        block_tables: &MetalTensor,
        seq_lens: &MetalTensor,
        block_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<MetalTensor> {
        let q_shape = q.shape();
        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let head_dim = q_shape[2];
        let num_kv_heads = k_pool.shape()[1];

        let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

        let ctx = q.context();
        let out_shape = [batch_size, num_heads, head_dim];
        let out = MetalTensor::zeros(ctx, &out_shape, q.dtype());

        // Use the caller-provided max_seq_len for threadgroup sizing
        // (avoids a GPU flush to read seq_lens on CPU).
        let max_sl = max_seq_len;

        let params = PagedAttentionDecodeParams {
            batch_size: batch_size as u32,
            n_heads: num_heads as u32,
            kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            block_size: block_size as u32,
            max_blocks_per_seq: max_blocks_per_seq as u32,
            scale,
            softcap: softcap.unwrap_or(0.0),
            sliding_window: sliding_window.map_or(-1, |w| w as i32),
        };

        let tg_size = reduction_threadgroup_size(max_sl.max(head_dim));
        let num_threadgroups = batch_size * num_heads;

        let kernel = if q.dtype() == DType::F16 {
            "paged_attention_decode_f16"
        } else {
            "paged_attention_decode_f32"
        };
        ctx.dispatch_threadgroups(
            kernel,
            &[
                (q.metal_buffer(), q.buffer_offset()),
                (k_pool.metal_buffer(), k_pool.buffer_offset()),
                (v_pool.metal_buffer(), v_pool.buffer_offset()),
                (block_tables.metal_buffer(), block_tables.buffer_offset()),
                (seq_lens.metal_buffer(), seq_lens.buffer_offset()),
                (out.metal_buffer(), out.buffer_offset()),
            ],
            bytemuck::bytes_of(&params),
            MTLSize::new(num_threadgroups as u64, 1, 1),
            MTLSize::new(tg_size as u64, 1, 1),
            tg_size * std::mem::size_of::<f32>(),
        );

        Ok(out)
    }

    fn gather_paged_kv(
        paged_kv: &MetalPagedKvCache,
        layer_idx: usize,
        block_table: &BlockTable,
    ) -> Result<(MetalTensor, MetalTensor)> {
        let seq_len = block_table.seq_len();
        let head_stride = paged_kv.num_kv_heads * paged_kv.head_dim;
        let pool_dtype = paged_kv.k_pools[layer_idx].dtype();
        let elem_size = pool_dtype.size_in_bytes();
        let byte_stride = head_stride * elem_size;

        let k_pool_bytes = paged_kv.k_pools[layer_idx].as_bytes();
        let v_pool_bytes = paged_kv.v_pools[layer_idx].as_bytes();

        let mut k_bytes = Vec::with_capacity(seq_len * byte_stride);
        let mut v_bytes = Vec::with_capacity(seq_len * byte_stride);

        for pos in 0..seq_len {
            let block_idx = pos / paged_kv.block_size;
            let block_offset = pos % paged_kv.block_size;
            let phys_block = block_table.blocks()[block_idx];
            let off = (phys_block * paged_kv.block_size + block_offset) * byte_stride;

            k_bytes.extend_from_slice(&k_pool_bytes[off..off + byte_stride]);
            v_bytes.extend_from_slice(&v_pool_bytes[off..off + byte_stride]);
        }

        let ctx = paged_kv.k_pools[layer_idx].context();
        let shape = [seq_len, paged_kv.num_kv_heads, paged_kv.head_dim];
        Ok((
            MetalTensor::from_raw_bytes(ctx, &shape, pool_dtype, &k_bytes),
            MetalTensor::from_raw_bytes(ctx, &shape, pool_dtype, &v_bytes),
        ))
    }
}

// ---- Contiguous KV Cache (DeepSeek MLA) ----

impl KvCacheOps for MetalBackend {
    fn append_kv(
        cache: &mut MetalKvCache,
        layer_idx: usize,
        k: &MetalTensor,
        v: &MetalTensor,
    ) -> Result<()> {
        let layer = &mut cache.layers[layer_idx];
        layer.k.extend_from_slice(k.as_f32_slice());
        layer.v.extend_from_slice(v.as_f32_slice());
        let new_tokens = k.shape()[0];
        layer.len += new_tokens;
        Ok(())
    }

    fn get_kv(cache: &MetalKvCache, layer_idx: usize) -> (MetalTensor, MetalTensor) {
        let layer = &cache.layers[layer_idx];
        let shape = [layer.len, cache.num_kv_heads, cache.head_dim];
        (
            MetalTensor::from_f32(&cache.ctx, &shape, &layer.k),
            MetalTensor::from_f32(&cache.ctx, &shape, &layer.v),
        )
    }

    fn get_kv_up_to(
        cache: &MetalKvCache,
        layer_idx: usize,
        len: usize,
    ) -> (MetalTensor, MetalTensor) {
        let layer = &cache.layers[layer_idx];
        let stride = cache.num_kv_heads * cache.head_dim;
        let n = len * stride;
        let shape = [len, cache.num_kv_heads, cache.head_dim];
        (
            MetalTensor::from_f32(&cache.ctx, &shape, &layer.k[..n]),
            MetalTensor::from_f32(&cache.ctx, &shape, &layer.v[..n]),
        )
    }
}

// ---- Fused decode ops ----

/// Params for the fused RoPE + KV-cache append kernel.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RopeKvAppendParams {
    q_heads: u32,
    k_heads: u32,
    head_dim: u32,
    half_dim: u32,
    block_size: u32,
    max_blocks_per_seq: u32,
}

impl FusedDecodeOps for MetalBackend {
    #[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
    fn rope_kv_append_batched(
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        cos_cache: &MetalTensor,
        sin_cache: &MetalTensor,
        positions: &MetalTensor,
        paged_kv: &mut MetalPagedKvCache,
        layer_idx: usize,
        block_tables: &MetalTensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<MetalTensor> {
        let q_shape = q.shape();
        let k_shape = k.shape();
        let q_heads = q_shape[1];
        let k_heads = k_shape[1];
        let head_dim = q_shape[2];
        let half_dim = head_dim / 2;

        let ctx = q.context();
        let q_out = MetalTensor::zeros(ctx, q_shape, q.dtype());

        let params = RopeKvAppendParams {
            q_heads: q_heads as u32,
            k_heads: k_heads as u32,
            head_dim: head_dim as u32,
            half_dim: half_dim as u32,
            block_size: paged_kv.block_size as u32,
            max_blocks_per_seq: max_blocks_per_seq as u32,
        };

        let n = batch_size * (q_heads * half_dim + k_heads * half_dim + k_heads * head_dim);

        let kernel = if q.dtype() == DType::F16 {
            "rope_kv_append_fused_f16"
        } else {
            "rope_kv_append_fused_f32"
        };

        ctx.dispatch_1d(
            kernel,
            &[
                (q.metal_buffer(), q.buffer_offset()),
                (k.metal_buffer(), k.buffer_offset()),
                (v.metal_buffer(), v.buffer_offset()),
                (cos_cache.metal_buffer(), cos_cache.buffer_offset()),
                (sin_cache.metal_buffer(), sin_cache.buffer_offset()),
                (positions.metal_buffer(), positions.buffer_offset()),
                (q_out.metal_buffer(), q_out.buffer_offset()),
                (
                    paged_kv.k_pools[layer_idx].metal_buffer(),
                    paged_kv.k_pools[layer_idx].buffer_offset(),
                ),
                (
                    paged_kv.v_pools[layer_idx].metal_buffer(),
                    paged_kv.v_pools[layer_idx].buffer_offset(),
                ),
                (block_tables.metal_buffer(), block_tables.buffer_offset()),
            ],
            bytemuck::bytes_of(&params),
            n,
        );

        Ok(q_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalKvLayer;
    use infernum::backend::{AttentionOps, KvCacheOps, PagedAttentionOps, PagedKvCacheOps};
    use infernum::block_allocator::{BlockConfig, BlockTable};
    use infernum::tensor::Tensor;

    fn ctx() -> MetalContext {
        MetalContext::new()
    }

    // ---- Paged KV Cache ----

    #[test]
    fn test_paged_kv_cache_allocate() {
        let context = ctx();
        let block_config = BlockConfig {
            block_size: 16,
            num_blocks: 4,
        };
        let cache = MetalBackend::allocate_paged_kv_cache(
            &context,
            2, // num_layers
            &block_config,
            4, // num_kv_heads
            8, // head_dim
            DType::F32,
        )
        .unwrap();

        assert_eq!(cache.k_pools.len(), 2);
        assert_eq!(cache.v_pools.len(), 2);
        assert_eq!(cache.block_size, 16);
        // Pool shape: (4*16, 4, 8) = (64, 4, 8)
        assert_eq!(cache.k_pools[0].shape(), &[64, 4, 8]);
    }

    #[test]
    fn test_paged_kv_cache_append_and_get_pools() {
        let c = ctx();
        let block_config = BlockConfig {
            block_size: 4,
            num_blocks: 2,
        };
        let mut cache =
            MetalBackend::allocate_paged_kv_cache(&c, 1, &block_config, 1, 2, DType::F32).unwrap();

        // Block table: seq positions 0..3 map to physical block 1
        let block_table = BlockTable::from_raw(vec![1], 3, 4);

        // K/V data: 3 tokens, 1 head, dim 2
        let k = MetalTensor::from_f32(&c, &[3, 1, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v = MetalTensor::from_f32(&c, &[3, 1, 2], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

        MetalBackend::append_paged(&mut cache, 0, &block_table, &k, &v, 0).unwrap();

        let (k_pool, v_pool) = MetalBackend::get_pools(&cache, 0);
        let k_data = k_pool.as_f32_slice();
        let v_data = v_pool.as_f32_slice();

        // Physical block 1, offsets 0..2: should have our data
        // Pool layout: (num_blocks*block_size, heads, dim) = (8, 1, 2)
        // Block 1 starts at index 4 (block_size=4)
        // Slot 4: token 0 -> [1.0, 2.0]
        // Slot 5: token 1 -> [3.0, 4.0]
        // Slot 6: token 2 -> [5.0, 6.0]
        assert_eq!(&k_data[8..10], &[1.0, 2.0]);
        assert_eq!(&k_data[10..12], &[3.0, 4.0]);
        assert_eq!(&k_data[12..14], &[5.0, 6.0]);
        assert_eq!(&v_data[8..10], &[10.0, 20.0]);
        assert_eq!(&v_data[10..12], &[30.0, 40.0]);
        assert_eq!(&v_data[12..14], &[50.0, 60.0]);
    }

    #[test]
    fn test_gather_paged_kv() {
        let c = ctx();
        let block_config = BlockConfig {
            block_size: 2,
            num_blocks: 4,
        };
        let mut cache =
            MetalBackend::allocate_paged_kv_cache(&c, 1, &block_config, 1, 2, DType::F32).unwrap();

        // Two blocks: positions 0..1 in block 2, positions 2..3 in block 0
        let block_table = BlockTable::from_raw(vec![2, 0], 4, 2);
        let k = MetalTensor::from_f32(&c, &[4, 1, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let v = MetalTensor::from_f32(
            &c,
            &[4, 1, 2],
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        );

        MetalBackend::append_paged(&mut cache, 0, &block_table, &k, &v, 0).unwrap();

        let (k_gathered, v_gathered) =
            MetalBackend::gather_paged_kv(&cache, 0, &block_table).unwrap();

        assert_eq!(k_gathered.shape(), &[4, 1, 2]);
        let kg = k_gathered.as_f32_slice();
        let vg = v_gathered.as_f32_slice();
        // Should recover the original data in sequence order
        assert_eq!(kg, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(vg, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    }

    // ---- Paged Attention Decode ----

    #[test]
    fn test_paged_attention_decode_single() {
        let c = ctx();
        let block_config = BlockConfig {
            block_size: 4,
            num_blocks: 2,
        };
        let mut cache =
            MetalBackend::allocate_paged_kv_cache(&c, 1, &block_config, 1, 4, DType::F32).unwrap();

        // 3 tokens in block 0
        let block_table = BlockTable::from_raw(vec![0], 3, 4);
        let k = MetalTensor::from_f32(
            &c,
            &[3, 1, 4],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        let v = MetalTensor::from_f32(
            &c,
            &[3, 1, 4],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
        );
        MetalBackend::append_paged(&mut cache, 0, &block_table, &k, &v, 0).unwrap();

        // Query: batch=1, heads=1, dim=4
        let q = MetalTensor::from_f32(&c, &[1, 1, 4], &[0.0, 0.0, 1.0, 0.0]);
        let (k_pool, v_pool) = MetalBackend::get_pools(&cache, 0);
        let bt = MetalTensor::from_raw_bytes(&c, &[1], DType::U32, bytemuck::cast_slice(&[0i32]));
        let sl = MetalTensor::from_raw_bytes(&c, &[1], DType::U32, bytemuck::cast_slice(&[3i32]));

        let out = MetalBackend::paged_attention_decode(
            &q, k_pool, v_pool, &bt, &sl, 4, 1, 3, None, None, None,
        )
        .unwrap();

        assert_eq!(out.shape(), &[1, 1, 4]);
        let out_data = out.as_f32_slice();
        // Q=[0,0,1,0] should attend most strongly to K[2]=[0,0,1,0]
        // => V[2]=[0,0,3,0] should dominate
        assert!(
            out_data[2] > out_data[0] && out_data[2] > out_data[1],
            "Expected V[2] to dominate: {out_data:?}"
        );
    }

    // ---- Contiguous KV Cache ----

    #[test]
    fn test_contiguous_kv_cache_append_and_get() {
        let c = ctx();
        let mut cache = MetalKvCache {
            layers: vec![MetalKvLayer {
                k: Vec::new(),
                v: Vec::new(),
                len: 0,
            }],
            num_kv_heads: 2,
            head_dim: 3,
            ctx: c.clone(),
        };

        // Append 2 tokens: shape (2, 2, 3)
        let k = MetalTensor::from_f32(
            &c,
            &[2, 2, 3],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let v = MetalTensor::from_f32(
            &c,
            &[2, 2, 3],
            &[
                -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0,
            ],
        );

        MetalBackend::append_kv(&mut cache, 0, &k, &v).unwrap();

        let (k_out, v_out) = MetalBackend::get_kv(&cache, 0);
        assert_eq!(k_out.shape(), &[2, 2, 3]);
        assert_eq!(
            k_out.as_f32_slice(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
        assert_eq!(
            v_out.as_f32_slice(),
            &[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0]
        );

        // get_kv_up_to(1) should return only the first token
        let (k_up, _v_up) = MetalBackend::get_kv_up_to(&cache, 0, 1);
        assert_eq!(k_up.shape(), &[1, 2, 3]);
        assert_eq!(k_up.as_f32_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // ---- Fused Attention (existing, verifies the full pipeline) ----

    #[test]
    fn test_attention_with_kv_cache_round_trip() {
        // Verify that storing K/V in contiguous cache and using fused_attention_decode
        // produces valid output.
        let c = ctx();
        let mut cache = MetalKvCache {
            layers: vec![MetalKvLayer {
                k: Vec::new(),
                v: Vec::new(),
                len: 0,
            }],
            num_kv_heads: 1,
            head_dim: 4,
            ctx: c.clone(),
        };

        // Prefill: 3 tokens
        let k = MetalTensor::from_f32(
            &c,
            &[3, 1, 4],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        let v = MetalTensor::from_f32(
            &c,
            &[3, 1, 4],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        MetalBackend::append_kv(&mut cache, 0, &k, &v).unwrap();

        // Decode: query is 1 token
        let q = MetalTensor::from_f32(&c, &[1, 1, 4], &[0.0, 0.0, 1.0, 0.0]);
        let (k_cached, v_cached) = MetalBackend::get_kv(&cache, 0);

        let out = MetalBackend::fused_attention_decode(&q, &k_cached, &v_cached, None, None, None)
            .unwrap();

        assert_eq!(out.shape(), &[1, 1, 4]);
        let out_data = out.as_f32_slice();
        // Q=[0,0,1,0] should attend most to K[2]=[0,0,1,0] => V[2]=[0,0,1,0]
        assert!(
            out_data[2] > out_data[0] && out_data[2] > out_data[1],
            "Expected dim 2 to dominate: {out_data:?}"
        );
    }

    // ---- GPU Attention Kernel Tests ----

    #[test]
    fn test_attention_prefill_causal_mask() {
        // Prefill with 3 query positions. Verify causal masking:
        // - q[0] at offset=0 can only see k[0]
        // - q[1] at offset=0 can see k[0..1]
        // - q[2] at offset=0 can see k[0..2]
        let c = ctx();
        // K/V: 3 tokens, 1 head, dim=4. K are orthogonal unit vectors.
        let k = MetalTensor::from_f32(
            &c,
            &[3, 1, 4],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        let v = MetalTensor::from_f32(
            &c,
            &[3, 1, 4],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
        );
        // Q: 3 tokens, each matching a different K.
        // q[0]=[1,0,0,0] -> only sees k[0], so out = v[0] = [1,0,0,0]
        // q[2]=[0,0,1,0] -> sees k[0..2], highest dot with k[2]
        let q = MetalTensor::from_f32(
            &c,
            &[3, 1, 4],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );

        let out = MetalBackend::fused_attention_prefill(&q, &k, &v, 0, None, None, None).unwrap();
        assert_eq!(out.shape(), &[3, 1, 4]);
        let data = out.as_f32_slice();

        // q[0] can only attend to k[0]: output should be exactly v[0]
        assert!(
            (data[0] - 1.0).abs() < 1e-5,
            "q[0] should output v[0]: got {:?}",
            &data[0..4]
        );
        // q[2] should have dim 2 dominating (highest dot with k[2])
        assert!(
            data[10] > data[8] && data[10] > data[9],
            "q[2] dim 2 should dominate: {:?}",
            &data[8..12]
        );
    }

    #[test]
    fn test_attention_decode_gqa() {
        // GQA: 4 query heads, 2 KV heads. Heads 0,1 share kv_head 0; heads 2,3 share kv_head 1.
        let c = ctx();
        // 2 KV tokens, 2 KV heads, dim=2
        let k = MetalTensor::from_f32(
            &c,
            &[2, 2, 2],
            &[
                1.0, 0.0, // kv_token=0, kv_head=0
                0.0, 1.0, // kv_token=0, kv_head=1
                0.0, 1.0, // kv_token=1, kv_head=0
                1.0, 0.0, // kv_token=1, kv_head=1
            ],
        );
        let v = MetalTensor::from_f32(
            &c,
            &[2, 2, 2],
            &[
                1.0, 0.0, // kv_token=0, kv_head=0: v=[1,0]
                0.0, 1.0, // kv_token=0, kv_head=1: v=[0,1]
                2.0, 0.0, // kv_token=1, kv_head=0: v=[2,0]
                0.0, 2.0, // kv_token=1, kv_head=1: v=[0,2]
            ],
        );
        // Q: 1 token, 4 heads, dim=2
        let q = MetalTensor::from_f32(
            &c,
            &[1, 4, 2],
            &[
                1.0, 0.0, // head 0 -> kv_head 0, matches k[0,0]=[1,0]
                0.0, 1.0, // head 1 -> kv_head 0, matches k[1,0]=[0,1]
                0.0, 1.0, // head 2 -> kv_head 1, matches k[0,1]=[0,1]
                1.0, 0.0, // head 3 -> kv_head 1, matches k[1,1]=[1,0]
            ],
        );

        let out = MetalBackend::fused_attention_decode(&q, &k, &v, None, None, None).unwrap();
        assert_eq!(out.shape(), &[1, 4, 2]);
        let data = out.as_f32_slice();

        // Head 0 (kv_head 0): q=[1,0] dot k[0,0]=[1,0]=1, k[1,0]=[0,1]=0 => v[0,0]=[1,0] dominates
        assert!(data[0] > data[1], "Head 0 dim 0 should dominate: {data:?}");
        // Head 1 (kv_head 0): q=[0,1] dot k[0,0]=[1,0]=0, k[1,0]=[0,1]=1 => v[1,0]=[2,0] dominates
        assert!(
            data[2] > data[3],
            "Head 1 dim 0 should dominate (v=[2,0]): {data:?}"
        );
        // Head 2 (kv_head 1): q=[0,1] dot k[0,1]=[0,1]=1, k[1,1]=[1,0]=0 => v[0,1]=[0,1] dominates
        assert!(data[5] > data[4], "Head 2 dim 1 should dominate: {data:?}");
        // Head 3 (kv_head 1): q=[1,0] dot k[0,1]=[0,1]=0, k[1,1]=[1,0]=1 => v[1,1]=[0,2] dominates
        assert!(
            data[7] > data[6],
            "Head 3 dim 1 should dominate (v=[0,2]): {data:?}"
        );
    }

    #[test]
    fn test_attention_prefill_with_lse() {
        let c = ctx();
        // Single token, single head, dim=2. K has 1 token.
        let q = MetalTensor::from_f32(&c, &[1, 1, 2], &[1.0, 0.0]);
        let k = MetalTensor::from_f32(&c, &[1, 1, 2], &[1.0, 0.0]);
        let v = MetalTensor::from_f32(&c, &[1, 1, 2], &[3.0, 7.0]);

        let (out, lse) =
            MetalBackend::fused_attention_prefill_with_lse(&q, &k, &v, 0, None, None, None)
                .unwrap();

        assert_eq!(out.shape(), &[1, 1, 2]);
        assert_eq!(lse.shape(), &[1, 1]);
        let out_data = out.as_f32_slice();
        let lse_data = lse.as_f32_slice();

        // Only one KV token: output = V exactly, LSE = score (no other terms)
        assert!(
            (out_data[0] - 3.0).abs() < 1e-4,
            "Single-token output should be V: {out_data:?}"
        );
        assert!(
            (out_data[1] - 7.0).abs() < 1e-4,
            "Single-token output should be V: {out_data:?}"
        );
        // LSE = max + ln(sum_exp) = score + ln(1) = score
        // score = dot(q,k) * scale = 1.0 * (1/sqrt(2))
        let expected_lse = 1.0 / 2.0f32.sqrt();
        assert!(
            (lse_data[0] - expected_lse).abs() < 1e-4,
            "LSE should be score for single token: got {}, expected {}",
            lse_data[0],
            expected_lse
        );
    }

    #[test]
    fn test_attention_decode_sliding_window() {
        let c = ctx();
        // 4 KV tokens, 1 head, dim=4. Sliding window = 2.
        // Decode query at position 3 should only attend to positions [2, 3].
        let k = MetalTensor::from_f32(
            &c,
            &[4, 1, 4],
            &[
                1.0, 0.0, 0.0, 0.0, // pos 0 — should be masked
                0.0, 1.0, 0.0, 0.0, // pos 1 — should be masked
                0.0, 0.0, 1.0, 0.0, // pos 2 — visible
                0.0, 0.0, 0.0, 1.0, // pos 3 — visible
            ],
        );
        let v = MetalTensor::from_f32(
            &c,
            &[4, 1, 4],
            &[
                100.0, 0.0, 0.0, 0.0, // pos 0: if visible, would dominate
                0.0, 100.0, 0.0, 0.0, // pos 1: if visible, would dominate
                0.0, 0.0, 5.0, 0.0, // pos 2
                0.0, 0.0, 0.0, 5.0, // pos 3
            ],
        );
        // Q matches pos 0 strongly, but window should block it
        let q = MetalTensor::from_f32(&c, &[1, 1, 4], &[1.0, 0.0, 0.0, 0.0]);

        let out = MetalBackend::fused_attention_decode(&q, &k, &v, None, None, Some(2)).unwrap();
        let data = out.as_f32_slice();

        // Pos 0 and 1 should be masked. Output should be a mix of v[2] and v[3] only.
        assert!(
            data[0].abs() < 1e-4 && data[1].abs() < 1e-4,
            "Dims 0,1 should be ~0 (masked positions): {data:?}"
        );
        assert!(
            data[2] > 0.0 || data[3] > 0.0,
            "Some visible position value should appear: {data:?}"
        );
    }
}
