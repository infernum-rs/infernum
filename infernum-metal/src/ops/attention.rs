//! AttentionOps, PagedAttentionOps, PagedKvCacheOps, KvCacheOps implementations.
//!
//! Fused prefill/decode attention dispatched to Metal GPU kernels.
//! Paged attention, KV cache, and combine_attention_with_lse remain CPU-side.

use bytemuck::{Pod, Zeroable};
use infernum::backend::{AttentionOps, KvCacheOps, PagedAttentionOps, PagedKvCacheOps};
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
        _cache_dtype: DType,
    ) -> Result<MetalPagedKvCache> {
        let block_size = block_config.block_size;
        let num_blocks = block_config.num_blocks;
        let pool_shape = [num_blocks * block_size, num_kv_heads, head_dim];

        let mut k_pools = Vec::with_capacity(num_layers);
        let mut v_pools = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            k_pools.push(MetalTensor::zeros(device, &pool_shape, DType::F32));
            v_pools.push(MetalTensor::zeros(device, &pool_shape, DType::F32));
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
        let k_data = k.as_f32_slice();
        let v_data = v.as_f32_slice();
        let head_stride = cache.num_kv_heads * cache.head_dim;
        let seq_len = k.shape()[0];

        let k_pool = cache.k_pools[layer_idx].as_f32_slice_mut();
        let v_pool = cache.v_pools[layer_idx].as_f32_slice_mut();

        for t in 0..seq_len {
            let pos = start_pos + t;
            let block_idx = pos / cache.block_size;
            let block_offset = pos % cache.block_size;
            let physical_block = block_table.blocks()[block_idx];
            let dst_offset = (physical_block * cache.block_size + block_offset) * head_stride;
            let src_offset = t * head_stride;

            k_pool[dst_offset..dst_offset + head_stride]
                .copy_from_slice(&k_data[src_offset..src_offset + head_stride]);
            v_pool[dst_offset..dst_offset + head_stride]
                .copy_from_slice(&v_data[src_offset..src_offset + head_stride]);
        }

        Ok(())
    }

    fn get_pools(cache: &MetalPagedKvCache, layer_idx: usize) -> (&MetalTensor, &MetalTensor) {
        (&cache.k_pools[layer_idx], &cache.v_pools[layer_idx])
    }

    fn block_size(cache: &MetalPagedKvCache) -> usize {
        cache.block_size
    }

    #[allow(clippy::too_many_arguments)]
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
        let k_data = k.as_f32_slice();
        let v_data = v.as_f32_slice();
        let bt_data = block_tables.as_i32_slice();
        let pos_data = positions.as_i32_slice();
        let head_stride = cache.num_kv_heads * cache.head_dim;

        let k_pool = cache.k_pools[layer_idx].as_f32_slice_mut();
        let v_pool = cache.v_pools[layer_idx].as_f32_slice_mut();

        for b in 0..batch_size {
            #[allow(clippy::cast_sign_loss)]
            let pos = pos_data[b] as usize;
            let block_idx = pos / cache.block_size;
            let block_offset = pos % cache.block_size;
            #[allow(clippy::cast_sign_loss)]
            let physical_block = bt_data[b * max_blocks_per_seq + block_idx] as usize;
            let dst_offset = (physical_block * cache.block_size + block_offset) * head_stride;
            let src_offset = b * head_stride;

            k_pool[dst_offset..dst_offset + head_stride]
                .copy_from_slice(&k_data[src_offset..src_offset + head_stride]);
            v_pool[dst_offset..dst_offset + head_stride]
                .copy_from_slice(&v_data[src_offset..src_offset + head_stride]);
        }

        Ok(())
    }
}

// ---- Paged Attention (Decode) ----

#[allow(clippy::needless_range_loop)]
impl PagedAttentionOps for MetalBackend {
    #[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
    fn paged_attention_decode(
        q: &MetalTensor,
        k_pool: &MetalTensor,
        v_pool: &MetalTensor,
        block_tables: &MetalTensor,
        seq_lens: &MetalTensor,
        block_size: usize,
        max_blocks_per_seq: usize,
        _max_seq_len: usize,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<MetalTensor> {
        let q_shape = q.shape();
        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let head_dim = q_shape[2];
        let k_pool_shape = k_pool.shape();
        let num_kv_heads = k_pool_shape[1];
        let gqa_ratio = num_heads / num_kv_heads;

        let q_data = q.as_f32_slice();
        let k_data = k_pool.as_f32_slice();
        let v_data = v_pool.as_f32_slice();
        let bt_data = block_tables.as_i32_slice();
        let sl_data = seq_lens.as_i32_slice();

        let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());
        let kv_stride = num_kv_heads * head_dim;

        let mut output = vec![0.0f32; batch_size * num_heads * head_dim];

        // Max seq_len for scratch buffer
        let max_sl: usize = sl_data
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .try_into()
            .expect("seq_len must be non-negative");
        let mut scores = vec![0.0f32; max_sl];

        for bh_idx in 0..(batch_size * num_heads) {
            let out_head = &mut output[bh_idx * head_dim..(bh_idx + 1) * head_dim];
            let b = bh_idx / num_heads;
            let h = bh_idx % num_heads;
            let kv_h = h / gqa_ratio;

            #[allow(clippy::cast_sign_loss)]
            let seq_len = sl_data[b] as usize;
            if seq_len == 0 {
                continue;
            }

            let q_off = (b * num_heads + h) * head_dim;
            let q_vec = &q_data[q_off..q_off + head_dim];
            let bt_row = &bt_data[b * max_blocks_per_seq..];

            let scores = &mut scores[..seq_len];

            // Q × K dot products
            for pos in 0..seq_len {
                let blk_idx = pos / block_size;
                let blk_off = pos % block_size;
                #[allow(clippy::cast_sign_loss)]
                let phys_block = bt_row[blk_idx] as usize;
                let k_off = (phys_block * block_size + blk_off) * kv_stride + kv_h * head_dim;

                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_vec[d] * k_data[k_off + d];
                }
                dot *= scale;
                if let Some(cap) = softcap {
                    dot = cap * (dot / cap).tanh();
                }
                scores[pos] = dot;
            }

            // Sliding window mask
            if let Some(window) = sliding_window {
                let query_pos = seq_len - 1;
                if query_pos >= window {
                    let cutoff = query_pos - window + 1;
                    for s in &mut scores[..cutoff] {
                        *s = f32::NEG_INFINITY;
                    }
                }
            }

            // Softmax
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_s).exp();
                sum += *s;
            }
            if sum > 0.0 {
                let inv_sum = 1.0 / sum;
                for s in scores.iter_mut() {
                    *s *= inv_sum;
                }
            }

            // Weighted V accumulation
            for pos in 0..seq_len {
                let w = scores[pos];
                if w > 0.0 {
                    let blk_idx = pos / block_size;
                    let blk_off = pos % block_size;
                    #[allow(clippy::cast_sign_loss)]
                    let phys_block = bt_row[blk_idx] as usize;
                    let v_off = (phys_block * block_size + blk_off) * kv_stride + kv_h * head_dim;
                    for d in 0..head_dim {
                        out_head[d] += w * v_data[v_off + d];
                    }
                }
            }
        }

        Ok(MetalTensor::from_f32(
            q.context(),
            &[batch_size, num_heads, head_dim],
            &output,
        ))
    }

    fn gather_paged_kv(
        paged_kv: &MetalPagedKvCache,
        layer_idx: usize,
        block_table: &BlockTable,
    ) -> Result<(MetalTensor, MetalTensor)> {
        let seq_len = block_table.seq_len();
        let head_stride = paged_kv.num_kv_heads * paged_kv.head_dim;
        let mut k_out = Vec::with_capacity(seq_len * head_stride);
        let mut v_out = Vec::with_capacity(seq_len * head_stride);
        let k_pool = paged_kv.k_pools[layer_idx].as_f32_slice();
        let v_pool = paged_kv.v_pools[layer_idx].as_f32_slice();

        for pos in 0..seq_len {
            let block_idx = pos / paged_kv.block_size;
            let block_offset = pos % paged_kv.block_size;
            let phys_block = block_table.blocks()[block_idx];
            let off = (phys_block * paged_kv.block_size + block_offset) * head_stride;

            k_out.extend_from_slice(&k_pool[off..off + head_stride]);
            v_out.extend_from_slice(&v_pool[off..off + head_stride]);
        }

        let ctx = paged_kv.k_pools[layer_idx].context();
        let shape = [seq_len, paged_kv.num_kv_heads, paged_kv.head_dim];
        Ok((
            MetalTensor::from_f32(ctx, &shape, &k_out),
            MetalTensor::from_f32(ctx, &shape, &v_out),
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
