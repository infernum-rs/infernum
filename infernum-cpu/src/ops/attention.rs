//! AttentionOps, KvCacheOps, PagedKvCacheOps, PagedAttentionOps implementations.

use std::sync::Arc;

use infernum::backend::{AttentionOps, KvCacheOps, PagedAttentionOps, PagedKvCacheOps};
use infernum::block_allocator::{BlockConfig, BlockTable};
use infernum::dtype::DType;
use infernum::tensor::Tensor;
use infernum::Result;

use crate::tensor::CpuTensor;
use crate::CpuBackend;

// ---- Contiguous KV Cache ----

/// CPU contiguous KV cache: Vec of (K, V) per layer, each grows as tokens are appended.
pub struct CpuKvCache {
    layers: Vec<CpuLayerKv>,
    num_kv_heads: usize,
    head_dim: usize,
}

struct CpuLayerKv {
    k: Vec<f32>, // (current_len, num_kv_heads, head_dim) flattened
    v: Vec<f32>,
    len: usize,
}

impl CpuKvCache {
    #[must_use]
    pub fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| CpuLayerKv {
                k: Vec::new(),
                v: Vec::new(),
                len: 0,
            })
            .collect();
        Self {
            layers,
            num_kv_heads,
            head_dim,
        }
    }
}

impl KvCacheOps for CpuBackend {
    fn append_kv(
        cache: &mut CpuKvCache,
        layer_idx: usize,
        k: &CpuTensor,
        v: &CpuTensor,
    ) -> Result<()> {
        let layer = &mut cache.layers[layer_idx];
        layer.k.extend_from_slice(k.as_f32_slice());
        layer.v.extend_from_slice(v.as_f32_slice());
        let new_tokens = k.shape()[0];
        layer.len += new_tokens;
        Ok(())
    }

    fn get_kv(cache: &CpuKvCache, layer_idx: usize) -> (CpuTensor, CpuTensor) {
        let layer = &cache.layers[layer_idx];
        let shape = [layer.len, cache.num_kv_heads, cache.head_dim];
        (
            CpuTensor::from_f32(&shape, &layer.k),
            CpuTensor::from_f32(&shape, &layer.v),
        )
    }

    fn get_kv_up_to(cache: &CpuKvCache, layer_idx: usize, len: usize) -> (CpuTensor, CpuTensor) {
        let layer = &cache.layers[layer_idx];
        let stride = cache.num_kv_heads * cache.head_dim;
        let n = len * stride;
        let shape = [len, cache.num_kv_heads, cache.head_dim];
        (
            CpuTensor::from_f32(&shape, &layer.k[..n]),
            CpuTensor::from_f32(&shape, &layer.v[..n]),
        )
    }
}

// ---- Paged KV Cache ----

/// CPU paged KV cache using block-allocated storage.
///
/// Pools are stored as `Arc<Vec<u8>>` so that `CpuTensor` views can share the
/// same backing allocation without copying. On each token append the pool is
/// mutated in-place via `Arc::get_mut` (safe because we drop the old tensor
/// first), then a new tensor is created by cloning the `Arc` (pointer bump,
/// no data copy).
pub struct CpuPagedKvCache {
    /// Per-layer K pool: (num_blocks, block_size, num_kv_heads, head_dim) as bytes
    k_pools: Vec<Arc<Vec<u8>>>,
    /// Per-layer V pool: same shape, as bytes
    v_pools: Vec<Arc<Vec<u8>>>,
    /// K pool tensors (share backing storage with k_pools)
    k_tensors: Vec<CpuTensor>,
    /// V pool tensors (share backing storage with v_pools)
    v_tensors: Vec<CpuTensor>,
    block_size: usize,
    num_blocks: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl PagedKvCacheOps for CpuBackend {
    fn allocate_paged_kv_cache(
        _device: &(),
        num_layers: usize,
        block_config: &BlockConfig,
        num_kv_heads: usize,
        head_dim: usize,
        _cache_dtype: DType,
    ) -> Result<CpuPagedKvCache> {
        let block_size = block_config.block_size;
        let num_blocks = block_config.num_blocks;
        let pool_byte_size = num_blocks * block_size * num_kv_heads * head_dim * 4;

        let mut k_pools = Vec::with_capacity(num_layers);
        let mut v_pools = Vec::with_capacity(num_layers);
        let mut k_tensors = Vec::with_capacity(num_layers);
        let mut v_tensors = Vec::with_capacity(num_layers);
        let pool_shape = [num_blocks * block_size, num_kv_heads, head_dim];

        for _ in 0..num_layers {
            let k_arc = Arc::new(vec![0u8; pool_byte_size]);
            let v_arc = Arc::new(vec![0u8; pool_byte_size]);
            k_tensors.push(CpuTensor::from_arc(
                &pool_shape,
                DType::F32,
                Arc::clone(&k_arc),
            ));
            v_tensors.push(CpuTensor::from_arc(
                &pool_shape,
                DType::F32,
                Arc::clone(&v_arc),
            ));
            k_pools.push(k_arc);
            v_pools.push(v_arc);
        }

        Ok(CpuPagedKvCache {
            k_pools,
            v_pools,
            k_tensors,
            v_tensors,
            block_size,
            num_blocks,
            num_kv_heads,
            head_dim,
        })
    }

    fn append_paged(
        cache: &mut CpuPagedKvCache,
        layer_idx: usize,
        block_table: &BlockTable,
        k: &CpuTensor,
        v: &CpuTensor,
        start_pos: usize,
    ) -> Result<()> {
        let k_data = k.as_f32_slice();
        let v_data = v.as_f32_slice();
        let head_stride = cache.num_kv_heads * cache.head_dim;
        let seq_len = k.shape()[0];
        let pool_shape = [
            cache.num_blocks * cache.block_size,
            cache.num_kv_heads,
            cache.head_dim,
        ];

        // Drop old tensors so Arc refcount goes to 1, enabling get_mut.
        cache.k_tensors[layer_idx] = CpuTensor::zeros_f32(&[1]);
        cache.v_tensors[layer_idx] = CpuTensor::zeros_f32(&[1]);

        // Mutate pool bytes in-place.
        let k_pool: &mut Vec<u8> = Arc::get_mut(&mut cache.k_pools[layer_idx])
            .expect("KV pool Arc should have refcount 1");
        let v_pool: &mut Vec<u8> = Arc::get_mut(&mut cache.v_pools[layer_idx])
            .expect("KV pool Arc should have refcount 1");
        let k_pool_f32: &mut [f32] = bytemuck::cast_slice_mut(k_pool);
        let v_pool_f32: &mut [f32] = bytemuck::cast_slice_mut(v_pool);

        for t in 0..seq_len {
            let pos = start_pos + t;
            let block_idx = pos / cache.block_size;
            let block_offset = pos % cache.block_size;
            let physical_block = block_table.blocks()[block_idx];
            let dst_offset = (physical_block * cache.block_size + block_offset) * head_stride;
            let src_offset = t * head_stride;

            k_pool_f32[dst_offset..dst_offset + head_stride]
                .copy_from_slice(&k_data[src_offset..src_offset + head_stride]);
            v_pool_f32[dst_offset..dst_offset + head_stride]
                .copy_from_slice(&v_data[src_offset..src_offset + head_stride]);
        }

        // Rebuild tensors sharing the pool Arc (zero-copy).
        cache.k_tensors[layer_idx] = CpuTensor::from_arc(
            &pool_shape,
            DType::F32,
            Arc::clone(&cache.k_pools[layer_idx]),
        );
        cache.v_tensors[layer_idx] = CpuTensor::from_arc(
            &pool_shape,
            DType::F32,
            Arc::clone(&cache.v_pools[layer_idx]),
        );

        Ok(())
    }

    fn get_pools(cache: &CpuPagedKvCache, layer_idx: usize) -> (&CpuTensor, &CpuTensor) {
        (&cache.k_tensors[layer_idx], &cache.v_tensors[layer_idx])
    }

    fn block_size(cache: &CpuPagedKvCache) -> usize {
        cache.block_size
    }

    fn append_paged_batched(
        cache: &mut CpuPagedKvCache,
        layer_idx: usize,
        k: &CpuTensor,
        v: &CpuTensor,
        block_tables: &CpuTensor,
        positions: &CpuTensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        let k_data = k.as_f32_slice();
        let v_data = v.as_f32_slice();
        let bt_data = block_tables.as_i32_slice();
        let pos_data = positions.as_i32_slice();
        let head_stride = cache.num_kv_heads * cache.head_dim;
        let pool_shape = [
            cache.num_blocks * cache.block_size,
            cache.num_kv_heads,
            cache.head_dim,
        ];

        // Drop old tensors so Arc refcount goes to 1, enabling get_mut.
        cache.k_tensors[layer_idx] = CpuTensor::zeros_f32(&[1]);
        cache.v_tensors[layer_idx] = CpuTensor::zeros_f32(&[1]);

        let k_pool: &mut Vec<u8> = Arc::get_mut(&mut cache.k_pools[layer_idx])
            .expect("KV pool Arc should have refcount 1");
        let v_pool: &mut Vec<u8> = Arc::get_mut(&mut cache.v_pools[layer_idx])
            .expect("KV pool Arc should have refcount 1");
        let k_pool_f32: &mut [f32] = bytemuck::cast_slice_mut(k_pool);
        let v_pool_f32: &mut [f32] = bytemuck::cast_slice_mut(v_pool);

        for b in 0..batch_size {
            #[allow(clippy::cast_sign_loss)]
            let pos = pos_data[b] as usize;
            let block_idx = pos / cache.block_size;
            let block_offset = pos % cache.block_size;
            #[allow(clippy::cast_sign_loss)]
            let physical_block = bt_data[b * max_blocks_per_seq + block_idx] as usize;
            let dst_offset = (physical_block * cache.block_size + block_offset) * head_stride;
            let src_offset = b * head_stride;

            k_pool_f32[dst_offset..dst_offset + head_stride]
                .copy_from_slice(&k_data[src_offset..src_offset + head_stride]);
            v_pool_f32[dst_offset..dst_offset + head_stride]
                .copy_from_slice(&v_data[src_offset..src_offset + head_stride]);
        }

        // Rebuild tensors sharing the pool Arc (zero-copy).
        cache.k_tensors[layer_idx] = CpuTensor::from_arc(
            &pool_shape,
            DType::F32,
            Arc::clone(&cache.k_pools[layer_idx]),
        );
        cache.v_tensors[layer_idx] = CpuTensor::from_arc(
            &pool_shape,
            DType::F32,
            Arc::clone(&cache.v_pools[layer_idx]),
        );

        Ok(())
    }
}

// ---- Attention ----

/// Process a single (seq_pos, head) attention unit: Q·K^T, mask, softmax, V weighted sum.
///
/// Writes directly into `output` at the correct offset. Each (s, h) pair
/// accesses a disjoint region so this is safe to call from multiple threads.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn attention_head_unit(
    output: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    s: usize,
    h: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    gqa_ratio: usize,
    offset: usize,
    scale: f32,
    softcap: Option<f32>,
    sliding_window: Option<usize>,
) {
    let kv_h = h / gqa_ratio;
    let q_off = (s * num_heads + h) * head_dim;
    let q_vec = &q[q_off..q_off + head_dim];

    // Compute attention scores
    let mut scores = Vec::with_capacity(kv_len);
    for kv_pos in 0..kv_len {
        let k_off = (kv_pos * num_kv_heads + kv_h) * head_dim;
        let k_vec = &k[k_off..k_off + head_dim];
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += q_vec[d] * k_vec[d];
        }
        dot *= scale;

        if let Some(cap) = softcap {
            dot = cap * (dot / cap).tanh();
        }

        scores.push(dot);
    }

    // Causal mask + sliding window
    let query_pos = offset + s;
    for kv_pos in 0..kv_len {
        if kv_pos > query_pos {
            scores[kv_pos] = f32::NEG_INFINITY;
        }
        if let Some(window) = sliding_window {
            if query_pos >= window && kv_pos < query_pos - window + 1 {
                scores[kv_pos] = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for score in &mut scores {
        *score = (*score - max_score).exp();
        sum += *score;
    }
    if sum > 0.0 {
        for score in &mut scores {
            *score /= sum;
        }
    }

    // Weighted sum of V
    let o_off = (s * num_heads + h) * head_dim;
    for kv_pos in 0..kv_len {
        if scores[kv_pos] > 0.0 {
            let v_off = (kv_pos * num_kv_heads + kv_h) * head_dim;
            for d in 0..head_dim {
                output[o_off + d] += scores[kv_pos] * v[v_off + d];
            }
        }
    }
}

/// Causal attention: Q @ K^T * scale, mask, softmax, @ V.
///
/// Parallelized across the `(seq_pos, head)` dimension — each unit writes
/// to a disjoint region of the output, so no synchronization is needed.
#[allow(clippy::too_many_arguments)]
fn causal_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    offset: usize,
    scale: f32,
    softcap: Option<f32>,
    sliding_window: Option<usize>,
) -> Vec<f32> {
    let gqa_ratio = num_heads / num_kv_heads;
    let total_units = seq_len * num_heads;
    let mut output = vec![0.0f32; total_units * head_dim];

    let pool = crate::thread_pool::global_pool();
    let num_threads = pool.num_threads();

    // For very small workloads (decode with few heads), skip dispatch overhead.
    if total_units <= num_threads {
        for s in 0..seq_len {
            for h in 0..num_heads {
                attention_head_unit(
                    &mut output,
                    q,
                    k,
                    v,
                    s,
                    h,
                    kv_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    gqa_ratio,
                    offset,
                    scale,
                    softcap,
                    sliding_window,
                );
            }
        }
        return output;
    }

    let chunk_size = total_units.div_ceil(num_threads);
    let num_tasks = total_units.div_ceil(chunk_size);
    let out_addr = output.as_mut_ptr() as usize;

    pool.dispatch(num_tasks, |task_id, _| {
        let unit_start = task_id * chunk_size;
        let unit_end = (unit_start + chunk_size).min(total_units);

        // Safety: each task writes to disjoint (s, h) output slots.
        let out_slice =
            unsafe { std::slice::from_raw_parts_mut(out_addr as *mut f32, total_units * head_dim) };

        for unit in unit_start..unit_end {
            let s = unit / num_heads;
            let h = unit % num_heads;
            attention_head_unit(
                out_slice,
                q,
                k,
                v,
                s,
                h,
                kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                gqa_ratio,
                offset,
                scale,
                softcap,
                sliding_window,
            );
        }
    });

    output
}

#[allow(clippy::needless_range_loop)]
impl AttentionOps for CpuBackend {
    fn fused_attention_prefill(
        q: &CpuTensor,
        k: &CpuTensor,
        v: &CpuTensor,
        offset: usize,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<CpuTensor> {
        let q_shape = q.shape();
        let k_shape = k.shape();
        let seq_len = q_shape[0];
        let num_heads = q_shape[1];
        let head_dim = q_shape[2];
        let kv_len = k_shape[0];
        let num_kv_heads = k_shape[1];

        #[allow(clippy::cast_precision_loss)]
        let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

        let output = causal_attention(
            q.as_f32_slice(),
            k.as_f32_slice(),
            v.as_f32_slice(),
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            offset,
            scale,
            softcap,
            sliding_window,
        );

        Ok(CpuTensor::from_f32_vec(
            &[seq_len, num_heads, head_dim],
            output,
        ))
    }

    fn fused_attention_decode(
        q: &CpuTensor,
        k: &CpuTensor,
        v: &CpuTensor,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<CpuTensor> {
        let q_shape = q.shape();
        let k_shape = k.shape();
        let num_heads = q_shape[1];
        let head_dim = q_shape[2];
        let kv_len = k_shape[0];
        let num_kv_heads = k_shape[1];

        #[allow(clippy::cast_precision_loss)]
        let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());
        let offset = kv_len - 1;

        let output = causal_attention(
            q.as_f32_slice(),
            k.as_f32_slice(),
            v.as_f32_slice(),
            1,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            offset,
            scale,
            softcap,
            sliding_window,
        );

        Ok(CpuTensor::from_f32_vec(&[1, num_heads, head_dim], output))
    }

    fn fused_attention_prefill_with_lse(
        q: &CpuTensor,
        k: &CpuTensor,
        v: &CpuTensor,
        offset: usize,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<(CpuTensor, CpuTensor)> {
        // Compute attention output and also log-sum-exp per (seq, head)
        let q_shape = q.shape();
        let k_shape = k.shape();
        let seq_len = q_shape[0];
        let num_heads = q_shape[1];
        let head_dim = q_shape[2];
        let kv_len = k_shape[0];
        let num_kv_heads = k_shape[1];
        let gqa_ratio = num_heads / num_kv_heads;

        #[allow(clippy::cast_precision_loss)]
        let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

        let q_data = q.as_f32_slice();
        let k_data = k.as_f32_slice();
        let v_data = v.as_f32_slice();

        let mut output = vec![0.0f32; seq_len * num_heads * head_dim];
        let mut lse = vec![0.0f32; seq_len * num_heads];

        for s in 0..seq_len {
            for h in 0..num_heads {
                let kv_h = h / gqa_ratio;
                let q_off = (s * num_heads + h) * head_dim;
                let q_vec = &q_data[q_off..q_off + head_dim];

                let mut scores = Vec::with_capacity(kv_len);
                for kv_pos in 0..kv_len {
                    let k_off = (kv_pos * num_kv_heads + kv_h) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_vec[d] * k_data[k_off + d];
                    }
                    dot *= scale;
                    if let Some(cap) = softcap {
                        dot = cap * (dot / cap).tanh();
                    }
                    scores.push(dot);
                }

                let query_pos = offset + s;
                for kv_pos in 0..kv_len {
                    if kv_pos > query_pos {
                        scores[kv_pos] = f32::NEG_INFINITY;
                    }
                    if let Some(window) = sliding_window {
                        if query_pos >= window && kv_pos < query_pos - window + 1 {
                            scores[kv_pos] = f32::NEG_INFINITY;
                        }
                    }
                }

                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for score in &mut scores {
                    *score = (*score - max_score).exp();
                    sum_exp += *score;
                }
                lse[s * num_heads + h] = max_score + sum_exp.ln();

                if sum_exp > 0.0 {
                    for score in &mut scores {
                        *score /= sum_exp;
                    }
                }

                let o_off = (s * num_heads + h) * head_dim;
                for kv_pos in 0..kv_len {
                    if scores[kv_pos] > 0.0 {
                        let v_off = (kv_pos * num_kv_heads + kv_h) * head_dim;
                        for d in 0..head_dim {
                            output[o_off + d] += scores[kv_pos] * v_data[v_off + d];
                        }
                    }
                }
            }
        }

        Ok((
            CpuTensor::from_f32_vec(&[seq_len, num_heads, head_dim], output),
            CpuTensor::from_f32_vec(&[seq_len, num_heads], lse),
        ))
    }

    fn combine_attention_with_lse(
        out1: &CpuTensor,
        lse1: &CpuTensor,
        out2: &CpuTensor,
        lse2: &CpuTensor,
    ) -> Result<CpuTensor> {
        let shape = out1.shape();
        let n = shape[0]; // seq or batch
        let heads = shape[1];
        let head_dim = shape[2];

        let o1 = out1.as_f32_slice();
        let l1 = lse1.as_f32_slice();
        let o2 = out2.as_f32_slice();
        let l2 = lse2.as_f32_slice();

        let mut output = vec![0.0f32; n * heads * head_dim];

        for i in 0..(n * heads) {
            let lse_a = l1[i];
            let lse_b = l2[i];
            let max_lse = lse_a.max(lse_b);
            let w_a = (lse_a - max_lse).exp();
            let w_b = (lse_b - max_lse).exp();
            let w_sum = w_a + w_b;

            for d in 0..head_dim {
                let idx = i * head_dim + d;
                output[idx] = (w_a * o1[idx] + w_b * o2[idx]) / w_sum;
            }
        }

        Ok(CpuTensor::from_f32_vec(shape, output))
    }
}

// ---- Paged Attention (Decode) ----

#[allow(clippy::needless_range_loop)]
impl PagedAttentionOps for CpuBackend {
    fn paged_attention_decode(
        q: &CpuTensor,
        k_pool: &CpuTensor,
        v_pool: &CpuTensor,
        block_tables: &CpuTensor,
        seq_lens: &CpuTensor,
        block_size: usize,
        max_blocks_per_seq: usize,
        _max_seq_len: usize,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<CpuTensor> {
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

        #[allow(clippy::cast_precision_loss)]
        let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());
        let kv_stride = num_kv_heads * head_dim;

        let mut output = vec![0.0f32; batch_size * num_heads * head_dim];

        // Find max seq_len across batch for scratch buffer sizing
        let max_sl: usize = sl_data
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .try_into()
            .expect("seq_len must be non-negative");
        let mut scores = vec![0.0f32; max_sl];

        // Process each (batch, head) pair sequentially.
        // At decode time, work per head is tiny (seq_len dot products of dim 64),
        // so rayon's scheduling overhead far exceeds the actual compute.
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

            // Q × K dot products using SIMD
            for pos in 0..seq_len {
                let block_idx = pos / block_size;
                let block_offset = pos % block_size;
                #[allow(clippy::cast_sign_loss)]
                let phys_block = bt_row[block_idx] as usize;
                let k_off = (phys_block * block_size + block_offset) * kv_stride + kv_h * head_dim;

                let mut dot = crate::simd::dot_f32(q_vec, &k_data[k_off..k_off + head_dim]);
                dot *= scale;
                if let Some(cap) = softcap {
                    dot = cap * (dot / cap).tanh();
                }
                scores[pos] = dot;
            }

            // Apply sliding window mask
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

            // Weighted V accumulation using SIMD AXPY
            for pos in 0..seq_len {
                let w = scores[pos];
                if w > 0.0 {
                    let block_idx = pos / block_size;
                    let block_offset = pos % block_size;
                    #[allow(clippy::cast_sign_loss)]
                    let phys_block = bt_row[block_idx] as usize;
                    let v_off =
                        (phys_block * block_size + block_offset) * kv_stride + kv_h * head_dim;
                    crate::simd::vec_axpy(out_head, w, &v_data[v_off..v_off + head_dim]);
                }
            }
        }

        Ok(CpuTensor::from_f32_vec(
            &[batch_size, num_heads, head_dim],
            output,
        ))
    }

    fn gather_paged_kv(
        paged_kv: &CpuPagedKvCache,
        layer_idx: usize,
        block_table: &BlockTable,
    ) -> Result<(CpuTensor, CpuTensor)> {
        let seq_len = block_table.seq_len();
        let head_stride = paged_kv.num_kv_heads * paged_kv.head_dim;
        let mut k_out = Vec::with_capacity(seq_len * head_stride);
        let mut v_out = Vec::with_capacity(seq_len * head_stride);
        let k_pool_f32: &[f32] = bytemuck::cast_slice(&paged_kv.k_pools[layer_idx]);
        let v_pool_f32: &[f32] = bytemuck::cast_slice(&paged_kv.v_pools[layer_idx]);

        for pos in 0..seq_len {
            let block_idx = pos / paged_kv.block_size;
            let block_offset = pos % paged_kv.block_size;
            let phys_block = block_table.blocks()[block_idx];
            let off = (phys_block * paged_kv.block_size + block_offset) * head_stride;

            k_out.extend_from_slice(&k_pool_f32[off..off + head_stride]);
            v_out.extend_from_slice(&v_pool_f32[off..off + head_stride]);
        }

        let shape = [seq_len, paged_kv.num_kv_heads, paged_kv.head_dim];
        Ok((
            CpuTensor::from_f32_vec(&shape, k_out),
            CpuTensor::from_f32_vec(&shape, v_out),
        ))
    }
}
