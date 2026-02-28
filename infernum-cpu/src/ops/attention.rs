//! AttentionOps, KvCacheOps, PagedKvCacheOps, PagedAttentionOps implementations.

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
pub struct CpuPagedKvCache {
    /// Per-layer K pool: (num_blocks, block_size, num_kv_heads, head_dim)
    k_pools: Vec<Vec<f32>>,
    /// Per-layer V pool: same shape
    v_pools: Vec<Vec<f32>>,
    /// K pool tensors (recreated on access for trait compliance)
    k_tensors: Vec<CpuTensor>,
    /// V pool tensors
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
        let pool_size = num_blocks * block_size * num_kv_heads * head_dim;

        let mut k_pools = Vec::with_capacity(num_layers);
        let mut v_pools = Vec::with_capacity(num_layers);
        let mut k_tensors = Vec::with_capacity(num_layers);
        let mut v_tensors = Vec::with_capacity(num_layers);
        let pool_shape = [num_blocks * block_size, num_kv_heads, head_dim];

        for _ in 0..num_layers {
            let k = vec![0.0f32; pool_size];
            let v = vec![0.0f32; pool_size];
            k_tensors.push(CpuTensor::from_f32(&pool_shape, &k));
            v_tensors.push(CpuTensor::from_f32(&pool_shape, &v));
            k_pools.push(k);
            v_pools.push(v);
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

        for t in 0..seq_len {
            let pos = start_pos + t;
            let block_idx = pos / cache.block_size;
            let block_offset = pos % cache.block_size;
            let physical_block = block_table.blocks()[block_idx];
            let dst_offset = (physical_block * cache.block_size + block_offset) * head_stride;
            let src_offset = t * head_stride;

            cache.k_pools[layer_idx][dst_offset..dst_offset + head_stride]
                .copy_from_slice(&k_data[src_offset..src_offset + head_stride]);
            cache.v_pools[layer_idx][dst_offset..dst_offset + head_stride]
                .copy_from_slice(&v_data[src_offset..src_offset + head_stride]);
        }

        // Rebuild tensors
        let pool_shape = [
            cache.num_blocks * cache.block_size,
            cache.num_kv_heads,
            cache.head_dim,
        ];
        cache.k_tensors[layer_idx] = CpuTensor::from_f32(&pool_shape, &cache.k_pools[layer_idx]);
        cache.v_tensors[layer_idx] = CpuTensor::from_f32(&pool_shape, &cache.v_pools[layer_idx]);

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

        for b in 0..batch_size {
            #[allow(clippy::cast_sign_loss)]
            let pos = pos_data[b] as usize;
            let block_idx = pos / cache.block_size;
            let block_offset = pos % cache.block_size;
            #[allow(clippy::cast_sign_loss)]
            let physical_block = bt_data[b * max_blocks_per_seq + block_idx] as usize;
            let dst_offset = (physical_block * cache.block_size + block_offset) * head_stride;
            let src_offset = b * head_stride;

            cache.k_pools[layer_idx][dst_offset..dst_offset + head_stride]
                .copy_from_slice(&k_data[src_offset..src_offset + head_stride]);
            cache.v_pools[layer_idx][dst_offset..dst_offset + head_stride]
                .copy_from_slice(&v_data[src_offset..src_offset + head_stride]);
        }

        let pool_shape = [
            cache.num_blocks * cache.block_size,
            cache.num_kv_heads,
            cache.head_dim,
        ];
        cache.k_tensors[layer_idx] = CpuTensor::from_f32(&pool_shape, &cache.k_pools[layer_idx]);
        cache.v_tensors[layer_idx] = CpuTensor::from_f32(&pool_shape, &cache.v_pools[layer_idx]);

        Ok(())
    }
}

// ---- Attention ----

/// Causal attention: Q @ K^T * scale, mask, softmax, @ V.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
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
    let mut output = vec![0.0f32; seq_len * num_heads * head_dim];

    for s in 0..seq_len {
        for h in 0..num_heads {
            let kv_h = h / gqa_ratio;
            let q_offset = (s * num_heads + h) * head_dim;
            let q_vec = &q[q_offset..q_offset + head_dim];

            // Compute attention scores
            let mut scores = Vec::with_capacity(kv_len);
            for kv_pos in 0..kv_len {
                let k_offset = (kv_pos * num_kv_heads + kv_h) * head_dim;
                let k_vec = &k[k_offset..k_offset + head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_vec[d] * k_vec[d];
                }
                dot *= scale;

                // Apply soft-capping
                if let Some(cap) = softcap {
                    dot = cap * (dot / cap).tanh();
                }

                scores.push(dot);
            }

            // Apply causal mask and sliding window
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
            let o_offset = (s * num_heads + h) * head_dim;
            for kv_pos in 0..kv_len {
                if scores[kv_pos] > 0.0 {
                    let v_offset = (kv_pos * num_kv_heads + kv_h) * head_dim;
                    for d in 0..head_dim {
                        output[o_offset + d] += scores[kv_pos] * v[v_offset + d];
                    }
                }
            }
        }
    }

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

        Ok(CpuTensor::from_f32(
            &[seq_len, num_heads, head_dim],
            &output,
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

        Ok(CpuTensor::from_f32(&[1, num_heads, head_dim], &output))
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
            CpuTensor::from_f32(&[seq_len, num_heads, head_dim], &output),
            CpuTensor::from_f32(&[seq_len, num_heads], &lse),
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

        Ok(CpuTensor::from_f32(shape, &output))
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

        for b in 0..batch_size {
            #[allow(clippy::cast_sign_loss)]
            let seq_len = sl_data[b] as usize;

            for h in 0..num_heads {
                let kv_h = h / gqa_ratio;
                let q_off = (b * num_heads + h) * head_dim;
                let q_vec = &q_data[q_off..q_off + head_dim];

                // Gather scores from paged blocks
                let mut scores = Vec::with_capacity(seq_len);
                for pos in 0..seq_len {
                    let block_idx = pos / block_size;
                    let block_offset = pos % block_size;
                    #[allow(clippy::cast_sign_loss)]
                    let phys_block = bt_data[b * max_blocks_per_seq + block_idx] as usize;
                    let k_off =
                        (phys_block * block_size + block_offset) * kv_stride + kv_h * head_dim;

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

                // Apply sliding window
                let query_pos = seq_len - 1;
                if let Some(window) = sliding_window {
                    for (pos, score) in scores.iter_mut().enumerate() {
                        if query_pos >= window && pos < query_pos - window + 1 {
                            *score = f32::NEG_INFINITY;
                        }
                    }
                }

                // Softmax
                let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                if sum > 0.0 {
                    for s in &mut scores {
                        *s /= sum;
                    }
                }

                // Weighted V sum
                let o_off = (b * num_heads + h) * head_dim;
                for pos in 0..seq_len {
                    if scores[pos] > 0.0 {
                        let block_idx = pos / block_size;
                        let block_offset = pos % block_size;
                        #[allow(clippy::cast_sign_loss)]
                        let phys_block = bt_data[b * max_blocks_per_seq + block_idx] as usize;
                        let v_off =
                            (phys_block * block_size + block_offset) * kv_stride + kv_h * head_dim;
                        for d in 0..head_dim {
                            output[o_off + d] += scores[pos] * v_data[v_off + d];
                        }
                    }
                }
            }
        }

        Ok(CpuTensor::from_f32(
            &[batch_size, num_heads, head_dim],
            &output,
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

        for pos in 0..seq_len {
            let block_idx = pos / paged_kv.block_size;
            let block_offset = pos % paged_kv.block_size;
            let phys_block = block_table.blocks()[block_idx];
            let off = (phys_block * paged_kv.block_size + block_offset) * head_stride;

            k_out.extend_from_slice(&paged_kv.k_pools[layer_idx][off..off + head_stride]);
            v_out.extend_from_slice(&paged_kv.v_pools[layer_idx][off..off + head_stride]);
        }

        let shape = [seq_len, paged_kv.num_kv_heads, paged_kv.head_dim];
        Ok((
            CpuTensor::from_f32(&shape, &k_out),
            CpuTensor::from_f32(&shape, &v_out),
        ))
    }
}
