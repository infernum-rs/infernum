//! AttentionOps, PagedAttentionOps, PagedKvCacheOps, KvCacheOps implementations.
//!
//! Phase 1: CPU-side naive implementations.

use infernum::backend::{AttentionOps, KvCacheOps, PagedAttentionOps, PagedKvCacheOps};
use infernum::block_allocator::{BlockConfig, BlockTable};
use infernum::tensor::Tensor;
use infernum::DType;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::{MetalBackend, MetalContext, MetalKvCache, MetalPagedKvCache};

// ---- Fused Attention ----

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
        let (out, _) = attention_impl(q, k, v, offset, scale, softcap, sliding_window, false)?;
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
        let kv_len = k.shape()[0];
        let offset = kv_len.saturating_sub(1);
        let (out, _) = attention_impl(q, k, v, offset, scale, softcap, sliding_window, false)?;
        Ok(out)
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
        attention_impl(q, k, v, offset, scale, softcap, sliding_window, true)
    }

    fn combine_attention_with_lse(
        out1: &MetalTensor,
        lse1: &MetalTensor,
        out2: &MetalTensor,
        lse2: &MetalTensor,
    ) -> Result<MetalTensor> {
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

        let device = metal::Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
        Ok(MetalTensor::from_f32(&device, shape, &out))
    }
}

/// Generic attention implementation (prefill / decode / with LSE).
#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::many_single_char_names
)]
fn attention_impl(
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    offset: usize,
    scale: Option<f32>,
    softcap: Option<f32>,
    sliding_window: Option<usize>,
    compute_lse: bool,
) -> Result<(MetalTensor, MetalTensor)> {
    let q_shape = q.shape();
    let seq_len = q_shape[0];
    let n_heads = q_shape[1];
    let head_dim = q_shape[2];
    let kv_len = k.shape()[0];

    let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

    let q_data = q.as_f32_slice();
    let k_data = k.as_f32_slice();
    let v_data = v.as_f32_slice();
    let kv_heads = k.shape()[1];
    let kv_repeat = n_heads / kv_heads;

    let mut out = vec![0.0f32; seq_len * n_heads * head_dim];
    let mut lse_data = vec![0.0f32; seq_len * n_heads];

    for s in 0..seq_len {
        let q_pos = s + offset;
        for h in 0..n_heads {
            let kv_h = h / kv_repeat;
            let q_base = (s * n_heads + h) * head_dim;

            // Compute attention scores
            let mut scores = vec![f32::NEG_INFINITY; kv_len];
            for kv in 0..kv_len {
                // Causal: only attend to positions <= q_pos
                if kv > q_pos {
                    continue;
                }
                // Sliding window
                if let Some(w) = sliding_window {
                    if q_pos.saturating_sub(w) > kv {
                        continue;
                    }
                }
                let k_base = (kv * kv_heads + kv_h) * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_data[q_base + d] * k_data[k_base + d];
                }
                let mut score = dot * scale;
                if let Some(cap) = softcap {
                    score = cap * (score / cap).tanh();
                }
                scores[kv] = score;
            }

            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            for e in &mut exp_scores {
                *e /= sum_exp;
            }

            // LSE
            if compute_lse {
                lse_data[s * n_heads + h] = max_score + sum_exp.ln();
            }

            // Weighted sum of V
            let out_base = (s * n_heads + h) * head_dim;
            for kv in 0..kv_len {
                let w = exp_scores[kv];
                if w == 0.0 {
                    continue;
                }
                let v_base = (kv * kv_heads + kv_h) * head_dim;
                for d in 0..head_dim {
                    out[out_base + d] += w * v_data[v_base + d];
                }
            }
        }
    }

    let device = metal::Device::system_default()
        .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
    let out_tensor = MetalTensor::from_f32(&device, q_shape, &out);
    let lse_tensor = MetalTensor::from_f32(&device, &[seq_len, n_heads], &lse_data);
    Ok((out_tensor, lse_tensor))
}

// ---- Paged KV Cache ----
//
// Paged KV cache is complex and will be properly implemented when
// Metal attention kernels are added. For now, all methods use todo!().

impl PagedKvCacheOps for MetalBackend {
    fn allocate_paged_kv_cache(
        _device: &MetalContext,
        _num_layers: usize,
        _block_config: &BlockConfig,
        _num_kv_heads: usize,
        _head_dim: usize,
        _cache_dtype: DType,
    ) -> Result<MetalPagedKvCache> {
        todo!("Metal paged KV cache allocation — to be implemented with attention kernels")
    }

    fn append_paged(
        _cache: &mut MetalPagedKvCache,
        _layer_idx: usize,
        _block_table: &BlockTable,
        _k: &MetalTensor,
        _v: &MetalTensor,
        _start_pos: usize,
    ) -> Result<()> {
        todo!("Metal paged KV cache append")
    }

    fn get_pools(_cache: &MetalPagedKvCache, _layer_idx: usize) -> (&MetalTensor, &MetalTensor) {
        todo!("Metal paged KV cache get_pools")
    }

    fn block_size(_cache: &MetalPagedKvCache) -> usize {
        todo!("Metal paged KV cache block_size")
    }

    fn append_paged_batched(
        _cache: &mut MetalPagedKvCache,
        _layer_idx: usize,
        _k: &MetalTensor,
        _v: &MetalTensor,
        _block_tables: &MetalTensor,
        _positions: &MetalTensor,
        _batch_size: usize,
        _max_blocks_per_seq: usize,
    ) -> Result<()> {
        todo!("Metal paged KV cache batched append")
    }
}

impl PagedAttentionOps for MetalBackend {
    #[allow(clippy::too_many_arguments)]
    fn paged_attention_decode(
        _q: &MetalTensor,
        _k_pool: &MetalTensor,
        _v_pool: &MetalTensor,
        _block_tables: &MetalTensor,
        _seq_lens: &MetalTensor,
        _block_size: usize,
        _max_blocks_per_seq: usize,
        _max_seq_len: usize,
        _scale: Option<f32>,
        _softcap: Option<f32>,
        _sliding_window: Option<usize>,
    ) -> Result<MetalTensor> {
        todo!("Metal paged attention decode")
    }

    fn gather_paged_kv(
        _paged_kv: &MetalPagedKvCache,
        _layer_idx: usize,
        _block_table: &BlockTable,
    ) -> Result<(MetalTensor, MetalTensor)> {
        todo!("Metal gather paged KV")
    }
}

// ---- Contiguous KV Cache (DeepSeek MLA) ----

impl KvCacheOps for MetalBackend {
    fn append_kv(
        _cache: &mut MetalKvCache,
        _layer_idx: usize,
        _k: &MetalTensor,
        _v: &MetalTensor,
    ) -> Result<()> {
        todo!("Metal contiguous KV cache append")
    }

    fn get_kv(_cache: &MetalKvCache, _layer_idx: usize) -> (MetalTensor, MetalTensor) {
        todo!("Metal contiguous KV cache get_kv")
    }

    fn get_kv_up_to(
        _cache: &MetalKvCache,
        _layer_idx: usize,
        _len: usize,
    ) -> (MetalTensor, MetalTensor) {
        todo!("Metal contiguous KV cache get_kv_up_to")
    }
}
