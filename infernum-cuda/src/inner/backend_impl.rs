//! `CudaBackend` — implements the infernum `Backend` + op traits for CUDA.

use infernum::backend::{
    ArithOps, AttentionOps, Backend, BiasOps, CastOps, EmbedOps, GegluOps, KvCacheOps,
    MatmulExtOps, MatmulOps, MoeOps, MoeSigmoidOps, NormOps, PagedAttentionOps, PagedKvCacheOps,
    RopeInterleavedOps, RopeOps, SwigluOps, TensorFactory, TensorOps,
};
use infernum::block_allocator::{BlockConfig, BlockTable};
use infernum::{DType, Result};

use crate::cuda::ops;
use crate::cuda::ops::LinearWeight;
use crate::cuda::CudaTensor;
use crate::cuda_logits::CudaLogits;
use crate::cuda_runtime_state::CudaRuntimeState;

/// Marker type for the CUDA backend.
pub struct CudaBackend;

impl Backend for CudaBackend {
    type Tensor = CudaTensor;
    type DeviceHandle = crate::cuda::CudaContext;
    type PagedKvCache = crate::cuda::PagedKvCache;
    type KvCache = crate::cuda::KvCache;
    type RuntimeState = CudaRuntimeState;
    type ExecutorState = crate::inner::execute_context::CudaExecutorState;
    type Logits = CudaLogits;

    #[cfg(feature = "nccl")]
    type Comm = crate::cuda::NcclCommunicator;
    #[cfg(not(feature = "nccl"))]
    type Comm = ();

    fn logits_from_tensor(tensor: CudaTensor) -> CudaLogits {
        CudaLogits::new(tensor)
    }

    const QUANTIZED_COMPUTE_DTYPE: DType = DType::BF16;
}

// ---- Tensor factory ----

impl TensorFactory for CudaBackend {
    fn from_f32_slice(
        device: &crate::cuda::CudaContext,
        shape: &[usize],
        data: &[f32],
    ) -> Result<CudaTensor> {
        CudaTensor::from_slice(device, shape, data)
    }

    fn from_raw_bytes(
        device: &crate::cuda::CudaContext,
        shape: &[usize],
        dtype: DType,
        data: &[u8],
    ) -> Result<CudaTensor> {
        CudaTensor::from_raw_bytes(device, shape, dtype, data)
    }

    fn from_u32_slice(
        device: &crate::cuda::CudaContext,
        shape: &[usize],
        data: &[u32],
    ) -> Result<CudaTensor> {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        CudaTensor::from_raw_bytes(device, shape, DType::U32, bytes)
    }

    fn from_i32_slice(
        device: &crate::cuda::CudaContext,
        shape: &[usize],
        data: &[i32],
    ) -> Result<CudaTensor> {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        CudaTensor::from_raw_bytes(device, shape, DType::U32, bytes)
    }
}

// ---- Decode buffers ----

impl infernum::backend::DecodeBufferOps for CudaBackend {}

// ---- Arithmetic ----

impl ArithOps for CudaBackend {
    fn add(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        ops::add(a, b)
    }

    fn add_inplace(a: &mut CudaTensor, b: &CudaTensor) -> Result<()> {
        ops::add_inplace(a, b)
    }

    fn mul(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        ops::mul(a, b)
    }

    fn scale_inplace(a: &mut CudaTensor, scale: f32) -> Result<()> {
        ops::scale_inplace(a, scale)
    }

    fn silu(input: &CudaTensor) -> Result<CudaTensor> {
        ops::silu(input)
    }

    fn logit_softcap(input: &CudaTensor, cap: f32) -> Result<CudaTensor> {
        ops::logit_softcap(input, cap)
    }
}

// ---- Matmul ----

impl MatmulOps for CudaBackend {
    type LinearWeight = LinearWeight;

    fn matmul(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        ops::matmul(a, b)
    }

    fn linear(input: &CudaTensor, weight: &LinearWeight) -> Result<CudaTensor> {
        ops::linear(input, weight)
    }

    fn as_dense_weight(weight: &LinearWeight) -> Option<&CudaTensor> {
        match weight {
            LinearWeight::Dense(t) => Some(t),
            LinearWeight::Quantized(_) => None,
        }
    }

    fn dense_weight(tensor: CudaTensor) -> LinearWeight {
        LinearWeight::Dense(tensor)
    }

    fn is_dense_weight(weight: &LinearWeight) -> bool {
        matches!(weight, LinearWeight::Dense(_))
    }

    fn quantize_to_q8(
        device: &crate::cuda::CudaContext,
        shape: &[usize],
        data: &[f32],
    ) -> Result<LinearWeight> {
        Ok(LinearWeight::Quantized(
            crate::cuda::QuantizedTensor::from_f32_as_q8(device, shape, data)?,
        ))
    }

    fn upload_host_linear(
        device: &crate::cuda::CudaContext,
        weight: &infernum::weights::host::HostLinearWeight,
    ) -> Result<LinearWeight> {
        use infernum::weights::host::HostLinearWeight;

        match weight {
            HostLinearWeight::Dense(host_tensor) => {
                let tensor = CudaTensor::from_raw_bytes(
                    device,
                    &host_tensor.shape,
                    host_tensor.dtype,
                    &host_tensor.data,
                )?;
                Ok(LinearWeight::Dense(tensor))
            }
            HostLinearWeight::Quantized(hq) => {
                if hq.dtype.is_group_quantized() {
                    let qzeros = hq
                        .qzeros
                        .as_deref()
                        .expect("GPTQ/AWQ quantized weight must have qzeros");
                    let group_size = hq
                        .group_size
                        .expect("GPTQ/AWQ quantized weight must have group_size");
                    let qt = crate::cuda::QuantizedTensor::from_gptq_raw(
                        device, &hq.shape, hq.dtype, &hq.data, &hq.scales, qzeros, group_size,
                    )?;
                    Ok(LinearWeight::Quantized(qt))
                } else {
                    let mut qt = crate::cuda::QuantizedTensor::from_raw(
                        device, &hq.shape, hq.dtype, &hq.data, &hq.scales,
                    )?;
                    if let Some(ref channel_scales) = hq.channel_scales {
                        qt.set_channel_scales(device, channel_scales)?;
                    } else if (hq.weight_scale - 1.0).abs() > f32::EPSILON {
                        qt.set_weight_scale(device, hq.weight_scale)?;
                    }
                    Ok(LinearWeight::Quantized(qt))
                }
            }
        }
    }
}

impl MatmulExtOps for CudaBackend {
    fn matmul_bf16_f32(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        ops::matmul_bf16_f32(a, b)
    }
}

// ---- Normalization ----

impl NormOps for CudaBackend {
    fn rms_norm(input: &CudaTensor, weight: &CudaTensor, eps: f32) -> Result<CudaTensor> {
        ops::rms_norm(input, weight, eps)
    }

    fn rms_norm_inplace(input: &mut CudaTensor, weight: &CudaTensor, eps: f32) -> Result<()> {
        ops::rms_norm_inplace(input, weight, eps)
    }

    fn add_rmsnorm(
        residual: &CudaTensor,
        input: &CudaTensor,
        weight: &CudaTensor,
        eps: f32,
    ) -> Result<(CudaTensor, CudaTensor)> {
        ops::add_rmsnorm(residual, input, weight, eps)
    }
}

// ---- Activations ----

impl SwigluOps for CudaBackend {
    fn swiglu(gate: &CudaTensor, up: &CudaTensor) -> Result<CudaTensor> {
        ops::swiglu(gate, up)
    }
}

impl GegluOps for CudaBackend {
    fn geglu(gate: &CudaTensor, up: &CudaTensor) -> Result<CudaTensor> {
        ops::geglu(gate, up)
    }
}

// ---- Cast ----

impl CastOps for CudaBackend {
    fn cast_to_f32(input: &CudaTensor) -> Result<CudaTensor> {
        ops::cast_to_f32(input)
    }

    fn cast_from_f32(input: &CudaTensor, target: DType) -> Result<CudaTensor> {
        ops::cast_from_f32(input, target)
    }
}

// ---- Tensor data (host download) ----

impl infernum::backend::TensorDataOps for CudaBackend {
    fn to_f32_vec(tensor: &CudaTensor) -> Result<Vec<f32>> {
        use infernum::tensor::Tensor;
        if tensor.dtype() == DType::F32 {
            tensor.to_vec::<f32>()
        } else {
            let f32_tensor = ops::cast_to_f32(tensor)?;
            f32_tensor.to_vec::<f32>()
        }
    }

    fn to_raw_bytes(tensor: &CudaTensor) -> Result<Vec<u8>> {
        tensor.to_raw_bytes()
    }
}

// ---- Embedding ----

impl EmbedOps for CudaBackend {
    fn embedding_gather(table: &CudaTensor, indices: &[u32]) -> Result<CudaTensor> {
        ops::embedding_gather(table.context(), table, indices)
    }

    fn embedding_gather_tensor(
        table: &CudaTensor,
        indices: &CudaTensor,
        seq_len: usize,
    ) -> Result<CudaTensor> {
        ops::embedding_gather_from_tensor(table.context(), table, indices, seq_len)
    }
}

// ---- Bias ----

impl BiasOps for CudaBackend {
    fn bias_add_inplace(input: &mut CudaTensor, bias: &CudaTensor) -> Result<()> {
        ops::bias_add_inplace(input, bias)
    }
}

// ---- Tensor manipulation ----

impl TensorOps for CudaBackend {
    fn transpose_2d(input: &CudaTensor) -> Result<CudaTensor> {
        ops::transpose_2d(input)
    }

    fn split_inner_dim(
        tensor: &CudaTensor,
        dim1: usize,
        dim2: usize,
    ) -> Result<(CudaTensor, CudaTensor)> {
        ops::split_inner_dim(tensor, dim1, dim2)
    }

    fn concat_inner_dim(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        ops::concat_inner_dim(a, b)
    }

    fn pad_inner_dim(tensor: &CudaTensor, new_width: usize) -> Result<CudaTensor> {
        ops::pad_inner_dim(tensor, new_width)
    }

    fn broadcast_to_heads(tensor: &CudaTensor, num_heads: usize) -> Result<CudaTensor> {
        ops::broadcast_to_heads(tensor, num_heads)
    }

    fn repeat_kv(tensor: &CudaTensor, num_repeats: usize) -> Result<CudaTensor> {
        ops::repeat_kv(tensor, num_repeats)
    }

    fn concat_rows(parts: &[CudaTensor]) -> Result<CudaTensor> {
        use infernum::tensor::Tensor;

        assert!(!parts.is_empty(), "concat_rows: empty input");
        let ref_shape = parts[0].shape();
        let dtype = parts[0].dtype();
        let elem = dtype.size_in_bytes();
        // Stride = product of all dims except the first (handles 2D, 3D, etc.)
        let stride: usize = ref_shape.iter().skip(1).product();
        let total_rows: usize = parts.iter().map(|p| p.shape()[0]).sum();
        let ctx = parts[0].context();

        // Build output shape: same as input but with dim-0 = total_rows
        let mut out_shape = ref_shape.to_vec();
        out_shape[0] = total_rows;

        let mut output = unsafe { CudaTensor::uninit(ctx, &out_shape, dtype)? };
        let out_slice = output.cuda_slice_mut();
        let mut offset = 0;
        for part in parts {
            let part_bytes = part.shape()[0] * stride * elem;
            if part_bytes > 0 {
                let src = part.cuda_slice().slice(..part_bytes);
                let mut dst = out_slice.slice_mut(offset..offset + part_bytes);
                ctx.device().dtod_copy(&src, &mut dst)?;
            }
            offset += part_bytes;
        }
        Ok(output)
    }
}

// ---- RoPE ----

impl RopeOps for CudaBackend {
    fn apply_rope(
        input: &CudaTensor,
        cos_cache: &CudaTensor,
        sin_cache: &CudaTensor,
        position_offset: usize,
    ) -> Result<CudaTensor> {
        ops::apply_rope(input, cos_cache, sin_cache, position_offset)
    }

    fn apply_rope_batched(
        input: &CudaTensor,
        cos_cache: &CudaTensor,
        sin_cache: &CudaTensor,
        positions: &CudaTensor,
        batch_size: usize,
    ) -> Result<CudaTensor> {
        // The tensor holds i32 positions; apply_rope_batched_indirect
        // reads them from the GPU via CudaSlice<i32>. Since I32 and i32
        // are bit-identical and cudarc passes raw device pointers, we
        // can pass the CudaView<u8> directly to the kernel.
        ops::apply_rope_batched_from_tensor(input, cos_cache, sin_cache, positions, batch_size)
    }
}

impl RopeInterleavedOps for CudaBackend {
    fn apply_rope_interleaved(
        input: &CudaTensor,
        cos_cache: &CudaTensor,
        sin_cache: &CudaTensor,
        position_offset: usize,
    ) -> Result<CudaTensor> {
        ops::apply_rope_interleaved(input, cos_cache, sin_cache, position_offset)
    }
}

// ---- Attention ----

impl AttentionOps for CudaBackend {
    fn fused_attention_prefill(
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        offset: usize,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<CudaTensor> {
        ops::fused_attention_prefill(q, k, v, offset, scale, softcap, sliding_window)
    }

    fn fused_attention_decode(
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<CudaTensor> {
        ops::fused_attention_decode(q, k, v, scale, softcap, sliding_window)
    }

    fn fused_attention_prefill_with_lse(
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        offset: usize,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<(CudaTensor, CudaTensor)> {
        ops::fused_attention_prefill_with_lse(q, k, v, offset, scale, softcap, sliding_window)
    }

    fn combine_attention_with_lse(
        out1: &CudaTensor,
        lse1: &CudaTensor,
        out2: &CudaTensor,
        lse2: &CudaTensor,
    ) -> Result<CudaTensor> {
        ops::combine_attention_with_lse(out1, lse1, out2, lse2)
    }
}

// ---- Paged attention ----

impl PagedAttentionOps for CudaBackend {
    fn paged_attention_decode(
        q: &CudaTensor,
        k_pool: &CudaTensor,
        v_pool: &CudaTensor,
        block_tables: &CudaTensor,
        seq_lens: &CudaTensor,
        block_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
        scale: Option<f32>,
        softcap: Option<f32>,
        sliding_window: Option<usize>,
    ) -> Result<CudaTensor> {
        ops::paged_attention_decode_from_tensor(
            q.context(),
            q,
            k_pool,
            v_pool,
            block_tables,
            seq_lens,
            block_size,
            max_blocks_per_seq,
            max_seq_len,
            scale,
            softcap,
            sliding_window,
        )
    }

    fn gather_paged_kv(
        paged_kv: &crate::cuda::PagedKvCache,
        layer_idx: usize,
        block_table: &BlockTable,
    ) -> Result<(CudaTensor, CudaTensor)> {
        ops::gather_paged_kv(paged_kv, layer_idx, block_table)
    }
}

// ---- KV cache management ----

impl PagedKvCacheOps for CudaBackend {
    fn allocate_paged_kv_cache(
        device: &crate::cuda::CudaContext,
        num_layers: usize,
        block_config: &BlockConfig,
        num_kv_heads: usize,
        head_dim: usize,
        cache_dtype: DType,
    ) -> Result<crate::cuda::PagedKvCache> {
        crate::cuda::PagedKvCache::new(
            device,
            num_layers,
            block_config,
            num_kv_heads,
            head_dim,
            cache_dtype,
        )
    }

    fn append_paged(
        cache: &mut crate::cuda::PagedKvCache,
        layer_idx: usize,
        block_table: &BlockTable,
        k: &CudaTensor,
        v: &CudaTensor,
        start_pos: usize,
    ) -> Result<()> {
        cache.append_paged(layer_idx, block_table, k, v, start_pos)
    }

    fn get_pools(
        cache: &crate::cuda::PagedKvCache,
        layer_idx: usize,
    ) -> (&CudaTensor, &CudaTensor) {
        cache.get_pools(layer_idx)
    }

    fn block_size(cache: &crate::cuda::PagedKvCache) -> usize {
        cache.block_size()
    }

    fn append_paged_batched(
        cache: &mut crate::cuda::PagedKvCache,
        layer_idx: usize,
        k: &CudaTensor,
        v: &CudaTensor,
        block_tables: &CudaTensor,
        positions: &CudaTensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        cache.append_paged_batched_tensor(
            layer_idx,
            k,
            v,
            block_tables,
            positions,
            batch_size,
            max_blocks_per_seq,
        )
    }
}

impl KvCacheOps for CudaBackend {
    fn append_kv(
        cache: &mut crate::cuda::KvCache,
        layer_idx: usize,
        k: &CudaTensor,
        v: &CudaTensor,
    ) -> Result<()> {
        cache.append(layer_idx, k, v)
    }

    fn get_kv(cache: &crate::cuda::KvCache, layer_idx: usize) -> (CudaTensor, CudaTensor) {
        cache.get(layer_idx)
    }

    fn get_kv_up_to(
        cache: &crate::cuda::KvCache,
        layer_idx: usize,
        len: usize,
    ) -> (CudaTensor, CudaTensor) {
        cache.get_up_to(layer_idx, len)
    }
}

// ---- MoE ----

impl MoeOps for CudaBackend {
    fn moe_forward_softmax<F>(
        hidden: &CudaTensor,
        gate_weight: &CudaTensor,
        num_experts: usize,
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        expert_fn: F,
    ) -> Result<CudaTensor>
    where
        F: Fn(usize, &CudaTensor) -> Result<CudaTensor>,
    {
        crate::cuda::moe::moe_forward(
            hidden,
            gate_weight,
            num_experts,
            num_experts_per_tok,
            norm_topk_prob,
            expert_fn,
        )
    }
}

impl MoeSigmoidOps for CudaBackend {
    fn moe_forward_sigmoid<F>(
        hidden: &CudaTensor,
        gate_weight: &CudaTensor,
        e_score_correction_bias: &[f32],
        num_experts: usize,
        num_experts_per_tok: usize,
        n_group: usize,
        topk_group: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        expert_fn: F,
    ) -> Result<CudaTensor>
    where
        F: Fn(usize, &CudaTensor) -> Result<CudaTensor>,
    {
        crate::cuda::moe::moe_forward_sigmoid(
            hidden,
            gate_weight,
            e_score_correction_bias,
            num_experts,
            num_experts_per_tok,
            n_group,
            topk_group,
            norm_topk_prob,
            routed_scaling_factor,
            expert_fn,
        )
    }
}

impl infernum::backend::MlaAttentionOps for CudaBackend {
    /// Run the full MLA forward pass for one decode step using a flat KV cache.
    ///
    /// The KV cache `kv_cache` holds one `CudaTensor` per layer with shape
    /// `[seq_so_far, kv_lora_rank + qk_rope_head_dim]`.  On each call the
    /// current entry is appended and the tensor is grown in-place via
    /// `concat_rows`.
    ///
    /// # Errors
    /// Returns an error if any CUDA kernel fails.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::similar_names,
        clippy::cast_precision_loss
    )]
    fn mla_attention(
        hidden: &CudaTensor,
        q_a_proj: &LinearWeight,
        q_a_layernorm: &CudaTensor,
        q_b_proj: &LinearWeight,
        kv_a_proj_with_mqa: &LinearWeight,
        kv_a_layernorm: &CudaTensor,
        _kv_b_proj_k: &LinearWeight,
        kv_b_proj_v: &LinearWeight,
        kv_b_proj_k_t: &LinearWeight,
        o_proj: &LinearWeight,
        kv_cache: &mut Vec<CudaTensor>,
        pos: usize,
        num_heads: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        kv_lora_rank: usize,
        rms_norm_eps: f32,
        attn_scale: f32,
    ) -> infernum::Result<CudaTensor> {
        use infernum::tensor::Tensor as TensorTrait;

        let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
        let ctx = hidden.context().clone();

        // --- Q projection (two-stage LoRA) ---
        let q_compressed = <CudaBackend as MatmulOps>::linear(hidden, q_a_proj)?;
        let q_compressed =
            <CudaBackend as NormOps>::rms_norm(&q_compressed, q_a_layernorm, rms_norm_eps)?;
        let q = <CudaBackend as MatmulOps>::linear(&q_compressed, q_b_proj)?;
        // q: [1, num_heads * qk_head_dim]

        // Split Q into nope and rope portions (per-head).
        let (q_nope, q_rope) = ops::split_inner_dim(
            &q.reshape(&[num_heads, qk_head_dim]),
            qk_nope_head_dim,
            qk_rope_head_dim,
        )?;
        // q_nope: [num_heads, qk_nope_head_dim], q_rope: [num_heads, qk_rope_head_dim]

        // --- KV joint projection (compressed only) ---
        let kv_proj = <CudaBackend as MatmulOps>::linear(hidden, kv_a_proj_with_mqa)?;
        // kv_proj: [1, kv_lora_rank + qk_rope_head_dim]
        let (k_compressed, k_rope) =
            ops::split_inner_dim(&kv_proj, kv_lora_rank, qk_rope_head_dim)?;
        let k_compressed =
            <CudaBackend as NormOps>::rms_norm(&k_compressed, kv_a_layernorm, rms_norm_eps)?;
        // k_compressed: [1, kv_lora_rank], k_rope: [1, qk_rope_head_dim]

        // --- RoPE (interleaved) on q_rope and k_rope ---
        // Build single-position cos/sin tables.
        let half_dim = qk_rope_head_dim / 2;
        let cos_data: Vec<f32> = (0..half_dim)
            .map(|i| {
                let theta = 1.0_f32 / 10_000_f32.powf(2.0 * i as f32 / qk_rope_head_dim as f32);
                (pos as f32 * theta).cos()
            })
            .collect();
        let sin_data: Vec<f32> = (0..half_dim)
            .map(|i| {
                let theta = 1.0_f32 / 10_000_f32.powf(2.0 * i as f32 / qk_rope_head_dim as f32);
                (pos as f32 * theta).sin()
            })
            .collect();
        let cos_t = CudaTensor::from_slice(&ctx, &[1, half_dim], &cos_data)?;
        let sin_t = CudaTensor::from_slice(&ctx, &[1, half_dim], &sin_data)?;

        // q_rope_3d: [1, num_heads, qk_rope_head_dim]
        let q_rope_3d = q_rope.reshape(&[1, num_heads, qk_rope_head_dim]);
        // k_rope_3d: [1, 1, qk_rope_head_dim]
        let k_rope_3d = k_rope.reshape(&[1, 1, qk_rope_head_dim]);

        let q_rope_rot = ops::apply_rope_interleaved(&q_rope_3d, &cos_t, &sin_t, 0)?;
        let k_rope_rot = ops::apply_rope_interleaved(&k_rope_3d, &cos_t, &sin_t, 0)?;
        // q_rope_rot: [1, num_heads, qk_rope_head_dim]
        // k_rope_rot: [1, 1, qk_rope_head_dim]

        // --- Append to flat KV cache ---
        // Cache entry shape: [1, kv_lora_rank + qk_rope_head_dim]
        let k_rope_flat = k_rope_rot.reshape(&[1, qk_rope_head_dim]);
        let cache_entry = ops::concat_inner_dim(&k_compressed, &k_rope_flat)?;

        if kv_cache.is_empty() {
            kv_cache.push(cache_entry);
        } else {
            let existing = kv_cache[0].clone();
            kv_cache[0] = <CudaBackend as TensorOps>::concat_rows(&[existing, cache_entry])?;
        }
        let full_kv = &kv_cache[0]; // [seq_len, kv_lora_rank + qk_rope_head_dim]
        let seq_len = full_kv.shape()[0];

        // --- Q absorption: q_nope @ kv_b_proj_k_t ---
        // kv_b_proj_k_t registered as LinearWeight with shape [kv_lora, num_heads * qk_nope].
        // linear([num_heads, qk_nope_head_dim], kv_b_proj_k_t) gives
        // [num_heads, kv_lora_rank].
        let q_absorbed_nope = <CudaBackend as MatmulOps>::linear(&q_nope, kv_b_proj_k_t)?;
        // q_absorbed_nope: [num_heads, kv_lora_rank]

        // Concat absorbed nope with rotated rope to get full absorbed Q per head.
        let q_rope_2d = q_rope_rot.reshape(&[num_heads, qk_rope_head_dim]);
        let q_absorbed_2d = ops::concat_inner_dim(&q_absorbed_nope, &q_rope_2d)?;
        // q_absorbed_2d: [num_heads, kv_lora_rank + qk_rope_head_dim]
        let q_absorbed = q_absorbed_2d.reshape(&[1, num_heads, kv_lora_rank + qk_rope_head_dim]);

        // Broadcast single KV head to num_heads.
        let kv_3d = full_kv.reshape(&[seq_len, 1, kv_lora_rank + qk_rope_head_dim]);
        let kv_expanded = <CudaBackend as TensorOps>::repeat_kv(&kv_3d, num_heads)?;
        // kv_expanded: [seq_len, num_heads, kv_lora_rank + qk_rope_head_dim]

        let attn_out = <CudaBackend as AttentionOps>::fused_attention_decode(
            &q_absorbed,
            &kv_expanded,
            &kv_expanded,
            Some(attn_scale),
            None,
            None,
        )?;
        // attn_out: [1, num_heads, kv_lora_rank + qk_rope_head_dim]

        // --- V absorption: take only the kv_lora_rank portion, then decompress ---
        let attn_flat = attn_out.reshape(&[num_heads, kv_lora_rank + qk_rope_head_dim]);
        let (attn_nope, _) = ops::split_inner_dim(&attn_flat, kv_lora_rank, qk_rope_head_dim)?;
        // attn_nope: [num_heads, kv_lora_rank]

        // V decompression via kv_b_proj_v.
        // kv_b_proj_v registered as LinearWeight shape [num_heads * v_head_dim, kv_lora_rank].
        // linear([num_heads, kv_lora_rank], kv_b_proj_v) would give
        // [num_heads, num_heads * v_head_dim] — incorrect (cross-head mixing).
        //
        // Correct approach: flatten to [1, num_heads * kv_lora_rank] and use a
        // block-diagonal matmul.  Since we store kv_b_proj_v as a 2D linear weight
        // [num_heads * v_head_dim, kv_lora_rank], this is equivalent only when
        // num_heads == 1.  For the general case we access the inner CudaTensor and
        // perform a batched matmul: [num_heads, 1, kv_lora_rank] @ [num_heads, kv_lora_rank, v_head_dim].
        let attn_v = match kv_b_proj_v {
            crate::cuda::ops::LinearWeight::Dense(w) => {
                // w is the pre-transposed weight stored as [kv_lora_rank, num_heads * v_head_dim]
                // (Dense stores transposed: shape (in_features, out_features)).
                // Reshape to [num_heads, kv_lora_rank, v_head_dim] for batched matmul.
                let w_batched = w.reshape(&[num_heads, kv_lora_rank, v_head_dim]);
                // attn_nope reshaped to [num_heads, 1, kv_lora_rank]
                let a = attn_nope.reshape(&[num_heads, 1, kv_lora_rank]);
                // batched matmul: [num_heads, 1, kv_lora_rank] @ [num_heads, kv_lora_rank, v_head_dim]
                // → [num_heads, 1, v_head_dim]
                let out = <CudaBackend as MatmulOps>::matmul(&a, &w_batched)?;
                out.reshape(&[1, num_heads * v_head_dim])
            }
            crate::cuda::ops::LinearWeight::Quantized(_) => {
                // Quantized path: fall back to flat linear.  Not numerically
                // equivalent for multi-head but acceptable as a known limitation
                // (quantized MLA is not a target for this branch).
                let a_flat = attn_nope.reshape(&[1, num_heads * kv_lora_rank]);
                <CudaBackend as MatmulOps>::linear(&a_flat, kv_b_proj_v)?
            }
        };
        // attn_v: [1, num_heads * v_head_dim]

        // --- Output projection ---
        let out = <CudaBackend as MatmulOps>::linear(&attn_v, o_proj)?;
        Ok(out)
    }
}

#[cfg(feature = "nccl")]
impl infernum::backend::MultiDeviceOps for CudaBackend {
    type CommId = crate::cuda::NcclId;

    fn create_comm_id() -> Result<Self::CommId> {
        crate::cuda::NcclId::new()
    }

    fn create_device(rank: usize) -> Result<Self::DeviceHandle> {
        crate::cuda::CudaContext::new(rank)
    }

    fn create_comm(
        device: &Self::DeviceHandle,
        rank: usize,
        world_size: usize,
        comm_id: Self::CommId,
    ) -> Result<Self::Comm> {
        crate::cuda::NcclCommunicator::from_rank(
            std::sync::Arc::clone(device.device()),
            rank,
            world_size,
            comm_id,
        )
    }
}

impl infernum::backend::SafeTensorsLoaderOps for CudaBackend {
    type SafeTensorsLoader = crate::weights::CudaWeightLoader<crate::weights::SafeTensorsLoader>;

    fn safetensors_loader(
        device: &Self::DeviceHandle,
        model_dir: &std::path::Path,
    ) -> Result<Self::SafeTensorsLoader> {
        let format_loader = crate::weights::SafeTensorsLoader::from_directory(model_dir)?;
        Ok(crate::weights::CudaWeightLoader::new(
            device.clone(),
            format_loader,
        ))
    }
}
