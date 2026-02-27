//! `CudaBackend` — implements the infernum `Backend` + op traits for CUDA.

use infernum::backend::{
    ArithOps, AttentionOps, Backend, BiasOps, CastOps, EmbedOps, GegluOps, KvCacheOps,
    MatmulExtOps, MatmulOps, MoeOps, NormOps, PagedAttentionOps, PagedKvCacheOps,
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
    type Logits = CudaLogits;

    #[cfg(feature = "nccl")]
    type Comm = crate::cuda::NcclCommunicator;
    #[cfg(not(feature = "nccl"))]
    type Comm = ();

    fn logits_from_tensor(tensor: CudaTensor) -> CudaLogits {
        CudaLogits::new(tensor)
    }
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
        let cols = parts[0].shape().last().copied().unwrap_or(0);
        let dtype = parts[0].dtype();
        let elem = dtype.size_in_bytes();
        let total_rows: usize = parts.iter().map(|p| p.shape()[0]).sum();
        let ctx = parts[0].context();

        let mut output = unsafe { CudaTensor::uninit(ctx, &[total_rows, cols], dtype)? };
        let out_slice = output.cuda_slice_mut();
        let mut offset = 0;
        for part in parts {
            let row_bytes = part.shape()[0] * cols * elem;
            let src = part.cuda_slice().slice(..row_bytes);
            let mut dst = out_slice.slice_mut(offset..offset + row_bytes);
            ctx.device().dtod_copy(&src, &mut dst)?;
            offset += row_bytes;
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

#[cfg(feature = "nccl")]
impl infernum::backend::AllReduceOps for CudaBackend {
    type OldComm = crate::cuda::NcclCommunicator;

    fn all_reduce_sum_inplace(comm: &Self::OldComm, tensor: &mut CudaTensor) -> Result<()> {
        comm.all_reduce_sum_inplace(tensor)
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
