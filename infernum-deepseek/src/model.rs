//! DeepSeek V3 / R1 model implementation

#![allow(
    clippy::similar_names,
    clippy::struct_field_names,
    clippy::no_effect_underscore_binding,
    clippy::doc_markdown,
    dead_code,
    unused_mut
)]

use std::path::Path;

use infernum::cuda::block_allocator::BlockTable;
use infernum::cuda::ops::{
    add_inplace, add_rmsnorm, apply_rope_interleaved, apply_rope_interleaved_batched_indirect,
    apply_rope_interleaved_indirect, broadcast_to_heads, cast_from_f32, cast_to_f32,
    combine_attention_with_lse, concat_inner_dim, embedding_gather, embedding_gather_from_device,
    fused_attention_decode, fused_attention_decode_indirect, fused_attention_prefill,
    fused_attention_prefill_with_lse, gather_paged_kv, linear, matmul, matmul_bf16_f32,
    pad_inner_dim, paged_attention_decode, paged_attention_decode_indirect, precompute_rope_cache,
    precompute_rope_cache_scaled, quantized_matmul, rms_norm, rms_norm_inplace, split_inner_dim,
    swiglu, transpose_2d, LinearWeight, RopeScaling,
};
use infernum::cuda::{
    BatchedGraphInputs, CudaContext, CudaSlice, CudaTensor, GpuConfig, KvCache, PagedKvCache,
    QuantizedTensor,
};
#[cfg(feature = "nccl")]
use infernum::cuda::{NcclCommunicator, ShardConfig, ShardStrategy};
use infernum::dtype::DType;
use infernum::tensor::Tensor;
use infernum::weights::{SafeTensorsLoader, WeightLoader};
use infernum::Result;

use crate::DeepSeekConfig;

// --- NCCL conditional trait bounds (same pattern as infernum-llama / infernum-qwen) ---

#[cfg(feature = "nccl")]
fn nccl_all_reduce(comm: Option<&NcclCommunicator>, tensor: &mut CudaTensor) -> Result<()> {
    if let Some(comm) = comm {
        comm.all_reduce_sum_inplace(tensor)?;
    }
    Ok(())
}

// --- Weight helpers ---

fn pretranspose_weight(weight: &CudaTensor) -> Result<CudaTensor> {
    transpose_2d(weight)
}

fn load_typed(
    dtype: DType,
    loader: &impl WeightLoader,
    ctx: &CudaContext,
    name: &str,
) -> Result<CudaTensor> {
    match dtype {
        DType::F32 => loader.load_f32(ctx, name),
        DType::F16 => loader.load_f16(ctx, name),
        DType::BF16 => loader.load_bf16(ctx, name),
        other => panic!("Unsupported dtype for load_typed: {other}"),
    }
}

#[cfg(feature = "nccl")]
fn load_typed_sharded(
    dtype: DType,
    loader: &impl WeightLoader,
    ctx: &CudaContext,
    name: &str,
    shard: &ShardConfig,
    strategy: ShardStrategy,
) -> Result<CudaTensor> {
    match dtype {
        DType::F32 => loader.load_f32_sharded(ctx, name, shard, strategy),
        DType::F16 => loader.load_f16_sharded(ctx, name, shard, strategy),
        DType::BF16 => loader.load_bf16_sharded(ctx, name, shard, strategy),
        other => panic!("Unsupported dtype for load_typed_sharded: {other}"),
    }
}

// --- kv_b_proj splitting ---

/// Split a pre-transposed dense `kv_b_proj` weight into K-nope and V portions.
///
/// Input shape: `(kv_lora_rank, num_heads * (qk_nope_dim + v_head_dim))` (pre-transposed).
/// Columns are interleaved per head: for head `h`, columns
/// `[h*stride .. h*stride+qk_nope_dim]` are K-nope, followed by `v_head_dim` V columns.
///
/// Returns `(kv_b_proj_k, kv_b_proj_v, kv_b_proj_k_t)`:
/// - `kv_b_proj_k`: `(kv_lora_rank, num_heads * qk_nope_dim)` — K-nope decompression
/// - `kv_b_proj_v`: `(num_heads, kv_lora_rank, v_head_dim)` — V decompression (batched matmul)
/// - `kv_b_proj_k_t`: `(num_heads, qk_nope_dim, kv_lora_rank)` — Q absorption (batched matmul)
fn split_kv_b_proj_dense(
    ctx: &CudaContext,
    weight: &CudaTensor,
    num_heads: usize,
    qk_nope_dim: usize,
    v_head_dim: usize,
) -> Result<(CudaTensor, CudaTensor, CudaTensor)> {
    let shape = weight.shape();
    let dtype = weight.dtype();
    let elem = dtype.size_in_bytes();
    let kv_lora_rank = shape[0];
    let total_cols = shape[1];
    let stride = qk_nope_dim + v_head_dim;
    assert_eq!(
        total_cols,
        num_heads * stride,
        "split_kv_b_proj_dense: expected {} columns, got {total_cols}",
        num_heads * stride
    );

    let data = weight.to_raw_bytes()?;

    // Extract K-nope columns: shape (kv_lora_rank, num_heads * qk_nope_dim)
    let k_cols = num_heads * qk_nope_dim;
    let mut k_data = vec![0u8; kv_lora_rank * k_cols * elem];
    for row in 0..kv_lora_rank {
        for h in 0..num_heads {
            let src_offset = (row * total_cols + h * stride) * elem;
            let dst_offset = (row * k_cols + h * qk_nope_dim) * elem;
            let len = qk_nope_dim * elem;
            k_data[dst_offset..dst_offset + len]
                .copy_from_slice(&data[src_offset..src_offset + len]);
        }
    }

    // Extract V columns: shape (num_heads, kv_lora_rank, v_head_dim) for batched matmul
    let mut v_data = vec![0u8; num_heads * kv_lora_rank * v_head_dim * elem];
    for h in 0..num_heads {
        for row in 0..kv_lora_rank {
            let src_offset = (row * total_cols + h * stride + qk_nope_dim) * elem;
            let dst_offset = (h * kv_lora_rank * v_head_dim + row * v_head_dim) * elem;
            let len = v_head_dim * elem;
            v_data[dst_offset..dst_offset + len]
                .copy_from_slice(&data[src_offset..src_offset + len]);
        }
    }

    // K transposed per-head: shape (num_heads, qk_nope_dim, kv_lora_rank) for batched matmul.
    let mut k_t_data = vec![0u8; num_heads * qk_nope_dim * kv_lora_rank * elem];
    for h in 0..num_heads {
        for row in 0..kv_lora_rank {
            for col in 0..qk_nope_dim {
                let src_offset = (row * k_cols + h * qk_nope_dim + col) * elem;
                let dst_offset = (h * qk_nope_dim * kv_lora_rank + col * kv_lora_rank + row) * elem;
                k_t_data[dst_offset..dst_offset + elem]
                    .copy_from_slice(&k_data[src_offset..src_offset + elem]);
            }
        }
    }

    let k_tensor = CudaTensor::from_raw_bytes(ctx, &[kv_lora_rank, k_cols], dtype, &k_data)?;
    let v_tensor =
        CudaTensor::from_raw_bytes(ctx, &[num_heads, kv_lora_rank, v_head_dim], dtype, &v_data)?;
    let k_t_tensor = CudaTensor::from_raw_bytes(
        ctx,
        &[num_heads, qk_nope_dim, kv_lora_rank],
        dtype,
        &k_t_data,
    )?;

    Ok((k_tensor, v_tensor, k_t_tensor))
}

/// Split a quantized `kv_b_proj` by dequantizing to dense first, then splitting.
///
/// The quantized weight has shape `(out_features, in_features)` where
/// `out_features = num_heads * (qk_nope_dim + v_head_dim)` and `in_features = kv_lora_rank`.
/// We dequantize via identity matmul: `I @ W^T = W^T`, yielding the pre-transposed dense
/// weight `(kv_lora_rank, out_features)` that `split_kv_b_proj_dense` expects.
///
/// This is a one-time cost at model load time.
fn split_kv_b_proj_quantized(
    dtype: DType,
    ctx: &CudaContext,
    w: &QuantizedTensor,
    num_heads: usize,
    qk_nope_dim: usize,
    v_head_dim: usize,
) -> Result<(CudaTensor, CudaTensor, CudaTensor)> {
    let shape = w.shape();
    let out_features = shape[0];
    let kv_lora_rank = shape[1];
    assert_eq!(
        out_features,
        num_heads * (qk_nope_dim + v_head_dim),
        "split_kv_b_proj_quantized: out_features mismatch"
    );

    // Build identity matrix on host
    let mut identity_data = vec![0.0f32; kv_lora_rank * kv_lora_rank];
    for i in 0..kv_lora_rank {
        identity_data[i * kv_lora_rank + i] = 1.0;
    }
    let identity = CudaTensor::from_slice(ctx, &[kv_lora_rank, kv_lora_rank], &identity_data)?;

    // Dequantize: I @ W^T → (kv_lora_rank, out_features) as f32
    let dense_f32 = quantized_matmul(&identity, w)?;

    // Convert to target dtype and split
    let dense_t = cast_from_f32(&dense_f32, dtype)?;

    split_kv_b_proj_dense(ctx, &dense_t, num_heads, qk_nope_dim, v_head_dim)
}

// --- Weight structures ---

/// MLA attention weights (shared by dense and MoE layers)
struct DeepSeekAttentionWeights {
    q_a_proj: LinearWeight,
    q_a_layernorm: CudaTensor,
    q_b_proj: LinearWeight,
    kv_a_proj_with_mqa: LinearWeight,
    kv_a_layernorm: CudaTensor,
    kv_b_proj: LinearWeight,
    /// K-nope decompression columns, pre-transposed: `(kv_lora_rank, num_heads * qk_nope_dim)`
    kv_b_proj_k: LinearWeight,
    /// V decompression per-head: `(num_heads, kv_lora_rank, v_head_dim)` for batched V absorption
    kv_b_proj_v: CudaTensor,
    /// Transposed K per-head: `(num_heads, qk_nope_dim, kv_lora_rank)` for batched Q absorption
    kv_b_proj_k_t: CudaTensor,
    o_proj: LinearWeight,
}

/// Dense MLP weights
struct DenseMlpWeights {
    gate_proj: LinearWeight,
    up_proj: LinearWeight,
    down_proj: LinearWeight,
}

/// MoE layer weights
struct MoeWeights {
    gate_weight: CudaTensor,
    e_score_correction_bias: Vec<f32>,
    /// GPU-resident copy of the bias for the sigmoid routing kernel.
    e_score_correction_bias_gpu: CudaTensor,
    /// Pre-allocated GPU buffers for routing output, reused across decode steps.
    routing_bufs: std::sync::Mutex<infernum::cuda::ops::GpuRoutingBuffers>,
    experts: Vec<DenseMlpWeights>,
    shared_expert: DenseMlpWeights,
}

/// Dense vs MoE FFN layer
#[allow(clippy::large_enum_variant)]
enum FfnWeights {
    Dense(Box<DenseMlpWeights>),
    Moe(Box<MoeWeights>),
}

/// Single transformer layer
struct DeepSeekLayerWeights {
    input_layernorm: CudaTensor,
    attention: DeepSeekAttentionWeights,
    post_attention_layernorm: CudaTensor,
    ffn: FfnWeights,
}

/// Complete DeepSeek V3/R1 model.
pub struct DeepSeekModel {
    config: DeepSeekConfig,
    ctx: CudaContext,
    #[allow(dead_code)]
    gpu_config: GpuConfig,

    #[cfg(feature = "nccl")]
    nccl_comm: Option<NcclCommunicator>,

    tp_num_heads: usize,
    dtype: DType,

    embed_tokens: CudaTensor,
    layers: Vec<DeepSeekLayerWeights>,
    norm: CudaTensor,
    lm_head: LinearWeight,

    cos_cache: CudaTensor,
    sin_cache: CudaTensor,

    /// Pre-computed attention scale (includes YaRN mscale adjustment)
    attn_scale: f32,
}

impl DeepSeekModel {
    /// Load a DeepSeek model from a directory containing SafeTensors and config.json
    ///
    /// # Errors
    /// Returns an error if loading fails
    pub fn from_pretrained(ctx: &CudaContext, model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();
        let config_path = model_path.join("config.json");
        let config = DeepSeekConfig::from_file(&config_path)?;
        let loader = SafeTensorsLoader::from_directory(model_path)?;
        Self::load_weights(ctx, config, &loader)
    }

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &DeepSeekConfig {
        &self.config
    }

    // --- Weight loading ---

    #[allow(clippy::too_many_lines)]
    fn load_weights(
        ctx: &CudaContext,
        config: DeepSeekConfig,
        loader: &impl WeightLoader,
    ) -> Result<Self> {
        fn load_linear(
            model_dtype: DType,
            ctx: &CudaContext,
            loader: &impl WeightLoader,
            name: &str,
            quant_config: Option<&crate::config::QuantizationConfig>,
        ) -> Result<LinearWeight> {
            if let Some(qc) = quant_config {
                let prefix = name
                    .strip_suffix(".weight")
                    .expect("GPTQ/AWQ weight name must end with .weight");
                match qc.quant_method.as_str() {
                    "gptq" => {
                        let qt = loader.load_gptq_linear(ctx, prefix, qc.group_size)?;
                        return Ok(LinearWeight::Quantized(qt));
                    }
                    "awq" => {
                        let qt = loader.load_awq_linear(ctx, prefix, qc.group_size)?;
                        return Ok(LinearWeight::Quantized(qt));
                    }
                    _ => {}
                }
            }

            let dtype = loader.get_dtype(name)?;
            if dtype.is_quantized() {
                let mut qt = loader.load_quantized(ctx, name)?;
                let scale_name = format!("{name}_scale");
                if loader.contains(&scale_name) {
                    let scale_tensor = loader.load_f32(ctx, &scale_name)?;
                    let scale_val = scale_tensor.to_vec::<f32>()?;
                    if scale_val.len() == 1 {
                        qt.set_weight_scale(ctx, scale_val[0])?;
                    } else {
                        qt.set_channel_scales(ctx, &scale_val)?;
                    }
                }
                Ok(LinearWeight::Quantized(qt))
            } else {
                let f32_weight = loader.load_f32(ctx, name)?;
                let transposed = pretranspose_weight(&f32_weight)?;
                if model_dtype == DType::F32 {
                    Ok(LinearWeight::Dense(transposed))
                } else {
                    let native = load_typed(model_dtype, loader, ctx, name)?;
                    let raw = native.to_raw_bytes()?;
                    let shape = native.shape();
                    let rows = shape[0];
                    let cols = shape[1];
                    let elem = model_dtype.size_in_bytes();
                    let mut buf = vec![0u8; raw.len()];
                    for r in 0..rows {
                        for c in 0..cols {
                            let src = (r * cols + c) * elem;
                            let dst = (c * rows + r) * elem;
                            buf[dst..dst + elem].copy_from_slice(&raw[src..src + elem]);
                        }
                    }
                    Ok(LinearWeight::Dense(CudaTensor::from_raw_bytes(
                        ctx,
                        &[cols, rows],
                        model_dtype,
                        &buf,
                    )?))
                }
            }
        }

        let qc = config.quantization_config.as_ref();

        let embed_dtype = loader.get_dtype("model.embed_tokens.weight")?;
        let dtype = if embed_dtype.is_quantized() {
            DType::F32
        } else {
            embed_dtype
        };

        let embed_tokens = load_typed(dtype, loader, ctx, "model.embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            // MLA attention weights
            let kv_b_proj = load_linear(
                dtype,
                ctx,
                loader,
                &format!("{prefix}.self_attn.kv_b_proj.weight"),
                qc,
            )?;

            // Split kv_b_proj into K-nope and V portions for absorbed attention
            let (kv_b_proj_k, kv_b_proj_v, kv_b_proj_k_t) = match &kv_b_proj {
                LinearWeight::Dense(w) => {
                    let (k, v, k_t) = split_kv_b_proj_dense(
                        ctx,
                        w,
                        config.num_attention_heads,
                        config.qk_nope_head_dim,
                        config.v_head_dim,
                    )?;
                    (LinearWeight::Dense(k), v, k_t)
                }
                LinearWeight::Quantized(w) => {
                    let (k, v, k_t) = split_kv_b_proj_quantized(
                        dtype,
                        ctx,
                        w,
                        config.num_attention_heads,
                        config.qk_nope_head_dim,
                        config.v_head_dim,
                    )?;
                    (LinearWeight::Dense(k), v, k_t)
                }
            };

            let attention = DeepSeekAttentionWeights {
                q_a_proj: load_linear(
                    dtype,
                    ctx,
                    loader,
                    &format!("{prefix}.self_attn.q_a_proj.weight"),
                    qc,
                )?,
                q_a_layernorm: load_typed(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.self_attn.q_a_layernorm.weight"),
                )?,
                q_b_proj: load_linear(
                    dtype,
                    ctx,
                    loader,
                    &format!("{prefix}.self_attn.q_b_proj.weight"),
                    qc,
                )?,
                kv_a_proj_with_mqa: load_linear(
                    dtype,
                    ctx,
                    loader,
                    &format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
                    qc,
                )?,
                kv_a_layernorm: load_typed(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.self_attn.kv_a_layernorm.weight"),
                )?,
                kv_b_proj,
                kv_b_proj_k,
                kv_b_proj_v,
                kv_b_proj_k_t,
                o_proj: load_linear(
                    dtype,
                    ctx,
                    loader,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    qc,
                )?,
            };

            // FFN: dense or MoE
            let ffn = if config.is_moe_layer(i) {
                let num_experts = config
                    .n_routed_experts
                    .expect("MoE layer requires n_routed_experts");

                // Router gate
                let gate_name = format!("{prefix}.mlp.gate.weight");
                let gate_f32 = loader.load_f32(ctx, &gate_name)?;
                let gate_transposed = pretranspose_weight(&gate_f32)?;
                let gate_weight = cast_from_f32(&gate_transposed, dtype)?;

                // Bias correction
                let bias_name = format!("{prefix}.mlp.gate.e_score_correction_bias");
                let e_score_correction_bias = if loader.contains(&bias_name) {
                    loader.load_f32(ctx, &bias_name)?.to_vec::<f32>()?
                } else {
                    vec![0.0_f32; num_experts]
                };

                // Per-expert MLPs
                let mut experts = Vec::with_capacity(num_experts);
                for e in 0..num_experts {
                    let ep = format!("{prefix}.mlp.experts.{e}");
                    experts.push(DenseMlpWeights {
                        gate_proj: load_linear(
                            dtype,
                            ctx,
                            loader,
                            &format!("{ep}.gate_proj.weight"),
                            qc,
                        )?,
                        up_proj: load_linear(
                            dtype,
                            ctx,
                            loader,
                            &format!("{ep}.up_proj.weight"),
                            qc,
                        )?,
                        down_proj: load_linear(
                            dtype,
                            ctx,
                            loader,
                            &format!("{ep}.down_proj.weight"),
                            qc,
                        )?,
                    });
                }

                // Shared expert
                let sp = format!("{prefix}.mlp.shared_experts");
                let shared_expert = DenseMlpWeights {
                    gate_proj: load_linear(
                        dtype,
                        ctx,
                        loader,
                        &format!("{sp}.gate_proj.weight"),
                        qc,
                    )?,
                    up_proj: load_linear(dtype, ctx, loader, &format!("{sp}.up_proj.weight"), qc)?,
                    down_proj: load_linear(
                        dtype,
                        ctx,
                        loader,
                        &format!("{sp}.down_proj.weight"),
                        qc,
                    )?,
                };

                let e_score_correction_bias_gpu = CudaTensor::from_slice(
                    ctx,
                    &[e_score_correction_bias.len()],
                    &e_score_correction_bias,
                )?;

                let routing_bufs = std::sync::Mutex::new(
                    infernum::cuda::ops::GpuRoutingBuffers::new(ctx, config.num_experts_per_tok)?,
                );

                FfnWeights::Moe(Box::new(MoeWeights {
                    gate_weight,
                    e_score_correction_bias,
                    e_score_correction_bias_gpu,
                    routing_bufs,
                    experts,
                    shared_expert,
                }))
            } else {
                let mp = format!("{prefix}.mlp");
                FfnWeights::Dense(Box::new(DenseMlpWeights {
                    gate_proj: load_linear(
                        dtype,
                        ctx,
                        loader,
                        &format!("{mp}.gate_proj.weight"),
                        qc,
                    )?,
                    up_proj: load_linear(dtype, ctx, loader, &format!("{mp}.up_proj.weight"), qc)?,
                    down_proj: load_linear(
                        dtype,
                        ctx,
                        loader,
                        &format!("{mp}.down_proj.weight"),
                        qc,
                    )?,
                }))
            };

            layers.push(DeepSeekLayerWeights {
                input_layernorm: load_typed(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.input_layernorm.weight"),
                )?,
                attention,
                post_attention_layernorm: load_typed(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                ffn,
            });
        }

        let norm = load_typed(dtype, loader, ctx, "model.norm.weight")?;

        let lm_head = if config.tie_word_embeddings {
            let embed_f32 = cast_to_f32(&embed_tokens)?;
            let transposed = pretranspose_weight(&embed_f32)?;
            LinearWeight::Dense(cast_from_f32(&transposed, dtype)?)
        } else {
            load_linear(dtype, ctx, loader, "lm_head.weight", None)?
        };

        // RoPE cache — use qk_rope_head_dim (only rope portion gets RoPE)
        let rope_head_dim = config.qk_rope_head_dim;
        let (cos_f32, sin_f32) = if let Some(ref rs) = config.rope_scaling {
            let scaling = RopeScaling {
                rope_type: rs.rope_type.clone(),
                factor: rs.factor,
                original_max_position_embeddings: rs.original_max_position_embeddings,
            };
            precompute_rope_cache_scaled(
                ctx,
                config.max_position_embeddings,
                rope_head_dim,
                config.rope_theta,
                &scaling,
            )?
        } else {
            precompute_rope_cache(
                ctx,
                config.max_position_embeddings,
                rope_head_dim,
                config.rope_theta,
            )?
        };
        let cos_cache = cast_from_f32(&cos_f32, dtype)?;
        let sin_cache = cast_from_f32(&sin_f32, dtype)?;

        let attn_scale = config.mla_attn_scale();

        Ok(Self {
            tp_num_heads: config.num_attention_heads,
            dtype,
            config,
            ctx: ctx.clone(),
            gpu_config: GpuConfig::Single,
            #[cfg(feature = "nccl")]
            nccl_comm: None,
            embed_tokens,
            layers,
            norm,
            lm_head,
            cos_cache,
            sin_cache,
            attn_scale,
        })
    }

    /// Load a DeepSeek model with tensor-parallel sharding across multiple GPUs.
    ///
    /// # Errors
    /// Returns an error if loading fails or head counts are not divisible.
    #[cfg(feature = "nccl")]
    pub fn from_pretrained_sharded(
        ctx: &CudaContext,
        model_path: impl AsRef<Path>,
        gpu_config: GpuConfig,
        nccl_comm: NcclCommunicator,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        let config_path = model_path.join("config.json");
        let config = DeepSeekConfig::from_file(&config_path)?;
        let loader = SafeTensorsLoader::from_directory(model_path)?;
        Self::load_weights_sharded(ctx, config, &loader, gpu_config, nccl_comm)
    }

    #[cfg(feature = "nccl")]
    #[allow(clippy::too_many_lines)]
    fn load_weights_sharded(
        ctx: &CudaContext,
        config: DeepSeekConfig,
        loader: &impl WeightLoader,
        gpu_config: GpuConfig,
        nccl_comm: NcclCommunicator,
    ) -> Result<Self> {
        fn load_linear_sharded(
            model_dtype: DType,
            ctx: &CudaContext,
            loader: &impl WeightLoader,
            name: &str,
            shard: &ShardConfig,
            strategy: ShardStrategy,
            quant_config: Option<&crate::config::QuantizationConfig>,
        ) -> Result<LinearWeight> {
            if let Some(qc) = quant_config {
                let prefix = name
                    .strip_suffix(".weight")
                    .expect("GPTQ/AWQ weight name must end with .weight");
                match qc.quant_method.as_str() {
                    "gptq" => {
                        let qt = loader.load_gptq_linear_sharded(
                            ctx,
                            prefix,
                            qc.group_size,
                            shard,
                            strategy,
                        )?;
                        return Ok(LinearWeight::Quantized(qt));
                    }
                    "awq" => {
                        let qt = loader.load_awq_linear_sharded(
                            ctx,
                            prefix,
                            qc.group_size,
                            shard,
                            strategy,
                        )?;
                        return Ok(LinearWeight::Quantized(qt));
                    }
                    _ => {}
                }
            }

            let dtype = loader.get_dtype(name)?;
            if dtype.is_quantized() {
                let mut qt =
                    loader.load_quantized_sharded(ctx, name, shard, ShardStrategy::Replicate)?;
                let scale_name = format!("{name}_scale");
                if loader.contains(&scale_name) {
                    let scale_tensor = loader.load_f32(ctx, &scale_name)?;
                    let scale_val = scale_tensor.to_vec::<f32>()?;
                    if scale_val.len() == 1 {
                        qt.set_weight_scale(ctx, scale_val[0])?;
                    } else {
                        qt.set_channel_scales(ctx, &scale_val)?;
                    }
                }
                Ok(LinearWeight::Quantized(qt))
            } else {
                let f32_weight = loader.load_f32_sharded(ctx, name, shard, strategy)?;
                let transposed = pretranspose_weight(&f32_weight)?;
                if model_dtype == DType::F32 {
                    Ok(LinearWeight::Dense(transposed))
                } else {
                    let native =
                        load_typed_sharded(model_dtype, loader, ctx, name, shard, strategy)?;
                    let raw = native.to_raw_bytes()?;
                    let shape = native.shape();
                    let rows = shape[0];
                    let cols = shape[1];
                    let elem = model_dtype.size_in_bytes();
                    let mut buf = vec![0u8; raw.len()];
                    for r in 0..rows {
                        for c in 0..cols {
                            let src = (r * cols + c) * elem;
                            let dst = (c * rows + r) * elem;
                            buf[dst..dst + elem].copy_from_slice(&raw[src..src + elem]);
                        }
                    }
                    Ok(LinearWeight::Dense(CudaTensor::from_raw_bytes(
                        ctx,
                        &[cols, rows],
                        model_dtype,
                        &buf,
                    )?))
                }
            }
        }

        let shard = match &gpu_config {
            GpuConfig::Sharded(s) => *s,
            GpuConfig::Single => {
                return Self::load_weights(ctx, config, loader).map(|mut m| {
                    m.nccl_comm = Some(nccl_comm);
                    m
                })
            }
        };
        let world_size = shard.world_size;

        assert!(
            config.num_attention_heads.is_multiple_of(world_size),
            "num_attention_heads ({}) must be divisible by world_size ({world_size})",
            config.num_attention_heads
        );

        let qc = config.quantization_config.as_ref();
        let tp_num_heads = config.num_attention_heads / world_size;

        let embed_dtype = loader.get_dtype("model.embed_tokens.weight")?;
        let dtype = if embed_dtype.is_quantized() {
            DType::F32
        } else {
            embed_dtype
        };

        let embed_tokens = load_typed(dtype, loader, ctx, "model.embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            // MLA attention weights
            // q_a_proj, kv_a_proj_with_mqa: replicated (shared bottleneck)
            // q_b_proj, kv_b_proj: column-sharded (output is per-head)
            // o_proj: row-sharded (all-reduce after)
            let kv_b_proj = load_linear_sharded(
                dtype,
                ctx,
                loader,
                &format!("{prefix}.self_attn.kv_b_proj.weight"),
                &shard,
                ShardStrategy::Column,
                qc,
            )?;

            // Split kv_b_proj into K-nope and V portions (using tp_num_heads after sharding)
            let (kv_b_proj_k, kv_b_proj_v, kv_b_proj_k_t) = match &kv_b_proj {
                LinearWeight::Dense(w) => {
                    let (k, v, k_t) = split_kv_b_proj_dense(
                        ctx,
                        w,
                        tp_num_heads,
                        config.qk_nope_head_dim,
                        config.v_head_dim,
                    )?;
                    (LinearWeight::Dense(k), v, k_t)
                }
                LinearWeight::Quantized(w) => {
                    let (k, v, k_t) = split_kv_b_proj_quantized(
                        ctx,
                        w,
                        tp_num_heads,
                        config.qk_nope_head_dim,
                        config.v_head_dim,
                    )?;
                    (LinearWeight::Dense(k), v, k_t)
                }
            };

            let attention = DeepSeekAttentionWeights {
                q_a_proj: load_linear_sharded(
                    dtype,
                    ctx,
                    loader,
                    &format!("{prefix}.self_attn.q_a_proj.weight"),
                    &shard,
                    ShardStrategy::Replicate,
                    qc,
                )?,
                q_a_layernorm: load_typed(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.self_attn.q_a_layernorm.weight"),
                )?,
                q_b_proj: load_linear_sharded(
                    dtype,
                    ctx,
                    loader,
                    &format!("{prefix}.self_attn.q_b_proj.weight"),
                    &shard,
                    ShardStrategy::Column,
                    qc,
                )?,
                kv_a_proj_with_mqa: load_linear_sharded(
                    dtype,
                    ctx,
                    loader,
                    &format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
                    &shard,
                    ShardStrategy::Replicate,
                    qc,
                )?,
                kv_a_layernorm: load_typed(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.self_attn.kv_a_layernorm.weight"),
                )?,
                kv_b_proj,
                kv_b_proj_k,
                kv_b_proj_v,
                kv_b_proj_k_t,
                o_proj: load_linear_sharded(
                    dtype,
                    ctx,
                    loader,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    &shard,
                    ShardStrategy::Row,
                    qc,
                )?,
            };

            // FFN: dense or MoE
            let ffn = if config.is_moe_layer(i) {
                let num_experts = config
                    .n_routed_experts
                    .expect("MoE layer requires n_routed_experts");

                // Router gate: replicated
                let gate_name = format!("{prefix}.mlp.gate.weight");
                let gate_f32 = loader.load_f32(ctx, &gate_name)?;
                let gate_transposed = pretranspose_weight(&gate_f32)?;
                let gate_weight = cast_from_f32(&gate_transposed, dtype)?;

                // Bias correction: replicated
                let bias_name = format!("{prefix}.mlp.gate.e_score_correction_bias");
                let e_score_correction_bias = if loader.contains(&bias_name) {
                    loader.load_f32(ctx, &bias_name)?.to_vec::<f32>()?
                } else {
                    vec![0.0_f32; num_experts]
                };

                // Per-expert MLPs: gate/up column-sharded, down row-sharded
                let mut experts = Vec::with_capacity(num_experts);
                for e in 0..num_experts {
                    let ep = format!("{prefix}.mlp.experts.{e}");
                    experts.push(DenseMlpWeights {
                        gate_proj: load_linear_sharded(
                            dtype,
                            ctx,
                            loader,
                            &format!("{ep}.gate_proj.weight"),
                            &shard,
                            ShardStrategy::Column,
                            qc,
                        )?,
                        up_proj: load_linear_sharded(
                            dtype,
                            ctx,
                            loader,
                            &format!("{ep}.up_proj.weight"),
                            &shard,
                            ShardStrategy::Column,
                            qc,
                        )?,
                        down_proj: load_linear_sharded(
                            dtype,
                            ctx,
                            loader,
                            &format!("{ep}.down_proj.weight"),
                            &shard,
                            ShardStrategy::Row,
                            qc,
                        )?,
                    });
                }

                // Shared expert: gate/up column-sharded, down row-sharded
                let sp = format!("{prefix}.mlp.shared_experts");
                let shared_expert = DenseMlpWeights {
                    gate_proj: load_linear_sharded(
                        dtype,
                        ctx,
                        loader,
                        &format!("{sp}.gate_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?,
                    up_proj: load_linear_sharded(
                        dtype,
                        ctx,
                        loader,
                        &format!("{sp}.up_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?,
                    down_proj: load_linear_sharded(
                        dtype,
                        ctx,
                        loader,
                        &format!("{sp}.down_proj.weight"),
                        &shard,
                        ShardStrategy::Row,
                        qc,
                    )?,
                };

                let e_score_correction_bias_gpu = CudaTensor::from_slice(
                    ctx,
                    &[e_score_correction_bias.len()],
                    &e_score_correction_bias,
                )?;

                let routing_bufs = std::sync::Mutex::new(
                    infernum::cuda::ops::GpuRoutingBuffers::new(ctx, config.num_experts_per_tok)?,
                );

                FfnWeights::Moe(Box::new(MoeWeights {
                    gate_weight,
                    e_score_correction_bias,
                    e_score_correction_bias_gpu,
                    routing_bufs,
                    experts,
                    shared_expert,
                }))
            } else {
                let mp = format!("{prefix}.mlp");
                FfnWeights::Dense(Box::new(DenseMlpWeights {
                    gate_proj: load_linear_sharded(
                        dtype,
                        ctx,
                        loader,
                        &format!("{mp}.gate_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?,
                    up_proj: load_linear_sharded(
                        dtype,
                        ctx,
                        loader,
                        &format!("{mp}.up_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?,
                    down_proj: load_linear_sharded(
                        dtype,
                        ctx,
                        loader,
                        &format!("{mp}.down_proj.weight"),
                        &shard,
                        ShardStrategy::Row,
                        qc,
                    )?,
                }))
            };

            layers.push(DeepSeekLayerWeights {
                input_layernorm: load_typed(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.input_layernorm.weight"),
                )?,
                attention,
                post_attention_layernorm: load_typed(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                ffn,
            });
        }

        let norm = load_typed(dtype, loader, ctx, "model.norm.weight")?;

        let lm_head = if config.tie_word_embeddings {
            let embed_f32 = cast_to_f32(&embed_tokens)?;
            let transposed = pretranspose_weight(&embed_f32)?;
            LinearWeight::Dense(cast_from_f32(&transposed, dtype)?)
        } else {
            load_linear_sharded(
                dtype,
                ctx,
                loader,
                "lm_head.weight",
                &shard,
                ShardStrategy::Replicate,
                None,
            )?
        };

        let rope_head_dim = config.qk_rope_head_dim;
        let (cos_f32, sin_f32) = if let Some(ref rs) = config.rope_scaling {
            let scaling = RopeScaling {
                rope_type: rs.rope_type.clone(),
                factor: rs.factor,
                original_max_position_embeddings: rs.original_max_position_embeddings,
            };
            precompute_rope_cache_scaled(
                ctx,
                config.max_position_embeddings,
                rope_head_dim,
                config.rope_theta,
                &scaling,
            )?
        } else {
            precompute_rope_cache(
                ctx,
                config.max_position_embeddings,
                rope_head_dim,
                config.rope_theta,
            )?
        };
        let cos_cache = cast_from_f32(&cos_f32, dtype)?;
        let sin_cache = cast_from_f32(&sin_f32, dtype)?;

        let attn_scale = config.mla_attn_scale();

        Ok(Self {
            tp_num_heads,
            dtype,
            config,
            ctx: ctx.clone(),
            gpu_config,
            nccl_comm: Some(nccl_comm),
            embed_tokens,
            layers,
            norm,
            lm_head,
            cos_cache,
            sin_cache,
            attn_scale,
        })
    }

    // --- Forward pass ---

    fn embed(&self, input_ids: &[u32]) -> Result<CudaTensor> {
        embedding_gather(&self.ctx, &self.embed_tokens, input_ids)
    }

    fn embed_from_device(&self, token_id_gpu: &CudaSlice<u32>) -> Result<CudaTensor> {
        embedding_gather_from_device(&self.ctx, &self.embed_tokens, token_id_gpu, 1)
    }

    fn extract_last_row(&self, hidden: &CudaTensor, seq_len: usize) -> Result<CudaTensor> {
        if seq_len == 1 {
            return Ok(hidden.reshape(&[1, self.config.hidden_size]));
        }
        let hidden_size = hidden.shape()[1];
        let flat = hidden.reshape(&[seq_len * hidden_size]);
        let mut out = unsafe { CudaTensor::uninit(&self.ctx, &[1, hidden_size], self.dtype)? };
        let device = self.ctx.device();
        let src = flat.cuda_slice();
        let last_offset = (seq_len - 1) * hidden_size;
        let src_sub = src.slice(last_offset..seq_len * hidden_size);
        device.dtod_copy(&src_sub, out.cuda_slice_mut())?;
        Ok(out)
    }

    fn lm_head_forward(&self, hidden: &CudaTensor) -> Result<CudaTensor> {
        if self.dtype == DType::BF16 {
            if let LinearWeight::Dense(w) = &self.lm_head {
                return matmul_bf16_f32(hidden, w);
            }
        }
        let logits_t = linear(hidden, &self.lm_head)?;
        if self.dtype == DType::F32 {
            return Ok(logits_t);
        }
        cast_to_f32(&logits_t)
    }

    // --- MLA attention ---

    /// Split a 2D tensor `[seq, total_dim]` into two parts along the last dimension.
    fn split_last_dim(
        tensor: &CudaTensor,
        dim1: usize,
        dim2: usize,
    ) -> Result<(CudaTensor, CudaTensor)> {
        let seq_len = tensor.shape()[0];
        let total = tensor.shape()[1];
        assert_eq!(total, dim1 + dim2, "split_last_dim: dim mismatch");

        if seq_len == 1 {
            let a = tensor.slice_view(0, &[1, dim1]);
            let b = tensor.slice_view(dim1, &[1, dim2]);
            Ok((a, b))
        } else {
            split_inner_dim(tensor, dim1, dim2)
        }
    }

    /// Split a 3D tensor `[seq, num_heads, total_dim]` into two parts along last dim.
    fn split_head_dim(
        tensor: &CudaTensor,
        dim1: usize,
        dim2: usize,
    ) -> Result<(CudaTensor, CudaTensor)> {
        let seq_len = tensor.shape()[0];
        let num_heads = tensor.shape()[1];
        let total = tensor.shape()[2];
        assert_eq!(total, dim1 + dim2, "split_head_dim: dim mismatch");

        let outer = seq_len * num_heads;
        let flat = tensor.reshape(&[outer, total]);
        let (a_flat, b_flat) = split_inner_dim(&flat, dim1, dim2)?;
        let a = a_flat.reshape(&[seq_len, num_heads, dim1]);
        let b = b_flat.reshape(&[seq_len, num_heads, dim2]);
        Ok((a, b))
    }

    /// Concatenate two 3D tensors along the last dimension.
    fn concat_head_dim(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        let seq_len = a.shape()[0];
        let num_heads = a.shape()[1];
        let dim1 = a.shape()[2];
        let dim2 = b.shape()[2];
        assert_eq!(seq_len, b.shape()[0]);
        assert_eq!(num_heads, b.shape()[1]);

        let outer = seq_len * num_heads;
        let a_flat = a.reshape(&[outer, dim1]);
        let b_flat = b.reshape(&[outer, dim2]);
        let out_flat = concat_inner_dim(&a_flat, &b_flat)?;
        Ok(out_flat.reshape(&[seq_len, num_heads, dim1 + dim2]))
    }

    /// Broadcast `[seq, 1, rope_dim]` to `[seq, num_heads, rope_dim]`.
    fn broadcast_kv_rope(k_rope: &CudaTensor, num_heads: usize) -> Result<CudaTensor> {
        broadcast_to_heads(k_rope, num_heads)
    }

    /// Pad V from `[seq, num_heads, v_head_dim]` to `[seq, num_heads, qk_head_dim]`.
    fn pad_v_to_qk_dim(v: &CudaTensor, qk_head_dim: usize) -> Result<CudaTensor> {
        let seq_len = v.shape()[0];
        let num_heads = v.shape()[1];
        let v_dim = v.shape()[2];
        if v_dim == qk_head_dim {
            return Ok(v.slice_view(0, v.shape()));
        }

        let outer = seq_len * num_heads;
        let flat = v.reshape(&[outer, v_dim]);
        let padded = pad_inner_dim(&flat, qk_head_dim)?;
        Ok(padded.reshape(&[seq_len, num_heads, qk_head_dim]))
    }

    /// Truncate attention output from `[seq, num_heads, qk_head_dim]` to
    /// `[seq, num_heads, v_head_dim]`.
    fn truncate_attn_output(attn_out: &CudaTensor, v_head_dim: usize) -> Result<CudaTensor> {
        let seq_len = attn_out.shape()[0];
        let num_heads = attn_out.shape()[1];
        let qk_dim = attn_out.shape()[2];
        if qk_dim == v_head_dim {
            return Ok(attn_out.slice_view(0, attn_out.shape()));
        }

        let outer = seq_len * num_heads;
        let flat = attn_out.reshape(&[outer, qk_dim]);
        let discard_dim = qk_dim - v_head_dim;
        let (kept, _) = split_inner_dim(&flat, v_head_dim, discard_dim)?;
        Ok(kept.reshape(&[seq_len, num_heads, v_head_dim]))
    }

    /// Absorb K into Q: per-head `q_nope_h @ W_k_h^T` → `q_absorbed_nope`.
    ///
    /// `q_nope`: `(1, num_heads, qk_nope_dim)`
    /// `kv_b_proj_k_t`: `(num_heads, qk_nope_dim, kv_lora_rank)`
    /// Returns: `(1, num_heads, kv_lora_rank)`
    fn absorb_k_into_q(q_nope: &CudaTensor, kv_b_proj_k_t: &CudaTensor) -> Result<CudaTensor> {
        let num_heads = q_nope.shape()[1];
        let d = q_nope.shape()[2];
        let kv_lora_rank = kv_b_proj_k_t.shape()[2];
        // (num_heads, 1, D) @ (num_heads, D, R) → (num_heads, 1, R) → (1, H, R)
        let q = q_nope.reshape(&[num_heads, 1, d]);
        let out = matmul(&q, kv_b_proj_k_t)?;
        Ok(out.reshape(&[1, num_heads, kv_lora_rank]))
    }

    /// Absorb V: per-head `attn_nope_h @ W_v_h` → V output.
    ///
    /// `attn_nope`: `(1, num_heads, kv_lora_rank)`
    /// `kv_b_proj_v`: `(num_heads, kv_lora_rank, v_head_dim)`
    /// Returns: `(1, num_heads, v_head_dim)`
    fn absorb_v(attn_nope: &CudaTensor, kv_b_proj_v: &CudaTensor) -> Result<CudaTensor> {
        let num_heads = attn_nope.shape()[1];
        let r = attn_nope.shape()[2];
        let v_head_dim = kv_b_proj_v.shape()[2];
        // (num_heads, 1, R) @ (num_heads, R, V) → (num_heads, 1, V) → (1, H, V)
        let a = attn_nope.reshape(&[num_heads, 1, r]);
        let out = matmul(&a, kv_b_proj_v)?;
        Ok(out.reshape(&[1, num_heads, v_head_dim]))
    }

    /// Batched K absorption for batch > 1 (CUDA graph indirect decode).
    ///
    /// `q_nope`: `(batch, num_heads, qk_nope_dim)` — contiguous
    /// `kv_b_proj_k_t`: `(num_heads, qk_nope_dim, kv_lora_rank)`
    /// Returns: `(batch, num_heads, kv_lora_rank)`
    fn absorb_k_batched(
        q_nope: &CudaTensor,
        kv_b_proj_k_t: &CudaTensor,
        batch_size: usize,
        num_heads: usize,
        kv_lora_rank: usize,
    ) -> Result<CudaTensor> {
        let d = q_nope.shape()[2];
        let item_elems = num_heads * d;
        let out_elems = num_heads * kv_lora_rank;

        let elem = q_nope.dtype().size_in_bytes();
        let mut all_bytes: Vec<u8> = Vec::with_capacity(batch_size * out_elems * elem);
        for b in 0..batch_size {
            let q_b = q_nope.slice_view(b * item_elems, &[num_heads, 1, d]);
            let out_b = matmul(&q_b, kv_b_proj_k_t)?; // (H, 1, R)
            all_bytes.extend_from_slice(&out_b.to_raw_bytes()?);
        }
        CudaTensor::from_raw_bytes(
            q_nope.context(),
            &[batch_size, num_heads, kv_lora_rank],
            q_nope.dtype(),
            &all_bytes,
        )
    }

    /// Batched V absorption for batch > 1 (CUDA graph indirect decode).
    ///
    /// `attn_nope`: `(batch, num_heads, kv_lora_rank)`
    /// `kv_b_proj_v`: `(num_heads, kv_lora_rank, v_head_dim)`
    /// Returns: `(batch, num_heads, v_head_dim)`
    fn absorb_v_batched(
        attn_nope: &CudaTensor,
        kv_b_proj_v: &CudaTensor,
        batch_size: usize,
        num_heads: usize,
        v_head_dim: usize,
    ) -> Result<CudaTensor> {
        let r = attn_nope.shape()[2];
        let item_elems = num_heads * r;
        let out_elems = num_heads * v_head_dim;

        let elem = attn_nope.dtype().size_in_bytes();
        let mut all_bytes: Vec<u8> = Vec::with_capacity(batch_size * out_elems * elem);
        for b in 0..batch_size {
            let a_b = attn_nope.slice_view(b * item_elems, &[num_heads, 1, r]);
            let out_b = matmul(&a_b, kv_b_proj_v)?; // (H, 1, V)
            all_bytes.extend_from_slice(&out_b.to_raw_bytes()?);
        }
        CudaTensor::from_raw_bytes(
            attn_nope.context(),
            &[batch_size, num_heads, v_head_dim],
            attn_nope.dtype(),
            &all_bytes,
        )
    }

    /// MLA attention forward pass.
    #[allow(clippy::too_many_lines)]
    fn forward_mla_attention(
        &self,
        hidden: &CudaTensor,
        weights: &DeepSeekAttentionWeights,
        layer_idx: usize,
        kv_cache: &mut KvCache,
        position_offset: usize,
    ) -> Result<CudaTensor> {
        let seq_len = hidden.shape()[0];
        let num_heads = self.tp_num_heads;
        let qk_nope_dim = self.config.qk_nope_head_dim;
        let qk_rope_dim = self.config.qk_rope_head_dim;
        let qk_head_dim = self.config.qk_head_dim();
        let v_head_dim = self.config.v_head_dim;
        let kv_lora_rank = self.config.kv_lora_rank;

        // --- Q projection: two-stage LoRA ---
        // q_compressed = hidden @ q_a_proj → [seq, q_lora_rank]
        let q_compressed = linear(hidden, &weights.q_a_proj)?;
        // RMSNorm
        let q_compressed = rms_norm(
            &q_compressed,
            &weights.q_a_layernorm,
            self.config.rms_norm_eps,
        )?;
        // q = q_compressed @ q_b_proj → [seq, num_heads * qk_head_dim]
        let q = linear(&q_compressed, &weights.q_b_proj)?;

        // Reshape to [seq, num_heads, qk_head_dim]
        let q = q.reshape(&[seq_len, num_heads, qk_head_dim]);
        // Split into nope and rope portions
        let (q_nope, q_rope) = Self::split_head_dim(&q, qk_nope_dim, qk_rope_dim)?;

        // --- KV joint projection ---
        // kv = hidden @ kv_a_proj → [seq, kv_lora_rank + qk_rope_head_dim]
        let kv = linear(hidden, &weights.kv_a_proj_with_mqa)?;
        // Split into compressed KV and k_rope
        let (k_compressed, k_rope) = Self::split_last_dim(&kv, kv_lora_rank, qk_rope_dim)?;

        // RMSNorm the compressed KV
        let k_compressed = rms_norm(
            &k_compressed,
            &weights.kv_a_layernorm,
            self.config.rms_norm_eps,
        )?;

        // Decompress: kv_decompressed = k_compressed @ kv_b_proj → [seq, num_heads * (qk_nope_dim + v_head_dim)]
        let kv_decompressed = linear(&k_compressed, &weights.kv_b_proj)?;

        // Reshape to [seq, num_heads, qk_nope_dim + v_head_dim] and split
        let kv_decompressed =
            kv_decompressed.reshape(&[seq_len, num_heads, qk_nope_dim + v_head_dim]);
        let (k_nope, v) = Self::split_head_dim(&kv_decompressed, qk_nope_dim, v_head_dim)?;

        // --- RoPE ---
        let k_rope = k_rope.reshape(&[seq_len, 1, qk_rope_dim]);
        let q_rope =
            apply_rope_interleaved(&q_rope, &self.cos_cache, &self.sin_cache, position_offset)?;
        let k_rope =
            apply_rope_interleaved(&k_rope, &self.cos_cache, &self.sin_cache, position_offset)?;

        // --- Write compressed entry to KV cache ---
        let k_rope_2d = k_rope.reshape(&[seq_len, qk_rope_dim]);
        let cache_entry_2d = concat_inner_dim(&k_compressed, &k_rope_2d)?;
        let cache_entry = cache_entry_2d.reshape(&[seq_len, 1, kv_lora_rank + qk_rope_dim]);
        kv_cache.append(layer_idx, &cache_entry, &cache_entry)?;
        let total_len = kv_cache.current_len() + seq_len;

        // --- Attention ---
        let attn_output = if seq_len == 1 {
            // Decode: absorbed attention in compressed space
            let q_absorbed_nope = Self::absorb_k_into_q(&q_nope, &weights.kv_b_proj_k_t)?;
            let q_absorbed = Self::concat_head_dim(&q_absorbed_nope, &q_rope)?;

            let (k_full, _v_full) = kv_cache.get_up_to(layer_idx, total_len);
            let attn_compressed = fused_attention_decode(
                &q_absorbed,
                &k_full,
                &k_full,
                Some(self.attn_scale),
                None,
                None,
            )?;

            // Absorb V: discard rope, decompress
            let (attn_nope, _) = Self::split_head_dim(&attn_compressed, kv_lora_rank, qk_rope_dim)?;
            Self::absorb_v(&attn_nope, &weights.kv_b_proj_v)?
        } else {
            // Prefill: use locally-decompressed K/V
            let k_rope_broadcast = Self::broadcast_kv_rope(&k_rope, num_heads)?;
            let q = Self::concat_head_dim(&q_nope, &q_rope)?;
            let k = Self::concat_head_dim(&k_nope, &k_rope_broadcast)?;
            let v_padded = Self::pad_v_to_qk_dim(&v, qk_head_dim)?;
            let attn = fused_attention_prefill(
                &q,
                &k,
                &v_padded,
                position_offset,
                Some(self.attn_scale),
                None,
                None,
            )?;
            Self::truncate_attn_output(&attn, v_head_dim)?
        };

        // --- Output projection ---
        let attn_output = attn_output.reshape(&[seq_len, num_heads * v_head_dim]);
        let mut out = linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    /// MLA attention forward using indirect kernels for CUDA graph capture.
    #[allow(clippy::too_many_lines)]
    fn forward_mla_attention_indirect(
        &self,
        hidden: &CudaTensor,
        weights: &DeepSeekAttentionWeights,
        layer_idx: usize,
        kv_cache: &mut KvCache,
    ) -> Result<CudaTensor> {
        let num_heads = self.tp_num_heads;
        let qk_nope_dim = self.config.qk_nope_head_dim;
        let qk_rope_dim = self.config.qk_rope_head_dim;
        let qk_head_dim = self.config.qk_head_dim();
        let v_head_dim = self.config.v_head_dim;
        let kv_lora_rank = self.config.kv_lora_rank;

        let q_compressed = linear(hidden, &weights.q_a_proj)?;
        let q_compressed = rms_norm(
            &q_compressed,
            &weights.q_a_layernorm,
            self.config.rms_norm_eps,
        )?;
        let q = linear(&q_compressed, &weights.q_b_proj)?;
        let q = q.reshape(&[1, num_heads, qk_head_dim]);
        let (q_nope, q_rope) = Self::split_head_dim(&q, qk_nope_dim, qk_rope_dim)?;

        // KV joint projection (compressed only)
        let kv = linear(hidden, &weights.kv_a_proj_with_mqa)?;
        let (k_compressed, k_rope) = Self::split_last_dim(&kv, kv_lora_rank, qk_rope_dim)?;
        let k_compressed = rms_norm(
            &k_compressed,
            &weights.kv_a_layernorm,
            self.config.rms_norm_eps,
        )?;

        let k_rope = k_rope.reshape(&[1, 1, qk_rope_dim]);
        let position = kv_cache.current_position();
        let q_rope =
            apply_rope_interleaved_indirect(&q_rope, &self.cos_cache, &self.sin_cache, position)?;
        let k_rope =
            apply_rope_interleaved_indirect(&k_rope, &self.cos_cache, &self.sin_cache, position)?;

        // Write compressed entry to KV cache
        let k_rope_2d = k_rope.reshape(&[1, qk_rope_dim]);
        let cache_entry_2d = concat_inner_dim(&k_compressed, &k_rope_2d)?;
        let cache_entry = cache_entry_2d.reshape(&[1, 1, kv_lora_rank + qk_rope_dim]);
        kv_cache.append_indirect(layer_idx, &cache_entry, &cache_entry)?;

        // Absorb K into Q
        let q_absorbed_nope = Self::absorb_k_into_q(&q_nope, &weights.kv_b_proj_k_t)?;
        let q_absorbed = Self::concat_head_dim(&q_absorbed_nope, &q_rope)?;

        let (k_full, _v_full) = kv_cache.full_buffers(layer_idx);
        let total_len = kv_cache.current_total_len();
        let attn_compressed = fused_attention_decode_indirect(
            &q_absorbed,
            k_full,
            k_full,
            total_len,
            kv_cache.graph_max_seq_len(),
            Some(self.attn_scale),
            None,
            None,
        )?;

        // Absorb V: discard rope, decompress
        let (attn_nope, _) = Self::split_head_dim(&attn_compressed, kv_lora_rank, qk_rope_dim)?;
        let attn_output = Self::absorb_v(&attn_nope, &weights.kv_b_proj_v)?;
        let attn_output = attn_output.reshape(&[1, num_heads * v_head_dim]);
        let mut out = linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    // --- FFN ---

    #[allow(clippy::unused_self)]
    fn forward_mlp(&self, hidden: &CudaTensor, weights: &DenseMlpWeights) -> Result<CudaTensor> {
        let gate = linear(hidden, &weights.gate_proj)?;
        let up = linear(hidden, &weights.up_proj)?;
        let intermediate = swiglu(&gate, &up)?;
        let mut out = linear(&intermediate, &weights.down_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    #[allow(clippy::unused_self)]
    fn forward_mlp_no_reduce(
        &self,
        hidden: &CudaTensor,
        weights: &DenseMlpWeights,
    ) -> Result<CudaTensor> {
        let gate = linear(hidden, &weights.gate_proj)?;
        let up = linear(hidden, &weights.up_proj)?;
        let intermediate = swiglu(&gate, &up)?;
        linear(&intermediate, &weights.down_proj)
    }

    fn forward_moe(&self, hidden: &CudaTensor, moe_weights: &MoeWeights) -> Result<CudaTensor> {
        let num_experts = moe_weights.experts.len();

        let mut routing_bufs = moe_weights.routing_bufs.lock().unwrap();
        let mut routed_output = infernum::cuda::moe::moe_forward_sigmoid_gpu(
            hidden,
            &moe_weights.gate_weight,
            &moe_weights.e_score_correction_bias,
            &moe_weights.e_score_correction_bias_gpu,
            num_experts,
            self.config.num_experts_per_tok,
            self.config.n_group,
            self.config.topk_group,
            self.config.norm_topk_prob,
            self.config.routed_scaling_factor,
            &mut routing_bufs,
            |expert_idx, expert_input| {
                self.forward_mlp_no_reduce(expert_input, &moe_weights.experts[expert_idx])
            },
        )?;

        // Add shared expert output
        let shared_output = self.forward_mlp_no_reduce(hidden, &moe_weights.shared_expert)?;
        add_inplace(&mut routed_output, &shared_output)?;

        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut routed_output)?;
        Ok(routed_output)
    }

    fn forward_ffn(&self, hidden: &CudaTensor, ffn: &FfnWeights) -> Result<CudaTensor> {
        match ffn {
            FfnWeights::Dense(mlp) => self.forward_mlp(hidden, mlp),
            FfnWeights::Moe(moe) => self.forward_moe(hidden, moe),
        }
    }

    // --- Layer forward ---

    fn forward_layer_kv(
        &self,
        hidden: &CudaTensor,
        layer: &DeepSeekLayerWeights,
        layer_idx: usize,
        kv_cache: &mut KvCache,
        position_offset: usize,
    ) -> Result<CudaTensor> {
        let normed = rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;
        let attn_output = self.forward_mla_attention(
            &normed,
            &layer.attention,
            layer_idx,
            kv_cache,
            position_offset,
        )?;

        let (mut h, normed) = add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
        add_inplace(&mut h, &mlp_output)?;
        Ok(h)
    }

    fn forward_layer_kv_indirect(
        &self,
        hidden: &CudaTensor,
        layer: &DeepSeekLayerWeights,
        layer_idx: usize,
        kv_cache: &mut KvCache,
    ) -> Result<CudaTensor> {
        let normed = rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;
        let attn_output =
            self.forward_mla_attention_indirect(&normed, &layer.attention, layer_idx, kv_cache)?;

        let (mut h, normed) = add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
        add_inplace(&mut h, &mlp_output)?;
        Ok(h)
    }

    // --- Paged KV cache support ---

    /// MLA attention for single-sequence prefill with paged KV cache.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn forward_mla_attention_paged_prefill(
        &self,
        hidden: &CudaTensor,
        weights: &DeepSeekAttentionWeights,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache,
        block_table: &BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<CudaTensor> {
        let num_heads = self.tp_num_heads;
        let qk_nope_dim = self.config.qk_nope_head_dim;
        let qk_rope_dim = self.config.qk_rope_head_dim;
        let qk_head_dim = self.config.qk_head_dim();
        let v_head_dim = self.config.v_head_dim;
        let kv_lora_rank = self.config.kv_lora_rank;

        // Q projection (two-stage LoRA)
        let q_compressed = linear(hidden, &weights.q_a_proj)?;
        let q_compressed = rms_norm(
            &q_compressed,
            &weights.q_a_layernorm,
            self.config.rms_norm_eps,
        )?;
        let q = linear(&q_compressed, &weights.q_b_proj)?;
        let q = q.reshape(&[seq_len, num_heads, qk_head_dim]);
        let (q_nope, q_rope) = Self::split_head_dim(&q, qk_nope_dim, qk_rope_dim)?;

        // KV joint projection
        let kv = linear(hidden, &weights.kv_a_proj_with_mqa)?;
        let (k_compressed, k_rope) = Self::split_last_dim(&kv, kv_lora_rank, qk_rope_dim)?;
        let k_compressed = rms_norm(
            &k_compressed,
            &weights.kv_a_layernorm,
            self.config.rms_norm_eps,
        )?;
        let kv_decompressed = linear(&k_compressed, &weights.kv_b_proj)?;
        let kv_decompressed =
            kv_decompressed.reshape(&[seq_len, num_heads, qk_nope_dim + v_head_dim]);
        let (k_nope, v) = Self::split_head_dim(&kv_decompressed, qk_nope_dim, v_head_dim)?;

        // RoPE (interleaved)
        let k_rope = k_rope.reshape(&[seq_len, 1, qk_rope_dim]);
        let q_rope = apply_rope_interleaved(&q_rope, &self.cos_cache, &self.sin_cache, start_pos)?;
        let k_rope = apply_rope_interleaved(&k_rope, &self.cos_cache, &self.sin_cache, start_pos)?;

        // Write compressed entry to paged cache: (seq_len, 1, kv_lora_rank + qk_rope_dim)
        let k_rope_2d = k_rope.reshape(&[seq_len, qk_rope_dim]);
        let cache_entry_2d = concat_inner_dim(&k_compressed, &k_rope_2d)?;
        let cache_entry = cache_entry_2d.reshape(&[seq_len, 1, kv_lora_rank + qk_rope_dim]);
        paged_kv.append_paged(
            layer_idx,
            block_table,
            &cache_entry,
            &cache_entry,
            start_pos,
        )?;

        // Attention: single-chunk vs multi-chunk
        let attn_output = if start_pos == 0 {
            // Single-chunk prefill: local decompressed attention only
            let k_rope_broadcast = Self::broadcast_kv_rope(&k_rope, num_heads)?;
            let q_full = Self::concat_head_dim(&q_nope, &q_rope)?;
            let k_full = Self::concat_head_dim(&k_nope, &k_rope_broadcast)?;
            let v_padded = Self::pad_v_to_qk_dim(&v, qk_head_dim)?;

            let out = fused_attention_prefill(
                &q_full,
                &k_full,
                &v_padded,
                0,
                Some(self.attn_scale),
                None,
                None,
            )?;
            Self::truncate_attn_output(&out, v_head_dim)?
        } else {
            // Multi-chunk prefill: two-pass with LSE combining
            //
            // Pass 1: attend over cached compressed tokens (absorbed path)
            let q_absorbed_nope = Self::absorb_k_batched(
                &q_nope,
                &weights.kv_b_proj_k_t,
                seq_len,
                num_heads,
                kv_lora_rank,
            )?;
            let q_absorbed = Self::concat_head_dim(&q_absorbed_nope, &q_rope)?;
            let (cached_k, _cached_v) = gather_paged_kv(paged_kv, layer_idx, block_table)?;
            // cached_k: (start_pos, 1, kv_lora_rank + qk_rope_dim) — no causal mask needed
            let (out_cached, lse_cached) = fused_attention_prefill_with_lse(
                &q_absorbed,
                &cached_k,
                &cached_k,
                0,
                Some(self.attn_scale),
                None,
                None,
            )?;
            // Discard rope portion, absorb V
            let (attn_nope_cached, _) =
                Self::split_head_dim(&out_cached, kv_lora_rank, qk_rope_dim)?;
            let out_cached_v = Self::absorb_v_batched(
                &attn_nope_cached,
                &weights.kv_b_proj_v,
                seq_len,
                num_heads,
                v_head_dim,
            )?;

            // Pass 2: attend over local decompressed tokens (causal mask)
            let k_rope_broadcast = Self::broadcast_kv_rope(&k_rope, num_heads)?;
            let q_full = Self::concat_head_dim(&q_nope, &q_rope)?;
            let k_full = Self::concat_head_dim(&k_nope, &k_rope_broadcast)?;
            let v_padded = Self::pad_v_to_qk_dim(&v, qk_head_dim)?;
            let (out_local, lse_local) = fused_attention_prefill_with_lse(
                &q_full,
                &k_full,
                &v_padded,
                0,
                Some(self.attn_scale),
                None,
                None,
            )?;
            let out_local_v = Self::truncate_attn_output(&out_local, v_head_dim)?;

            // Combine with numerically-stable LSE correction
            combine_attention_with_lse(&out_cached_v, &lse_cached, &out_local_v, &lse_local)?
        };
        let attn_output = attn_output.reshape(&[seq_len, num_heads * v_head_dim]);
        let mut out = linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    /// MLA attention for single-token decode with paged KV cache.
    fn forward_mla_attention_paged_decode(
        &self,
        hidden: &CudaTensor,
        weights: &DeepSeekAttentionWeights,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache,
        block_table: &BlockTable,
        position: usize,
    ) -> Result<CudaTensor> {
        let num_heads = self.tp_num_heads;
        let qk_nope_dim = self.config.qk_nope_head_dim;
        let qk_rope_dim = self.config.qk_rope_head_dim;
        let qk_head_dim = self.config.qk_head_dim();
        let v_head_dim = self.config.v_head_dim;
        let kv_lora_rank = self.config.kv_lora_rank;

        // Q projection (two-stage LoRA)
        let q_compressed = linear(hidden, &weights.q_a_proj)?;
        let q_compressed = rms_norm(
            &q_compressed,
            &weights.q_a_layernorm,
            self.config.rms_norm_eps,
        )?;
        let q = linear(&q_compressed, &weights.q_b_proj)?;
        let q = q.reshape(&[1, num_heads, qk_head_dim]);
        let (q_nope, q_rope) = Self::split_head_dim(&q, qk_nope_dim, qk_rope_dim)?;

        // KV joint projection (compressed only — no decompression for absorbed decode)
        let kv = linear(hidden, &weights.kv_a_proj_with_mqa)?;
        let (k_compressed, k_rope) = Self::split_last_dim(&kv, kv_lora_rank, qk_rope_dim)?;
        let k_compressed = rms_norm(
            &k_compressed,
            &weights.kv_a_layernorm,
            self.config.rms_norm_eps,
        )?;

        // RoPE (interleaved)
        let k_rope = k_rope.reshape(&[1, 1, qk_rope_dim]);
        let q_rope = apply_rope_interleaved(&q_rope, &self.cos_cache, &self.sin_cache, position)?;
        let k_rope = apply_rope_interleaved(&k_rope, &self.cos_cache, &self.sin_cache, position)?;

        // Write compressed entry to paged cache: (1, 1, kv_lora_rank + qk_rope_dim)
        let k_rope_2d = k_rope.reshape(&[1, qk_rope_dim]);
        let cache_entry_2d = concat_inner_dim(&k_compressed, &k_rope_2d)?;
        let cache_entry = cache_entry_2d.reshape(&[1, 1, kv_lora_rank + qk_rope_dim]);
        paged_kv.append_paged(layer_idx, block_table, &cache_entry, &cache_entry, position)?;

        // Absorbed decode attention (Q absorption + paged attention in compressed space)
        let q_absorbed_nope = Self::absorb_k_into_q(&q_nope, &weights.kv_b_proj_k_t)?;
        let q_absorbed = Self::concat_head_dim(&q_absorbed_nope, &q_rope)?;

        let mut table_with_current = block_table.clone();
        table_with_current.advance(1);
        let (k_pool, _v_pool) = paged_kv.get_pools(layer_idx);
        let attn_output = paged_attention_decode(
            &self.ctx,
            &q_absorbed,
            k_pool,
            k_pool,
            &[table_with_current],
            paged_kv.block_size(),
            Some(self.attn_scale),
            None,
            None,
        )?;

        // Absorb V: discard rope portion, decompress via kv_b_proj_v
        let (attn_nope, _) = Self::split_head_dim(&attn_output, kv_lora_rank, qk_rope_dim)?;
        let attn_output = Self::absorb_v(&attn_nope, &weights.kv_b_proj_v)?;

        let attn_output = attn_output.reshape(&[1, num_heads * v_head_dim]);
        let mut out = linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    /// Layer forward for paged prefill.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_paged_prefill(
        &self,
        hidden: &CudaTensor,
        layer: &DeepSeekLayerWeights,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache,
        block_table: &BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<CudaTensor> {
        let normed = rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;
        let attn_output = self.forward_mla_attention_paged_prefill(
            &normed,
            &layer.attention,
            layer_idx,
            paged_kv,
            block_table,
            start_pos,
            seq_len,
        )?;

        let (mut h, normed) = add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
        add_inplace(&mut h, &mlp_output)?;
        Ok(h)
    }

    /// Layer forward for paged decode (single token).
    fn forward_layer_paged_decode(
        &self,
        hidden: &CudaTensor,
        layer: &DeepSeekLayerWeights,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache,
        block_table: &BlockTable,
        position: usize,
    ) -> Result<CudaTensor> {
        let normed = rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;
        let attn_output = self.forward_mla_attention_paged_decode(
            &normed,
            &layer.attention,
            layer_idx,
            paged_kv,
            block_table,
            position,
        )?;

        let (mut h, normed) = add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
        add_inplace(&mut h, &mlp_output)?;
        Ok(h)
    }

    // --- Batched indirect decode (CUDA graph compatible) ---

    /// MLA attention for batched decode using GPU-resident positions, block tables,
    /// and sequence lengths.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn forward_mla_attention_paged_decode_indirect(
        &self,
        hidden: &CudaTensor,
        weights: &DeepSeekAttentionWeights,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache,
        graph_inputs: &BatchedGraphInputs,
        max_seq_len: usize,
    ) -> Result<CudaTensor> {
        let batch_size = hidden.shape()[0];
        let num_heads = self.tp_num_heads;
        let qk_nope_dim = self.config.qk_nope_head_dim;
        let qk_rope_dim = self.config.qk_rope_head_dim;
        let qk_head_dim = self.config.qk_head_dim();
        let v_head_dim = self.config.v_head_dim;
        let kv_lora_rank = self.config.kv_lora_rank;

        // Q projection (two-stage LoRA)
        let q_compressed = linear(hidden, &weights.q_a_proj)?;
        let q_compressed = rms_norm(
            &q_compressed,
            &weights.q_a_layernorm,
            self.config.rms_norm_eps,
        )?;
        let q = linear(&q_compressed, &weights.q_b_proj)?;
        let q = q.reshape(&[batch_size, num_heads, qk_head_dim]);
        let (q_nope, q_rope) = Self::split_head_dim(&q, qk_nope_dim, qk_rope_dim)?;

        // KV joint projection (compressed only — no decompression via kv_b_proj)
        let kv = linear(hidden, &weights.kv_a_proj_with_mqa)?;
        let (k_compressed, k_rope) = Self::split_last_dim(&kv, kv_lora_rank, qk_rope_dim)?;
        let k_compressed = rms_norm(
            &k_compressed,
            &weights.kv_a_layernorm,
            self.config.rms_norm_eps,
        )?;

        // RoPE (interleaved, batched from GPU-resident positions)
        let k_rope = k_rope.reshape(&[batch_size, 1, qk_rope_dim]);
        let q_rope = apply_rope_interleaved_batched_indirect(
            &q_rope,
            &self.cos_cache,
            &self.sin_cache,
            graph_inputs.positions(),
            batch_size,
        )?;
        let k_rope = apply_rope_interleaved_batched_indirect(
            &k_rope,
            &self.cos_cache,
            &self.sin_cache,
            graph_inputs.positions(),
            batch_size,
        )?;

        // Write compressed entry to paged cache: (batch_size, 1, kv_lora_rank + qk_rope_dim)
        let k_rope_2d = k_rope.reshape(&[batch_size, qk_rope_dim]);
        let cache_entry_2d = concat_inner_dim(&k_compressed, &k_rope_2d)?;
        let cache_entry = cache_entry_2d.reshape(&[batch_size, 1, kv_lora_rank + qk_rope_dim]);
        paged_kv.append_paged_batched(
            layer_idx,
            &cache_entry,
            &cache_entry,
            graph_inputs.block_tables(),
            graph_inputs.positions(),
            batch_size,
            graph_inputs.max_blocks_per_seq(),
        )?;

        // Absorb K into Q: per-head batched matmul
        // q_nope: (B, H, D) — process each batch item with absorb_k_into_q
        let q_absorbed_nope = Self::absorb_k_batched(
            &q_nope,
            &weights.kv_b_proj_k_t,
            batch_size,
            num_heads,
            kv_lora_rank,
        )?;
        let q_absorbed = Self::concat_head_dim(&q_absorbed_nope, &q_rope)?;

        // Paged attention in compressed space (K pool = V pool)
        let (k_pool, _v_pool) = paged_kv.get_pools(layer_idx);
        let attn_output = paged_attention_decode_indirect(
            &self.ctx,
            &q_absorbed,
            k_pool,
            k_pool,
            graph_inputs.block_tables(),
            graph_inputs.seq_lens(),
            paged_kv.block_size(),
            graph_inputs.max_blocks_per_seq(),
            max_seq_len,
            Some(self.attn_scale),
            None,
            None,
        )?;

        // Absorb V: discard rope portion, decompress via kv_b_proj_v
        let (attn_nope, _) = Self::split_head_dim(&attn_output, kv_lora_rank, qk_rope_dim)?;
        let attn_output = Self::absorb_v_batched(
            &attn_nope,
            &weights.kv_b_proj_v,
            batch_size,
            num_heads,
            v_head_dim,
        )?;
        let attn_output = attn_output.reshape(&[batch_size, num_heads * v_head_dim]);
        let mut out = linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    /// Layer forward for batched decode with paged KV cache (indirect).
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_paged_decode_indirect(
        &self,
        hidden: &CudaTensor,
        layer: &DeepSeekLayerWeights,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache,
        graph_inputs: &BatchedGraphInputs,
        max_seq_len: usize,
    ) -> Result<CudaTensor> {
        let normed = rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;
        let attn_output = self.forward_mla_attention_paged_decode_indirect(
            &normed,
            &layer.attention,
            layer_idx,
            paged_kv,
            graph_inputs,
            max_seq_len,
        )?;

        let (mut h, normed) = add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
        add_inplace(&mut h, &mlp_output)?;
        Ok(h)
    }

    /// Batched decode using indirect kernels for CUDA graph capture.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward_batch_decode_indirect(
        &self,
        graph_inputs: &BatchedGraphInputs,
        paged_kvs: &mut [PagedKvCache],
        max_seq_len: usize,
    ) -> Result<CudaTensor> {
        let batch_size = graph_inputs.max_batch_size();
        let paged_kv = &mut paged_kvs[0];

        // Embed from GPU-resident token IDs
        let mut hidden = embedding_gather_from_device(
            &self.ctx,
            &self.embed_tokens,
            graph_inputs.token_ids(),
            batch_size,
        )?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer_paged_decode_indirect(
                &hidden,
                layer,
                layer_idx,
                paged_kv,
                graph_inputs,
                max_seq_len,
            )?;
        }

        rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;

        self.lm_head_forward(&hidden.reshape(&[batch_size, self.config.hidden_size]))
    }

    // --- Public forward methods ---

    /// Forward pass with KV cache (prefill phase)
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward_with_kv_cache(
        &self,
        input_ids: &[u32],
        kv_cache: &mut KvCache,
    ) -> Result<CudaTensor> {
        let seq_len = input_ids.len();
        let position_offset = kv_cache.current_len();

        let mut hidden = self.embed(input_ids)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer_kv(&hidden, layer, layer_idx, kv_cache, position_offset)?;
        }

        kv_cache.advance(seq_len)?;
        rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;

        let last_hidden = self.extract_last_row(&hidden, seq_len)?;
        self.lm_head_forward(&last_hidden)
    }

    /// Forward pass for a single token with KV cache (decode phase)
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward_next_token(&self, token_id: u32, kv_cache: &mut KvCache) -> Result<CudaTensor> {
        self.forward_with_kv_cache(&[token_id], kv_cache)
    }

    /// Decode-phase forward pass reading the token from a GPU buffer.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward_next_token_device(
        &self,
        token_id_gpu: &CudaSlice<u32>,
        kv_cache: &mut KvCache,
    ) -> Result<CudaTensor> {
        let position_offset = kv_cache.current_len();

        let mut hidden = self.embed_from_device(token_id_gpu)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer_kv(&hidden, layer, layer_idx, kv_cache, position_offset)?;
        }

        kv_cache.advance(1)?;
        rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;

        let last_hidden = hidden.reshape(&[1, self.config.hidden_size]);
        self.lm_head_forward(&last_hidden)
    }

    /// Forward pass using indirect kernels for CUDA graph capture.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward_next_token_indirect(
        &self,
        token_id_gpu: &CudaSlice<u32>,
        kv_cache: &mut KvCache,
    ) -> Result<CudaTensor> {
        let mut hidden = self.embed_from_device(token_id_gpu)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer_kv_indirect(&hidden, layer, layer_idx, kv_cache)?;
        }

        rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;

        let last_hidden = hidden.reshape(&[1, self.config.hidden_size]);
        self.lm_head_forward(&last_hidden)
    }
}

// --- Model trait implementation ---

impl infernum::Model for DeepSeekModel {
    fn config(&self) -> infernum::ModelConfig {
        let config = &self.config;
        infernum::ModelConfig {
            num_layers: config.num_hidden_layers,
            max_seq_len: config.max_position_embeddings,
            // Compressed KV cache: single latent vector per token
            num_kv_heads: 1,
            head_dim: config.kv_lora_rank + config.qk_rope_head_dim,
            eos_token_id: config.eos_token_id,
            cache_dtype: self.dtype,
        }
    }

    fn devices(&self) -> Vec<&CudaContext> {
        vec![&self.ctx]
    }

    fn forward(&self, input_ids: &[u32]) -> Result<CudaTensor> {
        let seq_len = input_ids.len();
        let mut hidden = self.embed(input_ids)?;

        // Full-recompute forward is not meaningful without KV cache for MLA,
        // but we provide a minimal implementation that doesn't use cache.
        // This path is not used in production — prefer forward_with_kv_cache.
        for layer in &self.layers {
            let normed = rms_norm(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            // Simplified: skip attention, just do FFN (for trait compliance)
            let (mut h, normed) = add_rmsnorm(
                &hidden,
                &normed,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;

            let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
            add_inplace(&mut h, &mlp_output)?;
            hidden = h;
        }

        rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;

        if seq_len > 1 {
            let last = self.extract_last_row(&hidden, seq_len)?;
            self.lm_head_forward(&last)
        } else {
            self.lm_head_forward(&hidden)
        }
    }

    fn forward_prefill_paged(
        &self,
        input_ids: &[u32],
        paged_kvs: &mut [PagedKvCache],
        block_table: &BlockTable,
        start_pos: usize,
    ) -> Result<CudaTensor> {
        let seq_len = input_ids.len();
        let paged_kv = &mut paged_kvs[0];

        let mut hidden = self.embed(input_ids)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer_paged_prefill(
                &hidden,
                layer,
                layer_idx,
                paged_kv,
                block_table,
                start_pos,
                seq_len,
            )?;
        }

        rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;

        let last_hidden = self.extract_last_row(&hidden, seq_len)?;
        self.lm_head_forward(&last_hidden)
    }

    fn forward_batch_decode(
        &self,
        token_ids: &[u32],
        paged_kvs: &mut [PagedKvCache],
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<CudaTensor> {
        let batch_size = token_ids.len();
        let paged_kv = &mut paged_kvs[0];
        let hidden_size = self.config.hidden_size;

        // Process each sequence independently through the full MLA pipeline.
        // MLA's multi-stage projections (Q LoRA, KV joint, per-head decomposition)
        // make batching non-trivial; per-sequence is correct and simpler.
        let mut logits_parts = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut hidden = self.embed(&token_ids[i..=i])?;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden = self.forward_layer_paged_decode(
                    &hidden,
                    layer,
                    layer_idx,
                    paged_kv,
                    &block_tables[i],
                    positions[i],
                )?;
            }

            rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;
            let logits = self.lm_head_forward(&hidden.reshape(&[1, hidden_size]))?;
            logits_parts.push(logits);
        }

        if batch_size == 1 {
            return Ok(logits_parts.into_iter().next().unwrap());
        }

        // Concatenate per-sequence logits into (batch_size, vocab_size)
        let vocab_size = logits_parts[0].shape()[1];
        let mut output =
            unsafe { CudaTensor::uninit(&self.ctx, &[batch_size, vocab_size], DType::F32)? };
        let out_slice = output.cuda_slice_mut();
        for (i, part) in logits_parts.iter().enumerate() {
            let src = part.cuda_slice().slice(..vocab_size);
            let mut dst = out_slice.slice_mut(i * vocab_size..(i + 1) * vocab_size);
            self.ctx.device().dtod_copy(&src, &mut dst)?;
        }
        Ok(output)
    }

    fn forward_batch_decode_indirect(
        &self,
        graph_inputs: &BatchedGraphInputs,
        paged_kvs: &mut [PagedKvCache],
        max_seq_len: usize,
    ) -> Result<CudaTensor> {
        self.forward_batch_decode_indirect(graph_inputs, paged_kvs, max_seq_len)
    }
}

#[cfg(feature = "nccl")]
impl infernum::ShardedLoadable for DeepSeekModel {
    fn load_shard(
        ctx: &CudaContext,
        model_path: &Path,
        shard: ShardConfig,
        comm: NcclCommunicator,
    ) -> Result<Self> {
        Self::from_pretrained_sharded(ctx, model_path, GpuConfig::Sharded(shard), comm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_kv_b_proj_shapes() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let kv_lora_rank = 16;
        let num_heads = 4;
        let qk_nope_dim = 8;
        let v_head_dim = 8;
        let total_cols = num_heads * (qk_nope_dim + v_head_dim);

        // Create a dummy pre-transposed kv_b_proj: (kv_lora_rank, total_cols)
        let data: Vec<f32> = (0..kv_lora_rank * total_cols).map(|i| i as f32).collect();
        let weight = CudaTensor::from_slice(&ctx, &[kv_lora_rank, total_cols], &data).unwrap();

        let (k, v, k_t) =
            split_kv_b_proj_dense(&ctx, &weight, num_heads, qk_nope_dim, v_head_dim).unwrap();

        assert_eq!(k.shape(), &[kv_lora_rank, num_heads * qk_nope_dim]);
        assert_eq!(v.shape(), &[num_heads, kv_lora_rank, v_head_dim]);
        assert_eq!(k_t.shape(), &[num_heads, qk_nope_dim, kv_lora_rank]);
    }

    #[test]
    fn split_kv_b_proj_roundtrip() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let kv_lora_rank = 4;
        let num_heads = 2;
        let qk_nope_dim = 3;
        let v_head_dim = 5;
        let stride = qk_nope_dim + v_head_dim;
        let total_cols = num_heads * stride;

        let data: Vec<f32> = (0..kv_lora_rank * total_cols)
            .map(|i| i as f32 * 0.1)
            .collect();
        let weight = CudaTensor::from_slice(&ctx, &[kv_lora_rank, total_cols], &data).unwrap();

        let (k, v, _) =
            split_kv_b_proj_dense(&ctx, &weight, num_heads, qk_nope_dim, v_head_dim).unwrap();

        let k_data = k.to_vec::<f32>().unwrap();
        let v_data = v.to_vec::<f32>().unwrap();

        // Reconstruct the original by interleaving K and V columns back
        // K is (kv_lora_rank, num_heads * qk_nope_dim), V is (num_heads, kv_lora_rank, v_head_dim)
        let mut reconstructed = vec![0.0_f32; kv_lora_rank * total_cols];
        for row in 0..kv_lora_rank {
            for h in 0..num_heads {
                // K columns: k_data layout is (kv_lora_rank, num_heads * qk_nope_dim)
                for d in 0..qk_nope_dim {
                    reconstructed[row * total_cols + h * stride + d] =
                        k_data[row * (num_heads * qk_nope_dim) + h * qk_nope_dim + d];
                }
                // V columns: v_data layout is (num_heads, kv_lora_rank, v_head_dim)
                for d in 0..v_head_dim {
                    reconstructed[row * total_cols + h * stride + qk_nope_dim + d] =
                        v_data[h * kv_lora_rank * v_head_dim + row * v_head_dim + d];
                }
            }
        }

        for (i, (&orig, &recon)) in data.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (orig - recon).abs() < 1e-6,
                "Mismatch at index {i}: orig={orig}, recon={recon}"
            );
        }
    }

    #[test]
    fn split_kv_b_proj_k_transpose() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let kv_lora_rank = 4;
        let num_heads = 2;
        let qk_nope_dim = 3;
        let v_head_dim = 5;
        let total_cols = num_heads * (qk_nope_dim + v_head_dim);

        let data: Vec<f32> = (0..kv_lora_rank * total_cols)
            .map(|i| i as f32 * 0.1)
            .collect();
        let weight = CudaTensor::from_slice(&ctx, &[kv_lora_rank, total_cols], &data).unwrap();

        let (k, _, k_t) =
            split_kv_b_proj_dense(&ctx, &weight, num_heads, qk_nope_dim, v_head_dim).unwrap();

        let k_cols = num_heads * qk_nope_dim;
        assert_eq!(k_t.shape(), &[num_heads, qk_nope_dim, kv_lora_rank]);

        let k_data = k.to_vec::<f32>().unwrap();
        let k_t_data = k_t.to_vec::<f32>().unwrap();

        // Verify k_t[h][d][r] == k[r][h * qk_nope_dim + d] (per-head transpose)
        for h in 0..num_heads {
            for d in 0..qk_nope_dim {
                for r in 0..kv_lora_rank {
                    let k_val = k_data[r * k_cols + h * qk_nope_dim + d];
                    let k_t_val = k_t_data[h * qk_nope_dim * kv_lora_rank + d * kv_lora_rank + r];
                    assert!(
                        (k_val - k_t_val).abs() < 1e-6,
                        "Transpose mismatch at h={h}, d={d}, r={r}: k={k_val}, k_t={k_t_val}"
                    );
                }
            }
        }
    }
}
