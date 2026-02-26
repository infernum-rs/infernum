//! Gemma model implementation (Gemma 2 + Gemma 3 text)

#![allow(
    clippy::struct_field_names,
    clippy::no_effect_underscore_binding,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::module_name_repetitions,
    clippy::manual_div_ceil,
    unused_mut
)]

use std::marker::PhantomData;
use std::path::Path;

use infernum::backend::{
    ArithOps, AttentionOps, Backend, CastOps, EmbedOps, GegluOps, MatmulExtOps, NormOps,
    PagedAttentionOps, PagedKvCacheOps, RopeOps, TensorOps,
};
use infernum::dtype::DType;
use infernum::tensor::Tensor;
use infernum::Result;
use infernum_cuda::cuda::ops::{
    apply_rope_batched_indirect, cast_from_f32, embedding_gather_from_device, matmul,
    paged_attention_decode_indirect, precompute_rope_cache, quantized_matmul, transpose_2d,
};
#[cfg(feature = "nccl")]
use infernum_cuda::cuda::{
    shard_strategy_for_weight, NcclCommunicator, ShardConfig, ShardStrategy,
};
use infernum_cuda::cuda::{
    BatchedGraphInputs, CudaContext, CudaTensor, GpuConfig, PagedKvCache, QuantizedTensor,
};
use infernum_cuda::weights::{SafeTensorsLoader, WeightLoader};
use infernum_cuda::BlockTable;
use infernum_cuda::CudaBackend;

use crate::GemmaConfig;

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

fn concat_weights(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
    let dtype = a.dtype();
    let elem = dtype.size_in_bytes();
    let k = a.shape()[0];
    assert_eq!(k, b.shape()[0], "concat_weights: K dimension mismatch");
    let n1 = a.shape()[1];
    let n2 = b.shape()[1];

    let a_bytes = a.to_raw_bytes()?;
    let b_bytes = b.to_raw_bytes()?;
    let mut out = vec![0u8; k * (n1 + n2) * elem];
    for row in 0..k {
        let out_off = row * (n1 + n2) * elem;
        let a_off = row * n1 * elem;
        let b_off = row * n2 * elem;
        out[out_off..out_off + n1 * elem].copy_from_slice(&a_bytes[a_off..a_off + n1 * elem]);
        out[out_off + n1 * elem..out_off + (n1 + n2) * elem]
            .copy_from_slice(&b_bytes[b_off..b_off + n2 * elem]);
    }
    CudaTensor::from_raw_bytes(a.context(), &[k, n1 + n2], dtype, &out)
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

/// Load a Gemma RMSNorm weight and add 1.0 to every element.
///
/// Gemma's RMSNorm computes `x_normed * (1 + weight)` (weights initialized to
/// zeros), unlike Llama's `x_normed * weight` (weights initialized to ones).
/// By pre-adding 1.0 at load time, we can reuse the standard RMSNorm kernel.
fn load_gemma_norm(
    dtype: DType,
    loader: &impl WeightLoader,
    ctx: &CudaContext,
    name: &str,
) -> Result<CudaTensor> {
    let gpu_tensor = load_typed(dtype, loader, ctx, name)?;
    let f32_tensor = CudaBackend::cast_to_f32(&gpu_tensor)?;
    let mut host: Vec<f32> = f32_tensor.to_vec::<f32>()?;
    for v in &mut host {
        *v += 1.0;
    }
    let adjusted = CudaTensor::from_slice(ctx, gpu_tensor.shape(), &host)?;
    cast_from_f32(&adjusted, dtype)
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

// --- Weight structures ---

enum LinearWeight {
    Dense(CudaTensor),
    Quantized(QuantizedTensor),
}

enum KvProjWeight {
    Fused {
        weight: CudaTensor,
        kv_dim: usize,
    },
    Separate {
        k_proj: Box<LinearWeight>,
        v_proj: Box<LinearWeight>,
    },
}

enum GateUpWeight {
    Fused {
        weight: CudaTensor,
        intermediate_size: usize,
    },
    Separate {
        gate_proj: Box<LinearWeight>,
        up_proj: Box<LinearWeight>,
    },
}

struct GemmaAttentionWeights {
    q_proj: LinearWeight,
    kv_proj: KvProjWeight,
    o_proj: LinearWeight,
    q_norm: Option<CudaTensor>,
    k_norm: Option<CudaTensor>,
}

struct GemmaMlpWeights {
    gate_up: GateUpWeight,
    down_proj: LinearWeight,
}

struct GemmaLayerWeights {
    input_layernorm: CudaTensor,
    post_attention_layernorm: CudaTensor,
    pre_feedforward_layernorm: CudaTensor,
    post_feedforward_layernorm: CudaTensor,
    attention: GemmaAttentionWeights,
    mlp: GemmaMlpWeights,
}

/// Gemma model supporting both Gemma 2 and Gemma 3 text architectures.
pub struct GemmaModel<B: Backend> {
    config: GemmaConfig,
    ctx: CudaContext,
    #[allow(dead_code)]
    gpu_config: GpuConfig,

    #[cfg(feature = "nccl")]
    nccl_comm: Option<NcclCommunicator>,

    tp_num_heads: usize,
    tp_num_kv_heads: usize,
    dtype: DType,

    embed_tokens: B::Tensor,
    layers: Vec<GemmaLayerWeights>,
    norm: B::Tensor,
    lm_head: LinearWeight,

    /// Embedding scale factor: sqrt(hidden_size)
    embed_scale: f32,

    /// Attention scale: 1 / sqrt(query_pre_attn_scalar)
    attn_scale: f32,

    // RoPE caches — Gemma 2: single set, Gemma 3: two sets (local + global)
    cos_cache: B::Tensor,
    sin_cache: B::Tensor,
    // Gemma 3 dual-theta RoPE: separate cache for full-attention layers
    cos_cache_global: Option<B::Tensor>,
    sin_cache_global: Option<B::Tensor>,

    _backend: PhantomData<B>,
}

impl GemmaModel<CudaBackend> {
    /// Load a Gemma model from a directory containing SafeTensors and config.json
    ///
    /// # Errors
    /// Returns an error if loading fails
    pub fn from_pretrained(ctx: &CudaContext, model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();
        let config_path = model_path.join("config.json");
        let config = GemmaConfig::from_json(&config_path);
        let loader = SafeTensorsLoader::from_directory(model_path)?;
        Self::load_weights(ctx, config, &loader)
    }

    /// Load a Gemma model with tensor-parallel sharding across multiple GPUs.
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
        let config = GemmaConfig::from_json(&config_path);
        let loader = SafeTensorsLoader::from_directory(model_path)?;
        Self::load_weights_sharded(ctx, config, &loader, gpu_config, nccl_comm)
    }

    #[allow(clippy::too_many_lines)]
    fn load_weights(
        ctx: &CudaContext,
        config: GemmaConfig,
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

            // Load attention weights
            let k = load_linear(
                dtype,
                ctx,
                loader,
                &format!("{prefix}.self_attn.k_proj.weight"),
                qc,
            )?;
            let v = load_linear(
                dtype,
                ctx,
                loader,
                &format!("{prefix}.self_attn.v_proj.weight"),
                qc,
            )?;
            let kv_dim = config.num_key_value_heads * config.head_dim;
            let kv_proj = match (k, v) {
                (LinearWeight::Dense(k_w), LinearWeight::Dense(v_w)) => KvProjWeight::Fused {
                    kv_dim,
                    weight: concat_weights(&k_w, &v_w)?,
                },
                (k, v) => KvProjWeight::Separate {
                    k_proj: Box::new(k),
                    v_proj: Box::new(v),
                },
            };

            // Load QK-norm weights (Gemma 3 only)
            let q_norm_name = format!("{prefix}.self_attn.q_norm.weight");
            let k_norm_name = format!("{prefix}.self_attn.k_norm.weight");
            let q_norm = if loader.contains(&q_norm_name) {
                Some(load_gemma_norm(dtype, loader, ctx, &q_norm_name)?)
            } else {
                None
            };
            let k_norm = if loader.contains(&k_norm_name) {
                Some(load_gemma_norm(dtype, loader, ctx, &k_norm_name)?)
            } else {
                None
            };

            // Load MLP weights (GeGLU: gate_proj, up_proj, down_proj)
            let gate = load_linear(
                dtype,
                ctx,
                loader,
                &format!("{prefix}.mlp.gate_proj.weight"),
                qc,
            )?;
            let up = load_linear(
                dtype,
                ctx,
                loader,
                &format!("{prefix}.mlp.up_proj.weight"),
                qc,
            )?;
            let gate_up = match (gate, up) {
                (LinearWeight::Dense(g), LinearWeight::Dense(u)) => GateUpWeight::Fused {
                    weight: concat_weights(&g, &u)?,
                    intermediate_size: config.intermediate_size,
                },
                (g, u) => GateUpWeight::Separate {
                    gate_proj: Box::new(g),
                    up_proj: Box::new(u),
                },
            };

            let layer = GemmaLayerWeights {
                input_layernorm: load_gemma_norm(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.input_layernorm.weight"),
                )?,
                post_attention_layernorm: load_gemma_norm(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                pre_feedforward_layernorm: load_gemma_norm(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.pre_feedforward_layernorm.weight"),
                )?,
                post_feedforward_layernorm: load_gemma_norm(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.post_feedforward_layernorm.weight"),
                )?,
                attention: GemmaAttentionWeights {
                    q_proj: load_linear(
                        dtype,
                        ctx,
                        loader,
                        &format!("{prefix}.self_attn.q_proj.weight"),
                        qc,
                    )?,
                    kv_proj,
                    o_proj: load_linear(
                        dtype,
                        ctx,
                        loader,
                        &format!("{prefix}.self_attn.o_proj.weight"),
                        qc,
                    )?,
                    q_norm,
                    k_norm,
                },
                mlp: GemmaMlpWeights {
                    gate_up,
                    down_proj: load_linear(
                        dtype,
                        ctx,
                        loader,
                        &format!("{prefix}.mlp.down_proj.weight"),
                        qc,
                    )?,
                },
            };

            layers.push(layer);
        }

        let norm = load_gemma_norm(dtype, loader, ctx, "model.norm.weight")?;

        // Tied word embeddings — Gemma always ties lm_head to embed_tokens
        let lm_head = if qc.is_some() {
            let embed_f32 = CudaBackend::cast_to_f32(&embed_tokens)?;
            let data = embed_f32.to_vec::<f32>()?;
            LinearWeight::Quantized(QuantizedTensor::from_f32_as_q8(
                ctx,
                embed_f32.shape(),
                &data,
            )?)
        } else {
            let embed_f32 = CudaBackend::cast_to_f32(&embed_tokens)?;
            let transposed = pretranspose_weight(&embed_f32)?;
            LinearWeight::Dense(cast_from_f32(&transposed, dtype)?)
        };

        // Precompute RoPE caches
        // For Gemma 2: single cache with rope_theta
        // For Gemma 3: two caches — local (rope_local_base_freq) and global (rope_theta)
        let local_theta = config.rope_local_base_freq.unwrap_or(config.rope_theta);
        let (cos_f32, sin_f32) = precompute_rope_cache(
            ctx,
            config.max_position_embeddings,
            config.head_dim,
            local_theta,
        )?;

        let (cos_cache, sin_cache) = convert_rope_cache(dtype, &cos_f32, &sin_f32)?;

        let (cos_cache_global, sin_cache_global) = if config.rope_local_base_freq.is_some() {
            let (cos_g_f32, sin_g_f32) = precompute_rope_cache(
                ctx,
                config.max_position_embeddings,
                config.head_dim,
                config.rope_theta,
            )?;
            let (cos_g, sin_g) = convert_rope_cache(dtype, &cos_g_f32, &sin_g_f32)?;
            (Some(cos_g), Some(sin_g))
        } else {
            (None, None)
        };

        let embed_scale = (config.hidden_size as f32).sqrt();
        let attn_scale = config.attn_scale();

        Ok(Self {
            tp_num_heads: config.num_attention_heads,
            tp_num_kv_heads: config.num_key_value_heads,
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
            embed_scale,
            attn_scale,
            cos_cache,
            sin_cache,
            cos_cache_global,
            sin_cache_global,
            _backend: PhantomData,
        })
    }

    // --- Sharded loading ---

    #[cfg(feature = "nccl")]
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    fn load_weights_sharded(
        ctx: &CudaContext,
        config: GemmaConfig,
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
                let mut qt = loader.load_quantized_sharded(ctx, name, shard, strategy)?;
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

        let shard = match gpu_config {
            GpuConfig::Sharded(s) => s,
            GpuConfig::Single => panic!("load_weights_sharded requires GpuConfig::Sharded"),
        };

        let qc = config.quantization_config.as_ref();
        let world_size = shard.world_size;

        assert!(
            config.num_attention_heads.is_multiple_of(world_size),
            "num_attention_heads ({}) must be divisible by world_size ({world_size})",
            config.num_attention_heads
        );
        assert!(
            config.num_key_value_heads.is_multiple_of(world_size),
            "num_key_value_heads ({}) must be divisible by world_size ({world_size})",
            config.num_key_value_heads
        );

        let tp_num_heads = config.num_attention_heads / world_size;
        let tp_num_kv_heads = config.num_key_value_heads / world_size;

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

            let k_name = format!("{prefix}.self_attn.k_proj.weight");
            let k = load_linear_sharded(
                dtype,
                ctx,
                loader,
                &k_name,
                &shard,
                shard_strategy_for_weight(&k_name),
                qc,
            )?;
            let v_name = format!("{prefix}.self_attn.v_proj.weight");
            let v = load_linear_sharded(
                dtype,
                ctx,
                loader,
                &v_name,
                &shard,
                shard_strategy_for_weight(&v_name),
                qc,
            )?;
            let kv_dim = tp_num_kv_heads * config.head_dim;
            let kv_proj = match (k, v) {
                (LinearWeight::Dense(k_w), LinearWeight::Dense(v_w)) => KvProjWeight::Fused {
                    kv_dim,
                    weight: concat_weights(&k_w, &v_w)?,
                },
                (k, v) => KvProjWeight::Separate {
                    k_proj: Box::new(k),
                    v_proj: Box::new(v),
                },
            };

            // QK-norm weights are per-head, not sharded
            let q_norm_name = format!("{prefix}.self_attn.q_norm.weight");
            let k_norm_name = format!("{prefix}.self_attn.k_norm.weight");
            let q_norm = if loader.contains(&q_norm_name) {
                Some(load_gemma_norm(dtype, loader, ctx, &q_norm_name)?)
            } else {
                None
            };
            let k_norm = if loader.contains(&k_norm_name) {
                Some(load_gemma_norm(dtype, loader, ctx, &k_norm_name)?)
            } else {
                None
            };

            let gate_name = format!("{prefix}.mlp.gate_proj.weight");
            let gate = load_linear_sharded(
                dtype,
                ctx,
                loader,
                &gate_name,
                &shard,
                shard_strategy_for_weight(&gate_name),
                qc,
            )?;
            let up_name = format!("{prefix}.mlp.up_proj.weight");
            let up = load_linear_sharded(
                dtype,
                ctx,
                loader,
                &up_name,
                &shard,
                shard_strategy_for_weight(&up_name),
                qc,
            )?;
            let tp_intermediate = config.intermediate_size / world_size;
            let gate_up = match (gate, up) {
                (LinearWeight::Dense(g), LinearWeight::Dense(u)) => GateUpWeight::Fused {
                    weight: concat_weights(&g, &u)?,
                    intermediate_size: tp_intermediate,
                },
                (g, u) => GateUpWeight::Separate {
                    gate_proj: Box::new(g),
                    up_proj: Box::new(u),
                },
            };

            let layer = GemmaLayerWeights {
                input_layernorm: load_gemma_norm(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.input_layernorm.weight"),
                )?,
                post_attention_layernorm: load_gemma_norm(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                pre_feedforward_layernorm: load_gemma_norm(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.pre_feedforward_layernorm.weight"),
                )?,
                post_feedforward_layernorm: load_gemma_norm(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.post_feedforward_layernorm.weight"),
                )?,
                attention: GemmaAttentionWeights {
                    q_proj: {
                        let name = format!("{prefix}.self_attn.q_proj.weight");
                        load_linear_sharded(
                            dtype,
                            ctx,
                            loader,
                            &name,
                            &shard,
                            shard_strategy_for_weight(&name),
                            qc,
                        )?
                    },
                    kv_proj,
                    o_proj: {
                        let name = format!("{prefix}.self_attn.o_proj.weight");
                        load_linear_sharded(
                            dtype,
                            ctx,
                            loader,
                            &name,
                            &shard,
                            shard_strategy_for_weight(&name),
                            qc,
                        )?
                    },
                    q_norm,
                    k_norm,
                },
                mlp: GemmaMlpWeights {
                    gate_up,
                    down_proj: {
                        let name = format!("{prefix}.mlp.down_proj.weight");
                        load_linear_sharded(
                            dtype,
                            ctx,
                            loader,
                            &name,
                            &shard,
                            shard_strategy_for_weight(&name),
                            qc,
                        )?
                    },
                },
            };

            layers.push(layer);
        }

        let norm = load_gemma_norm(dtype, loader, ctx, "model.norm.weight")?;

        // Tied embeddings — not sharded for lm_head
        let lm_head = if qc.is_some() {
            let embed_f32 = CudaBackend::cast_to_f32(&embed_tokens)?;
            let data = embed_f32.to_vec::<f32>()?;
            LinearWeight::Quantized(QuantizedTensor::from_f32_as_q8(
                ctx,
                embed_f32.shape(),
                &data,
            )?)
        } else {
            let embed_f32 = CudaBackend::cast_to_f32(&embed_tokens)?;
            let transposed = pretranspose_weight(&embed_f32)?;
            LinearWeight::Dense(cast_from_f32(&transposed, dtype)?)
        };

        let local_theta = config.rope_local_base_freq.unwrap_or(config.rope_theta);
        let (cos_f32, sin_f32) = precompute_rope_cache(
            ctx,
            config.max_position_embeddings,
            config.head_dim,
            local_theta,
        )?;
        let (cos_cache, sin_cache) = convert_rope_cache(dtype, &cos_f32, &sin_f32)?;

        let (cos_cache_global, sin_cache_global) = if config.rope_local_base_freq.is_some() {
            let (cos_g_f32, sin_g_f32) = precompute_rope_cache(
                ctx,
                config.max_position_embeddings,
                config.head_dim,
                config.rope_theta,
            )?;
            let (cos_g, sin_g) = convert_rope_cache(dtype, &cos_g_f32, &sin_g_f32)?;
            (Some(cos_g), Some(sin_g))
        } else {
            (None, None)
        };

        let embed_scale = (config.hidden_size as f32).sqrt();
        let attn_scale = config.attn_scale();

        Ok(Self {
            tp_num_heads,
            tp_num_kv_heads,
            dtype,
            config,
            ctx: ctx.clone(),
            gpu_config: GpuConfig::Sharded(shard),
            #[cfg(feature = "nccl")]
            nccl_comm: Some(nccl_comm),
            embed_tokens,
            layers,
            norm,
            lm_head,
            embed_scale,
            attn_scale,
            cos_cache,
            sin_cache,
            cos_cache_global,
            sin_cache_global,
            _backend: PhantomData,
        })
    }

    // --- Forward pass ---

    // --- Paged KV cache methods ---

    /// Batched decode forward pass with paged KV cache.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward_batch_decode(
        &self,
        token_ids: &[u32],
        paged_kvs: &mut [PagedKvCache],
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<CudaTensor> {
        let batch_size = token_ids.len();
        let paged_kv = &mut paged_kvs[0];

        let mut hidden = self.embed(token_ids)?;
        CudaBackend::scale_inplace(&mut hidden, self.embed_scale)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer_paged_decode(
                &hidden,
                layer,
                layer_idx,
                paged_kv,
                block_tables,
                positions,
            )?;
        }

        CudaBackend::rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;
        self.lm_head_forward(&hidden.reshape(&[batch_size, self.config.hidden_size]))
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

        let mut hidden = embedding_gather_from_device(
            &self.ctx,
            &self.embed_tokens,
            graph_inputs.token_ids(),
            batch_size,
        )?;
        CudaBackend::scale_inplace(&mut hidden, self.embed_scale)?;

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

        CudaBackend::rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;
        self.lm_head_forward(&hidden.reshape(&[batch_size, self.config.hidden_size]))
    }

    /// Single-sequence prefill with paged KV cache.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward_prefill_paged(
        &self,
        input_ids: &[u32],
        paged_kvs: &mut [PagedKvCache],
        block_table: &BlockTable,
        start_pos: usize,
    ) -> Result<CudaTensor> {
        let seq_len = input_ids.len();
        let paged_kv = &mut paged_kvs[0];

        let mut hidden = self.embed(input_ids)?;
        CudaBackend::scale_inplace(&mut hidden, self.embed_scale)?;

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

        CudaBackend::rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;
        let last_hidden = self.extract_last_row(&hidden, seq_len);
        self.lm_head_forward(&last_hidden)
    }

    fn forward_layer_paged_decode(
        &self,
        hidden: &CudaTensor,
        layer: &GemmaLayerWeights,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache,
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<CudaTensor> {
        let normed =
            CudaBackend::rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

        let attn_output = self.forward_attention_paged_decode(
            &normed,
            &layer.attention,
            layer_idx,
            paged_kv,
            block_tables,
            positions,
        )?;

        let post_attn = CudaBackend::rms_norm(
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;
        let mut hidden = hidden.clone();
        CudaBackend::add_inplace(&mut hidden, &post_attn)?;

        let normed_ffn = CudaBackend::rms_norm(
            &hidden,
            &layer.pre_feedforward_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_mlp(&normed_ffn, &layer.mlp)?;

        let post_ffn = CudaBackend::rms_norm(
            &mlp_output,
            &layer.post_feedforward_layernorm,
            self.config.rms_norm_eps,
        )?;
        CudaBackend::add_inplace(&mut hidden, &post_ffn)?;

        Ok(hidden)
    }

    fn forward_attention_paged_decode(
        &self,
        hidden: &CudaTensor,
        weights: &GemmaAttentionWeights,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache,
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<CudaTensor> {
        let batch_size = hidden.shape()[0];
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim;

        let q = gemma_linear(hidden, &weights.q_proj)?;
        let (k, v) = match &weights.kv_proj {
            KvProjWeight::Fused { weight, kv_dim } => {
                let kv = matmul(hidden, weight)?;
                if batch_size == 1 {
                    let k = kv.slice_view(0, &[1, *kv_dim]);
                    let v = kv.slice_view(*kv_dim, &[1, *kv_dim]);
                    (k, v)
                } else {
                    CudaBackend::split_inner_dim(&kv, *kv_dim, *kv_dim)?
                }
            }
            KvProjWeight::Separate { k_proj, v_proj } => {
                let k = gemma_linear(hidden, k_proj)?;
                let v = gemma_linear(hidden, v_proj)?;
                (k, v)
            }
        };

        let mut q = q.reshape(&[batch_size, num_heads, head_dim]);
        let mut k = k.reshape(&[batch_size, num_kv_heads, head_dim]);
        let v = v.reshape(&[batch_size, num_kv_heads, head_dim]);

        if let Some(ref q_norm_w) = weights.q_norm {
            let flat_q = q.reshape(&[batch_size * num_heads, head_dim]);
            let normed_q = CudaBackend::rms_norm(&flat_q, q_norm_w, self.config.rms_norm_eps)?;
            q = normed_q.reshape(&[batch_size, num_heads, head_dim]);
        }
        if let Some(ref k_norm_w) = weights.k_norm {
            let flat_k = k.reshape(&[batch_size * num_kv_heads, head_dim]);
            let normed_k = CudaBackend::rms_norm(&flat_k, k_norm_w, self.config.rms_norm_eps)?;
            k = normed_k.reshape(&[batch_size, num_kv_heads, head_dim]);
        }

        let (cos, sin) = self.rope_caches_for_layer(layer_idx);
        let q = CudaBackend::apply_rope_batched(&q, cos, sin, positions)?;
        let k = CudaBackend::apply_rope_batched(&k, cos, sin, positions)?;

        let q_stride = num_heads * head_dim;
        let kv_stride = num_kv_heads * head_dim;
        let sliding_window = self.config.effective_sliding_window(layer_idx);

        let mut attn_parts = Vec::with_capacity(batch_size);
        for (i, &pos) in positions.iter().enumerate() {
            let q_i = q.slice_view(i * q_stride, &[1, num_heads, head_dim]);
            let k_i = k.slice_view(i * kv_stride, &[1, num_kv_heads, head_dim]);
            let v_i = v.slice_view(i * kv_stride, &[1, num_kv_heads, head_dim]);

            CudaBackend::append_paged(paged_kv, layer_idx, &block_tables[i], &k_i, &v_i, pos)?;

            let mut table_with_current = block_tables[i].clone();
            table_with_current.advance(1);

            let (k_pool, v_pool) = CudaBackend::get_pools(paged_kv, layer_idx);
            let attn_i = CudaBackend::paged_attention_decode(
                &q_i,
                k_pool,
                v_pool,
                &[table_with_current],
                CudaBackend::block_size(paged_kv),
                None,
                self.config.attn_logit_softcapping,
                sliding_window,
            )?;

            attn_parts.push(attn_i.reshape(&[1, num_heads * head_dim]));
        }

        let attn_output = CudaBackend::concat_rows(&attn_parts)?;

        let mut out = gemma_linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_paged_decode_indirect(
        &self,
        hidden: &CudaTensor,
        layer: &GemmaLayerWeights,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache,
        graph_inputs: &BatchedGraphInputs,
        max_seq_len: usize,
    ) -> Result<CudaTensor> {
        let normed =
            CudaBackend::rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

        let attn_output = self.forward_attention_paged_decode_indirect(
            &normed,
            &layer.attention,
            layer_idx,
            paged_kv,
            graph_inputs,
            max_seq_len,
        )?;

        let post_attn = CudaBackend::rms_norm(
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;
        let mut hidden = hidden.clone();
        CudaBackend::add_inplace(&mut hidden, &post_attn)?;

        let normed_ffn = CudaBackend::rms_norm(
            &hidden,
            &layer.pre_feedforward_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_mlp(&normed_ffn, &layer.mlp)?;

        let post_ffn = CudaBackend::rms_norm(
            &mlp_output,
            &layer.post_feedforward_layernorm,
            self.config.rms_norm_eps,
        )?;
        CudaBackend::add_inplace(&mut hidden, &post_ffn)?;

        Ok(hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attention_paged_decode_indirect(
        &self,
        hidden: &CudaTensor,
        weights: &GemmaAttentionWeights,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache,
        graph_inputs: &BatchedGraphInputs,
        max_seq_len: usize,
    ) -> Result<CudaTensor> {
        let batch_size = hidden.shape()[0];
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim;

        let q = gemma_linear(hidden, &weights.q_proj)?;
        let (k, v) = match &weights.kv_proj {
            KvProjWeight::Fused { weight, kv_dim } => {
                let kv = matmul(hidden, weight)?;
                if batch_size == 1 {
                    let k = kv.slice_view(0, &[1, *kv_dim]);
                    let v = kv.slice_view(*kv_dim, &[1, *kv_dim]);
                    (k, v)
                } else {
                    CudaBackend::split_inner_dim(&kv, *kv_dim, *kv_dim)?
                }
            }
            KvProjWeight::Separate { k_proj, v_proj } => {
                let k = gemma_linear(hidden, k_proj)?;
                let v = gemma_linear(hidden, v_proj)?;
                (k, v)
            }
        };

        let mut q = q.reshape(&[batch_size, num_heads, head_dim]);
        let mut k = k.reshape(&[batch_size, num_kv_heads, head_dim]);
        let v = v.reshape(&[batch_size, num_kv_heads, head_dim]);

        if let Some(ref q_norm_w) = weights.q_norm {
            let flat_q = q.reshape(&[batch_size * num_heads, head_dim]);
            let normed_q = CudaBackend::rms_norm(&flat_q, q_norm_w, self.config.rms_norm_eps)?;
            q = normed_q.reshape(&[batch_size, num_heads, head_dim]);
        }
        if let Some(ref k_norm_w) = weights.k_norm {
            let flat_k = k.reshape(&[batch_size * num_kv_heads, head_dim]);
            let normed_k = CudaBackend::rms_norm(&flat_k, k_norm_w, self.config.rms_norm_eps)?;
            k = normed_k.reshape(&[batch_size, num_kv_heads, head_dim]);
        }

        let (cos, sin) = self.rope_caches_for_layer(layer_idx);
        let q = apply_rope_batched_indirect(&q, cos, sin, graph_inputs.positions(), batch_size)?;
        let k = apply_rope_batched_indirect(&k, cos, sin, graph_inputs.positions(), batch_size)?;

        paged_kv.append_paged_batched(
            layer_idx,
            &k,
            &v,
            graph_inputs.block_tables(),
            graph_inputs.positions(),
            batch_size,
            graph_inputs.max_blocks_per_seq(),
        )?;

        let (k_pool, v_pool) = CudaBackend::get_pools(paged_kv, layer_idx);
        let sliding_window = self.config.effective_sliding_window(layer_idx);
        let attn_output = paged_attention_decode_indirect(
            &self.ctx,
            &q,
            k_pool,
            v_pool,
            graph_inputs.block_tables(),
            graph_inputs.seq_lens(),
            CudaBackend::block_size(paged_kv),
            graph_inputs.max_blocks_per_seq(),
            max_seq_len,
            None,
            self.config.attn_logit_softcapping,
            sliding_window,
        )?;

        let attn_output = attn_output.reshape(&[batch_size, num_heads * head_dim]);
        let mut out = gemma_linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_paged_prefill(
        &self,
        hidden: &CudaTensor,
        layer: &GemmaLayerWeights,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache,
        block_table: &BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<CudaTensor> {
        let normed =
            CudaBackend::rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

        let attn_output = self.forward_attention_paged_prefill(
            &normed,
            &layer.attention,
            layer_idx,
            paged_kv,
            block_table,
            start_pos,
            seq_len,
        )?;

        let post_attn = CudaBackend::rms_norm(
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;
        let mut hidden = hidden.clone();
        CudaBackend::add_inplace(&mut hidden, &post_attn)?;

        let normed_ffn = CudaBackend::rms_norm(
            &hidden,
            &layer.pre_feedforward_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_mlp(&normed_ffn, &layer.mlp)?;

        let post_ffn = CudaBackend::rms_norm(
            &mlp_output,
            &layer.post_feedforward_layernorm,
            self.config.rms_norm_eps,
        )?;
        CudaBackend::add_inplace(&mut hidden, &post_ffn)?;

        Ok(hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attention_paged_prefill(
        &self,
        hidden: &CudaTensor,
        weights: &GemmaAttentionWeights,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache,
        block_table: &BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<CudaTensor> {
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim;

        let q = gemma_linear(hidden, &weights.q_proj)?;
        let (k, v) = match &weights.kv_proj {
            KvProjWeight::Fused { weight, kv_dim } => {
                let kv = matmul(hidden, weight)?;
                CudaBackend::split_inner_dim(&kv, *kv_dim, *kv_dim)?
            }
            KvProjWeight::Separate { k_proj, v_proj } => {
                let k = gemma_linear(hidden, k_proj)?;
                let v = gemma_linear(hidden, v_proj)?;
                (k, v)
            }
        };

        let mut q = q.reshape(&[seq_len, num_heads, head_dim]);
        let mut k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
        let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

        if let Some(ref q_norm_w) = weights.q_norm {
            let flat_q = q.reshape(&[seq_len * num_heads, head_dim]);
            let normed_q = CudaBackend::rms_norm(&flat_q, q_norm_w, self.config.rms_norm_eps)?;
            q = normed_q.reshape(&[seq_len, num_heads, head_dim]);
        }
        if let Some(ref k_norm_w) = weights.k_norm {
            let flat_k = k.reshape(&[seq_len * num_kv_heads, head_dim]);
            let normed_k = CudaBackend::rms_norm(&flat_k, k_norm_w, self.config.rms_norm_eps)?;
            k = normed_k.reshape(&[seq_len, num_kv_heads, head_dim]);
        }

        let (cos, sin) = self.rope_caches_for_layer(layer_idx);
        let q = CudaBackend::apply_rope(&q, cos, sin, start_pos)?;
        let k = CudaBackend::apply_rope(&k, cos, sin, start_pos)?;

        CudaBackend::append_paged(paged_kv, layer_idx, block_table, &k, &v, start_pos)?;

        let mut gather_table = block_table.clone();
        gather_table.advance(seq_len);
        let (k_contig, v_contig) =
            CudaBackend::gather_paged_kv(paged_kv, layer_idx, &gather_table)?;

        let sliding_window = self.config.effective_sliding_window(layer_idx);
        let attn_output = CudaBackend::fused_attention_prefill(
            &q,
            &k_contig,
            &v_contig,
            start_pos,
            Some(self.attn_scale),
            self.config.attn_logit_softcapping,
            sliding_window,
        )?;

        let attn_output = attn_output.reshape(&[seq_len, num_heads * head_dim]);
        let mut out = gemma_linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    fn rope_caches_for_layer(&self, layer_idx: usize) -> (&CudaTensor, &CudaTensor) {
        // Gemma 3: full-attention layers use the global RoPE cache
        if let (Some(ref cos_g), Some(ref sin_g)) = (&self.cos_cache_global, &self.sin_cache_global)
        {
            if self.config.effective_sliding_window(layer_idx).is_none() {
                return (cos_g, sin_g);
            }
        }
        (&self.cos_cache, &self.sin_cache)
    }

    #[allow(clippy::unused_self)]
    fn forward_mlp(&self, hidden: &CudaTensor, weights: &GemmaMlpWeights) -> Result<CudaTensor> {
        let (gate, up) = match &weights.gate_up {
            GateUpWeight::Fused {
                weight,
                intermediate_size,
            } => {
                let seq_len = hidden.shape()[0];
                let gate_up = matmul(hidden, weight)?;
                if seq_len == 1 {
                    let gate = gate_up.slice_view(0, &[1, *intermediate_size]);
                    let up = gate_up.slice_view(*intermediate_size, &[1, *intermediate_size]);
                    (gate, up)
                } else {
                    CudaBackend::split_inner_dim(&gate_up, *intermediate_size, *intermediate_size)?
                }
            }
            GateUpWeight::Separate { gate_proj, up_proj } => {
                let gate = gemma_linear(hidden, gate_proj)?;
                let up = gemma_linear(hidden, up_proj)?;
                (gate, up)
            }
        };
        // GeGLU: gelu(gate) * up
        let intermediate = CudaBackend::geglu(&gate, &up)?;
        let mut out = gemma_linear(&intermediate, &weights.down_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    #[allow(clippy::unused_self)]
    fn extract_last_row(&self, hidden: &CudaTensor, seq_len: usize) -> CudaTensor {
        let hidden_size = hidden.shape()[1];
        if seq_len == 1 {
            return hidden.reshape(&[1, hidden_size]);
        }
        hidden.slice_view((seq_len - 1) * hidden_size, &[1, hidden_size])
    }

    fn embed(&self, input_ids: &[u32]) -> Result<CudaTensor> {
        CudaBackend::embedding_gather(&self.embed_tokens, input_ids)
    }

    fn lm_head_forward(&self, hidden: &CudaTensor) -> Result<CudaTensor> {
        // bf16 fast path
        if self.dtype == DType::BF16 {
            if let LinearWeight::Dense(w) = &self.lm_head {
                let mut logits = CudaBackend::matmul_bf16_f32(hidden, w)?;
                self.apply_final_softcap(&mut logits)?;
                return Ok(logits);
            }
        }
        let logits_t = gemma_linear(hidden, &self.lm_head)?;
        let mut logits = if self.dtype == DType::F32 {
            logits_t
        } else {
            CudaBackend::cast_to_f32(&logits_t)?
        };
        self.apply_final_softcap(&mut logits)?;
        Ok(logits)
    }

    /// Apply final logit soft-capping (Gemma 2 only): tanh(logits / cap) * cap
    fn apply_final_softcap(&self, logits: &mut CudaTensor) -> Result<()> {
        if let Some(cap) = self.config.final_logit_softcapping {
            let data: Vec<f32> = logits.to_vec::<f32>()?;
            let capped: Vec<f32> = data.iter().map(|&x| (x / cap).tanh() * cap).collect();
            *logits = CudaTensor::from_slice(&self.ctx, logits.shape(), &capped)?;
        }
        Ok(())
    }

    /// Access the model configuration
    #[must_use]
    pub fn config(&self) -> &GemmaConfig {
        &self.config
    }

    /// Get the model's compute dtype
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Convert f32 RoPE caches to model dtype
fn convert_rope_cache(
    dtype: DType,
    cos_f32: &CudaTensor,
    sin_f32: &CudaTensor,
) -> Result<(CudaTensor, CudaTensor)> {
    Ok((
        cast_from_f32(cos_f32, dtype)?,
        cast_from_f32(sin_f32, dtype)?,
    ))
}

/// Linear projection that handles both dense and quantized weights.
///
/// Quantized weights are computed in f32 and cast back to the input dtype.
fn gemma_linear(input: &CudaTensor, weight: &LinearWeight) -> Result<CudaTensor> {
    match weight {
        LinearWeight::Dense(w) => matmul(input, w),
        LinearWeight::Quantized(w) => {
            let input_f32 = if input.dtype() == DType::F32 {
                input.slice_view(0, input.shape())
            } else {
                CudaBackend::cast_to_f32(input)?
            };
            let output_f32 = quantized_matmul(&input_f32, w)?;
            if input.dtype() == DType::F32 {
                Ok(output_f32)
            } else {
                cast_from_f32(&output_f32, input.dtype())
            }
        }
    }
}

// --- Model trait implementation ---

#[cfg(feature = "nccl")]
impl infernum_cuda::ShardedLoadable for GemmaModel<CudaBackend> {
    fn load_shard(
        ctx: &CudaContext,
        model_path: &Path,
        shard: ShardConfig,
        comm: NcclCommunicator,
    ) -> Result<Self> {
        Self::from_pretrained_sharded(ctx, model_path, GpuConfig::Sharded(shard), comm)
    }
}

// --- Public helpers & infernum::Model implementation ---

#[allow(private_bounds)]
impl GemmaModel<CudaBackend> {
    /// Build the runtime-facing [`ModelConfig`](infernum::ModelConfig).
    #[must_use]
    pub fn model_config(&self) -> infernum::ModelConfig {
        infernum::ModelConfig {
            num_layers: self.config.num_hidden_layers,
            max_seq_len: self.config.max_position_embeddings,
            num_kv_heads: self.tp_num_kv_heads,
            head_dim: self.config.head_dim,
            eos_token_id: self.config.eos_token_id,
            cache_dtype: self.dtype,
        }
    }

    /// Full forward pass without KV cache (recomputes everything).
    ///
    /// Returns raw logits as a [`CudaTensor`] of shape `(seq_len, vocab_size)`.
    ///
    /// # Errors
    /// Returns an error if any GPU operation fails.
    pub fn forward_full(&self, input_ids: &[u32]) -> Result<CudaTensor> {
        let seq_len = input_ids.len();
        let mut hidden = self.embed(input_ids)?;
        CudaBackend::scale_inplace(&mut hidden, self.embed_scale)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let normed =
                CudaBackend::rms_norm(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            let num_heads = self.tp_num_heads;
            let num_kv_heads = self.tp_num_kv_heads;
            let head_dim = self.config.head_dim;
            let sliding_window = self.config.effective_sliding_window(layer_idx);

            let q = gemma_linear(&normed, &layer.attention.q_proj)?;
            let (k, v) = match &layer.attention.kv_proj {
                KvProjWeight::Fused { weight, kv_dim } => {
                    let kv = matmul(&normed, weight)?;
                    CudaBackend::split_inner_dim(&kv, *kv_dim, *kv_dim)?
                }
                KvProjWeight::Separate { k_proj, v_proj } => {
                    let k = gemma_linear(&normed, k_proj)?;
                    let v = gemma_linear(&normed, v_proj)?;
                    (k, v)
                }
            };

            let mut q = q.reshape(&[seq_len, num_heads, head_dim]);
            let mut k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
            let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

            if let Some(ref q_norm_w) = layer.attention.q_norm {
                let flat_q = q.reshape(&[seq_len * num_heads, head_dim]);
                let normed_q = CudaBackend::rms_norm(&flat_q, q_norm_w, self.config.rms_norm_eps)?;
                q = normed_q.reshape(&[seq_len, num_heads, head_dim]);
            }
            if let Some(ref k_norm_w) = layer.attention.k_norm {
                let flat_k = k.reshape(&[seq_len * num_kv_heads, head_dim]);
                let normed_k = CudaBackend::rms_norm(&flat_k, k_norm_w, self.config.rms_norm_eps)?;
                k = normed_k.reshape(&[seq_len, num_kv_heads, head_dim]);
            }

            let (cos, sin) = self.rope_caches_for_layer(layer_idx);
            let q = CudaBackend::apply_rope(&q, cos, sin, 0)?;
            let k = CudaBackend::apply_rope(&k, cos, sin, 0)?;

            let attn_output = CudaBackend::fused_attention_prefill(
                &q,
                &k,
                &v,
                0,
                Some(self.attn_scale),
                self.config.attn_logit_softcapping,
                sliding_window,
            )?;
            let attn_output = attn_output.reshape(&[seq_len, num_heads * head_dim]);
            let mut attn_output = gemma_linear(&attn_output, &layer.attention.o_proj)?;
            #[cfg(feature = "nccl")]
            nccl_all_reduce(self.nccl_comm.as_ref(), &mut attn_output)?;

            // Post-attention norm + residual
            let post_attn = CudaBackend::rms_norm(
                &attn_output,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;
            CudaBackend::add_inplace(&mut hidden, &post_attn)?;

            // Pre-feedforward norm
            let normed_ffn = CudaBackend::rms_norm(
                &hidden,
                &layer.pre_feedforward_layernorm,
                self.config.rms_norm_eps,
            )?;

            let mlp_output = self.forward_mlp(&normed_ffn, &layer.mlp)?;

            // Post-feedforward norm + residual
            let post_ffn = CudaBackend::rms_norm(
                &mlp_output,
                &layer.post_feedforward_layernorm,
                self.config.rms_norm_eps,
            )?;
            CudaBackend::add_inplace(&mut hidden, &post_ffn)?;
        }

        CudaBackend::rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;
        self.lm_head_forward(&hidden)
    }
}

#[allow(private_bounds)]
impl infernum::Model for GemmaModel<CudaBackend> {
    type B = CudaBackend;
    type KvCache = PagedKvCache;

    fn config(&self) -> infernum::ModelConfig {
        self.model_config()
    }

    fn allocate_kv_cache(&self, block_config: &infernum::BlockConfig) -> Result<Self::KvCache> {
        let mc = self.model_config();
        PagedKvCache::new(
            &self.ctx,
            mc.num_layers,
            block_config,
            mc.num_kv_heads,
            mc.head_dim,
            mc.cache_dtype,
        )
    }

    fn forward(&self, input_ids: &[u32]) -> Result<infernum_cuda::CudaLogits> {
        Ok(infernum_cuda::CudaLogits::new(
            self.forward_full(input_ids)?,
        ))
    }

    fn forward_prefill(
        &self,
        input_ids: &[u32],
        kv_cache: &mut Self::KvCache,
        _runtime_state: &mut infernum_cuda::CudaRuntimeState,
        block_table: &infernum::BlockTable,
        start_pos: usize,
    ) -> Result<infernum_cuda::CudaLogits> {
        let tensor = self.forward_prefill_paged(
            input_ids,
            std::slice::from_mut(kv_cache),
            block_table,
            start_pos,
        )?;
        Ok(infernum_cuda::CudaLogits::new(tensor))
    }

    fn forward_batch_decode(
        &self,
        token_ids: &[u32],
        kv_cache: &mut Self::KvCache,
        _runtime_state: &mut infernum_cuda::CudaRuntimeState,
        block_tables: &[infernum::BlockTable],
        positions: &[usize],
    ) -> Result<infernum_cuda::CudaLogits> {
        let tensor = self.forward_batch_decode(
            token_ids,
            std::slice::from_mut(kv_cache),
            block_tables,
            positions,
        )?;
        Ok(infernum_cuda::CudaLogits::new(tensor))
    }
}
