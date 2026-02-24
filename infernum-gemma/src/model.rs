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

use std::path::Path;

use infernum::cuda::block_allocator::BlockTable;
use infernum::cuda::ops::{
    add_inplace, apply_rope, apply_rope_batched, apply_rope_batched_indirect, cast_to_f32,
    embedding_gather, embedding_gather_from_device, fused_attention_prefill, gather_paged_kv,
    geglu, matmul, matmul_bf16_f32, paged_attention_decode, paged_attention_decode_indirect,
    precompute_rope_cache, quantized_matmul, rms_norm, rms_norm_inplace, scale_inplace,
    transpose_2d, GemmScalar,
};
use infernum::cuda::{
    BatchedGraphInputs, CudaBlas, CudaContext, CudaTensor, DeviceRepr, Gemm, GpuConfig,
    PagedKvCache, QuantizedTensor, ValidAsZeroBits,
};
#[cfg(feature = "nccl")]
use infernum::cuda::{NcclCommunicator, NcclType, ShardConfig, ShardStrategy};
use infernum::dtype::TensorDType;
use infernum::tensor::Tensor;
use infernum::weights::{SafeTensorsLoader, WeightLoader};
use infernum::Result;

use crate::GemmaConfig;

// --- NCCL conditional trait bounds ---

#[cfg(feature = "nccl")]
trait MaybeNcclType: NcclType {}
#[cfg(feature = "nccl")]
impl<T: NcclType> MaybeNcclType for T {}

#[cfg(not(feature = "nccl"))]
trait MaybeNcclType {}
#[cfg(not(feature = "nccl"))]
impl<T> MaybeNcclType for T {}

#[cfg(feature = "nccl")]
fn nccl_all_reduce<T>(comm: Option<&NcclCommunicator>, tensor: &mut CudaTensor<T>) -> Result<()>
where
    T: TensorDType + DeviceRepr + ValidAsZeroBits + NcclType,
{
    if let Some(comm) = comm {
        comm.all_reduce_sum_inplace(tensor)?;
    }
    Ok(())
}

// --- Weight helpers ---

fn pretranspose_weight(weight: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    transpose_2d(weight)
}

fn concat_weights<T: TensorDType + DeviceRepr + Default>(
    a: &CudaTensor<T>,
    b: &CudaTensor<T>,
) -> Result<CudaTensor<T>> {
    let k = a.shape()[0];
    assert_eq!(k, b.shape()[0], "concat_weights: K dimension mismatch");
    let n1 = a.shape()[1];
    let n2 = b.shape()[1];

    let a_data = a.to_vec()?;
    let b_data = b.to_vec()?;
    let mut out = vec![T::default(); k * (n1 + n2)];
    for row in 0..k {
        out[row * (n1 + n2)..row * (n1 + n2) + n1]
            .copy_from_slice(&a_data[row * n1..(row + 1) * n1]);
        out[row * (n1 + n2) + n1..row * (n1 + n2) + n1 + n2]
            .copy_from_slice(&b_data[row * n2..(row + 1) * n2]);
    }
    CudaTensor::from_slice(a.context(), &[k, n1 + n2], &out)
}

fn split_gate_up<T: TensorDType + DeviceRepr + Default>(
    fused: &CudaTensor<T>,
    intermediate_size: usize,
) -> Result<(CudaTensor<T>, CudaTensor<T>)> {
    let seq_len = fused.shape()[0];
    let data = fused.to_vec()?;
    let stride = 2 * intermediate_size;

    let mut gate_data = vec![T::default(); seq_len * intermediate_size];
    let mut up_data = vec![T::default(); seq_len * intermediate_size];

    for row in 0..seq_len {
        gate_data[row * intermediate_size..(row + 1) * intermediate_size]
            .copy_from_slice(&data[row * stride..row * stride + intermediate_size]);
        up_data[row * intermediate_size..(row + 1) * intermediate_size]
            .copy_from_slice(&data[row * stride + intermediate_size..(row + 1) * stride]);
    }

    let gate = CudaTensor::from_slice(fused.context(), &[seq_len, intermediate_size], &gate_data)?;
    let up = CudaTensor::from_slice(fused.context(), &[seq_len, intermediate_size], &up_data)?;
    Ok((gate, up))
}

fn split_kv<T: TensorDType + DeviceRepr + Default>(
    fused: &CudaTensor<T>,
    kv_dim: usize,
) -> Result<(CudaTensor<T>, CudaTensor<T>)> {
    let seq_len = fused.shape()[0];
    let data = fused.to_vec()?;
    let stride = 2 * kv_dim;

    let mut k_data = vec![T::default(); seq_len * kv_dim];
    let mut v_data = vec![T::default(); seq_len * kv_dim];

    for row in 0..seq_len {
        k_data[row * kv_dim..(row + 1) * kv_dim]
            .copy_from_slice(&data[row * stride..row * stride + kv_dim]);
        v_data[row * kv_dim..(row + 1) * kv_dim]
            .copy_from_slice(&data[row * stride + kv_dim..(row + 1) * stride]);
    }

    let k = CudaTensor::from_slice(fused.context(), &[seq_len, kv_dim], &k_data)?;
    let v = CudaTensor::from_slice(fused.context(), &[seq_len, kv_dim], &v_data)?;
    Ok((k, v))
}

fn load_typed<T: TensorDType + DeviceRepr>(
    loader: &impl WeightLoader,
    ctx: &CudaContext,
    name: &str,
) -> Result<CudaTensor<T>> {
    use infernum::dtype::DType;
    match T::DTYPE {
        DType::F32 => {
            let t = loader.load_f32(ctx, name)?;
            Ok(reinterpret_tensor(t))
        }
        DType::F16 => {
            let t = loader.load_f16(ctx, name)?;
            Ok(reinterpret_tensor(t))
        }
        DType::BF16 => {
            let t = loader.load_bf16(ctx, name)?;
            Ok(reinterpret_tensor(t))
        }
        other => panic!("Unsupported dtype for load_typed: {other}"),
    }
}

#[cfg(feature = "nccl")]
fn load_typed_sharded<T: TensorDType + DeviceRepr>(
    loader: &impl WeightLoader,
    ctx: &CudaContext,
    name: &str,
    shard: &ShardConfig,
    strategy: ShardStrategy,
) -> Result<CudaTensor<T>> {
    use infernum::dtype::DType;
    match T::DTYPE {
        DType::F32 => {
            let t = loader.load_f32_sharded(ctx, name, shard, strategy)?;
            Ok(reinterpret_tensor(t))
        }
        DType::F16 => {
            let t = loader.load_f16_sharded(ctx, name, shard, strategy)?;
            Ok(reinterpret_tensor(t))
        }
        DType::BF16 => {
            let t = loader.load_bf16_sharded(ctx, name, shard, strategy)?;
            Ok(reinterpret_tensor(t))
        }
        other => panic!("Unsupported dtype for load_typed_sharded: {other}"),
    }
}

fn reinterpret_tensor<A: TensorDType + DeviceRepr, B: TensorDType + DeviceRepr>(
    tensor: CudaTensor<A>,
) -> CudaTensor<B> {
    assert_eq!(
        std::mem::size_of::<A>(),
        std::mem::size_of::<B>(),
        "reinterpret_tensor: size mismatch between {} and {}",
        A::DTYPE,
        B::DTYPE,
    );
    unsafe { tensor.reinterpret() }
}

// --- Weight structures ---

enum LinearWeight<T: TensorDType> {
    Dense(CudaTensor<T>),
    Quantized(QuantizedTensor),
}

enum KvProjWeight<T: TensorDType> {
    Fused {
        weight: CudaTensor<T>,
        kv_dim: usize,
    },
    Separate {
        k_proj: Box<LinearWeight<T>>,
        v_proj: Box<LinearWeight<T>>,
    },
}

enum GateUpWeight<T: TensorDType> {
    Fused {
        weight: CudaTensor<T>,
        intermediate_size: usize,
    },
    Separate {
        gate_proj: Box<LinearWeight<T>>,
        up_proj: Box<LinearWeight<T>>,
    },
}

struct GemmaAttentionWeights<T: TensorDType> {
    q_proj: LinearWeight<T>,
    kv_proj: KvProjWeight<T>,
    o_proj: LinearWeight<T>,
    q_norm: Option<CudaTensor<T>>,
    k_norm: Option<CudaTensor<T>>,
}

struct GemmaMlpWeights<T: TensorDType> {
    gate_up: GateUpWeight<T>,
    down_proj: LinearWeight<T>,
}

struct GemmaLayerWeights<T: TensorDType> {
    input_layernorm: CudaTensor<T>,
    post_attention_layernorm: CudaTensor<T>,
    pre_feedforward_layernorm: CudaTensor<T>,
    post_feedforward_layernorm: CudaTensor<T>,
    attention: GemmaAttentionWeights<T>,
    mlp: GemmaMlpWeights<T>,
}

/// Gemma model supporting both Gemma 2 and Gemma 3 text architectures.
pub struct GemmaModel<T: TensorDType> {
    config: GemmaConfig,
    ctx: CudaContext,
    #[allow(dead_code)]
    gpu_config: GpuConfig,

    #[cfg(feature = "nccl")]
    nccl_comm: Option<NcclCommunicator>,

    tp_num_heads: usize,
    tp_num_kv_heads: usize,

    embed_tokens: CudaTensor<T>,
    layers: Vec<GemmaLayerWeights<T>>,
    norm: CudaTensor<T>,
    lm_head: LinearWeight<T>,

    /// Embedding scale factor: sqrt(hidden_size)
    embed_scale: f32,

    /// Attention scale: 1 / sqrt(query_pre_attn_scalar)
    attn_scale: f32,

    // RoPE caches — Gemma 2: single set, Gemma 3: two sets (local + global)
    cos_cache: CudaTensor<T>,
    sin_cache: CudaTensor<T>,
    // Gemma 3 dual-theta RoPE: separate cache for full-attention layers
    cos_cache_global: Option<CudaTensor<T>>,
    sin_cache_global: Option<CudaTensor<T>>,
}

#[allow(private_bounds)]
impl<T> GemmaModel<T>
where
    T: TensorDType + DeviceRepr + GemmScalar + Default + ValidAsZeroBits + MaybeNcclType,
    CudaBlas: Gemm<T>,
{
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
        fn load_linear<T: TensorDType + DeviceRepr>(
            ctx: &CudaContext,
            loader: &impl WeightLoader,
            name: &str,
            quant_config: Option<&crate::config::QuantizationConfig>,
        ) -> Result<LinearWeight<T>> {
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
                    let scale_val = scale_tensor.to_vec()?;
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
                if T::DTYPE == infernum::dtype::DType::F32 {
                    Ok(LinearWeight::Dense(reinterpret_tensor(transposed)))
                } else {
                    let native = load_typed::<T>(loader, ctx, name)?;
                    let shape = native.shape().to_vec();
                    let data = native.to_vec()?;
                    let rows = shape[0];
                    let cols = shape[1];
                    let mut transposed_data = vec![T::default(); data.len()];
                    for r in 0..rows {
                        for c in 0..cols {
                            transposed_data[c * rows + r] = data[r * cols + c];
                        }
                    }
                    Ok(LinearWeight::Dense(CudaTensor::from_slice(
                        ctx,
                        &[cols, rows],
                        &transposed_data,
                    )?))
                }
            }
        }

        let qc = config.quantization_config.as_ref();

        let embed_tokens = load_typed::<T>(loader, ctx, "model.embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            // Load attention weights
            let k = load_linear::<T>(
                ctx,
                loader,
                &format!("{prefix}.self_attn.k_proj.weight"),
                qc,
            )?;
            let v = load_linear::<T>(
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
                Some(load_typed::<T>(loader, ctx, &q_norm_name)?)
            } else {
                None
            };
            let k_norm = if loader.contains(&k_norm_name) {
                Some(load_typed::<T>(loader, ctx, &k_norm_name)?)
            } else {
                None
            };

            // Load MLP weights (GeGLU: gate_proj, up_proj, down_proj)
            let gate =
                load_linear::<T>(ctx, loader, &format!("{prefix}.mlp.gate_proj.weight"), qc)?;
            let up = load_linear::<T>(ctx, loader, &format!("{prefix}.mlp.up_proj.weight"), qc)?;
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
                input_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.input_layernorm.weight"),
                )?,
                post_attention_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                pre_feedforward_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.pre_feedforward_layernorm.weight"),
                )?,
                post_feedforward_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.post_feedforward_layernorm.weight"),
                )?,
                attention: GemmaAttentionWeights {
                    q_proj: load_linear::<T>(
                        ctx,
                        loader,
                        &format!("{prefix}.self_attn.q_proj.weight"),
                        qc,
                    )?,
                    kv_proj,
                    o_proj: load_linear::<T>(
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
                    down_proj: load_linear::<T>(
                        ctx,
                        loader,
                        &format!("{prefix}.mlp.down_proj.weight"),
                        qc,
                    )?,
                },
            };

            layers.push(layer);
        }

        let norm = load_typed::<T>(loader, ctx, "model.norm.weight")?;

        // Tied word embeddings — Gemma always ties lm_head to embed_tokens
        let lm_head = if qc.is_some() {
            let embed_f32 = cast_to_f32(&embed_tokens)?;
            let data = embed_f32.to_vec()?;
            LinearWeight::Quantized(QuantizedTensor::from_f32_as_q8(
                ctx,
                embed_f32.shape(),
                &data,
            )?)
        } else {
            let embed_f32 = cast_to_f32(&embed_tokens)?;
            let transposed = pretranspose_weight(&embed_f32)?;
            if T::DTYPE == infernum::dtype::DType::F32 {
                LinearWeight::Dense(reinterpret_tensor(transposed))
            } else {
                let shape = transposed.shape().to_vec();
                let data_f32 = transposed.to_vec()?;
                let data_t: Vec<T> = data_f32.iter().map(|&v| T::from_f32(v)).collect();
                LinearWeight::Dense(CudaTensor::from_slice(ctx, &shape, &data_t)?)
            }
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

        let (cos_cache, sin_cache) = convert_rope_cache::<T>(ctx, &cos_f32, &sin_f32)?;

        let (cos_cache_global, sin_cache_global) = if config.rope_local_base_freq.is_some() {
            let (cos_g_f32, sin_g_f32) = precompute_rope_cache(
                ctx,
                config.max_position_embeddings,
                config.head_dim,
                config.rope_theta,
            )?;
            let (cos_g, sin_g) = convert_rope_cache::<T>(ctx, &cos_g_f32, &sin_g_f32)?;
            (Some(cos_g), Some(sin_g))
        } else {
            (None, None)
        };

        let embed_scale = (config.hidden_size as f32).sqrt();
        let attn_scale = config.attn_scale();

        Ok(Self {
            tp_num_heads: config.num_attention_heads,
            tp_num_kv_heads: config.num_key_value_heads,
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
        fn load_linear_sharded<T: TensorDType + DeviceRepr>(
            ctx: &CudaContext,
            loader: &impl WeightLoader,
            name: &str,
            shard: &ShardConfig,
            strategy: ShardStrategy,
            quant_config: Option<&crate::config::QuantizationConfig>,
        ) -> Result<LinearWeight<T>> {
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
                    let scale_val = scale_tensor.to_vec()?;
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
                if T::DTYPE == infernum::dtype::DType::F32 {
                    Ok(LinearWeight::Dense(reinterpret_tensor(transposed)))
                } else {
                    let native = load_typed_sharded::<T>(loader, ctx, name, shard, strategy)?;
                    let shape = native.shape().to_vec();
                    let data = native.to_vec()?;
                    let rows = shape[0];
                    let cols = shape[1];
                    let mut transposed_data = vec![T::default(); data.len()];
                    for r in 0..rows {
                        for c in 0..cols {
                            transposed_data[c * rows + r] = data[r * cols + c];
                        }
                    }
                    Ok(LinearWeight::Dense(CudaTensor::from_slice(
                        ctx,
                        &[cols, rows],
                        &transposed_data,
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
            config.num_attention_heads % world_size == 0,
            "num_attention_heads ({}) must be divisible by world_size ({world_size})",
            config.num_attention_heads
        );
        assert!(
            config.num_key_value_heads % world_size == 0,
            "num_key_value_heads ({}) must be divisible by world_size ({world_size})",
            config.num_key_value_heads
        );

        let tp_num_heads = config.num_attention_heads / world_size;
        let tp_num_kv_heads = config.num_key_value_heads / world_size;

        let embed_tokens = load_typed::<T>(loader, ctx, "model.embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            let k = load_linear_sharded::<T>(
                ctx,
                loader,
                &format!("{prefix}.self_attn.k_proj.weight"),
                &shard,
                ShardStrategy::Row,
                qc,
            )?;
            let v = load_linear_sharded::<T>(
                ctx,
                loader,
                &format!("{prefix}.self_attn.v_proj.weight"),
                &shard,
                ShardStrategy::Row,
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
                Some(load_typed::<T>(loader, ctx, &q_norm_name)?)
            } else {
                None
            };
            let k_norm = if loader.contains(&k_norm_name) {
                Some(load_typed::<T>(loader, ctx, &k_norm_name)?)
            } else {
                None
            };

            let gate = load_linear_sharded::<T>(
                ctx,
                loader,
                &format!("{prefix}.mlp.gate_proj.weight"),
                &shard,
                ShardStrategy::Row,
                qc,
            )?;
            let up = load_linear_sharded::<T>(
                ctx,
                loader,
                &format!("{prefix}.mlp.up_proj.weight"),
                &shard,
                ShardStrategy::Row,
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
                input_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.input_layernorm.weight"),
                )?,
                post_attention_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                pre_feedforward_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.pre_feedforward_layernorm.weight"),
                )?,
                post_feedforward_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.post_feedforward_layernorm.weight"),
                )?,
                attention: GemmaAttentionWeights {
                    q_proj: load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &format!("{prefix}.self_attn.q_proj.weight"),
                        &shard,
                        ShardStrategy::Row,
                        qc,
                    )?,
                    kv_proj,
                    o_proj: load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &format!("{prefix}.self_attn.o_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?,
                    q_norm,
                    k_norm,
                },
                mlp: GemmaMlpWeights {
                    gate_up,
                    down_proj: load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &format!("{prefix}.mlp.down_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?,
                },
            };

            layers.push(layer);
        }

        let norm = load_typed::<T>(loader, ctx, "model.norm.weight")?;

        // Tied embeddings — not sharded for lm_head
        let lm_head = if qc.is_some() {
            let embed_f32 = cast_to_f32(&embed_tokens)?;
            let data = embed_f32.to_vec()?;
            LinearWeight::Quantized(QuantizedTensor::from_f32_as_q8(
                ctx,
                embed_f32.shape(),
                &data,
            )?)
        } else {
            let embed_f32 = cast_to_f32(&embed_tokens)?;
            let transposed = pretranspose_weight(&embed_f32)?;
            if T::DTYPE == infernum::dtype::DType::F32 {
                LinearWeight::Dense(reinterpret_tensor(transposed))
            } else {
                let shape = transposed.shape().to_vec();
                let data_f32 = transposed.to_vec()?;
                let data_t: Vec<T> = data_f32.iter().map(|&v| T::from_f32(v)).collect();
                LinearWeight::Dense(CudaTensor::from_slice(ctx, &shape, &data_t)?)
            }
        };

        let local_theta = config.rope_local_base_freq.unwrap_or(config.rope_theta);
        let (cos_f32, sin_f32) = precompute_rope_cache(
            ctx,
            config.max_position_embeddings,
            config.head_dim,
            local_theta,
        )?;
        let (cos_cache, sin_cache) = convert_rope_cache::<T>(ctx, &cos_f32, &sin_f32)?;

        let (cos_cache_global, sin_cache_global) = if config.rope_local_base_freq.is_some() {
            let (cos_g_f32, sin_g_f32) = precompute_rope_cache(
                ctx,
                config.max_position_embeddings,
                config.head_dim,
                config.rope_theta,
            )?;
            let (cos_g, sin_g) = convert_rope_cache::<T>(ctx, &cos_g_f32, &sin_g_f32)?;
            (Some(cos_g), Some(sin_g))
        } else {
            (None, None)
        };

        let embed_scale = (config.hidden_size as f32).sqrt();
        let attn_scale = config.attn_scale();

        Ok(Self {
            tp_num_heads,
            tp_num_kv_heads,
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
        paged_kvs: &mut [PagedKvCache<T>],
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<CudaTensor<f32>> {
        let batch_size = token_ids.len();
        let paged_kv = &mut paged_kvs[0];

        let mut hidden = self.embed(token_ids)?;
        scale_inplace(&mut hidden, self.embed_scale)?;

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

        rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;
        self.lm_head_forward(&hidden.reshape(&[batch_size, self.config.hidden_size]))
    }

    /// Batched decode using indirect kernels for CUDA graph capture.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward_batch_decode_indirect(
        &self,
        graph_inputs: &BatchedGraphInputs,
        paged_kvs: &mut [PagedKvCache<T>],
        max_seq_len: usize,
    ) -> Result<CudaTensor<f32>> {
        let batch_size = graph_inputs.max_batch_size();
        let paged_kv = &mut paged_kvs[0];

        let mut hidden = embedding_gather_from_device(
            &self.ctx,
            &self.embed_tokens,
            graph_inputs.token_ids(),
            batch_size,
        )?;
        scale_inplace(&mut hidden, self.embed_scale)?;

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

    /// Single-sequence prefill with paged KV cache.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward_prefill_paged(
        &self,
        input_ids: &[u32],
        paged_kvs: &mut [PagedKvCache<T>],
        block_table: &BlockTable,
        start_pos: usize,
    ) -> Result<CudaTensor<f32>> {
        let seq_len = input_ids.len();
        let paged_kv = &mut paged_kvs[0];

        let mut hidden = self.embed(input_ids)?;
        scale_inplace(&mut hidden, self.embed_scale)?;

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

    fn forward_layer_paged_decode(
        &self,
        hidden: &CudaTensor<T>,
        layer: &GemmaLayerWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<CudaTensor<T>> {
        let normed = rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

        let attn_output = self.forward_attention_paged_decode(
            &normed,
            &layer.attention,
            layer_idx,
            paged_kv,
            block_tables,
            positions,
        )?;

        let post_attn = rms_norm(
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;
        let mut hidden = hidden.clone();
        add_inplace(&mut hidden, &post_attn)?;

        let normed_ffn = rms_norm(
            &hidden,
            &layer.pre_feedforward_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_mlp(&normed_ffn, &layer.mlp)?;

        let post_ffn = rms_norm(
            &mlp_output,
            &layer.post_feedforward_layernorm,
            self.config.rms_norm_eps,
        )?;
        add_inplace(&mut hidden, &post_ffn)?;

        Ok(hidden)
    }

    fn forward_attention_paged_decode(
        &self,
        hidden: &CudaTensor<T>,
        weights: &GemmaAttentionWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<CudaTensor<T>> {
        let batch_size = hidden.shape()[0];
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim;

        let q = linear(hidden, &weights.q_proj)?;
        let (k, v) = match &weights.kv_proj {
            KvProjWeight::Fused { weight, kv_dim } => {
                let kv = matmul(hidden, weight)?;
                if batch_size == 1 {
                    let k = kv.slice_view(0, &[1, *kv_dim]);
                    let v = kv.slice_view(*kv_dim, &[1, *kv_dim]);
                    (k, v)
                } else {
                    split_kv(&kv, *kv_dim)?
                }
            }
            KvProjWeight::Separate { k_proj, v_proj } => {
                let k = linear(hidden, k_proj)?;
                let v = linear(hidden, v_proj)?;
                (k, v)
            }
        };

        let mut q = q.reshape(&[batch_size, num_heads, head_dim]);
        let mut k = k.reshape(&[batch_size, num_kv_heads, head_dim]);
        let v = v.reshape(&[batch_size, num_kv_heads, head_dim]);

        if let Some(ref q_norm_w) = weights.q_norm {
            let flat_q = q.reshape(&[batch_size * num_heads, head_dim]);
            let normed_q = rms_norm(&flat_q, q_norm_w, self.config.rms_norm_eps)?;
            q = normed_q.reshape(&[batch_size, num_heads, head_dim]);
        }
        if let Some(ref k_norm_w) = weights.k_norm {
            let flat_k = k.reshape(&[batch_size * num_kv_heads, head_dim]);
            let normed_k = rms_norm(&flat_k, k_norm_w, self.config.rms_norm_eps)?;
            k = normed_k.reshape(&[batch_size, num_kv_heads, head_dim]);
        }

        let (cos, sin) = self.rope_caches_for_layer(layer_idx);
        let q = apply_rope_batched(&q, cos, sin, positions)?;
        let k = apply_rope_batched(&k, cos, sin, positions)?;

        let q_stride = num_heads * head_dim;
        let kv_stride = num_kv_heads * head_dim;
        let sliding_window = self.config.effective_sliding_window(layer_idx);

        let mut attn_parts = Vec::with_capacity(batch_size);
        for (i, &pos) in positions.iter().enumerate() {
            let q_i = q.slice_view(i * q_stride, &[1, num_heads, head_dim]);
            let k_i = k.slice_view(i * kv_stride, &[1, num_kv_heads, head_dim]);
            let v_i = v.slice_view(i * kv_stride, &[1, num_kv_heads, head_dim]);

            paged_kv.append_paged(layer_idx, &block_tables[i], &k_i, &v_i, pos)?;

            let (k_pool, v_pool) = paged_kv.get_pools(layer_idx);
            let attn_i = paged_attention_decode(
                &self.ctx,
                &q_i,
                k_pool,
                v_pool,
                &[block_tables[i].clone()],
                paged_kv.block_size(),
                Some(self.attn_scale),
                sliding_window,
            )?;

            attn_parts.push(attn_i.reshape(&[1, num_heads * head_dim]));
        }

        let attn_output = if batch_size == 1 {
            attn_parts.into_iter().next().unwrap()
        } else {
            let row_size = num_heads * head_dim;
            let mut output =
                unsafe { CudaTensor::<T>::uninit(&self.ctx, &[batch_size, row_size])? };
            let out_slice = output.cuda_slice_mut();
            for (i, part) in attn_parts.iter().enumerate() {
                let src = part.cuda_slice().slice(..row_size);
                let mut dst = out_slice.slice_mut(i * row_size..(i + 1) * row_size);
                self.ctx.device().dtod_copy(&src, &mut dst)?;
            }
            output
        };

        let mut out = linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_paged_decode_indirect(
        &self,
        hidden: &CudaTensor<T>,
        layer: &GemmaLayerWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        graph_inputs: &BatchedGraphInputs,
        max_seq_len: usize,
    ) -> Result<CudaTensor<T>> {
        let normed = rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

        let attn_output = self.forward_attention_paged_decode_indirect(
            &normed,
            &layer.attention,
            layer_idx,
            paged_kv,
            graph_inputs,
            max_seq_len,
        )?;

        let post_attn = rms_norm(
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;
        let mut hidden = hidden.clone();
        add_inplace(&mut hidden, &post_attn)?;

        let normed_ffn = rms_norm(
            &hidden,
            &layer.pre_feedforward_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_mlp(&normed_ffn, &layer.mlp)?;

        let post_ffn = rms_norm(
            &mlp_output,
            &layer.post_feedforward_layernorm,
            self.config.rms_norm_eps,
        )?;
        add_inplace(&mut hidden, &post_ffn)?;

        Ok(hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attention_paged_decode_indirect(
        &self,
        hidden: &CudaTensor<T>,
        weights: &GemmaAttentionWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        graph_inputs: &BatchedGraphInputs,
        max_seq_len: usize,
    ) -> Result<CudaTensor<T>> {
        let batch_size = hidden.shape()[0];
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim;

        let q = linear(hidden, &weights.q_proj)?;
        let (k, v) = match &weights.kv_proj {
            KvProjWeight::Fused { weight, kv_dim } => {
                let kv = matmul(hidden, weight)?;
                if batch_size == 1 {
                    let k = kv.slice_view(0, &[1, *kv_dim]);
                    let v = kv.slice_view(*kv_dim, &[1, *kv_dim]);
                    (k, v)
                } else {
                    split_kv(&kv, *kv_dim)?
                }
            }
            KvProjWeight::Separate { k_proj, v_proj } => {
                let k = linear(hidden, k_proj)?;
                let v = linear(hidden, v_proj)?;
                (k, v)
            }
        };

        let mut q = q.reshape(&[batch_size, num_heads, head_dim]);
        let mut k = k.reshape(&[batch_size, num_kv_heads, head_dim]);
        let v = v.reshape(&[batch_size, num_kv_heads, head_dim]);

        if let Some(ref q_norm_w) = weights.q_norm {
            let flat_q = q.reshape(&[batch_size * num_heads, head_dim]);
            let normed_q = rms_norm(&flat_q, q_norm_w, self.config.rms_norm_eps)?;
            q = normed_q.reshape(&[batch_size, num_heads, head_dim]);
        }
        if let Some(ref k_norm_w) = weights.k_norm {
            let flat_k = k.reshape(&[batch_size * num_kv_heads, head_dim]);
            let normed_k = rms_norm(&flat_k, k_norm_w, self.config.rms_norm_eps)?;
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

        let (k_pool, v_pool) = paged_kv.get_pools(layer_idx);
        let sliding_window = self.config.effective_sliding_window(layer_idx);
        let attn_output = paged_attention_decode_indirect(
            &self.ctx,
            &q,
            k_pool,
            v_pool,
            graph_inputs.block_tables(),
            graph_inputs.seq_lens(),
            paged_kv.block_size(),
            graph_inputs.max_blocks_per_seq(),
            max_seq_len,
            Some(self.attn_scale),
            sliding_window,
        )?;

        let attn_output = attn_output.reshape(&[batch_size, num_heads * head_dim]);
        let mut out = linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_paged_prefill(
        &self,
        hidden: &CudaTensor<T>,
        layer: &GemmaLayerWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        block_table: &BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<CudaTensor<T>> {
        let normed = rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

        let attn_output = self.forward_attention_paged_prefill(
            &normed,
            &layer.attention,
            layer_idx,
            paged_kv,
            block_table,
            start_pos,
            seq_len,
        )?;

        let post_attn = rms_norm(
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;
        let mut hidden = hidden.clone();
        add_inplace(&mut hidden, &post_attn)?;

        let normed_ffn = rms_norm(
            &hidden,
            &layer.pre_feedforward_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_mlp(&normed_ffn, &layer.mlp)?;

        let post_ffn = rms_norm(
            &mlp_output,
            &layer.post_feedforward_layernorm,
            self.config.rms_norm_eps,
        )?;
        add_inplace(&mut hidden, &post_ffn)?;

        Ok(hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attention_paged_prefill(
        &self,
        hidden: &CudaTensor<T>,
        weights: &GemmaAttentionWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        block_table: &BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<CudaTensor<T>> {
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim;

        let q = linear(hidden, &weights.q_proj)?;
        let (k, v) = match &weights.kv_proj {
            KvProjWeight::Fused { weight, kv_dim } => {
                let kv = matmul(hidden, weight)?;
                split_kv(&kv, *kv_dim)?
            }
            KvProjWeight::Separate { k_proj, v_proj } => {
                let k = linear(hidden, k_proj)?;
                let v = linear(hidden, v_proj)?;
                (k, v)
            }
        };

        let mut q = q.reshape(&[seq_len, num_heads, head_dim]);
        let mut k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
        let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

        if let Some(ref q_norm_w) = weights.q_norm {
            let flat_q = q.reshape(&[seq_len * num_heads, head_dim]);
            let normed_q = rms_norm(&flat_q, q_norm_w, self.config.rms_norm_eps)?;
            q = normed_q.reshape(&[seq_len, num_heads, head_dim]);
        }
        if let Some(ref k_norm_w) = weights.k_norm {
            let flat_k = k.reshape(&[seq_len * num_kv_heads, head_dim]);
            let normed_k = rms_norm(&flat_k, k_norm_w, self.config.rms_norm_eps)?;
            k = normed_k.reshape(&[seq_len, num_kv_heads, head_dim]);
        }

        let (cos, sin) = self.rope_caches_for_layer(layer_idx);
        let q = apply_rope(&q, cos, sin, start_pos)?;
        let k = apply_rope(&k, cos, sin, start_pos)?;

        paged_kv.append_paged(layer_idx, block_table, &k, &v, start_pos)?;

        let mut gather_table = block_table.clone();
        gather_table.advance(seq_len);
        let (k_contig, v_contig) = gather_paged_kv(paged_kv, layer_idx, &gather_table)?;

        let sliding_window = self.config.effective_sliding_window(layer_idx);
        let attn_output = fused_attention_prefill(
            &q,
            &k_contig,
            &v_contig,
            start_pos,
            Some(self.attn_scale),
            self.config.attn_logit_softcapping,
            sliding_window,
        )?;

        let attn_output = attn_output.reshape(&[seq_len, num_heads * head_dim]);
        let mut out = linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    fn rope_caches_for_layer(&self, layer_idx: usize) -> (&CudaTensor<T>, &CudaTensor<T>) {
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
    fn forward_mlp(
        &self,
        hidden: &CudaTensor<T>,
        weights: &GemmaMlpWeights<T>,
    ) -> Result<CudaTensor<T>> {
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
                    split_gate_up(&gate_up, *intermediate_size)?
                }
            }
            GateUpWeight::Separate { gate_proj, up_proj } => {
                let gate = linear(hidden, gate_proj)?;
                let up = linear(hidden, up_proj)?;
                (gate, up)
            }
        };
        // GeGLU: gelu(gate) * up
        let intermediate = geglu(&gate, &up)?;
        let mut out = linear(&intermediate, &weights.down_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    fn extract_last_row(&self, hidden: &CudaTensor<T>, seq_len: usize) -> Result<CudaTensor<T>> {
        if seq_len == 1 {
            return Ok(hidden.reshape(&[1, self.config.hidden_size]));
        }
        let hidden_size = hidden.shape()[1];
        let flat = hidden.reshape(&[seq_len * hidden_size]);
        let mut out = unsafe { CudaTensor::<T>::uninit(&self.ctx, &[1, hidden_size])? };
        let device = self.ctx.device();
        let src = flat.cuda_slice();
        let last_offset = (seq_len - 1) * hidden_size;
        let src_sub = src.slice(last_offset..seq_len * hidden_size);
        device.dtod_copy(&src_sub, out.cuda_slice_mut())?;
        Ok(out)
    }

    fn embed(&self, input_ids: &[u32]) -> Result<CudaTensor<T>> {
        embedding_gather(&self.ctx, &self.embed_tokens, input_ids)
    }

    fn lm_head_forward(&self, hidden: &CudaTensor<T>) -> Result<CudaTensor<f32>> {
        // bf16 fast path
        if T::DTYPE == infernum::dtype::DType::BF16 {
            if let LinearWeight::Dense(w) = &self.lm_head {
                let h_bf16: CudaTensor<half::bf16> =
                    unsafe { hidden.slice_view(0, hidden.shape()).reinterpret() };
                let w_bf16: CudaTensor<half::bf16> =
                    unsafe { w.slice_view(0, w.shape()).reinterpret() };
                let mut logits = matmul_bf16_f32(&h_bf16, &w_bf16)?;
                self.apply_final_softcap(&mut logits)?;
                return Ok(logits);
            }
        }
        let logits_t = linear(hidden, &self.lm_head)?;
        let mut logits = if T::DTYPE == infernum::dtype::DType::F32 {
            unsafe { logits_t.reinterpret() }
        } else {
            cast_to_f32(&logits_t)?
        };
        self.apply_final_softcap(&mut logits)?;
        Ok(logits)
    }

    /// Apply final logit soft-capping (Gemma 2 only): tanh(logits / cap) * cap
    fn apply_final_softcap(&self, logits: &mut CudaTensor<f32>) -> Result<()> {
        if let Some(cap) = self.config.final_logit_softcapping {
            let data = logits.to_vec()?;
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
}

/// Convert f32 RoPE caches to model dtype T
fn convert_rope_cache<T: TensorDType + DeviceRepr + Default>(
    ctx: &CudaContext,
    cos_f32: &CudaTensor<f32>,
    sin_f32: &CudaTensor<f32>,
) -> Result<(CudaTensor<T>, CudaTensor<T>)> {
    if T::DTYPE == infernum::dtype::DType::F32 {
        Ok((
            reinterpret_tensor(cos_f32.slice_view(0, cos_f32.shape())),
            reinterpret_tensor(sin_f32.slice_view(0, sin_f32.shape())),
        ))
    } else {
        let cos_data: Vec<T> = cos_f32.to_vec()?.iter().map(|&v| T::from_f32(v)).collect();
        let sin_data: Vec<T> = sin_f32.to_vec()?.iter().map(|&v| T::from_f32(v)).collect();
        let cos = CudaTensor::from_slice(ctx, cos_f32.shape(), &cos_data)?;
        let sin = CudaTensor::from_slice(ctx, sin_f32.shape(), &sin_data)?;
        Ok((cos, sin))
    }
}

fn linear<T>(input: &CudaTensor<T>, weight: &LinearWeight<T>) -> Result<CudaTensor<T>>
where
    T: TensorDType + DeviceRepr + GemmScalar + Default,
    CudaBlas: Gemm<T>,
{
    match weight {
        LinearWeight::Dense(w) => matmul(input, w),
        LinearWeight::Quantized(w) => {
            let input_f32 = if T::DTYPE == infernum::dtype::DType::F32 {
                reinterpret_tensor(input.slice_view(0, input.shape()))
            } else {
                cast_to_f32(input)?
            };
            let output_f32 = quantized_matmul(&input_f32, w)?;
            Ok(reinterpret_tensor(output_f32))
        }
    }
}

// --- Model trait implementation ---

#[allow(private_bounds)]
impl<T> infernum::Model for GemmaModel<T>
where
    T: TensorDType + DeviceRepr + GemmScalar + Default + ValidAsZeroBits + MaybeNcclType,
    CudaBlas: Gemm<T>,
{
    type CacheDtype = T;

    fn config(&self) -> infernum::ModelConfig {
        infernum::ModelConfig {
            num_layers: self.config.num_hidden_layers,
            max_seq_len: self.config.max_position_embeddings,
            num_kv_heads: self.tp_num_kv_heads,
            head_dim: self.config.head_dim,
            eos_token_id: self.config.eos_token_id,
        }
    }

    fn devices(&self) -> Vec<&CudaContext> {
        vec![&self.ctx]
    }

    fn forward(&self, input_ids: &[u32]) -> Result<CudaTensor<f32>> {
        let seq_len = input_ids.len();
        let mut hidden = self.embed(input_ids)?;
        scale_inplace(&mut hidden, self.embed_scale)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let normed = rms_norm(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            let num_heads = self.tp_num_heads;
            let num_kv_heads = self.tp_num_kv_heads;
            let head_dim = self.config.head_dim;
            let sliding_window = self.config.effective_sliding_window(layer_idx);

            let q = linear(&normed, &layer.attention.q_proj)?;
            let (k, v) = match &layer.attention.kv_proj {
                KvProjWeight::Fused { weight, kv_dim } => {
                    let kv = matmul(&normed, weight)?;
                    split_kv(&kv, *kv_dim)?
                }
                KvProjWeight::Separate { k_proj, v_proj } => {
                    let k = linear(&normed, k_proj)?;
                    let v = linear(&normed, v_proj)?;
                    (k, v)
                }
            };

            let mut q = q.reshape(&[seq_len, num_heads, head_dim]);
            let mut k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
            let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

            if let Some(ref q_norm_w) = layer.attention.q_norm {
                let flat_q = q.reshape(&[seq_len * num_heads, head_dim]);
                let normed_q = rms_norm(&flat_q, q_norm_w, self.config.rms_norm_eps)?;
                q = normed_q.reshape(&[seq_len, num_heads, head_dim]);
            }
            if let Some(ref k_norm_w) = layer.attention.k_norm {
                let flat_k = k.reshape(&[seq_len * num_kv_heads, head_dim]);
                let normed_k = rms_norm(&flat_k, k_norm_w, self.config.rms_norm_eps)?;
                k = normed_k.reshape(&[seq_len, num_kv_heads, head_dim]);
            }

            let (cos, sin) = self.rope_caches_for_layer(layer_idx);
            let q = apply_rope(&q, cos, sin, 0)?;
            let k = apply_rope(&k, cos, sin, 0)?;

            let attn_output = fused_attention_prefill(
                &q,
                &k,
                &v,
                0,
                Some(self.attn_scale),
                self.config.attn_logit_softcapping,
                sliding_window,
            )?;
            let attn_output = attn_output.reshape(&[seq_len, num_heads * head_dim]);
            let mut attn_output = linear(&attn_output, &layer.attention.o_proj)?;
            #[cfg(feature = "nccl")]
            nccl_all_reduce(self.nccl_comm.as_ref(), &mut attn_output)?;

            // Post-attention norm + residual
            let post_attn = rms_norm(
                &attn_output,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;
            add_inplace(&mut hidden, &post_attn)?;

            // Pre-feedforward norm
            let normed_ffn = rms_norm(
                &hidden,
                &layer.pre_feedforward_layernorm,
                self.config.rms_norm_eps,
            )?;

            let mlp_output = self.forward_mlp(&normed_ffn, &layer.mlp)?;

            // Post-feedforward norm + residual
            let post_ffn = rms_norm(
                &mlp_output,
                &layer.post_feedforward_layernorm,
                self.config.rms_norm_eps,
            )?;
            add_inplace(&mut hidden, &post_ffn)?;
        }

        rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;
        self.lm_head_forward(&hidden)
    }

    fn forward_prefill_paged(
        &self,
        input_ids: &[u32],
        paged_kvs: &mut [PagedKvCache<T>],
        block_table: &BlockTable,
        start_pos: usize,
    ) -> Result<CudaTensor<f32>> {
        self.forward_prefill_paged(input_ids, paged_kvs, block_table, start_pos)
    }

    fn forward_batch_decode(
        &self,
        token_ids: &[u32],
        paged_kvs: &mut [PagedKvCache<T>],
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<CudaTensor<f32>> {
        self.forward_batch_decode(token_ids, paged_kvs, block_tables, positions)
    }

    fn forward_batch_decode_indirect(
        &self,
        graph_inputs: &BatchedGraphInputs,
        paged_kvs: &mut [PagedKvCache<T>],
        max_seq_len: usize,
    ) -> Result<CudaTensor<f32>> {
        self.forward_batch_decode_indirect(graph_inputs, paged_kvs, max_seq_len)
    }
}

#[cfg(feature = "nccl")]
#[allow(private_bounds)]
impl<T> infernum::ShardedLoadable for GemmaModel<T>
where
    T: TensorDType
        + DeviceRepr
        + GemmScalar
        + Default
        + ValidAsZeroBits
        + MaybeNcclType
        + Send
        + Sync
        + 'static,
    CudaBlas: Gemm<T>,
{
    fn load_shard(
        ctx: &CudaContext,
        model_path: &Path,
        shard: ShardConfig,
        comm: NcclCommunicator,
    ) -> Result<Self> {
        Self::from_pretrained_sharded(ctx, model_path, GpuConfig::Sharded(shard), comm)
    }
}
