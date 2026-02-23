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

use infernum::cuda::ops::{
    add_inplace, add_rmsnorm, apply_rope_interleaved, apply_rope_interleaved_indirect,
    cast_f32_to_bf16, cast_to_f32, embedding_gather, embedding_gather_from_device,
    fused_attention_decode, fused_attention_decode_indirect, fused_attention_prefill, matmul,
    matmul_bf16_f32, precompute_rope_cache, precompute_rope_cache_scaled, quantized_matmul,
    rms_norm, rms_norm_inplace, swiglu, transpose_2d, GemmScalar, RopeScaling,
};
use infernum::cuda::{
    CudaBlas, CudaContext, CudaSlice, CudaTensor, DeviceRepr, Gemm, GpuConfig, QuantizedTensor,
    ValidAsZeroBits,
};
#[cfg(feature = "nccl")]
use infernum::cuda::{NcclCommunicator, NcclType, ShardConfig, ShardStrategy};
use infernum::dtype::TensorDType;
use infernum::tensor::Tensor;
use infernum::weights::{SafeTensorsLoader, WeightLoader};
use infernum::KvCache;
use infernum::Result;

use crate::DeepSeekConfig;

// --- NCCL conditional trait bounds (same pattern as infernum-llama / infernum-qwen) ---

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

/// MLA attention weights (shared by dense and MoE layers)
struct DeepSeekAttentionWeights<T: TensorDType> {
    q_a_proj: LinearWeight<T>,
    q_a_layernorm: CudaTensor<T>,
    q_b_proj: LinearWeight<T>,
    kv_a_proj_with_mqa: LinearWeight<T>,
    kv_a_layernorm: CudaTensor<T>,
    kv_b_proj: LinearWeight<T>,
    o_proj: LinearWeight<T>,
}

/// Dense MLP weights
struct DenseMlpWeights<T: TensorDType> {
    gate_proj: LinearWeight<T>,
    up_proj: LinearWeight<T>,
    down_proj: LinearWeight<T>,
}

/// MoE layer weights
struct MoeWeights<T: TensorDType> {
    gate_weight: CudaTensor<T>,
    e_score_correction_bias: Vec<f32>,
    experts: Vec<DenseMlpWeights<T>>,
    shared_expert: DenseMlpWeights<T>,
}

/// Dense vs MoE FFN layer
#[allow(clippy::large_enum_variant)]
enum FfnWeights<T: TensorDType> {
    Dense(Box<DenseMlpWeights<T>>),
    Moe(Box<MoeWeights<T>>),
}

/// Single transformer layer
struct DeepSeekLayerWeights<T: TensorDType> {
    input_layernorm: CudaTensor<T>,
    attention: DeepSeekAttentionWeights<T>,
    post_attention_layernorm: CudaTensor<T>,
    ffn: FfnWeights<T>,
}

/// Complete DeepSeek V3/R1 model, generic over the compute dtype `T`.
pub struct DeepSeekModel<T: TensorDType> {
    config: DeepSeekConfig,
    ctx: CudaContext,
    #[allow(dead_code)]
    gpu_config: GpuConfig,

    #[cfg(feature = "nccl")]
    nccl_comm: Option<NcclCommunicator>,

    tp_num_heads: usize,

    embed_tokens: CudaTensor<T>,
    layers: Vec<DeepSeekLayerWeights<T>>,
    norm: CudaTensor<T>,
    lm_head: LinearWeight<T>,

    cos_cache: CudaTensor<T>,
    sin_cache: CudaTensor<T>,

    /// Pre-computed attention scale (includes YaRN mscale adjustment)
    attn_scale: f32,
}

#[allow(private_bounds)]
impl<T> DeepSeekModel<T>
where
    T: TensorDType + DeviceRepr + GemmScalar + Default + ValidAsZeroBits + MaybeNcclType,
    CudaBlas: Gemm<T>,
{
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

            // MLA attention weights
            let attention = DeepSeekAttentionWeights {
                q_a_proj: load_linear::<T>(
                    ctx,
                    loader,
                    &format!("{prefix}.self_attn.q_a_proj.weight"),
                    qc,
                )?,
                q_a_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.self_attn.q_a_layernorm.weight"),
                )?,
                q_b_proj: load_linear::<T>(
                    ctx,
                    loader,
                    &format!("{prefix}.self_attn.q_b_proj.weight"),
                    qc,
                )?,
                kv_a_proj_with_mqa: load_linear::<T>(
                    ctx,
                    loader,
                    &format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
                    qc,
                )?,
                kv_a_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.self_attn.kv_a_layernorm.weight"),
                )?,
                kv_b_proj: load_linear::<T>(
                    ctx,
                    loader,
                    &format!("{prefix}.self_attn.kv_b_proj.weight"),
                    qc,
                )?,
                o_proj: load_linear::<T>(
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
                let gate_weight = if T::DTYPE == infernum::dtype::DType::F32 {
                    reinterpret_tensor(gate_transposed)
                } else {
                    let data_f32 = gate_transposed.to_vec()?;
                    let data_t: Vec<T> = data_f32.iter().map(|&v| T::from_f32(v)).collect();
                    CudaTensor::from_slice(ctx, gate_transposed.shape(), &data_t)?
                };

                // Bias correction
                let bias_name = format!("{prefix}.mlp.gate.e_score_correction_bias");
                let e_score_correction_bias = if loader.contains(&bias_name) {
                    loader.load_f32(ctx, &bias_name)?.to_vec()?
                } else {
                    vec![0.0_f32; num_experts]
                };

                // Per-expert MLPs
                let mut experts = Vec::with_capacity(num_experts);
                for e in 0..num_experts {
                    let ep = format!("{prefix}.mlp.experts.{e}");
                    experts.push(DenseMlpWeights {
                        gate_proj: load_linear::<T>(
                            ctx,
                            loader,
                            &format!("{ep}.gate_proj.weight"),
                            qc,
                        )?,
                        up_proj: load_linear::<T>(
                            ctx,
                            loader,
                            &format!("{ep}.up_proj.weight"),
                            qc,
                        )?,
                        down_proj: load_linear::<T>(
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
                    gate_proj: load_linear::<T>(
                        ctx,
                        loader,
                        &format!("{sp}.gate_proj.weight"),
                        qc,
                    )?,
                    up_proj: load_linear::<T>(ctx, loader, &format!("{sp}.up_proj.weight"), qc)?,
                    down_proj: load_linear::<T>(
                        ctx,
                        loader,
                        &format!("{sp}.down_proj.weight"),
                        qc,
                    )?,
                };

                FfnWeights::Moe(Box::new(MoeWeights {
                    gate_weight,
                    e_score_correction_bias,
                    experts,
                    shared_expert,
                }))
            } else {
                let mp = format!("{prefix}.mlp");
                FfnWeights::Dense(Box::new(DenseMlpWeights {
                    gate_proj: load_linear::<T>(
                        ctx,
                        loader,
                        &format!("{mp}.gate_proj.weight"),
                        qc,
                    )?,
                    up_proj: load_linear::<T>(ctx, loader, &format!("{mp}.up_proj.weight"), qc)?,
                    down_proj: load_linear::<T>(
                        ctx,
                        loader,
                        &format!("{mp}.down_proj.weight"),
                        qc,
                    )?,
                }))
            };

            layers.push(DeepSeekLayerWeights {
                input_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.input_layernorm.weight"),
                )?,
                attention,
                post_attention_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                ffn,
            });
        }

        let norm = load_typed::<T>(loader, ctx, "model.norm.weight")?;

        let lm_head = if config.tie_word_embeddings {
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
        } else {
            load_linear::<T>(ctx, loader, "lm_head.weight", None)?
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
        let (cos_cache, sin_cache) = if T::DTYPE == infernum::dtype::DType::F32 {
            (reinterpret_tensor(cos_f32), reinterpret_tensor(sin_f32))
        } else {
            let cos_data: Vec<T> = cos_f32.to_vec()?.iter().map(|&v| T::from_f32(v)).collect();
            let sin_data: Vec<T> = sin_f32.to_vec()?.iter().map(|&v| T::from_f32(v)).collect();
            let cos = CudaTensor::from_slice(ctx, cos_f32.shape(), &cos_data)?;
            let sin = CudaTensor::from_slice(ctx, sin_f32.shape(), &sin_data)?;
            (cos, sin)
        };

        let attn_scale = config.mla_attn_scale();

        Ok(Self {
            tp_num_heads: config.num_attention_heads,
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
                let mut qt =
                    loader.load_quantized_sharded(ctx, name, shard, ShardStrategy::Replicate)?;
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

        let embed_tokens = load_typed::<T>(loader, ctx, "model.embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            // MLA attention weights
            // q_a_proj, kv_a_proj_with_mqa: replicated (shared bottleneck)
            // q_b_proj, kv_b_proj: column-sharded (output is per-head)
            // o_proj: row-sharded (all-reduce after)
            let attention = DeepSeekAttentionWeights {
                q_a_proj: load_linear_sharded::<T>(
                    ctx,
                    loader,
                    &format!("{prefix}.self_attn.q_a_proj.weight"),
                    &shard,
                    ShardStrategy::Replicate,
                    qc,
                )?,
                q_a_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.self_attn.q_a_layernorm.weight"),
                )?,
                q_b_proj: load_linear_sharded::<T>(
                    ctx,
                    loader,
                    &format!("{prefix}.self_attn.q_b_proj.weight"),
                    &shard,
                    ShardStrategy::Column,
                    qc,
                )?,
                kv_a_proj_with_mqa: load_linear_sharded::<T>(
                    ctx,
                    loader,
                    &format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
                    &shard,
                    ShardStrategy::Replicate,
                    qc,
                )?,
                kv_a_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.self_attn.kv_a_layernorm.weight"),
                )?,
                kv_b_proj: load_linear_sharded::<T>(
                    ctx,
                    loader,
                    &format!("{prefix}.self_attn.kv_b_proj.weight"),
                    &shard,
                    ShardStrategy::Column,
                    qc,
                )?,
                o_proj: load_linear_sharded::<T>(
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
                let gate_weight = if T::DTYPE == infernum::dtype::DType::F32 {
                    reinterpret_tensor(gate_transposed)
                } else {
                    let data_f32 = gate_transposed.to_vec()?;
                    let data_t: Vec<T> = data_f32.iter().map(|&v| T::from_f32(v)).collect();
                    CudaTensor::from_slice(ctx, gate_transposed.shape(), &data_t)?
                };

                // Bias correction: replicated
                let bias_name = format!("{prefix}.mlp.gate.e_score_correction_bias");
                let e_score_correction_bias = if loader.contains(&bias_name) {
                    loader.load_f32(ctx, &bias_name)?.to_vec()?
                } else {
                    vec![0.0_f32; num_experts]
                };

                // Per-expert MLPs: gate/up column-sharded, down row-sharded
                let mut experts = Vec::with_capacity(num_experts);
                for e in 0..num_experts {
                    let ep = format!("{prefix}.mlp.experts.{e}");
                    experts.push(DenseMlpWeights {
                        gate_proj: load_linear_sharded::<T>(
                            ctx,
                            loader,
                            &format!("{ep}.gate_proj.weight"),
                            &shard,
                            ShardStrategy::Column,
                            qc,
                        )?,
                        up_proj: load_linear_sharded::<T>(
                            ctx,
                            loader,
                            &format!("{ep}.up_proj.weight"),
                            &shard,
                            ShardStrategy::Column,
                            qc,
                        )?,
                        down_proj: load_linear_sharded::<T>(
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
                    gate_proj: load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &format!("{sp}.gate_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?,
                    up_proj: load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &format!("{sp}.up_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?,
                    down_proj: load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &format!("{sp}.down_proj.weight"),
                        &shard,
                        ShardStrategy::Row,
                        qc,
                    )?,
                };

                FfnWeights::Moe(Box::new(MoeWeights {
                    gate_weight,
                    e_score_correction_bias,
                    experts,
                    shared_expert,
                }))
            } else {
                let mp = format!("{prefix}.mlp");
                FfnWeights::Dense(Box::new(DenseMlpWeights {
                    gate_proj: load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &format!("{mp}.gate_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?,
                    up_proj: load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &format!("{mp}.up_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?,
                    down_proj: load_linear_sharded::<T>(
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
                input_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.input_layernorm.weight"),
                )?,
                attention,
                post_attention_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                ffn,
            });
        }

        let norm = load_typed::<T>(loader, ctx, "model.norm.weight")?;

        let lm_head = if config.tie_word_embeddings {
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
        } else {
            load_linear_sharded::<T>(
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
        let (cos_cache, sin_cache) = if T::DTYPE == infernum::dtype::DType::F32 {
            (reinterpret_tensor(cos_f32), reinterpret_tensor(sin_f32))
        } else {
            let cos_data: Vec<T> = cos_f32.to_vec()?.iter().map(|&v| T::from_f32(v)).collect();
            let sin_data: Vec<T> = sin_f32.to_vec()?.iter().map(|&v| T::from_f32(v)).collect();
            let cos = CudaTensor::from_slice(ctx, cos_f32.shape(), &cos_data)?;
            let sin = CudaTensor::from_slice(ctx, sin_f32.shape(), &sin_data)?;
            (cos, sin)
        };

        let attn_scale = config.mla_attn_scale();

        Ok(Self {
            tp_num_heads,
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

    fn embed(&self, input_ids: &[u32]) -> Result<CudaTensor<T>> {
        embedding_gather(&self.ctx, &self.embed_tokens, input_ids)
    }

    fn embed_from_device(&self, token_id_gpu: &CudaSlice<u32>) -> Result<CudaTensor<T>> {
        embedding_gather_from_device(&self.ctx, &self.embed_tokens, token_id_gpu, 1)
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

    fn lm_head_forward(&self, hidden: &CudaTensor<T>) -> Result<CudaTensor<f32>> {
        if T::DTYPE == infernum::dtype::DType::BF16 {
            if let LinearWeight::Dense(w) = &self.lm_head {
                let h_bf16: CudaTensor<half::bf16> =
                    unsafe { hidden.slice_view(0, hidden.shape()).reinterpret() };
                let w_bf16: CudaTensor<half::bf16> =
                    unsafe { w.slice_view(0, w.shape()).reinterpret() };
                return matmul_bf16_f32(&h_bf16, &w_bf16);
            }
        }
        let logits_t = linear(hidden, &self.lm_head)?;
        if T::DTYPE == infernum::dtype::DType::F32 {
            return Ok(unsafe { logits_t.reinterpret() });
        }
        cast_to_f32(&logits_t)
    }

    // --- MLA attention ---

    /// Split a 2D tensor `[seq, total_dim]` into two parts along the last dimension.
    fn split_last_dim(
        tensor: &CudaTensor<T>,
        dim1: usize,
        dim2: usize,
    ) -> Result<(CudaTensor<T>, CudaTensor<T>)> {
        let seq_len = tensor.shape()[0];
        let total = tensor.shape()[1];
        assert_eq!(total, dim1 + dim2, "split_last_dim: dim mismatch");

        let data = tensor.to_vec()?;
        let mut a_data = vec![T::default(); seq_len * dim1];
        let mut b_data = vec![T::default(); seq_len * dim2];

        for row in 0..seq_len {
            a_data[row * dim1..(row + 1) * dim1]
                .copy_from_slice(&data[row * total..row * total + dim1]);
            b_data[row * dim2..(row + 1) * dim2]
                .copy_from_slice(&data[row * total + dim1..(row + 1) * total]);
        }

        let a = CudaTensor::from_slice(tensor.context(), &[seq_len, dim1], &a_data)?;
        let b = CudaTensor::from_slice(tensor.context(), &[seq_len, dim2], &b_data)?;
        Ok((a, b))
    }

    /// Split a 3D tensor `[seq, num_heads, total_dim]` into two parts along last dim.
    fn split_head_dim(
        tensor: &CudaTensor<T>,
        dim1: usize,
        dim2: usize,
    ) -> Result<(CudaTensor<T>, CudaTensor<T>)> {
        let seq_len = tensor.shape()[0];
        let num_heads = tensor.shape()[1];
        let total = tensor.shape()[2];
        assert_eq!(total, dim1 + dim2, "split_head_dim: dim mismatch");

        let data = tensor.to_vec()?;
        let mut a_data = vec![T::default(); seq_len * num_heads * dim1];
        let mut b_data = vec![T::default(); seq_len * num_heads * dim2];

        for s in 0..seq_len {
            for h in 0..num_heads {
                let src_offset = (s * num_heads + h) * total;
                let a_offset = (s * num_heads + h) * dim1;
                let b_offset = (s * num_heads + h) * dim2;
                a_data[a_offset..a_offset + dim1]
                    .copy_from_slice(&data[src_offset..src_offset + dim1]);
                b_data[b_offset..b_offset + dim2]
                    .copy_from_slice(&data[src_offset + dim1..src_offset + total]);
            }
        }

        let a = CudaTensor::from_slice(tensor.context(), &[seq_len, num_heads, dim1], &a_data)?;
        let b = CudaTensor::from_slice(tensor.context(), &[seq_len, num_heads, dim2], &b_data)?;
        Ok((a, b))
    }

    /// Concatenate two 3D tensors along the last dimension.
    fn concat_head_dim(a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>> {
        let seq_len = a.shape()[0];
        let num_heads = a.shape()[1];
        let dim1 = a.shape()[2];
        let dim2 = b.shape()[2];
        assert_eq!(seq_len, b.shape()[0]);
        assert_eq!(num_heads, b.shape()[1]);

        let total = dim1 + dim2;
        let a_data = a.to_vec()?;
        let b_data = b.to_vec()?;
        let mut out_data = vec![T::default(); seq_len * num_heads * total];

        for s in 0..seq_len {
            for h in 0..num_heads {
                let out_offset = (s * num_heads + h) * total;
                let a_offset = (s * num_heads + h) * dim1;
                let b_offset = (s * num_heads + h) * dim2;
                out_data[out_offset..out_offset + dim1]
                    .copy_from_slice(&a_data[a_offset..a_offset + dim1]);
                out_data[out_offset + dim1..out_offset + total]
                    .copy_from_slice(&b_data[b_offset..b_offset + dim2]);
            }
        }

        CudaTensor::from_slice(a.context(), &[seq_len, num_heads, total], &out_data)
    }

    /// Broadcast `[seq, 1, rope_dim]` to `[seq, num_heads, rope_dim]`.
    fn broadcast_kv_rope(k_rope: &CudaTensor<T>, num_heads: usize) -> Result<CudaTensor<T>> {
        let seq_len = k_rope.shape()[0];
        let rope_dim = k_rope.shape()[2];
        let data = k_rope.to_vec()?;

        let mut out_data = vec![T::default(); seq_len * num_heads * rope_dim];
        for s in 0..seq_len {
            let src = &data[s * rope_dim..(s + 1) * rope_dim];
            for h in 0..num_heads {
                let dst_offset = (s * num_heads + h) * rope_dim;
                out_data[dst_offset..dst_offset + rope_dim].copy_from_slice(src);
            }
        }

        CudaTensor::from_slice(k_rope.context(), &[seq_len, num_heads, rope_dim], &out_data)
    }

    /// Pad V from `[seq, num_heads, v_head_dim]` to `[seq, num_heads, qk_head_dim]`.
    fn pad_v_to_qk_dim(v: &CudaTensor<T>, qk_head_dim: usize) -> Result<CudaTensor<T>> {
        let seq_len = v.shape()[0];
        let num_heads = v.shape()[1];
        let v_dim = v.shape()[2];
        if v_dim == qk_head_dim {
            return Ok(v.slice_view(0, v.shape()));
        }

        let v_data = v.to_vec()?;
        let mut out_data = vec![T::default(); seq_len * num_heads * qk_head_dim];
        for s in 0..seq_len {
            for h in 0..num_heads {
                let src_offset = (s * num_heads + h) * v_dim;
                let dst_offset = (s * num_heads + h) * qk_head_dim;
                out_data[dst_offset..dst_offset + v_dim]
                    .copy_from_slice(&v_data[src_offset..src_offset + v_dim]);
                // Remaining elements stay as default (zero)
            }
        }

        CudaTensor::from_slice(v.context(), &[seq_len, num_heads, qk_head_dim], &out_data)
    }

    /// Truncate attention output from `[seq, num_heads, qk_head_dim]` to
    /// `[seq, num_heads, v_head_dim]`.
    fn truncate_attn_output(attn_out: &CudaTensor<T>, v_head_dim: usize) -> Result<CudaTensor<T>> {
        let seq_len = attn_out.shape()[0];
        let num_heads = attn_out.shape()[1];
        let qk_dim = attn_out.shape()[2];
        if qk_dim == v_head_dim {
            return Ok(attn_out.slice_view(0, attn_out.shape()));
        }

        let data = attn_out.to_vec()?;
        let mut out_data = vec![T::default(); seq_len * num_heads * v_head_dim];
        for s in 0..seq_len {
            for h in 0..num_heads {
                let src_offset = (s * num_heads + h) * qk_dim;
                let dst_offset = (s * num_heads + h) * v_head_dim;
                out_data[dst_offset..dst_offset + v_head_dim]
                    .copy_from_slice(&data[src_offset..src_offset + v_head_dim]);
            }
        }

        CudaTensor::from_slice(
            attn_out.context(),
            &[seq_len, num_heads, v_head_dim],
            &out_data,
        )
    }

    /// MLA attention forward pass.
    #[allow(clippy::too_many_lines)]
    fn forward_mla_attention(
        &self,
        hidden: &CudaTensor<T>,
        weights: &DeepSeekAttentionWeights<T>,
        layer_idx: usize,
        kv_cache: &mut KvCache<T>,
        position_offset: usize,
    ) -> Result<CudaTensor<T>> {
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
        // k_rope is [seq, qk_rope_dim], reshape to [seq, 1, qk_rope_dim] for RoPE
        let k_rope = k_rope.reshape(&[seq_len, 1, qk_rope_dim]);
        let q_rope =
            apply_rope_interleaved(&q_rope, &self.cos_cache, &self.sin_cache, position_offset)?;
        let k_rope =
            apply_rope_interleaved(&k_rope, &self.cos_cache, &self.sin_cache, position_offset)?;

        // Broadcast k_rope from [seq, 1, rope_dim] → [seq, num_heads, rope_dim]
        let k_rope = Self::broadcast_kv_rope(&k_rope, num_heads)?;

        // --- Concatenate nope and rope ---
        let q = Self::concat_head_dim(&q_nope, &q_rope)?;
        let k = Self::concat_head_dim(&k_nope, &k_rope)?;

        // --- Pad V to qk_head_dim for fused attention ---
        let v_padded = Self::pad_v_to_qk_dim(&v, qk_head_dim)?;

        // --- KV cache ---
        kv_cache.append(layer_idx, &k, &v_padded)?;
        let total_len = kv_cache.current_len() + seq_len;
        let (k_full, v_full) = kv_cache.get_up_to(layer_idx, total_len);

        // --- Attention ---
        let attn_output = if seq_len == 1 {
            fused_attention_decode(&q, &k_full, &v_full, None)?
        } else {
            fused_attention_prefill(&q, &k_full, &v_full, kv_cache.current_len(), None)?
        };

        // --- Truncate attention output back to v_head_dim ---
        let attn_output = Self::truncate_attn_output(&attn_output, v_head_dim)?;

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
        hidden: &CudaTensor<T>,
        weights: &DeepSeekAttentionWeights<T>,
        layer_idx: usize,
        kv_cache: &mut KvCache<T>,
    ) -> Result<CudaTensor<T>> {
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

        let kv = linear(hidden, &weights.kv_a_proj_with_mqa)?;
        let (k_compressed, k_rope) = Self::split_last_dim(&kv, kv_lora_rank, qk_rope_dim)?;
        let k_compressed = rms_norm(
            &k_compressed,
            &weights.kv_a_layernorm,
            self.config.rms_norm_eps,
        )?;
        let kv_decompressed = linear(&k_compressed, &weights.kv_b_proj)?;
        let kv_decompressed = kv_decompressed.reshape(&[1, num_heads, qk_nope_dim + v_head_dim]);
        let (k_nope, v) = Self::split_head_dim(&kv_decompressed, qk_nope_dim, v_head_dim)?;

        let k_rope = k_rope.reshape(&[1, 1, qk_rope_dim]);
        let position = kv_cache.current_position();
        let q_rope =
            apply_rope_interleaved_indirect(&q_rope, &self.cos_cache, &self.sin_cache, position)?;
        let k_rope =
            apply_rope_interleaved_indirect(&k_rope, &self.cos_cache, &self.sin_cache, position)?;
        let k_rope = Self::broadcast_kv_rope(&k_rope, num_heads)?;

        let q = Self::concat_head_dim(&q_nope, &q_rope)?;
        let k = Self::concat_head_dim(&k_nope, &k_rope)?;
        let v_padded = Self::pad_v_to_qk_dim(&v, qk_head_dim)?;

        kv_cache.append_indirect(layer_idx, &k, &v_padded)?;
        let (k_full, v_full) = kv_cache.full_buffers(layer_idx);
        let total_len = kv_cache.current_total_len();
        let attn_output = fused_attention_decode_indirect(
            &q,
            k_full,
            v_full,
            total_len,
            kv_cache.graph_max_seq_len(),
            None,
        )?;

        let attn_output = Self::truncate_attn_output(&attn_output, v_head_dim)?;
        let attn_output = attn_output.reshape(&[1, num_heads * v_head_dim]);
        let mut out = linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    // --- FFN ---

    #[allow(clippy::unused_self)]
    fn forward_mlp(
        &self,
        hidden: &CudaTensor<T>,
        weights: &DenseMlpWeights<T>,
    ) -> Result<CudaTensor<T>> {
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
        hidden: &CudaTensor<T>,
        weights: &DenseMlpWeights<T>,
    ) -> Result<CudaTensor<T>> {
        let gate = linear(hidden, &weights.gate_proj)?;
        let up = linear(hidden, &weights.up_proj)?;
        let intermediate = swiglu(&gate, &up)?;
        linear(&intermediate, &weights.down_proj)
    }

    fn forward_moe(
        &self,
        hidden: &CudaTensor<T>,
        moe_weights: &MoeWeights<T>,
    ) -> Result<CudaTensor<T>> {
        let num_experts = moe_weights.experts.len();

        let mut routed_output = infernum::cuda::moe::moe_forward_sigmoid(
            hidden,
            &moe_weights.gate_weight,
            &moe_weights.e_score_correction_bias,
            num_experts,
            self.config.num_experts_per_tok,
            self.config.n_group,
            self.config.topk_group,
            self.config.norm_topk_prob,
            self.config.routed_scaling_factor,
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

    fn forward_ffn(&self, hidden: &CudaTensor<T>, ffn: &FfnWeights<T>) -> Result<CudaTensor<T>> {
        match ffn {
            FfnWeights::Dense(mlp) => self.forward_mlp(hidden, mlp),
            FfnWeights::Moe(moe) => self.forward_moe(hidden, moe),
        }
    }

    // --- Layer forward ---

    fn forward_layer_kv(
        &self,
        hidden: &CudaTensor<T>,
        layer: &DeepSeekLayerWeights<T>,
        layer_idx: usize,
        kv_cache: &mut KvCache<T>,
        position_offset: usize,
    ) -> Result<CudaTensor<T>> {
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
        hidden: &CudaTensor<T>,
        layer: &DeepSeekLayerWeights<T>,
        layer_idx: usize,
        kv_cache: &mut KvCache<T>,
    ) -> Result<CudaTensor<T>> {
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

    // --- Public forward methods ---

    /// Forward pass with KV cache (prefill phase)
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward_with_kv_cache(
        &self,
        input_ids: &[u32],
        kv_cache: &mut KvCache<T>,
    ) -> Result<CudaTensor<f32>> {
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
    pub fn forward_next_token(
        &self,
        token_id: u32,
        kv_cache: &mut KvCache<T>,
    ) -> Result<CudaTensor<f32>> {
        self.forward_with_kv_cache(&[token_id], kv_cache)
    }

    /// Decode-phase forward pass reading the token from a GPU buffer.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward_next_token_device(
        &self,
        token_id_gpu: &CudaSlice<u32>,
        kv_cache: &mut KvCache<T>,
    ) -> Result<CudaTensor<f32>> {
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
        kv_cache: &mut KvCache<T>,
    ) -> Result<CudaTensor<f32>> {
        let mut hidden = self.embed_from_device(token_id_gpu)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer_kv_indirect(&hidden, layer, layer_idx, kv_cache)?;
        }

        rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;

        let last_hidden = hidden.reshape(&[1, self.config.hidden_size]);
        self.lm_head_forward(&last_hidden)
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
            match T::DTYPE {
                infernum::dtype::DType::F32 => Ok(reinterpret_tensor(output_f32)),
                infernum::dtype::DType::BF16 => {
                    Ok(reinterpret_tensor(cast_f32_to_bf16(&output_f32)?))
                }
                other => panic!("Quantized matmul not supported for dtype {other}"),
            }
        }
    }
}

// --- Model trait implementation ---

#[allow(private_bounds)]
impl<T> infernum::Model for DeepSeekModel<T>
where
    T: TensorDType + DeviceRepr + GemmScalar + Default + ValidAsZeroBits + MaybeNcclType,
    CudaBlas: Gemm<T>,
{
    type CacheDtype = T;

    fn config(&self) -> infernum::ModelConfig {
        let config = &self.config;
        infernum::ModelConfig {
            num_layers: config.num_hidden_layers,
            max_seq_len: config.max_position_embeddings,
            // KV cache stores decompressed KV: tp_num_heads × qk_head_dim (Phase 1)
            num_kv_heads: self.tp_num_heads,
            head_dim: config.qk_head_dim(),
            eos_token_id: config.eos_token_id,
        }
    }

    fn devices(&self) -> Vec<&CudaContext> {
        vec![&self.ctx]
    }

    fn forward(&self, input_ids: &[u32]) -> Result<CudaTensor<f32>> {
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

    fn forward_with_kv_cache(
        &self,
        input_ids: &[u32],
        kv_caches: &mut [KvCache<T>],
    ) -> Result<CudaTensor<f32>> {
        self.forward_with_kv_cache(input_ids, &mut kv_caches[0])
    }

    fn forward_next_token(
        &self,
        token_id: u32,
        kv_caches: &mut [KvCache<T>],
    ) -> Result<CudaTensor<f32>> {
        self.forward_next_token(token_id, &mut kv_caches[0])
    }

    fn forward_next_token_device(
        &self,
        token_id_gpu: &CudaSlice<u32>,
        kv_caches: &mut [KvCache<T>],
    ) -> Result<CudaTensor<f32>> {
        self.forward_next_token_device(token_id_gpu, &mut kv_caches[0])
    }

    fn forward_next_token_indirect(
        &self,
        token_id_gpu: &CudaSlice<u32>,
        kv_caches: &mut [KvCache<T>],
    ) -> Result<CudaTensor<f32>> {
        self.forward_next_token_indirect(token_id_gpu, &mut kv_caches[0])
    }
}

#[cfg(feature = "nccl")]
#[allow(private_bounds)]
impl<T> infernum::ShardedLoadable for DeepSeekModel<T>
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
