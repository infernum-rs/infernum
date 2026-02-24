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
    apply_rope_interleaved_indirect, broadcast_to_heads, cast_to_f32, concat_inner_dim,
    embedding_gather, embedding_gather_from_device, fused_attention_decode,
    fused_attention_decode_indirect, fused_attention_prefill, gather_paged_kv, linear,
    matmul_bf16_f32, pad_inner_dim, paged_attention_decode, paged_attention_decode_indirect,
    precompute_rope_cache, precompute_rope_cache_scaled, reinterpret_tensor, rms_norm,
    rms_norm_inplace, split_inner_dim, swiglu, transpose_2d, GemmScalar, LinearWeight, RopeScaling,
};
use infernum::cuda::{
    BatchedGraphInputs, CudaBlas, CudaContext, CudaSlice, CudaTensor, DeviceRepr, Gemm, GpuConfig,
    KvCache, PagedKvCache, ValidAsZeroBits,
};
#[cfg(feature = "nccl")]
use infernum::cuda::{NcclCommunicator, NcclType, ShardConfig, ShardStrategy};
use infernum::dtype::TensorDType;
use infernum::tensor::Tensor;
use infernum::weights::{SafeTensorsLoader, WeightLoader};
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

// --- kv_b_proj splitting ---

/// Split a pre-transposed dense `kv_b_proj` weight into K-nope and V portions.
///
/// Input shape: `(kv_lora_rank, num_heads * (qk_nope_dim + v_head_dim))` (pre-transposed).
/// Columns are interleaved per head: for head `h`, columns
/// `[h*stride .. h*stride+qk_nope_dim]` are K-nope, followed by `v_head_dim` V columns.
///
/// Returns `(kv_b_proj_k, kv_b_proj_v, kv_b_proj_k_t)`:
/// - `kv_b_proj_k`: `(kv_lora_rank, num_heads * qk_nope_dim)` — K-nope decompression
/// - `kv_b_proj_v`: `(kv_lora_rank, num_heads * v_head_dim)` — V decompression
/// - `kv_b_proj_k_t`: `(num_heads * qk_nope_dim, kv_lora_rank)` — transposed K for Q absorption
fn split_kv_b_proj_dense<T: TensorDType + DeviceRepr + Default>(
    ctx: &CudaContext,
    weight: &CudaTensor<T>,
    num_heads: usize,
    qk_nope_dim: usize,
    v_head_dim: usize,
) -> Result<(CudaTensor<T>, CudaTensor<T>, CudaTensor<T>)> {
    let shape = weight.shape();
    let kv_lora_rank = shape[0];
    let total_cols = shape[1];
    let stride = qk_nope_dim + v_head_dim;
    assert_eq!(
        total_cols,
        num_heads * stride,
        "split_kv_b_proj_dense: expected {} columns, got {total_cols}",
        num_heads * stride
    );

    let data = weight.to_vec()?;

    // Extract K-nope columns: shape (kv_lora_rank, num_heads * qk_nope_dim)
    let k_cols = num_heads * qk_nope_dim;
    let mut k_data = vec![T::default(); kv_lora_rank * k_cols];
    for row in 0..kv_lora_rank {
        for h in 0..num_heads {
            let src_offset = row * total_cols + h * stride;
            let dst_offset = row * k_cols + h * qk_nope_dim;
            k_data[dst_offset..dst_offset + qk_nope_dim]
                .copy_from_slice(&data[src_offset..src_offset + qk_nope_dim]);
        }
    }

    // Extract V columns: shape (kv_lora_rank, num_heads * v_head_dim)
    let v_cols = num_heads * v_head_dim;
    let mut v_data = vec![T::default(); kv_lora_rank * v_cols];
    for row in 0..kv_lora_rank {
        for h in 0..num_heads {
            let src_offset = row * total_cols + h * stride + qk_nope_dim;
            let dst_offset = row * v_cols + h * v_head_dim;
            v_data[dst_offset..dst_offset + v_head_dim]
                .copy_from_slice(&data[src_offset..src_offset + v_head_dim]);
        }
    }

    // Transpose K: (kv_lora_rank, num_heads * qk_nope_dim) → (num_heads * qk_nope_dim, kv_lora_rank)
    let mut k_t_data = vec![T::default(); kv_lora_rank * k_cols];
    for row in 0..kv_lora_rank {
        for col in 0..k_cols {
            k_t_data[col * kv_lora_rank + row] = k_data[row * k_cols + col];
        }
    }

    let k_tensor = CudaTensor::from_slice(ctx, &[kv_lora_rank, k_cols], &k_data)?;
    let v_tensor = CudaTensor::from_slice(ctx, &[kv_lora_rank, v_cols], &v_data)?;
    let k_t_tensor = CudaTensor::from_slice(ctx, &[k_cols, kv_lora_rank], &k_t_data)?;

    Ok((k_tensor, v_tensor, k_t_tensor))
}

// --- Weight structures ---

/// MLA attention weights (shared by dense and MoE layers)
struct DeepSeekAttentionWeights<T: TensorDType> {
    q_a_proj: LinearWeight<T>,
    q_a_layernorm: CudaTensor<T>,
    q_b_proj: LinearWeight<T>,
    kv_a_proj_with_mqa: LinearWeight<T>,
    kv_a_layernorm: CudaTensor<T>,
    kv_b_proj: LinearWeight<T>,
    /// K-nope decompression columns, pre-transposed: `(kv_lora_rank, num_heads * qk_nope_dim)`
    kv_b_proj_k: LinearWeight<T>,
    /// V decompression columns, pre-transposed: `(kv_lora_rank, num_heads * v_head_dim)`
    kv_b_proj_v: LinearWeight<T>,
    /// Transpose of `kv_b_proj_k`: `(num_heads * qk_nope_dim, kv_lora_rank)` for Q absorption
    kv_b_proj_k_t: LinearWeight<T>,
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
            let kv_b_proj = load_linear::<T>(
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
                    (
                        LinearWeight::Dense(k),
                        LinearWeight::Dense(v),
                        LinearWeight::Dense(k_t),
                    )
                }
                LinearWeight::Quantized(_) => {
                    panic!("Quantized kv_b_proj splitting not yet implemented")
                }
            };

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
                kv_b_proj,
                kv_b_proj_k,
                kv_b_proj_v,
                kv_b_proj_k_t,
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
            let kv_b_proj = load_linear_sharded::<T>(
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
                    (
                        LinearWeight::Dense(k),
                        LinearWeight::Dense(v),
                        LinearWeight::Dense(k_t),
                    )
                }
                LinearWeight::Quantized(_) => {
                    panic!("Quantized kv_b_proj splitting not yet implemented")
                }
            };

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
                kv_b_proj,
                kv_b_proj_k,
                kv_b_proj_v,
                kv_b_proj_k_t,
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
        tensor: &CudaTensor<T>,
        dim1: usize,
        dim2: usize,
    ) -> Result<(CudaTensor<T>, CudaTensor<T>)> {
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
    fn concat_head_dim(a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>> {
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
    fn broadcast_kv_rope(k_rope: &CudaTensor<T>, num_heads: usize) -> Result<CudaTensor<T>> {
        broadcast_to_heads(k_rope, num_heads)
    }

    /// Pad V from `[seq, num_heads, v_head_dim]` to `[seq, num_heads, qk_head_dim]`.
    fn pad_v_to_qk_dim(v: &CudaTensor<T>, qk_head_dim: usize) -> Result<CudaTensor<T>> {
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
    fn truncate_attn_output(attn_out: &CudaTensor<T>, v_head_dim: usize) -> Result<CudaTensor<T>> {
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
            fused_attention_decode(&q, &k_full, &v_full, None, None, None)?
        } else {
            fused_attention_prefill(
                &q,
                &k_full,
                &v_full,
                kv_cache.current_len(),
                None,
                None,
                None,
            )?
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
            None,
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

    // --- Paged KV cache support ---

    /// MLA attention for single-sequence prefill with paged KV cache.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn forward_mla_attention_paged_prefill(
        &self,
        hidden: &CudaTensor<T>,
        weights: &DeepSeekAttentionWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        block_table: &BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<CudaTensor<T>> {
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
        let k_rope = Self::broadcast_kv_rope(&k_rope, num_heads)?;

        // Concatenate nope + rope
        let q = Self::concat_head_dim(&q_nope, &q_rope)?;
        let k = Self::concat_head_dim(&k_nope, &k_rope)?;

        // Pad V to qk_head_dim
        let v_padded = Self::pad_v_to_qk_dim(&v, qk_head_dim)?;

        // Write K/V into paged cache
        paged_kv.append_paged(layer_idx, block_table, &k, &v_padded, start_pos)?;

        // Gather contiguous K/V for fused prefill kernel
        let mut gather_table = block_table.clone();
        gather_table.advance(seq_len);
        let (k_contig, v_contig) = gather_paged_kv(paged_kv, layer_idx, &gather_table)?;

        let attn_output =
            fused_attention_prefill(&q, &k_contig, &v_contig, start_pos, None, None, None)?;

        // Truncate back to v_head_dim and project
        let attn_output = Self::truncate_attn_output(&attn_output, v_head_dim)?;
        let attn_output = attn_output.reshape(&[seq_len, num_heads * v_head_dim]);
        let mut out = linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    /// MLA attention for single-token decode with paged KV cache.
    fn forward_mla_attention_paged_decode(
        &self,
        hidden: &CudaTensor<T>,
        weights: &DeepSeekAttentionWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        block_table: &BlockTable,
        position: usize,
    ) -> Result<CudaTensor<T>> {
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

        // KV joint projection
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

        // RoPE (interleaved)
        let k_rope = k_rope.reshape(&[1, 1, qk_rope_dim]);
        let q_rope = apply_rope_interleaved(&q_rope, &self.cos_cache, &self.sin_cache, position)?;
        let k_rope = apply_rope_interleaved(&k_rope, &self.cos_cache, &self.sin_cache, position)?;
        let k_rope = Self::broadcast_kv_rope(&k_rope, num_heads)?;

        // Concatenate nope + rope
        let q = Self::concat_head_dim(&q_nope, &q_rope)?;
        let k = Self::concat_head_dim(&k_nope, &k_rope)?;

        // Pad V to qk_head_dim
        let v_padded = Self::pad_v_to_qk_dim(&v, qk_head_dim)?;

        // Append to paged cache
        paged_kv.append_paged(layer_idx, block_table, &k, &v_padded, position)?;

        // Paged attention decode
        let mut table_with_current = block_table.clone();
        table_with_current.advance(1);
        let (k_pool, v_pool) = paged_kv.get_pools(layer_idx);
        let attn_output = paged_attention_decode(
            &self.ctx,
            &q,
            k_pool,
            v_pool,
            &[table_with_current],
            paged_kv.block_size(),
            None,
            None,
        )?;

        // Truncate back to v_head_dim and project
        let attn_output = Self::truncate_attn_output(&attn_output, v_head_dim)?;
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
        hidden: &CudaTensor<T>,
        layer: &DeepSeekLayerWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        block_table: &BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<CudaTensor<T>> {
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
        hidden: &CudaTensor<T>,
        layer: &DeepSeekLayerWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        block_table: &BlockTable,
        position: usize,
    ) -> Result<CudaTensor<T>> {
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
        hidden: &CudaTensor<T>,
        weights: &DeepSeekAttentionWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        graph_inputs: &BatchedGraphInputs,
        max_seq_len: usize,
    ) -> Result<CudaTensor<T>> {
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
            kv_decompressed.reshape(&[batch_size, num_heads, qk_nope_dim + v_head_dim]);
        let (k_nope, v) = Self::split_head_dim(&kv_decompressed, qk_nope_dim, v_head_dim)?;

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
        let k_rope = Self::broadcast_kv_rope(&k_rope, num_heads)?;

        // Concatenate nope + rope
        let q = Self::concat_head_dim(&q_nope, &q_rope)?;
        let k = Self::concat_head_dim(&k_nope, &k_rope)?;

        // Pad V to qk_head_dim
        let v_padded = Self::pad_v_to_qk_dim(&v, qk_head_dim)?;

        // Batched paged K/V append from GPU-resident block tables + positions
        paged_kv.append_paged_batched(
            layer_idx,
            &k,
            &v_padded,
            graph_inputs.block_tables(),
            graph_inputs.positions(),
            batch_size,
            graph_inputs.max_blocks_per_seq(),
        )?;

        // Paged attention decode from GPU-resident block tables + seq_lens
        let (k_pool, v_pool) = paged_kv.get_pools(layer_idx);
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
            None,
            None,
        )?;

        // Truncate back to v_head_dim and project
        let attn_output = Self::truncate_attn_output(&attn_output, v_head_dim)?;
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
        hidden: &CudaTensor<T>,
        layer: &DeepSeekLayerWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        graph_inputs: &BatchedGraphInputs,
        max_seq_len: usize,
    ) -> Result<CudaTensor<T>> {
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
        paged_kvs: &mut [PagedKvCache<T>],
        max_seq_len: usize,
    ) -> Result<CudaTensor<f32>> {
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
            // Compressed KV cache: single latent vector per token
            num_kv_heads: 1,
            head_dim: config.kv_lora_rank + config.qk_rope_head_dim,
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

    fn forward_prefill_paged(
        &self,
        input_ids: &[u32],
        paged_kvs: &mut [PagedKvCache<T>],
        block_table: &BlockTable,
        start_pos: usize,
    ) -> Result<CudaTensor<f32>> {
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
        paged_kvs: &mut [PagedKvCache<T>],
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<CudaTensor<f32>> {
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
            unsafe { CudaTensor::<f32>::uninit(&self.ctx, &[batch_size, vocab_size])? };
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
        paged_kvs: &mut [PagedKvCache<T>],
        max_seq_len: usize,
    ) -> Result<CudaTensor<f32>> {
        self.forward_batch_decode_indirect(graph_inputs, paged_kvs, max_seq_len)
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
        assert_eq!(v.shape(), &[kv_lora_rank, num_heads * v_head_dim]);
        assert_eq!(k_t.shape(), &[num_heads * qk_nope_dim, kv_lora_rank]);
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

        let k_data = k.to_vec().unwrap();
        let v_data = v.to_vec().unwrap();

        // Reconstruct the original by interleaving K and V columns back
        let mut reconstructed = vec![0.0_f32; kv_lora_rank * total_cols];
        for row in 0..kv_lora_rank {
            for h in 0..num_heads {
                // K columns
                for d in 0..qk_nope_dim {
                    reconstructed[row * total_cols + h * stride + d] =
                        k_data[row * (num_heads * qk_nope_dim) + h * qk_nope_dim + d];
                }
                // V columns
                for d in 0..v_head_dim {
                    reconstructed[row * total_cols + h * stride + qk_nope_dim + d] =
                        v_data[row * (num_heads * v_head_dim) + h * v_head_dim + d];
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
        assert_eq!(k_t.shape(), &[k_cols, kv_lora_rank]);

        let k_data = k.to_vec().unwrap();
        let k_t_data = k_t.to_vec().unwrap();

        // Verify k_t[col][row] == k[row][col]
        for row in 0..kv_lora_rank {
            for col in 0..k_cols {
                let k_val = k_data[row * k_cols + col];
                let k_t_val = k_t_data[col * kv_lora_rank + row];
                assert!(
                    (k_val - k_t_val).abs() < 1e-6,
                    "Transpose mismatch at ({row}, {col}): k={k_val}, k_t={k_t_val}"
                );
            }
        }
    }
}
