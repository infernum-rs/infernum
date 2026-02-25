//! Qwen model implementation

#![allow(
    clippy::struct_field_names,
    clippy::no_effect_underscore_binding,
    clippy::doc_markdown,
    unused_mut
)]

use std::path::Path;

use infernum::cuda::block_allocator::BlockTable;
use infernum::cuda::ops::{
    add_inplace, add_rmsnorm, apply_rope, apply_rope_batched, apply_rope_batched_indirect,
    bias_add_inplace, cast_to_f32, embedding_gather, embedding_gather_from_device,
    fused_attention_prefill, gather_paged_kv, linear, matmul, matmul_bf16_f32, mul,
    paged_attention_decode, paged_attention_decode_indirect, precompute_rope_cache,
    precompute_rope_cache_scaled, reinterpret_tensor, rms_norm, rms_norm_inplace, split_inner_dim,
    swiglu, transpose_2d, GemmScalar, LinearWeight, RopeScaling,
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

use crate::QwenConfig;

// --- NCCL conditional trait bounds (same pattern as infernum-llama) ---

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

fn split_gate_up<T: TensorDType + DeviceRepr + ValidAsZeroBits>(
    fused: &CudaTensor<T>,
    intermediate_size: usize,
) -> Result<(CudaTensor<T>, CudaTensor<T>)> {
    split_inner_dim(fused, intermediate_size, intermediate_size)
}

fn split_kv<T: TensorDType + DeviceRepr + ValidAsZeroBits>(
    fused: &CudaTensor<T>,
    kv_dim: usize,
) -> Result<(CudaTensor<T>, CudaTensor<T>)> {
    split_inner_dim(fused, kv_dim, kv_dim)
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

// --- Weight structures ---

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

struct QwenAttentionWeights<T: TensorDType> {
    q_proj: LinearWeight<T>,
    kv_proj: KvProjWeight<T>,
    o_proj: LinearWeight<T>,
    /// Q/K/V biases (Qwen2, some Qwen3-MoE)
    q_bias: Option<CudaTensor<T>>,
    k_bias: Option<CudaTensor<T>>,
    v_bias: Option<CudaTensor<T>>,
    /// QK-norm weights (Qwen3): RMSNorm per-head before RoPE
    q_norm: Option<CudaTensor<T>>,
    k_norm: Option<CudaTensor<T>>,
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

struct QwenMlpWeights<T: TensorDType> {
    gate_up: GateUpWeight<T>,
    down_proj: LinearWeight<T>,
}

struct MoeExpertWeights<T: TensorDType> {
    mlp: QwenMlpWeights<T>,
}

#[allow(clippy::large_enum_variant)]
enum QwenFfnWeights<T: TensorDType> {
    Dense(Box<QwenMlpWeights<T>>),
    Moe {
        gate: CudaTensor<T>,
        experts: Vec<MoeExpertWeights<T>>,
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        shared_expert: Option<Box<QwenMlpWeights<T>>>,
        shared_expert_gate: Option<CudaTensor<T>>,
    },
}

struct QwenLayerWeights<T: TensorDType> {
    input_layernorm: CudaTensor<T>,
    attention: QwenAttentionWeights<T>,
    post_attention_layernorm: CudaTensor<T>,
    ffn: QwenFfnWeights<T>,
}

/// Complete Qwen model, generic over the compute dtype `T`.
pub struct QwenModel<T: TensorDType> {
    config: QwenConfig,
    ctx: CudaContext,
    #[allow(dead_code)]
    gpu_config: GpuConfig,

    #[cfg(feature = "nccl")]
    nccl_comm: Option<NcclCommunicator>,

    tp_num_heads: usize,
    tp_num_kv_heads: usize,

    embed_tokens: CudaTensor<T>,
    layers: Vec<QwenLayerWeights<T>>,
    norm: CudaTensor<T>,
    lm_head: LinearWeight<T>,

    cos_cache: CudaTensor<T>,
    sin_cache: CudaTensor<T>,
}

#[allow(private_bounds)]
impl<T> QwenModel<T>
where
    T: TensorDType + DeviceRepr + GemmScalar + Default + ValidAsZeroBits + MaybeNcclType,
    CudaBlas: Gemm<T>,
{
    /// Load a Qwen model from a directory containing SafeTensors and config.json
    ///
    /// # Errors
    /// Returns an error if loading fails
    pub fn from_pretrained(ctx: &CudaContext, model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();
        let config_path = model_path.join("config.json");
        let config = QwenConfig::from_file(&config_path)?;
        let loader = SafeTensorsLoader::from_directory(model_path)?;
        Self::load_weights(ctx, config, &loader)
    }

    /// Load a Qwen model with tensor-parallel sharding across multiple GPUs.
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
        let config = QwenConfig::from_file(&config_path)?;
        let loader = SafeTensorsLoader::from_directory(model_path)?;
        Self::load_weights_sharded(ctx, config, &loader, gpu_config, nccl_comm)
    }

    #[allow(clippy::too_many_lines)]
    fn load_weights(
        ctx: &CudaContext,
        config: QwenConfig,
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

        fn load_dense_mlp<T: TensorDType + DeviceRepr>(
            ctx: &CudaContext,
            loader: &impl WeightLoader,
            prefix: &str,
            intermediate_size: usize,
            qc: Option<&crate::config::QuantizationConfig>,
        ) -> Result<QwenMlpWeights<T>> {
            let gate = load_linear::<T>(ctx, loader, &format!("{prefix}.gate_proj.weight"), qc)?;
            let up = load_linear::<T>(ctx, loader, &format!("{prefix}.up_proj.weight"), qc)?;
            let gate_up = match (gate, up) {
                (LinearWeight::Dense(g), LinearWeight::Dense(u)) => GateUpWeight::Fused {
                    weight: concat_weights(&g, &u)?,
                    intermediate_size,
                },
                (g, u) => GateUpWeight::Separate {
                    gate_proj: Box::new(g),
                    up_proj: Box::new(u),
                },
            };
            Ok(QwenMlpWeights {
                gate_up,
                down_proj: load_linear::<T>(
                    ctx,
                    loader,
                    &format!("{prefix}.down_proj.weight"),
                    qc,
                )?,
            })
        }

        fn load_moe_weights<T: TensorDType + DeviceRepr>(
            ctx: &CudaContext,
            loader: &impl WeightLoader,
            layer_prefix: &str,
            config: &QwenConfig,
            qc: Option<&crate::config::QuantizationConfig>,
        ) -> Result<QwenFfnWeights<T>> {
            let num_experts = config.num_experts.expect("MoE requires num_experts");
            let num_experts_per_tok = config
                .num_experts_per_tok
                .expect("MoE requires num_experts_per_tok");
            let expert_inter = config.moe_expert_intermediate_size();

            // Qwen MoE router: mlp.gate.weight [num_experts, hidden_size]
            let gate_name = format!("{layer_prefix}.mlp.gate.weight");
            let gate_f32 = loader.load_f32(ctx, &gate_name)?;
            let gate_transposed = pretranspose_weight(&gate_f32)?;
            let gate = if T::DTYPE == infernum::dtype::DType::F32 {
                reinterpret_tensor(gate_transposed)
            } else {
                let data_f32 = gate_transposed.to_vec()?;
                let data_t: Vec<T> = data_f32.iter().map(|&v| T::from_f32(v)).collect();
                CudaTensor::from_slice(ctx, gate_transposed.shape(), &data_t)?
            };

            // Per-expert MLPs: mlp.experts.{E}.{gate,up,down}_proj.weight
            let mut experts = Vec::with_capacity(num_experts);
            for e in 0..num_experts {
                let ep = format!("{layer_prefix}.mlp.experts.{e}");
                let mlp = load_dense_mlp::<T>(ctx, loader, &ep, expert_inter, qc)?;
                experts.push(MoeExpertWeights { mlp });
            }

            // Shared expert: mlp.shared_expert.{gate,up,down}_proj.weight
            let shared_expert = if config.has_shared_expert() {
                let shared_inter = config
                    .shared_expert_intermediate_size
                    .expect("shared_expert_intermediate_size required");
                let sp = format!("{layer_prefix}.mlp.shared_expert");
                let mlp = load_dense_mlp::<T>(ctx, loader, &sp, shared_inter, qc)?;
                Some(Box::new(mlp))
            } else {
                None
            };

            // Shared expert gate: mlp.shared_expert_gate.weight [1]
            let shared_expert_gate = if config.has_shared_expert() {
                let gate_name = format!("{layer_prefix}.mlp.shared_expert_gate.weight");
                if loader.contains(&gate_name) {
                    Some(load_typed::<T>(loader, ctx, &gate_name)?)
                } else {
                    None
                }
            } else {
                None
            };

            Ok(QwenFfnWeights::Moe {
                gate,
                experts,
                num_experts_per_tok,
                norm_topk_prob: config.norm_topk_prob,
                shared_expert,
                shared_expert_gate,
            })
        }

        let qc = config.quantization_config.as_ref();

        let embed_tokens = load_typed::<T>(loader, ctx, "model.embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            // Detect Q/K/V biases by checking if .bias tensors exist
            let q_bias_name = format!("{prefix}.self_attn.q_proj.bias");
            let k_bias_name = format!("{prefix}.self_attn.k_proj.bias");
            let v_bias_name = format!("{prefix}.self_attn.v_proj.bias");
            let q_bias = if loader.contains(&q_bias_name) {
                Some(load_typed::<T>(loader, ctx, &q_bias_name)?)
            } else {
                None
            };
            let k_bias = if loader.contains(&k_bias_name) {
                Some(load_typed::<T>(loader, ctx, &k_bias_name)?)
            } else {
                None
            };
            let v_bias = if loader.contains(&v_bias_name) {
                Some(load_typed::<T>(loader, ctx, &v_bias_name)?)
            } else {
                None
            };

            // Detect QK-norm weights
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
            let kv_proj = match (k, v) {
                (LinearWeight::Dense(k_w), LinearWeight::Dense(v_w)) => KvProjWeight::Fused {
                    kv_dim: config.num_kv_heads() * config.head_dim(),
                    weight: concat_weights(&k_w, &v_w)?,
                },
                (k, v) => KvProjWeight::Separate {
                    k_proj: Box::new(k),
                    v_proj: Box::new(v),
                },
            };

            let layer = QwenLayerWeights {
                input_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.input_layernorm.weight"),
                )?,
                attention: QwenAttentionWeights {
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
                    q_bias,
                    k_bias,
                    v_bias,
                    q_norm,
                    k_norm,
                },
                post_attention_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                ffn: if config.is_moe_layer(i) {
                    load_moe_weights::<T>(ctx, loader, &prefix, &config, qc)?
                } else {
                    QwenFfnWeights::Dense(Box::new(load_dense_mlp::<T>(
                        ctx,
                        loader,
                        &format!("{prefix}.mlp"),
                        config.intermediate_size,
                        qc,
                    )?))
                },
            };

            layers.push(layer);
        }

        let norm = load_typed::<T>(loader, ctx, "model.norm.weight")?;

        let lm_head = if config.tie_word_embeddings {
            if qc.is_some() {
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
            }
        } else {
            let lw = load_linear::<T>(ctx, loader, "lm_head.weight", None)?;
            if qc.is_some() {
                if let LinearWeight::Dense(ref w) = lw {
                    let f32_w = cast_to_f32(w)?;
                    let data = f32_w.to_vec()?;
                    let k = f32_w.shape()[0];
                    let n = f32_w.shape()[1];
                    let mut row_major = vec![0.0_f32; data.len()];
                    for r in 0..k {
                        for c in 0..n {
                            row_major[c * k + r] = data[r * n + c];
                        }
                    }
                    LinearWeight::Quantized(QuantizedTensor::from_f32_as_q8(
                        ctx,
                        &[n, k],
                        &row_major,
                    )?)
                } else {
                    lw
                }
            } else {
                lw
            }
        };

        // Precompute RoPE cache (with optional YaRN scaling)
        let (cos_f32, sin_f32) = if let Some(ref rs) = config.rope_scaling {
            let scaling = RopeScaling {
                rope_type: rs.rope_type.clone(),
                factor: rs.factor,
                original_max_position_embeddings: rs.original_max_position_embeddings,
            };
            precompute_rope_cache_scaled(
                ctx,
                config.max_position_embeddings,
                config.head_dim(),
                config.rope_theta,
                &scaling,
            )?
        } else {
            precompute_rope_cache(
                ctx,
                config.max_position_embeddings,
                config.head_dim(),
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

        Ok(Self {
            tp_num_heads: config.num_attention_heads,
            tp_num_kv_heads: config.num_kv_heads(),
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
        })
    }

    // --- Sharded loading ---

    #[cfg(feature = "nccl")]
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    fn load_weights_sharded(
        ctx: &CudaContext,
        config: QwenConfig,
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
                    qt.set_weight_scale(ctx, scale_val[0])?;
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
        assert!(
            config.num_kv_heads().is_multiple_of(world_size),
            "num_kv_heads ({}) must be divisible by world_size ({world_size})",
            config.num_kv_heads()
        );

        let qc = config.quantization_config.as_ref();
        let embed_tokens = load_typed::<T>(loader, ctx, "model.embed_tokens.weight")?;

        let tp_num_heads = config.num_attention_heads / world_size;
        let tp_num_kv_heads = config.num_kv_heads() / world_size;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            // Biases: column-shard Q/K/V biases
            let q_bias_name = format!("{prefix}.self_attn.q_proj.bias");
            let k_bias_name = format!("{prefix}.self_attn.k_proj.bias");
            let v_bias_name = format!("{prefix}.self_attn.v_proj.bias");
            let q_bias = if loader.contains(&q_bias_name) {
                Some(load_typed_sharded::<T>(
                    loader,
                    ctx,
                    &q_bias_name,
                    &shard,
                    ShardStrategy::Column,
                )?)
            } else {
                None
            };
            let k_bias = if loader.contains(&k_bias_name) {
                Some(load_typed_sharded::<T>(
                    loader,
                    ctx,
                    &k_bias_name,
                    &shard,
                    ShardStrategy::Column,
                )?)
            } else {
                None
            };
            let v_bias = if loader.contains(&v_bias_name) {
                Some(load_typed_sharded::<T>(
                    loader,
                    ctx,
                    &v_bias_name,
                    &shard,
                    ShardStrategy::Column,
                )?)
            } else {
                None
            };

            // QK-norm: replicate (per-head, head_dim-sized)
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

            let q_proj = load_linear_sharded::<T>(
                ctx,
                loader,
                &format!("{prefix}.self_attn.q_proj.weight"),
                &shard,
                ShardStrategy::Column,
                qc,
            )?;
            let k_proj = load_linear_sharded::<T>(
                ctx,
                loader,
                &format!("{prefix}.self_attn.k_proj.weight"),
                &shard,
                ShardStrategy::Column,
                qc,
            )?;
            let v_proj = load_linear_sharded::<T>(
                ctx,
                loader,
                &format!("{prefix}.self_attn.v_proj.weight"),
                &shard,
                ShardStrategy::Column,
                qc,
            )?;
            let kv_proj = KvProjWeight::Separate {
                k_proj: Box::new(k_proj),
                v_proj: Box::new(v_proj),
            };

            let o_proj = load_linear_sharded::<T>(
                ctx,
                loader,
                &format!("{prefix}.self_attn.o_proj.weight"),
                &shard,
                ShardStrategy::Row,
                qc,
            )?;

            let ffn = if config.is_moe_layer(i) {
                let num_experts = config.num_experts.expect("MoE requires num_experts");
                let num_experts_per_tok = config
                    .num_experts_per_tok
                    .expect("MoE requires num_experts_per_tok");
                let _expert_inter = config.moe_expert_intermediate_size();

                // Router: replicated
                let gate_name = format!("{prefix}.mlp.gate.weight");
                let gate_f32 = loader.load_f32(ctx, &gate_name)?;
                let gate_transposed = pretranspose_weight(&gate_f32)?;
                let gate = if T::DTYPE == infernum::dtype::DType::F32 {
                    reinterpret_tensor(gate_transposed)
                } else {
                    let data_f32 = gate_transposed.to_vec()?;
                    let data_t: Vec<T> = data_f32.iter().map(|&v| T::from_f32(v)).collect();
                    CudaTensor::from_slice(ctx, gate_transposed.shape(), &data_t)?
                };

                let mut experts = Vec::with_capacity(num_experts);
                for e in 0..num_experts {
                    let ep = format!("{prefix}.mlp.experts.{e}");
                    let gate_proj = load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &format!("{ep}.gate_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?;
                    let up_proj = load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &format!("{ep}.up_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?;
                    let gate_up = GateUpWeight::Separate {
                        gate_proj: Box::new(gate_proj),
                        up_proj: Box::new(up_proj),
                    };
                    let down_proj = load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &format!("{ep}.down_proj.weight"),
                        &shard,
                        ShardStrategy::Row,
                        qc,
                    )?;
                    experts.push(MoeExpertWeights {
                        mlp: QwenMlpWeights { gate_up, down_proj },
                    });
                }

                // Shared expert: sharded like dense MLP
                let shared_expert = if config.has_shared_expert() {
                    let _shared_inter = config
                        .shared_expert_intermediate_size
                        .expect("shared_expert_intermediate_size required");
                    let sp = format!("{prefix}.mlp.shared_expert");
                    let gate_proj = load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &format!("{sp}.gate_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?;
                    let up_proj = load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &format!("{sp}.up_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?;
                    let gate_up = GateUpWeight::Separate {
                        gate_proj: Box::new(gate_proj),
                        up_proj: Box::new(up_proj),
                    };
                    let down_proj = load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &format!("{sp}.down_proj.weight"),
                        &shard,
                        ShardStrategy::Row,
                        qc,
                    )?;
                    Some(Box::new(QwenMlpWeights { gate_up, down_proj }))
                } else {
                    None
                };

                // Shared expert gate: replicated
                let shared_expert_gate = if config.has_shared_expert() {
                    let gate_name = format!("{prefix}.mlp.shared_expert_gate.weight");
                    if loader.contains(&gate_name) {
                        Some(load_typed::<T>(loader, ctx, &gate_name)?)
                    } else {
                        None
                    }
                } else {
                    None
                };

                QwenFfnWeights::Moe {
                    gate,
                    experts,
                    num_experts_per_tok,
                    norm_topk_prob: config.norm_topk_prob,
                    shared_expert,
                    shared_expert_gate,
                }
            } else {
                let gate_name = format!("{prefix}.mlp.gate_proj.weight");
                let up_name = format!("{prefix}.mlp.up_proj.weight");
                let down_name = format!("{prefix}.mlp.down_proj.weight");

                let gate = load_linear_sharded::<T>(
                    ctx,
                    loader,
                    &gate_name,
                    &shard,
                    ShardStrategy::Column,
                    qc,
                )?;
                let up = load_linear_sharded::<T>(
                    ctx,
                    loader,
                    &up_name,
                    &shard,
                    ShardStrategy::Column,
                    qc,
                )?;
                let gate_up = GateUpWeight::Separate {
                    gate_proj: Box::new(gate),
                    up_proj: Box::new(up),
                };

                QwenFfnWeights::Dense(Box::new(QwenMlpWeights {
                    gate_up,
                    down_proj: load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &down_name,
                        &shard,
                        ShardStrategy::Row,
                        qc,
                    )?,
                }))
            };

            layers.push(QwenLayerWeights {
                input_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.input_layernorm.weight"),
                )?,
                attention: QwenAttentionWeights {
                    q_proj,
                    kv_proj,
                    o_proj,
                    q_bias,
                    k_bias,
                    v_bias,
                    q_norm,
                    k_norm,
                },
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

        let (cos_f32, sin_f32) = if let Some(ref rs) = config.rope_scaling {
            let scaling = RopeScaling {
                rope_type: rs.rope_type.clone(),
                factor: rs.factor,
                original_max_position_embeddings: rs.original_max_position_embeddings,
            };
            precompute_rope_cache_scaled(
                ctx,
                config.max_position_embeddings,
                config.head_dim(),
                config.rope_theta,
                &scaling,
            )?
        } else {
            precompute_rope_cache(
                ctx,
                config.max_position_embeddings,
                config.head_dim(),
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

        Ok(Self {
            tp_num_heads,
            tp_num_kv_heads,
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
        })
    }

    // --- Forward pass ---

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &QwenConfig {
        &self.config
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
        layer: &QwenLayerWeights<T>,
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

        let (mut hidden, normed) = add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
        add_inplace(&mut hidden, &mlp_output)?;
        Ok(hidden)
    }

    fn forward_attention_paged_decode(
        &self,
        hidden: &CudaTensor<T>,
        weights: &QwenAttentionWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<CudaTensor<T>> {
        let batch_size = hidden.shape()[0];
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim();

        let mut q = linear(hidden, &weights.q_proj)?;
        let (mut k, mut v) = match &weights.kv_proj {
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

        if let Some(ref bias) = weights.q_bias {
            bias_add_inplace(&mut q, bias)?;
        }
        if let Some(ref bias) = weights.k_bias {
            bias_add_inplace(&mut k, bias)?;
        }
        if let Some(ref bias) = weights.v_bias {
            bias_add_inplace(&mut v, bias)?;
        }

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

        let q = apply_rope_batched(&q, &self.cos_cache, &self.sin_cache, positions)?;
        let k = apply_rope_batched(&k, &self.cos_cache, &self.sin_cache, positions)?;

        let q_stride = num_heads * head_dim;
        let kv_stride = num_kv_heads * head_dim;
        let sliding_window = self.config.effective_sliding_window(layer_idx);

        let mut attn_parts = Vec::with_capacity(batch_size);
        for (i, &pos) in positions.iter().enumerate() {
            let q_i = q.slice_view(i * q_stride, &[1, num_heads, head_dim]);
            let k_i = k.slice_view(i * kv_stride, &[1, num_kv_heads, head_dim]);
            let v_i = v.slice_view(i * kv_stride, &[1, num_kv_heads, head_dim]);

            paged_kv.append_paged(layer_idx, &block_tables[i], &k_i, &v_i, pos)?;

            let mut table_with_current = block_tables[i].clone();
            table_with_current.advance(1);

            let (k_pool, v_pool) = paged_kv.get_pools(layer_idx);
            let attn_i = paged_attention_decode(
                &self.ctx,
                &q_i,
                k_pool,
                v_pool,
                &[table_with_current],
                paged_kv.block_size(),
                None,
                None,
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
        layer: &QwenLayerWeights<T>,
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

        let (mut hidden, normed) = add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
        add_inplace(&mut hidden, &mlp_output)?;
        Ok(hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attention_paged_decode_indirect(
        &self,
        hidden: &CudaTensor<T>,
        weights: &QwenAttentionWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        graph_inputs: &BatchedGraphInputs,
        max_seq_len: usize,
    ) -> Result<CudaTensor<T>> {
        let batch_size = hidden.shape()[0];
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim();

        let mut q = linear(hidden, &weights.q_proj)?;
        let (mut k, mut v) = match &weights.kv_proj {
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

        if let Some(ref bias) = weights.q_bias {
            bias_add_inplace(&mut q, bias)?;
        }
        if let Some(ref bias) = weights.k_bias {
            bias_add_inplace(&mut k, bias)?;
        }
        if let Some(ref bias) = weights.v_bias {
            bias_add_inplace(&mut v, bias)?;
        }

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

        let q = apply_rope_batched_indirect(
            &q,
            &self.cos_cache,
            &self.sin_cache,
            graph_inputs.positions(),
            batch_size,
        )?;
        let k = apply_rope_batched_indirect(
            &k,
            &self.cos_cache,
            &self.sin_cache,
            graph_inputs.positions(),
            batch_size,
        )?;

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
            None,
            None,
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
        layer: &QwenLayerWeights<T>,
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

        let (mut hidden, normed) = add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
        add_inplace(&mut hidden, &mlp_output)?;
        Ok(hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attention_paged_prefill(
        &self,
        hidden: &CudaTensor<T>,
        weights: &QwenAttentionWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        block_table: &BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<CudaTensor<T>> {
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim();

        let mut q = linear(hidden, &weights.q_proj)?;
        let (mut k, mut v) = match &weights.kv_proj {
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

        if let Some(ref bias) = weights.q_bias {
            bias_add_inplace(&mut q, bias)?;
        }
        if let Some(ref bias) = weights.k_bias {
            bias_add_inplace(&mut k, bias)?;
        }
        if let Some(ref bias) = weights.v_bias {
            bias_add_inplace(&mut v, bias)?;
        }

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

        let q = apply_rope(&q, &self.cos_cache, &self.sin_cache, start_pos)?;
        let k = apply_rope(&k, &self.cos_cache, &self.sin_cache, start_pos)?;

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
            None,
            None,
            sliding_window,
        )?;

        let attn_output = attn_output.reshape(&[seq_len, num_heads * head_dim]);
        let mut out = linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    // --- FFN ---

    fn forward_ffn(
        &self,
        hidden: &CudaTensor<T>,
        ffn: &QwenFfnWeights<T>,
    ) -> Result<CudaTensor<T>> {
        match ffn {
            QwenFfnWeights::Dense(mlp) => self.forward_mlp(hidden, mlp),
            QwenFfnWeights::Moe {
                gate,
                experts,
                num_experts_per_tok,
                norm_topk_prob,
                shared_expert,
                shared_expert_gate,
            } => self.forward_moe(
                hidden,
                gate,
                experts,
                *num_experts_per_tok,
                *norm_topk_prob,
                shared_expert.as_deref(),
                shared_expert_gate.as_ref(),
            ),
        }
    }

    #[allow(clippy::unused_self)]
    fn forward_mlp(
        &self,
        hidden: &CudaTensor<T>,
        weights: &QwenMlpWeights<T>,
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
        weights: &QwenMlpWeights<T>,
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
        let intermediate = swiglu(&gate, &up)?;
        linear(&intermediate, &weights.down_proj)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_moe(
        &self,
        hidden: &CudaTensor<T>,
        gate: &CudaTensor<T>,
        experts: &[MoeExpertWeights<T>],
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        shared_expert: Option<&QwenMlpWeights<T>>,
        shared_expert_gate: Option<&CudaTensor<T>>,
    ) -> Result<CudaTensor<T>> {
        let mut out = infernum::cuda::moe::moe_forward(
            hidden,
            gate,
            experts.len(),
            num_experts_per_tok,
            norm_topk_prob,
            |expert_idx, expert_input| {
                self.forward_mlp_no_reduce(expert_input, &experts[expert_idx].mlp)
            },
        )?;

        if let Some(shared_mlp) = shared_expert {
            let shared_out = self.forward_mlp_no_reduce(hidden, shared_mlp)?;

            if let Some(gate_weight) = shared_expert_gate {
                let gate_data = cast_to_f32(gate_weight)?.to_vec()?;
                let gate_val = 1.0 / (1.0 + (-gate_data[0]).exp());
                let gate_t =
                    CudaTensor::<T>::from_slice(&self.ctx, &[1], &[T::from_f32(gate_val)])?;
                let gated = mul(&shared_out, &gate_t)?;
                add_inplace(&mut out, &gated)?;
            } else {
                add_inplace(&mut out, &shared_out)?;
            }
        }

        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
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
}

#[allow(private_bounds)]
impl<T> infernum::Model for QwenModel<T>
where
    T: TensorDType + DeviceRepr + GemmScalar + Default + ValidAsZeroBits + MaybeNcclType,
    CudaBlas: Gemm<T>,
{
    type CacheDtype = T;

    fn config(&self) -> infernum::ModelConfig {
        let config = self.config();
        infernum::ModelConfig {
            num_layers: config.num_hidden_layers,
            max_seq_len: config.max_position_embeddings,
            num_kv_heads: self.tp_num_kv_heads,
            head_dim: config.head_dim(),
            eos_token_id: config.eos_token_id,
        }
    }

    fn devices(&self) -> Vec<&CudaContext> {
        vec![&self.ctx]
    }

    fn forward(&self, input_ids: &[u32]) -> Result<CudaTensor<f32>> {
        let seq_len = input_ids.len();
        let mut hidden = self.embed(input_ids)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let normed = rms_norm(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            let num_heads = self.tp_num_heads;
            let num_kv_heads = self.tp_num_kv_heads;
            let head_dim = self.config.head_dim();

            let mut q = linear(&normed, &layer.attention.q_proj)?;
            let (mut k, mut v) = match &layer.attention.kv_proj {
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

            if let Some(ref bias) = layer.attention.q_bias {
                bias_add_inplace(&mut q, bias)?;
            }
            if let Some(ref bias) = layer.attention.k_bias {
                bias_add_inplace(&mut k, bias)?;
            }
            if let Some(ref bias) = layer.attention.v_bias {
                bias_add_inplace(&mut v, bias)?;
            }

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

            let q = apply_rope(&q, &self.cos_cache, &self.sin_cache, 0)?;
            let k = apply_rope(&k, &self.cos_cache, &self.sin_cache, 0)?;

            let sliding_window = self.config.effective_sliding_window(layer_idx);
            let attn_output = fused_attention_prefill(&q, &k, &v, 0, None, None, sliding_window)?;
            let attn_output = attn_output.reshape(&[seq_len, num_heads * head_dim]);
            let mut attn_output = linear(&attn_output, &layer.attention.o_proj)?;
            #[cfg(feature = "nccl")]
            nccl_all_reduce(self.nccl_comm.as_ref(), &mut attn_output)?;

            let (mut h, normed) = add_rmsnorm(
                &hidden,
                &attn_output,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;

            let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
            add_inplace(&mut h, &mlp_output)?;
            hidden = h;
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
impl<T> infernum::ShardedLoadable for QwenModel<T>
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
