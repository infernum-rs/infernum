//! Qwen model implementation

#![allow(
    clippy::struct_field_names,
    clippy::no_effect_underscore_binding,
    clippy::doc_markdown,
    unused_mut
)]

use std::marker::PhantomData;
use std::path::Path;

use infernum::backend::{
    ArithOps, AttentionOps, Backend, BiasOps, CastOps, Comm, EmbedOps, MatmulExtOps, MatmulOps,
    MoeOps, NormOps, PagedAttentionOps, PagedKvCacheOps, RopeOps, SwigluOps, TensorOps,
};
use infernum::dtype::DType;
use infernum::shard::{ShardConfig, ShardStrategy};
use infernum::tensor::Tensor;
use infernum::Result;
use infernum_cuda::cuda::ops::{
    cast_from_f32, cast_to_f32, precompute_rope_cache, precompute_rope_cache_scaled, transpose_2d,
    LinearWeight, RopeScaling,
};
#[cfg(feature = "nccl")]
use infernum_cuda::cuda::NcclCommunicator;
use infernum_cuda::cuda::{CudaContext, CudaTensor, GpuConfig, PagedKvCache, QuantizedTensor};
use infernum_cuda::weights::{SafeTensorsLoader, WeightLoader};
use infernum_cuda::BlockTable;
use infernum_cuda::CudaBackend;

use crate::QwenConfig;

// --- Weight helpers ---

fn pretranspose_weight(weight: &CudaTensor) -> Result<CudaTensor> {
    transpose_2d(weight)
}

fn concat_weights(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
    let k = a.shape()[0];
    assert_eq!(k, b.shape()[0], "concat_weights: K dimension mismatch");
    let n1 = a.shape()[1];
    let n2 = b.shape()[1];
    let dtype = a.dtype();
    let elem = dtype.size_in_bytes();

    let a_data = a.to_raw_bytes()?;
    let b_data = b.to_raw_bytes()?;
    let mut out = vec![0u8; k * (n1 + n2) * elem];
    for row in 0..k {
        let out_off = row * (n1 + n2) * elem;
        out[out_off..out_off + n1 * elem]
            .copy_from_slice(&a_data[row * n1 * elem..(row + 1) * n1 * elem]);
        out[out_off + n1 * elem..out_off + (n1 + n2) * elem]
            .copy_from_slice(&b_data[row * n2 * elem..(row + 1) * n2 * elem]);
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

enum KvProjWeight<B: Backend + MatmulOps> {
    Fused {
        weight: B::Tensor,
        kv_dim: usize,
    },
    Separate {
        k_proj: Box<<B as MatmulOps>::LinearWeight>,
        v_proj: Box<<B as MatmulOps>::LinearWeight>,
    },
}

struct QwenAttentionWeights<B: Backend + MatmulOps> {
    q_proj: <B as MatmulOps>::LinearWeight,
    kv_proj: KvProjWeight<B>,
    o_proj: <B as MatmulOps>::LinearWeight,
    /// Q/K/V biases (Qwen2, some Qwen3-MoE)
    q_bias: Option<B::Tensor>,
    k_bias: Option<B::Tensor>,
    v_bias: Option<B::Tensor>,
    /// QK-norm weights (Qwen3): RMSNorm per-head before RoPE
    q_norm: Option<B::Tensor>,
    k_norm: Option<B::Tensor>,
}

enum GateUpWeight<B: Backend + MatmulOps> {
    Fused {
        weight: B::Tensor,
        intermediate_size: usize,
    },
    Separate {
        gate_proj: Box<<B as MatmulOps>::LinearWeight>,
        up_proj: Box<<B as MatmulOps>::LinearWeight>,
    },
}

struct QwenMlpWeights<B: Backend + MatmulOps> {
    gate_up: GateUpWeight<B>,
    down_proj: <B as MatmulOps>::LinearWeight,
}

struct MoeExpertWeights<B: Backend + MatmulOps> {
    mlp: QwenMlpWeights<B>,
}

#[allow(clippy::large_enum_variant)]
enum QwenFfnWeights<B: Backend + MatmulOps> {
    Dense(Box<QwenMlpWeights<B>>),
    Moe {
        gate: B::Tensor,
        experts: Vec<MoeExpertWeights<B>>,
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        shared_expert: Option<Box<QwenMlpWeights<B>>>,
        shared_expert_gate: Option<B::Tensor>,
    },
}

struct QwenLayerWeights<B: Backend + MatmulOps> {
    input_layernorm: B::Tensor,
    attention: QwenAttentionWeights<B>,
    post_attention_layernorm: B::Tensor,
    ffn: QwenFfnWeights<B>,
}

/// Complete Qwen model, generic over the compute backend `B`.
pub struct QwenModel<B: Backend + MatmulOps> {
    config: QwenConfig,
    ctx: CudaContext,
    #[allow(dead_code)]
    gpu_config: GpuConfig,

    /// Optional communicator for tensor-parallel all-reduce.
    comm: Option<B::Comm>,

    tp_num_heads: usize,
    tp_num_kv_heads: usize,

    dtype: DType,
    embed_tokens: B::Tensor,
    layers: Vec<QwenLayerWeights<B>>,
    norm: B::Tensor,
    lm_head: <B as MatmulOps>::LinearWeight,

    cos_cache: B::Tensor,
    sin_cache: B::Tensor,

    _backend: PhantomData<B>,
}

impl QwenModel<CudaBackend> {
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
    pub fn from_pretrained_sharded(
        ctx: &CudaContext,
        model_path: impl AsRef<Path>,
        gpu_config: GpuConfig,
        comm: Option<<CudaBackend as Backend>::Comm>,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        let config_path = model_path.join("config.json");
        let config = QwenConfig::from_file(&config_path)?;
        let loader = SafeTensorsLoader::from_directory(model_path)?;
        Self::load_weights_sharded(ctx, config, &loader, gpu_config, comm)
    }

    /// Optional all-reduce for tensor-parallel models. No-op for single GPU.
    fn maybe_all_reduce(&self, tensor: &mut CudaTensor) -> Result<()> {
        if let Some(comm) = &self.comm {
            comm.all_reduce_sum(tensor)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn load_weights(
        ctx: &CudaContext,
        config: QwenConfig,
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

            let file_dtype = loader.get_dtype(name)?;
            if file_dtype.is_quantized() {
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
            } else if model_dtype == DType::F32 {
                let f32_weight = loader.load_f32(ctx, name)?;
                Ok(LinearWeight::Dense(pretranspose_weight(&f32_weight)?))
            } else {
                let native = load_typed(model_dtype, loader, ctx, name)?;
                let shape = native.shape().to_vec();
                let rows = shape[0];
                let cols = shape[1];
                let elem = model_dtype.size_in_bytes();
                let data = native.to_raw_bytes()?;
                let mut transposed_data = vec![0u8; data.len()];
                for r in 0..rows {
                    for c in 0..cols {
                        let src = (r * cols + c) * elem;
                        let dst = (c * rows + r) * elem;
                        transposed_data[dst..dst + elem].copy_from_slice(&data[src..src + elem]);
                    }
                }
                Ok(LinearWeight::Dense(CudaTensor::from_raw_bytes(
                    ctx,
                    &[cols, rows],
                    model_dtype,
                    &transposed_data,
                )?))
            }
        }

        fn load_dense_mlp(
            dtype: DType,
            ctx: &CudaContext,
            loader: &impl WeightLoader,
            prefix: &str,
            intermediate_size: usize,
            qc: Option<&crate::config::QuantizationConfig>,
        ) -> Result<QwenMlpWeights<CudaBackend>> {
            let gate = load_linear(
                dtype,
                ctx,
                loader,
                &format!("{prefix}.gate_proj.weight"),
                qc,
            )?;
            let up = load_linear(dtype, ctx, loader, &format!("{prefix}.up_proj.weight"), qc)?;
            let gate_up = match (gate, up) {
                (LinearWeight::Dense(g), LinearWeight::Dense(u)) => {
                    GateUpWeight::<CudaBackend>::Fused {
                        weight: concat_weights(&g, &u)?,
                        intermediate_size,
                    }
                }
                (g, u) => GateUpWeight::<CudaBackend>::Separate {
                    gate_proj: Box::new(g),
                    up_proj: Box::new(u),
                },
            };
            Ok(QwenMlpWeights {
                gate_up,
                down_proj: load_linear(
                    dtype,
                    ctx,
                    loader,
                    &format!("{prefix}.down_proj.weight"),
                    qc,
                )?,
            })
        }

        fn load_moe_weights(
            dtype: DType,
            ctx: &CudaContext,
            loader: &impl WeightLoader,
            layer_prefix: &str,
            config: &QwenConfig,
            qc: Option<&crate::config::QuantizationConfig>,
        ) -> Result<QwenFfnWeights<CudaBackend>> {
            let num_experts = config.num_experts.expect("MoE requires num_experts");
            let num_experts_per_tok = config
                .num_experts_per_tok
                .expect("MoE requires num_experts_per_tok");
            let expert_inter = config.moe_expert_intermediate_size();

            // Qwen MoE router: mlp.gate.weight [num_experts, hidden_size]
            let gate_name = format!("{layer_prefix}.mlp.gate.weight");
            let gate_f32 = loader.load_f32(ctx, &gate_name)?;
            let gate_transposed = pretranspose_weight(&gate_f32)?;
            let gate = cast_from_f32(&gate_transposed, dtype)?;

            // Per-expert MLPs: mlp.experts.{E}.{gate,up,down}_proj.weight
            let mut experts = Vec::with_capacity(num_experts);
            for e in 0..num_experts {
                let ep = format!("{layer_prefix}.mlp.experts.{e}");
                let mlp = load_dense_mlp(dtype, ctx, loader, &ep, expert_inter, qc)?;
                experts.push(MoeExpertWeights { mlp });
            }

            // Shared expert: mlp.shared_expert.{gate,up,down}_proj.weight
            let shared_expert = if config.has_shared_expert() {
                let shared_inter = config
                    .shared_expert_intermediate_size
                    .expect("shared_expert_intermediate_size required");
                let sp = format!("{layer_prefix}.mlp.shared_expert");
                let mlp = load_dense_mlp(dtype, ctx, loader, &sp, shared_inter, qc)?;
                Some(Box::new(mlp))
            } else {
                None
            };

            // Shared expert gate: mlp.shared_expert_gate.weight [1]
            let shared_expert_gate = if config.has_shared_expert() {
                let gate_name = format!("{layer_prefix}.mlp.shared_expert_gate.weight");
                if loader.contains(&gate_name) {
                    Some(load_typed(dtype, loader, ctx, &gate_name)?)
                } else {
                    None
                }
            } else {
                None
            };

            Ok(QwenFfnWeights::<CudaBackend>::Moe {
                gate,
                experts,
                num_experts_per_tok,
                norm_topk_prob: config.norm_topk_prob,
                shared_expert,
                shared_expert_gate,
            })
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

            let q_bias_name = format!("{prefix}.self_attn.q_proj.bias");
            let k_bias_name = format!("{prefix}.self_attn.k_proj.bias");
            let v_bias_name = format!("{prefix}.self_attn.v_proj.bias");
            let q_bias = if loader.contains(&q_bias_name) {
                Some(load_typed(dtype, loader, ctx, &q_bias_name)?)
            } else {
                None
            };
            let k_bias = if loader.contains(&k_bias_name) {
                Some(load_typed(dtype, loader, ctx, &k_bias_name)?)
            } else {
                None
            };
            let v_bias = if loader.contains(&v_bias_name) {
                Some(load_typed(dtype, loader, ctx, &v_bias_name)?)
            } else {
                None
            };

            let q_norm_name = format!("{prefix}.self_attn.q_norm.weight");
            let k_norm_name = format!("{prefix}.self_attn.k_norm.weight");
            let q_norm = if loader.contains(&q_norm_name) {
                Some(load_typed(dtype, loader, ctx, &q_norm_name)?)
            } else {
                None
            };
            let k_norm = if loader.contains(&k_norm_name) {
                Some(load_typed(dtype, loader, ctx, &k_norm_name)?)
            } else {
                None
            };

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
            let kv_proj = match (k, v) {
                (LinearWeight::Dense(k_w), LinearWeight::Dense(v_w)) => {
                    KvProjWeight::<CudaBackend>::Fused {
                        kv_dim: config.num_kv_heads() * config.head_dim(),
                        weight: concat_weights(&k_w, &v_w)?,
                    }
                }
                (k, v) => KvProjWeight::<CudaBackend>::Separate {
                    k_proj: Box::new(k),
                    v_proj: Box::new(v),
                },
            };

            let layer = QwenLayerWeights {
                input_layernorm: load_typed(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.input_layernorm.weight"),
                )?,
                attention: QwenAttentionWeights {
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
                    q_bias,
                    k_bias,
                    v_bias,
                    q_norm,
                    k_norm,
                },
                post_attention_layernorm: load_typed(
                    dtype,
                    loader,
                    ctx,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                ffn: if config.is_moe_layer(i) {
                    load_moe_weights(dtype, ctx, loader, &prefix, &config, qc)?
                } else {
                    QwenFfnWeights::<CudaBackend>::Dense(Box::new(load_dense_mlp(
                        dtype,
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

        let norm = load_typed(dtype, loader, ctx, "model.norm.weight")?;

        let lm_head = if config.tie_word_embeddings {
            if qc.is_some() {
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
            }
        } else {
            let lw = load_linear(dtype, ctx, loader, "lm_head.weight", None)?;
            if qc.is_some() {
                if let LinearWeight::Dense(ref w) = lw {
                    let f32_w = cast_to_f32(w)?;
                    let data = f32_w.to_vec::<f32>()?;
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
        let cos_cache = cast_from_f32(&cos_f32, dtype)?;
        let sin_cache = cast_from_f32(&sin_f32, dtype)?;

        Ok(Self {
            tp_num_heads: config.num_attention_heads,
            tp_num_kv_heads: config.num_kv_heads(),
            dtype,
            config,
            ctx: ctx.clone(),
            gpu_config: GpuConfig::Single,
            comm: None,
            embed_tokens,
            layers,
            norm,
            lm_head,
            cos_cache,
            sin_cache,
            _backend: PhantomData,
        })
    }

    // --- Sharded loading ---

    #[allow(clippy::too_many_lines, clippy::similar_names)]
    fn load_weights_sharded(
        ctx: &CudaContext,
        config: QwenConfig,
        loader: &impl WeightLoader,
        gpu_config: GpuConfig,
        comm: Option<<CudaBackend as Backend>::Comm>,
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
                    qt.set_weight_scale(ctx, scale_val[0])?;
                }
                Ok(LinearWeight::Quantized(qt))
            } else {
                let f32_weight = loader.load_f32_sharded(ctx, name, shard, strategy)?;
                let transposed = pretranspose_weight(&f32_weight)?;
                if model_dtype == DType::F32 {
                    Ok(LinearWeight::Dense(transposed))
                } else {
                    let shape = transposed.shape();
                    let rows = shape[0];
                    let cols = shape[1];
                    let native =
                        load_typed_sharded(model_dtype, loader, ctx, name, shard, strategy)?;
                    let native_bytes = native.to_raw_bytes()?;
                    let elem = model_dtype.size_in_bytes();
                    let mut buf = vec![0u8; native_bytes.len()];
                    for r in 0..rows {
                        for c in 0..cols {
                            let src = (r * cols + c) * elem;
                            let dst = (c * rows + r) * elem;
                            buf[dst..dst + elem].copy_from_slice(&native_bytes[src..src + elem]);
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
                    m.comm = comm;
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

        let embed_dtype = loader.get_dtype("model.embed_tokens.weight")?;
        let dtype = if embed_dtype.is_quantized() {
            DType::F32
        } else {
            embed_dtype
        };

        let embed_tokens = load_typed(dtype, loader, ctx, "model.embed_tokens.weight")?;

        let tp_num_heads = config.num_attention_heads / world_size;
        let tp_num_kv_heads = config.num_kv_heads() / world_size;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            let q_bias_name = format!("{prefix}.self_attn.q_proj.bias");
            let k_bias_name = format!("{prefix}.self_attn.k_proj.bias");
            let v_bias_name = format!("{prefix}.self_attn.v_proj.bias");
            let q_bias = if loader.contains(&q_bias_name) {
                Some(load_typed_sharded(
                    dtype,
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
                Some(load_typed_sharded(
                    dtype,
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
                Some(load_typed_sharded(
                    dtype,
                    loader,
                    ctx,
                    &v_bias_name,
                    &shard,
                    ShardStrategy::Column,
                )?)
            } else {
                None
            };

            let q_norm_name = format!("{prefix}.self_attn.q_norm.weight");
            let k_norm_name = format!("{prefix}.self_attn.k_norm.weight");
            let q_norm = if loader.contains(&q_norm_name) {
                Some(load_typed(dtype, loader, ctx, &q_norm_name)?)
            } else {
                None
            };
            let k_norm = if loader.contains(&k_norm_name) {
                Some(load_typed(dtype, loader, ctx, &k_norm_name)?)
            } else {
                None
            };

            let q_proj = load_linear_sharded(
                dtype,
                ctx,
                loader,
                &format!("{prefix}.self_attn.q_proj.weight"),
                &shard,
                ShardStrategy::Column,
                qc,
            )?;
            let k_proj = load_linear_sharded(
                dtype,
                ctx,
                loader,
                &format!("{prefix}.self_attn.k_proj.weight"),
                &shard,
                ShardStrategy::Column,
                qc,
            )?;
            let v_proj = load_linear_sharded(
                dtype,
                ctx,
                loader,
                &format!("{prefix}.self_attn.v_proj.weight"),
                &shard,
                ShardStrategy::Column,
                qc,
            )?;
            let kv_proj = KvProjWeight::<CudaBackend>::Separate {
                k_proj: Box::new(k_proj),
                v_proj: Box::new(v_proj),
            };

            let o_proj = load_linear_sharded(
                dtype,
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
                let gate = cast_from_f32(&gate_transposed, dtype)?;

                let mut experts = Vec::with_capacity(num_experts);
                for e in 0..num_experts {
                    let ep = format!("{prefix}.mlp.experts.{e}");
                    let gate_proj = load_linear_sharded(
                        dtype,
                        ctx,
                        loader,
                        &format!("{ep}.gate_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?;
                    let up_proj = load_linear_sharded(
                        dtype,
                        ctx,
                        loader,
                        &format!("{ep}.up_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?;
                    let gate_up = GateUpWeight::<CudaBackend>::Separate {
                        gate_proj: Box::new(gate_proj),
                        up_proj: Box::new(up_proj),
                    };
                    let down_proj = load_linear_sharded(
                        dtype,
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
                    let gate_proj = load_linear_sharded(
                        dtype,
                        ctx,
                        loader,
                        &format!("{sp}.gate_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?;
                    let up_proj = load_linear_sharded(
                        dtype,
                        ctx,
                        loader,
                        &format!("{sp}.up_proj.weight"),
                        &shard,
                        ShardStrategy::Column,
                        qc,
                    )?;
                    let gate_up = GateUpWeight::<CudaBackend>::Separate {
                        gate_proj: Box::new(gate_proj),
                        up_proj: Box::new(up_proj),
                    };
                    let down_proj = load_linear_sharded(
                        dtype,
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
                        Some(load_typed(dtype, loader, ctx, &gate_name)?)
                    } else {
                        None
                    }
                } else {
                    None
                };

                QwenFfnWeights::<CudaBackend>::Moe {
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

                let gate = load_linear_sharded(
                    dtype,
                    ctx,
                    loader,
                    &gate_name,
                    &shard,
                    ShardStrategy::Column,
                    qc,
                )?;
                let up = load_linear_sharded(
                    dtype,
                    ctx,
                    loader,
                    &up_name,
                    &shard,
                    ShardStrategy::Column,
                    qc,
                )?;
                let gate_up = GateUpWeight::<CudaBackend>::Separate {
                    gate_proj: Box::new(gate),
                    up_proj: Box::new(up),
                };

                QwenFfnWeights::<CudaBackend>::Dense(Box::new(QwenMlpWeights {
                    gate_up,
                    down_proj: load_linear_sharded(
                        dtype,
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
                input_layernorm: load_typed(
                    dtype,
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
            let embed_f32 = CudaBackend::cast_to_f32(&embed_tokens)?;
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
        let cos_cache = cast_from_f32(&cos_f32, dtype)?;
        let sin_cache = cast_from_f32(&sin_f32, dtype)?;

        Ok(Self {
            tp_num_heads,
            tp_num_kv_heads,
            dtype,
            config,
            ctx: ctx.clone(),
            gpu_config,
            comm,
            embed_tokens,
            layers,
            norm,
            lm_head,
            cos_cache,
            sin_cache,
            _backend: PhantomData,
        })
    }

    // --- Forward pass ---

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &QwenConfig {
        &self.config
    }

    /// Get the model's compute dtype
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
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
        layer: &QwenLayerWeights<CudaBackend>,
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

        let (mut hidden, normed) = CudaBackend::add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
        CudaBackend::add_inplace(&mut hidden, &mlp_output)?;
        Ok(hidden)
    }

    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn forward_attention_paged_decode(
        &self,
        hidden: &CudaTensor,
        weights: &QwenAttentionWeights<CudaBackend>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache,
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<CudaTensor> {
        let batch_size = hidden.shape()[0];
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim();

        let mut q = CudaBackend::linear(hidden, &weights.q_proj)?;
        let (mut k, mut v) = match &weights.kv_proj {
            KvProjWeight::<CudaBackend>::Fused { weight, kv_dim } => {
                let kv = CudaBackend::matmul(hidden, weight)?;
                if batch_size == 1 {
                    let k = kv.slice_view(0, &[1, *kv_dim]);
                    let v = kv.slice_view(*kv_dim, &[1, *kv_dim]);
                    (k, v)
                } else {
                    CudaBackend::split_inner_dim(&kv, *kv_dim, *kv_dim)?
                }
            }
            KvProjWeight::<CudaBackend>::Separate { k_proj, v_proj } => {
                let k = CudaBackend::linear(hidden, k_proj)?;
                let v = CudaBackend::linear(hidden, v_proj)?;
                (k, v)
            }
        };

        if let Some(ref bias) = weights.q_bias {
            CudaBackend::bias_add_inplace(&mut q, bias)?;
        }
        if let Some(ref bias) = weights.k_bias {
            CudaBackend::bias_add_inplace(&mut k, bias)?;
        }
        if let Some(ref bias) = weights.v_bias {
            CudaBackend::bias_add_inplace(&mut v, bias)?;
        }

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

        // Upload positions to device for batched RoPE
        #[allow(clippy::cast_possible_wrap)]
        let positions_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let positions_tensor =
            CudaTensor::from_raw_bytes(&self.ctx, &[batch_size], infernum::DType::U32, unsafe {
                std::slice::from_raw_parts(
                    positions_i32.as_ptr().cast::<u8>(),
                    positions_i32.len() * 4,
                )
            })?;

        let q = CudaBackend::apply_rope_batched(
            &q,
            &self.cos_cache,
            &self.sin_cache,
            &positions_tensor,
            batch_size,
        )?;
        let k = CudaBackend::apply_rope_batched(
            &k,
            &self.cos_cache,
            &self.sin_cache,
            &positions_tensor,
            batch_size,
        )?;

        // Build flattened block tables and seq_lens for batched paged attention
        let max_blocks_per_seq = block_tables
            .iter()
            .map(|bt| bt.blocks().len())
            .max()
            .unwrap_or(0);
        #[allow(clippy::cast_possible_wrap)]
        let block_tables_flat: Vec<i32> = {
            let mut flat = vec![0i32; batch_size * max_blocks_per_seq];
            for (i, bt) in block_tables.iter().enumerate() {
                for (j, &block_id) in bt.blocks().iter().enumerate() {
                    flat[i * max_blocks_per_seq + j] = block_id as i32;
                }
            }
            flat
        };
        let block_tables_tensor = CudaTensor::from_raw_bytes(
            &self.ctx,
            &[batch_size * max_blocks_per_seq],
            infernum::DType::U32,
            unsafe {
                std::slice::from_raw_parts(
                    block_tables_flat.as_ptr().cast::<u8>(),
                    block_tables_flat.len() * 4,
                )
            },
        )?;

        #[allow(clippy::cast_possible_wrap)]
        let seq_lens_i32: Vec<i32> = positions.iter().map(|&p| (p + 1) as i32).collect();
        let seq_lens_tensor =
            CudaTensor::from_raw_bytes(&self.ctx, &[batch_size], infernum::DType::U32, unsafe {
                std::slice::from_raw_parts(
                    seq_lens_i32.as_ptr().cast::<u8>(),
                    seq_lens_i32.len() * 4,
                )
            })?;

        // Batched KV cache append
        CudaBackend::append_paged_batched(
            paged_kv,
            layer_idx,
            &k,
            &v,
            &block_tables_tensor,
            &positions_tensor,
            batch_size,
            max_blocks_per_seq,
        )?;

        let sliding_window = self.config.effective_sliding_window(layer_idx);
        let max_seq_len = seq_lens_i32.iter().copied().max().unwrap_or(0) as usize;

        // Batched paged attention decode
        let (k_pool, v_pool) = CudaBackend::get_pools(paged_kv, layer_idx);
        let attn_output = CudaBackend::paged_attention_decode(
            &q,
            k_pool,
            v_pool,
            &block_tables_tensor,
            &seq_lens_tensor,
            CudaBackend::block_size(paged_kv),
            max_blocks_per_seq,
            max_seq_len,
            None,
            None,
            sliding_window,
        )?;

        let attn_output = attn_output.reshape(&[batch_size, num_heads * head_dim]);

        let mut out = CudaBackend::linear(&attn_output, &weights.o_proj)?;
        self.maybe_all_reduce(&mut out)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_paged_prefill(
        &self,
        hidden: &CudaTensor,
        layer: &QwenLayerWeights<CudaBackend>,
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

        let (mut hidden, normed) = CudaBackend::add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
        CudaBackend::add_inplace(&mut hidden, &mlp_output)?;
        Ok(hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attention_paged_prefill(
        &self,
        hidden: &CudaTensor,
        weights: &QwenAttentionWeights<CudaBackend>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache,
        block_table: &BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<CudaTensor> {
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim();

        let mut q = CudaBackend::linear(hidden, &weights.q_proj)?;
        let (mut k, mut v) = match &weights.kv_proj {
            KvProjWeight::<CudaBackend>::Fused { weight, kv_dim } => {
                let kv = CudaBackend::matmul(hidden, weight)?;
                CudaBackend::split_inner_dim(&kv, *kv_dim, *kv_dim)?
            }
            KvProjWeight::<CudaBackend>::Separate { k_proj, v_proj } => {
                let k = CudaBackend::linear(hidden, k_proj)?;
                let v = CudaBackend::linear(hidden, v_proj)?;
                (k, v)
            }
        };

        if let Some(ref bias) = weights.q_bias {
            CudaBackend::bias_add_inplace(&mut q, bias)?;
        }
        if let Some(ref bias) = weights.k_bias {
            CudaBackend::bias_add_inplace(&mut k, bias)?;
        }
        if let Some(ref bias) = weights.v_bias {
            CudaBackend::bias_add_inplace(&mut v, bias)?;
        }

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

        let q = CudaBackend::apply_rope(&q, &self.cos_cache, &self.sin_cache, start_pos)?;
        let k = CudaBackend::apply_rope(&k, &self.cos_cache, &self.sin_cache, start_pos)?;

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
            None,
            None,
            sliding_window,
        )?;

        let attn_output = attn_output.reshape(&[seq_len, num_heads * head_dim]);
        let mut out = CudaBackend::linear(&attn_output, &weights.o_proj)?;
        self.maybe_all_reduce(&mut out)?;
        Ok(out)
    }

    // --- FFN ---

    fn forward_ffn(
        &self,
        hidden: &CudaTensor,
        ffn: &QwenFfnWeights<CudaBackend>,
    ) -> Result<CudaTensor> {
        match ffn {
            QwenFfnWeights::<CudaBackend>::Dense(mlp) => self.forward_mlp(hidden, mlp),
            QwenFfnWeights::<CudaBackend>::Moe {
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
        hidden: &CudaTensor,
        weights: &QwenMlpWeights<CudaBackend>,
    ) -> Result<CudaTensor> {
        let (gate, up) = match &weights.gate_up {
            GateUpWeight::<CudaBackend>::Fused {
                weight,
                intermediate_size,
            } => {
                let seq_len = hidden.shape()[0];
                let gate_up = CudaBackend::matmul(hidden, weight)?;
                if seq_len == 1 {
                    let gate = gate_up.slice_view(0, &[1, *intermediate_size]);
                    let up = gate_up.slice_view(*intermediate_size, &[1, *intermediate_size]);
                    (gate, up)
                } else {
                    CudaBackend::split_inner_dim(&gate_up, *intermediate_size, *intermediate_size)?
                }
            }
            GateUpWeight::<CudaBackend>::Separate { gate_proj, up_proj } => {
                let gate = CudaBackend::linear(hidden, gate_proj)?;
                let up = CudaBackend::linear(hidden, up_proj)?;
                (gate, up)
            }
        };
        let intermediate = CudaBackend::swiglu(&gate, &up)?;
        let mut out = CudaBackend::linear(&intermediate, &weights.down_proj)?;
        self.maybe_all_reduce(&mut out)?;
        Ok(out)
    }

    #[allow(clippy::unused_self)]
    fn forward_mlp_no_reduce(
        &self,
        hidden: &CudaTensor,
        weights: &QwenMlpWeights<CudaBackend>,
    ) -> Result<CudaTensor> {
        let (gate, up) = match &weights.gate_up {
            GateUpWeight::<CudaBackend>::Fused {
                weight,
                intermediate_size,
            } => {
                let seq_len = hidden.shape()[0];
                let gate_up = CudaBackend::matmul(hidden, weight)?;
                if seq_len == 1 {
                    let gate = gate_up.slice_view(0, &[1, *intermediate_size]);
                    let up = gate_up.slice_view(*intermediate_size, &[1, *intermediate_size]);
                    (gate, up)
                } else {
                    CudaBackend::split_inner_dim(&gate_up, *intermediate_size, *intermediate_size)?
                }
            }
            GateUpWeight::<CudaBackend>::Separate { gate_proj, up_proj } => {
                let gate = CudaBackend::linear(hidden, gate_proj)?;
                let up = CudaBackend::linear(hidden, up_proj)?;
                (gate, up)
            }
        };
        let intermediate = CudaBackend::swiglu(&gate, &up)?;
        CudaBackend::linear(&intermediate, &weights.down_proj)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_moe(
        &self,
        hidden: &CudaTensor,
        gate: &CudaTensor,
        experts: &[MoeExpertWeights<CudaBackend>],
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        shared_expert: Option<&QwenMlpWeights<CudaBackend>>,
        shared_expert_gate: Option<&CudaTensor>,
    ) -> Result<CudaTensor> {
        let mut out = CudaBackend::moe_forward_softmax(
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
                let gate_data: Vec<f32> = cast_to_f32(gate_weight)?.to_vec::<f32>()?;
                let gate_val = 1.0 / (1.0 + (-gate_data[0]).exp());
                let gate_f32 = CudaTensor::from_slice(&self.ctx, &[1], &[gate_val])?;
                let gate_t = cast_from_f32(&gate_f32, self.dtype)?;
                let gated = CudaBackend::mul(&shared_out, &gate_t)?;
                CudaBackend::add_inplace(&mut out, &gated)?;
            } else {
                CudaBackend::add_inplace(&mut out, &shared_out)?;
            }
        }

        self.maybe_all_reduce(&mut out)?;
        Ok(out)
    }

    fn lm_head_forward(&self, hidden: &CudaTensor) -> Result<CudaTensor> {
        if self.dtype == DType::BF16 {
            if let LinearWeight::Dense(w) = &self.lm_head {
                return CudaBackend::matmul_bf16_f32(hidden, w);
            }
        }
        let logits_t = CudaBackend::linear(hidden, &self.lm_head)?;
        if self.dtype == DType::F32 {
            return Ok(logits_t);
        }
        CudaBackend::cast_to_f32(&logits_t)
    }
}

#[cfg(feature = "nccl")]
impl infernum_cuda::ShardedLoadable for QwenModel<CudaBackend> {
    fn load_shard(
        ctx: &CudaContext,
        model_path: &Path,
        shard: ShardConfig,
        comm: NcclCommunicator,
    ) -> Result<Self> {
        Self::from_pretrained_sharded(ctx, model_path, GpuConfig::Sharded(shard), Some(comm))
    }
}

// --- Public helpers & infernum::Model implementation ---

impl QwenModel<CudaBackend> {
    /// Build the runtime-facing [`ModelConfig`](infernum::ModelConfig).
    #[must_use]
    pub fn model_config(&self) -> infernum::ModelConfig {
        let c = self.config();
        infernum::ModelConfig {
            num_layers: c.num_hidden_layers,
            max_seq_len: c.max_position_embeddings,
            num_kv_heads: self.tp_num_kv_heads,
            head_dim: c.head_dim(),
            eos_token_id: c.eos_token_id,
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

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let normed =
                CudaBackend::rms_norm(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            let num_heads = self.tp_num_heads;
            let num_kv_heads = self.tp_num_kv_heads;
            let head_dim = self.config.head_dim();

            let mut q = CudaBackend::linear(&normed, &layer.attention.q_proj)?;
            let (mut k, mut v) = match &layer.attention.kv_proj {
                KvProjWeight::<CudaBackend>::Fused { weight, kv_dim } => {
                    let kv = CudaBackend::matmul(&normed, weight)?;
                    CudaBackend::split_inner_dim(&kv, *kv_dim, *kv_dim)?
                }
                KvProjWeight::<CudaBackend>::Separate { k_proj, v_proj } => {
                    let k = CudaBackend::linear(&normed, k_proj)?;
                    let v = CudaBackend::linear(&normed, v_proj)?;
                    (k, v)
                }
            };

            if let Some(ref bias) = layer.attention.q_bias {
                CudaBackend::bias_add_inplace(&mut q, bias)?;
            }
            if let Some(ref bias) = layer.attention.k_bias {
                CudaBackend::bias_add_inplace(&mut k, bias)?;
            }
            if let Some(ref bias) = layer.attention.v_bias {
                CudaBackend::bias_add_inplace(&mut v, bias)?;
            }

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

            let q = CudaBackend::apply_rope(&q, &self.cos_cache, &self.sin_cache, 0)?;
            let k = CudaBackend::apply_rope(&k, &self.cos_cache, &self.sin_cache, 0)?;

            let sliding_window = self.config.effective_sliding_window(layer_idx);
            let attn_output =
                CudaBackend::fused_attention_prefill(&q, &k, &v, 0, None, None, sliding_window)?;
            let attn_output = attn_output.reshape(&[seq_len, num_heads * head_dim]);
            let mut attn_output = CudaBackend::linear(&attn_output, &layer.attention.o_proj)?;
            self.maybe_all_reduce(&mut attn_output)?;

            let (mut h, normed) = CudaBackend::add_rmsnorm(
                &hidden,
                &attn_output,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;

            let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
            CudaBackend::add_inplace(&mut h, &mlp_output)?;
            hidden = h;
        }

        CudaBackend::rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;
        self.lm_head_forward(&hidden)
    }
}

impl infernum::Model for QwenModel<CudaBackend> {
    type B = CudaBackend;
    type KvCache = PagedKvCache;

    fn config(&self) -> infernum::ModelConfig {
        self.model_config()
    }

    fn device(&self) -> &CudaContext {
        &self.ctx
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

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_decode(
        &self,
        token_ids: &CudaTensor,
        kv_cache: &mut Self::KvCache,
        _runtime_state: &mut infernum_cuda::CudaRuntimeState,
        block_tables: &CudaTensor,
        _seq_lens: &CudaTensor,
        positions: &CudaTensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
        _max_seq_len: usize,
    ) -> Result<infernum_cuda::CudaLogits> {
        // Download tensors to host and call old-style internal method.
        // TODO: refactor QwenModel to use tensor-based ops directly.
        let token_ids_host = token_ids.to_vec::<u32>()?;
        let positions_u32 = positions.to_vec::<u32>()?;
        let positions_host: Vec<usize> = positions_u32.iter().map(|&p| p as usize).collect();
        let bt_flat = block_tables.to_vec::<u32>()?;
        let block_size = CudaBackend::block_size(kv_cache);
        let block_tables_host: Vec<infernum::BlockTable> = (0..batch_size)
            .map(|i| {
                let start = i * max_blocks_per_seq;
                let end = start + max_blocks_per_seq;
                let blocks: Vec<usize> = bt_flat[start..end].iter().map(|&b| b as usize).collect();
                let seq_len = positions_host[i];
                infernum::BlockTable::from_raw(blocks, seq_len, block_size)
            })
            .collect();
        let tensor = QwenModel::forward_batch_decode(
            self,
            &token_ids_host,
            std::slice::from_mut(kv_cache),
            &block_tables_host,
            &positions_host,
        )?;
        Ok(infernum_cuda::CudaLogits::new(tensor))
    }
}
