//! Qwen model implementation — fully generic over the compute backend `B`.
//!
//! All forward pass methods, weight loading, and the model struct are
//! generic over `B: Backend`. No CUDA-specific code remains in this file.

#![allow(
    clippy::struct_field_names,
    clippy::no_effect_underscore_binding,
    clippy::doc_markdown,
    dead_code,
    unused_mut
)]

use std::marker::PhantomData;
use std::path::Path;

use infernum::backend::{
    ArithOps, AttentionOps, Backend, BiasOps, CastOps, Comm, EmbedOps, MatmulExtOps, MatmulOps,
    MoeOps, NormOps, PagedAttentionOps, PagedKvCacheOps, RopeOps, SwigluOps, TensorDataOps,
    TensorFactory, TensorOps,
};
use infernum::block_allocator::BlockTable;
use infernum::dtype::DType;
use infernum::shard::GpuConfig;
use infernum::tensor::Tensor;
use infernum::Result;

use crate::QwenConfig;

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
    device: B::DeviceHandle,
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

// ---- Trait alias ----

/// Convenience alias: all op traits required by `QwenModel` forward methods.
pub trait QwenOps:
    Backend
    + MatmulOps
    + MatmulExtOps
    + NormOps
    + ArithOps
    + SwigluOps
    + CastOps
    + EmbedOps
    + TensorOps
    + TensorFactory
    + TensorDataOps
    + RopeOps
    + AttentionOps
    + PagedAttentionOps
    + PagedKvCacheOps
    + MoeOps
    + BiasOps
{
}

impl<B> QwenOps for B where
    B: Backend
        + MatmulOps
        + MatmulExtOps
        + NormOps
        + ArithOps
        + SwigluOps
        + CastOps
        + EmbedOps
        + TensorOps
        + TensorFactory
        + TensorDataOps
        + RopeOps
        + AttentionOps
        + PagedAttentionOps
        + PagedKvCacheOps
        + MoeOps
        + BiasOps
{
}

// ---- Generic impl ----

impl<B: QwenOps> QwenModel<B> {
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

    // ---- Generic weight loading ----

    /// Load model weights from a backend-agnostic weight loader.
    ///
    /// # Errors
    /// Returns an error if any weight fails to load.
    ///
    /// # Panics
    /// Panics if `as_dense_weight` returns `None` after `is_dense_weight`
    /// returned `true`.
    #[allow(clippy::too_many_lines)]
    pub fn load_weights(
        device: B::DeviceHandle,
        config: QwenConfig,
        loader: &impl infernum::WeightLoader<B>,
    ) -> Result<Self> {
        let qc = config.quantization_config.as_ref();

        let embed_dtype = loader.get_dtype("model.embed_tokens.weight")?;
        let dtype = if embed_dtype.is_quantized() {
            DType::F32
        } else {
            embed_dtype
        };

        let embed_tokens = loader.load_tensor("model.embed_tokens.weight", dtype)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            let q_bias = Self::load_optional_tensor(
                loader,
                &format!("{prefix}.self_attn.q_proj.bias"),
                dtype,
            )?;
            let k_bias = Self::load_optional_tensor(
                loader,
                &format!("{prefix}.self_attn.k_proj.bias"),
                dtype,
            )?;
            let v_bias = Self::load_optional_tensor(
                loader,
                &format!("{prefix}.self_attn.v_proj.bias"),
                dtype,
            )?;

            let q_norm = Self::load_optional_tensor(
                loader,
                &format!("{prefix}.self_attn.q_norm.weight"),
                dtype,
            )?;
            let k_norm = Self::load_optional_tensor(
                loader,
                &format!("{prefix}.self_attn.k_norm.weight"),
                dtype,
            )?;

            let k = loader.load_linear(&format!("{prefix}.self_attn.k_proj.weight"), dtype, qc)?;
            let v = loader.load_linear(&format!("{prefix}.self_attn.v_proj.weight"), dtype, qc)?;
            let kv_proj = if B::is_dense_weight(&k) && B::is_dense_weight(&v) {
                let k_t = B::as_dense_weight(&k).expect("checked dense");
                let v_t = B::as_dense_weight(&v).expect("checked dense");
                KvProjWeight::<B>::Fused {
                    kv_dim: config.num_kv_heads() * config.head_dim(),
                    weight: B::concat_inner_dim(k_t, v_t)?,
                }
            } else {
                KvProjWeight::<B>::Separate {
                    k_proj: Box::new(k),
                    v_proj: Box::new(v),
                }
            };

            let layer = QwenLayerWeights {
                input_layernorm: loader
                    .load_tensor(&format!("{prefix}.input_layernorm.weight"), dtype)?,
                attention: QwenAttentionWeights {
                    q_proj: loader.load_linear(
                        &format!("{prefix}.self_attn.q_proj.weight"),
                        dtype,
                        qc,
                    )?,
                    kv_proj,
                    o_proj: loader.load_linear(
                        &format!("{prefix}.self_attn.o_proj.weight"),
                        dtype,
                        qc,
                    )?,
                    q_bias,
                    k_bias,
                    v_bias,
                    q_norm,
                    k_norm,
                },
                post_attention_layernorm: loader
                    .load_tensor(&format!("{prefix}.post_attention_layernorm.weight"), dtype)?,
                ffn: if config.is_moe_layer(i) {
                    Self::load_moe_weights(dtype, loader, &prefix, &config, qc)?
                } else {
                    QwenFfnWeights::<B>::Dense(Box::new(Self::load_dense_mlp(
                        dtype,
                        loader,
                        &format!("{prefix}.mlp"),
                        config.intermediate_size,
                        qc,
                    )?))
                },
            };

            layers.push(layer);
        }

        let norm = loader.load_tensor("model.norm.weight", dtype)?;

        let lm_head = if config.tie_word_embeddings {
            if qc.is_some() {
                let embed_f32 = B::cast_to_f32(&embed_tokens)?;
                let data = B::to_f32_vec(&embed_f32)?;
                B::quantize_to_q8(&device, embed_f32.shape(), &data)?
            } else {
                let embed_f32 = B::cast_to_f32(&embed_tokens)?;
                let transposed = B::transpose_2d(&embed_f32)?;
                B::dense_weight(B::cast_from_f32(&transposed, dtype)?)
            }
        } else {
            let lw = loader.load_linear("lm_head.weight", dtype, None)?;
            if qc.is_some() {
                if let Some(w) = B::as_dense_weight(&lw) {
                    let f32_w = B::cast_to_f32(w)?;
                    let row_major = B::transpose_2d(&f32_w)?;
                    let data = B::to_f32_vec(&row_major)?;
                    B::quantize_to_q8(&device, row_major.shape(), &data)?
                } else {
                    lw
                }
            } else {
                lw
            }
        };

        // Precompute RoPE cache (with optional YaRN scaling)
        let half_dim = config.head_dim() / 2;
        let max_pos = config.max_position_embeddings;
        let (cos_data, sin_data) = if let Some(ref rs) = config.rope_scaling {
            let scaling = infernum::rope::RopeScaling {
                rope_type: rs.rope_type.clone(),
                factor: rs.factor,
                original_max_position_embeddings: rs.original_max_position_embeddings,
            };
            infernum::rope::precompute_rope_data_scaled(
                max_pos,
                config.head_dim(),
                config.rope_theta,
                &scaling,
            )
        } else {
            infernum::rope::precompute_rope_data(max_pos, config.head_dim(), config.rope_theta)
        };
        let cos_f32 = B::from_f32_slice(&device, &[max_pos, half_dim], &cos_data)?;
        let sin_f32 = B::from_f32_slice(&device, &[max_pos, half_dim], &sin_data)?;
        let cos_cache = B::cast_from_f32(&cos_f32, dtype)?;
        let sin_cache = B::cast_from_f32(&sin_f32, dtype)?;

        Ok(Self {
            tp_num_heads: config.num_attention_heads,
            tp_num_kv_heads: config.num_kv_heads(),
            dtype,
            config,
            device,
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

    /// Load a dense MLP (gate_proj, up_proj, down_proj).
    fn load_dense_mlp(
        dtype: DType,
        loader: &impl infernum::WeightLoader<B>,
        prefix: &str,
        intermediate_size: usize,
        qc: Option<&crate::config::QuantizationConfig>,
    ) -> Result<QwenMlpWeights<B>> {
        let gate = loader.load_linear(&format!("{prefix}.gate_proj.weight"), dtype, qc)?;
        let up = loader.load_linear(&format!("{prefix}.up_proj.weight"), dtype, qc)?;
        let gate_up = if B::is_dense_weight(&gate) && B::is_dense_weight(&up) {
            let g = B::as_dense_weight(&gate).expect("checked dense");
            let u = B::as_dense_weight(&up).expect("checked dense");
            GateUpWeight::<B>::Fused {
                weight: B::concat_inner_dim(g, u)?,
                intermediate_size,
            }
        } else {
            GateUpWeight::<B>::Separate {
                gate_proj: Box::new(gate),
                up_proj: Box::new(up),
            }
        };
        Ok(QwenMlpWeights {
            gate_up,
            down_proj: loader.load_linear(&format!("{prefix}.down_proj.weight"), dtype, qc)?,
        })
    }

    /// Load MoE weights for a single layer.
    fn load_moe_weights(
        dtype: DType,
        loader: &impl infernum::WeightLoader<B>,
        layer_prefix: &str,
        config: &QwenConfig,
        qc: Option<&crate::config::QuantizationConfig>,
    ) -> Result<QwenFfnWeights<B>> {
        let num_experts = config.num_experts.expect("MoE requires num_experts");
        let num_experts_per_tok = config
            .num_experts_per_tok
            .expect("MoE requires num_experts_per_tok");
        let expert_inter = config.moe_expert_intermediate_size();

        let gate_f32 =
            loader.load_tensor(&format!("{layer_prefix}.mlp.gate.weight"), DType::F32)?;
        let gate_transposed = B::transpose_2d(&gate_f32)?;
        let gate = B::cast_from_f32(&gate_transposed, dtype)?;

        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let ep = format!("{layer_prefix}.mlp.experts.{e}");
            let mlp = Self::load_dense_mlp(dtype, loader, &ep, expert_inter, qc)?;
            experts.push(MoeExpertWeights { mlp });
        }

        let shared_expert = if config.has_shared_expert() {
            let shared_inter = config
                .shared_expert_intermediate_size
                .expect("shared_expert_intermediate_size required");
            let sp = format!("{layer_prefix}.mlp.shared_expert");
            let mlp = Self::load_dense_mlp(dtype, loader, &sp, shared_inter, qc)?;
            Some(Box::new(mlp))
        } else {
            None
        };

        let shared_expert_gate = if config.has_shared_expert() {
            let gate_name = format!("{layer_prefix}.mlp.shared_expert_gate.weight");
            if loader.contains(&gate_name) {
                Some(loader.load_tensor(&gate_name, dtype)?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(QwenFfnWeights::<B>::Moe {
            gate,
            experts,
            num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
            shared_expert,
            shared_expert_gate,
        })
    }

    /// Load a SafeTensors model from a directory.
    ///
    /// # Errors
    /// Returns an error if the config is missing or weights fail to load.
    pub fn from_pretrained(device: &B::DeviceHandle, model_path: impl AsRef<Path>) -> Result<Self>
    where
        B: infernum::SafeTensorsLoaderOps,
    {
        let model_path = model_path.as_ref();
        let config = QwenConfig::from_file(model_path.join("config.json"))?;
        let loader = B::safetensors_loader(device, model_path)?;
        Self::load_weights(device.clone(), config, &loader)
    }

    /// Load a Qwen model with tensor-parallel sharding.
    ///
    /// # Errors
    /// Returns an error if loading fails or head counts are not divisible.
    pub fn from_pretrained_sharded(
        device: &B::DeviceHandle,
        model_path: impl AsRef<Path>,
        gpu_config: GpuConfig,
        comm: Option<B::Comm>,
    ) -> Result<Self>
    where
        B: infernum::SafeTensorsLoaderOps,
    {
        let model_path = model_path.as_ref();
        let config = QwenConfig::from_file(model_path.join("config.json"))?;
        let loader = B::safetensors_loader(device, model_path)?;
        Self::load_weights_sharded(device.clone(), config, &loader, gpu_config, comm)
    }
    // ---- Sharded weight loading ----

    /// Load model weights with tensor-parallel sharding, generic over backend.
    ///
    /// # Errors
    /// Returns an error if weight loading fails.
    ///
    /// # Panics
    /// Panics if head counts are not divisible by `world_size`.
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    pub fn load_weights_sharded(
        device: B::DeviceHandle,
        config: QwenConfig,
        loader: &impl infernum::WeightLoader<B>,
        gpu_config: GpuConfig,
        comm: Option<B::Comm>,
    ) -> Result<Self> {
        use infernum::shard::{shard_strategy_for_weight, ShardStrategy};

        let shard = match &gpu_config {
            GpuConfig::Sharded(s) => *s,
            GpuConfig::Single => {
                return Self::load_weights(device, config, loader).map(|mut m| {
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

        let embed_tokens = loader.load_tensor("model.embed_tokens.weight", dtype)?;

        let tp_num_heads = config.num_attention_heads / world_size;
        let tp_num_kv_heads = config.num_kv_heads() / world_size;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            let q_bias_name = format!("{prefix}.self_attn.q_proj.bias");
            let k_bias_name = format!("{prefix}.self_attn.k_proj.bias");
            let v_bias_name = format!("{prefix}.self_attn.v_proj.bias");
            let q_bias = if loader.contains(&q_bias_name) {
                Some(loader.load_tensor_sharded(
                    &q_bias_name,
                    dtype,
                    &shard,
                    ShardStrategy::Column,
                )?)
            } else {
                None
            };
            let k_bias = if loader.contains(&k_bias_name) {
                Some(loader.load_tensor_sharded(
                    &k_bias_name,
                    dtype,
                    &shard,
                    ShardStrategy::Column,
                )?)
            } else {
                None
            };
            let v_bias = if loader.contains(&v_bias_name) {
                Some(loader.load_tensor_sharded(
                    &v_bias_name,
                    dtype,
                    &shard,
                    ShardStrategy::Column,
                )?)
            } else {
                None
            };

            // QK-norm weights: replicated
            let q_norm = Self::load_optional_tensor(
                loader,
                &format!("{prefix}.self_attn.q_norm.weight"),
                dtype,
            )?;
            let k_norm = Self::load_optional_tensor(
                loader,
                &format!("{prefix}.self_attn.k_norm.weight"),
                dtype,
            )?;

            let q_name = format!("{prefix}.self_attn.q_proj.weight");
            let k_name = format!("{prefix}.self_attn.k_proj.weight");
            let v_name = format!("{prefix}.self_attn.v_proj.weight");
            let o_name = format!("{prefix}.self_attn.o_proj.weight");

            let q_proj = loader.load_linear_sharded(
                &q_name,
                dtype,
                qc,
                &shard,
                shard_strategy_for_weight(&q_name),
            )?;
            let k_proj = loader.load_linear_sharded(
                &k_name,
                dtype,
                qc,
                &shard,
                shard_strategy_for_weight(&k_name),
            )?;
            let v_proj = loader.load_linear_sharded(
                &v_name,
                dtype,
                qc,
                &shard,
                shard_strategy_for_weight(&v_name),
            )?;

            let kv_proj = KvProjWeight::<B>::Separate {
                k_proj: Box::new(k_proj),
                v_proj: Box::new(v_proj),
            };

            let ffn = if config.is_moe_layer(i) {
                Self::load_moe_weights_sharded(dtype, loader, &prefix, &config, &shard, qc)?
            } else {
                let gate_name = format!("{prefix}.mlp.gate_proj.weight");
                let up_name = format!("{prefix}.mlp.up_proj.weight");
                let down_name = format!("{prefix}.mlp.down_proj.weight");

                let gate = loader.load_linear_sharded(
                    &gate_name,
                    dtype,
                    qc,
                    &shard,
                    shard_strategy_for_weight(&gate_name),
                )?;
                let up = loader.load_linear_sharded(
                    &up_name,
                    dtype,
                    qc,
                    &shard,
                    shard_strategy_for_weight(&up_name),
                )?;

                let gate_up = GateUpWeight::<B>::Separate {
                    gate_proj: Box::new(gate),
                    up_proj: Box::new(up),
                };

                QwenFfnWeights::<B>::Dense(Box::new(QwenMlpWeights {
                    gate_up,
                    down_proj: loader.load_linear_sharded(
                        &down_name,
                        dtype,
                        qc,
                        &shard,
                        shard_strategy_for_weight(&down_name),
                    )?,
                }))
            };

            layers.push(QwenLayerWeights {
                input_layernorm: loader
                    .load_tensor(&format!("{prefix}.input_layernorm.weight"), dtype)?,
                attention: QwenAttentionWeights {
                    q_proj,
                    kv_proj,
                    o_proj: loader.load_linear_sharded(
                        &o_name,
                        dtype,
                        qc,
                        &shard,
                        shard_strategy_for_weight(&o_name),
                    )?,
                    q_bias,
                    k_bias,
                    v_bias,
                    q_norm,
                    k_norm,
                },
                post_attention_layernorm: loader
                    .load_tensor(&format!("{prefix}.post_attention_layernorm.weight"), dtype)?,
                ffn,
            });
        }

        let norm = loader.load_tensor("model.norm.weight", dtype)?;
        let lm_head = if config.tie_word_embeddings {
            let embed_f32 = B::cast_to_f32(&embed_tokens)?;
            let transposed = B::transpose_2d(&embed_f32)?;
            B::dense_weight(B::cast_from_f32(&transposed, dtype)?)
        } else {
            loader.load_linear_sharded(
                "lm_head.weight",
                dtype,
                None,
                &shard,
                ShardStrategy::Replicate,
            )?
        };

        let half_dim = config.head_dim() / 2;
        let max_pos = config.max_position_embeddings;
        let (cos_data, sin_data) = if let Some(ref rs) = config.rope_scaling {
            let scaling = infernum::rope::RopeScaling {
                rope_type: rs.rope_type.clone(),
                factor: rs.factor,
                original_max_position_embeddings: rs.original_max_position_embeddings,
            };
            infernum::rope::precompute_rope_data_scaled(
                max_pos,
                config.head_dim(),
                config.rope_theta,
                &scaling,
            )
        } else {
            infernum::rope::precompute_rope_data(max_pos, config.head_dim(), config.rope_theta)
        };
        let cos_f32 = B::from_f32_slice(&device, &[max_pos, half_dim], &cos_data)?;
        let sin_f32 = B::from_f32_slice(&device, &[max_pos, half_dim], &sin_data)?;
        let cos_cache = B::cast_from_f32(&cos_f32, dtype)?;
        let sin_cache = B::cast_from_f32(&sin_f32, dtype)?;

        Ok(Self {
            tp_num_heads,
            tp_num_kv_heads,
            dtype,
            config,
            device,
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

    /// Load MoE weights with tensor-parallel sharding for a single layer.
    fn load_moe_weights_sharded(
        dtype: DType,
        loader: &impl infernum::WeightLoader<B>,
        layer_prefix: &str,
        config: &QwenConfig,
        shard: &infernum::ShardConfig,
        qc: Option<&crate::config::QuantizationConfig>,
    ) -> Result<QwenFfnWeights<B>> {
        use infernum::shard::ShardStrategy;

        let num_experts = config.num_experts.expect("MoE requires num_experts");
        let num_experts_per_tok = config
            .num_experts_per_tok
            .expect("MoE requires num_experts_per_tok");

        let gate_f32 =
            loader.load_tensor(&format!("{layer_prefix}.mlp.gate.weight"), DType::F32)?;
        let gate_transposed = B::transpose_2d(&gate_f32)?;
        let gate = B::cast_from_f32(&gate_transposed, dtype)?;

        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let ep = format!("{layer_prefix}.mlp.experts.{e}");
            let gate_proj = loader.load_linear_sharded(
                &format!("{ep}.gate_proj.weight"),
                dtype,
                qc,
                shard,
                ShardStrategy::Column,
            )?;
            let up_proj = loader.load_linear_sharded(
                &format!("{ep}.up_proj.weight"),
                dtype,
                qc,
                shard,
                ShardStrategy::Column,
            )?;
            let gate_up = GateUpWeight::<B>::Separate {
                gate_proj: Box::new(gate_proj),
                up_proj: Box::new(up_proj),
            };
            let down_proj = loader.load_linear_sharded(
                &format!("{ep}.down_proj.weight"),
                dtype,
                qc,
                shard,
                ShardStrategy::Row,
            )?;
            experts.push(MoeExpertWeights {
                mlp: QwenMlpWeights { gate_up, down_proj },
            });
        }

        let shared_expert = if config.has_shared_expert() {
            let sp = format!("{layer_prefix}.mlp.shared_expert");
            let gate_proj = loader.load_linear_sharded(
                &format!("{sp}.gate_proj.weight"),
                dtype,
                qc,
                shard,
                ShardStrategy::Column,
            )?;
            let up_proj = loader.load_linear_sharded(
                &format!("{sp}.up_proj.weight"),
                dtype,
                qc,
                shard,
                ShardStrategy::Column,
            )?;
            let gate_up = GateUpWeight::<B>::Separate {
                gate_proj: Box::new(gate_proj),
                up_proj: Box::new(up_proj),
            };
            let down_proj = loader.load_linear_sharded(
                &format!("{sp}.down_proj.weight"),
                dtype,
                qc,
                shard,
                ShardStrategy::Row,
            )?;
            Some(Box::new(QwenMlpWeights { gate_up, down_proj }))
        } else {
            None
        };

        let shared_expert_gate = if config.has_shared_expert() {
            let gate_name = format!("{layer_prefix}.mlp.shared_expert_gate.weight");
            if loader.contains(&gate_name) {
                Some(loader.load_tensor(&gate_name, dtype)?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(QwenFfnWeights::<B>::Moe {
            gate,
            experts,
            num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
            shared_expert,
            shared_expert_gate,
        })
    }
    // ---- Helpers ----

    /// Load an optional tensor (returns `None` if the tensor does not exist).
    fn load_optional_tensor(
        loader: &impl infernum::WeightLoader<B>,
        name: &str,
        dtype: DType,
    ) -> Result<Option<B::Tensor>> {
        if loader.contains(name) {
            Ok(Some(loader.load_tensor(name, dtype)?))
        } else {
            Ok(None)
        }
    }

    /// Extract the last row from a (seq_len, hidden_size) tensor
    #[allow(clippy::unused_self)]
    fn extract_last_row(&self, hidden: &B::Tensor, seq_len: usize) -> B::Tensor {
        let hidden_size = hidden.shape()[1];
        if seq_len == 1 {
            return hidden.reshape(&[1, hidden_size]);
        }
        hidden.slice_view((seq_len - 1) * hidden_size, &[1, hidden_size])
    }

    /// Embed token IDs
    fn embed(&self, input_ids: &[u32]) -> Result<B::Tensor> {
        B::embedding_gather(&self.embed_tokens, input_ids)
    }

    /// Optional all-reduce for tensor-parallel models. No-op for single GPU.
    fn maybe_all_reduce(&self, tensor: &mut B::Tensor) -> Result<()> {
        if let Some(comm) = &self.comm {
            comm.all_reduce_sum(tensor)?;
        }
        Ok(())
    }

    // ---- Full forward pass (no KV cache) ----

    /// Full forward pass without KV cache (recomputes everything).
    ///
    /// Returns raw logits tensor of shape `(seq_len, vocab_size)`.
    ///
    /// # Errors
    /// Returns an error if any operation fails.
    pub fn forward_full(&self, input_ids: &[u32]) -> Result<B::Tensor> {
        let seq_len = input_ids.len();
        let mut hidden = self.embed(input_ids)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let normed = B::rms_norm(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            let num_heads = self.tp_num_heads;
            let num_kv_heads = self.tp_num_kv_heads;
            let head_dim = self.config.head_dim();

            let mut q = B::linear(&normed, &layer.attention.q_proj)?;
            let (mut k, mut v) = match &layer.attention.kv_proj {
                KvProjWeight::<B>::Fused { weight, kv_dim } => {
                    let kv = B::matmul(&normed, weight)?;
                    B::split_inner_dim(&kv, *kv_dim, *kv_dim)?
                }
                KvProjWeight::<B>::Separate { k_proj, v_proj } => {
                    let k = B::linear(&normed, k_proj)?;
                    let v = B::linear(&normed, v_proj)?;
                    (k, v)
                }
            };

            if let Some(ref bias) = layer.attention.q_bias {
                B::bias_add_inplace(&mut q, bias)?;
            }
            if let Some(ref bias) = layer.attention.k_bias {
                B::bias_add_inplace(&mut k, bias)?;
            }
            if let Some(ref bias) = layer.attention.v_bias {
                B::bias_add_inplace(&mut v, bias)?;
            }

            let mut q = q.reshape(&[seq_len, num_heads, head_dim]);
            let mut k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
            let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

            if let Some(ref q_norm_w) = layer.attention.q_norm {
                let flat_q = q.reshape(&[seq_len * num_heads, head_dim]);
                let normed_q = B::rms_norm(&flat_q, q_norm_w, self.config.rms_norm_eps)?;
                q = normed_q.reshape(&[seq_len, num_heads, head_dim]);
            }
            if let Some(ref k_norm_w) = layer.attention.k_norm {
                let flat_k = k.reshape(&[seq_len * num_kv_heads, head_dim]);
                let normed_k = B::rms_norm(&flat_k, k_norm_w, self.config.rms_norm_eps)?;
                k = normed_k.reshape(&[seq_len, num_kv_heads, head_dim]);
            }

            let q = B::apply_rope(&q, &self.cos_cache, &self.sin_cache, 0)?;
            let k = B::apply_rope(&k, &self.cos_cache, &self.sin_cache, 0)?;

            let sliding_window = self.config.effective_sliding_window(layer_idx);
            let attn_output =
                B::fused_attention_prefill(&q, &k, &v, 0, None, None, sliding_window)?;
            let attn_output = attn_output.reshape(&[seq_len, num_heads * head_dim]);
            let mut attn_output = B::linear(&attn_output, &layer.attention.o_proj)?;
            self.maybe_all_reduce(&mut attn_output)?;

            let (mut h, normed) = B::add_rmsnorm(
                &hidden,
                &attn_output,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;

            let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
            B::add_inplace(&mut h, &mlp_output)?;
            hidden = h;
        }

        B::rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;
        self.lm_head_forward(&hidden)
    }

    // ---- Batched decode (host-side convenience) ----

    /// Batched decode with host-side inputs.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn forward_batch_decode(
        &self,
        token_ids: &[u32],
        paged_kv: &mut B::PagedKvCache,
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<B::Tensor> {
        let batch_size = token_ids.len();
        let max_blocks_per_seq = block_tables
            .iter()
            .map(|bt| bt.blocks().len())
            .max()
            .unwrap_or(0);
        let mut bt_flat = vec![0i32; batch_size * max_blocks_per_seq];
        for (i, bt) in block_tables.iter().enumerate() {
            for (j, &block_id) in bt.blocks().iter().enumerate() {
                bt_flat[i * max_blocks_per_seq + j] = block_id as i32;
            }
        }
        let seq_lens: Vec<i32> = positions.iter().map(|&p| (p + 1) as i32).collect();
        let positions_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let max_seq_len = seq_lens.iter().copied().max().unwrap_or(0) as usize;

        let token_ids_t = B::from_u32_slice(&self.device, &[batch_size], token_ids)?;
        let bt_t = B::from_i32_slice(&self.device, &[batch_size * max_blocks_per_seq], &bt_flat)?;
        let sl_t = B::from_i32_slice(&self.device, &[batch_size], &seq_lens)?;
        let pos_t = B::from_i32_slice(&self.device, &[batch_size], &positions_i32)?;

        self.forward_batch_decode_tensors(
            &token_ids_t,
            paged_kv,
            &bt_t,
            &sl_t,
            &pos_t,
            batch_size,
            max_blocks_per_seq,
            max_seq_len,
        )
    }

    // ---- Batched decode with paged KV cache (device tensors) ----

    /// Batched decode forward pass with paged KV cache.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batch_decode_tensors(
        &self,
        token_ids: &B::Tensor,
        paged_kv: &mut B::PagedKvCache,
        block_tables: &B::Tensor,
        seq_lens: &B::Tensor,
        positions: &B::Tensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
    ) -> Result<B::Tensor> {
        let mut hidden = B::embedding_gather_tensor(&self.embed_tokens, token_ids, batch_size)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer_paged_decode_batched(
                &hidden,
                layer,
                layer_idx,
                paged_kv,
                block_tables,
                seq_lens,
                positions,
                batch_size,
                max_blocks_per_seq,
                max_seq_len,
            )?;
        }

        B::rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;
        self.lm_head_forward(&hidden.reshape(&[batch_size, self.config.hidden_size]))
    }

    // ---- Single-sequence prefill with paged KV cache ----

    /// Single-sequence prefill with paged KV cache.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward_prefill_paged(
        &self,
        input_ids: &[u32],
        paged_kv: &mut B::PagedKvCache,
        block_table: &BlockTable,
        start_pos: usize,
    ) -> Result<B::Tensor> {
        let seq_len = input_ids.len();

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

        let last = self.extract_last_row(&hidden, seq_len);
        let normed = B::rms_norm(&last, &self.norm, self.config.rms_norm_eps)?;
        self.lm_head_forward(&normed.reshape(&[1, self.config.hidden_size]))
    }
    // ---- Layer-level forward methods ----

    /// Transformer layer forward for batched decode with paged KV cache.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_paged_decode_batched(
        &self,
        hidden: &B::Tensor,
        layer: &QwenLayerWeights<B>,
        layer_idx: usize,
        paged_kv: &mut B::PagedKvCache,
        block_tables: &B::Tensor,
        seq_lens: &B::Tensor,
        positions: &B::Tensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
    ) -> Result<B::Tensor> {
        let normed = B::rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

        let attn_output = self.forward_attention_paged_decode_batched(
            &normed,
            &layer.attention,
            layer_idx,
            paged_kv,
            block_tables,
            seq_lens,
            positions,
            batch_size,
            max_blocks_per_seq,
            max_seq_len,
        )?;

        let (mut hidden, normed) = B::add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
        B::add_inplace(&mut hidden, &mlp_output)?;
        Ok(hidden)
    }

    /// Batched attention decode with paged KV cache.
    #[allow(clippy::too_many_arguments)]
    fn forward_attention_paged_decode_batched(
        &self,
        hidden: &B::Tensor,
        weights: &QwenAttentionWeights<B>,
        layer_idx: usize,
        paged_kv: &mut B::PagedKvCache,
        block_tables: &B::Tensor,
        seq_lens: &B::Tensor,
        positions: &B::Tensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
    ) -> Result<B::Tensor> {
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim();

        let mut q = B::linear(hidden, &weights.q_proj)?;
        let (mut k, mut v) = match &weights.kv_proj {
            KvProjWeight::<B>::Fused { weight, kv_dim } => {
                let kv = B::matmul(hidden, weight)?;
                if batch_size == 1 {
                    let k = kv.slice_view(0, &[1, *kv_dim]);
                    let v = kv.slice_view(*kv_dim, &[1, *kv_dim]);
                    (k, v)
                } else {
                    B::split_inner_dim(&kv, *kv_dim, *kv_dim)?
                }
            }
            KvProjWeight::<B>::Separate { k_proj, v_proj } => {
                let k = B::linear(hidden, k_proj)?;
                let v = B::linear(hidden, v_proj)?;
                (k, v)
            }
        };

        if let Some(ref bias) = weights.q_bias {
            B::bias_add_inplace(&mut q, bias)?;
        }
        if let Some(ref bias) = weights.k_bias {
            B::bias_add_inplace(&mut k, bias)?;
        }
        if let Some(ref bias) = weights.v_bias {
            B::bias_add_inplace(&mut v, bias)?;
        }

        let mut q = q.reshape(&[batch_size, num_heads, head_dim]);
        let mut k = k.reshape(&[batch_size, num_kv_heads, head_dim]);
        let v = v.reshape(&[batch_size, num_kv_heads, head_dim]);

        if let Some(ref q_norm_w) = weights.q_norm {
            let flat_q = q.reshape(&[batch_size * num_heads, head_dim]);
            let normed_q = B::rms_norm(&flat_q, q_norm_w, self.config.rms_norm_eps)?;
            q = normed_q.reshape(&[batch_size, num_heads, head_dim]);
        }
        if let Some(ref k_norm_w) = weights.k_norm {
            let flat_k = k.reshape(&[batch_size * num_kv_heads, head_dim]);
            let normed_k = B::rms_norm(&flat_k, k_norm_w, self.config.rms_norm_eps)?;
            k = normed_k.reshape(&[batch_size, num_kv_heads, head_dim]);
        }

        let sliding_window = self.config.effective_sliding_window(layer_idx);

        let q = B::apply_rope_batched(&q, &self.cos_cache, &self.sin_cache, positions, batch_size)?;
        let k = B::apply_rope_batched(&k, &self.cos_cache, &self.sin_cache, positions, batch_size)?;

        B::append_paged_batched(
            paged_kv,
            layer_idx,
            &k,
            &v,
            block_tables,
            positions,
            batch_size,
            max_blocks_per_seq,
        )?;

        let (k_pool, v_pool) = B::get_pools(paged_kv, layer_idx);
        let attn_output = B::paged_attention_decode(
            &q,
            k_pool,
            v_pool,
            block_tables,
            seq_lens,
            B::block_size(paged_kv),
            max_blocks_per_seq,
            max_seq_len,
            None,
            None,
            sliding_window,
        )?;

        let attn_output = attn_output.reshape(&[batch_size, num_heads * head_dim]);

        let mut out = B::linear(&attn_output, &weights.o_proj)?;
        self.maybe_all_reduce(&mut out)?;
        Ok(out)
    }

    /// Transformer layer forward for single-sequence prefill with paged KV cache.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_paged_prefill(
        &self,
        hidden: &B::Tensor,
        layer: &QwenLayerWeights<B>,
        layer_idx: usize,
        paged_kv: &mut B::PagedKvCache,
        block_table: &BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<B::Tensor> {
        let normed = B::rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

        let attn_output = self.forward_attention_paged_prefill(
            &normed,
            &layer.attention,
            layer_idx,
            paged_kv,
            block_table,
            start_pos,
            seq_len,
        )?;

        let (mut hidden, normed) = B::add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
        B::add_inplace(&mut hidden, &mlp_output)?;
        Ok(hidden)
    }

    /// Attention for single-sequence prefill with paged KV cache.
    #[allow(clippy::too_many_arguments)]
    fn forward_attention_paged_prefill(
        &self,
        hidden: &B::Tensor,
        weights: &QwenAttentionWeights<B>,
        layer_idx: usize,
        paged_kv: &mut B::PagedKvCache,
        block_table: &BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<B::Tensor> {
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim();

        let mut q = B::linear(hidden, &weights.q_proj)?;
        let (mut k, mut v) = match &weights.kv_proj {
            KvProjWeight::<B>::Fused { weight, kv_dim } => {
                let kv = B::matmul(hidden, weight)?;
                B::split_inner_dim(&kv, *kv_dim, *kv_dim)?
            }
            KvProjWeight::<B>::Separate { k_proj, v_proj } => {
                let k = B::linear(hidden, k_proj)?;
                let v = B::linear(hidden, v_proj)?;
                (k, v)
            }
        };

        if let Some(ref bias) = weights.q_bias {
            B::bias_add_inplace(&mut q, bias)?;
        }
        if let Some(ref bias) = weights.k_bias {
            B::bias_add_inplace(&mut k, bias)?;
        }
        if let Some(ref bias) = weights.v_bias {
            B::bias_add_inplace(&mut v, bias)?;
        }

        let mut q = q.reshape(&[seq_len, num_heads, head_dim]);
        let mut k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
        let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

        if let Some(ref q_norm_w) = weights.q_norm {
            let flat_q = q.reshape(&[seq_len * num_heads, head_dim]);
            let normed_q = B::rms_norm(&flat_q, q_norm_w, self.config.rms_norm_eps)?;
            q = normed_q.reshape(&[seq_len, num_heads, head_dim]);
        }
        if let Some(ref k_norm_w) = weights.k_norm {
            let flat_k = k.reshape(&[seq_len * num_kv_heads, head_dim]);
            let normed_k = B::rms_norm(&flat_k, k_norm_w, self.config.rms_norm_eps)?;
            k = normed_k.reshape(&[seq_len, num_kv_heads, head_dim]);
        }

        let q = B::apply_rope(&q, &self.cos_cache, &self.sin_cache, start_pos)?;
        let k = B::apply_rope(&k, &self.cos_cache, &self.sin_cache, start_pos)?;

        B::append_paged(paged_kv, layer_idx, block_table, &k, &v, start_pos)?;

        let mut gather_table = block_table.clone();
        gather_table.advance(seq_len);
        let (k_contig, v_contig) = B::gather_paged_kv(paged_kv, layer_idx, &gather_table)?;

        let sliding_window = self.config.effective_sliding_window(layer_idx);
        let attn_output = B::fused_attention_prefill(
            &q,
            &k_contig,
            &v_contig,
            start_pos,
            None,
            None,
            sliding_window,
        )?;

        let attn_output = attn_output.reshape(&[seq_len, num_heads * head_dim]);
        let mut out = B::linear(&attn_output, &weights.o_proj)?;
        self.maybe_all_reduce(&mut out)?;
        Ok(out)
    }

    // ---- MLP / FFN ----

    /// Compute gate and up projections from a gate+up weight.
    #[allow(clippy::unused_self)]
    fn compute_gate_up(
        &self,
        hidden: &B::Tensor,
        gate_up: &GateUpWeight<B>,
    ) -> Result<(B::Tensor, B::Tensor)> {
        match gate_up {
            GateUpWeight::<B>::Fused {
                weight,
                intermediate_size,
            } => {
                let seq_len = hidden.shape()[0];
                let gate_up = B::matmul(hidden, weight)?;
                if seq_len == 1 {
                    let gate = gate_up.slice_view(0, &[1, *intermediate_size]);
                    let up = gate_up.slice_view(*intermediate_size, &[1, *intermediate_size]);
                    Ok((gate, up))
                } else {
                    B::split_inner_dim(&gate_up, *intermediate_size, *intermediate_size)
                }
            }
            GateUpWeight::<B>::Separate { gate_proj, up_proj } => {
                let gate = B::linear(hidden, gate_proj)?;
                let up = B::linear(hidden, up_proj)?;
                Ok((gate, up))
            }
        }
    }

    /// Forward pass through MLP (SwiGLU) with all-reduce.
    #[allow(clippy::unused_self)]
    fn forward_mlp(&self, hidden: &B::Tensor, weights: &QwenMlpWeights<B>) -> Result<B::Tensor> {
        let (gate, up) = self.compute_gate_up(hidden, &weights.gate_up)?;
        let intermediate = B::swiglu(&gate, &up)?;
        let mut out = B::linear(&intermediate, &weights.down_proj)?;
        self.maybe_all_reduce(&mut out)?;
        Ok(out)
    }

    /// Forward pass through MLP without all-reduce (used by MoE experts).
    fn forward_mlp_no_reduce(
        &self,
        hidden: &B::Tensor,
        weights: &QwenMlpWeights<B>,
    ) -> Result<B::Tensor> {
        let (gate, up) = self.compute_gate_up(hidden, &weights.gate_up)?;
        let intermediate = B::swiglu(&gate, &up)?;
        B::linear(&intermediate, &weights.down_proj)
    }

    /// Dispatch to dense MLP or MoE forward pass.
    fn forward_ffn(&self, hidden: &B::Tensor, ffn: &QwenFfnWeights<B>) -> Result<B::Tensor> {
        match ffn {
            QwenFfnWeights::<B>::Dense(mlp) => self.forward_mlp(hidden, mlp),
            QwenFfnWeights::<B>::Moe {
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

    /// Forward pass through a Mixture-of-Experts layer with shared expert.
    #[allow(clippy::too_many_arguments)]
    fn forward_moe(
        &self,
        hidden: &B::Tensor,
        gate: &B::Tensor,
        experts: &[MoeExpertWeights<B>],
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        shared_expert: Option<&QwenMlpWeights<B>>,
        shared_expert_gate: Option<&B::Tensor>,
    ) -> Result<B::Tensor> {
        let mut out = B::moe_forward_softmax(
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
                let gate_data = B::to_f32_vec(gate_weight)?;
                let gate_val = 1.0 / (1.0 + (-gate_data[0]).exp());
                let gate_f32 = B::from_f32_slice(&self.device, &[1], &[gate_val])?;
                let gate_t = B::cast_from_f32(&gate_f32, self.dtype)?;
                let gated = B::mul(&shared_out, &gate_t)?;
                B::add_inplace(&mut out, &gated)?;
            } else {
                B::add_inplace(&mut out, &shared_out)?;
            }
        }

        self.maybe_all_reduce(&mut out)?;
        Ok(out)
    }

    /// Project hidden states to vocabulary logits (always f32)
    fn lm_head_forward(&self, hidden: &B::Tensor) -> Result<B::Tensor> {
        if self.dtype == DType::BF16 {
            if let Some(w) = B::as_dense_weight(&self.lm_head) {
                return B::matmul_bf16_f32(hidden, w);
            }
        }
        let logits_t = B::linear(hidden, &self.lm_head)?;
        if self.dtype == DType::F32 {
            return Ok(logits_t);
        }
        B::cast_to_f32(&logits_t)
    }
}

// ---- Model trait impl (generic over any backend) ----

impl<B: QwenOps + Send + 'static> infernum::Model for QwenModel<B>
where
    <B as MatmulOps>::LinearWeight: Send + Sync,
{
    type B = B;
    type KvCache = B::PagedKvCache;

    fn config(&self) -> infernum::ModelConfig {
        self.model_config()
    }

    fn device(&self) -> &B::DeviceHandle {
        &self.device
    }

    fn allocate_kv_cache(&self, block_config: &infernum::BlockConfig) -> Result<Self::KvCache> {
        let c = &self.config;
        B::allocate_paged_kv_cache(
            &self.device,
            c.num_hidden_layers,
            block_config,
            self.tp_num_kv_heads,
            c.head_dim(),
            self.dtype,
        )
    }

    fn forward(&self, input_ids: &[u32]) -> Result<B::Logits> {
        let tensor = self.forward_full(input_ids)?;
        Ok(B::logits_from_tensor(tensor))
    }

    fn forward_prefill(
        &self,
        input_ids: &[u32],
        kv_cache: &mut Self::KvCache,
        _runtime_state: &mut B::RuntimeState,
        block_table: &BlockTable,
        start_pos: usize,
    ) -> Result<B::Logits> {
        let tensor = self.forward_prefill_paged(input_ids, kv_cache, block_table, start_pos)?;
        Ok(B::logits_from_tensor(tensor))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_decode(
        &self,
        token_ids: &B::Tensor,
        kv_cache: &mut Self::KvCache,
        _runtime_state: &mut B::RuntimeState,
        block_tables: &B::Tensor,
        seq_lens: &B::Tensor,
        positions: &B::Tensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
    ) -> Result<B::Logits> {
        let tensor = self.forward_batch_decode_tensors(
            token_ids,
            kv_cache,
            block_tables,
            seq_lens,
            positions,
            batch_size,
            max_blocks_per_seq,
            max_seq_len,
        )?;
        Ok(B::logits_from_tensor(tensor))
    }
}
