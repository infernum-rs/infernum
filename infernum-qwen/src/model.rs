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
    ArithOps, AttentionOps, Backend, BiasOps, CastOps, EmbedOps, MatmulExtOps, MatmulOps, MoeOps,
    NormOps, PagedAttentionOps, PagedKvCacheOps, RopeOps, SwigluOps, TensorDataOps, TensorFactory,
    TensorOps,
};
use infernum::block_allocator::BlockTable;
use infernum::dtype::DType;
use infernum::shard::GpuConfig;
use infernum::tensor::Tensor;
use infernum::transformer::{self, GateUpWeight, KvProjWeight, MlpWeights};
use infernum::Result;

use crate::QwenConfig;

// --- Weight structures ---

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

#[allow(clippy::large_enum_variant)]
enum QwenFfnWeights<B: Backend + MatmulOps> {
    Dense(Box<MlpWeights<B>>),
    Moe {
        gate: B::Tensor,
        experts: Vec<MlpWeights<B>>,
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        shared_expert: Option<Box<MlpWeights<B>>>,
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

        let lm_head = transformer::load_lm_head::<B>(
            &device,
            loader,
            &embed_tokens,
            config.tie_word_embeddings,
            dtype,
            qc,
        )?;

        // Precompute RoPE cache (with optional YaRN scaling)
        let rope_scaling: Option<infernum::RopeScaling> =
            config.rope_scaling.as_ref().map(Into::into);
        let (cos_cache, sin_cache) = transformer::build_rope_cache::<B>(
            &device,
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
            rope_scaling.as_ref(),
            dtype,
        )?;

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
    ) -> Result<MlpWeights<B>> {
        transformer::load_mlp_weights(loader, prefix, intermediate_size, dtype, qc)
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
            experts.push(mlp);
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

                QwenFfnWeights::<B>::Dense(Box::new(MlpWeights {
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

        let rope_scaling: Option<infernum::RopeScaling> =
            config.rope_scaling.as_ref().map(Into::into);
        let (cos_cache, sin_cache) = transformer::build_rope_cache::<B>(
            &device,
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
            rope_scaling.as_ref(),
            dtype,
        )?;

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
            experts.push(MlpWeights { gate_up, down_proj });
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
            Some(Box::new(MlpWeights { gate_up, down_proj }))
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
        transformer::load_optional_tensor::<B>(loader, name, dtype)
    }

    fn embed(&self, input_ids: &[u32]) -> Result<B::Tensor> {
        transformer::embed::<B>(&self.embed_tokens, input_ids)
    }

    fn maybe_all_reduce(&self, tensor: &mut B::Tensor) -> Result<()> {
        transformer::maybe_all_reduce::<B>(self.comm.as_ref(), tensor)
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
            let (mut k, mut v) =
                transformer::compute_kv_proj::<B>(&normed, &layer.attention.kv_proj)?;

            transformer::apply_qkv_bias::<B>(
                &mut q,
                &mut k,
                &mut v,
                layer.attention.q_bias.as_ref(),
                layer.attention.k_bias.as_ref(),
                layer.attention.v_bias.as_ref(),
            )?;

            let mut q = q.reshape(&[seq_len, num_heads, head_dim]);
            let mut k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
            let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

            transformer::apply_qk_norm::<B>(
                &mut q,
                &mut k,
                layer.attention.q_norm.as_ref(),
                layer.attention.k_norm.as_ref(),
                num_heads,
                num_kv_heads,
                head_dim,
                self.config.rms_norm_eps,
            )?;

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
    pub fn forward_batch_decode(
        &self,
        token_ids: &[u32],
        paged_kv: &mut B::PagedKvCache,
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<B::Tensor> {
        transformer::forward_batch_decode_host::<B, _>(
            &self.device,
            token_ids,
            block_tables,
            positions,
            |tid, bt, sl, pos, bs, mbps, msl| {
                self.forward_batch_decode_tensors(tid, paged_kv, bt, sl, pos, bs, mbps, msl)
            },
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

        let last = transformer::extract_last_row::<B>(&hidden, seq_len);
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
        let (mut k, mut v) =
            transformer::compute_kv_proj_decode::<B>(hidden, &weights.kv_proj, batch_size)?;

        transformer::apply_qkv_bias::<B>(
            &mut q,
            &mut k,
            &mut v,
            weights.q_bias.as_ref(),
            weights.k_bias.as_ref(),
            weights.v_bias.as_ref(),
        )?;

        let mut q = q.reshape(&[batch_size, num_heads, head_dim]);
        let mut k = k.reshape(&[batch_size, num_kv_heads, head_dim]);
        let v = v.reshape(&[batch_size, num_kv_heads, head_dim]);

        transformer::apply_qk_norm::<B>(
            &mut q,
            &mut k,
            weights.q_norm.as_ref(),
            weights.k_norm.as_ref(),
            num_heads,
            num_kv_heads,
            head_dim,
            self.config.rms_norm_eps,
        )?;

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
        let (mut k, mut v) = transformer::compute_kv_proj::<B>(hidden, &weights.kv_proj)?;

        transformer::apply_qkv_bias::<B>(
            &mut q,
            &mut k,
            &mut v,
            weights.q_bias.as_ref(),
            weights.k_bias.as_ref(),
            weights.v_bias.as_ref(),
        )?;

        let mut q = q.reshape(&[seq_len, num_heads, head_dim]);
        let mut k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
        let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

        transformer::apply_qk_norm::<B>(
            &mut q,
            &mut k,
            weights.q_norm.as_ref(),
            weights.k_norm.as_ref(),
            num_heads,
            num_kv_heads,
            head_dim,
            self.config.rms_norm_eps,
        )?;

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

    fn forward_mlp(&self, hidden: &B::Tensor, weights: &MlpWeights<B>) -> Result<B::Tensor> {
        transformer::forward_mlp::<B>(hidden, weights, self.comm.as_ref())
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
        experts: &[MlpWeights<B>],
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        shared_expert: Option<&MlpWeights<B>>,
        shared_expert_gate: Option<&B::Tensor>,
    ) -> Result<B::Tensor> {
        let mut out = B::moe_forward_softmax(
            hidden,
            gate,
            experts.len(),
            num_experts_per_tok,
            norm_topk_prob,
            |expert_idx, expert_input| {
                transformer::forward_mlp_no_reduce::<B>(expert_input, &experts[expert_idx])
            },
        )?;

        if let Some(shared_mlp) = shared_expert {
            let shared_out = transformer::forward_mlp_no_reduce::<B>(hidden, shared_mlp)?;

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

    fn lm_head_forward(&self, hidden: &B::Tensor) -> Result<B::Tensor> {
        transformer::lm_head_forward::<B>(hidden, &self.lm_head, self.dtype)
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
