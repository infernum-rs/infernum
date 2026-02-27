//! DeepSeek V3 / R1 model implementation — fully generic over the compute backend `B`.
//!
//! All forward pass methods, weight loading, and the model struct are
//! generic over `B: Backend`. No CUDA-specific code remains in this file.

#![allow(
    clippy::similar_names,
    clippy::struct_field_names,
    clippy::no_effect_underscore_binding,
    clippy::doc_markdown,
    dead_code,
    unused_mut
)]

use std::marker::PhantomData;
use std::path::Path;

use infernum::backend::{
    ArithOps, AttentionOps, Backend, CastOps, Comm, EmbedOps, MatmulExtOps, MatmulOps,
    MoeSigmoidOps, NormOps, PagedAttentionOps, PagedKvCacheOps, RopeInterleavedOps, SwigluOps,
    TensorDataOps, TensorFactory, TensorOps,
};
use infernum::dtype::DType;
use infernum::shard::GpuConfig;
use infernum::tensor::Tensor;
use infernum::Result;

use crate::DeepSeekConfig;

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
fn split_kv_b_proj_dense<B: TensorFactory + TensorDataOps>(
    device: &B::DeviceHandle,
    weight: &B::Tensor,
    num_heads: usize,
    qk_nope_dim: usize,
    v_head_dim: usize,
) -> Result<(B::Tensor, B::Tensor, B::Tensor)> {
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

    let data = B::to_raw_bytes(weight)?;

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

    let k_tensor = B::from_raw_bytes(device, &[kv_lora_rank, k_cols], dtype, &k_data)?;
    let v_tensor = B::from_raw_bytes(
        device,
        &[num_heads, kv_lora_rank, v_head_dim],
        dtype,
        &v_data,
    )?;
    let k_t_tensor = B::from_raw_bytes(
        device,
        &[num_heads, qk_nope_dim, kv_lora_rank],
        dtype,
        &k_t_data,
    )?;

    Ok((k_tensor, v_tensor, k_t_tensor))
}

/// Split a quantized `kv_b_proj` by dequantizing to dense first, then splitting.
///
/// Uses identity matmul: `I @ W^T = W^T` to dequantize via `B::linear`.
fn split_kv_b_proj_quantized<B: MatmulOps + CastOps + TensorFactory + TensorDataOps>(
    dtype: DType,
    device: &B::DeviceHandle,
    w: &<B as MatmulOps>::LinearWeight,
    w_shape: &[usize],
    num_heads: usize,
    qk_nope_dim: usize,
    v_head_dim: usize,
) -> Result<(B::Tensor, B::Tensor, B::Tensor)> {
    let out_features = w_shape[0];
    let kv_lora_rank = w_shape[1];
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
    let identity = B::from_f32_slice(device, &[kv_lora_rank, kv_lora_rank], &identity_data)?;

    // Dequantize: I @ W^T → (kv_lora_rank, out_features) as f32
    let dense_f32 = B::linear(&identity, w)?;

    // Convert to target dtype and split
    let dense_t = B::cast_from_f32(&dense_f32, dtype)?;

    split_kv_b_proj_dense::<B>(device, &dense_t, num_heads, qk_nope_dim, v_head_dim)
}

// --- Weight structures ---

/// MLA attention weights (shared by dense and MoE layers)
struct DeepSeekAttentionWeights<B: Backend + MatmulOps> {
    q_a_proj: <B as MatmulOps>::LinearWeight,
    q_a_layernorm: B::Tensor,
    q_b_proj: <B as MatmulOps>::LinearWeight,
    kv_a_proj_with_mqa: <B as MatmulOps>::LinearWeight,
    kv_a_layernorm: B::Tensor,
    kv_b_proj: <B as MatmulOps>::LinearWeight,
    /// K-nope decompression columns, pre-transposed: `(kv_lora_rank, num_heads * qk_nope_dim)`
    kv_b_proj_k: <B as MatmulOps>::LinearWeight,
    /// V decompression per-head: `(num_heads, kv_lora_rank, v_head_dim)` for batched V absorption
    kv_b_proj_v: B::Tensor,
    /// Transposed K per-head: `(num_heads, qk_nope_dim, kv_lora_rank)` for batched Q absorption
    kv_b_proj_k_t: B::Tensor,
    o_proj: <B as MatmulOps>::LinearWeight,
}

/// Dense MLP weights
struct DenseMlpWeights<B: Backend + MatmulOps> {
    gate_proj: <B as MatmulOps>::LinearWeight,
    up_proj: <B as MatmulOps>::LinearWeight,
    down_proj: <B as MatmulOps>::LinearWeight,
}

/// MoE layer weights
struct MoeWeights<B: Backend + MatmulOps> {
    gate_weight: B::Tensor,
    e_score_correction_bias: Vec<f32>,
    experts: Vec<DenseMlpWeights<B>>,
    shared_expert: DenseMlpWeights<B>,
}

/// Dense vs MoE FFN layer
#[allow(clippy::large_enum_variant)]
enum FfnWeights<B: Backend + MatmulOps> {
    Dense(Box<DenseMlpWeights<B>>),
    Moe(Box<MoeWeights<B>>),
}

/// Single transformer layer
struct DeepSeekLayerWeights<B: Backend + MatmulOps> {
    input_layernorm: B::Tensor,
    attention: DeepSeekAttentionWeights<B>,
    post_attention_layernorm: B::Tensor,
    ffn: FfnWeights<B>,
}

// --- DeepSeekOps trait alias ---

/// Combined ops trait for DeepSeek models.
///
/// Bundles all backend bounds required by `DeepSeekModel`, including
/// MLA attention, MoE sigmoid routing, and interleaved RoPE.
pub trait DeepSeekOps:
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
    + RopeInterleavedOps
    + AttentionOps
    + PagedAttentionOps
    + PagedKvCacheOps
    + MoeSigmoidOps
{
}

impl<B> DeepSeekOps for B where
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
        + RopeInterleavedOps
        + AttentionOps
        + PagedAttentionOps
        + PagedKvCacheOps
        + MoeSigmoidOps
{
}

/// Complete DeepSeek V3/R1 model.
pub struct DeepSeekModel<B: Backend + MatmulOps> {
    config: DeepSeekConfig,
    device: B::DeviceHandle,
    #[allow(dead_code)]
    gpu_config: GpuConfig,

    /// Optional communicator for tensor-parallel all-reduce.
    comm: Option<B::Comm>,

    tp_num_heads: usize,
    dtype: DType,

    embed_tokens: B::Tensor,
    layers: Vec<DeepSeekLayerWeights<B>>,
    norm: B::Tensor,
    lm_head: <B as MatmulOps>::LinearWeight,

    cos_cache: B::Tensor,
    sin_cache: B::Tensor,

    /// Pre-computed attention scale (includes YaRN mscale adjustment)
    attn_scale: f32,

    _backend: PhantomData<B>,
}

impl<B: DeepSeekOps> DeepSeekModel<B> {
    /// Load a DeepSeek model from a directory containing SafeTensors and config.json
    ///
    /// # Errors
    /// Returns an error if loading fails
    pub fn from_pretrained(device: &B::DeviceHandle, model_path: impl AsRef<Path>) -> Result<Self>
    where
        B: infernum::SafeTensorsLoaderOps,
    {
        let model_path = model_path.as_ref();
        let config_path = model_path.join("config.json");
        let config = DeepSeekConfig::from_file(&config_path)?;
        let loader = B::safetensors_loader(device, model_path)?;
        Self::load_weights(device.clone(), config, &loader)
    }

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &DeepSeekConfig {
        &self.config
    }

    /// Get the model's compute dtype
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    // --- Weight loading ---

    #[allow(clippy::too_many_lines)]
    fn load_weights(
        device: B::DeviceHandle,
        config: DeepSeekConfig,
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

            // MLA attention weights
            let kv_b_proj =
                loader.load_linear(&format!("{prefix}.self_attn.kv_b_proj.weight"), dtype, qc)?;

            // Split kv_b_proj into K-nope and V portions for absorbed attention
            let kv_b_proj_shape =
                loader.get_shape(&format!("{prefix}.self_attn.kv_b_proj.weight"))?;
            let (kv_b_proj_k_dense, kv_b_proj_v, kv_b_proj_k_t) = if B::is_dense_weight(&kv_b_proj)
            {
                let w = B::as_dense_weight(&kv_b_proj)
                    .expect("is_dense_weight returned true but as_dense_weight is None");
                let (k, v, k_t) = split_kv_b_proj_dense::<B>(
                    &device,
                    w,
                    config.num_attention_heads,
                    config.qk_nope_head_dim,
                    config.v_head_dim,
                )?;
                (k, v, k_t)
            } else {
                let (k, v, k_t) = split_kv_b_proj_quantized::<B>(
                    dtype,
                    &device,
                    &kv_b_proj,
                    &kv_b_proj_shape,
                    config.num_attention_heads,
                    config.qk_nope_head_dim,
                    config.v_head_dim,
                )?;
                (k, v, k_t)
            };
            let kv_b_proj_k = B::dense_weight(kv_b_proj_k_dense);

            let attention = DeepSeekAttentionWeights {
                q_a_proj: loader.load_linear(
                    &format!("{prefix}.self_attn.q_a_proj.weight"),
                    dtype,
                    qc,
                )?,
                q_a_layernorm: loader
                    .load_tensor(&format!("{prefix}.self_attn.q_a_layernorm.weight"), dtype)?,
                q_b_proj: loader.load_linear(
                    &format!("{prefix}.self_attn.q_b_proj.weight"),
                    dtype,
                    qc,
                )?,
                kv_a_proj_with_mqa: loader.load_linear(
                    &format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
                    dtype,
                    qc,
                )?,
                kv_a_layernorm: loader
                    .load_tensor(&format!("{prefix}.self_attn.kv_a_layernorm.weight"), dtype)?,
                kv_b_proj,
                kv_b_proj_k,
                kv_b_proj_v,
                kv_b_proj_k_t,
                o_proj: loader.load_linear(
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    dtype,
                    qc,
                )?,
            };

            // FFN: dense or MoE
            let ffn = if config.is_moe_layer(i) {
                Self::load_moe_weights(&config, &prefix, dtype, qc, loader, &device)?
            } else {
                let mp = format!("{prefix}.mlp");
                FfnWeights::Dense(Box::new(DenseMlpWeights {
                    gate_proj: loader.load_linear(&format!("{mp}.gate_proj.weight"), dtype, qc)?,
                    up_proj: loader.load_linear(&format!("{mp}.up_proj.weight"), dtype, qc)?,
                    down_proj: loader.load_linear(&format!("{mp}.down_proj.weight"), dtype, qc)?,
                }))
            };

            layers.push(DeepSeekLayerWeights {
                input_layernorm: loader
                    .load_tensor(&format!("{prefix}.input_layernorm.weight"), dtype)?,
                attention,
                post_attention_layernorm: loader
                    .load_tensor(&format!("{prefix}.post_attention_layernorm.weight"), dtype)?,
                ffn,
            });
        }

        let norm = loader.load_tensor("model.norm.weight", dtype)?;

        let lm_head = transformer::load_lm_head::<B>(
            &device,
            loader,
            &embed_tokens,
            config.tie_word_embeddings,
            dtype,
            config.quantization_config.as_ref(),
        )?;

        // RoPE cache — use qk_rope_head_dim (only rope portion gets RoPE)
        let rope_scaling = config
            .rope_scaling
            .as_ref()
            .map(|rs| infernum::RopeScaling {
                rope_type: rs.rope_type.clone(),
                factor: rs.factor,
                original_max_position_embeddings: rs.original_max_position_embeddings,
            });
        let (cos_cache, sin_cache) = transformer::build_rope_cache::<B>(
            &device,
            config.max_position_embeddings,
            config.qk_rope_head_dim,
            config.rope_theta,
            rope_scaling.as_ref(),
            dtype,
        )?;

        let attn_scale = config.mla_attn_scale();

        Ok(Self {
            tp_num_heads: config.num_attention_heads,
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
            attn_scale,
            _backend: PhantomData,
        })
    }

    /// Load MoE weights for a single layer.
    fn load_moe_weights(
        config: &DeepSeekConfig,
        prefix: &str,
        dtype: DType,
        qc: Option<&crate::config::QuantizationConfig>,
        loader: &impl infernum::WeightLoader<B>,
        _device: &B::DeviceHandle,
    ) -> Result<FfnWeights<B>> {
        let num_experts = config
            .n_routed_experts
            .expect("MoE layer requires n_routed_experts");

        // Router gate — load as dense tensor, transpose for matmul-ready layout
        let gate_f32 = loader.load_tensor(&format!("{prefix}.mlp.gate.weight"), DType::F32)?;
        let gate_transposed = B::transpose_2d(&gate_f32)?;
        let gate_weight = B::cast_from_f32(&gate_transposed, dtype)?;

        // Bias correction
        let bias_name = format!("{prefix}.mlp.gate.e_score_correction_bias");
        let e_score_correction_bias = if loader.contains(&bias_name) {
            B::to_f32_vec(&loader.load_tensor(&bias_name, DType::F32)?)?
        } else {
            vec![0.0_f32; num_experts]
        };

        // Per-expert MLPs
        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let ep = format!("{prefix}.mlp.experts.{e}");
            experts.push(DenseMlpWeights {
                gate_proj: loader.load_linear(&format!("{ep}.gate_proj.weight"), dtype, qc)?,
                up_proj: loader.load_linear(&format!("{ep}.up_proj.weight"), dtype, qc)?,
                down_proj: loader.load_linear(&format!("{ep}.down_proj.weight"), dtype, qc)?,
            });
        }

        // Shared expert
        let sp = format!("{prefix}.mlp.shared_experts");
        let shared_expert = DenseMlpWeights {
            gate_proj: loader.load_linear(&format!("{sp}.gate_proj.weight"), dtype, qc)?,
            up_proj: loader.load_linear(&format!("{sp}.up_proj.weight"), dtype, qc)?,
            down_proj: loader.load_linear(&format!("{sp}.down_proj.weight"), dtype, qc)?,
        };

        Ok(FfnWeights::Moe(Box::new(MoeWeights {
            gate_weight,
            e_score_correction_bias,
            experts,
            shared_expert,
        })))
    }

    /// Load a DeepSeek model with tensor-parallel sharding across multiple GPUs.
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
        let config_path = model_path.join("config.json");
        let config = DeepSeekConfig::from_file(&config_path)?;
        let loader = B::safetensors_loader(device, model_path)?;
        Self::load_weights_sharded(device.clone(), config, &loader, gpu_config, comm)
    }

    /// Optional all-reduce for tensor-parallel models. No-op for single GPU.
    fn maybe_all_reduce(&self, tensor: &mut B::Tensor) -> Result<()> {
        if let Some(comm) = &self.comm {
            comm.all_reduce_sum(tensor)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn load_weights_sharded(
        device: B::DeviceHandle,
        config: DeepSeekConfig,
        loader: &impl infernum::WeightLoader<B>,
        gpu_config: GpuConfig,
        comm: Option<B::Comm>,
    ) -> Result<Self> {
        use infernum::shard::ShardStrategy;

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

        let qc = config.quantization_config.as_ref();
        let tp_num_heads = config.num_attention_heads / world_size;

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

            // MLA attention weights
            // q_a_proj, kv_a_proj_with_mqa: replicated (shared bottleneck)
            // q_b_proj, kv_b_proj: column-sharded (output is per-head)
            // o_proj: row-sharded (all-reduce after)
            let kv_b_proj = loader.load_linear_sharded(
                &format!("{prefix}.self_attn.kv_b_proj.weight"),
                dtype,
                qc,
                &shard,
                ShardStrategy::Column,
            )?;

            // Split kv_b_proj into K-nope and V portions (using tp_num_heads after sharding)
            let kv_b_proj_shape =
                loader.get_shape(&format!("{prefix}.self_attn.kv_b_proj.weight"))?;
            let (kv_b_proj_k_dense, kv_b_proj_v, kv_b_proj_k_t) = if B::is_dense_weight(&kv_b_proj)
            {
                let w = B::as_dense_weight(&kv_b_proj)
                    .expect("is_dense_weight returned true but as_dense_weight is None");
                split_kv_b_proj_dense::<B>(
                    &device,
                    w,
                    tp_num_heads,
                    config.qk_nope_head_dim,
                    config.v_head_dim,
                )?
            } else {
                // For sharded quantized, adjust shape for the shard
                let mut shard_shape = kv_b_proj_shape.clone();
                shard_shape[0] /= world_size; // Column-sharded: out_features divided
                split_kv_b_proj_quantized::<B>(
                    dtype,
                    &device,
                    &kv_b_proj,
                    &shard_shape,
                    tp_num_heads,
                    config.qk_nope_head_dim,
                    config.v_head_dim,
                )?
            };
            let kv_b_proj_k = B::dense_weight(kv_b_proj_k_dense);

            let attention = DeepSeekAttentionWeights {
                q_a_proj: loader.load_linear_sharded(
                    &format!("{prefix}.self_attn.q_a_proj.weight"),
                    dtype,
                    qc,
                    &shard,
                    ShardStrategy::Replicate,
                )?,
                q_a_layernorm: loader
                    .load_tensor(&format!("{prefix}.self_attn.q_a_layernorm.weight"), dtype)?,
                q_b_proj: loader.load_linear_sharded(
                    &format!("{prefix}.self_attn.q_b_proj.weight"),
                    dtype,
                    qc,
                    &shard,
                    ShardStrategy::Column,
                )?,
                kv_a_proj_with_mqa: loader.load_linear_sharded(
                    &format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
                    dtype,
                    qc,
                    &shard,
                    ShardStrategy::Replicate,
                )?,
                kv_a_layernorm: loader
                    .load_tensor(&format!("{prefix}.self_attn.kv_a_layernorm.weight"), dtype)?,
                kv_b_proj,
                kv_b_proj_k,
                kv_b_proj_v,
                kv_b_proj_k_t,
                o_proj: loader.load_linear_sharded(
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    dtype,
                    qc,
                    &shard,
                    ShardStrategy::Row,
                )?,
            };

            // FFN: dense or MoE (shared experts replicated, per-expert sharded)
            let ffn = if config.is_moe_layer(i) {
                Self::load_moe_weights_sharded(
                    &config, &prefix, dtype, qc, loader, &device, &shard,
                )?
            } else {
                let mp = format!("{prefix}.mlp");
                FfnWeights::Dense(Box::new(DenseMlpWeights {
                    gate_proj: loader.load_linear_sharded(
                        &format!("{mp}.gate_proj.weight"),
                        dtype,
                        qc,
                        &shard,
                        ShardStrategy::Column,
                    )?,
                    up_proj: loader.load_linear_sharded(
                        &format!("{mp}.up_proj.weight"),
                        dtype,
                        qc,
                        &shard,
                        ShardStrategy::Column,
                    )?,
                    down_proj: loader.load_linear_sharded(
                        &format!("{mp}.down_proj.weight"),
                        dtype,
                        qc,
                        &shard,
                        ShardStrategy::Row,
                    )?,
                }))
            };

            layers.push(DeepSeekLayerWeights {
                input_layernorm: loader
                    .load_tensor(&format!("{prefix}.input_layernorm.weight"), dtype)?,
                attention,
                post_attention_layernorm: loader
                    .load_tensor(&format!("{prefix}.post_attention_layernorm.weight"), dtype)?,
                ffn,
            });
        }

        let norm = loader.load_tensor("model.norm.weight", dtype)?;

        let lm_head = if config.tie_word_embeddings {
            transformer::load_lm_head::<B>(
                &device,
                loader,
                &embed_tokens,
                true,
                dtype,
                config.quantization_config.as_ref(),
            )?
        } else {
            loader.load_linear_sharded(
                "lm_head.weight",
                dtype,
                None,
                &shard,
                ShardStrategy::Replicate,
            )?
        };

        let rope_scaling = config
            .rope_scaling
            .as_ref()
            .map(|rs| infernum::RopeScaling {
                rope_type: rs.rope_type.clone(),
                factor: rs.factor,
                original_max_position_embeddings: rs.original_max_position_embeddings,
            });
        let (cos_cache, sin_cache) = transformer::build_rope_cache::<B>(
            &device,
            config.max_position_embeddings,
            config.qk_rope_head_dim,
            config.rope_theta,
            rope_scaling.as_ref(),
            dtype,
        )?;

        let attn_scale = config.mla_attn_scale();

        Ok(Self {
            tp_num_heads,
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
            attn_scale,
            _backend: PhantomData,
        })
    }

    /// Load sharded MoE weights for a single layer.
    #[allow(clippy::too_many_arguments)]
    fn load_moe_weights_sharded(
        config: &DeepSeekConfig,
        prefix: &str,
        dtype: DType,
        qc: Option<&crate::config::QuantizationConfig>,
        loader: &impl infernum::WeightLoader<B>,
        _device: &B::DeviceHandle,
        shard: &infernum::shard::ShardConfig,
    ) -> Result<FfnWeights<B>> {
        use infernum::shard::ShardStrategy;

        let num_experts = config
            .n_routed_experts
            .expect("MoE layer requires n_routed_experts");

        // Router gate: replicated
        let gate_f32 = loader.load_tensor(&format!("{prefix}.mlp.gate.weight"), DType::F32)?;
        let gate_transposed = B::transpose_2d(&gate_f32)?;
        let gate_weight = B::cast_from_f32(&gate_transposed, dtype)?;

        // Bias correction: replicated
        let bias_name = format!("{prefix}.mlp.gate.e_score_correction_bias");
        let e_score_correction_bias = if loader.contains(&bias_name) {
            B::to_f32_vec(&loader.load_tensor(&bias_name, DType::F32)?)?
        } else {
            vec![0.0_f32; num_experts]
        };

        // Per-expert MLPs: gate/up column-sharded, down row-sharded
        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let ep = format!("{prefix}.mlp.experts.{e}");
            experts.push(DenseMlpWeights {
                gate_proj: loader.load_linear_sharded(
                    &format!("{ep}.gate_proj.weight"),
                    dtype,
                    qc,
                    shard,
                    ShardStrategy::Column,
                )?,
                up_proj: loader.load_linear_sharded(
                    &format!("{ep}.up_proj.weight"),
                    dtype,
                    qc,
                    shard,
                    ShardStrategy::Column,
                )?,
                down_proj: loader.load_linear_sharded(
                    &format!("{ep}.down_proj.weight"),
                    dtype,
                    qc,
                    shard,
                    ShardStrategy::Row,
                )?,
            });
        }

        // Shared expert: gate/up column-sharded, down row-sharded
        let sp = format!("{prefix}.mlp.shared_experts");
        let shared_expert = DenseMlpWeights {
            gate_proj: loader.load_linear_sharded(
                &format!("{sp}.gate_proj.weight"),
                dtype,
                qc,
                shard,
                ShardStrategy::Column,
            )?,
            up_proj: loader.load_linear_sharded(
                &format!("{sp}.up_proj.weight"),
                dtype,
                qc,
                shard,
                ShardStrategy::Column,
            )?,
            down_proj: loader.load_linear_sharded(
                &format!("{sp}.down_proj.weight"),
                dtype,
                qc,
                shard,
                ShardStrategy::Row,
            )?,
        };

        Ok(FfnWeights::Moe(Box::new(MoeWeights {
            gate_weight,
            e_score_correction_bias,
            experts,
            shared_expert,
        })))
    }

    // --- Forward pass ---

    fn embed(&self, input_ids: &[u32]) -> Result<B::Tensor> {
        transformer::embed::<B>(&self.embed_tokens, input_ids)
    }

    #[allow(clippy::unused_self)]
    fn extract_last_row(&self, hidden: &B::Tensor, seq_len: usize) -> B::Tensor {
        transformer::extract_last_row::<B>(hidden, seq_len)
    }

    fn lm_head_forward(&self, hidden: &B::Tensor) -> Result<B::Tensor> {
        transformer::lm_head_forward::<B>(hidden, &self.lm_head, self.dtype)
    }

    // --- MLA attention helpers ---

    /// Split a 2D tensor `[seq, total_dim]` into two parts along the last dimension.
    fn split_last_dim(
        tensor: &B::Tensor,
        dim1: usize,
        dim2: usize,
    ) -> Result<(B::Tensor, B::Tensor)> {
        let seq_len = tensor.shape()[0];
        let total = tensor.shape()[1];
        assert_eq!(total, dim1 + dim2, "split_last_dim: dim mismatch");

        if seq_len == 1 {
            let a = tensor.slice_view(0, &[1, dim1]);
            let b = tensor.slice_view(dim1, &[1, dim2]);
            Ok((a, b))
        } else {
            B::split_inner_dim(tensor, dim1, dim2)
        }
    }

    /// Split a 3D tensor `[seq, num_heads, total_dim]` into two parts along last dim.
    fn split_head_dim(
        tensor: &B::Tensor,
        dim1: usize,
        dim2: usize,
    ) -> Result<(B::Tensor, B::Tensor)> {
        let seq_len = tensor.shape()[0];
        let num_heads = tensor.shape()[1];
        let total = tensor.shape()[2];
        assert_eq!(total, dim1 + dim2, "split_head_dim: dim mismatch");

        let outer = seq_len * num_heads;
        let flat = tensor.reshape(&[outer, total]);
        let (a_flat, b_flat) = B::split_inner_dim(&flat, dim1, dim2)?;
        let a = a_flat.reshape(&[seq_len, num_heads, dim1]);
        let b = b_flat.reshape(&[seq_len, num_heads, dim2]);
        Ok((a, b))
    }

    /// Concatenate two 3D tensors along the last dimension.
    fn concat_head_dim(a: &B::Tensor, b: &B::Tensor) -> Result<B::Tensor> {
        let seq_len = a.shape()[0];
        let num_heads = a.shape()[1];
        let dim1 = a.shape()[2];
        let dim2 = b.shape()[2];
        assert_eq!(seq_len, b.shape()[0]);
        assert_eq!(num_heads, b.shape()[1]);

        let outer = seq_len * num_heads;
        let a_flat = a.reshape(&[outer, dim1]);
        let b_flat = b.reshape(&[outer, dim2]);
        let out_flat = B::concat_inner_dim(&a_flat, &b_flat)?;
        Ok(out_flat.reshape(&[seq_len, num_heads, dim1 + dim2]))
    }

    /// Broadcast `[seq, 1, rope_dim]` to `[seq, num_heads, rope_dim]`.
    fn broadcast_kv_rope(k_rope: &B::Tensor, num_heads: usize) -> Result<B::Tensor> {
        B::broadcast_to_heads(k_rope, num_heads)
    }

    /// Pad V from `[seq, num_heads, v_head_dim]` to `[seq, num_heads, qk_head_dim]`.
    fn pad_v_to_qk_dim(v: &B::Tensor, qk_head_dim: usize) -> Result<B::Tensor> {
        let seq_len = v.shape()[0];
        let num_heads = v.shape()[1];
        let v_dim = v.shape()[2];
        if v_dim == qk_head_dim {
            return Ok(v.slice_view(0, v.shape()));
        }

        let outer = seq_len * num_heads;
        let flat = v.reshape(&[outer, v_dim]);
        let padded = B::pad_inner_dim(&flat, qk_head_dim)?;
        Ok(padded.reshape(&[seq_len, num_heads, qk_head_dim]))
    }

    /// Truncate attention output from `[seq, num_heads, qk_head_dim]` to
    /// `[seq, num_heads, v_head_dim]`.
    fn truncate_attn_output(attn_out: &B::Tensor, v_head_dim: usize) -> Result<B::Tensor> {
        let seq_len = attn_out.shape()[0];
        let num_heads = attn_out.shape()[1];
        let qk_dim = attn_out.shape()[2];
        if qk_dim == v_head_dim {
            return Ok(attn_out.slice_view(0, attn_out.shape()));
        }

        let outer = seq_len * num_heads;
        let flat = attn_out.reshape(&[outer, qk_dim]);
        let discard_dim = qk_dim - v_head_dim;
        let (kept, _) = B::split_inner_dim(&flat, v_head_dim, discard_dim)?;
        Ok(kept.reshape(&[seq_len, num_heads, v_head_dim]))
    }

    /// Absorb K into Q: per-head `q_nope_h @ W_k_h^T` → `q_absorbed_nope`.
    ///
    /// `q_nope`: `(1, num_heads, qk_nope_dim)`
    /// `kv_b_proj_k_t`: `(num_heads, qk_nope_dim, kv_lora_rank)`
    /// Returns: `(1, num_heads, kv_lora_rank)`
    fn absorb_k_into_q(q_nope: &B::Tensor, kv_b_proj_k_t: &B::Tensor) -> Result<B::Tensor> {
        let num_heads = q_nope.shape()[1];
        let d = q_nope.shape()[2];
        let kv_lora_rank = kv_b_proj_k_t.shape()[2];
        // (num_heads, 1, D) @ (num_heads, D, R) → (num_heads, 1, R) → (1, H, R)
        let q = q_nope.reshape(&[num_heads, 1, d]);
        let out = B::matmul(&q, kv_b_proj_k_t)?;
        Ok(out.reshape(&[1, num_heads, kv_lora_rank]))
    }

    /// Absorb V: per-head `attn_nope_h @ W_v_h` → V output.
    ///
    /// `attn_nope`: `(1, num_heads, kv_lora_rank)`
    /// `kv_b_proj_v`: `(num_heads, kv_lora_rank, v_head_dim)`
    /// Returns: `(1, num_heads, v_head_dim)`
    fn absorb_v(attn_nope: &B::Tensor, kv_b_proj_v: &B::Tensor) -> Result<B::Tensor> {
        let num_heads = attn_nope.shape()[1];
        let r = attn_nope.shape()[2];
        let v_head_dim = kv_b_proj_v.shape()[2];
        // (num_heads, 1, R) @ (num_heads, R, V) → (num_heads, 1, V) → (1, H, V)
        let a = attn_nope.reshape(&[num_heads, 1, r]);
        let out = B::matmul(&a, kv_b_proj_v)?;
        Ok(out.reshape(&[1, num_heads, v_head_dim]))
    }

    /// Batched K absorption for batch > 1 (multi-chunk prefill).
    ///
    /// `q_nope`: `(batch, num_heads, qk_nope_dim)` — contiguous
    /// `kv_b_proj_k_t`: `(num_heads, qk_nope_dim, kv_lora_rank)`
    /// Returns: `(batch, num_heads, kv_lora_rank)`
    fn absorb_k_batched(
        device: &B::DeviceHandle,
        q_nope: &B::Tensor,
        kv_b_proj_k_t: &B::Tensor,
        batch_size: usize,
        num_heads: usize,
        kv_lora_rank: usize,
    ) -> Result<B::Tensor> {
        let d = q_nope.shape()[2];
        let item_elems = num_heads * d;

        let elem = q_nope.dtype().size_in_bytes();
        let out_elems = num_heads * kv_lora_rank;
        let mut all_bytes: Vec<u8> = Vec::with_capacity(batch_size * out_elems * elem);
        for b in 0..batch_size {
            let q_b = q_nope.slice_view(b * item_elems, &[num_heads, 1, d]);
            let out_b = B::matmul(&q_b, kv_b_proj_k_t)?; // (H, 1, R)
            all_bytes.extend_from_slice(&B::to_raw_bytes(&out_b)?);
        }
        B::from_raw_bytes(
            device,
            &[batch_size, num_heads, kv_lora_rank],
            q_nope.dtype(),
            &all_bytes,
        )
    }

    /// Batched V absorption for batch > 1 (multi-chunk prefill).
    ///
    /// `attn_nope`: `(batch, num_heads, kv_lora_rank)`
    /// `kv_b_proj_v`: `(num_heads, kv_lora_rank, v_head_dim)`
    /// Returns: `(batch, num_heads, v_head_dim)`
    fn absorb_v_batched(
        device: &B::DeviceHandle,
        attn_nope: &B::Tensor,
        kv_b_proj_v: &B::Tensor,
        batch_size: usize,
        num_heads: usize,
        v_head_dim: usize,
    ) -> Result<B::Tensor> {
        let r = attn_nope.shape()[2];
        let item_elems = num_heads * r;
        let out_elems = num_heads * v_head_dim;

        let elem = attn_nope.dtype().size_in_bytes();
        let mut all_bytes: Vec<u8> = Vec::with_capacity(batch_size * out_elems * elem);
        for b in 0..batch_size {
            let a_b = attn_nope.slice_view(b * item_elems, &[num_heads, 1, r]);
            let out_b = B::matmul(&a_b, kv_b_proj_v)?; // (H, 1, V)
            all_bytes.extend_from_slice(&B::to_raw_bytes(&out_b)?);
        }
        B::from_raw_bytes(
            device,
            &[batch_size, num_heads, v_head_dim],
            attn_nope.dtype(),
            &all_bytes,
        )
    }

    // --- MLA attention for paged KV ---

    /// MLA attention for single-sequence prefill with paged KV cache.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn forward_mla_attention_paged_prefill(
        &self,
        hidden: &B::Tensor,
        weights: &DeepSeekAttentionWeights<B>,
        layer_idx: usize,
        paged_kv: &mut B::PagedKvCache,
        block_table: &infernum::BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<B::Tensor> {
        let num_heads = self.tp_num_heads;
        let qk_nope_dim = self.config.qk_nope_head_dim;
        let qk_rope_dim = self.config.qk_rope_head_dim;
        let qk_head_dim = self.config.qk_head_dim();
        let v_head_dim = self.config.v_head_dim;
        let kv_lora_rank = self.config.kv_lora_rank;

        // Q projection (two-stage LoRA)
        let q_compressed = B::linear(hidden, &weights.q_a_proj)?;
        let q_compressed = B::rms_norm(
            &q_compressed,
            &weights.q_a_layernorm,
            self.config.rms_norm_eps,
        )?;
        let q = B::linear(&q_compressed, &weights.q_b_proj)?;
        let q = q.reshape(&[seq_len, num_heads, qk_head_dim]);
        let (q_nope, q_rope) = Self::split_head_dim(&q, qk_nope_dim, qk_rope_dim)?;

        // KV joint projection
        let kv = B::linear(hidden, &weights.kv_a_proj_with_mqa)?;
        let (k_compressed, k_rope) = Self::split_last_dim(&kv, kv_lora_rank, qk_rope_dim)?;
        let k_compressed = B::rms_norm(
            &k_compressed,
            &weights.kv_a_layernorm,
            self.config.rms_norm_eps,
        )?;
        let kv_decompressed = B::linear(&k_compressed, &weights.kv_b_proj)?;
        let kv_decompressed =
            kv_decompressed.reshape(&[seq_len, num_heads, qk_nope_dim + v_head_dim]);
        let (k_nope, v) = Self::split_head_dim(&kv_decompressed, qk_nope_dim, v_head_dim)?;

        // RoPE (interleaved)
        let k_rope = k_rope.reshape(&[seq_len, 1, qk_rope_dim]);
        let q_rope =
            B::apply_rope_interleaved(&q_rope, &self.cos_cache, &self.sin_cache, start_pos)?;
        let k_rope =
            B::apply_rope_interleaved(&k_rope, &self.cos_cache, &self.sin_cache, start_pos)?;

        // Write compressed entry to paged cache: (seq_len, 1, kv_lora_rank + qk_rope_dim)
        let k_rope_2d = k_rope.reshape(&[seq_len, qk_rope_dim]);
        let cache_entry_2d = B::concat_inner_dim(&k_compressed, &k_rope_2d)?;
        let cache_entry = cache_entry_2d.reshape(&[seq_len, 1, kv_lora_rank + qk_rope_dim]);
        B::append_paged(
            paged_kv,
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

            let out = B::fused_attention_prefill(
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
                &self.device,
                &q_nope,
                &weights.kv_b_proj_k_t,
                seq_len,
                num_heads,
                kv_lora_rank,
            )?;
            let q_absorbed = Self::concat_head_dim(&q_absorbed_nope, &q_rope)?;
            let (cached_k, _cached_v) = B::gather_paged_kv(paged_kv, layer_idx, block_table)?;
            // cached_k: (start_pos, 1, kv_lora_rank + qk_rope_dim) — no causal mask needed
            let (out_cached, lse_cached) = B::fused_attention_prefill_with_lse(
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
                &self.device,
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
            let (out_local, lse_local) = B::fused_attention_prefill_with_lse(
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
            B::combine_attention_with_lse(&out_cached_v, &lse_cached, &out_local_v, &lse_local)?
        };
        let attn_output = attn_output.reshape(&[seq_len, num_heads * v_head_dim]);
        let mut out = B::linear(&attn_output, &weights.o_proj)?;
        self.maybe_all_reduce(&mut out)?;
        Ok(out)
    }

    /// MLA attention for single-token decode with paged KV cache.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn forward_mla_attention_paged_decode(
        &self,
        hidden: &B::Tensor,
        weights: &DeepSeekAttentionWeights<B>,
        layer_idx: usize,
        paged_kv: &mut B::PagedKvCache,
        block_table: &infernum::BlockTable,
        position: usize,
    ) -> Result<B::Tensor> {
        let num_heads = self.tp_num_heads;
        let qk_nope_dim = self.config.qk_nope_head_dim;
        let qk_rope_dim = self.config.qk_rope_head_dim;
        let qk_head_dim = self.config.qk_head_dim();
        let v_head_dim = self.config.v_head_dim;
        let kv_lora_rank = self.config.kv_lora_rank;

        // Q projection (two-stage LoRA)
        let q_compressed = B::linear(hidden, &weights.q_a_proj)?;
        let q_compressed = B::rms_norm(
            &q_compressed,
            &weights.q_a_layernorm,
            self.config.rms_norm_eps,
        )?;
        let q = B::linear(&q_compressed, &weights.q_b_proj)?;
        let q = q.reshape(&[1, num_heads, qk_head_dim]);
        let (q_nope, q_rope) = Self::split_head_dim(&q, qk_nope_dim, qk_rope_dim)?;

        // KV joint projection (compressed only — no decompression for absorbed decode)
        let kv = B::linear(hidden, &weights.kv_a_proj_with_mqa)?;
        let (k_compressed, k_rope) = Self::split_last_dim(&kv, kv_lora_rank, qk_rope_dim)?;
        let k_compressed = B::rms_norm(
            &k_compressed,
            &weights.kv_a_layernorm,
            self.config.rms_norm_eps,
        )?;

        // RoPE (interleaved)
        let k_rope = k_rope.reshape(&[1, 1, qk_rope_dim]);
        let q_rope =
            B::apply_rope_interleaved(&q_rope, &self.cos_cache, &self.sin_cache, position)?;
        let k_rope =
            B::apply_rope_interleaved(&k_rope, &self.cos_cache, &self.sin_cache, position)?;

        // Write compressed entry to paged cache: (1, 1, kv_lora_rank + qk_rope_dim)
        let k_rope_2d = k_rope.reshape(&[1, qk_rope_dim]);
        let cache_entry_2d = B::concat_inner_dim(&k_compressed, &k_rope_2d)?;
        let cache_entry = cache_entry_2d.reshape(&[1, 1, kv_lora_rank + qk_rope_dim]);
        B::append_paged(
            paged_kv,
            layer_idx,
            block_table,
            &cache_entry,
            &cache_entry,
            position,
        )?;

        // Absorbed decode attention (Q absorption + paged attention in compressed space)
        let q_absorbed_nope = Self::absorb_k_into_q(&q_nope, &weights.kv_b_proj_k_t)?;
        let q_absorbed = Self::concat_head_dim(&q_absorbed_nope, &q_rope)?;

        let mut table_with_current = block_table.clone();
        table_with_current.advance(1);

        // Build device tensors for single-sequence paged attention
        let max_blocks_per_seq = table_with_current.blocks().len();
        #[allow(clippy::cast_possible_wrap)]
        let bt_i32: Vec<i32> = table_with_current
            .blocks()
            .iter()
            .map(|&b| b as i32)
            .collect();
        let block_tables_tensor = B::from_i32_slice(&self.device, &[max_blocks_per_seq], &bt_i32)?;
        let seq_len_val = table_with_current.seq_len() as i32;
        let seq_lens_tensor = B::from_i32_slice(&self.device, &[1], &[seq_len_val])?;

        let (k_pool, _v_pool) = B::get_pools(paged_kv, layer_idx);
        let attn_output = B::paged_attention_decode(
            &q_absorbed,
            k_pool,
            k_pool,
            &block_tables_tensor,
            &seq_lens_tensor,
            B::block_size(paged_kv),
            max_blocks_per_seq,
            table_with_current.seq_len(),
            Some(self.attn_scale),
            None,
            None,
        )?;

        // Absorb V: discard rope portion, decompress via kv_b_proj_v
        let (attn_nope, _) = Self::split_head_dim(&attn_output, kv_lora_rank, qk_rope_dim)?;
        let attn_output = Self::absorb_v(&attn_nope, &weights.kv_b_proj_v)?;

        let attn_output = attn_output.reshape(&[1, num_heads * v_head_dim]);
        let mut out = B::linear(&attn_output, &weights.o_proj)?;
        self.maybe_all_reduce(&mut out)?;
        Ok(out)
    }

    // --- FFN ---

    fn forward_mlp(&self, hidden: &B::Tensor, weights: &DenseMlpWeights<B>) -> Result<B::Tensor> {
        let gate = B::linear(hidden, &weights.gate_proj)?;
        let up = B::linear(hidden, &weights.up_proj)?;
        let intermediate = B::swiglu(&gate, &up)?;
        let mut out = B::linear(&intermediate, &weights.down_proj)?;
        self.maybe_all_reduce(&mut out)?;
        Ok(out)
    }

    #[allow(clippy::unused_self)]
    fn forward_mlp_no_reduce(
        hidden: &B::Tensor,
        weights: &DenseMlpWeights<B>,
    ) -> Result<B::Tensor> {
        let gate = B::linear(hidden, &weights.gate_proj)?;
        let up = B::linear(hidden, &weights.up_proj)?;
        let intermediate = B::swiglu(&gate, &up)?;
        B::linear(&intermediate, &weights.down_proj)
    }

    fn forward_moe(&self, hidden: &B::Tensor, moe_weights: &MoeWeights<B>) -> Result<B::Tensor> {
        let num_experts = moe_weights.experts.len();

        let mut routed_output = B::moe_forward_sigmoid(
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
                Self::forward_mlp_no_reduce(expert_input, &moe_weights.experts[expert_idx])
            },
        )?;

        // Add shared expert output
        let shared_output = Self::forward_mlp_no_reduce(hidden, &moe_weights.shared_expert)?;
        B::add_inplace(&mut routed_output, &shared_output)?;

        self.maybe_all_reduce(&mut routed_output)?;
        Ok(routed_output)
    }

    fn forward_ffn(&self, hidden: &B::Tensor, ffn: &FfnWeights<B>) -> Result<B::Tensor> {
        match ffn {
            FfnWeights::Dense(mlp) => self.forward_mlp(hidden, mlp),
            FfnWeights::Moe(moe) => self.forward_moe(hidden, moe),
        }
    }

    // --- Layer forward ---

    /// Layer forward for paged prefill.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_paged_prefill(
        &self,
        hidden: &B::Tensor,
        layer: &DeepSeekLayerWeights<B>,
        layer_idx: usize,
        paged_kv: &mut B::PagedKvCache,
        block_table: &infernum::BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<B::Tensor> {
        let normed = B::rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;
        let attn_output = self.forward_mla_attention_paged_prefill(
            &normed,
            &layer.attention,
            layer_idx,
            paged_kv,
            block_table,
            start_pos,
            seq_len,
        )?;

        let (mut h, normed) = B::add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
        B::add_inplace(&mut h, &mlp_output)?;
        Ok(h)
    }

    /// Layer forward for paged decode (single token).
    fn forward_layer_paged_decode(
        &self,
        hidden: &B::Tensor,
        layer: &DeepSeekLayerWeights<B>,
        layer_idx: usize,
        paged_kv: &mut B::PagedKvCache,
        block_table: &infernum::BlockTable,
        position: usize,
    ) -> Result<B::Tensor> {
        let normed = B::rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;
        let attn_output = self.forward_mla_attention_paged_decode(
            &normed,
            &layer.attention,
            layer_idx,
            paged_kv,
            block_table,
            position,
        )?;

        let (mut h, normed) = B::add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
        B::add_inplace(&mut h, &mlp_output)?;
        Ok(h)
    }

    // --- Public forward methods ---

    /// Build the runtime-facing [`ModelConfig`](infernum::ModelConfig).
    #[must_use]
    pub fn model_config(&self) -> infernum::ModelConfig {
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

    /// Full forward pass without KV cache.
    ///
    /// MLA attention is skipped (not meaningful without cache); only the FFN
    /// path is exercised. Use paged forward methods for real inference.
    ///
    /// # Errors
    /// Returns an error if any GPU operation fails.
    pub fn forward_full(&self, input_ids: &[u32]) -> Result<B::Tensor> {
        let seq_len = input_ids.len();
        let mut hidden = self.embed(input_ids)?;

        for layer in &self.layers {
            let normed = B::rms_norm(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            // Simplified: skip attention, just do FFN
            let (mut h, normed) = B::add_rmsnorm(
                &hidden,
                &normed,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;

            let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
            B::add_inplace(&mut h, &mlp_output)?;
            hidden = h;
        }

        B::rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;

        if seq_len > 1 {
            let last = self.extract_last_row(&hidden, seq_len);
            self.lm_head_forward(&last)
        } else {
            self.lm_head_forward(&hidden)
        }
    }

    /// Prefill forward pass with paged KV cache.
    ///
    /// # Errors
    /// Returns an error if any GPU operation fails.
    pub fn forward_prefill_paged(
        &self,
        input_ids: &[u32],
        paged_kv: &mut B::PagedKvCache,
        block_table: &infernum::BlockTable,
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

        B::rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;

        let last_hidden = self.extract_last_row(&hidden, seq_len);
        self.lm_head_forward(&last_hidden)
    }

    /// Batched decode forward pass with paged KV cache.
    ///
    /// # Errors
    /// Returns an error if any GPU operation fails.
    ///
    /// # Panics
    /// Panics if `token_ids` is empty (batch size 0).
    pub fn forward_batch_decode_paged(
        &self,
        token_ids: &[u32],
        paged_kv: &mut B::PagedKvCache,
        block_tables: &[infernum::BlockTable],
        positions: &[usize],
    ) -> Result<B::Tensor> {
        let batch_size = token_ids.len();
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

            B::rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;
            let logits = self.lm_head_forward(&hidden.reshape(&[1, hidden_size]))?;
            logits_parts.push(logits);
        }

        if batch_size == 1 {
            return Ok(logits_parts.into_iter().next().unwrap());
        }

        // Concatenate per-sequence logits into (batch_size, vocab_size)
        B::concat_rows(&logits_parts)
    }
}

// --- Model trait implementation ---

impl<B: DeepSeekOps + Send + 'static> infernum::Model for DeepSeekModel<B>
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
        let mc = self.model_config();
        B::allocate_paged_kv_cache(
            &self.device,
            mc.num_layers,
            block_config,
            mc.num_kv_heads,
            mc.head_dim,
            mc.cache_dtype,
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
        block_table: &infernum::BlockTable,
        start_pos: usize,
    ) -> Result<B::Logits> {
        let tensor = self.forward_prefill_paged(input_ids, kv_cache, block_table, start_pos)?;
        Ok(B::logits_from_tensor(tensor))
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn forward_batch_decode(
        &self,
        token_ids: &B::Tensor,
        kv_cache: &mut Self::KvCache,
        _runtime_state: &mut B::RuntimeState,
        block_tables: &B::Tensor,
        _seq_lens: &B::Tensor,
        positions: &B::Tensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
        _max_seq_len: usize,
    ) -> Result<B::Logits> {
        // Bridge: download tensors to host and call per-sequence method.
        let token_ids_host = B::to_f32_vec(token_ids)?;
        let token_ids_u32: Vec<u32> = token_ids_host.iter().map(|&v| v as u32).collect();
        let positions_f32 = B::to_f32_vec(positions)?;
        let positions_host: Vec<usize> = positions_f32.iter().map(|&p| p as usize).collect();
        let bt_f32 = B::to_f32_vec(block_tables)?;
        let bt_flat: Vec<u32> = bt_f32.iter().map(|&b| b as u32).collect();
        let block_size = B::block_size(kv_cache);
        let block_tables_host: Vec<infernum::BlockTable> = (0..batch_size)
            .map(|i| {
                let start = i * max_blocks_per_seq;
                let end = start + max_blocks_per_seq;
                let blocks: Vec<usize> = bt_flat[start..end].iter().map(|&b| b as usize).collect();
                infernum::BlockTable::from_raw(blocks, positions_host[i], block_size)
            })
            .collect();
        let tensor = self.forward_batch_decode_paged(
            &token_ids_u32,
            kv_cache,
            &block_tables_host,
            &positions_host,
        )?;
        Ok(B::logits_from_tensor(tensor))
    }
}

// --- Shared module imports ---

use infernum::transformer;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn split_kv_b_proj_shapes() {
        use infernum_cuda::cuda::{CudaContext, CudaTensor};
        use infernum_cuda::CudaBackend;

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
            split_kv_b_proj_dense::<CudaBackend>(&ctx, &weight, num_heads, qk_nope_dim, v_head_dim)
                .unwrap();

        assert_eq!(k.shape(), &[kv_lora_rank, num_heads * qk_nope_dim]);
        assert_eq!(v.shape(), &[num_heads, kv_lora_rank, v_head_dim]);
        assert_eq!(k_t.shape(), &[num_heads, qk_nope_dim, kv_lora_rank]);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn split_kv_b_proj_roundtrip() {
        use infernum_cuda::cuda::{CudaContext, CudaTensor};
        use infernum_cuda::CudaBackend;

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
            split_kv_b_proj_dense::<CudaBackend>(&ctx, &weight, num_heads, qk_nope_dim, v_head_dim)
                .unwrap();

        let k_data = k.to_vec::<f32>().unwrap();
        let v_data = v.to_vec::<f32>().unwrap();

        // Reconstruct the original by interleaving K and V columns back
        let mut reconstructed = vec![0.0_f32; kv_lora_rank * total_cols];
        for row in 0..kv_lora_rank {
            for h in 0..num_heads {
                for d in 0..qk_nope_dim {
                    reconstructed[row * total_cols + h * stride + d] =
                        k_data[row * (num_heads * qk_nope_dim) + h * qk_nope_dim + d];
                }
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
    #[cfg(feature = "cuda")]
    fn split_kv_b_proj_k_transpose() {
        use infernum_cuda::cuda::{CudaContext, CudaTensor};
        use infernum_cuda::CudaBackend;

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
            split_kv_b_proj_dense::<CudaBackend>(&ctx, &weight, num_heads, qk_nope_dim, v_head_dim)
                .unwrap();

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
