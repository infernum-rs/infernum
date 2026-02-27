//! Llama model implementation — fully generic over the compute backend.
//!
//! All forward pass methods, weight types, and the model struct are generic
//! over `B: Backend`. CUDA-specific code (loading, indirect graph kernels,
//! `Model` impl) lives in `cuda_model.rs` behind `#[cfg(feature = "cuda")]`.

#![allow(
    clippy::struct_field_names, // _proj suffix is conventional for Llama weights
    clippy::no_effect_underscore_binding,
    clippy::doc_markdown, // tensor shape docs trigger false positives
    dead_code, // generic types are only instantiated when a backend feature is enabled
    unused_mut // variables are conditionally mutated via all_reduce
)]

use std::marker::PhantomData;
use std::path::Path;

use infernum::backend::{
    ArithOps, AttentionOps, Backend, CastOps, Comm, EmbedOps, MatmulExtOps, MatmulOps, MoeOps,
    NormOps, PagedAttentionOps, PagedKvCacheOps, RopeOps, SwigluOps, TensorDataOps, TensorFactory,
    TensorOps,
};
use infernum::block_allocator::BlockTable;
use infernum::dtype::DType;
use infernum::shard::GpuConfig;
use infernum::tensor::Tensor;
use infernum::Result;

use crate::LlamaConfig;

// ---- Weight types ----

/// K+V projection storage: fused for dense weights, separate for quantized.
pub(crate) enum KvProjWeight<B: Backend + MatmulOps> {
    /// K and V weights concatenated into a single (hidden, 2*kv_dim) dense
    /// tensor. After matmul the output columns split as `[k(kv_dim), v(kv_dim)]`.
    Fused { weight: B::Tensor, kv_dim: usize },
    /// Separate K and V projections (used for quantized weights).
    Separate {
        k_proj: Box<<B as MatmulOps>::LinearWeight>,
        v_proj: Box<<B as MatmulOps>::LinearWeight>,
    },
}

/// Weights for a single Llama attention layer
pub(crate) struct LlamaAttentionWeights<B: Backend + MatmulOps> {
    pub q_proj: <B as MatmulOps>::LinearWeight,
    pub kv_proj: KvProjWeight<B>,
    pub o_proj: <B as MatmulOps>::LinearWeight,
}

/// Gate+Up projection storage: fused for dense weights, separate for quantized.
pub(crate) enum GateUpWeight<B: Backend + MatmulOps> {
    /// Gate and up weights concatenated into a single (hidden, 2*intermediate)
    /// dense tensor. After matmul the output columns split as
    /// `[gate(intermediate), up(intermediate)]`.
    Fused {
        weight: B::Tensor,
        intermediate_size: usize,
    },
    /// Separate gate and up projections (used for quantized weights).
    Separate {
        gate_proj: Box<<B as MatmulOps>::LinearWeight>,
        up_proj: Box<<B as MatmulOps>::LinearWeight>,
    },
}

/// Weights for a single Llama MLP layer
pub(crate) struct LlamaMlpWeights<B: Backend + MatmulOps> {
    pub gate_up: GateUpWeight<B>,
    pub down_proj: <B as MatmulOps>::LinearWeight,
}

/// Weights for a single `MoE` expert (same structure as a dense MLP).
pub(crate) struct MoeExpertWeights<B: Backend + MatmulOps> {
    pub mlp: LlamaMlpWeights<B>,
}

/// Feed-forward network weights: either a single dense MLP or a Mixture-of-Experts layer.
pub(crate) enum FfnWeights<B: Backend + MatmulOps> {
    /// Standard dense MLP (Llama, etc.)
    Dense(Box<LlamaMlpWeights<B>>),
    /// Mixture-of-Experts (Mixtral, etc.)
    Moe {
        /// Router gate weight, pre-transposed: shape `[hidden_size, num_experts]`
        gate: B::Tensor,
        /// Per-expert MLP weights
        experts: Vec<MoeExpertWeights<B>>,
        /// How many experts to activate per token
        num_experts_per_tok: usize,
    },
}

/// Weights for a single Llama decoder layer
pub(crate) struct LlamaLayerWeights<B: Backend + MatmulOps> {
    pub input_layernorm: B::Tensor,
    pub attention: LlamaAttentionWeights<B>,
    pub post_attention_layernorm: B::Tensor,
    pub ffn: FfnWeights<B>,
}

/// All-reduce callback for tensor-parallel models.
///
/// Wraps the backend-specific communicator in a type-erased closure so
/// Complete Llama model, generic over the compute backend `B`.
///
/// The backend determines the tensor type and linear weight representation.
/// CUDA-specific methods (loading, CUDA graph decode, `Model` impl) live
/// in the `cuda_model` module.
pub struct LlamaModel<B: Backend + MatmulOps> {
    pub(crate) config: LlamaConfig,
    pub(crate) dtype: DType,
    pub(crate) device: B::DeviceHandle,
    #[allow(dead_code)]
    pub(crate) gpu_config: GpuConfig,

    /// Optional communicator for tensor-parallel all-reduce.
    /// `None` for single-GPU, `Some(comm)` for sharded models.
    pub(crate) comm: Option<B::Comm>,

    // Per-GPU head counts (== full counts for single-GPU, divided for TP)
    pub(crate) tp_num_heads: usize,
    pub(crate) tp_num_kv_heads: usize,

    // Embeddings
    pub(crate) embed_tokens: B::Tensor,

    // Transformer layers
    pub(crate) layers: Vec<LlamaLayerWeights<B>>,

    // Final layer norm
    pub(crate) norm: B::Tensor,

    // Output projection (may be tied to embed_tokens)
    pub(crate) lm_head: <B as MatmulOps>::LinearWeight,

    // RoPE caches (stored in model dtype)
    pub(crate) cos_cache: B::Tensor,
    pub(crate) sin_cache: B::Tensor,

    pub(crate) _backend: PhantomData<B>,
}

// ---- Trait alias for the full set of ops used by the generic forward pass ----

/// Convenience alias: all op traits required by `LlamaModel` forward methods.
pub trait LlamaOps:
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
{
}

impl<B> LlamaOps for B where
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
{
}

// ---- Generic forward pass methods ----

impl<B: LlamaOps> LlamaModel<B> {
    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &LlamaConfig {
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
    /// The loader handles format-specific details (SafeTensors, GGUF, etc.)
    /// and backend-specific details (host→device transfer, quantization).
    /// This method handles model-specific logic: weight names, fusing K+V
    /// and gate+up projections, MoE layout, tied embeddings, RoPE caches.
    ///
    /// # Errors
    /// Returns an error if any weight fails to load.
    ///
    /// # Panics
    /// Panics if `as_dense_weight` returns `None` after `is_dense_weight`
    /// returned `true` (indicates a backend bug).
    #[allow(clippy::too_many_lines)]
    pub fn load_weights(
        device: B::DeviceHandle,
        config: LlamaConfig,
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

            let layer = LlamaLayerWeights {
                input_layernorm: loader
                    .load_tensor(&format!("{prefix}.input_layernorm.weight"), dtype)?,
                attention: {
                    let k = loader.load_linear(
                        &format!("{prefix}.self_attn.k_proj.weight"),
                        dtype,
                        qc,
                    )?;
                    let v = loader.load_linear(
                        &format!("{prefix}.self_attn.v_proj.weight"),
                        dtype,
                        qc,
                    )?;
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
                    LlamaAttentionWeights {
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
                    }
                },
                post_attention_layernorm: loader
                    .load_tensor(&format!("{prefix}.post_attention_layernorm.weight"), dtype)?,
                ffn: if config.is_moe() {
                    Self::load_moe_weights(dtype, loader, &prefix, &config, qc)?
                } else {
                    FfnWeights::<B>::Dense(Box::new(Self::load_dense_mlp(
                        dtype, loader, &prefix, &config, qc,
                    )?))
                },
            };

            layers.push(layer);
        }

        let norm = loader.load_tensor("model.norm.weight", dtype)?;

        let lm_head = if config.tie_word_embeddings {
            if qc.is_some() {
                // Quantized tied embeddings: cast to f32, quantize to Q8
                let embed_f32 = B::cast_to_f32(&embed_tokens)?;
                let data = B::to_f32_vec(&embed_f32)?;
                B::quantize_to_q8(&device, embed_f32.shape(), &data)?
            } else {
                // Dense tied embeddings: transpose to matmul-ready layout
                let embed_f32 = B::cast_to_f32(&embed_tokens)?;
                let transposed = B::transpose_2d(&embed_f32)?;
                B::dense_weight(B::cast_from_f32(&transposed, dtype)?)
            }
        } else {
            let lw = loader.load_linear("lm_head.weight", dtype, None)?;
            if qc.is_some() {
                // Quantized model with separate lm_head: re-quantize as Q8
                if let Some(w) = B::as_dense_weight(&lw) {
                    let f32_w = B::cast_to_f32(w)?;
                    // f32_w is in (in_features, out_features) col-major.
                    // Transpose back to (out_features, in_features) row-major
                    // for Q8 quantization.
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

        // Precompute RoPE caches
        let half_dim = config.head_dim() / 2;
        let max_pos = config.max_position_embeddings;
        let (cos_data, sin_data) =
            infernum::rope::precompute_rope_data(max_pos, config.head_dim(), config.rope_theta);
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

    /// Load a dense MLP (gate_proj, up_proj, down_proj) for a single layer.
    fn load_dense_mlp(
        dtype: DType,
        loader: &impl infernum::WeightLoader<B>,
        layer_prefix: &str,
        config: &LlamaConfig,
        qc: Option<&infernum::QuantizationConfig>,
    ) -> Result<LlamaMlpWeights<B>> {
        let gate =
            loader.load_linear(&format!("{layer_prefix}.mlp.gate_proj.weight"), dtype, qc)?;
        let up = loader.load_linear(&format!("{layer_prefix}.mlp.up_proj.weight"), dtype, qc)?;
        let gate_up = if B::is_dense_weight(&gate) && B::is_dense_weight(&up) {
            let g = B::as_dense_weight(&gate).expect("checked dense");
            let u = B::as_dense_weight(&up).expect("checked dense");
            GateUpWeight::<B>::Fused {
                weight: B::concat_inner_dim(g, u)?,
                intermediate_size: config.intermediate_size,
            }
        } else {
            GateUpWeight::<B>::Separate {
                gate_proj: Box::new(gate),
                up_proj: Box::new(up),
            }
        };
        Ok(LlamaMlpWeights {
            gate_up,
            down_proj: loader.load_linear(
                &format!("{layer_prefix}.mlp.down_proj.weight"),
                dtype,
                qc,
            )?,
        })
    }

    /// Load MoE weights (router gate + per-expert MLPs) for a single layer.
    fn load_moe_weights(
        dtype: DType,
        loader: &impl infernum::WeightLoader<B>,
        layer_prefix: &str,
        config: &LlamaConfig,
        qc: Option<&infernum::QuantizationConfig>,
    ) -> Result<FfnWeights<B>> {
        let num_experts = config
            .num_local_experts
            .expect("MoE requires num_local_experts");
        let num_experts_per_tok = config
            .num_experts_per_tok
            .expect("MoE requires num_experts_per_tok");

        // Router gate: load as f32, transpose, cast to model dtype
        let gate_name = format!("{layer_prefix}.block_sparse_moe.gate.weight");
        let gate_f32 = loader.load_tensor(&gate_name, DType::F32)?;
        let gate_transposed = B::transpose_2d(&gate_f32)?;
        let gate = B::cast_from_f32(&gate_transposed, dtype)?;

        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let ep = format!("{layer_prefix}.block_sparse_moe.experts.{e}");
            let gate_proj = loader.load_linear(&format!("{ep}.w1.weight"), dtype, qc)?;
            let up_proj = loader.load_linear(&format!("{ep}.w3.weight"), dtype, qc)?;
            let gate_up = if B::is_dense_weight(&gate_proj) && B::is_dense_weight(&up_proj) {
                let g = B::as_dense_weight(&gate_proj).expect("checked dense");
                let u = B::as_dense_weight(&up_proj).expect("checked dense");
                GateUpWeight::<B>::Fused {
                    weight: B::concat_inner_dim(g, u)?,
                    intermediate_size: config.intermediate_size,
                }
            } else {
                GateUpWeight::<B>::Separate {
                    gate_proj: Box::new(gate_proj),
                    up_proj: Box::new(up_proj),
                }
            };
            let down_proj = loader.load_linear(&format!("{ep}.w2.weight"), dtype, qc)?;
            experts.push(MoeExpertWeights {
                mlp: LlamaMlpWeights { gate_up, down_proj },
            });
        }

        Ok(FfnWeights::<B>::Moe {
            gate,
            experts,
            num_experts_per_tok,
        })
    }

    // ---- GGUF weight loading (generic over backend) ----

    /// Load model weights from a GGUF file, generic over any backend.
    ///
    /// Uses GGUF tensor names (`token_embd.weight`, `blk.N.*`, etc.)
    /// and the core [`GgufLoader`](infernum::weights::gguf::GgufLoader)
    /// which returns host-side buffers. Uploads to the backend via
    /// `B::upload_host_linear` and `B::from_f32_slice`.
    ///
    /// # Errors
    /// Returns an error if any weight fails to load or upload.
    ///
    /// # Panics
    /// Panics if the model config is MoE (MoE GGUF loading is not yet supported).
    #[allow(clippy::too_many_lines)]
    pub fn load_weights_gguf(
        device: B::DeviceHandle,
        config: LlamaConfig,
        loader: &infernum::weights::gguf::GgufLoader,
    ) -> Result<Self> {
        use infernum::weights::format::FormatLoader;
        use infernum::weights::format::{host_transpose_2d, host_unpermute_f32};
        use infernum::weights::host::{host_concat_inner_dim, HostLinearWeight};

        /// Load a GGUF tensor as a linear weight (dense or quantized).
        /// Dense weights are transposed to matmul-ready layout on the host.
        fn host_load_linear(
            loader: &infernum::weights::gguf::GgufLoader,
            name: &str,
        ) -> Result<HostLinearWeight> {
            let dtype = FormatLoader::get_dtype(loader, name)?;
            if dtype.is_quantized() {
                Ok(HostLinearWeight::Quantized(FormatLoader::load_quantized(
                    loader, name,
                )?))
            } else {
                let tensor = FormatLoader::load_f32(loader, name)?;
                Ok(HostLinearWeight::Dense(host_transpose_2d(&tensor)?))
            }
        }

        /// Load a GGUF Q/K weight with head-dimension unpermuting.
        fn host_load_linear_unpermute(
            loader: &infernum::weights::gguf::GgufLoader,
            name: &str,
            n_head: usize,
        ) -> Result<HostLinearWeight> {
            let dtype = FormatLoader::get_dtype(loader, name)?;
            if dtype.is_quantized() {
                Ok(HostLinearWeight::Quantized(
                    FormatLoader::load_quantized_unpermute(loader, name, n_head)?,
                ))
            } else {
                let tensor = FormatLoader::load_f32(loader, name)?;
                let unpermuted = host_unpermute_f32(&tensor, n_head)?;
                Ok(HostLinearWeight::Dense(host_transpose_2d(&unpermuted)?))
            }
        }

        let embed_host = FormatLoader::load_f32(loader, "token_embd.weight")?;
        let embed_tokens =
            B::from_f32_slice(&device, &embed_host.shape, embed_host.as_f32_slice())?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("blk.{i}");

            let attn_norm_host =
                FormatLoader::load_f32(loader, &format!("{prefix}.attn_norm.weight"))?;
            let input_layernorm = B::from_f32_slice(
                &device,
                &attn_norm_host.shape,
                attn_norm_host.as_f32_slice(),
            )?;

            let attention = {
                let k_host = host_load_linear_unpermute(
                    loader,
                    &format!("{prefix}.attn_k.weight"),
                    config.num_kv_heads(),
                )?;
                let v_host = host_load_linear(loader, &format!("{prefix}.attn_v.weight"))?;

                // Try to fuse K+V if both are dense
                let kv_proj = if let (HostLinearWeight::Dense(k_t), HostLinearWeight::Dense(v_t)) =
                    (&k_host, &v_host)
                {
                    let fused = host_concat_inner_dim(k_t, v_t);
                    let fused_tensor =
                        B::from_raw_bytes(&device, &fused.shape, fused.dtype, &fused.data)?;
                    KvProjWeight::<B>::Fused {
                        kv_dim: config.num_kv_heads() * config.head_dim(),
                        weight: fused_tensor,
                    }
                } else {
                    let k_dev = B::upload_host_linear(&device, &k_host)?;
                    let v_dev = B::upload_host_linear(&device, &v_host)?;
                    KvProjWeight::<B>::Separate {
                        k_proj: Box::new(k_dev),
                        v_proj: Box::new(v_dev),
                    }
                };

                let q_host = host_load_linear_unpermute(
                    loader,
                    &format!("{prefix}.attn_q.weight"),
                    config.num_attention_heads,
                )?;
                let o_host = host_load_linear(loader, &format!("{prefix}.attn_output.weight"))?;

                LlamaAttentionWeights {
                    q_proj: B::upload_host_linear(&device, &q_host)?,
                    kv_proj,
                    o_proj: B::upload_host_linear(&device, &o_host)?,
                }
            };

            let ffn_norm_host =
                FormatLoader::load_f32(loader, &format!("{prefix}.ffn_norm.weight"))?;
            let post_attention_layernorm =
                B::from_f32_slice(&device, &ffn_norm_host.shape, ffn_norm_host.as_f32_slice())?;

            let ffn = {
                assert!(!config.is_moe(), "MoE GGUF loading is not yet supported");
                let gate_host = host_load_linear(loader, &format!("{prefix}.ffn_gate.weight"))?;
                let up_host = host_load_linear(loader, &format!("{prefix}.ffn_up.weight"))?;

                let gate_up = if let (HostLinearWeight::Dense(g), HostLinearWeight::Dense(u)) =
                    (&gate_host, &up_host)
                {
                    let fused = host_concat_inner_dim(g, u);
                    let fused_tensor =
                        B::from_raw_bytes(&device, &fused.shape, fused.dtype, &fused.data)?;
                    GateUpWeight::<B>::Fused {
                        weight: fused_tensor,
                        intermediate_size: config.intermediate_size,
                    }
                } else {
                    let g_dev = B::upload_host_linear(&device, &gate_host)?;
                    let u_dev = B::upload_host_linear(&device, &up_host)?;
                    GateUpWeight::<B>::Separate {
                        gate_proj: Box::new(g_dev),
                        up_proj: Box::new(u_dev),
                    }
                };

                FfnWeights::<B>::Dense(Box::new(LlamaMlpWeights {
                    gate_up,
                    down_proj: B::upload_host_linear(
                        &device,
                        &host_load_linear(loader, &format!("{prefix}.ffn_down.weight"))?,
                    )?,
                }))
            };

            layers.push(LlamaLayerWeights {
                input_layernorm,
                attention,
                post_attention_layernorm,
                ffn,
            });
        }

        let norm_host = FormatLoader::load_f32(loader, "output_norm.weight")?;
        let norm = B::from_f32_slice(&device, &norm_host.shape, norm_host.as_f32_slice())?;

        // lm_head: check tied embeddings, quantized fallback
        let lm_head = if config.tie_word_embeddings {
            let embd_dtype = FormatLoader::get_dtype(loader, "token_embd.weight")?;
            if embd_dtype.is_quantized() {
                B::upload_host_linear(
                    &device,
                    &HostLinearWeight::Quantized(FormatLoader::load_quantized(
                        loader,
                        "token_embd.weight",
                    )?),
                )?
            } else {
                let transposed = host_transpose_2d(&embed_host)?;
                B::upload_host_linear(&device, &HostLinearWeight::Dense(transposed))?
            }
        } else if loader.contains("output.weight") {
            B::upload_host_linear(&device, &host_load_linear(loader, "output.weight")?)?
        } else {
            let embd_dtype = FormatLoader::get_dtype(loader, "token_embd.weight")?;
            if embd_dtype.is_quantized() {
                B::upload_host_linear(
                    &device,
                    &HostLinearWeight::Quantized(FormatLoader::load_quantized(
                        loader,
                        "token_embd.weight",
                    )?),
                )?
            } else {
                let transposed = host_transpose_2d(&embed_host)?;
                B::upload_host_linear(&device, &HostLinearWeight::Dense(transposed))?
            }
        };

        // Precompute RoPE caches
        let half_dim = config.head_dim() / 2;
        let max_pos = config.max_position_embeddings;
        let (cos_data, sin_data) =
            infernum::rope::precompute_rope_data(max_pos, config.head_dim(), config.rope_theta);
        let cos_f32 = B::from_f32_slice(&device, &[max_pos, half_dim], &cos_data)?;
        let sin_f32 = B::from_f32_slice(&device, &[max_pos, half_dim], &sin_data)?;
        let cos_cache = B::cast_from_f32(&cos_f32, DType::F32)?;
        let sin_cache = B::cast_from_f32(&sin_f32, DType::F32)?;

        Ok(Self {
            tp_num_heads: config.num_attention_heads,
            tp_num_kv_heads: config.num_kv_heads(),
            dtype: DType::F32,
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

    /// Load a Llama model from a GGUF file.
    ///
    /// Parses the GGUF file (CPU-only), then uploads weights to the device
    /// via the generic [`load_weights_gguf`](Self::load_weights_gguf) method.
    ///
    /// # Errors
    /// Returns an error if the file cannot be parsed or weights fail to load.
    pub fn from_gguf(device: &B::DeviceHandle, gguf_path: impl AsRef<Path>) -> Result<Self> {
        let loader = infernum::weights::gguf::GgufLoader::from_file(gguf_path)?;
        let config = LlamaConfig::from_gguf_metadata(loader.metadata())?;
        Self::load_weights_gguf(device.clone(), config, &loader)
    }

    /// Load a Llama model from a SafeTensors directory.
    ///
    /// Reads `config.json` for model configuration and loads weights from
    /// `*.safetensors` files. The backend provides the weight loader via
    /// [`SafeTensorsLoaderOps`](infernum::SafeTensorsLoaderOps).
    ///
    /// # Errors
    /// Returns an error if the config is missing or weights fail to load.
    pub fn from_pretrained(device: &B::DeviceHandle, model_path: impl AsRef<Path>) -> Result<Self>
    where
        B: infernum::SafeTensorsLoaderOps,
    {
        let model_path = model_path.as_ref();
        let config = LlamaConfig::from_file(model_path.join("config.json"))?;
        let loader = B::safetensors_loader(device, model_path)?;
        Self::load_weights(device.clone(), config, &loader)
    }

    // ---- Sharded weight loading (tensor parallelism) ----

    /// Load model weights with tensor-parallel sharding, generic over backend.
    ///
    /// Splits attention and MLP projections across GPUs according to shard
    /// strategy. The communicator is used for all-reduce after each parallel
    /// region in the forward pass to synchronise partial results.
    ///
    /// If `gpu_config` is `Single`, falls back to the non-sharded
    /// [`load_weights`](Self::load_weights) and attaches the communicator.
    ///
    /// # Errors
    /// Returns an error if weight loading fails.
    ///
    /// # Panics
    /// Panics if head counts or intermediate size are not divisible by
    /// `world_size`.
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    pub fn load_weights_sharded(
        device: B::DeviceHandle,
        config: LlamaConfig,
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
        assert!(
            config.intermediate_size.is_multiple_of(world_size),
            "intermediate_size ({}) must be divisible by world_size ({world_size})",
            config.intermediate_size
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

            let layer = LlamaLayerWeights {
                input_layernorm: loader
                    .load_tensor(&format!("{prefix}.input_layernorm.weight"), dtype)?,
                attention: {
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

                    LlamaAttentionWeights {
                        q_proj,
                        kv_proj,
                        o_proj: loader.load_linear_sharded(
                            &o_name,
                            dtype,
                            qc,
                            &shard,
                            shard_strategy_for_weight(&o_name),
                        )?,
                    }
                },
                post_attention_layernorm: loader
                    .load_tensor(&format!("{prefix}.post_attention_layernorm.weight"), dtype)?,
                ffn: if config.is_moe() {
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

                    FfnWeights::<B>::Dense(Box::new(LlamaMlpWeights {
                        gate_up,
                        down_proj: loader.load_linear_sharded(
                            &down_name,
                            dtype,
                            qc,
                            &shard,
                            shard_strategy_for_weight(&down_name),
                        )?,
                    }))
                },
            };

            layers.push(layer);
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
        let (cos_data, sin_data) =
            infernum::rope::precompute_rope_data(max_pos, config.head_dim(), config.rope_theta);
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
        config: &LlamaConfig,
        shard: &infernum::ShardConfig,
        qc: Option<&infernum::QuantizationConfig>,
    ) -> Result<FfnWeights<B>> {
        use infernum::shard::ShardStrategy;

        let num_experts = config
            .num_local_experts
            .expect("MoE requires num_local_experts");
        let num_experts_per_tok = config
            .num_experts_per_tok
            .expect("MoE requires num_experts_per_tok");

        let gate_name = format!("{layer_prefix}.block_sparse_moe.gate.weight");
        let gate_f32 = loader.load_tensor(&gate_name, DType::F32)?;
        let gate_transposed = B::transpose_2d(&gate_f32)?;
        let gate = B::cast_from_f32(&gate_transposed, dtype)?;

        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let ep = format!("{layer_prefix}.block_sparse_moe.experts.{e}");
            let gate_proj = loader.load_linear_sharded(
                &format!("{ep}.w1.weight"),
                dtype,
                qc,
                shard,
                ShardStrategy::Column,
            )?;
            let up_proj = loader.load_linear_sharded(
                &format!("{ep}.w3.weight"),
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
                &format!("{ep}.w2.weight"),
                dtype,
                qc,
                shard,
                ShardStrategy::Row,
            )?;

            experts.push(MoeExpertWeights {
                mlp: LlamaMlpWeights { gate_up, down_proj },
            });
        }

        Ok(FfnWeights::<B>::Moe {
            gate,
            experts,
            num_experts_per_tok,
        })
    }

    // ---- Helpers ----

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
    pub(crate) fn maybe_all_reduce(&self, tensor: &mut B::Tensor) -> Result<()> {
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
        let mut hidden = self.embed(input_ids)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let normed = B::rms_norm(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            let seq_len = hidden.shape()[0];
            let num_heads = self.tp_num_heads;
            let num_kv_heads = self.tp_num_kv_heads;
            let head_dim = self.config.head_dim();

            let q = B::linear(&normed, &layer.attention.q_proj)?;
            let (k, v) = match &layer.attention.kv_proj {
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

            let q = q.reshape(&[seq_len, num_heads, head_dim]);
            let k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
            let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

            let q = B::apply_rope(&q, &self.cos_cache, &self.sin_cache, 0)?;
            let k = B::apply_rope(&k, &self.cos_cache, &self.sin_cache, 0)?;

            let sliding_window = self.config.effective_sliding_window(layer_idx);
            let attn_output =
                B::fused_attention_prefill(&q, &k, &v, 0, None, None, sliding_window)?;
            let attn_output = attn_output.reshape(&[seq_len, num_heads * head_dim]);

            let mut out = B::linear(&attn_output, &layer.attention.o_proj)?;
            self.maybe_all_reduce(&mut out)?;

            let (mut hidden2, normed) = B::add_rmsnorm(
                &hidden,
                &out,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;

            let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
            B::add_inplace(&mut hidden2, &mlp_output)?;
            hidden = hidden2;
        }

        B::rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;
        self.lm_head_forward(&hidden)
    }

    // ---- Batched decode (host-side convenience) ----

    /// Batched decode with host-side inputs.
    ///
    /// Converts host arrays to device tensors and calls
    /// [`forward_batch_decode_tensors`](Self::forward_batch_decode_tensors).
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
        paged_kvs: &mut [B::PagedKvCache],
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
            paged_kvs,
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
    /// Processes one token per sequence for `batch_size` sequences. All
    /// inputs are device-side tensors — the engine uploads them before
    /// calling this method.
    ///
    /// Returns logits of shape `(batch_size, vocab_size)`.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batch_decode_tensors(
        &self,
        token_ids: &B::Tensor,
        paged_kvs: &mut [B::PagedKvCache],
        block_tables: &B::Tensor,
        seq_lens: &B::Tensor,
        positions: &B::Tensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
    ) -> Result<B::Tensor> {
        let paged_kv = &mut paged_kvs[0];

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
    /// Processes all prompt tokens, writing K/V into the paged cache via
    /// the block table. Returns logits for the **last** token only.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward_prefill_paged(
        &self,
        input_ids: &[u32],
        paged_kvs: &mut [B::PagedKvCache],
        block_table: &BlockTable,
        start_pos: usize,
    ) -> Result<B::Tensor> {
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

        let last = self.extract_last_row(&hidden, seq_len);
        let normed = B::rms_norm(&last, &self.norm, self.config.rms_norm_eps)?;
        self.lm_head_forward(&normed.reshape(&[1, self.config.hidden_size]))
    }

    // ---- Layer-level forward methods ----

    /// Transformer layer forward pass for batched decode with paged KV cache.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_paged_decode_batched(
        &self,
        hidden: &B::Tensor,
        layer: &LlamaLayerWeights<B>,
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

    /// Batched attention decode with paged KV cache — single kernel launch.
    #[allow(clippy::too_many_arguments)]
    fn forward_attention_paged_decode_batched(
        &self,
        hidden: &B::Tensor,
        weights: &LlamaAttentionWeights<B>,
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

        let q = B::linear(hidden, &weights.q_proj)?;
        let (k, v) = match &weights.kv_proj {
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

        let q = q.reshape(&[batch_size, num_heads, head_dim]);
        let k = k.reshape(&[batch_size, num_kv_heads, head_dim]);
        let v = v.reshape(&[batch_size, num_kv_heads, head_dim]);

        let sliding_window = self.config.effective_sliding_window(layer_idx);

        let q = B::apply_rope_batched(&q, &self.cos_cache, &self.sin_cache, positions, batch_size)?;
        let k = B::apply_rope_batched(&k, &self.cos_cache, &self.sin_cache, positions, batch_size)?;

        // Batched KV cache append — single kernel launch for all sequences
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

        // Batched paged attention decode — single kernel launch
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

    /// Transformer layer forward pass for single-sequence prefill with paged KV cache.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_paged_prefill(
        &self,
        hidden: &B::Tensor,
        layer: &LlamaLayerWeights<B>,
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
        weights: &LlamaAttentionWeights<B>,
        layer_idx: usize,
        paged_kv: &mut B::PagedKvCache,
        block_table: &BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<B::Tensor> {
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim();

        let q = B::linear(hidden, &weights.q_proj)?;
        let (k, v) = match &weights.kv_proj {
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

        let q = q.reshape(&[seq_len, num_heads, head_dim]);
        let k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
        let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

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

    /// Forward pass through MLP (SwiGLU)
    #[allow(clippy::unused_self)]
    fn forward_mlp(&self, hidden: &B::Tensor, weights: &LlamaMlpWeights<B>) -> Result<B::Tensor> {
        let (gate, up) = self.compute_gate_up(hidden, &weights.gate_up)?;
        let intermediate = B::swiglu(&gate, &up)?;
        let mut out = B::linear(&intermediate, &weights.down_proj)?;
        self.maybe_all_reduce(&mut out)?;
        Ok(out)
    }

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

    /// Forward pass through MLP without all-reduce (used by MoE experts).
    fn forward_mlp_no_reduce(
        &self,
        hidden: &B::Tensor,
        weights: &LlamaMlpWeights<B>,
    ) -> Result<B::Tensor> {
        let (gate, up) = self.compute_gate_up(hidden, &weights.gate_up)?;
        let intermediate = B::swiglu(&gate, &up)?;
        B::linear(&intermediate, &weights.down_proj)
    }

    /// Dispatch to dense MLP or MoE forward pass.
    pub(crate) fn forward_ffn(&self, hidden: &B::Tensor, ffn: &FfnWeights<B>) -> Result<B::Tensor> {
        match ffn {
            FfnWeights::<B>::Dense(mlp) => self.forward_mlp(hidden, mlp),
            FfnWeights::<B>::Moe {
                gate,
                experts,
                num_experts_per_tok,
            } => self.forward_moe(hidden, gate, experts, *num_experts_per_tok),
        }
    }

    /// Forward pass through a Mixture-of-Experts layer.
    fn forward_moe(
        &self,
        hidden: &B::Tensor,
        gate: &B::Tensor,
        experts: &[MoeExpertWeights<B>],
        num_experts_per_tok: usize,
    ) -> Result<B::Tensor> {
        let mut out = B::moe_forward_softmax(
            hidden,
            gate,
            experts.len(),
            num_experts_per_tok,
            true,
            |expert_idx, expert_input| {
                self.forward_mlp_no_reduce(expert_input, &experts[expert_idx].mlp)
            },
        )?;
        self.maybe_all_reduce(&mut out)?;
        Ok(out)
    }

    /// Project hidden states to vocabulary logits (always f32)
    pub(crate) fn lm_head_forward(&self, hidden: &B::Tensor) -> Result<B::Tensor> {
        if self.dtype == DType::BF16 {
            if let Some(w) = self.lm_head_dense_weight() {
                return B::matmul_bf16_f32(hidden, w);
            }
        }
        let logits_t = B::linear(hidden, &self.lm_head)?;
        if self.dtype == DType::F32 {
            return Ok(logits_t);
        }
        B::cast_to_f32(&logits_t)
    }

    /// Try to extract the dense weight from lm_head for mixed-precision matmul.
    ///
    /// Returns `None` if lm_head is quantized.
    fn lm_head_dense_weight(&self) -> Option<&B::Tensor> {
        B::as_dense_weight(&self.lm_head)
    }
}

// ---- Model trait impl (generic over any backend) ----

impl<B: LlamaOps + Send + 'static> infernum::Model for LlamaModel<B>
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
        let tensor = self.forward_prefill_paged(
            input_ids,
            std::slice::from_mut(kv_cache),
            block_table,
            start_pos,
        )?;
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
            std::slice::from_mut(kv_cache),
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
