//! Llama model implementation

#![allow(
    clippy::struct_field_names, // _proj suffix is conventional for Llama weights
    clippy::no_effect_underscore_binding,
    clippy::doc_markdown, // tensor shape docs trigger false positives
    unused_mut // variables are conditionally mutated via cfg(feature = "nccl")
)]

use std::path::Path;

use infernum::cuda::ops::{
    add_inplace, add_rmsnorm, apply_rope, apply_rope_indirect, attention, cast_to_f32,
    embedding_gather, embedding_gather_from_device, fused_attention_decode,
    fused_attention_decode_indirect, fused_attention_prefill, matmul, matmul_bf16_f32,
    precompute_rope_cache, quantized_matmul, repeat_kv, rms_norm, rms_norm_inplace, swiglu,
    transpose_2d, GemmScalar,
};
use infernum::cuda::{
    CudaBlas, CudaContext, CudaSlice, CudaTensor, DeviceRepr, Gemm, GpuConfig, QuantizedTensor,
    ShardStrategy, ValidAsZeroBits,
};
#[cfg(feature = "nccl")]
use infernum::cuda::{NcclCommunicator, NcclType, ShardConfig};
use infernum::dtype::TensorDType;
use infernum::tensor::Tensor;

/// When NCCL is enabled, `MaybeNcclType` is `NcclType` — only types that
/// support NCCL all-reduce satisfy it. When NCCL is disabled, it's a blanket
/// trait so all tensor types work without NCCL-specific bounds.
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

use infernum::weights::{GgufLoader, SafeTensorsLoader, WeightLoader};
use infernum::KvCache;
use infernum::Result;

use crate::LlamaConfig;

/// Transpose a weight matrix once, for use in pre-transposed linear projections.
/// (out_features, in_features) -> (in_features, out_features)
fn pretranspose_weight(weight: &CudaTensor<f32>) -> Result<CudaTensor<f32>> {
    transpose_2d(weight)
}

/// Concatenate two pre-transposed dense weight matrices along the column
/// (output features) dimension: `(K, N1)` + `(K, N2)` → `(K, N1+N2)`.
///
/// Both matrices must share the same first dimension (in_features).
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

/// Split a `(seq_len, 2 * intermediate)` tensor into two `(seq_len, intermediate)`
/// tensors (gate and up) by deinterleaving rows.
///
/// Each row of the input is `[gate(intermediate), up(intermediate)]`.
/// Uses a host roundtrip — only called during prefill, not the hot decode path.
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

/// Split a `(seq_len, 2 * kv_dim)` tensor into two `(seq_len, kv_dim)` tensors
/// (K and V) by deinterleaving rows.
///
/// Each row of the input is `[k(kv_dim), v(kv_dim)]`.
/// Uses a host roundtrip — only called during prefill, not the hot decode path.
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

/// Reverse the llama.cpp Q/K weight permutation for f32 tensors.
///
/// GGUF files interleave each head's rows: `[h0, h_half, h1, h_{half+1}, ...]`.
/// This restores the HuggingFace sequential-half layout: `[h0..h_{half-1}, h_half..h_{dim-1}]`.
fn unpermute_f32(weight: &CudaTensor<f32>, n_head: usize) -> Result<CudaTensor<f32>> {
    let shape = weight.shape();
    let n_rows = shape[0];
    let n_cols = shape[1];
    let head_dim = n_rows / n_head;
    let half_dim = head_dim / 2;

    let data = weight.to_vec()?;
    let mut out = vec![0.0_f32; data.len()];

    for h in 0..n_head {
        for i in 0..half_dim {
            let src0 = (h * head_dim + 2 * i) * n_cols;
            let src1 = (h * head_dim + 2 * i + 1) * n_cols;
            let dst0 = (h * head_dim + i) * n_cols;
            let dst1 = (h * head_dim + i + half_dim) * n_cols;
            out[dst0..dst0 + n_cols].copy_from_slice(&data[src0..src0 + n_cols]);
            out[dst1..dst1 + n_cols].copy_from_slice(&data[src1..src1 + n_cols]);
        }
    }

    CudaTensor::from_slice(weight.context(), &[n_rows, n_cols], &out)
}

/// Load a tensor from a `WeightLoader` in the model's native dtype `T`.
///
/// Dispatches to `load_f32`, `load_f16`, or `load_bf16` based on `T::DTYPE`.
fn load_typed<T: TensorDType + DeviceRepr>(
    loader: &impl WeightLoader,
    ctx: &CudaContext,
    name: &str,
) -> Result<CudaTensor<T>> {
    use infernum::dtype::DType;
    match T::DTYPE {
        DType::F32 => {
            let t = loader.load_f32(ctx, name)?;
            // SAFETY: T is f32, so CudaTensor<f32> and CudaTensor<T> are the same type.
            // We use a ptr cast to avoid requiring T: Pod.
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

/// Load a tensor with sharding, dispatching by `T::DTYPE`.
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

/// Reinterpret a `CudaTensor<A>` as `CudaTensor<B>` when both have the same
/// layout (same `DType`). This is a zero-cost cast (no copy, no kernel).
///
/// # Panics
/// Panics if `size_of::<A>() != size_of::<B>()`.
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
    // The caller ensures A::DTYPE == B::DTYPE, so the GPU memory layout is identical.
    // We reconstruct the tensor using the raw components.
    unsafe { tensor.reinterpret() }
}

/// A linear layer weight that is either a dense matrix (pre-transposed for
/// standard matmul) or a quantized tensor (dequantized on-the-fly in the kernel).
enum LinearWeight<T: TensorDType> {
    /// Pre-transposed dense weight: shape (in_features, out_features)
    Dense(CudaTensor<T>),
    /// Quantized weight: shape (out_features, in_features) — transposed inside kernel.
    /// Only valid when `T = f32`.
    Quantized(QuantizedTensor),
}

/// K+V projection storage: fused for dense weights, separate for quantized.
enum KvProjWeight<T: TensorDType> {
    /// K and V weights concatenated into a single (hidden, 2*kv_dim) dense
    /// tensor. After matmul the output columns split as `[k(kv_dim), v(kv_dim)]`.
    Fused {
        weight: CudaTensor<T>,
        kv_dim: usize,
    },
    /// Separate K and V projections (used for quantized weights).
    Separate {
        k_proj: Box<LinearWeight<T>>,
        v_proj: Box<LinearWeight<T>>,
    },
}

/// Weights for a single Llama attention layer
struct LlamaAttentionWeights<T: TensorDType> {
    q_proj: LinearWeight<T>,
    kv_proj: KvProjWeight<T>,
    o_proj: LinearWeight<T>,
}

/// Gate+Up projection storage: fused for dense weights, separate for quantized.
enum GateUpWeight<T: TensorDType> {
    /// Gate and up weights concatenated into a single (hidden, 2*intermediate)
    /// dense tensor. After matmul the output columns split as
    /// `[gate(intermediate), up(intermediate)]`.
    Fused {
        weight: CudaTensor<T>,
        intermediate_size: usize,
    },
    /// Separate gate and up projections (used for quantized weights).
    Separate {
        gate_proj: Box<LinearWeight<T>>,
        up_proj: Box<LinearWeight<T>>,
    },
}

/// Weights for a single Llama MLP layer
struct LlamaMlpWeights<T: TensorDType> {
    gate_up: GateUpWeight<T>,
    down_proj: LinearWeight<T>,
}

/// Weights for a single Llama decoder layer
struct LlamaLayerWeights<T: TensorDType> {
    input_layernorm: CudaTensor<T>,
    attention: LlamaAttentionWeights<T>,
    post_attention_layernorm: CudaTensor<T>,
    mlp: LlamaMlpWeights<T>,
}

/// Complete Llama model, generic over the compute dtype `T`.
///
/// `T` is the dtype for activations, weights, and KV cache. Supported: `f32`, `f16`, `bf16`.
///
/// - `LlamaModel<f32>`: standard f32 model (also supports GGUF quantized weights)
/// - `LlamaModel<half::bf16>`: bf16 model (SafeTensors only)
/// - `LlamaModel<half::f16>`: f16 model (SafeTensors only)
///
/// Logits are always returned as `CudaTensor<f32>` (cast at the lm_head output).
pub struct LlamaModel<T: TensorDType> {
    config: LlamaConfig,
    ctx: CudaContext,
    #[allow(dead_code)]
    gpu_config: GpuConfig,

    #[cfg(feature = "nccl")]
    nccl_comm: Option<NcclCommunicator>,

    // Per-GPU head counts (== full counts for single-GPU, divided for TP)
    tp_num_heads: usize,
    tp_num_kv_heads: usize,

    // Embeddings
    embed_tokens: CudaTensor<T>,

    // Transformer layers
    layers: Vec<LlamaLayerWeights<T>>,

    // Final layer norm
    norm: CudaTensor<T>,

    // Output projection (may be tied to embed_tokens)
    lm_head: LinearWeight<T>,

    // RoPE caches (stored in model dtype)
    cos_cache: CudaTensor<T>,
    sin_cache: CudaTensor<T>,
}

#[allow(private_bounds)]
impl<T> LlamaModel<T>
where
    T: TensorDType + DeviceRepr + GemmScalar + Default + ValidAsZeroBits + MaybeNcclType,
    CudaBlas: Gemm<T>,
{
    /// Load a Llama model from a directory containing SafeTensors and config.json
    ///
    /// # Errors
    /// Returns an error if loading fails
    pub fn from_pretrained(ctx: &CudaContext, model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();

        // Load config
        let config_path = model_path.join("config.json");
        let config = LlamaConfig::from_file(&config_path)?;

        // Load weights
        let loader = SafeTensorsLoader::from_directory(model_path)?;

        Self::load_weights(ctx, config, &loader)
    }

    /// Load a Llama model with tensor-parallel sharding across multiple GPUs.
    ///
    /// Requires SafeTensors format (FP8/BF16). GGUF is not supported for
    /// multi-GPU because sharding block-quantized formats is non-trivial.
    ///
    /// # Errors
    /// Returns an error if loading fails or if head counts are not evenly
    /// divisible by the world size.
    #[cfg(feature = "nccl")]
    pub fn from_pretrained_sharded(
        ctx: &CudaContext,
        model_path: impl AsRef<Path>,
        gpu_config: GpuConfig,
        nccl_comm: NcclCommunicator,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        let config_path = model_path.join("config.json");
        let config = LlamaConfig::from_file(&config_path)?;
        let loader = SafeTensorsLoader::from_directory(model_path)?;
        Self::load_weights_sharded(ctx, config, &loader, gpu_config, nccl_comm)
    }

    /// Load model weights from a weight loader
    #[allow(clippy::too_many_lines)]
    fn load_weights(
        ctx: &CudaContext,
        config: LlamaConfig,
        loader: &impl WeightLoader,
    ) -> Result<Self> {
        /// Load a linear weight — quantized if the tensor uses a quantized dtype,
        /// otherwise dense (pre-transposed). For FP8 weights, also loads the
        /// companion `weight_scale` tensor if present.
        fn load_linear<T: TensorDType + DeviceRepr>(
            ctx: &CudaContext,
            loader: &impl WeightLoader,
            name: &str,
        ) -> Result<LinearWeight<T>> {
            let dtype = loader.get_dtype(name)?;
            if dtype.is_quantized() {
                let mut qt = loader.load_quantized(ctx, name)?;

                // FP8 models store a per-tensor scale as a sibling tensor
                // e.g. "model.layers.0.self_attn.q_proj.weight" ->
                //      "model.layers.0.self_attn.q_proj.weight_scale"
                let scale_name = format!("{name}_scale");
                if loader.contains(&scale_name) {
                    let scale_tensor = loader.load_f32(ctx, &scale_name)?;
                    let scale_val = scale_tensor.to_vec()?;
                    qt.set_weight_scale(ctx, scale_val[0])?;
                }

                Ok(LinearWeight::Quantized(qt))
            } else {
                // Load as f32 for pretranspose (transpose_2d is f32-only),
                // then convert to T.
                let f32_weight = loader.load_f32(ctx, name)?;
                let transposed = pretranspose_weight(&f32_weight)?;
                if T::DTYPE == infernum::dtype::DType::F32 {
                    Ok(LinearWeight::Dense(reinterpret_tensor(transposed)))
                } else {
                    // Load directly in native dtype (already transposed shape)
                    // Re-load in native dtype and transpose on host
                    let native = load_typed::<T>(loader, ctx, name)?;
                    let shape = native.shape().to_vec();
                    let data = native.to_vec()?;
                    // Transpose on host: (out_features, in_features) -> (in_features, out_features)
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

        // Load embeddings
        let embed_tokens = load_typed::<T>(loader, ctx, "model.embed_tokens.weight")?;

        // Load transformer layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            let layer = LlamaLayerWeights {
                input_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.input_layernorm.weight"),
                )?,
                attention: {
                    let k = load_linear::<T>(
                        ctx,
                        loader,
                        &format!("{prefix}.self_attn.k_proj.weight"),
                    )?;
                    let v = load_linear::<T>(
                        ctx,
                        loader,
                        &format!("{prefix}.self_attn.v_proj.weight"),
                    )?;
                    let kv_proj = match (k, v) {
                        (LinearWeight::Dense(k_w), LinearWeight::Dense(v_w)) => {
                            KvProjWeight::Fused {
                                kv_dim: config.num_kv_heads() * config.head_dim(),
                                weight: concat_weights(&k_w, &v_w)?,
                            }
                        }
                        (k, v) => KvProjWeight::Separate {
                            k_proj: Box::new(k),
                            v_proj: Box::new(v),
                        },
                    };
                    LlamaAttentionWeights {
                        q_proj: load_linear::<T>(
                            ctx,
                            loader,
                            &format!("{prefix}.self_attn.q_proj.weight"),
                        )?,
                        kv_proj,
                        o_proj: load_linear::<T>(
                            ctx,
                            loader,
                            &format!("{prefix}.self_attn.o_proj.weight"),
                        )?,
                    }
                },
                post_attention_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                mlp: {
                    let gate =
                        load_linear::<T>(ctx, loader, &format!("{prefix}.mlp.gate_proj.weight"))?;
                    let up =
                        load_linear::<T>(ctx, loader, &format!("{prefix}.mlp.up_proj.weight"))?;
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
                    LlamaMlpWeights {
                        gate_up,
                        down_proj: load_linear::<T>(
                            ctx,
                            loader,
                            &format!("{prefix}.mlp.down_proj.weight"),
                        )?,
                    }
                },
            };

            layers.push(layer);
        }

        // Load final norm
        let norm = load_typed::<T>(loader, ctx, "model.norm.weight")?;

        // Load or tie lm_head
        let lm_head = if config.tie_word_embeddings {
            // Transpose embedding table for lm_head
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
            load_linear::<T>(ctx, loader, "lm_head.weight")?
        };

        // Precompute RoPE cache in f32, then convert to T
        let (cos_f32, sin_f32) = precompute_rope_cache(
            ctx,
            config.max_position_embeddings,
            config.head_dim(),
            config.rope_theta,
        )?;
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

    /// Load model weights with tensor-parallel sharding.
    ///
    /// Only supports dense (non-quantized) linear weights — FP8 quantized
    /// weights use `Replicate` strategy and are not sharded.
    #[cfg(feature = "nccl")]
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    fn load_weights_sharded(
        ctx: &CudaContext,
        config: LlamaConfig,
        loader: &impl WeightLoader,
        gpu_config: GpuConfig,
        nccl_comm: NcclCommunicator,
    ) -> Result<Self> {
        /// Load a sharded linear weight. Quantized weights fall back to
        /// `Replicate` (sharding block-quantized formats is unsupported).
        fn load_linear_sharded<T: TensorDType + DeviceRepr>(
            ctx: &CudaContext,
            loader: &impl WeightLoader,
            name: &str,
            shard: &ShardConfig,
            strategy: ShardStrategy,
        ) -> Result<LinearWeight<T>> {
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
        assert!(
            config.intermediate_size.is_multiple_of(world_size),
            "intermediate_size ({}) must be divisible by world_size ({world_size})",
            config.intermediate_size
        );

        // Embeddings and norms are replicated
        let embed_tokens = load_typed::<T>(loader, ctx, "model.embed_tokens.weight")?;

        let tp_num_heads = config.num_attention_heads / world_size;
        let tp_num_kv_heads = config.num_kv_heads() / world_size;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            let layer = LlamaLayerWeights {
                input_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.input_layernorm.weight"),
                )?,
                attention: {
                    let q_name = format!("{prefix}.self_attn.q_proj.weight");
                    let k_name = format!("{prefix}.self_attn.k_proj.weight");
                    let v_name = format!("{prefix}.self_attn.v_proj.weight");
                    let o_name = format!("{prefix}.self_attn.o_proj.weight");

                    let q_proj = load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &q_name,
                        &shard,
                        shard_strategy_for_weight(&q_name),
                    )?;
                    let k_proj = load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &k_name,
                        &shard,
                        shard_strategy_for_weight(&k_name),
                    )?;
                    let v_proj = load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &v_name,
                        &shard,
                        shard_strategy_for_weight(&v_name),
                    )?;

                    // With TP we keep K/V separate — fusing sharded weights
                    // would require matching the shard boundary to the K/V split.
                    let kv_proj = KvProjWeight::Separate {
                        k_proj: Box::new(k_proj),
                        v_proj: Box::new(v_proj),
                    };

                    LlamaAttentionWeights {
                        q_proj,
                        kv_proj,
                        o_proj: load_linear_sharded::<T>(
                            ctx,
                            loader,
                            &o_name,
                            &shard,
                            shard_strategy_for_weight(&o_name),
                        )?,
                    }
                },
                post_attention_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                mlp: {
                    let gate_name = format!("{prefix}.mlp.gate_proj.weight");
                    let up_name = format!("{prefix}.mlp.up_proj.weight");
                    let down_name = format!("{prefix}.mlp.down_proj.weight");

                    let gate = load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &gate_name,
                        &shard,
                        shard_strategy_for_weight(&gate_name),
                    )?;
                    let up = load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &up_name,
                        &shard,
                        shard_strategy_for_weight(&up_name),
                    )?;

                    // Keep gate/up separate for the same reason as K/V
                    let gate_up = GateUpWeight::Separate {
                        gate_proj: Box::new(gate),
                        up_proj: Box::new(up),
                    };

                    LlamaMlpWeights {
                        gate_up,
                        down_proj: load_linear_sharded::<T>(
                            ctx,
                            loader,
                            &down_name,
                            &shard,
                            shard_strategy_for_weight(&down_name),
                        )?,
                    }
                },
            };

            layers.push(layer);
        }

        // Final norm and lm_head are replicated
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
            )?
        };

        let (cos_f32, sin_f32) = precompute_rope_cache(
            ctx,
            config.max_position_embeddings,
            config.head_dim(),
            config.rope_theta,
        )?;
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

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// Forward pass with KV cache (prefill phase)
    ///
    /// Processes the full prompt, populating the KV cache for each layer,
    /// and returns logits for the **last** token only: shape `(1, vocab_size)`.
    ///
    /// After this call, `kv_cache.current_len()` equals `input_ids.len()`.
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

        // Embed tokens: (seq_len,) -> (seq_len, hidden_size)
        let mut hidden = self.embed(input_ids)?;

        // Run through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer_kv(&hidden, layer, layer_idx, kv_cache, position_offset)?;
        }

        // Advance KV cache position (once after all layers)
        kv_cache.advance(seq_len)?;

        // Final layer norm (in-place: hidden is consumed)
        rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;

        // Extract last hidden state, then project to vocab logits
        let last_hidden = self.extract_last_row(&hidden, seq_len)?;
        self.lm_head_forward(&last_hidden)
    }

    /// Forward pass for a single token with KV cache (decode phase)
    ///
    /// Processes one new token, appending its KV to the cache, and returns
    /// logits of shape `(1, vocab_size)`.
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

    /// Forward pass through a single transformer layer using KV cache
    fn forward_layer_kv(
        &self,
        hidden: &CudaTensor<T>,
        layer: &LlamaLayerWeights<T>,
        layer_idx: usize,
        kv_cache: &mut KvCache<T>,
        position_offset: usize,
    ) -> Result<CudaTensor<T>> {
        // Pre-attention RMS norm
        let normed = rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

        // Self-attention with KV cache
        let attn_output = self.forward_attention_kv(
            &normed,
            &layer.attention,
            layer_idx,
            kv_cache,
            position_offset,
        )?;

        // Residual add + pre-MLP RMS norm (fused in release builds)
        let (mut hidden, normed) = add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        // MLP
        let mlp_output = self.forward_mlp(&normed, &layer.mlp)?;

        // Residual connection (in-place: hidden += mlp_output)
        add_inplace(&mut hidden, &mlp_output)?;
        Ok(hidden)
    }

    /// Forward pass through attention with KV cache.
    ///
    /// Directly calls fused attention kernels (which are generic over T)
    /// rather than going through the `attention_kv` block, which has an
    /// f32-only decomposed fallback.
    fn forward_attention_kv(
        &self,
        hidden: &CudaTensor<T>,
        weights: &LlamaAttentionWeights<T>,
        layer_idx: usize,
        kv_cache: &mut KvCache<T>,
        position_offset: usize,
    ) -> Result<CudaTensor<T>> {
        let seq_len = hidden.shape()[0];
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim();

        // Project Q, K, V
        let q = linear(hidden, &weights.q_proj)?;
        let (k, v) = match &weights.kv_proj {
            KvProjWeight::Fused { weight, kv_dim } => {
                let kv = matmul(hidden, weight)?;
                if seq_len == 1 {
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

        // Reshape: (seq_len, num_heads * head_dim) -> (seq_len, num_heads, head_dim)
        let q = q.reshape(&[seq_len, num_heads, head_dim]);
        let k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
        let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

        // Apply RoPE with position offset
        let q = apply_rope(&q, &self.cos_cache, &self.sin_cache, position_offset)?;
        let k = apply_rope(&k, &self.cos_cache, &self.sin_cache, position_offset)?;

        // Append new K/V to cache
        kv_cache.append(layer_idx, &k, &v)?;

        // Retrieve full cached K/V including the just-appended tokens
        let total_len = kv_cache.current_len() + seq_len;
        let (k_full, v_full) = kv_cache.get_up_to(layer_idx, total_len);

        // Compute attention using fused kernels (generic over T)
        let attn_output = if seq_len == 1 {
            fused_attention_decode(&q, &k_full, &v_full)?
        } else {
            fused_attention_prefill(&q, &k_full, &v_full, kv_cache.current_len())?
        };

        // Reshape back: (seq_len, num_heads, head_dim) -> (seq_len, num_heads * head_dim)
        let attn_output = attn_output.reshape(&[seq_len, num_heads * head_dim]);

        // Output projection (row-parallel in TP: needs all-reduce)
        let mut out = linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    /// Extract the last row from a (seq_len, hidden_size) tensor
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

    /// Embed token IDs
    fn embed(&self, input_ids: &[u32]) -> Result<CudaTensor<T>> {
        embedding_gather(&self.ctx, &self.embed_tokens, input_ids)
    }

    /// Embed a single token ID already on the GPU (avoids `htod_sync_copy`)
    fn embed_from_device(&self, token_id_gpu: &CudaSlice<u32>) -> Result<CudaTensor<T>> {
        embedding_gather_from_device(&self.ctx, &self.embed_tokens, token_id_gpu, 1)
    }

    /// Decode-phase forward pass reading the token from a GPU buffer.
    ///
    /// Identical to [`Self::forward_next_token`] but avoids the host→device
    /// copy, making the entire call capturable by a CUDA graph.
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

    /// Decode-phase forward pass using indirect kernels for CUDA graph capture.
    ///
    /// Uses `_indirect` kernel variants that read position from stable device
    /// pointers. The entire call is capturable by a CUDA graph and the graph
    /// can be replayed without re-capture.
    ///
    /// **Does not call `kv_cache.advance()`** — the caller must do that
    /// outside the captured region.
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

    /// Transformer layer forward pass using indirect kernels.
    ///
    /// Reads position from `kv_cache.current_position()` device pointer.
    fn forward_layer_kv_indirect(
        &self,
        hidden: &CudaTensor<T>,
        layer: &LlamaLayerWeights<T>,
        layer_idx: usize,
        kv_cache: &mut KvCache<T>,
    ) -> Result<CudaTensor<T>> {
        let normed = rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

        let attn_output =
            self.forward_attention_kv_indirect(&normed, &layer.attention, layer_idx, kv_cache)?;

        let (mut hidden, normed) = add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        let mlp_output = self.forward_mlp(&normed, &layer.mlp)?;
        add_inplace(&mut hidden, &mlp_output)?;
        Ok(hidden)
    }

    /// Attention with indirect kernels (decode-only, seq_len=1).
    ///
    /// Uses `apply_rope_indirect`, `append_indirect`, and
    /// `fused_attention_decode_indirect` so all position-dependent parameters
    /// are read from stable device pointers.
    fn forward_attention_kv_indirect(
        &self,
        hidden: &CudaTensor<T>,
        weights: &LlamaAttentionWeights<T>,
        layer_idx: usize,
        kv_cache: &mut KvCache<T>,
    ) -> Result<CudaTensor<T>> {
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim();

        // Project Q, K, V
        let q = linear(hidden, &weights.q_proj)?;
        let (k, v) = match &weights.kv_proj {
            KvProjWeight::Fused { weight, kv_dim } => {
                let kv = matmul(hidden, weight)?;
                let k = kv.slice_view(0, &[1, *kv_dim]);
                let v = kv.slice_view(*kv_dim, &[1, *kv_dim]);
                (k, v)
            }
            KvProjWeight::Separate { k_proj, v_proj } => {
                let k = linear(hidden, k_proj)?;
                let v = linear(hidden, v_proj)?;
                (k, v)
            }
        };

        // Reshape to (1, num_heads, head_dim)
        let q = q.reshape(&[1, num_heads, head_dim]);
        let k = k.reshape(&[1, num_kv_heads, head_dim]);
        let v = v.reshape(&[1, num_kv_heads, head_dim]);

        // RoPE with indirect position
        let position = kv_cache.current_position();
        let q = apply_rope_indirect(&q, &self.cos_cache, &self.sin_cache, position)?;
        let k = apply_rope_indirect(&k, &self.cos_cache, &self.sin_cache, position)?;

        // Append to cache using indirect write offset
        kv_cache.append_indirect(layer_idx, &k, &v)?;

        // Full buffers (stable addresses for graph replay)
        let (k_full, v_full) = kv_cache.full_buffers(layer_idx);

        // Fused decode attention with indirect total_len.
        // total_len = current_len + 1 (includes the just-appended token).
        let total_len = kv_cache.current_total_len();
        let attn_output = fused_attention_decode_indirect(
            &q,
            k_full,
            v_full,
            total_len,
            kv_cache.graph_max_seq_len(),
        )?;

        let attn_output = attn_output.reshape(&[1, num_heads * head_dim]);
        let mut out = linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    /// Forward pass through MLP (SwiGLU)
    #[allow(clippy::unused_self)]
    fn forward_mlp(
        &self,
        hidden: &CudaTensor<T>,
        weights: &LlamaMlpWeights<T>,
    ) -> Result<CudaTensor<T>> {
        let (gate, up) = match &weights.gate_up {
            GateUpWeight::Fused {
                weight,
                intermediate_size,
            } => {
                let seq_len = hidden.shape()[0];
                let gate_up = matmul(hidden, weight)?;
                // gate_up shape: (seq_len, 2 * intermediate_size)
                if seq_len == 1 {
                    // Decode: zero-copy split via slice_view
                    let gate = gate_up.slice_view(0, &[1, *intermediate_size]);
                    let up = gate_up.slice_view(*intermediate_size, &[1, *intermediate_size]);
                    (gate, up)
                } else {
                    // Prefill: deinterleave rows into two contiguous tensors
                    split_gate_up(&gate_up, *intermediate_size)?
                }
            }
            GateUpWeight::Separate { gate_proj, up_proj } => {
                let gate = linear(hidden, gate_proj)?;
                let up = linear(hidden, up_proj)?;
                (gate, up)
            }
        };

        // SwiGLU activation
        let intermediate = swiglu(&gate, &up)?;

        // Down projection (row-parallel in TP: needs all-reduce)
        let mut out = linear(&intermediate, &weights.down_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    /// Project hidden states to vocabulary logits (always f32)
    fn lm_head_forward(&self, hidden: &CudaTensor<T>) -> Result<CudaTensor<f32>> {
        // bf16 + dense weight: use mixed-precision matmul to skip the cast kernel
        if T::DTYPE == infernum::dtype::DType::BF16 {
            if let LinearWeight::Dense(w) = &self.lm_head {
                // Zero-copy reinterpret via slice_view (shared Arc, no GPU copy)
                let h_bf16: CudaTensor<half::bf16> =
                    unsafe { hidden.slice_view(0, hidden.shape()).reinterpret() };
                let w_bf16: CudaTensor<half::bf16> =
                    unsafe { w.slice_view(0, w.shape()).reinterpret() };
                return matmul_bf16_f32(&h_bf16, &w_bf16);
            }
        }
        let logits_t = linear(hidden, &self.lm_head)?;
        if T::DTYPE == infernum::dtype::DType::F32 {
            // T is f32 — zero-copy reinterpret (same layout, no GPU op)
            return Ok(unsafe { logits_t.reinterpret() });
        }
        cast_to_f32(&logits_t)
    }
}

/// Methods only available for `LlamaModel<f32>` (GGUF loading, non-cached forward)
impl LlamaModel<f32> {
    /// Load a Llama model from a GGUF file containing quantized weights
    ///
    /// # Errors
    /// Returns an error if the file cannot be parsed or weights fail to load
    pub fn from_gguf(ctx: &CudaContext, gguf_path: impl AsRef<Path>) -> Result<Self> {
        let loader = GgufLoader::from_file(gguf_path)?;

        // Extract config from GGUF metadata
        let config = LlamaConfig::from_gguf_metadata(loader.metadata())?;

        Self::load_weights_gguf(ctx, config, &loader)
    }

    /// Load model weights from a GGUF loader, using quantized weights where available
    #[allow(clippy::too_many_lines)]
    fn load_weights_gguf(
        ctx: &CudaContext,
        config: LlamaConfig,
        loader: &GgufLoader,
    ) -> Result<Self> {
        /// Load a linear weight — quantized if the tensor uses a quantized dtype,
        /// otherwise f32 (pre-transposed).
        fn load_linear(
            ctx: &CudaContext,
            loader: &GgufLoader,
            name: &str,
        ) -> Result<LinearWeight<f32>> {
            let dtype = loader.get_dtype(name)?;
            if dtype.is_quantized() {
                Ok(LinearWeight::Quantized(loader.load_quantized(ctx, name)?))
            } else {
                Ok(LinearWeight::Dense(pretranspose_weight(
                    &loader.load_f32(ctx, name)?,
                )?))
            }
        }

        /// Load a Q/K projection weight, reversing the llama.cpp interleaved
        /// permutation so that infernum's sequential half-half RoPE is correct.
        fn load_linear_unpermute(
            ctx: &CudaContext,
            loader: &GgufLoader,
            name: &str,
            n_head: usize,
        ) -> Result<LinearWeight<f32>> {
            let dtype = loader.get_dtype(name)?;
            if dtype.is_quantized() {
                Ok(LinearWeight::Quantized(
                    loader.load_quantized_unpermute(ctx, name, n_head)?,
                ))
            } else {
                let tensor = loader.load_f32(ctx, name)?;
                let unpermuted = unpermute_f32(&tensor, n_head)?;
                Ok(LinearWeight::Dense(pretranspose_weight(&unpermuted)?))
            }
        }

        // Embeddings (always f32/f16)
        let embed_tokens = loader.load_f32(ctx, "token_embd.weight")?;

        // Transformer layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("blk.{i}");

            let layer = LlamaLayerWeights {
                input_layernorm: loader.load_f32(ctx, &format!("{prefix}.attn_norm.weight"))?,
                attention: {
                    let k = load_linear_unpermute(
                        ctx,
                        loader,
                        &format!("{prefix}.attn_k.weight"),
                        config.num_kv_heads(),
                    )?;
                    let v = load_linear(ctx, loader, &format!("{prefix}.attn_v.weight"))?;
                    let kv_proj = match (k, v) {
                        (LinearWeight::Dense(k_w), LinearWeight::Dense(v_w)) => {
                            KvProjWeight::Fused {
                                kv_dim: config.num_kv_heads() * config.head_dim(),
                                weight: concat_weights(&k_w, &v_w)?,
                            }
                        }
                        (k, v) => KvProjWeight::Separate {
                            k_proj: Box::new(k),
                            v_proj: Box::new(v),
                        },
                    };
                    LlamaAttentionWeights {
                        q_proj: load_linear_unpermute(
                            ctx,
                            loader,
                            &format!("{prefix}.attn_q.weight"),
                            config.num_attention_heads,
                        )?,
                        kv_proj,
                        o_proj: load_linear(ctx, loader, &format!("{prefix}.attn_output.weight"))?,
                    }
                },
                post_attention_layernorm: loader
                    .load_f32(ctx, &format!("{prefix}.ffn_norm.weight"))?,
                mlp: {
                    let gate = load_linear(ctx, loader, &format!("{prefix}.ffn_gate.weight"))?;
                    let up = load_linear(ctx, loader, &format!("{prefix}.ffn_up.weight"))?;
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
                    LlamaMlpWeights {
                        gate_up,
                        down_proj: load_linear(ctx, loader, &format!("{prefix}.ffn_down.weight"))?,
                    }
                },
            };

            layers.push(layer);
        }

        // Final norm
        let norm = loader.load_f32(ctx, "output_norm.weight")?;

        // Output head
        let lm_head = if config.tie_word_embeddings {
            let embd_dtype = loader.get_dtype("token_embd.weight")?;
            if embd_dtype.is_quantized() {
                LinearWeight::Quantized(loader.load_quantized(ctx, "token_embd.weight")?)
            } else {
                LinearWeight::Dense(pretranspose_weight(&embed_tokens)?)
            }
        } else if loader.contains("output.weight") {
            let dtype = loader.get_dtype("output.weight")?;
            if dtype.is_quantized() {
                LinearWeight::Quantized(loader.load_quantized(ctx, "output.weight")?)
            } else {
                LinearWeight::Dense(pretranspose_weight(
                    &loader.load_f32(ctx, "output.weight")?,
                )?)
            }
        } else {
            // Fallback: tie to embeddings
            let embd_dtype = loader.get_dtype("token_embd.weight")?;
            if embd_dtype.is_quantized() {
                LinearWeight::Quantized(loader.load_quantized(ctx, "token_embd.weight")?)
            } else {
                LinearWeight::Dense(pretranspose_weight(&embed_tokens)?)
            }
        };

        // Precompute RoPE cache
        let (cos_cache, sin_cache) = precompute_rope_cache(
            ctx,
            config.max_position_embeddings,
            config.head_dim(),
            config.rope_theta,
        )?;

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

    /// Run forward pass and return logits
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape (seq_len,)
    ///
    /// # Returns
    /// Logits tensor of shape (seq_len, vocab_size)
    ///
    /// # Errors
    /// Returns an error if forward pass fails
    pub fn forward(&self, input_ids: &[u32]) -> Result<CudaTensor<f32>> {
        let _seq_len = input_ids.len();

        // Embed tokens: (seq_len,) -> (seq_len, hidden_size)
        let mut hidden = self.embed(input_ids)?;

        // Run through transformer layers
        for layer in &self.layers {
            hidden = self.forward_layer(&hidden, layer)?;
        }

        // Final layer norm (in-place: hidden is consumed)
        rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;

        // Project to vocabulary
        let logits = linear(&hidden, &self.lm_head)?;

        Ok(logits)
    }

    /// Forward pass through a single transformer layer (no KV cache)
    fn forward_layer(
        &self,
        hidden: &CudaTensor<f32>,
        layer: &LlamaLayerWeights<f32>,
    ) -> Result<CudaTensor<f32>> {
        let _seq_len = hidden.shape()[0];
        let _hidden_size = self.config.hidden_size;

        // Pre-attention RMS norm
        let normed = rms_norm(hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

        // Self-attention
        let attn_output = self.forward_attention(&normed, &layer.attention)?;

        // Residual add + pre-MLP RMS norm (fused in release builds)
        let (mut hidden, normed) = add_rmsnorm(
            hidden,
            &attn_output,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        // MLP
        let mlp_output = self.forward_mlp(&normed, &layer.mlp)?;

        // Residual connection (in-place: hidden += mlp_output)
        add_inplace(&mut hidden, &mlp_output)?;
        Ok(hidden)
    }

    /// Forward pass through attention (no KV cache, f32 only)
    fn forward_attention(
        &self,
        hidden: &CudaTensor<f32>,
        weights: &LlamaAttentionWeights<f32>,
    ) -> Result<CudaTensor<f32>> {
        let seq_len = hidden.shape()[0];
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim();

        // Project Q, K, V
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

        // Reshape for attention: (seq_len, num_heads, head_dim)
        let q = q.reshape(&[seq_len, num_heads, head_dim]);
        let k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
        let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

        // Apply RoPE to Q and K
        let q = apply_rope(&q, &self.cos_cache, &self.sin_cache, 0)?;
        let k = apply_rope(&k, &self.cos_cache, &self.sin_cache, 0)?;

        // Expand K, V for GQA if needed
        let (k, v) = if num_kv_heads < num_heads {
            let k = repeat_kv(&k, num_heads / num_kv_heads)?;
            let v = repeat_kv(&v, num_heads / num_kv_heads)?;
            (k, v)
        } else {
            (k, v)
        };

        // Compute attention
        let attn_output = attention(&q, &k, &v, true)?;

        // Reshape back: (seq_len, num_heads, head_dim) -> (seq_len, num_heads * head_dim)
        let attn_output = attn_output.reshape(&[seq_len, num_heads * head_dim]);

        // Output projection (row-parallel in TP: needs all-reduce)
        let mut out = linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }
}

#[allow(private_bounds)]
impl<T> infernum::Model for LlamaModel<T>
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

    fn forward(&self, input_ids: &[u32]) -> Result<CudaTensor<f32>> {
        // Non-KV-cache forward: compute in T, cast logits to f32 at the end.
        let mut hidden = self.embed(input_ids)?;
        for layer in &self.layers {
            let normed = rms_norm(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            // For the trait's forward (no KV cache), we use fused attention
            // to avoid the f32-only decomposed attention path.
            let seq_len = hidden.shape()[0];
            let num_heads = self.tp_num_heads;
            let num_kv_heads = self.tp_num_kv_heads;
            let head_dim = self.config.head_dim();

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

            let q = q.reshape(&[seq_len, num_heads, head_dim]);
            let k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
            let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

            let q = apply_rope(&q, &self.cos_cache, &self.sin_cache, 0)?;
            let k = apply_rope(&k, &self.cos_cache, &self.sin_cache, 0)?;

            let attn_output = fused_attention_prefill(&q, &k, &v, 0)?;
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

            let mlp_output = self.forward_mlp(&normed, &layer.mlp)?;

            add_inplace(&mut h, &mlp_output)?;
            hidden = h;
        }

        rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;
        self.lm_head_forward(&hidden)
    }

    fn forward_with_kv_cache(
        &self,
        input_ids: &[u32],
        kv_cache: &mut KvCache<T>,
    ) -> Result<CudaTensor<f32>> {
        self.forward_with_kv_cache(input_ids, kv_cache)
    }

    fn forward_next_token(
        &self,
        token_id: u32,
        kv_cache: &mut KvCache<T>,
    ) -> Result<CudaTensor<f32>> {
        self.forward_next_token(token_id, kv_cache)
    }

    fn forward_next_token_device(
        &self,
        token_id_gpu: &CudaSlice<u32>,
        kv_cache: &mut KvCache<T>,
    ) -> Result<CudaTensor<f32>> {
        self.forward_next_token_device(token_id_gpu, kv_cache)
    }

    fn forward_next_token_indirect(
        &self,
        token_id_gpu: &CudaSlice<u32>,
        kv_cache: &mut KvCache<T>,
    ) -> Result<CudaTensor<f32>> {
        self.forward_next_token_indirect(token_id_gpu, kv_cache)
    }
}

/// Determine the sharding strategy for a given SafeTensors weight name.
///
/// - Column-parallel (split output dim): `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`
/// - Row-parallel (split input dim): `o_proj`, `down_proj`
/// - Replicate: norms, embeddings, RoPE caches, `lm_head`, scales, everything else
#[allow(dead_code)]
fn shard_strategy_for_weight(name: &str) -> ShardStrategy {
    // Scale tensors are always replicated (per-tensor scalars)
    if name.ends_with("_scale") {
        return ShardStrategy::Replicate;
    }

    // Column-parallel projections (split along output dimension)
    if name.ends_with("q_proj.weight")
        || name.ends_with("k_proj.weight")
        || name.ends_with("v_proj.weight")
        || name.ends_with("gate_proj.weight")
        || name.ends_with("up_proj.weight")
    {
        return ShardStrategy::Column;
    }

    // Row-parallel projections (split along input dimension)
    if name.ends_with("o_proj.weight") || name.ends_with("down_proj.weight") {
        return ShardStrategy::Row;
    }

    // Everything else: norms, embeddings, lm_head
    ShardStrategy::Replicate
}

/// Linear projection: output = input @ weight
///
/// For `Dense` weights: pre-transposed as (in_features, out_features), uses standard matmul.
/// For `Quantized` weights: stored as (out_features, in_features), dequantized on-the-fly.
///
/// # Panics
/// Panics if `Quantized` variant is used with `T != f32`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use infernum::dtype::DType;
    use std::collections::HashMap;

    #[test]
    fn test_linear() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // input: (2, 3), weight pre-transposed: (3, 4) -> output: (2, 4)
        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // Columns correspond to output features:
        // col 0: picks dim 0, col 1: picks dim 1, col 2: picks dim 2, col 3: sum
        let weight_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 1.0, // row 0
            0.0, 1.0, 0.0, 1.0, // row 1
            0.0, 0.0, 1.0, 1.0, // row 2
        ];

        let input = CudaTensor::from_slice(&ctx, &[2, 3], &input_data).unwrap();
        let weight =
            LinearWeight::Dense(CudaTensor::from_slice(&ctx, &[3, 4], &weight_data).unwrap());

        let output = linear(&input, &weight).unwrap();

        assert_eq!(output.shape(), &[2, 4]);

        let result = output.to_vec().unwrap();
        // row 0: [1, 2, 3, 6], row 1: [4, 5, 6, 15]
        assert!((result[0] - 1.0).abs() < 1e-4);
        assert!((result[1] - 2.0).abs() < 1e-4);
        assert!((result[2] - 3.0).abs() < 1e-4);
        assert!((result[3] - 6.0).abs() < 1e-4);
        assert!((result[4] - 4.0).abs() < 1e-4);
        assert!((result[5] - 5.0).abs() < 1e-4);
        assert!((result[6] - 6.0).abs() < 1e-4);
        assert!((result[7] - 15.0).abs() < 1e-4);
    }

    #[test]
    fn test_linear_bf16() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        // input: (2, 3), weight pre-transposed: (3, 4) -> output: (2, 4)
        let input_data: Vec<half::bf16> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(half::bf16::from_f32)
            .collect();
        let weight_data: Vec<half::bf16> = vec![
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(0.0),
            half::bf16::from_f32(0.0),
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(0.0),
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(0.0),
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(0.0),
            half::bf16::from_f32(0.0),
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(1.0),
        ];

        let input = CudaTensor::from_slice(&ctx, &[2, 3], &input_data).unwrap();
        let weight =
            LinearWeight::Dense(CudaTensor::from_slice(&ctx, &[3, 4], &weight_data).unwrap());

        let output = linear(&input, &weight).unwrap();

        assert_eq!(output.shape(), &[2, 4]);

        let result: Vec<f32> = output
            .to_vec()
            .unwrap()
            .into_iter()
            .map(half::bf16::to_f32)
            .collect();
        // Same expected results as f32 test: row 0: [1, 2, 3, 6], row 1: [4, 5, 6, 15]
        assert!((result[0] - 1.0).abs() < 0.1);
        assert!((result[1] - 2.0).abs() < 0.1);
        assert!((result[2] - 3.0).abs() < 0.1);
        assert!((result[3] - 6.0).abs() < 0.1);
        assert!((result[4] - 4.0).abs() < 0.1);
        assert!((result[5] - 5.0).abs() < 0.1);
        assert!((result[6] - 6.0).abs() < 0.1);
        assert!((result[7] - 15.0).abs() < 0.2);
    }

    // --- End-to-end tests with a tiny model ---

    /// Deterministic pseudo-random f32 in [-scale, scale] for reproducible test weights
    fn pseudo_random_weights(n: usize, scale: f32) -> Vec<f32> {
        let mut values = Vec::with_capacity(n);
        let mut state: u64 = 42;
        for _ in 0..n {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let f = (state as f32) / (u64::MAX as f32); // [0, 1)
            values.push((f * 2.0 - 1.0) * scale);
        }
        values
    }

    /// Mock weight loader that stores pre-built tensors by name
    struct MockWeightLoader {
        tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
    }

    impl MockWeightLoader {
        fn new() -> Self {
            Self {
                tensors: HashMap::new(),
            }
        }

        fn add(&mut self, name: &str, shape: &[usize], data: Vec<f32>) {
            self.tensors
                .insert(name.to_string(), (shape.to_vec(), data));
        }
    }

    impl WeightLoader for MockWeightLoader {
        fn load_f32(&self, ctx: &CudaContext, name: &str) -> Result<CudaTensor<f32>> {
            let (shape, data) = self
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("MockWeightLoader: tensor not found: {name}"));
            CudaTensor::from_slice(ctx, shape, data)
        }

        fn load_f16(
            &self,
            _ctx: &CudaContext,
            name: &str,
        ) -> Result<CudaTensor<infernum::dtype::F16>> {
            Err(infernum::Error::UnsupportedDtype(format!(
                "MockWeightLoader: load_f16 not supported (tensor: {name})"
            )))
        }

        fn load_bf16(
            &self,
            _ctx: &CudaContext,
            name: &str,
        ) -> Result<CudaTensor<infernum::dtype::BF16>> {
            Err(infernum::Error::UnsupportedDtype(format!(
                "MockWeightLoader: load_bf16 not supported (tensor: {name})"
            )))
        }

        fn get_shape(&self, name: &str) -> Result<Vec<usize>> {
            Ok(self.tensors.get(name).unwrap().0.clone())
        }

        fn get_dtype(&self, _name: &str) -> Result<DType> {
            Ok(DType::F32)
        }

        fn tensor_names(&self) -> Vec<String> {
            self.tensors.keys().cloned().collect()
        }

        fn contains(&self, name: &str) -> bool {
            self.tensors.contains_key(name)
        }
    }

    /// Build a tiny LlamaConfig for testing
    fn tiny_config() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: Some(2),
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }

    /// Build a MockWeightLoader with random weights matching the tiny config
    fn tiny_weight_loader(config: &LlamaConfig) -> MockWeightLoader {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();
        let scale = 0.02;

        let mut loader = MockWeightLoader::new();
        let mut seed_offset: usize = 0;

        let mut rand = |n: usize| -> Vec<f32> {
            seed_offset += 1;
            let mut vals = pseudo_random_weights(n, scale);
            // Rotate by seed_offset to get different values per tensor
            vals.rotate_left(seed_offset % n.max(1));
            vals
        };

        // Embeddings
        loader.add("model.embed_tokens.weight", &[vocab, h], rand(vocab * h));

        // Layer 0
        let prefix = "model.layers.0";
        loader.add(
            &format!("{prefix}.input_layernorm.weight"),
            &[h],
            vec![1.0; h],
        );
        loader.add(
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[num_heads * head_dim, h],
            rand(num_heads * head_dim * h),
        );
        loader.add(
            &format!("{prefix}.self_attn.k_proj.weight"),
            &[num_kv_heads * head_dim, h],
            rand(num_kv_heads * head_dim * h),
        );
        loader.add(
            &format!("{prefix}.self_attn.v_proj.weight"),
            &[num_kv_heads * head_dim, h],
            rand(num_kv_heads * head_dim * h),
        );
        loader.add(
            &format!("{prefix}.self_attn.o_proj.weight"),
            &[h, num_heads * head_dim],
            rand(h * num_heads * head_dim),
        );
        loader.add(
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[h],
            vec![1.0; h],
        );
        loader.add(
            &format!("{prefix}.mlp.gate_proj.weight"),
            &[inter, h],
            rand(inter * h),
        );
        loader.add(
            &format!("{prefix}.mlp.up_proj.weight"),
            &[inter, h],
            rand(inter * h),
        );
        loader.add(
            &format!("{prefix}.mlp.down_proj.weight"),
            &[h, inter],
            rand(h * inter),
        );

        // Final norm
        loader.add("model.norm.weight", &[h], vec![1.0; h]);

        // lm_head
        loader.add("lm_head.weight", &[vocab, h], rand(vocab * h));

        loader
    }

    fn build_tiny_model(ctx: &CudaContext) -> LlamaModel<f32> {
        let config = tiny_config();
        let loader = tiny_weight_loader(&config);
        LlamaModel::<f32>::load_weights(ctx, config, &loader).expect("Failed to build tiny model")
    }

    #[test]
    fn test_forward_output_shape() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_model(&ctx);

        let input_ids: Vec<u32> = vec![1, 5, 10];
        let logits = model.forward(&input_ids).expect("Forward pass failed");

        // logits should be (seq_len, vocab_size)
        assert_eq!(logits.shape(), &[3, 64]);
    }

    #[test]
    fn test_forward_single_token() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_model(&ctx);

        let logits = model.forward(&[0]).expect("Forward pass failed");
        assert_eq!(logits.shape(), &[1, 64]);

        // Logits should be finite
        let data = logits.to_vec().unwrap();
        assert!(
            data.iter().all(|x| x.is_finite()),
            "Logits contain non-finite values"
        );
    }

    #[test]
    fn test_forward_deterministic() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_model(&ctx);

        let input_ids: Vec<u32> = vec![1, 5, 10];
        let logits1 = model.forward(&input_ids).unwrap().to_vec().unwrap();
        let logits2 = model.forward(&input_ids).unwrap().to_vec().unwrap();

        for (i, (a, b)) in logits1.iter().zip(logits2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "Non-deterministic at index {i}: {a} vs {b}"
            );
        }
    }

    fn build_tiny_engine(ctx: &CudaContext) -> infernum_runtime::Engine<LlamaModel<f32>> {
        let model = build_tiny_model(ctx);
        infernum_runtime::Engine::new(ctx, model).expect("Failed to build engine")
    }

    #[test]
    fn test_generate_respects_max_tokens() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let mut engine = build_tiny_engine(&ctx);

        let prompt = vec![1_u32, 5, 10];
        let max_new = 4;
        let options = infernum::GenerateOptions {
            max_new_tokens: max_new,
            use_kv_cache: false,
            ..Default::default()
        };
        let tokens = engine.generate(&prompt, &options).unwrap();

        // Should produce at most prompt_len + max_new_tokens
        assert!(tokens.len() <= prompt.len() + max_new);
        assert!(
            tokens.len() > prompt.len(),
            "Should generate at least 1 token"
        );
        // Prompt should be preserved
        assert_eq!(&tokens[..prompt.len()], &prompt);
    }

    #[test]
    fn test_generate_stops_on_eos() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let mut engine = build_tiny_engine(&ctx);

        // Run generate with a very large max_new_tokens but EOS set to a
        // token that the model is likely to produce with random weights
        let prompt = vec![1_u32];
        let options_no_eos = infernum::GenerateOptions {
            max_new_tokens: 5,
            use_kv_cache: false,
            ..Default::default()
        };
        let result_no_eos = engine.generate(&prompt, &options_no_eos).unwrap();

        // The generated tokens should not include the EOS token if we set it
        // to one that was actually produced
        if result_no_eos.len() > 1 {
            let first_generated = result_no_eos[1];
            let options_with_eos = infernum::GenerateOptions {
                max_new_tokens: 100,
                eos_token_id: Some(first_generated),
                use_kv_cache: false,
                ..Default::default()
            };
            let result_with_eos = engine.generate(&prompt, &options_with_eos).unwrap();
            // Should stop immediately since the first generated token == EOS
            assert_eq!(
                result_with_eos.len(),
                prompt.len(),
                "Should stop before appending the EOS token"
            );
        }
    }

    #[test]
    fn test_forward_with_kv_cache_output_shape() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_model(&ctx);
        let config = tiny_config();

        let mut kv_cache = KvCache::new(
            &ctx,
            config.num_hidden_layers,
            config.max_position_embeddings,
            config.num_kv_heads(),
            config.head_dim(),
        )
        .unwrap();

        let input_ids: Vec<u32> = vec![1, 5, 10];
        let logits = model
            .forward_with_kv_cache(&input_ids, &mut kv_cache)
            .expect("Forward with KV cache failed");

        // Returns (1, vocab_size) — logits for last token only
        assert_eq!(logits.shape(), &[1, config.vocab_size]);
        assert_eq!(kv_cache.current_len(), input_ids.len());
    }

    #[test]
    fn test_forward_next_token_output_shape() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_model(&ctx);
        let config = tiny_config();

        let mut kv_cache = KvCache::new(
            &ctx,
            config.num_hidden_layers,
            config.max_position_embeddings,
            config.num_kv_heads(),
            config.head_dim(),
        )
        .unwrap();

        // Prefill
        let prompt: Vec<u32> = vec![1, 5, 10];
        model.forward_with_kv_cache(&prompt, &mut kv_cache).unwrap();

        // Decode one token
        let logits = model
            .forward_next_token(42, &mut kv_cache)
            .expect("Forward next token failed");

        assert_eq!(logits.shape(), &[1, config.vocab_size]);
        assert_eq!(kv_cache.current_len(), prompt.len() + 1);
    }

    #[test]
    fn test_kv_cache_reset_reuse() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_model(&ctx);
        let config = tiny_config();

        let mut kv_cache = KvCache::new(
            &ctx,
            config.num_hidden_layers,
            config.max_position_embeddings,
            config.num_kv_heads(),
            config.head_dim(),
        )
        .unwrap();

        // First sequence
        let prompt1: Vec<u32> = vec![1, 5, 10];
        model
            .forward_with_kv_cache(&prompt1, &mut kv_cache)
            .unwrap();
        assert_eq!(kv_cache.current_len(), 3);

        // Reset and process a different sequence
        kv_cache.reset().unwrap();
        assert_eq!(kv_cache.current_len(), 0);

        let prompt2: Vec<u32> = vec![2, 3];
        let logits = model
            .forward_with_kv_cache(&prompt2, &mut kv_cache)
            .unwrap();
        assert_eq!(logits.shape(), &[1, config.vocab_size]);
        assert_eq!(kv_cache.current_len(), 2);
    }

    #[test]
    fn test_kv_cache_generate_matches_naive_generate() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let mut engine = build_tiny_engine(&ctx);

        let prompt: Vec<u32> = vec![1, 5, 10];
        let max_new = 5;

        // Greedy generate without KV cache (recomputes full sequence each step)
        let naive_options = infernum::GenerateOptions {
            max_new_tokens: max_new,
            use_kv_cache: false,
            ..Default::default()
        };
        let tokens_naive = engine.generate(&prompt, &naive_options).unwrap();

        // Greedy generate with KV cache
        let kv_options = infernum::GenerateOptions {
            max_new_tokens: max_new,
            use_kv_cache: true,
            ..Default::default()
        };
        let tokens_kv = engine.generate(&prompt, &kv_options).unwrap();

        // Both should produce the exact same token sequence
        assert_eq!(
            tokens_naive, tokens_kv,
            "KV-cache generate should match naive generate"
        );
    }

    #[test]
    fn test_shard_strategy_column_parallel() {
        let column_names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.5.self_attn.k_proj.weight",
            "model.layers.31.self_attn.v_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        ];
        for name in &column_names {
            assert!(
                matches!(shard_strategy_for_weight(name), ShardStrategy::Column),
                "{name} should be Column"
            );
        }
    }

    #[test]
    fn test_shard_strategy_row_parallel() {
        let row_names = [
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.31.mlp.down_proj.weight",
        ];
        for name in &row_names {
            assert!(
                matches!(shard_strategy_for_weight(name), ShardStrategy::Row),
                "{name} should be Row"
            );
        }
    }

    #[test]
    fn test_shard_strategy_replicate() {
        let replicate_names = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight_scale",
        ];
        for name in &replicate_names {
            assert!(
                matches!(shard_strategy_for_weight(name), ShardStrategy::Replicate),
                "{name} should be Replicate"
            );
        }
    }
}
