//! Llama model implementation

#![allow(
    clippy::struct_field_names, // _proj suffix is conventional for Llama weights
    clippy::no_effect_underscore_binding,
    clippy::doc_markdown, // tensor shape docs trigger false positives
    unused_mut // variables are conditionally mutated via cfg(feature = "nccl")
)]

use std::path::Path;

use infernum::cuda::block_allocator::BlockTable;
use infernum::cuda::ops::{
    add_inplace, add_rmsnorm, apply_rope, apply_rope_batched, apply_rope_batched_indirect,
    apply_rope_indirect, attention, cast_to_f32, embedding_gather, embedding_gather_from_device,
    fused_attention_decode, fused_attention_decode_indirect, fused_attention_prefill,
    gather_paged_kv, matmul, matmul_bf16_f32, paged_attention_decode,
    paged_attention_decode_indirect, precompute_rope_cache, quantized_matmul, repeat_kv, rms_norm,
    rms_norm_inplace, swiglu, transpose_2d, GemmScalar,
};
use infernum::cuda::{
    BatchedGraphInputs, CudaBlas, CudaContext, CudaSlice, CudaTensor, DeviceRepr, Gemm, GpuConfig,
    PagedKvCache, QuantizedTensor, ShardStrategy, ValidAsZeroBits,
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

/// Weights for a single `MoE` expert (same structure as a dense MLP).
struct MoeExpertWeights<T: TensorDType> {
    mlp: LlamaMlpWeights<T>,
}

/// Feed-forward network weights: either a single dense MLP or a Mixture-of-Experts layer.
enum FfnWeights<T: TensorDType> {
    /// Standard dense MLP (Llama, etc.)
    Dense(Box<LlamaMlpWeights<T>>),
    /// Mixture-of-Experts (Mixtral, etc.)
    Moe {
        /// Router gate weight, pre-transposed: shape `[hidden_size, num_experts]`
        gate: CudaTensor<T>,
        /// Per-expert MLP weights
        experts: Vec<MoeExpertWeights<T>>,
        /// How many experts to activate per token
        num_experts_per_tok: usize,
    },
}

/// Weights for a single Llama decoder layer
struct LlamaLayerWeights<T: TensorDType> {
    input_layernorm: CudaTensor<T>,
    attention: LlamaAttentionWeights<T>,
    post_attention_layernorm: CudaTensor<T>,
    ffn: FfnWeights<T>,
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
        ///
        /// When `quant_config` is `Some`, uses GPTQ or AWQ loading instead of
        /// the generic `load_quantized` path.
        fn load_linear<T: TensorDType + DeviceRepr>(
            ctx: &CudaContext,
            loader: &impl WeightLoader,
            name: &str,
            quant_config: Option<&crate::QuantizationConfig>,
        ) -> Result<LinearWeight<T>> {
            // GPTQ/AWQ: load via dedicated loader using the layer prefix
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
                    _ => {
                        // Unknown quant method (e.g. "compressed-tensors") —
                        // fall through to standard weight loading
                    }
                }
            }

            let dtype = loader.get_dtype(name)?;
            if dtype.is_quantized() {
                let mut qt = loader.load_quantized(ctx, name)?;

                // FP8 models store a scale as a sibling tensor
                // e.g. "model.layers.0.self_attn.q_proj.weight" ->
                //      "model.layers.0.self_attn.q_proj.weight_scale"
                let scale_name = format!("{name}_scale");
                if loader.contains(&scale_name) {
                    let scale_tensor = loader.load_f32(ctx, &scale_name)?;
                    let scale_val = scale_tensor.to_vec()?;
                    if scale_val.len() == 1 {
                        // Per-tensor scale (single scalar)
                        qt.set_weight_scale(ctx, scale_val[0])?;
                    } else {
                        // Per-channel scale (one per output row)
                        qt.set_channel_scales(ctx, &scale_val)?;
                    }
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

        /// Load a dense MLP (gate_proj, up_proj, down_proj) for a single layer.
        fn load_dense_mlp<T: TensorDType + DeviceRepr>(
            ctx: &CudaContext,
            loader: &impl WeightLoader,
            layer_prefix: &str,
            config: &LlamaConfig,
            qc: Option<&crate::QuantizationConfig>,
        ) -> Result<LlamaMlpWeights<T>> {
            let gate = load_linear::<T>(
                ctx,
                loader,
                &format!("{layer_prefix}.mlp.gate_proj.weight"),
                qc,
            )?;
            let up = load_linear::<T>(
                ctx,
                loader,
                &format!("{layer_prefix}.mlp.up_proj.weight"),
                qc,
            )?;
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
            Ok(LlamaMlpWeights {
                gate_up,
                down_proj: load_linear::<T>(
                    ctx,
                    loader,
                    &format!("{layer_prefix}.mlp.down_proj.weight"),
                    qc,
                )?,
            })
        }

        /// Load MoE weights (router gate + per-expert MLPs) for a single layer.
        ///
        /// Mixtral SafeTensors naming:
        /// - `{prefix}.block_sparse_moe.gate.weight` — router `[num_experts, hidden_size]`
        /// - `{prefix}.block_sparse_moe.experts.{e}.w1.weight` — gate_proj
        /// - `{prefix}.block_sparse_moe.experts.{e}.w2.weight` — down_proj
        /// - `{prefix}.block_sparse_moe.experts.{e}.w3.weight` — up_proj
        fn load_moe_weights<T: TensorDType + DeviceRepr>(
            ctx: &CudaContext,
            loader: &impl WeightLoader,
            layer_prefix: &str,
            config: &LlamaConfig,
            qc: Option<&crate::QuantizationConfig>,
        ) -> Result<FfnWeights<T>> {
            let num_experts = config
                .num_local_experts
                .expect("MoE requires num_local_experts");
            let num_experts_per_tok = config
                .num_experts_per_tok
                .expect("MoE requires num_experts_per_tok");

            // Router gate: [num_experts, hidden_size] → pre-transpose to [hidden_size, num_experts]
            let gate_name = format!("{layer_prefix}.block_sparse_moe.gate.weight");
            let gate_f32 = loader.load_f32(ctx, &gate_name)?;
            let gate_transposed = pretranspose_weight(&gate_f32)?;
            let gate = if T::DTYPE == infernum::dtype::DType::F32 {
                reinterpret_tensor(gate_transposed)
            } else {
                let data_f32 = gate_transposed.to_vec()?;
                let data_t: Vec<T> = data_f32.iter().map(|&v| T::from_f32(v)).collect();
                CudaTensor::from_slice(ctx, gate_transposed.shape(), &data_t)?
            };

            // Load each expert's MLP
            let mut experts = Vec::with_capacity(num_experts);
            for e in 0..num_experts {
                let ep = format!("{layer_prefix}.block_sparse_moe.experts.{e}");
                // w1 = gate_proj, w2 = down_proj, w3 = up_proj
                let gate_proj = load_linear::<T>(ctx, loader, &format!("{ep}.w1.weight"), qc)?;
                let up_proj = load_linear::<T>(ctx, loader, &format!("{ep}.w3.weight"), qc)?;
                let gate_up = match (gate_proj, up_proj) {
                    (LinearWeight::Dense(g), LinearWeight::Dense(u)) => GateUpWeight::Fused {
                        weight: concat_weights(&g, &u)?,
                        intermediate_size: config.intermediate_size,
                    },
                    (g, u) => GateUpWeight::Separate {
                        gate_proj: Box::new(g),
                        up_proj: Box::new(u),
                    },
                };
                let down_proj = load_linear::<T>(ctx, loader, &format!("{ep}.w2.weight"), qc)?;
                experts.push(MoeExpertWeights {
                    mlp: LlamaMlpWeights { gate_up, down_proj },
                });
            }

            Ok(FfnWeights::Moe {
                gate,
                experts,
                num_experts_per_tok,
            })
        }

        let qc = config.quantization_config.as_ref();

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
                        qc,
                    )?;
                    let v = load_linear::<T>(
                        ctx,
                        loader,
                        &format!("{prefix}.self_attn.v_proj.weight"),
                        qc,
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
                            qc,
                        )?,
                        kv_proj,
                        o_proj: load_linear::<T>(
                            ctx,
                            loader,
                            &format!("{prefix}.self_attn.o_proj.weight"),
                            qc,
                        )?,
                    }
                },
                post_attention_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                ffn: if config.is_moe() {
                    load_moe_weights::<T>(ctx, loader, &prefix, &config, qc)?
                } else {
                    FfnWeights::Dense(Box::new(load_dense_mlp::<T>(
                        ctx, loader, &prefix, &config, qc,
                    )?))
                },
            };

            layers.push(layer);
        }

        // Load final norm
        let norm = load_typed::<T>(loader, ctx, "model.norm.weight")?;

        // Load or tie lm_head
        // For quantized models (GPTQ, AWQ, FP8), quantize lm_head to Q8_0
        // so decode uses the fast dp4a GEMV kernel instead of cuBLAS.
        let lm_head = if config.tie_word_embeddings {
            if qc.is_some() {
                // Quantize embedding table to Q8_0 for fast decode GEMV.
                // Shape is [vocab_size, hidden_dim] — already (N, K) for quantized_matmul.
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
                    // Untranspose: dense lm_head is [K, N] (pretransposed),
                    // but quantized_matmul expects [N, K] (row-major weights).
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
    /// Dense and GPTQ/AWQ INT4 linear weights are sharded according to
    /// `shard_strategy_for_weight`. FP8 block-quantized weights use
    /// `Replicate` strategy.
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
        /// `Replicate` (sharding quantized formats is unsupported).
        fn load_linear_sharded<T: TensorDType + DeviceRepr>(
            ctx: &CudaContext,
            loader: &impl WeightLoader,
            name: &str,
            shard: &ShardConfig,
            strategy: ShardStrategy,
            quant_config: Option<&crate::QuantizationConfig>,
        ) -> Result<LinearWeight<T>> {
            // GPTQ/AWQ: load via dedicated sharded loader
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
                    _ => {
                        // Unknown quant method (e.g. "compressed-tensors") —
                        // fall through to standard weight loading
                    }
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

        /// Load MoE weights with tensor-parallel sharding for a single layer.
        ///
        /// Each expert's MLP weights are sharded the same way as a dense MLP:
        /// `w1`/`w3` (gate/up) are column-parallel, `w2` (down) is row-parallel.
        /// The router gate is replicated on all ranks.
        fn load_moe_weights_sharded<T: TensorDType + DeviceRepr>(
            ctx: &CudaContext,
            loader: &impl WeightLoader,
            layer_prefix: &str,
            config: &LlamaConfig,
            shard: &ShardConfig,
            qc: Option<&crate::QuantizationConfig>,
        ) -> Result<FfnWeights<T>> {
            let num_experts = config
                .num_local_experts
                .expect("MoE requires num_local_experts");
            let num_experts_per_tok = config
                .num_experts_per_tok
                .expect("MoE requires num_experts_per_tok");

            // Router gate: replicated on all ranks
            // [num_experts, hidden_size] → pre-transpose to [hidden_size, num_experts]
            let gate_name = format!("{layer_prefix}.block_sparse_moe.gate.weight");
            let gate_f32 = loader.load_f32(ctx, &gate_name)?;
            let gate_transposed = pretranspose_weight(&gate_f32)?;
            let gate = if T::DTYPE == infernum::dtype::DType::F32 {
                reinterpret_tensor(gate_transposed)
            } else {
                let data_f32 = gate_transposed.to_vec()?;
                let data_t: Vec<T> = data_f32.iter().map(|&v| T::from_f32(v)).collect();
                CudaTensor::from_slice(ctx, gate_transposed.shape(), &data_t)?
            };

            // Load each expert's MLP with sharded weights
            let mut experts = Vec::with_capacity(num_experts);
            for e in 0..num_experts {
                let ep = format!("{layer_prefix}.block_sparse_moe.experts.{e}");
                // w1 = gate_proj (column-parallel), w3 = up_proj (column-parallel)
                let gate_proj = load_linear_sharded::<T>(
                    ctx,
                    loader,
                    &format!("{ep}.w1.weight"),
                    shard,
                    ShardStrategy::Column,
                    qc,
                )?;
                let up_proj = load_linear_sharded::<T>(
                    ctx,
                    loader,
                    &format!("{ep}.w3.weight"),
                    shard,
                    ShardStrategy::Column,
                    qc,
                )?;

                // Keep gate/up separate (same as dense sharded MLP)
                let gate_up = GateUpWeight::Separate {
                    gate_proj: Box::new(gate_proj),
                    up_proj: Box::new(up_proj),
                };

                // w2 = down_proj (row-parallel)
                let down_proj = load_linear_sharded::<T>(
                    ctx,
                    loader,
                    &format!("{ep}.w2.weight"),
                    shard,
                    ShardStrategy::Row,
                    qc,
                )?;

                experts.push(MoeExpertWeights {
                    mlp: LlamaMlpWeights { gate_up, down_proj },
                });
            }

            Ok(FfnWeights::Moe {
                gate,
                experts,
                num_experts_per_tok,
            })
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

        let qc = config.quantization_config.as_ref();

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
                        qc,
                    )?;
                    let k_proj = load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &k_name,
                        &shard,
                        shard_strategy_for_weight(&k_name),
                        qc,
                    )?;
                    let v_proj = load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &v_name,
                        &shard,
                        shard_strategy_for_weight(&v_name),
                        qc,
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
                            qc,
                        )?,
                    }
                },
                post_attention_layernorm: load_typed::<T>(
                    loader,
                    ctx,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                ffn: if config.is_moe() {
                    load_moe_weights_sharded::<T>(ctx, loader, &prefix, &config, &shard, qc)?
                } else {
                    let gate_name = format!("{prefix}.mlp.gate_proj.weight");
                    let up_name = format!("{prefix}.mlp.up_proj.weight");
                    let down_name = format!("{prefix}.mlp.down_proj.weight");

                    let gate = load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &gate_name,
                        &shard,
                        shard_strategy_for_weight(&gate_name),
                        qc,
                    )?;
                    let up = load_linear_sharded::<T>(
                        ctx,
                        loader,
                        &up_name,
                        &shard,
                        shard_strategy_for_weight(&up_name),
                        qc,
                    )?;

                    // Keep gate/up separate for the same reason as K/V
                    let gate_up = GateUpWeight::Separate {
                        gate_proj: Box::new(gate),
                        up_proj: Box::new(up),
                    };

                    FfnWeights::Dense(Box::new(LlamaMlpWeights {
                        gate_up,
                        down_proj: load_linear_sharded::<T>(
                            ctx,
                            loader,
                            &down_name,
                            &shard,
                            shard_strategy_for_weight(&down_name),
                            qc,
                        )?,
                    }))
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
                None,
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
        let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;

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
        let sliding_window = self.config.effective_sliding_window(layer_idx);
        let attn_output = if seq_len == 1 {
            fused_attention_decode(&q, &k_full, &v_full, None, None, sliding_window)?
        } else {
            fused_attention_prefill(
                &q,
                &k_full,
                &v_full,
                kv_cache.current_len(),
                None,
                None,
                sliding_window,
            )?
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

    /// Batched decode forward pass with paged KV cache.
    ///
    /// Processes one token per sequence for `batch_size` sequences, each with
    /// its own block table into the shared KV pool. Returns logits of shape
    /// `(batch_size, vocab_size)`.
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

        // Embed tokens: (batch_size,) -> (batch_size, hidden_size)
        let mut hidden = self.embed(token_ids)?;

        // Run through transformer layers
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

        // Append K/V to paged cache for each sequence is done inside
        // forward_layer_paged_decode, but we still need to advance block tables
        // and positions on the caller side (the engine handles this).

        // Final layer norm
        rms_norm_inplace(&mut hidden, &self.norm, self.config.rms_norm_eps)?;

        // Project to logits: (batch_size, hidden_size) -> (batch_size, vocab_size)
        self.lm_head_forward(&hidden.reshape(&[batch_size, self.config.hidden_size]))
    }

    /// Batched decode using indirect kernels for CUDA graph capture.
    ///
    /// All per-sequence metadata (token IDs, positions, block tables, seq_lens)
    /// is read from GPU-resident buffers in `graph_inputs`. Padding sequences
    /// (beyond `actual_batch_size`) are processed but their logits are ignored.
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

    /// Transformer layer forward pass for batched decode with paged KV cache.
    fn forward_layer_paged_decode(
        &self,
        hidden: &CudaTensor<T>,
        layer: &LlamaLayerWeights<T>,
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

    /// Attention for batched decode with paged KV cache.
    ///
    /// Batches the Q/K/V projections and output projection (large matmuls),
    /// then processes RoPE, K/V append, and attention per-sequence since each
    /// has a different position offset. Per-sequence attention results are
    /// collected into a single output via O-projection batching.
    fn forward_attention_paged_decode(
        &self,
        hidden: &CudaTensor<T>,
        weights: &LlamaAttentionWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        block_tables: &[BlockTable],
        positions: &[usize],
    ) -> Result<CudaTensor<T>> {
        let batch_size = hidden.shape()[0];
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim();

        // Batch Q/K/V projections: (batch_size, hidden) -> (batch_size, proj_dim)
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

        // Reshape: (batch_size, dim) -> (batch_size, heads, head_dim)
        let q = q.reshape(&[batch_size, num_heads, head_dim]);
        let k = k.reshape(&[batch_size, num_kv_heads, head_dim]);
        let v = v.reshape(&[batch_size, num_kv_heads, head_dim]);

        let q_stride = num_heads * head_dim;
        let kv_stride = num_kv_heads * head_dim;
        let sliding_window = self.config.effective_sliding_window(layer_idx);

        // Batched RoPE: one kernel launch for all sequences
        let q = apply_rope_batched(&q, &self.cos_cache, &self.sin_cache, positions)?;
        let k = apply_rope_batched(&k, &self.cos_cache, &self.sin_cache, positions)?;

        // Per-sequence: K/V append + paged decode attention
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
        let _ = sliding_window; // will be used when batched paged attention supports it

        // Batched output projection: (batch_size, num_heads * head_dim) -> (batch_size, hidden)
        let mut out = linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    /// Transformer layer for batched decode using indirect (GPU-resident) inputs.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_paged_decode_indirect(
        &self,
        hidden: &CudaTensor<T>,
        layer: &LlamaLayerWeights<T>,
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

    /// Attention for batched decode using GPU-resident positions, block tables,
    /// and sequence lengths.
    #[allow(clippy::too_many_arguments)]
    fn forward_attention_paged_decode_indirect(
        &self,
        hidden: &CudaTensor<T>,
        weights: &LlamaAttentionWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        graph_inputs: &BatchedGraphInputs,
        max_seq_len: usize,
    ) -> Result<CudaTensor<T>> {
        let batch_size = hidden.shape()[0];
        let num_heads = self.tp_num_heads;
        let num_kv_heads = self.tp_num_kv_heads;
        let head_dim = self.config.head_dim();

        // Batched Q/K/V projections
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

        // Reshape: (batch_size, dim) -> (batch_size, heads, head_dim)
        let q = q.reshape(&[batch_size, num_heads, head_dim]);
        let k = k.reshape(&[batch_size, num_kv_heads, head_dim]);
        let v = v.reshape(&[batch_size, num_kv_heads, head_dim]);

        // Batched RoPE from GPU-resident positions
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

        // Batched paged K/V append from GPU-resident block tables + positions
        paged_kv.append_paged_batched(
            layer_idx,
            &k,
            &v,
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
        )?;

        let attn_output = attn_output.reshape(&[batch_size, num_heads * head_dim]);

        // Batched output projection
        let mut out = linear(&attn_output, &weights.o_proj)?;
        #[cfg(feature = "nccl")]
        nccl_all_reduce(self.nccl_comm.as_ref(), &mut out)?;
        Ok(out)
    }

    /// Transformer layer forward pass for single-sequence prefill with paged KV cache.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_paged_prefill(
        &self,
        hidden: &CudaTensor<T>,
        layer: &LlamaLayerWeights<T>,
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

    /// Attention for single-sequence prefill with paged KV cache.
    ///
    /// Uses the gather approach: write K/V into paged cache, then gather
    /// contiguous K/V for use with the existing fused prefill kernel.
    #[allow(clippy::too_many_arguments)]
    fn forward_attention_paged_prefill(
        &self,
        hidden: &CudaTensor<T>,
        weights: &LlamaAttentionWeights<T>,
        layer_idx: usize,
        paged_kv: &mut PagedKvCache<T>,
        block_table: &BlockTable,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<CudaTensor<T>> {
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

        let q = q.reshape(&[seq_len, num_heads, head_dim]);
        let k = k.reshape(&[seq_len, num_kv_heads, head_dim]);
        let v = v.reshape(&[seq_len, num_kv_heads, head_dim]);

        let q = apply_rope(&q, &self.cos_cache, &self.sin_cache, start_pos)?;
        let k = apply_rope(&k, &self.cos_cache, &self.sin_cache, start_pos)?;

        // Write K/V into paged cache
        paged_kv.append_paged(layer_idx, block_table, &k, &v, start_pos)?;

        // Gather contiguous K/V for the fused prefill kernel.
        // Create a temporary table with updated seq_len so gather knows
        // how many tokens to copy.
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

        let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;
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
        let sliding_window = self.config.effective_sliding_window(layer_idx);
        let total_len = kv_cache.current_total_len();
        let attn_output = fused_attention_decode_indirect(
            &q,
            k_full,
            v_full,
            total_len,
            kv_cache.graph_max_seq_len(),
            None,
            None,
            sliding_window,
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

    /// Forward pass through MLP without all-reduce.
    ///
    /// Used by MoE: each expert produces a partial (rank-local) output that
    /// gets weighted-summed, and a single all-reduce is applied after
    /// combining all experts rather than once per expert.
    #[allow(clippy::unused_self)]
    fn forward_mlp_no_reduce(
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

    /// Dispatch to dense MLP or MoE forward pass.
    fn forward_ffn(&self, hidden: &CudaTensor<T>, ffn: &FfnWeights<T>) -> Result<CudaTensor<T>> {
        match ffn {
            FfnWeights::Dense(mlp) => self.forward_mlp(hidden, mlp),
            FfnWeights::Moe {
                gate,
                experts,
                num_experts_per_tok,
            } => self.forward_moe(hidden, gate, experts, *num_experts_per_tok),
        }
    }

    /// Forward pass through a Mixture-of-Experts layer.
    ///
    /// Each expert runs without all-reduce (via `forward_mlp_no_reduce`);
    /// the partial rank-local outputs are weighted-summed by `moe_forward`,
    /// and a single all-reduce is applied to the combined result.
    fn forward_moe(
        &self,
        hidden: &CudaTensor<T>,
        gate: &CudaTensor<T>,
        experts: &[MoeExpertWeights<T>],
        num_experts_per_tok: usize,
    ) -> Result<CudaTensor<T>> {
        let mut out = infernum::cuda::moe::moe_forward(
            hidden,
            gate,
            experts.len(),
            num_experts_per_tok,
            true, // Mixtral always renormalizes top-K weights
            |expert_idx, expert_input| {
                self.forward_mlp_no_reduce(expert_input, &experts[expert_idx].mlp)
            },
        )?;
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
                ffn: {
                    assert!(!config.is_moe(), "MoE GGUF loading is not yet supported");
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
                    FfnWeights::Dense(Box::new(LlamaMlpWeights {
                        gate_up,
                        down_proj: load_linear(ctx, loader, &format!("{prefix}.ffn_down.weight"))?,
                    }))
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
        let mlp_output = self.forward_ffn(&normed, &layer.ffn)?;

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

    fn devices(&self) -> Vec<&CudaContext> {
        vec![&self.ctx]
    }

    fn forward(&self, input_ids: &[u32]) -> Result<CudaTensor<f32>> {
        // Non-KV-cache forward: compute in T, cast logits to f32 at the end.
        let mut hidden = self.embed(input_ids)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
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

    fn forward_prefill_paged(
        &self,
        input_ids: &[u32],
        paged_kvs: &mut [PagedKvCache<T>],
        block_table: &BlockTable,
        start_pos: usize,
    ) -> Result<CudaTensor<f32>> {
        self.forward_prefill_paged(input_ids, paged_kvs, block_table, start_pos)
    }
}

#[cfg(feature = "nccl")]
#[allow(private_bounds)]
impl<T> infernum::ShardedLoadable for LlamaModel<T>
where
    T: TensorDType
        + DeviceRepr
        + GemmScalar
        + Default
        + ValidAsZeroBits
        + MaybeNcclType
        + Send
        + Sync,
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

/// Determine the sharding strategy for a given SafeTensors weight name.
///
/// - Column-parallel (split output dim): `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`
/// - Row-parallel (split input dim): `o_proj`, `down_proj`
/// - MoE experts: `w1`/`w3` Column, `w2` Row (Mixtral naming)
/// - Replicate: norms, embeddings, RoPE caches, `lm_head`, router gate, scales, everything else
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

    // MoE expert projections (Mixtral SafeTensors naming)
    // w1 = gate_proj (column-parallel), w3 = up_proj (column-parallel)
    // w2 = down_proj (row-parallel)
    // Router gate falls through to Replicate below.
    if name.contains(".block_sparse_moe.experts.") {
        if name.ends_with(".w1.weight") || name.ends_with(".w3.weight") {
            return ShardStrategy::Column;
        }
        if name.ends_with(".w2.weight") {
            return ShardStrategy::Row;
        }
    }

    // Everything else: norms, embeddings, lm_head, router gate
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
    /// Stored GPTQ data for a single linear layer
    struct GptqLayerData {
        shape: Vec<usize>,
        qweight: Vec<u8>,
        scales: Vec<u8>,
        qzeros: Vec<u8>,
        group_size: usize,
    }

    struct MockWeightLoader {
        tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
        gptq_weights: HashMap<String, GptqLayerData>,
    }

    impl MockWeightLoader {
        fn new() -> Self {
            Self {
                tensors: HashMap::new(),
                gptq_weights: HashMap::new(),
            }
        }

        fn add(&mut self, name: &str, shape: &[usize], data: Vec<f32>) {
            self.tensors
                .insert(name.to_string(), (shape.to_vec(), data));
        }

        fn add_gptq(
            &mut self,
            prefix: &str,
            shape: &[usize],
            qweight: Vec<u8>,
            scales: Vec<u8>,
            qzeros: Vec<u8>,
            group_size: usize,
        ) {
            self.gptq_weights.insert(
                prefix.to_string(),
                GptqLayerData {
                    shape: shape.to_vec(),
                    qweight,
                    scales,
                    qzeros,
                    group_size,
                },
            );
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

        fn load_gptq_linear(
            &self,
            ctx: &CudaContext,
            prefix: &str,
            _group_size: usize,
        ) -> Result<QuantizedTensor> {
            let data = self
                .gptq_weights
                .get(prefix)
                .unwrap_or_else(|| panic!("MockWeightLoader: GPTQ prefix not found: {prefix}"));
            QuantizedTensor::from_gptq_raw(
                ctx,
                &data.shape,
                DType::GPTQ_INT4,
                &data.qweight,
                &data.scales,
                &data.qzeros,
                data.group_size,
            )
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
            model_type: "llama".to_string(),
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
            quantization_config: None,
            num_local_experts: None,
            num_experts_per_tok: None,
            sliding_window: None,
            use_sliding_window: false,
            max_window_layers: None,
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

    /// Pack f32 weights into GPTQ INT4 format on the host.
    ///
    /// - `weights`: `[out_features, in_features]` in row-major order
    /// - Returns `(qweight, scales, qzeros)` as raw byte vectors
    fn pack_gptq_test(
        weights: &[f32],
        out_features: usize,
        in_features: usize,
        group_size: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        assert_eq!(weights.len(), out_features * in_features);
        assert_eq!(in_features % 8, 0);
        assert_eq!(in_features % group_size, 0);
        assert_eq!(out_features % 8, 0);

        let num_groups = in_features / group_size;
        let packed_rows = in_features / 8;
        let zero_point = 8_i32;

        let mut scales_f16 = vec![half::f16::from_f32(0.0); num_groups * out_features];
        let mut quantized = vec![0_i32; out_features * in_features];

        for n in 0..out_features {
            for g in 0..num_groups {
                let k_start = g * group_size;
                let k_end = k_start + group_size;
                let group_vals: Vec<f32> = (k_start..k_end)
                    .map(|k| weights[n * in_features + k])
                    .collect();
                let max_abs = group_vals.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
                scales_f16[g * out_features + n] = half::f16::from_f32(scale);

                for (j, &v) in group_vals.iter().enumerate() {
                    let q = ((v / scale).round() as i32 + zero_point).clamp(0, 15);
                    quantized[n * in_features + k_start + j] = q;
                }
            }
        }

        // Pack qweight: [in_features/8, out_features] as int32
        let mut qweight = vec![0_u8; packed_rows * out_features * 4];
        for pr in 0..packed_rows {
            for n in 0..out_features {
                let mut packed: u32 = 0;
                for j in 0..8 {
                    let k = pr * 8 + j;
                    let q = quantized[n * in_features + k] as u32;
                    packed |= (q & 0xF) << (j * 4);
                }
                let idx = (pr * out_features + n) * 4;
                qweight[idx..idx + 4].copy_from_slice(&packed.to_le_bytes());
            }
        }

        // Pack scales: [num_groups, out_features] as f16
        let mut scales_bytes = vec![0_u8; num_groups * out_features * 2];
        for (i, &s) in scales_f16.iter().enumerate() {
            let bytes = s.to_le_bytes();
            scales_bytes[i * 2] = bytes[0];
            scales_bytes[i * 2 + 1] = bytes[1];
        }

        // Pack qzeros: [num_groups, out_features/8] as int32
        let qzeros_cols = out_features / 8;
        let mut qzeros = vec![0_u8; num_groups * qzeros_cols * 4];
        for g in 0..num_groups {
            for col in 0..qzeros_cols {
                let mut packed: u32 = 0;
                for j in 0..8 {
                    packed |= (zero_point as u32 & 0xF) << (j * 4);
                }
                let idx = (g * qzeros_cols + col) * 4;
                qzeros[idx..idx + 4].copy_from_slice(&packed.to_le_bytes());
            }
        }

        (qweight, scales_bytes, qzeros)
    }

    /// Build a tiny LlamaConfig with GPTQ quantization enabled.
    /// Uses group_size=32 to keep dimensions small.
    fn tiny_gptq_config() -> LlamaConfig {
        LlamaConfig {
            model_type: "llama".to_string(),
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
            quantization_config: Some(crate::QuantizationConfig {
                quant_method: "gptq".to_string(),
                bits: 4,
                group_size: 32,
            }),
            num_local_experts: None,
            num_experts_per_tok: None,
            sliding_window: None,
            use_sliding_window: false,
            max_window_layers: None,
        }
    }

    /// Generate pseudo-random weights and pack into GPTQ format, advancing the seed.
    fn rand_gptq(
        seed_offset: &mut usize,
        out_f: usize,
        in_f: usize,
        group_size: usize,
        scale: f32,
    ) -> (Vec<usize>, Vec<u8>, Vec<u8>, Vec<u8>) {
        *seed_offset += 1;
        let mut w = pseudo_random_weights(out_f * in_f, scale);
        w.rotate_left(*seed_offset % (out_f * in_f).max(1));
        let (qw, sc, qz) = pack_gptq_test(&w, out_f, in_f, group_size);
        (vec![out_f, in_f], qw, sc, qz)
    }

    /// Generate pseudo-random f32 data, advancing the seed offset
    fn rand_f32(seed_offset: &mut usize, n: usize, scale: f32) -> Vec<f32> {
        *seed_offset += 1;
        let mut vals = pseudo_random_weights(n, scale);
        vals.rotate_left(*seed_offset % n.max(1));
        vals
    }

    /// Build a MockWeightLoader with GPTQ-quantized linear layers
    fn tiny_gptq_weight_loader(config: &LlamaConfig) -> MockWeightLoader {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();
        let group_size = config.quantization_config.as_ref().unwrap().group_size;
        let scale = 0.02;

        let mut loader = MockWeightLoader::new();
        let mut seed_offset: usize = 0;

        // Embeddings (always dense f32)
        loader.add(
            "model.embed_tokens.weight",
            &[vocab, h],
            rand_f32(&mut seed_offset, vocab * h, scale),
        );

        // Layer 0
        let prefix = "model.layers.0";
        loader.add(
            &format!("{prefix}.input_layernorm.weight"),
            &[h],
            vec![1.0; h],
        );

        let gptq_layers: Vec<(&str, usize, usize)> = vec![
            ("self_attn.q_proj", num_heads * head_dim, h),
            ("self_attn.k_proj", num_kv_heads * head_dim, h),
            ("self_attn.v_proj", num_kv_heads * head_dim, h),
            ("self_attn.o_proj", h, num_heads * head_dim),
            ("mlp.gate_proj", inter, h),
            ("mlp.up_proj", inter, h),
            ("mlp.down_proj", h, inter),
        ];

        for (name, out_f, in_f) in &gptq_layers {
            let (shape, qw, sc, qz) = rand_gptq(&mut seed_offset, *out_f, *in_f, group_size, scale);
            loader.add_gptq(&format!("{prefix}.{name}"), &shape, qw, sc, qz, group_size);
        }

        loader.add(
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[h],
            vec![1.0; h],
        );

        // Final norm
        loader.add("model.norm.weight", &[h], vec![1.0; h]);

        // lm_head (always dense, never quantized)
        loader.add(
            "lm_head.weight",
            &[vocab, h],
            rand_f32(&mut seed_offset, vocab * h, scale),
        );

        loader
    }

    fn build_tiny_gptq_model(ctx: &CudaContext) -> LlamaModel<f32> {
        let config = tiny_gptq_config();
        let loader = tiny_gptq_weight_loader(&config);
        LlamaModel::<f32>::load_weights(ctx, config, &loader)
            .expect("Failed to build tiny GPTQ model")
    }

    #[test]
    fn test_linear_gptq() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let k = 32;
        let n = 8;
        let group_size = 32;

        // Constant weight = 1.0 → output[i] ≈ sum of input row
        let w_data = vec![1.0_f32; n * k];
        let (qw, sc, qz) = pack_gptq_test(&w_data, n, k, group_size);

        let weight = LinearWeight::Quantized(
            QuantizedTensor::from_gptq_raw(
                &ctx,
                &[n, k],
                DType::GPTQ_INT4,
                &qw,
                &sc,
                &qz,
                group_size,
            )
            .unwrap(),
        );

        // Input: row of 1.0s → expected output ≈ 32.0 per output
        let input_data = vec![1.0_f32; k];
        let input = CudaTensor::from_slice(&ctx, &[1, k], &input_data).unwrap();

        let output = linear(&input, &weight).unwrap();
        assert_eq!(output.shape(), &[1, n]);

        let result = output.to_vec().unwrap();
        let expected = 32.0_f32;
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected).abs() < expected * 0.15,
                "GPTQ linear [{i}]: {v} vs expected ~{expected}",
            );
        }
    }

    #[test]
    fn test_forward_gptq_output_shape() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_gptq_model(&ctx);

        let input_ids: Vec<u32> = vec![1, 5, 10];
        let logits = model.forward(&input_ids).expect("Forward pass failed");

        assert_eq!(logits.shape(), &[3, 64]);
    }

    #[test]
    fn test_forward_gptq_single_token() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_gptq_model(&ctx);

        let logits = model.forward(&[0]).expect("Forward pass failed");
        assert_eq!(logits.shape(), &[1, 64]);

        let data = logits.to_vec().unwrap();
        assert!(
            data.iter().all(|x| x.is_finite()),
            "Logits contain non-finite values"
        );
    }

    #[test]
    fn test_forward_gptq_deterministic() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_gptq_model(&ctx);

        let input_ids: Vec<u32> = vec![1, 5, 10];
        let logits1 = model.forward(&input_ids).unwrap().to_vec().unwrap();
        let logits2 = model.forward(&input_ids).unwrap().to_vec().unwrap();

        for (i, (a, b)) in logits1.iter().zip(logits2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "GPTQ non-deterministic at index {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_generate_gptq_respects_max_tokens() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model = build_tiny_gptq_model(&ctx);
        let engine = infernum_runtime::Engine::new(model).expect("Failed to build engine");

        let prompt = vec![1_u32, 5, 10];
        let max_new = 4;
        let options = infernum::GenerateOptions {
            max_new_tokens: max_new,
            use_kv_cache: false,
            ..Default::default()
        };
        let tokens = engine.generate(&prompt, &options).unwrap();

        assert!(tokens.len() <= prompt.len() + max_new);
        assert!(
            tokens.len() > prompt.len(),
            "Should generate at least 1 token"
        );
        assert_eq!(&tokens[..prompt.len()], &prompt);
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

    fn build_tiny_engine(ctx: &CudaContext) -> infernum_runtime::Engine {
        let model = build_tiny_model(ctx);
        infernum_runtime::Engine::new(model).expect("Failed to build engine")
    }

    #[test]
    fn test_generate_respects_max_tokens() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let engine = build_tiny_engine(&ctx);

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
        let engine = build_tiny_engine(&ctx);

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
        let engine = build_tiny_engine(&ctx);

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

    /// Slice columns from a 2D row-major byte buffer (test helper).
    fn slice_columns(
        data: &[u8],
        rows: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
        elem_bytes: usize,
    ) -> Vec<u8> {
        let row_bytes = total_cols * elem_bytes;
        let col_start_bytes = col_start * elem_bytes;
        let col_count_bytes = col_count * elem_bytes;
        let mut result = Vec::with_capacity(rows * col_count_bytes);
        for r in 0..rows {
            let off = r * row_bytes + col_start_bytes;
            result.extend_from_slice(&data[off..off + col_count_bytes]);
        }
        result
    }

    /// Slice contiguous rows from a 2D row-major byte buffer (test helper).
    fn slice_rows(
        data: &[u8],
        cols: usize,
        row_start: usize,
        row_count: usize,
        elem_bytes: usize,
    ) -> Vec<u8> {
        let row_bytes = cols * elem_bytes;
        let start = row_start * row_bytes;
        data[start..start + row_count * row_bytes].to_vec()
    }

    #[test]
    fn test_linear_gptq_column_shard() {
        // Column-parallel: split output dimension N into 2 shards, verify
        // that concatenating shard outputs matches the full output.
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let k = 32;
        let n = 16; // must be divisible by 8 * world_size
        let group_size = 32;
        let world_size = 2;

        let w_data: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.01) - 0.5).collect();
        let (qw, sc, qz) = pack_gptq_test(&w_data, n, k, group_size);

        // Full (non-sharded) output
        let full_weight = LinearWeight::Quantized(
            QuantizedTensor::from_gptq_raw(
                &ctx,
                &[n, k],
                DType::GPTQ_INT4,
                &qw,
                &sc,
                &qz,
                group_size,
            )
            .unwrap(),
        );
        let input_data: Vec<f32> = (0..k).map(|i| i as f32 * 0.1).collect();
        let input = CudaTensor::from_slice(&ctx, &[1, k], &input_data).unwrap();
        let full_output = linear(&input, &full_weight).unwrap().to_vec().unwrap();

        // Column-sharded: split qweight, scales, qzeros along N
        let mut shard_outputs = Vec::new();
        for rank in 0..world_size {
            let n_shard = n / world_size;
            let n_start = rank * n_shard;

            // qweight [K/8, N] int32 → slice columns
            let qw_s = slice_columns(&qw, k / 8, n, n_start, n_shard, 4);
            // scales [num_groups, N] f16 → slice columns
            let sc_s = slice_columns(&sc, k / group_size, n, n_start, n_shard, 2);
            // qzeros [num_groups, N/8] int32 → slice columns
            let qz_s = slice_columns(&qz, k / group_size, n / 8, n_start / 8, n_shard / 8, 4);

            let shard_weight = LinearWeight::Quantized(
                QuantizedTensor::from_gptq_raw(
                    &ctx,
                    &[n_shard, k],
                    DType::GPTQ_INT4,
                    &qw_s,
                    &sc_s,
                    &qz_s,
                    group_size,
                )
                .unwrap(),
            );
            let shard_out = linear(&input, &shard_weight).unwrap().to_vec().unwrap();
            assert_eq!(shard_out.len(), n_shard);
            shard_outputs.extend(shard_out);
        }

        // Concatenated shard outputs should match full output
        assert_eq!(shard_outputs.len(), full_output.len());
        for (i, (a, b)) in full_output.iter().zip(shard_outputs.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "Column shard mismatch at {i}: full={a} vs sharded={b}"
            );
        }
    }

    #[test]
    fn test_linear_gptq_row_shard() {
        // Row-parallel: split input dimension K into 2 shards, verify
        // that summing shard partial outputs matches the full output.
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

        let k = 64; // must be divisible by group_size * world_size
        let n = 8;
        let group_size = 32;
        let world_size = 2;

        let w_data: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.01) - 0.5).collect();
        let (qw, sc, qz) = pack_gptq_test(&w_data, n, k, group_size);

        // Full (non-sharded) output
        let full_weight = LinearWeight::Quantized(
            QuantizedTensor::from_gptq_raw(
                &ctx,
                &[n, k],
                DType::GPTQ_INT4,
                &qw,
                &sc,
                &qz,
                group_size,
            )
            .unwrap(),
        );
        let input_data: Vec<f32> = (0..k).map(|i| i as f32 * 0.1).collect();
        let input = CudaTensor::from_slice(&ctx, &[1, k], &input_data).unwrap();
        let full_output = linear(&input, &full_weight).unwrap().to_vec().unwrap();

        // Row-sharded: split qweight, scales, qzeros along K
        let mut summed = vec![0.0_f32; n];
        for rank in 0..world_size {
            let k_shard = k / world_size;
            let k_start = rank * k_shard;

            // qweight [K/8, N] int32 → slice rows
            let qw_s = slice_rows(&qw, n, k_start / 8, k_shard / 8, 4);
            // scales [num_groups, N] f16 → slice rows
            let g_start = k_start / group_size;
            let g_shard = k_shard / group_size;
            let sc_s = slice_rows(&sc, n, g_start, g_shard, 2);
            // qzeros [num_groups, N/8] int32 → slice rows
            let qz_s = slice_rows(&qz, n / 8, g_start, g_shard, 4);

            let shard_weight = LinearWeight::Quantized(
                QuantizedTensor::from_gptq_raw(
                    &ctx,
                    &[n, k_shard],
                    DType::GPTQ_INT4,
                    &qw_s,
                    &sc_s,
                    &qz_s,
                    group_size,
                )
                .unwrap(),
            );

            // Input shard: columns [k_start..k_start+k_shard]
            let input_shard_data: Vec<f32> = input_data[k_start..k_start + k_shard].to_vec();
            let input_shard =
                CudaTensor::from_slice(&ctx, &[1, k_shard], &input_shard_data).unwrap();
            let shard_out = linear(&input_shard, &shard_weight)
                .unwrap()
                .to_vec()
                .unwrap();

            for (j, v) in shard_out.iter().enumerate() {
                summed[j] += v;
            }
        }

        // Summed shard outputs should match full output
        for (i, (a, b)) in full_output.iter().zip(summed.iter()).enumerate() {
            assert!(
                (a - b).abs() < 0.5,
                "Row shard mismatch at {i}: full={a} vs summed={b}"
            );
        }
    }

    #[test]
    fn test_shard_strategy_column_parallel() {
        let column_names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.5.self_attn.k_proj.weight",
            "model.layers.31.self_attn.v_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            // MoE expert projections: w1 = gate_proj, w3 = up_proj
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.7.w3.weight",
            "model.layers.31.block_sparse_moe.experts.3.w1.weight",
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
            // MoE expert projections: w2 = down_proj
            "model.layers.0.block_sparse_moe.experts.0.w2.weight",
            "model.layers.31.block_sparse_moe.experts.5.w2.weight",
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
            // MoE router gate: replicated on all ranks
            "model.layers.0.block_sparse_moe.gate.weight",
        ];
        for name in &replicate_names {
            assert!(
                matches!(shard_strategy_for_weight(name), ShardStrategy::Replicate),
                "{name} should be Replicate"
            );
        }
    }
}
