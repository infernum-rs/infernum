//! CUDA graph-mode inference engine generic over model configuration.
//!
//! Provides [`CudaGraphEngine<C>`] backed by a paged KV cache
//! ([`crate::cuda::PagedKvCache`]), enabling any model family that implements
//! [`CudaGraphEngineConfig`] to run inference on a CUDA GPU using the graph
//! execution path.
//!
//! # Usage
//!
//! Model crates implement [`CudaGraphEngineConfig`] for their config type and
//! expose a convenience constructor (e.g. `LlamaCudaGraphEngine::from_pretrained`).
//! The engine then satisfies `infernum::Model` and can be plugged into
//! `infernum_runtime::Runtime`.
//!
//! # Limitations
//!
//! - `MoE` models (Qwen3-MoE, Mixtral) require the CUDA `MoE` dispatch arms to be
//!   present in the executor (added in Step 6).
//! - `DeepSeek` (MLA) is not supported via this generic path; it remains a separate
//!   implementation.

use std::cell::RefCell;
use std::path::Path;
use std::sync::Arc;

use infernum::block_allocator::{BlockConfig, BlockTable};
use infernum::dtype::Q4K_BLOCK_ELEMENTS;
use infernum::graph::{optimizer, plan, ExecutionPlan, Graph, NodeId, WeightId, WeightStore};
use infernum::shard::{shard_strategy_for_weight, ShardConfig, ShardStrategy};
use infernum::weights::QuantizationConfig;
use infernum::WeightLoader as _;
use infernum::{
    precompute_rope_data, precompute_rope_row, DType, ModelConfig, Result, QUANTIZATION_BLOCK_SIZE,
};

use half::bf16;

use super::executor::execute;
use crate::cuda::ops::{argmax_last_tensor, cast_to_f32, LinearWeight};
use crate::cuda::{CudaContext, CudaEvent, CudaGraph, CudaTensor, PinnedBuffer};
use crate::cuda_logits::CudaLogits;
use crate::inner::execute_context::GraphInputs;
use crate::weights::{CudaWeightLoader, SafeTensorsLoader, WeightLoader};
use crate::CudaBackend;

// ---------------------------------------------------------------------------
// CudaGraphEngineConfig
// ---------------------------------------------------------------------------

/// Configuration trait for CUDA-backed graph engines.
///
/// Model crates implement this for their `*Config` type. It provides:
/// - Scalar config getters (matching [`infernum_cpu::GraphEngineConfig`]).
/// - Methods to build `Graph<CudaBackend>` for prefill and decode.
/// - A method to load weights from a `SafeTensors` directory into a CUDA
///   [`WeightStore`].
pub trait CudaGraphEngineConfig: Send + Sync + 'static {
    fn num_hidden_layers(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
    fn max_position_embeddings(&self) -> usize;
    fn rope_theta(&self) -> f32;
    fn vocab_size(&self) -> usize;
    fn eos_token_id(&self) -> u32;

    /// Optional quantization config for GPTQ/AWQ/FP8 models.
    ///
    /// When `Some`, `load_graph_weights_cuda` passes this to `load_linear`
    /// so that quantized weight variants (`qweight`/`qzeros`/`scales` for
    /// GPTQ, or raw FP8 tensors with scales) are loaded correctly.
    ///
    /// Default implementation returns `None` (dense/unquantized models).
    fn quantization_config(&self) -> Option<&QuantizationConfig> {
        None
    }

    /// Build a prefill-mode graph for the given sequence length.
    fn build_prefill_graph_cuda(
        &self,
        seq_len: usize,
        shard: Option<&ShardConfig>,
    ) -> Graph<CudaBackend>;

    /// Build a prefill-mode graph that also outputs K and V for every layer.
    ///
    /// Returns `None` if batch-GEMM prefill with KV export is not supported
    /// for this model. In that case, [`CudaGraphEngine`] falls back to
    /// token-by-token paged-decode prefill.
    ///
    /// When `Some`, the returned graph has outputs:
    /// `[logits, k_0, v_0, k_1, v_1, ..., k_{n-1}, v_{n-1}]`
    /// where each `k_i`/`v_i` has shape `[seq_len, num_kv_heads_local, head_dim]`.
    fn build_prefill_graph_with_kv_cuda(
        &self,
        seq_len: usize,
        shard: Option<&ShardConfig>,
    ) -> Option<Graph<CudaBackend>> {
        let _ = (seq_len, shard);
        None
    }

    /// Build a decode-mode graph for a KV cache of the given length.
    fn build_decode_graph_cuda(
        &self,
        kv_len: usize,
        shard: Option<&ShardConfig>,
    ) -> Graph<CudaBackend>;

    /// Build a batched paged-KV decode graph for the given batch size and block config.
    fn build_paged_decode_graph_cuda(
        &self,
        batch_size: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
        shard: Option<&ShardConfig>,
    ) -> Graph<CudaBackend>;

    /// Load all model weights from a `SafeTensors` directory into CUDA memory.
    ///
    /// The `dummy_graph` (a prefill graph built with a small `seq_len`) is
    /// used to enumerate weight metadata (names, shapes, dtypes). The weights
    /// are then loaded from disk and uploaded to the provided CUDA context.
    ///
    /// When `shard` is `Some`, each linear weight is sliced to the rank's
    /// portion before uploading; tensor weights (norms, embeddings) are
    /// replicated on all ranks.
    ///
    /// # Errors
    ///
    /// Returns an error if any weight file is missing or cannot be read.
    fn load_weights_cuda_safetensors(
        &self,
        dummy_graph: &Graph<CudaBackend>,
        ctx: &CudaContext,
        model_dir: &Path,
        shard: Option<&ShardConfig>,
    ) -> Result<WeightStore<CudaTensor, LinearWeight>>;

    /// Load all model weights from a GGUF file into CUDA memory.
    ///
    /// Quantized weights (`Q8_0`, `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`) are kept in their quantized form
    /// on the GPU; non-quantized weights are dequantized to BF16. Q and K
    /// projection weights are un-permuted from GGUF's interleaved `RoPE` layout
    /// to the `HuggingFace` sequential layout before upload.
    ///
    /// When `shard` is `Some`, each linear weight is sliced at block boundaries
    /// to this rank's portion before uploading (enabling GGUF tensor-parallel).
    ///
    /// Returns `Err` by default; override for each model family.
    ///
    /// # Errors
    ///
    /// Returns an error if GGUF loading is not implemented for this config,
    /// or if any weight is missing or cannot be uploaded.
    fn load_weights_cuda_gguf(
        &self,
        dummy_graph: &Graph<CudaBackend>,
        ctx: &CudaContext,
        gguf_path: &Path,
        shard: Option<&ShardConfig>,
    ) -> Result<WeightStore<CudaTensor, LinearWeight>> {
        let _ = (dummy_graph, ctx, gguf_path, shard);
        Err(infernum::Error::UnsupportedModel(
            "GGUF loading is not yet supported by the CUDA graph engine for this model".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Generic CUDA weight loader helper
// ---------------------------------------------------------------------------

/// Load all graph weights from a `SafeTensors` directory into CUDA memory.
///
/// Iterates the graph's weight metadata (tensor weights and linear weights),
/// loads each from disk via a [`CudaWeightLoader`], and returns a populated
/// [`WeightStore`].
///
/// The `lm_head_fallback` parameter, when `true`, silently falls back to
/// loading `model.embed_tokens.weight` when `lm_head.weight` is absent
/// (tied-embedding models such as `SmolLM2`).
///
/// The `quant_config` parameter, when `Some`, is forwarded to `load_linear`
/// for each linear weight. This enables GPTQ and AWQ quantized loading (via
/// `qweight`/`qzeros`/`scales`) and FP8 compressed-tensors loading. When
/// `None`, the loader falls back to detecting quantization from the file's
/// dtype (e.g. raw FP8 tensors stored as `F8E4M3`).
///
/// # Panics
///
/// Panics if the number of weights in the graph exceeds `u32::MAX`.
///
/// # Errors
///
/// Returns an error if any weight cannot be found or uploaded to the GPU.
pub fn load_graph_weights_cuda(
    graph: &Graph<CudaBackend>,
    ctx: &CudaContext,
    model_dir: &Path,
    lm_head_fallback: bool,
    quant_config: Option<&QuantizationConfig>,
    shard: Option<&ShardConfig>,
) -> Result<WeightStore<CudaTensor, LinearWeight>> {
    let format_loader = SafeTensorsLoader::from_directory(model_dir)?;
    let loader = CudaWeightLoader::new(ctx.clone(), format_loader);

    let tensor_count = graph.tensor_weight_count();
    let linear_count = graph.linear_weight_count();

    let mut store = WeightStore::with_capacity(tensor_count, linear_count);

    for i in 0..tensor_count {
        let meta = graph.tensor_weight_meta(WeightId::from_index(
            u32::try_from(i).expect("weight count exceeds u32"),
        ));
        // Tensor weights (norms, embeddings, RoPE caches) are always replicated.
        let tensor = loader.load_tensor(&meta.name, meta.dtype)?;
        store.push_tensor_weight(tensor);
    }

    for i in 0..linear_count {
        let meta = graph.linear_weight_meta(WeightId::from_index(
            u32::try_from(i).expect("weight count exceeds u32"),
        ));
        let name = if lm_head_fallback
            && meta.name == "lm_head.weight"
            && !loader.contains("lm_head.weight")
        {
            "model.embed_tokens.weight"
        } else {
            &meta.name
        };
        // Only pass quant_config when the weight is actually quantized.
        // GPTQ/AWQ checkpoints may contain unquantized tensors (e.g.
        // `model.embed_tokens.weight`) alongside quantized projections; we
        // detect this by probing for the `<prefix>.qweight` sibling that GPTQ
        // format always produces.  For FP8 (compressed-tensors) we check the
        // on-disk dtype.
        let effective_quant = quant_config.and_then(|qc| {
            let is_quantized = match qc.quant_method.as_str() {
                "gptq" | "awq" => {
                    let prefix = name.strip_suffix(".weight").unwrap_or(name);
                    loader.contains(&format!("{prefix}.qweight"))
                }
                _ => {
                    // compressed-tensors / fp8: weight exists as a quantized
                    // dtype on disk — let load_linear detect it via file_dtype.
                    false
                }
            };
            if is_quantized {
                Some(qc)
            } else {
                None
            }
        });
        let weight = if let Some(s) = shard {
            let strategy = shard_strategy_for_weight(name);
            loader.load_linear_sharded(name, meta.dtype, effective_quant, s, strategy)?
        } else {
            loader.load_linear(name, meta.dtype, effective_quant)?
        };
        store.push_linear_weight(weight);
    }

    Ok(store)
}

/// Load all graph weights from a GGUF file into CUDA memory.
///
/// Quantized weights (`Q8_0`, `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`) are kept in quantized form on the GPU
/// so the existing quantized GEMV kernels can run decode at native quantized
/// memory bandwidth. Non-quantized weights (BF16/F16/F32) are dequantized to
/// BF16 on the host.
///
/// Q and K projections are un-permuted from GGUF's interleaved `RoPE` layout
/// before upload. When `shard` is `Some`, each linear weight is sliced at
/// block boundaries to the rank's portion (enables GGUF tensor-parallelism).
///
/// `name_mapper` converts `SafeTensors` weight names (as registered in the graph)
/// to GGUF key names. `n_heads` / `n_kv_heads` are used to compute the head
/// dimension for the Q/K unpermutation. `is_qk` identifies which mapped GGUF
/// names require unpermutation. When `lm_head_fallback` is `true`, a missing
/// `output.weight` falls back to `token_embd.weight`.
///
/// # Errors
///
/// Returns an error if the GGUF file cannot be opened, a required tensor is
/// missing, or GPU allocation fails.
///
/// # Panics
///
/// Panics if the number of registered weights exceeds `u32::MAX`.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn load_graph_weights_gguf_cuda(
    graph: &Graph<CudaBackend>,
    ctx: &CudaContext,
    gguf_path: &Path,
    name_mapper: impl Fn(&str) -> String,
    is_qk: impl Fn(&str) -> bool,
    n_heads: usize,
    n_kv_heads: usize,
    lm_head_fallback: bool,
    shard: Option<&ShardConfig>,
) -> Result<WeightStore<CudaTensor, LinearWeight>> {
    use infernum::graph::WeightId;

    let loader = crate::weights::GgufLoader::from_file_or_split(gguf_path)?;

    let tensor_count = graph.tensor_weight_count();
    let linear_count = graph.linear_weight_count();
    let mut store = WeightStore::with_capacity(tensor_count, linear_count);

    // Tensor weights (embeddings, norms, RoPE caches) are always replicated.
    for i in 0..tensor_count {
        let meta = graph.tensor_weight_meta(WeightId::from_index(
            u32::try_from(i).expect("weight count exceeds u32"),
        ));
        let gguf_name = name_mapper(&meta.name);
        let tensor = loader.load_bf16(ctx, &gguf_name)?;
        store.push_tensor_weight(tensor);
    }

    for i in 0..linear_count {
        let meta = graph.linear_weight_meta(WeightId::from_index(
            u32::try_from(i).expect("weight count exceeds u32"),
        ));
        let safetensors_name = &meta.name;

        // CONCAT weights (for QKV fusion): concatenate multiple GGUF tensors.
        // For Q4_0 and Q8_0, preserve the quantised format by concatenating the raw
        // per-block data + scales bytes (rows stack independently, K is shared).
        // For other dtypes, dequantise to BF16 and concatenate.
        if let Some(names_str) = safetensors_name.strip_prefix("CONCAT:") {
            let gguf_names: Vec<String> = names_str
                .split(',')
                .map(|n| name_mapper(n.trim()))
                .collect();

            // Detect the on-disk dtype from the first component.
            let first_dtype = loader.get_dtype(&gguf_names[0])?;

            if matches!(
                first_dtype,
                infernum::dtype::DType::Q4_0 | infernum::dtype::DType::Q8_0
            ) {
                // Native quantised CONCAT: concatenate block data + scales on host,
                // then upload as a single QuantizedTensor.  No dequantisation needed.
                let mut all_data: Vec<u8> = Vec::new();
                let mut all_scales: Vec<u8> = Vec::new();
                let mut total_rows = 0usize;
                let mut cols = 0usize;
                for gname in &gguf_names {
                    let qt = loader.load_quantized(ctx, gname)?;
                    let data = ctx.device().dtoh_sync_copy(&qt.data_slice().slice(..))?;
                    let scales = ctx.device().dtoh_sync_copy(&qt.scales_slice().slice(..))?;
                    total_rows += qt.shape()[0];
                    cols = qt.shape()[1];
                    all_data.extend_from_slice(&data);
                    all_scales.extend_from_slice(&scales);
                }
                let qt_cat = crate::cuda::QuantizedTensor::from_raw(
                    ctx,
                    &[total_rows, cols],
                    first_dtype,
                    &all_data,
                    &all_scales,
                )?;
                store.push_linear_weight(LinearWeight::Quantized(qt_cat));
                continue;
            }

            // BF16 / other dtypes: dequantise each component, concatenate, upload Dense.
            let mut total_rows = 0usize;
            let mut all_bytes: Vec<Vec<u8>> = Vec::new();
            let mut cols = 0usize;
            for gname in &gguf_names {
                let (bytes, shape) = loader.load_bf16_bytes(gname)?;
                total_rows += shape[0];
                cols = shape[1];
                all_bytes.push(bytes);
            }
            let elem = 2usize; // BF16
            let mut cat_bytes = vec![0u8; total_rows * cols * elem];
            let mut offset = 0;
            for bytes in &all_bytes {
                cat_bytes[offset..offset + bytes.len()].copy_from_slice(bytes);
                offset += bytes.len();
            }
            let tensor = CudaTensor::from_raw_bytes(
                ctx,
                &[total_rows, cols],
                infernum::dtype::DType::BF16,
                &cat_bytes,
            )?;
            store.push_linear_weight(LinearWeight::Dense(crate::cuda::ops::transpose_2d(
                &tensor,
            )?));
            continue;
        }

        let gguf_name = name_mapper(safetensors_name);

        // lm_head fallback: GGUF may not have output.weight for tied-embedding models.
        let effective_gguf_name = if lm_head_fallback
            && gguf_name == "output.weight"
            && !loader.contains("output.weight")
        {
            "token_embd.weight".to_string()
        } else {
            gguf_name.clone()
        };

        let strategy = shard_strategy_for_weight(safetensors_name);

        // Expert weights carry a "[N]" suffix (e.g. "blk.0.ffn_gate_exps.weight[42]").
        // The 3D stacked tensor is sliced per-expert directly from the mmap.
        // For Row sharding, if k is not block-aligned (e.g. Q6_K k=1536 at TP=8),
        // fall back to host-dequant → BF16 → element-level shard → GPU transpose.
        if let Some((base_name, expert_idx)) = parse_expert_suffix(&effective_gguf_name) {
            match loader.load_quantized_expert_slice(ctx, &base_name, expert_idx, shard, strategy) {
                Ok(qt) => {
                    store.push_linear_weight(LinearWeight::Quantized(qt));
                }
                Err(infernum::Error::UnsupportedDtype(ref msg))
                    if msg.contains("expert_slice_row_unaligned") =>
                {
                    let (host_bytes, shape) =
                        loader.load_bf16_bytes_expert_slice(&base_name, expert_idx)?;
                    let (rows, cols) = (shape[0], shape[1]);
                    let tensor = match shard {
                        None => CudaTensor::from_raw_bytes(
                            ctx,
                            &[rows, cols],
                            infernum::dtype::DType::BF16,
                            &host_bytes,
                        )?,
                        Some(s) => shard_bf16_slice(ctx, &host_bytes, rows, cols, s, strategy)?,
                    };
                    store.push_linear_weight(LinearWeight::Dense(crate::cuda::ops::transpose_2d(
                        &tensor,
                    )?));
                }
                Err(e) => return Err(e),
            }
            continue;
        }

        let file_dtype = loader.get_dtype(&effective_gguf_name)?;

        // Q8_0 and Q4_0 support GPU quantized sharding, but only when the
        // column count is aligned to block_size × world_size for Row strategy.
        // Q4_K and Q5_K use 256-element super-blocks (same alignment check).
        // Misaligned shapes (e.g. Qwen3-72B at TP=8) fall through to BF16.
        let supports_quant_sharding = match file_dtype {
            DType::Q8_0 | DType::Q4_0 => shard.is_none_or(|s| {
                strategy != ShardStrategy::Row || {
                    loader
                        .get_shape(&effective_gguf_name)
                        .ok()
                        .and_then(|shape| shape.get(1).copied())
                        .is_some_and(|n_cols| {
                            (n_cols / QUANTIZATION_BLOCK_SIZE).is_multiple_of(s.world_size)
                        })
                }
            }),
            DType::Q4_K | DType::Q5_K => shard.is_none_or(|s| {
                strategy != ShardStrategy::Row || {
                    loader
                        .get_shape(&effective_gguf_name)
                        .ok()
                        .and_then(|shape| shape.get(1).copied())
                        .is_some_and(|n_cols| {
                            (n_cols / Q4K_BLOCK_ELEMENTS).is_multiple_of(s.world_size)
                        })
                }
            }),
            _ => false,
        };
        let use_gpu_quant =
            file_dtype.has_gpu_quant_kernel() && (shard.is_none() || supports_quant_sharding);

        let weight = if use_gpu_quant {
            // Keep quantized on GPU — GEMV kernels dequantize on the fly.
            // No transpose needed: quantized matmul kernels use the native
            // [out, in] layout, unlike dense weights which are transposed.
            if is_qk(&gguf_name) {
                let n_head = if gguf_name.contains("attn_q") {
                    n_heads
                } else {
                    n_kv_heads
                };
                let qt = match shard {
                    None => loader.load_quantized_unpermute(ctx, &effective_gguf_name, n_head)?,
                    Some(s) => loader.load_quantized_unpermute_sharded(
                        ctx,
                        &effective_gguf_name,
                        n_head,
                        s,
                        strategy,
                    )?,
                };
                LinearWeight::Quantized(qt)
            } else {
                let qt = match shard {
                    None => loader.load_quantized(ctx, &effective_gguf_name)?,
                    Some(s) => {
                        loader.load_quantized_sharded(ctx, &effective_gguf_name, s, strategy)?
                    }
                };
                LinearWeight::Quantized(qt)
            }
        } else {
            // Non-quantized or unsupported quant (Q6_K+TP, etc.):
            // dequantize to BF16 on host, shard host bytes, upload once, GPU-transpose.
            let (host_bytes, shape) = if is_qk(&gguf_name) {
                let n_head = if gguf_name.contains("attn_q") {
                    n_heads
                } else {
                    n_kv_heads
                };
                loader.load_bf16_bytes_unpermute(&effective_gguf_name, n_head)?
            } else {
                loader.load_bf16_bytes(&effective_gguf_name)?
            };
            let (rows, cols) = (shape[0], shape[1]);
            let tensor = match shard {
                None => CudaTensor::from_raw_bytes(
                    ctx,
                    &[rows, cols],
                    infernum::dtype::DType::BF16,
                    &host_bytes,
                )?,
                Some(s) => shard_bf16_slice(ctx, &host_bytes, rows, cols, s, strategy)?,
            };
            // GPU transpose: (rows, cols) → (cols, rows)
            LinearWeight::Dense(crate::cuda::ops::transpose_2d(&tensor)?)
        };

        store.push_linear_weight(weight);
    }

    Ok(store)
}

/// Parse a `[N]` expert-index suffix from a GGUF weight name.
///
/// Returns `(base_name, expert_idx)` when the name ends with `[N]`, or `None`
/// for ordinary (non-expert) names.
fn parse_expert_suffix(name: &str) -> Option<(String, usize)> {
    let name = name.strip_suffix(']')?;
    let bracket = name.rfind('[')?;
    let idx: usize = name[bracket + 1..].parse().ok()?;
    Some((name[..bracket].to_string(), idx))
}

/// Shard a 2D BF16 slice on the host (slice, then upload).
///
/// Takes raw BF16 bytes (not yet on GPU) to avoid an unnecessary upload/download cycle.
fn shard_bf16_slice(
    ctx: &CudaContext,
    data: &[u8],
    rows: usize,
    cols: usize,
    shard: &ShardConfig,
    strategy: ShardStrategy,
) -> Result<CudaTensor> {
    let elem = 2usize; // bf16
    match strategy {
        ShardStrategy::Replicate => {
            CudaTensor::from_raw_bytes(ctx, &[rows, cols], infernum::dtype::DType::BF16, data)
        }
        ShardStrategy::Column => {
            let (start_row, shard_rows) = shard.shard_range(rows);
            let row_bytes = cols * elem;
            let start = start_row * row_bytes;
            let end = start + shard_rows * row_bytes;
            CudaTensor::from_raw_bytes(
                ctx,
                &[shard_rows, cols],
                infernum::dtype::DType::BF16,
                &data[start..end],
            )
        }
        ShardStrategy::Row => {
            let (start_col, shard_cols) = shard.shard_range(cols);
            let mut shard_data = vec![0u8; rows * shard_cols * elem];
            for r in 0..rows {
                let src_start = (r * cols + start_col) * elem;
                let dst_start = r * shard_cols * elem;
                let chunk = shard_cols * elem;
                shard_data[dst_start..dst_start + chunk]
                    .copy_from_slice(&data[src_start..src_start + chunk]);
            }
            CudaTensor::from_raw_bytes(
                ctx,
                &[rows, shard_cols],
                infernum::dtype::DType::BF16,
                &shard_data,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// DecodeState — cached CUDA-graph decode state for batch_size == 1
// ---------------------------------------------------------------------------

/// Cached CUDA-graph state for the single-token decode path.
///
/// Allocated lazily on the first `forward_batch_decode` call with
/// `batch_size == 1` and rebuilt whenever `max_blocks_per_seq` changes.
/// The external [`crate::cuda::PagedKvCache`] is still passed through to
/// `execute()` on every step, so KV continuity from prefill is preserved.
struct DecodeState {
    /// Compiled graph executable.
    cuda_graph: CudaGraph,
    /// Pre-allocated stable GPU-side input buffers (token, cos, sin, block
    /// table, positions, `seq_lens`). Updated via `htod_copy_into` before each
    /// step; the captured graph references their fixed device addresses.
    graph_inputs: GraphInputs,
    /// Topological execution plan (reused across steps).
    plan: ExecutionPlan,
    /// The optimised paged-decode graph (`batch_size` = 1).
    graph: Graph<CudaBackend>,
    /// Output node of the graph (argmax token, `[1]` U32).
    output_node: NodeId,
    /// `max_blocks_per_seq` the graph was built for.
    max_blocks_per_seq: usize,
    /// Pool-miss count at the end of the previous step. When it equals the
    /// current count the pool has stabilised and the captured graph is reusable
    /// without re-capture.
    last_miss_count: u64,
    /// `true` once the pool has stabilised and bare `launch()` suffices.
    stabilized: bool,
    /// `true` when the graph contains ops that are incompatible with CUDA
    /// graph capture (e.g. `MoE` routing requires a D→H sync copy for
    /// host-side top-K selection). When set, `begin_capture`/`end_capture`
    /// are never called and the engine always runs eagerly.
    capture_unsafe: bool,
    /// Token output tensor from the last capture (stable pool-backed GPU
    /// address). On the fast path the CUDA graph writes into it; we copy 4
    /// bytes out via the pinned buffer.
    saved_token: Option<CudaTensor>,
    /// Pinned host buffer for async D→H token readback (1 `u32`).
    pinned_token: PinnedBuffer,
    /// CUDA event recorded after each async copy. `synchronize()` waits only
    /// for the DMA rather than flushing the entire device.
    completion_event: CudaEvent,
}

// SAFETY: `DecodeState` is only ever accessed from the single thread that runs
// the runtime. `CudaGraph` and `CudaEvent` hold raw CUDA handles that do not
// auto-implement `Send`; `DecodeState` owns them exclusively and is never
// aliased across threads.
unsafe impl Send for DecodeState {}

// SAFETY: `CudaGraphEngine` is always accessed from a single thread — either
// the caller's thread (single-GPU) or a dedicated scoped thread per engine
// (multi-GPU via `ShardedGraphEngine`). The `RefCell<decode_state>` field is
// therefore never accessed concurrently; `!Sync` is an over-approximation.
unsafe impl<C: CudaGraphEngineConfig> Sync for CudaGraphEngine<C> {}

impl DecodeState {
    /// Build a new `DecodeState` for a paged-decode graph with `batch_size = 1`.
    ///
    /// # Errors
    ///
    /// Returns an error if graph planning, buffer allocation, or CUDA API
    /// calls fail.
    fn new(
        ctx: &CudaContext,
        graph: Graph<CudaBackend>,
        half_dim: usize,
        max_blocks_per_seq: usize,
    ) -> Result<Self> {
        let mut graph = graph;
        // Check for MoE ops before the optimizer consumes the graph.
        // MoE routing (`moe_route`) calls `logits_to_f32_host` (a D→H sync
        // copy) which is illegal inside a CUDA stream capture window on T4
        // (`CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 0`). For these
        // models we skip capture entirely and always run eagerly.
        let capture_unsafe = graph.has_capture_unsafe_ops();
        optimizer::optimize(&mut graph);
        let plan = plan(&graph);

        let output_node = *graph
            .output_ids()
            .first()
            .expect("paged decode graph must have at least one output");

        let cuda_graph = CudaGraph::new(ctx.device())?;
        // max_seq_len is set each step in run_decode_captured before capture begins.
        let graph_inputs = GraphInputs::new(ctx.device(), 1, half_dim, max_blocks_per_seq, 0)?;
        let pinned_token = PinnedBuffer::new(ctx.device())?;
        let completion_event = CudaEvent::new(ctx.device())?;

        Ok(Self {
            cuda_graph,
            graph_inputs,
            plan,
            graph,
            output_node,
            max_blocks_per_seq,
            last_miss_count: 0,
            stabilized: false,
            capture_unsafe,
            saved_token: None,
            pinned_token,
            completion_event,
        })
    }
}

// ---------------------------------------------------------------------------
// CudaGraphEngine
// ---------------------------------------------------------------------------

/// CUDA graph-mode engine for any model family implementing [`CudaGraphEngineConfig`].
///
/// Loads weights once and satisfies [`infernum::Model`], allowing it to be used
/// with `infernum_runtime::Runtime`.
///
/// For `batch_size == 1` decode steps the engine uses CUDA graph capture via
/// [`DecodeState`]: the first few steps re-capture the graph until the buffer
/// pool stabilises, then subsequent steps replay with a single `launch()` call.
/// For `batch_size > 1` (rare in practice) the engine falls back to the eager
/// `execute()` path.
///
/// Prefill always uses the eager path (variable sequence length makes capture
/// impractical).
pub struct CudaGraphEngine<C: CudaGraphEngineConfig> {
    config: C,
    ctx: CudaContext,
    weights: Arc<WeightStore<CudaTensor, LinearWeight>>,
    /// Cached `head_dim / 2` for `RoPE` slice arithmetic.
    half_dim: usize,
    /// Cached CUDA-graph decode state for the `batch_size == 1` fast path.
    ///
    /// `RefCell` provides interior mutability — the [`infernum::Model`] trait
    /// requires `&self` on all forward methods, but graph capture needs
    /// mutation. `RefCell` is safe here because the `Runtime` calls these
    /// methods single-threadedly.
    decode_state: RefCell<Option<Box<DecodeState>>>,
    /// Tensor-parallel shard config. `None` for single-GPU inference.
    shard: Option<ShardConfig>,
    /// NCCL communicator for tensor-parallel all-reduce.
    #[cfg(feature = "nccl")]
    comm: Option<crate::cuda::NcclCommunicator>,
}

impl<C: CudaGraphEngineConfig> CudaGraphEngine<C> {
    /// Load a model from a `SafeTensors` directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or the config file cannot be parsed.
    pub fn from_config_and_dir(config: C, ctx: CudaContext, model_dir: &Path) -> Result<Self> {
        let dummy_graph = config.build_prefill_graph_cuda(1, None);
        let weights = config.load_weights_cuda_safetensors(&dummy_graph, &ctx, model_dir, None)?;
        let half_dim = config.head_dim() / 2;
        Ok(Self {
            config,
            ctx,
            weights: Arc::new(weights),
            half_dim,
            decode_state: RefCell::new(None),
            shard: None,
            #[cfg(feature = "nccl")]
            comm: None,
        })
    }

    /// Load a model from a GGUF file, dequantizing weights to BF16.
    ///
    /// # Errors
    ///
    /// Returns an error if GGUF loading is not supported for this config,
    /// or if the file cannot be opened or weights cannot be uploaded.
    pub fn from_config_gguf(config: C, ctx: CudaContext, gguf_path: &Path) -> Result<Self> {
        let dummy_graph = config.build_prefill_graph_cuda(1, None);
        let weights = config.load_weights_cuda_gguf(&dummy_graph, &ctx, gguf_path, None)?;
        let half_dim = config.head_dim() / 2;
        Ok(Self {
            config,
            ctx,
            weights: Arc::new(weights),
            half_dim,
            decode_state: RefCell::new(None),
            shard: None,
            #[cfg(feature = "nccl")]
            comm: None,
        })
    }

    /// Load a tensor-parallel shard from a `SafeTensors` directory.
    ///
    /// Each rank loads only its slice of the column-parallel and row-parallel
    /// weights. The provided `comm` is used for all-reduce synchronisation
    /// at the `AllReduceSumOp` nodes injected by the graph builder.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing or weights cannot be loaded.
    #[cfg(feature = "nccl")]
    pub fn from_config_comm_and_dir(
        config: C,
        ctx: CudaContext,
        comm: crate::cuda::NcclCommunicator,
        shard: ShardConfig,
        model_dir: &Path,
    ) -> Result<Self> {
        let dummy_graph = config.build_prefill_graph_cuda(1, Some(&shard));
        let weights =
            config.load_weights_cuda_safetensors(&dummy_graph, &ctx, model_dir, Some(&shard))?;
        let half_dim = config.head_dim() / 2;
        Ok(Self {
            config,
            ctx,
            weights: Arc::new(weights),
            half_dim,
            decode_state: RefCell::new(None),
            shard: Some(shard),
            comm: Some(comm),
        })
    }

    /// Load a tensor-parallel shard from a GGUF file.
    ///
    /// Each rank loads only its slice of the column-parallel and row-parallel
    /// weights (sliced at block boundaries in the GGUF loader). Quantized
    /// weights stay quantized on GPU; non-quantized weights are dequantized
    /// to BF16. The provided `comm` is used for all-reduce at `AllReduceSumOp`
    /// nodes injected by the graph builder.
    ///
    /// # Errors
    ///
    /// Returns an error if the GGUF file is missing, metadata cannot be
    /// parsed, or weights cannot be uploaded.
    #[cfg(feature = "nccl")]
    pub fn from_config_comm_and_gguf(
        config: C,
        ctx: CudaContext,
        comm: crate::cuda::NcclCommunicator,
        shard: ShardConfig,
        gguf_path: &Path,
    ) -> Result<Self> {
        let dummy_graph = config.build_prefill_graph_cuda(1, Some(&shard));
        let weights = config.load_weights_cuda_gguf(&dummy_graph, &ctx, gguf_path, Some(&shard))?;
        let half_dim = config.head_dim() / 2;
        Ok(Self {
            config,
            ctx,
            weights: Arc::new(weights),
            half_dim,
            decode_state: RefCell::new(None),
            shard: Some(shard),
            comm: Some(comm),
        })
    }

    /// Return a reference to the model configuration.
    #[must_use]
    pub fn config(&self) -> &C {
        &self.config
    }

    /// Return a reference to the CUDA context.
    #[must_use]
    pub fn cuda_context(&self) -> &CudaContext {
        &self.ctx
    }

    /// Return a reference to the NCCL communicator, if any.
    #[cfg(feature = "nccl")]
    fn comm_ref(&self) -> Option<&crate::cuda::NcclCommunicator> {
        self.comm.as_ref()
    }

    #[cfg(not(feature = "nccl"))]
    #[allow(clippy::unused_self)]
    fn comm_ref(&self) -> Option<&()> {
        None
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn run_prefill_graph(&self, input_ids: &[u32]) -> Result<Vec<CudaTensor>> {
        let seq_len = input_ids.len();
        let head_dim = self.config.head_dim();
        let half_dim = self.half_dim;

        let mut graph = self
            .config
            .build_prefill_graph_cuda(seq_len, self.shard.as_ref());
        optimizer::optimize(&mut graph);
        let ep = plan(&graph);

        let (cos_data, sin_data) =
            precompute_rope_data(seq_len, head_dim, self.config.rope_theta());
        let cos_bf16: Vec<bf16> = cos_data.iter().map(|&x| bf16::from_f32(x)).collect();
        let sin_bf16: Vec<bf16> = sin_data.iter().map(|&x| bf16::from_f32(x)).collect();
        let input_ids_t = CudaTensor::from_slice(&self.ctx, &[seq_len], input_ids)?;
        let cos_t = CudaTensor::from_slice(&self.ctx, &[seq_len, half_dim], &cos_bf16)?;
        let sin_t = CudaTensor::from_slice(&self.ctx, &[seq_len, half_dim], &sin_bf16)?;
        let inputs = vec![input_ids_t, cos_t, sin_t];

        let output_nodes = graph.output_ids().to_vec();
        let (outputs, _) = execute(
            &self.ctx,
            &ep,
            graph.nodes(),
            &self.weights,
            &inputs,
            &output_nodes,
            None,
            None,
            0,
            None,
            self.comm_ref(),
            0, // no paged KV cache in prefill-graph mode
        )?;
        Ok(outputs)
    }

    /// Run one decode step using the cached CUDA-graph path (`batch_size` == 1).
    ///
    /// Writes the dynamic inputs into the stable `GraphInputs` GPU buffers,
    /// then either re-captures the graph (stabilisation phase) or replays it
    /// (fast path). Returns the raw logit tensor (BF16).
    ///
    /// The `kv_cache` is the externally-managed paged KV cache shared between
    /// prefill and decode; it is passed directly to `execute()` so KV
    /// continuity from the prefill step is preserved.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer writes, kernel launches, or CUDA API calls
    /// fail.
    /// Run one decode step via CUDA graph replay (stabilised) or capture (warm-up).
    ///
    /// Returns `(logits_tensor, Some(argmax_token))` in the stabilised fast
    /// path — the argmax is computed via a GPU reduction + async DToH inside
    /// the event-sync window, so the caller can skip a second GPU round-trip.
    /// Returns `(logits_tensor, None)` during the non-stabilised capture phase.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn run_decode_captured(
        &self,
        token_ids_host: &[u32],
        kv_cache: &mut crate::cuda::PagedKvCache,
        block_table_u32: &[u32],
        positions_u32: &[u32],
        seq_lens_u32: &[u32],
        cos_data: &[f32],
        sin_data: &[f32],
        max_blocks_per_seq: usize,
    ) -> Result<(CudaTensor, Option<u32>)> {
        let mut borrow = self.decode_state.borrow_mut();

        // (Re-)build DecodeState when absent or when max_blocks changed.
        let needs_rebuild = borrow
            .as_ref()
            .is_none_or(|s| s.max_blocks_per_seq != max_blocks_per_seq);
        if needs_rebuild {
            let block_size = kv_cache.block_size();
            let graph = self.config.build_paged_decode_graph_cuda(
                1,
                block_size,
                max_blocks_per_seq,
                self.shard.as_ref(),
            );
            *borrow = Some(Box::new(DecodeState::new(
                &self.ctx,
                graph,
                self.half_dim,
                max_blocks_per_seq,
            )?));
        }

        let state = borrow.as_mut().expect("DecodeState was just built");

        // Write all 6 dynamic inputs into the stable GPU buffers.
        let device = self.ctx.device();
        // token_ids: re-upload the pre-downloaded host values into the stable GPU buffer.
        device.htod_copy_into(token_ids_host.to_vec(), &mut state.graph_inputs.token_ids)?;
        // Convert cos/sin to BF16 before upload — the graph inputs are BF16 so that
        // `apply_rope` can use them directly without a per-layer F32→BF16 cast kernel.
        let cos_bf16: Vec<bf16> = cos_data.iter().map(|&x| bf16::from_f32(x)).collect();
        let sin_bf16: Vec<bf16> = sin_data.iter().map(|&x| bf16::from_f32(x)).collect();
        device.htod_copy_into(cos_bf16, &mut state.graph_inputs.cos)?;
        device.htod_copy_into(sin_bf16, &mut state.graph_inputs.sin)?;
        device.htod_copy_into(
            block_table_u32.to_vec(),
            &mut state.graph_inputs.block_table,
        )?;
        device.htod_copy_into(positions_u32.to_vec(), &mut state.graph_inputs.positions)?;
        device.htod_copy_into(seq_lens_u32.to_vec(), &mut state.graph_inputs.seq_lens)?;
        // Compute max_seq_len on the host now (before capture) so that
        // CudaPagedKvCacheAccess can use it without a D→H copy during capture.
        state.graph_inputs.max_seq_len = seq_lens_u32
            .iter()
            .copied()
            .map(|s| s as usize)
            .max()
            .unwrap_or(1);

        if state.stabilized {
            // Fast path: graph is stable — replay with a single launch, compute
            // argmax on the logit output, DMA the 4-byte result to the pinned
            // buffer, then wait only for that event rather than the whole device.
            state.cuda_graph.launch()?;

            // Kick off the argmax kernel right after the graph launch so it
            // runs as soon as the graph finishes on the GPU stream.
            // Keep `argmax_out` alive past event.synchronize() — the DMA reads
            // from its device address and must complete before the buffer drops.
            let saved = state
                .saved_token
                .as_ref()
                .expect("saved_token must be set before stabilization");
            let argmax_out = argmax_last_tensor(saved)?;

            state
                .pinned_token
                .async_copy_from_device(argmax_out.device_ptr())?;
            state.completion_event.record()?;
            state.completion_event.synchronize()?;
            // `argmax_out` drops here — after the DMA completes.

            let token = state.pinned_token.read();
            let logits = saved.clone();
            Ok((logits, Some(token)))
        } else {
            // Stabilisation path: wrap execute() in begin/end capture so the
            // CUDA buffer-pool allocates the right sizes and the graph exe is
            // built. Re-capture until the pool stops growing.
            //
            // Exception: capture-unsafe models (e.g. MoE) perform D→H sync
            // copies during routing which are illegal inside a capture window.
            // For these we skip capture entirely and run eagerly every step.

            // Temporarily swap out the GraphInputs (execute() consumes it).
            let dummy = GraphInputs::new(
                device,
                state.graph_inputs.batch_size,
                state.graph_inputs.half_dim,
                state.graph_inputs.max_blocks_per_seq,
                0, // placeholder; real_inputs carries the actual max_seq_len
            )?;
            let real_inputs = std::mem::replace(&mut state.graph_inputs, dummy);

            if !state.capture_unsafe {
                state.cuda_graph.begin_capture()?;
            }

            let output_nodes = [state.output_node];
            let (mut outputs, returned_inputs) = execute(
                &self.ctx,
                &state.plan,
                state.graph.nodes(),
                &self.weights,
                &[], // inputs come from graph_inputs
                &output_nodes,
                None,              // no MLA KV cache
                Some(kv_cache),    // external paged KV cache
                0,                 // mla_seq_pos
                Some(real_inputs), // stable GPU input buffers
                self.comm_ref(),
                0, // graph_inputs carries max_seq_len; no eager fallback needed
            )?;

            if !state.capture_unsafe {
                state.cuda_graph.end_capture()?;
                state.cuda_graph.launch()?;
            }
            self.ctx.synchronize()?;

            // Restore graph_inputs from the value returned by execute().
            // The captured CUDA graph holds references to the GPU buffer addresses
            // inside real_inputs; those buffers must remain live for the graph's
            // entire lifetime. Storing them back here (rather than allocating a
            // fresh GraphInputs) keeps those addresses valid across launches.
            // max_seq_len is overwritten at the top of the next run_decode_captured call.
            state.graph_inputs =
                returned_inputs.expect("execute() must return the GraphInputs that was passed in");

            let token_tensor = outputs.pop().expect("execute returned no outputs");
            // Save the pool-backed tensor so the fast path can DMA from it.
            state.saved_token = Some(token_tensor.clone());

            // Track pool misses to detect stabilisation.
            // Capture-unsafe models (e.g. MoE) never stabilise — they always
            // run eagerly without a CUDA graph.
            if !state.capture_unsafe {
                if let Some(pool) = self.ctx.buffer_pool() {
                    let current_misses = pool.misses();
                    if current_misses == state.last_miss_count && state.cuda_graph.is_instantiated()
                    {
                        state.stabilized = true;
                    }
                    state.last_miss_count = current_misses;
                } else {
                    // No pool: switch to bare launch after first successful capture.
                    state.stabilized = state.cuda_graph.is_instantiated();
                }
            }

            Ok((token_tensor, None))
        }
    }

    /// Decode step using pre-downloaded host data, avoiding cross-rank GPU stream sync.
    ///
    /// Called by [`ShardedGraphEngine`] which downloads all rank-0 GPU tensors to host
    /// before spawning rank threads, preventing an NCCL collective-op deadlock: without
    /// this, some rank threads block in `cuStreamSynchronize(GPU-0 stream)` while other
    /// threads have already submitted NCCL `AllReduce`s to that stream, deadlocking the ring.
    #[allow(clippy::too_many_arguments, dead_code)]
    pub(crate) fn forward_batch_decode_precomputed(
        &self,
        token_ids_host: &[u32],
        kv_cache: &mut crate::cuda::PagedKvCache,
        positions_data: &[i32],
        block_table_u32: &[u32],
        batch_size: usize,
        max_blocks_per_seq: usize,
        _max_seq_len: usize,
    ) -> Result<CudaLogits> {
        let block_size = kv_cache.block_size();
        let head_dim = self.config.head_dim();
        let half_dim = self.half_dim;

        let mut cos_data = Vec::with_capacity(batch_size * half_dim);
        let mut sin_data = Vec::with_capacity(batch_size * half_dim);
        for &pos_i32 in positions_data {
            let pos = usize::try_from(pos_i32).expect("position must be non-negative");
            let (c, s) = precompute_rope_row(pos, head_dim, self.config.rope_theta());
            cos_data.extend(c);
            sin_data.extend(s);
        }
        let positions_u32: Vec<u32> = positions_data
            .iter()
            .map(|&p| u32::try_from(p).expect("position must be non-negative"))
            .collect();
        let seq_lens_u32: Vec<u32> = positions_data
            .iter()
            .map(|&p| u32::try_from(p).expect("position must be non-negative") + 1)
            .collect();

        if batch_size == 1 {
            let (logits_tensor, precomputed) = self.run_decode_captured(
                token_ids_host,
                kv_cache,
                block_table_u32,
                &positions_u32,
                &seq_lens_u32,
                &cos_data,
                &sin_data,
                max_blocks_per_seq,
            )?;
            let logits_f32 = cast_to_f32(&logits_tensor)?;
            return Ok(match precomputed {
                Some(token) => CudaLogits::new_with_precomputed_argmax(logits_f32, token),
                None => CudaLogits::new(logits_f32),
            });
        }

        // Batch > 1: create GPU tensors on this rank's own device from host data.
        let token_ids_t = CudaTensor::from_slice(&self.ctx, &[batch_size], token_ids_host)?;
        let mut graph = self.config.build_paged_decode_graph_cuda(
            batch_size,
            block_size,
            max_blocks_per_seq,
            self.shard.as_ref(),
        );
        optimizer::optimize(&mut graph);
        let ep = plan(&graph);

        let cos_bf16: Vec<bf16> = cos_data.iter().map(|&x| bf16::from_f32(x)).collect();
        let sin_bf16: Vec<bf16> = sin_data.iter().map(|&x| bf16::from_f32(x)).collect();
        let cos_t = CudaTensor::from_slice(&self.ctx, &[batch_size, half_dim], &cos_bf16)?;
        let sin_t = CudaTensor::from_slice(&self.ctx, &[batch_size, half_dim], &sin_bf16)?;
        let block_table_t = CudaTensor::from_slice(
            &self.ctx,
            &[batch_size, max_blocks_per_seq],
            block_table_u32,
        )?;
        let positions_t = CudaTensor::from_slice(&self.ctx, &[batch_size], &positions_u32)?;
        let seq_lens_t = CudaTensor::from_slice(&self.ctx, &[batch_size], &seq_lens_u32)?;

        let inputs = vec![
            token_ids_t,
            cos_t,
            sin_t,
            block_table_t,
            positions_t,
            seq_lens_t,
        ];
        let output_nodes = graph.output_ids().to_vec();
        let eager_max = seq_lens_u32
            .iter()
            .copied()
            .map(|s| s as usize)
            .max()
            .unwrap_or(0);
        let (outputs, _) = execute(
            &self.ctx,
            &ep,
            graph.nodes(),
            &self.weights,
            &inputs,
            &output_nodes,
            None,
            Some(kv_cache),
            0,
            None,
            self.comm_ref(),
            eager_max,
        )?;

        let logits_bf16 = outputs.into_iter().next().expect("no outputs");
        let logits_f32 = cast_to_f32(&logits_bf16)?;
        Ok(CudaLogits::new(logits_f32))
    }
}

impl<C: CudaGraphEngineConfig> infernum::Model for CudaGraphEngine<C> {
    type B = CudaBackend;
    type KvCache = crate::cuda::PagedKvCache;

    fn config(&self) -> ModelConfig {
        let c = &self.config;
        ModelConfig {
            num_layers: c.num_hidden_layers(),
            max_seq_len: c.max_position_embeddings(),
            num_kv_heads: c.num_kv_heads(),
            head_dim: c.head_dim(),
            eos_token_id: c.eos_token_id(),
            cache_dtype: DType::BF16,
        }
    }

    fn device(&self) -> &CudaContext {
        &self.ctx
    }

    fn allocate_kv_cache(&self, block_config: &BlockConfig) -> Result<crate::cuda::PagedKvCache> {
        let num_kv_heads = match &self.shard {
            Some(s) => self.config.num_kv_heads() / s.world_size,
            None => self.config.num_kv_heads(),
        };
        crate::cuda::PagedKvCache::new(
            &self.ctx,
            self.config.num_hidden_layers(),
            block_config,
            num_kv_heads,
            self.config.head_dim(),
            DType::BF16,
        )
    }

    /// Full forward pass without KV cache (prefill graph).
    ///
    /// Returns logits of shape `(seq_len, vocab_size)`.
    fn forward(
        &self,
        input_ids: &[u32],
    ) -> Result<<CudaBackend as infernum::backend::Backend>::Logits> {
        let outputs = self.run_prefill_graph(input_ids)?;
        let logits_tensor = outputs
            .into_iter()
            .next()
            .expect("prefill graph has no output");
        let logits_f32 = cast_to_f32(&logits_tensor)?;
        Ok(CudaLogits::new(logits_f32))
    }

    /// Single-sequence prefill with paged KV cache.
    ///
    /// Fast path (when the model supports it): runs a single batch-GEMM forward
    /// pass over all `input_ids` at once, then scatters the K/V tensors for
    /// every layer into the paged pool with `append_paged`. This turns
    /// `seq_len` serial M=1 GEMVs into one M=seq_len GEMM, unlocking
    /// tensor-core throughput and giving 10–50× prefill speedup for large prompts.
    ///
    /// Slow path fallback: iterates token-by-token using the paged decode graph.
    /// Used for models that do not implement `build_prefill_graph_with_kv_cuda`.
    fn forward_prefill(
        &self,
        input_ids: &[u32],
        kv_cache: &mut crate::cuda::PagedKvCache,
        _runtime_state: &mut <CudaBackend as infernum::backend::Backend>::RuntimeState,
        block_table: &BlockTable,
        _start_pos: usize,
    ) -> Result<<CudaBackend as infernum::backend::Backend>::Logits> {
        let seq_len = input_ids.len();
        let head_dim = self.config.head_dim();
        let half_dim = self.half_dim;

        // --- Fast path: batch GEMM + KV scatter ---
        if let Some(mut graph) = self
            .config
            .build_prefill_graph_with_kv_cuda(seq_len, self.shard.as_ref())
        {
            optimizer::optimize(&mut graph);
            let ep = plan(&graph);

            let (cos_data, sin_data) =
                precompute_rope_data(seq_len, head_dim, self.config.rope_theta());
            let cos_bf16: Vec<bf16> = cos_data.iter().map(|&x| bf16::from_f32(x)).collect();
            let sin_bf16: Vec<bf16> = sin_data.iter().map(|&x| bf16::from_f32(x)).collect();
            let input_ids_t = CudaTensor::from_slice(&self.ctx, &[seq_len], input_ids)?;
            let cos_t = CudaTensor::from_slice(&self.ctx, &[seq_len, half_dim], &cos_bf16)?;
            let sin_t = CudaTensor::from_slice(&self.ctx, &[seq_len, half_dim], &sin_bf16)?;
            let inputs = vec![input_ids_t, cos_t, sin_t];

            let output_nodes = graph.output_ids().to_vec();

            let t0 = std::time::Instant::now();
            let (outputs, _) = execute(
                &self.ctx,
                &ep,
                graph.nodes(),
                &self.weights,
                &inputs,
                &output_nodes,
                None,
                None,
                0,
                None,
                self.comm_ref(),
                0,
            )?;
            let cpu_dispatch_ms = t0.elapsed().as_secs_f64() * 1000.0;
            // outputs[0] = logits (last-token)
            // outputs[1 + layer*2]     = k  [seq_len, num_kv_heads_local, head_dim]
            // outputs[1 + layer*2 + 1] = v  [seq_len, num_kv_heads_local, head_dim]
            let num_layers = self.config.num_hidden_layers();
            for layer_idx in 0..num_layers {
                let k = &outputs[1 + layer_idx * 2];
                let v = &outputs[1 + layer_idx * 2 + 1];
                kv_cache.append_paged(layer_idx, block_table, k, v, 0)?;
            }

            self.ctx.synchronize()?;
            let gpu_plus_sync_ms = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[prefill] CPU dispatch: {cpu_dispatch_ms:.1}ms | GPU+sync: {gpu_plus_sync_ms:.1}ms | seq={seq_len}"
            );

            let logits_f32 = cast_to_f32(&outputs[0])?;
            return Ok(CudaLogits::new(logits_f32));
        }

        // --- Slow path: token-by-token paged decode ---
        let block_size = kv_cache.block_size();
        let blocks = block_table.blocks();
        let max_blocks = blocks.len();

        // Convert block IDs (usize) to U32 for the graph inputs.
        let block_ids_u32: Vec<u32> = blocks
            .iter()
            .map(|&b| u32::try_from(b).expect("block ID fits u32"))
            .collect();

        // Build the paged decode graph once — the structure is constant across all
        // prefill tokens (batch_size=1, block_size, max_blocks are all fixed).
        let mut graph = self.config.build_paged_decode_graph_cuda(
            1,
            block_size,
            max_blocks,
            self.shard.as_ref(),
        );
        optimizer::optimize(&mut graph);
        let ep = plan(&graph);
        let output_nodes = graph.output_ids().to_vec();

        // Pre-allocate stable GPU input buffers shared across all prefill tokens.
        // Each step updates them via async htod_copy_into (stream-ordered) instead
        // of allocating fresh tensors with htod_sync_copy. This pipelines CPU input
        // preparation with GPU execution: the CPU never stalls waiting for the GPU
        // between tokens — the stream serialises HtoD→kernel automatically.
        let device = self.ctx.device();
        let mut buf_token = device.alloc_zeros::<u32>(1)?;
        let mut buf_cos = device.alloc_zeros::<bf16>(half_dim)?;
        let mut buf_sin = device.alloc_zeros::<bf16>(half_dim)?;
        let mut buf_block = device.alloc_zeros::<u32>(max_blocks)?;
        let mut buf_pos = device.alloc_zeros::<u32>(1)?;
        let mut buf_seq = device.alloc_zeros::<u32>(1)?;

        let mut last_logits: Option<CudaTensor> = None;

        for (pos, &token) in input_ids.iter().enumerate() {
            let (cos_row, sin_row) = precompute_rope_row(pos, head_dim, self.config.rope_theta());
            let cos_bf16: Vec<bf16> = cos_row.iter().map(|&x| bf16::from_f32(x)).collect();
            let sin_bf16: Vec<bf16> = sin_row.iter().map(|&x| bf16::from_f32(x)).collect();
            let pos_u32 = u32::try_from(pos).expect("position fits u32");

            // Async HtoD into stable buffers — stream-ordered, no CPU sync.
            device.htod_copy_into(vec![token], &mut buf_token)?;
            device.htod_copy_into(cos_bf16, &mut buf_cos)?;
            device.htod_copy_into(sin_bf16, &mut buf_sin)?;
            device.htod_copy_into(block_ids_u32.clone(), &mut buf_block)?;
            device.htod_copy_into(vec![pos_u32], &mut buf_pos)?;
            device.htod_copy_into(vec![pos_u32 + 1], &mut buf_seq)?;

            // Non-owning views over the stable buffers.  The PoolableBuffer is
            // marked non_owning so it never frees the underlying slice on drop;
            // the CudaSlice in buf_* remains the sole owner.
            let input_id_t = CudaTensor::from_cuda_slice_view(&self.ctx, &[1], &buf_token);
            let cos_t = CudaTensor::from_cuda_slice_view(&self.ctx, &[1, half_dim], &buf_cos);
            let sin_t = CudaTensor::from_cuda_slice_view(&self.ctx, &[1, half_dim], &buf_sin);
            let block_table_t =
                CudaTensor::from_cuda_slice_view(&self.ctx, &[1, max_blocks], &buf_block);
            let positions_t = CudaTensor::from_cuda_slice_view(&self.ctx, &[1], &buf_pos);
            let seq_len_t = CudaTensor::from_cuda_slice_view(&self.ctx, &[1], &buf_seq);

            let inputs = vec![
                input_id_t,
                cos_t,
                sin_t,
                block_table_t,
                positions_t,
                seq_len_t,
            ];
            let (outputs, _) = execute(
                &self.ctx,
                &ep,
                graph.nodes(),
                &self.weights,
                &inputs,
                &output_nodes,
                None,
                Some(kv_cache),
                0,
                None,
                self.comm_ref(),
                pos + 1, // seq_len at this prefill step
            )?;
            last_logits = Some(outputs.into_iter().next().expect("no outputs"));
        }

        // Single synchronise after all pipelined steps complete.
        self.ctx.synchronize()?;

        let logits_tensor = last_logits.expect("input_ids must not be empty");
        let logits_f32 = cast_to_f32(&logits_tensor)?;
        Ok(CudaLogits::new(logits_f32))
    }

    /// Batched decode with paged KV cache.
    ///
    /// For `batch_size == 1`, delegates to the CUDA-graph capture/replay path
    /// via [`DecodeState`]. After a stabilisation phase (a few steps where the
    /// buffer pool allocates its working set), each decode step reduces to a
    /// single `cuGraphLaunch` call with near-zero CPU overhead.
    ///
    /// For `batch_size > 1`, falls back to the eager `execute()` path
    /// (building and running a fresh graph each call). In practice the runtime
    /// uses `batch_size` > 1 only briefly; the common steady-state is
    /// `batch_size == 1`.
    ///
    /// K/V continuity from prefill is preserved because the same external
    /// `kv_cache` is passed through to `execute()` on every call.
    #[allow(clippy::too_many_arguments)]
    fn forward_batch_decode(
        &self,
        token_ids: &CudaTensor,
        kv_cache: &mut crate::cuda::PagedKvCache,
        _runtime_state: &mut <CudaBackend as infernum::backend::Backend>::RuntimeState,
        block_tables: &CudaTensor,
        _seq_lens: &CudaTensor,
        positions: &CudaTensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
        _max_seq_len: usize,
    ) -> Result<<CudaBackend as infernum::backend::Backend>::Logits> {
        let block_size = kv_cache.block_size();
        let head_dim = self.config.head_dim();
        let half_dim = self.half_dim;

        // Download positions (I32) to build per-sequence RoPE and seq_lens.
        let positions_bytes = positions.to_raw_bytes()?;
        let positions_data: Vec<i32> = positions_bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // Download block_tables (I32) and reinterpret as U32.
        let block_table_bytes = block_tables.to_raw_bytes()?;
        let block_table_u32: Vec<u32> = block_table_bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]).cast_unsigned())
            .collect();

        // Build batched RoPE: cos/sin for each sequence's current position.
        let mut cos_data = Vec::with_capacity(batch_size * half_dim);
        let mut sin_data = Vec::with_capacity(batch_size * half_dim);
        for &pos_i32 in &positions_data {
            let pos = usize::try_from(pos_i32).expect("position must be non-negative");
            let (c, s) = precompute_rope_row(pos, head_dim, self.config.rope_theta());
            cos_data.extend(c);
            sin_data.extend(s);
        }

        // positions_u32: write index for K/V append.
        let positions_u32: Vec<u32> = positions_data
            .iter()
            .map(|&p| u32::try_from(p).expect("position must be non-negative"))
            .collect();

        // seq_lens_u32: K/V length for attention = positions + 1 (after appending).
        let seq_lens_u32: Vec<u32> = positions_data
            .iter()
            .map(|&p| u32::try_from(p).expect("position must be non-negative") + 1)
            .collect();

        if batch_size == 1 {
            // Single-sequence fast path: use CUDA graph capture/replay.
            // Download token_ids from our own rank's GPU (no cross-rank stream sync).
            let token_ids_host = token_ids.to_vec::<u32>()?;
            let (logits_tensor, precomputed) = self.run_decode_captured(
                &token_ids_host,
                kv_cache,
                &block_table_u32,
                &positions_u32,
                &seq_lens_u32,
                &cos_data,
                &sin_data,
                max_blocks_per_seq,
            )?;
            let logits_f32 = cast_to_f32(&logits_tensor)?;
            return Ok(match precomputed {
                Some(token) => CudaLogits::new_with_precomputed_argmax(logits_f32, token),
                None => CudaLogits::new(logits_f32),
            });
        }

        // Batch size > 1: eager path (builds a fresh graph each call).
        let mut graph = self.config.build_paged_decode_graph_cuda(
            batch_size,
            block_size,
            max_blocks_per_seq,
            self.shard.as_ref(),
        );
        optimizer::optimize(&mut graph);
        let ep = plan(&graph);

        let cos_bf16: Vec<bf16> = cos_data.iter().map(|&x| bf16::from_f32(x)).collect();
        let sin_bf16: Vec<bf16> = sin_data.iter().map(|&x| bf16::from_f32(x)).collect();
        let cos_t = CudaTensor::from_slice(&self.ctx, &[batch_size, half_dim], &cos_bf16)?;
        let sin_t = CudaTensor::from_slice(&self.ctx, &[batch_size, half_dim], &sin_bf16)?;
        let block_table_t = CudaTensor::from_slice(
            &self.ctx,
            &[batch_size, max_blocks_per_seq],
            &block_table_u32,
        )?;
        let positions_t = CudaTensor::from_slice(&self.ctx, &[batch_size], &positions_u32)?;
        let seq_lens_t = CudaTensor::from_slice(&self.ctx, &[batch_size], &seq_lens_u32)?;

        let inputs = vec![
            token_ids.clone(),
            cos_t,
            sin_t,
            block_table_t,
            positions_t,
            seq_lens_t,
        ];
        let output_nodes = graph.output_ids().to_vec();
        let eager_max_fbd = seq_lens_u32
            .iter()
            .copied()
            .map(|s| s as usize)
            .max()
            .unwrap_or(0);
        let (outputs, _) = execute(
            &self.ctx,
            &ep,
            graph.nodes(),
            &self.weights,
            &inputs,
            &output_nodes,
            None,
            Some(kv_cache),
            0,
            None,
            self.comm_ref(),
            eager_max_fbd,
        )?;

        let logits_bf16 = outputs.into_iter().next().expect("no outputs");
        let logits_f32 = cast_to_f32(&logits_bf16)?;
        Ok(CudaLogits::new(logits_f32))
    }
}
