//! CUDA graph-mode inference engine for the `DeepSeek` model family.
//!
//! Provides [`DeepSeekCudaEngine`] which loads weights once and implements
//! [`infernum::Model`] so the runtime can drive generation. Unlike the generic
//! [`infernum_cuda::CudaGraphEngine<C>`] used by Llama/Qwen/Gemma, `DeepSeek`
//! cannot use the paged-KV generic path because it has no paged decode graph.
//!
//! **MLA KV state** is managed via [`MlaKvState`]: a `Vec<Vec<CudaTensor>>`
//! (one inner `Vec` per layer, each entry is one compressed KV tensor) plus a
//! `seq_pos` counter. The executor reads and writes through a raw pointer in
//! [`CudaExecutorState`]; the engine increments `seq_pos` after each step.
//!
//! **Weight loading** reuses `load_graph_weights_cuda` for all weights except
//! the three split MLA projection matrices (`kv_b_proj_k`, `kv_b_proj_v`,
//! `kv_b_proj_k_t`) which are derived from a single on-disk
//! `kv_b_proj.weight` via [`split_kv_b_proj_dense`].

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use infernum::backend::TensorOps as _;
use infernum::block_allocator::{BlockConfig, BlockTable};
use infernum::dtype::{Q4K_BLOCK_ELEMENTS, QUANTIZATION_BLOCK_SIZE};
use infernum::graph::{optimizer, plan, ExecutionPlan, Graph, NodeId, WeightId, WeightStore};
use infernum::shard::{ShardConfig, ShardStrategy};
use infernum::WeightLoader as _;
use infernum::{DType, ModelConfig, Result};
use infernum_cuda::{
    cast_to_f32, execute, CudaBackend, CudaContext, CudaLogits, CudaRuntimeState, CudaTensor,
    CudaWeightLoader, GgufLoader, LinearWeight, SafeTensorsLoader,
};

use crate::config::DeepSeekConfig;
use crate::graph_builder::{
    build_decode_graph, build_prefill_graph, safetensors_to_gguf_name, shard_strategy_for_deepseek,
    DeepSeekGraphOps,
};
use crate::weights::{split_kv_b_proj_dense, split_kv_b_proj_dense_sharded};

// ---------------------------------------------------------------------------
// MlaKvState
// ---------------------------------------------------------------------------

/// KV cache for `DeepSeek` MLA attention.
///
/// `kv[layer][step]` holds the compressed latent KV tensor appended by
/// `mla_attention` at step `step` for `layer`. The executor maintains this
/// state across decode calls via a raw pointer in [`CudaExecutorState`].
pub struct MlaKvState {
    pub kv: Vec<Vec<CudaTensor>>,
    pub seq_pos: usize,
}

// SAFETY: CudaTensor is Send (CUDA handles are safe to send across threads
// when each thread owns exclusive access). MlaKvState is accessed exclusively
// by the single runtime thread.
unsafe impl Send for MlaKvState {}

// ---------------------------------------------------------------------------
// Weight loading — SafeTensors
// ---------------------------------------------------------------------------

/// Return the on-disk `kv_b_proj.weight` name for a split virtual weight name,
/// or `None` if the name is not a split weight.
fn kv_b_proj_disk_name(registered_name: &str) -> Option<String> {
    for suffix in &[
        ".kv_b_proj_k.weight",
        ".kv_b_proj_v.weight",
        ".kv_b_proj_k_t.weight",
    ] {
        if let Some(prefix) = registered_name.strip_suffix(suffix) {
            return Some(format!("{prefix}.kv_b_proj.weight"));
        }
    }
    None
}

/// Load all graph weights from a `SafeTensors` directory.
///
/// Mirrors `load_graph_weights_cuda` but intercepts the three split MLA
/// projection weights (`kv_b_proj_k`, `kv_b_proj_v`, `kv_b_proj_k_t`) that
/// don't exist on disk as separate tensors. For each layer, the single on-disk
/// `kv_b_proj.weight` is loaded once and split into three parts via
/// [`split_kv_b_proj_dense`].
///
/// # Panics
///
/// Panics if the number of weights in the graph exceeds `u32::MAX`.
///
/// # Errors
///
/// Returns an error if any weight cannot be found or uploaded to the GPU.
fn load_weights_cuda(
    dummy_graph: &Graph<CudaBackend>,
    ctx: &CudaContext,
    model_dir: &Path,
    config: &DeepSeekConfig,
) -> Result<WeightStore<CudaTensor, LinearWeight>> {
    let format_loader = SafeTensorsLoader::from_directory(model_dir)?;
    let loader = CudaWeightLoader::new(ctx.clone(), format_loader);

    let tensor_count = dummy_graph.tensor_weight_count();
    let linear_count = dummy_graph.linear_weight_count();

    let mut store = WeightStore::with_capacity(tensor_count, linear_count);

    // Tensor weights (norms, gate biases, embeddings).
    for i in 0..tensor_count {
        let meta = dummy_graph.tensor_weight_meta(WeightId::from_index(
            u32::try_from(i).expect("weight count exceeds u32"),
        ));
        let tensor = loader.load_tensor(&meta.name, meta.dtype)?;
        store.push_tensor_weight(tensor);
    }

    // Split cache: disk name → (k_tensor, v_tensor, k_t_tensor).
    let mut split_cache: HashMap<String, (CudaTensor, CudaTensor, CudaTensor)> = HashMap::new();

    // Linear weights — with kv_b_proj interception.
    for i in 0..linear_count {
        let meta = dummy_graph.linear_weight_meta(WeightId::from_index(
            u32::try_from(i).expect("weight count exceeds u32"),
        ));
        let name = &meta.name;

        // kv_b_proj split: load once per layer, cache for the other two parts.
        if let Some(disk_name) = kv_b_proj_disk_name(name) {
            if !split_cache.contains_key(&disk_name) {
                // On disk: (num_heads*(qk_nope+v), kv_lora_rank). Transpose to
                // (kv_lora_rank, num_heads*(qk_nope+v)) before splitting.
                let raw = loader.load_tensor(&disk_name, meta.dtype)?;
                let raw_t = CudaBackend::transpose_2d(&raw)?;
                let split = split_kv_b_proj_dense::<CudaBackend>(
                    ctx,
                    &raw_t,
                    config.num_attention_heads,
                    config.qk_nope_head_dim,
                    config.v_head_dim,
                )?;
                split_cache.insert(disk_name.clone(), split);
            }
            let (k, v, k_t) = split_cache.get(&disk_name).expect("just inserted");
            let weight = if name.ends_with(".kv_b_proj_k.weight") {
                LinearWeight::Dense(k.clone())
            } else if name.ends_with(".kv_b_proj_v.weight") {
                LinearWeight::Dense(v.clone())
            } else {
                // kv_b_proj_k_t
                LinearWeight::Dense(k_t.clone())
            };
            store.push_linear_weight(weight);
            continue;
        }

        // lm_head fallback for tied-embedding models.
        let load_name = if name == "lm_head.weight" && !loader.contains("lm_head.weight") {
            "model.embed_tokens.weight"
        } else {
            name.as_str()
        };

        let weight = loader.load_linear(load_name, meta.dtype, None)?;
        store.push_linear_weight(weight);
    }

    Ok(store)
}

// ---------------------------------------------------------------------------
// Weight loading — GGUF
// ---------------------------------------------------------------------------

/// Parse a `[N]` expert-index suffix from a GGUF weight name.
fn parse_expert_suffix(name: &str) -> Option<(String, usize)> {
    let name = name.strip_suffix(']')?;
    let bracket = name.rfind('[')?;
    let idx: usize = name[bracket + 1..].parse().ok()?;
    Some((name[..bracket].to_string(), idx))
}

/// Shard a 2D BF16 byte slice on the host, then upload to GPU.
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
            CudaTensor::from_raw_bytes(ctx, &[rows, cols], DType::BF16, data)
        }
        ShardStrategy::Column => {
            let (start_row, shard_rows) = shard.shard_range(rows);
            let row_bytes = cols * elem;
            let start = start_row * row_bytes;
            let end = start + shard_rows * row_bytes;
            CudaTensor::from_raw_bytes(ctx, &[shard_rows, cols], DType::BF16, &data[start..end])
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
            CudaTensor::from_raw_bytes(ctx, &[rows, shard_cols], DType::BF16, &shard_data)
        }
    }
}

/// Load all graph weights from a GGUF file.
///
/// Mirrors `load_weights_cuda` but reads from GGUF format. Weight names in the
/// graph are SafeTensors names; [`safetensors_to_gguf_name`] converts them to
/// GGUF tensor names for lookup.
///
/// kv_b_proj split: the three virtual weights all map to `blk.N.attn_kv_b.weight`
/// in GGUF. The on-disk tensor is loaded once per layer and split via
/// [`split_kv_b_proj_dense`] (or the sharded variant).
///
/// Expert weights (identified by `[N]` suffix in their GGUF name) are loaded
/// via `load_quantized_expert_slice`, with a BF16 host fallback for unaligned
/// Row sharding.
///
/// # Panics
///
/// Panics if the number of weights in the graph exceeds `u32::MAX`.
///
/// # Errors
///
/// Returns an error if any weight cannot be found or uploaded to the GPU.
fn load_weights_gguf(
    dummy_graph: &Graph<CudaBackend>,
    ctx: &CudaContext,
    gguf_path: &Path,
    config: &DeepSeekConfig,
    shard: Option<&ShardConfig>,
) -> Result<WeightStore<CudaTensor, LinearWeight>> {
    use infernum_cuda::WeightLoader as _;

    let loader = GgufLoader::from_file_or_split(gguf_path)?;

    let tensor_count = dummy_graph.tensor_weight_count();
    let linear_count = dummy_graph.linear_weight_count();
    let mut store = WeightStore::with_capacity(tensor_count, linear_count);

    // Tensor weights (norms, embeddings) — always replicated.
    for i in 0..tensor_count {
        let meta = dummy_graph.tensor_weight_meta(WeightId::from_index(
            u32::try_from(i).expect("weight count exceeds u32"),
        ));
        let gguf_name = safetensors_to_gguf_name(&meta.name);
        let tensor = loader.load_bf16(ctx, &gguf_name)?;
        store.push_tensor_weight(tensor);
    }

    // Split cache: GGUF disk name → (k_tensor, v_tensor, k_t_tensor).
    let mut split_cache: HashMap<String, (CudaTensor, CudaTensor, CudaTensor)> = HashMap::new();

    for i in 0..linear_count {
        let meta = dummy_graph.linear_weight_meta(WeightId::from_index(
            u32::try_from(i).expect("weight count exceeds u32"),
        ));
        let safetensors_name = &meta.name;
        let gguf_name = safetensors_to_gguf_name(safetensors_name);

        // kv_b split: all three virtual weights (kv_b_proj_k/v/k_t) map to the
        // same on-disk blk.N.attn_kv_b.weight. `gguf_name` is already correctly
        // mapped (safetensors_to_gguf_name handles the _k/_v/_k_t suffixes).
        if let Some(_st_disk_name) = kv_b_proj_disk_name(safetensors_name) {
            let gguf_disk_name = gguf_name.clone();

            if !split_cache.contains_key(&gguf_disk_name) {
                // Shape on disk: (n_heads*(qk_nope+v), kv_lora). Transpose before split.
                let raw = loader.load_bf16(ctx, &gguf_disk_name)?;
                let raw_t = CudaBackend::transpose_2d(&raw)?;
                let split = if let Some(s) = shard {
                    split_kv_b_proj_dense_sharded::<CudaBackend>(
                        ctx,
                        &raw_t,
                        config.num_attention_heads,
                        config.qk_nope_head_dim,
                        config.v_head_dim,
                        s,
                    )?
                } else {
                    split_kv_b_proj_dense::<CudaBackend>(
                        ctx,
                        &raw_t,
                        config.num_attention_heads,
                        config.qk_nope_head_dim,
                        config.v_head_dim,
                    )?
                };
                split_cache.insert(gguf_disk_name.clone(), split);
            }

            let (k, v, k_t) = split_cache.get(&gguf_disk_name).expect("just inserted");
            let weight = if safetensors_name.ends_with(".kv_b_proj_k.weight") {
                LinearWeight::Dense(k.clone())
            } else if safetensors_name.ends_with(".kv_b_proj_v.weight") {
                LinearWeight::Dense(v.clone())
            } else {
                // kv_b_proj_k_t
                LinearWeight::Dense(k_t.clone())
            };
            store.push_linear_weight(weight);
            continue;
        }

        // lm_head fallback for tied-embedding models.
        let effective_gguf_name =
            if gguf_name == "output.weight" && !loader.contains("output.weight") {
                "token_embd.weight".to_string()
            } else {
                gguf_name.clone()
            };

        let strategy = shard_strategy_for_deepseek(safetensors_name);

        // Expert weights: blk.N.ffn_gate_exps.weight[E] etc.
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
                            DType::BF16,
                            &host_bytes,
                        )?,
                        Some(s) => shard_bf16_slice(ctx, &host_bytes, rows, cols, s, strategy)?,
                    };
                    store.push_linear_weight(LinearWeight::Dense(CudaBackend::transpose_2d(
                        &tensor,
                    )?));
                }
                Err(e) => return Err(e),
            }
            continue;
        }

        let file_dtype = loader.get_dtype(&effective_gguf_name)?;

        // MLA does not use interleaved RoPE on Q/K projections; no un-permutation needed.
        let supports_quant_sharding = match file_dtype {
            DType::Q8_0 | DType::Q4_0 => shard.map_or(true, |s| {
                strategy != ShardStrategy::Row
                    || loader
                        .get_shape(&effective_gguf_name)
                        .ok()
                        .and_then(|sh| sh.get(1).copied())
                        .map_or(false, |n_cols| {
                            (n_cols / QUANTIZATION_BLOCK_SIZE) % s.world_size == 0
                        })
            }),
            DType::Q4_K | DType::Q5_K => shard.map_or(true, |s| {
                strategy != ShardStrategy::Row
                    || loader
                        .get_shape(&effective_gguf_name)
                        .ok()
                        .and_then(|sh| sh.get(1).copied())
                        .map_or(false, |n_cols| {
                            (n_cols / Q4K_BLOCK_ELEMENTS) % s.world_size == 0
                        })
            }),
            _ => false,
        };
        let use_gpu_quant =
            file_dtype.has_gpu_quant_kernel() && (shard.is_none() || supports_quant_sharding);

        let weight = if use_gpu_quant {
            let qt = match shard {
                None => loader.load_quantized(ctx, &effective_gguf_name)?,
                Some(s) => loader.load_quantized_sharded(ctx, &effective_gguf_name, s, strategy)?,
            };
            LinearWeight::Quantized(qt)
        } else {
            let (host_bytes, shape) = loader.load_bf16_bytes(&effective_gguf_name)?;
            let (rows, cols) = (shape[0], shape[1]);
            let tensor = match shard {
                None => CudaTensor::from_raw_bytes(ctx, &[rows, cols], DType::BF16, &host_bytes)?,
                Some(s) => shard_bf16_slice(ctx, &host_bytes, rows, cols, s, strategy)?,
            };
            LinearWeight::Dense(CudaBackend::transpose_2d(&tensor)?)
        };

        store.push_linear_weight(weight);
    }

    Ok(store)
}

// ---------------------------------------------------------------------------
// DecodeCache
// ---------------------------------------------------------------------------

struct DecodeCache {
    graph: Graph<CudaBackend>,
    plan: ExecutionPlan,
    logits_id: NodeId,
}

fn build_decode_cache(config: &DeepSeekConfig, shard: Option<&ShardConfig>) -> DecodeCache
where
    CudaBackend: DeepSeekGraphOps,
{
    let mut graph: Graph<CudaBackend> = build_decode_graph(config, DType::BF16, shard);
    optimizer::optimize(&mut graph);
    let ep = plan(&graph);
    let logits_id = graph.output_ids()[0];
    DecodeCache {
        graph,
        plan: ep,
        logits_id,
    }
}

// ---------------------------------------------------------------------------
// DeepSeekCudaEngine
// ---------------------------------------------------------------------------

/// CUDA graph-mode engine for the `DeepSeek` model family (DeepSeek-V3, R1).
///
/// Loads weights once and satisfies [`infernum::Model`]. Unlike
/// `CudaGraphEngine<C>`, this engine manages a contiguous MLA KV cache
/// ([`MlaKvState`]) instead of a paged block cache.
pub struct DeepSeekCudaEngine {
    config: DeepSeekConfig,
    ctx: CudaContext,
    weights: Arc<WeightStore<CudaTensor, LinearWeight>>,
    decode: DecodeCache,
    #[cfg(feature = "nccl")]
    comm: Option<infernum_cuda::NcclCommunicator>,
}

// SAFETY: DeepSeekCudaEngine holds CudaContext (which contains an Arc<CudaDevice>)
// and an Arc<WeightStore>. Both are safe to send across threads when used with
// proper CUDA context management. The engine is only used single-threadedly by
// the Runtime.
unsafe impl Send for DeepSeekCudaEngine {}

impl DeepSeekCudaEngine {
    /// Load a DeepSeek-family model from a `SafeTensors` directory onto a CUDA device.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or `config.json` cannot be parsed.
    pub fn from_pretrained(ctx: CudaContext, model_dir: &Path) -> Result<Self>
    where
        CudaBackend: DeepSeekGraphOps,
    {
        let config = DeepSeekConfig::from_file(model_dir.join("config.json"))?;

        // Build a dummy prefill graph to enumerate weight metadata.
        let dummy_graph: Graph<CudaBackend> = build_prefill_graph(&config, DType::BF16, None);
        let weights = load_weights_cuda(&dummy_graph, &ctx, model_dir, &config)?;

        let decode = build_decode_cache(&config, None);
        Ok(Self {
            config,
            ctx,
            weights: Arc::new(weights),
            decode,
            #[cfg(feature = "nccl")]
            comm: None,
        })
    }

    /// Load a DeepSeek-family model from a GGUF file onto a single CUDA device.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened, weights cannot be loaded,
    /// or metadata cannot be parsed.
    pub fn from_gguf(ctx: CudaContext, gguf_path: &Path) -> Result<Self>
    where
        CudaBackend: DeepSeekGraphOps,
    {
        let loader = GgufLoader::from_file_or_split(gguf_path)?;
        let config = DeepSeekConfig::from_gguf_metadata(loader.metadata())?;

        let dummy_graph: Graph<CudaBackend> = build_prefill_graph(&config, DType::BF16, None);
        let weights = load_weights_gguf(&dummy_graph, &ctx, gguf_path, &config, None)?;

        let decode = build_decode_cache(&config, None);
        Ok(Self {
            config,
            ctx,
            weights: Arc::new(weights),
            decode,
            #[cfg(feature = "nccl")]
            comm: None,
        })
    }

    /// Load a tensor-parallel shard of a DeepSeek GGUF model onto one CUDA device.
    ///
    /// Called by [`DeepSeekShardedEngine`] — one invocation per GPU rank.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened, weights cannot be loaded,
    /// or metadata cannot be parsed.
    #[cfg(feature = "nccl")]
    pub fn from_gguf_comm(
        ctx: CudaContext,
        comm: infernum_cuda::NcclCommunicator,
        shard: ShardConfig,
        gguf_path: &Path,
    ) -> Result<Self>
    where
        CudaBackend: DeepSeekGraphOps,
    {
        let loader = GgufLoader::from_file_or_split(gguf_path)?;
        let config = DeepSeekConfig::from_gguf_metadata(loader.metadata())?;

        let dummy_graph: Graph<CudaBackend> =
            build_prefill_graph(&config, DType::BF16, Some(&shard));
        let weights = load_weights_gguf(&dummy_graph, &ctx, gguf_path, &config, Some(&shard))?;

        let decode = build_decode_cache(&config, Some(&shard));
        Ok(Self {
            config,
            ctx,
            weights: Arc::new(weights),
            decode,
            comm: Some(comm),
        })
    }

    /// Return a reference to the model configuration.
    #[must_use]
    pub fn config(&self) -> &DeepSeekConfig {
        &self.config
    }

    /// Run one decode step, accumulating MLA KV state.
    ///
    /// Executes the decode graph with `token_id` as the sole input.
    /// The MLA KV cache (`kv.kv`) is updated in-place by the executor.
    /// `kv.seq_pos` is incremented after the call.
    fn run_step(&self, token_id: u32, kv: &mut MlaKvState) -> Result<CudaTensor> {
        let dc = &self.decode;
        let token_t = CudaTensor::from_slice(&self.ctx, &[1], &[token_id])?;

        #[cfg(feature = "nccl")]
        let comm = self.comm.as_ref();
        #[cfg(not(feature = "nccl"))]
        let comm: Option<&infernum_cuda::NcclCommunicator> = None;

        let (mut outputs, _) = execute(
            &self.ctx,
            &dc.plan,
            dc.graph.nodes(),
            &self.weights,
            &[token_t],
            &[dc.logits_id],
            Some(&mut kv.kv),
            None, // no paged KV cache
            kv.seq_pos,
            None, // no CUDA graph capture
            comm,
        )?;
        kv.seq_pos += 1;

        Ok(outputs.pop().expect("decode graph has no output"))
    }

    /// Allocate a fresh, empty MLA KV state.
    pub fn fresh_kv(&self) -> MlaKvState {
        MlaKvState {
            kv: vec![Vec::new(); self.config.num_hidden_layers],
            seq_pos: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Model impl
// ---------------------------------------------------------------------------

impl infernum::Model for DeepSeekCudaEngine {
    type B = CudaBackend;
    type KvCache = MlaKvState;

    fn config(&self) -> ModelConfig {
        ModelConfig {
            num_layers: self.config.num_hidden_layers,
            max_seq_len: self.config.max_position_embeddings,
            // MLA has no paged KV cache; these values are placeholders for the
            // runtime's block allocator sizing (the allocator is not used).
            num_kv_heads: self.config.num_kv_heads(),
            head_dim: self.config.v_head_dim,
            eos_token_id: self.config.eos_token_id,
            cache_dtype: DType::BF16,
        }
    }

    fn device(&self) -> &CudaContext {
        &self.ctx
    }

    fn allocate_kv_cache(&self, _block_config: &BlockConfig) -> Result<MlaKvState> {
        Ok(self.fresh_kv())
    }

    /// Full forward pass without KV cache (stateless prefill).
    ///
    /// Processes all `input_ids` one token at a time with a fresh MLA state,
    /// and returns the logits for the last token.
    fn forward(&self, input_ids: &[u32]) -> Result<CudaLogits> {
        let mut kv = self.fresh_kv();
        let mut last_logits: Option<CudaTensor> = None;
        for &token in input_ids {
            last_logits = Some(self.run_step(token, &mut kv)?);
        }
        let logits_bf16 = last_logits.expect("input_ids must not be empty");
        let logits_f32 = cast_to_f32(&logits_bf16)?;
        Ok(CudaLogits::new(logits_f32))
    }

    /// Single-sequence prefill with MLA KV cache.
    ///
    /// Processes each token in `input_ids` through the decode graph,
    /// accumulating into `kv_cache`. Returns logits for the last token.
    /// The `block_table` and `start_pos` parameters are unused (MLA has no
    /// paged KV).
    fn forward_prefill(
        &self,
        input_ids: &[u32],
        kv_cache: &mut MlaKvState,
        _runtime_state: &mut CudaRuntimeState,
        _block_table: &BlockTable,
        _start_pos: usize,
    ) -> Result<CudaLogits> {
        let mut last_logits: Option<CudaTensor> = None;
        for &token in input_ids {
            last_logits = Some(self.run_step(token, kv_cache)?);
        }
        let logits_bf16 = last_logits.expect("input_ids must not be empty");
        let logits_f32 = cast_to_f32(&logits_bf16)?;
        Ok(CudaLogits::new(logits_f32))
    }

    /// Batched decode with MLA KV cache.
    ///
    /// Only `batch_size == 1` is supported. The token to decode is extracted
    /// from `token_ids[0]`. `block_tables`, `seq_lens`, `positions`, and
    /// `max_blocks_per_seq` are unused (MLA has no paged KV).
    #[allow(clippy::too_many_arguments)]
    fn forward_batch_decode(
        &self,
        token_ids: &CudaTensor,
        kv_cache: &mut MlaKvState,
        _runtime_state: &mut CudaRuntimeState,
        _block_tables: &CudaTensor,
        _seq_lens: &CudaTensor,
        _positions: &CudaTensor,
        batch_size: usize,
        _max_blocks_per_seq: usize,
        _max_seq_len: usize,
    ) -> Result<CudaLogits> {
        assert_eq!(
            batch_size, 1,
            "DeepSeekCudaEngine only supports batch_size == 1"
        );

        // Extract the single token from the GPU tensor.
        let token_vec = token_ids.to_vec::<u32>()?;
        let token = token_vec[0];

        let logits_bf16 = self.run_step(token, kv_cache)?;
        let logits_f32 = cast_to_f32(&logits_bf16)?;
        Ok(CudaLogits::new(logits_f32))
    }
}

// ---------------------------------------------------------------------------
// Extension trait / constructor
// ---------------------------------------------------------------------------

/// Extension trait providing a convenient `from_pretrained` constructor.
pub trait DeepSeekCudaGraphEngineExt: Sized {
    /// Load a DeepSeek-family model from a `SafeTensors` directory onto a CUDA device.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or `config.json` cannot be parsed.
    fn from_pretrained(ctx: CudaContext, model_dir: &Path) -> Result<Self>;
}

impl DeepSeekCudaGraphEngineExt for DeepSeekCudaEngine {
    fn from_pretrained(ctx: CudaContext, model_dir: &Path) -> Result<Self> {
        DeepSeekCudaEngine::from_pretrained(ctx, model_dir)
    }
}

// ---------------------------------------------------------------------------
// DeepSeekShardedEngine — tensor-parallel multi-GPU engine
// ---------------------------------------------------------------------------

/// Tensor-parallel CUDA engine for DeepSeek spanning multiple GPUs.
///
/// Implements [`infernum::Model`] with `KvCache = Vec<MlaKvState>` (one
/// [`MlaKvState`] per rank). Use [`DeepSeekShardedEngine::from_gguf`] to
/// construct; it creates one [`DeepSeekCudaEngine`] per GPU, wires up NCCL
/// communicators, and loads sharded weights on each device in parallel.
#[cfg(feature = "nccl")]
pub struct DeepSeekShardedEngine {
    replicas: Vec<DeepSeekCudaEngine>,
}

// SAFETY: Each replica is accessed only from its own scoped thread inside the
// forward methods. See DeepSeekCudaEngine's Send impl for the per-engine reasoning.
#[cfg(feature = "nccl")]
unsafe impl Send for DeepSeekShardedEngine {}

#[cfg(feature = "nccl")]
impl DeepSeekShardedEngine {
    /// Load a tensor-parallel DeepSeek model from a GGUF file across `num_devices` GPUs.
    ///
    /// Creates one CUDA context per device, establishes an NCCL communicator
    /// group, and loads each shard in a dedicated thread.
    ///
    /// # Errors
    ///
    /// Returns an error if device creation, NCCL setup, or weight loading
    /// fails on any device.
    ///
    /// # Panics
    ///
    /// Panics if a device thread panics.
    pub fn from_gguf(num_devices: usize, gguf_path: &Path) -> Result<Self>
    where
        CudaBackend: DeepSeekGraphOps,
    {
        use infernum::backend::MultiDeviceOps as _;
        use std::thread;

        let comm_id = CudaBackend::create_comm_id()?;
        let comm_id_raw = *comm_id.to_raw();
        let gguf_path = gguf_path.to_owned();

        let replicas = thread::scope(|s| {
            let handles: Vec<_> = (0..num_devices)
                .map(|rank| {
                    let gguf_path = gguf_path.clone();
                    s.spawn(move || {
                        let ctx = CudaBackend::create_device(rank)?;
                        let comm_id = infernum_cuda::NcclId::from_raw(comm_id_raw);
                        let comm = CudaBackend::create_comm(&ctx, rank, num_devices, comm_id)?;
                        let shard = ShardConfig {
                            rank,
                            world_size: num_devices,
                        };
                        DeepSeekCudaEngine::from_gguf_comm(ctx, comm, shard, &gguf_path)
                    })
                })
                .collect();

            handles
                .into_iter()
                .map(|h| h.join().expect("device thread panicked"))
                .collect::<Result<Vec<_>>>()
        })?;

        Ok(Self { replicas })
    }
}

#[cfg(feature = "nccl")]
impl infernum::Model for DeepSeekShardedEngine {
    type B = CudaBackend;
    type KvCache = Vec<MlaKvState>;

    fn config(&self) -> ModelConfig {
        let r = &self.replicas[0];
        ModelConfig {
            num_layers: r.config.num_hidden_layers,
            max_seq_len: r.config.max_position_embeddings,
            num_kv_heads: r.config.num_kv_heads(),
            head_dim: r.config.v_head_dim,
            eos_token_id: r.config.eos_token_id,
            cache_dtype: DType::BF16,
        }
    }

    fn device(&self) -> &CudaContext {
        &self.replicas[0].ctx
    }

    fn allocate_kv_cache(&self, _block_config: &BlockConfig) -> Result<Vec<MlaKvState>> {
        Ok(self.replicas.iter().map(|r| r.fresh_kv()).collect())
    }

    fn forward(&self, input_ids: &[u32]) -> Result<CudaLogits> {
        use std::thread;
        thread::scope(|s| {
            let handles: Vec<_> = self
                .replicas
                .iter()
                .map(|r| s.spawn(move || infernum::Model::forward(r, input_ids)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("device thread panicked"))
                .next()
                .expect("no replicas")
        })
    }

    fn forward_prefill(
        &self,
        input_ids: &[u32],
        kv_cache: &mut Vec<MlaKvState>,
        _runtime_state: &mut CudaRuntimeState,
        _block_table: &BlockTable,
        _start_pos: usize,
    ) -> Result<CudaLogits> {
        use std::thread;
        thread::scope(|s| {
            let handles: Vec<_> = self
                .replicas
                .iter()
                .zip(kv_cache.iter_mut())
                .map(|(r, kv)| {
                    s.spawn(move || {
                        let mut last_logits: Option<CudaTensor> = None;
                        for &token in input_ids {
                            last_logits = Some(r.run_step(token, kv)?);
                        }
                        let logits_bf16 = last_logits.expect("input_ids must not be empty");
                        let logits_f32 = cast_to_f32(&logits_bf16)?;
                        Ok(CudaLogits::new(logits_f32))
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("device thread panicked"))
                .next()
                .expect("no replicas")
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_decode(
        &self,
        token_ids: &CudaTensor,
        kv_cache: &mut Vec<MlaKvState>,
        _runtime_state: &mut CudaRuntimeState,
        _block_tables: &CudaTensor,
        _seq_lens: &CudaTensor,
        _positions: &CudaTensor,
        batch_size: usize,
        _max_blocks_per_seq: usize,
        _max_seq_len: usize,
    ) -> Result<CudaLogits> {
        use std::thread;

        assert_eq!(
            batch_size, 1,
            "DeepSeekShardedEngine only supports batch_size == 1"
        );

        // Pre-download to host before spawning rank threads to avoid AllReduce deadlock.
        let token_vec = token_ids.to_vec::<u32>()?;
        let token = token_vec[0];

        thread::scope(|s| {
            let handles: Vec<_> = self
                .replicas
                .iter()
                .zip(kv_cache.iter_mut())
                .map(|(r, kv)| {
                    s.spawn(move || {
                        let logits_bf16 = r.run_step(token, kv)?;
                        let logits_f32 = cast_to_f32(&logits_bf16)?;
                        Ok(CudaLogits::new(logits_f32))
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("device thread panicked"))
                .next()
                .expect("no replicas")
        })
    }
}
