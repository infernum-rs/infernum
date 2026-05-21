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
use infernum::graph::{optimizer, plan, ExecutionPlan, Graph, NodeId, WeightId, WeightStore};
use infernum::WeightLoader as _;
use infernum::{DType, ModelConfig, Result};
use infernum_cuda::{
    cast_to_f32, execute, CudaBackend, CudaContext, CudaLogits, CudaRuntimeState, CudaTensor,
    CudaWeightLoader, LinearWeight, SafeTensorsLoader,
};

use crate::config::DeepSeekConfig;
use crate::graph_builder::{build_decode_graph, build_prefill_graph, DeepSeekGraphOps};
use crate::weights::split_kv_b_proj_dense;

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
// Weight loading
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
// DecodeCache
// ---------------------------------------------------------------------------

struct DecodeCache {
    graph: Graph<CudaBackend>,
    plan: ExecutionPlan,
    logits_id: NodeId,
}

fn build_decode_cache(config: &DeepSeekConfig) -> DecodeCache
where
    CudaBackend: DeepSeekGraphOps,
{
    let mut graph: Graph<CudaBackend> = build_decode_graph(config, DType::BF16);
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
        let dummy_graph: Graph<CudaBackend> = build_prefill_graph(&config, DType::BF16);
        let weights = load_weights_cuda(&dummy_graph, &ctx, model_dir, &config)?;

        let decode = build_decode_cache(&config);
        Ok(Self {
            config,
            ctx,
            weights: Arc::new(weights),
            decode,
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
        )?;
        kv.seq_pos += 1;

        Ok(outputs.pop().expect("decode graph has no output"))
    }

    /// Allocate a fresh, empty MLA KV state.
    fn fresh_kv(&self) -> MlaKvState {
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
