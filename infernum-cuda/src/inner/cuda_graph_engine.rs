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
//! - Graphs are rebuilt for each decode step (no indirect / CUDA graph capture).
//!   A future optimization can adopt the `CudaDecodeEngine` approach.

use std::path::Path;
use std::sync::Arc;

use infernum::block_allocator::{BlockConfig, BlockTable};
use infernum::graph::{optimizer, plan, Graph, WeightId, WeightStore};
use infernum::weights::QuantizationConfig;
use infernum::{DType, ModelConfig, Result};

use super::executor::execute;
use crate::cuda::ops::{cast_to_f32, LinearWeight};
use crate::cuda::{CudaContext, CudaTensor};
use crate::cuda_logits::CudaLogits;
use crate::weights::{CudaWeightLoader, SafeTensorsLoader};
use crate::CudaBackend;

// ---------------------------------------------------------------------------
// CudaGraphEngineConfig
// ---------------------------------------------------------------------------

/// Configuration trait for CUDA-backed graph engines.
///
/// Model crates implement this for their `*Config` type. It provides:
/// - Scalar config getters (matching [`infernum_cpu::GraphEngineConfig`]).
/// - Methods to build `Graph<CudaBackend>` for prefill and decode.
/// - A method to load weights from a `SafeTensors` directory into a CUDA///   [`WeightStore`].
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
    fn build_prefill_graph_cuda(&self, seq_len: usize) -> Graph<CudaBackend>;

    /// Build a decode-mode graph for a KV cache of the given length.
    fn build_decode_graph_cuda(&self, kv_len: usize) -> Graph<CudaBackend>;

    /// Build a batched paged-KV decode graph for the given batch size and block config.
    fn build_paged_decode_graph_cuda(
        &self,
        batch_size: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Graph<CudaBackend>;

    /// Load all model weights from a `SafeTensors` directory into CUDA memory.
    ///
    /// The `dummy_graph` (a prefill graph built with a small `seq_len`) is
    /// used to enumerate weight metadata (names, shapes, dtypes). The weights
    /// are then loaded from disk and uploaded to the provided CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if any weight file is missing or cannot be read.
    fn load_weights_cuda_safetensors(
        &self,
        dummy_graph: &Graph<CudaBackend>,
        ctx: &CudaContext,
        model_dir: &Path,
    ) -> Result<WeightStore<CudaTensor, LinearWeight>>;
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
) -> Result<WeightStore<CudaTensor, LinearWeight>> {
    use infernum::WeightLoader as _;

    let format_loader = SafeTensorsLoader::from_directory(model_dir)?;
    let loader = CudaWeightLoader::new(ctx.clone(), format_loader);

    let tensor_count = graph.tensor_weight_count();
    let linear_count = graph.linear_weight_count();

    let mut store = WeightStore::with_capacity(tensor_count, linear_count);

    for i in 0..tensor_count {
        let meta = graph.tensor_weight_meta(WeightId::from_index(
            u32::try_from(i).expect("weight count exceeds u32"),
        ));
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
        let weight = loader.load_linear(name, meta.dtype, effective_quant)?;
        store.push_linear_weight(weight);
    }

    Ok(store)
}

// ---------------------------------------------------------------------------
// RoPE precomputation (mirrors infernum-cpu/src/graph_engine.rs)
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn precompute_rope_row(pos: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let mut cos_row = Vec::with_capacity(half_dim);
    let mut sin_row = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let freq = 1.0_f32 / theta.powf(2.0 * i as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        cos_row.push(angle.cos());
        sin_row.push(angle.sin());
    }
    (cos_row, sin_row)
}

fn precompute_rope_data(max_pos: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let mut cos_data = Vec::with_capacity(max_pos * half_dim);
    let mut sin_data = Vec::with_capacity(max_pos * half_dim);
    for pos in 0..max_pos {
        let (c, s) = precompute_rope_row(pos, head_dim, theta);
        cos_data.extend(c);
        sin_data.extend(s);
    }
    (cos_data, sin_data)
}

// ---------------------------------------------------------------------------
// CudaGraphEngine
// ---------------------------------------------------------------------------

/// CUDA graph-mode engine for any model family implementing [`CudaGraphEngineConfig`].
///
/// Loads weights once and satisfies [`infernum::Model`], allowing it to be used
/// with `infernum_runtime::Runtime`. Graphs are rebuilt per step (no indirect
/// CUDA graph capture yet).
pub struct CudaGraphEngine<C: CudaGraphEngineConfig> {
    config: C,
    ctx: CudaContext,
    weights: Arc<WeightStore<CudaTensor, LinearWeight>>,
    /// Cached `head_dim / 2` for `RoPE` slice arithmetic.
    half_dim: usize,
}

impl<C: CudaGraphEngineConfig> CudaGraphEngine<C> {
    /// Load a model from a `SafeTensors` directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or the config file cannot be parsed.
    pub fn from_config_and_dir(config: C, ctx: CudaContext, model_dir: &Path) -> Result<Self> {
        let dummy_graph = config.build_prefill_graph_cuda(1);
        let weights = config.load_weights_cuda_safetensors(&dummy_graph, &ctx, model_dir)?;
        let half_dim = config.head_dim() / 2;
        Ok(Self {
            config,
            ctx,
            weights: Arc::new(weights),
            half_dim,
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

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn run_prefill_graph(&self, input_ids: &[u32]) -> Result<Vec<CudaTensor>> {
        let seq_len = input_ids.len();
        let head_dim = self.config.head_dim();
        let half_dim = self.half_dim;

        let mut graph = self.config.build_prefill_graph_cuda(seq_len);
        optimizer::optimize(&mut graph);
        let ep = plan(&graph);

        let (cos_data, sin_data) =
            precompute_rope_data(seq_len, head_dim, self.config.rope_theta());
        let input_ids_t = CudaTensor::from_slice(&self.ctx, &[seq_len], input_ids)?;
        let cos_t = CudaTensor::from_slice(&self.ctx, &[seq_len, half_dim], &cos_data)?;
        let sin_t = CudaTensor::from_slice(&self.ctx, &[seq_len, half_dim], &sin_data)?;
        let inputs = vec![input_ids_t, cos_t, sin_t];

        let output_nodes = graph.output_ids().to_vec();
        execute(
            &ep,
            graph.nodes(),
            &self.weights,
            &inputs,
            &output_nodes,
            None,
            None,
            0,
        )
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
        crate::cuda::PagedKvCache::new(
            &self.ctx,
            self.config.num_hidden_layers(),
            block_config,
            self.config.num_kv_heads(),
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
    /// Iterates through `input_ids` one token at a time using the paged decode
    /// graph. Each step appends K/V into the paged pool (side-effect) and reads
    /// it back for attention. Returns the logits for the last token.
    fn forward_prefill(
        &self,
        input_ids: &[u32],
        kv_cache: &mut crate::cuda::PagedKvCache,
        _runtime_state: &mut <CudaBackend as infernum::backend::Backend>::RuntimeState,
        block_table: &BlockTable,
        _start_pos: usize,
    ) -> Result<<CudaBackend as infernum::backend::Backend>::Logits> {
        let block_size = kv_cache.block_size();
        let head_dim = self.config.head_dim();
        let half_dim = self.half_dim;

        let blocks = block_table.blocks();
        let max_blocks = blocks.len();

        // Convert block IDs (usize) to U32 for the graph inputs.
        let block_ids_u32: Vec<u32> = blocks
            .iter()
            .map(|&b| u32::try_from(b).expect("block ID fits u32"))
            .collect();

        let mut last_logits: Option<CudaTensor> = None;

        for (pos, &token) in input_ids.iter().enumerate() {
            let mut graph = self
                .config
                .build_paged_decode_graph_cuda(1, block_size, max_blocks);
            optimizer::optimize(&mut graph);
            let ep = plan(&graph);

            let (cos_row, sin_row) = precompute_rope_row(pos, head_dim, self.config.rope_theta());

            let input_id_t = CudaTensor::from_slice(&self.ctx, &[1], &[token])?;
            let cos_t = CudaTensor::from_slice(&self.ctx, &[1, half_dim], &cos_row)?;
            let sin_t = CudaTensor::from_slice(&self.ctx, &[1, half_dim], &sin_row)?;
            let block_table_t =
                CudaTensor::from_slice(&self.ctx, &[1, max_blocks], &block_ids_u32)?;
            let pos_u32 = u32::try_from(pos).expect("position fits u32");
            let positions_t = CudaTensor::from_slice(&self.ctx, &[1], &[pos_u32])?;
            let seq_len_t = CudaTensor::from_slice(&self.ctx, &[1], &[pos_u32 + 1])?;

            let inputs = vec![
                input_id_t,
                cos_t,
                sin_t,
                block_table_t,
                positions_t,
                seq_len_t,
            ];
            let output_nodes = graph.output_ids().to_vec();
            let outputs = execute(
                &ep,
                graph.nodes(),
                &self.weights,
                &inputs,
                &output_nodes,
                None,
                Some(kv_cache),
                0,
            )?;
            last_logits = Some(outputs.into_iter().next().expect("no outputs"));
        }

        let logits_tensor = last_logits.expect("input_ids must not be empty");
        let logits_f32 = cast_to_f32(&logits_tensor)?;
        Ok(CudaLogits::new(logits_f32))
    }

    /// Batched decode with paged KV cache.
    ///
    /// Runs one graph step for all sequences in the batch simultaneously.
    /// K/V append and paged attention are handled inside the graph executor
    /// via the `append_paged_batched` and `paged_attention_decode` arms.
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

        let mut graph =
            self.config
                .build_paged_decode_graph_cuda(batch_size, block_size, max_blocks_per_seq);
        optimizer::optimize(&mut graph);
        let ep = plan(&graph);

        let cos_t = CudaTensor::from_slice(&self.ctx, &[batch_size, half_dim], &cos_data)?;
        let sin_t = CudaTensor::from_slice(&self.ctx, &[batch_size, half_dim], &sin_data)?;
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
        let outputs = execute(
            &ep,
            graph.nodes(),
            &self.weights,
            &inputs,
            &output_nodes,
            None,
            Some(kv_cache),
            0,
        )?;

        let logits_bf16 = outputs.into_iter().next().expect("no outputs");
        let logits_f32 = cast_to_f32(&logits_bf16)?;
        Ok(CudaLogits::new(logits_f32))
    }
}
