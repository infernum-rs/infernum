//! Metal graph-mode inference engine generic over model configuration.
//!
//! Provides [`MetalGraphEngine<C>`] backed by a paged KV cache
//! ([`MetalPagedKvCache`]), enabling any model family that implements
//! [`MetalGraphEngineConfig`] to run inference on Metal using the graph
//! execution path.
//!
//! # Usage
//!
//! Model crates implement [`MetalGraphEngineConfig`] for their config type and
//! expose a convenience constructor (e.g. `LlamaMetalGraphEngine::from_pretrained`).
//! The engine then satisfies `infernum::Model` and can be plugged into
//! `infernum_runtime::Runtime`.
//!
//! # Differences from `CudaGraphEngine`
//!
//! - No CUDA graph capture/replay — always eager execution.
//! - No MLA KV cache support.
//! - No quantisation probing (Phase 1: dense weights only).

use std::path::Path;
use std::sync::Arc;

use infernum::backend::PagedKvCacheOps;
use infernum::block_allocator::{BlockConfig, BlockTable};
use infernum::graph::{optimizer, plan, Graph, WeightId, WeightStore};
use infernum::{precompute_rope_data, precompute_rope_row, DType, ModelConfig, Result};

use crate::context::MetalContext;
use crate::executor::execute;
use crate::logits::MetalLogits;
use crate::tensor::MetalTensor;
use crate::weights::{MetalLinearWeight, MetalSafeTensorsLoader};
use crate::{MetalBackend, MetalPagedKvCache};

// ---------------------------------------------------------------------------
// MetalGraphEngineConfig
// ---------------------------------------------------------------------------

/// Configuration trait for Metal-backed graph engines.
///
/// Model crates implement this for their `*Config` type. It provides:
/// - Scalar config getters (matching the CUDA and CPU equivalents).
/// - Methods to build `Graph<MetalBackend>` for prefill and decode.
/// - A method to load weights from a SafeTensors directory into Metal
///   memory.
pub trait MetalGraphEngineConfig: Send + Sync + 'static {
    fn num_hidden_layers(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
    fn max_position_embeddings(&self) -> usize;
    fn rope_theta(&self) -> f32;
    fn vocab_size(&self) -> usize;
    fn eos_token_id(&self) -> u32;

    /// Build a prefill-mode graph for the given sequence length.
    fn build_prefill_graph_metal(&self, seq_len: usize) -> Graph<MetalBackend>;

    /// Build a decode-mode graph for a KV cache of the given length.
    fn build_decode_graph_metal(&self, kv_len: usize) -> Graph<MetalBackend>;

    /// Build a batched paged-KV decode graph.
    fn build_paged_decode_graph_metal(
        &self,
        batch_size: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Graph<MetalBackend>;

    /// Load all model weights from a SafeTensors directory into Metal memory.
    ///
    /// The `dummy_graph` (a prefill graph built with a small `seq_len`) is
    /// used to enumerate weight metadata (names, shapes, dtypes). The weights
    /// are then loaded from disk into the provided Metal context.
    ///
    /// # Errors
    ///
    /// Returns an error if any weight file is missing or cannot be read.
    fn load_weights_metal_safetensors(
        &self,
        dummy_graph: &Graph<MetalBackend>,
        ctx: &MetalContext,
        model_dir: &Path,
    ) -> Result<WeightStore<MetalTensor, MetalLinearWeight>>;
}

// ---------------------------------------------------------------------------
// Generic Metal weight loader helper
// ---------------------------------------------------------------------------

/// Load all graph weights from a SafeTensors directory into Metal memory.
///
/// Iterates the graph's weight metadata (tensor weights and linear weights),
/// loads each from disk via a [`MetalSafeTensorsLoader`], and returns a
/// populated [`WeightStore`].
///
/// The `lm_head_fallback` parameter, when `true`, silently falls back to
/// loading `model.embed_tokens.weight` when `lm_head.weight` is absent
/// (tied-embedding models such as SmolLM2).
///
/// # Panics
///
/// Panics if the number of weights in the graph exceeds `u32::MAX`.
///
/// # Errors
///
/// Returns an error if any weight cannot be found or loaded.
pub fn load_graph_weights_metal(
    graph: &Graph<MetalBackend>,
    ctx: &MetalContext,
    model_dir: &Path,
    lm_head_fallback: bool,
) -> Result<WeightStore<MetalTensor, MetalLinearWeight>> {
    use infernum::WeightLoader as _;

    let loader = MetalSafeTensorsLoader::new(ctx.clone(), model_dir)?;

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
        let weight = loader.load_linear(name, meta.dtype, None)?;
        store.push_linear_weight(weight);
    }

    Ok(store)
}

// ---------------------------------------------------------------------------
// MetalGraphEngine
// ---------------------------------------------------------------------------

/// Metal graph-mode engine for any model family implementing [`MetalGraphEngineConfig`].
///
/// Loads weights once and satisfies [`infernum::Model`], allowing it to be used
/// with `infernum_runtime::Runtime`.
///
/// Unlike the CUDA engine, Metal always uses eager execution (no graph
/// capture/replay). Prefill and decode both build a fresh graph per call.
pub struct MetalGraphEngine<C: MetalGraphEngineConfig> {
    config: C,
    ctx: MetalContext,
    weights: Arc<WeightStore<MetalTensor, MetalLinearWeight>>,
    /// Cached `head_dim / 2` for RoPE slice arithmetic.
    half_dim: usize,
}

impl<C: MetalGraphEngineConfig> MetalGraphEngine<C> {
    /// Load a model from a SafeTensors directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing or weights cannot be
    /// loaded.
    pub fn from_config_and_dir(config: C, ctx: MetalContext, model_dir: &Path) -> Result<Self> {
        let dummy_graph = config.build_prefill_graph_metal(1);
        let weights = config.load_weights_metal_safetensors(&dummy_graph, &ctx, model_dir)?;
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

    /// Return a reference to the Metal context.
    #[must_use]
    pub fn metal_context(&self) -> &MetalContext {
        &self.ctx
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn run_prefill_graph(&self, input_ids: &[u32]) -> Result<Vec<MetalTensor>> {
        let seq_len = input_ids.len();
        let head_dim = self.config.head_dim();
        let half_dim = self.half_dim;

        let mut graph = self.config.build_prefill_graph_metal(seq_len);
        optimizer::optimize(&mut graph);
        let ep = plan(&graph);

        let (cos_data, sin_data) =
            precompute_rope_data(seq_len, head_dim, self.config.rope_theta());
        let input_ids_t = MetalTensor::from_raw_bytes(
            &self.ctx,
            &[seq_len],
            DType::U32,
            bytemuck::cast_slice(input_ids),
        );
        let cos_t = MetalTensor::from_f32(&self.ctx, &[seq_len, half_dim], &cos_data);
        let sin_t = MetalTensor::from_f32(&self.ctx, &[seq_len, half_dim], &sin_data);
        let inputs = vec![input_ids_t, cos_t, sin_t];

        let output_nodes = graph.output_ids().to_vec();
        execute(
            &self.ctx,
            &ep,
            graph.nodes(),
            &self.weights,
            &inputs,
            &output_nodes,
            None,
        )
    }
}

impl<C: MetalGraphEngineConfig> infernum::Model for MetalGraphEngine<C> {
    type B = MetalBackend;
    type KvCache = MetalPagedKvCache;

    fn config(&self) -> ModelConfig {
        let c = &self.config;
        ModelConfig {
            num_layers: c.num_hidden_layers(),
            max_seq_len: c.max_position_embeddings(),
            num_kv_heads: c.num_kv_heads(),
            head_dim: c.head_dim(),
            eos_token_id: c.eos_token_id(),
            cache_dtype: DType::F32,
        }
    }

    fn device(&self) -> &MetalContext {
        &self.ctx
    }

    fn allocate_kv_cache(&self, block_config: &BlockConfig) -> Result<MetalPagedKvCache> {
        <MetalBackend as PagedKvCacheOps>::allocate_paged_kv_cache(
            &self.ctx,
            self.config.num_hidden_layers(),
            block_config,
            self.config.num_kv_heads(),
            self.config.head_dim(),
            DType::F32,
        )
    }

    /// Full forward pass without KV cache (prefill graph).
    ///
    /// Returns logits of shape `(seq_len, vocab_size)`.
    fn forward(&self, input_ids: &[u32]) -> Result<MetalLogits> {
        let outputs = self.run_prefill_graph(input_ids)?;
        let logits_tensor = outputs
            .into_iter()
            .next()
            .expect("prefill graph has no output");
        Ok(MetalLogits::from_tensor(logits_tensor))
    }

    /// Single-sequence prefill with paged KV cache.
    ///
    /// Iterates through `input_ids` one token at a time using the paged decode
    /// graph. Each step appends K/V into the paged pool (side-effect) and reads
    /// it back for attention. Returns the logits for the last token.
    fn forward_prefill(
        &self,
        input_ids: &[u32],
        kv_cache: &mut MetalPagedKvCache,
        _runtime_state: &mut <MetalBackend as infernum::backend::Backend>::RuntimeState,
        block_table: &BlockTable,
        _start_pos: usize,
    ) -> Result<MetalLogits> {
        let block_size = kv_cache.block_size;
        let half_dim = self.half_dim;

        let blocks = block_table.blocks();
        let max_blocks = blocks.len();

        let block_ids_u32: Vec<u32> = blocks
            .iter()
            .map(|&b| u32::try_from(b).expect("block ID fits u32"))
            .collect();

        let mut last_logits: Option<MetalTensor> = None;

        for (pos, &token) in input_ids.iter().enumerate() {
            let mut graph = self
                .config
                .build_paged_decode_graph_metal(1, block_size, max_blocks);
            optimizer::optimize(&mut graph);
            let ep = plan(&graph);

            let (cos_row, sin_row) =
                precompute_rope_row(pos, self.config.head_dim(), self.config.rope_theta());

            let input_id_t = MetalTensor::from_raw_bytes(
                &self.ctx,
                &[1],
                DType::U32,
                bytemuck::cast_slice(&[token]),
            );
            let cos_t = MetalTensor::from_f32(&self.ctx, &[1, half_dim], &cos_row);
            let sin_t = MetalTensor::from_f32(&self.ctx, &[1, half_dim], &sin_row);
            let block_table_t = MetalTensor::from_raw_bytes(
                &self.ctx,
                &[1, max_blocks],
                DType::U32,
                bytemuck::cast_slice(&block_ids_u32),
            );
            let pos_u32 = u32::try_from(pos).expect("position fits u32");
            let positions_t = MetalTensor::from_raw_bytes(
                &self.ctx,
                &[1],
                DType::U32,
                bytemuck::cast_slice(&[pos_u32]),
            );
            let seq_len_t = MetalTensor::from_raw_bytes(
                &self.ctx,
                &[1],
                DType::U32,
                bytemuck::cast_slice(&[pos_u32 + 1]),
            );

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
                &self.ctx,
                &ep,
                graph.nodes(),
                &self.weights,
                &inputs,
                &output_nodes,
                Some(kv_cache),
            )?;
            last_logits = Some(outputs.into_iter().next().expect("no outputs"));
        }

        let logits_tensor = last_logits.expect("input_ids must not be empty");
        Ok(MetalLogits::from_tensor(logits_tensor))
    }

    /// Batched decode with paged KV cache (always eager execution).
    #[allow(clippy::too_many_arguments)]
    fn forward_batch_decode(
        &self,
        token_ids: &MetalTensor,
        kv_cache: &mut MetalPagedKvCache,
        _runtime_state: &mut <MetalBackend as infernum::backend::Backend>::RuntimeState,
        block_tables: &MetalTensor,
        _seq_lens: &MetalTensor,
        positions: &MetalTensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
        _max_seq_len: usize,
    ) -> Result<MetalLogits> {
        let block_size = kv_cache.block_size;
        let head_dim = self.config.head_dim();
        let half_dim = self.half_dim;

        // Read positions from unified memory (I32/U32 layout).
        let positions_data: &[i32] = positions.as_i32_slice();

        // Read block_tables from unified memory and reinterpret as U32.
        let block_table_data: &[i32] = block_tables.as_i32_slice();
        let block_table_u32: Vec<u32> = block_table_data
            .iter()
            .map(|&v| v.cast_unsigned())
            .collect();

        // Build batched RoPE: cos/sin for each sequence's current position.
        let mut cos_data = Vec::with_capacity(batch_size * half_dim);
        let mut sin_data = Vec::with_capacity(batch_size * half_dim);
        for &pos_i32 in positions_data {
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

        // Build the paged decode graph.
        let mut graph =
            self.config
                .build_paged_decode_graph_metal(batch_size, block_size, max_blocks_per_seq);
        optimizer::optimize(&mut graph);
        let ep = plan(&graph);

        let cos_t = MetalTensor::from_f32(&self.ctx, &[batch_size, half_dim], &cos_data);
        let sin_t = MetalTensor::from_f32(&self.ctx, &[batch_size, half_dim], &sin_data);
        let block_table_t = MetalTensor::from_raw_bytes(
            &self.ctx,
            &[batch_size, max_blocks_per_seq],
            DType::U32,
            bytemuck::cast_slice(&block_table_u32),
        );
        let positions_t = MetalTensor::from_raw_bytes(
            &self.ctx,
            &[batch_size],
            DType::U32,
            bytemuck::cast_slice(&positions_u32),
        );
        let seq_lens_t = MetalTensor::from_raw_bytes(
            &self.ctx,
            &[batch_size],
            DType::U32,
            bytemuck::cast_slice(&seq_lens_u32),
        );

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
            &self.ctx,
            &ep,
            graph.nodes(),
            &self.weights,
            &inputs,
            &output_nodes,
            Some(kv_cache),
        )?;

        let logits_tensor = outputs.into_iter().next().expect("no outputs");
        Ok(MetalLogits::from_tensor(logits_tensor))
    }
}
