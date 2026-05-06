//! CUDA graph-mode inference engine generic over model configuration.
//!
//! Provides [`CudaGraphEngine<C>`] and its associated [`CudaGraphKvCache`] type,
//! enabling any model family that implements [`CudaGraphEngineConfig`] to run
//! inference on a CUDA GPU using the graph execution path.
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
use std::sync::{Arc, Mutex};

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
// KV cache
// ---------------------------------------------------------------------------

/// Per-sequence, per-layer KV tensors on the GPU for the graph-mode decode path.
///
/// Stores the K/V history for one sequence as dense CUDA tensors.
/// Shape after `n` tokens: `[n, num_kv_heads, head_dim]`.
struct SeqKvStore {
    /// Per-layer accumulated K/V tensors. `None` until the first decode step.
    layers: Vec<(Option<CudaTensor>, Option<CudaTensor>)>,
    /// Number of tokens whose K/V has been committed to `layers`.
    committed: usize,
}

impl SeqKvStore {
    fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![(None, None); num_layers],
            committed: 0,
        }
    }

    fn get_layer(&self, layer: usize) -> Option<(&CudaTensor, &CudaTensor)> {
        let (k, v) = &self.layers[layer];
        k.as_ref().zip(v.as_ref())
    }

    /// Update K/V from graph outputs.
    ///
    /// Output layout (from all graph builders):
    /// `[logits, k_layer0, k_layer1, ..., k_layer{L-1}, v_layer0, v_layer1, ..., v_layer{L-1}]`
    fn update_from_outputs(&mut self, outputs: &[CudaTensor], num_layers: usize) {
        for layer in 0..num_layers {
            let k = outputs[1 + layer].clone();
            let v = outputs[1 + num_layers + layer].clone();
            self.layers[layer] = (Some(k), Some(v));
        }
        self.committed += 1;
    }
}

/// Multi-sequence KV cache for the graph-mode decode path.
///
/// Keyed by the first block ID of each sequence (stable per-sequence identifier
/// provided by the block allocator). Each sequence maintains its own dense K/V
/// accumulator, enabling inflight batching without a paged KV layout.
///
/// Wraps a `Mutex` so it can be shared across the `Model` trait boundary.
/// The mutex is never contended (single worker thread).
pub struct CudaGraphKvCache {
    inner: Mutex<std::collections::HashMap<usize, SeqKvStore>>,
    num_layers: usize,
}

impl CudaGraphKvCache {
    fn new(num_layers: usize) -> Self {
        Self {
            inner: Mutex::new(std::collections::HashMap::new()),
            num_layers,
        }
    }

    /// Get or insert a per-sequence store for `seq_key`.
    fn with_seq<F, R>(&self, seq_key: usize, f: F) -> R
    where
        F: FnOnce(&mut SeqKvStore) -> R,
    {
        let mut map = self.inner.lock().expect("CudaGraphKvCache mutex poisoned");
        let entry = map
            .entry(seq_key)
            .or_insert_with(|| SeqKvStore::new(self.num_layers));
        f(entry)
    }
}

// `CudaGraphKvCache` must be `Send` for the `Model::KvCache: Send` bound.
// `CudaTensor` is `Send` so this is safe.
unsafe impl Send for CudaGraphKvCache {}

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
            0,
        )
    }

    fn build_empty_kv_inputs(&self) -> Result<Vec<CudaTensor>> {
        let num_layers = self.config.num_hidden_layers();
        let num_kv_heads = self.config.num_kv_heads();
        let head_dim = self.config.head_dim();
        let mut inputs = Vec::with_capacity(2 * num_layers);
        for _ in 0..num_layers {
            // Must match the model's activation dtype (BF16). An F32 empty
            // tensor concatenated with a BF16 K/V update causes an OOB panic
            // because concat_rows uses dtype byte-width for slice arithmetic.
            inputs.push(CudaTensor::zeros(
                &self.ctx,
                &[0, num_kv_heads, head_dim],
                DType::BF16,
            )?);
            inputs.push(CudaTensor::zeros(
                &self.ctx,
                &[0, num_kv_heads, head_dim],
                DType::BF16,
            )?);
        }
        Ok(inputs)
    }

    /// Run one decode step for a single sequence.
    ///
    /// `kv_len` is the number of tokens already in the sequence's KV store,
    /// and `seq_store` provides the current accumulated K/V tensors per layer.
    fn run_decode_graph(
        &self,
        token: u32,
        kv_len: usize,
        seq_store: &SeqKvStore,
    ) -> Result<Vec<CudaTensor>> {
        let head_dim = self.config.head_dim();
        let half_dim = self.half_dim;
        let num_layers = self.config.num_hidden_layers();
        let num_kv_heads = self.config.num_kv_heads();

        let mut graph = self.config.build_decode_graph_cuda(kv_len);
        optimizer::optimize(&mut graph);
        let ep = plan(&graph);

        let pos = kv_len;
        let (cos_row, sin_row) = precompute_rope_row(pos, head_dim, self.config.rope_theta());

        let input_id_t = CudaTensor::from_slice(&self.ctx, &[1], &[token])?;
        let cos_t = CudaTensor::from_slice(&self.ctx, &[1, half_dim], &cos_row)?;
        let sin_t = CudaTensor::from_slice(&self.ctx, &[1, half_dim], &sin_row)?;
        let mut inputs = vec![input_id_t, cos_t, sin_t];

        if kv_len > 0 {
            for layer in 0..num_layers {
                if let Some((k, v)) = seq_store.get_layer(layer) {
                    inputs.push(k.clone());
                    inputs.push(v.clone());
                } else {
                    inputs.push(CudaTensor::zeros(
                        &self.ctx,
                        &[0, num_kv_heads, head_dim],
                        DType::BF16,
                    )?);
                    inputs.push(CudaTensor::zeros(
                        &self.ctx,
                        &[0, num_kv_heads, head_dim],
                        DType::BF16,
                    )?);
                }
            }
        } else {
            inputs.extend(self.build_empty_kv_inputs()?);
        }

        let output_nodes = graph.output_ids().to_vec();
        execute(
            &ep,
            graph.nodes(),
            &self.weights,
            &inputs,
            &output_nodes,
            None,
            0,
        )
    }
}

impl<C: CudaGraphEngineConfig> infernum::Model for CudaGraphEngine<C> {
    type B = CudaBackend;
    type KvCache = CudaGraphKvCache;

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

    fn allocate_kv_cache(&self, _block_config: &BlockConfig) -> Result<CudaGraphKvCache> {
        Ok(CudaGraphKvCache::new(self.config.num_hidden_layers()))
    }

    /// Full forward pass without KV cache.
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

    /// Single-sequence prefill with KV cache.
    ///
    /// Iterates through `input_ids` one token at a time using the decode graph
    /// (which outputs `[logits, k_layer0, v_layer0, ...]`), storing K/V outputs
    /// into `kv_cache` after each step. Returns the logits for the last token.
    ///
    /// The prefill graph cannot be used here because it only outputs logits —
    /// it does not return per-layer K/V tensors needed to populate the cache.
    fn forward_prefill(
        &self,
        input_ids: &[u32],
        kv_cache: &mut CudaGraphKvCache,
        _runtime_state: &mut <CudaBackend as infernum::backend::Backend>::RuntimeState,
        block_table: &BlockTable,
        _start_pos: usize,
    ) -> Result<<CudaBackend as infernum::backend::Backend>::Logits> {
        let num_layers = self.config.num_hidden_layers();
        // Use the first block ID as the stable per-sequence key. Block 0 is
        // always assigned before prefill and never changes for this sequence.
        let seq_key = block_table.blocks().first().copied().unwrap_or(0);
        let mut last_outputs: Option<Vec<CudaTensor>> = None;

        for &token in input_ids {
            let outputs = kv_cache.with_seq(seq_key, |seq_store| -> Result<Vec<CudaTensor>> {
                let kv_len = seq_store.committed;
                let outs = self.run_decode_graph(token, kv_len, seq_store)?;
                seq_store.update_from_outputs(&outs, num_layers);
                Ok(outs)
            })?;
            last_outputs = Some(outputs);
        }

        let outputs = last_outputs.expect("input_ids must not be empty");
        // outputs[0] has shape [1, vocab_size]; cast to F32 and return.
        let logits_f32 = cast_to_f32(&outputs[0])?;
        Ok(CudaLogits::new(logits_f32))
    }

    /// Batched decode with per-sequence KV cache.
    ///
    /// Runs one decode step per sequence independently, using the first block
    /// ID from each row of `block_tables` as the stable per-sequence key into
    /// the KV cache. Each sequence's K/V history is updated after its step.
    /// Logits are concatenated row-wise before returning.
    #[allow(clippy::too_many_arguments)]
    fn forward_batch_decode(
        &self,
        token_ids: &CudaTensor,
        kv_cache: &mut CudaGraphKvCache,
        _runtime_state: &mut <CudaBackend as infernum::backend::Backend>::RuntimeState,
        block_tables: &CudaTensor,
        _seq_lens: &CudaTensor,
        positions: &CudaTensor,
        batch_size: usize,
        max_blocks_per_seq: usize,
        _max_seq_len: usize,
    ) -> Result<<CudaBackend as infernum::backend::Backend>::Logits> {
        let num_layers = self.config.num_hidden_layers();

        // Bring per-sequence data to the host for the graph-engine loop.
        let token_data = token_ids.to_vec::<u32>()?;
        // positions[i] is the number of tokens already committed for sequence i.
        // DType::I32 tensors hold raw i32 bytes; reinterpret after downloading.
        let positions_bytes = positions.to_raw_bytes()?;
        let positions_data: Vec<i32> = positions_bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        // block_tables is a flat (batch_size * max_blocks_per_seq) i32 tensor.
        let block_table_bytes = block_tables.to_raw_bytes()?;
        let block_table_data: Vec<i32> = block_table_bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let mut all_logits: Vec<CudaTensor> = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let token = token_data[i];
            let kv_len = positions_data[i] as usize;
            // First block ID is the stable per-sequence key.
            let seq_key = if max_blocks_per_seq > 0 {
                block_table_data[i * max_blocks_per_seq] as usize
            } else {
                i
            };

            let logits_t = kv_cache.with_seq(seq_key, |seq_store| -> Result<CudaTensor> {
                // Sync committed count from the block table position so that
                // sequences re-entering the batch after being descheduled
                // are handled correctly.
                seq_store.committed = kv_len;
                let outs = self.run_decode_graph(token, kv_len, seq_store)?;
                seq_store.update_from_outputs(&outs, num_layers);
                // outs[0] has shape [1, vocab_size] in BF16.
                Ok(outs
                    .into_iter()
                    .next()
                    .expect("no outputs from decode graph"))
            })?;
            all_logits.push(logits_t);
        }

        // Concatenate logits row-wise: [batch_size, vocab_size] in BF16.
        let logits_bf16 = <CudaBackend as infernum::backend::TensorOps>::concat_rows(&all_logits)?;
        let logits_f32 = cast_to_f32(&logits_bf16)?;
        Ok(CudaLogits::new(logits_f32))
    }
}
