//! CPU graph-mode inference engine for the Llama model family.
//!
//! [`LlamaGraphEngine`] is a self-contained, standalone alternative to
//! [`crate::LlamaModel`] + [`infernum_runtime::Engine`] for CPU inference.
//! It pre-compiles a single decode graph at construction time and reuses it
//! for every token — both prompt warmup and autoregressive decode — via
//! [`KvCacheStore`], which intercepts KV cache nodes in the graph and
//! maintains a persistent, growing KV buffer without copying it through the
//! arena each step.
//!
//! Unlike the eager path, this engine does not use paged KV caches or the
//! runtime scheduler.
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use infernum_llama::graph_engine::LlamaGraphEngine;
//!
//! let engine = LlamaGraphEngine::from_pretrained(Path::new("/path/to/model")).unwrap();
//! let tokens = engine.generate(&[1, 1234, 567], 128, 2).unwrap();
//! println!("{tokens:?}");
//! ```

use std::path::Path;

use infernum::graph::{optimizer, plan, Arena, ExecutionPlan, Graph, NodeId, WeightStore};
use infernum::rope::precompute_rope_data;
use infernum::{DType, Result};
use infernum_cpu::executor::{execute, KvCacheStore};
use infernum_cpu::tensor::{CpuLinearWeight, CpuTensor};
use infernum_cpu::CpuBackend;

use crate::config::LlamaConfig;
use crate::graph_builder::{
    build_decode_graph, build_prefill_graph, load_graph_weights_gguf,
    load_graph_weights_safetensors,
};

// ---------------------------------------------------------------------------
// Flat KV buffer (used by the Model-trait bridge only)
// ---------------------------------------------------------------------------

/// Flat, growable KV cache used exclusively by [`GraphKvCache`].
///
/// Each K/V entry stores the *full* accumulated sequence per layer, not just
/// the latest token. The buffer is replaced wholesale each decode step because
/// the decode graph's `concat_seq` output already contains the concatenated
/// history up to the current position.
struct KvStore {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    num_kv_heads: usize,
    head_dim: usize,
}

impl KvStore {
    fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            k: vec![Vec::new(); num_layers],
            v: vec![Vec::new(); num_layers],
            num_kv_heads,
            head_dim,
        }
    }

    fn get_layer(&self, layer: usize, kv_len: usize) -> (CpuTensor, CpuTensor) {
        let shape = [kv_len, self.num_kv_heads, self.head_dim];
        let elem = self.num_kv_heads * self.head_dim;
        let k = CpuTensor::from_f32(&shape, &self.k[layer][..kv_len * elem]);
        let v = CpuTensor::from_f32(&shape, &self.v[layer][..kv_len * elem]);
        (k, v)
    }

    /// Replace stored K/V with the full accumulated tensors from decode outputs.
    fn update_from_outputs(&mut self, outputs: &[CpuTensor], num_layers: usize) {
        for layer in 0..num_layers {
            self.k[layer] = outputs[1 + layer].to_f32_vec();
            self.v[layer] = outputs[1 + num_layers + layer].to_f32_vec();
        }
    }
}

// ---------------------------------------------------------------------------
// KV cache node discovery helper
// ---------------------------------------------------------------------------

/// Return the [`NodeId`]s of the KV cache input nodes and `concat_seq` nodes
/// in a decode graph built with `kv_len = 0`.
///
/// The decode graph always has the same structure regardless of `kv_len`:
/// - Input nodes 0–2: `token_id`, `cos`, `sin`
/// - Input nodes 3..: one `[0, kv_heads, head_dim]` K input and one V input per layer
/// - `concat_seq` nodes: one K and one V concat per layer
fn find_kv_cache_node_ids(
    nodes: &[infernum::graph::GraphNode<CpuBackend>],
    num_layers: usize,
) -> (Vec<NodeId>, Vec<NodeId>) {
    let input_ids: Vec<NodeId> = nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.op.name() == "input")
        .skip(3) // skip token_id, cos, sin
        .map(|(i, _)| NodeId::from_index(u32::try_from(i).expect("node index fits in u32")))
        .collect();
    assert_eq!(
        input_ids.len(),
        2 * num_layers,
        "unexpected KV input count in decode graph"
    );

    let concat_ids: Vec<NodeId> = nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.op.name() == "concat_seq")
        .map(|(i, _)| NodeId::from_index(u32::try_from(i).expect("node index fits in u32")))
        .collect();
    assert_eq!(
        concat_ids.len(),
        2 * num_layers,
        "unexpected ConcatSeq count in decode graph"
    );

    (input_ids, concat_ids)
}

// ---------------------------------------------------------------------------
// LlamaGraphEngine
// ---------------------------------------------------------------------------

/// Pre-compiled decode state cached in [`LlamaGraphEngine`].
///
/// Built once in the constructor and reused across all `generate()` calls.
/// Each call allocates a fresh [`Arena`] so that stale residual activations
/// from the previous call never bleed into the new sequence.
struct DecodeCache {
    graph: Graph<CpuBackend>,
    plan: ExecutionPlan,
    logits_id: NodeId,
    cache_input_ids: Vec<NodeId>,
    concat_ids: Vec<NodeId>,
    /// Pre-computed RoPE tables up to `max_position_embeddings`.
    cos: Vec<f32>,
    sin: Vec<f32>,
    half_dim: usize,
}

/// CPU graph-mode engine for Llama-family models.
///
/// Loads weights once, pre-compiles a single decode graph (at `kv_len = 0`),
/// and reuses it for all generation steps — both prompt warmup and autoregressive
/// decode — via [`KvCacheStore`].
pub struct LlamaGraphEngine {
    config: LlamaConfig,
    weights: WeightStore<CpuTensor, CpuLinearWeight>,
    decode: DecodeCache,
}

/// Build the decode cache for a given config.
///
/// Compiles the decode graph once at `kv_len=0`, runs the optimizer and
/// memory planner, discovers KV cache node IDs, and pre-computes the full
/// RoPE table up to `max_position_embeddings`. Stored in [`LlamaGraphEngine`]
/// so that `generate()` pays zero setup cost per call.
///
/// # Panics
///
/// Panics if the compiled decode graph does not contain the expected number of
/// KV cache input nodes or `concat_seq` nodes (indicates a graph-builder bug).
fn build_decode_cache(config: &LlamaConfig) -> DecodeCache {
    let head_dim = config.head_dim();
    let half_dim = head_dim / 2;
    let num_layers = config.num_hidden_layers;

    let (mut graph, _) = build_decode_graph::<CpuBackend>(config, 0, DType::F32);
    optimizer::optimize(&mut graph);
    let ep = plan(&graph);
    let logits_id = graph.output_ids()[0];
    let (cache_input_ids, concat_ids) = find_kv_cache_node_ids(graph.nodes(), num_layers);

    // Pre-compute RoPE up to max_position_embeddings.
    let max_pos = config.max_position_embeddings;
    let (cos, sin) = precompute_rope_data(max_pos, head_dim, config.rope_theta);

    DecodeCache {
        graph,
        plan: ep,
        logits_id,
        cache_input_ids,
        concat_ids,
        cos,
        sin,
        half_dim,
    }
}

impl LlamaGraphEngine {
    /// Load a Llama-family model from a SafeTensors directory.
    ///
    /// # Errors
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or `config.json` cannot be parsed.
    pub fn from_pretrained(model_dir: &Path) -> Result<Self> {
        // Read config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            infernum::Error::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to read {}: {e}", config_path.display()),
            ))
        })?;
        let config: LlamaConfig = serde_json::from_str(&config_str).map_err(|e| {
            infernum::Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to parse config.json: {e}"),
            ))
        })?;

        // Build a dummy 1-token prefill graph to discover weight metadata.
        let (dummy_graph, _) = build_prefill_graph::<CpuBackend>(&config, 1, DType::F32);
        let weights = load_graph_weights_safetensors(&dummy_graph, model_dir, &config)?;

        let decode = build_decode_cache(&config);
        Ok(Self {
            config,
            weights,
            decode,
        })
    }

    /// Load a Llama-family model from a GGUF file.
    ///
    /// Weights are stored in their native quantization format (Q8_0, Q4_0,
    /// F32, etc.) and dequantized lazily inside each matmul kernel.
    ///
    /// # Errors
    /// Returns an error if the GGUF file cannot be opened, cannot be parsed,
    /// or contains unsupported quantization types (e.g. Q6_K).
    pub fn from_gguf(gguf_path: &Path) -> Result<Self> {
        let loader = infernum::weights::gguf::GgufLoader::from_file(
            gguf_path.to_str().ok_or_else(|| {
                infernum::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "GGUF path is not valid UTF-8",
                ))
            })?,
        )?;
        let config = LlamaConfig::from_gguf_metadata(loader.metadata())?;

        // Build a 1-token prefill graph to determine weight slot order.
        let (dummy_graph, _) = build_prefill_graph::<CpuBackend>(&config, 1, DType::F32);
        let weights = load_graph_weights_gguf(&dummy_graph, &config, gguf_path)?;

        let decode = build_decode_cache(&config);
        Ok(Self {
            config,
            weights,
            decode,
        })
    }

    /// Greedy generation using a persistent KV cache.
    ///
    /// Processes the prompt token-by-token (warming up the KV cache), then
    /// continues decoding until EOS or `max_new_tokens` is reached. The same
    /// compiled decode graph is reused for every step; the [`KvCacheStore`]
    /// intercepts KV nodes so no KV data is ever copied through the arena.
    ///
    /// Returns the full token sequence (prompt + generated tokens).
    ///
    /// # Errors
    /// Returns an error if any graph execution step fails.
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        eos_token_id: u32,
    ) -> Result<Vec<u32>> {
        let config = &self.config;
        let num_layers = config.num_hidden_layers;
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();

        let dc = &self.decode;
        let half_dim = dc.half_dim;

        // Fresh per-sequence KV cache — reuses the pre-compiled graph/plan/arena.
        // Pre-allocate to max_position_embeddings to avoid Vec reallocations.
        let mut kv_cache = KvCacheStore::new(
            num_layers,
            num_kv_heads,
            head_dim,
            config.max_position_embeddings,
            dc.cache_input_ids.clone(),
            dc.concat_ids.clone(),
        );

        // Allocate a fresh arena for this generation sequence.
        // The plan arena_size is constant (graph compiled at kv_len=0),
        // so allocation is cheap and reuse across calls would be safe,
        // but a fresh arena avoids any stale-data hazards during debugging.
        let mut arena = Arena::new(dc.plan.arena_size);

        // Use the pre-computed RoPE table when the sequence fits; otherwise
        // extend on the fly (rare: only when prompt + new tokens > max_position_embeddings).
        let max_pos = prompt_ids.len() + max_new_tokens;
        let rope_extended;
        let (cos, sin): (&[f32], &[f32]) = if max_pos * half_dim <= dc.cos.len() {
            (&dc.cos, &dc.sin)
        } else {
            rope_extended = precompute_rope_data(max_pos, head_dim, config.rope_theta);
            (&rope_extended.0, &rope_extended.1)
        };

        // Helper closure: run one decode step at position `pos` with token `token`.
        let run_step = |pos: usize,
                        token: u32,
                        kv_cache: &mut KvCacheStore,
                        arena: &mut Arena|
         -> Result<Vec<CpuTensor>> {
            let cos_start = pos * half_dim;
            let inputs = vec![
                CpuTensor::from_u32(&[1], &[token]),
                CpuTensor::from_f32(&[1, half_dim], &cos[cos_start..cos_start + half_dim]),
                CpuTensor::from_f32(&[1, half_dim], &sin[cos_start..cos_start + half_dim]),
            ];
            execute(
                &dc.plan,
                dc.graph.nodes(),
                arena,
                &self.weights,
                &inputs,
                &[dc.logits_id],
                Some(kv_cache),
            )
        };

        let mut token_ids = prompt_ids.to_vec();

        // ── Prompt warmup ────────────────────────────────────────────────────
        // Process all prompt tokens except the last (no need for their logits).
        for (pos, &token) in prompt_ids[..prompt_ids.len().saturating_sub(1)]
            .iter()
            .enumerate()
        {
            run_step(pos, token, &mut kv_cache, &mut arena)?;
        }

        // Process the last prompt token and pick the first generated token.
        let last_prompt_pos = prompt_ids.len().saturating_sub(1);
        let last_prompt_token = *prompt_ids.last().unwrap_or(&0);
        let outputs = run_step(
            last_prompt_pos,
            last_prompt_token,
            &mut kv_cache,
            &mut arena,
        )?;
        let first_token = argmax(outputs[0].as_f32_slice());
        if first_token == eos_token_id {
            return Ok(token_ids);
        }
        token_ids.push(first_token);

        // ── Autoregressive decode ─────────────────────────────────────────────
        for step in 0..max_new_tokens.saturating_sub(1) {
            let pos = prompt_ids.len() + step;
            let last_token = *token_ids.last().unwrap();
            let outputs = run_step(pos, last_token, &mut kv_cache, &mut arena)?;
            let next_token = argmax(outputs[0].as_f32_slice());
            if next_token == eos_token_id {
                break;
            }
            token_ids.push(next_token);
        }

        Ok(token_ids)
    }

    /// Return a reference to the model configuration.
    #[must_use]
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }
}

/// Returns the index of the maximum element in `slice` as a `u32`.
/// Returns 0 if `slice` is empty.
///
/// # Panics
///
/// Panics if the number of elements exceeds `u32::MAX` (unreachable in practice).
fn argmax(slice: &[f32]) -> u32 {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| u32::try_from(i).expect("index fits in u32"))
}

// ---------------------------------------------------------------------------
// Model trait implementation — graph-mode bridge to infernum-runtime
// ---------------------------------------------------------------------------

/// A graph-mode KV cache that satisfies the `Model::KvCache` associated type.
///
/// Wraps a [`KvStore`] behind a `Mutex` so it can be passed as `&mut Self::KvCache`
/// across the `Model` trait boundary without requiring `LlamaGraphEngine` itself
/// to be mutable.  The mutex is always uncontested (single-threaded engine
/// worker), so it adds negligible overhead.
pub struct GraphKvCache {
    inner: std::sync::Mutex<KvStore>,
    /// Tracks how many tokens have been committed to the cache (seq len so far).
    committed: std::sync::atomic::AtomicUsize,
}

impl GraphKvCache {
    fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            inner: std::sync::Mutex::new(KvStore::new(num_layers, num_kv_heads, head_dim)),
            committed: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn seq_len(&self) -> usize {
        self.committed.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl infernum::Model for LlamaGraphEngine {
    type B = infernum_cpu::CpuBackend;
    type KvCache = GraphKvCache;

    fn config(&self) -> infernum::ModelConfig {
        infernum::ModelConfig {
            num_layers: self.config.num_hidden_layers,
            max_seq_len: self.config.max_position_embeddings,
            num_kv_heads: self.config.num_kv_heads(),
            head_dim: self.config.head_dim(),
            eos_token_id: self.config.eos_token_id,
            cache_dtype: infernum::DType::F32,
        }
    }

    fn device(&self) -> &() {
        &()
    }

    fn allocate_kv_cache(
        &self,
        _block_config: &infernum::block_allocator::BlockConfig,
    ) -> infernum::Result<GraphKvCache> {
        Ok(GraphKvCache::new(
            self.config.num_hidden_layers,
            self.config.num_kv_heads(),
            self.config.head_dim(),
        ))
    }

    /// Full no-cache forward pass.  Returns logits for all positions.
    fn forward(
        &self,
        input_ids: &[u32],
    ) -> infernum::Result<<infernum_cpu::CpuBackend as infernum::backend::Backend>::Logits> {
        use infernum::Backend as _;
        let config = &self.config;
        let seq_len = input_ids.len();
        let head_dim = config.head_dim();
        let half_dim = head_dim / 2;

        let (mut graph, _) =
            build_prefill_graph::<infernum_cpu::CpuBackend>(config, seq_len, DType::F32);
        optimizer::optimize(&mut graph);
        let ep = plan(&graph);

        let (cos_data, sin_data) = precompute_rope_data(seq_len, head_dim, config.rope_theta);
        let input_ids_t = CpuTensor::from_u32(&[seq_len], input_ids);
        let cos_t = CpuTensor::from_f32(&[seq_len, half_dim], &cos_data);
        let sin_t = CpuTensor::from_f32(&[seq_len, half_dim], &sin_data);
        let inputs = vec![input_ids_t, cos_t, sin_t];

        let mut arena = Arena::new(ep.arena_size);
        let output_nodes = graph.output_ids().to_vec();
        let outputs = execute(
            &ep,
            graph.nodes(),
            &mut arena,
            &self.weights,
            &inputs,
            &output_nodes,
            None,
        )?;

        let logits_tensor = outputs
            .into_iter()
            .next()
            .expect("prefill graph has no output");
        Ok(infernum_cpu::CpuBackend::logits_from_tensor(logits_tensor))
    }

    /// Prefill: process the full prompt, write K/V into `kv_cache`, return last-position logits.
    fn forward_prefill(
        &self,
        input_ids: &[u32],
        kv_cache: &mut GraphKvCache,
        _runtime_state: &mut (),
        _block_table: &infernum::block_allocator::BlockTable,
        _start_pos: usize,
    ) -> infernum::Result<<infernum_cpu::CpuBackend as infernum::backend::Backend>::Logits> {
        use infernum::Backend as _;
        let config = &self.config;
        let seq_len = input_ids.len();
        let head_dim = config.head_dim();
        let half_dim = head_dim / 2;
        let num_layers = config.num_hidden_layers;
        let vocab_size = config.vocab_size;

        let (mut graph, _) =
            build_prefill_graph::<infernum_cpu::CpuBackend>(config, seq_len, DType::F32);
        optimizer::optimize(&mut graph);
        let ep = plan(&graph);

        let (cos_data, sin_data) = precompute_rope_data(seq_len, head_dim, config.rope_theta);
        let input_ids_t = CpuTensor::from_u32(&[seq_len], input_ids);
        let cos_t = CpuTensor::from_f32(&[seq_len, half_dim], &cos_data);
        let sin_t = CpuTensor::from_f32(&[seq_len, half_dim], &sin_data);
        let inputs = vec![input_ids_t, cos_t, sin_t];

        let mut arena = Arena::new(ep.arena_size);
        let output_nodes = graph.output_ids().to_vec();
        let outputs = execute(
            &ep,
            graph.nodes(),
            &mut arena,
            &self.weights,
            &inputs,
            &output_nodes,
            None,
        )?;

        // Store updated K/V (outputs[1..1+L] = K per layer, outputs[1+L..1+2L] = V per layer)
        {
            let mut store = kv_cache.inner.lock().expect("GraphKvCache mutex poisoned");
            store.update_from_outputs(&outputs, num_layers);
        }
        kv_cache
            .committed
            .store(seq_len, std::sync::atomic::Ordering::Relaxed);

        // Return last-position logits (shape [1, vocab_size])
        let logits_vec = outputs[0].to_f32_vec();
        let last_row = logits_vec[(seq_len - 1) * vocab_size..seq_len * vocab_size].to_vec();
        let last_tensor = CpuTensor::from_f32(&[1, vocab_size], &last_row);
        Ok(infernum_cpu::CpuBackend::logits_from_tensor(last_tensor))
    }

    /// Batched decode.  Only batch_size=1 is supported on the graph CPU path.
    ///
    /// The block table inputs are ignored — the graph engine tracks sequence
    /// length via the K/V cache length.
    #[allow(clippy::too_many_arguments)]
    fn forward_batch_decode(
        &self,
        token_ids: &CpuTensor,
        kv_cache: &mut GraphKvCache,
        _runtime_state: &mut (),
        _block_tables: &CpuTensor,
        _seq_lens: &CpuTensor,
        _positions: &CpuTensor,
        batch_size: usize,
        _max_blocks_per_seq: usize,
        _max_seq_len: usize,
    ) -> infernum::Result<<infernum_cpu::CpuBackend as infernum::backend::Backend>::Logits> {
        use infernum::Backend as _;
        if batch_size != 1 {
            return Err(infernum::Error::InvalidShape(format!(
                "LlamaGraphEngine only supports batch_size=1 in decode, got {batch_size}"
            )));
        }

        let config = &self.config;
        let head_dim = config.head_dim();
        let half_dim = head_dim / 2;
        let num_layers = config.num_hidden_layers;
        let num_kv_heads = config.num_kv_heads();
        let vocab_size = config.vocab_size;

        let kv_len = kv_cache.seq_len();

        let (mut graph, _) =
            build_decode_graph::<infernum_cpu::CpuBackend>(config, kv_len, DType::F32);
        optimizer::optimize(&mut graph);
        let ep = plan(&graph);

        let pos = kv_len;
        let (cos_data, sin_data) = precompute_rope_data(pos + 1, head_dim, config.rope_theta);
        let cos_row = &cos_data[pos * half_dim..(pos + 1) * half_dim];
        let sin_row = &sin_data[pos * half_dim..(pos + 1) * half_dim];

        // Extract the single token ID from the tensor.
        let token_data = token_ids.as_u32_slice();
        let last_token = token_data[0];

        let input_id_t = CpuTensor::from_u32(&[1], &[last_token]);
        let cos_t = CpuTensor::from_f32(&[1, half_dim], cos_row);
        let sin_t = CpuTensor::from_f32(&[1, half_dim], sin_row);

        let mut inputs = vec![input_id_t, cos_t, sin_t];
        {
            let store = kv_cache.inner.lock().expect("GraphKvCache mutex poisoned");
            for layer in 0..num_layers {
                let (k, v) = store.get_layer(layer, kv_len);
                inputs.push(k);
                inputs.push(v);
            }
        }

        // Placeholder inputs for empty cache (first decode step after prefill)
        if kv_len == 0 {
            for _ in 0..num_layers {
                inputs.push(CpuTensor::from_f32(&[0, num_kv_heads, head_dim], &[]));
                inputs.push(CpuTensor::from_f32(&[0, num_kv_heads, head_dim], &[]));
            }
        }

        let mut arena = Arena::new(ep.arena_size.max(self.decode.plan.arena_size));
        let output_nodes = graph.output_ids().to_vec();
        let outputs = execute(
            &ep,
            graph.nodes(),
            &mut arena,
            &self.weights,
            &inputs,
            &output_nodes,
            None,
        )?;

        // Update K/V cache
        {
            let mut store = kv_cache.inner.lock().expect("GraphKvCache mutex poisoned");
            store.update_from_outputs(&outputs, num_layers);
        }
        kv_cache
            .committed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Return logits (outputs[0], shape [1, vocab_size])
        let logits_vec = outputs[0].to_f32_vec();
        let logit_tensor = CpuTensor::from_f32(&[1, vocab_size], &logits_vec[..vocab_size]);
        Ok(infernum_cpu::CpuBackend::logits_from_tensor(logit_tensor))
    }
}
