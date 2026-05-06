//! Generic CPU graph-mode inference engine.
//!
//! [`GraphEngine<C>`] is the single shared implementation of the graph-mode
//! inference loop for all model families. Each model crate provides an
//! `impl GraphEngineConfig for XxxConfig` and publishes a type alias
//! `XxxGraphEngine = GraphEngine<XxxConfig>`.
//!
//! The engine pre-compiles a single decode graph at construction time and
//! reuses it for every token step via [`KvCacheStore`], which intercepts KV
//! cache nodes in the graph and maintains a persistent, growing KV buffer
//! without copying it through the arena each step.
//!
//! # Usage
//!
//! Model crates expose named type aliases and an extension trait with model-specific
//! constructors. Use those instead of constructing `GraphEngine` directly.
//!
//! ```no_run
//! use std::path::Path;
//! use infernum_llama::{LlamaGraphEngine, LlamaGraphEngineExt as _};
//!
//! let engine = LlamaGraphEngine::from_pretrained(Path::new("/path/to/model")).unwrap();
//! let tokens = engine.generate(&[1, 1234, 567], 128, 2).unwrap();
//! println!("{tokens:?}");
//! ```

use std::path::Path;

use infernum::graph::{optimizer, plan, Arena, ExecutionPlan, Graph, NodeId, WeightStore};
use infernum::rope::precompute_rope_data;
use infernum::{DType, Result};

use crate::executor::{execute, KvCacheStore};
use crate::tensor::{CpuLinearWeight, CpuTensor};
use crate::CpuBackend;

// ---------------------------------------------------------------------------
// GraphEngineConfig — the abstraction boundary
// ---------------------------------------------------------------------------

/// Configuration interface required to drive [`GraphEngine`].
///
/// Implement this trait for your model's config struct to get a fully working
/// `XxxGraphEngine = GraphEngine<XxxConfig>` with zero additional code.
///
/// # Required methods
///
/// The scalar getters expose the 7 fields the engine loop needs. The three
/// builder methods delegate to the model-specific graph builder functions.
pub trait GraphEngineConfig: Send + Sync + 'static {
    // ── Scalar getters ───────────────────────────────────────────────────

    /// Number of transformer layers.
    fn num_hidden_layers(&self) -> usize;

    /// Number of KV heads (after GQA grouping, if any).
    fn num_kv_heads(&self) -> usize;

    /// Per-head dimension for K and V.
    fn head_dim(&self) -> usize;

    /// Maximum supported sequence length (used to size the pre-computed RoPE table).
    fn max_position_embeddings(&self) -> usize;

    /// RoPE base theta.
    fn rope_theta(&self) -> f32;

    /// Vocabulary size (used in the `Model` trait forward pass).
    fn vocab_size(&self) -> usize;

    /// EOS token ID (used to terminate greedy decode).
    fn eos_token_id(&self) -> u32;

    // ── Graph builder delegates ──────────────────────────────────────────

    /// Build a prefill graph for the given `seq_len`.
    ///
    /// Used once at construction time with `seq_len = 1` to discover weight
    /// metadata. The graph itself is discarded after weight loading.
    fn build_prefill_graph(&self, seq_len: usize) -> Graph<CpuBackend>;

    /// Build a decode graph with the given initial KV length.
    ///
    /// Used at construction time with `kv_len = 0` to compile the persistent
    /// decode graph.
    fn build_decode_graph(&self, kv_len: usize) -> Graph<CpuBackend>;

    /// Load weights from a `SafeTensors` model directory.
    ///
    /// The provided `dummy_graph` is the 1-token prefill graph used solely to
    /// discover weight slot order and metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if any weight file is missing or cannot be parsed.
    fn load_weights_safetensors(
        &self,
        dummy_graph: &Graph<CpuBackend>,
        model_dir: &Path,
    ) -> Result<WeightStore<CpuTensor, CpuLinearWeight>>;

    /// Load weights from a GGUF file.
    ///
    /// Returns `None` by default (no `GGUF` support). Override to enable.
    ///
    /// The provided `dummy_graph` is the 1-token prefill graph used solely to
    /// discover weight slot order and metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the `GGUF` file cannot be opened, parsed, or contains
    /// unsupported quantization types.
    fn load_weights_gguf(
        &self,
        dummy_graph: &Graph<CpuBackend>,
        gguf_path: &Path,
    ) -> Option<Result<WeightStore<CpuTensor, CpuLinearWeight>>> {
        let _ = (dummy_graph, gguf_path);
        None
    }
}

// ---------------------------------------------------------------------------
// Internal structures
// ---------------------------------------------------------------------------

/// Flat, growable KV cache used by [`GraphKvCache`].
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
    ///
    /// `outputs[1..=num_layers]` are the updated K tensors;
    /// `outputs[1+num_layers..1+2*num_layers]` are the updated V tensors.
    fn update_from_outputs(&mut self, outputs: &[CpuTensor], num_layers: usize) {
        for layer in 0..num_layers {
            self.k[layer] = outputs[1 + layer].to_f32_vec();
            self.v[layer] = outputs[1 + num_layers + layer].to_f32_vec();
        }
    }
}

/// Return the [`NodeId`]s of the KV cache input nodes and `concat_seq` nodes
/// in a decode graph built with `kv_len = 0`.
///
/// The decode graph always has the same structure regardless of `kv_len`:
/// - Input nodes 0–2: `token_id`, `cos`, `sin`
/// - Input nodes 3..: one `[0, kv_heads, head_dim]` K input and one V input per layer
/// - `concat_seq` nodes: one K and one V concat per layer
pub(crate) fn find_kv_cache_node_ids(
    nodes: &[infernum::graph::GraphNode<CpuBackend>],
    num_layers: usize,
) -> (Vec<NodeId>, Vec<NodeId>) {
    let input_ids: Vec<NodeId> = nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.op.name() == "input")
        .skip(3) // skip token_id, cos, sin
        .map(|(i, _)| NodeId::from_index(u32::try_from(i).expect("node index fits u32")))
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
        .map(|(i, _)| NodeId::from_index(u32::try_from(i).expect("node index fits u32")))
        .collect();
    assert_eq!(
        concat_ids.len(),
        2 * num_layers,
        "unexpected ConcatSeq count in decode graph"
    );

    (input_ids, concat_ids)
}

/// Pre-compiled decode state cached in [`GraphEngine`].
///
/// Built once in the constructor and reused across all `generate()` calls.
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

fn build_decode_cache<C: GraphEngineConfig>(config: &C) -> DecodeCache {
    let head_dim = config.head_dim();
    let half_dim = head_dim / 2;
    let num_layers = config.num_hidden_layers();

    let mut graph = config.build_decode_graph(0);
    optimizer::optimize(&mut graph);
    let ep = plan(&graph);
    let logits_id = graph.output_ids()[0];
    let (cache_input_ids, concat_ids) = find_kv_cache_node_ids(graph.nodes(), num_layers);

    let max_pos = config.max_position_embeddings();
    let (cos, sin) = precompute_rope_data(max_pos, head_dim, config.rope_theta());

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

// ---------------------------------------------------------------------------
// GraphEngine
// ---------------------------------------------------------------------------

/// CPU graph-mode engine for any model family that implements [`GraphEngineConfig`].
///
/// Loads weights once, pre-compiles a single decode graph (at `kv_len = 0`),
/// and reuses it for all generation steps — both prompt warmup and autoregressive
/// decode — via [`KvCacheStore`].
pub struct GraphEngine<C: GraphEngineConfig> {
    config: C,
    weights: WeightStore<CpuTensor, CpuLinearWeight>,
    decode: DecodeCache,
}

impl<C: GraphEngineConfig> GraphEngine<C> {
    /// Load a model from a `SafeTensors` directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or the config file cannot be parsed (config parsing is the caller's
    /// responsibility — pass an already-parsed config and model dir).
    pub fn from_config_and_dir(config: C, model_dir: &Path) -> Result<Self> {
        let dummy_graph = config.build_prefill_graph(1);
        let weights = config.load_weights_safetensors(&dummy_graph, model_dir)?;
        let decode = build_decode_cache(&config);
        Ok(Self {
            config,
            weights,
            decode,
        })
    }

    /// Load a model from a GGUF file using an already-parsed config.
    ///
    /// This is the low-level constructor used by model-specific `from_gguf`
    /// helpers (e.g. [`infernum_llama::LlamaGraphEngineExt::from_gguf`]).
    /// Requires the config type to override [`GraphEngineConfig::load_weights_gguf`].
    ///
    /// # Errors
    ///
    /// Returns an error if `load_weights_gguf` returns an error, or if the
    /// config type does not support `GGUF` loading (returns `None`).
    pub fn from_gguf_with_config(config: C, gguf_path: &Path) -> Result<Self> {
        let dummy_graph = config.build_prefill_graph(1);
        let weights = config
            .load_weights_gguf(&dummy_graph, gguf_path)
            .ok_or_else(|| {
                infernum::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "this model config does not support GGUF loading",
                ))
            })??;
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
    ///
    /// Returns an error if any graph execution step fails.
    ///
    /// # Panics
    ///
    /// Panics if `prompt_ids` is empty.
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        eos_token_id: u32,
    ) -> Result<Vec<u32>> {
        let config = &self.config;
        let num_layers = config.num_hidden_layers();
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();

        let dc = &self.decode;
        let half_dim = dc.half_dim;

        let mut kv_cache = KvCacheStore::new(
            num_layers,
            num_kv_heads,
            head_dim,
            config.max_position_embeddings(),
            dc.cache_input_ids.clone(),
            dc.concat_ids.clone(),
        );

        let mut arena = Arena::new(dc.plan.arena_size);

        let max_pos = prompt_ids.len() + max_new_tokens;
        let rope_extended;
        let (cos, sin): (&[f32], &[f32]) = if max_pos * half_dim <= dc.cos.len() {
            (&dc.cos, &dc.sin)
        } else {
            rope_extended = precompute_rope_data(max_pos, head_dim, config.rope_theta());
            (&rope_extended.0, &rope_extended.1)
        };

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

        // Prompt warmup: process all tokens except the last.
        for (pos, &token) in prompt_ids[..prompt_ids.len().saturating_sub(1)]
            .iter()
            .enumerate()
        {
            run_step(pos, token, &mut kv_cache, &mut arena)?;
        }

        // Last prompt token — get first generated token.
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

        let debug = std::env::var("INFERNUM_DEBUG_KV").is_ok();

        // Autoregressive decode.
        for step in 0..max_new_tokens.saturating_sub(1) {
            let pos = prompt_ids.len() + step;
            let last_token = *token_ids.last().unwrap();
            let outputs = run_step(pos, last_token, &mut kv_cache, &mut arena)?;
            let next_token = argmax(outputs[0].as_f32_slice());
            if debug {
                eprintln!(
                    "[DEBUG] step={step} pos={pos} token={last_token} -> next={next_token} kv_len={}",
                    kv_cache.len()
                );
            }
            if next_token == eos_token_id {
                break;
            }
            token_ids.push(next_token);
        }

        Ok(token_ids)
    }

    /// Return a reference to the model configuration.
    #[must_use]
    pub fn config(&self) -> &C {
        &self.config
    }
}

fn argmax(slice: &[f32]) -> u32 {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| {
            u32::try_from(i).expect("vocab size exceeds u32")
        })
}

// ---------------------------------------------------------------------------
// Model trait bridge (GraphKvCache + impl Model for GraphEngine<C>)
// ---------------------------------------------------------------------------

/// A graph-mode KV cache that satisfies the `Model::KvCache` associated type.
///
/// Wraps a [`KvStore`] behind a `Mutex` so it can be passed as `&mut Self::KvCache`
/// across the `Model` trait boundary without requiring `GraphEngine` itself
/// to be mutable. The mutex is always uncontested (single-threaded engine
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

impl<C: GraphEngineConfig> infernum::Model for GraphEngine<C> {
    type B = CpuBackend;
    type KvCache = GraphKvCache;

    fn config(&self) -> infernum::ModelConfig {
        let c = &self.config;
        infernum::ModelConfig {
            num_layers: c.num_hidden_layers(),
            max_seq_len: c.max_position_embeddings(),
            num_kv_heads: c.num_kv_heads(),
            head_dim: c.head_dim(),
            eos_token_id: c.eos_token_id(),
            cache_dtype: DType::F32,
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
            self.config.num_hidden_layers(),
            self.config.num_kv_heads(),
            self.config.head_dim(),
        ))
    }

    /// Full no-cache forward pass. Returns logits for all positions.
    fn forward(
        &self,
        input_ids: &[u32],
    ) -> infernum::Result<<CpuBackend as infernum::backend::Backend>::Logits> {
        use infernum::Backend as _;
        let config = &self.config;
        let seq_len = input_ids.len();
        let head_dim = config.head_dim();
        let half_dim = head_dim / 2;

        let mut graph = config.build_prefill_graph(seq_len);
        optimizer::optimize(&mut graph);
        let ep = plan(&graph);

        let (cos_data, sin_data) = precompute_rope_data(seq_len, head_dim, config.rope_theta());
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
        Ok(CpuBackend::logits_from_tensor(logits_tensor))
    }

    /// Prefill: process the full prompt, write K/V into `kv_cache`, return last-position logits.
    fn forward_prefill(
        &self,
        input_ids: &[u32],
        kv_cache: &mut GraphKvCache,
        _runtime_state: &mut (),
        _block_table: &infernum::block_allocator::BlockTable,
        _start_pos: usize,
    ) -> infernum::Result<<CpuBackend as infernum::backend::Backend>::Logits> {
        use infernum::Backend as _;
        // The prefill graph only outputs logits — it does not produce KV cache
        // tensors. To correctly populate the KV cache for subsequent decode
        // steps, we process prompt tokens sequentially through the decode graph
        // (the same path as forward_batch_decode). This is O(n²) in prompt
        // length but is functionally correct with the existing graph structure.
        let config = &self.config;
        let head_dim = config.head_dim();
        let half_dim = head_dim / 2;
        let num_layers = config.num_hidden_layers();
        let vocab_size = config.vocab_size();
        let seq_len = input_ids.len();

        let (cos_data, sin_data) = precompute_rope_data(seq_len, head_dim, config.rope_theta());

        let mut last_logits: Option<<CpuBackend as infernum::backend::Backend>::Logits> = None;

        for (pos, &token) in input_ids.iter().enumerate() {
            let kv_len = kv_cache.seq_len();

            let mut graph = config.build_decode_graph(kv_len);
            optimizer::optimize(&mut graph);
            let ep = plan(&graph);

            let cos_row = &cos_data[pos * half_dim..(pos + 1) * half_dim];
            let sin_row = &sin_data[pos * half_dim..(pos + 1) * half_dim];

            let input_id_t = CpuTensor::from_u32(&[1], &[token]);
            let cos_t = CpuTensor::from_f32(&[1, half_dim], cos_row);
            let sin_t = CpuTensor::from_f32(&[1, half_dim], sin_row);

            let mut inputs = vec![input_id_t, cos_t, sin_t];
            {
                // When kv_len == 0, get_layer returns empty [0, num_kv_heads, head_dim] tensors,
                // which is exactly what the decode graph expects for an empty cache.
                let store = kv_cache.inner.lock().expect("GraphKvCache mutex poisoned");
                for layer in 0..num_layers {
                    let (k, v) = store.get_layer(layer, kv_len);
                    inputs.push(k);
                    inputs.push(v);
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

            {
                let mut store = kv_cache.inner.lock().expect("GraphKvCache mutex poisoned");
                store.update_from_outputs(&outputs, num_layers);
            }
            kv_cache
                .committed
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            let logits_vec = outputs[0].to_f32_vec();
            let logit_tensor = CpuTensor::from_f32(&[1, vocab_size], &logits_vec[..vocab_size]);
            last_logits = Some(CpuBackend::logits_from_tensor(logit_tensor));
        }

        last_logits.ok_or_else(|| {
            infernum::Error::InvalidShape("forward_prefill called with empty input_ids".into())
        })
    }

    /// Batched decode. Only `batch_size=1` is supported on the graph CPU path.
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
    ) -> infernum::Result<<CpuBackend as infernum::backend::Backend>::Logits> {
        use infernum::Backend as _;
        if batch_size != 1 {
            return Err(infernum::Error::InvalidShape(format!(
                "GraphEngine only supports batch_size=1 in decode, got {batch_size}"
            )));
        }

        let config = &self.config;
        let head_dim = config.head_dim();
        let half_dim = head_dim / 2;
        let num_layers = config.num_hidden_layers();
        let num_kv_heads = config.num_kv_heads();
        let vocab_size = config.vocab_size();

        let kv_len = kv_cache.seq_len();

        let mut graph = config.build_decode_graph(kv_len);
        optimizer::optimize(&mut graph);
        let ep = plan(&graph);

        let pos = kv_len;
        let (cos_data, sin_data) = precompute_rope_data(pos + 1, head_dim, config.rope_theta());
        let cos_row = &cos_data[pos * half_dim..(pos + 1) * half_dim];
        let sin_row = &sin_data[pos * half_dim..(pos + 1) * half_dim];

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

        // Placeholder inputs for empty cache (first decode step after prefill).
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

        // Update K/V cache.
        {
            let mut store = kv_cache.inner.lock().expect("GraphKvCache mutex poisoned");
            store.update_from_outputs(&outputs, num_layers);
        }
        kv_cache
            .committed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Return logits (outputs[0], shape [1, vocab_size]).
        let logits_vec = outputs[0].to_f32_vec();
        let logit_tensor = CpuTensor::from_f32(&[1, vocab_size], &logits_vec[..vocab_size]);
        Ok(CpuBackend::logits_from_tensor(logit_tensor))
    }
}
