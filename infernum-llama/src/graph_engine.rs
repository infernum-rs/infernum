//! CPU graph-mode inference engine for the Llama model family.
//!
//! [`LlamaGraphEngine`] is a self-contained, standalone alternative to
//! [`crate::LlamaModel`] + [`infernum_runtime::Engine`] for CPU inference.
//! It pre-compiles both a prefill graph and a decode graph at construction
//! time, then uses those compiled plans for every subsequent generation call.
//!
//! Unlike the eager path, this engine does not use paged KV caches or the
//! runtime scheduler. Instead it maintains a simple flat
//! [`KvStore`] that grows by one position per decode step.
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

use infernum::backend::Backend;
use infernum::graph::{optimizer, plan, Arena, WeightStore};
use infernum::rope::precompute_rope_data;
use infernum::{DType, Result};
use infernum_cpu::executor::execute;
use infernum_cpu::tensor::{CpuLinearWeight, CpuTensor};
use infernum_cpu::CpuBackend;

use crate::config::LlamaConfig;
use crate::graph_builder::{
    build_decode_graph, build_prefill_graph, load_graph_weights_safetensors,
};

// ---------------------------------------------------------------------------
// Per-layer KV buffer
// ---------------------------------------------------------------------------

/// Simple flat KV cache — one F32 vec per (layer, k/v) pair.
///
/// Each entry has shape `[current_seq_len, num_kv_heads, head_dim]`.
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

    fn len(&self) -> usize {
        // All layers stay in sync; use layer 0 as reference.
        self.k[0].len() / (self.num_kv_heads * self.head_dim)
    }

    /// Reset the cache so the engine can start a new sequence.
    #[allow(dead_code)]
    fn clear(&mut self) {
        for layer_k in &mut self.k {
            layer_k.clear();
        }
        for layer_v in &mut self.v {
            layer_v.clear();
        }
    }

    fn get_layer(&self, layer: usize, kv_len: usize) -> (CpuTensor, CpuTensor) {
        let shape = [kv_len, self.num_kv_heads, self.head_dim];
        let elem = self.num_kv_heads * self.head_dim;
        let k = CpuTensor::from_f32(&shape, &self.k[layer][..kv_len * elem]);
        let v = CpuTensor::from_f32(&shape, &self.v[layer][..kv_len * elem]);
        (k, v)
    }

    /// Append updated K/V tensors (output of the decode graph) for all layers.
    fn update_from_outputs(&mut self, outputs: &[CpuTensor], num_layers: usize) {
        // outputs[0] = logits
        // outputs[1..=num_layers] = k_layer_0..k_layer_n
        // outputs[num_layers+1..=2*num_layers] = v_layer_0..v_layer_n
        for layer in 0..num_layers {
            let k_tensor = &outputs[1 + layer];
            let v_tensor = &outputs[1 + num_layers + layer];
            self.k[layer] = k_tensor.to_f32_vec();
            self.v[layer] = v_tensor.to_f32_vec();
        }
    }
}

// ---------------------------------------------------------------------------
// LlamaGraphEngine
// ---------------------------------------------------------------------------

/// CPU graph-mode engine for Llama-family models.
///
/// Loads weights once, compiles prefill and decode plans at construction,
/// and executes generation without per-token allocation.
pub struct LlamaGraphEngine {
    config: LlamaConfig,
    weights: WeightStore<CpuTensor, CpuLinearWeight>,
    prefill_arena_size: usize,
    decode_arena_size: usize,
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

        // Pre-compute representative arena sizes.
        // Prefill arena: use a 512-token graph as a proxy (actual size scales with seq_len).
        let representative_prefill_len = 512.min(config.max_position_embeddings);
        let (mut pf_graph, _) =
            build_prefill_graph::<CpuBackend>(&config, representative_prefill_len, DType::F32);
        optimizer::optimize(&mut pf_graph);
        let prefill_arena_size = plan(&pf_graph).arena_size;

        // Decode arena: single decode step with kv_len = representative_prefill_len.
        let (mut dec_graph, _) =
            build_decode_graph::<CpuBackend>(&config, representative_prefill_len, DType::F32);
        optimizer::optimize(&mut dec_graph);
        let decode_arena_size = plan(&dec_graph).arena_size;

        Ok(Self {
            config,
            weights,
            prefill_arena_size,
            decode_arena_size,
        })
    }

    /// Greedy generation with a simple flat KV cache.
    ///
    /// Runs a full-sequence prefill, then decode steps one token at a time.
    /// Returns the full token sequence (prompt + generated tokens).
    ///
    /// # Arguments
    /// * `prompt_ids` — Tokenized prompt token IDs.
    /// * `max_new_tokens` — Maximum number of tokens to generate.
    /// * `eos_token_id` — Stop generation when this token is produced.
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
        let head_dim = config.head_dim();
        let half_dim = head_dim / 2;
        let num_layers = config.num_hidden_layers;
        let num_kv_heads = config.num_kv_heads();
        let vocab_size = config.vocab_size;

        let mut kv = KvStore::new(num_layers, num_kv_heads, head_dim);
        let mut token_ids = prompt_ids.to_vec();

        // ── Prefill ──────────────────────────────────────────────────────────
        {
            let seq_len = token_ids.len();
            let (mut graph, _) = build_prefill_graph::<CpuBackend>(config, seq_len, DType::F32);
            optimizer::optimize(&mut graph);
            let ep = plan(&graph);

            let (cos_data, sin_data) = precompute_rope_data(seq_len, head_dim, config.rope_theta);
            let input_ids_t = CpuTensor::from_u32(&[seq_len], &token_ids);
            let cos_t = CpuTensor::from_f32(&[seq_len, half_dim], &cos_data);
            let sin_t = CpuTensor::from_f32(&[seq_len, half_dim], &sin_data);
            let inputs = vec![input_ids_t, cos_t, sin_t];

            let arena_size = ep.arena_size.max(self.prefill_arena_size);
            let mut arena = Arena::new(arena_size);
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

            // Argmax on last position
            let logits_vec = outputs[0].to_f32_vec();
            let last_row = &logits_vec[(seq_len - 1) * vocab_size..seq_len * vocab_size];
            let next_token = argmax(last_row);
            if next_token == eos_token_id {
                return Ok(token_ids);
            }
            token_ids.push(next_token);
        }

        // ── Decode loop ───────────────────────────────────────────────────────
        for _ in 0..max_new_tokens.saturating_sub(1) {
            let kv_len = kv.len();
            let (mut graph, _) = build_decode_graph::<CpuBackend>(config, kv_len, DType::F32);
            optimizer::optimize(&mut graph);
            let ep = plan(&graph);

            let pos = kv_len; // 0-based position of the new token
            let (cos_data, sin_data) = precompute_rope_data(pos + 1, head_dim, config.rope_theta);
            // Slice to just the current position
            let cos_row = &cos_data[pos * half_dim..(pos + 1) * half_dim];
            let sin_row = &sin_data[pos * half_dim..(pos + 1) * half_dim];

            let last_token = *token_ids.last().unwrap();
            let input_id_t = CpuTensor::from_u32(&[1], &[last_token]);
            let cos_t = CpuTensor::from_f32(&[1, half_dim], cos_row);
            let sin_t = CpuTensor::from_f32(&[1, half_dim], sin_row);

            // KV cache inputs: [k_layer_0, v_layer_0, k_layer_1, v_layer_1, ...]
            let mut inputs = vec![input_id_t, cos_t, sin_t];
            if kv_len > 0 {
                for layer in 0..num_layers {
                    let (k, v) = kv.get_layer(layer, kv_len);
                    inputs.push(k);
                    inputs.push(v);
                }
            } else {
                // First decode step: empty cache — still need placeholder inputs
                for _ in 0..num_layers {
                    inputs.push(CpuTensor::from_f32(&[0, num_kv_heads, head_dim], &[]));
                    inputs.push(CpuTensor::from_f32(&[0, num_kv_heads, head_dim], &[]));
                }
            }

            let arena_size = ep.arena_size.max(self.decode_arena_size);
            let mut arena = Arena::new(arena_size);
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

            // Update KV cache from outputs
            kv.update_from_outputs(&outputs, num_layers);

            // Argmax on logits (outputs[0], shape [1, vocab_size])
            let logits_vec = outputs[0].to_f32_vec();
            let next_token = argmax(&logits_vec[..vocab_size]);
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

fn argmax(slice: &[f32]) -> u32 {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
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

        let mut arena = Arena::new(ep.arena_size.max(self.prefill_arena_size));
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

        let mut arena = Arena::new(ep.arena_size.max(self.prefill_arena_size));
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

        let mut arena = Arena::new(ep.arena_size.max(self.decode_arena_size));
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
