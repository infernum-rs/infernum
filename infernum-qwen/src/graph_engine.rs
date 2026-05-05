//! CPU graph-mode inference engine for the Qwen model family.
//!
//! [`QwenGraphEngine`] is a self-contained, standalone alternative to
//! [`crate::QwenModel`] + [`infernum_runtime::Engine`] for CPU inference.
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
//! use infernum_qwen::QwenGraphEngine;
//!
//! let engine = QwenGraphEngine::from_pretrained(Path::new("/path/to/model")).unwrap();
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

use crate::config::QwenConfig;
use crate::graph_builder::{
    build_decode_graph, build_prefill_graph, load_graph_weights_safetensors, QwenGraphOps,
};

// ---------------------------------------------------------------------------
// KV cache node discovery
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
        .map(|(i, _)| NodeId::from_index(i as u32))
        .collect();
    assert_eq!(
        input_ids.len(),
        2 * num_layers,
        "unexpected KV input count in Qwen decode graph"
    );

    let concat_ids: Vec<NodeId> = nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.op.name() == "concat_seq")
        .map(|(i, _)| NodeId::from_index(i as u32))
        .collect();
    assert_eq!(
        concat_ids.len(),
        2 * num_layers,
        "unexpected ConcatSeq count in Qwen decode graph"
    );

    (input_ids, concat_ids)
}

// ---------------------------------------------------------------------------
// DecodeCache
// ---------------------------------------------------------------------------

struct DecodeCache {
    graph: Graph<CpuBackend>,
    plan: ExecutionPlan,
    logits_id: NodeId,
    cache_input_ids: Vec<NodeId>,
    concat_ids: Vec<NodeId>,
    /// Pre-computed RoPE tables (standard half-dim layout).
    cos: Vec<f32>,
    sin: Vec<f32>,
    half_dim: usize,
}

fn build_decode_cache(config: &QwenConfig) -> Result<DecodeCache>
where
    CpuBackend: QwenGraphOps,
{
    let head_dim = config.head_dim();
    let half_dim = head_dim / 2;
    let num_layers = config.num_hidden_layers;

    let (mut graph, _) = build_decode_graph::<CpuBackend>(config, 0, DType::F32);
    optimizer::optimize(&mut graph);
    let ep = plan(&graph);
    let logits_id = graph.output_ids()[0];
    let (cache_input_ids, concat_ids) = find_kv_cache_node_ids(graph.nodes(), num_layers);

    let max_pos = config.max_position_embeddings;
    let (cos, sin) = precompute_rope_data(max_pos, head_dim, config.rope_theta);

    Ok(DecodeCache {
        graph,
        plan: ep,
        logits_id,
        cache_input_ids,
        concat_ids,
        cos,
        sin,
        half_dim,
    })
}

// ---------------------------------------------------------------------------
// QwenGraphEngine
// ---------------------------------------------------------------------------

/// CPU graph-mode engine for Qwen-family models.
///
/// Loads weights once, pre-compiles a single decode graph (at `kv_len = 0`),
/// and reuses it for all generation steps — both prompt warmup and autoregressive
/// decode — via [`KvCacheStore`].
pub struct QwenGraphEngine {
    config: QwenConfig,
    weights: WeightStore<CpuTensor, CpuLinearWeight>,
    decode: DecodeCache,
}

impl QwenGraphEngine {
    /// Load a Qwen-family model from a SafeTensors directory.
    ///
    /// # Errors
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or `config.json` cannot be parsed.
    pub fn from_pretrained(model_dir: &Path) -> Result<Self>
    where
        CpuBackend: QwenGraphOps,
    {
        let config = QwenConfig::from_file(model_dir.join("config.json"))?;

        let (dummy_graph, _) = build_prefill_graph::<CpuBackend>(&config, 1, DType::F32);
        let weights = load_graph_weights_safetensors(&dummy_graph, model_dir, &config)?;

        let decode = build_decode_cache(&config)?;
        Ok(Self {
            config,
            weights,
            decode,
        })
    }

    /// Greedy generation using a persistent KV cache.
    ///
    /// Processes the prompt token-by-token (warming up the KV cache), then
    /// continues decoding until EOS or `max_new_tokens` is reached.
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
    ) -> Result<Vec<u32>>
    where
        CpuBackend: QwenGraphOps,
    {
        let config = &self.config;
        let num_layers = config.num_hidden_layers;
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();

        let dc = &self.decode;
        let half_dim = dc.half_dim;

        let mut kv_cache = KvCacheStore::new(
            num_layers,
            num_kv_heads,
            head_dim,
            config.max_position_embeddings,
            dc.cache_input_ids.clone(),
            dc.concat_ids.clone(),
        );

        let mut arena = Arena::new(dc.plan.arena_size);

        let max_pos = prompt_ids.len() + max_new_tokens;
        let rope_extended;
        let (cos, sin): (&[f32], &[f32]) = if max_pos * half_dim <= dc.cos.len() {
            (&dc.cos, &dc.sin)
        } else {
            rope_extended = precompute_rope_data(max_pos, head_dim, config.rope_theta);
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

        // Autoregressive decode.
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
    pub fn config(&self) -> &QwenConfig {
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
