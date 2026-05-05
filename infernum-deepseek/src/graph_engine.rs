//! CPU graph-mode inference engine for the DeepSeek model family.
//!
//! [`DeepSeekGraphEngine`] is a self-contained, standalone alternative to
//! [`crate::DeepSeekModel`] + [`infernum_runtime::Engine`] for CPU inference.
//! It pre-compiles a single decode graph at construction time and reuses it
//! for every token — both prompt warmup and autoregressive decode.
//!
//! Unlike the Llama/Qwen/Gemma engines, the DeepSeek engine does **not** use
//! [`KvCacheStore`] externally. The [`MlaAttentionOp`] node is opaque: KV
//! cache management (compressed latent `c_kv` storage and accumulation) is
//! entirely internal to the CPU executor dispatch arm for that op. The graph's
//! only input is the current token ID; the only output is the logits tensor.
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use infernum_deepseek::DeepSeekGraphEngine;
//!
//! let engine = DeepSeekGraphEngine::from_pretrained(Path::new("/path/to/model")).unwrap();
//! let tokens = engine.generate(&[0, 1234, 567], 128, 2).unwrap();
//! println!("{tokens:?}");
//! ```

use std::path::Path;

use infernum::graph::{optimizer, plan, Arena, ExecutionPlan, Graph, NodeId, WeightStore};
use infernum::{DType, Result};
use infernum_cpu::executor::execute;
use infernum_cpu::tensor::{CpuLinearWeight, CpuTensor};
use infernum_cpu::CpuBackend;

use crate::config::DeepSeekConfig;
use crate::graph_builder::{
    build_decode_graph, build_prefill_graph, load_graph_weights_safetensors, DeepSeekGraphOps,
};

// ---------------------------------------------------------------------------
// DecodeCache
// ---------------------------------------------------------------------------

struct DecodeCache {
    graph: Graph<CpuBackend>,
    plan: ExecutionPlan,
    logits_id: NodeId,
}

fn build_decode_cache(config: &DeepSeekConfig) -> Result<DecodeCache>
where
    CpuBackend: DeepSeekGraphOps,
{
    let mut graph: Graph<CpuBackend> = build_decode_graph(config, DType::F32);
    optimizer::optimize(&mut graph);
    let ep = plan(&graph);
    let logits_id = graph.output_ids()[0];
    Ok(DecodeCache {
        graph,
        plan: ep,
        logits_id,
    })
}

// ---------------------------------------------------------------------------
// DeepSeekGraphEngine
// ---------------------------------------------------------------------------

/// CPU graph-mode engine for the DeepSeek model family (DeepSeek-V3, DeepSeek-R1).
///
/// Loads weights once and pre-compiles a single decode graph. The MLA
/// attention KV cache is managed internally by the `MlaAttentionOp` CPU
/// executor dispatch arm, not by the engine itself.
pub struct DeepSeekGraphEngine {
    config: DeepSeekConfig,
    weights: WeightStore<CpuTensor, CpuLinearWeight>,
    decode: DecodeCache,
}

impl DeepSeekGraphEngine {
    /// Load a DeepSeek-family model from a SafeTensors directory.
    ///
    /// # Errors
    /// Returns an error if the directory is missing, weights cannot be loaded,
    /// or `config.json` cannot be parsed.
    pub fn from_pretrained(model_dir: &Path) -> Result<Self>
    where
        CpuBackend: DeepSeekGraphOps,
    {
        let config = DeepSeekConfig::from_file(model_dir.join("config.json"))?;

        // Build a dummy 1-token prefill graph to discover weight metadata.
        let dummy_graph: Graph<CpuBackend> = build_prefill_graph(&config, DType::F32);
        let weights = load_graph_weights_safetensors(&dummy_graph, model_dir, &config)?;

        let decode = build_decode_cache(&config)?;
        Ok(Self {
            config,
            weights,
            decode,
        })
    }

    /// Greedy generation.
    ///
    /// The KV cache is managed internally by the `MlaAttentionOp` executor
    /// dispatch arm. The engine simply feeds one token at a time and reads
    /// the argmax of the returned logits.
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
        CpuBackend: DeepSeekGraphOps,
    {
        let dc = &self.decode;
        let mut arena = Arena::new(dc.plan.arena_size);

        let run_step = |token: u32, arena: &mut Arena| -> Result<Vec<CpuTensor>> {
            let inputs = vec![CpuTensor::from_u32(&[1], &[token])];
            execute(
                &dc.plan,
                dc.graph.nodes(),
                arena,
                &self.weights,
                &inputs,
                &[dc.logits_id],
                None,
            )
        };

        let mut token_ids = prompt_ids.to_vec();

        // Prompt warmup: process all tokens except the last.
        for &token in &prompt_ids[..prompt_ids.len().saturating_sub(1)] {
            run_step(token, &mut arena)?;
        }

        // Last prompt token — get the first generated token.
        let last_prompt_token = *prompt_ids.last().unwrap_or(&0);
        let outputs = run_step(last_prompt_token, &mut arena)?;
        let first_token = argmax(outputs[0].as_f32_slice());
        if first_token == eos_token_id {
            return Ok(token_ids);
        }
        token_ids.push(first_token);

        // Autoregressive decode.
        for _ in 0..max_new_tokens.saturating_sub(1) {
            let last_token = *token_ids.last().unwrap();
            let outputs = run_step(last_token, &mut arena)?;
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
    pub fn config(&self) -> &DeepSeekConfig {
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
