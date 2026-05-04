//! CUDA-graph-accelerated decode engine for Llama-family models.
//!
//! [`CudaDecodeEngine`] pre-builds an indirect decode graph once and replays
//! it with `begin_capture` / `end_capture` until the pool stabilizes, then
//! switches to bare `launch()` calls — avoiding the per-step
//! `cuGraphExecUpdate_v2` overhead once the graph topology is fixed.
//!
//! The engine owns:
//! - A single `CudaGraph` instance (`cudaGraphExec_t` under the hood).
//! - The pre-computed RoPE cos/sin cache (stable GPU address, registered as a
//!   tensor weight).
//! - A `KvCache` with pre-allocated `[max_seq_len, …]` buffers per layer.
//! - A `SeqPosition` holding the current step position on the GPU.
//! - The compiled `ExecutionPlan` (topological schedule).
//!
//! ## Stabilization
//!
//! On first use the buffer pool has not yet allocated the right sizes, so each
//! call to `execute_indirect` may miss the pool and allocate new buffers. The
//! pool miss count is tracked via [`BufferPool::misses`]. After the count stops
//! growing (i.e., the pool has enough buffers cached), the `CudaGraph`'s
//! `cuGraphExec_t` is fully stable and `begin_capture`/`end_capture` are no
//! longer needed each step — `launch()` alone suffices.

use infernum::graph::{optimizer, plan, ExecutionPlan, Graph, NodeId, WeightStore};
use infernum::Result;

use crate::cuda::ops::LinearWeight;
use crate::cuda::{CudaContext, CudaGraph, CudaTensor, KvCache, SeqPosition};
use crate::CudaBackend;

// ---------------------------------------------------------------------------
// CudaDecodeEngine
// ---------------------------------------------------------------------------

/// CUDA-graph-accelerated single-token decode engine.
///
/// Call [`CudaDecodeEngine::step`] once per decode step. The first few steps
/// re-capture the graph until the buffer pool stabilizes; subsequent steps
/// replay the captured graph with a single `launch()` call.
pub struct CudaDecodeEngine {
    ctx: CudaContext,
    cuda_graph: CudaGraph,
    plan: ExecutionPlan,
    graph: Graph<CudaBackend>,
    weights: WeightStore<CudaTensor, LinearWeight>,
    kv_cache: KvCache,
    seq_pos: SeqPosition,
    /// How many pool misses were seen at the end of the last step. When this
    /// value matches the current `pool.misses()` the graph has stabilized.
    last_miss_count: u64,
    /// `true` once the graph no longer changes between steps.
    stabilized: bool,
    /// Index of the graph output node (logits).
    output_node: NodeId,
    /// The logits tensor from the last captured execute. Its GPU address is
    /// stable (pool-backed), so the same tensor can be returned each step after
    /// stabilization — the CUDA graph will have written fresh values into it.
    saved_logits: Option<CudaTensor>,
}

impl CudaDecodeEngine {
    /// Build a CUDA-graph decode engine from an already-compiled indirect
    /// decode graph and a freshly-populated `WeightStore`.
    ///
    /// # Parameters
    ///
    /// * `ctx` — CUDA context (device 0 or whichever device the weights live on).
    /// * `graph` — Indirect decode graph built by `build_indirect_decode_graph`.
    /// * `weights` — Fully populated `WeightStore` (model weights, RoPE caches,
    ///   KV cache tensors registered at construction time).
    /// * `kv_cache` — Pre-allocated KV cache with `max_seq_len` capacity.
    /// * `seq_pos` — GPU-resident sequence position counter (device pointer).
    ///
    /// # Errors
    ///
    /// Returns an error if graph planning or `CudaGraph` allocation fails.
    pub fn new(
        ctx: CudaContext,
        graph: Graph<CudaBackend>,
        weights: WeightStore<CudaTensor, LinearWeight>,
        kv_cache: KvCache,
        seq_pos: SeqPosition,
    ) -> Result<Self> {
        let mut graph = graph;
        optimizer::optimize(&mut graph);
        let ep = plan(&graph);

        // -- DIAGNOSTIC: dump the full graph and schedule (remove before shipping) --
        eprintln!("[graph_engine] graph has {} nodes", graph.nodes().len());
        for (i, node) in graph.nodes().iter().enumerate() {
            eprintln!(
                "  node {:3}: op={:30} inputs={:?} side_effect={}",
                i,
                node.op.name(),
                node.inputs,
                node.op.is_side_effect(),
            );
        }
        eprintln!(
            "[graph_engine] plan schedule ({} entries):",
            ep.schedule.len()
        );
        for &nid in &ep.schedule {
            let node = &graph.nodes()[nid.index() as usize];
            eprintln!("  {:?} op={}", nid, node.op.name());
        }

        // Locate the single output node (logits).
        let output_node = *graph
            .output_ids()
            .first()
            .expect("indirect decode graph must have at least one output");

        let cuda_graph = CudaGraph::new(ctx.device())?;

        Ok(Self {
            ctx,
            cuda_graph,
            plan: ep,
            graph,
            weights,
            kv_cache,
            seq_pos,
            last_miss_count: 0,
            stabilized: false,
            output_node,
            saved_logits: None,
        })
    }

    /// Write the next token ID to the GPU buffer and run one decode step.
    ///
    /// Returns the logits tensor for the predicted next token. Caller must
    /// call `kv_cache.advance(1)` **after** reading the logits — it is a
    /// host-side pointer update that cannot be captured inside the graph.
    ///
    /// The first calls re-capture the graph (cheap once the pool is warm); once
    /// the pool has stabilized, only `launch()` is called.
    ///
    /// # Errors
    ///
    /// Returns an error if any kernel launch or CUDA API call fails.
    pub fn step(&mut self, next_token: u32) -> Result<CudaTensor> {
        // Write the token ID to the stable GPU buffer that embedding_gather_indirect reads.
        self.ctx
            .device()
            .htod_copy_into(vec![next_token], self.cuda_graph.token_input_mut())?;

        if self.stabilized {
            // Fast path: graph is stable — replay with a single launch.
            // The logits tensor's GPU address is stable (pool-backed) and the
            // CUDA graph will have written fresh values into it. Return the
            // same CudaTensor handle — the caller sees the updated data after
            // synchronize() below.
            self.cuda_graph.launch()?;
            self.ctx.synchronize()?;
            Ok(self
                .saved_logits
                .clone()
                .expect("saved_logits must be set before stabilization"))
        } else {
            // Stabilization path: run execute_indirect within a CUDA graph
            // capture so the pool allocates all required buffers. The resulting
            // logits tensor has a stable pool-backed GPU address — save it so
            // the fast path can return it without re-executing.
            self.cuda_graph.begin_capture()?;

            let mut outputs = super::executor::execute_indirect(
                &self.plan,
                self.graph.nodes(),
                &self.weights,
                &[], // no graph inputs — data comes from weights / kv_cache / seq_pos
                &[self.output_node],
                &self.ctx,
                &self.seq_pos,
                &mut self.kv_cache,
            )?;

            self.cuda_graph.end_capture()?;
            self.cuda_graph.launch()?;
            self.ctx.synchronize()?;

            let logits = outputs.pop().unwrap();
            // Save logits for the fast path (stable pool-backed GPU address).
            self.saved_logits = Some(logits.clone());

            // Track pool misses to detect stabilization.
            if let Some(pool) = self.ctx.buffer_pool() {
                let current_misses = pool.misses();
                if current_misses == self.last_miss_count && self.cuda_graph.is_instantiated() {
                    self.stabilized = true;
                }
                self.last_miss_count = current_misses;
            } else {
                // No pool: switch to bare launch after first successful capture.
                self.stabilized = self.cuda_graph.is_instantiated();
            }

            Ok(logits)
        }
    }

    /// Advance the sequence position by one step (must be called after `step`).
    ///
    /// This is a host-side update of the GPU position pointer and cannot be
    /// captured inside a CUDA graph. It must stay outside the capture boundary.
    ///
    /// # Errors
    ///
    /// Returns an error if the device copy fails.
    pub fn advance(&mut self) -> Result<()> {
        self.kv_cache.advance(1)
    }

    /// Return a shared reference to the KV cache (e.g., for reading `current_len`).
    pub fn kv_cache(&self) -> &KvCache {
        &self.kv_cache
    }

    /// Return a shared reference to the sequence position counter.
    pub fn seq_pos(&self) -> &SeqPosition {
        &self.seq_pos
    }

    /// Return whether the CUDA graph has stabilized (pool misses stopped growing).
    pub fn is_stabilized(&self) -> bool {
        self.stabilized
    }
}
