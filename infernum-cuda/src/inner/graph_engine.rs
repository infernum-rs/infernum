//! CUDA-graph-accelerated decode engine for Llama-family models.
//!
//! [`CudaDecodeEngine`] pre-builds an indirect decode graph once and replays
//! it with `begin_capture` / `end_capture` until the pool stabilizes, then
//! switches to bare `launch()` calls — avoiding the per-step
//! `cuGraphExecUpdate_v2` overhead once the graph topology is fixed.
//!
//! The engine owns:
//! - A single [`CudaGraph`] instance (wraps a `cudaGraphExec_t`).
//! - The pre-computed `RoPE` cos/sin cache (stable GPU address, registered as a
//!   tensor weight).
//! - A [`KvCache`] with pre-allocated `[max_seq_len, …]` buffers per layer.
//! - A [`SeqPosition`] holding the current step position on the GPU.
//! - The compiled [`ExecutionPlan`] (topological schedule).
//!
//! ## Stabilization
//!
//! On first use the buffer pool has not yet allocated the right sizes, so each
//! call to `execute_indirect` may miss the pool and allocate new buffers. The
//! pool miss count is tracked via [`BufferPool::misses`]. After the count stops
//! growing (i.e., the pool has enough buffers cached), the [`CudaGraph`]'s
//! `cuGraphExec_t` is fully stable and `begin_capture`/`end_capture` are no
//! longer needed each step — `launch()` alone suffices.

use infernum::graph::{optimizer, plan, ExecutionPlan, Graph, NodeId, WeightStore};
use infernum::Result;

use crate::cuda::ops::LinearWeight;
use crate::cuda::{
    CudaContext, CudaEvent, CudaGraph, CudaTensor, KvCache, PinnedBuffer, SeqPosition,
};
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
    /// Index of the graph output node (`U32` token tensor `[1]`).
    output_node: NodeId,
    /// The token tensor (`[1]` `U32`) from the last captured execute. Its GPU
    /// address is stable (pool-backed). On the fast path the CUDA graph writes
    /// fresh values into it each step; we read 4 bytes from the pinned buffer.
    saved_token: Option<CudaTensor>,
    /// Pinned host buffer for async D→H token readback (1 `u32`).
    pinned_token: PinnedBuffer,
    /// CUDA event recorded after each async `DtoH` copy. On the next step,
    /// `event.synchronize()` waits only for that copy — much cheaper than
    /// `ctx.synchronize()` which flushes all GPU work on all streams.
    completion_event: CudaEvent,
}

impl CudaDecodeEngine {
    /// Build a CUDA-graph decode engine from an already-compiled indirect
    /// decode graph and a freshly-populated [`WeightStore`].
    ///
    /// # Parameters
    ///
    /// * `ctx` — CUDA context (device 0 or whichever device the weights live on).
    /// * `graph` — Indirect decode graph built by `build_indirect_decode_graph`.
    /// * `weights` — Fully populated [`WeightStore`] (model weights, `RoPE` caches,
    ///   KV cache tensors registered at construction time).
    /// * `kv_cache` — Pre-allocated KV cache with `max_seq_len` capacity.
    /// * `seq_pos` — GPU-resident sequence position counter (device pointer).
    ///
    /// # Panics
    ///
    /// Panics if the indirect decode graph has no output nodes.
    ///
    /// # Errors
    ///
    /// Returns an error if graph planning or [`CudaGraph`] allocation fails.
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
        let pinned_token = PinnedBuffer::new(ctx.device())?;
        let completion_event = CudaEvent::new(ctx.device())?;

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
            saved_token: None,
            pinned_token,
            completion_event,
        })
    }

    /// Write the next token ID to the GPU buffer and run one decode step.
    ///
    /// Returns the predicted next token as a `u32`. The argmax is computed
    /// on-device inside the CUDA graph; only 4 bytes are copied host-side.
    /// Caller must call [`CudaDecodeEngine::advance`] **after** this — the KV
    /// cache host-side pointer update cannot be captured inside the graph.
    ///
    /// The first calls re-capture the graph (cheap once the pool is warm); once
    /// the pool has stabilized, only `launch()` is called.
    ///
    /// # Panics
    ///
    /// Panics if the stable fast-path token tensor has not been saved yet
    /// (internal invariant violation; indicates a bug in stabilization logic).
    ///
    /// # Errors
    ///
    /// Returns an error if any kernel launch or CUDA API call fails.
    pub fn step(&mut self, next_token: u32) -> Result<u32> {
        // TODO(Step 3): write next_token into GraphInputs::token_ids buffer here.
        // The old indirect-ops token_input buffer is no longer used; the token
        // is now passed via GraphInputs on each step.
        let _ = next_token;

        if self.stabilized {
            // Fast path: graph is stable — replay with a single launch, then
            // use a targeted event sync instead of a full device sync.
            //
            // Stream ordering guarantees:
            //   cuGraphLaunch  (async, writes token to saved_token GPU buffer)
            //   cuMemcpyDtoHAsync  (async, copies token → pinned_token, after graph)
            //   cuEventRecord  (marks completion fence)
            //   cuEventSynchronize  (wait only until the DtoH copy is done)
            //
            // This is substantially cheaper than ctx.synchronize() which waits
            // for ALL work on ALL CUDA streams (cuBLAS internal streams, etc.).
            let token_device_ptr = self
                .saved_token
                .as_ref()
                .expect("saved_token must be set before stabilization")
                .device_ptr();

            self.cuda_graph.launch()?;
            self.pinned_token.async_copy_from_device(token_device_ptr)?;
            self.completion_event.record()?;
            self.completion_event.synchronize()?;

            Ok(self.pinned_token.read())
        } else {
            // Stabilization path: execute the graph inside a CUDA graph
            // capture so the pool allocates all required buffers. The resulting
            // token tensor has a stable pool-backed GPU address — save it so
            // the fast path can read it without re-executing.
            self.cuda_graph.begin_capture()?;

            // TODO(Step 3): replace `inputs` with views into GraphInputs buffers.
            // For now pass an empty slice — graph inputs are supplied via
            // registered weights (KV cache, RoPE tables).
            let mut outputs = super::executor::execute(
                &self.plan,
                self.graph.nodes(),
                &self.weights,
                &[], // inputs come from registered weights
                &[self.output_node],
                None, // no MLA KV cache
                None, // no paged KV cache
                0,    // mla_seq_pos
            )?;

            self.cuda_graph.end_capture()?;
            self.cuda_graph.launch()?;
            self.ctx.synchronize()?;

            let token_tensor = outputs.pop().unwrap();
            let result = token_tensor.to_vec::<u32>()?;
            // Save the token tensor for the fast path (stable pool-backed GPU address).
            self.saved_token = Some(token_tensor);

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

            Ok(result[0])
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
    #[must_use]
    pub fn kv_cache(&self) -> &KvCache {
        &self.kv_cache
    }

    /// Return a shared reference to the sequence position counter.
    #[must_use]
    pub fn seq_pos(&self) -> &SeqPosition {
        &self.seq_pos
    }

    /// Return whether the CUDA graph has stabilized (pool misses stopped growing).
    #[must_use]
    pub fn is_stabilized(&self) -> bool {
        self.stabilized
    }
}
