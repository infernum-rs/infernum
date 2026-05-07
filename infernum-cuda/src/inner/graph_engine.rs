//! CUDA-graph-accelerated decode engine for Llama-family models.
//!
//! [`CudaDecodeEngine`] owns a [`GraphInputs`] struct with pre-allocated
//! stable GPU-side input buffers. On each `step()`:
//!
//! - New token ID, position, `RoPE` cos/sin, block table, and seq-len are
//!   written into the buffers via `htod_copy_into`.
//! - During stabilization, the graph is captured inside
//!   `begin_capture` / `end_capture` until the buffer pool stops growing.
//! - On the fast path, only `cuda_graph.launch()` is called per step.
//!
//! The engine owns:
//! - A single [`CudaGraph`] instance (wraps a `cudaGraphExec_t`).
//! - A [`GraphInputs`] with stable GPU-side input buffers.
//! - A [`PagedKvCache`] for the KV cache.
//! - A [`BlockAllocator`] and [`BlockTable`] for paged memory management.
//! - The compiled [`ExecutionPlan`] (topological schedule).
//!
//! ## Stabilization
//!
//! On first use the buffer pool has not yet allocated the right sizes, so each
//! call to `execute` may miss the pool and allocate new buffers. The pool miss
//! count is tracked via [`BufferPool::misses`]. After the count stops growing
//! (i.e., the pool has enough buffers cached), the [`CudaGraph`]'s
//! `cudaGraphExec_t` is fully stable and `begin_capture`/`end_capture` are no
//! longer needed each step — `launch()` alone suffices.

use infernum::block_allocator::{BlockAllocator, BlockConfig, BlockTable};
use infernum::graph::{optimizer, plan, ExecutionPlan, Graph, NodeId, WeightStore};
use infernum::{precompute_rope_row, Result};

use crate::cuda::ops::LinearWeight;
use crate::cuda::{CudaContext, CudaEvent, CudaGraph, CudaTensor, PagedKvCache, PinnedBuffer};
use crate::inner::execute_context::GraphInputs;
use crate::CudaBackend;

// ---------------------------------------------------------------------------
// CudaDecodeEngine
// ---------------------------------------------------------------------------

/// CUDA-graph-accelerated single-token decode engine.
///
/// Owns all state needed for repeated single-token decode steps:
///
/// - [`GraphInputs`] — pre-allocated stable GPU-side input buffers (`token_ids`,
///   `cos`, `sin`, `block_table`, `positions`, `seq_lens`).
/// - [`PagedKvCache`] — paged KV cache for all transformer layers.
/// - [`BlockAllocator`] / [`BlockTable`] — host-side paged memory management.
/// - [`CudaGraph`] — the compiled CUDA graph executable.
/// - [`ExecutionPlan`] — topological schedule (built once, reused every step).
///
/// # Step lifecycle
///
/// [`step`](CudaDecodeEngine::step) runs one decode step:
///
/// 1. Writes the current token ID, `RoPE` cos/sin, block table, positions, and
///    seq-lens into the `GraphInputs` buffers via `htod_copy_into`.
/// 2. **Stabilisation phase** (first few steps): wraps [`execute`] in
///    [`begin_capture`](CudaGraph::begin_capture) /
///    [`end_capture`](CudaGraph::end_capture) so the buffer pool can allocate
///    its working set under capture. Re-captures until the pool miss count
///    plateaus (pool stable), then sets `stabilized = true`.
/// 3. **Fast path** (once stabilised): calls
///    [`launch`](CudaGraph::launch) only — a single `cuGraphLaunch` with
///    near-zero CPU overhead. The token result is read back asynchronously via
///    [`PinnedBuffer`] + [`CudaEvent`].
///
/// [`advance`](CudaDecodeEngine::advance) **must** be called after each
/// `step()`. It increments the host-side sequence position and advances the
/// block table — operations that cannot occur inside a capture boundary.
pub struct CudaDecodeEngine {
    ctx: CudaContext,
    cuda_graph: CudaGraph,
    plan: ExecutionPlan,
    graph: Graph<CudaBackend>,
    weights: WeightStore<CudaTensor, LinearWeight>,
    /// Pre-allocated stable GPU-side input buffers (`token_ids`, cos, sin,
    /// `block_table`, positions, `seq_lens`). Updated via `htod_copy_into` before
    /// each step; the captured graph references these fixed addresses.
    graph_inputs: GraphInputs,
    /// Paged KV cache for all transformer layers.
    paged_kv_cache: PagedKvCache,
    /// Host-side block allocator for the single decode sequence.
    block_allocator: BlockAllocator,
    /// Per-sequence block table (single sequence for `batch_size=1`).
    block_table: BlockTable,
    /// Maximum number of blocks per sequence that the graph was built for.
    max_blocks_per_seq: usize,
    /// Current sequence position (0-indexed). Incremented in `advance()`.
    current_pos: u32,
    /// `RoPE` theta for cos/sin computation.
    rope_theta: f32,
    /// Head dimension (used to compute `half_dim` for `RoPE`).
    head_dim: usize,
    /// How many pool misses were seen at the end of the last step. When this
    /// value matches the current `pool.misses()` the graph has stabilized.
    last_miss_count: u64,
    /// `true` once the graph no longer changes between steps.
    stabilized: bool,
    /// Index of the graph output node (argmax token, `U32` `[1]`).
    output_node: NodeId,
    /// The token tensor (`[1]` `U32`) from the last captured execute. Its GPU
    /// address is stable (pool-backed). On the fast path the CUDA graph writes
    /// fresh values into it; we read 4 bytes from the pinned buffer.
    saved_token: Option<CudaTensor>,
    /// Pinned host buffer for async D→H token readback (1 `u32`).
    pinned_token: PinnedBuffer,
    /// CUDA event recorded after each async `DtoH` copy. On the next step,
    /// `event.synchronize()` waits only for that copy — much cheaper than
    /// `ctx.synchronize()` which flushes all GPU work on all streams.
    completion_event: CudaEvent,
}

impl CudaDecodeEngine {
    /// Build a CUDA-graph decode engine from a paged decode graph and a
    /// freshly-populated [`WeightStore`].
    ///
    /// # Parameters
    ///
    /// * `ctx` — CUDA context.
    /// * `graph` — Paged decode graph built by `build_paged_decode_graph`.
    /// * `weights` — Fully populated [`WeightStore`] (model weights).
    /// * `graph_inputs` — Pre-allocated stable GPU-side input buffers.
    /// * `paged_kv_cache` — Pre-allocated paged KV cache.
    /// * `block_config` — Block configuration (size + total pool).
    /// * `max_blocks_per_seq` — Max blocks per sequence (graph was built with this).
    /// * `rope_theta` — `RoPE` theta for cos/sin computation.
    /// * `head_dim` — Attention head dimension.
    ///
    /// # Panics
    ///
    /// Panics if the decode graph has no output nodes.
    ///
    /// # Errors
    ///
    /// Returns an error if graph planning or [`CudaGraph`] allocation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: CudaContext,
        graph: Graph<CudaBackend>,
        weights: WeightStore<CudaTensor, LinearWeight>,
        graph_inputs: GraphInputs,
        paged_kv_cache: PagedKvCache,
        block_config: &BlockConfig,
        max_blocks_per_seq: usize,
        rope_theta: f32,
        head_dim: usize,
    ) -> Result<Self> {
        let mut graph = graph;
        optimizer::optimize(&mut graph);
        let ep = plan(&graph);

        // Locate the single output node (argmax token).
        let output_node = *graph
            .output_ids()
            .first()
            .expect("paged decode graph must have at least one output");

        let cuda_graph = CudaGraph::new(ctx.device())?;
        let pinned_token = PinnedBuffer::new(ctx.device())?;
        let completion_event = CudaEvent::new(ctx.device())?;

        let block_allocator = BlockAllocator::new(block_config);
        let block_table = BlockTable::new(block_config.block_size);

        Ok(Self {
            ctx,
            cuda_graph,
            plan: ep,
            graph,
            weights,
            graph_inputs,
            paged_kv_cache,
            block_allocator,
            block_table,
            max_blocks_per_seq,
            current_pos: 0,
            rope_theta,
            head_dim,
            last_miss_count: 0,
            stabilized: false,
            output_node,
            saved_token: None,
            pinned_token,
            completion_event,
        })
    }

    /// Write updated decode inputs into the [`GraphInputs`] GPU buffers.
    ///
    /// Must be called before every `execute` or `launch()` call. The block
    /// table must already have a block allocated for the current position
    /// (call `ensure_block()` first).
    ///
    /// # Errors
    ///
    /// Returns an error if any `htod_copy_into` fails.
    fn update_graph_inputs(&mut self, next_token: u32) -> Result<()> {
        let device = self.ctx.device();
        let half_dim = self.head_dim / 2;
        let (cos_row, sin_row) =
            precompute_rope_row(self.current_pos as usize, self.head_dim, self.rope_theta);

        device.htod_copy_into(vec![next_token], &mut self.graph_inputs.token_ids)?;
        device.htod_copy_into(cos_row, &mut self.graph_inputs.cos)?;
        device.htod_copy_into(sin_row, &mut self.graph_inputs.sin)?;

        // Block table: pad to max_blocks_per_seq with zeros.
        let raw_blocks: Vec<u32> = self
            .block_table
            .blocks()
            .iter()
            .map(|&b| u32::try_from(b).expect("block index exceeds u32::MAX"))
            .collect();
        let mut block_row = vec![0u32; self.max_blocks_per_seq];
        let n = raw_blocks.len().min(self.max_blocks_per_seq);
        block_row[..n].copy_from_slice(&raw_blocks[..n]);
        device.htod_copy_into(block_row, &mut self.graph_inputs.block_table)?;

        // positions: the token's absolute position in the sequence.
        device.htod_copy_into(vec![self.current_pos], &mut self.graph_inputs.positions)?;

        // seq_lens: after appending, the KV cache contains current_pos + 1 tokens.
        device.htod_copy_into(vec![self.current_pos + 1], &mut self.graph_inputs.seq_lens)?;

        let _ = half_dim; // half_dim is the length of cos/sin, already correct
        Ok(())
    }

    /// Ensure a block is allocated for the current position. Must be called
    /// before `update_graph_inputs` when the current block is full.
    ///
    /// # Panics
    ///
    /// Panics if the block pool is exhausted.
    fn ensure_block(&mut self) {
        if self.block_table.needs_new_block() {
            let block_idx = self
                .block_allocator
                .allocate()
                .expect("block pool exhausted; increase max_blocks_per_seq");
            self.block_table.append_block(block_idx);
        }
    }

    /// Write the next token ID to the GPU buffer and run one decode step.
    ///
    /// Returns the predicted next token as a `u32`. The argmax is computed
    /// on-device inside the CUDA graph; only 4 bytes are copied host-side.
    /// Caller must call [`CudaDecodeEngine::advance`] **after** this — the
    /// host-side block table and position counter cannot be captured inside
    /// the graph.
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
        // Ensure a block is available for the current position.
        self.ensure_block();

        // Write dynamic decode values into the stable GPU buffers.
        self.update_graph_inputs(next_token)?;

        if self.stabilized {
            // Fast path: graph is stable — replay with a single launch, then
            // use a targeted event sync instead of a full device sync.
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

            // GraphInputs ownership: CudaExecutorState takes it via `take()`,
            // then returns it after the node. We reconstruct it from the state
            // after execute completes by transferring the buffers back. To avoid
            // moving self.graph_inputs, we construct a temporary GraphInputs
            // that borrows the same device memory by re-creating it with the
            // same slices. The simplest approach: replace graph_inputs with a
            // new allocation temporarily and restore after.
            //
            // However CudaSlice cannot be easily cloned. The cleanest approach
            // is to make GraphInputs available for the duration of execute by
            // passing it through the call and getting it back.
            //
            // Since execute() takes `mut graph_inputs: Option<GraphInputs>`,
            // we move self.graph_inputs into a temporary and rebuild it from
            // state after. Use std::mem::replace with a dummy.
            let dummy = GraphInputs::new(
                self.ctx.device(),
                self.graph_inputs.batch_size,
                self.graph_inputs.half_dim,
                self.graph_inputs.max_blocks_per_seq,
            )?;
            let real_inputs = std::mem::replace(&mut self.graph_inputs, dummy);

            self.cuda_graph.begin_capture()?;

            let output_nodes = [self.output_node];
            let mut outputs = super::executor::execute(
                &self.ctx,
                &self.plan,
                self.graph.nodes(),
                &self.weights,
                &[], // inputs come from graph_inputs
                &output_nodes,
                None,                           // no MLA KV cache
                Some(&mut self.paged_kv_cache), // paged KV cache
                0,                              // mla_seq_pos
                Some(real_inputs),              // stable GPU input buffers
            )?;

            self.cuda_graph.end_capture()?;
            self.cuda_graph.launch()?;
            self.ctx.synchronize()?;

            // Restore graph_inputs. The executor consumed it and we need a fresh
            // one for the next step. Allocate a replacement.
            self.graph_inputs = GraphInputs::new(
                self.ctx.device(),
                1,
                self.head_dim / 2,
                self.max_blocks_per_seq,
            )?;

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
    /// Updates the host-side block table and position counter. These cannot
    /// be captured inside a CUDA graph and must stay outside the capture
    /// boundary.
    pub fn advance(&mut self) {
        self.block_table.advance(1);
        self.current_pos += 1;
    }

    /// Return a shared reference to the paged KV cache.
    #[must_use]
    pub fn paged_kv_cache(&self) -> &PagedKvCache {
        &self.paged_kv_cache
    }

    /// Return the current sequence position (number of tokens generated so far).
    #[must_use]
    pub fn current_pos(&self) -> u32 {
        self.current_pos
    }

    /// Return whether the CUDA graph has stabilized (pool misses stopped growing).
    #[must_use]
    pub fn is_stabilized(&self) -> bool {
        self.stabilized
    }
}
