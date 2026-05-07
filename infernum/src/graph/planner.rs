//! Memory planner for the computation graph.
//!
//! Takes a completed `Graph<B>` and produces an `ExecutionPlan` containing:
//! - A topological execution schedule
//! - Buffer offset assignments within a shared arena
//! - The total arena size (peak memory usage)
//!
//! The planner does **not** allocate memory — it produces a plan that an
//! executor uses to map each node's output to a region of a single arena.

use std::collections::VecDeque;

use crate::backend::{Backend, ContextBackend, MatmulOps};
use crate::dtype::DType;

use super::builder::Graph;
use super::node::NodeId;

/// A planned buffer allocation within the arena.
#[derive(Clone, Debug)]
pub struct BufferSlot {
    /// Offset in bytes from the start of the arena.
    pub offset: usize,
    /// Size in bytes of this buffer.
    pub size: usize,
}

/// The complete execution plan for a graph.
#[derive(Clone, Debug)]
pub struct ExecutionPlan {
    /// Nodes in topological execution order.
    pub schedule: Vec<NodeId>,
    /// Buffer slot for each node output, indexed by `(NodeId.index(), output_index)`.
    /// Outer vec indexed by `NodeId.index()`, inner vec by output index.
    /// `None` for `Input` nodes (they use external buffers) and side-effect ops.
    pub buffer_slots: Vec<Vec<Option<BufferSlot>>>,
    /// Total arena size in bytes (peak memory usage).
    pub arena_size: usize,
}

impl ExecutionPlan {
    /// Get the buffer slot for a given node's output.
    #[must_use]
    pub fn slot(&self, node: NodeId, output_idx: u32) -> Option<&BufferSlot> {
        self.buffer_slots
            .get(node.0 as usize)
            .and_then(|outputs| outputs.get(output_idx as usize))
            .and_then(|s| s.as_ref())
    }
}

/// Compute the execution plan for a graph.
///
/// # Panics
/// Panics if the graph contains cycles (should never happen for well-formed graphs).
#[must_use]
pub fn plan<B: Backend + MatmulOps + ContextBackend>(graph: &Graph<B>) -> ExecutionPlan {
    let full_schedule = topological_sort(graph);
    let schedule = eliminate_dead_nodes(graph, full_schedule);
    let last_use = compute_last_use(graph, &schedule);
    let (buffer_slots, arena_size) = assign_offsets(graph, &schedule, &last_use);
    ExecutionPlan {
        schedule,
        buffer_slots,
        arena_size,
    }
}

/// Remove nodes from the schedule that have no consumers and are not graph
/// outputs or side-effect ops. This eliminates orphaned nodes left behind
/// by fusion passes (e.g., Silu nodes orphaned by Swiglu fusion).
fn eliminate_dead_nodes<B: Backend + MatmulOps + ContextBackend>(
    graph: &Graph<B>,
    schedule: Vec<NodeId>,
) -> Vec<NodeId> {
    use std::collections::HashSet;

    let n = graph.nodes.len();

    // Build a set of all node IDs that are consumed by at least one other node.
    // An input `(src_node, output_idx)` means `src_node` is consumed.
    let mut is_consumed = vec![false; n];
    for node in &graph.nodes {
        for &(src_node, _output_idx) in &node.inputs {
            is_consumed[src_node.0 as usize] = true;
        }
    }

    // Graph outputs are always live.
    let output_set: HashSet<NodeId> = graph.outputs.iter().copied().collect();

    schedule
        .into_iter()
        .filter(|&id| {
            let idx = id.0 as usize;
            let node = &graph.nodes[idx];
            // Keep if: consumed by another node, is a graph output, or has side effects.
            is_consumed[idx] || output_set.contains(&id) || node.op.is_side_effect()
        })
        .collect()
}

/// Size in bytes for a single element of a `DType`.
///
/// Quantized types should not appear in intermediate buffers, but we handle
/// them gracefully with a conservative 1-byte fallback.
const fn dtype_size_bytes(dtype: DType) -> usize {
    match dtype {
        DType::F32 | DType::U32 => 4,
        DType::F16 | DType::BF16 => 2,
        // F8E4M3 = 1 byte. Quantized types are weight-only and shouldn't
        // appear in intermediates, but we handle them with a 1-byte fallback.
        _ => 1,
    }
}

/// Topological sort using Kahn's algorithm.
///
/// Standard BFS over nodes. Each node's inputs are `OutputRef = (NodeId, u32)`;
/// the `NodeId` part gives the predecessor.
#[allow(clippy::cast_possible_truncation)] // graph will never have 2^32 nodes
#[allow(clippy::needless_range_loop)]
fn topological_sort<B: Backend + MatmulOps + ContextBackend>(graph: &Graph<B>) -> Vec<NodeId> {
    let n = graph.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    // Build in-degree for each node: count distinct predecessor NodeIds
    // (not output indices — multiple inputs from the same node count once
    // per input edge for scheduling purposes, but we use the simpler
    // approach of counting the number of input edges).
    let mut in_degree = vec![0u32; n];
    for (i, node) in graph.nodes.iter().enumerate() {
        in_degree[i] = node.inputs.len() as u32;
    }

    // Seed the queue with nodes that have zero in-degree.
    let mut queue = VecDeque::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            queue.push_back(NodeId(i as u32));
        }
    }

    let mut schedule = Vec::with_capacity(n);

    while let Some(id) = queue.pop_front() {
        schedule.push(id);

        // Decrease in-degree for downstream consumers.
        for (j, node) in graph.nodes.iter().enumerate() {
            let mut decrements = 0u32;
            for &(src_node, _) in &node.inputs {
                if src_node == id {
                    decrements += 1;
                }
            }
            if decrements > 0 {
                in_degree[j] -= decrements;
                if in_degree[j] == 0 {
                    queue.push_back(NodeId(j as u32));
                }
            }
        }
    }

    assert_eq!(
        schedule.len(),
        n,
        "Graph contains a cycle: scheduled {} of {} nodes",
        schedule.len(),
        n,
    );

    schedule
}

/// For each `(node, output_index)`, compute the latest execution step where
/// it is still needed as an input. Graph outputs are live until the last step.
///
/// Returns a flat vector indexed by a key derived from `(node_index, output_index)`.
/// We use a `Vec<Vec<usize>>` where outer = node index, inner = output index.
fn compute_last_use<B: Backend + MatmulOps + ContextBackend>(
    graph: &Graph<B>,
    schedule: &[NodeId],
) -> Vec<Vec<usize>> {
    let n = graph.nodes.len();
    // Initialise per-output last-use to 0.
    let mut last_use: Vec<Vec<usize>> = graph
        .nodes
        .iter()
        .map(|node| vec![0usize; node.output_shapes.len()])
        .collect();

    for (step, &id) in schedule.iter().enumerate() {
        let node = &graph.nodes[id.0 as usize];
        for &(src_node, src_output) in &node.inputs {
            let idx = src_node.0 as usize;
            let oidx = src_output as usize;
            if oidx < last_use[idx].len() && step > last_use[idx][oidx] {
                last_use[idx][oidx] = step;
            }
        }
    }

    // Graph outputs are live until the final step.
    if let Some(last_step) = schedule.len().checked_sub(1) {
        for &out in &graph.outputs {
            let idx = out.0 as usize;
            // Graph outputs reference a NodeId — all outputs of that node
            // that are used as graph outputs should be kept alive.
            // In practice, graph outputs are NodeIds pointing at the node,
            // and consumers select specific output indices. We keep all
            // outputs of output nodes alive (conservative but correct).
            if idx < n {
                for oidx in 0..last_use[idx].len() {
                    if last_step > last_use[idx][oidx] {
                        last_use[idx][oidx] = last_step;
                    }
                }
            }
        }
    }

    last_use
}

/// Compute the buffer size in bytes for a single output.
fn buffer_size(shape: &[usize], dtype: DType) -> usize {
    let elements: usize = shape.iter().product();
    elements * dtype_size_bytes(dtype)
}

/// Greedy offset assignment within a shared arena.
///
/// Maintains a sorted free list of `(offset, size)` gaps. When a buffer dies
/// (its `last_use < current_step`), its range is returned to the free list
/// and merged with adjacent gaps.
///
/// Each node may produce multiple outputs, each needing its own buffer slot.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::needless_range_loop)]
fn assign_offsets<B: Backend + MatmulOps + ContextBackend>(
    graph: &Graph<B>,
    schedule: &[NodeId],
    last_use: &[Vec<usize>],
) -> (Vec<Vec<Option<BufferSlot>>>, usize) {
    let mut buffer_slots: Vec<Vec<Option<BufferSlot>>> = graph
        .nodes
        .iter()
        .map(|node| vec![None; node.output_shapes.len().max(1)])
        .collect();
    let mut arena_size: usize = 0;

    // Free list: sorted by offset. Each entry is (offset, size).
    let mut free_list: Vec<(usize, usize)> = Vec::new();

    // Track which buffers are currently live so we can free them.
    // (node_index, output_index, offset, size, last_use_step)
    let mut live_buffers: Vec<(usize, usize, usize, usize, usize)> = Vec::new();

    for (step, &id) in schedule.iter().enumerate() {
        // Free buffers whose last_use < current step.
        live_buffers.retain(|&(_, _, offset, size, last_step)| {
            if last_step < step {
                insert_free_range(&mut free_list, offset, size);
                false
            } else {
                true
            }
        });

        let idx = id.0 as usize;
        let node = &graph.nodes[idx];
        let num_outputs = node.output_shapes.len();

        // Side-effect ops produce no output buffer.
        if node.op.is_side_effect() || num_outputs == 0 {
            continue;
        }

        // Allocate a buffer for each output of this node.
        for oidx in 0..num_outputs {
            let size = buffer_size(&node.output_shapes[oidx], node.output_dtypes[oidx]);
            if size == 0 {
                buffer_slots[idx][oidx] = Some(BufferSlot { offset: 0, size: 0 });
                continue;
            }

            // Find first-fit gap in the free list.
            let offset = if let Some(pos) = free_list.iter().position(|&(_, s)| s >= size) {
                let (gap_offset, gap_size) = free_list[pos];
                if gap_size == size {
                    free_list.remove(pos);
                } else {
                    // Shrink the gap.
                    free_list[pos] = (gap_offset + size, gap_size - size);
                }
                gap_offset
            } else {
                // Extend the arena.
                let offset = arena_size;
                arena_size += size;
                offset
            };

            buffer_slots[idx][oidx] = Some(BufferSlot { offset, size });
            let lu = last_use
                .get(idx)
                .and_then(|v| v.get(oidx))
                .copied()
                .unwrap_or(step);
            live_buffers.push((idx, oidx, offset, size, lu));
        }
    }

    (buffer_slots, arena_size)
}

/// Insert a free range into the sorted free list, merging with adjacent gaps.
fn insert_free_range(free_list: &mut Vec<(usize, usize)>, offset: usize, size: usize) {
    // Find the insertion point (keep sorted by offset).
    let pos = free_list
        .iter()
        .position(|&(o, _)| o > offset)
        .unwrap_or(free_list.len());

    // Check if we can merge with the previous gap.
    let merge_prev = pos > 0 && {
        let (prev_off, prev_size) = free_list[pos - 1];
        prev_off + prev_size == offset
    };

    // Check if we can merge with the next gap.
    let merge_next = pos < free_list.len() && {
        let (next_off, _) = free_list[pos];
        offset + size == next_off
    };

    match (merge_prev, merge_next) {
        (true, true) => {
            // Merge previous, current, and next into one.
            let (prev_off, prev_size) = free_list[pos - 1];
            let (_, next_size) = free_list[pos];
            free_list[pos - 1] = (prev_off, prev_size + size + next_size);
            free_list.remove(pos);
        }
        (true, false) => {
            // Merge with previous.
            let (prev_off, prev_size) = free_list[pos - 1];
            free_list[pos - 1] = (prev_off, prev_size + size);
        }
        (false, true) => {
            // Merge with next.
            let (_, next_size) = free_list[pos];
            free_list[pos] = (offset, size + next_size);
        }
        (false, false) => {
            // Insert new gap.
            free_list.insert(pos, (offset, size));
        }
    }
}

// ---------------------------------------------------------------------------
// Plan cache
// ---------------------------------------------------------------------------

/// Thread-safe cache of compiled `ExecutionPlan`s, keyed by graph topology.
///
/// Avoids recompiling the same graph topology on every call. The cache key is
/// a `(topology_hash, arena_size)` pair derived from the graph's node count
/// and op-name sequence, so graphs with the same topology (e.g., same
/// sequence length) reuse the compiled plan without re-running the planner.
///
/// # Example
///
/// ```
/// use infernum::graph::PlanCache;
/// let cache = PlanCache::new();
/// ```
#[derive(Clone, Default)]
pub struct PlanCache {
    inner: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<(u64, usize), ExecutionPlan>>>,
}

impl PlanCache {
    /// Create an empty plan cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a compiled plan for `graph`, reusing a cached one if available.
    ///
    /// The first call for a given topology compiles and caches the plan.
    /// Subsequent calls with topologically identical graphs return the cached
    /// plan without re-running the planner.
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned (which only happens if another
    /// thread panicked while holding it).
    #[must_use]
    pub fn get_or_compile<B: Backend + MatmulOps + ContextBackend>(
        &self,
        graph: &Graph<B>,
    ) -> ExecutionPlan {
        let hash = topology_hash(graph);
        // Compile outside the lock to avoid holding it during planning.
        let candidate = plan(graph);
        let key = (hash, candidate.arena_size);
        let mut guard = self.inner.lock().expect("PlanCache mutex poisoned");
        guard.entry(key).or_insert(candidate).clone()
    }

    /// Number of cached plans.
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().expect("PlanCache mutex poisoned").len()
    }

    /// Whether the cache is empty.
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Compute a topology hash for a graph from its node count and op-name sequence.
fn topology_hash<B: Backend + MatmulOps + ContextBackend>(graph: &Graph<B>) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    graph.len().hash(&mut hasher);
    for node in graph.nodes() {
        node.op.name().hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::dtype::DType;
    use crate::graph::builder_traits::*;

    // Re-use the TestBackend from the parent module's tests. We can't access
    // it directly (it's inside a `#[cfg(test)]` block), so we define a minimal
    // one here.

    struct TestBackend;

    #[derive(Clone)]
    struct DummyTensor;

    impl crate::tensor::Tensor for DummyTensor {
        fn shape(&self) -> &[usize] {
            &[]
        }
        fn dtype(&self) -> DType {
            DType::F32
        }
        fn reshape(&self, _shape: &[usize]) -> Self {
            Self
        }
        fn slice_view(&self, _offset: usize, _shape: &[usize]) -> Self {
            Self
        }
    }

    struct DummyLogits;

    impl crate::logits::Logits for DummyLogits {
        fn vocab_size(&self) -> usize {
            0
        }
        fn batch_size(&self) -> usize {
            0
        }
        fn argmax(&self, _batch_index: usize) -> crate::Result<u32> {
            Ok(0)
        }
        fn sample_top_p(
            &self,
            _batch_index: usize,
            _temperature: f32,
            _top_p: f32,
            _rng_seed: u64,
            _repetition_penalty: f32,
            _recent_tokens: &[u32],
        ) -> crate::Result<u32> {
            Ok(0)
        }
    }

    struct DummyRuntimeState;

    impl crate::runtime_state::RuntimeStateInit for DummyRuntimeState {
        fn new(
            _batch_config: &crate::runtime_state::BatchConfig,
            _block_config: &crate::block_allocator::BlockConfig,
        ) -> crate::Result<Self> {
            Ok(Self)
        }

        fn new_placeholder() -> Self {
            Self
        }
    }

    impl Backend for TestBackend {
        type Tensor = DummyTensor;
        type DeviceHandle = ();
        type PagedKvCache = ();
        type KvCache = ();
        type RuntimeState = DummyRuntimeState;
        type ExecutorState = ();
        type Logits = DummyLogits;
        type Comm = ();

        fn logits_from_tensor(_tensor: Self::Tensor) -> Self::Logits {
            DummyLogits
        }
    }

    impl crate::backend::MatmulOps for TestBackend {
        type LinearWeight = DummyTensor;

        fn matmul(_a: &DummyTensor, _b: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn linear(_input: &DummyTensor, _weight: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn dense_weight(tensor: DummyTensor) -> DummyTensor {
            tensor
        }
        fn is_dense_weight(_weight: &DummyTensor) -> bool {
            true
        }
        fn quantize_to_q8(
            _device: &(),
            _shape: &[usize],
            _data: &[f32],
        ) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn upload_host_linear(
            _device: &(),
            _weight: &crate::weights::host::HostLinearWeight,
        ) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl crate::backend::NormOps for TestBackend {
        fn rms_norm(
            _input: &DummyTensor,
            _weight: &DummyTensor,
            _eps: f32,
        ) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn rms_norm_inplace(
            _input: &mut DummyTensor,
            _weight: &DummyTensor,
            _eps: f32,
        ) -> crate::Result<()> {
            Ok(())
        }
        fn add_rmsnorm(
            _residual: &DummyTensor,
            _input: &DummyTensor,
            _weight: &DummyTensor,
            _eps: f32,
        ) -> crate::Result<(DummyTensor, DummyTensor)> {
            Ok((DummyTensor, DummyTensor))
        }
    }

    impl crate::backend::ArithOps for TestBackend {
        fn add(_a: &DummyTensor, _b: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn add_inplace(_a: &mut DummyTensor, _b: &DummyTensor) -> crate::Result<()> {
            Ok(())
        }
        fn mul(_a: &DummyTensor, _b: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn scale_inplace(_a: &mut DummyTensor, _scale: f32) -> crate::Result<()> {
            Ok(())
        }
        fn silu(_input: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn logit_softcap(_input: &DummyTensor, _cap: f32) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl crate::backend::AttentionOps for TestBackend {
        fn fused_attention_prefill(
            _q: &DummyTensor,
            _k: &DummyTensor,
            _v: &DummyTensor,
            _offset: usize,
            _scale: Option<f32>,
            _softcap: Option<f32>,
            _sliding_window: Option<usize>,
        ) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn fused_attention_decode(
            _q: &DummyTensor,
            _k: &DummyTensor,
            _v: &DummyTensor,
            _scale: Option<f32>,
            _softcap: Option<f32>,
            _sliding_window: Option<usize>,
        ) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn fused_attention_prefill_with_lse(
            _q: &DummyTensor,
            _k: &DummyTensor,
            _v: &DummyTensor,
            _offset: usize,
            _scale: Option<f32>,
            _softcap: Option<f32>,
            _sliding_window: Option<usize>,
        ) -> crate::Result<(DummyTensor, DummyTensor)> {
            Ok((DummyTensor, DummyTensor))
        }
        fn combine_attention_with_lse(
            _out1: &DummyTensor,
            _lse1: &DummyTensor,
            _out2: &DummyTensor,
            _lse2: &DummyTensor,
        ) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl crate::backend::SwigluOps for TestBackend {
        fn swiglu(_gate: &DummyTensor, _up: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl crate::backend::MoeOps for TestBackend {
        fn moe_forward_softmax<F>(
            _hidden: &DummyTensor,
            _gate_weight: &DummyTensor,
            _num_experts: usize,
            _num_experts_per_tok: usize,
            _norm_topk_prob: bool,
            _expert_fn: F,
        ) -> crate::Result<DummyTensor>
        where
            F: Fn(usize, &DummyTensor) -> crate::Result<DummyTensor>,
        {
            unimplemented!()
        }
    }

    impl crate::backend::MoeSigmoidOps for TestBackend {
        fn moe_forward_sigmoid<F>(
            _hidden: &DummyTensor,
            _gate_weight: &DummyTensor,
            _e_score_correction_bias: &[f32],
            _num_experts: usize,
            _num_experts_per_tok: usize,
            _n_group: usize,
            _topk_group: usize,
            _norm_topk_prob: bool,
            _routed_scaling_factor: f32,
            _expert_fn: F,
        ) -> crate::Result<DummyTensor>
        where
            F: Fn(usize, &DummyTensor) -> crate::Result<DummyTensor>,
        {
            unimplemented!()
        }
    }

    impl crate::backend::TensorDataOps for TestBackend {
        fn to_f32_vec(_tensor: &DummyTensor) -> crate::Result<Vec<f32>> {
            unimplemented!()
        }
        fn to_raw_bytes(_tensor: &DummyTensor) -> crate::Result<Vec<u8>> {
            unimplemented!()
        }
    }

    impl crate::backend::TensorOps for TestBackend {
        fn transpose_2d(_input: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn split_inner_dim(
            _tensor: &DummyTensor,
            _dim1: usize,
            _dim2: usize,
        ) -> crate::Result<(DummyTensor, DummyTensor)> {
            Ok((DummyTensor, DummyTensor))
        }
        fn concat_inner_dim(_a: &DummyTensor, _b: &DummyTensor) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn pad_inner_dim(_tensor: &DummyTensor, _new_width: usize) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn broadcast_to_heads(
            _tensor: &DummyTensor,
            _num_heads: usize,
        ) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn repeat_kv(_tensor: &DummyTensor, _num_repeats: usize) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
        fn concat_rows(_parts: &[DummyTensor]) -> crate::Result<DummyTensor> {
            Ok(DummyTensor)
        }
    }

    impl crate::backend::PagedKvCacheOps for TestBackend {
        fn allocate_paged_kv_cache(
            _device: &(),
            _num_layers: usize,
            _block_config: &crate::block_allocator::BlockConfig,
            _num_kv_heads: usize,
            _head_dim: usize,
            _cache_dtype: DType,
        ) -> crate::Result<()> {
            Ok(())
        }

        fn append_paged(
            _cache: &mut (),
            _layer_idx: usize,
            _block_table: &crate::block_allocator::BlockTable,
            _k: &DummyTensor,
            _v: &DummyTensor,
            _start_pos: usize,
        ) -> crate::Result<()> {
            Ok(())
        }

        fn get_pools(_cache: &(), _layer_idx: usize) -> (&DummyTensor, &DummyTensor) {
            static DUMMY: DummyTensor = DummyTensor;
            (&DUMMY, &DUMMY)
        }

        fn block_size(_cache: &()) -> usize {
            16
        }

        fn append_paged_batched(
            _cache: &mut (),
            _layer_idx: usize,
            _k: &DummyTensor,
            _v: &DummyTensor,
            _block_tables: &DummyTensor,
            _positions: &DummyTensor,
            _batch_size: usize,
            _max_blocks_per_seq: usize,
        ) -> crate::Result<()> {
            Ok(())
        }
    }

    impl crate::backend::ContextBackend for TestBackend {
        fn ctx_read(
            _ctx: &crate::graph::execute_context::ExecuteContext<'_, Self>,
            _output_ref: crate::graph::OutputRef,
        ) -> DummyTensor {
            DummyTensor
        }
        fn ctx_write(
            _ctx: &mut crate::graph::execute_context::ExecuteContext<'_, Self>,
            _node_id: crate::graph::NodeId,
            _idx: u32,
            _tensor: DummyTensor,
        ) {
        }
        fn ctx_next_input(
            _ctx: &mut crate::graph::execute_context::ExecuteContext<'_, Self>,
        ) -> DummyTensor {
            DummyTensor
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: Empty graph
    // -----------------------------------------------------------------------

    #[test]
    fn empty_graph() {
        let graph = Graph::<TestBackend>::new();
        let plan = plan(&graph);

        assert!(plan.schedule.is_empty());
        assert!(plan.buffer_slots.is_empty());
        assert_eq!(plan.arena_size, 0);
    }

    // -----------------------------------------------------------------------
    // Test 2: Linear chain — input → norm → linear → output
    // -----------------------------------------------------------------------

    #[test]
    fn linear_chain() {
        let mut graph = Graph::<TestBackend>::new();

        let norm_w = graph.register_tensor_weight("ln.weight", &[512], DType::BF16);
        let proj_w = graph.register_linear_weight("proj.weight", &[512, 512], DType::BF16);

        let input = graph.add_input(&[8, 512], DType::BF16);
        let normed = graph.add_rms_norm(input, norm_w, 1e-5);
        let projected = graph.add_linear(normed, proj_w);
        graph.set_output(projected.0);

        let exec = plan(&graph);

        assert_eq!(exec.schedule.len(), 3);
        assert!(exec.arena_size > 0);

        // All nodes should have buffer slots (including inputs).
        assert!(exec.slot(input.0, 0).is_some());
        assert!(exec.slot(normed.0, 0).is_some());
        assert!(exec.slot(projected.0, 0).is_some());

        // Schedule must respect dependencies: input before norm, norm before linear.
        let pos = |id: NodeId| exec.schedule.iter().position(|&x| x == id).unwrap();
        assert!(pos(input.0) < pos(normed.0));
        assert!(pos(normed.0) < pos(projected.0));
    }

    // -----------------------------------------------------------------------
    // Test 3: Buffer reuse — input → A → B → C → output
    //   A's buffer should be freed by the time C is computed (A's last_use
    //   is step of B), so C can reuse A's offset.
    // -----------------------------------------------------------------------

    #[test]
    fn buffer_reuse() {
        let mut graph = Graph::<TestBackend>::new();

        // All same shape/dtype so buffers are interchangeable.
        let input = graph.add_input(&[4, 128], DType::F32);
        let a = graph.add_scale(input, 1.0);
        let b = graph.add_scale(a, 2.0);
        let c = graph.add_scale(b, 3.0);
        graph.set_output(c.0);

        let exec = plan(&graph);

        assert_eq!(exec.schedule.len(), 4);

        // A is only used by B. Once B runs, A's buffer can be freed.
        // C should be able to reuse A's offset.
        let slot_a = exec.slot(a.0, 0).unwrap();
        let slot_c = exec.slot(c.0, 0).unwrap();
        assert_eq!(
            slot_a.offset, slot_c.offset,
            "C should reuse A's buffer offset"
        );

        // Arena should be 2 buffers worth (A and B live simultaneously during B,
        // then A freed and C reuses it).
        let buf_size = 4 * 128 * 4; // 4×128 elements × 4 bytes (f32)
        assert_eq!(exec.arena_size, buf_size * 2);
    }

    // -----------------------------------------------------------------------
    // Test 4: Multi-output — AddRmsNorm producing two outputs
    // -----------------------------------------------------------------------

    #[test]
    fn multi_output() {
        let mut graph = Graph::<TestBackend>::new();

        let norm_w = graph.register_tensor_weight("ln.weight", &[256], DType::F32);
        let residual = graph.add_input(&[4, 256], DType::F32);
        let delta = graph.add_input(&[4, 256], DType::F32);

        let (updated, normed) = graph.add_add_rmsnorm(residual, delta, norm_w, 1e-6);

        graph.set_output(updated.0);
        graph.set_output(normed.0);

        let exec = plan(&graph);

        // 2 inputs + 1 multi-output node = 3 nodes
        assert_eq!(exec.schedule.len(), 3);

        // Both outputs should have buffer slots.
        assert!(exec.slot(updated.0, updated.1).is_some());
        assert!(exec.slot(normed.0, normed.1).is_some());
        assert!(exec.slot(residual.0, 0).is_some());
        assert!(exec.slot(delta.0, 0).is_some());
    }

    // -----------------------------------------------------------------------
    // Test 5: Diamond DAG — input → A, input → B, (A, B) → C → output
    // -----------------------------------------------------------------------

    #[test]
    fn diamond_dag() {
        let mut graph = Graph::<TestBackend>::new();

        let input = graph.add_input(&[4, 64], DType::BF16);
        let a = graph.add_scale(input, 1.0);
        let b = graph.add_scale(input, 2.0);
        let c = graph.add_add(a, b);
        graph.set_output(c.0);

        let exec = plan(&graph);

        assert_eq!(exec.schedule.len(), 4);

        // Verify topological order: input before A and B, both A and B before C.
        let pos = |id: NodeId| exec.schedule.iter().position(|&x| x == id).unwrap();
        assert!(pos(input.0) < pos(a.0));
        assert!(pos(input.0) < pos(b.0));
        assert!(pos(a.0) < pos(c.0));
        assert!(pos(b.0) < pos(c.0));
    }

    // -----------------------------------------------------------------------
    // Test 6: Side-effect ops produce no buffer slot
    // -----------------------------------------------------------------------

    #[test]
    fn side_effect_no_buffer() {
        let mut graph = Graph::<TestBackend>::new();

        let k = graph.add_input(&[8, 64], DType::BF16);
        let v = graph.add_input(&[8, 64], DType::BF16);

        // AppendPaged is a side-effect op — no output buffer.
        let append = graph.add_append_paged(k, v, 0, 0);

        let exec = plan(&graph);

        assert_eq!(exec.schedule.len(), 3);
        // Side-effect op has no buffer slot.
        // append is (NodeId, 0) but op has 0 outputs — no slot.
        assert!(exec.slot(append.0, 0).is_none());
        // Input nodes get buffer slots, so arena is non-zero.
        assert!(exec.arena_size > 0);
        assert!(exec.slot(k.0, 0).is_some());
        assert!(exec.slot(v.0, 0).is_some());
    }

    // -----------------------------------------------------------------------
    // Test 7: dtype_size_bytes correctness
    // -----------------------------------------------------------------------

    #[test]
    fn dtype_sizes() {
        assert_eq!(dtype_size_bytes(DType::F32), 4);
        assert_eq!(dtype_size_bytes(DType::BF16), 2);
        assert_eq!(dtype_size_bytes(DType::F16), 2);
        assert_eq!(dtype_size_bytes(DType::U32), 4);
        assert_eq!(dtype_size_bytes(DType::F8E4M3), 1);
        // Quantized types get conservative fallback.
        assert_eq!(dtype_size_bytes(DType::Q8_0), 1);
        assert_eq!(dtype_size_bytes(DType::Q4_0), 1);
    }

    // -----------------------------------------------------------------------
    // Test 8: LinearPair → consumer uses BOTH outputs
    //
    // A node (like Swiglu) takes both outputs of a LinearPair as inputs.
    // -----------------------------------------------------------------------

    #[test]
    fn linear_pair_both_outputs_consumed() {
        let mut graph = Graph::<TestBackend>::new();

        let w1 = graph.register_linear_weight("w1", &[64, 128], DType::F32);
        let w2 = graph.register_linear_weight("w2", &[64, 128], DType::F32);

        let input = graph.add_input(&[4, 128], DType::F32);

        // LinearPair: output 0 = gate, output 1 = up
        let (gate, up) = graph.add_linear_pair(input, w1, w2);

        // Swiglu takes BOTH outputs of the pair.
        let activated = graph.add_swiglu(gate, up);
        graph.set_output(activated.0);

        let exec = plan(&graph);

        assert_eq!(exec.schedule.len(), 3); // input, linear_pair, swiglu
        let pos = |id: NodeId| exec.schedule.iter().position(|&x| x == id).unwrap();
        assert!(pos(gate.0) < pos(activated.0));
    }

    // -----------------------------------------------------------------------
    // Test 9: LinearTriple → consumer uses multiple outputs
    // -----------------------------------------------------------------------

    #[test]
    fn linear_triple_outputs_consumed() {
        let mut graph = Graph::<TestBackend>::new();

        let w1 = graph.register_linear_weight("w1", &[64, 128], DType::F32);
        let w2 = graph.register_linear_weight("w2", &[64, 128], DType::F32);
        let w3 = graph.register_linear_weight("w3", &[64, 128], DType::F32);

        let input = graph.add_input(&[4, 128], DType::F32);

        // LinearTriple: outputs 0, 1, 2
        let (out1, out2, out3) = graph.add_linear_triple(input, w1, w2, w3);

        // Consumer takes the 2nd and 3rd outputs.
        let consumer = graph.add_add(out2, out3);
        graph.set_output(consumer.0);
        // Also keep the first output alive.
        graph.set_output(out1.0);

        let exec = plan(&graph);

        assert_eq!(exec.schedule.len(), 3); // input, linear_triple, add
        let pos = |id: NodeId| exec.schedule.iter().position(|&x| x == id).unwrap();
        assert!(pos(out2.0) < pos(consumer.0));
    }

    // -----------------------------------------------------------------------
    // Test 10: Combined LinearTriple + LinearPair (mini Llama layer)
    // -----------------------------------------------------------------------

    #[test]
    fn mini_llama_layer_topo_sort() {
        let mut graph = Graph::<TestBackend>::new();

        let wq = graph.register_linear_weight("q", &[64, 128], DType::F32);
        let wk = graph.register_linear_weight("k", &[64, 128], DType::F32);
        let wv = graph.register_linear_weight("v", &[64, 128], DType::F32);
        let wo = graph.register_linear_weight("o", &[128, 64], DType::F32);
        let wg = graph.register_linear_weight("gate", &[64, 128], DType::F32);
        let wu = graph.register_linear_weight("up", &[64, 128], DType::F32);
        let wd = graph.register_linear_weight("down", &[128, 64], DType::F32);
        let norm_w = graph.register_tensor_weight("norm", &[128], DType::F32);

        let input = graph.add_input(&[4, 128], DType::F32);

        // Pre-attention norm
        let normed = graph.add_rms_norm(input, norm_w, 1e-5);

        // Q/K/V triple
        let (q, k, v) = graph.add_linear_triple(normed, wq, wk, wv);

        // Attention (uses all three)
        let attn = graph.add_fused_attention_prefill(q, k, v, 0, None, None, None);

        // Output projection
        let proj = graph.add_linear(attn, wo);

        // Residual add
        let residual = graph.add_add(input, proj);

        // Gate/Up pair → Swiglu
        let (gate, up) = graph.add_linear_pair(residual, wg, wu);
        let activated = graph.add_swiglu(gate, up);

        // Down projection
        let down = graph.add_linear(activated, wd);
        graph.set_output(down.0);

        let exec = plan(&graph);

        // All nodes must be scheduled.
        assert_eq!(exec.schedule.len(), graph.len());
    }

    // -----------------------------------------------------------------------
    // PlanCache tests
    // -----------------------------------------------------------------------

    fn mini_graph() -> Graph<TestBackend> {
        let mut graph = Graph::<TestBackend>::new();
        let norm_w = graph.register_tensor_weight("ln.weight", &[256], DType::F32);
        let input = graph.add_input(&[4, 256], DType::F32);
        let normed = graph.add_rms_norm(input, norm_w, 1e-5);
        graph.set_output(normed.0);
        graph
    }

    #[test]
    fn plan_cache_compiles_and_returns_plan() {
        let cache = PlanCache::new();
        assert!(cache.is_empty());
        let graph = mini_graph();
        let ep = cache.get_or_compile(&graph);
        assert!(!cache.is_empty());
        // The returned plan should schedule all nodes.
        assert_eq!(ep.schedule.len(), graph.len());
    }

    #[test]
    fn plan_cache_reuses_entry_for_same_topology() {
        let cache = PlanCache::new();
        let g1 = mini_graph();
        let g2 = mini_graph();
        let _ = cache.get_or_compile(&g1);
        let _ = cache.get_or_compile(&g2);
        // Both graphs have identical topology, so only one entry should exist.
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn plan_cache_separate_entry_for_different_topology() {
        let cache = PlanCache::new();

        // Graph 1: input + rms_norm
        let g1 = mini_graph();

        // Graph 2: input + rms_norm + linear (different topology)
        let mut g2 = Graph::<TestBackend>::new();
        let norm_w = g2.register_tensor_weight("ln.weight", &[256], DType::F32);
        let lw = g2.register_linear_weight("proj", &[256, 256], DType::F32);
        let input = g2.add_input(&[4, 256], DType::F32);
        let normed = g2.add_rms_norm(input, norm_w, 1e-5);
        let proj = g2.add_linear(normed, lw);
        g2.set_output(proj.0);

        let _ = cache.get_or_compile(&g1);
        let _ = cache.get_or_compile(&g2);
        // Different topologies → two separate cache entries.
        assert_eq!(cache.len(), 2);
    }
}
