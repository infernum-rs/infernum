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

use crate::backend::Backend;
use crate::dtype::DType;

use super::builder::Graph;
use super::node::NodeId;
use super::ops::Op;

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
    /// Buffer slot for each node's primary output, indexed by `NodeId`.
    /// `None` for `Input` nodes (they use external buffers) and side-effect ops.
    pub buffer_slots: Vec<Option<BufferSlot>>,
    /// Total arena size in bytes (peak memory usage).
    pub arena_size: usize,
}

impl ExecutionPlan {
    /// Get the buffer slot for a given node.
    #[must_use]
    pub fn slot(&self, id: NodeId) -> Option<&BufferSlot> {
        self.buffer_slots[id.0 as usize].as_ref()
    }
}

/// Compute the execution plan for a graph.
///
/// # Panics
/// Panics if the graph contains cycles (should never happen for well-formed graphs).
#[must_use]
pub fn plan<B: Backend>(graph: &Graph<B>) -> ExecutionPlan {
    let schedule = topological_sort(graph);
    let last_use = compute_last_use(graph, &schedule);
    let (buffer_slots, arena_size) = assign_offsets(graph, &schedule, &last_use);
    ExecutionPlan {
        schedule,
        buffer_slots,
        arena_size,
    }
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

/// Returns `true` if the op is a side-effect-only operation that produces no
/// output buffer.
const fn is_side_effect_op(op: &Op) -> bool {
    matches!(op, Op::AppendPaged { .. } | Op::AppendPagedBatched { .. })
}

/// Topological sort using Kahn's algorithm.
///
/// `SecondOutput` nodes are not included in the main BFS — they are inserted
/// immediately after their source node so that both outputs are available
/// before any downstream consumer runs.
#[allow(clippy::cast_possible_truncation)] // graph will never have 2^32 nodes
fn topological_sort<B: Backend>(graph: &Graph<B>) -> Vec<NodeId> {
    let n = graph.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    // Identify SecondOutput nodes and build in-degree for the rest.
    let mut in_degree = vec![0u32; n];
    let mut second_outputs: Vec<Vec<NodeId>> = vec![Vec::new(); n];
    let mut is_second_output = vec![false; n];

    for (i, node) in graph.nodes.iter().enumerate() {
        if let Op::SecondOutput { source } = &node.op {
            is_second_output[i] = true;
            second_outputs[source.0 as usize].push(NodeId(i as u32));
        } else {
            for &_input in &node.inputs {
                // Only count edges to non-SecondOutput nodes.
                if !is_second_output[i] {
                    in_degree[i] += 1;
                }
            }
        }
    }

    // Seed the queue with nodes that have zero in-degree (excluding SecondOutput).
    let mut queue = VecDeque::new();
    for i in 0..n {
        if !is_second_output[i] && in_degree[i] == 0 {
            queue.push_back(NodeId(i as u32));
        }
    }

    let mut schedule = Vec::with_capacity(n);

    while let Some(id) = queue.pop_front() {
        schedule.push(id);

        // Insert any SecondOutput nodes immediately after their source.
        for &second in &second_outputs[id.0 as usize] {
            schedule.push(second);
        }

        // Decrease in-degree for downstream consumers.
        //
        // Count ALL input edges from node j that reference either `id`
        // itself or any of its SecondOutput nodes. A single consumer may
        // reference multiple outputs of the same multi-output op (e.g.,
        // Swiglu takes both outputs of a LinearPair).
        for (j, node) in graph.nodes.iter().enumerate() {
            if is_second_output[j] {
                continue;
            }
            let mut decrements = 0u32;
            for &input in &node.inputs {
                if input == id || second_outputs[id.0 as usize].contains(&input) {
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

    let non_second_count = is_second_output.iter().filter(|&&b| !b).count();
    let second_count = n - non_second_count;
    assert_eq!(
        schedule.len(),
        non_second_count + second_count,
        "Graph contains a cycle: scheduled {} of {} nodes",
        schedule.len(),
        n,
    );

    schedule
}

/// For each node, compute the latest execution step where it is still needed
/// as an input (i.e., its "last use"). Graph outputs are live until the last step.
fn compute_last_use<B: Backend>(graph: &Graph<B>, schedule: &[NodeId]) -> Vec<usize> {
    let n = graph.nodes.len();
    // Default to 0; we'll update with max step.
    let mut last_use = vec![0usize; n];

    for (step, &id) in schedule.iter().enumerate() {
        let node = &graph.nodes[id.0 as usize];
        for &input in &node.inputs {
            let idx = input.0 as usize;
            if step > last_use[idx] {
                last_use[idx] = step;
            }
        }

        // SecondOutput references its source — the source must stay live at
        // least until this step.
        if let Op::SecondOutput { source } = &node.op {
            let idx = source.0 as usize;
            if step > last_use[idx] {
                last_use[idx] = step;
            }
        }
    }

    // Graph outputs are live until the final step.
    if let Some(&last_step) = schedule.len().checked_sub(1).as_ref() {
        for &out in &graph.outputs {
            let idx = out.0 as usize;
            if last_step > last_use[idx] {
                last_use[idx] = last_step;
            }
        }
    }

    last_use
}

/// Compute the buffer size in bytes for a node's output.
fn buffer_size(shape: &[usize], dtype: DType) -> usize {
    let elements: usize = shape.iter().product();
    elements * dtype_size_bytes(dtype)
}

/// Greedy offset assignment within a shared arena.
///
/// Maintains a sorted free list of `(offset, size)` gaps. When a buffer dies
/// (its `last_use < current_step`), its range is returned to the free list
/// and merged with adjacent gaps.
#[allow(clippy::cast_possible_truncation)]
fn assign_offsets<B: Backend>(
    graph: &Graph<B>,
    schedule: &[NodeId],
    last_use: &[usize],
) -> (Vec<Option<BufferSlot>>, usize) {
    let n = graph.nodes.len();
    let mut buffer_slots: Vec<Option<BufferSlot>> = vec![None; n];
    let mut arena_size: usize = 0;

    // Free list: sorted by offset. Each entry is (offset, size).
    let mut free_list: Vec<(usize, usize)> = Vec::new();

    // Track which buffers are currently live so we can free them.
    // (node_index, offset, size, last_use_step)
    let mut live_buffers: Vec<(usize, usize, usize, usize)> = Vec::new();

    for (step, &id) in schedule.iter().enumerate() {
        // Free buffers whose last_use < current step.
        live_buffers.retain(|&(_, offset, size, last_step)| {
            if last_step < step {
                insert_free_range(&mut free_list, offset, size);
                false
            } else {
                true
            }
        });

        let idx = id.0 as usize;
        let node = &graph.nodes[idx];

        // Side-effect ops produce no output buffer.
        if is_side_effect_op(&node.op) {
            continue;
        }

        let size = buffer_size(&node.shape, node.dtype);
        if size == 0 {
            buffer_slots[idx] = Some(BufferSlot { offset: 0, size: 0 });
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

        buffer_slots[idx] = Some(BufferSlot { offset, size });
        live_buffers.push((idx, offset, size, last_use[idx]));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::dtype::DType;

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
        type Logits = DummyLogits;
        type Comm = ();

        fn logits_from_tensor(_tensor: Self::Tensor) -> Self::Logits {
            DummyLogits
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
        let normed = graph.push_node(
            Op::RmsNorm {
                weight: norm_w,
                eps: 1e-5,
            },
            &[input],
            vec![8, 512],
            DType::BF16,
        );
        let projected = graph.push_node(
            Op::Linear { weight: proj_w },
            &[normed],
            vec![8, 512],
            DType::BF16,
        );
        graph.set_output(projected);

        let exec = plan(&graph);

        assert_eq!(exec.schedule.len(), 3);
        assert!(exec.arena_size > 0);

        // All nodes should have buffer slots (including inputs).
        assert!(exec.slot(input).is_some());
        assert!(exec.slot(normed).is_some());
        assert!(exec.slot(projected).is_some());

        // Schedule must respect dependencies: input before norm, norm before linear.
        let pos = |id: NodeId| exec.schedule.iter().position(|&x| x == id).unwrap();
        assert!(pos(input) < pos(normed));
        assert!(pos(normed) < pos(projected));
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
        let a = graph.push_node(
            Op::Scale { factor: 1.0 },
            &[input],
            vec![4, 128],
            DType::F32,
        );
        let b = graph.push_node(Op::Scale { factor: 2.0 }, &[a], vec![4, 128], DType::F32);
        let c = graph.push_node(Op::Scale { factor: 3.0 }, &[b], vec![4, 128], DType::F32);
        graph.set_output(c);

        let exec = plan(&graph);

        assert_eq!(exec.schedule.len(), 4);

        // A is only used by B. Once B runs, A's buffer can be freed.
        // C should be able to reuse A's offset.
        let slot_a = exec.slot(a).unwrap();
        let slot_c = exec.slot(c).unwrap();
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

        let (updated, normed) = graph.push_node_pair(
            Op::AddRmsNorm {
                weight: norm_w,
                eps: 1e-6,
            },
            &[residual, delta],
            vec![4, 256],
            DType::F32,
            vec![4, 256],
            DType::F32,
        );

        graph.set_output(updated);
        graph.set_output(normed);

        let exec = plan(&graph);

        // 2 inputs + primary + secondary = 4 nodes
        assert_eq!(exec.schedule.len(), 4);

        // All nodes should have buffer slots (including inputs).
        assert!(exec.slot(updated).is_some());
        assert!(exec.slot(normed).is_some());
        assert!(exec.slot(residual).is_some());
        assert!(exec.slot(delta).is_some());
    }

    // -----------------------------------------------------------------------
    // Test 5: Diamond DAG — input → A, input → B, (A, B) → C → output
    // -----------------------------------------------------------------------

    #[test]
    fn diamond_dag() {
        let mut graph = Graph::<TestBackend>::new();

        let input = graph.add_input(&[4, 64], DType::BF16);
        let a = graph.push_node(
            Op::Scale { factor: 1.0 },
            &[input],
            vec![4, 64],
            DType::BF16,
        );
        let b = graph.push_node(
            Op::Scale { factor: 2.0 },
            &[input],
            vec![4, 64],
            DType::BF16,
        );
        let c = graph.push_node(Op::Add, &[a, b], vec![4, 64], DType::BF16);
        graph.set_output(c);

        let exec = plan(&graph);

        assert_eq!(exec.schedule.len(), 4);

        // Verify topological order: input before A and B, both A and B before C.
        let pos = |id: NodeId| exec.schedule.iter().position(|&x| x == id).unwrap();
        assert!(pos(input) < pos(a));
        assert!(pos(input) < pos(b));
        assert!(pos(a) < pos(c));
        assert!(pos(b) < pos(c));
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
        let append = graph.push_node(
            Op::AppendPaged {
                layer_idx: 0,
                start_pos: 0,
            },
            &[k, v],
            vec![],
            DType::BF16,
        );

        let exec = plan(&graph);

        assert_eq!(exec.schedule.len(), 3);
        // Side-effect op has no buffer slot.
        assert!(exec.slot(append).is_none());
        // Input nodes get buffer slots, so arena is non-zero.
        assert!(exec.arena_size > 0);
        // But both inputs share/reuse arena space since they're read before append.
        assert!(exec.slot(k).is_some());
        assert!(exec.slot(v).is_some());
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
    // Test 8: LinearPair → consumer uses BOTH primary and SecondOutput
    //
    // This reproduces the bug where a node (like Swiglu) takes both outputs
    // of a LinearPair as inputs. The if/else in the BFS decrement only
    // counted one dependency edge instead of two.
    // -----------------------------------------------------------------------

    #[test]
    fn linear_pair_both_outputs_consumed() {
        let mut graph = Graph::<TestBackend>::new();

        let w1 = graph.register_linear_weight("w1", &[64, 128], DType::F32);
        let w2 = graph.register_linear_weight("w2", &[64, 128], DType::F32);

        let input = graph.add_input(&[4, 128], DType::F32);

        // LinearPair: primary = gate, secondary = up
        let (gate, up) = graph.push_node_pair(
            Op::LinearPair { w1, w2 },
            &[input],
            vec![4, 64],
            DType::F32,
            vec![4, 64],
            DType::F32,
        );

        // Swiglu takes BOTH outputs of the pair.
        let activated = graph.push_node(Op::Swiglu, &[gate, up], vec![4, 64], DType::F32);
        graph.set_output(activated);

        // This panics before the fix: "scheduled 3 of 4 nodes"
        let exec = plan(&graph);

        assert_eq!(exec.schedule.len(), 4); // input, gate, up(second), swiglu
        let pos = |id: NodeId| exec.schedule.iter().position(|&x| x == id).unwrap();
        assert!(pos(gate) < pos(activated));
        assert!(pos(up) < pos(activated));
    }

    // -----------------------------------------------------------------------
    // Test 9: LinearTriple → two SecondOutputs feed the same consumer
    //
    // Reproduces the break-in-else bug: a node takes the 2nd and 3rd
    // outputs of a LinearTriple. The break after the first match means
    // the second dependency is never decremented.
    // -----------------------------------------------------------------------

    #[test]
    fn linear_triple_two_second_outputs_same_consumer() {
        let mut graph = Graph::<TestBackend>::new();

        let w1 = graph.register_linear_weight("w1", &[64, 128], DType::F32);
        let w2 = graph.register_linear_weight("w2", &[64, 128], DType::F32);
        let w3 = graph.register_linear_weight("w3", &[64, 128], DType::F32);

        let input = graph.add_input(&[4, 128], DType::F32);

        // LinearTriple: primary = out1, second = out2, third = out3
        let primary = graph.push_node(
            Op::LinearTriple { w1, w2, w3 },
            &[input],
            vec![4, 64],
            DType::F32,
        );
        let second = graph.push_node(
            Op::SecondOutput { source: primary },
            &[],
            vec![4, 64],
            DType::F32,
        );
        let third = graph.push_node(
            Op::SecondOutput { source: primary },
            &[],
            vec![4, 64],
            DType::F32,
        );

        // Consumer takes the 2nd and 3rd outputs (both SecondOutput nodes).
        let consumer = graph.push_node(Op::Add, &[second, third], vec![4, 64], DType::F32);
        graph.set_output(consumer);

        let exec = plan(&graph);

        assert_eq!(exec.schedule.len(), 5); // input, primary, second, third, consumer
        let pos = |id: NodeId| exec.schedule.iter().position(|&x| x == id).unwrap();
        assert!(pos(second) < pos(consumer));
        assert!(pos(third) < pos(consumer));
    }

    // -----------------------------------------------------------------------
    // Test 10: Combined LinearTriple + LinearPair (mini Llama layer)
    //
    // Mimics one transformer layer: LinearTriple for Q/K/V, then
    // downstream ops, then LinearPair for gate/up → Swiglu.
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
        let normed = graph.push_node(
            Op::RmsNorm {
                weight: norm_w,
                eps: 1e-5,
            },
            &[input],
            vec![4, 128],
            DType::F32,
        );

        // Q/K/V triple
        let (q, k, v) = graph.push_node_triple(
            Op::LinearTriple {
                w1: wq,
                w2: wk,
                w3: wv,
            },
            &[normed],
            vec![4, 64],
            DType::F32,
            vec![4, 64],
            DType::F32,
            vec![4, 64],
            DType::F32,
        );

        // Attention (uses all three)
        let attn = graph.push_node(
            Op::FusedAttentionPrefill {
                offset: 0,
                scale: None,
                softcap: None,
                sliding_window: None,
            },
            &[q, k, v],
            vec![4, 64],
            DType::F32,
        );

        // Output projection
        let proj = graph.push_node(Op::Linear { weight: wo }, &[attn], vec![4, 128], DType::F32);

        // Residual add
        let residual = graph.push_node(Op::Add, &[input, proj], vec![4, 128], DType::F32);

        // Gate/Up pair → Swiglu
        let (gate, up) = graph.push_node_pair(
            Op::LinearPair { w1: wg, w2: wu },
            &[residual],
            vec![4, 64],
            DType::F32,
            vec![4, 64],
            DType::F32,
        );
        let activated = graph.push_node(Op::Swiglu, &[gate, up], vec![4, 64], DType::F32);

        // Down projection
        let down = graph.push_node(
            Op::Linear { weight: wd },
            &[activated],
            vec![4, 128],
            DType::F32,
        );
        graph.set_output(down);

        let exec = plan(&graph);

        // All nodes must be scheduled.
        assert_eq!(exec.schedule.len(), graph.len());
    }
}
