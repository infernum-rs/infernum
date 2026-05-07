//! Graph executor for the CPU backend.
//!
//! Walks an [`ExecutionPlan`] in topological order and dispatches each
//! operation to the appropriate CPU SIMD kernel. Intermediate tensors are
//! read from / written to a shared [`Arena`].
//!
//! ## Dispatch strategy
//!
//! All ops are dispatched through the `_ =>` fallback arm, which constructs
//! an [`ExecuteContext`] and calls `node.op.execute(ctx)`. Each op's own
//! `execute()` implementation in `builtin_ops.rs` handles the logic.

use std::collections::HashMap;

use infernum::graph::execute_context::KvCacheAccess;
use infernum::graph::{Arena, GraphNode, OutputRef, WeightStore};
use infernum::{DType, ExecutionPlan, NodeId, Result};

use crate::tensor::{CpuLinearWeight, CpuTensor};
use crate::CpuBackend;

/// Persistent KV cache storage for decode-mode graph execution.
///
/// Instead of passing KV caches as graph inputs and copying them through
/// the arena every step, this store holds pre-allocated buffers that grow
/// via in-place append. This eliminates the O(seq_len) per-step copy cost.
pub struct KvCacheStore {
    /// Per-layer K cache: `[current_len, num_kv_heads, head_dim]` stored flat.
    k_caches: Vec<Vec<f32>>,
    /// Per-layer V cache: `[current_len, num_kv_heads, head_dim]` stored flat.
    v_caches: Vec<Vec<f32>>,
    /// Number of positions currently stored (same for all layers).
    len: usize,
    /// Number of KV heads (used to construct shapes).
    num_kv_heads: usize,
    /// Head dimension (used to construct shapes).
    head_dim: usize,
    /// Node IDs of the KV cache Input nodes in the graph, in order:
    /// `[k_cache_0, v_cache_0, k_cache_1, v_cache_1, ...]`
    cache_input_node_ids: Vec<NodeId>,
    /// Node IDs of the `ConcatSeq` nodes that append to KV caches, in order:
    /// `[k_concat_0, v_concat_0, k_concat_1, v_concat_1, ...]`
    concat_node_ids: Vec<NodeId>,
    /// Tensor overrides: nodes whose output lives outside the arena (e.g.,
    /// KV caches in the persistent `KvCacheStore`). Populated by
    /// `try_append_kv` and read by downstream attention ops.
    overrides: HashMap<NodeId, CpuTensor>,
}

impl KvCacheStore {
    /// Create a new empty KV cache store.
    ///
    /// # Arguments
    /// * `num_layers` — Number of transformer layers.
    /// * `num_kv_heads` — Number of KV attention heads per layer.
    /// * `head_dim` — Dimension of each attention head.
    /// * `max_seq_len` — Maximum sequence length to pre-allocate for (avoids
    ///   `Vec` reallocations during generation).
    /// * `cache_input_node_ids` — `NodeId`s of KV cache Input nodes in the graph.
    /// * `concat_node_ids` — `NodeId`s of `ConcatSeq` nodes that append to KV caches.
    #[must_use]
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        cache_input_node_ids: Vec<NodeId>,
        concat_node_ids: Vec<NodeId>,
    ) -> Self {
        assert_eq!(cache_input_node_ids.len(), 2 * num_layers);
        assert_eq!(concat_node_ids.len(), 2 * num_layers);
        let row_size = num_kv_heads * head_dim;
        let cap = max_seq_len * row_size;
        Self {
            k_caches: (0..num_layers).map(|_| Vec::with_capacity(cap)).collect(),
            v_caches: (0..num_layers).map(|_| Vec::with_capacity(cap)).collect(),
            len: 0,
            num_kv_heads,
            head_dim,
            cache_input_node_ids,
            concat_node_ids,
            overrides: HashMap::new(),
        }
    }

    /// Append a new K or V row to the specified layer's cache.
    fn append(&mut self, layer: usize, is_key: bool, data: &[f32]) {
        let row_size = self.num_kv_heads * self.head_dim;
        assert_eq!(data.len(), row_size, "KV row size mismatch");
        let cache = if is_key {
            &mut self.k_caches[layer]
        } else {
            &mut self.v_caches[layer]
        };
        cache.extend_from_slice(data);
    }

    /// Get the full K or V cache for a layer as a `CpuTensor` view.
    fn get_cache(&self, layer: usize, is_key: bool, len: usize) -> CpuTensor {
        let cache = if is_key {
            &self.k_caches[layer]
        } else {
            &self.v_caches[layer]
        };
        let shape = [len, self.num_kv_heads, self.head_dim];
        CpuTensor::from_f32(&shape, &cache[..len * self.num_kv_heads * self.head_dim])
    }

    /// Check if a node ID is a KV cache input node.
    #[allow(dead_code)]
    fn is_cache_input(&self, node_id: NodeId) -> bool {
        self.cache_input_node_ids.contains(&node_id)
    }

    /// Check if a node ID is a KV cache `ConcatSeq` node.
    /// Returns `(layer_index, is_key)` if found.
    fn cache_concat_info(&self, node_id: NodeId) -> Option<(usize, bool)> {
        self.concat_node_ids
            .iter()
            .position(|&id| id == node_id)
            .map(|pos| {
                let layer = pos / 2;
                let is_key = pos % 2 == 0;
                (layer, is_key)
            })
    }

    /// Current number of cached positions.
    #[must_use]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if no positions are cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl KvCacheAccess<CpuBackend> for KvCacheStore {
    fn is_cache_input(&self, node_id: NodeId) -> bool {
        self.cache_input_node_ids.contains(&node_id)
    }

    fn read_cache(&self, node_id: NodeId) -> Option<&CpuTensor> {
        self.overrides.get(&node_id)
    }

    fn write_cache(&mut self, node_id: NodeId, tensor: CpuTensor) {
        self.overrides.insert(node_id, tensor);
    }

    fn cache_concat_info(&self, node_id: NodeId) -> Option<(usize, bool)> {
        self.concat_node_ids
            .iter()
            .position(|&id| id == node_id)
            .map(|pos| {
                let layer = pos / 2;
                let is_key = pos % 2 == 0;
                (layer, is_key)
            })
    }

    fn try_append_kv(&mut self, node_id: NodeId, new_row: &CpuTensor) -> Option<CpuTensor> {
        let (layer, is_key) = self.cache_concat_info(node_id)?;
        self.append(layer, is_key, new_row.as_f32_slice());
        let row_elems = self.num_kv_heads * self.head_dim;
        let new_len = if is_key {
            self.k_caches[layer].len() / row_elems
        } else {
            self.v_caches[layer].len() / row_elems
        };
        let full_cache = self.get_cache(layer, is_key, new_len);
        self.overrides.insert(node_id, full_cache.clone());
        Some(full_cache)
    }

    fn finalize_step(&mut self) {
        self.len += 1;
    }
}

/// Read a tensor from the arena using the plan's buffer slot for an `OutputRef`.
fn read_tensor(
    arena: &Arena,
    plan: &ExecutionPlan,
    nodes: &[GraphNode<CpuBackend>],
    output_ref: OutputRef,
) -> CpuTensor {
    let (node_id, output_idx) = output_ref;
    let node = &nodes[node_id.index() as usize];
    let slot = plan
        .slot(node_id, output_idx)
        .expect("node output has no buffer slot");
    let shape = &node.output_shapes[output_idx as usize];
    let dtype = node.output_dtypes[output_idx as usize];
    let num_elements: usize = shape.iter().product();
    if dtype == DType::U32 {
        let data = arena.u32_slice(slot.offset, num_elements);
        CpuTensor::from_u32(shape, data)
    } else {
        let data = arena.f32_slice(slot.offset, num_elements);
        CpuTensor::from_f32(shape, data)
    }
}

/// Execute a computation graph on the CPU backend.
///
/// Walks the `plan.schedule` in topological order. Built-in ops are dispatched
/// via a `match op_name { ... }` block (zero overhead, no trait-object call).
/// Rope-fusion look-ahead is handled inline in that block. Unknown or custom
/// ops fall through to the `_ =>` arm, which constructs an [`ExecuteContext`]
/// and calls `node.op.execute(ctx)` — the open-dispatch path for ops added by
/// external crates. Intermediate tensors are stored in the shared `arena`.
///
/// # Arguments
///
/// * `plan` — Memory plan with schedule, buffer offsets, and arena size.
/// * `nodes` — Graph nodes (indexed by `NodeId`).
/// * `arena` — Pre-allocated byte arena for intermediate tensors.
/// * `weights` — Loaded model weights (embedding, norm, linear, bias, etc.).
/// * `inputs` — External input tensors, one per `InputOp` in schedule order.
/// * `output_nodes` — `NodeId`s of the graph outputs to collect.
///
/// # Errors
///
/// Returns an error if any op kernel fails.
#[allow(clippy::too_many_lines, clippy::cast_ptr_alignment)]
pub fn execute(
    plan: &ExecutionPlan,
    nodes: &[GraphNode<CpuBackend>],
    arena: &mut Arena,
    weights: &WeightStore<CpuTensor, CpuLinearWeight>,
    inputs: &[CpuTensor],
    output_nodes: &[NodeId],
    mut kv_cache: Option<&mut KvCacheStore>,
) -> Result<Vec<CpuTensor>> {
    let mut input_idx: usize = 0;
    // Tensor overrides: nodes whose output lives outside the arena (e.g., KV
    // caches in the persistent `KvCacheStore`). `read_tensor` is bypassed for
    // these — the override is consumed directly instead of copying through arena.
    let mut overrides: HashMap<NodeId, CpuTensor> = HashMap::new();

    for &node_id in &plan.schedule {
        let node = &nodes[node_id.index() as usize];
        let op_name = node.op.name();

        match op_name {
            // --- Open dispatch: custom / unknown ops self-execute via OpNode::execute ---
            _ => {
                let mut ctx = infernum::graph::execute_context::ExecuteContext {
                    state: arena,
                    plan,
                    nodes,
                    weights,
                    device: &(),
                    kv_cache: kv_cache.as_deref_mut().map(|kv| {
                        kv as &mut dyn infernum::graph::execute_context::KvCacheAccess<CpuBackend>
                    }),
                    input_tensors: inputs,
                    input_idx: &mut input_idx,
                };
                node.op.execute(&mut ctx, node_id, &node.inputs)?;
            }
        }
    }

    // Update persistent KV cache length after all layers have appended.
    if let Some(kv) = kv_cache {
        kv.len += 1;
    }

    // Collect output tensors (check overrides first, then arena).
    let outputs = output_nodes
        .iter()
        .map(|&id| {
            overrides
                .remove(&id)
                .unwrap_or_else(|| read_tensor(arena, plan, nodes, (id, 0)))
        })
        .collect();

    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use infernum::dtype::DType;
    use infernum::graph::Graph;
    use infernum::{plan, GraphArithOps, GraphMatmulOps, GraphNormOps};

    /// Test 1: Simple add graph — two inputs added together.
    #[test]
    fn simple_add_graph() {
        let mut graph = Graph::<CpuBackend>::new();

        let a = graph.add_input(&[4], DType::F32);
        let b = graph.add_input(&[4], DType::F32);
        let c = graph.add_add(a, b);
        graph.set_output(c.0);

        let exec_plan = plan(&graph);
        let mut arena = Arena::new(exec_plan.arena_size);
        let weights = WeightStore::<CpuTensor, CpuLinearWeight>::new();

        let input_a = CpuTensor::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let input_b = CpuTensor::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]);

        let outputs = execute(
            &exec_plan,
            graph.nodes(),
            &mut arena,
            &weights,
            &[input_a, input_b],
            graph.output_ids(),
            None,
        )
        .unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_f32_slice(), &[11.0, 22.0, 33.0, 44.0]);
    }

    /// Test 2: Linear projection — input × weight.
    #[test]
    fn linear_projection() {
        let mut graph = Graph::<CpuBackend>::new();

        // Register a dense linear weight: (out_features=2, in_features=3).
        let w_id = graph.register_linear_weight("proj", &[2, 3], DType::F32);
        let input_node = graph.add_input(&[1, 3], DType::F32);
        let output_node = graph.add_linear(input_node, w_id);
        graph.set_output(output_node.0);

        let exec_plan = plan(&graph);
        let mut arena = Arena::new(exec_plan.arena_size);

        // Build the weight store with a dense linear weight.
        // new_dense takes (in_features, out_features) = (3, 2) and creates both layouts.
        let weight_data = CpuTensor::from_f32(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let linear_w = CpuLinearWeight::new_dense(weight_data);
        let mut weights = WeightStore::<CpuTensor, CpuLinearWeight>::new();
        weights.push_linear_weight(linear_w);

        let input = CpuTensor::from_f32(&[1, 3], &[3.0, 5.0, 7.0]);

        let outputs = execute(
            &exec_plan,
            graph.nodes(),
            &mut arena,
            &weights,
            &[input],
            graph.output_ids(),
            None,
        )
        .unwrap();

        assert_eq!(outputs.len(), 1);
        let result = outputs[0].as_f32_slice();
        // [3, 5, 7] × [[1, 0], [0, 1], [0, 0]] = [3, 5]
        assert_eq!(result, &[3.0, 5.0]);
    }

    /// Test 3: Norm + linear chain — rms_norm then linear projection.
    #[test]
    fn norm_then_linear() {
        let mut graph = Graph::<CpuBackend>::new();

        let norm_w_id = graph.register_tensor_weight("ln.weight", &[4], DType::F32);
        let proj_w_id = graph.register_linear_weight("proj", &[2, 4], DType::F32);

        let input_node = graph.add_input(&[1, 4], DType::F32);
        let normed = graph.add_rms_norm(input_node, norm_w_id, 1e-5);
        let output_node = graph.add_linear(normed, proj_w_id);
        graph.set_output(output_node.0);

        let exec_plan = plan(&graph);
        let mut arena = Arena::new(exec_plan.arena_size);

        // Norm weight: all ones (RMS norm with unit weight is just x / rms(x)).
        let norm_weight = CpuTensor::from_f32(&[4], &[1.0, 1.0, 1.0, 1.0]);

        // Linear weight: identity-ish (4, 2) — pick first two dims.
        let weight_data = CpuTensor::from_f32(&[4, 2], &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let linear_w = CpuLinearWeight::new_dense(weight_data);

        let mut weights = WeightStore::<CpuTensor, CpuLinearWeight>::new();
        weights.push_tensor_weight(norm_weight);
        weights.push_linear_weight(linear_w);

        let input = CpuTensor::from_f32(&[1, 4], &[1.0, 2.0, 3.0, 4.0]);

        let outputs = execute(
            &exec_plan,
            graph.nodes(),
            &mut arena,
            &weights,
            &[input],
            graph.output_ids(),
            None,
        )
        .unwrap();

        assert_eq!(outputs.len(), 1);
        let result = outputs[0].as_f32_slice();

        // RMS of [1, 2, 3, 4] = sqrt((1+4+9+16)/4) = sqrt(30/4) ≈ 2.7386
        // Normalized: [1/rms, 2/rms, 3/rms, 4/rms]
        // Linear picks first two: [1/rms, 2/rms]
        let rms = (30.0_f32 / 4.0).sqrt();
        let expected = [1.0 / rms, 2.0 / rms];
        assert!(
            (result[0] - expected[0]).abs() < 1e-5,
            "expected {}, got {}",
            expected[0],
            result[0]
        );
        assert!(
            (result[1] - expected[1]).abs() < 1e-5,
            "expected {}, got {}",
            expected[1],
            result[1]
        );
    }

    /// Test 4: End-to-end Llama prefill graph with synthetic weights.
    ///
    /// Gated behind `integration` until `infernum-llama` is migrated to the
    /// new graph IR.
    #[cfg(feature = "integration")]
    #[test]
    fn llama_prefill_graph_synthetic() {
        use infernum::graph::WeightId;
        use infernum_llama::build_prefill_graph;
        use infernum_llama::LlamaConfig;

        let config: LlamaConfig = serde_json::from_str(
            r#"{"vocab_size": 256, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4, "rms_norm_eps": 1e-5, "rope_theta": 10000.0}"#,
        )
        .unwrap();

        let seq_len = 8;
        let head_dim = config.head_dim();
        let half_dim = head_dim / 2;

        let (mut graph, _model_weights) =
            build_prefill_graph::<CpuBackend>(&config, seq_len, DType::F32);
        infernum::graph::optimizer::optimize(&mut graph);
        let exec_plan = plan(&graph);
        let mut arena = Arena::new(exec_plan.arena_size);

        let mut weights = WeightStore::<CpuTensor, CpuLinearWeight>::new();

        // Push tensor weights in registration order.
        for i in 0..graph.tensor_weight_count() {
            let meta = graph.tensor_weight_meta(WeightId::from_index(i as u32));
            let size: usize = meta.shape.iter().product();
            let data: Vec<f32> = (0..size).map(|j| (j as f32 * 0.01) % 1.0).collect();
            weights.push_tensor_weight(CpuTensor::from_f32(&meta.shape, &data));
        }

        // Graph registers [N, K]; `new_dense` expects (K, N).
        for i in 0..graph.linear_weight_count() {
            let meta = graph.linear_weight_meta(WeightId::from_index(i as u32));
            let n = meta.shape[0];
            let k = meta.shape[1];
            let size = k * n;
            let data: Vec<f32> = (0..size)
                .map(|j| ((j as f32 * 0.001) - 0.5) * 0.1)
                .collect();
            let tensor = CpuTensor::from_f32(&[k, n], &data);
            weights.push_linear_weight(CpuLinearWeight::new_dense(tensor));
        }

        let token_ids: Vec<u32> = (0..seq_len).map(|i| (i * 7 % 256) as u32).collect();
        let input_ids = CpuTensor::from_u32(&[seq_len], &token_ids);
        let cos_data: Vec<f32> = (0..seq_len * half_dim)
            .map(|i| (i as f32 * 0.1).cos())
            .collect();
        let sin_data: Vec<f32> = (0..seq_len * half_dim)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let cos_cache = CpuTensor::from_f32(&[seq_len, half_dim], &cos_data);
        let sin_cache = CpuTensor::from_f32(&[seq_len, half_dim], &sin_data);

        let inputs = vec![input_ids, cos_cache, sin_cache];

        let outputs = execute(
            &exec_plan,
            graph.nodes(),
            &mut arena,
            &weights,
            &inputs,
            graph.output_ids(),
            None,
        )
        .unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape(), &[seq_len, config.vocab_size]);

        let logits = outputs[0].as_f32_slice();
        for (i, &v) in logits.iter().enumerate() {
            assert!(!v.is_nan(), "NaN at logit index {i}");
            assert!(!v.is_infinite(), "Inf at logit index {i}");
        }
    }

    /// Verify that the Gemma decode graph produces different logits across
    /// autoregressive steps when KV cache accumulates correctly.
    ///
    /// Uses a tiny Gemma 2 config with synthetic weights. Runs 3 decode steps
    /// and asserts that logits differ between steps, proving the model sees a
    /// growing context rather than a static (stuck) state.
    #[test]
    fn gemma_decode_graph_kv_accumulates() {
        use crate::graph_engine::find_kv_cache_node_ids;
        use infernum::graph::optimizer;
        use infernum::graph::WeightId;
        use infernum::rope::precompute_rope_data;
        use infernum_gemma::{build_decode_graph, GemmaConfig};

        // Tiny Gemma 2 config — small enough to run without a GPU.
        let config = GemmaConfig::from_str(
            r#"{
                "model_type": "gemma2",
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "query_pre_attn_scalar": 16.0,
                "attn_logit_softcapping": null,
                "final_logit_softcapping": null,
                "sliding_window": null,
                "tie_word_embeddings": true,
                "bos_token_id": 2,
                "eos_token_id": 1
            }"#,
        );

        let head_dim = config.head_dim;
        let half_dim = head_dim / 2;
        let num_layers = config.num_hidden_layers;

        let mut graph = build_decode_graph::<CpuBackend>(&config, 0, DType::F32);
        optimizer::optimize(&mut graph);
        let exec_plan = plan(&graph);
        let mut arena = Arena::new(exec_plan.arena_size);

        // Build synthetic weights.
        let mut weights = WeightStore::<CpuTensor, CpuLinearWeight>::new();
        for i in 0..graph.tensor_weight_count() {
            let meta = graph.tensor_weight_meta(WeightId::from_index(i as u32));
            let size: usize = meta.shape.iter().product();
            // Use non-uniform values to avoid degenerate outputs.
            let data: Vec<f32> = (0..size)
                .map(|j| ((j as f32 * 0.07 + 0.3) % 1.5) - 0.5)
                .collect();
            weights.push_tensor_weight(CpuTensor::from_f32(&meta.shape, &data));
        }
        for i in 0..graph.linear_weight_count() {
            let meta = graph.linear_weight_meta(WeightId::from_index(i as u32));
            let (n, k) = (meta.shape[0], meta.shape[1]);
            let size = k * n;
            let data: Vec<f32> = (0..size)
                .map(|j| ((j as f32 * 0.03 + 0.1) % 0.8) - 0.4)
                .collect();
            weights.push_linear_weight(CpuLinearWeight::new_dense(CpuTensor::from_f32(
                &[k, n],
                &data,
            )));
        }

        // Set up KV cache store via find_kv_cache_node_ids.
        let (cache_input_ids, concat_ids) = find_kv_cache_node_ids(graph.nodes(), num_layers);
        let mut kv_cache = KvCacheStore::new(
            num_layers,
            config.num_key_value_heads,
            head_dim,
            64,
            cache_input_ids,
            concat_ids,
        );

        let (cos_table, sin_table) = precompute_rope_data(64, head_dim, config.rope_theta);
        let logits_id = graph.output_ids()[0];

        let mut prev_logits: Option<Vec<f32>> = None;
        for pos in 0..3 {
            let cos_start = pos * half_dim;
            let inputs = vec![
                CpuTensor::from_u32(&[1], &[(pos as u32 + 5) % 64]),
                CpuTensor::from_f32(&[1, half_dim], &cos_table[cos_start..cos_start + half_dim]),
                CpuTensor::from_f32(&[1, half_dim], &sin_table[cos_start..cos_start + half_dim]),
            ];

            let outputs = execute(
                &exec_plan,
                graph.nodes(),
                &mut arena,
                &weights,
                &inputs,
                &[logits_id],
                Some(&mut kv_cache),
            )
            .unwrap();

            let logits = outputs[0].as_f32_slice().to_vec();
            assert_eq!(logits.len(), config.vocab_size, "logit count at step {pos}");
            for (i, &v) in logits.iter().enumerate() {
                assert!(!v.is_nan(), "NaN at logit[{i}] in step {pos}");
                assert!(!v.is_infinite(), "Inf at logit[{i}] in step {pos}");
            }

            if let Some(prev) = &prev_logits {
                // Logits must change when the model sees a new token in context.
                let same = prev
                    .iter()
                    .zip(logits.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-6);
                assert!(
                    !same,
                    "Logits at step {pos} are identical to step {}: \
                     KV cache is not accumulating correctly",
                    pos - 1
                );
            }
            prev_logits = Some(logits);
        }
    }
}
