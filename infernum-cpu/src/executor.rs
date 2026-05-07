//! Graph executor for the CPU backend.
//!
//! Walks an [`ExecutionPlan`] in topological order and dispatches each
//! operation to the appropriate CPU SIMD kernel. Intermediate tensors are
//! read from / written to a shared [`Arena`].
//!
//! ## Dispatch strategy
//!
//! Almost all ops are dispatched through the `_ =>` fallback arm, which
//! constructs an [`ExecuteContext`] and calls `node.op.execute(ctx)`. The
//! op's own `execute()` implementation in `builtin_ops.rs` handles the logic.
//!
//! A small set of ops remain as named match arms because they require
//! resources unavailable through the generic `OpNode::execute` interface:
//!
//! - `"input"` — KV cache routing (reads from persistent [`KvCacheStore`])
//! - `"rope"` — look-ahead Q+K fusion (fuses consecutive rope ops into one
//!   SIMD dispatch)
//! - `"silu"` / `"logit_softcap"` — require `CpuTensor::as_f32_slice`,
//!   which is not on the generic `Tensor` trait
//! - `"moe_dispatch_softmax"` / `"moe_dispatch_sigmoid"` — closure-based
//!   expert dispatch
//! - `"concat_seq"` — KV cache sequence concatenation

use std::collections::HashMap;

use infernum::backend::{ArithOps, MatmulOps, RopeOps};
use infernum::graph::builtin_ops::{
    LogitSoftcapOp, MoeDispatchSigmoidOp, MoeDispatchSoftmaxOp, RopeOp,
};
use infernum::graph::execute_context::KvCacheAccess;
use infernum::graph::{Arena, GraphNode, OutputRef, WeightStore};
use infernum::tensor::Tensor;
use infernum::{DType, ExecutionPlan, NodeId, Result};

use crate::simd;
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

/// Write a tensor into the arena at the plan's buffer slot for a node output.
fn write_tensor(
    arena: &mut Arena,
    plan: &ExecutionPlan,
    node_id: NodeId,
    output_idx: u32,
    tensor: &CpuTensor,
) {
    if let Some(slot) = plan.slot(node_id, output_idx) {
        if tensor.dtype() == DType::U32 {
            let src = tensor.as_u32_slice();
            let dst = arena.u32_slice_mut(slot.offset, src.len());
            dst.copy_from_slice(src);
        } else {
            let src = tensor.as_f32_slice();
            let dst = arena.f32_slice_mut(slot.offset, src.len());
            dst.copy_from_slice(src);
        }
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

    let mut sched_idx = 0;
    while sched_idx < plan.schedule.len() {
        let node_id = plan.schedule[sched_idx];
        sched_idx += 1;
        let node = &nodes[node_id.index() as usize];
        let op_name = node.op.name();

        match op_name {
            // --- Input ---
            "input" => {
                if kv_cache
                    .as_ref()
                    .is_some_and(|kv| kv.is_cache_input(node_id))
                {
                    // KV cache lives in the persistent store; skip arena copy.
                    continue;
                }
                let tensor = &inputs[input_idx];
                input_idx += 1;
                write_tensor(arena, plan, node_id, 0, tensor);
            }

            // --- Activations ---
            "silu" => {
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                let data = input.as_f32_slice();
                let mut out = vec![0.0f32; data.len()];
                for (o, &x) in out.iter_mut().zip(data.iter()) {
                    *o = x / (1.0 + (-x).exp());
                }
                let result = CpuTensor::from_f32(&node.output_shapes[0], &out);
                write_tensor(arena, plan, node_id, 0, &result);
            }

            // --- RoPE (zero-copy, with pair fusion) ---
            "rope" => {
                let op = node.op.as_any().downcast_ref::<RopeOp>().unwrap();
                let cos_ref = node.inputs[1];
                let sin_ref = node.inputs[2];
                let cos_node = &nodes[cos_ref.0.index() as usize];
                let sin_node = &nodes[sin_ref.0.index() as usize];

                // Check if cos/sin are f32 in the arena.
                if cos_node.output_dtypes[cos_ref.1 as usize] != DType::F32
                    || sin_node.output_dtypes[sin_ref.1 as usize] != DType::F32
                {
                    // Fallback: cos/sin need dtype conversion.
                    let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                    let cos_cache = read_tensor(arena, plan, nodes, cos_ref);
                    let sin_cache = read_tensor(arena, plan, nodes, sin_ref);
                    let result = <CpuBackend as RopeOps>::apply_rope(
                        &input, &cos_cache, &sin_cache, op.offset,
                    )?;
                    write_tensor(arena, plan, node_id, 0, &result);
                } else {
                    // Check if the next scheduled op is also Rope with same
                    // cos/sin inputs — if so, fuse both into one dispatch.
                    let next_rope = if sched_idx < plan.schedule.len() {
                        let next_id = plan.schedule[sched_idx];
                        let next_node = &nodes[next_id.index() as usize];
                        if next_node.op.name() == "rope" {
                            let next_op = next_node.op.as_any().downcast_ref::<RopeOp>().unwrap();
                            let n2_cos_ref = next_node.inputs[1];
                            let n2_sin_ref = next_node.inputs[2];
                            let n2_cos_node = &nodes[n2_cos_ref.0.index() as usize];
                            let n2_sin_node = &nodes[n2_sin_ref.0.index() as usize];
                            if n2_cos_ref == cos_ref
                                && n2_sin_ref == sin_ref
                                && n2_cos_node.output_dtypes[n2_cos_ref.1 as usize] == DType::F32
                                && n2_sin_node.output_dtypes[n2_sin_ref.1 as usize] == DType::F32
                            {
                                Some((next_id, next_node, next_op.offset))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    let cos_slot = plan
                        .slot(cos_ref.0, cos_ref.1)
                        .expect("rope cos has no slot");
                    let sin_slot = plan
                        .slot(sin_ref.0, sin_ref.1)
                        .expect("rope sin has no slot");
                    let cos_shape = &cos_node.output_shapes[cos_ref.1 as usize];
                    let sin_shape = &sin_node.output_shapes[sin_ref.1 as usize];
                    let cos_elements: usize = cos_shape.iter().product();
                    let sin_elements: usize = sin_shape.iter().product();

                    let base = arena.as_mut_ptr();
                    let shape = &node.output_shapes[0];

                    if let Some((next_id, next_node, offset_b)) = next_rope {
                        // Fused pair: execute both ropes in a single dispatch.
                        sched_idx += 1; // skip the second Rope

                        let (inp_a_nid, inp_a_oidx) = node.inputs[0];
                        let inp_a = plan.slot(inp_a_nid, inp_a_oidx).expect("rope A input");
                        let out_a = plan.slot(node_id, 0).expect("rope A output");
                        let (next_inp_nid, next_inp_oidx) = next_node.inputs[0];
                        let inp_b = plan
                            .slot(next_inp_nid, next_inp_oidx)
                            .expect("rope B input");
                        let out_b = plan.slot(next_id, 0).expect("rope B output");

                        let next_shape = &next_node.output_shapes[0];
                        unsafe {
                            let cos_data = std::slice::from_raw_parts(
                                base.add(cos_slot.offset).cast::<f32>(),
                                cos_elements,
                            );
                            let sin_data = std::slice::from_raw_parts(
                                base.add(sin_slot.offset).cast::<f32>(),
                                sin_elements,
                            );
                            let op_a = crate::ops::rope::RopeOperand {
                                inp_ptr: base.add(inp_a.offset) as usize,
                                out_ptr: base.add(out_a.offset) as usize,
                                seq_len: shape[0],
                                num_heads: shape[1],
                                head_dim: shape[2],
                                offset: op.offset,
                            };
                            let op_b = crate::ops::rope::RopeOperand {
                                inp_ptr: base.add(inp_b.offset) as usize,
                                out_ptr: base.add(out_b.offset) as usize,
                                seq_len: next_shape[0],
                                num_heads: next_shape[1],
                                head_dim: next_shape[2],
                                offset: offset_b,
                            };
                            crate::ops::rope::apply_rope_pair_slices(
                                &op_a, &op_b, cos_data, sin_data,
                            );
                        }
                    } else {
                        // Single rope — no fusion possible.
                        let (inp_nid, inp_oidx) = node.inputs[0];
                        let inp_slot = plan.slot(inp_nid, inp_oidx).expect("rope input");
                        let out_slot = plan.slot(node_id, 0).expect("rope output");
                        let num_elements: usize = shape.iter().product();
                        unsafe {
                            let input = std::slice::from_raw_parts(
                                base.add(inp_slot.offset).cast::<f32>(),
                                num_elements,
                            );
                            let cos_data = std::slice::from_raw_parts(
                                base.add(cos_slot.offset).cast::<f32>(),
                                cos_elements,
                            );
                            let sin_data = std::slice::from_raw_parts(
                                base.add(sin_slot.offset).cast::<f32>(),
                                sin_elements,
                            );
                            let output = std::slice::from_raw_parts_mut(
                                base.add(out_slot.offset).cast::<f32>(),
                                num_elements,
                            );
                            crate::ops::rope::apply_rope_slices(
                                input, cos_data, sin_data, output, shape[0], shape[1], shape[2],
                                op.offset,
                            );
                        }
                    }
                }
            }

            // --- Shape / Data movement ---
            "concat_seq" => {
                let concat_info = kv_cache
                    .as_ref()
                    .and_then(|kv| kv.cache_concat_info(node_id));

                if let Some((layer, is_key)) = concat_info {
                    // KV cache path: append new row to persistent buffer.
                    // Store the result in `overrides` so downstream attention
                    // reads directly — no arena copy of the full KV cache.
                    let new_row = read_tensor(arena, plan, nodes, node.inputs[1]);
                    let kv = kv_cache.as_mut().unwrap();
                    kv.append(layer, is_key, new_row.as_f32_slice());
                    // Compute actual length from vec size to avoid stale `len` field.
                    let row_elems = kv.num_kv_heads * kv.head_dim;
                    let new_len = if is_key {
                        kv.k_caches[layer].len() / row_elems
                    } else {
                        kv.v_caches[layer].len() / row_elems
                    };
                    let full_cache = kv.get_cache(layer, is_key, new_len);
                    overrides.insert(node_id, full_cache);
                } else {
                    let a = read_tensor(arena, plan, nodes, node.inputs[0]);
                    let b = read_tensor(arena, plan, nodes, node.inputs[1]);
                    let a_data = a.as_f32_slice();
                    let b_data = b.as_f32_slice();
                    let mut data = Vec::with_capacity(a_data.len() + b_data.len());
                    data.extend_from_slice(a_data);
                    data.extend_from_slice(b_data);
                    let result = CpuTensor::from_f32_vec(&node.output_shapes[0], data);
                    write_tensor(arena, plan, node_id, 0, &result);
                }
            }

            // --- Logit soft-cap: tanh(x / cap) * cap (Gemma 2 final logit) ---
            "logit_softcap" => {
                let op = node.op.as_any().downcast_ref::<LogitSoftcapOp>().unwrap();
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                let cap = op.cap;
                let data = input.as_f32_slice();
                let out: Vec<f32> = data.iter().map(|&x| (x / cap).tanh() * cap).collect();
                let result = CpuTensor::from_f32_vec(&node.output_shapes[0], out);
                write_tensor(arena, plan, node_id, 0, &result);
            }

            // --- MoE softmax dispatch (Mixtral, Qwen3-MoE) ---
            "moe_dispatch_softmax" => {
                use infernum::backend::MoeOps as _;
                let op = node
                    .op
                    .as_any()
                    .downcast_ref::<MoeDispatchSoftmaxOp>()
                    .unwrap();
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                let gate_t = weights.tensor_weight(op.gate);
                let num_experts = op.experts.len();
                let num_experts_per_tok = op.num_experts_per_tok;
                let norm_topk = op.norm_topk;
                // Snapshot expert weight IDs so we can pass them to the closure.
                let expert_ids = op.experts.clone();
                let result = crate::CpuBackend::moe_forward_softmax(
                    &input,
                    gate_t,
                    num_experts,
                    num_experts_per_tok,
                    norm_topk,
                    |expert_idx, expert_input| {
                        let eids = &expert_ids[expert_idx];
                        let gate_w = weights.linear_weight(eids.gate_proj);
                        let up_w = weights.linear_weight(eids.up_proj);
                        let down_w = weights.linear_weight(eids.down_proj);
                        let gate_out = crate::CpuBackend::linear(expert_input, gate_w)?;
                        let up_out = crate::CpuBackend::linear(expert_input, up_w)?;
                        // Fused SiLU-Mul: gate_act[i] = silu(gate[i]) * up[i]
                        let gate_data = gate_out.as_f32_slice();
                        let up_data = up_out.as_f32_slice();
                        let mut fused = vec![0.0f32; gate_data.len()];
                        simd::vec_silu_mul(gate_data, up_data, &mut fused);
                        let shape = gate_out.shape().to_vec();
                        let activated = CpuTensor::from_f32_vec(&shape, fused);
                        crate::CpuBackend::linear(&activated, down_w)
                    },
                )?;
                write_tensor(arena, plan, node_id, 0, &result);
            }

            // --- MoE sigmoid dispatch with bias correction (DeepSeek) ---
            "moe_dispatch_sigmoid" => {
                use infernum::backend::MoeSigmoidOps as _;
                let op = node
                    .op
                    .as_any()
                    .downcast_ref::<MoeDispatchSigmoidOp>()
                    .unwrap();
                let input = read_tensor(arena, plan, nodes, node.inputs[0]);
                let gate_t = weights.tensor_weight(op.gate);
                let bias_data: Vec<f32> = if let Some(bias_id) = op.bias {
                    weights.tensor_weight(bias_id).as_f32_slice().to_vec()
                } else {
                    vec![0.0f32; op.experts.len()]
                };
                let num_experts = op.experts.len();
                let num_experts_per_tok = op.num_experts_per_tok;
                let n_group = op.n_group;
                let topk_group = op.topk_group;
                let routed_scaling_factor = op.routed_scaling_factor;
                let expert_ids = op.experts.clone();
                let shared_ids = op.shared_expert.clone();
                let mut result = crate::CpuBackend::moe_forward_sigmoid(
                    &input,
                    gate_t,
                    &bias_data,
                    num_experts,
                    num_experts_per_tok,
                    n_group,
                    topk_group,
                    false, // norm_topk_prob: DeepSeek normalises inside the kernel
                    routed_scaling_factor,
                    |expert_idx, expert_input| {
                        let eids = &expert_ids[expert_idx];
                        let gate_w = weights.linear_weight(eids.gate_proj);
                        let up_w = weights.linear_weight(eids.up_proj);
                        let down_w = weights.linear_weight(eids.down_proj);
                        let gate_out = crate::CpuBackend::linear(expert_input, gate_w)?;
                        let up_out = crate::CpuBackend::linear(expert_input, up_w)?;
                        let gate_data = gate_out.as_f32_slice();
                        let up_data = up_out.as_f32_slice();
                        let mut fused = vec![0.0f32; gate_data.len()];
                        simd::vec_silu_mul(gate_data, up_data, &mut fused);
                        let shape = gate_out.shape().to_vec();
                        let activated = CpuTensor::from_f32_vec(&shape, fused);
                        crate::CpuBackend::linear(&activated, down_w)
                    },
                )?;
                // Add shared expert output if present.
                if let Some(sids) = shared_ids {
                    let sg = weights.linear_weight(sids.gate_proj);
                    let su = weights.linear_weight(sids.up_proj);
                    let sd = weights.linear_weight(sids.down_proj);
                    let sgate = crate::CpuBackend::linear(&input, sg)?;
                    let sup_out = crate::CpuBackend::linear(&input, su)?;
                    let shared_gate_data = sgate.as_f32_slice();
                    let shared_up_data = sup_out.as_f32_slice();
                    let mut sfused = vec![0.0f32; shared_gate_data.len()];
                    simd::vec_silu_mul(shared_gate_data, shared_up_data, &mut sfused);
                    let sshape = sgate.shape().to_vec();
                    let sact = CpuTensor::from_f32_vec(&sshape, sfused);
                    let shared_out = crate::CpuBackend::linear(&sact, sd)?;
                    crate::CpuBackend::add_inplace(&mut result, &shared_out)?;
                }
                write_tensor(arena, plan, node_id, 0, &result);
            }

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
