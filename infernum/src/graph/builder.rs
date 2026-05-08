//! The `Graph` struct and its core construction methods.

use std::marker::PhantomData;

use smallvec::SmallVec;

use crate::backend::{Backend, ContextBackend, MatmulOps};
use crate::dtype::DType;

use super::builtin_ops::InputOp;
use super::node::{GraphNode, NodeId, WeightId, WeightMeta};
use super::op_node::{OpNode, OutputRef};

/// A computation graph parameterised by a backend.
///
/// Nodes represent operations; edges (stored as `OutputRef` lists in each
/// `GraphNode`) represent data dependencies. Weights are registered
/// separately and referenced by `WeightId`.\
pub struct Graph<B: Backend + MatmulOps + ContextBackend> {
    /// All nodes in topological (insertion) order.
    pub(crate) nodes: Vec<GraphNode<B>>,
    /// Nodes marked as graph outputs.
    pub(crate) outputs: Vec<NodeId>,
    /// Registered tensor weights (layernorm, embedding, `RoPE`, bias, etc.).
    pub(crate) tensor_weights: Vec<WeightMeta>,
    /// Registered linear weights (dense or quantized).
    pub(crate) linear_weights: Vec<WeightMeta>,
    _backend: PhantomData<B>,
}

impl<B: Backend + MatmulOps + ContextBackend> Graph<B> {
    /// Create an empty computation graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            outputs: Vec::new(),
            tensor_weights: Vec::new(),
            linear_weights: Vec::new(),
            _backend: PhantomData,
        }
    }

    /// Register a tensor weight (layernorm, embedding table, `RoPE` cache,
    /// bias, etc.). Returns a `WeightId` for use in ops.
    #[allow(clippy::cast_possible_truncation)] // graph will never have 2^32 weights
    pub fn register_tensor_weight(
        &mut self,
        name: impl Into<String>,
        shape: &[usize],
        dtype: DType,
    ) -> WeightId {
        let id = WeightId(self.tensor_weights.len() as u32);
        self.tensor_weights.push(WeightMeta {
            name: name.into(),
            shape: shape.to_vec(),
            dtype,
        });
        id
    }

    /// Register a linear weight (dense or quantized).
    /// Returns a `WeightId` for use in `Linear` ops.
    #[allow(clippy::cast_possible_truncation)] // graph will never have 2^32 weights
    pub fn register_linear_weight(
        &mut self,
        name: impl Into<String>,
        shape: &[usize],
        dtype: DType,
    ) -> WeightId {
        let id = WeightId(self.linear_weights.len() as u32);
        self.linear_weights.push(WeightMeta {
            name: name.into(),
            shape: shape.to_vec(),
            dtype,
        });
        id
    }

    /// Add an input node (e.g., token IDs). Returns an `OutputRef`.
    pub fn add_input(&mut self, shape: &[usize], dtype: DType) -> OutputRef {
        let node_id = self.add_node(
            Box::new(InputOp {
                shape: shape.to_vec(),
                dtype,
            }),
            &[],
        );
        (node_id, 0)
    }

    /// Mark a node as a graph output.
    pub fn set_output(&mut self, node: NodeId) {
        self.outputs.push(node);
    }

    /// Add a node to the graph. Performs shape/dtype inference via the op's
    /// `output_shapes` and `output_dtypes` methods.
    ///
    /// Returns the `NodeId` of the newly created node.
    ///
    /// # Panics
    ///
    /// Panics if the number of provided `inputs` does not match the op's
    /// expected input count (`op.num_inputs()`).
    #[allow(clippy::cast_possible_truncation)] // graph will never have 2^32 nodes
    pub fn add_node(&mut self, op: Box<dyn OpNode<B>>, inputs: &[OutputRef]) -> NodeId {
        let input_shapes: Vec<&[usize]> = inputs
            .iter()
            .map(|&(nid, oidx)| self.output_shape(nid, oidx))
            .collect();
        let input_dtypes: Vec<DType> = inputs
            .iter()
            .map(|&(nid, oidx)| self.output_dtype(nid, oidx))
            .collect();

        assert_eq!(
            inputs.len(),
            op.num_inputs(),
            "Op '{}' expects {} inputs, got {}",
            op.name(),
            op.num_inputs(),
            inputs.len()
        );

        let output_shapes = op.output_shapes(&input_shapes);
        let output_dtypes = op.output_dtypes(&input_dtypes);

        let node_id = NodeId(self.nodes.len() as u32);
        self.nodes.push(GraphNode {
            op,
            inputs: SmallVec::from_slice(inputs),
            output_shapes,
            output_dtypes,
        });
        node_id
    }

    /// Get the shape of a specific output of a node.
    #[must_use]
    pub fn output_shape(&self, id: NodeId, output_idx: u32) -> &[usize] {
        &self.nodes[id.0 as usize].output_shapes[output_idx as usize]
    }

    /// Get the dtype of a specific output of a node.
    #[must_use]
    pub fn output_dtype(&self, id: NodeId, output_idx: u32) -> DType {
        self.nodes[id.0 as usize].output_dtypes[output_idx as usize]
    }

    /// Get the shape of the output referenced by an `OutputRef`.
    ///
    /// Convenience method equivalent to `output_shape(ref.0, ref.1)`.
    #[must_use]
    pub fn node_shape(&self, output_ref: OutputRef) -> &[usize] {
        self.output_shape(output_ref.0, output_ref.1)
    }

    /// Get the dtype of the output referenced by an `OutputRef`.
    ///
    /// Convenience method equivalent to `output_dtype(ref.0, ref.1)`.
    #[must_use]
    pub fn node_dtype(&self, output_ref: OutputRef) -> DType {
        self.output_dtype(output_ref.0, output_ref.1)
    }

    /// All nodes in the graph (indexed by `NodeId`).
    #[must_use]
    pub fn nodes(&self) -> &[GraphNode<B>] {
        &self.nodes
    }

    /// Nodes marked as graph outputs.
    #[must_use]
    pub fn output_ids(&self) -> &[NodeId] {
        &self.outputs
    }

    /// Total number of nodes in the graph.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get the registered tensor weight metadata by `WeightId`.
    #[must_use]
    pub fn tensor_weight_meta(&self, id: WeightId) -> &WeightMeta {
        &self.tensor_weights[id.0 as usize]
    }

    /// Get the registered linear weight metadata by `WeightId`.
    #[must_use]
    pub fn linear_weight_meta(&self, id: WeightId) -> &WeightMeta {
        &self.linear_weights[id.0 as usize]
    }

    /// Number of registered tensor weights.
    #[must_use]
    pub fn tensor_weight_count(&self) -> usize {
        self.tensor_weights.len()
    }

    /// Number of registered linear weights.
    #[must_use]
    pub fn linear_weight_count(&self) -> usize {
        self.linear_weights.len()
    }

    /// Returns `true` if the graph contains any operation that is incompatible
    /// with CUDA stream capture.
    ///
    /// Operations are capture-unsafe when they perform synchronous host↔device
    /// copies (`cuStreamSynchronize`, `dtoh_sync_copy`, `htod_sync_copy`) or
    /// allocate device memory (`cuMemAlloc`) during the forward pass:
    ///
    /// - `moe_dispatch_softmax` / `moe_dispatch_sigmoid` — `MoE` routing calls
    ///   `logits_to_f32_host` (a D→H sync copy) for host-side top-K selection.
    /// - `logit_softcap` — Gemma 2 final logit soft-capping implements via a
    ///   full CPU round-trip (`to_vec` + `from_slice`), both sync.
    ///
    /// When this returns `true`, callers should skip CUDA graph capture and
    /// always run the executor eagerly.
    #[must_use]
    pub fn has_capture_unsafe_ops(&self) -> bool {
        self.nodes.iter().any(|n| {
            matches!(
                n.op.name(),
                "moe_dispatch_softmax" | "moe_dispatch_sigmoid" | "logit_softcap"
            )
        })
    }
}

impl<B: Backend + MatmulOps + ContextBackend> Default for Graph<B> {
    fn default() -> Self {
        Self::new()
    }
}
