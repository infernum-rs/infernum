//! The `Graph` struct and its core construction methods.

use std::marker::PhantomData;

use smallvec::SmallVec;

use crate::backend::Backend;
use crate::dtype::DType;

use super::node::{GraphNode, NodeId, WeightId, WeightMeta};
use super::ops::Op;

/// A computation graph parameterised by a backend.
///
/// Nodes represent operations; edges (stored as `NodeId` lists in each
/// `GraphNode`) represent data dependencies. Weights are registered
/// separately and referenced by `WeightId`.
pub struct Graph<B: Backend> {
    /// All nodes in topological (insertion) order.
    pub(crate) nodes: Vec<GraphNode>,
    /// Nodes marked as graph outputs.
    pub(crate) outputs: Vec<NodeId>,
    /// Registered tensor weights (layernorm, embedding, `RoPE`, bias, etc.).
    pub(crate) tensor_weights: Vec<WeightMeta>,
    /// Registered linear weights (dense or quantized).
    pub(crate) linear_weights: Vec<WeightMeta>,
    _backend: PhantomData<B>,
}

impl<B: Backend> Graph<B> {
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

    /// Add an input node (e.g., token IDs).
    pub fn add_input(&mut self, shape: &[usize], dtype: DType) -> NodeId {
        self.push_node(Op::Input, &[], shape.to_vec(), dtype)
    }

    /// Mark a node as a graph output.
    pub fn set_output(&mut self, node: NodeId) {
        self.outputs.push(node);
    }

    /// Push a single-output node and return its `NodeId`.
    #[allow(clippy::cast_possible_truncation)] // graph will never have 2^32 nodes
    pub(crate) fn push_node(
        &mut self,
        op: Op,
        inputs: &[NodeId],
        shape: Vec<usize>,
        dtype: DType,
    ) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(GraphNode {
            op,
            inputs: SmallVec::from_slice(inputs),
            shape,
            dtype,
        });
        id
    }

    /// Push a multi-output op (two outputs). Returns `(primary, secondary)`.
    ///
    /// The primary node carries the actual `Op`. The secondary node is a
    /// `SecondOutput` that references the primary.
    #[allow(dead_code)] // used by builder traits in later steps
    pub(crate) fn push_node_pair(
        &mut self,
        op: Op,
        inputs: &[NodeId],
        shape1: Vec<usize>,
        dtype1: DType,
        shape2: Vec<usize>,
        dtype2: DType,
    ) -> (NodeId, NodeId) {
        let primary = self.push_node(op, inputs, shape1, dtype1);
        let secondary = self.push_node(Op::SecondOutput { source: primary }, &[], shape2, dtype2);
        (primary, secondary)
    }

    /// Get the shape of a node's output.
    #[must_use]
    pub fn node_shape(&self, id: NodeId) -> &[usize] {
        &self.nodes[id.0 as usize].shape
    }

    /// Get the dtype of a node's output.
    #[must_use]
    pub fn node_dtype(&self, id: NodeId) -> DType {
        self.nodes[id.0 as usize].dtype
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
}

impl<B: Backend> Default for Graph<B> {
    fn default() -> Self {
        Self::new()
    }
}
