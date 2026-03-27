//! Core node types for the computation graph.

use smallvec::SmallVec;

use crate::dtype::DType;

use super::ops::Op;

/// Index into `Graph::nodes`. Lightweight handle returned by `add_*` methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) u32);

impl NodeId {
    /// Create a `NodeId` from a raw index.
    #[must_use]
    pub fn from_index(index: u32) -> Self {
        Self(index)
    }

    /// Return the raw index.
    #[must_use]
    pub fn index(self) -> u32 {
        self.0
    }
}

/// Reference to a weight in the `WeightStore`. Can be either a plain tensor
/// (embedding table, layernorm weight, `RoPE` cache) or a linear weight
/// (dense, quantized, etc.).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightRef {
    /// A plain tensor weight (embedding table, layernorm, `RoPE` cache, bias).
    Tensor(WeightId),
    /// A linear weight (dense or quantized matmul weight).
    Linear(WeightId),
}

/// Index into `WeightStore`'s tensor or linear weight vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightId(pub(crate) u32);

/// Metadata for a registered weight (stored in the graph for shape inference).
#[derive(Clone, Debug)]
pub struct WeightMeta {
    /// The name of the weight (e.g., `"model.layers.0.self_attn.q_proj"`).
    pub name: String,
    /// Shape of the weight tensor.
    pub shape: Vec<usize>,
    /// Data type of the weight tensor.
    pub dtype: DType,
}

/// A node in the computation graph.
#[derive(Clone, Debug)]
pub struct GraphNode {
    /// The operation this node performs.
    pub op: Op,
    /// Input node IDs (edges in the graph).
    pub inputs: SmallVec<[NodeId; 4]>,
    /// Shape of the primary output.
    pub shape: Vec<usize>,
    /// Data type of the primary output.
    pub dtype: DType,
}
