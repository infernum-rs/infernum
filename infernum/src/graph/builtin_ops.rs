//! Built-in [`OpNode`] implementations for the computation graph.
//!
//! Each struct wraps a single graph operation (matmul, `RmsNorm`, `RoPE`, etc.)
//! with its parameters, shape/dtype inference, and backend execution logic.
//! Ops that require resources unavailable through the [`OpNode::execute`]
//! interface (KV cache, communicator, closure-based dispatch) panic at
//! runtime and must be handled specially by the executor.

use std::any::Any;

use crate::backend::{
    ArithOps, AttentionOps, Backend, CastOps, EmbedOps, GegluOps, MatmulExtOps, MatmulOps, NormOps,
    RopeInterleavedOps, RopeOps, SwigluOps, TensorOps,
};
use crate::dtype::DType;
use crate::tensor::Tensor as TensorTrait;
use crate::Result;

use super::node::WeightId;
use super::op_node::OpNode;
use super::weight_store::WeightStore;

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Weight IDs for a single `MoE` expert's MLP.
#[derive(Clone, Debug)]
pub struct MoeExpertIds {
    /// Gate projection weight ID.
    pub gate_proj: WeightId,
    /// Up projection weight ID.
    pub up_proj: WeightId,
    /// Down projection weight ID.
    pub down_proj: WeightId,
}

// ---------------------------------------------------------------------------
// InputOp
// ---------------------------------------------------------------------------

/// Graph input placeholder — not a real computation.
///
/// The executor injects external input tensors at these nodes;
/// calling [`OpNode::execute`] on an `InputOp` is a programmer error.
#[derive(Debug)]
pub struct InputOp {
    /// Declared shape of the input tensor.
    pub shape: Vec<usize>,
    /// Declared data type of the input tensor.
    pub dtype: DType,
}

impl<B: Backend + MatmulOps> OpNode<B> for InputOp {
    fn name(&self) -> &'static str {
        "input"
    }
    fn num_inputs(&self) -> usize {
        0
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, _input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![self.shape.clone()]
    }
    fn output_dtypes(&self, _input_dtypes: &[DType]) -> Vec<DType> {
        vec![self.dtype]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!("InputOp is handled specially by the executor")
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// EmbeddingGatherOp
// ---------------------------------------------------------------------------

/// Embedding table lookup by token IDs.
///
/// Gathers rows from the embedding table to produce hidden states
/// of shape `(seq_len, embed_dim)`.
#[derive(Debug)]
pub struct EmbeddingGatherOp {
    /// Weight ID of the embedding table.
    pub table: WeightId,
    /// Embedding dimension (hidden size).
    pub embed_dim: usize,
    /// Output data type.
    pub dtype: DType,
}

impl<B: Backend + MatmulOps + EmbedOps> OpNode<B> for EmbeddingGatherOp {
    fn name(&self) -> &'static str {
        "embedding_gather"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![vec![input_shapes[0][0], self.embed_dim]]
    }
    fn output_dtypes(&self, _input_dtypes: &[DType]) -> Vec<DType> {
        vec![self.dtype]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let table = weights.tensor_weight(self.table);
        let seq_len = inputs[0].shape()[0];
        let result = B::embedding_gather_tensor(table, inputs[0], seq_len)?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// RmsNormOp
// ---------------------------------------------------------------------------

/// RMS normalization.
///
/// Normalizes the input tensor using root-mean-square statistics
/// and a learned weight vector.
#[derive(Debug)]
pub struct RmsNormOp {
    /// Weight ID of the normalization weight vector.
    pub weight: WeightId,
    /// Epsilon for numerical stability.
    pub eps: f32,
}

impl<B: Backend + MatmulOps + NormOps> OpNode<B> for RmsNormOp {
    fn name(&self) -> &'static str {
        "rms_norm"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let w = weights.tensor_weight(self.weight);
        let result = B::rms_norm(inputs[0], w, self.eps)?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// AddRmsNormOp
// ---------------------------------------------------------------------------

/// Fused residual add + RMS normalization.
///
/// Computes `updated = residual + delta` and `normed = rms_norm(updated)`,
/// returning both tensors.
#[derive(Debug)]
pub struct AddRmsNormOp {
    /// Weight ID of the normalization weight vector.
    pub weight: WeightId,
    /// Epsilon for numerical stability.
    pub eps: f32,
}

impl<B: Backend + MatmulOps + NormOps> OpNode<B> for AddRmsNormOp {
    fn name(&self) -> &'static str {
        "add_rms_norm"
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        2
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec(), input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0], input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let w = weights.tensor_weight(self.weight);
        let (updated, normed) = B::add_rmsnorm(inputs[0], inputs[1], w, self.eps)?;
        Ok(vec![updated, normed])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// LinearOp
// ---------------------------------------------------------------------------

/// Dense or quantized linear projection.
///
/// Computes `output = input × weight` where the weight may be dense or
/// quantized depending on the backend's `LinearWeight` type.
#[derive(Debug)]
pub struct LinearOp {
    /// Weight ID of the linear weight.
    pub weight: WeightId,
    /// Output features (last dimension of the result).
    pub out_features: usize,
}

impl<B: Backend + MatmulOps> OpNode<B> for LinearOp {
    fn name(&self) -> &'static str {
        "linear"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        let mut s = input_shapes[0].to_vec();
        *s.last_mut().unwrap() = self.out_features;
        vec![s]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let w = weights.linear_weight(self.weight);
        let result = B::linear(inputs[0], w)?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// LinearPairOp
// ---------------------------------------------------------------------------

/// Paired linear projection dispatched in a single parallel call.
///
/// Computes two linear projections from the same input, enabling
/// backends to fuse the dispatches for better utilisation.
#[derive(Debug)]
pub struct LinearPairOp {
    /// Weight ID of the first linear weight.
    pub w1: WeightId,
    /// Weight ID of the second linear weight.
    pub w2: WeightId,
    /// Output features for the first projection.
    pub out1: usize,
    /// Output features for the second projection.
    pub out2: usize,
}

impl<B: Backend + MatmulOps> OpNode<B> for LinearPairOp {
    fn name(&self) -> &'static str {
        "linear_pair"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        2
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        let mut s1 = input_shapes[0].to_vec();
        *s1.last_mut().unwrap() = self.out1;
        let mut s2 = input_shapes[0].to_vec();
        *s2.last_mut().unwrap() = self.out2;
        vec![s1, s2]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0], input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let w1 = weights.linear_weight(self.w1);
        let w2 = weights.linear_weight(self.w2);
        let (r1, r2) = B::linear_pair(inputs[0], w1, w2)?;
        Ok(vec![r1, r2])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// LinearTripleOp
// ---------------------------------------------------------------------------

/// Triple linear projection dispatched in a single parallel call.
///
/// Used for Q+K+V attention projections. Backends may fuse input
/// quantisation across all three weight matrices.
#[derive(Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct LinearTripleOp {
    /// Weight ID of the first linear weight (e.g., Q).
    pub w1: WeightId,
    /// Weight ID of the second linear weight (e.g., K).
    pub w2: WeightId,
    /// Weight ID of the third linear weight (e.g., V).
    pub w3: WeightId,
    /// Output features for the first projection.
    pub out1: usize,
    /// Output features for the second projection.
    pub out2: usize,
    /// Output features for the third projection.
    pub out3: usize,
}

impl<B: Backend + MatmulOps> OpNode<B> for LinearTripleOp {
    fn name(&self) -> &'static str {
        "linear_triple"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        3
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        let mut s1 = input_shapes[0].to_vec();
        *s1.last_mut().unwrap() = self.out1;
        let mut s2 = input_shapes[0].to_vec();
        *s2.last_mut().unwrap() = self.out2;
        let mut s3 = input_shapes[0].to_vec();
        *s3.last_mut().unwrap() = self.out3;
        vec![s1, s2, s3]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0], input_dtypes[0], input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let w1 = weights.linear_weight(self.w1);
        let w2 = weights.linear_weight(self.w2);
        let w3 = weights.linear_weight(self.w3);
        let (r1, r2, r3) = B::linear_triple(inputs[0], w1, w2, w3)?;
        Ok(vec![r1, r2, r3])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// MatmulOp
// ---------------------------------------------------------------------------

/// General matrix multiplication.
///
/// Computes `C = A × B` for 2-D tensors.
#[derive(Debug)]
pub struct MatmulOp;

impl<B: Backend + MatmulOps> OpNode<B> for MatmulOp {
    fn name(&self) -> &'static str {
        "matmul"
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![vec![input_shapes[0][0], input_shapes[1][1]]]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let result = B::matmul(inputs[0], inputs[1])?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// MatmulBf16F32Op
// ---------------------------------------------------------------------------

/// Mixed-precision matrix multiplication: bf16 inputs → f32 output.
///
/// Used by `DeepSeek` MLA and similar architectures that need
/// higher-precision accumulation.
#[derive(Debug)]
pub struct MatmulBf16F32Op;

impl<B: Backend + MatmulOps + MatmulExtOps> OpNode<B> for MatmulBf16F32Op {
    fn name(&self) -> &'static str {
        "matmul_bf16_f32"
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![vec![input_shapes[0][0], input_shapes[1][1]]]
    }
    fn output_dtypes(&self, _input_dtypes: &[DType]) -> Vec<DType> {
        vec![DType::F32]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let result = B::matmul_bf16_f32(inputs[0], inputs[1])?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// SwigluOp
// ---------------------------------------------------------------------------

/// Fused `SwiGLU` activation: `silu(gate) * up`.
///
/// Standard FFN activation for Llama, Qwen, and similar architectures.
#[derive(Debug)]
pub struct SwigluOp;

impl<B: Backend + MatmulOps + SwigluOps> OpNode<B> for SwigluOp {
    fn name(&self) -> &'static str {
        "swiglu"
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let result = B::swiglu(inputs[0], inputs[1])?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// GegluOp
// ---------------------------------------------------------------------------

/// Fused `GeGLU` activation: `gelu(gate) * up`.
///
/// Used by Gemma and similar architectures.
#[derive(Debug)]
pub struct GegluOp;

impl<B: Backend + MatmulOps + GegluOps> OpNode<B> for GegluOp {
    fn name(&self) -> &'static str {
        "geglu"
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let result = B::geglu(inputs[0], inputs[1])?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// SiluOp
// ---------------------------------------------------------------------------

/// Standalone `SiLU` activation.
///
/// Should be fused into [`SwigluOp`] by the graph optimiser; direct
/// execution is not supported.
#[derive(Debug)]
pub struct SiluOp;

impl<B: Backend + MatmulOps> OpNode<B> for SiluOp {
    fn name(&self) -> &'static str {
        "silu"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!(
            "SiluOp should be fused into SwigluOp by the optimizer; 
             direct execution is not supported"
        )
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// AddOp
// ---------------------------------------------------------------------------

/// Element-wise addition of two tensors.
#[derive(Debug)]
pub struct AddOp;

impl<B: Backend + MatmulOps + ArithOps> OpNode<B> for AddOp {
    fn name(&self) -> &'static str {
        "add"
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let result = B::add(inputs[0], inputs[1])?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// AddInplaceOp
// ---------------------------------------------------------------------------

/// In-place element-wise addition (semantically `a += b`).
///
/// Since [`OpNode::execute`] receives immutable tensor refs, this falls
/// back to an allocating add. The executor may optimise this into a true
/// in-place operation when ownership is available.
#[derive(Debug)]
pub struct AddInplaceOp;

impl<B: Backend + MatmulOps + ArithOps> OpNode<B> for AddInplaceOp {
    fn name(&self) -> &'static str {
        "add_inplace"
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let result = B::add(inputs[0], inputs[1])?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// MulOp
// ---------------------------------------------------------------------------

/// Element-wise multiplication of two tensors.
#[derive(Debug)]
pub struct MulOp;

impl<B: Backend + MatmulOps + ArithOps> OpNode<B> for MulOp {
    fn name(&self) -> &'static str {
        "mul"
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let result = B::mul(inputs[0], inputs[1])?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// ScaleOp
// ---------------------------------------------------------------------------

/// Scalar multiplication: `tensor *= factor`.
///
/// The backend only provides `scale_inplace`, which requires `&mut`.
/// Since [`OpNode::execute`] receives immutable refs, this op must be
/// handled specially by the executor (clone + scale in-place).
#[derive(Debug)]
pub struct ScaleOp {
    /// Scalar multiplication factor.
    pub factor: f32,
}

impl<B: Backend + MatmulOps> OpNode<B> for ScaleOp {
    fn name(&self) -> &'static str {
        "scale"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!(
            "ScaleOp requires in-place mutation (scale_inplace); 
             handled specially by the executor"
        )
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// BiasAddOp
// ---------------------------------------------------------------------------

/// Bias addition: `input[row, col] += bias[col]`.
///
/// The backend only provides `bias_add_inplace`, which requires `&mut`.
/// Since [`OpNode::execute`] receives immutable refs, this op must be
/// handled specially by the executor.
#[derive(Debug)]
pub struct BiasAddOp {
    /// Weight ID of the bias vector.
    pub bias: WeightId,
}

impl<B: Backend + MatmulOps> OpNode<B> for BiasAddOp {
    fn name(&self) -> &'static str {
        "bias_add"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!(
            "BiasAddOp requires in-place mutation (bias_add_inplace); 
             handled specially by the executor"
        )
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// RopeOp
// ---------------------------------------------------------------------------

/// Rotary positional embedding (standard layout).
///
/// Applies `RoPE` to a 3-D tensor using precomputed cos/sin caches.
#[derive(Debug)]
pub struct RopeOp {
    /// Position offset for KV-cache continuation.
    pub offset: usize,
}

impl<B: Backend + MatmulOps + RopeOps> OpNode<B> for RopeOp {
    fn name(&self) -> &'static str {
        "rope"
    }
    fn num_inputs(&self) -> usize {
        3
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let result = B::apply_rope(inputs[0], inputs[1], inputs[2], self.offset)?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// RopeBatchedOp
// ---------------------------------------------------------------------------

/// Batched rotary positional embedding with per-token positions.
///
/// Used during batched decode where each sequence in the batch has
/// a different position offset.
#[derive(Debug)]
pub struct RopeBatchedOp {
    /// Number of sequences in the batch.
    pub batch_size: usize,
}

impl<B: Backend + MatmulOps + RopeOps> OpNode<B> for RopeBatchedOp {
    fn name(&self) -> &'static str {
        "rope_batched"
    }
    fn num_inputs(&self) -> usize {
        4
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let result =
            B::apply_rope_batched(inputs[0], inputs[1], inputs[2], inputs[3], self.batch_size)?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// RopeInterleavedOp
// ---------------------------------------------------------------------------

/// Rotary positional embedding (interleaved layout, used by `DeepSeek`).
///
/// Interleaved layout pairs `(x[0], x[1]), (x[2], x[3]), …` instead
/// of the standard split-half layout.
#[derive(Debug)]
pub struct RopeInterleavedOp {
    /// Position offset for KV-cache continuation.
    pub offset: usize,
}

impl<B: Backend + MatmulOps + RopeInterleavedOps> OpNode<B> for RopeInterleavedOp {
    fn name(&self) -> &'static str {
        "rope_interleaved"
    }
    fn num_inputs(&self) -> usize {
        3
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let result = B::apply_rope_interleaved(inputs[0], inputs[1], inputs[2], self.offset)?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// FusedAttentionPrefillOp
// ---------------------------------------------------------------------------

/// Fused attention for prefill (prompt processing).
///
/// Computes scaled dot-product attention over the full Q/K/V tensors
/// with optional soft-capping and sliding-window masking.
#[derive(Debug)]
pub struct FusedAttentionPrefillOp {
    /// Position offset for causal masking.
    pub offset: usize,
    /// Optional custom attention scale (default: `1/sqrt(head_dim)`).
    pub scale: Option<f32>,
    /// Optional logit soft-cap (Gemma-style).
    pub softcap: Option<f32>,
    /// Optional sliding window size for local attention.
    pub sliding_window: Option<usize>,
}

impl<B: Backend + MatmulOps + AttentionOps> OpNode<B> for FusedAttentionPrefillOp {
    fn name(&self) -> &'static str {
        "fused_attention_prefill"
    }
    fn num_inputs(&self) -> usize {
        3
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let result = B::fused_attention_prefill(
            inputs[0],
            inputs[1],
            inputs[2],
            self.offset,
            self.scale,
            self.softcap,
            self.sliding_window,
        )?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// FusedAttentionDecodeOp
// ---------------------------------------------------------------------------

/// Fused attention for single-token decode.
///
/// Computes attention for a single query position against the full
/// K/V context with optional soft-capping.
#[derive(Debug)]
pub struct FusedAttentionDecodeOp {
    /// Optional logit soft-cap (Gemma-style).
    pub softcap: Option<f32>,
}

impl<B: Backend + MatmulOps + AttentionOps> OpNode<B> for FusedAttentionDecodeOp {
    fn name(&self) -> &'static str {
        "fused_attention_decode"
    }
    fn num_inputs(&self) -> usize {
        3
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let result =
            B::fused_attention_decode(inputs[0], inputs[1], inputs[2], None, self.softcap, None)?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// PagedAttentionDecodeOp
// ---------------------------------------------------------------------------

/// Paged attention for batched decode.
///
/// Requires KV cache access which is not available through the
/// [`OpNode::execute`] interface; handled specially by the executor.
#[derive(Debug)]
#[allow(clippy::struct_field_names)]
pub struct PagedAttentionDecodeOp {
    /// Transformer layer index for KV cache lookup.
    pub layer_idx: usize,
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of key/value heads.
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Number of tokens per paged block.
    pub block_size: usize,
    /// Optional sliding window size.
    pub sliding_window: Option<usize>,
}

impl<B: Backend + MatmulOps> OpNode<B> for PagedAttentionDecodeOp {
    fn name(&self) -> &'static str {
        "paged_attention_decode"
    }
    fn num_inputs(&self) -> usize {
        4
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!("PagedAttentionDecodeOp requires KV cache access; handled specially by the executor")
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// AppendPagedOp
// ---------------------------------------------------------------------------

/// Append key/value tensors to the paged KV cache.
///
/// Side-effect op: writes to the KV cache but produces no output tensors.
/// Requires KV cache access; handled specially by the executor.
#[derive(Debug)]
pub struct AppendPagedOp {
    /// Transformer layer index for KV cache write.
    pub layer_idx: usize,
    /// Starting position in the sequence.
    pub start_pos: usize,
}

impl<B: Backend + MatmulOps> OpNode<B> for AppendPagedOp {
    fn name(&self) -> &'static str {
        "append_paged"
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        0
    }
    fn output_shapes(&self, _input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![]
    }
    fn output_dtypes(&self, _input_dtypes: &[DType]) -> Vec<DType> {
        vec![]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!("AppendPagedOp requires KV cache access; handled specially by the executor")
    }
    fn is_side_effect(&self) -> bool {
        true
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// AppendPagedBatchedOp
// ---------------------------------------------------------------------------

/// Batched append of key/value tensors to the paged KV cache.
///
/// Side-effect op: writes to the KV cache but produces no output tensors.
/// Requires KV cache access; handled specially by the executor.
#[derive(Debug)]
pub struct AppendPagedBatchedOp {
    /// Transformer layer index for KV cache write.
    pub layer_idx: usize,
}

impl<B: Backend + MatmulOps> OpNode<B> for AppendPagedBatchedOp {
    fn name(&self) -> &'static str {
        "append_paged_batched"
    }
    fn num_inputs(&self) -> usize {
        4
    }
    fn num_outputs(&self) -> usize {
        0
    }
    fn output_shapes(&self, _input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![]
    }
    fn output_dtypes(&self, _input_dtypes: &[DType]) -> Vec<DType> {
        vec![]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!("AppendPagedBatchedOp requires KV cache access; handled specially by the executor")
    }
    fn is_side_effect(&self) -> bool {
        true
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// GatherPagedKvOp
// ---------------------------------------------------------------------------

/// Gather contiguous K/V tensors from the paged KV cache.
///
/// Materialises `(kv_len, num_kv_heads, head_dim)` tensors for K and V
/// from paged blocks. Requires KV cache access; handled specially by
/// the executor.
#[derive(Debug)]
pub struct GatherPagedKvOp {
    /// Transformer layer index for KV cache read.
    pub layer_idx: usize,
    /// Total key/value sequence length.
    pub kv_len: usize,
    /// Number of key/value heads.
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Data type of the cached key/value tensors.
    pub dtype: DType,
}

impl<B: Backend + MatmulOps> OpNode<B> for GatherPagedKvOp {
    fn name(&self) -> &'static str {
        "gather_paged_kv"
    }
    fn num_inputs(&self) -> usize {
        0
    }
    fn num_outputs(&self) -> usize {
        2
    }
    fn output_shapes(&self, _input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        let s = vec![self.kv_len, self.num_kv_heads, self.head_dim];
        vec![s.clone(), s]
    }
    fn output_dtypes(&self, _input_dtypes: &[DType]) -> Vec<DType> {
        vec![self.dtype, self.dtype]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!("GatherPagedKvOp requires KV cache access; handled specially by the executor")
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// ReshapeOp
// ---------------------------------------------------------------------------

/// Zero-copy view with a different logical shape.
///
/// The executor implements this as a view/reshape on the backing tensor
/// without allocating new memory.
#[derive(Debug)]
pub struct ReshapeOp {
    /// Target shape (must have the same number of elements).
    pub shape: Vec<usize>,
}

impl<B: Backend + MatmulOps> OpNode<B> for ReshapeOp {
    fn name(&self) -> &'static str {
        "reshape"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, _input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![self.shape.clone()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!("ReshapeOp is a zero-copy view; handled specially by the executor")
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// SliceViewOp
// ---------------------------------------------------------------------------

/// Zero-copy sub-slice view of a tensor.
///
/// The executor implements this as a sub-view on the backing tensor
/// without allocating new memory.
#[derive(Debug)]
pub struct SliceViewOp {
    /// Element offset into the source tensor.
    pub offset: usize,
    /// Shape of the resulting view.
    pub shape: Vec<usize>,
}

impl<B: Backend + MatmulOps> OpNode<B> for SliceViewOp {
    fn name(&self) -> &'static str {
        "slice_view"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, _input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![self.shape.clone()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!("SliceViewOp is a zero-copy view; handled specially by the executor")
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Transpose2dOp
// ---------------------------------------------------------------------------

/// 2-D matrix transpose.
///
/// Swaps the two dimensions of a 2-D tensor: `(M, N) → (N, M)`.
#[derive(Debug)]
pub struct Transpose2dOp;

impl<B: Backend + MatmulOps + TensorOps> OpNode<B> for Transpose2dOp {
    fn name(&self) -> &'static str {
        "transpose_2d"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![vec![input_shapes[0][1], input_shapes[0][0]]]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let result = B::transpose_2d(inputs[0])?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// SplitInnerDimOp
// ---------------------------------------------------------------------------

/// Split the innermost dimension of a tensor into two parts.
///
/// Given input of shape `(…, left_size + right_size)`, produces two
/// tensors of shapes `(…, left_size)` and `(…, right_size)`.
#[derive(Debug)]
pub struct SplitInnerDimOp {
    /// Size of the left (first) split.
    pub left_size: usize,
}

impl<B: Backend + MatmulOps + TensorOps> OpNode<B> for SplitInnerDimOp {
    fn name(&self) -> &'static str {
        "split_inner_dim"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        2
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        let total = *input_shapes[0].last().unwrap();
        let right_size = total - self.left_size;
        let mut left = input_shapes[0].to_vec();
        *left.last_mut().unwrap() = self.left_size;
        let mut right = input_shapes[0].to_vec();
        *right.last_mut().unwrap() = right_size;
        vec![left, right]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0], input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let total = inputs[0].shape().last().copied().unwrap();
        let right_size = total - self.left_size;
        let (left, right) = B::split_inner_dim(inputs[0], self.left_size, right_size)?;
        Ok(vec![left, right])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// ConcatInnerDimOp
// ---------------------------------------------------------------------------

/// Concatenate two tensors along their innermost dimension.
///
/// Given inputs of shapes `(…, A)` and `(…, B)`, produces a tensor
/// of shape `(…, A + B)`.
#[derive(Debug)]
pub struct ConcatInnerDimOp;

impl<B: Backend + MatmulOps + TensorOps> OpNode<B> for ConcatInnerDimOp {
    fn name(&self) -> &'static str {
        "concat_inner_dim"
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        let mut s = input_shapes[0].to_vec();
        *s.last_mut().unwrap() += input_shapes[1].last().unwrap();
        vec![s]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let result = B::concat_inner_dim(inputs[0], inputs[1])?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// ConcatSeqOp
// ---------------------------------------------------------------------------

/// Concatenate two tensors along the sequence (first) dimension.
///
/// Given inputs of shapes `(S1, …)` and `(S2, …)`, produces a tensor
/// of shape `(S1 + S2, …)`. Requires backend-specific concat; handled
/// by the executor.
#[derive(Debug)]
pub struct ConcatSeqOp;

impl<B: Backend + MatmulOps> OpNode<B> for ConcatSeqOp {
    fn name(&self) -> &'static str {
        "concat_seq"
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        let mut s = input_shapes[0].to_vec();
        s[0] += input_shapes[1][0];
        vec![s]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!("ConcatSeqOp requires backend-specific concat; handled by the executor")
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// RepeatKvOp
// ---------------------------------------------------------------------------

/// Repeat KV heads to match the number of query heads (GQA).
///
/// Input shape `(seq, kv_heads, head_dim)` → output shape
/// `(seq, kv_heads * num_repeats, head_dim)`.
#[derive(Debug)]
pub struct RepeatKvOp {
    /// Number of times to repeat each KV head.
    pub num_repeats: usize,
}

impl<B: Backend + MatmulOps + TensorOps> OpNode<B> for RepeatKvOp {
    fn name(&self) -> &'static str {
        "repeat_kv"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        let s = input_shapes[0];
        vec![vec![s[0], s[1] * self.num_repeats, s[2]]]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let result = B::repeat_kv(inputs[0], self.num_repeats)?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// ExtractLastRowOp
// ---------------------------------------------------------------------------

/// Extract the last row of a 2-D tensor.
///
/// Used to select the final token's hidden state for next-token
/// prediction. Implemented as a zero-copy view by the executor.
#[derive(Debug)]
pub struct ExtractLastRowOp {
    /// Sequence length (used for offset calculation).
    pub seq_len: usize,
}

impl<B: Backend + MatmulOps> OpNode<B> for ExtractLastRowOp {
    fn name(&self) -> &'static str {
        "extract_last_row"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![vec![1, input_shapes[0][1]]]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!("ExtractLastRowOp is a zero-copy view; handled specially by the executor")
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// CastToF32Op
// ---------------------------------------------------------------------------

/// Cast a tensor to `f32` dtype.
#[derive(Debug)]
pub struct CastToF32Op;

impl<B: Backend + MatmulOps + CastOps> OpNode<B> for CastToF32Op {
    fn name(&self) -> &'static str {
        "cast_to_f32"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, _input_dtypes: &[DType]) -> Vec<DType> {
        vec![DType::F32]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let result = B::cast_to_f32(inputs[0])?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// CastFromF32Op
// ---------------------------------------------------------------------------

/// Cast a tensor from `f32` to a target dtype.
#[derive(Debug)]
pub struct CastFromF32Op {
    /// Target data type for the cast.
    pub target: DType,
}

impl<B: Backend + MatmulOps + CastOps> OpNode<B> for CastFromF32Op {
    fn name(&self) -> &'static str {
        "cast_from_f32"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, _input_dtypes: &[DType]) -> Vec<DType> {
        vec![self.target]
    }
    fn execute(
        &self,
        inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        let result = B::cast_from_f32(inputs[0], self.target)?;
        Ok(vec![result])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// MoeDispatchSoftmaxOp
// ---------------------------------------------------------------------------

/// Softmax `MoE` routing (`Mixtral`, `Qwen`).
///
/// Dispatches tokens to experts using softmax-based gating. Requires
/// closure-based expert dispatch which is not available through the
/// [`OpNode::execute`] interface; handled by the executor.
#[derive(Debug)]
pub struct MoeDispatchSoftmaxOp {
    /// Weight ID of the gating projection.
    pub gate: WeightId,
    /// Per-expert weight IDs for the MLP.
    pub experts: Vec<MoeExpertIds>,
    /// Number of experts selected per token.
    pub num_experts_per_tok: usize,
    /// Whether to normalise top-k probabilities.
    pub norm_topk: bool,
}

impl<B: Backend + MatmulOps> OpNode<B> for MoeDispatchSoftmaxOp {
    fn name(&self) -> &'static str {
        "moe_dispatch_softmax"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!(
            "MoeDispatchSoftmaxOp requires closure-based expert dispatch; 
             handled by the executor"
        )
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// MoeDispatchSigmoidOp
// ---------------------------------------------------------------------------

/// Sigmoid `MoE` routing (`DeepSeek`).
///
/// Dispatches tokens to experts using sigmoid-based gating with bias
/// correction, grouped top-k selection, and optional shared experts.
/// Requires closure-based expert dispatch; handled by the executor.
#[derive(Debug)]
#[allow(clippy::struct_field_names)]
pub struct MoeDispatchSigmoidOp {
    /// Weight ID of the gating projection.
    pub gate: WeightId,
    /// Optional bias weight ID for score correction.
    pub bias: Option<WeightId>,
    /// Per-expert weight IDs for the MLP.
    pub experts: Vec<MoeExpertIds>,
    /// Optional shared-expert weight IDs.
    pub shared_expert: Option<MoeExpertIds>,
    /// Number of experts selected per token.
    pub num_experts_per_tok: usize,
    /// Number of expert groups for grouped top-k.
    pub n_group: usize,
    /// Number of groups selected in top-k.
    pub topk_group: usize,
    /// Scaling factor for routed expert outputs.
    pub routed_scaling_factor: f32,
}

impl<B: Backend + MatmulOps> OpNode<B> for MoeDispatchSigmoidOp {
    fn name(&self) -> &'static str {
        "moe_dispatch_sigmoid"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!(
            "MoeDispatchSigmoidOp requires closure-based expert dispatch; \
             handled by the executor"
        )
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// AllReduceSumOp
// ---------------------------------------------------------------------------

/// Multi-device all-reduce sum for tensor parallelism.
///
/// Requires communicator access which is not available through the
/// [`OpNode::execute`] interface; handled specially by the executor.
#[derive(Debug)]
pub struct AllReduceSumOp;

impl<B: Backend + MatmulOps> OpNode<B> for AllReduceSumOp {
    fn name(&self) -> &'static str {
        "all_reduce_sum"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!("AllReduceSumOp requires communicator access; handled by the executor")
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// LmHeadOp
// ---------------------------------------------------------------------------

/// Language model head projection (final logits computation).
///
/// Projects hidden states to vocabulary-sized logits. The output
/// dtype is always `f32`. Requires dtype-dependent matmul dispatch;
/// handled specially by the executor.
#[derive(Debug)]
pub struct LmHeadOp {
    /// Weight ID of the LM head linear weight.
    pub weight: WeightId,
    /// Data type of the weight (used for dispatch decisions).
    pub weight_dtype: DType,
    /// Vocabulary size (output dimension).
    pub vocab_size: usize,
}

impl<B: Backend + MatmulOps> OpNode<B> for LmHeadOp {
    fn name(&self) -> &'static str {
        "lm_head"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![vec![input_shapes[0][0], self.vocab_size]]
    }
    fn output_dtypes(&self, _input_dtypes: &[DType]) -> Vec<DType> {
        vec![DType::F32]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!("LmHeadOp requires dtype-dependent dispatch; handled by the executor")
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// EmbeddingGatherIndirectOp
// ---------------------------------------------------------------------------

/// Embedding table lookup reading the token ID from a stable GPU `u32` pointer.
///
/// The token ID is not a graph input — it is stored in a `SeqPosition`-like
/// GPU buffer and passed to the executor out-of-band. This makes the op
/// capturable by a CUDA graph (fixed device address, varying value).
///
/// Output shape: `[1, embed_dim]`.
#[derive(Debug)]
pub struct EmbeddingGatherIndirectOp {
    /// Weight ID of the embedding table.
    pub table: WeightId,
    /// Embedding dimension (hidden size).
    pub embed_dim: usize,
    /// Output data type.
    pub dtype: DType,
}

impl<B: Backend + MatmulOps> OpNode<B> for EmbeddingGatherIndirectOp {
    fn name(&self) -> &'static str {
        "embedding_gather_indirect"
    }
    fn num_inputs(&self) -> usize {
        0
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, _input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![vec![1, self.embed_dim]]
    }
    fn output_dtypes(&self, _input_dtypes: &[DType]) -> Vec<DType> {
        vec![self.dtype]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!(
            "EmbeddingGatherIndirectOp must be executed by the CUDA indirect executor; \
             it reads the token ID from a stable GPU device pointer (SeqPosition)"
        )
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// RopeIndirectOp
// ---------------------------------------------------------------------------

/// Rotary positional embedding reading the current position from a stable GPU
/// `u32` pointer (`SeqPosition`).
///
/// Takes one graph input (the Q or K tensor, shape `[1, heads, head_dim]`).
/// The full cos/sin cache (`[max_seq_len, head_dim/2]`) are stored as
/// tensor weights identified by `cos_cache` and `sin_cache`. The position is
/// read at execution time from the `SeqPosition` passed to the executor.
///
/// Output shape: same as input.
#[derive(Debug)]
pub struct RopeIndirectOp {
    /// Weight ID of the full cosine cache `[max_seq_len, head_dim/2]`.
    pub cos_cache: WeightId,
    /// Weight ID of the full sine cache `[max_seq_len, head_dim/2]`.
    pub sin_cache: WeightId,
    /// If `true`, use interleaved `RoPE` (`DeepSeek` style); otherwise standard.
    pub interleaved: bool,
    /// Attention head dimension.
    pub head_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
}

impl<B: Backend + MatmulOps> OpNode<B> for RopeIndirectOp {
    fn name(&self) -> &'static str {
        "rope_indirect"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!(
            "RopeIndirectOp must be executed by the CUDA indirect executor; \
             it reads the sequence position from a stable GPU device pointer (SeqPosition)"
        )
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// AppendKvIndirectOp
// ---------------------------------------------------------------------------

/// Append a new K or V token into the pre-allocated KV cache buffer.
///
/// Takes one graph input (new K or V, shape `[1, kv_heads, head_dim]`).
/// The target KV buffer and write offset are both addressed via `SeqPosition`
/// and the pre-allocated `KvCache` passed to the executor. This is a
/// side-effect op (no output tensor).
///
/// The `layer_idx` determines which layer buffer to write into; `is_key`
/// selects K vs V.
#[derive(Debug)]
pub struct AppendKvIndirectOp {
    /// Transformer layer index.
    pub layer_idx: usize,
    /// `true` = append to K cache; `false` = append to V cache.
    ///
    /// In practice the executor handles both K and V together for the same
    /// `layer_idx`, but having separate ops keeps the graph DAG explicit.
    pub is_key: bool,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Attention head dimension.
    pub head_dim: usize,
}

impl<B: Backend + MatmulOps> OpNode<B> for AppendKvIndirectOp {
    fn name(&self) -> &'static str {
        "append_kv_indirect"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        0
    }
    fn is_side_effect(&self) -> bool {
        true
    }
    fn output_shapes(&self, _input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![]
    }
    fn output_dtypes(&self, _input_dtypes: &[DType]) -> Vec<DType> {
        vec![]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!(
            "AppendKvIndirectOp must be executed by the CUDA indirect executor; \
             it writes into pre-allocated GPU KV cache buffers via SeqPosition"
        )
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// FusedAttentionDecodeIndirectOp
// ---------------------------------------------------------------------------

/// Fused decode attention reading K/V buffers and total sequence length from
/// the executor's out-of-band `KvCache` (indexed by `layer_idx`).
///
/// Takes one graph input: Q `[1, num_heads, head_dim]`. The full K and V cache
/// buffers are fetched from `kv_cache.full_buffers(layer_idx)` at execution
/// time — their GPU addresses are stable across all decode steps, making this
/// op safe to use inside a CUDA graph capture. The actual sequence length is
/// read from the `KvCache`'s internal `SeqPosition` device pointer.
///
/// Output shape: `[1, num_heads, head_dim]`.
#[derive(Debug)]
pub struct FusedAttentionDecodeIndirectOp {
    /// Layer index used to look up K/V buffers from `kv_cache.full_buffers(layer_idx)`.
    pub layer_idx: usize,
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of key/value heads (may differ from `num_heads` for GQA).
    pub num_kv_heads: usize,
    /// Attention head dimension.
    pub head_dim: usize,
    /// Attention scale (`1/sqrt(head_dim)` if standard).
    pub scale: f32,
    /// Optional `logit_softcapping` value (Gemma-style).
    pub softcap: Option<f32>,
    /// Optional sliding window size (Mistral/Qwen-style).
    pub sliding_window: Option<usize>,
}

impl<B: Backend + MatmulOps> OpNode<B> for FusedAttentionDecodeIndirectOp {
    fn name(&self) -> &'static str {
        "fused_attention_decode_indirect"
    }
    fn num_inputs(&self) -> usize {
        1 // Q only; K/V fetched from kv_cache.full_buffers(layer_idx) by executor
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, _input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![vec![1, self.num_heads, self.head_dim]]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!(
            "FusedAttentionDecodeIndirectOp must be executed by the CUDA indirect executor; \
             it reads the total sequence length from a stable GPU device pointer (SeqPosition)"
        )
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// ArgmaxLastOp
// ---------------------------------------------------------------------------

/// Argmax over the last dimension of a 2D tensor, returning a `[1]` U32 tensor.
///
/// During decode the input is logits of shape `[1, vocab_size]`. The output is
/// the token index on the GPU — no device→host transfer is needed until the
/// caller explicitly reads the 4-byte result. This allows the op to be captured
/// inside a CUDA graph and replayed without a D→H sync on every step.
///
/// Must be dispatched by the CUDA indirect executor via `argmax_last_gpu`; the
/// generic `execute()` implementation panics.
#[derive(Debug, Clone)]
pub struct ArgmaxLastOp;

impl<B: Backend + MatmulOps> OpNode<B> for ArgmaxLastOp {
    fn name(&self) -> &'static str {
        "argmax_last"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, _input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![vec![1]]
    }
    fn output_dtypes(&self, _input_dtypes: &[DType]) -> Vec<DType> {
        vec![DType::U32]
    }
    fn execute(
        &self,
        _inputs: &[&B::Tensor],
        _weights: &WeightStore<B::Tensor, <B as MatmulOps>::LinearWeight>,
        _device: &B::DeviceHandle,
    ) -> Result<Vec<B::Tensor>> {
        panic!(
            "ArgmaxLastOp must be dispatched by the CUDA executor via argmax_last_gpu; \
             it cannot be executed through the generic OpNode::execute path"
        )
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
