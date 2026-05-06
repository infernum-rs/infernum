//! Built-in [`OpNode`] implementations for the computation graph.
//!
//! Each struct wraps a single graph operation (matmul, `RmsNorm`, `RoPE`, etc.)
//! with its parameters, shape/dtype inference, and backend execution logic.
//! Ops that require resources unavailable through the [`OpNode::execute`]
//! interface (KV cache, communicator, closure-based dispatch) panic at
//! runtime and must be handled specially by the executor.

use std::any::Any;

use crate::backend::{
    ArithOps, AttentionOps, Backend, BiasOps, CastOps, ContextBackend, EmbedOps, GegluOps,
    MatmulExtOps, MatmulOps, NormOps, RopeInterleavedOps, RopeOps, SwigluOps, TensorOps,
};
use crate::dtype::DType;
use crate::tensor::Tensor as TensorTrait;
use crate::Result;

use super::execute_context::ExecuteContext;
use super::node::{NodeId, WeightId};
use super::op_node::{OpNode, OutputRef};

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

impl<B: ContextBackend> OpNode<B> for InputOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        _inputs: &[OutputRef],
    ) -> Result<()> {
        if ctx
            .kv_cache
            .as_ref()
            .is_some_and(|kv| kv.is_cache_input(node_id))
        {
            return Ok(());
        }
        let tensor = B::ctx_next_input(ctx);
        B::ctx_write(ctx, node_id, 0, tensor);
        Ok(())
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

impl<B: ContextBackend + EmbedOps> OpNode<B> for EmbeddingGatherOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let token_ids = B::ctx_read(ctx, inputs[0]);
        let table_w = ctx.weights.tensor_weight(self.table).clone();
        let seq_len = token_ids.shape()[0];
        let result = <B as EmbedOps>::embedding_gather_tensor(&table_w, &token_ids, seq_len)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + NormOps> OpNode<B> for RmsNormOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let input = B::ctx_read(ctx, inputs[0]);
        let weight = ctx.weights.tensor_weight(self.weight).clone();
        let result = <B as NormOps>::rms_norm(&input, &weight, self.eps)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + NormOps> OpNode<B> for AddRmsNormOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let residual = B::ctx_read(ctx, inputs[0]);
        let delta = B::ctx_read(ctx, inputs[1]);
        let weight = ctx.weights.tensor_weight(self.weight).clone();
        let (updated, normed) = <B as NormOps>::add_rmsnorm(&residual, &delta, &weight, self.eps)?;
        B::ctx_write(ctx, node_id, 0, updated);
        B::ctx_write(ctx, node_id, 1, normed);
        Ok(())
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

impl<B: ContextBackend> OpNode<B> for LinearOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let input = B::ctx_read(ctx, inputs[0]);
        let weight = ctx.weights.linear_weight(self.weight);
        let result = <B as MatmulOps>::linear(&input, weight)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend> OpNode<B> for LinearPairOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let input = B::ctx_read(ctx, inputs[0]);
        let lw1 = ctx.weights.linear_weight(self.w1);
        let lw2 = ctx.weights.linear_weight(self.w2);
        let (out1, out2) = <B as MatmulOps>::linear_pair(&input, lw1, lw2)?;
        B::ctx_write(ctx, node_id, 0, out1);
        B::ctx_write(ctx, node_id, 1, out2);
        Ok(())
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

impl<B: ContextBackend> OpNode<B> for LinearTripleOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let input = B::ctx_read(ctx, inputs[0]);
        let lw1 = ctx.weights.linear_weight(self.w1);
        let lw2 = ctx.weights.linear_weight(self.w2);
        let lw3 = ctx.weights.linear_weight(self.w3);
        let (out1, out2, out3) = <B as MatmulOps>::linear_triple(&input, lw1, lw2, lw3)?;
        B::ctx_write(ctx, node_id, 0, out1);
        B::ctx_write(ctx, node_id, 1, out2);
        B::ctx_write(ctx, node_id, 2, out3);
        Ok(())
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

impl<B: ContextBackend> OpNode<B> for MatmulOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let a = B::ctx_read(ctx, inputs[0]);
        let b = B::ctx_read(ctx, inputs[1]);
        let result = <B as MatmulOps>::matmul(&a, &b)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + MatmulExtOps> OpNode<B> for MatmulBf16F32Op {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let a = B::ctx_read(ctx, inputs[0]);
        let b = B::ctx_read(ctx, inputs[1]);
        let result = <B as MatmulExtOps>::matmul_bf16_f32(&a, &b)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + SwigluOps> OpNode<B> for SwigluOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let gate = B::ctx_read(ctx, inputs[0]);
        let up = B::ctx_read(ctx, inputs[1]);
        let result = <B as SwigluOps>::swiglu(&gate, &up)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + GegluOps> OpNode<B> for GegluOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let gate = B::ctx_read(ctx, inputs[0]);
        let up = B::ctx_read(ctx, inputs[1]);
        let result = <B as GegluOps>::geglu(&gate, &up)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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
        _ctx: &mut ExecuteContext<'_, B>,
        _node_id: NodeId,
        _inputs: &[OutputRef],
    ) -> Result<()> {
        unimplemented!("SiluOp should be fused into SwigluOp by the graph optimiser")
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

impl<B: ContextBackend + ArithOps> OpNode<B> for AddOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let a = B::ctx_read(ctx, inputs[0]);
        let b = B::ctx_read(ctx, inputs[1]);
        let result = <B as ArithOps>::add(&a, &b)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + ArithOps> OpNode<B> for AddInplaceOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let a = B::ctx_read(ctx, inputs[0]);
        let b = B::ctx_read(ctx, inputs[1]);
        let result = <B as ArithOps>::add(&a, &b)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + ArithOps> OpNode<B> for MulOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let a = B::ctx_read(ctx, inputs[0]);
        let b = B::ctx_read(ctx, inputs[1]);
        let result = <B as ArithOps>::mul(&a, &b)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + ArithOps> OpNode<B> for ScaleOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let mut result = B::ctx_read(ctx, inputs[0]);
        <B as ArithOps>::scale_inplace(&mut result, self.factor)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + BiasOps> OpNode<B> for BiasAddOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let mut result = B::ctx_read(ctx, inputs[0]);
        let bias = ctx.weights.tensor_weight(self.bias).clone();
        <B as BiasOps>::bias_add_inplace(&mut result, &bias)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + RopeOps> OpNode<B> for RopeOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let input = B::ctx_read(ctx, inputs[0]);
        let cos = B::ctx_read(ctx, inputs[1]);
        let sin = B::ctx_read(ctx, inputs[2]);
        let result = <B as RopeOps>::apply_rope(&input, &cos, &sin, self.offset)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + RopeOps> OpNode<B> for RopeBatchedOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let input = B::ctx_read(ctx, inputs[0]);
        let cos = B::ctx_read(ctx, inputs[1]);
        let sin = B::ctx_read(ctx, inputs[2]);
        let positions = B::ctx_read(ctx, inputs[3]);
        let result =
            <B as RopeOps>::apply_rope_batched(&input, &cos, &sin, &positions, self.batch_size)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + RopeInterleavedOps> OpNode<B> for RopeInterleavedOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let input = B::ctx_read(ctx, inputs[0]);
        let cos = B::ctx_read(ctx, inputs[1]);
        let sin = B::ctx_read(ctx, inputs[2]);
        let result =
            <B as RopeInterleavedOps>::apply_rope_interleaved(&input, &cos, &sin, self.offset)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + AttentionOps> OpNode<B> for FusedAttentionPrefillOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let q = B::ctx_read(ctx, inputs[0]);
        let k = B::ctx_read(ctx, inputs[1]);
        let v = B::ctx_read(ctx, inputs[2]);
        let result = <B as AttentionOps>::fused_attention_prefill(
            &q,
            &k,
            &v,
            self.offset,
            self.scale,
            self.softcap,
            self.sliding_window,
        )?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + AttentionOps> OpNode<B> for FusedAttentionDecodeOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let q = B::ctx_read(ctx, inputs[0]);
        let k = B::ctx_read(ctx, inputs[1]);
        let v = B::ctx_read(ctx, inputs[2]);
        let result =
            <B as AttentionOps>::fused_attention_decode(&q, &k, &v, None, self.softcap, None)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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
    /// Optional attention logit soft-cap (`tanh(x / cap) * cap`). Used by Gemma 2.
    pub softcap: Option<f32>,
}

impl<B: Backend + MatmulOps> OpNode<B> for PagedAttentionDecodeOp {
    fn name(&self) -> &'static str {
        "paged_attention_decode"
    }
    fn num_inputs(&self) -> usize {
        // inputs[0]: q
        // inputs[1]: block_tables
        // inputs[2]: seq_lens
        // inputs[3]: positions
        // inputs[4]: append_ref (dummy — creates ordering edge; not used at runtime)
        5
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
        _ctx: &mut ExecuteContext<'_, B>,
        _node_id: NodeId,
        _inputs: &[OutputRef],
    ) -> Result<()> {
        unimplemented!("PagedAttentionDecodeOp requires paged KV cache — handled by executor")
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
    /// One dummy output so `(node_id, 0)` is a valid [`OutputRef`] scheduling handle.
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, _input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![vec![]] // dummy; never read by executor
    }
    fn output_dtypes(&self, _input_dtypes: &[DType]) -> Vec<DType> {
        vec![DType::U32] // dummy; never used
    }
    fn execute(
        &self,
        _ctx: &mut ExecuteContext<'_, B>,
        _node_id: NodeId,
        _inputs: &[OutputRef],
    ) -> Result<()> {
        unimplemented!("AppendPagedOp requires paged KV cache write access — handled by executor")
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
    /// One dummy output declared so that `(node_id, 0)` is a valid [`OutputRef`].
    ///
    /// The scheduling trick in [`GraphPagedKvCacheOps::add_append_paged_batched`]
    /// returns `(node_id, 0)` which is passed as a dependency input to the
    /// subsequent `paged_attention_decode` node. Graph construction calls
    /// `output_shape(node_id, 0)` on that ref, so we must declare at least one
    /// output shape or it panics. The executor's buffer initialiser already
    /// allocates a slot via `.max(1)`, so this declaration aligns with that.
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, _input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        vec![vec![]] // dummy scalar-shape marker; never read by the executor
    }
    fn output_dtypes(&self, _input_dtypes: &[DType]) -> Vec<DType> {
        vec![DType::U32] // dummy; never used
    }
    fn execute(
        &self,
        _ctx: &mut ExecuteContext<'_, B>,
        _node_id: NodeId,
        _inputs: &[OutputRef],
    ) -> Result<()> {
        unimplemented!(
            "AppendPagedBatchedOp requires paged KV cache write access — handled by executor"
        )
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
        _ctx: &mut ExecuteContext<'_, B>,
        _node_id: NodeId,
        _inputs: &[OutputRef],
    ) -> Result<()> {
        unimplemented!("GatherPagedKvOp requires paged KV cache read access — handled by executor")
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

impl<B: ContextBackend> OpNode<B> for ReshapeOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let input = B::ctx_read(ctx, inputs[0]);
        let result = input.reshape(&self.shape);
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend> OpNode<B> for SliceViewOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let input = B::ctx_read(ctx, inputs[0]);
        let result = input.slice_view(self.offset, &self.shape);
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + TensorOps> OpNode<B> for Transpose2dOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let input = B::ctx_read(ctx, inputs[0]);
        let result = <B as TensorOps>::transpose_2d(&input)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + TensorOps> OpNode<B> for SplitInnerDimOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let input = B::ctx_read(ctx, inputs[0]);
        let total = *input.shape().last().unwrap();
        let right_size = total - self.left_size;
        let (left, right) = <B as TensorOps>::split_inner_dim(&input, self.left_size, right_size)?;
        B::ctx_write(ctx, node_id, 0, left);
        B::ctx_write(ctx, node_id, 1, right);
        Ok(())
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

impl<B: ContextBackend + TensorOps> OpNode<B> for ConcatInnerDimOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let a = B::ctx_read(ctx, inputs[0]);
        let b = B::ctx_read(ctx, inputs[1]);
        let result = <B as TensorOps>::concat_inner_dim(&a, &b)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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
        _ctx: &mut ExecuteContext<'_, B>,
        _node_id: NodeId,
        _inputs: &[OutputRef],
    ) -> Result<()> {
        unimplemented!("ConcatSeqOp requires KV cache sequence concat — handled by executor")
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

impl<B: ContextBackend + TensorOps> OpNode<B> for RepeatKvOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let input = B::ctx_read(ctx, inputs[0]);
        let result = <B as TensorOps>::repeat_kv(&input, self.num_repeats)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend> OpNode<B> for ExtractLastRowOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let input = B::ctx_read(ctx, inputs[0]);
        let cols = input.shape()[1];
        let offset = (self.seq_len - 1) * cols;
        let out_shape = vec![1, cols];
        let result = input.slice_view(offset, &out_shape);
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + CastOps> OpNode<B> for CastToF32Op {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let input = B::ctx_read(ctx, inputs[0]);
        let result = <B as CastOps>::cast_to_f32(&input)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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

impl<B: ContextBackend + CastOps> OpNode<B> for CastFromF32Op {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let input = B::ctx_read(ctx, inputs[0]);
        let result = <B as CastOps>::cast_from_f32(&input, self.target)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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
        _ctx: &mut ExecuteContext<'_, B>,
        _node_id: NodeId,
        _inputs: &[OutputRef],
    ) -> Result<()> {
        panic!("MoeDispatchSoftmaxOp requires closure-based expert dispatch — handled by executor")
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
        _ctx: &mut ExecuteContext<'_, B>,
        _node_id: NodeId,
        _inputs: &[OutputRef],
    ) -> Result<()> {
        panic!("MoeDispatchSigmoidOp requires closure-based expert dispatch — handled by executor")
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
        _ctx: &mut ExecuteContext<'_, B>,
        _node_id: NodeId,
        _inputs: &[OutputRef],
    ) -> Result<()> {
        unimplemented!(
            "AllReduceSumOp requires multi-device NCCL communicator — handled by executor"
        )
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

impl<B: ContextBackend> OpNode<B> for LmHeadOp {
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
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let input = B::ctx_read(ctx, inputs[0]);
        let weight = ctx.weights.linear_weight(self.weight);
        let result = <B as MatmulOps>::linear(&input, weight)?;
        B::ctx_write(ctx, node_id, 0, result);
        Ok(())
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
        _ctx: &mut ExecuteContext<'_, B>,
        _node_id: NodeId,
        _inputs: &[OutputRef],
    ) -> Result<()> {
        unimplemented!(
            "EmbeddingGatherIndirectOp requires SeqPosition GPU pointer — handled by CUDA graph executor"
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
        _ctx: &mut ExecuteContext<'_, B>,
        _node_id: NodeId,
        _inputs: &[OutputRef],
    ) -> Result<()> {
        unimplemented!(
            "RopeIndirectOp requires SeqPosition GPU pointer — handled by CUDA graph executor"
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
        _ctx: &mut ExecuteContext<'_, B>,
        _node_id: NodeId,
        _inputs: &[OutputRef],
    ) -> Result<()> {
        unimplemented!(
            "AppendKvIndirectOp requires KV cache device buffer access — handled by CUDA graph executor"
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
        _ctx: &mut ExecuteContext<'_, B>,
        _node_id: NodeId,
        _inputs: &[OutputRef],
    ) -> Result<()> {
        unimplemented!(
            "FusedAttentionDecodeIndirectOp requires stable KV cache device buffers — handled by CUDA graph executor"
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
        _ctx: &mut ExecuteContext<'_, B>,
        _node_id: NodeId,
        _inputs: &[OutputRef],
    ) -> Result<()> {
        unimplemented!(
            "ArgmaxLastOp must use argmax_last_gpu kernel — handled by CUDA graph executor"
        )
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// RmsNormQkOp
// ---------------------------------------------------------------------------

/// Per-head `RMSNorm` applied to `Q` and `K` tensors before `RoPE`.
///
/// Used by `Qwen3` and `Gemma3` models which apply an independent `RMSNorm` to
/// each attention head in Q and K before the rotary embedding is applied.
/// The op takes two inputs (`Q`, `K`) and returns two outputs (`Q_normed`, `K_normed`)
/// with identical shapes to their respective inputs.
///
/// Shape contract:
/// - `inputs[0]` = Q: `[seq, num_q_heads, head_dim]`
/// - `inputs[1]` = K: `[seq, num_kv_heads, head_dim]`
/// - `outputs[0]` = `Q_normed`: `[seq, num_q_heads, head_dim]`
/// - `outputs[1]` = `K_normed`: `[seq, num_kv_heads, head_dim]`
#[derive(Debug, Clone)]
pub struct RmsNormQkOp {
    /// Weight ID for the per-head Q normalization weight (shape `[head_dim]`).
    pub q_weight: WeightId,
    /// Weight ID for the per-head K normalization weight (shape `[head_dim]`).
    pub k_weight: WeightId,
    /// Epsilon for numerical stability.
    pub eps: f32,
}

impl<B: ContextBackend + NormOps> OpNode<B> for RmsNormQkOp {
    fn name(&self) -> &'static str {
        "rms_norm_qk"
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        2
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        // Q and K shapes are preserved element-wise.
        vec![input_shapes[0].to_vec(), input_shapes[1].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0], input_dtypes[1]]
    }
    fn execute(
        &self,
        ctx: &mut ExecuteContext<'_, B>,
        node_id: NodeId,
        inputs: &[OutputRef],
    ) -> Result<()> {
        let q = B::ctx_read(ctx, inputs[0]);
        let k = B::ctx_read(ctx, inputs[1]);
        let q_weight = ctx.weights.tensor_weight(self.q_weight).clone();
        let k_weight = ctx.weights.tensor_weight(self.k_weight).clone();
        let q_normed = <B as NormOps>::rms_norm(&q, &q_weight, self.eps)?;
        let k_normed = <B as NormOps>::rms_norm(&k, &k_weight, self.eps)?;
        B::ctx_write(ctx, node_id, 0, q_normed);
        B::ctx_write(ctx, node_id, 1, k_normed);
        Ok(())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// LogitSoftcapOp
// ---------------------------------------------------------------------------

/// Element-wise logit soft-cap: `tanh(x / cap) * cap`.
///
/// Used by Gemma 2 as a final logit soft-capping step after the LM head
/// projection to bound the logit magnitude. The `cap` hyper-parameter is
/// typically `30.0` for Gemma 2 models.
///
/// Shape contract:
/// - `inputs[0]`: any shape `[...]` with dtype `F32`.
/// - `outputs[0]`: same shape and dtype as `inputs[0]`.
#[derive(Debug, Clone)]
pub struct LogitSoftcapOp {
    /// Soft-capping value (must be positive).
    pub cap: f32,
}

impl<B: Backend + MatmulOps> OpNode<B> for LogitSoftcapOp {
    fn name(&self) -> &'static str {
        "logit_softcap"
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
        _ctx: &mut ExecuteContext<'_, B>,
        _node_id: NodeId,
        _inputs: &[OutputRef],
    ) -> Result<()> {
        unimplemented!(
            "LogitSoftcapOp requires backend-specific softcap kernel — handled by executor"
        )
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// MlaAttentionOp
// ---------------------------------------------------------------------------

/// Multi-head Latent Attention (MLA) as used by `DeepSeek` V3 / R1.
///
/// This is an **opaque** op that encapsulates the entire MLA forward pass:
/// `Q` `LoRA` compression + `RMSNorm` + expansion, joint `KV` projection + `RMSNorm`,
/// `KV` decompression, interleaved `RoPE`, `Q` absorption, fused attention (with
/// optional two-pass LSE combining for multi-chunk prefill), V absorption,
/// and the output projection.
///
/// The KV cache stores the compressed latent `(c_kv, k_rope)` — this is a
/// special case that the generic graph IR does not generalize over, making
/// opacity the right design choice.
///
/// Shape contract:
/// - `inputs[0]`: `[seq_len, hidden_size]` hidden states.
/// - `outputs[0]`: `[seq_len, hidden_size]` attention output.
///
/// All weight tensors are fetched from the `WeightStore` via the stored IDs.
/// The executor handles KV cache access and the full MLA algorithm.
#[derive(Debug)]
pub struct MlaAttentionOp {
    // --- Attention projection weights ---
    /// Q compression linear: `(hidden_size, q_lora_rank)`.
    pub q_a_proj: WeightId,
    /// Q compression `RMSNorm` weight: `(q_lora_rank,)`.
    pub q_a_layernorm: WeightId,
    /// Q expansion linear: `(q_lora_rank, num_heads * qk_head_dim)`.
    pub q_b_proj: WeightId,
    /// Joint KV compression linear: `(hidden_size, kv_lora_rank + qk_rope_head_dim)`.
    pub kv_a_proj_with_mqa: WeightId,
    /// KV compression `RMSNorm` weight: `(kv_lora_rank,)`.
    pub kv_a_layernorm: WeightId,
    /// K-nope decompression matrix (dense): `(kv_lora_rank, num_heads * qk_nope_head_dim)`.
    pub kv_b_proj_k: WeightId,
    /// V decompression matrix (dense, batched): `(num_heads, kv_lora_rank, v_head_dim)`.
    pub kv_b_proj_v: WeightId,
    /// K-transposed for Q absorption (batched): `(num_heads, qk_nope_head_dim, kv_lora_rank)`.
    pub kv_b_proj_k_t: WeightId,
    /// Output projection linear: `(num_heads * v_head_dim, hidden_size)`.
    pub o_proj: WeightId,
    // --- Dimension parameters (inlined to avoid config dep in graph IR) ---
    /// Number of attention heads on this device (post-TP sharding).
    pub num_heads: usize,
    /// Non-RoPE portion of Q/K head dim.
    pub qk_nope_head_dim: usize,
    /// `RoPE` portion of Q/K head dim.
    pub qk_rope_head_dim: usize,
    /// Value head dim.
    pub v_head_dim: usize,
    /// KV compression rank.
    pub kv_lora_rank: usize,
    /// `RMSNorm` epsilon.
    pub rms_norm_eps: f32,
    /// Attention scale factor (pre-computed, includes `YaRN` mscale if applicable).
    pub attn_scale: f32,
    /// Layer index (for KV cache slot selection).
    pub layer_idx: usize,
}

impl<B: Backend + MatmulOps> OpNode<B> for MlaAttentionOp {
    fn name(&self) -> &'static str {
        "mla_attention"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn output_shapes(&self, input_shapes: &[&[usize]]) -> Vec<Vec<usize>> {
        // Output hidden states have the same shape as input hidden states.
        vec![input_shapes[0].to_vec()]
    }
    fn output_dtypes(&self, input_dtypes: &[DType]) -> Vec<DType> {
        vec![input_dtypes[0]]
    }
    fn execute(
        &self,
        _ctx: &mut ExecuteContext<'_, B>,
        _node_id: NodeId,
        _inputs: &[OutputRef],
    ) -> Result<()> {
        unimplemented!(
            "MlaAttentionOp is a composite op requiring MLA-specific KV cache logic — handled by executor"
        )
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
