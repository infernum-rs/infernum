//! Mixture-of-Experts (MoE) routing and dispatch primitives.
//!
//! Reusable by any model crate that implements MoE layers (Mixtral, Qwen-MoE, etc.).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown
)]

use cudarc::cublas::{CudaBlas, Gemm};
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

use super::ops::matmul;
use super::CudaTensor;
use crate::dtype::TensorDType;
use crate::tensor::Tensor;
use crate::Result;

/// Routing assignments for a single token: `(expert_index, normalized_weight)`.
type TokenRouting = Vec<(usize, f32)>;

/// Compute softmax over a slice in-place.
fn softmax_inplace(logits: &mut [f32]) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in logits.iter_mut() {
        *v /= sum;
    }
}

/// Select the top-k indices from `values`, returning `(index, value)` pairs.
fn topk(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

/// Renormalize a set of weights so they sum to 1.0.
fn renormalize(selections: &mut [(usize, f32)]) {
    let sum: f32 = selections.iter().map(|(_, w)| *w).sum();
    if sum > 0.0 {
        for (_, w) in selections.iter_mut() {
            *w /= sum;
        }
    }
}

/// Compute MoE routing: `hidden @ gate_weight → softmax → topk → renormalize`.
///
/// # Arguments
/// * `hidden` — input hidden states, shape `[seq_len, hidden_size]`
/// * `gate_weight` — pre-transposed router weight, shape `[hidden_size, num_experts]`
/// * `num_experts_per_tok` — how many experts to activate per token
/// * `norm_topk_prob` — if `true`, renormalize selected expert weights to sum
///   to 1.0 after top-K selection. Mixtral and Qwen3-MoE use `true`.
///
/// # Returns
/// Per-token routing assignments: for each token, a `Vec<(expert_idx, weight)>`
/// of length `num_experts_per_tok`.
///
/// # Errors
/// Returns an error if the matmul or device-to-host copy fails.
pub fn moe_route<T>(
    hidden: &CudaTensor<T>,
    gate_weight: &CudaTensor<T>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
) -> Result<Vec<TokenRouting>>
where
    T: TensorDType + DeviceRepr + super::ops::GemmScalar + Default,
    CudaBlas: Gemm<T>,
{
    // [seq_len, hidden_size] @ [hidden_size, num_experts] → [seq_len, num_experts]
    let logits = matmul(hidden, gate_weight)?;
    let logits_host = logits.to_vec()?;

    let seq_len = hidden.shape()[0];
    let num_experts = gate_weight.shape()[1];

    let mut routing = Vec::with_capacity(seq_len);
    for tok in 0..seq_len {
        let start = tok * num_experts;
        let mut row: Vec<f32> = logits_host[start..start + num_experts]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        softmax_inplace(&mut row);
        let mut selected = topk(&row, num_experts_per_tok);
        if norm_topk_prob {
            renormalize(&mut selected);
        }
        routing.push(selected);
    }

    Ok(routing)
}

/// Run MoE forward: route tokens to experts, dispatch, weighted-sum results.
///
/// For each token, the router selects `num_experts_per_tok` experts. Each
/// expert processes its assigned tokens via `expert_fn`, and the results are
/// combined using the routing weights.
///
/// # Arguments
/// * `hidden` — input hidden states, shape `[seq_len, hidden_size]`
/// * `gate_weight` — pre-transposed router weight, shape `[hidden_size, num_experts]`
/// * `num_experts` — total number of experts
/// * `num_experts_per_tok` — experts activated per token
/// * `norm_topk_prob` — renormalize top-K weights to sum to 1.0
/// * `expert_fn` — called as `expert_fn(expert_idx, expert_input)` → expert output
///
/// # Errors
/// Returns an error if routing, expert computation, or tensor operations fail.
pub fn moe_forward<T, F>(
    hidden: &CudaTensor<T>,
    gate_weight: &CudaTensor<T>,
    num_experts: usize,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    expert_fn: F,
) -> Result<CudaTensor<T>>
where
    T: TensorDType + DeviceRepr + ValidAsZeroBits + Default + super::ops::GemmScalar,
    CudaBlas: Gemm<T>,
    F: Fn(usize, &CudaTensor<T>) -> Result<CudaTensor<T>>,
{
    let seq_len = hidden.shape()[0];
    let hidden_size = hidden.shape()[1];
    let routing = moe_route(hidden, gate_weight, num_experts_per_tok, norm_topk_prob)?;

    if seq_len == 1 {
        // Decode fast path: single token, no gather/scatter needed
        decode_moe(hidden, &routing[0], &expert_fn)
    } else {
        // Prefill: group tokens by expert, gather, compute, scatter-add
        prefill_moe(hidden, &routing, num_experts, hidden_size, &expert_fn)
    }
}

/// Decode fast path: single token, run each selected expert and weighted-sum.
fn decode_moe<T, F>(
    hidden: &CudaTensor<T>,
    assignments: &[(usize, f32)],
    expert_fn: &F,
) -> Result<CudaTensor<T>>
where
    T: TensorDType + DeviceRepr + ValidAsZeroBits + Default,
    F: Fn(usize, &CudaTensor<T>) -> Result<CudaTensor<T>>,
{
    let hidden_size = hidden.shape()[1];
    let ctx = hidden.context();

    // Accumulate weighted expert outputs on CPU
    let mut accum = vec![0.0_f32; hidden_size];

    for &(expert_idx, weight) in assignments {
        let expert_out = expert_fn(expert_idx, hidden)?;
        let out_host = expert_out.to_vec()?;
        for (a, v) in accum.iter_mut().zip(&out_host) {
            *a += weight * v.to_f32();
        }
    }

    // Convert back to T
    let result_data: Vec<T> = accum.iter().map(|v| T::from_f32(*v)).collect();
    CudaTensor::from_slice(ctx, &[1, hidden_size], &result_data)
}

/// Prefill path: group tokens by expert, gather rows, run experts, scatter-add.
fn prefill_moe<T, F>(
    hidden: &CudaTensor<T>,
    routing: &[TokenRouting],
    num_experts: usize,
    hidden_size: usize,
    expert_fn: &F,
) -> Result<CudaTensor<T>>
where
    T: TensorDType + DeviceRepr + ValidAsZeroBits + Default,
    F: Fn(usize, &CudaTensor<T>) -> Result<CudaTensor<T>>,
{
    let seq_len = routing.len();
    let ctx = hidden.context();

    // Read all hidden states to CPU for gather/scatter
    let hidden_host = hidden.to_vec()?;
    let mut output_host = vec![0.0_f32; seq_len * hidden_size];

    for expert_idx in 0..num_experts {
        // Find tokens assigned to this expert and their weights
        let assignments: Vec<(usize, f32)> = routing
            .iter()
            .enumerate()
            .filter_map(|(tok, choices)| {
                choices
                    .iter()
                    .find(|(eidx, _)| *eidx == expert_idx)
                    .map(|(_, w)| (tok, *w))
            })
            .collect();

        if assignments.is_empty() {
            continue;
        }

        // Gather: build input tensor for this expert
        let num_tokens = assignments.len();
        let mut gathered = vec![T::default(); num_tokens * hidden_size];
        for (local_idx, &(tok_idx, _)) in assignments.iter().enumerate() {
            let src = &hidden_host[tok_idx * hidden_size..(tok_idx + 1) * hidden_size];
            gathered[local_idx * hidden_size..(local_idx + 1) * hidden_size].copy_from_slice(src);
        }

        let expert_input = CudaTensor::from_slice(ctx, &[num_tokens, hidden_size], &gathered)?;
        let expert_output = expert_fn(expert_idx, &expert_input)?;
        let expert_host = expert_output.to_vec()?;

        // Scatter-add: accumulate weighted expert output back to token positions
        for (local_idx, &(tok_idx, weight)) in assignments.iter().enumerate() {
            let src_start = local_idx * hidden_size;
            let dst_start = tok_idx * hidden_size;
            for i in 0..hidden_size {
                output_host[dst_start + i] += weight * expert_host[src_start + i].to_f32();
            }
        }
    }

    // Convert f32 accumulator back to T
    let result_data: Vec<T> = output_host.iter().map(|v| T::from_f32(*v)).collect();
    CudaTensor::from_slice(ctx, &[seq_len, hidden_size], &result_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sums_to_one() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        softmax_inplace(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum = {sum}");
        // Values should be monotonically increasing
        for i in 1..logits.len() {
            assert!(logits[i] > logits[i - 1]);
        }
    }

    #[test]
    fn test_softmax_uniform() {
        let mut logits = vec![0.0; 8];
        softmax_inplace(&mut logits);
        for v in &logits {
            assert!((*v - 0.125).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let mut logits = vec![1000.0, 1001.0, 1002.0];
        softmax_inplace(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum = {sum}");
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_topk_selects_highest() {
        let values = vec![0.1, 0.4, 0.05, 0.3, 0.15];
        let top2 = topk(&values, 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, 1); // index of 0.4
        assert_eq!(top2[1].0, 3); // index of 0.3
    }

    #[test]
    fn test_topk_single() {
        let values = vec![0.2, 0.8];
        let top1 = topk(&values, 1);
        assert_eq!(top1.len(), 1);
        assert_eq!(top1[0].0, 1);
    }

    #[test]
    fn test_renormalize_sums_to_one() {
        let mut selections = vec![(1, 0.3_f32), (3, 0.2)];
        renormalize(&mut selections);
        let sum: f32 = selections.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 1e-6, "renormalized sum = {sum}");
        assert!((selections[0].1 - 0.6).abs() < 1e-6);
        assert!((selections[1].1 - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_renormalize_already_normalized() {
        let mut selections = vec![(0, 0.5_f32), (1, 0.5)];
        renormalize(&mut selections);
        assert!((selections[0].1 - 0.5).abs() < 1e-6);
        assert!((selections[1].1 - 0.5).abs() < 1e-6);
    }
}
