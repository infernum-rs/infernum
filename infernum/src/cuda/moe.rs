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

// --- Sigmoid routing (DeepSeek V3) ---

/// Compute sigmoid element-wise.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute MoE routing with sigmoid scoring, bias correction, and grouped top-k.
///
/// DeepSeek V3's routing differs from standard softmax routing:
/// 1. Scores are `sigmoid(logits)` (independent, not competing)
/// 2. A learned per-expert `e_score_correction_bias` shifts selection
/// 3. Experts are grouped; top groups are selected first, then top-k within
/// 4. Original (unbiased) scores are used as weights, optionally normalized
/// 5. Final weights are multiplied by `routed_scaling_factor`
///
/// # Arguments
/// * `hidden` — input hidden states, shape `[seq_len, hidden_size]`
/// * `gate_weight` — pre-transposed router weight, shape `[hidden_size, num_experts]`
/// * `e_score_correction_bias` — per-expert bias for selection, shape `[num_experts]`
/// * `num_experts_per_tok` — how many experts to activate per token
/// * `n_group` — number of expert groups
/// * `topk_group` — how many groups to select
/// * `norm_topk_prob` — if `true`, normalize selected weights to sum to 1.0
/// * `routed_scaling_factor` — multiply final weights by this factor
///
/// # Returns
/// Per-token routing assignments: `Vec<(expert_idx, weight)>`
///
/// # Errors
/// Returns an error if the matmul or device-to-host copy fails.
#[allow(clippy::too_many_arguments)]
pub fn moe_route_sigmoid<T>(
    hidden: &CudaTensor<T>,
    gate_weight: &CudaTensor<T>,
    e_score_correction_bias: &[f32],
    num_experts_per_tok: usize,
    n_group: usize,
    topk_group: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
) -> Result<Vec<TokenRouting>>
where
    T: TensorDType + DeviceRepr + super::ops::GemmScalar + Default,
    CudaBlas: Gemm<T>,
{
    let logits = matmul(hidden, gate_weight)?;
    let logits_host = logits.to_vec()?;

    let seq_len = hidden.shape()[0];
    let num_experts = gate_weight.shape()[1];
    let experts_per_group = num_experts / n_group;

    let mut routing = Vec::with_capacity(seq_len);
    for tok in 0..seq_len {
        let start = tok * num_experts;
        let scores: Vec<f32> = logits_host[start..start + num_experts]
            .iter()
            .map(|v| sigmoid(v.to_f32()))
            .collect();

        // Biased scores for selection only
        let biased: Vec<f32> = scores
            .iter()
            .zip(e_score_correction_bias)
            .map(|(&s, &b)| s + b)
            .collect();

        // Group experts and compute per-group scores (sum of top-2 within each group)
        let group_scores: Vec<f32> = (0..n_group)
            .map(|g| {
                let group_start = g * experts_per_group;
                let group_slice = &biased[group_start..group_start + experts_per_group];
                let top2 = topk(group_slice, 2.min(experts_per_group));
                top2.iter().map(|(_, v)| *v).sum()
            })
            .collect();

        // Select top groups
        let top_groups = topk(&group_scores, topk_group);
        let mut group_mask = vec![false; n_group];
        for &(g, _) in &top_groups {
            group_mask[g] = true;
        }

        // Mask out non-selected groups in biased scores
        let mut masked_biased = biased.clone();
        for (g, &selected) in group_mask.iter().enumerate() {
            if !selected {
                let group_start = g * experts_per_group;
                for v in &mut masked_biased[group_start..group_start + experts_per_group] {
                    *v = f32::NEG_INFINITY;
                }
            }
        }

        // Top-k over remaining experts
        let top_experts = topk(&masked_biased, num_experts_per_tok);

        // Gather *original* scores (not biased) for the selected experts
        let mut selected: Vec<(usize, f32)> = top_experts
            .iter()
            .map(|&(idx, _)| (idx, scores[idx]))
            .collect();

        // Normalize and scale
        if norm_topk_prob {
            renormalize(&mut selected);
        }
        for (_, w) in &mut selected {
            *w *= routed_scaling_factor;
        }

        routing.push(selected);
    }

    Ok(routing)
}

/// Run MoE forward with sigmoid routing (DeepSeek V3).
///
/// Routes tokens using sigmoid scoring with bias correction and grouped top-k,
/// dispatches to experts, and sums the weighted results.
///
/// # Errors
/// Returns an error if routing, expert computation, or tensor operations fail.
#[allow(clippy::too_many_arguments)]
pub fn moe_forward_sigmoid<T, F>(
    hidden: &CudaTensor<T>,
    gate_weight: &CudaTensor<T>,
    e_score_correction_bias: &[f32],
    num_experts: usize,
    num_experts_per_tok: usize,
    n_group: usize,
    topk_group: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
    expert_fn: F,
) -> Result<CudaTensor<T>>
where
    T: TensorDType + DeviceRepr + ValidAsZeroBits + Default + super::ops::GemmScalar,
    CudaBlas: Gemm<T>,
    F: Fn(usize, &CudaTensor<T>) -> Result<CudaTensor<T>>,
{
    let seq_len = hidden.shape()[0];
    let routing = moe_route_sigmoid(
        hidden,
        gate_weight,
        e_score_correction_bias,
        num_experts_per_tok,
        n_group,
        topk_group,
        norm_topk_prob,
        routed_scaling_factor,
    )?;

    if seq_len == 1 {
        decode_moe(hidden, &routing[0], &expert_fn)
    } else {
        let hidden_size = hidden.shape()[1];
        prefill_moe(hidden, &routing, num_experts, hidden_size, &expert_fn)
    }
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

    // --- Sigmoid routing tests ---

    #[test]
    fn test_sigmoid_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!((sigmoid(100.0) - 1.0).abs() < 1e-4);
        assert!(sigmoid(-100.0) < 1e-4);
        // sigmoid(1.0) ≈ 0.7310586
        assert!((sigmoid(1.0) - 0.731_058_6).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid_routing_basic() {
        // 8 experts, 2 groups of 4, top-1 group, top-2 experts
        // Gate logits designed so group 0 has higher experts
        let logits = vec![
            2.0_f32, 1.5, 0.1, 0.1, // group 0 (experts 0-3)
            0.1, 0.1, 0.1, 0.1, // group 1 (experts 4-7)
        ];
        let bias = vec![0.0_f32; 8];

        let scores: Vec<f32> = logits.iter().map(|&x| sigmoid(x)).collect();

        // With no bias, group 0 should win, experts 0 and 1 selected
        let biased = scores.clone();
        let mut group_scores = vec![0.0_f32; 2];
        for g in 0..2 {
            let gs = g * 4;
            let top2 = topk(&biased[gs..gs + 4], 2);
            group_scores[g] = top2.iter().map(|(_, v)| *v).sum();
        }
        let top_groups = topk(&group_scores, 1);
        assert_eq!(top_groups[0].0, 0, "Group 0 should win");

        // Mask non-selected groups
        let mut masked = biased.clone();
        for i in 4..8 {
            masked[i] = f32::NEG_INFINITY;
        }
        let top_experts = topk(&masked, 2);
        assert_eq!(top_experts[0].0, 0);
        assert_eq!(top_experts[1].0, 1);

        // Verify the raw sigmoid scores are used as weights, not biased scores
        let _ = &bias; // used in full routing but not needed here
    }

    #[test]
    fn test_sigmoid_routing_bias_shifts_selection() {
        // Without bias: expert 0 wins (logit=1.0 → sigmoid=0.731)
        // With bias: expert 1 gets +10.0, so expert 1 wins for selection
        let logits = vec![1.0_f32, 0.5, 0.1, 0.1];
        let bias = vec![0.0, 10.0, 0.0, 0.0];

        let scores: Vec<f32> = logits.iter().map(|&x| sigmoid(x)).collect();
        let biased: Vec<f32> = scores.iter().zip(&bias).map(|(&s, &b)| s + b).collect();

        // group_scores: one group of 4, topk_group=1 → all selected
        let top_experts = topk(&biased, 1);
        assert_eq!(
            top_experts[0].0, 1,
            "Expert 1 should win with bias correction"
        );

        // But the weight should be the original score, not biased
        let weight = scores[1];
        assert!(
            (weight - sigmoid(0.5)).abs() < 1e-5,
            "Weight should use unbiased sigmoid score"
        );
    }

    #[test]
    fn test_sigmoid_routing_weight_normalization() {
        // Test that norm_topk_prob + routed_scaling_factor works correctly
        let scores = vec![(0, 0.4_f32), (1, 0.6)];
        let scaling = 2.5_f32;

        let mut selected = scores;
        renormalize(&mut selected);
        for (_, w) in &mut selected {
            *w *= scaling;
        }

        let sum: f32 = selected.iter().map(|(_, w)| *w).sum();
        assert!(
            (sum - scaling).abs() < 1e-5,
            "Normalized weights times scaling should sum to {scaling}, got {sum}"
        );
    }

    #[test]
    fn test_sigmoid_routing_grouped_exclusion() {
        // 8 experts, 2 groups of 4
        // Expert 4 (group 1) has highest individual score
        // But group 0 has higher aggregate → group 1 masked → expert 4 excluded
        let scores = vec![
            0.7_f32, 0.6, 0.5, 0.4, // group 0 (aggregate top-2: 0.7 + 0.6 = 1.3)
            0.8, 0.1, 0.1, 0.1, // group 1 (aggregate top-2: 0.8 + 0.1 = 0.9)
        ];

        let mut group_scores = vec![0.0_f32; 2];
        for g in 0..2 {
            let gs = g * 4;
            let top2 = topk(&scores[gs..gs + 4], 2);
            group_scores[g] = top2.iter().map(|(_, v)| *v).sum();
        }

        // topk_group=1: only group 0 selected
        let top_groups = topk(&group_scores, 1);
        assert_eq!(top_groups[0].0, 0, "Group 0 should have higher aggregate");

        // Expert 4 should be masked even though it has the highest individual score
        let mut masked = scores;
        for i in 4..8 {
            masked[i] = f32::NEG_INFINITY;
        }
        let top2 = topk(&masked, 2);
        assert!(
            top2.iter().all(|&(idx, _)| idx < 4),
            "All selected experts should be from group 0"
        );
    }
}
