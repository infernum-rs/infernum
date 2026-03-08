//! MoeOps and MoeSigmoidOps implementations for Metal.
//!
//! Phase 1: CPU-side routing logic, dispatching to expert_fn for each expert.

use infernum::backend::{MoeOps, MoeSigmoidOps};
use infernum::tensor::Tensor;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;

impl MoeOps for MetalBackend {
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn moe_forward_softmax<F>(
        hidden: &MetalTensor,
        gate_weight: &MetalTensor,
        num_experts: usize,
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        expert_fn: F,
    ) -> Result<MetalTensor>
    where
        F: Fn(usize, &MetalTensor) -> Result<MetalTensor>,
    {
        let data = hidden.as_f32_slice();
        let gate_data = gate_weight.as_f32_slice();
        let hidden_dim = hidden.shape()[hidden.shape().len() - 1];
        let batch = hidden.numel() / hidden_dim;
        let gate_dim = gate_weight.shape()[gate_weight.shape().len() - 1];
        assert_eq!(gate_dim, num_experts);

        let mut result = vec![0.0f32; batch * hidden_dim];

        for b in 0..batch {
            let gate_row = &gate_data[b * num_experts..(b + 1) * num_experts];

            // Softmax over gate scores
            let max_g = gate_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_g: Vec<f32> = gate_row.iter().map(|&g| (g - max_g).exp()).collect();
            let sum_exp: f32 = exp_g.iter().sum();
            let probs: Vec<f32> = exp_g.iter().map(|e| e / sum_exp).collect();

            // Top-k experts
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let topk = &indexed[..num_experts_per_tok];

            let topk_sum: f32 = topk.iter().map(|(_, p)| p).sum();
            let device = metal::Device::system_default()
                .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
            let token = MetalTensor::from_f32(
                &device,
                &[1, hidden_dim],
                &data[b * hidden_dim..(b + 1) * hidden_dim],
            );

            for &(expert_idx, prob) in topk {
                let weight = if norm_topk_prob {
                    prob / topk_sum
                } else {
                    prob
                };
                let expert_out = expert_fn(expert_idx, &token)?;
                let expert_data = expert_out.as_f32_slice();
                for d in 0..hidden_dim {
                    result[b * hidden_dim + d] += weight * expert_data[d];
                }
            }
        }

        let device = metal::Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
        Ok(MetalTensor::from_f32(&device, hidden.shape(), &result))
    }
}

impl MoeSigmoidOps for MetalBackend {
    #[allow(
        clippy::too_many_arguments,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::needless_range_loop
    )]
    fn moe_forward_sigmoid<F>(
        hidden: &MetalTensor,
        gate_weight: &MetalTensor,
        e_score_correction_bias: &[f32],
        num_experts: usize,
        num_experts_per_tok: usize,
        n_group: usize,
        topk_group: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        expert_fn: F,
    ) -> Result<MetalTensor>
    where
        F: Fn(usize, &MetalTensor) -> Result<MetalTensor>,
    {
        let data = hidden.as_f32_slice();
        let gate_data = gate_weight.as_f32_slice();
        let hidden_dim = hidden.shape()[hidden.shape().len() - 1];
        let batch = hidden.numel() / hidden_dim;

        let mut result = vec![0.0f32; batch * hidden_dim];
        let experts_per_group = num_experts / n_group;

        for b in 0..batch {
            let gate_row = &gate_data[b * num_experts..(b + 1) * num_experts];

            // Sigmoid scores with bias correction
            let mut scores: Vec<f32> = gate_row
                .iter()
                .enumerate()
                .map(|(i, &g)| 1.0 / (1.0 + (-g).exp()) + e_score_correction_bias[i])
                .collect();

            // Group-level top-k: pick topk_group groups, zero out rest
            if n_group > 1 && topk_group < n_group {
                let mut group_scores: Vec<(usize, f32)> = (0..n_group)
                    .map(|g| {
                        let start = g * experts_per_group;
                        let end = start + experts_per_group;
                        let max_score = scores[start..end]
                            .iter()
                            .copied()
                            .fold(f32::NEG_INFINITY, f32::max);
                        (g, max_score)
                    })
                    .collect();
                group_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let mut keep_groups = vec![false; n_group];
                for &(g, _) in &group_scores[..topk_group] {
                    keep_groups[g] = true;
                }
                for g in 0..n_group {
                    if !keep_groups[g] {
                        let start = g * experts_per_group;
                        for s in &mut scores[start..start + experts_per_group] {
                            *s = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            // Top-k experts
            let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let topk = &indexed[..num_experts_per_tok];

            let topk_sum: f32 = topk.iter().map(|(_, s)| s).sum();
            let device = metal::Device::system_default()
                .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
            let token = MetalTensor::from_f32(
                &device,
                &[1, hidden_dim],
                &data[b * hidden_dim..(b + 1) * hidden_dim],
            );

            for &(expert_idx, score) in topk {
                let weight = if norm_topk_prob {
                    score / topk_sum
                } else {
                    score
                };
                let weight = weight * routed_scaling_factor;
                let expert_out = expert_fn(expert_idx, &token)?;
                let expert_data = expert_out.as_f32_slice();
                for d in 0..hidden_dim {
                    result[b * hidden_dim + d] += weight * expert_data[d];
                }
            }
        }

        let device = metal::Device::system_default()
            .ok_or_else(|| infernum::Error::Other("No Metal device".into()))?;
        Ok(MetalTensor::from_f32(&device, hidden.shape(), &result))
    }
}
