//! MoeOps and MoeSigmoidOps implementations for CpuBackend.

use infernum::backend::{MoeOps, MoeSigmoidOps};
use infernum::tensor::Tensor;
use infernum::Result;

use crate::tensor::CpuTensor;
use crate::CpuBackend;

impl MoeOps for CpuBackend {
    fn moe_forward_softmax<F>(
        hidden: &CpuTensor,
        gate_weight: &CpuTensor,
        num_experts: usize,
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        expert_fn: F,
    ) -> Result<CpuTensor>
    where
        F: Fn(usize, &CpuTensor) -> Result<CpuTensor>,
    {
        // gate_weight: (num_experts, hidden_size)
        // hidden: (seq_len, hidden_size)
        let hidden_data = hidden.as_f32_slice();
        let gate_data = gate_weight.as_f32_slice();
        let seq_len = hidden.shape()[0];
        let hidden_size = hidden.shape()[1];

        let mut output_data = vec![0.0f32; seq_len * hidden_size];

        for s in 0..seq_len {
            let token = &hidden_data[s * hidden_size..(s + 1) * hidden_size];

            // Compute gate scores: gate_weight @ token
            let mut scores = vec![0.0f32; num_experts];
            for e in 0..num_experts {
                let gate_row = &gate_data[e * hidden_size..(e + 1) * hidden_size];
                scores[e] = crate::simd::dot_f32(gate_row, token);
            }

            // Softmax
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum: f32 = exp_scores.iter().sum();
            for s in &mut exp_scores {
                *s /= sum;
            }

            // Top-k selection
            let mut indexed: Vec<(usize, f32)> = exp_scores.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let topk = &indexed[..num_experts_per_tok];

            let mut weights: Vec<(usize, f32)> = topk.to_vec();
            if norm_topk_prob {
                let weight_sum: f32 = weights.iter().map(|(_, w)| w).sum();
                for (_, w) in &mut weights {
                    *w /= weight_sum;
                }
            }

            // Run experts and combine
            let token_tensor = CpuTensor::from_f32(&[1, hidden_size], token);
            for (expert_idx, weight) in weights {
                let expert_out = expert_fn(expert_idx, &token_tensor)?;
                let expert_data = expert_out.as_f32_slice();
                for d in 0..hidden_size {
                    output_data[s * hidden_size + d] += weight * expert_data[d];
                }
            }
        }

        Ok(CpuTensor::from_f32(hidden.shape(), &output_data))
    }
}

impl MoeSigmoidOps for CpuBackend {
    #[allow(clippy::too_many_arguments)]
    fn moe_forward_sigmoid<F>(
        hidden: &CpuTensor,
        gate_weight: &CpuTensor,
        e_score_correction_bias: &[f32],
        num_experts: usize,
        num_experts_per_tok: usize,
        n_group: usize,
        topk_group: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        expert_fn: F,
    ) -> Result<CpuTensor>
    where
        F: Fn(usize, &CpuTensor) -> Result<CpuTensor>,
    {
        let hidden_data = hidden.as_f32_slice();
        let gate_data = gate_weight.as_f32_slice();
        let seq_len = hidden.shape()[0];
        let hidden_size = hidden.shape()[1];

        let mut output_data = vec![0.0f32; seq_len * hidden_size];

        for s in 0..seq_len {
            let token = &hidden_data[s * hidden_size..(s + 1) * hidden_size];

            // Compute gate scores with sigmoid
            let mut scores = vec![0.0f32; num_experts];
            for e in 0..num_experts {
                let gate_row = &gate_data[e * hidden_size..(e + 1) * hidden_size];
                let logit = crate::simd::dot_f32(gate_row, token);
                scores[e] = 1.0 / (1.0 + (-logit).exp()); // sigmoid
            }

            // Apply bias correction
            let mut corrected = scores.clone();
            for e in 0..num_experts {
                corrected[e] += e_score_correction_bias[e];
            }

            // Grouped top-k: select topk_group groups first
            let experts_per_group = num_experts / n_group;
            let mut group_scores: Vec<(usize, f32)> = (0..n_group)
                .map(|g| {
                    let start = g * experts_per_group;
                    let end = start + experts_per_group;
                    let max_score = corrected[start..end]
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    (g, max_score)
                })
                .collect();
            group_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Mask out non-selected groups
            let mut mask = vec![false; num_experts];
            for &(g, _) in group_scores.iter().take(topk_group) {
                let start = g * experts_per_group;
                for m in &mut mask[start..start + experts_per_group] {
                    *m = true;
                }
            }

            // Top-k from masked experts using corrected scores
            let mut candidates: Vec<(usize, f32)> = (0..num_experts)
                .filter(|&e| mask[e])
                .map(|e| (e, corrected[e]))
                .collect();
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let topk = &candidates[..num_experts_per_tok.min(candidates.len())];

            // Use original sigmoid scores (not corrected) as weights
            let mut weights: Vec<(usize, f32)> =
                topk.iter().map(|&(e, _)| (e, scores[e])).collect();

            if norm_topk_prob {
                let weight_sum: f32 = weights.iter().map(|(_, w)| w).sum();
                if weight_sum > 0.0 {
                    for (_, w) in &mut weights {
                        *w /= weight_sum;
                    }
                }
            }

            // Run experts and combine
            let token_tensor = CpuTensor::from_f32(&[1, hidden_size], token);
            for (expert_idx, weight) in weights {
                let expert_out = expert_fn(expert_idx, &token_tensor)?;
                let expert_data = expert_out.as_f32_slice();
                for d in 0..hidden_size {
                    output_data[s * hidden_size + d] += weight * expert_data[d];
                }
            }

            // Apply routed scaling factor
            for d in 0..hidden_size {
                output_data[s * hidden_size + d] *= routed_scaling_factor;
            }
        }

        Ok(CpuTensor::from_f32(hidden.shape(), &output_data))
    }
}
