//! GPU-side MoE sigmoid routing kernel wrapper.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::missing_panics_doc
)]

use cudarc::driver::LaunchAsync;

use crate::cuda::CudaTensor;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::Result;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels/moe_route_sigmoid.ptx"));
const KERNEL_NAMES: &[&str] = &["moe_route_sigmoid_f32"];

/// GPU-resident routing result for a single decode token.
///
/// Contains the selected expert indices and their weights, already copied
/// to CPU. The DtoH transfer is only 64 bytes (8 × u32 + 8 × f32 for
/// DeepSeek V3), so the sync only waits for the routing kernel — not
/// the entire preceding GPU pipeline.
pub struct GpuRouting {
    /// Selected expert indices on CPU.
    pub expert_indices: Vec<u32>,
    /// Corresponding routing weights on CPU.
    pub expert_weights: Vec<f32>,
}

/// Pre-allocated GPU buffers for MoE routing output.
///
/// Reuse across all MoE layers in a decode step to avoid per-layer
/// `cuMemAllocAsync` / `cuMemFreeAsync` overhead.
pub struct GpuRoutingBuffers {
    out_indices: CudaTensor,
    out_weights: CudaTensor,
}

impl GpuRoutingBuffers {
    /// Allocate routing output buffers for `top_k` experts.
    ///
    /// # Errors
    /// Returns an error if GPU memory allocation fails.
    pub fn new(ctx: &crate::cuda::CudaContext, top_k: usize) -> Result<Self> {
        Ok(Self {
            out_indices: unsafe { CudaTensor::uninit(ctx, &[top_k], DType::U32)? },
            out_weights: unsafe { CudaTensor::uninit(ctx, &[top_k], DType::F32)? },
        })
    }

    /// Run sigmoid routing using these pre-allocated buffers.
    ///
    /// # Errors
    /// Returns an error if kernel launch or DtoH copy fails.
    #[allow(clippy::too_many_arguments)]
    pub fn route(
        &mut self,
        logits: &CudaTensor,
        bias_gpu: &CudaTensor,
        num_experts_per_tok: usize,
        n_group: usize,
        topk_group: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
    ) -> Result<GpuRouting> {
        launch_routing_kernel(
            logits,
            bias_gpu,
            &mut self.out_indices,
            &mut self.out_weights,
            num_experts_per_tok,
            n_group,
            topk_group,
            norm_topk_prob,
            routed_scaling_factor,
        )
    }
}

/// Launch the routing kernel and copy results to CPU.
#[allow(clippy::too_many_arguments)]
fn launch_routing_kernel(
    logits: &CudaTensor,
    bias_gpu: &CudaTensor,
    out_indices: &mut CudaTensor,
    out_weights: &mut CudaTensor,
    num_experts_per_tok: usize,
    n_group: usize,
    topk_group: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
) -> Result<GpuRouting> {
    let num_experts = logits.shape()[1];
    let top_k = num_experts_per_tok;
    let device = logits.context().device();

    // Load kernel
    let module_name = "moe_route_sigmoid";
    let kernel_name = "moe_route_sigmoid_f32";
    if !device.has_func(module_name, kernel_name) {
        device.load_ptx(cudarc::nvrtc::Ptx::from_src(PTX), module_name, KERNEL_NAMES)?;
    }
    let func = device.get_func(module_name, kernel_name).unwrap();

    // Shared memory: scores[E] + biased[E] + group_scores[G] + group_selected[G]
    let smem_bytes = num_experts * 4 + num_experts * 4 + n_group * 4 + n_group * 4;

    // Single block, num_experts threads (max 256 for DeepSeek V3)
    let block_size = num_experts.next_power_of_two().min(256) as u32;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };

    let norm_flag: i32 = i32::from(norm_topk_prob);

    unsafe {
        func.launch(
            cfg,
            (
                &logits.cuda_slice(),
                &bias_gpu.cuda_slice(),
                out_indices.cuda_slice_mut(),
                out_weights.cuda_slice_mut(),
                num_experts as i32,
                top_k as i32,
                n_group as i32,
                topk_group as i32,
                norm_flag,
                routed_scaling_factor,
            ),
        )?;
    }

    // Small DtoH: only top_k values (e.g. 8 × 4 + 8 × 4 = 64 bytes)
    let indices = out_indices.to_vec()?;
    let weights = out_weights.to_vec()?;

    Ok(GpuRouting {
        expert_indices: indices,
        expert_weights: weights,
    })
}

/// Compute sigmoid MoE routing entirely on GPU for single-token decode.
///
/// Allocates temporary output buffers per call. For hot-path decode, prefer
/// `GpuRoutingBuffers` which reuses pre-allocated buffers.
///
/// # Errors
/// Returns an error if kernel launch or DtoH copy fails.
#[allow(clippy::too_many_arguments)]
pub fn moe_route_sigmoid_gpu(
    logits: &CudaTensor,
    bias_gpu: &CudaTensor,
    num_experts_per_tok: usize,
    n_group: usize,
    topk_group: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
) -> Result<GpuRouting> {
    let ctx = logits.context();
    let mut bufs = GpuRoutingBuffers::new(ctx, num_experts_per_tok)?;
    bufs.route(
        logits,
        bias_gpu,
        num_experts_per_tok,
        n_group,
        topk_group,
        norm_topk_prob,
        routed_scaling_factor,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaContext;

    #[test]
    fn test_moe_route_sigmoid_gpu_basic() {
        let ctx = CudaContext::new(0).expect("CUDA context");

        // 8 experts, 2 groups of 4, topk_group=1, top_k=2
        let logits_data = vec![
            2.0_f32, 1.5, 0.1, 0.1, // group 0
            0.1, 0.1, 0.1, 0.1, // group 1
        ];
        let bias_data = vec![0.0_f32; 8];

        let logits = CudaTensor::from_slice(&ctx, &[1, 8], &logits_data).unwrap();
        let bias = CudaTensor::from_slice(&ctx, &[8], &bias_data).unwrap();

        let routing = moe_route_sigmoid_gpu(
            &logits, &bias, 2,    // top_k
            2,    // n_group
            1,    // topk_group
            true, // norm
            1.0,  // scaling
        )
        .unwrap();

        assert_eq!(routing.expert_indices.len(), 2);
        assert_eq!(routing.expert_weights.len(), 2);

        // Group 0 should win (higher scores), experts 0 and 1 selected
        assert_eq!(routing.expert_indices[0], 0);
        assert_eq!(routing.expert_indices[1], 1);

        // Weights should be normalized (sum to 1.0) then scaled by 1.0
        let sum: f32 = routing.expert_weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Normalized weights should sum to 1.0, got {sum}"
        );
    }

    /// Compare GPU routing against the CPU `moe_route_sigmoid` implementation
    /// using DeepSeek V3 parameters: 256 experts, 8 groups, topk_group=4, top_k=8.
    #[test]
    fn test_sigmoid_routing_gpu_matches_cpu() {
        let ctx = CudaContext::new(0).expect("CUDA context");

        let num_experts = 256;
        let n_group = 8;
        let topk_group = 4;
        let top_k = 8;
        let scaling = 2.5_f32;

        // Deterministic "random" logits: sin-based pattern
        let logits_data: Vec<f32> = (0..num_experts)
            .map(|i| (i as f32 * 0.37 + 0.5).sin() * 2.0)
            .collect();

        // Non-zero bias for some experts
        let bias_data: Vec<f32> = (0..num_experts)
            .map(|i| if i % 7 == 0 { 0.3 } else { 0.0 })
            .collect();

        // CPU path: needs hidden @ gate_weight to produce logits.
        // We bypass that by using the logits directly with the CPU routing.
        // The CPU function takes gate output as a Vec, so we replicate
        // the routing logic manually.
        let cpu_routing = {
            let scores: Vec<f32> = logits_data
                .iter()
                .map(|&x| 1.0 / (1.0 + (-x).exp()))
                .collect();
            let biased: Vec<f32> = scores
                .iter()
                .zip(&bias_data)
                .map(|(&s, &b)| s + b)
                .collect();
            let experts_per_group = num_experts / n_group;

            // Group scores
            let group_scores: Vec<f32> = (0..n_group)
                .map(|g| {
                    let gs = g * experts_per_group;
                    let mut group_slice: Vec<(usize, f32)> = biased[gs..gs + experts_per_group]
                        .iter()
                        .copied()
                        .enumerate()
                        .collect();
                    group_slice.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    group_slice.iter().take(2).map(|(_, v)| *v).sum()
                })
                .collect();

            // Select top groups
            let mut group_indexed: Vec<(usize, f32)> =
                group_scores.iter().copied().enumerate().collect();
            group_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let selected_groups: Vec<usize> = group_indexed
                .iter()
                .take(topk_group)
                .map(|&(g, _)| g)
                .collect();

            // Mask non-selected groups
            let mut masked = biased.clone();
            for g in 0..n_group {
                if !selected_groups.contains(&g) {
                    let gs = g * experts_per_group;
                    for v in &mut masked[gs..gs + experts_per_group] {
                        *v = f32::NEG_INFINITY;
                    }
                }
            }

            // Top-k experts
            let mut expert_indexed: Vec<(usize, f32)> =
                masked.iter().copied().enumerate().collect();
            expert_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_experts: Vec<(usize, f32)> = expert_indexed
                .iter()
                .take(top_k)
                .map(|&(idx, _)| (idx, scores[idx]))
                .collect();

            // Normalize and scale
            let sum: f32 = top_experts.iter().map(|(_, w)| *w).sum();
            let result: Vec<(u32, f32)> = top_experts
                .iter()
                .map(|&(idx, w)| (idx as u32, w / sum * scaling))
                .collect();
            result
        };

        // GPU path: upload logits as [1, 256] and run kernel
        let logits_gpu = CudaTensor::from_slice(&ctx, &[1, num_experts], &logits_data).unwrap();
        let bias_gpu = CudaTensor::from_slice(&ctx, &[num_experts], &bias_data).unwrap();

        let gpu_routing = moe_route_sigmoid_gpu(
            &logits_gpu,
            &bias_gpu,
            top_k,
            n_group,
            topk_group,
            true,
            scaling,
        )
        .unwrap();

        // Compare: same experts selected (order may differ, so sort by index)
        let mut cpu_sorted: Vec<(u32, f32)> = cpu_routing;
        cpu_sorted.sort_by_key(|&(idx, _)| idx);

        let mut gpu_sorted: Vec<(u32, f32)> = gpu_routing
            .expert_indices
            .iter()
            .zip(&gpu_routing.expert_weights)
            .map(|(&idx, &w)| (idx, w))
            .collect();
        gpu_sorted.sort_by_key(|&(idx, _)| idx);

        assert_eq!(
            cpu_sorted.len(),
            gpu_sorted.len(),
            "Different number of selected experts"
        );

        for (i, ((cpu_idx, cpu_w), (gpu_idx, gpu_w))) in
            cpu_sorted.iter().zip(&gpu_sorted).enumerate()
        {
            assert_eq!(
                cpu_idx, gpu_idx,
                "Expert index mismatch at position {i}: CPU={cpu_idx}, GPU={gpu_idx}"
            );
            assert!(
                (cpu_w - gpu_w).abs() < 1e-4,
                "Weight mismatch for expert {cpu_idx}: CPU={cpu_w}, GPU={gpu_w}"
            );
        }
    }
}
