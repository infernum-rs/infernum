// GPU-side MoE sigmoid routing for single-token decode.
//
// Performs the full DeepSeek V3 routing pipeline on GPU:
//   1. Sigmoid activation on gate logits
//   2. Bias correction for expert selection
//   3. Grouped top-k selection (top groups, then top experts within)
//   4. Weight normalization and scaling
//
// Single block, up to 256 threads (one per expert).
// Output: top_k expert indices (u32) and weights (f32).

static __device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Insertion sort to find top-k values and their indices from shared memory.
// Writes results to out_vals[0..k] and out_idxs[0..k] in descending order.
static __device__ void topk_shared(
    const float* __restrict__ vals,
    int n,
    int k,
    float* out_vals,
    unsigned int* out_idxs
) {
    for (int i = 0; i < k; i++) {
        out_vals[i] = -1e38f;
        out_idxs[i] = 0;
    }
    for (int i = 0; i < n; i++) {
        float v = vals[i];
        if (v > out_vals[k - 1]) {
            // Insert into sorted position
            int pos = k - 1;
            while (pos > 0 && v > out_vals[pos - 1]) {
                out_vals[pos] = out_vals[pos - 1];
                out_idxs[pos] = out_idxs[pos - 1];
                pos--;
            }
            out_vals[pos] = v;
            out_idxs[pos] = (unsigned int)i;
        }
    }
}

extern "C" __global__ void moe_route_sigmoid_f32(
    const float* __restrict__ logits,       // [num_experts] gate output
    const float* __restrict__ bias,         // [num_experts] e_score_correction_bias
    unsigned int* __restrict__ out_indices,  // [top_k] selected expert indices
    float* __restrict__ out_weights,         // [top_k] selected expert weights
    const int num_experts,
    const int top_k,
    const int n_group,
    const int topk_group,
    const int norm_topk_prob,
    const float scaling_factor
) {
    // Shared memory layout:
    //   float scores[num_experts]        — original sigmoid scores
    //   float biased[num_experts]        — scores + bias (for selection)
    //   float group_scores[n_group]      — per-group aggregate
    //   int   group_selected[n_group]    — 1 if group is in top topk_group
    extern __shared__ char smem[];

    const int tid = threadIdx.x;
    const int experts_per_group = num_experts / n_group;

    float* scores = reinterpret_cast<float*>(smem);
    float* biased = scores + num_experts;
    float* group_scores = biased + num_experts;
    int* group_selected = reinterpret_cast<int*>(group_scores + n_group);

    // Step 1: Each thread computes sigmoid and biased score for its expert
    if (tid < num_experts) {
        float s = sigmoid(logits[tid]);
        scores[tid] = s;
        biased[tid] = s + bias[tid];
    }
    __syncthreads();

    // Step 2: Compute per-group scores (sum of top-2 biased scores within each group)
    // Thread 0 does this serially — n_group is small (8 for DeepSeek V3)
    if (tid == 0) {
        for (int g = 0; g < n_group; g++) {
            int gs = g * experts_per_group;
            float top2_vals[2];
            unsigned int top2_idxs[2];
            topk_shared(biased + gs, experts_per_group, 2, top2_vals, top2_idxs);
            group_scores[g] = top2_vals[0] + top2_vals[1];
            group_selected[g] = 0;
        }

        // Step 3: Select top topk_group groups
        for (int i = 0; i < topk_group; i++) {
            float best = -1e38f;
            int best_g = 0;
            for (int g = 0; g < n_group; g++) {
                if (!group_selected[g] && group_scores[g] > best) {
                    best = group_scores[g];
                    best_g = g;
                }
            }
            group_selected[best_g] = 1;
        }
    }
    __syncthreads();

    // Step 4: Mask non-selected groups in biased scores
    if (tid < num_experts) {
        int group_id = tid / experts_per_group;
        if (!group_selected[group_id]) {
            biased[tid] = -1e38f;
        }
    }
    __syncthreads();

    // Step 5: Thread 0 selects top_k experts globally from masked biased scores
    if (tid == 0) {
        float topk_vals[64];   // max top_k we'll ever need
        unsigned int topk_idxs[64];
        topk_shared(biased, num_experts, top_k, topk_vals, topk_idxs);

        // Step 6: Gather original scores as weights and optionally normalize
        float weight_sum = 0.0f;
        for (int i = 0; i < top_k; i++) {
            topk_vals[i] = scores[topk_idxs[i]];
            weight_sum += topk_vals[i];
        }

        if (norm_topk_prob && weight_sum > 0.0f) {
            for (int i = 0; i < top_k; i++) {
                topk_vals[i] /= weight_sum;
            }
        }

        // Step 7: Apply scaling factor and write output
        for (int i = 0; i < top_k; i++) {
            out_indices[i] = topk_idxs[i];
            out_weights[i] = topk_vals[i] * scaling_factor;
        }
    }
}
