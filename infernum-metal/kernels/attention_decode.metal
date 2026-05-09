#include <metal_stdlib>
using namespace metal;

struct AttentionDecodeParams {
    uint kv_len;        // number of key/value positions
    uint n_heads;       // number of query heads
    uint kv_heads;      // number of KV heads (for GQA)
    uint head_dim;      // dimension per head
    float scale;        // 1/sqrt(head_dim)
    float softcap;      // 0.0 = disabled
    int sliding_window; // -1 = disabled
};

/// Fused decode attention: single query (seq_len=1) attending to all KV.
///
/// Dispatch: one threadgroup per head. Grid: n_heads threadgroups.
/// Each thread strides over KV positions for dot product + softmax.
/// V accumulation is parallelized over head_dim.
///
/// Buffers:
///   0: Q   (1, n_heads, head_dim)
///   1: K   (kv_len, kv_heads, head_dim)
///   2: V   (kv_len, kv_heads, head_dim)
///   3: out (1, n_heads, head_dim) — output
///   4: params
kernel void fused_attention_decode_f32(
    device const float* Q               [[buffer(0)]],
    device const float* K               [[buffer(1)]],
    device const float* V               [[buffer(2)]],
    device float* output                [[buffer(3)]],
    constant AttentionDecodeParams& p   [[buffer(4)]],
    threadgroup float* shared           [[threadgroup(0)]],
    uint gid                            [[threadgroup_position_in_grid]],
    uint lid                            [[thread_position_in_threadgroup]],
    uint tg_size                        [[threads_per_threadgroup]])
{
    const uint head_idx = gid;
    const uint kv_head  = head_idx / (p.n_heads / p.kv_heads);
    const uint head_dim = p.head_dim;
    const uint kv_len   = p.kv_len;
    const float scale   = p.scale;

    // Query position is kv_len - 1 (decode = one step past all cached KV)
    const uint q_pos = kv_len - 1;

    // KV range with sliding window
    uint kv_start = 0;
    uint kv_end = kv_len;
    if (p.sliding_window > 0) {
        uint window = uint(p.sliding_window);
        if (q_pos >= window) {
            kv_start = q_pos - window + 1;
        }
    }

    const uint q_base = head_idx * head_dim;

    // === Pass 1: Find max score ===
    float local_max = -INFINITY;
    for (uint kv = kv_start + lid; kv < kv_end; kv += tg_size) {
        const uint k_base = (kv * p.kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += Q[q_base + d] * K[k_base + d];
        }
        float score = dot * scale;
        if (p.softcap > 0.0f) {
            score = p.softcap * precise::tanh(score / p.softcap);
        }
        local_max = max(local_max, score);
    }

    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] = max(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float global_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Pass 2: Compute exp sum ===
    float local_sum = 0.0f;
    for (uint kv = kv_start + lid; kv < kv_end; kv += tg_size) {
        const uint k_base = (kv * p.kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += Q[q_base + d] * K[k_base + d];
        }
        float score = dot * scale;
        if (p.softcap > 0.0f) {
            score = p.softcap * precise::tanh(score / p.softcap);
        }
        local_sum += exp(score - global_max);
    }

    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float global_sum = shared[0];
    const float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Pass 3: Accumulate weighted V ===
    // Parallelize over head_dim: each thread handles a subset of dimensions.
    const uint out_base = head_idx * head_dim;

    for (uint d = lid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint kv = kv_start; kv < kv_end; kv++) {
            const uint k_base = (kv * p.kv_heads + kv_head) * head_dim;
            float dot = 0.0f;
            for (uint dd = 0; dd < head_dim; dd++) {
                dot += Q[q_base + dd] * K[k_base + dd];
            }
            float score = dot * scale;
            if (p.softcap > 0.0f) {
                score = p.softcap * precise::tanh(score / p.softcap);
            }
            float weight = exp(score - global_max) * inv_sum;
            const uint v_base = (kv * p.kv_heads + kv_head) * head_dim;
            acc += weight * V[v_base + d];
        }
        output[out_base + d] = acc;
    }
}
