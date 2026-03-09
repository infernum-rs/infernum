#include <metal_stdlib>
using namespace metal;

struct AttentionPrefillParams {
    uint seq_len;       // number of query positions
    uint kv_len;        // number of key/value positions
    uint n_heads;       // number of query heads
    uint kv_heads;      // number of KV heads (for GQA)
    uint head_dim;      // dimension per head
    uint offset;        // position offset for causal mask
    float scale;        // 1/sqrt(head_dim)
    float softcap;      // softcap value (0.0 = disabled)
    int sliding_window; // -1 = disabled
    uint compute_lse;   // 1 = compute, 0 = skip
};

/// Fused prefill attention: QK^T + causal mask + softcap + sliding window + softmax + V.
///
/// Dispatch: one threadgroup per (seq_pos, head) pair.
/// Grid: (seq_len * n_heads) threadgroups, each with tg_size threads.
/// Each thread strides over the KV positions.
///
/// Buffers:
///   0: Q   (seq_len, n_heads, head_dim)
///   1: K   (kv_len, kv_heads, head_dim)
///   2: V   (kv_len, kv_heads, head_dim)
///   3: out (seq_len, n_heads, head_dim) — output
///   4: lse (seq_len, n_heads) — log-sum-exp (only written if compute_lse)
///   5: params
kernel void fused_attention_prefill_f32(
    device const float* Q               [[buffer(0)]],
    device const float* K               [[buffer(1)]],
    device const float* V               [[buffer(2)]],
    device float* output                [[buffer(3)]],
    device float* lse                   [[buffer(4)]],
    constant AttentionPrefillParams& p  [[buffer(5)]],
    threadgroup float* shared           [[threadgroup(0)]],
    uint gid                            [[threadgroup_position_in_grid]],
    uint lid                            [[thread_position_in_threadgroup]],
    uint tg_size                        [[threads_per_threadgroup]])
{
    const uint seq_idx  = gid / p.n_heads;
    const uint head_idx = gid % p.n_heads;
    const uint kv_head  = head_idx / (p.n_heads / p.kv_heads);
    const uint head_dim = p.head_dim;
    const uint kv_len   = p.kv_len;
    const float scale   = p.scale;

    // Query position in the full sequence
    const uint q_pos = p.offset + seq_idx;

    // Causal mask: attend to [kv_start..kv_end)
    uint kv_start = 0;
    uint kv_end = min(q_pos + 1, kv_len);

    // Sliding window
    if (p.sliding_window > 0) {
        uint window = uint(p.sliding_window);
        if (q_pos >= window) {
            kv_start = max(kv_start, q_pos - window + 1);
        }
    }

    // Pointers
    const uint q_base = (seq_idx * p.n_heads + head_idx) * head_dim;
    // K/V layout: (kv_len, kv_heads, head_dim)

    // Online softmax + V accumulation.
    // Each thread handles a strided subset of KV positions.
    // Phase 1: compute local (max, exp_sum, weighted_v)
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    // Per-thread V accumulator in registers (up to head_dim elements).
    // For large head_dim this is fine — typical values are 64-128.
    // We store in shared memory at the end.
    // Actually, we need to accumulate V across threads, so we use
    // a different approach: two passes.

    // === Pass 1: Find global max score ===
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

    // Reduce max across threads
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

    // === Pass 2: Compute exp(score - max) sum and accumulate weighted V ===
    // We need head_dim floats for V accumulation per thread.
    // Since head_dim can be up to 128 and we have up to 256 threads,
    // we can't put per-thread V accumulators in shared memory.
    // Instead: compute softmax weights, then do a third pass for V.

    // Compute exp sum
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
    // Each thread handles a subset of head_dim dimensions.
    // For each dimension, iterate over all KV positions, recompute weight.
    const uint out_base = (seq_idx * p.n_heads + head_idx) * head_dim;

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

    // Write LSE if requested
    if (p.compute_lse != 0 && lid == 0) {
        lse[seq_idx * p.n_heads + head_idx] = global_max + log(global_sum);
    }
}
