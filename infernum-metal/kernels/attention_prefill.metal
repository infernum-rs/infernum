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

/// Flash prefill attention: single-pass online softmax + V accumulation.
///
/// One SIMD group (32 threads) per (seq_pos, head) pair.
/// Q is loaded into registers once; K and V are streamed through in a single
/// forward pass with online (Dao-style) softmax — K and V each read exactly once.
///
/// The previous 3-pass kernel recomputed Q·K for every output dimension in Pass 3,
/// reading K (head_dim × kv_len) times instead of kv_len times. This kernel fixes
/// that, reading K and V only once regardless of head_dim.
///
/// Buffers:
///   0: Q   (seq_len, n_heads, head_dim)
///   1: K   (kv_len, kv_heads, head_dim)
///   2: V   (kv_len, kv_heads, head_dim)
///   3: out (seq_len, n_heads, head_dim)
///   4: lse (seq_len, n_heads) — only written if compute_lse != 0
///   5: params
///
/// Grid: (seq_len * n_heads, 1, 1) threadgroups, (32, 1, 1) threads each.
/// head_dim must be ≤ 256 (MAX_HD_PER_LANE * SIMD_WIDTH = 8 * 32).

constant constexpr uint MAX_HD_PER_LANE = 8;
constant constexpr uint SIMD_WIDTH = 32;

kernel void fused_attention_prefill_f32(
    device const float* Q               [[buffer(0)]],
    device const float* K               [[buffer(1)]],
    device const float* V               [[buffer(2)]],
    device float* output                [[buffer(3)]],
    device float* lse                   [[buffer(4)]],
    constant AttentionPrefillParams& p  [[buffer(5)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint seq_idx = gid / p.n_heads;
    const uint head    = gid % p.n_heads;
    const uint kv_h    = head / (p.n_heads / p.kv_heads);
    const uint hd      = p.head_dim;
    const uint kv_stride = p.kv_heads * hd;

    const uint q_pos = p.offset + seq_idx;
    const uint kv_end = min(q_pos + 1, p.kv_len);

    uint kv_start = 0;
    if (p.sliding_window > 0) {
        uint window = uint(p.sliding_window);
        if (q_pos >= window) kv_start = q_pos - window + 1;
    }

    const uint elems_per_lane = (hd >= SIMD_WIDTH) ? (hd / SIMD_WIDTH) : 1;
    const bool lane_active = (lane < hd);
    const uint lane_start  = (hd >= SIMD_WIDTH) ? (lane * elems_per_lane) : lane;

    const uint q_base = (seq_idx * p.n_heads + head) * hd;

    // Load Q into registers once.
    float q_reg[MAX_HD_PER_LANE];
    for (uint i = 0; i < elems_per_lane; i++) {
        q_reg[i] = lane_active ? Q[q_base + lane_start + i] : 0.0f;
    }

    float v_acc[MAX_HD_PER_LANE];
    for (uint i = 0; i < elems_per_lane; i++) v_acc[i] = 0.0f;

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // Single pass: for each KV position read K once, compute score,
    // update online softmax, read V once, accumulate.
    for (uint pos = kv_start; pos < kv_end; pos++) {
        const uint kv_off = (pos * p.kv_heads + kv_h) * hd;

        float partial_dot = 0.0f;
        for (uint i = 0; i < elems_per_lane; i++) {
            float k_val = lane_active ? K[kv_off + lane_start + i] : 0.0f;
            partial_dot += q_reg[i] * k_val;
        }
        float score = simd_sum(partial_dot) * p.scale;

        if (p.softcap > 0.0f) {
            score = p.softcap * precise::tanh(score / p.softcap);
        }

        float old_max = running_max;
        running_max = max(running_max, score);
        float correction = exp(old_max - running_max);
        float exp_score  = exp(score - running_max);

        running_sum = running_sum * correction + exp_score;

        if (lane_active) {
            for (uint i = 0; i < elems_per_lane; i++) {
                v_acc[i] = v_acc[i] * correction
                         + exp_score * V[kv_off + lane_start + i];
            }
        }
    }

    if (lane_active) {
        const float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
        const uint out_base = (seq_idx * p.n_heads + head) * hd;
        for (uint i = 0; i < elems_per_lane; i++) {
            output[out_base + lane_start + i] = v_acc[i] * inv_sum;
        }
    }

    if (p.compute_lse != 0 && lane == 0) {
        lse[seq_idx * p.n_heads + head] = running_max + log(running_sum);
    }
}
