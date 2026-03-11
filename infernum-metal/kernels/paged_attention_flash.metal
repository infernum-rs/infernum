#include <metal_stdlib>
using namespace metal;

// Flash decode attention with online softmax.
//
// Single-pass over KV positions: computes Q·K, updates running softmax,
// and accumulates weighted V in one loop. O(seq_len × head_dim) instead
// of O(seq_len × head_dim²) from the old 3-pass kernel.
//
// Dispatch: 1 threadgroup per (batch, head).
// Threads per threadgroup: 32 (one SIMD group).
//
// For head_dim >= 32: each lane handles head_dim/32 contiguous elements.
//   Lane i handles elements [i * (hd/32) .. (i+1) * (hd/32)).
// For head_dim < 32: each of the first head_dim lanes handles 1 element;
//   remaining lanes contribute 0 to the dot product and skip V writes.

struct FlashDecodeParams {
    uint batch_size;
    uint n_heads;           // query heads
    uint kv_heads;          // KV heads (for GQA)
    uint head_dim;
    uint block_size;        // tokens per KV block
    uint max_blocks_per_seq;
    float scale;
    float softcap;          // 0.0 = disabled
    int sliding_window;     // -1 = disabled
};

// Maximum elements per lane: head_dim=256 / 32 lanes = 8.
constant constexpr uint MAX_HD_PER_LANE = 8;
constant constexpr uint SIMD_WIDTH = 32;

// ---------------------------------------------------------------------------
// f16 variant — Q/K/V/output are half, scores in float
// ---------------------------------------------------------------------------
kernel void paged_attention_flash_decode_f16(
    device const half*   Q            [[buffer(0)]],
    device const half*   K_pool       [[buffer(1)]],
    device const half*   V_pool       [[buffer(2)]],
    device const int*    block_tables [[buffer(3)]],
    device const int*    seq_lens     [[buffer(4)]],
    device half*         output       [[buffer(5)]],
    constant FlashDecodeParams& p    [[buffer(6)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint b    = gid / p.n_heads;
    const uint h    = gid % p.n_heads;
    const uint kv_h = h / (p.n_heads / p.kv_heads);
    const uint kv_stride = p.kv_heads * p.head_dim;
    const uint hd   = p.head_dim;

    const int seq_len = seq_lens[b];
    if (seq_len <= 0) return;
    const uint sl = uint(seq_len);

    // Sliding window bounds
    uint kv_start = 0;
    uint kv_end = sl;
    if (p.sliding_window > 0) {
        uint window = uint(p.sliding_window);
        uint q_pos = sl - 1;
        if (q_pos >= window) {
            kv_start = q_pos - window + 1;
        }
    }

    const uint q_base = (b * p.n_heads + h) * hd;
    device const int* bt_row = block_tables + b * p.max_blocks_per_seq;

    // For hd >= 32: each lane handles a contiguous chunk of hd/32 elements.
    // For hd < 32:  each lane handles at most 1 element.
    const uint elems_per_lane = (hd >= SIMD_WIDTH) ? (hd / SIMD_WIDTH) : 1;
    const bool lane_active = (lane < hd);
    // Starting offset into the head dimension for this lane.
    const uint lane_start = (hd >= SIMD_WIDTH) ? (lane * elems_per_lane) : lane;

    // Load Q into registers
    float q_reg[MAX_HD_PER_LANE];
    for (uint i = 0; i < elems_per_lane; i++) {
        q_reg[i] = lane_active ? float(Q[q_base + lane_start + i]) : 0.0f;
    }

    // V accumulator
    float v_acc[MAX_HD_PER_LANE];
    for (uint i = 0; i < elems_per_lane; i++) {
        v_acc[i] = 0.0f;
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // Single pass over all KV positions
    for (uint pos = kv_start; pos < kv_end; pos++) {
        uint blk_idx = pos / p.block_size;
        uint blk_off = pos % p.block_size;
        uint phys_block = uint(bt_row[blk_idx]);
        uint kv_off = (phys_block * p.block_size + blk_off) * kv_stride + kv_h * hd;

        // --- Compute Q · K[pos] ---
        // Each lane computes dot for its chunk, then simd_sum reduces
        float partial_dot = 0.0f;
        for (uint i = 0; i < elems_per_lane; i++) {
            float k_val = lane_active ? float(K_pool[kv_off + lane_start + i]) : 0.0f;
            partial_dot += q_reg[i] * k_val;
        }
        float score = simd_sum(partial_dot) * p.scale;

        // Soft-capping (Gemma)
        if (p.softcap > 0.0f) {
            score = p.softcap * precise::tanh(score / p.softcap);
        }

        // --- Online softmax update ---
        float old_max = running_max;
        running_max = max(running_max, score);
        float correction = exp(old_max - running_max);
        float exp_score = exp(score - running_max);

        running_sum = running_sum * correction + exp_score;
        for (uint i = 0; i < elems_per_lane; i++) {
            v_acc[i] = v_acc[i] * correction;
        }

        // Accumulate weighted V
        if (lane_active) {
            for (uint i = 0; i < elems_per_lane; i++) {
                float v_val = float(V_pool[kv_off + lane_start + i]);
                v_acc[i] += exp_score * v_val;
            }
        }
    }

    // Normalize and write output
    if (lane_active) {
        const float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
        const uint out_base = (b * p.n_heads + h) * hd;
        for (uint i = 0; i < elems_per_lane; i++) {
            output[out_base + lane_start + i] = half(v_acc[i] * inv_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// f32 variant — Q/K/V/output are float
// ---------------------------------------------------------------------------
kernel void paged_attention_flash_decode_f32(
    device const float*  Q            [[buffer(0)]],
    device const float*  K_pool       [[buffer(1)]],
    device const float*  V_pool       [[buffer(2)]],
    device const int*    block_tables [[buffer(3)]],
    device const int*    seq_lens     [[buffer(4)]],
    device float*        output       [[buffer(5)]],
    constant FlashDecodeParams& p    [[buffer(6)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint b    = gid / p.n_heads;
    const uint h    = gid % p.n_heads;
    const uint kv_h = h / (p.n_heads / p.kv_heads);
    const uint kv_stride = p.kv_heads * p.head_dim;
    const uint hd   = p.head_dim;

    const int seq_len = seq_lens[b];
    if (seq_len <= 0) return;
    const uint sl = uint(seq_len);

    uint kv_start = 0;
    uint kv_end = sl;
    if (p.sliding_window > 0) {
        uint window = uint(p.sliding_window);
        uint q_pos = sl - 1;
        if (q_pos >= window) {
            kv_start = q_pos - window + 1;
        }
    }

    const uint q_base = (b * p.n_heads + h) * hd;
    device const int* bt_row = block_tables + b * p.max_blocks_per_seq;

    const uint elems_per_lane = (hd >= SIMD_WIDTH) ? (hd / SIMD_WIDTH) : 1;
    const bool lane_active = (lane < hd);
    const uint lane_start = (hd >= SIMD_WIDTH) ? (lane * elems_per_lane) : lane;

    float q_reg[MAX_HD_PER_LANE];
    for (uint i = 0; i < elems_per_lane; i++) {
        q_reg[i] = lane_active ? Q[q_base + lane_start + i] : 0.0f;
    }

    float v_acc[MAX_HD_PER_LANE];
    for (uint i = 0; i < elems_per_lane; i++) {
        v_acc[i] = 0.0f;
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (uint pos = kv_start; pos < kv_end; pos++) {
        uint blk_idx = pos / p.block_size;
        uint blk_off = pos % p.block_size;
        uint phys_block = uint(bt_row[blk_idx]);
        uint kv_off = (phys_block * p.block_size + blk_off) * kv_stride + kv_h * hd;

        float partial_dot = 0.0f;
        for (uint i = 0; i < elems_per_lane; i++) {
            float k_val = lane_active ? K_pool[kv_off + lane_start + i] : 0.0f;
            partial_dot += q_reg[i] * k_val;
        }
        float score = simd_sum(partial_dot) * p.scale;

        if (p.softcap > 0.0f) {
            score = p.softcap * precise::tanh(score / p.softcap);
        }

        float old_max = running_max;
        running_max = max(running_max, score);
        float correction = exp(old_max - running_max);
        float exp_score = exp(score - running_max);

        running_sum = running_sum * correction + exp_score;
        for (uint i = 0; i < elems_per_lane; i++) {
            v_acc[i] = v_acc[i] * correction;
        }

        if (lane_active) {
            for (uint i = 0; i < elems_per_lane; i++) {
                float v_val = V_pool[kv_off + lane_start + i];
                v_acc[i] += exp_score * v_val;
            }
        }
    }

    if (lane_active) {
        const float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
        const uint out_base = (b * p.n_heads + h) * hd;
        for (uint i = 0; i < elems_per_lane; i++) {
            output[out_base + lane_start + i] = v_acc[i] * inv_sum;
        }
    }
}
