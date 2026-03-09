#include <metal_stdlib>
using namespace metal;

struct PagedAttentionDecodeParams {
    uint batch_size;
    uint n_heads;           // query heads
    uint kv_heads;          // KV heads (for GQA)
    uint head_dim;
    uint block_size;        // tokens per block
    uint max_blocks_per_seq;
    float scale;
    float softcap;          // 0.0 = disabled
    int sliding_window;     // -1 = disabled
};

/// Paged attention decode: single query per batch element, KV stored in paged blocks.
///
/// Dispatch: one threadgroup per (batch, head) pair.
/// Grid: (batch_size * n_heads) threadgroups.
/// Each thread strides over KV positions for dot product + softmax + V accum.
///
/// Buffers:
///   0: Q            (batch_size, n_heads, head_dim)
///   1: K_pool        (num_blocks * block_size, kv_heads, head_dim)
///   2: V_pool        (num_blocks * block_size, kv_heads, head_dim)
///   3: block_tables  (batch_size, max_blocks_per_seq) — int32
///   4: seq_lens      (batch_size) — int32
///   5: output        (batch_size, n_heads, head_dim)
///   6: params
kernel void paged_attention_decode_f32(
    device const float*  Q            [[buffer(0)]],
    device const float*  K_pool       [[buffer(1)]],
    device const float*  V_pool       [[buffer(2)]],
    device const int*    block_tables [[buffer(3)]],
    device const int*    seq_lens     [[buffer(4)]],
    device float*        output       [[buffer(5)]],
    constant PagedAttentionDecodeParams& p [[buffer(6)]],
    threadgroup float*   shared       [[threadgroup(0)]],
    uint gid                          [[threadgroup_position_in_grid]],
    uint lid                          [[thread_position_in_threadgroup]],
    uint tg_size                      [[threads_per_threadgroup]])
{
    const uint b       = gid / p.n_heads;
    const uint h       = gid % p.n_heads;
    const uint kv_h    = h / (p.n_heads / p.kv_heads);
    const uint kv_stride = p.kv_heads * p.head_dim;

    const int seq_len = seq_lens[b];
    if (seq_len <= 0) return;
    const uint sl = uint(seq_len);

    const uint q_pos = sl - 1; // decode query position

    // Sliding window bounds
    uint kv_start = 0;
    uint kv_end = sl;
    if (p.sliding_window > 0) {
        uint window = uint(p.sliding_window);
        if (q_pos >= window) {
            kv_start = q_pos - window + 1;
        }
    }

    const uint q_base = (b * p.n_heads + h) * p.head_dim;
    device const int* bt_row = block_tables + b * p.max_blocks_per_seq;

    // === Pass 1: Find max score ===
    float local_max = -INFINITY;
    for (uint pos = kv_start + lid; pos < kv_end; pos += tg_size) {
        uint blk_idx = pos / p.block_size;
        uint blk_off = pos % p.block_size;
        uint phys_block = uint(bt_row[blk_idx]);
        uint k_off = (phys_block * p.block_size + blk_off) * kv_stride + kv_h * p.head_dim;

        float dot = 0.0f;
        for (uint d = 0; d < p.head_dim; d++) {
            dot += Q[q_base + d] * K_pool[k_off + d];
        }
        float score = dot * p.scale;
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
    for (uint pos = kv_start + lid; pos < kv_end; pos += tg_size) {
        uint blk_idx = pos / p.block_size;
        uint blk_off = pos % p.block_size;
        uint phys_block = uint(bt_row[blk_idx]);
        uint k_off = (phys_block * p.block_size + blk_off) * kv_stride + kv_h * p.head_dim;

        float dot = 0.0f;
        for (uint d = 0; d < p.head_dim; d++) {
            dot += Q[q_base + d] * K_pool[k_off + d];
        }
        float score = dot * p.scale;
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
    const uint out_base = (b * p.n_heads + h) * p.head_dim;

    for (uint d = lid; d < p.head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint pos = kv_start; pos < kv_end; pos++) {
            uint blk_idx = pos / p.block_size;
            uint blk_off = pos % p.block_size;
            uint phys_block = uint(bt_row[blk_idx]);
            uint k_off = (phys_block * p.block_size + blk_off) * kv_stride + kv_h * p.head_dim;

            float dot = 0.0f;
            for (uint dd = 0; dd < p.head_dim; dd++) {
                dot += Q[q_base + dd] * K_pool[k_off + dd];
            }
            float score = dot * p.scale;
            if (p.softcap > 0.0f) {
                score = p.softcap * precise::tanh(score / p.softcap);
            }
            float weight = exp(score - global_max) * inv_sum;

            uint v_off = (phys_block * p.block_size + blk_off) * kv_stride + kv_h * p.head_dim;
            acc += weight * V_pool[v_off + d];
        }
        output[out_base + d] = acc;
    }
}
