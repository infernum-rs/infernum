#include <metal_stdlib>
using namespace metal;

struct RopeParams {
    uint n_heads;
    uint head_dim;
    uint half_dim;
    uint pos_offset;
};

/// Standard RoPE: pairs at (d, d+half_dim).
/// Thread grid: total_pairs = seq_len * n_heads * half_dim, 1D dispatch.
kernel void apply_rope_f32(
    device const float* input       [[buffer(0)]],
    device const float* cos_cache   [[buffer(1)]],
    device const float* sin_cache   [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant RopeParams& params     [[buffer(4)]],
    uint tid                        [[thread_position_in_grid]])
{
    const uint n_heads  = params.n_heads;
    const uint head_dim = params.head_dim;
    const uint half_dim = params.half_dim;
    const uint pos_offset = params.pos_offset;

    // Decompose tid into (seq_pos, head, pair_idx)
    const uint pair_idx = tid % half_dim;
    const uint head     = (tid / half_dim) % n_heads;
    const uint seq_pos  = tid / (half_dim * n_heads);

    const uint pos = seq_pos + pos_offset;
    const uint base = (seq_pos * n_heads + head) * head_dim;

    float cos_val = cos_cache[pos * half_dim + pair_idx];
    float sin_val = sin_cache[pos * half_dim + pair_idx];

    float x0 = input[base + pair_idx];
    float x1 = input[base + half_dim + pair_idx];

    output[base + pair_idx]            = x0 * cos_val - x1 * sin_val;
    output[base + half_dim + pair_idx] = x1 * cos_val + x0 * sin_val;
}

/// Batched RoPE with per-token positions from a position tensor.
/// Thread grid: total_pairs = batch_size * n_heads * half_dim, 1D.
kernel void apply_rope_batched_f32(
    device const float* input       [[buffer(0)]],
    device const float* cos_cache   [[buffer(1)]],
    device const float* sin_cache   [[buffer(2)]],
    device const int*   positions   [[buffer(3)]],
    device float* output            [[buffer(4)]],
    constant RopeParams& params     [[buffer(5)]],
    uint tid                        [[thread_position_in_grid]])
{
    const uint n_heads  = params.n_heads;
    const uint head_dim = params.head_dim;
    const uint half_dim = params.half_dim;

    const uint pair_idx = tid % half_dim;
    const uint head     = (tid / half_dim) % n_heads;
    const uint batch    = tid / (half_dim * n_heads);

    const uint pos = uint(positions[batch]);
    const uint base = (batch * n_heads + head) * head_dim;

    float cos_val = cos_cache[pos * half_dim + pair_idx];
    float sin_val = sin_cache[pos * half_dim + pair_idx];

    float x0 = input[base + pair_idx];
    float x1 = input[base + half_dim + pair_idx];

    output[base + pair_idx]            = x0 * cos_val - x1 * sin_val;
    output[base + half_dim + pair_idx] = x1 * cos_val + x0 * sin_val;
}
