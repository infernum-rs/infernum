#include <metal_stdlib>
using namespace metal;

struct RopeInterleavedParams {
    uint n_heads;
    uint head_dim;
    uint n_pairs;
    uint pos_offset;
};

/// Interleaved RoPE: pairs at (2p, 2p+1).
/// Thread grid: total_pairs = seq_len * n_heads * n_pairs, 1D dispatch.
kernel void apply_rope_interleaved_f32(
    device const float* input       [[buffer(0)]],
    device const float* cos_cache   [[buffer(1)]],
    device const float* sin_cache   [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant RopeInterleavedParams& params [[buffer(4)]],
    uint tid                        [[thread_position_in_grid]])
{
    const uint n_heads   = params.n_heads;
    const uint head_dim  = params.head_dim;
    const uint n_pairs   = params.n_pairs;
    const uint pos_offset = params.pos_offset;

    const uint pair_idx = tid % n_pairs;
    const uint head     = (tid / n_pairs) % n_heads;
    const uint seq_pos  = tid / (n_pairs * n_heads);

    const uint pos = seq_pos + pos_offset;
    const uint base = (seq_pos * n_heads + head) * head_dim;

    float cos_val = cos_cache[pos * n_pairs + pair_idx];
    float sin_val = sin_cache[pos * n_pairs + pair_idx];

    float x0 = input[base + 2 * pair_idx];
    float x1 = input[base + 2 * pair_idx + 1];

    output[base + 2 * pair_idx]     = x0 * cos_val - x1 * sin_val;
    output[base + 2 * pair_idx + 1] = x1 * cos_val + x0 * sin_val;
}
