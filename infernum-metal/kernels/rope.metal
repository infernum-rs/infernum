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

struct RopeQkParams {
    uint q_heads;
    uint k_heads;
    uint head_dim;
    uint half_dim;
};

/// Fused Q+K RoPE: apply RoPE to both Q and K in a single dispatch.
///
/// Thread grid: batch_size * (q_heads + k_heads) * half_dim, 1D.
/// Threads in [0, batch*q_heads*half_dim) operate on Q.
/// Remaining threads operate on K.
kernel void apply_rope_qk_batched_f32(
    device const float* q_input     [[buffer(0)]],
    device const float* k_input     [[buffer(1)]],
    device const float* cos_cache   [[buffer(2)]],
    device const float* sin_cache   [[buffer(3)]],
    device const int*   positions   [[buffer(4)]],
    device float* q_output          [[buffer(5)]],
    device float* k_output          [[buffer(6)]],
    constant RopeQkParams& params   [[buffer(7)]],
    uint tid                        [[thread_position_in_grid]])
{
    const uint q_heads  = params.q_heads;
    const uint k_heads  = params.k_heads;
    const uint head_dim = params.head_dim;
    const uint half_dim = params.half_dim;

    const uint q_stride = q_heads * half_dim;   // pairs per batch element for Q
    const uint total_per_batch = q_stride + k_heads * half_dim;

    const uint batch_idx = tid / total_per_batch;
    const uint remainder = tid % total_per_batch;

    const uint pos = uint(positions[batch_idx]);

    // Determine if this thread processes Q or K
    device const float* src;
    device float* dst;
    uint n_heads;
    uint local_idx;

    if (remainder < q_stride) {
        src       = q_input;
        dst       = q_output;
        n_heads   = q_heads;
        local_idx = remainder;
    } else {
        src       = k_input;
        dst       = k_output;
        n_heads   = k_heads;
        local_idx = remainder - q_stride;
    }

    const uint p_idx = local_idx % half_dim;
    const uint h     = local_idx / half_dim;
    const uint base  = (batch_idx * n_heads + h) * head_dim;

    float cv = cos_cache[pos * half_dim + p_idx];
    float sv = sin_cache[pos * half_dim + p_idx];

    float a = src[base + p_idx];
    float b = src[base + half_dim + p_idx];

    dst[base + p_idx]            = a * cv - b * sv;
    dst[base + half_dim + p_idx] = b * cv + a * sv;
}


/// Fused Q+K RoPE f16: activations half, cos/sin cache stays float.
kernel void apply_rope_qk_batched_f16(
    device const half* q_input      [[buffer(0)]],
    device const half* k_input      [[buffer(1)]],
    device const float* cos_cache   [[buffer(2)]],
    device const float* sin_cache   [[buffer(3)]],
    device const int*   positions   [[buffer(4)]],
    device half* q_output           [[buffer(5)]],
    device half* k_output           [[buffer(6)]],
    constant RopeQkParams& params   [[buffer(7)]],
    uint tid                        [[thread_position_in_grid]])
{
    const uint q_heads  = params.q_heads;
    const uint k_heads  = params.k_heads;
    const uint head_dim = params.head_dim;
    const uint half_dim = params.half_dim;

    const uint q_stride = q_heads * half_dim;
    const uint total_per_batch = q_stride + k_heads * half_dim;

    const uint batch_idx = tid / total_per_batch;
    const uint remainder = tid % total_per_batch;

    const uint pos = uint(positions[batch_idx]);

    device const half* src;
    device half* dst;
    uint n_heads;
    uint local_idx;

    if (remainder < q_stride) {
        src       = q_input;
        dst       = q_output;
        n_heads   = q_heads;
        local_idx = remainder;
    } else {
        src       = k_input;
        dst       = k_output;
        n_heads   = k_heads;
        local_idx = remainder - q_stride;
    }

    const uint p_idx = local_idx % half_dim;
    const uint h     = local_idx / half_dim;
    const uint base  = (batch_idx * n_heads + h) * head_dim;

    float cv = cos_cache[pos * half_dim + p_idx];
    float sv = sin_cache[pos * half_dim + p_idx];

    float a = float(src[base + p_idx]);
    float b = float(src[base + half_dim + p_idx]);

    dst[base + p_idx]            = half(a * cv - b * sv);
    dst[base + half_dim + p_idx] = half(b * cv + a * sv);
}

/// Standard RoPE f16.
kernel void apply_rope_f16(
    device const half* input        [[buffer(0)]],
    device const float* cos_cache   [[buffer(1)]],
    device const float* sin_cache   [[buffer(2)]],
    device half* output             [[buffer(3)]],
    constant RopeParams& params     [[buffer(4)]],
    uint tid                        [[thread_position_in_grid]])
{
    const uint n_heads  = params.n_heads;
    const uint head_dim = params.head_dim;
    const uint half_dim = params.half_dim;
    const uint pos_offset = params.pos_offset;

    const uint pair_idx = tid % half_dim;
    const uint head     = (tid / half_dim) % n_heads;
    const uint seq_pos  = tid / (half_dim * n_heads);

    const uint pos = seq_pos + pos_offset;
    const uint base = (seq_pos * n_heads + head) * head_dim;

    float cos_val = cos_cache[pos * half_dim + pair_idx];
    float sin_val = sin_cache[pos * half_dim + pair_idx];

    float x0 = float(input[base + pair_idx]);
    float x1 = float(input[base + half_dim + pair_idx]);

    output[base + pair_idx]            = half(x0 * cos_val - x1 * sin_val);
    output[base + half_dim + pair_idx] = half(x1 * cos_val + x0 * sin_val);
}
