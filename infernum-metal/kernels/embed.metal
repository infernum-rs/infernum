#include <metal_stdlib>
using namespace metal;

/// Embedding gather: out[i*hidden + j] = table[indices[i]*hidden + j].
/// Thread grid: n_tokens * hidden, 1D dispatch.
kernel void embedding_gather_f32(
    device const float* table       [[buffer(0)]],
    device const uint*  indices     [[buffer(1)]],
    device float* output            [[buffer(2)]],
    constant uint& hidden           [[buffer(3)]],
    uint tid                        [[thread_position_in_grid]])
{
    const uint token = tid / hidden;
    const uint col   = tid % hidden;
    output[tid] = table[indices[token] * hidden + col];
}
