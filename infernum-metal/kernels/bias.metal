#include <metal_stdlib>
using namespace metal;

/// Row-wise bias add in-place: input[i] += bias[i % cols].
kernel void bias_add_inplace_f32(
    device float* input         [[buffer(0)]],
    device const float* bias    [[buffer(1)]],
    constant uint& cols         [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]])
{
    input[tid] += bias[tid % cols];
}
