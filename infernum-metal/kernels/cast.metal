#include <metal_stdlib>
using namespace metal;

/// Cast bf16 → f32, elementwise.
kernel void cast_bf16_to_f32(
    device const bfloat* input  [[buffer(0)]],
    device float* output        [[buffer(1)]],
    uint tid                    [[thread_position_in_grid]])
{
    output[tid] = float(input[tid]);
}

/// Cast f16 → f32, elementwise.
kernel void cast_f16_to_f32(
    device const half* input    [[buffer(0)]],
    device float* output        [[buffer(1)]],
    uint tid                    [[thread_position_in_grid]])
{
    output[tid] = float(input[tid]);
}
