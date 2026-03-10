#include <metal_stdlib>
using namespace metal;

/// Elementwise multiply: out[i] = a[i] * b[i].
kernel void mul_f32(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float* out       [[buffer(2)]],
    uint tid                [[thread_position_in_grid]])
{
    out[tid] = a[tid] * b[tid];
}
kernel void mul_f16(
    device const half* a    [[buffer(0)]],
    device const half* b    [[buffer(1)]],
    device half* out        [[buffer(2)]],
    uint tid                [[thread_position_in_grid]])
{
    out[tid] = a[tid] * b[tid];
}

