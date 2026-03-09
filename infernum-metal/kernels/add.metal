#include <metal_stdlib>
using namespace metal;

/// Elementwise add: out[i] = a[i] + b[i].
kernel void add_f32(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float* out       [[buffer(2)]],
    uint tid                [[thread_position_in_grid]])
{
    out[tid] = a[tid] + b[tid];
}

/// Elementwise add in-place: a[i] += b[i].
kernel void add_inplace_f32(
    device float* a         [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    uint tid                [[thread_position_in_grid]])
{
    a[tid] += b[tid];
}
