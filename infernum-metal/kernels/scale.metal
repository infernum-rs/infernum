#include <metal_stdlib>
using namespace metal;

/// Scale in-place: a[i] *= scale.
kernel void scale_inplace_f32(
    device float* a             [[buffer(0)]],
    constant float& scale       [[buffer(1)]],
    uint tid                    [[thread_position_in_grid]])
{
    a[tid] *= scale;
}
