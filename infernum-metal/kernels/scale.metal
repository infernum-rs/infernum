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
kernel void scale_inplace_f16(
    device half* a              [[buffer(0)]],
    constant float& scale       [[buffer(1)]],
    uint tid                    [[thread_position_in_grid]])
{
    a[tid] = half(float(a[tid]) * scale);
}

/// Elementwise multiply f16: out[i] = a[i] * b[i].
