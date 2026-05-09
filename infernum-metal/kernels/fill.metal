#include <metal_stdlib>
using namespace metal;

/// Fill every element of `out` with `value`.
kernel void fill_f32(
    device float* out       [[buffer(0)]],
    constant float& value   [[buffer(1)]],
    uint tid                [[thread_position_in_grid]])
{
    out[tid] = value;
}
