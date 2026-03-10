#include <metal_stdlib>
using namespace metal;

/// SiLU(x) = x / (1 + exp(-x)).
static inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

/// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))).
static inline float gelu(float x) {
    const float c = sqrt(2.0f / M_PI_F);
    return 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x * x * x)));
}

/// SwiGLU: out[i] = silu(gate[i]) * up[i].
kernel void swiglu_f32(
    device const float* gate    [[buffer(0)]],
    device const float* up      [[buffer(1)]],
    device float* out           [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]])
{
    out[tid] = silu(gate[tid]) * up[tid];
}

/// GeGLU: out[i] = gelu(gate[i]) * up[i].
kernel void geglu_f32(
    device const float* gate    [[buffer(0)]],
    device const float* up      [[buffer(1)]],
    device float* out           [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]])
{
    out[tid] = gelu(gate[tid]) * up[tid];
}


/// SwiGLU f16: out[i] = silu(gate[i]) * up[i].
kernel void swiglu_f16(
    device const half* gate    [[buffer(0)]],
    device const half* up      [[buffer(1)]],
    device half* out           [[buffer(2)]],
    uint tid                   [[thread_position_in_grid]])
{
    float g = float(gate[tid]);
    out[tid] = half(silu(g) * float(up[tid]));
}

/// GeGLU f16: out[i] = gelu(gate[i]) * up[i].
kernel void geglu_f16(
    device const half* gate    [[buffer(0)]],
    device const half* up      [[buffer(1)]],
    device half* out           [[buffer(2)]],
    uint tid                   [[thread_position_in_grid]])
{
    float g = float(gate[tid]);
    out[tid] = half(gelu(g) * float(up[tid]));
}
