#include <metal_stdlib>
using namespace metal;

struct RMSNormParams {
    uint hidden;
    float eps;
};

/// RMS normalization: out[row,i] = (input[row,i] / rms) * weight[i]
/// where rms = sqrt(mean(input[row,:]^2) + eps).
///
/// Dispatch: threadgroups = rows, threads_per_threadgroup = power-of-2 <= hidden.
/// Each threadgroup handles one row.
kernel void rms_norm_f32(
    device const float* input       [[buffer(0)]],
    device const float* weight      [[buffer(1)]],
    device float* output            [[buffer(2)]],
    constant RMSNormParams& params  [[buffer(3)]],
    uint row                        [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tg_size                    [[threads_per_threadgroup]],
    threadgroup float* shared       [[threadgroup(0)]])
{
    const uint hidden = params.hidden;
    const float eps = params.eps;

    // Accumulate sum of squares for this row
    float sum_sq = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        float v = input[row * hidden + i];
        sum_sq += v * v;
    }

    // Store partial sum in shared memory
    shared[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction in shared memory
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute scale: 1 / sqrt(mean_sq + eps)
    float scale = rsqrt(shared[0] / float(hidden) + eps);

    // Apply normalization and weight
    for (uint i = tid; i < hidden; i += tg_size) {
        output[row * hidden + i] = input[row * hidden + i] * scale * weight[i];
    }
}

/// Fused add + RMS normalization.
/// updated[row,i] = residual[row,i] + input[row,i]
/// normed[row,i]  = (updated[row,i] / rms) * weight[i]
///
/// Dispatch: threadgroups = rows, threads_per_threadgroup = power-of-2 <= hidden.
kernel void add_rmsnorm_f32(
    device const float* residual    [[buffer(0)]],
    device const float* input       [[buffer(1)]],
    device const float* weight      [[buffer(2)]],
    device float* updated           [[buffer(3)]],
    device float* normed            [[buffer(4)]],
    constant RMSNormParams& params  [[buffer(5)]],
    uint row                        [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tg_size                    [[threads_per_threadgroup]],
    threadgroup float* shared       [[threadgroup(0)]])
{
    const uint hidden = params.hidden;
    const float eps = params.eps;

    // Compute updated = residual + input, and accumulate sum of squares
    float sum_sq = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        uint idx = row * hidden + i;
        float u = residual[idx] + input[idx];
        updated[idx] = u;
        sum_sq += u * u;
    }

    // Shared memory reduction
    shared[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = rsqrt(shared[0] / float(hidden) + eps);

    // Apply normalization with weight
    for (uint i = tid; i < hidden; i += tg_size) {
        uint idx = row * hidden + i;
        normed[idx] = updated[idx] * scale * weight[i];
    }
}


/// RMS normalization f16: activations are half, norm weight is float.
/// out[row,i] = half((float(input[row,i]) / rms) * weight[i])
kernel void rms_norm_f16(
    device const half* input        [[buffer(0)]],
    device const float* weight      [[buffer(1)]],
    device half* output             [[buffer(2)]],
    constant RMSNormParams& params  [[buffer(3)]],
    uint row                        [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tg_size                    [[threads_per_threadgroup]],
    threadgroup float* shared       [[threadgroup(0)]])
{
    const uint hidden = params.hidden;
    const float eps = params.eps;

    float sum_sq = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        float v = float(input[row * hidden + i]);
        sum_sq += v * v;
    }

    shared[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = rsqrt(shared[0] / float(hidden) + eps);

    for (uint i = tid; i < hidden; i += tg_size) {
        output[row * hidden + i] = half(float(input[row * hidden + i]) * scale * weight[i]);
    }
}

/// Fused add + RMS normalization f16.
/// updated[row,i] = residual[row,i] + input[row,i]
/// normed[row,i]  = half((float(updated) / rms) * weight[i])
kernel void add_rmsnorm_f16(
    device const half* residual     [[buffer(0)]],
    device const half* input        [[buffer(1)]],
    device const float* weight      [[buffer(2)]],
    device half* updated            [[buffer(3)]],
    device half* normed             [[buffer(4)]],
    constant RMSNormParams& params  [[buffer(5)]],
    uint row                        [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tg_size                    [[threads_per_threadgroup]],
    threadgroup float* shared       [[threadgroup(0)]])
{
    const uint hidden = params.hidden;
    const float eps = params.eps;

    float sum_sq = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        uint idx = row * hidden + i;
        float u = float(residual[idx]) + float(input[idx]);
        updated[idx] = half(u);
        sum_sq += u * u;
    }

    shared[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = rsqrt(shared[0] / float(hidden) + eps);

    for (uint i = tid; i < hidden; i += tg_size) {
        uint idx = row * hidden + i;
        normed[idx] = half(float(updated[idx]) * scale * weight[i]);
    }
}
