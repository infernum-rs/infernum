#include <metal_stdlib>
using namespace metal;

struct RMSNormParams {
    uint hidden;
    float eps;
};

// -------------------------------------------------------------------
// Shared reduction helper: 2 barriers via simd_sum + shmem handoff.
//
// shmem must have at least ceil(tg_size / 32) floats allocated.
// Returns the total sum across all threads in the threadgroup.
// -------------------------------------------------------------------
static inline float tg_sum_reduce(
    float partial,
    threadgroup float* shmem,
    uint tiisg,   // thread index in simdgroup  (0–31)
    uint sgitg,   // simdgroup index in threadgroup
    uint tg_size)
{
    // Reduce within each SIMD group (no barrier needed)
    partial = simd_sum(partial);

    // One thread per SIMD group writes to shmem
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shmem[sgitg] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All threads sum the SIMD-group partials — at most 8 values (256/32)
    uint num_sg = max(1u, tg_size / 32u);
    float total = 0.0f;
    for (uint i = 0; i < num_sg; i++) {
        total += shmem[i];
    }
    return total;
}

// -------------------------------------------------------------------
// rms_norm_f32 — float activations, float weights.
// Uses float4 vectorised loads when hidden % 4 == 0 (fast path).
// -------------------------------------------------------------------
kernel void rms_norm_f32(
    device const float* input       [[buffer(0)]],
    device const float* weight      [[buffer(1)]],
    device float* output            [[buffer(2)]],
    constant RMSNormParams& params  [[buffer(3)]],
    uint row                        [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint tiisg                      [[thread_index_in_simdgroup]],
    uint sgitg                      [[simdgroup_index_in_threadgroup]],
    threadgroup float* shmem        [[threadgroup(0)]])
{
    const uint hidden = params.hidden;
    const float eps = params.eps;
    const uint base = row * hidden;

    float sum_sq = 0.0f;

    if (hidden % 4 == 0) {
        // Vectorised path: 4 elements per iteration
        const uint hidden4 = hidden / 4;
        device const float4* inp4 = (device const float4*)(input + base);
        for (uint i = tid; i < hidden4; i += tg_size) {
            float4 v = inp4[i];
            sum_sq += dot(v, v);
        }
    } else {
        for (uint i = tid; i < hidden; i += tg_size) {
            float v = input[base + i];
            sum_sq += v * v;
        }
    }

    float total = tg_sum_reduce(sum_sq, shmem, tiisg, sgitg, tg_size);
    float scale = rsqrt(total / float(hidden) + eps);

    if (hidden % 4 == 0) {
        const uint hidden4 = hidden / 4;
        device const float4* inp4  = (device const float4*)(input  + base);
        device const float4* wgt4  = (device const float4*)weight;
        device float4*       out4  = (device float4*)(output + base);
        for (uint i = tid; i < hidden4; i += tg_size) {
            out4[i] = inp4[i] * scale * wgt4[i];
        }
    } else {
        for (uint i = tid; i < hidden; i += tg_size) {
            output[base + i] = input[base + i] * scale * weight[i];
        }
    }
}

// -------------------------------------------------------------------
// add_rmsnorm_f32 — fused residual-add + RMSNorm.
// -------------------------------------------------------------------
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
    uint tiisg                      [[thread_index_in_simdgroup]],
    uint sgitg                      [[simdgroup_index_in_threadgroup]],
    threadgroup float* shmem        [[threadgroup(0)]])
{
    const uint hidden = params.hidden;
    const float eps = params.eps;
    const uint base = row * hidden;

    float sum_sq = 0.0f;

    if (hidden % 4 == 0) {
        const uint hidden4 = hidden / 4;
        device const float4* res4 = (device const float4*)(residual + base);
        device const float4* inp4 = (device const float4*)(input    + base);
        device float4*       upd4 = (device float4*)(updated + base);
        for (uint i = tid; i < hidden4; i += tg_size) {
            float4 u = res4[i] + inp4[i];
            upd4[i] = u;
            sum_sq += dot(u, u);
        }
    } else {
        for (uint i = tid; i < hidden; i += tg_size) {
            uint idx = base + i;
            float u = residual[idx] + input[idx];
            updated[idx] = u;
            sum_sq += u * u;
        }
    }

    float total = tg_sum_reduce(sum_sq, shmem, tiisg, sgitg, tg_size);
    float scale = rsqrt(total / float(hidden) + eps);

    if (hidden % 4 == 0) {
        const uint hidden4 = hidden / 4;
        device const float4* upd4 = (device const float4*)(updated + base);
        device const float4* wgt4 = (device const float4*)weight;
        device float4*       nrm4 = (device float4*)(normed + base);
        for (uint i = tid; i < hidden4; i += tg_size) {
            nrm4[i] = upd4[i] * scale * wgt4[i];
        }
    } else {
        for (uint i = tid; i < hidden; i += tg_size) {
            uint idx = base + i;
            normed[idx] = updated[idx] * scale * weight[i];
        }
    }
}

// -------------------------------------------------------------------
// rms_norm_f16 — half activations, float weights.
// -------------------------------------------------------------------
kernel void rms_norm_f16(
    device const half* input        [[buffer(0)]],
    device const float* weight      [[buffer(1)]],
    device half* output             [[buffer(2)]],
    constant RMSNormParams& params  [[buffer(3)]],
    uint row                        [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint tiisg                      [[thread_index_in_simdgroup]],
    uint sgitg                      [[simdgroup_index_in_threadgroup]],
    threadgroup float* shmem        [[threadgroup(0)]])
{
    const uint hidden = params.hidden;
    const float eps = params.eps;
    const uint base = row * hidden;

    float sum_sq = 0.0f;

    if (hidden % 4 == 0) {
        const uint hidden4 = hidden / 4;
        device const half4* inp4 = (device const half4*)(input + base);
        for (uint i = tid; i < hidden4; i += tg_size) {
            float4 v = float4(inp4[i]);
            sum_sq += dot(v, v);
        }
    } else {
        for (uint i = tid; i < hidden; i += tg_size) {
            float v = float(input[base + i]);
            sum_sq += v * v;
        }
    }

    float total = tg_sum_reduce(sum_sq, shmem, tiisg, sgitg, tg_size);
    float scale = rsqrt(total / float(hidden) + eps);

    if (hidden % 4 == 0) {
        const uint hidden4 = hidden / 4;
        device const half4*  inp4 = (device const half4*)(input  + base);
        device const float4* wgt4 = (device const float4*)weight;
        device half4*        out4 = (device half4*)(output + base);
        for (uint i = tid; i < hidden4; i += tg_size) {
            out4[i] = half4(float4(inp4[i]) * scale * wgt4[i]);
        }
    } else {
        for (uint i = tid; i < hidden; i += tg_size) {
            output[base + i] = half(float(input[base + i]) * scale * weight[i]);
        }
    }
}

// -------------------------------------------------------------------
// add_rmsnorm_f16 — fused residual-add + RMSNorm, half activations.
// -------------------------------------------------------------------
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
    uint tiisg                      [[thread_index_in_simdgroup]],
    uint sgitg                      [[simdgroup_index_in_threadgroup]],
    threadgroup float* shmem        [[threadgroup(0)]])
{
    const uint hidden = params.hidden;
    const float eps = params.eps;
    const uint base = row * hidden;

    float sum_sq = 0.0f;

    if (hidden % 4 == 0) {
        const uint hidden4 = hidden / 4;
        device const half4* res4 = (device const half4*)(residual + base);
        device const half4* inp4 = (device const half4*)(input    + base);
        device half4*       upd4 = (device half4*)(updated + base);
        for (uint i = tid; i < hidden4; i += tg_size) {
            float4 u = float4(res4[i]) + float4(inp4[i]);
            upd4[i] = half4(u);
            sum_sq += dot(u, u);
        }
    } else {
        for (uint i = tid; i < hidden; i += tg_size) {
            uint idx = base + i;
            float u = float(residual[idx]) + float(input[idx]);
            updated[idx] = half(u);
            sum_sq += u * u;
        }
    }

    float total = tg_sum_reduce(sum_sq, shmem, tiisg, sgitg, tg_size);
    float scale = rsqrt(total / float(hidden) + eps);

    if (hidden % 4 == 0) {
        const uint hidden4 = hidden / 4;
        device const half4*  upd4 = (device const half4*)(updated + base);
        device const float4* wgt4 = (device const float4*)weight;
        device half4*        nrm4 = (device half4*)(normed + base);
        for (uint i = tid; i < hidden4; i += tg_size) {
            nrm4[i] = half4(float4(upd4[i]) * scale * wgt4[i]);
        }
    } else {
        for (uint i = tid; i < hidden; i += tg_size) {
            uint idx = base + i;
            normed[idx] = half(float(updated[idx]) * scale * weight[i]);
        }
    }
}
