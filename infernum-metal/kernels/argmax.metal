#include <metal_stdlib>
using namespace metal;

/// Argmax reduction: find the index of the maximum value in a 1D array.
/// Dispatch: single threadgroup, threads_per_group = power-of-2 ≤ 1024.
/// Uses shared memory for parallel reduction of (index, value) pairs.
kernel void argmax_f32(
    device const float* input       [[buffer(0)]],
    device uint* output             [[buffer(1)]],
    constant uint& n                [[buffer(2)]],
    threadgroup float* shared_val   [[threadgroup(0)]],
    uint lid                        [[thread_position_in_threadgroup]],
    uint tg_size                    [[threads_per_threadgroup]])
{
    // Each thread finds local max over its strided range
    float local_max = -INFINITY;
    uint local_idx = 0;
    for (uint i = lid; i < n; i += tg_size) {
        float val = input[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    // Store in shared memory — interleave val and idx
    // Layout: [val_0, val_1, ..., val_tg-1, idx_0, idx_1, ..., idx_tg-1]
    threadgroup uint* shared_idx = reinterpret_cast<threadgroup uint*>(shared_val + tg_size);

    shared_val[lid] = local_max;
    shared_idx[lid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (shared_val[lid + stride] > shared_val[lid]) {
                shared_val[lid] = shared_val[lid + stride];
                shared_idx[lid] = shared_idx[lid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        output[0] = shared_idx[0];
    }
}
