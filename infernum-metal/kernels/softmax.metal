#include <metal_stdlib>
using namespace metal;

struct SoftmaxParams {
    uint rows;
    uint cols;
};

/// Row-wise softmax: for each row, compute max, exp(x-max), sum, normalize.
/// Dispatch: one threadgroup per row. Threadgroup size = cols (capped to 1024).
/// Uses threadgroup shared memory for max/sum reductions.
kernel void softmax_f32(
    device const float* input           [[buffer(0)]],
    device float* output                [[buffer(1)]],
    constant SoftmaxParams& params      [[buffer(2)]],
    threadgroup float* shared           [[threadgroup(0)]],
    uint gid                            [[threadgroup_position_in_grid]],
    uint lid                            [[thread_position_in_threadgroup]],
    uint tg_size                        [[threads_per_threadgroup]])
{
    const uint row = gid;
    const uint cols = params.cols;
    const uint row_offset = row * cols;

    // Step 1: Find row max (parallel reduction)
    float local_max = -INFINITY;
    for (uint i = lid; i < cols; i += tg_size) {
        local_max = max(local_max, input[row_offset + i]);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] = max(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = lid; i < cols; i += tg_size) {
        float val = exp(input[row_offset + i] - row_max);
        output[row_offset + i] = val;
        local_sum += val;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_sum = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Normalize
    const float inv_sum = 1.0f / row_sum;
    for (uint i = lid; i < cols; i += tg_size) {
        output[row_offset + i] *= inv_sum;
    }
}

struct CausalSoftmaxParams {
    uint rows;
    uint cols;
    uint offset;   // starting position of the first query in the sequence
};

/// Causal softmax: same as softmax but masks future positions.
/// For query at position `offset + row_idx`, only attend to keys [0..offset+row_idx].
/// Future positions are set to -INFINITY before softmax.
kernel void causal_softmax_f32(
    device const float* input               [[buffer(0)]],
    device float* output                    [[buffer(1)]],
    constant CausalSoftmaxParams& params    [[buffer(2)]],
    threadgroup float* shared               [[threadgroup(0)]],
    uint gid                                [[threadgroup_position_in_grid]],
    uint lid                                [[thread_position_in_threadgroup]],
    uint tg_size                            [[threads_per_threadgroup]])
{
    const uint row = gid;
    const uint cols = params.cols;
    const uint offset = params.offset;
    const uint row_offset = row * cols;
    // Query at position (offset + row) can attend to keys [0..offset+row]
    const uint max_key = offset + row + 1;

    // Step 1: Find row max with causal mask
    float local_max = -INFINITY;
    for (uint i = lid; i < cols; i += tg_size) {
        float val = (i < max_key) ? input[row_offset + i] : -INFINITY;
        local_max = max(local_max, val);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] = max(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: exp(x - max) with mask, and sum
    float local_sum = 0.0f;
    for (uint i = lid; i < cols; i += tg_size) {
        float val = 0.0f;
        if (i < max_key) {
            val = exp(input[row_offset + i] - row_max);
        }
        output[row_offset + i] = val;
        local_sum += val;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_sum = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Normalize
    const float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (uint i = lid; i < cols; i += tg_size) {
        output[row_offset + i] *= inv_sum;
    }
}
