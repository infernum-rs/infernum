#include <metal_stdlib>
using namespace metal;

struct TransposeParams {
    uint rows;
    uint cols;
};

/// 2D transpose: out[col * rows + row] = input[row * cols + col].
/// Thread grid: rows * cols, 1D dispatch.
kernel void transpose_2d_f32(
    device const float* input           [[buffer(0)]],
    device float* output                [[buffer(1)]],
    constant TransposeParams& params    [[buffer(2)]],
    uint tid                            [[thread_position_in_grid]])
{
    const uint row = tid / params.cols;
    const uint col = tid % params.cols;
    output[col * params.rows + row] = input[row * params.cols + col];
}

struct RepeatKvParams {
    uint seq;
    uint kv_heads;
    uint head_dim;
    uint num_repeats;
};

/// Repeat KV heads: out[s, kv*R+r, d] = input[s, kv, d].
/// Thread grid: seq * kv_heads * num_repeats * head_dim, 1D.
kernel void repeat_kv_f32(
    device const float* input           [[buffer(0)]],
    device float* output                [[buffer(1)]],
    constant RepeatKvParams& params     [[buffer(2)]],
    uint tid                            [[thread_position_in_grid]])
{
    const uint kv_heads    = params.kv_heads;
    const uint head_dim    = params.head_dim;
    const uint num_repeats = params.num_repeats;
    const uint new_heads   = kv_heads * num_repeats;

    const uint d   = tid % head_dim;
    const uint nh  = (tid / head_dim) % new_heads;
    const uint s   = tid / (head_dim * new_heads);

    const uint kv  = nh / num_repeats;
    output[tid] = input[(s * kv_heads + kv) * head_dim + d];
}

struct CopyStridedParams {
    uint in_cols;
    uint out_cols;
    uint col_offset;
};

/// Copy with stride: out[row * out_cols + col_offset + col] = in[row * in_cols + col].
/// Thread grid: outer * in_cols, 1D.
kernel void copy_strided_f32(
    device const float* input           [[buffer(0)]],
    device float* output                [[buffer(1)]],
    constant CopyStridedParams& params  [[buffer(2)]],
    uint tid                            [[thread_position_in_grid]])
{
    const uint row = tid / params.in_cols;
    const uint col = tid % params.in_cols;
    output[row * params.out_cols + params.col_offset + col] = input[row * params.in_cols + col];
}

struct PadInnerParams {
    uint width;
    uint new_width;
};

/// Pad inner dimension: copy width elements per row, rest stays zero.
/// Thread grid: outer * width, 1D.
kernel void pad_inner_f32(
    device const float* input           [[buffer(0)]],
    device float* output                [[buffer(1)]],
    constant PadInnerParams& params     [[buffer(2)]],
    uint tid                            [[thread_position_in_grid]])
{
    const uint row = tid / params.width;
    const uint col = tid % params.width;
    output[row * params.new_width + col] = input[row * params.width + col];
}

kernel void repeat_kv_f16(
    device const half* input            [[buffer(0)]],
    device half* output                 [[buffer(1)]],
    constant RepeatKvParams& params     [[buffer(2)]],
    uint tid                            [[thread_position_in_grid]])
{
    const uint kv_heads    = params.kv_heads;
    const uint head_dim    = params.head_dim;
    const uint num_repeats = params.num_repeats;
    const uint new_heads   = kv_heads * num_repeats;

    const uint d   = tid % head_dim;
    const uint nh  = (tid / head_dim) % new_heads;
    const uint s   = tid / (head_dim * new_heads);

    const uint kv  = nh / num_repeats;
    output[tid] = input[(s * kv_heads + kv) * head_dim + d];
}

kernel void copy_strided_f16(
    device const half* input            [[buffer(0)]],
    device half* output                 [[buffer(1)]],
    constant CopyStridedParams& params  [[buffer(2)]],
    uint tid                            [[thread_position_in_grid]])
{
    const uint row = tid / params.in_cols;
    const uint col = tid % params.in_cols;
    output[row * params.out_cols + params.col_offset + col] = input[row * params.in_cols + col];
}
