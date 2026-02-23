#include <cuda_bf16.h>

// Split along inner dim: input[outer, dim1+dim2] → out_a[outer, dim1] + out_b[outer, dim2]
// Thread index maps to total output elements (outer * (dim1 + dim2)).
extern "C" __global__ void split_inner_dim_f32(
    float* __restrict__ out_a,
    float* __restrict__ out_b,
    const float* __restrict__ input,
    const int outer,
    const int dim1,
    const int dim2
) {
    const int total_dim = dim1 + dim2;
    const int total = outer * total_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        const int row = idx / total_dim;
        const int col = idx % total_dim;
        if (col < dim1) {
            out_a[row * dim1 + col] = input[idx];
        } else {
            out_b[row * dim2 + (col - dim1)] = input[idx];
        }
    }
}

extern "C" __global__ void split_inner_dim_bf16(
    __nv_bfloat16* __restrict__ out_a,
    __nv_bfloat16* __restrict__ out_b,
    const __nv_bfloat16* __restrict__ input,
    const int outer,
    const int dim1,
    const int dim2
) {
    const int total_dim = dim1 + dim2;
    const int total = outer * total_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        const int row = idx / total_dim;
        const int col = idx % total_dim;
        if (col < dim1) {
            out_a[row * dim1 + col] = input[idx];
        } else {
            out_b[row * dim2 + (col - dim1)] = input[idx];
        }
    }
}

// Concat along inner dim: in_a[outer, dim1] + in_b[outer, dim2] → output[outer, dim1+dim2]
extern "C" __global__ void concat_inner_dim_f32(
    float* __restrict__ output,
    const float* __restrict__ in_a,
    const float* __restrict__ in_b,
    const int outer,
    const int dim1,
    const int dim2
) {
    const int total_dim = dim1 + dim2;
    const int total = outer * total_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        const int row = idx / total_dim;
        const int col = idx % total_dim;
        if (col < dim1) {
            output[idx] = in_a[row * dim1 + col];
        } else {
            output[idx] = in_b[row * dim2 + (col - dim1)];
        }
    }
}

extern "C" __global__ void concat_inner_dim_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ in_a,
    const __nv_bfloat16* __restrict__ in_b,
    const int outer,
    const int dim1,
    const int dim2
) {
    const int total_dim = dim1 + dim2;
    const int total = outer * total_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        const int row = idx / total_dim;
        const int col = idx % total_dim;
        if (col < dim1) {
            output[idx] = in_a[row * dim1 + col];
        } else {
            output[idx] = in_b[row * dim2 + (col - dim1)];
        }
    }
}

// Broadcast: input[dim] → output[num_heads * dim] by repeating input num_heads times.
// Total threads = num_heads * dim.
extern "C" __global__ void broadcast_to_heads_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int num_heads,
    const int dim
) {
    const int total = num_heads * dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        output[idx] = input[idx % dim];
    }
}

extern "C" __global__ void broadcast_to_heads_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const int num_heads,
    const int dim
) {
    const int total = num_heads * dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        output[idx] = input[idx % dim];
    }
}

// Pad inner dim with zeros: input[outer, src_dim] → output[outer, dst_dim]
// First src_dim elements per row are copied, rest are zero.
extern "C" __global__ void pad_inner_dim_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int outer,
    const int src_dim,
    const int dst_dim
) {
    const int total = outer * dst_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        const int row = idx / dst_dim;
        const int col = idx % dst_dim;
        output[idx] = (col < src_dim) ? input[row * src_dim + col] : 0.0f;
    }
}

extern "C" __global__ void pad_inner_dim_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const int outer,
    const int src_dim,
    const int dst_dim
) {
    const int total = outer * dst_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        const int row = idx / dst_dim;
        const int col = idx % dst_dim;
        output[idx] = (col < src_dim) ? input[row * src_dim + col] : __float2bfloat16(0.0f);
    }
}
