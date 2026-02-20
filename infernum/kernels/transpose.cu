#include <cuda_bf16.h>

// Transpose 2D: (rows, cols) -> (cols, rows)
extern "C" __global__ void transpose_2d_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int rows,
    const int cols
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx < total) {
        const int r = idx / cols;
        const int c = idx % cols;
        output[c * rows + r] = input[r * cols + c];
    }
}

// Transpose 2D bf16: (rows, cols) -> (cols, rows)
extern "C" __global__ void transpose_2d_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const int rows,
    const int cols
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx < total) {
        const int r = idx / cols;
        const int c = idx % cols;
        output[c * rows + r] = input[r * cols + c];
    }
}

// Transpose 3D: (a, b, c) -> (b, a, c)
// Swaps the first two dimensions
extern "C" __global__ void transpose_012_to_102_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int dim_a,
    const int dim_b,
    const int dim_c
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = dim_a * dim_b * dim_c;
    if (idx < total) {
        const int i = idx / (dim_b * dim_c);
        const int remainder = idx % (dim_b * dim_c);
        const int j = remainder / dim_c;
        const int k = remainder % dim_c;

        // src: (i, j, k) -> dst: (j, i, k)
        const int dst_idx = j * dim_a * dim_c + i * dim_c + k;
        output[dst_idx] = input[idx];
    }
}

// Transpose last two dims of 3D: (a, b, c) -> (a, c, b)
extern "C" __global__ void transpose_last_two_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int dim_a,
    const int dim_b,
    const int dim_c
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = dim_a * dim_b * dim_c;
    if (idx < total) {
        const int i = idx / (dim_b * dim_c);
        const int remainder = idx % (dim_b * dim_c);
        const int j = remainder / dim_c;
        const int k = remainder % dim_c;

        // src: (i, j, k) -> dst: (i, k, j)
        const int dst_idx = i * dim_c * dim_b + k * dim_b + j;
        output[dst_idx] = input[idx];
    }
}
