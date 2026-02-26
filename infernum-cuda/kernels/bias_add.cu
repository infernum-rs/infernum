#include <cuda_fp16.h>
#include <cuda_bf16.h>

// bias_add: output[row * cols + col] = input[row * cols + col] + bias[col]
extern "C" __global__ void bias_add_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ bias,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        output[idx] = input[idx] + bias[idx % cols];
    }
}

// bias_add_inplace: input[row * cols + col] += bias[col]
extern "C" __global__ void bias_add_inplace_f32(
    float* __restrict__ input,
    const float* __restrict__ bias,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        input[idx] += bias[idx % cols];
    }
}

extern "C" __global__ void bias_add_f16(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ bias,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        output[idx] = __hadd(input[idx], bias[idx % cols]);
    }
}

extern "C" __global__ void bias_add_inplace_f16(
    __half* __restrict__ input,
    const __half* __restrict__ bias,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        input[idx] = __hadd(input[idx], bias[idx % cols]);
    }
}

extern "C" __global__ void bias_add_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ bias,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        output[idx] = __hadd(input[idx], bias[idx % cols]);
    }
}

extern "C" __global__ void bias_add_inplace_bf16(
    __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ bias,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        input[idx] = __hadd(input[idx], bias[idx % cols]);
    }
}
