#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void add_f32(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

extern "C" __global__ void add_inplace_f32(
    float* __restrict__ a,
    const float* __restrict__ b,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += b[idx];
    }
}

extern "C" __global__ void add_f16(
    __half* __restrict__ output,
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __hadd(a[idx], b[idx]);
    }
}

extern "C" __global__ void add_inplace_f16(
    __half* __restrict__ a,
    const __half* __restrict__ b,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = __hadd(a[idx], b[idx]);
    }
}

extern "C" __global__ void add_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __hadd(a[idx], b[idx]);
    }
}

extern "C" __global__ void add_inplace_bf16(
    __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = __hadd(a[idx], b[idx]);
    }
}
