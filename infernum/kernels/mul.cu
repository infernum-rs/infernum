#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void mul_f32(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] * b[idx];
    }
}

extern "C" __global__ void mul_f16(
    __half* __restrict__ output,
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __hmul(a[idx], b[idx]);
    }
}

extern "C" __global__ void mul_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __hmul(a[idx], b[idx]);
    }
}
