// Cast kernels between f32 and bf16
//
// bf16 is stored as unsigned short (__nv_bfloat16) — the upper 16 bits of
// an IEEE-754 float32, which makes the conversion a simple bit shift.

#include <cuda_bf16.h>

extern "C" __global__ void cast_f32_to_bf16(
    __nv_bfloat16* __restrict__ output,
    const float* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2bfloat16(input[idx]);
    }
}

extern "C" __global__ void cast_bf16_to_f32(
    float* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __bfloat162float(input[idx]);
    }
}
