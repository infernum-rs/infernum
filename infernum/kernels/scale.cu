#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void scale_f32(
    float* __restrict__ data,
    const float scale,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

extern "C" __global__ void scale_f16(
    __half* __restrict__ data,
    const float scale,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = __float2half(__half2float(data[idx]) * scale);
    }
}

extern "C" __global__ void scale_bf16(
    __nv_bfloat16* __restrict__ data,
    const float scale,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = __float2bfloat16(__bfloat162float(data[idx]) * scale);
    }
}

// Per-column broadcast scale: data[m, n] *= scales[n]
// data shape: [M, N] row-major, scales shape: [N]
// Grid/block: 1D over total elements (M * N)
extern "C" __global__ void scale_rows_f32(
    float* __restrict__ data,
    const float* __restrict__ scales,
    const int N,
    const int total
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx] *= scales[idx % N];
    }
}
