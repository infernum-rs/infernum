#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void silu_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

extern "C" __global__ void silu_inplace_f32(
    float* __restrict__ data,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        data[idx] = x / (1.0f + expf(-x));
    }
}

// SiLU with elementwise multiplication: output = silu(a) * b
// Used in SwiGLU: silu(gate) * up
extern "C" __global__ void silu_mul_f32(
    float* __restrict__ output,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = gate[idx];
        float silu_x = x / (1.0f + expf(-x));
        output[idx] = silu_x * up[idx];
    }
}

// F16 variants (compute in F32 for accuracy)
extern "C" __global__ void silu_f16(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(input[idx]);
        output[idx] = __float2half(x / (1.0f + expf(-x)));
    }
}

extern "C" __global__ void silu_inplace_f16(
    __half* __restrict__ data,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(data[idx]);
        data[idx] = __float2half(x / (1.0f + expf(-x)));
    }
}

extern "C" __global__ void silu_mul_f16(
    __half* __restrict__ output,
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(gate[idx]);
        float silu_x = x / (1.0f + expf(-x));
        output[idx] = __float2half(silu_x * __half2float(up[idx]));
    }
}

// BF16 variants (compute in F32 for accuracy)
extern "C" __global__ void silu_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(input[idx]);
        output[idx] = __float2bfloat16(x / (1.0f + expf(-x)));
    }
}

extern "C" __global__ void silu_inplace_bf16(
    __nv_bfloat16* __restrict__ data,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(data[idx]);
        data[idx] = __float2bfloat16(x / (1.0f + expf(-x)));
    }
}

extern "C" __global__ void silu_mul_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(gate[idx]);
        float silu_x = x / (1.0f + expf(-x));
        output[idx] = __float2bfloat16(silu_x * __bfloat162float(up[idx]));
    }
}
