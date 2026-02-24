#include <cuda_fp16.h>
#include <cuda_bf16.h>

// GELU (tanh approximation, "gelu_pytorch_tanh"):
// y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

#define GELU_COEFF_A 0.7978845608028654f   // sqrt(2/π)
#define GELU_COEFF_B 0.044715f

extern "C" __global__ void gelu_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float inner = GELU_COEFF_A * (x + GELU_COEFF_B * x * x * x);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

extern "C" __global__ void gelu_inplace_f32(
    float* __restrict__ data,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        float inner = GELU_COEFF_A * (x + GELU_COEFF_B * x * x * x);
        data[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// GELU with elementwise multiplication: output = gelu(gate) * up
// Used in GeGLU: gelu(gate) * up
extern "C" __global__ void gelu_mul_f32(
    float* __restrict__ output,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = gate[idx];
        float inner = GELU_COEFF_A * (x + GELU_COEFF_B * x * x * x);
        float gelu_x = 0.5f * x * (1.0f + tanhf(inner));
        output[idx] = gelu_x * up[idx];
    }
}

// F16 variants (compute in F32 for accuracy)
extern "C" __global__ void gelu_f16(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(input[idx]);
        float inner = GELU_COEFF_A * (x + GELU_COEFF_B * x * x * x);
        output[idx] = __float2half(0.5f * x * (1.0f + tanhf(inner)));
    }
}

extern "C" __global__ void gelu_inplace_f16(
    __half* __restrict__ data,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(data[idx]);
        float inner = GELU_COEFF_A * (x + GELU_COEFF_B * x * x * x);
        data[idx] = __float2half(0.5f * x * (1.0f + tanhf(inner)));
    }
}

extern "C" __global__ void gelu_mul_f16(
    __half* __restrict__ output,
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(gate[idx]);
        float inner = GELU_COEFF_A * (x + GELU_COEFF_B * x * x * x);
        float gelu_x = 0.5f * x * (1.0f + tanhf(inner));
        output[idx] = __float2half(gelu_x * __half2float(up[idx]));
    }
}

// BF16 variants (compute in F32 for accuracy)
extern "C" __global__ void gelu_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(input[idx]);
        float inner = GELU_COEFF_A * (x + GELU_COEFF_B * x * x * x);
        output[idx] = __float2bfloat16(0.5f * x * (1.0f + tanhf(inner)));
    }
}

extern "C" __global__ void gelu_inplace_bf16(
    __nv_bfloat16* __restrict__ data,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(data[idx]);
        float inner = GELU_COEFF_A * (x + GELU_COEFF_B * x * x * x);
        data[idx] = __float2bfloat16(0.5f * x * (1.0f + tanhf(inner)));
    }
}

extern "C" __global__ void gelu_mul_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(gate[idx]);
        float inner = GELU_COEFF_A * (x + GELU_COEFF_B * x * x * x);
        float gelu_x = 0.5f * x * (1.0f + tanhf(inner));
        output[idx] = __float2bfloat16(gelu_x * __bfloat162float(up[idx]));
    }
}
