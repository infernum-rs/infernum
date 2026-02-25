#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define INFINITY __int_as_float(0x7f800000)

// Combine two attention outputs using their log-sum-exp values.
//
// For each (n, h) position:
//   m = max(lse1[n,h], lse2[n,h])
//   w1 = exp(lse1[n,h] - m)
//   w2 = exp(lse2[n,h] - m)
//   combined[n,h,d] = (w1 * out1[n,h,d] + w2 * out2[n,h,d]) / (w1 + w2)
//
// Grid: (num_heads, N, 1)
// Threads iterate over head_dim.

extern "C" __global__ void combine_attention_lse_f32(
    float* __restrict__ combined,      // (N, num_heads, head_dim)
    const float* __restrict__ out1,    // (N, num_heads, head_dim)
    const float* __restrict__ lse1,    // (N, num_heads)
    const float* __restrict__ out2,    // (N, num_heads, head_dim)
    const float* __restrict__ lse2,    // (N, num_heads)
    const int num_heads,
    const int head_dim
) {
    const int head = blockIdx.x;
    const int n = blockIdx.y;
    const int tid = threadIdx.x;

    const int lse_idx = n * num_heads + head;
    const float l1 = lse1[lse_idx];
    const float l2 = lse2[lse_idx];

    const float m = fmaxf(l1, l2);
    const float w1 = expf(l1 - m);
    const float w2 = expf(l2 - m);
    const float inv_sum = 1.0f / (w1 + w2);

    const int base = n * num_heads * head_dim + head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        combined[base + d] = (w1 * out1[base + d] + w2 * out2[base + d]) * inv_sum;
    }
}

extern "C" __global__ void combine_attention_lse_bf16(
    __nv_bfloat16* __restrict__ combined,
    const __nv_bfloat16* __restrict__ out1,
    const float* __restrict__ lse1,    // LSE is always f32
    const __nv_bfloat16* __restrict__ out2,
    const float* __restrict__ lse2,
    const int num_heads,
    const int head_dim
) {
    const int head = blockIdx.x;
    const int n = blockIdx.y;
    const int tid = threadIdx.x;

    const int lse_idx = n * num_heads + head;
    const float l1 = lse1[lse_idx];
    const float l2 = lse2[lse_idx];

    const float m = fmaxf(l1, l2);
    const float w1 = expf(l1 - m);
    const float w2 = expf(l2 - m);
    const float inv_sum = 1.0f / (w1 + w2);

    const int base = n * num_heads * head_dim + head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float v1 = __bfloat162float(out1[base + d]);
        float v2 = __bfloat162float(out2[base + d]);
        combined[base + d] = __float2bfloat16((w1 * v1 + w2 * v2) * inv_sum);
    }
}

extern "C" __global__ void combine_attention_lse_f16(
    __half* __restrict__ combined,
    const __half* __restrict__ out1,
    const float* __restrict__ lse1,
    const __half* __restrict__ out2,
    const float* __restrict__ lse2,
    const int num_heads,
    const int head_dim
) {
    const int head = blockIdx.x;
    const int n = blockIdx.y;
    const int tid = threadIdx.x;

    const int lse_idx = n * num_heads + head;
    const float l1 = lse1[lse_idx];
    const float l2 = lse2[lse_idx];

    const float m = fmaxf(l1, l2);
    const float w1 = expf(l1 - m);
    const float w2 = expf(l2 - m);
    const float inv_sum = 1.0f / (w1 + w2);

    const int base = n * num_heads * head_dim + head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float v1 = __half2float(out1[base + d]);
        float v2 = __half2float(out2[base + d]);
        combined[base + d] = __float2half((w1 * v1 + w2 * v2) * inv_sum);
    }
}
