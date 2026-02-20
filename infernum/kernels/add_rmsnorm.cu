// Fused residual add + RMS normalization.
//
// For each row: computes sum = residual + x, then RMS-normalizes sum.
// Writes both the sum (updated hidden state) and the normed result,
// saving one global memory round-trip compared to separate add + rmsnorm.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void add_rmsnorm_f32(
    float* __restrict__ sum_out,
    float* __restrict__ norm_out,
    const float* __restrict__ residual,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* row_residual = residual + row * hidden_size;
    const float* row_x = x + row * hidden_size;
    float* row_sum = sum_out + row * hidden_size;
    float* row_norm = norm_out + row * hidden_size;

    extern __shared__ float shared[];

    // Pass 1: compute sum and accumulate sum-of-squares
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float s = row_residual[i] + row_x[i];
        row_sum[i] = s;
        local_ss += s * s;
    }

    shared[tid] = local_ss;
    __syncthreads();

    // Block reduction for sum-of-squares
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / (float)hidden_size + eps);

    // Pass 2: apply normalization and weight
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        row_norm[i] = row_sum[i] * rms * weight[i];
    }
}

extern "C" __global__ void add_rmsnorm_f16(
    __half* __restrict__ sum_out,
    __half* __restrict__ norm_out,
    const __half* __restrict__ residual,
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __half* row_residual = residual + row * hidden_size;
    const __half* row_x = x + row * hidden_size;
    __half* row_sum = sum_out + row * hidden_size;
    __half* row_norm = norm_out + row * hidden_size;

    extern __shared__ float shared[];

    // Pass 1: compute sum and accumulate sum-of-squares (in F32)
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float s = __half2float(row_residual[i]) + __half2float(row_x[i]);
        row_sum[i] = __float2half(s);
        local_ss += s * s;
    }

    shared[tid] = local_ss;
    __syncthreads();

    // Block reduction for sum-of-squares
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / (float)hidden_size + eps);

    // Pass 2: apply normalization and weight
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float sum_val = __half2float(row_sum[i]);
        row_norm[i] = __float2half(sum_val * rms * __half2float(weight[i]));
    }
}

extern "C" __global__ void add_rmsnorm_bf16(
    __nv_bfloat16* __restrict__ sum_out,
    __nv_bfloat16* __restrict__ norm_out,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* row_residual = residual + row * hidden_size;
    const __nv_bfloat16* row_x = x + row * hidden_size;
    __nv_bfloat16* row_sum = sum_out + row * hidden_size;
    __nv_bfloat16* row_norm = norm_out + row * hidden_size;

    extern __shared__ float shared[];

    // Pass 1: compute sum and accumulate sum-of-squares (in F32)
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float s = __bfloat162float(row_residual[i]) + __bfloat162float(row_x[i]);
        row_sum[i] = __float2bfloat16(s);
        local_ss += s * s;
    }

    shared[tid] = local_ss;
    __syncthreads();

    // Block reduction for sum-of-squares
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / (float)hidden_size + eps);

    // Pass 2: apply normalization and weight
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float sum_val = __bfloat162float(row_sum[i]);
        row_norm[i] = __float2bfloat16(sum_val * rms * __bfloat162float(weight[i]));
    }
}
