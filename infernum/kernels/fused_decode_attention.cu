#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define INFINITY __int_as_float(0x7f800000)

// Fused decode attention with score caching
// Caches attention weights in shared memory to avoid Q·K recomputation in output pass.
// This reduces compute from 3 passes over head_dim to 2 passes.
extern "C" __global__ void fused_decode_attention_f32(
    float* __restrict__ output,       // (1, num_heads, head_dim)
    const float* __restrict__ q,      // (1, num_heads, head_dim)
    const float* __restrict__ k,      // (total_len, num_kv_heads, head_dim)
    const float* __restrict__ v,      // (total_len, num_kv_heads, head_dim)
    const float scale,
    const int total_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;

    // GQA: map query head to KV head
    const int kv_head = head * num_kv_heads / num_heads;

    // Shared memory layout:
    //   [0 .. head_dim)                    : Q vector
    //   [head_dim .. head_dim + total_len) : cached weights (softmax output)
    //   [head_dim + total_len .. ]         : reduction scratch
    extern __shared__ float shared[];
    float* s_q = shared;
    float* s_weights = shared + head_dim;
    float* s_scratch = shared + head_dim + total_len;

    const int kv_stride = num_kv_heads * head_dim;
    const float* k_base = k + kv_head * head_dim;
    const float* v_base = v + kv_head * head_dim;

    // Load Q into shared memory
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = q[head * head_dim + d];
    }
    __syncthreads();

    // Pass 1: Compute scores and find max (parallel over positions)
    float local_max = -INFINITY;
    for (int t = tid; t < total_len; t += blockDim.x) {
        float dot = 0.0f;
        const float* k_ptr = k_base + t * kv_stride;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * k_ptr[d];
        }
        dot *= scale;
        s_weights[t] = dot;  // Cache the score
        local_max = fmaxf(local_max, dot);
    }

    // Reduce max
    s_scratch[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] = fmaxf(s_scratch[tid], s_scratch[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = s_scratch[0];
    __syncthreads();

    // Pass 2: Compute exp and sum, store normalized weights
    float local_sum = 0.0f;
    for (int t = tid; t < total_len; t += blockDim.x) {
        float w = expf(s_weights[t] - max_val);
        s_weights[t] = w;
        local_sum += w;
    }

    // Reduce sum
    s_scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = s_scratch[0];
    __syncthreads();

    // Normalize weights in shared memory
    float inv_sum = 1.0f / sum_val;
    for (int t = tid; t < total_len; t += blockDim.x) {
        s_weights[t] *= inv_sum;
    }
    __syncthreads();

    // Pass 3: Compute weighted output (parallel over dimensions)
    // Uses cached weights - no Q·K recomputation!
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < total_len; t++) {
            acc += s_weights[t] * v_base[t * kv_stride + d];
        }
        output[head * head_dim + d] = acc;
    }
}

// GPU-indirect variant: reads total_len from a device pointer.
// Shared memory must be sized for max_seq_len (the worst case), since
// the grid is captured once and replayed. The kernel reads the actual
// total_len at execution time and only loops over that many positions.
extern "C" __global__ void fused_decode_attention_indirect_f32(
    float* __restrict__ output,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float scale,
    const unsigned int* __restrict__ total_len_ptr,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int total_len = (int)(*total_len_ptr);
    const int kv_head = head * num_kv_heads / num_heads;

    extern __shared__ float shared[];
    float* s_q = shared;
    float* s_weights = shared + head_dim;
    float* s_scratch = shared + head_dim + total_len;

    const int kv_stride = num_kv_heads * head_dim;
    const float* k_base = k + kv_head * head_dim;
    const float* v_base = v + kv_head * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = q[head * head_dim + d];
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int t = tid; t < total_len; t += blockDim.x) {
        float dot = 0.0f;
        const float* k_ptr = k_base + t * kv_stride;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * k_ptr[d];
        }
        dot *= scale;
        s_weights[t] = dot;
        local_max = fmaxf(local_max, dot);
    }

    s_scratch[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] = fmaxf(s_scratch[tid], s_scratch[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = s_scratch[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int t = tid; t < total_len; t += blockDim.x) {
        float w = expf(s_weights[t] - max_val);
        s_weights[t] = w;
        local_sum += w;
    }

    s_scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = s_scratch[0];
    __syncthreads();

    float inv_sum = 1.0f / sum_val;
    for (int t = tid; t < total_len; t += blockDim.x) {
        s_weights[t] *= inv_sum;
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < total_len; t++) {
            acc += s_weights[t] * v_base[t * kv_stride + d];
        }
        output[head * head_dim + d] = acc;
    }
}

// F16 version with F32 accumulation and score caching
extern "C" __global__ void fused_decode_attention_f16(
    __half* __restrict__ output,       // (1, num_heads, head_dim)
    const __half* __restrict__ q,      // (1, num_heads, head_dim)
    const __half* __restrict__ k,      // (total_len, num_kv_heads, head_dim)
    const __half* __restrict__ v,      // (total_len, num_kv_heads, head_dim)
    const float scale,
    const int total_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int kv_head = head * num_kv_heads / num_heads;

    // Shared memory layout:
    //   [0 .. head_dim)                    : Q vector (f32)
    //   [head_dim .. head_dim + total_len) : cached weights (f32)
    //   [head_dim + total_len .. ]         : reduction scratch (f32)
    extern __shared__ float shared[];
    float* s_q = shared;
    float* s_weights = shared + head_dim;
    float* s_scratch = shared + head_dim + total_len;

    const int kv_stride = num_kv_heads * head_dim;
    const __half* k_base = k + kv_head * head_dim;
    const __half* v_base = v + kv_head * head_dim;

    // Load Q into shared memory (convert to F32)
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = __half2float(q[head * head_dim + d]);
    }
    __syncthreads();

    // Pass 1: Compute scores and find max (parallel over positions)
    float local_max = -INFINITY;
    for (int t = tid; t < total_len; t += blockDim.x) {
        float dot = 0.0f;
        const __half* k_ptr = k_base + t * kv_stride;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * __half2float(k_ptr[d]);
        }
        dot *= scale;
        s_weights[t] = dot;  // Cache the score
        local_max = fmaxf(local_max, dot);
    }

    // Reduce max
    s_scratch[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] = fmaxf(s_scratch[tid], s_scratch[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = s_scratch[0];
    __syncthreads();

    // Pass 2: Compute exp and sum, store normalized weights
    float local_sum = 0.0f;
    for (int t = tid; t < total_len; t += blockDim.x) {
        float w = expf(s_weights[t] - max_val);
        s_weights[t] = w;
        local_sum += w;
    }

    // Reduce sum
    s_scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = s_scratch[0];
    __syncthreads();

    // Normalize weights in shared memory
    float inv_sum = 1.0f / sum_val;
    for (int t = tid; t < total_len; t += blockDim.x) {
        s_weights[t] *= inv_sum;
    }
    __syncthreads();

    // Pass 3: Compute weighted output (parallel over dimensions)
    // Uses cached weights - no Q·K recomputation!
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < total_len; t++) {
            acc += s_weights[t] * __half2float(v_base[t * kv_stride + d]);
        }
        output[head * head_dim + d] = __float2half(acc);
    }
}

extern "C" __global__ void fused_decode_attention_indirect_f16(
    __half* __restrict__ output,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const float scale,
    const unsigned int* __restrict__ total_len_ptr,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int total_len = (int)(*total_len_ptr);
    const int kv_head = head * num_kv_heads / num_heads;

    extern __shared__ float shared[];
    float* s_q = shared;
    float* s_weights = shared + head_dim;
    float* s_scratch = shared + head_dim + total_len;

    const int kv_stride = num_kv_heads * head_dim;
    const __half* k_base = k + kv_head * head_dim;
    const __half* v_base = v + kv_head * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = __half2float(q[head * head_dim + d]);
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int t = tid; t < total_len; t += blockDim.x) {
        float dot = 0.0f;
        const __half* k_ptr = k_base + t * kv_stride;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * __half2float(k_ptr[d]);
        }
        dot *= scale;
        s_weights[t] = dot;
        local_max = fmaxf(local_max, dot);
    }

    s_scratch[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] = fmaxf(s_scratch[tid], s_scratch[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = s_scratch[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int t = tid; t < total_len; t += blockDim.x) {
        float w = expf(s_weights[t] - max_val);
        s_weights[t] = w;
        local_sum += w;
    }

    s_scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = s_scratch[0];
    __syncthreads();

    float inv_sum = 1.0f / sum_val;
    for (int t = tid; t < total_len; t += blockDim.x) {
        s_weights[t] *= inv_sum;
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < total_len; t++) {
            acc += s_weights[t] * __half2float(v_base[t * kv_stride + d]);
        }
        output[head * head_dim + d] = __float2half(acc);
    }
}

// BF16 version with F32 accumulation and score caching
extern "C" __global__ void fused_decode_attention_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float scale,
    const int total_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int kv_head = head * num_kv_heads / num_heads;

    // Shared memory layout:
    //   [0 .. head_dim)                    : Q vector (f32)
    //   [head_dim .. head_dim + total_len) : cached weights (f32)
    //   [head_dim + total_len .. ]         : reduction scratch (f32)
    extern __shared__ float shared[];
    float* s_q = shared;
    float* s_weights = shared + head_dim;
    float* s_scratch = shared + head_dim + total_len;

    const int kv_stride = num_kv_heads * head_dim;
    const __nv_bfloat16* k_base = k + kv_head * head_dim;
    const __nv_bfloat16* v_base = v + kv_head * head_dim;

    // Load Q into shared memory (convert to F32)
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = __bfloat162float(q[head * head_dim + d]);
    }
    __syncthreads();

    // Pass 1: Compute scores and find max (parallel over positions)
    float local_max = -INFINITY;
    for (int t = tid; t < total_len; t += blockDim.x) {
        float dot = 0.0f;
        const __nv_bfloat16* k_ptr = k_base + t * kv_stride;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * __bfloat162float(k_ptr[d]);
        }
        dot *= scale;
        s_weights[t] = dot;  // Cache the score
        local_max = fmaxf(local_max, dot);
    }

    // Reduce max
    s_scratch[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] = fmaxf(s_scratch[tid], s_scratch[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = s_scratch[0];
    __syncthreads();

    // Pass 2: Compute exp and sum, store normalized weights
    float local_sum = 0.0f;
    for (int t = tid; t < total_len; t += blockDim.x) {
        float w = expf(s_weights[t] - max_val);
        s_weights[t] = w;
        local_sum += w;
    }

    // Reduce sum
    s_scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = s_scratch[0];
    __syncthreads();

    // Normalize weights in shared memory
    float inv_sum = 1.0f / sum_val;
    for (int t = tid; t < total_len; t += blockDim.x) {
        s_weights[t] *= inv_sum;
    }
    __syncthreads();

    // Pass 3: Compute weighted output (parallel over dimensions)
    // Uses cached weights - no Q·K recomputation!
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < total_len; t++) {
            acc += s_weights[t] * __bfloat162float(v_base[t * kv_stride + d]);
        }
        output[head * head_dim + d] = __float2bfloat16(acc);
    }
}

extern "C" __global__ void fused_decode_attention_indirect_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float scale,
    const unsigned int* __restrict__ total_len_ptr,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int total_len = (int)(*total_len_ptr);
    const int kv_head = head * num_kv_heads / num_heads;

    extern __shared__ float shared[];
    float* s_q = shared;
    float* s_weights = shared + head_dim;
    float* s_scratch = shared + head_dim + total_len;

    const int kv_stride = num_kv_heads * head_dim;
    const __nv_bfloat16* k_base = k + kv_head * head_dim;
    const __nv_bfloat16* v_base = v + kv_head * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = __bfloat162float(q[head * head_dim + d]);
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int t = tid; t < total_len; t += blockDim.x) {
        float dot = 0.0f;
        const __nv_bfloat16* k_ptr = k_base + t * kv_stride;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * __bfloat162float(k_ptr[d]);
        }
        dot *= scale;
        s_weights[t] = dot;
        local_max = fmaxf(local_max, dot);
    }

    s_scratch[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] = fmaxf(s_scratch[tid], s_scratch[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = s_scratch[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int t = tid; t < total_len; t += blockDim.x) {
        float w = expf(s_weights[t] - max_val);
        s_weights[t] = w;
        local_sum += w;
    }

    s_scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = s_scratch[0];
    __syncthreads();

    float inv_sum = 1.0f / sum_val;
    for (int t = tid; t < total_len; t += blockDim.x) {
        s_weights[t] *= inv_sum;
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < total_len; t++) {
            acc += s_weights[t] * __bfloat162float(v_base[t * kv_stride + d]);
        }
        output[head * head_dim + d] = __float2bfloat16(acc);
    }
}
