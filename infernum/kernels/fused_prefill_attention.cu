#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define INFINITY __int_as_float(0x7f800000)

extern "C" __global__ void fused_prefill_attention_f32(
    float* __restrict__ output,       // (seq_q, num_heads, head_dim)
    const float* __restrict__ q,      // (seq_q, num_heads, head_dim)
    const float* __restrict__ k,      // (total_len, num_kv_heads, head_dim)
    const float* __restrict__ v,      // (total_len, num_kv_heads, head_dim)
    const float scale,
    const int seq_q,
    const int total_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int offset           // causal offset for KV cache
) {
    // blockIdx.x = head, blockIdx.y = query position
    const int head = blockIdx.x;
    const int qpos = blockIdx.y;
    const int tid = threadIdx.x;

    const int kv_head = head * num_kv_heads / num_heads;
    const int max_valid_k = offset + qpos + 1;

    extern __shared__ float shared[];
    float* s_q = shared;
    float* s_scratch = shared + head_dim;

    // Load Q for this (qpos, head) into shared memory
    const float* q_ptr = q + qpos * num_heads * head_dim + head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = q_ptr[d];
    }
    __syncthreads();

    // Pass 1: find max score over valid key positions
    float local_max = -INFINITY;
    for (int t = tid; t < max_valid_k && t < total_len; t += blockDim.x) {
        float dot = 0.0f;
        const float* k_ptr = k + t * num_kv_heads * head_dim + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * k_ptr[d];
        }
        dot *= scale;
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

    // Pass 2: compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (int t = tid; t < max_valid_k && t < total_len; t += blockDim.x) {
        float dot = 0.0f;
        const float* k_ptr = k + t * num_kv_heads * head_dim + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * k_ptr[d];
        }
        dot *= scale;
        local_sum += expf(dot - max_val);
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

    // Pass 3: compute weighted output
    float* out_ptr = output + qpos * num_heads * head_dim + head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < max_valid_k && t < total_len; t++) {
            float dot = 0.0f;
            const float* k_ptr = k + t * num_kv_heads * head_dim + kv_head * head_dim;
            for (int dd = 0; dd < head_dim; dd++) {
                dot += s_q[dd] * k_ptr[dd];
            }
            dot *= scale;
            float weight = expf(dot - max_val) / sum_val;

            const float* v_ptr = v + t * num_kv_heads * head_dim + kv_head * head_dim;
            acc += weight * v_ptr[d];
        }
        out_ptr[d] = acc;
    }
}

// F16 version with F32 accumulation
extern "C" __global__ void fused_prefill_attention_f16(
    __half* __restrict__ output,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const float scale,
    const int seq_q,
    const int total_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int offset
) {
    const int head = blockIdx.x;
    const int qpos = blockIdx.y;
    const int tid = threadIdx.x;
    const int kv_head = head * num_kv_heads / num_heads;
    const int max_valid_k = offset + qpos + 1;

    extern __shared__ float shared[];
    float* s_q = shared;
    float* s_scratch = shared + head_dim;

    const __half* q_ptr = q + qpos * num_heads * head_dim + head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = __half2float(q_ptr[d]);
    }
    __syncthreads();

    // Pass 1: find max
    float local_max = -INFINITY;
    for (int t = tid; t < max_valid_k && t < total_len; t += blockDim.x) {
        float dot = 0.0f;
        const __half* k_ptr = k + t * num_kv_heads * head_dim + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * __half2float(k_ptr[d]);
        }
        dot *= scale;
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

    // Pass 2: compute sum
    float local_sum = 0.0f;
    for (int t = tid; t < max_valid_k && t < total_len; t += blockDim.x) {
        float dot = 0.0f;
        const __half* k_ptr = k + t * num_kv_heads * head_dim + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * __half2float(k_ptr[d]);
        }
        dot *= scale;
        local_sum += expf(dot - max_val);
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

    // Pass 3: weighted output
    __half* out_ptr = output + qpos * num_heads * head_dim + head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < max_valid_k && t < total_len; t++) {
            float dot = 0.0f;
            const __half* k_ptr = k + t * num_kv_heads * head_dim + kv_head * head_dim;
            for (int dd = 0; dd < head_dim; dd++) {
                dot += s_q[dd] * __half2float(k_ptr[dd]);
            }
            dot *= scale;
            float weight = expf(dot - max_val) / sum_val;
            const __half* v_ptr = v + t * num_kv_heads * head_dim + kv_head * head_dim;
            acc += weight * __half2float(v_ptr[d]);
        }
        out_ptr[d] = __float2half(acc);
    }
}

// BF16 version with F32 accumulation
extern "C" __global__ void fused_prefill_attention_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float scale,
    const int seq_q,
    const int total_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int offset
) {
    const int head = blockIdx.x;
    const int qpos = blockIdx.y;
    const int tid = threadIdx.x;
    const int kv_head = head * num_kv_heads / num_heads;
    const int max_valid_k = offset + qpos + 1;

    extern __shared__ float shared[];
    float* s_q = shared;
    float* s_scratch = shared + head_dim;

    const __nv_bfloat16* q_ptr = q + qpos * num_heads * head_dim + head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = __bfloat162float(q_ptr[d]);
    }
    __syncthreads();

    // Pass 1: find max
    float local_max = -INFINITY;
    for (int t = tid; t < max_valid_k && t < total_len; t += blockDim.x) {
        float dot = 0.0f;
        const __nv_bfloat16* k_ptr = k + t * num_kv_heads * head_dim + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * __bfloat162float(k_ptr[d]);
        }
        dot *= scale;
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

    // Pass 2: compute sum
    float local_sum = 0.0f;
    for (int t = tid; t < max_valid_k && t < total_len; t += blockDim.x) {
        float dot = 0.0f;
        const __nv_bfloat16* k_ptr = k + t * num_kv_heads * head_dim + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * __bfloat162float(k_ptr[d]);
        }
        dot *= scale;
        local_sum += expf(dot - max_val);
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

    // Pass 3: weighted output
    __nv_bfloat16* out_ptr = output + qpos * num_heads * head_dim + head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < max_valid_k && t < total_len; t++) {
            float dot = 0.0f;
            const __nv_bfloat16* k_ptr = k + t * num_kv_heads * head_dim + kv_head * head_dim;
            for (int dd = 0; dd < head_dim; dd++) {
                dot += s_q[dd] * __bfloat162float(k_ptr[dd]);
            }
            dot *= scale;
            float weight = expf(dot - max_val) / sum_val;
            const __nv_bfloat16* v_ptr = v + t * num_kv_heads * head_dim + kv_head * head_dim;
            acc += weight * __bfloat162float(v_ptr[d]);
        }
        out_ptr[d] = __float2bfloat16(acc);
    }
}
