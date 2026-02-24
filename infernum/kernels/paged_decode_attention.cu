#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define INFINITY_F __int_as_float(0x7f800000)

// Paged decode attention: one query token per request, K/V in block-table-indexed pool.
//
// Grid: (num_heads, batch_size, 1)
//   blockIdx.x = query head index
//   blockIdx.y = request index in the batch
//
// Each block computes attention for one (request, head) pair by iterating
// over that request's block table to find K/V data in the shared pool.
//
// Pool layout: (num_blocks, block_size, num_kv_heads, head_dim) — row-major
// block_tables: (batch_size, max_blocks_per_seq) — each row is one request's block table
// seq_lens: (batch_size,) — number of valid tokens per request
//
// Algorithm:
//   1. Load Q for this (request, head) into shared memory
//   2. For each token t in [0, seq_len):
//      - logical_block = t / block_size
//      - offset_in_block = t % block_size
//      - physical_block = block_tables[request * max_blocks_per_seq + logical_block]
//      - k_idx = ((physical_block * block_size + offset_in_block) * num_kv_heads + kv_head) * head_dim
//      - Compute dot(Q, K), track online softmax
//   3. Accumulate weighted V, write output

extern "C" __global__ void paged_decode_attention_f32(
    float* __restrict__ output,          // (batch_size, num_heads, head_dim)
    const float* __restrict__ q,         // (batch_size, num_heads, head_dim)
    const float* __restrict__ k_pool,    // (num_blocks, block_size, num_kv_heads, head_dim)
    const float* __restrict__ v_pool,    // same layout
    const int* __restrict__ block_tables,// (batch_size, max_blocks_per_seq)
    const int* __restrict__ seq_lens,    // (batch_size,)
    const float scale,
    const int block_size,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq
) {
    const int head = blockIdx.x;
    const int req = blockIdx.y;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[req];
    if (seq_len == 0) return;

    const int kv_head = head * num_kv_heads / num_heads;
    const int pool_kv_stride = num_kv_heads * head_dim;  // stride between positions in pool

    extern __shared__ float shared[];
    float* s_q = shared;                          // head_dim
    float* s_weights = shared + head_dim;         // seq_len (filled dynamically)
    float* s_scratch = shared + head_dim + seq_len; // blockDim.x

    // Load Q
    const int q_base = (req * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = q[q_base + d];
    }
    __syncthreads();

    // Pass 1: Compute Q·K scores and find max (for online softmax)
    const int* req_block_table = block_tables + req * max_blocks_per_seq;
    float local_max = -INFINITY_F;

    for (int t = tid; t < seq_len; t += blockDim.x) {
        const int logical_block = t / block_size;
        const int offset_in_block = t % block_size;
        const int physical_block = req_block_table[logical_block];
        const int k_base = ((physical_block * block_size + offset_in_block) * num_kv_heads + kv_head) * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * k_pool[k_base + d];
        }
        dot *= scale;
        s_weights[t] = dot;
        local_max = fmaxf(local_max, dot);
    }

    // Reduce max across threads
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

    // Pass 2: Compute exp and sum
    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += blockDim.x) {
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
    float inv_sum = 1.0f / s_scratch[0];
    __syncthreads();

    // Normalize weights
    for (int t = tid; t < seq_len; t += blockDim.x) {
        s_weights[t] *= inv_sum;
    }
    __syncthreads();

    // Pass 3: Weighted V output
    const int out_base = (req * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            const int logical_block = t / block_size;
            const int offset_in_block = t % block_size;
            const int physical_block = req_block_table[logical_block];
            const int v_idx = ((physical_block * block_size + offset_in_block) * num_kv_heads + kv_head) * head_dim + d;
            acc += s_weights[t] * v_pool[v_idx];
        }
        output[out_base + d] = acc;
    }
}

// f16 variant
extern "C" __global__ void paged_decode_attention_f16(
    __half* __restrict__ output,
    const __half* __restrict__ q,
    const __half* __restrict__ k_pool,
    const __half* __restrict__ v_pool,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const float scale,
    const int block_size,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq
) {
    const int head = blockIdx.x;
    const int req = blockIdx.y;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[req];
    if (seq_len == 0) return;

    const int kv_head = head * num_kv_heads / num_heads;

    extern __shared__ float shared[];
    float* s_q = shared;
    float* s_weights = shared + head_dim;
    float* s_scratch = shared + head_dim + seq_len;

    const int* req_block_table = block_tables + req * max_blocks_per_seq;
    const int q_base = (req * num_heads + head) * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = __half2float(q[q_base + d]);
    }
    __syncthreads();

    float local_max = -INFINITY_F;
    for (int t = tid; t < seq_len; t += blockDim.x) {
        const int logical_block = t / block_size;
        const int offset_in_block = t % block_size;
        const int physical_block = req_block_table[logical_block];
        const int k_base = ((physical_block * block_size + offset_in_block) * num_kv_heads + kv_head) * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * __half2float(k_pool[k_base + d]);
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
    for (int t = tid; t < seq_len; t += blockDim.x) {
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
    float inv_sum = 1.0f / s_scratch[0];
    __syncthreads();

    for (int t = tid; t < seq_len; t += blockDim.x) {
        s_weights[t] *= inv_sum;
    }
    __syncthreads();

    const int out_base = (req * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            const int logical_block = t / block_size;
            const int offset_in_block = t % block_size;
            const int physical_block = req_block_table[logical_block];
            const int v_idx = ((physical_block * block_size + offset_in_block) * num_kv_heads + kv_head) * head_dim + d;
            acc += s_weights[t] * __half2float(v_pool[v_idx]);
        }
        output[out_base + d] = __float2half(acc);
    }
}

// bf16 variant
extern "C" __global__ void paged_decode_attention_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const float scale,
    const int block_size,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq
) {
    const int head = blockIdx.x;
    const int req = blockIdx.y;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[req];
    if (seq_len == 0) return;

    const int kv_head = head * num_kv_heads / num_heads;

    extern __shared__ float shared[];
    float* s_q = shared;
    float* s_weights = shared + head_dim;
    float* s_scratch = shared + head_dim + seq_len;

    const int* req_block_table = block_tables + req * max_blocks_per_seq;
    const int q_base = (req * num_heads + head) * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = __bfloat162float(q[q_base + d]);
    }
    __syncthreads();

    float local_max = -INFINITY_F;
    for (int t = tid; t < seq_len; t += blockDim.x) {
        const int logical_block = t / block_size;
        const int offset_in_block = t % block_size;
        const int physical_block = req_block_table[logical_block];
        const int k_base = ((physical_block * block_size + offset_in_block) * num_kv_heads + kv_head) * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * __bfloat162float(k_pool[k_base + d]);
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
    for (int t = tid; t < seq_len; t += blockDim.x) {
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
    float inv_sum = 1.0f / s_scratch[0];
    __syncthreads();

    for (int t = tid; t < seq_len; t += blockDim.x) {
        s_weights[t] *= inv_sum;
    }
    __syncthreads();

    const int out_base = (req * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            const int logical_block = t / block_size;
            const int offset_in_block = t % block_size;
            const int physical_block = req_block_table[logical_block];
            const int v_idx = ((physical_block * block_size + offset_in_block) * num_kv_heads + kv_head) * head_dim + d;
            acc += s_weights[t] * __bfloat162float(v_pool[v_idx]);
        }
        output[out_base + d] = __float2bfloat16(acc);
    }
}
