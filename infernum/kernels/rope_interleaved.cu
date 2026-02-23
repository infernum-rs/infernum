#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Interleaved RoPE: pairs are (x[2i], x[2i+1]) — used by DeepSeek V3.
// Each thread handles one pair of adjacent elements.

extern "C" __global__ void rope_interleaved_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int position_offset
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int pair_idx = threadIdx.x;

    if (pair_idx >= head_dim / 2) return;

    const int pos = position_offset + seq_idx;

    // Input layout: (seq_len, num_heads, head_dim)
    // Interleaved convention: pairs are (x[2i], x[2i+1])
    const int base_idx = (seq_idx * num_heads + head_idx) * head_dim;
    const int idx0 = base_idx + 2 * pair_idx;
    const int idx1 = base_idx + 2 * pair_idx + 1;

    // Cache layout: (max_seq_len, head_dim/2)
    const int cache_idx = pos * (head_dim / 2) + pair_idx;

    float cos_val = cos_cache[cache_idx];
    float sin_val = sin_cache[cache_idx];

    float x0 = input[idx0];
    float x1 = input[idx1];

    output[idx0] = x0 * cos_val - x1 * sin_val;
    output[idx1] = x0 * sin_val + x1 * cos_val;
}

extern "C" __global__ void rope_interleaved_indirect_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const unsigned int* __restrict__ position_offset_ptr
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int pair_idx = threadIdx.x;

    if (pair_idx >= head_dim / 2) return;

    const int position_offset = (int)(*position_offset_ptr);
    const int pos = position_offset + seq_idx;

    const int base_idx = (seq_idx * num_heads + head_idx) * head_dim;
    const int idx0 = base_idx + 2 * pair_idx;
    const int idx1 = base_idx + 2 * pair_idx + 1;

    const int cache_idx = pos * (head_dim / 2) + pair_idx;

    float cos_val = cos_cache[cache_idx];
    float sin_val = sin_cache[cache_idx];

    float x0 = input[idx0];
    float x1 = input[idx1];

    output[idx0] = x0 * cos_val - x1 * sin_val;
    output[idx1] = x0 * sin_val + x1 * cos_val;
}

extern "C" __global__ void rope_interleaved_f16(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ cos_cache,
    const __half* __restrict__ sin_cache,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int position_offset
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int pair_idx = threadIdx.x;

    if (pair_idx >= head_dim / 2) return;

    const int pos = position_offset + seq_idx;
    const int base_idx = (seq_idx * num_heads + head_idx) * head_dim;
    const int idx0 = base_idx + 2 * pair_idx;
    const int idx1 = base_idx + 2 * pair_idx + 1;
    const int cache_idx = pos * (head_dim / 2) + pair_idx;

    float cos_val = __half2float(cos_cache[cache_idx]);
    float sin_val = __half2float(sin_cache[cache_idx]);
    float x0 = __half2float(input[idx0]);
    float x1 = __half2float(input[idx1]);

    output[idx0] = __float2half(x0 * cos_val - x1 * sin_val);
    output[idx1] = __float2half(x0 * sin_val + x1 * cos_val);
}

extern "C" __global__ void rope_interleaved_indirect_f16(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ cos_cache,
    const __half* __restrict__ sin_cache,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const unsigned int* __restrict__ position_offset_ptr
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int pair_idx = threadIdx.x;

    if (pair_idx >= head_dim / 2) return;

    const int position_offset = (int)(*position_offset_ptr);
    const int pos = position_offset + seq_idx;
    const int base_idx = (seq_idx * num_heads + head_idx) * head_dim;
    const int idx0 = base_idx + 2 * pair_idx;
    const int idx1 = base_idx + 2 * pair_idx + 1;
    const int cache_idx = pos * (head_dim / 2) + pair_idx;

    float cos_val = __half2float(cos_cache[cache_idx]);
    float sin_val = __half2float(sin_cache[cache_idx]);
    float x0 = __half2float(input[idx0]);
    float x1 = __half2float(input[idx1]);

    output[idx0] = __float2half(x0 * cos_val - x1 * sin_val);
    output[idx1] = __float2half(x0 * sin_val + x1 * cos_val);
}

extern "C" __global__ void rope_interleaved_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int position_offset
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int pair_idx = threadIdx.x;

    if (pair_idx >= head_dim / 2) return;

    const int pos = position_offset + seq_idx;
    const int base_idx = (seq_idx * num_heads + head_idx) * head_dim;
    const int idx0 = base_idx + 2 * pair_idx;
    const int idx1 = base_idx + 2 * pair_idx + 1;
    const int cache_idx = pos * (head_dim / 2) + pair_idx;

    float cos_val = __bfloat162float(cos_cache[cache_idx]);
    float sin_val = __bfloat162float(sin_cache[cache_idx]);
    float x0 = __bfloat162float(input[idx0]);
    float x1 = __bfloat162float(input[idx1]);

    output[idx0] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    output[idx1] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}

extern "C" __global__ void rope_interleaved_indirect_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const unsigned int* __restrict__ position_offset_ptr
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int pair_idx = threadIdx.x;

    if (pair_idx >= head_dim / 2) return;

    const int position_offset = (int)(*position_offset_ptr);
    const int pos = position_offset + seq_idx;
    const int base_idx = (seq_idx * num_heads + head_idx) * head_dim;
    const int idx0 = base_idx + 2 * pair_idx;
    const int idx1 = base_idx + 2 * pair_idx + 1;
    const int cache_idx = pos * (head_dim / 2) + pair_idx;

    float cos_val = __bfloat162float(cos_cache[cache_idx]);
    float sin_val = __bfloat162float(sin_cache[cache_idx]);
    float x0 = __bfloat162float(input[idx0]);
    float x1 = __bfloat162float(input[idx1]);

    output[idx0] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    output[idx1] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}
