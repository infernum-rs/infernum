#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void rope_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int position_offset
) {
    // Each block handles one (seq_pos, head) pair
    // Each thread handles one pair of dimensions
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int pair_idx = threadIdx.x;  // Which pair of dimensions (0..head_dim/2)
    
    if (pair_idx >= head_dim / 2) return;
    
    const int pos = position_offset + seq_idx;
    
    // Input layout: (seq_len, num_heads, head_dim)
    // Sequential (half-half) convention: pairs are (x[i], x[i + head_dim/2])
    const int half_dim = head_dim / 2;
    const int base_idx = (seq_idx * num_heads + head_idx) * head_dim;
    const int idx0 = base_idx + pair_idx;
    const int idx1 = base_idx + pair_idx + half_dim;
    
    // Cache layout: (max_seq_len, head_dim/2)
    const int cache_idx = pos * (head_dim / 2) + pair_idx;
    
    float cos_val = cos_cache[cache_idx];
    float sin_val = sin_cache[cache_idx];
    
    float x0 = input[idx0];
    float x1 = input[idx1];
    
    output[idx0] = x0 * cos_val - x1 * sin_val;
    output[idx1] = x0 * sin_val + x1 * cos_val;
}

// GPU-indirect variant: reads position_offset from a device pointer.
// The pointer address is stable across CUDA graph replays; only the
// value at that address changes between steps.
extern "C" __global__ void rope_indirect_f32(
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
    
    const int half_dim = head_dim / 2;
    const int base_idx = (seq_idx * num_heads + head_idx) * head_dim;
    const int idx0 = base_idx + pair_idx;
    const int idx1 = base_idx + pair_idx + half_dim;
    
    const int cache_idx = pos * (head_dim / 2) + pair_idx;
    
    float cos_val = cos_cache[cache_idx];
    float sin_val = sin_cache[cache_idx];
    
    float x0 = input[idx0];
    float x1 = input[idx1];
    
    output[idx0] = x0 * cos_val - x1 * sin_val;
    output[idx1] = x0 * sin_val + x1 * cos_val;
}

extern "C" __global__ void rope_f16(
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
    const int half_dim = head_dim / 2;
    const int base_idx = (seq_idx * num_heads + head_idx) * head_dim;
    const int idx0 = base_idx + pair_idx;
    const int idx1 = base_idx + pair_idx + half_dim;
    const int cache_idx = pos * (head_dim / 2) + pair_idx;
    
    // Compute in F32 for precision
    float cos_val = __half2float(cos_cache[cache_idx]);
    float sin_val = __half2float(sin_cache[cache_idx]);
    float x0 = __half2float(input[idx0]);
    float x1 = __half2float(input[idx1]);
    
    output[idx0] = __float2half(x0 * cos_val - x1 * sin_val);
    output[idx1] = __float2half(x0 * sin_val + x1 * cos_val);
}

extern "C" __global__ void rope_indirect_f16(
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
    const int half_dim = head_dim / 2;
    const int base_idx = (seq_idx * num_heads + head_idx) * head_dim;
    const int idx0 = base_idx + pair_idx;
    const int idx1 = base_idx + pair_idx + half_dim;
    const int cache_idx = pos * (head_dim / 2) + pair_idx;
    
    float cos_val = __half2float(cos_cache[cache_idx]);
    float sin_val = __half2float(sin_cache[cache_idx]);
    float x0 = __half2float(input[idx0]);
    float x1 = __half2float(input[idx1]);
    
    output[idx0] = __float2half(x0 * cos_val - x1 * sin_val);
    output[idx1] = __float2half(x0 * sin_val + x1 * cos_val);
}

extern "C" __global__ void rope_bf16(
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
    const int half_dim = head_dim / 2;
    const int base_idx = (seq_idx * num_heads + head_idx) * head_dim;
    const int idx0 = base_idx + pair_idx;
    const int idx1 = base_idx + pair_idx + half_dim;
    const int cache_idx = pos * (head_dim / 2) + pair_idx;
    
    // Compute in F32 for precision
    float cos_val = __bfloat162float(cos_cache[cache_idx]);
    float sin_val = __bfloat162float(sin_cache[cache_idx]);
    float x0 = __bfloat162float(input[idx0]);
    float x1 = __bfloat162float(input[idx1]);
    
    output[idx0] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    output[idx1] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}

extern "C" __global__ void rope_indirect_bf16(
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
    const int half_dim = head_dim / 2;
    const int base_idx = (seq_idx * num_heads + head_idx) * head_dim;
    const int idx0 = base_idx + pair_idx;
    const int idx1 = base_idx + pair_idx + half_dim;
    const int cache_idx = pos * (head_dim / 2) + pair_idx;
    
    float cos_val = __bfloat162float(cos_cache[cache_idx]);
    float sin_val = __bfloat162float(sin_cache[cache_idx]);
    float x0 = __bfloat162float(input[idx0]);
    float x1 = __bfloat162float(input[idx1]);
    
    output[idx0] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    output[idx1] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}
