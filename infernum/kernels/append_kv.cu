#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void append_kv_f32(
    float* __restrict__ cache,
    const float* __restrict__ new_data,
    const int current_len,
    const int max_seq_len,
    const int num_kv_heads,
    const int head_dim,
    const int new_seq_len
) {
    // Each thread copies one element of the new K or V slice
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = new_seq_len * num_kv_heads * head_dim;
    if (idx >= total) return;

    // Decompose flat index into (seq_offset, head, dim)
    const int hd = num_kv_heads * head_dim;
    const int seq_offset = idx / hd;
    const int remainder = idx % hd;
    const int head = remainder / head_dim;
    const int dim = remainder % head_dim;

    // Cache layout: (max_seq_len, num_kv_heads, head_dim) — row-major
    const int dst_seq = current_len + seq_offset;
    const int dst_idx = (dst_seq * num_kv_heads + head) * head_dim + dim;

    cache[dst_idx] = new_data[idx];
}

// GPU-indirect variant: reads current_len from a device pointer.
extern "C" __global__ void append_kv_indirect_f32(
    float* __restrict__ cache,
    const float* __restrict__ new_data,
    const unsigned int* __restrict__ current_len_ptr,
    const int max_seq_len,
    const int num_kv_heads,
    const int head_dim,
    const int new_seq_len
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = new_seq_len * num_kv_heads * head_dim;
    if (idx >= total) return;

    const int current_len = (int)(*current_len_ptr);
    const int hd = num_kv_heads * head_dim;
    const int seq_offset = idx / hd;
    const int remainder = idx % hd;
    const int head = remainder / head_dim;
    const int dim = remainder % head_dim;

    const int dst_seq = current_len + seq_offset;
    const int dst_idx = (dst_seq * num_kv_heads + head) * head_dim + dim;

    cache[dst_idx] = new_data[idx];
}

extern "C" __global__ void append_kv_f16(
    __half* __restrict__ cache,
    const __half* __restrict__ new_data,
    const int current_len,
    const int max_seq_len,
    const int num_kv_heads,
    const int head_dim,
    const int new_seq_len
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = new_seq_len * num_kv_heads * head_dim;
    if (idx >= total) return;

    const int hd = num_kv_heads * head_dim;
    const int seq_offset = idx / hd;
    const int remainder = idx % hd;
    const int head = remainder / head_dim;
    const int dim = remainder % head_dim;

    const int dst_seq = current_len + seq_offset;
    const int dst_idx = (dst_seq * num_kv_heads + head) * head_dim + dim;

    cache[dst_idx] = new_data[idx];
}

extern "C" __global__ void append_kv_indirect_f16(
    __half* __restrict__ cache,
    const __half* __restrict__ new_data,
    const unsigned int* __restrict__ current_len_ptr,
    const int max_seq_len,
    const int num_kv_heads,
    const int head_dim,
    const int new_seq_len
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = new_seq_len * num_kv_heads * head_dim;
    if (idx >= total) return;

    const int current_len = (int)(*current_len_ptr);
    const int hd = num_kv_heads * head_dim;
    const int seq_offset = idx / hd;
    const int remainder = idx % hd;
    const int head = remainder / head_dim;
    const int dim = remainder % head_dim;

    const int dst_seq = current_len + seq_offset;
    const int dst_idx = (dst_seq * num_kv_heads + head) * head_dim + dim;

    cache[dst_idx] = new_data[idx];
}

extern "C" __global__ void append_kv_bf16(
    __nv_bfloat16* __restrict__ cache,
    const __nv_bfloat16* __restrict__ new_data,
    const int current_len,
    const int max_seq_len,
    const int num_kv_heads,
    const int head_dim,
    const int new_seq_len
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = new_seq_len * num_kv_heads * head_dim;
    if (idx >= total) return;

    const int hd = num_kv_heads * head_dim;
    const int seq_offset = idx / hd;
    const int remainder = idx % hd;
    const int head = remainder / head_dim;
    const int dim = remainder % head_dim;

    const int dst_seq = current_len + seq_offset;
    const int dst_idx = (dst_seq * num_kv_heads + head) * head_dim + dim;

    cache[dst_idx] = new_data[idx];
}

extern "C" __global__ void append_kv_indirect_bf16(
    __nv_bfloat16* __restrict__ cache,
    const __nv_bfloat16* __restrict__ new_data,
    const unsigned int* __restrict__ current_len_ptr,
    const int max_seq_len,
    const int num_kv_heads,
    const int head_dim,
    const int new_seq_len
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = new_seq_len * num_kv_heads * head_dim;
    if (idx >= total) return;

    const int current_len = (int)(*current_len_ptr);
    const int hd = num_kv_heads * head_dim;
    const int seq_offset = idx / hd;
    const int remainder = idx % hd;
    const int head = remainder / head_dim;
    const int dim = remainder % head_dim;

    const int dst_seq = current_len + seq_offset;
    const int dst_idx = (dst_seq * num_kv_heads + head) * head_dim + dim;

    cache[dst_idx] = new_data[idx];
}
