#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Append K or V data into a paged (block-table-indexed) pool.
//
// Pool layout: (num_blocks, block_size, num_kv_heads, head_dim) — row-major.
// Block table: maps logical block index → physical block index.
//
// For each new token at position `start_pos + seq_offset`:
//   logical_block  = position / block_size
//   offset_in_blk  = position % block_size
//   physical_block = block_table[logical_block]
//   dst_idx = ((physical_block * block_size + offset_in_blk) * num_kv_heads + head) * head_dim + dim

extern "C" __global__ void append_kv_paged_f32(
    float* __restrict__ pool,
    const float* __restrict__ new_data,
    const int* __restrict__ block_table,
    const int start_pos,
    const int block_size,
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

    const int position = start_pos + seq_offset;
    const int logical_block = position / block_size;
    const int offset_in_block = position % block_size;
    const int physical_block = block_table[logical_block];

    const int dst_idx = ((physical_block * block_size + offset_in_block) * num_kv_heads + head) * head_dim + dim;
    pool[dst_idx] = new_data[idx];
}

extern "C" __global__ void append_kv_paged_f16(
    __half* __restrict__ pool,
    const __half* __restrict__ new_data,
    const int* __restrict__ block_table,
    const int start_pos,
    const int block_size,
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

    const int position = start_pos + seq_offset;
    const int logical_block = position / block_size;
    const int offset_in_block = position % block_size;
    const int physical_block = block_table[logical_block];

    const int dst_idx = ((physical_block * block_size + offset_in_block) * num_kv_heads + head) * head_dim + dim;
    pool[dst_idx] = new_data[idx];
}

extern "C" __global__ void append_kv_paged_bf16(
    __nv_bfloat16* __restrict__ pool,
    const __nv_bfloat16* __restrict__ new_data,
    const int* __restrict__ block_table,
    const int start_pos,
    const int block_size,
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

    const int position = start_pos + seq_offset;
    const int logical_block = position / block_size;
    const int offset_in_block = position % block_size;
    const int physical_block = block_table[logical_block];

    const int dst_idx = ((physical_block * block_size + offset_in_block) * num_kv_heads + head) * head_dim + dim;
    pool[dst_idx] = new_data[idx];
}
