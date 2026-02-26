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

// --- Batched paged append: one token per sequence, B sequences in one launch ---
//
// new_data shape: (batch_size, num_kv_heads, head_dim) — one token per sequence.
// positions: (batch_size,) — each entry is the position at which to store this token.
// block_tables: (batch_size, max_blocks_per_seq) — flattened per-sequence block tables.
// Grid: (ceil(num_kv_heads * head_dim / 256), batch_size, 1)
// blockIdx.y selects the sequence.

extern "C" __global__ void append_kv_paged_batched_f32(
    float* __restrict__ pool,
    const float* __restrict__ new_data,
    const int* __restrict__ block_tables,
    const int* __restrict__ positions,
    const int block_size,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq
) {
    const int seq_idx = blockIdx.y;
    const int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_per_token = num_kv_heads * head_dim;
    if (elem_idx >= total_per_token) return;

    const int head = elem_idx / head_dim;
    const int dim = elem_idx % head_dim;

    const int position = positions[seq_idx];
    const int logical_block = position / block_size;
    const int offset_in_block = position % block_size;
    const int physical_block = block_tables[seq_idx * max_blocks_per_seq + logical_block];

    const int dst_idx = ((physical_block * block_size + offset_in_block) * num_kv_heads + head) * head_dim + dim;
    const int src_idx = seq_idx * total_per_token + elem_idx;
    pool[dst_idx] = new_data[src_idx];
}

extern "C" __global__ void append_kv_paged_batched_f16(
    __half* __restrict__ pool,
    const __half* __restrict__ new_data,
    const int* __restrict__ block_tables,
    const int* __restrict__ positions,
    const int block_size,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq
) {
    const int seq_idx = blockIdx.y;
    const int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_per_token = num_kv_heads * head_dim;
    if (elem_idx >= total_per_token) return;

    const int head = elem_idx / head_dim;
    const int dim = elem_idx % head_dim;

    const int position = positions[seq_idx];
    const int logical_block = position / block_size;
    const int offset_in_block = position % block_size;
    const int physical_block = block_tables[seq_idx * max_blocks_per_seq + logical_block];

    const int dst_idx = ((physical_block * block_size + offset_in_block) * num_kv_heads + head) * head_dim + dim;
    const int src_idx = seq_idx * total_per_token + elem_idx;
    pool[dst_idx] = new_data[src_idx];
}

extern "C" __global__ void append_kv_paged_batched_bf16(
    __nv_bfloat16* __restrict__ pool,
    const __nv_bfloat16* __restrict__ new_data,
    const int* __restrict__ block_tables,
    const int* __restrict__ positions,
    const int block_size,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq
) {
    const int seq_idx = blockIdx.y;
    const int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_per_token = num_kv_heads * head_dim;
    if (elem_idx >= total_per_token) return;

    const int head = elem_idx / head_dim;
    const int dim = elem_idx % head_dim;

    const int position = positions[seq_idx];
    const int logical_block = position / block_size;
    const int offset_in_block = position % block_size;
    const int physical_block = block_tables[seq_idx * max_blocks_per_seq + logical_block];

    const int dst_idx = ((physical_block * block_size + offset_in_block) * num_kv_heads + head) * head_dim + dim;
    const int src_idx = seq_idx * total_per_token + elem_idx;
    pool[dst_idx] = new_data[src_idx];
}
