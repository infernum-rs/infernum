#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Gather K or V data from a paged (block-table-indexed) pool into a contiguous buffer.
//
// Pool layout: (num_blocks, block_size, num_kv_heads, head_dim) — row-major.
// Block table: maps logical block index → physical block index.
// Output: (seq_len, num_kv_heads, head_dim) — row-major, contiguous.
//
// For each token t in [0, seq_len):
//   logical_block  = t / block_size
//   offset_in_blk  = t % block_size
//   physical_block = block_table[logical_block]
//   src_idx = ((physical_block * block_size + offset_in_blk) * num_kv_heads + head) * head_dim + dim
//   dst_idx = (t * num_kv_heads + head) * head_dim + dim

extern "C" __global__ void gather_paged_kv_f32(
    float* __restrict__ output,
    const float* __restrict__ pool,
    const int* __restrict__ block_table,
    const int seq_len,
    const int block_size,
    const int num_kv_heads,
    const int head_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seq_len * num_kv_heads * head_dim;
    if (idx >= total) return;

    const int hd = num_kv_heads * head_dim;
    const int t = idx / hd;
    const int remainder = idx % hd;
    const int head = remainder / head_dim;
    const int dim = remainder % head_dim;

    const int logical_block = t / block_size;
    const int offset_in_block = t % block_size;
    const int physical_block = block_table[logical_block];

    const int src_idx = ((physical_block * block_size + offset_in_block) * num_kv_heads + head) * head_dim + dim;
    output[idx] = pool[src_idx];
}

extern "C" __global__ void gather_paged_kv_f16(
    __half* __restrict__ output,
    const __half* __restrict__ pool,
    const int* __restrict__ block_table,
    const int seq_len,
    const int block_size,
    const int num_kv_heads,
    const int head_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seq_len * num_kv_heads * head_dim;
    if (idx >= total) return;

    const int hd = num_kv_heads * head_dim;
    const int t = idx / hd;
    const int remainder = idx % hd;
    const int head = remainder / head_dim;
    const int dim = remainder % head_dim;

    const int logical_block = t / block_size;
    const int offset_in_block = t % block_size;
    const int physical_block = block_table[logical_block];

    const int src_idx = ((physical_block * block_size + offset_in_block) * num_kv_heads + head) * head_dim + dim;
    output[idx] = pool[src_idx];
}

extern "C" __global__ void gather_paged_kv_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ pool,
    const int* __restrict__ block_table,
    const int seq_len,
    const int block_size,
    const int num_kv_heads,
    const int head_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seq_len * num_kv_heads * head_dim;
    if (idx >= total) return;

    const int hd = num_kv_heads * head_dim;
    const int t = idx / hd;
    const int remainder = idx % hd;
    const int head = remainder / head_dim;
    const int dim = remainder % head_dim;

    const int logical_block = t / block_size;
    const int offset_in_block = t % block_size;
    const int physical_block = block_table[logical_block];

    const int src_idx = ((physical_block * block_size + offset_in_block) * num_kv_heads + head) * head_dim + dim;
    output[idx] = pool[src_idx];
}
