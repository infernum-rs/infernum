#include <metal_stdlib>
using namespace metal;

// Batched paged KV append: one token per sequence, B sequences in one launch.
//
// new_data shape: (batch_size, num_kv_heads * head_dim) — one token per seq.
// positions: (batch_size,) — position at which to store this token.
// block_tables: (batch_size, max_blocks_per_seq) — per-sequence block tables.
//
// Pool layout: (num_blocks, block_size, num_kv_heads, head_dim) row-major.

struct AppendKvPagedParams {
    uint batch_size;
    uint block_size;
    uint num_kv_heads;
    uint head_dim;
    uint max_blocks_per_seq;
};

kernel void append_kv_paged_batched_f32(
    device float*       pool       [[buffer(0)]],
    device const float* new_data   [[buffer(1)]],
    device const int*   block_tables [[buffer(2)]],
    device const int*   positions  [[buffer(3)]],
    constant AppendKvPagedParams& p [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint elem_idx = tid.x;
    uint seq_idx  = tid.y;

    uint total_per_token = p.num_kv_heads * p.head_dim;
    if (elem_idx >= total_per_token || seq_idx >= p.batch_size) return;

    int position = positions[seq_idx];
    uint logical_block  = uint(position) / p.block_size;
    uint offset_in_blk  = uint(position) % p.block_size;
    int physical_block   = block_tables[seq_idx * p.max_blocks_per_seq + logical_block];

    uint head = elem_idx / p.head_dim;
    uint dim  = elem_idx % p.head_dim;

    uint dst = ((uint(physical_block) * p.block_size + offset_in_blk) * p.num_kv_heads + head) * p.head_dim + dim;
    uint src = seq_idx * total_per_token + elem_idx;
    pool[dst] = new_data[src];
}
