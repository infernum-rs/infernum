#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// Gather K+V for contiguous-attention decode path (BF16)
//
// Reads seq_len from a GPU buffer, then gathers both K and V from paged pools
// into pre-allocated contiguous output buffers of shape (max_capacity, num_kv_heads, head_dim).
// Only [0, seq_len) rows are filled; the rest are left undefined and not read
// by the subsequent fused_decode_attention kernel.
//
// Using GPU seq_lens makes this safe inside a CUDA graph: the graph captures
// the stable buffer address for seq_lens (updated via htod_copy_into each step)
// and the function produces the correct results on replay without re-capture.
//
// Grid: ceil((max_capacity × num_kv_heads × head_dim) / blockDim.x)
// Block: 256 threads
// ---------------------------------------------------------------------------
extern "C" __global__ void gather_kv_for_attn_bf16(
    __nv_bfloat16* __restrict__ k_out,         // [max_capacity, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ v_out,         // [max_capacity, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_pool,  // paged K pool
    const __nv_bfloat16* __restrict__ v_pool,  // paged V pool
    const int* __restrict__ block_table,       // [max_blocks_per_seq] — for req=0
    const int* __restrict__ seq_lens,          // [1] — current seq len for req=0
    const int block_size,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq,
    const int max_capacity           // allocated output size (max_blocks × block_size)
) {
    const int seq_len = seq_lens[0];
    const int hd_stride = num_kv_heads * head_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= max_capacity * hd_stride) return;

    const int t    = idx / hd_stride;
    const int head = (idx % hd_stride) / head_dim;
    const int dim  = idx % head_dim;

    if (t >= seq_len) return;   // beyond valid range — leave undefined

    const int logical_block = t / block_size;
    const int offset        = t % block_size;
    const int phys          = block_table[logical_block];
    const int src = ((phys * block_size + offset) * num_kv_heads + head) * head_dim + dim;

    k_out[idx] = k_pool[src];
    v_out[idx] = v_pool[src];
}

// ---------------------------------------------------------------------------
// Original single-tensor gather (kept for eager / prefill paths)
// ---------------------------------------------------------------------------

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
