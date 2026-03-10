#include <metal_stdlib>
using namespace metal;

/// Fused RoPE + paged KV-cache append.
///
/// Combines apply_rope_qk_batched + append_kv_paged_batched_fused into a
/// single dispatch.  Q gets RoPE and writes to a linear output buffer.
/// K gets RoPE and writes directly to the paged k_pool cache.
/// V copies directly to the paged v_pool cache.
///
/// Thread grid (1-D):
///   Zone 1: batch * (q_heads + k_heads) * half_dim   — RoPE for Q and K
///   Zone 2: batch * k_heads * head_dim                — V copy
///
/// Total = batch * (q_heads * half_dim + k_heads * half_dim + k_heads * head_dim)

struct RopeKvAppendParams {
    uint q_heads;
    uint k_heads;           // == num_kv_heads
    uint head_dim;
    uint half_dim;
    uint block_size;
    uint max_blocks_per_seq;
};

kernel void rope_kv_append_fused_f32(
    device const float* q_input      [[buffer(0)]],
    device const float* k_input      [[buffer(1)]],
    device const float* v_input      [[buffer(2)]],
    device const float* cos_cache    [[buffer(3)]],
    device const float* sin_cache    [[buffer(4)]],
    device const int*   positions    [[buffer(5)]],
    device float* q_output           [[buffer(6)]],
    device float* k_pool             [[buffer(7)]],
    device float* v_pool             [[buffer(8)]],
    device const int* block_tables   [[buffer(9)]],
    constant RopeKvAppendParams& p   [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    const uint q_heads  = p.q_heads;
    const uint k_heads  = p.k_heads;
    const uint head_dim = p.head_dim;
    const uint half_dim = p.half_dim;

    // Zone 1: RoPE for Q and K
    const uint q_stride  = q_heads * half_dim;
    const uint rope_per_batch = q_stride + k_heads * half_dim;

    const uint v_per_batch = k_heads * head_dim;
    const uint total_per_batch = rope_per_batch + v_per_batch;

    const uint batch_idx = tid / total_per_batch;
    const uint local     = tid % total_per_batch;

    const int position = positions[batch_idx];
    const uint pos = uint(position);

    if (local < rope_per_batch) {
        // ---- Zone 1: RoPE ----
        if (local < q_stride) {
            // Q part
            const uint p_idx = local % half_dim;
            const uint h     = local / half_dim;
            const uint base  = (batch_idx * q_heads + h) * head_dim;

            float cv = cos_cache[pos * half_dim + p_idx];
            float sv = sin_cache[pos * half_dim + p_idx];

            float a = q_input[base + p_idx];
            float b = q_input[base + half_dim + p_idx];

            q_output[base + p_idx]            = a * cv - b * sv;
            q_output[base + half_dim + p_idx] = b * cv + a * sv;
        } else {
            // K part — RoPE then write to paged cache
            const uint k_local = local - q_stride;
            const uint p_idx   = k_local % half_dim;
            const uint h       = k_local / half_dim;
            const uint base    = (batch_idx * k_heads + h) * head_dim;

            float cv = cos_cache[pos * half_dim + p_idx];
            float sv = sin_cache[pos * half_dim + p_idx];

            float a = k_input[base + p_idx];
            float b = k_input[base + half_dim + p_idx];

            float k_lo = a * cv - b * sv;
            float k_hi = b * cv + a * sv;

            // Paged cache destination
            uint logical_block  = pos / p.block_size;
            uint offset_in_blk  = pos % p.block_size;
            int  physical_block = block_tables[batch_idx * p.max_blocks_per_seq + logical_block];

            uint cache_base = ((uint(physical_block) * p.block_size + offset_in_blk)
                               * k_heads + h) * head_dim;

            k_pool[cache_base + p_idx]            = k_lo;
            k_pool[cache_base + half_dim + p_idx] = k_hi;
        }
    } else {
        // ---- Zone 2: V copy to paged cache ----
        const uint v_local = local - rope_per_batch;
        const uint elem    = v_local;  // in [0, k_heads * head_dim)
        const uint h       = elem / head_dim;
        const uint d       = elem % head_dim;

        uint logical_block  = pos / p.block_size;
        uint offset_in_blk  = pos % p.block_size;
        int  physical_block = block_tables[batch_idx * p.max_blocks_per_seq + logical_block];

        uint dst = ((uint(physical_block) * p.block_size + offset_in_blk)
                    * k_heads + h) * head_dim + d;
        uint src = batch_idx * k_heads * head_dim + elem;

        v_pool[dst] = v_input[src];
    }
}


kernel void rope_kv_append_fused_f16(
    device const half*  q_input      [[buffer(0)]],
    device const half*  k_input      [[buffer(1)]],
    device const half*  v_input      [[buffer(2)]],
    device const float* cos_cache    [[buffer(3)]],
    device const float* sin_cache    [[buffer(4)]],
    device const int*   positions    [[buffer(5)]],
    device half*  q_output           [[buffer(6)]],
    device half*  k_pool             [[buffer(7)]],
    device half*  v_pool             [[buffer(8)]],
    device const int* block_tables   [[buffer(9)]],
    constant RopeKvAppendParams& p   [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    const uint q_heads  = p.q_heads;
    const uint k_heads  = p.k_heads;
    const uint head_dim = p.head_dim;
    const uint half_dim = p.half_dim;

    const uint q_stride  = q_heads * half_dim;
    const uint rope_per_batch = q_stride + k_heads * half_dim;
    const uint v_per_batch = k_heads * head_dim;
    const uint total_per_batch = rope_per_batch + v_per_batch;

    const uint batch_idx = tid / total_per_batch;
    const uint local     = tid % total_per_batch;

    const int position = positions[batch_idx];
    const uint pos = uint(position);

    if (local < rope_per_batch) {
        // ---- Zone 1: RoPE ----
        if (local < q_stride) {
            // Q part
            const uint p_idx = local % half_dim;
            const uint h     = local / half_dim;
            const uint base  = (batch_idx * q_heads + h) * head_dim;

            float cv = cos_cache[pos * half_dim + p_idx];
            float sv = sin_cache[pos * half_dim + p_idx];

            float a = float(q_input[base + p_idx]);
            float b = float(q_input[base + half_dim + p_idx]);

            q_output[base + p_idx]            = half(a * cv - b * sv);
            q_output[base + half_dim + p_idx] = half(b * cv + a * sv);
        } else {
            // K part — RoPE then write to paged cache
            const uint k_local = local - q_stride;
            const uint p_idx   = k_local % half_dim;
            const uint h       = k_local / half_dim;
            const uint base    = (batch_idx * k_heads + h) * head_dim;

            float cv = cos_cache[pos * half_dim + p_idx];
            float sv = sin_cache[pos * half_dim + p_idx];

            float a = float(k_input[base + p_idx]);
            float b = float(k_input[base + half_dim + p_idx]);

            half k_lo = half(a * cv - b * sv);
            half k_hi = half(b * cv + a * sv);

            uint logical_block  = pos / p.block_size;
            uint offset_in_blk  = pos % p.block_size;
            int  physical_block = block_tables[batch_idx * p.max_blocks_per_seq + logical_block];

            uint cache_base = ((uint(physical_block) * p.block_size + offset_in_blk)
                               * k_heads + h) * head_dim;

            k_pool[cache_base + p_idx]            = k_lo;
            k_pool[cache_base + half_dim + p_idx] = k_hi;
        }
    } else {
        // ---- Zone 2: V copy to paged cache ----
        const uint v_local = local - rope_per_batch;
        const uint elem    = v_local;
        const uint h       = elem / head_dim;
        const uint d       = elem % head_dim;

        uint logical_block  = pos / p.block_size;
        uint offset_in_blk  = pos % p.block_size;
        int  physical_block = block_tables[batch_idx * p.max_blocks_per_seq + logical_block];

        uint dst = ((uint(physical_block) * p.block_size + offset_in_blk)
                    * k_heads + h) * head_dim + d;
        uint src = batch_idx * k_heads * head_dim + elem;

        v_pool[dst] = v_input[src];
    }
}
