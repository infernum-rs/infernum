#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define INFINITY_F __int_as_float(0x7f800000)
#define NEG_INF_F  (-1e30f)

// Apply logit soft-capping: tanh(x / cap) * cap
// Disabled when softcap <= 0.0
#define MAYBE_SOFTCAP(dot, softcap) \
    if (softcap > 0.0f) { dot = tanhf(dot / softcap) * softcap; }

// ---------------------------------------------------------------------------
// Contiguous-KV decode attention (BF16) — gather-then-attend fast path
//
// Same algorithm as paged_decode_attention_bf16 but K and V are in a
// contiguous pre-allocated buffer (filled by gather_kv_for_attn_bf16 first).
// Sequential memory access eliminates the scattered-page cache misses that
// dominate the paged kernel's runtime.
//
// Grid: (num_heads, batch_size=1)
// Block: 256 threads (same as paged kernel after the shared-mem fix)
// Shared memory: head_dim + max_capacity + blockDim.x (f32 each)
// ---------------------------------------------------------------------------
extern "C" __global__ void contiguous_decode_attn_bf16(
    __nv_bfloat16* __restrict__ output,       // (1, num_heads, head_dim)
    const __nv_bfloat16* __restrict__ q,      // (1, num_heads, head_dim)
    const __nv_bfloat16* __restrict__ k_cont, // (max_capacity, num_kv_heads, head_dim)
    const __nv_bfloat16* __restrict__ v_cont, // (max_capacity, num_kv_heads, head_dim)
    const int* __restrict__ seq_lens,         // (1,) — reads current seq_len from GPU
    const float scale,
    const float softcap,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_capacity,   // allocated size of k_cont/v_cont (max_blocks * block_size)
    const int window_size
) {
    const int head = blockIdx.x;
    const int req  = blockIdx.y;
    const int tid  = threadIdx.x;

    const int seq_len = seq_lens[req];
    if (seq_len == 0) return;

    const int kv_head   = head * num_kv_heads / num_heads;
    const int win_start = (window_size > 0) ? max(0, seq_len - window_size) : 0;
    const int active_len = seq_len - win_start;

    extern __shared__ float shared[];
    float* s_q       = shared;                                  // head_dim floats
    float* s_weights = shared + head_dim;                       // max_capacity floats
    float* s_scratch = shared + head_dim + max_capacity;        // blockDim.x floats

    // Load Q
    const int q_base = (req * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x)
        s_q[d] = __bfloat162float(q[q_base + d]);
    __syncthreads();

    // ---- Pass 1: Q·K scores (sequential reads from k_cont) ----
    const int kv_stride = num_kv_heads * head_dim;
    float local_max = -1e30f;

    for (int i = tid; i < active_len; i += blockDim.x) {
        const int t      = win_start + i;
        const int k_base = (t * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += s_q[d] * __bfloat162float(k_cont[k_base + d]);
        dot *= scale;
        MAYBE_SOFTCAP(dot, softcap);
        s_weights[i] = dot;
        local_max = fmaxf(local_max, dot);
    }

    // Reduce local_max across block
    s_scratch[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_scratch[tid] = fmaxf(s_scratch[tid], s_scratch[tid + stride]);
        __syncthreads();
    }
    const float max_val = s_scratch[0];
    __syncthreads();

    // Softmax weights + sum
    float local_sum = 0.0f;
    for (int i = tid; i < active_len; i += blockDim.x) {
        float w = expf(s_weights[i] - max_val);
        s_weights[i] = w;
        local_sum += w;
    }
    s_scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_scratch[tid] += s_scratch[tid + stride];
        __syncthreads();
    }
    const float inv_sum = 1.0f / s_scratch[0];
    __syncthreads();

    for (int i = tid; i < active_len; i += blockDim.x)
        s_weights[i] *= inv_sum;
    __syncthreads();

    // ---- Pass 2: V accumulation (sequential reads from v_cont) ----
    const int out_base = (req * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < active_len; i++) {
            const int t      = win_start + i;
            const int v_base = (t * num_kv_heads + kv_head) * head_dim + d;
            acc += s_weights[i] * __bfloat162float(v_cont[v_base]);
        }
        output[out_base + d] = __float2bfloat16(acc);
    }
}

// ---------------------------------------------------------------------------
// Partitioned paged decode attention (BF16)
//
// Splits the KV sequence across multiple thread blocks to maximize SM
// utilisation.  For batch_size=1 with num_heads=15, the standard kernel
// launches only 15 blocks (25% of L4's 60 SMs).  The partitioned kernel
// launches num_heads * batch_size * num_kv_blocks blocks (e.g. 255 for
// seq_len=264, block_size=16), filling all SMs.
//
// Two-kernel approach:
//   Part 1 – paged_decode_attn_part_bf16:
//     Grid: (num_heads, batch_size, num_kv_blocks)
//     Each block handles one KV block (block_size tokens).
//     Outputs: partial_max[...], partial_sum[...], partial_out[..., head_dim]
//
//   Part 2 – paged_decode_attn_reduce_bf16:
//     Grid: (num_heads, batch_size, 1)
//     Combines partial results with log-sum-exp rescaling.
//
// Each block uses exactly 32 threads (one warp), keeping register pressure
// low and enabling up to THREADS_PER_SM/32 blocks per SM.
// head_dim must be divisible by 32.
// ---------------------------------------------------------------------------

extern "C" __global__ void paged_decode_attn_part_bf16(
    float* __restrict__ partial_max,      // (batch_size, num_heads, num_kv_blocks)
    float* __restrict__ partial_sum,      // (batch_size, num_heads, num_kv_blocks)
    float* __restrict__ partial_out,      // (batch_size, num_heads, num_kv_blocks, head_dim) float
    const __nv_bfloat16* __restrict__ q,  // (batch_size, num_heads, head_dim)
    const __nv_bfloat16* __restrict__ k_pool, // (num_blocks, block_size, num_kv_heads, head_dim)
    const __nv_bfloat16* __restrict__ v_pool, // same layout
    const int* __restrict__ block_tables, // (batch_size, max_blocks_per_seq)
    const int* __restrict__ seq_lens,     // (batch_size,)
    const float scale,
    const float softcap,
    const int block_size,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq,
    const int window_size,
    const int num_kv_blocks              // ceil(max_active_len / block_size) — grid depth
) {
    const int head      = blockIdx.x;
    const int req       = blockIdx.y;
    const int kv_blk    = blockIdx.z;   // which KV block this thread block handles
    const int tid       = threadIdx.x;  // [0..31]

    const int seq_len   = seq_lens[req];
    const int kv_head   = head * num_kv_heads / num_heads;
    const int win_start = (window_size > 0) ? max(0, seq_len - window_size) : 0;
    const int active_len = seq_len - win_start;

    // Tokens handled by this partition: [part_start, part_end)
    const int part_start = kv_blk * block_size;
    const int part_end   = min(part_start + block_size, active_len);
    const int part_len   = max(0, part_end - part_start);

    const int part_idx  = (req * num_heads + head) * num_kv_blocks + kv_blk;
    const int out_base  = part_idx * head_dim;

    if (part_len == 0) {
        partial_max[part_idx] = NEG_INF_F;
        partial_sum[part_idx] = 0.0f;
        for (int d = tid; d < head_dim; d += 32) partial_out[out_base + d] = 0.0f;
        return;
    }

    // Each of the 32 threads owns (head_dim / 32) consecutive Q elements.
    // head_dim must be a multiple of 32.
    const int hdpt = head_dim / 32;  // elements per thread

    // Load Q into registers (hdpt floats per thread)
    float q_reg[8];  // supports up to head_dim = 256
    const int q_base = (req * num_heads + head) * head_dim;
    for (int j = 0; j < hdpt; j++)
        q_reg[j] = __bfloat162float(q[q_base + tid * hdpt + j]);

    const int* req_bt = block_tables + req * max_blocks_per_seq;

    // ---- Pass 1: compute Q·K scores for all tokens in this partition ----
    float scores[16];  // max block_size
    float local_max = NEG_INF_F;

    for (int i = 0; i < part_len; i++) {
        const int t               = win_start + part_start + i;
        const int logical_block   = t / block_size;
        const int offset_in_block = t % block_size;
        const int phys_block      = req_bt[logical_block];
        const int k_base          = ((phys_block * block_size + offset_in_block)
                                      * num_kv_heads + kv_head) * head_dim;

        float partial = 0.0f;
        for (int j = 0; j < hdpt; j++)
            partial += q_reg[j] * __bfloat162float(k_pool[k_base + tid * hdpt + j]);

        // Warp-reduce partial dot across 32 threads -> full dot for token i
        for (int s = 16; s > 0; s >>= 1)
            partial += __shfl_down_sync(0xFFFFFFFF, partial, s);
        float dot = __shfl_sync(0xFFFFFFFF, partial, 0);  // broadcast to all threads

        dot *= scale;
        MAYBE_SOFTCAP(dot, softcap);
        scores[i] = dot;
        local_max = fmaxf(local_max, dot);
    }

    // ---- Pass 2: softmax weights + V accumulation (per thread dimension) ----
    float local_sum = 0.0f;
    float v_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = 0; i < part_len; i++) {
        const float w = expf(scores[i] - local_max);
        local_sum += w;

        const int t               = win_start + part_start + i;
        const int logical_block   = t / block_size;
        const int offset_in_block = t % block_size;
        const int phys_block      = req_bt[logical_block];
        const int v_base          = ((phys_block * block_size + offset_in_block)
                                      * num_kv_heads + kv_head) * head_dim;

        for (int j = 0; j < hdpt; j++)
            v_acc[j] += w * __bfloat162float(v_pool[v_base + tid * hdpt + j]);
    }

    // Write partial results
    partial_max[part_idx] = local_max;
    partial_sum[part_idx] = local_sum;
    for (int j = 0; j < hdpt; j++)
        partial_out[out_base + tid * hdpt + j] = v_acc[j];
}

// Reduction kernel: combine partitions with log-sum-exp rescaling.
// Grid: (num_heads, batch_size, 1)   Block: 32 threads
extern "C" __global__ void paged_decode_attn_reduce_bf16(
    __nv_bfloat16* __restrict__ output,       // (batch_size, num_heads, head_dim)
    const float* __restrict__ partial_max,    // (batch_size, num_heads, num_kv_blocks)
    const float* __restrict__ partial_sum,    // (batch_size, num_heads, num_kv_blocks)
    const float* __restrict__ partial_out,    // (batch_size, num_heads, num_kv_blocks, head_dim)
    const int num_heads,
    const int head_dim,
    const int num_kv_blocks
) {
    const int head = blockIdx.x;
    const int req  = blockIdx.y;
    const int tid  = threadIdx.x;
    const int hdpt = head_dim / 32;

    const int base_p = (req * num_heads + head) * num_kv_blocks;

    // Find global max across all partitions
    float g_max = NEG_INF_F;
    for (int k = 0; k < num_kv_blocks; k++)
        g_max = fmaxf(g_max, partial_max[base_p + k]);

    // Compute global normalised sum and weighted output
    float g_sum = 0.0f;
    float g_out[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int k = 0; k < num_kv_blocks; k++) {
        const float corr = expf(partial_max[base_p + k] - g_max);
        const float ws   = partial_sum[base_p + k] * corr;
        g_sum += ws;

        const int off = (base_p + k) * head_dim;
        for (int j = 0; j < hdpt; j++)
            g_out[j] += partial_out[off + tid * hdpt + j] * corr;
    }

    // Normalise and write output
    const float inv_sum = (g_sum > 0.0f) ? 1.0f / g_sum : 0.0f;
    const int out_base = (req * num_heads + head) * head_dim;
    for (int j = 0; j < hdpt; j++)
        output[out_base + tid * hdpt + j] = __float2bfloat16(g_out[j] * inv_sum);
}

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
//   2. For each token t in [win_start, seq_len):
//      - logical_block = t / block_size
//      - offset_in_block = t % block_size
//      - physical_block = block_tables[request * max_blocks_per_seq + logical_block]
//      - k_idx = ((physical_block * block_size + offset_in_block) * num_kv_heads + kv_head) * head_dim
//      - Compute dot(Q, K), apply optional softcap, track online softmax
//   3. Accumulate weighted V, write output

extern "C" __global__ void paged_decode_attention_f32(
    float* __restrict__ output,          // (batch_size, num_heads, head_dim)
    const float* __restrict__ q,         // (batch_size, num_heads, head_dim)
    const float* __restrict__ k_pool,    // (num_blocks, block_size, num_kv_heads, head_dim)
    const float* __restrict__ v_pool,    // same layout
    const int* __restrict__ block_tables,// (batch_size, max_blocks_per_seq)
    const int* __restrict__ seq_lens,    // (batch_size,)
    const float scale,
    const float softcap,                 // <= 0 means disabled
    const int block_size,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq,
    const int window_size                // <= 0 means full attention (no window)
) {
    const int head = blockIdx.x;
    const int req = blockIdx.y;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[req];
    if (seq_len == 0) return;

    const int kv_head = head * num_kv_heads / num_heads;
    const int win_start = (window_size > 0) ? max(0, seq_len - window_size) : 0;
    const int active_len = seq_len - win_start;

    extern __shared__ float shared[];
    float* s_q = shared;                               // head_dim
    float* s_weights = shared + head_dim;              // active_len (filled dynamically)
    float* s_scratch = shared + head_dim + active_len; // blockDim.x

    // Load Q
    const int q_base = (req * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = q[q_base + d];
    }
    __syncthreads();

    // Pass 1: Compute Q·K scores and find max (for online softmax)
    const int* req_block_table = block_tables + req * max_blocks_per_seq;
    float local_max = -INFINITY_F;

    for (int i = tid; i < active_len; i += blockDim.x) {
        const int t = win_start + i;
        const int logical_block = t / block_size;
        const int offset_in_block = t % block_size;
        const int physical_block = req_block_table[logical_block];
        const int k_base = ((physical_block * block_size + offset_in_block) * num_kv_heads + kv_head) * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * k_pool[k_base + d];
        }
        dot *= scale;
        MAYBE_SOFTCAP(dot, softcap);
        s_weights[i] = dot;
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
    for (int i = tid; i < active_len; i += blockDim.x) {
        float w = expf(s_weights[i] - max_val);
        s_weights[i] = w;
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
    for (int i = tid; i < active_len; i += blockDim.x) {
        s_weights[i] *= inv_sum;
    }
    __syncthreads();

    // Pass 3: Weighted V output
    const int out_base = (req * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < active_len; i++) {
            const int t = win_start + i;
            const int logical_block = t / block_size;
            const int offset_in_block = t % block_size;
            const int physical_block = req_block_table[logical_block];
            const int v_idx = ((physical_block * block_size + offset_in_block) * num_kv_heads + kv_head) * head_dim + d;
            acc += s_weights[i] * v_pool[v_idx];
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
    const float softcap,
    const int block_size,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq,
    const int window_size
) {
    const int head = blockIdx.x;
    const int req = blockIdx.y;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[req];
    if (seq_len == 0) return;

    const int kv_head = head * num_kv_heads / num_heads;
    const int win_start = (window_size > 0) ? max(0, seq_len - window_size) : 0;
    const int active_len = seq_len - win_start;

    extern __shared__ float shared[];
    float* s_q = shared;
    float* s_weights = shared + head_dim;
    float* s_scratch = shared + head_dim + active_len;

    const int* req_block_table = block_tables + req * max_blocks_per_seq;
    const int q_base = (req * num_heads + head) * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = __half2float(q[q_base + d]);
    }
    __syncthreads();

    float local_max = -INFINITY_F;
    for (int i = tid; i < active_len; i += blockDim.x) {
        const int t = win_start + i;
        const int logical_block = t / block_size;
        const int offset_in_block = t % block_size;
        const int physical_block = req_block_table[logical_block];
        const int k_base = ((physical_block * block_size + offset_in_block) * num_kv_heads + kv_head) * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * __half2float(k_pool[k_base + d]);
        }
        dot *= scale;
        MAYBE_SOFTCAP(dot, softcap);
        s_weights[i] = dot;
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
    for (int i = tid; i < active_len; i += blockDim.x) {
        float w = expf(s_weights[i] - max_val);
        s_weights[i] = w;
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

    for (int i = tid; i < active_len; i += blockDim.x) {
        s_weights[i] *= inv_sum;
    }
    __syncthreads();

    const int out_base = (req * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < active_len; i++) {
            const int t = win_start + i;
            const int logical_block = t / block_size;
            const int offset_in_block = t % block_size;
            const int physical_block = req_block_table[logical_block];
            const int v_idx = ((physical_block * block_size + offset_in_block) * num_kv_heads + kv_head) * head_dim + d;
            acc += s_weights[i] * __half2float(v_pool[v_idx]);
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
    const float softcap,
    const int block_size,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq,
    const int window_size
) {
    const int head = blockIdx.x;
    const int req = blockIdx.y;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[req];
    if (seq_len == 0) return;

    const int kv_head = head * num_kv_heads / num_heads;
    const int win_start = (window_size > 0) ? max(0, seq_len - window_size) : 0;
    const int active_len = seq_len - win_start;

    extern __shared__ float shared[];
    float* s_q = shared;
    float* s_weights = shared + head_dim;
    float* s_scratch = shared + head_dim + active_len;

    const int* req_block_table = block_tables + req * max_blocks_per_seq;
    const int q_base = (req * num_heads + head) * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = __bfloat162float(q[q_base + d]);
    }
    __syncthreads();

    float local_max = -INFINITY_F;
    for (int i = tid; i < active_len; i += blockDim.x) {
        const int t = win_start + i;
        const int logical_block = t / block_size;
        const int offset_in_block = t % block_size;
        const int physical_block = req_block_table[logical_block];
        const int k_base = ((physical_block * block_size + offset_in_block) * num_kv_heads + kv_head) * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * __bfloat162float(k_pool[k_base + d]);
        }
        dot *= scale;
        MAYBE_SOFTCAP(dot, softcap);
        s_weights[i] = dot;
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
    for (int i = tid; i < active_len; i += blockDim.x) {
        float w = expf(s_weights[i] - max_val);
        s_weights[i] = w;
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

    for (int i = tid; i < active_len; i += blockDim.x) {
        s_weights[i] *= inv_sum;
    }
    __syncthreads();

    const int out_base = (req * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < active_len; i++) {
            const int t = win_start + i;
            const int logical_block = t / block_size;
            const int offset_in_block = t % block_size;
            const int physical_block = req_block_table[logical_block];
            const int v_idx = ((physical_block * block_size + offset_in_block) * num_kv_heads + kv_head) * head_dim + d;
            acc += s_weights[i] * __bfloat162float(v_pool[v_idx]);
        }
        output[out_base + d] = __float2bfloat16(acc);
    }
}
