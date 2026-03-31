// Flash Attention v2 prefill kernel — tiled (BR=4 query rows per block)
//
// Single-pass online softmax attention for prefill (seq_q > 1).
// Each thread block processes BR=4 query rows for one attention head.
//
// Algorithm (Flash Attention v2 / online softmax, tiled over queries):
//   For BR query positions processed in parallel, iterate over K/V in
//   tiles of size BC:
//     1. Load K tile [BC, head_dim] into shared memory (as f32)
//     2. Compute scores[BR, BC] = Q_tile · K_tile^T * scale, apply masks
//     3. Online softmax: warp-shuffle reduction for tile max/sum per query
//     4. Load V tile [BC, head_dim] into shared memory (as f32)
//     5. Accumulate weighted V into per-thread output registers
//
// Thread mapping: 128 threads = 4 warps.
//   qi = tid / BC  (query index 0..BR-1, one warp per query row)
//   kj = tid % BC  (key index 0..BC-1)
//
// All internal computation is in f32. F16/BF16 variants only differ in
// global memory I/O — loads convert to f32, stores convert from f32.
//
// All kernels (f32, f16, bf16) are standalone to guarantee
// bit-exact PTX output matching the original implementation.
//
// Supports: causal masking, GQA, sliding window, logit soft-capping.
//
// Grid:  (num_heads, ceil(seq_q / BR), 1)
// Block: (THREADS, 1, 1)
// Shared: smem_q[BR * head_dim] + smem_kv[BC * head_dim] + smem_scores[BR * BC]
//         (all f32, regardless of I/O dtype)

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define BR 4           // Query tile size (query rows per block)
#define BC 32          // K/V tile size (number of K/V rows per tile)
#define THREADS 128    // threads per block (BR * BC = 4 * 32)
#define MAX_ACC 8      // max dims per thread: head_dim <= 256, 256/32 = 8

// Soft-capping: tanh(x / cap) * cap
#define MAYBE_SOFTCAP(x, cap) ((cap) > 0.0f ? tanhf((x) / (cap)) * (cap) : (x))

// ===========================================================================
// F32 kernels (standalone — all computation in f32, I/O in f32)
// ===========================================================================

extern "C" __global__ void flash_prefill_attention_v2_f32(
    float* __restrict__ output,         // (seq_q, num_heads, head_dim)
    const float* __restrict__ q,        // (seq_q, num_heads, head_dim)
    const float* __restrict__ k,        // (total_len, num_kv_heads, head_dim)
    const float* __restrict__ v,        // (total_len, num_kv_heads, head_dim)
    float scale,
    float softcap,
    int total_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_q,
    int offset,        // causal offset: valid K positions for query qpos are
                       //   [min_valid_k, offset + qpos + 1)
    int window_size    // <= 0 means full attention
) {
    const int head = blockIdx.x;
    const int qblock = blockIdx.y;
    const int tid = threadIdx.x;
    const int qi = tid / BC;   // query index within tile  0..BR-1
    const int kj = tid % BC;   // key index within tile    0..BC-1

    const int qpos_base = qblock * BR;
    const int qpos = qpos_base + qi;

    // All threads must participate in cooperative smem loads and __syncthreads();
    // inactive threads (qpos >= seq_q) skip per-query work and output writes.
    const bool active = (qpos < seq_q);

    const int kv_head = head * num_kv_heads / num_heads;  // GQA mapping

    // Per-query causal bounds (only meaningful for active threads)
    const int max_valid_k = active ? (offset + qpos + 1) : 0;
    const int min_valid_k = active
        ? ((window_size > 0) ? max(0, max_valid_k - window_size) : 0)
        : 0;

    // Union of all BR queries' K ranges for tile iteration.
    // We iterate over tiles covering the widest window; per-query masking
    // in Phase 2 handles individual causal/sliding-window boundaries.
    const int last_qpos = min(qpos_base + BR, seq_q) - 1;
    const int block_max_valid_k = offset + last_qpos + 1;
    const int block_min_valid_k = (window_size > 0)
        ? max(0, offset + qpos_base + 1 - window_size)
        : 0;

    // Shared memory layout (dynamically sized):
    //   smem_q:      [BR * head_dim]  — BR query vectors
    //   smem_kv:     [BC * head_dim]  — K or V tile (reused)
    //   smem_scores: [BR * BC]        — score matrix
    extern __shared__ float smem[];
    float* smem_q      = smem;
    float* smem_kv     = smem + BR * head_dim;
    float* smem_scores = smem + BR * head_dim + BC * head_dim;

    // Load BR query vectors into shared memory
    for (int idx = tid; idx < BR * head_dim; idx += THREADS) {
        int row = idx / head_dim;
        int d   = idx % head_dim;
        int qp  = qpos_base + row;
        if (qp < seq_q) {
            smem_q[row * head_dim + d] = q[(qp * num_heads + head) * head_dim + d];
        } else {
            smem_q[row * head_dim + d] = 0.0f;
        }
    }
    __syncthreads();

    // Per-thread accumulators: each thread handles dims for its query row qi.
    // Thread handles dimensions d = kj, kj+BC, kj+2*BC, ...
    const int dims_per_thread = (head_dim + BC - 1) / BC;
    float acc[MAX_ACC];
    for (int i = 0; i < MAX_ACC; ++i) acc[i] = 0.0f;

    float running_max = -1e30f;
    float running_sum = 0.0f;

    // Iterate over K/V in tiles of BC
    const int num_tiles = (block_max_valid_k - block_min_valid_k + BC - 1) / BC;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_start = block_min_valid_k + tile * BC;
        const int tile_end = min(tile_start + BC, block_max_valid_k);
        const int tile_len = tile_end - tile_start;

        // --- Phase 1: Load K tile into smem_kv ---
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d   = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] =
                k[(kv_pos * num_kv_heads + kv_head) * head_dim + d];
        }
        __syncthreads();

        // --- Phase 2: Compute Q·K^T score for (qi, kj) ---
        float score = -1e30f;
        if (kj < tile_len) {
            int k_global = tile_start + kj;
            // Per-query causal + sliding window mask
            if (k_global < max_valid_k && k_global >= min_valid_k) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += smem_q[qi * head_dim + d]
                         * smem_kv[kj * head_dim + d];
                }
                score = MAYBE_SOFTCAP(dot * scale, softcap);
            }
        }
        smem_scores[qi * BC + kj] = score;
        __syncthreads();

        // --- Phase 3: Online softmax (warp-shuffle reduction) ---
        // Each warp (32 threads) handles one query row qi.
        // kj is the lane within the warp.

        // Find tile_max via warp shuffle
        float local_s = smem_scores[qi * BC + kj];
        float tile_max = local_s;
        for (int delta = 16; delta >= 1; delta >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, tile_max, delta);
            tile_max = fmaxf(tile_max, other);
        }

        // Compute exp(score - tile_max) and reduce tile_sum
        float tile_sum = expf(local_s - tile_max);
        for (int delta = 16; delta >= 1; delta >>= 1) {
            tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, delta);
        }

        // Update running max/sum, rescale accumulators
        float new_max = fmaxf(running_max, tile_max);
        float old_sf = expf(running_max - new_max);
        float tile_sf = expf(tile_max - new_max);

        for (int i = 0; i < dims_per_thread; ++i) {
            acc[i] *= old_sf;
        }

        running_sum = running_sum * old_sf + tile_sum * tile_sf;
        running_max = new_max;

        // --- Phase 4: Load V tile and accumulate weighted V ---
        // Reuse smem_kv for V tile (scores still in smem_scores)
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d   = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] =
                v[(kv_pos * num_kv_heads + kv_head) * head_dim + d];
        }
        __syncthreads();

        // Each thread accumulates over tile_len key positions
        // for dims d = kj, kj+BC, kj+2*BC, ... of query row qi.
        // Weights are recomputed from the original scores (smem_scores
        // is read-only throughout the tile) to avoid shared memory races.
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = kj + i * BC;
            if (d < head_dim) {
                float val = 0.0f;
                for (int j = 0; j < tile_len; ++j) {
                    float w = expf(smem_scores[qi * BC + j] - new_max);
                    val += w * smem_kv[j * head_dim + d];
                }
                acc[i] += val;
            }
        }

        __syncthreads();
    }

    // Normalize by running_sum and write output
    if (active) {
        float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
        float* out_ptr = output + (qpos * num_heads + head) * head_dim;
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = kj + i * BC;
            if (d < head_dim) {
                out_ptr[d] = acc[i] * inv_sum;
            }
        }
    }
}

// LSE variant: same as above but also outputs log-sum-exp for multi-chunk combining
extern "C" __global__ void flash_prefill_attention_v2_with_lse_f32(
    float* __restrict__ output,         // (seq_q, num_heads, head_dim)
    float* __restrict__ lse,            // (seq_q, num_heads)
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float scale,
    float softcap,
    int total_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_q,
    int offset,
    int window_size
) {
    const int head = blockIdx.x;
    const int qblock = blockIdx.y;
    const int tid = threadIdx.x;
    const int qi = tid / BC;
    const int kj = tid % BC;

    const int qpos_base = qblock * BR;
    const int qpos = qpos_base + qi;

    // All threads must participate in cooperative smem loads and __syncthreads();
    // inactive threads (qpos >= seq_q) skip per-query work and output writes.
    const bool active = (qpos < seq_q);

    const int kv_head = head * num_kv_heads / num_heads;
    const int max_valid_k = active ? (offset + qpos + 1) : 0;
    const int min_valid_k = active
        ? ((window_size > 0) ? max(0, max_valid_k - window_size) : 0)
        : 0;

    const int last_qpos = min(qpos_base + BR, seq_q) - 1;
    const int block_max_valid_k = offset + last_qpos + 1;
    const int block_min_valid_k = (window_size > 0)
        ? max(0, offset + qpos_base + 1 - window_size)
        : 0;

    extern __shared__ float smem[];
    float* smem_q      = smem;
    float* smem_kv     = smem + BR * head_dim;
    float* smem_scores = smem + BR * head_dim + BC * head_dim;

    for (int idx = tid; idx < BR * head_dim; idx += THREADS) {
        int row = idx / head_dim;
        int d   = idx % head_dim;
        int qp  = qpos_base + row;
        if (qp < seq_q) {
            smem_q[row * head_dim + d] = q[(qp * num_heads + head) * head_dim + d];
        } else {
            smem_q[row * head_dim + d] = 0.0f;
        }
    }
    __syncthreads();

    const int dims_per_thread = (head_dim + BC - 1) / BC;
    float acc[MAX_ACC];
    for (int i = 0; i < MAX_ACC; ++i) acc[i] = 0.0f;

    float running_max = -1e30f;
    float running_sum = 0.0f;

    const int num_tiles = (block_max_valid_k - block_min_valid_k + BC - 1) / BC;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_start = block_min_valid_k + tile * BC;
        const int tile_end = min(tile_start + BC, block_max_valid_k);
        const int tile_len = tile_end - tile_start;

        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d   = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] =
                k[(kv_pos * num_kv_heads + kv_head) * head_dim + d];
        }
        __syncthreads();

        float score = -1e30f;
        if (kj < tile_len) {
            int k_global = tile_start + kj;
            if (k_global < max_valid_k && k_global >= min_valid_k) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += smem_q[qi * head_dim + d]
                         * smem_kv[kj * head_dim + d];
                }
                score = MAYBE_SOFTCAP(dot * scale, softcap);
            }
        }
        smem_scores[qi * BC + kj] = score;
        __syncthreads();

        float local_s = smem_scores[qi * BC + kj];
        float tile_max = local_s;
        for (int delta = 16; delta >= 1; delta >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, tile_max, delta);
            tile_max = fmaxf(tile_max, other);
        }

        float tile_sum = expf(local_s - tile_max);
        for (int delta = 16; delta >= 1; delta >>= 1) {
            tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, delta);
        }

        float new_max = fmaxf(running_max, tile_max);
        float old_sf = expf(running_max - new_max);
        float tile_sf = expf(tile_max - new_max);
        for (int i = 0; i < dims_per_thread; ++i) acc[i] *= old_sf;
        running_sum = running_sum * old_sf + tile_sum * tile_sf;
        running_max = new_max;

        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d   = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] =
                v[(kv_pos * num_kv_heads + kv_head) * head_dim + d];
        }
        __syncthreads();

        for (int i = 0; i < dims_per_thread; ++i) {
            int d = kj + i * BC;
            if (d < head_dim) {
                float val = 0.0f;
                for (int j = 0; j < tile_len; ++j) {
                    float w = expf(smem_scores[qi * BC + j] - new_max);
                    val += w * smem_kv[j * head_dim + d];
                }
                acc[i] += val;
            }
        }
        __syncthreads();
    }

    if (active) {
        float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
        float* out_ptr = output + (qpos * num_heads + head) * head_dim;
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = kj + i * BC;
            if (d < head_dim) {
                out_ptr[d] = acc[i] * inv_sum;
            }
        }

        // Write LSE = log(running_sum) + running_max
        if (kj == 0) {
            lse[qpos * num_heads + head] = logf(running_sum) + running_max;
        }
    }
}

// ===========================================================================
// F16 kernels (standalone — all computation in f32, I/O in f16)
// ===========================================================================

extern "C" __global__ void flash_prefill_attention_v2_f16(
    __half* __restrict__ output,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    float scale,
    float softcap,
    int total_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_q,
    int offset,
    int window_size
) {
    const int head = blockIdx.x;
    const int qblock = blockIdx.y;
    const int tid = threadIdx.x;
    const int qi = tid / BC;
    const int kj = tid % BC;

    const int qpos_base = qblock * BR;
    const int qpos = qpos_base + qi;

    // All threads must participate in cooperative smem loads and __syncthreads();
    // inactive threads (qpos >= seq_q) skip per-query work and output writes.
    const bool active = (qpos < seq_q);

    const int kv_head = head * num_kv_heads / num_heads;
    const int max_valid_k = active ? (offset + qpos + 1) : 0;
    const int min_valid_k = active
        ? ((window_size > 0) ? max(0, max_valid_k - window_size) : 0)
        : 0;

    const int last_qpos = min(qpos_base + BR, seq_q) - 1;
    const int block_max_valid_k = offset + last_qpos + 1;
    const int block_min_valid_k = (window_size > 0)
        ? max(0, offset + qpos_base + 1 - window_size)
        : 0;

    extern __shared__ float smem[];
    float* smem_q      = smem;
    float* smem_kv     = smem + BR * head_dim;
    float* smem_scores = smem + BR * head_dim + BC * head_dim;

    // Load Q into shared memory (f16 → f32)
    for (int idx = tid; idx < BR * head_dim; idx += THREADS) {
        int row = idx / head_dim;
        int d   = idx % head_dim;
        int qp  = qpos_base + row;
        if (qp < seq_q) {
            smem_q[row * head_dim + d] =
                __half2float(q[(qp * num_heads + head) * head_dim + d]);
        } else {
            smem_q[row * head_dim + d] = 0.0f;
        }
    }
    __syncthreads();

    const int dims_per_thread = (head_dim + BC - 1) / BC;
    float acc[MAX_ACC];
    for (int i = 0; i < MAX_ACC; ++i) acc[i] = 0.0f;

    float running_max = -1e30f;
    float running_sum = 0.0f;

    const int num_tiles = (block_max_valid_k - block_min_valid_k + BC - 1) / BC;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_start = block_min_valid_k + tile * BC;
        const int tile_end = min(tile_start + BC, block_max_valid_k);
        const int tile_len = tile_end - tile_start;

        // Load K tile (f16 → f32)
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d   = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = __half2float(
                k[(kv_pos * num_kv_heads + kv_head) * head_dim + d]);
        }
        __syncthreads();

        float score = -1e30f;
        if (kj < tile_len) {
            int k_global = tile_start + kj;
            if (k_global < max_valid_k && k_global >= min_valid_k) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += smem_q[qi * head_dim + d]
                         * smem_kv[kj * head_dim + d];
                }
                score = MAYBE_SOFTCAP(dot * scale, softcap);
            }
        }
        smem_scores[qi * BC + kj] = score;
        __syncthreads();

        float local_s = smem_scores[qi * BC + kj];
        float tile_max = local_s;
        for (int delta = 16; delta >= 1; delta >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, tile_max, delta);
            tile_max = fmaxf(tile_max, other);
        }

        float tile_sum = expf(local_s - tile_max);
        for (int delta = 16; delta >= 1; delta >>= 1) {
            tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, delta);
        }

        float new_max = fmaxf(running_max, tile_max);
        float old_sf = expf(running_max - new_max);
        float tile_sf = expf(tile_max - new_max);
        for (int i = 0; i < dims_per_thread; ++i) acc[i] *= old_sf;
        running_sum = running_sum * old_sf + tile_sum * tile_sf;
        running_max = new_max;

        // Load V tile (f16 → f32)
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d   = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = __half2float(
                v[(kv_pos * num_kv_heads + kv_head) * head_dim + d]);
        }
        __syncthreads();

        for (int i = 0; i < dims_per_thread; ++i) {
            int d = kj + i * BC;
            if (d < head_dim) {
                float val = 0.0f;
                for (int j = 0; j < tile_len; ++j) {
                    float w = expf(smem_scores[qi * BC + j] - new_max);
                    val += w * smem_kv[j * head_dim + d];
                }
                acc[i] += val;
            }
        }
        __syncthreads();
    }

    // Output (f32 → f16)
    if (active) {
        float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
        __half* out_ptr = output + (qpos * num_heads + head) * head_dim;
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = kj + i * BC;
            if (d < head_dim) {
                out_ptr[d] = __float2half(acc[i] * inv_sum);
            }
        }
    }
}

extern "C" __global__ void flash_prefill_attention_v2_with_lse_f16(
    __half* __restrict__ output,
    float* __restrict__ lse,
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    float scale,
    float softcap,
    int total_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_q,
    int offset,
    int window_size
) {
    const int head = blockIdx.x;
    const int qblock = blockIdx.y;
    const int tid = threadIdx.x;
    const int qi = tid / BC;
    const int kj = tid % BC;

    const int qpos_base = qblock * BR;
    const int qpos = qpos_base + qi;

    // All threads must participate in cooperative smem loads and __syncthreads();
    // inactive threads (qpos >= seq_q) skip per-query work and output writes.
    const bool active = (qpos < seq_q);

    const int kv_head = head * num_kv_heads / num_heads;
    const int max_valid_k = active ? (offset + qpos + 1) : 0;
    const int min_valid_k = active
        ? ((window_size > 0) ? max(0, max_valid_k - window_size) : 0)
        : 0;

    const int last_qpos = min(qpos_base + BR, seq_q) - 1;
    const int block_max_valid_k = offset + last_qpos + 1;
    const int block_min_valid_k = (window_size > 0)
        ? max(0, offset + qpos_base + 1 - window_size)
        : 0;

    extern __shared__ float smem[];
    float* smem_q      = smem;
    float* smem_kv     = smem + BR * head_dim;
    float* smem_scores = smem + BR * head_dim + BC * head_dim;

    for (int idx = tid; idx < BR * head_dim; idx += THREADS) {
        int row = idx / head_dim;
        int d   = idx % head_dim;
        int qp  = qpos_base + row;
        if (qp < seq_q) {
            smem_q[row * head_dim + d] =
                __half2float(q[(qp * num_heads + head) * head_dim + d]);
        } else {
            smem_q[row * head_dim + d] = 0.0f;
        }
    }
    __syncthreads();

    const int dims_per_thread = (head_dim + BC - 1) / BC;
    float acc[MAX_ACC];
    for (int i = 0; i < MAX_ACC; ++i) acc[i] = 0.0f;

    float running_max = -1e30f;
    float running_sum = 0.0f;

    const int num_tiles = (block_max_valid_k - block_min_valid_k + BC - 1) / BC;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_start = block_min_valid_k + tile * BC;
        const int tile_end = min(tile_start + BC, block_max_valid_k);
        const int tile_len = tile_end - tile_start;

        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d   = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = __half2float(
                k[(kv_pos * num_kv_heads + kv_head) * head_dim + d]);
        }
        __syncthreads();

        float score = -1e30f;
        if (kj < tile_len) {
            int k_global = tile_start + kj;
            if (k_global < max_valid_k && k_global >= min_valid_k) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += smem_q[qi * head_dim + d]
                         * smem_kv[kj * head_dim + d];
                }
                score = MAYBE_SOFTCAP(dot * scale, softcap);
            }
        }
        smem_scores[qi * BC + kj] = score;
        __syncthreads();

        float local_s = smem_scores[qi * BC + kj];
        float tile_max = local_s;
        for (int delta = 16; delta >= 1; delta >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, tile_max, delta);
            tile_max = fmaxf(tile_max, other);
        }

        float tile_sum = expf(local_s - tile_max);
        for (int delta = 16; delta >= 1; delta >>= 1) {
            tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, delta);
        }

        float new_max = fmaxf(running_max, tile_max);
        float old_sf = expf(running_max - new_max);
        float tile_sf = expf(tile_max - new_max);
        for (int i = 0; i < dims_per_thread; ++i) acc[i] *= old_sf;
        running_sum = running_sum * old_sf + tile_sum * tile_sf;
        running_max = new_max;

        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d   = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = __half2float(
                v[(kv_pos * num_kv_heads + kv_head) * head_dim + d]);
        }
        __syncthreads();

        for (int i = 0; i < dims_per_thread; ++i) {
            int d = kj + i * BC;
            if (d < head_dim) {
                float val = 0.0f;
                for (int j = 0; j < tile_len; ++j) {
                    float w = expf(smem_scores[qi * BC + j] - new_max);
                    val += w * smem_kv[j * head_dim + d];
                }
                acc[i] += val;
            }
        }
        __syncthreads();
    }

    if (active) {
        float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
        __half* out_ptr = output + (qpos * num_heads + head) * head_dim;
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = kj + i * BC;
            if (d < head_dim) {
                out_ptr[d] = __float2half(acc[i] * inv_sum);
            }
        }

        if (kj == 0) {
            lse[qpos * num_heads + head] = logf(running_sum) + running_max;
        }
    }
}

// ===========================================================================
// BF16 kernels (standalone — all computation in f32, I/O in bf16)
// ===========================================================================

extern "C" __global__ void flash_prefill_attention_v2_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    float scale,
    float softcap,
    int total_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_q,
    int offset,
    int window_size
) {
    const int head = blockIdx.x;
    const int qblock = blockIdx.y;
    const int tid = threadIdx.x;
    const int qi = tid / BC;
    const int kj = tid % BC;

    const int qpos_base = qblock * BR;
    const int qpos = qpos_base + qi;

    // All threads must participate in cooperative smem loads and __syncthreads();
    // inactive threads (qpos >= seq_q) skip per-query work and output writes.
    const bool active = (qpos < seq_q);

    const int kv_head = head * num_kv_heads / num_heads;
    const int max_valid_k = active ? (offset + qpos + 1) : 0;
    const int min_valid_k = active
        ? ((window_size > 0) ? max(0, max_valid_k - window_size) : 0)
        : 0;

    const int last_qpos = min(qpos_base + BR, seq_q) - 1;
    const int block_max_valid_k = offset + last_qpos + 1;
    const int block_min_valid_k = (window_size > 0)
        ? max(0, offset + qpos_base + 1 - window_size)
        : 0;

    extern __shared__ float smem[];
    float* smem_q      = smem;
    float* smem_kv     = smem + BR * head_dim;
    float* smem_scores = smem + BR * head_dim + BC * head_dim;

    // Load Q into shared memory (bf16 → f32)
    for (int idx = tid; idx < BR * head_dim; idx += THREADS) {
        int row = idx / head_dim;
        int d   = idx % head_dim;
        int qp  = qpos_base + row;
        if (qp < seq_q) {
            smem_q[row * head_dim + d] =
                __bfloat162float(q[(qp * num_heads + head) * head_dim + d]);
        } else {
            smem_q[row * head_dim + d] = 0.0f;
        }
    }
    __syncthreads();

    const int dims_per_thread = (head_dim + BC - 1) / BC;
    float acc[MAX_ACC];
    for (int i = 0; i < MAX_ACC; ++i) acc[i] = 0.0f;

    float running_max = -1e30f;
    float running_sum = 0.0f;

    const int num_tiles = (block_max_valid_k - block_min_valid_k + BC - 1) / BC;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_start = block_min_valid_k + tile * BC;
        const int tile_end = min(tile_start + BC, block_max_valid_k);
        const int tile_len = tile_end - tile_start;

        // Load K tile (bf16 → f32)
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d   = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = __bfloat162float(
                k[(kv_pos * num_kv_heads + kv_head) * head_dim + d]);
        }
        __syncthreads();

        float score = -1e30f;
        if (kj < tile_len) {
            int k_global = tile_start + kj;
            if (k_global < max_valid_k && k_global >= min_valid_k) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += smem_q[qi * head_dim + d]
                         * smem_kv[kj * head_dim + d];
                }
                score = MAYBE_SOFTCAP(dot * scale, softcap);
            }
        }
        smem_scores[qi * BC + kj] = score;
        __syncthreads();

        float local_s = smem_scores[qi * BC + kj];
        float tile_max = local_s;
        for (int delta = 16; delta >= 1; delta >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, tile_max, delta);
            tile_max = fmaxf(tile_max, other);
        }

        float tile_sum = expf(local_s - tile_max);
        for (int delta = 16; delta >= 1; delta >>= 1) {
            tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, delta);
        }

        float new_max = fmaxf(running_max, tile_max);
        float old_sf = expf(running_max - new_max);
        float tile_sf = expf(tile_max - new_max);
        for (int i = 0; i < dims_per_thread; ++i) acc[i] *= old_sf;
        running_sum = running_sum * old_sf + tile_sum * tile_sf;
        running_max = new_max;

        // Load V tile (bf16 → f32)
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d   = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = __bfloat162float(
                v[(kv_pos * num_kv_heads + kv_head) * head_dim + d]);
        }
        __syncthreads();

        for (int i = 0; i < dims_per_thread; ++i) {
            int d = kj + i * BC;
            if (d < head_dim) {
                float val = 0.0f;
                for (int j = 0; j < tile_len; ++j) {
                    float w = expf(smem_scores[qi * BC + j] - new_max);
                    val += w * smem_kv[j * head_dim + d];
                }
                acc[i] += val;
            }
        }
        __syncthreads();
    }

    // Output (f32 → bf16)
    if (active) {
        float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
        __nv_bfloat16* out_ptr = output + (qpos * num_heads + head) * head_dim;
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = kj + i * BC;
            if (d < head_dim) {
                out_ptr[d] = __float2bfloat16(acc[i] * inv_sum);
            }
        }
    }
}

extern "C" __global__ void flash_prefill_attention_v2_with_lse_bf16(
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ lse,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    float scale,
    float softcap,
    int total_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_q,
    int offset,
    int window_size
) {
    const int head = blockIdx.x;
    const int qblock = blockIdx.y;
    const int tid = threadIdx.x;
    const int qi = tid / BC;
    const int kj = tid % BC;

    const int qpos_base = qblock * BR;
    const int qpos = qpos_base + qi;

    // All threads must participate in cooperative smem loads and __syncthreads();
    // inactive threads (qpos >= seq_q) skip per-query work and output writes.
    const bool active = (qpos < seq_q);

    const int kv_head = head * num_kv_heads / num_heads;
    const int max_valid_k = active ? (offset + qpos + 1) : 0;
    const int min_valid_k = active
        ? ((window_size > 0) ? max(0, max_valid_k - window_size) : 0)
        : 0;

    const int last_qpos = min(qpos_base + BR, seq_q) - 1;
    const int block_max_valid_k = offset + last_qpos + 1;
    const int block_min_valid_k = (window_size > 0)
        ? max(0, offset + qpos_base + 1 - window_size)
        : 0;

    extern __shared__ float smem[];
    float* smem_q      = smem;
    float* smem_kv     = smem + BR * head_dim;
    float* smem_scores = smem + BR * head_dim + BC * head_dim;

    for (int idx = tid; idx < BR * head_dim; idx += THREADS) {
        int row = idx / head_dim;
        int d   = idx % head_dim;
        int qp  = qpos_base + row;
        if (qp < seq_q) {
            smem_q[row * head_dim + d] =
                __bfloat162float(q[(qp * num_heads + head) * head_dim + d]);
        } else {
            smem_q[row * head_dim + d] = 0.0f;
        }
    }
    __syncthreads();

    const int dims_per_thread = (head_dim + BC - 1) / BC;
    float acc[MAX_ACC];
    for (int i = 0; i < MAX_ACC; ++i) acc[i] = 0.0f;

    float running_max = -1e30f;
    float running_sum = 0.0f;

    const int num_tiles = (block_max_valid_k - block_min_valid_k + BC - 1) / BC;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_start = block_min_valid_k + tile * BC;
        const int tile_end = min(tile_start + BC, block_max_valid_k);
        const int tile_len = tile_end - tile_start;

        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d   = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = __bfloat162float(
                k[(kv_pos * num_kv_heads + kv_head) * head_dim + d]);
        }
        __syncthreads();

        float score = -1e30f;
        if (kj < tile_len) {
            int k_global = tile_start + kj;
            if (k_global < max_valid_k && k_global >= min_valid_k) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += smem_q[qi * head_dim + d]
                         * smem_kv[kj * head_dim + d];
                }
                score = MAYBE_SOFTCAP(dot * scale, softcap);
            }
        }
        smem_scores[qi * BC + kj] = score;
        __syncthreads();

        float local_s = smem_scores[qi * BC + kj];
        float tile_max = local_s;
        for (int delta = 16; delta >= 1; delta >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, tile_max, delta);
            tile_max = fmaxf(tile_max, other);
        }

        float tile_sum = expf(local_s - tile_max);
        for (int delta = 16; delta >= 1; delta >>= 1) {
            tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, delta);
        }

        float new_max = fmaxf(running_max, tile_max);
        float old_sf = expf(running_max - new_max);
        float tile_sf = expf(tile_max - new_max);
        for (int i = 0; i < dims_per_thread; ++i) acc[i] *= old_sf;
        running_sum = running_sum * old_sf + tile_sum * tile_sf;
        running_max = new_max;

        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d   = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = __bfloat162float(
                v[(kv_pos * num_kv_heads + kv_head) * head_dim + d]);
        }
        __syncthreads();

        for (int i = 0; i < dims_per_thread; ++i) {
            int d = kj + i * BC;
            if (d < head_dim) {
                float val = 0.0f;
                for (int j = 0; j < tile_len; ++j) {
                    float w = expf(smem_scores[qi * BC + j] - new_max);
                    val += w * smem_kv[j * head_dim + d];
                }
                acc[i] += val;
            }
        }
        __syncthreads();
    }

    if (active) {
        float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
        __nv_bfloat16* out_ptr = output + (qpos * num_heads + head) * head_dim;
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = kj + i * BC;
            if (d < head_dim) {
                out_ptr[d] = __float2bfloat16(acc[i] * inv_sum);
            }
        }

        if (kj == 0) {
            lse[qpos * num_heads + head] = logf(running_sum) + running_max;
        }
    }
}
