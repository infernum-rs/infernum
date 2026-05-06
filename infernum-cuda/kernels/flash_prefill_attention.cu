// Flash Attention v2 prefill kernel (f32, f16, bf16)
//
// Single-pass online softmax attention for prefill (seq_q > 1).
// Each thread block processes one (head, query_position) pair.
//
// Algorithm (Flash Attention v2 / online softmax):
//   For each query position q, iterate over K/V in tiles of size Bc:
//     1. Load K tile [Bc, head_dim] into shared memory (as f32)
//     2. Compute scores[Bc] = Q · K^T * scale, apply causal mask
//     3. Online softmax: update running (max, sum) with new tile
//     4. Load V tile [Bc, head_dim] into shared memory (as f32)
//     5. Accumulate weighted V into running output, rescaling as needed
//
// All internal computation is in f32. F16/BF16 variants only differ in
// global memory I/O — loads convert to f32, stores convert from f32.
//
// All kernels (f32, f16, bf16) are standalone to guarantee
// bit-exact PTX output matching the original implementation.
//
// Supports: causal masking, GQA, sliding window, logit soft-capping.
//
// Grid:  (num_heads, seq_q, 1)
// Block: (THREADS, 1, 1)
// Shared: smem_q[head_dim] + smem_kv[BC * head_dim] + smem_scores[BC]
//         (all f32, regardless of I/O dtype)

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define BC 32          // K/V tile size (number of K/V rows per tile)
#define THREADS 128    // threads per block (must be >= BC and >= head_dim)

// Soft-capping: tanh(x / cap) * cap
#define MAYBE_SOFTCAP(x, cap) ((cap) > 0.0f ? tanhf((x) / (cap)) * (cap) : (x))

// ===========================================================================
// F32 kernels (standalone — bit-exact with original implementation)
// ===========================================================================

extern "C" __global__ void flash_prefill_attention_f32(
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
    int offset,        // causal offset: valid K positions = [min_k, offset + qpos + 1)
    int window_size    // <= 0 means full attention
) {
    const int head = blockIdx.x;
    const int qpos = blockIdx.y;
    const int tid = threadIdx.x;

    const int kv_head = head * num_kv_heads / num_heads;  // GQA mapping
    const int max_valid_k = offset + qpos + 1;            // causal: exclusive upper bound
    const int min_valid_k = (window_size > 0)
        ? max(0, max_valid_k - window_size)
        : 0;

    // Shared memory layout (dynamically sized):
    //   smem_q:      [head_dim] floats
    //   smem_kv:     [BC * head_dim] floats  (reused for K tile then V tile)
    //   smem_scores: [BC] floats
    extern __shared__ float smem[];
    float* smem_q      = smem;                              // [head_dim]
    float* smem_kv     = smem + head_dim;                   // [BC * head_dim]
    float* smem_scores = smem + head_dim + BC * head_dim;   // [BC]

    // Load Q vector for this (qpos, head) into shared memory
    const float* q_ptr = q + (qpos * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += THREADS) {
        smem_q[d] = q_ptr[d];
    }
    __syncthreads();

    // Per-thread accumulators for online softmax
    // Each thread accumulates a subset of the head_dim output dimensions
    // We store the full output vector in registers (thread i handles dims i, i+THREADS, ...)
    const int dims_per_thread = (head_dim + THREADS - 1) / THREADS;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // max 4 dims per thread (head_dim<=512)

    float running_max = -1e30f;  // global max so far
    float running_sum = 0.0f;    // sum of exp(score - running_max) so far

    // Iterate over K/V in tiles of BC
    // K layout: (total_len, num_kv_heads, head_dim)
    // K[kv_pos, kv_head, d] = k[(kv_pos * num_kv_heads + kv_head) * head_dim + d]
    const int num_tiles = (max_valid_k - min_valid_k + BC - 1) / BC;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_start = min_valid_k + tile * BC;
        const int tile_end = min(tile_start + BC, max_valid_k);
        const int tile_len = tile_end - tile_start;  // actual rows in this tile

        // --- Phase 1: Load K tile and compute scores ---

        // Load K tile [tile_len, head_dim] into smem_kv
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = k[(kv_pos * num_kv_heads + kv_head) * head_dim + d];
        }
        __syncthreads();

        // Compute scores: score[j] = dot(Q, K[j]) * scale
        // Each thread computes scores for tid-th K position (if tid < tile_len)
        float score = -1e30f;  // masked value
        if (tid < tile_len) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += smem_q[d] * smem_kv[tid * head_dim + d];
            }
            score = MAYBE_SOFTCAP(dot * scale, softcap);
        }

        // Store scores to shared memory for all threads to see
        if (tid < BC) {
            smem_scores[tid] = score;
        }
        __syncthreads();

        // --- Phase 2: Online softmax update ---

        // Find tile max (reduction in shared memory)
        // Use smem_scores for the reduction
        float tile_max = -1e30f;
        for (int j = 0; j < tile_len; ++j) {
            float s = smem_scores[j];
            if (s > tile_max) tile_max = s;
        }

        // Update running max and rescale existing accumulator
        float new_max = fmaxf(running_max, tile_max);
        float old_scale_factor = expf(running_max - new_max);
        float new_sum = running_sum * old_scale_factor;

        // Rescale existing output accumulator
        for (int i = 0; i < dims_per_thread; ++i) {
            acc[i] *= old_scale_factor;
        }

        // Compute tile_sum = sum(exp(score - new_max)) without overwriting
        // smem_scores (scores are needed again in Phase 3).
        float tile_sum = 0.0f;
        for (int j = 0; j < tile_len; ++j) {
            tile_sum += expf(smem_scores[j] - new_max);
        }

        running_max = new_max;
        running_sum = new_sum + tile_sum;

        // --- Phase 3: Load V tile and accumulate weighted V ---

        // Load V tile [tile_len, head_dim] into smem_kv (reuse K tile space)
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = v[(kv_pos * num_kv_heads + kv_head) * head_dim + d];
        }
        __syncthreads();

        // Accumulate: acc[d] += sum_j( exp(score_j - max) * V[j, d] )
        // Weights are recomputed from the original scores (smem_scores
        // is read-only throughout the tile) to avoid a cross-warp shared
        // memory race that would occur if we overwrote scores with weights.
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = tid + i * THREADS;
            if (d < head_dim) {
                float val = 0.0f;
                for (int j = 0; j < tile_len; ++j) {
                    float w = expf(smem_scores[j] - new_max);
                    val += w * smem_kv[j * head_dim + d];
                }
                acc[i] += val;
            }
        }

        __syncthreads();
    }

    // Normalize by running_sum and write output
    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    float* out_ptr = output + (qpos * num_heads + head) * head_dim;
    for (int i = 0; i < dims_per_thread; ++i) {
        int d = tid + i * THREADS;
        if (d < head_dim) {
            out_ptr[d] = acc[i] * inv_sum;
        }
    }
}

// LSE variant: same as above but also outputs log-sum-exp for multi-chunk combining
extern "C" __global__ void flash_prefill_attention_with_lse_f32(
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
    int offset,
    int window_size
) {
    const int head = blockIdx.x;
    const int qpos = blockIdx.y;
    const int tid = threadIdx.x;

    const int kv_head = head * num_kv_heads / num_heads;
    const int max_valid_k = offset + qpos + 1;
    const int min_valid_k = (window_size > 0)
        ? max(0, max_valid_k - window_size)
        : 0;

    extern __shared__ float smem[];
    float* smem_q      = smem;
    float* smem_kv     = smem + head_dim;
    float* smem_scores = smem + head_dim + BC * head_dim;

    const float* q_ptr = q + (qpos * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += THREADS) {
        smem_q[d] = q_ptr[d];
    }
    __syncthreads();

    const int dims_per_thread = (head_dim + THREADS - 1) / THREADS;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    float running_max = -1e30f;
    float running_sum = 0.0f;

    const int num_tiles = (max_valid_k - min_valid_k + BC - 1) / BC;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_start = min_valid_k + tile * BC;
        const int tile_end = min(tile_start + BC, max_valid_k);
        const int tile_len = tile_end - tile_start;

        // Load K tile
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = k[(kv_pos * num_kv_heads + kv_head) * head_dim + d];
        }
        __syncthreads();

        // Compute scores
        float score = -1e30f;
        if (tid < tile_len) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += smem_q[d] * smem_kv[tid * head_dim + d];
            }
            score = MAYBE_SOFTCAP(dot * scale, softcap);
        }
        if (tid < BC) smem_scores[tid] = score;
        __syncthreads();

        // Online softmax
        float tile_max = -1e30f;
        for (int j = 0; j < tile_len; ++j) {
            if (smem_scores[j] > tile_max) tile_max = smem_scores[j];
        }

        float new_max = fmaxf(running_max, tile_max);
        float old_scale_factor = expf(running_max - new_max);
        float new_sum = running_sum * old_scale_factor;

        for (int i = 0; i < dims_per_thread; ++i) acc[i] *= old_scale_factor;

        // Compute tile_sum = sum(exp(score - new_max)) without overwriting
        // smem_scores (scores are needed again in Phase 3).
        float tile_sum = 0.0f;
        for (int j = 0; j < tile_len; ++j) {
            tile_sum += expf(smem_scores[j] - new_max);
        }

        running_max = new_max;
        running_sum = new_sum + tile_sum;

        // Load V tile and accumulate
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = v[(kv_pos * num_kv_heads + kv_head) * head_dim + d];
        }
        __syncthreads();

        // Accumulate: acc[d] += sum_j( exp(score_j - max) * V[j, d] )
        // Weights are recomputed from the original scores (smem_scores
        // is read-only throughout the tile) to avoid a cross-warp shared
        // memory race that would occur if we overwrote scores with weights.
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = tid + i * THREADS;
            if (d < head_dim) {
                float val = 0.0f;
                for (int j = 0; j < tile_len; ++j) {
                    float w = expf(smem_scores[j] - new_max);
                    val += w * smem_kv[j * head_dim + d];
                }
                acc[i] += val;
            }
        }
        __syncthreads();
    }

    // Output
    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    float* out_ptr = output + (qpos * num_heads + head) * head_dim;
    for (int i = 0; i < dims_per_thread; ++i) {
        int d = tid + i * THREADS;
        if (d < head_dim) {
            out_ptr[d] = acc[i] * inv_sum;
        }
    }

    // Write LSE = log(running_sum) + running_max
    if (tid == 0) {
        lse[qpos * num_heads + head] = logf(running_sum) + running_max;
    }
}

// ===========================================================================
// F16 kernels (standalone — all computation in f32, I/O in f16)
// ===========================================================================

extern "C" __global__ void flash_prefill_attention_f16(
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
    int offset,
    int window_size
) {
    const int head = blockIdx.x;
    const int qpos = blockIdx.y;
    const int tid = threadIdx.x;

    const int kv_head = head * num_kv_heads / num_heads;
    const int max_valid_k = offset + qpos + 1;
    const int min_valid_k = (window_size > 0)
        ? max(0, max_valid_k - window_size)
        : 0;

    extern __shared__ float smem[];
    float* smem_q      = smem;
    float* smem_kv     = smem + head_dim;
    float* smem_scores = smem + head_dim + BC * head_dim;

    // Load Q into shared memory (f16 → f32)
    const __half* q_ptr = q + (qpos * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += THREADS) {
        smem_q[d] = __half2float(q_ptr[d]);
    }
    __syncthreads();

    const int dims_per_thread = (head_dim + THREADS - 1) / THREADS;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    float running_max = -1e30f;
    float running_sum = 0.0f;

    const int num_tiles = (max_valid_k - min_valid_k + BC - 1) / BC;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_start = min_valid_k + tile * BC;
        const int tile_end = min(tile_start + BC, max_valid_k);
        const int tile_len = tile_end - tile_start;

        // Load K tile (f16 → f32)
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = __half2float(
                k[(kv_pos * num_kv_heads + kv_head) * head_dim + d]);
        }
        __syncthreads();

        float score = -1e30f;
        if (tid < tile_len) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += smem_q[d] * smem_kv[tid * head_dim + d];
            }
            score = MAYBE_SOFTCAP(dot * scale, softcap);
        }
        if (tid < BC) smem_scores[tid] = score;
        __syncthreads();

        float tile_max = -1e30f;
        for (int j = 0; j < tile_len; ++j) {
            if (smem_scores[j] > tile_max) tile_max = smem_scores[j];
        }

        float new_max = fmaxf(running_max, tile_max);
        float old_scale_factor = expf(running_max - new_max);
        float new_sum = running_sum * old_scale_factor;
        for (int i = 0; i < dims_per_thread; ++i) acc[i] *= old_scale_factor;

        // Compute tile_sum = sum(exp(score - new_max)) without overwriting
        // smem_scores (scores are needed again in Phase 3).
        float tile_sum = 0.0f;
        for (int j = 0; j < tile_len; ++j) {
            tile_sum += expf(smem_scores[j] - new_max);
        }

        running_max = new_max;
        running_sum = new_sum + tile_sum;

        // Load V tile (f16 → f32)
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = __half2float(
                v[(kv_pos * num_kv_heads + kv_head) * head_dim + d]);
        }
        __syncthreads();

        // Accumulate: acc[d] += sum_j( exp(score_j - max) * V[j, d] )
        // Weights are recomputed from the original scores (smem_scores
        // is read-only throughout the tile) to avoid a cross-warp shared
        // memory race that would occur if we overwrote scores with weights.
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = tid + i * THREADS;
            if (d < head_dim) {
                float val = 0.0f;
                for (int j = 0; j < tile_len; ++j) {
                    float w = expf(smem_scores[j] - new_max);
                    val += w * smem_kv[j * head_dim + d];
                }
                acc[i] += val;
            }
        }
        __syncthreads();
    }

    // Output (f32 → f16)
    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    __half* out_ptr = output + (qpos * num_heads + head) * head_dim;
    for (int i = 0; i < dims_per_thread; ++i) {
        int d = tid + i * THREADS;
        if (d < head_dim) {
            out_ptr[d] = __float2half(acc[i] * inv_sum);
        }
    }
}

extern "C" __global__ void flash_prefill_attention_with_lse_f16(
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
    int offset,
    int window_size
) {
    const int head = blockIdx.x;
    const int qpos = blockIdx.y;
    const int tid = threadIdx.x;

    const int kv_head = head * num_kv_heads / num_heads;
    const int max_valid_k = offset + qpos + 1;
    const int min_valid_k = (window_size > 0)
        ? max(0, max_valid_k - window_size)
        : 0;

    extern __shared__ float smem[];
    float* smem_q      = smem;
    float* smem_kv     = smem + head_dim;
    float* smem_scores = smem + head_dim + BC * head_dim;

    // Load Q into shared memory (f16 → f32)
    const __half* q_ptr = q + (qpos * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += THREADS) {
        smem_q[d] = __half2float(q_ptr[d]);
    }
    __syncthreads();

    const int dims_per_thread = (head_dim + THREADS - 1) / THREADS;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    float running_max = -1e30f;
    float running_sum = 0.0f;

    const int num_tiles = (max_valid_k - min_valid_k + BC - 1) / BC;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_start = min_valid_k + tile * BC;
        const int tile_end = min(tile_start + BC, max_valid_k);
        const int tile_len = tile_end - tile_start;

        // Load K tile (f16 → f32)
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = __half2float(
                k[(kv_pos * num_kv_heads + kv_head) * head_dim + d]);
        }
        __syncthreads();

        float score = -1e30f;
        if (tid < tile_len) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += smem_q[d] * smem_kv[tid * head_dim + d];
            }
            score = MAYBE_SOFTCAP(dot * scale, softcap);
        }
        if (tid < BC) smem_scores[tid] = score;
        __syncthreads();

        float tile_max = -1e30f;
        for (int j = 0; j < tile_len; ++j) {
            if (smem_scores[j] > tile_max) tile_max = smem_scores[j];
        }

        float new_max = fmaxf(running_max, tile_max);
        float old_scale_factor = expf(running_max - new_max);
        float new_sum = running_sum * old_scale_factor;
        for (int i = 0; i < dims_per_thread; ++i) acc[i] *= old_scale_factor;

        // Compute tile_sum = sum(exp(score - new_max)) without overwriting
        // smem_scores (scores are needed again in Phase 3).
        float tile_sum = 0.0f;
        for (int j = 0; j < tile_len; ++j) {
            tile_sum += expf(smem_scores[j] - new_max);
        }

        running_max = new_max;
        running_sum = new_sum + tile_sum;

        // Load V tile (f16 → f32)
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = __half2float(
                v[(kv_pos * num_kv_heads + kv_head) * head_dim + d]);
        }
        __syncthreads();

        // Accumulate: acc[d] += sum_j( exp(score_j - max) * V[j, d] )
        // Weights are recomputed from the original scores (smem_scores
        // is read-only throughout the tile) to avoid a cross-warp shared
        // memory race that would occur if we overwrote scores with weights.
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = tid + i * THREADS;
            if (d < head_dim) {
                float val = 0.0f;
                for (int j = 0; j < tile_len; ++j) {
                    float w = expf(smem_scores[j] - new_max);
                    val += w * smem_kv[j * head_dim + d];
                }
                acc[i] += val;
            }
        }
        __syncthreads();
    }

    // Output (f32 → f16)
    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    __half* out_ptr = output + (qpos * num_heads + head) * head_dim;
    for (int i = 0; i < dims_per_thread; ++i) {
        int d = tid + i * THREADS;
        if (d < head_dim) {
            out_ptr[d] = __float2half(acc[i] * inv_sum);
        }
    }

    // Write LSE = log(running_sum) + running_max
    if (tid == 0) {
        lse[qpos * num_heads + head] = logf(running_sum) + running_max;
    }
}

// ===========================================================================
// BF16 kernels (standalone — all computation in f32, I/O in bf16)
// ===========================================================================

extern "C" __global__ void flash_prefill_attention_bf16(
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
    int offset,
    int window_size
) {
    const int head = blockIdx.x;
    const int qpos = blockIdx.y;
    const int tid = threadIdx.x;

    const int kv_head = head * num_kv_heads / num_heads;
    const int max_valid_k = offset + qpos + 1;
    const int min_valid_k = (window_size > 0)
        ? max(0, max_valid_k - window_size)
        : 0;

    extern __shared__ float smem[];
    float* smem_q      = smem;
    float* smem_kv     = smem + head_dim;
    float* smem_scores = smem + head_dim + BC * head_dim;

    // Load Q into shared memory (bf16 → f32)
    const __nv_bfloat16* q_ptr = q + (qpos * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += THREADS) {
        smem_q[d] = __bfloat162float(q_ptr[d]);
    }
    __syncthreads();

    const int dims_per_thread = (head_dim + THREADS - 1) / THREADS;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    float running_max = -1e30f;
    float running_sum = 0.0f;

    const int num_tiles = (max_valid_k - min_valid_k + BC - 1) / BC;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_start = min_valid_k + tile * BC;
        const int tile_end = min(tile_start + BC, max_valid_k);
        const int tile_len = tile_end - tile_start;

        // Load K tile (bf16 → f32)
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = __bfloat162float(
                k[(kv_pos * num_kv_heads + kv_head) * head_dim + d]);
        }
        __syncthreads();

        float score = -1e30f;
        if (tid < tile_len) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += smem_q[d] * smem_kv[tid * head_dim + d];
            }
            score = MAYBE_SOFTCAP(dot * scale, softcap);
        }
        if (tid < BC) smem_scores[tid] = score;
        __syncthreads();

        float tile_max = -1e30f;
        for (int j = 0; j < tile_len; ++j) {
            if (smem_scores[j] > tile_max) tile_max = smem_scores[j];
        }

        float new_max = fmaxf(running_max, tile_max);
        float old_scale_factor = expf(running_max - new_max);
        float new_sum = running_sum * old_scale_factor;
        for (int i = 0; i < dims_per_thread; ++i) acc[i] *= old_scale_factor;

        // Compute tile_sum = sum(exp(score - new_max)) without overwriting
        // smem_scores (scores are needed again in Phase 3).
        float tile_sum = 0.0f;
        for (int j = 0; j < tile_len; ++j) {
            tile_sum += expf(smem_scores[j] - new_max);
        }

        running_max = new_max;
        running_sum = new_sum + tile_sum;

        // Load V tile (bf16 → f32)
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = __bfloat162float(
                v[(kv_pos * num_kv_heads + kv_head) * head_dim + d]);
        }
        __syncthreads();

        // Accumulate: acc[d] += sum_j( exp(score_j - max) * V[j, d] )
        // Weights are recomputed from the original scores (smem_scores
        // is read-only throughout the tile) to avoid a cross-warp shared
        // memory race that would occur if we overwrote scores with weights.
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = tid + i * THREADS;
            if (d < head_dim) {
                float val = 0.0f;
                for (int j = 0; j < tile_len; ++j) {
                    float w = expf(smem_scores[j] - new_max);
                    val += w * smem_kv[j * head_dim + d];
                }
                acc[i] += val;
            }
        }
        __syncthreads();
    }

    // Output (f32 → bf16)
    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    __nv_bfloat16* out_ptr = output + (qpos * num_heads + head) * head_dim;
    for (int i = 0; i < dims_per_thread; ++i) {
        int d = tid + i * THREADS;
        if (d < head_dim) {
            out_ptr[d] = __float2bfloat16(acc[i] * inv_sum);
        }
    }
}

extern "C" __global__ void flash_prefill_attention_with_lse_bf16(
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
    int offset,
    int window_size
) {
    const int head = blockIdx.x;
    const int qpos = blockIdx.y;
    const int tid = threadIdx.x;

    const int kv_head = head * num_kv_heads / num_heads;
    const int max_valid_k = offset + qpos + 1;
    const int min_valid_k = (window_size > 0)
        ? max(0, max_valid_k - window_size)
        : 0;

    extern __shared__ float smem[];
    float* smem_q      = smem;
    float* smem_kv     = smem + head_dim;
    float* smem_scores = smem + head_dim + BC * head_dim;

    // Load Q into shared memory (bf16 → f32)
    const __nv_bfloat16* q_ptr = q + (qpos * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += THREADS) {
        smem_q[d] = __bfloat162float(q_ptr[d]);
    }
    __syncthreads();

    const int dims_per_thread = (head_dim + THREADS - 1) / THREADS;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    float running_max = -1e30f;
    float running_sum = 0.0f;

    const int num_tiles = (max_valid_k - min_valid_k + BC - 1) / BC;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_start = min_valid_k + tile * BC;
        const int tile_end = min(tile_start + BC, max_valid_k);
        const int tile_len = tile_end - tile_start;

        // Load K tile (bf16 → f32)
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = __bfloat162float(
                k[(kv_pos * num_kv_heads + kv_head) * head_dim + d]);
        }
        __syncthreads();

        float score = -1e30f;
        if (tid < tile_len) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += smem_q[d] * smem_kv[tid * head_dim + d];
            }
            score = MAYBE_SOFTCAP(dot * scale, softcap);
        }
        if (tid < BC) smem_scores[tid] = score;
        __syncthreads();

        float tile_max = -1e30f;
        for (int j = 0; j < tile_len; ++j) {
            if (smem_scores[j] > tile_max) tile_max = smem_scores[j];
        }

        float new_max = fmaxf(running_max, tile_max);
        float old_scale_factor = expf(running_max - new_max);
        float new_sum = running_sum * old_scale_factor;
        for (int i = 0; i < dims_per_thread; ++i) acc[i] *= old_scale_factor;

        // Compute tile_sum = sum(exp(score - new_max)) without overwriting
        // smem_scores (scores are needed again in Phase 3).
        float tile_sum = 0.0f;
        for (int j = 0; j < tile_len; ++j) {
            tile_sum += expf(smem_scores[j] - new_max);
        }

        running_max = new_max;
        running_sum = new_sum + tile_sum;

        // Load V tile (bf16 → f32)
        for (int idx = tid; idx < tile_len * head_dim; idx += THREADS) {
            int row = idx / head_dim;
            int d = idx % head_dim;
            int kv_pos = tile_start + row;
            smem_kv[row * head_dim + d] = __bfloat162float(
                v[(kv_pos * num_kv_heads + kv_head) * head_dim + d]);
        }
        __syncthreads();

        // Accumulate: acc[d] += sum_j( exp(score_j - max) * V[j, d] )
        // Weights are recomputed from the original scores (smem_scores
        // is read-only throughout the tile) to avoid a cross-warp shared
        // memory race that would occur if we overwrote scores with weights.
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = tid + i * THREADS;
            if (d < head_dim) {
                float val = 0.0f;
                for (int j = 0; j < tile_len; ++j) {
                    float w = expf(smem_scores[j] - new_max);
                    val += w * smem_kv[j * head_dim + d];
                }
                acc[i] += val;
            }
        }
        __syncthreads();
    }

    // Output (f32 → bf16)
    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    __nv_bfloat16* out_ptr = output + (qpos * num_heads + head) * head_dim;
    for (int i = 0; i < dims_per_thread; ++i) {
        int d = tid + i * THREADS;
        if (d < head_dim) {
            out_ptr[d] = __float2bfloat16(acc[i] * inv_sum);
        }
    }

    // Write LSE = log(running_sum) + running_max
    if (tid == 0) {
        lse[qpos * num_heads + head] = logf(running_sum) + running_max;
    }
}
