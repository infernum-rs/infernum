// MMQ kernel: Q8_0 × Q8_1 matmul using WMMA int8 tensor cores
//
// Computes: output[m][n] = sum_k( dequant(act[m][k]) * dequant(weight[n][k]) )
//
// Both operands are int8 quantized with per-32-element block scales.
// The kernel loads tiles of int8 data into shared memory, runs WMMA
// m16n16k16 matmuls, and applies per-block scales in float registers.
//
// Requires sm_75+ (Turing) for WMMA int8 support.

#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Block tile dimensions
#define BLOCK_M 64
#define BLOCK_N 64

// Warp layout: 4×4 = 16 warps, each producing 16×16
#define WARPS_M 4
#define WARPS_N 4
#define NWARPS  (WARPS_M * WARPS_N)

// Q8 block size (elements per scale factor)
#define QK 32

// WMMA m16n16k16 accumulator fragment layout (sm_75+):
// For thread t (lane 0-31) and element index i (0-7):
//   row(i) = (t >> 2) + ((i >> 1) & 1) * 8
//   col(i) = ((t & 3) * 2) + (i & 1) + ((i >> 2) & 1) * 8
#define FRAG_ROW(lane, i)  (((lane) >> 2) + (((i) >> 1) & 1) * 8)
#define FRAG_COL(lane, i)  ((((lane) & 3) * 2) + ((i) & 1) + (((i) >> 2) & 1) * 8)

extern "C" __global__ void mmq_q8_f32(
    float*              __restrict__ output,    // [M, N]
    const signed char*  __restrict__ w_data,    // [N, K] int8 weight values
    const unsigned short* __restrict__ w_scales, // [N, K/32] f16 weight scales
    const signed char*  __restrict__ a_data,    // [M, K] int8 quantized activations
    const float*        __restrict__ a_scales,  // [M, K/32] f32 activation scales
    const int M,
    const int N,
    const int K
) {
    const int bm = blockIdx.y * BLOCK_M;
    const int bn = blockIdx.x * BLOCK_N;
    const int warp_id = threadIdx.y;
    const int lane = threadIdx.x;
    const int wm = (warp_id / WARPS_N) * WMMA_M;
    const int wn = (warp_id % WARPS_N) * WMMA_N;

    // Per-thread float accumulator (8 elements per WMMA tile)
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Fragments
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::col_major> b_frag;

    // Shared memory
    __shared__ signed char smem_a[BLOCK_M * QK];
    __shared__ signed char smem_w[BLOCK_N * QK];
    __shared__ float smem_a_scales[BLOCK_M];
    __shared__ float smem_w_scales[BLOCK_N];

    const int tid = warp_id * 32 + lane;
    const int nthreads = NWARPS * 32;  // 512
    const int num_q_blocks = K / QK;

    for (int qb = 0; qb < num_q_blocks; ++qb) {
        const int k_off = qb * QK;

        // Cooperative load: activation tile [BLOCK_M, QK]
        for (int idx = tid; idx < BLOCK_M * QK; idx += nthreads) {
            int row = idx / QK;
            int col = idx % QK;
            int m = bm + row;
            smem_a[idx] = (m < M) ? a_data[m * K + k_off + col] : 0;
        }

        // Cooperative load: weight tile [BLOCK_N, QK]
        for (int idx = tid; idx < BLOCK_N * QK; idx += nthreads) {
            int row = idx / QK;
            int col = idx % QK;
            int n = bn + row;
            smem_w[idx] = (n < N) ? w_data[n * K + k_off + col] : 0;
        }

        // Load scales into shared memory
        for (int idx = tid; idx < BLOCK_M; idx += nthreads) {
            int m = bm + idx;
            smem_a_scales[idx] = (m < M) ? a_scales[m * num_q_blocks + qb] : 0.0f;
        }
        for (int idx = tid; idx < BLOCK_N; idx += nthreads) {
            int n = bn + idx;
            if (n < N) {
                unsigned short s = w_scales[n * num_q_blocks + qb];
                float sf;
                asm("cvt.f32.f16 %0, %1;" : "=f"(sf) : "h"(s));
                smem_w_scales[idx] = sf;
            } else {
                smem_w_scales[idx] = 0.0f;
            }
        }

        __syncthreads();

        // WMMA: two m16n16k16 steps for 32 columns
        wmma::fill_fragment(c_frag, 0);

        wmma::load_matrix_sync(a_frag, smem_a + wm * QK, QK);
        wmma::load_matrix_sync(b_frag, smem_w + wn * QK, QK);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        wmma::load_matrix_sync(a_frag, smem_a + wm * QK + 16, QK);
        wmma::load_matrix_sync(b_frag, smem_w + wn * QK + 16, QK);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // Scale and accumulate in float registers
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = FRAG_ROW(lane, i);
            int c = FRAG_COL(lane, i);
            float scale = smem_a_scales[wm + r] * smem_w_scales[wn + c];
            acc[i] += (float)c_frag.x[i] * scale;
        }

        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int out_m = bm + wm + FRAG_ROW(lane, i);
        int out_n = bn + wn + FRAG_COL(lane, i);
        if (out_m < M && out_n < N) {
            output[out_m * N + out_n] = acc[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Activation quantization: f32 → Q8 (symmetric per-block quantization)
// ---------------------------------------------------------------------------
extern "C" __global__ void quantize_activations_q8(
    signed char*  __restrict__ q_data,
    float*        __restrict__ q_scales,
    const float*  __restrict__ input,
    const int M,
    const int K
) {
    const int blocks_per_row = K / 32;
    const int total_blocks = M * blocks_per_row;
    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) return;

    const int m = block_idx / blocks_per_row;
    const int b = block_idx % blocks_per_row;
    const int base = m * K + b * 32;

    float amax = 0.0f;
    float vals[32];
    for (int i = 0; i < 32; ++i) {
        vals[i] = input[base + i];
        float a = fabsf(vals[i]);
        if (a > amax) amax = a;
    }

    float scale = amax / 127.0f;
    float inv_scale = (amax > 0.0f) ? 127.0f / amax : 0.0f;

    for (int i = 0; i < 32; ++i) {
        int q = (int)roundf(vals[i] * inv_scale);
        q = max(-128, min(127, q));
        q_data[base + i] = (signed char)q;
    }

    q_scales[block_idx] = scale;
}
