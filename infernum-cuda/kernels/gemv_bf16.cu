// GEMV kernels for M=1 decode.
//
// Weight convention: (K, N) row-major — input (1,K) @ weight (K,N) -> (1,N).
//
// Strategy — K-split / row-split GEMV (mirrors cuBLAS cublasGemvEx layout):
//
//   Block: (COLS_PER_CTA=32, K_SPLITS=16, 1) = 512 threads.
//     - threadIdx.x (0..31): which of the 32 output columns this thread owns.
//     - threadIdx.y (0..15): which 1/K_SPLITS chunk of K this thread processes.
//   Grid: (ceil(N / COLS_PER_CTA), 1)
//
//   Each thread accumulates K/K_SPLITS products in float32. Then all K_SPLITS
//   threads in the same x-column reduce their partial sums via shared memory.
//
//   Memory access pattern:
//     Thread (tx, ty) reads weight[(k_base + k_local) * N + col_base + tx]
//     for k_local in ty's chunk. Within any K-iteration, all 32 threads in
//     the same ty-row read 32 consecutive BF16 values → 64-byte coalesced
//     transaction (half a cache line). With 16 ty-rows active simultaneously,
//     16 warps issue 16 independent loads → high memory-level parallelism.
//
//   Shared memory reduction:
//     After the K-loop, each tx-column has K_SPLITS partial sums stored in
//     smem[ty][tx]. Thread (tx, 0) sums all K_SPLITS entries and writes output.
//
//   BF16 variant: accumulate in float32, write BF16 output.
//   F32 variant: all float32.

#include <cuda_bf16.h>
#include <stdint.h>

// Threads per CTA in N direction (= output columns per CTA).
// Must be a warp multiple; 32 gives perfectly coalesced loads.
#define COLS_PER_CTA 32

// K-split factor: how many thread-rows split the K dimension.
// 16 gives K_per_thread = K/16 ≈ 60 for K=960 → good arithmetic intensity.
#define K_SPLITS 16

// Shared memory for inter-row reduction: [K_SPLITS][COLS_PER_CTA].
#define SMEM_SIZE (K_SPLITS * COLS_PER_CTA)

// =========================================================================
// BF16 GEMV
// grid:  (ceil(N / COLS_PER_CTA), 1)
// block: (COLS_PER_CTA, K_SPLITS, 1) = 512 threads
// =========================================================================
extern "C" __global__ void gemv_bf16(
    const __nv_bfloat16 * __restrict__ weight,  // (K, N) row-major
    const __nv_bfloat16 * __restrict__ input,   // (K,)
    __nv_bfloat16       * __restrict__ output,  // (N,)
    int N,
    int K
) {
    __shared__ float smem[K_SPLITS][COLS_PER_CTA];

    const int tx = (int)threadIdx.x;   // column index within CTA (0..31)
    const int ty = (int)threadIdx.y;   // K-split index (0..15)
    const int col = (int)blockIdx.x * COLS_PER_CTA + tx;

    float acc = 0.0f;

    if (col < N) {
        // This thread processes K-elements [k_start, k_end).
        const int k_chunk = (K + K_SPLITS - 1) / K_SPLITS;
        const int k_start = ty * k_chunk;
        const int k_end   = (k_start + k_chunk < K) ? (k_start + k_chunk) : K;

        const __nv_bfloat16 *wcol = weight + col;  // weight[k*N + col] = wcol[k*N]
        for (int k = k_start; k < k_end; ++k) {
            acc += __bfloat162float(wcol[k * N]) * __bfloat162float(input[k]);
        }
    }

    smem[ty][tx] = acc;
    __syncthreads();

    // Only the ty==0 row reduces and writes output.
    if (ty == 0 && col < N) {
        float sum = 0.0f;
        #pragma unroll
        for (int s = 0; s < K_SPLITS; ++s) {
            sum += smem[s][tx];
        }
        output[col] = __float2bfloat16(sum);
    }
}

// =========================================================================
// F32 GEMV
// grid:  (ceil(N / COLS_PER_CTA), 1)
// block: (COLS_PER_CTA, K_SPLITS, 1) = 512 threads
// =========================================================================
extern "C" __global__ void gemv_f32(
    const float * __restrict__ weight,  // (K, N) row-major
    const float * __restrict__ input,   // (K,)
    float       * __restrict__ output,  // (N,)
    int N,
    int K
) {
    __shared__ float smem[K_SPLITS][COLS_PER_CTA];

    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;
    const int col = (int)blockIdx.x * COLS_PER_CTA + tx;

    float acc = 0.0f;

    if (col < N) {
        const int k_chunk = (K + K_SPLITS - 1) / K_SPLITS;
        const int k_start = ty * k_chunk;
        const int k_end   = (k_start + k_chunk < K) ? (k_start + k_chunk) : K;

        const float *wcol = weight + col;
        for (int k = k_start; k < k_end; ++k) {
            acc += wcol[k * N] * input[k];
        }
    }

    smem[ty][tx] = acc;
    __syncthreads();

    if (ty == 0 && col < N) {
        float sum = 0.0f;
        #pragma unroll
        for (int s = 0; s < K_SPLITS; ++s) {
            sum += smem[s][tx];
        }
        output[col] = sum;
    }
}
