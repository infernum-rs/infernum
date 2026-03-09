#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

/// Dense matmul using SIMD-group 8×8 matrix multiply-accumulate.
///
/// A is (M, K), B is (K, N), C is (M, N).
/// Each SIMD-group (32 threads) computes one 8×8 output tile of C.
/// Grid: threadgroups = (ceil(N/8), ceil(M/8), 1), threads_per_group = (32, 1, 1).
///
/// Uses threadgroup memory to zero-pad edge tiles so that
/// simdgroup_load always reads valid data.

struct MatmulParams {
    uint M;
    uint N;
    uint K;
};

kernel void matmul_f32(
    device const float* A           [[buffer(0)]],
    device const float* B           [[buffer(1)]],
    device float* C                 [[buffer(2)]],
    constant MatmulParams& params   [[buffer(3)]],
    uint2 group_id                  [[threadgroup_position_in_grid]],
    uint simd_lane                  [[thread_index_in_simdgroup]])
{
    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;

    const uint col_block = group_id.x;
    const uint row_block = group_id.y;

    const uint row_base = row_block * 8;
    const uint col_base = col_block * 8;

    if (row_base >= M || col_base >= N) return;

    // Threadgroup scratch for padding edge tiles.
    threadgroup float tgA[8][8];
    threadgroup float tgB[8][8];

    simdgroup_float8x8 acc(0.0f);

    const uint num_k_blocks = (K + 7) / 8;

    // Check if this tile touches edges (needs padding)
    const bool edge_m = (row_base + 8 > M);
    const bool edge_n = (col_base + 8 > N);
    const bool edge_k = (K % 8 != 0);

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        simdgroup_float8x8 a_tile, b_tile;
        const uint k_base = kb * 8;
        const bool k_edge = (kb == num_k_blocks - 1) && edge_k;

        if (!edge_m && !k_edge) {
            // Full A tile: load directly
            simdgroup_load(a_tile, A + row_base * K + k_base, K);
        } else {
            // Pad A tile through threadgroup memory
            // 32 threads cover 64 elements = 8×8
            uint r = simd_lane / 8;
            uint c = simd_lane % 8;
            // Two passes for 64 elements with 32 threads
            tgA[r][c] = (row_base + r < M && k_base + c < K) ?
                A[(row_base + r) * K + k_base + c] : 0.0f;
            tgA[r + 4][c] = (row_base + r + 4 < M && k_base + c < K) ?
                A[(row_base + r + 4) * K + k_base + c] : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(a_tile, &tgA[0][0], 8);
        }

        if (!edge_n && !k_edge) {
            // Full B tile: load directly
            simdgroup_load(b_tile, B + k_base * N + col_base, N);
        } else {
            uint r = simd_lane / 8;
            uint c = simd_lane % 8;
            tgB[r][c] = (k_base + r < K && col_base + c < N) ?
                B[(k_base + r) * N + col_base + c] : 0.0f;
            tgB[r + 4][c] = (k_base + r + 4 < K && col_base + c < N) ?
                B[(k_base + r + 4) * N + col_base + c] : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(b_tile, &tgB[0][0], 8);
        }

        simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);

        // Barrier needed if we used threadgroup memory and will reuse it
        if (edge_m || edge_n || k_edge) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Store result
    if (!edge_m && !edge_n) {
        simdgroup_store(acc, C + row_base * N + col_base, N);
    } else {
        threadgroup float out_tile[8][8];
        simdgroup_store(acc, &out_tile[0][0], 8);
        uint r = simd_lane / 8;
        uint c = simd_lane % 8;
        if (row_base + r < M && col_base + c < N) {
            C[(row_base + r) * N + col_base + c] = out_tile[r][c];
        }
        if (row_base + r + 4 < M && col_base + c < N) {
            C[(row_base + r + 4) * N + col_base + c] = out_tile[r + 4][c];
        }
    }
}

/// Dense linear: C = input × weight_t^T.
/// input is (M, K), weight_t is (N, K), output is (M, N).
///
/// Uses SIMD-group 8×8 with column-major load to transpose weight_t.
kernel void linear_dense_f32(
    device const float* input       [[buffer(0)]],
    device const float* weight_t    [[buffer(1)]],
    device float* output            [[buffer(2)]],
    constant MatmulParams& params   [[buffer(3)]],
    uint2 group_id                  [[threadgroup_position_in_grid]],
    uint simd_lane                  [[thread_index_in_simdgroup]])
{
    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;

    const uint col_block = group_id.x;
    const uint row_block = group_id.y;

    const uint row_base = row_block * 8;
    const uint col_base = col_block * 8;

    if (row_base >= M || col_base >= N) return;

    threadgroup float tgA[8][8];
    threadgroup float tgB[8][8];

    simdgroup_float8x8 acc(0.0f);

    const uint num_k_blocks = (K + 7) / 8;
    const bool edge_m = (row_base + 8 > M);
    const bool edge_n = (col_base + 8 > N);
    const bool edge_k = (K % 8 != 0);

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        simdgroup_float8x8 a_tile, b_tile;
        const uint k_base = kb * 8;
        const bool k_edge = (kb == num_k_blocks - 1) && edge_k;

        // Load A tile: input[row_base..+8, k_base..+8]
        if (!edge_m && !k_edge) {
            simdgroup_load(a_tile, input + row_base * K + k_base, K);
        } else {
            uint r = simd_lane / 8;
            uint c = simd_lane % 8;
            tgA[r][c] = (row_base + r < M && k_base + c < K) ?
                input[(row_base + r) * K + k_base + c] : 0.0f;
            tgA[r + 4][c] = (row_base + r + 4 < M && k_base + c < K) ?
                input[(row_base + r + 4) * K + k_base + c] : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(a_tile, &tgA[0][0], 8);
        }

        // Load B tile: need B[k,n] = weight_t[n,k]
        // weight_t is (N,K), we want a (K,N) view — load column-major
        if (!edge_n && !k_edge) {
            simdgroup_load(b_tile, weight_t + col_base * K + k_base, K, ulong2(0, 0), true);
        } else {
            // Manually load transposed: tgB[k_local][n_local] = weight_t[n, k]
            uint r = simd_lane / 8;  // k_local
            uint c = simd_lane % 8;  // n_local
            tgB[r][c] = (k_base + r < K && col_base + c < N) ?
                weight_t[(col_base + c) * K + k_base + r] : 0.0f;
            tgB[r + 4][c] = (k_base + r + 4 < K && col_base + c < N) ?
                weight_t[(col_base + c) * K + k_base + r + 4] : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(b_tile, &tgB[0][0], 8);
        }

        simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);

        if (edge_m || edge_n || k_edge) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (!edge_m && !edge_n) {
        simdgroup_store(acc, output + row_base * N + col_base, N);
    } else {
        threadgroup float out_tile[8][8];
        simdgroup_store(acc, &out_tile[0][0], 8);
        uint r = simd_lane / 8;
        uint c = simd_lane % 8;
        if (row_base + r < M && col_base + c < N) {
            output[(row_base + r) * N + col_base + c] = out_tile[r][c];
        }
        if (row_base + r + 4 < M && col_base + c < N) {
            output[(row_base + r + 4) * N + col_base + c] = out_tile[r + 4][c];
        }
    }
}
