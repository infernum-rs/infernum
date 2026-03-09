#include <metal_stdlib>
using namespace metal;

/// Dense matmul: C[row,col] = dot(A[row,:], B[:,col]).
/// A is (M, K), B is (K, N), C is (M, N).
///
/// Tiled with TILE×TILE threadgroups using shared memory.
/// Each threadgroup computes a TILE×TILE block of C.
constant constexpr uint TILE = 16;

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
    uint2 gid                       [[thread_position_in_grid]],
    uint2 lid                       [[thread_position_in_threadgroup]])
{
    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;

    const uint row = gid.y;
    const uint col = gid.x;

    threadgroup float tileA[TILE][TILE];
    threadgroup float tileB[TILE][TILE];

    float sum = 0.0f;

    const uint numTiles = (K + TILE - 1) / TILE;
    for (uint t = 0; t < numTiles; t++) {
        // Load A tile — all threads participate regardless of output bounds
        const uint a_col = t * TILE + lid.x;
        tileA[lid.y][lid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // Load B tile
        const uint b_row = t * TILE + lid.y;
        tileB[lid.y][lid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE; i++) {
            sum += tileA[lid.y][i] * tileB[i][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Only write output for valid positions
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/// Dense linear: C[row,col] = dot(input[row,:], weight_t[col,:]).
/// weight_t is (N, K) — pre-transposed (each row is one output neuron).
/// input is (M, K), output is (M, N).
///
/// Same tiled GEMM but with B accessed in transposed layout.
kernel void linear_dense_f32(
    device const float* input       [[buffer(0)]],
    device const float* weight_t    [[buffer(1)]],
    device float* output            [[buffer(2)]],
    constant MatmulParams& params   [[buffer(3)]],
    uint2 gid                       [[thread_position_in_grid]],
    uint2 lid                       [[thread_position_in_threadgroup]])
{
    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;

    const uint row = gid.y;
    const uint col = gid.x;

    threadgroup float tileA[TILE][TILE];
    threadgroup float tileW[TILE][TILE];

    float sum = 0.0f;

    const uint numTiles = (K + TILE - 1) / TILE;
    for (uint t = 0; t < numTiles; t++) {
        const uint a_col = t * TILE + lid.x;
        tileA[lid.y][lid.x] = (row < M && a_col < K) ? input[row * K + a_col] : 0.0f;

        // weight_t[col, k] — each row is one output neuron
        const uint w_col = t * TILE + lid.y;
        tileW[lid.y][lid.x] = (col < N && w_col < K) ? weight_t[col * K + w_col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE; i++) {
            sum += tileA[lid.y][i] * tileW[i][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        output[row * N + col] = sum;
    }
}
