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

/// Tiled dense linear: C = input × weight_t^T.
/// input is (M, K), weight_t is (N, K), output is (M, N).
///
/// Uses 16 SIMD-groups (4×4) per threadgroup, each computing one 8×8 output tile.
/// Total output tile per threadgroup: 32×32.
///
/// K_BLK=32: each SIMD group's 32 threads read 32 consecutive floats from the SAME
/// row of A or B — exactly one 128-byte cache line per load instruction. Fully
/// coalesced. The earlier K_BLK=8 version had threads read different rows within one
/// SIMD group, causing 4× more transactions than necessary.
///
/// Load layout per outer K-iteration (4 loads per thread):
///   - sA rows 0..15:  SIMD group g loads A[row_base+g, k_base..k_base+31]
///   - sA rows 16..31: SIMD group g loads A[row_base+g+16, k_base..k_base+31]
///   - sB rows 0..15:  SIMD group g loads wt[col_base+g, k_base..k_base+31]
///   - sB rows 16..31: SIMD group g loads wt[col_base+g+16, k_base..k_base+31]
/// All loads from global are contiguous within each SIMD group (lane = K index).
///
/// After the barrier, 4 inner 8-wide MMA steps consume the 32 K-elements in shmem.
///
/// shmem: sA[32×32]=4 KB + sB[32×32]=4 KB = 8 KB total (well within Metal's 16 KB min).
/// Grid: (ceil(N/32), ceil(M/32), 1), threads_per_group = (512, 1, 1).

constant constexpr uint TG_M = 32;
constant constexpr uint TG_N = 32;
constant constexpr uint K_BLK = 32;

kernel void linear_dense_f32_tiled(
    device const float* inp  [[buffer(0)]],
    device const float* wt   [[buffer(1)]],
    device float*       outp [[buffer(2)]],
    constant MatmulParams& p [[buffer(3)]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  sg_id [[simdgroup_index_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  tid   [[thread_index_in_threadgroup]])
{
    const uint M = p.M, N = p.N, K = p.K;

    const uint row_base = tgid.y * TG_M;
    const uint col_base = tgid.x * TG_N;

    // This SIMD group's sub-tile within the 32×32 output block.
    const uint sg_row = sg_id / 4;  // 0..3
    const uint sg_col = sg_id % 4;  // 0..3
    const uint out_row = row_base + sg_row * 8;
    const uint out_col = col_base + sg_col * 8;

    // sA[TG_M × K_BLK] = [32][32] = 4 KB
    // sB[TG_N × K_BLK] = [32][32] = 4 KB
    threadgroup float sA[TG_M * K_BLK];
    threadgroup float sB[TG_N * K_BLK];

    simdgroup_float8x8 acc(0.0f);

    // sg_load == sg_id: SIMD group index used for loading.
    // lane is also the K-local column index (0..31 = K_BLK).
    const uint sg_load = sg_id;  // 0..15
    const uint kl      = lane;   // 0..31 — K-position within this block

    for (uint ki = 0; ki < (K + K_BLK - 1) / K_BLK; ki++) {
        const uint k_base = ki * K_BLK;
        const uint gk = k_base + kl;

        // Coalesced load: each SIMD group reads one 32-float row (= one cache line).
        // Two rows of sA and two rows of sB per group (4 loads per thread total).
        uint gr0 = row_base + sg_load;
        uint gr1 = row_base + sg_load + 16;
        uint gc0 = col_base + sg_load;
        uint gc1 = col_base + sg_load + 16;

        sA[sg_load * K_BLK + kl]        = (gr0 < M && gk < K) ? inp[gr0 * K + gk] : 0.0f;
        sA[(sg_load + 16) * K_BLK + kl] = (gr1 < M && gk < K) ? inp[gr1 * K + gk] : 0.0f;
        sB[sg_load * K_BLK + kl]        = (gc0 < N && gk < K) ? wt[gc0 * K + gk] : 0.0f;
        sB[(sg_load + 16) * K_BLK + kl] = (gc1 < N && gk < K) ? wt[gc1 * K + gk] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4 inner MMA steps, each consuming 8 of the 32 K-elements from shmem.
        for (uint ki_inner = 0; ki_inner < 4; ki_inner++) {
            simdgroup_float8x8 a_tile, b_tile;
            // sA: [TG_M][K_BLK] row-major, stride=K_BLK
            simdgroup_load(a_tile, sA + sg_row * 8 * K_BLK + ki_inner * 8, K_BLK);
            // sB: [TG_N][K_BLK] row-major, transposed load to get [K][8] view
            simdgroup_load(b_tile, sB + sg_col * 8 * K_BLK + ki_inner * 8, K_BLK,
                           ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (out_row >= M || out_col >= N) return;

    if (out_row + 8 <= M && out_col + 8 <= N) {
        simdgroup_store(acc, outp + out_row * N + out_col, N);
    } else {
        threadgroup float tmp[64];
        simdgroup_store(acc, tmp, 8);
        uint r = lane / 8, c = lane % 8;
        if (out_row + r < M && out_col + c < N)
            outp[(out_row + r) * N + out_col + c] = tmp[r * 8 + c];
        if (out_row + r + 4 < M && out_col + c < N)
            outp[(out_row + r + 4) * N + out_col + c] = tmp[(r + 4) * 8 + c];
    }
}

/// Fused dequantize+GEMM for Q8_0 GGUF blocks.
///
/// weight is stored as interleaved blocks: [half d (2 bytes) | int8_t qs[32] (32 bytes)]
/// = 34 bytes per block. K must be a multiple of 32 (GGUF block size).
///
/// Dequantizes on the fly inside the K-loop — no intermediate F32 buffer.
/// Weight data is read only once (vs twice in the separate dequant+GEMM path).
///
/// Same tile structure as linear_dense_f32_tiled (32×32 output per threadgroup).
/// Grid: (ceil(N/32), ceil(M/32), 1), threads_per_group = (512, 1, 1).

struct block_q8_0 {
    half d;
    int8_t qs[32];
};

kernel void linear_q8_blocks_f32_tiled(
    device const float*    inp  [[buffer(0)]],
    device const uchar*    wt   [[buffer(1)]],
    device float*          outp [[buffer(2)]],
    constant MatmulParams& p    [[buffer(3)]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  sg_id [[simdgroup_index_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    const uint M = p.M, N = p.N, K = p.K;
    const uint num_blocks_k = K / K_BLK;  // K_BLK=32 = Q8_0 block size

    const uint row_base = tgid.y * TG_M;
    const uint col_base = tgid.x * TG_N;
    const uint sg_row = sg_id / 4;
    const uint sg_col = sg_id % 4;
    const uint out_row = row_base + sg_row * 8;
    const uint out_col = col_base + sg_col * 8;

    threadgroup float sA[TG_M * K_BLK];
    threadgroup float sB[TG_N * K_BLK];

    simdgroup_float8x8 acc(0.0f);

    const uint sg_load = sg_id;
    const uint kl = lane;

    device const block_q8_0* wt_blocks = (device const block_q8_0*)wt;

    for (uint ki = 0; ki < num_blocks_k; ki++) {
        const uint k_base = ki * K_BLK;

        // Load sA: F32 activations (two rows per SIMD group, coalesced)
        uint gr0 = row_base + sg_load;
        uint gr1 = row_base + sg_load + 16;
        sA[sg_load * K_BLK + kl]        = (gr0 < M) ? inp[gr0 * K + k_base + kl] : 0.0f;
        sA[(sg_load + 16) * K_BLK + kl] = (gr1 < M) ? inp[gr1 * K + k_base + kl] : 0.0f;

        // Load sB: dequantize Q8_0 blocks on the fly
        // Thread kl reads element kl from block (gc, ki): scale * qs[kl].
        // All 32 threads access consecutive bytes of qs[] — coalesced.
        uint gc0 = col_base + sg_load;
        uint gc1 = col_base + sg_load + 16;

        if (gc0 < N) {
            device const block_q8_0& blk = wt_blocks[gc0 * num_blocks_k + ki];
            sB[sg_load * K_BLK + kl] = float(blk.qs[kl]) * float(blk.d);
        } else {
            sB[sg_load * K_BLK + kl] = 0.0f;
        }
        if (gc1 < N) {
            device const block_q8_0& blk = wt_blocks[gc1 * num_blocks_k + ki];
            sB[(sg_load + 16) * K_BLK + kl] = float(blk.qs[kl]) * float(blk.d);
        } else {
            sB[(sg_load + 16) * K_BLK + kl] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ki_inner = 0; ki_inner < 4; ki_inner++) {
            simdgroup_float8x8 a_tile, b_tile;
            simdgroup_load(a_tile, sA + sg_row * 8 * K_BLK + ki_inner * 8, K_BLK);
            simdgroup_load(b_tile, sB + sg_col * 8 * K_BLK + ki_inner * 8, K_BLK,
                           ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (out_row >= M || out_col >= N) return;
    if (out_row + 8 <= M && out_col + 8 <= N) {
        simdgroup_store(acc, outp + out_row * N + out_col, N);
    } else {
        threadgroup float tmp[64];
        simdgroup_store(acc, tmp, 8);
        uint r = lane / 8, c = lane % 8;
        if (out_row + r < M && out_col + c < N)
            outp[(out_row + r) * N + out_col + c] = tmp[r * 8 + c];
        if (out_row + r + 4 < M && out_col + c < N)
            outp[(out_row + r + 4) * N + out_col + c] = tmp[(r + 4) * 8 + c];
    }
}

/// Fused dequantize+GEMM for Q4_0 GGUF blocks.
///
/// weight is stored as interleaved blocks: [half d (2 bytes) | uint8_t qs[16] (16 bytes)]
/// = 18 bytes per block. 32 elements per block stored as 16-byte nibble pairs.
/// Dequant: value = (nibble - 8) * d, nibbles unsigned [0,15] centered at 8.
///
/// Same tile structure as linear_q8_blocks_f32_tiled.

struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

kernel void linear_q4_blocks_f32_tiled(
    device const float*    inp  [[buffer(0)]],
    device const uchar*    wt   [[buffer(1)]],
    device float*          outp [[buffer(2)]],
    constant MatmulParams& p    [[buffer(3)]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  sg_id [[simdgroup_index_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    const uint M = p.M, N = p.N, K = p.K;
    const uint num_blocks_k = K / K_BLK;

    const uint row_base = tgid.y * TG_M;
    const uint col_base = tgid.x * TG_N;
    const uint sg_row = sg_id / 4;
    const uint sg_col = sg_id % 4;
    const uint out_row = row_base + sg_row * 8;
    const uint out_col = col_base + sg_col * 8;

    threadgroup float sA[TG_M * K_BLK];
    threadgroup float sB[TG_N * K_BLK];

    simdgroup_float8x8 acc(0.0f);

    const uint sg_load = sg_id;
    const uint kl = lane;

    device const block_q4_0* wt_blocks = (device const block_q4_0*)wt;

    for (uint ki = 0; ki < num_blocks_k; ki++) {
        const uint k_base = ki * K_BLK;

        uint gr0 = row_base + sg_load;
        uint gr1 = row_base + sg_load + 16;
        sA[sg_load * K_BLK + kl]        = (gr0 < M) ? inp[gr0 * K + k_base + kl] : 0.0f;
        sA[(sg_load + 16) * K_BLK + kl] = (gr1 < M) ? inp[gr1 * K + k_base + kl] : 0.0f;

        uint gc0 = col_base + sg_load;
        uint gc1 = col_base + sg_load + 16;

        // Each nibble byte holds elements kl and kl+16; lo nibble = even kl, hi = odd-indexed.
        // kl[0..15]: lo nibble of qs[kl],   kl[16..31]: hi nibble of qs[kl-16].
        if (gc0 < N) {
            device const block_q4_0& blk = wt_blocks[gc0 * num_blocks_k + ki];
            float scale = float(blk.d);
            uint8_t byte_val = blk.qs[kl % 16];
            int nibble = (kl < 16) ? int(byte_val & 0xF) : int(byte_val >> 4);
            sB[sg_load * K_BLK + kl] = float(nibble - 8) * scale;
        } else {
            sB[sg_load * K_BLK + kl] = 0.0f;
        }
        if (gc1 < N) {
            device const block_q4_0& blk = wt_blocks[gc1 * num_blocks_k + ki];
            float scale = float(blk.d);
            uint8_t byte_val = blk.qs[kl % 16];
            int nibble = (kl < 16) ? int(byte_val & 0xF) : int(byte_val >> 4);
            sB[(sg_load + 16) * K_BLK + kl] = float(nibble - 8) * scale;
        } else {
            sB[(sg_load + 16) * K_BLK + kl] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ki_inner = 0; ki_inner < 4; ki_inner++) {
            simdgroup_float8x8 a_tile, b_tile;
            simdgroup_load(a_tile, sA + sg_row * 8 * K_BLK + ki_inner * 8, K_BLK);
            simdgroup_load(b_tile, sB + sg_col * 8 * K_BLK + ki_inner * 8, K_BLK,
                           ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (out_row >= M || out_col >= N) return;
    if (out_row + 8 <= M && out_col + 8 <= N) {
        simdgroup_store(acc, outp + out_row * N + out_col, N);
    } else {
        threadgroup float tmp[64];
        simdgroup_store(acc, tmp, 8);
        uint r = lane / 8, c = lane % 8;
        if (out_row + r < M && out_col + c < N)
            outp[(out_row + r) * N + out_col + c] = tmp[r * 8 + c];
        if (out_row + r + 4 < M && out_col + c < N)
            outp[(out_row + r + 4) * N + out_col + c] = tmp[(r + 4) * 8 + c];
    }
}

/// Dense linear: C = input × weight_t^T.
/// input is (M, K), weight_t is (N, K), output is (M, N).
///
/// Uses SIMD-group 8×8 with column-major load to transpose weight_t.
/// Used for M=1 (decode) where the tiled kernel would be wasteful.
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
