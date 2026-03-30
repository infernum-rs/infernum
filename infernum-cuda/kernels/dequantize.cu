// Dequantize kernels: expand quantized weights to f16 for cuBLAS GEMM
//
// Used for prefill (M > 1) where cuBLAS with tensor cores is much faster
// than the custom tiled matmul kernels. The M=1 decode path still uses
// the dedicated GEMV kernels.
//
// Each kernel writes to a __half output buffer of shape (N, K) in row-major order.

#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Q8_0 → f16
// ---------------------------------------------------------------------------
// data:   int8 values, shape (N, K) contiguous
// scales: f16 values, shape (N, K/32) — one scale per block of 32
// output: f16 values, shape (N, K)
//
// Launch: grid=(ceil(N*K / 256),), block=(256,)
extern "C" __global__ void dequant_q8_f16(
    __half*              __restrict__ output,
    const signed char*   __restrict__ data,
    const unsigned short* __restrict__ scales,  // f16 stored as u16
    const int N,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * K;
    if (idx >= total) return;

    const int n = idx / K;
    const int k = idx % K;
    const int block_idx = k / 32;
    const int blocks_per_row = K / 32;

    __half scale = *reinterpret_cast<const __half*>(&scales[n * blocks_per_row + block_idx]);
    float w = (float)data[idx] * __half2float(scale);
    output[idx] = __float2half(w);
}

// ---------------------------------------------------------------------------
// Q4_0 → f16
// ---------------------------------------------------------------------------
// GGML Q4_0 non-consecutive packing:
//   byte[j] has element j in low nibble, element j+16 in high nibble
// data:   uint8 packed, shape (N, K/2)
// scales: f16 values, shape (N, K/32)
// output: f16 values, shape (N, K)
//
// Launch: grid=(ceil(N*K / 256),), block=(256,)
extern "C" __global__ void dequant_q4_f16(
    __half*              __restrict__ output,
    const unsigned char* __restrict__ data,
    const unsigned short* __restrict__ scales,
    const int N,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * K;
    if (idx >= total) return;

    const int n = idx / K;
    const int k = idx % K;
    const int block_idx = k / 32;
    const int blocks_per_row = K / 32;

    __half scale = *reinterpret_cast<const __half*>(&scales[n * blocks_per_row + block_idx]);

    // Q4_0 non-consecutive: within a 32-element block,
    // elements 0..15 are in the low nibble of bytes 0..15,
    // elements 16..31 are in the high nibble of bytes 0..15
    int k_in_block = k % 32;
    int byte_offset = n * (K / 2) + block_idx * 16 + (k_in_block % 16);
    unsigned char packed = data[byte_offset];
    int q;
    if (k_in_block < 16) {
        q = (packed & 0xF) - 8;
    } else {
        q = ((packed >> 4) & 0xF) - 8;
    }
    float w = (float)q * __half2float(scale);
    output[idx] = __float2half(w);
}

// ---------------------------------------------------------------------------
// Q6_K → f16
// ---------------------------------------------------------------------------
// Q6_K super-blocks: 256 elements each, 210 bytes per super-block.
// Layout per super-block: ql[128] | qh[64] | scales[16] | d(f16)
//
// The ql/qh byte layout uses ggml's interleaved mapping (8×32 → 16×16 reshape).
// This kernel replicates the exact same element-to-byte mapping as matmul_q6k_f32.
//
// Launch: grid=(ceil(N*K / 256),), block=(256,)
extern "C" __global__ void dequant_q6k_f16(
    __half*              __restrict__ output,
    const unsigned char* __restrict__ data,
    const int N,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * K;
    if (idx >= total) return;

    const int n = idx / K;
    const int k = idx % K;

    const int BLOCK_SIZE = 256;
    const int BLOCK_BYTES = 210;
    const int blocks_per_row = K / BLOCK_SIZE;
    const int block_idx = k / BLOCK_SIZE;
    const int elem = k % BLOCK_SIZE;  // element index within super-block [0, 255]

    const unsigned char* block_ptr = data + (n * blocks_per_row + block_idx) * BLOCK_BYTES;
    const unsigned char* ql = block_ptr;
    const unsigned char* qh = block_ptr + 128;
    const signed char* scales = (const signed char*)(block_ptr + 192);
    float d = __half2float(*reinterpret_cast<const __half*>(block_ptr + 208));

    // Map element index to (row8, col32) via 16×16 → 8×32 reshape
    int sb = elem / 16;           // sub-block 0-15
    int sb_elem = elem % 16;      // element within sub-block
    int flat_idx = sb * 16 + sb_elem;
    int row8 = flat_idx / 32;     // 0-7
    int col32 = flat_idx % 32;    // 0-31

    // ql layout after reshape to (8,32):
    // row 0: bytes 0-31 low nibbles    row 4: bytes 64-95 low nibbles
    // row 1: bytes 32-63 low nibbles   row 5: bytes 96-127 low nibbles
    // row 2: bytes 0-31 high nibbles   row 6: bytes 64-95 high nibbles
    // row 3: bytes 32-63 high nibbles  row 7: bytes 96-127 high nibbles
    int ql_half = row8 / 4;           // 0 for rows 0-3, 1 for rows 4-7
    int ql_nibble_sel = (row8 % 4) / 2; // 0 for rows 0-1,4-5 (low), 1 for 2-3,6-7 (high)
    int ql_offset = (row8 % 4) % 2;   // 0 for even rows in group, 1 for odd
    int ql_byte_idx = ql_half * 64 + ql_offset * 32 + col32;
    unsigned char ql_byte = ql[ql_byte_idx];
    int ql_val = (ql_nibble_sel == 0) ? (ql_byte & 0x0F) : (ql_byte >> 4);

    // qh layout: (8,32) with 2-bit shift selection
    // row 0: bytes 0-31 bits 0-1   row 4: bytes 32-63 bits 0-1
    // row 1: bytes 0-31 bits 2-3   row 5: bytes 32-63 bits 2-3
    // row 2: bytes 0-31 bits 4-5   row 6: bytes 32-63 bits 4-5
    // row 3: bytes 0-31 bits 6-7   row 7: bytes 32-63 bits 6-7
    int qh_half = row8 / 4;
    int qh_shift_sel = row8 % 4;
    int qh_byte_idx = qh_half * 32 + col32;
    unsigned char qh_byte = qh[qh_byte_idx];
    int qh_val = (qh_byte >> (qh_shift_sel * 2)) & 0x03;

    // Combine to 6-bit [0,63], center to signed [-32,31]
    int q = (ql_val | (qh_val << 4)) - 32;

    float sc = (float)scales[sb];
    float w = d * sc * (float)q;
    output[idx] = __float2half(w);
}

// ---------------------------------------------------------------------------
// GPTQ INT4 → f16
// ---------------------------------------------------------------------------
// GPTQ transposed layout (repacked at load time):
//   qweight: [N, K/8] as int32 (8 nibbles per int32)
//   scales:  [N, num_groups] as f16
//   qzeros:  [N/8, num_groups] as int32 (packed INT4 zero-points)
//
// Dequant: w = (q - (stored_qzero + 1)) * scale
//
// Launch: grid=(ceil(N*K / 256),), block=(256,)
extern "C" __global__ void dequant_gptq_f16(
    __half*              __restrict__ output,
    const int*           __restrict__ qweight,
    const unsigned short* __restrict__ scales,
    const int*           __restrict__ qzeros,
    const int N,
    const int K,
    const int group_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * K;
    if (idx >= total) return;

    const int n = idx / K;
    const int k = idx % K;
    const int packed_per_row = K / 8;
    const int num_groups = K / group_size;

    // Extract 4-bit weight
    int packed_idx = k / 8;
    int nibble_idx = k % 8;
    int packed_val = qweight[n * packed_per_row + packed_idx];
    int q = (packed_val >> (nibble_idx * 4)) & 0xF;

    // Scale and zero-point
    int group_idx = k / group_size;
    __half scale = *reinterpret_cast<const __half*>(&scales[n * num_groups + group_idx]);

    int qz_packed = qzeros[(n / 8) * num_groups + group_idx];
    int qz_shift = (n % 8) * 4;
    int qzero = ((qz_packed >> qz_shift) & 0xF) + 1;

    float w = (float)(q - qzero) * __half2float(scale);
    output[idx] = __float2half(w);
}
