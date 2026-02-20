// Branchless f16 → f32 decode (avoids cuda_fp16.h dependency in NVRTC).
// Uses bit manipulation instead of ldexpf; subnormals are flushed to zero
// which is safe for quantization scale factors (always normal floats).
__device__ float f16_to_f32(unsigned short bits) {
    unsigned int sign = (bits & 0x8000u) << 16;
    unsigned int exp  = (bits >> 10) & 0x1Fu;
    unsigned int mant = bits & 0x3FFu;
    // Zero/subnormal (exp==0) or inf/NaN (exp==31): treated as zero for scales
    unsigned int is_normal = (exp != 0u && exp != 31u) ? 0xFFFFFFFFu : 0u;
    unsigned int f32_bits = sign | ((exp + 112u) << 23) | (mant << 13);
    f32_bits &= is_normal;
    float result;
    memcpy(&result, &f32_bits, 4);
    return result;
}

// Q8_0 matmul: output[m][n] = sum_k( input[m][k] * dequant(weight[n][k]) )
// weight layout (row-major per output row n): blocks of 32 int8 values + 1 f16 scale
//
// data   pointer: int8  values, shape (N, K) stored contiguously
// scales pointer: f16   values, shape (N, K/32)
//
// Each thread computes one (m, n) output element.
extern "C" __global__ void matmul_q8_f32(
    float*       __restrict__ output,
    const float* __restrict__ input,
    const signed char* __restrict__ weight_data,
    const unsigned short* __restrict__ weight_scales,  // f16 stored as u16
    const int M,
    const int N,
    const int K
) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    const int blocks_per_row = K / 32;
    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        float scale = f16_to_f32(weight_scales[n * blocks_per_row + b]);

        int base_k = b * 32;
        int weight_base = n * K + base_k;
        const float* in_ptr = input + m * K + base_k;

        for (int j = 0; j < 32; ++j) {
            float w = (float)weight_data[weight_base + j] * scale;
            acc += in_ptr[j] * w;
        }
    }

    output[m * N + n] = acc;
}

// Q4_0 matmul: GGML Q4_0 non-consecutive packing
// byte[j] has element j in low nibble, element j+16 in high nibble
// data   pointer: uint8 packed values, shape (N, K/2)
// scales pointer: f16 values, shape (N, K/32)
extern "C" __global__ void matmul_q4_f32(
    float*       __restrict__ output,
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight_data,
    const unsigned short* __restrict__ weight_scales,
    const int M,
    const int N,
    const int K
) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    const int blocks_per_row = K / 32;
    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        float scale = f16_to_f32(weight_scales[n * blocks_per_row + b]);

        int base_k = b * 32;
        // Each block has 16 bytes containing 32 values in non-consecutive layout:
        // byte[j] has element j (low nibble) and element j+16 (high nibble)
        int packed_base = n * (K / 2) + base_k / 2;
        const float* in_ptr = input + m * K + base_k;

        // Process first half (elements 0-15): low nibbles
        for (int j = 0; j < 16; ++j) {
            unsigned char packed = weight_data[packed_base + j];
            float w_lo = (float)((int)(packed & 0x0F) - 8) * scale;
            acc += in_ptr[j] * w_lo;
        }
        // Process second half (elements 16-31): high nibbles
        for (int j = 0; j < 16; ++j) {
            unsigned char packed = weight_data[packed_base + j];
            float w_hi = (float)((int)(packed >> 4) - 8) * scale;
            acc += in_ptr[16 + j] * w_hi;
        }
    }

    output[m * N + n] = acc;
}

// FP8 E4M3 matmul: each weight byte is an fp8 value (no block structure)
// Manual decode: sign(1) | exponent(4) | mantissa(3), bias=7
// weight_scale: per-tensor scale factor (from dynamic quantization)
extern "C" __global__ void matmul_fp8_f32(
    float*       __restrict__ output,
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight_data,
    const float weight_scale,
    const int M,
    const int N,
    const int K
) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    float acc = 0.0f;
    const float* in_ptr = input + m * K;
    const unsigned char* w_ptr = weight_data + n * K;

    for (int k = 0; k < K; ++k) {
        unsigned char bits = w_ptr[k];

        // Decode E4M3: sign(1) | exp(4) | mantissa(3), bias=7
        int sign = (bits >> 7) & 1;
        int exp  = (bits >> 3) & 0x0F;
        int mant = bits & 0x07;

        float w;
        if (exp == 0 && mant == 0) {
            w = 0.0f;
        } else if (exp == 0) {
            // Subnormal: 2^(1-7) * (mant/8)
            w = ldexpf((float)mant / 8.0f, -6);
        } else if (exp == 15 && mant == 7) {
            // NaN in E4M3 (no infinity — E4M3 uses all-ones for NaN)
            w = 0.0f;
        } else {
            // Normal: 2^(exp-7) * (1 + mant/8)
            w = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
        }

        if (sign) w = -w;
        acc += in_ptr[k] * w;
    }

    output[m * N + n] = acc * weight_scale;
}

// Q6_K matmul: super-block of 256 elements, 210 bytes each
// Layout per super-block: ql[128] | qh[64] | scales[16] | d(f16)
//   ql: lower 4 bits of each 6-bit value, 2 per byte (low/high nibble)
//   qh: upper 2 bits of each 6-bit value, 4 per byte (2-bit fields)
//   scales: i8 sub-block scale, one per 16 elements
//   d: f16 super-block scale factor
//
// data pointer: packed super-blocks, contiguous for all N rows
// Each row of N has (K / 256) super-blocks × 210 bytes
extern "C" __global__ void matmul_q6k_f32(
    float*       __restrict__ output,
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight_data,
    const int M,
    const int N,
    const int K
) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    const int blocks_per_row = K / 256;
    const int block_bytes = 210;
    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int sb_offset = (n * blocks_per_row + b) * block_bytes;
        const unsigned char* ql     = weight_data + sb_offset;
        const unsigned char* qh     = weight_data + sb_offset + 128;
        const signed char*   scales = (const signed char*)(weight_data + sb_offset + 128 + 64);
        float d = f16_to_f32(*(const unsigned short*)(weight_data + sb_offset + 128 + 64 + 16));

        int base_k = b * 256;
        const float* in_ptr = input + m * K + base_k;

        for (int elem = 0; elem < 256; ++elem) {
            // Map element index using ggml's interleaved layout (8x32 -> 16x16 reshape)
            // The data flows: (128,) -> (2,1,64) -> shift -> (2,2,64) -> (8,32) -> (16,16)
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

            // qh layout: (64,) -> (2,1,32) -> shift -> (2,4,32) -> (8,32)
            // row 0: bytes 0-31 bits 0-1   row 4: bytes 32-63 bits 0-1
            // row 1: bytes 0-31 bits 2-3   row 5: bytes 32-63 bits 2-3
            // row 2: bytes 0-31 bits 4-5   row 6: bytes 32-63 bits 4-5
            // row 3: bytes 0-31 bits 6-7   row 7: bytes 32-63 bits 6-7
            int qh_half = row8 / 4;           // 0 or 1 (selects 32-byte half)
            int qh_shift_sel = row8 % 4;      // 0,1,2,3 -> shift 0,2,4,6
            int qh_byte_idx = qh_half * 32 + col32;
            unsigned char qh_byte = qh[qh_byte_idx];
            int qh_shift = qh_shift_sel * 2;
            int qh_val = (qh_byte >> qh_shift) & 0x03;

            // Combine to 6-bit [0,63], center to signed [-32,31]
            int q = (ql_val | (qh_val << 4)) - 32;

            float sc = (float)scales[sb];
            float w = d * sc * (float)q;
            acc += in_ptr[elem] * w;
        }
    }

    output[m * N + n] = acc;
}

// ---------------------------------------------------------------------------
// GEMV kernels: specialized for M=1 decode
// ---------------------------------------------------------------------------
//
// One thread per output element (same principle as the GEMM kernel with M=1,
// but using a 1D block layout to avoid wasting 15/16 of threads).
//
// Block: (GEMV_BLOCK, 1) threads     Grid: (ceil(N / GEMV_BLOCK), 1)
// Each thread loops over all K/32 quant blocks sequentially, maximizing
// instruction-level parallelism and sequential memory access for weights.
// Input vector is small (fits in L2 cache) and is broadcast to all threads.

#define GEMV_BLOCK 16

// Q8_0 GEMV: output[n] = dot(input[K], dequant(weight[n, K]))
extern "C" __global__ void gemv_q8_f32(
    float*              __restrict__ output,
    const float*        __restrict__ input,
    const signed char*  __restrict__ weight_data,
    const unsigned short* __restrict__ weight_scales,
    const int N,
    const int K
) {
    const int n = blockIdx.x * GEMV_BLOCK + threadIdx.x;
    if (n >= N) return;

    const int blocks_per_row = K / 32;
    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; b++) {
        float scale = f16_to_f32(weight_scales[n * blocks_per_row + b]);
        int base_k = b * 32;
        const signed char* w_ptr = weight_data + n * K + base_k;
        const float* in_ptr = input + base_k;

        float block_acc = 0.0f;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int w4;
            memcpy(&w4, w_ptr + i * 4, 4);
            const signed char* w4b = (const signed char*)&w4;

            float4 in4;
            memcpy(&in4, in_ptr + i * 4, 16);

            block_acc += w4b[0] * in4.x + w4b[1] * in4.y +
                         w4b[2] * in4.z + w4b[3] * in4.w;
        }

        acc += scale * block_acc;
    }

    output[n] = acc;
}

// Q4_0 GEMV: output[n] = dot(input[K], dequant(weight[n, K]))
// GGML Q4_0 packing: byte[j] has element j (low nibble), element j+16 (high nibble)
extern "C" __global__ void gemv_q4_f32(
    float*              __restrict__ output,
    const float*        __restrict__ input,
    const unsigned char* __restrict__ weight_data,
    const unsigned short* __restrict__ weight_scales,
    const int N,
    const int K
) {
    const int n = blockIdx.x * GEMV_BLOCK + threadIdx.x;
    if (n >= N) return;

    const int blocks_per_row = K / 32;
    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; b++) {
        float scale = f16_to_f32(weight_scales[n * blocks_per_row + b]);
        int base_k = b * 32;
        const unsigned char* w_ptr = weight_data + n * (K / 2) + b * 16;
        const float* in_ptr = input + base_k;

        float block_acc = 0.0f;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned int w4;
            memcpy(&w4, w_ptr + i * 4, 4);
            const unsigned char* w4b = (const unsigned char*)&w4;

            float4 in_lo;
            memcpy(&in_lo, in_ptr + i * 4, 16);

            float4 in_hi;
            memcpy(&in_hi, in_ptr + 16 + i * 4, 16);

            block_acc += ((int)(w4b[0] & 0x0F) - 8) * in_lo.x;
            block_acc += ((int)(w4b[1] & 0x0F) - 8) * in_lo.y;
            block_acc += ((int)(w4b[2] & 0x0F) - 8) * in_lo.z;
            block_acc += ((int)(w4b[3] & 0x0F) - 8) * in_lo.w;

            block_acc += ((int)(w4b[0] >> 4) - 8) * in_hi.x;
            block_acc += ((int)(w4b[1] >> 4) - 8) * in_hi.y;
            block_acc += ((int)(w4b[2] >> 4) - 8) * in_hi.z;
            block_acc += ((int)(w4b[3] >> 4) - 8) * in_hi.w;
        }

        acc += scale * block_acc;
    }

    output[n] = acc;
}
