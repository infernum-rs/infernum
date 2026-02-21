// f16 → f32 using PTX cvt instruction (single HW instruction on sm_70+,
// avoids cuda_fp16.h dependency while being more efficient than bit tricks).
__device__ float f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
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

// -----------------------------------------------------------------------
// Activation quantization: f32 → Q8_1 (int8 data + f32 scale + f32 sum)
// -----------------------------------------------------------------------
// One CUDA block per 32-element quant block. 32 threads cooperate to:
//   1. Find absmax via warp reduction
//   2. Quantize each element to int8
//   3. Compute sum(qs) via warp reduction
//   4. Write data, scale (= absmax / 127), and weighted sum (= d * sum(qs))
//
// Input:  f32 activations [K]       (K must be divisible by 32)
// Output: int8 data [K], f32 scales [K/32], f32 sums [K/32]

extern "C" __global__ void quantize_f32_to_q8_1(
    signed char*        __restrict__ out_data,
    float*              __restrict__ out_scales,
    float*              __restrict__ out_sums,
    const float*        __restrict__ input,
    const int K
) {
    const int block_idx = blockIdx.x;
    const int tid = threadIdx.x;  // 0..31
    const int base_k = block_idx * 32;

    if (base_k + tid >= K) return;

    // Load one element per thread
    float val = input[base_k + tid];
    float abs_val = fabsf(val);

    // Warp reduction: find max absolute value
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, abs_val, offset);
        abs_val = fmaxf(abs_val, other);
    }
    // abs_val now contains the block maximum in all lanes

    float d = abs_val / 127.0f;
    float inv_d = (d > 0.0f) ? (1.0f / d) : 0.0f;

    // Quantize
    int qi = __float2int_rn(val * inv_d);
    qi = max(-128, min(127, qi));

    // Write quantized value
    out_data[base_k + tid] = (signed char)qi;

    // Warp reduction: sum of quantized values
    int qi_sum = qi;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        qi_sum += __shfl_xor_sync(0xFFFFFFFF, qi_sum, offset);
    }

    // Lane 0 writes scale and weighted sum
    if (tid == 0) {
        out_scales[block_idx] = d;
        out_sums[block_idx] = d * (float)qi_sum;
    }
}

// -----------------------------------------------------------------------
// dp4a GEMV kernels: multi-warp K-splitting with dp4a
// -----------------------------------------------------------------------
// NWARPS warps (blockDim.y) per output row. Each warp's 32 lanes split K,
// then intra-warp shuffle + inter-warp shared-memory reduction.
//
// Launch config: grid = (N, 1, 1), block = (32, NWARPS, 1),
//                shared_mem = NWARPS * sizeof(float) bytes.

// Q8_0 × Q8_1 dp4a GEMV (multi-warp, vectorized loads)
extern "C" __global__ void gemv_q8_q8_dp4a(
    float*              __restrict__ output,
    const signed char*  __restrict__ act_data,     // [K] int8 quantized activations
    const float*        __restrict__ act_scales,   // [K/32] per-block scales
    const signed char*  __restrict__ weight_data,  // [N, K] int8 weights
    const unsigned short* __restrict__ weight_scales, // [N, K/32] f16 scales
    const int N,
    const int K
) {
    const int n = blockIdx.x;
    if (n >= N) return;

    const int lane = threadIdx.x;          // lane within warp [0..31]
    const int warp_id = threadIdx.y;       // warp index [0..NWARPS-1]
    const int nwarps = blockDim.y;
    const int tid = warp_id * 32 + lane;   // flat thread id within block
    const int nthreads = nwarps * 32;
    const int blocks_per_row = K / 32;
    float acc = 0.0f;

    for (int b = tid; b < blocks_per_row; b += nthreads) {
        float d_w = f16_to_f32(weight_scales[n * blocks_per_row + b]);
        float d_a = act_scales[b];

        // 32 bytes per quant block = 2 × int4 (128-bit) loads
        const int4* w_v = (const int4*)(weight_data + n * K + b * 32);
        const int4* a_v = (const int4*)(act_data + b * 32);

        int4 w0 = w_v[0], w1 = w_v[1];
        int4 a0 = a_v[0], a1 = a_v[1];

        int sumi = 0;
        sumi = __dp4a(w0.x, a0.x, sumi);
        sumi = __dp4a(w0.y, a0.y, sumi);
        sumi = __dp4a(w0.z, a0.z, sumi);
        sumi = __dp4a(w0.w, a0.w, sumi);
        sumi = __dp4a(w1.x, a1.x, sumi);
        sumi = __dp4a(w1.y, a1.y, sumi);
        sumi = __dp4a(w1.z, a1.z, sumi);
        sumi = __dp4a(w1.w, a1.w, sumi);

        acc += d_w * d_a * (float)sumi;
    }

    // Intra-warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    }

    // Inter-warp reduction via shared memory
    if (nwarps > 1) {
        extern __shared__ float smem[];  // [nwarps] floats
        if (lane == 0) {
            smem[warp_id] = acc;
        }
        __syncthreads();

        // Warp 0 reduces across all warps
        if (warp_id == 0 && lane == 0) {
            float sum = smem[0];
            for (int w = 1; w < nwarps; w++) {
                sum += smem[w];
            }
            output[n] = sum;
        }
    } else {
        if (lane == 0) {
            output[n] = acc;
        }
    }
}

// Q4_0 × Q8_1 dp4a GEMV (multi-warp, with zero-point correction)
extern "C" __global__ void gemv_q4_q8_dp4a(
    float*              __restrict__ output,
    const signed char*  __restrict__ act_data,     // [K] int8 quantized activations
    const float*        __restrict__ act_scales,   // [K/32] per-block scales
    const float*        __restrict__ act_sums,     // [K/32] per-block weighted sums
    const unsigned char* __restrict__ weight_data,  // [N, K/2] packed nibbles
    const unsigned short* __restrict__ weight_scales, // [N, K/32] f16 scales
    const int N,
    const int K
) {
    const int n = blockIdx.x;
    if (n >= N) return;

    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int nwarps = blockDim.y;
    const int tid = warp_id * 32 + lane;
    const int nthreads = nwarps * 32;
    const int blocks_per_row = K / 32;
    float acc = 0.0f;

    for (int b = tid; b < blocks_per_row; b += nthreads) {
        float d_w = f16_to_f32(weight_scales[n * blocks_per_row + b]);
        float d_a = act_scales[b];
        float s_a = act_sums[b];

        // 16 bytes weight data = 1 × int4 load
        const int4* w_v = (const int4*)(weight_data + n * (K / 2) + b * 16);
        int4 w = w_v[0];

        // 32 bytes activation = 2 × int4 loads (lo half, hi half)
        const int4* a_v = (const int4*)(act_data + b * 32);
        int4 a_lo = a_v[0], a_hi = a_v[1];

        int sumi = 0;
        // Nibble extraction + dp4a for 4 packed uint32
        unsigned int uw[4] = {(unsigned int)w.x, (unsigned int)w.y,
                              (unsigned int)w.z, (unsigned int)w.w};
        int al[4] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w};
        int ah[4] = {a_hi.x, a_hi.y, a_hi.z, a_hi.w};

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int lo = (int)(uw[i] & 0x0F0F0F0Fu);
            int hi = (int)((uw[i] >> 4) & 0x0F0F0F0Fu);
            sumi = __dp4a(lo, al[i], sumi);
            sumi = __dp4a(hi, ah[i], sumi);
        }

        acc += d_w * (d_a * (float)sumi - 8.0f * s_a);
    }

    // Intra-warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    }

    // Inter-warp reduction via shared memory
    if (nwarps > 1) {
        extern __shared__ float smem[];
        if (lane == 0) {
            smem[warp_id] = acc;
        }
        __syncthreads();

        if (warp_id == 0 && lane == 0) {
            float sum = smem[0];
            for (int w = 1; w < nwarps; w++) {
                sum += smem[w];
            }
            output[n] = sum;
        }
    } else {
        if (lane == 0) {
            output[n] = acc;
        }
    }
}
