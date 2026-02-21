// GPTQ INT4 dequantizing matmul kernel
//
// Computes: output[m][n] = sum_k( input[m][k] * dequant(weight[k][n]) )
//
// GPTQ transposed layout (repacked at load time for coalesced GPU access):
//   qweight:  [N, K/8] as int32  (row n holds all packed weights for output n)
//             Each int32 packs 8 INT4 values (bits 0-3 = element 0, ...)
//   scales:   [N, num_groups] as f16
//   qzeros:   [N/8, num_groups] as int32  (packed INT4 zero-points)
//
// Dequantization: w = (qweight_val - (stored_qzero + 1)) * scale
// AutoGPTQ stores qzeros with a -1 offset: stored = actual_zero_point - 1

// f16 → f32 using PTX cvt instruction (single HW instruction on sm_70+,
// avoids cuda_fp16.h dependency while being more efficient than bit tricks).
__device__ float f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// Each thread computes one (m, n) output element.
// M = batch size (number of input rows)
// N = out_features
// K = in_features
extern "C" __global__ void matmul_gptq_f32(
    float*       __restrict__ output,
    const float* __restrict__ input,
    const int*   __restrict__ qweight,      // [N, K/8]
    const unsigned short* __restrict__ scales,  // [N, num_groups] as f16
    const int*   __restrict__ qzeros,       // [N/8, num_groups]
    const int M,
    const int N,
    const int K,
    const int group_size
) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    const int packed_per_row = K / 8;
    const int num_groups = K / group_size;

    float acc = 0.0f;
    const float* in_ptr = input + m * K;

    for (int pr = 0; pr < packed_per_row; ++pr) {
        int packed_val = qweight[n * packed_per_row + pr];

        int base_k = pr * 8;
        int group_idx = base_k / group_size;

        float scale = f16_to_f32(scales[n * num_groups + group_idx]);

        int qz_packed = qzeros[(n / 8) * num_groups + group_idx];
        int qz_shift = (n % 8) * 4;
        int qzero = ((qz_packed >> qz_shift) & 0xF) + 1;

        for (int j = 0; j < 8; ++j) {
            int k = base_k + j;
            if (k >= K) break;

            int q = (packed_val >> (j * 4)) & 0xF;
            float w = (float)(q - qzero) * scale;
            acc += in_ptr[k] * w;
        }
    }

    output[m * N + n] = acc;
}

// ---------------------------------------------------------------------------
// dp4a GEMV kernel for GPTQ INT4 × Q8_1 (M=1 decode fast path)
// ---------------------------------------------------------------------------
// Warp-parallel GEMV using dp4a (INT8 dot product with INT32 accumulate).
// Weights are transposed at load time to [N, K/8] for coalesced access.
//
// Launch: grid=(N,1,1), block=(32, NWARPS, 1), shared=NWARPS*sizeof(float)
//
// Each block computes one output element n. Warps split K across threads.
// Activations are pre-quantized to Q8_1 format (int8 + per-block scale + sum).
extern "C" __global__ void gemv_gptq_q8_dp4a(
    float*              __restrict__ output,
    const signed char*  __restrict__ act_data,     // [K] int8 quantized activations
    const float*        __restrict__ act_scales,   // [K/32] per-block scales
    const float*        __restrict__ act_sums,     // [K/32] per-block weighted sums
    const int*          __restrict__ qweight,      // [N, K/8] row-major packed int4
    const unsigned short* __restrict__ scales,     // [N, num_groups] f16 weight scales
    const int*          __restrict__ qzeros,       // [N/8, num_groups] packed int4 zeros
    const int N,
    const int K,
    const int group_size
) {
    const int n = blockIdx.x;
    if (n >= N) return;

    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int nwarps = blockDim.y;
    const int tid = warp_id * 32 + lane;
    const int nthreads = nwarps * 32;
    const int blocks_per_row = K / 32;
    const int packed_per_row = K / 8;
    const int num_groups = K / group_size;
    const int n_div_8 = n / 8;
    const int n_mod_8_x4 = (n % 8) * 4;

    float acc = 0.0f;

    for (int b = tid; b < blocks_per_row; b += nthreads) {
        int group_idx = (b * 32) / group_size;

        float d_w = f16_to_f32(scales[n * num_groups + group_idx]);
        float d_a = act_scales[b];
        float s_a = act_sums[b];

        // Zero-point: stored with -1 offset, packed INT4
        int qz_packed = qzeros[n_div_8 * num_groups + group_idx];
        int qzero = ((qz_packed >> n_mod_8_x4) & 0xF) + 1;

        // Load 32 activation bytes as 8 × int32 (2 × int4 vector loads)
        const int4* a_v = (const int4*)(act_data + b * 32);
        int4 a_lo = a_v[0], a_hi = a_v[1];
        int a_int[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};

        int sumi = 0;

        // Vector-load 4 contiguous packed int32s (32 weight elements) from row n.
        const int4* w_v = (const int4*)(qweight + n * packed_per_row + b * 4);
        int4 w = w_v[0];
        int packed[4] = {w.x, w.y, w.z, w.w};

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            // Extract 8 nibbles and repack into dp4a-compatible byte order.
            // GPTQ packing: bits [4j+3:4j] = element j (j=0..7)
            unsigned int q0 = (packed[i] >>  0) & 0xF;
            unsigned int q1 = (packed[i] >>  4) & 0xF;
            unsigned int q2 = (packed[i] >>  8) & 0xF;
            unsigned int q3 = (packed[i] >> 12) & 0xF;
            unsigned int q4 = (packed[i] >> 16) & 0xF;
            unsigned int q5 = (packed[i] >> 20) & 0xF;
            unsigned int q6 = (packed[i] >> 24) & 0xF;
            unsigned int q7 = (packed[i] >> 28) & 0xF;

            int pack_a = (int)(q0 | (q1 << 8) | (q2 << 16) | (q3 << 24));
            int pack_b = (int)(q4 | (q5 << 8) | (q6 << 16) | (q7 << 24));

            sumi = __dp4a(pack_a, a_int[i * 2],     sumi);
            sumi = __dp4a(pack_b, a_int[i * 2 + 1], sumi);
        }

        // Zero-point correction: sum(q * a) - z * sum(a)
        acc += d_w * (d_a * (float)sumi - (float)qzero * s_a);
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
