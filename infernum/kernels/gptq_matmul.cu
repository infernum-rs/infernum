// GPTQ INT4 dequantizing matmul kernel
//
// Computes: output[m][n] = sum_k( input[m][k] * dequant(weight[k][n]) )
//
// GPTQ layout:
//   qweight:  [in_features/8, out_features] as int32
//             Each int32 packs 8 INT4 values (bits 0-3 = element 0, bits 4-7 = element 1, ...)
//   scales:   [num_groups, out_features] as f16
//             One scale per group of `group_size` input elements per output
//   qzeros:   [num_groups, out_features/8] as int32
//             Packed INT4 zero-points, same packing as qweight
//
// Dequantization: w = (qweight_val - (stored_qzero + 1)) * scale
// AutoGPTQ stores qzeros with a -1 offset: stored = actual_zero_point - 1

// Manual f16 → f32 decode (avoids cuda_fp16.h dependency in NVRTC)
__device__ float f16_to_f32(unsigned short bits) {
    unsigned int sign = (bits >> 15) & 0x1;
    unsigned int exp  = (bits >> 10) & 0x1F;
    unsigned int mant = bits & 0x3FF;

    float result;
    if (exp == 0) {
        result = ldexpf((float)mant / 1024.0f, -14);
    } else if (exp == 31) {
        result = 0.0f;
    } else {
        result = ldexpf(1.0f + (float)mant / 1024.0f, (int)exp - 15);
    }
    return sign ? -result : result;
}

// Each thread computes one (m, n) output element.
// M = batch size (number of input rows)
// N = out_features
// K = in_features
extern "C" __global__ void matmul_gptq_f32(
    float*       __restrict__ output,
    const float* __restrict__ input,
    const int*   __restrict__ qweight,      // [K/8, N]
    const unsigned short* __restrict__ scales,  // [K/group_size, N] as f16
    const int*   __restrict__ qzeros,       // [K/group_size, N/8]
    const int M,
    const int N,
    const int K,
    const int group_size
) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    const int packed_rows = K / 8;       // number of int32 rows in qweight
    const int num_groups = K / group_size;

    float acc = 0.0f;
    const float* in_ptr = input + m * K;

    // Iterate over packed rows (each int32 covers 8 input elements)
    for (int pr = 0; pr < packed_rows; ++pr) {
        int packed_val = qweight[pr * N + n];

        int base_k = pr * 8;

        // Determine which group these 8 elements belong to
        int group_idx = base_k / group_size;

        // Get scale for this group
        float scale = f16_to_f32(scales[group_idx * N + n]);

        // Get zero-point for this group
        // AutoGPTQ stores qzeros with a -1 offset: stored = actual - 1
        int qz_packed = qzeros[group_idx * (N / 8) + n / 8];
        int qz_shift = (n % 8) * 4;
        int qzero = ((qz_packed >> qz_shift) & 0xF) + 1;

        // Unpack and dequantize 8 INT4 values
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
// Same warp-parallel pattern as gemv_q4_q8_dp4a, but with GPTQ-specific
// data layout (column-major qweight) and per-group zero-point correction.
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
    const int*          __restrict__ qweight,      // [K/8, N] column-major packed int4
    const unsigned short* __restrict__ scales,     // [num_groups, N] f16 weight scales
    const int*          __restrict__ qzeros,       // [num_groups, N/8] packed int4 zeros
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
    const int n_div_8 = n / 8;
    const int n_mod_8_x4 = (n % 8) * 4;

    float acc = 0.0f;

    for (int b = tid; b < blocks_per_row; b += nthreads) {
        // Group index for zero-point and weight scale lookup
        int group_idx = (b * 32) / group_size;

        float d_w = f16_to_f32(scales[group_idx * N + n]);
        float d_a = act_scales[b];
        float s_a = act_sums[b];

        // Zero-point: stored with -1 offset, packed INT4
        int qz_packed = qzeros[group_idx * (N / 8) + n_div_8];
        int qzero = ((qz_packed >> n_mod_8_x4) & 0xF) + 1;

        // Load 32 activation bytes as 8 × int32 (2 × int4 vector loads)
        const int4* a_v = (const int4*)(act_data + b * 32);
        int4 a_lo = a_v[0], a_hi = a_v[1];
        int a_int[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};

        int sumi = 0;

        // Process 4 packed int32s (32 weight elements) from column-major qweight.
        // Each packed int32 covers 8 consecutive K elements.
        // Stride is N between consecutive packed rows.
        int qw_base = b * 4 * N + n;  // qweight[(b*4) * N + n]

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int packed = qweight[qw_base + i * N];

            // Extract 8 nibbles and repack into sequential order for dp4a.
            // GPTQ packing: byte[j] = (elem[2j+1] << 4) | elem[2j]
            // lo nibbles = even elements, hi nibbles = odd elements.
            // We need sequential order to match activation layout.
            unsigned int q0 = (packed >>  0) & 0xF;
            unsigned int q1 = (packed >>  4) & 0xF;
            unsigned int q2 = (packed >>  8) & 0xF;
            unsigned int q3 = (packed >> 12) & 0xF;
            unsigned int q4 = (packed >> 16) & 0xF;
            unsigned int q5 = (packed >> 20) & 0xF;
            unsigned int q6 = (packed >> 24) & 0xF;
            unsigned int q7 = (packed >> 28) & 0xF;

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
