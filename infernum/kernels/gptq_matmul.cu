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
