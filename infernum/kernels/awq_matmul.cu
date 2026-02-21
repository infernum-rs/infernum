// AWQ INT4 dequantizing matmul kernel
//
// Computes: output[m][n] = sum_k( input[m][k] * dequant(weight[k][n]) )
//
// AWQ layout:
//   qweight:  [in_features, out_features/8] as int32
//             Each int32 packs 8 INT4 output channels:
//             bits 0-3 = out channel (col*8+0), bits 4-7 = out channel (col*8+1), ...
//   scales:   [num_groups, out_features] as f16
//             One scale per group of `group_size` input elements per output
//   qzeros:   [num_groups, out_features/8] as int32
//             Packed INT4 zero-points, same packing as qweight columns
//
// Dequantization: w = (qweight_val - qzero_val) * scale

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
extern "C" __global__ void matmul_awq_f32(
    float*       __restrict__ output,
    const float* __restrict__ input,
    const int*   __restrict__ qweight,      // [K, N/8]
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

    const int packed_cols = N / 8;   // number of int32 columns in qweight
    const int n_col = n / 8;         // which packed column this output belongs to
    const int n_shift = (n % 8) * 4; // bit offset within the int32

    float acc = 0.0f;
    const float* in_ptr = input + m * K;

    for (int k = 0; k < K; ++k) {
        int group_idx = k / group_size;

        // Get scale for this group and output channel
        float scale = f16_to_f32(scales[group_idx * N + n]);

        // Get zero-point for this group and output channel
        int qz_packed = qzeros[group_idx * packed_cols + n_col];
        int qzero = (qz_packed >> n_shift) & 0xF;

        // Get quantized weight for this (k, n)
        int qw_packed = qweight[k * packed_cols + n_col];
        int q = (qw_packed >> n_shift) & 0xF;

        float w = (float)(q - qzero) * scale;
        acc += in_ptr[k] * w;
    }

    output[m * N + n] = acc;
}
