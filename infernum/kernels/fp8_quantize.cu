// Quantize f32 activations to FP8 E4M3, entirely on device.
//
// Two-kernel approach (no CPU readback):
//   1. absmax_f32: parallel reduction to find max|x| across the tensor
//   2. quantize_f32_to_fp8: reads absmax from device memory, computes scale,
//      writes inv_scale to a device pointer, and quantizes all elements

extern "C" __global__ void absmax_f32(
    const float* __restrict__ input,
    unsigned int* __restrict__ max_bits,
    const int numel
) {
    extern __shared__ unsigned int smax[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_max = 0.0f;
    for (int i = idx; i < numel; i += blockDim.x * gridDim.x) {
        float v = fabsf(input[i]);
        if (v > local_max) local_max = v;
    }

    smax[tid] = __float_as_uint(local_max);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = __uint_as_float(smax[tid]);
            float b = __uint_as_float(smax[tid + s]);
            smax[tid] = __float_as_uint(fmaxf(a, b));
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(max_bits, smax[0]);
    }
}

// Reads absmax from device memory, computes scale on-device, writes inv_scale,
// and quantizes input to FP8 E4M3 — all without any CPU<->GPU synchronization.
extern "C" __global__ void quantize_f32_to_fp8(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    const unsigned int* __restrict__ max_bits,
    float* __restrict__ inv_scale_out,
    const int numel
) {
    // First thread computes and broadcasts the scale via shared memory
    __shared__ float s_scale;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float absmax = __uint_as_float(*max_bits);
        float inv = (absmax > 0.0f) ? (absmax / 448.0f) : 1.0f;
        *inv_scale_out = inv;
    }
    if (threadIdx.x == 0) {
        float absmax = __uint_as_float(*max_bits);
        s_scale = (absmax > 0.0f) ? (448.0f / absmax) : 1.0f;
    }
    __syncthreads();

    float scale = s_scale;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float v = input[idx] * scale;
    v = fminf(fmaxf(v, -448.0f), 448.0f);

    unsigned char sign = (v < 0.0f) ? 1 : 0;
    float abs_v = fabsf(v);

    unsigned char result;
    if (abs_v < 0.001953125f) {
        result = 0;
    } else if (abs_v < 0.015625f) {
        int mant = (int)roundf(abs_v * 512.0f);
        if (mant > 7) mant = 7;
        if (mant < 1) mant = 1;
        result = (unsigned char)mant;
    } else {
        int raw_exp;
        float frac = frexpf(abs_v, &raw_exp);
        int biased_exp = raw_exp + 6;
        if (biased_exp < 1) biased_exp = 1;
        if (biased_exp > 15) biased_exp = 15;

        int mant = (int)roundf((2.0f * frac - 1.0f) * 8.0f);
        if (mant > 7) {
            mant = 0;
            biased_exp++;
            if (biased_exp > 15) { biased_exp = 15; mant = 6; }
        }

        if (biased_exp == 15 && mant == 7) mant = 6;

        result = ((unsigned char)biased_exp << 3) | (unsigned char)mant;
    }

    output[idx] = (sign << 7) | result;
}
