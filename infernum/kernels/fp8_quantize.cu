// Quantize f32 activations to FP8 E4M3, entirely on device.
//
// Provides both separate kernels (absmax + quantize) and a fused single-kernel
// version that uses atomic grid synchronization.

// ─── Warp-level max reduction ────────────────────────────────────────────────

static __device__ __forceinline__ float warp_reduce_max_val(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

static __device__ __forceinline__ float block_reduce_max(float val) {
    extern __shared__ char smem[];
    float* shared = reinterpret_cast<float*>(smem);

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = blockDim.x / 32;

    val = warp_reduce_max_val(val);

    if (num_warps <= 1) {
        return val;
    }

    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        val = warp_reduce_max_val(val);
        if (lane_id == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();

    return shared[0];
}

// ─── FP8 E4M3 encoding (device function) ────────────────────────────────────

static __device__ __forceinline__ unsigned char fp8_encode(float v) {
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

    return (sign << 7) | result;
}

// ─── Standalone absmax kernel (kept for compatibility) ──────────────────────

extern "C" __global__ void absmax_f32(
    const float* __restrict__ input,
    unsigned int* __restrict__ max_bits,
    const int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_max = 0.0f;
    // Vectorized loads
    const int vec_count = numel / 4;
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    for (int i = idx; i < vec_count; i += blockDim.x * gridDim.x) {
        float4 v = input_vec[i];
        local_max = fmaxf(local_max, fabsf(v.x));
        local_max = fmaxf(local_max, fabsf(v.y));
        local_max = fmaxf(local_max, fabsf(v.z));
        local_max = fmaxf(local_max, fabsf(v.w));
    }
    // Remainder
    for (int i = vec_count * 4 + idx; i < numel; i += blockDim.x * gridDim.x) {
        local_max = fmaxf(local_max, fabsf(input[i]));
    }

    local_max = block_reduce_max(local_max);

    if (threadIdx.x == 0) {
        atomicMax(max_bits, __float_as_uint(local_max));
    }
}

// ─── Standalone quantize kernel (kept for compatibility) ────────────────────

extern "C" __global__ void quantize_f32_to_fp8(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    const unsigned int* __restrict__ max_bits,
    float* __restrict__ inv_scale_out,
    const int numel
) {
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

    output[idx] = fp8_encode(input[idx] * scale);
}

// ─── Fused absmax + quantize (single kernel) ────────────────────────────────
//
// Uses a two-phase approach within one kernel launch:
// Phase 1: Each block computes its local absmax, atomicMax to global.
//          The last block to finish (tracked by atomic counter) knows
//          the global absmax is ready.
// Phase 2: All blocks read the global absmax and quantize their elements.
//
// Grid sync via atomic counter — no cooperative groups needed.

extern "C" __global__ void fused_absmax_quantize_f32(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    unsigned int* __restrict__ max_bits,      // pre-zeroed by host
    float* __restrict__ inv_scale_out,
    unsigned int* __restrict__ block_counter, // pre-zeroed by host
    const int numel
) {
    // ── Phase 1: absmax reduction ──

    float local_max = 0.0f;
    const int grid_stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Vectorized loads
    const int vec_count = numel / 4;
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    int vec_idx = idx;
    for (int i = vec_idx; i < vec_count; i += grid_stride) {
        float4 v = input_vec[i];
        local_max = fmaxf(local_max, fabsf(v.x));
        local_max = fmaxf(local_max, fabsf(v.y));
        local_max = fmaxf(local_max, fabsf(v.z));
        local_max = fmaxf(local_max, fabsf(v.w));
    }
    for (int i = vec_count * 4 + idx; i < numel; i += grid_stride) {
        local_max = fmaxf(local_max, fabsf(input[i]));
    }

    local_max = block_reduce_max(local_max);

    __shared__ bool is_last_block;

    if (threadIdx.x == 0) {
        atomicMax(max_bits, __float_as_uint(local_max));

        // Memory fence so all atomicMax writes are visible
        __threadfence();

        // Announce this block is done; check if we're the last one
        unsigned int finished = atomicAdd(block_counter, 1);
        is_last_block = (finished == gridDim.x - 1);
    }
    __syncthreads();

    // ── Phase 2: quantize ──
    // Only the last block computes the scale; all blocks must wait
    // for it, but we can't do a true grid barrier. Instead, each block
    // already has its local_max in the atomicMax. We just need
    // the last block to be the one that writes inv_scale. All blocks
    // then re-read max_bits (which is finalized after counter == gridDim.x - 1).
    //
    // Trick: all blocks that aren't last must spin-wait for the counter
    // to reach gridDim.x. This is valid because the last block will
    // always reach the counter increment.

    if (is_last_block) {
        // We are guaranteed all blocks have done atomicMax + threadfence
        if (threadIdx.x == 0) {
            float absmax = __uint_as_float(*max_bits);
            float inv = (absmax > 0.0f) ? (absmax / 448.0f) : 1.0f;
            *inv_scale_out = inv;
        }
    }

    // Non-last blocks spin until counter reaches gridDim.x
    if (!is_last_block) {
        if (threadIdx.x == 0) {
            while (atomicAdd(block_counter, 0) < gridDim.x) {
                // spin
            }
        }
        __syncthreads();
    }

    // Compute scale from finalized absmax
    __shared__ float s_scale;
    if (threadIdx.x == 0) {
        float absmax = __uint_as_float(*max_bits);
        s_scale = (absmax > 0.0f) ? (448.0f / absmax) : 1.0f;
    }
    __syncthreads();

    float scale = s_scale;

    // Quantize — each thread handles its strided elements
    for (int i = idx; i < numel; i += grid_stride) {
        output[i] = fp8_encode(input[i] * scale);
    }
}
