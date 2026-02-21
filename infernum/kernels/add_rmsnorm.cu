// Fused residual add + RMS normalization.
//
// For each row: computes sum = residual + x, then RMS-normalizes sum.
// Writes both the sum (updated hidden state) and the normed result,
// saving one global memory round-trip compared to separate add + rmsnorm.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Warp-level sum reduction using shuffle intrinsics
static __device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level sum reduction: warp shuffle + shared memory for inter-warp
// Shared memory needs (block_size / 32) floats when block_size > 32.
// Returns the total sum to ALL threads in the block.
static __device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (blockDim.x <= 32) {
        return val;
    }

    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < (int)(blockDim.x / 32)) ? shared[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();

    return shared[0];
}

// ============================================================================
// F32 variant
// ============================================================================

extern "C" __global__ void add_rmsnorm_f32(
    float* __restrict__ sum_out,
    float* __restrict__ norm_out,
    const float* __restrict__ residual,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* row_residual = residual + row * hidden_size;
    const float* row_x = x + row * hidden_size;
    float* row_sum = sum_out + row * hidden_size;
    float* row_norm = norm_out + row * hidden_size;

    extern __shared__ float shared[];

    // Pass 1: vectorized add and accumulate sum-of-squares
    const int vec_size = hidden_size / 4;
    const float4* res_vec = reinterpret_cast<const float4*>(row_residual);
    const float4* x_vec = reinterpret_cast<const float4*>(row_x);
    float4* sum_vec = reinterpret_cast<float4*>(row_sum);

    float local_ss = 0.0f;
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 r = res_vec[i];
        float4 xv = x_vec[i];
        float4 s = make_float4(r.x + xv.x, r.y + xv.y, r.z + xv.z, r.w + xv.w);
        sum_vec[i] = s;
        local_ss += s.x * s.x + s.y * s.y + s.z * s.z + s.w * s.w;
    }
    for (int i = vec_size * 4 + tid; i < hidden_size; i += blockDim.x) {
        float s = row_residual[i] + row_x[i];
        row_sum[i] = s;
        local_ss += s * s;
    }

    float rms = block_reduce_sum(local_ss, shared);
    rms = rsqrtf(rms / (float)hidden_size + eps);

    // Pass 2: vectorized normalize + weight
    const float4* w_vec = reinterpret_cast<const float4*>(weight);
    float4* norm_vec = reinterpret_cast<float4*>(row_norm);
    const float4* sum_vec_r = reinterpret_cast<const float4*>(row_sum);

    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 s = sum_vec_r[i];
        float4 w = w_vec[i];
        norm_vec[i] = make_float4(s.x * rms * w.x, s.y * rms * w.y,
                                   s.z * rms * w.z, s.w * rms * w.w);
    }
    for (int i = vec_size * 4 + tid; i < hidden_size; i += blockDim.x) {
        row_norm[i] = row_sum[i] * rms * weight[i];
    }
}

// ============================================================================
// F16 variant
// ============================================================================

extern "C" __global__ void add_rmsnorm_f16(
    __half* __restrict__ sum_out,
    __half* __restrict__ norm_out,
    const __half* __restrict__ residual,
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __half* row_residual = residual + row * hidden_size;
    const __half* row_x = x + row * hidden_size;
    __half* row_sum = sum_out + row * hidden_size;
    __half* row_norm = norm_out + row * hidden_size;

    extern __shared__ float shared[];

    // float4 = 16 bytes = 8 halves
    const int vec_size = hidden_size / 8;
    const float4* res_vec = reinterpret_cast<const float4*>(row_residual);
    const float4* x_vec = reinterpret_cast<const float4*>(row_x);
    float4* sum_vec = reinterpret_cast<float4*>(row_sum);

    float local_ss = 0.0f;
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 rv = res_vec[i];
        float4 xv = x_vec[i];
        const __half* rh = reinterpret_cast<const __half*>(&rv);
        const __half* xh = reinterpret_cast<const __half*>(&xv);
        float4 sv;
        __half* sh = reinterpret_cast<__half*>(&sv);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float s = __half2float(rh[j]) + __half2float(xh[j]);
            sh[j] = __float2half(s);
            local_ss += s * s;
        }
        sum_vec[i] = sv;
    }
    for (int i = vec_size * 8 + tid; i < hidden_size; i += blockDim.x) {
        float s = __half2float(row_residual[i]) + __half2float(row_x[i]);
        row_sum[i] = __float2half(s);
        local_ss += s * s;
    }

    float rms = block_reduce_sum(local_ss, shared);
    rms = rsqrtf(rms / (float)hidden_size + eps);

    const float4* w_vec = reinterpret_cast<const float4*>(weight);
    float4* norm_vec = reinterpret_cast<float4*>(row_norm);
    const float4* sum_vec_r = reinterpret_cast<const float4*>(row_sum);

    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 sv = sum_vec_r[i];
        float4 wv = w_vec[i];
        const __half* sh = reinterpret_cast<const __half*>(&sv);
        const __half* wh = reinterpret_cast<const __half*>(&wv);
        float4 nv;
        __half* nh = reinterpret_cast<__half*>(&nv);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            nh[j] = __float2half(__half2float(sh[j]) * rms * __half2float(wh[j]));
        }
        norm_vec[i] = nv;
    }
    for (int i = vec_size * 8 + tid; i < hidden_size; i += blockDim.x) {
        float sum_val = __half2float(row_sum[i]);
        row_norm[i] = __float2half(sum_val * rms * __half2float(weight[i]));
    }
}

// ============================================================================
// BF16 variant
// ============================================================================

extern "C" __global__ void add_rmsnorm_bf16(
    __nv_bfloat16* __restrict__ sum_out,
    __nv_bfloat16* __restrict__ norm_out,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* row_residual = residual + row * hidden_size;
    const __nv_bfloat16* row_x = x + row * hidden_size;
    __nv_bfloat16* row_sum = sum_out + row * hidden_size;
    __nv_bfloat16* row_norm = norm_out + row * hidden_size;

    extern __shared__ float shared[];

    // float4 = 16 bytes = 8 bfloat16s
    const int vec_size = hidden_size / 8;
    const float4* res_vec = reinterpret_cast<const float4*>(row_residual);
    const float4* x_vec = reinterpret_cast<const float4*>(row_x);
    float4* sum_vec = reinterpret_cast<float4*>(row_sum);

    float local_ss = 0.0f;
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 rv = res_vec[i];
        float4 xv = x_vec[i];
        const __nv_bfloat16* rh = reinterpret_cast<const __nv_bfloat16*>(&rv);
        const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xv);
        float4 sv;
        __nv_bfloat16* sh = reinterpret_cast<__nv_bfloat16*>(&sv);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float s = __bfloat162float(rh[j]) + __bfloat162float(xh[j]);
            sh[j] = __float2bfloat16(s);
            local_ss += s * s;
        }
        sum_vec[i] = sv;
    }
    for (int i = vec_size * 8 + tid; i < hidden_size; i += blockDim.x) {
        float s = __bfloat162float(row_residual[i]) + __bfloat162float(row_x[i]);
        row_sum[i] = __float2bfloat16(s);
        local_ss += s * s;
    }

    float rms = block_reduce_sum(local_ss, shared);
    rms = rsqrtf(rms / (float)hidden_size + eps);

    const float4* w_vec = reinterpret_cast<const float4*>(weight);
    float4* norm_vec = reinterpret_cast<float4*>(row_norm);
    const float4* sum_vec_r = reinterpret_cast<const float4*>(row_sum);

    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 sv = sum_vec_r[i];
        float4 wv = w_vec[i];
        const __nv_bfloat16* sh = reinterpret_cast<const __nv_bfloat16*>(&sv);
        const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wv);
        float4 nv;
        __nv_bfloat16* nh = reinterpret_cast<__nv_bfloat16*>(&nv);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            nh[j] = __float2bfloat16(__bfloat162float(sh[j]) * rms * __bfloat162float(wh[j]));
        }
        norm_vec[i] = nv;
    }
    for (int i = vec_size * 8 + tid; i < hidden_size; i += blockDim.x) {
        float sum_val = __bfloat162float(row_sum[i]);
        row_norm[i] = __float2bfloat16(sum_val * rms * __bfloat162float(weight[i]));
    }
}
