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

    // First warp reduces the per-warp sums
    if (warp_id == 0) {
        val = (lane_id < (int)(blockDim.x / 32)) ? shared[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        // Broadcast result to shared memory so all warps can read it
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

extern "C" __global__ void rmsnorm_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;

    extern __shared__ float shared[];

    // Vectorized sum-of-squares with float4 loads
    const int vec_size = hidden_size / 4;
    const float4* input_vec = reinterpret_cast<const float4*>(row_input);

    float local_sum = 0.0f;
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 v = input_vec[i];
        local_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    // Handle remaining elements
    for (int i = vec_size * 4 + tid; i < hidden_size; i += blockDim.x) {
        float val = row_input[i];
        local_sum += val * val;
    }

    float rms = block_reduce_sum(local_sum, shared);
    rms = rsqrtf(rms / (float)hidden_size + eps);

    // Vectorized normalize + weight
    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    float4* output_vec = reinterpret_cast<float4*>(row_output);

    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 v = input_vec[i];
        float4 w = weight_vec[i];
        output_vec[i] = make_float4(v.x * rms * w.x, v.y * rms * w.y,
                                     v.z * rms * w.z, v.w * rms * w.w);
    }
    for (int i = vec_size * 4 + tid; i < hidden_size; i += blockDim.x) {
        row_output[i] = row_input[i] * rms * weight[i];
    }
}

extern "C" __global__ void rmsnorm_inplace_f32(
    float* __restrict__ data,
    const float* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    float* row_data = data + row * hidden_size;

    extern __shared__ float shared[];

    const int vec_size = hidden_size / 4;
    const float4* data_vec = reinterpret_cast<const float4*>(row_data);

    float local_sum = 0.0f;
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 v = data_vec[i];
        local_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int i = vec_size * 4 + tid; i < hidden_size; i += blockDim.x) {
        float val = row_data[i];
        local_sum += val * val;
    }

    float rms = block_reduce_sum(local_sum, shared);
    rms = rsqrtf(rms / (float)hidden_size + eps);

    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    float4* data_vec_mut = reinterpret_cast<float4*>(row_data);

    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 v = data_vec[i];
        float4 w = weight_vec[i];
        data_vec_mut[i] = make_float4(v.x * rms * w.x, v.y * rms * w.y,
                                       v.z * rms * w.z, v.w * rms * w.w);
    }
    for (int i = vec_size * 4 + tid; i < hidden_size; i += blockDim.x) {
        row_data[i] = row_data[i] * rms * weight[i];
    }
}

// ============================================================================
// F16 variants: input/output/weight in half, accumulation in f32
// Uses half2 vectorized loads (2 elements per load = 4 bytes)
// ============================================================================

extern "C" __global__ void rmsnorm_f16(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __half* row_input = input + row * hidden_size;
    __half* row_output = output + row * hidden_size;

    extern __shared__ float shared[];

    // Vectorized loads with float4 (8 halves = 16 bytes per load)
    const int vec_size = hidden_size / 8;
    const float4* input_vec = reinterpret_cast<const float4*>(row_input);

    float local_sum = 0.0f;
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 v = input_vec[i];
        const __half* h = reinterpret_cast<const __half*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float val = __half2float(h[j]);
            local_sum += val * val;
        }
    }
    for (int i = vec_size * 8 + tid; i < hidden_size; i += blockDim.x) {
        float val = __half2float(row_input[i]);
        local_sum += val * val;
    }

    float rms = block_reduce_sum(local_sum, shared);
    rms = rsqrtf(rms / (float)hidden_size + eps);

    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    float4* output_vec = reinterpret_cast<float4*>(row_output);

    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 iv = input_vec[i];
        float4 wv = weight_vec[i];
        const __half* ih = reinterpret_cast<const __half*>(&iv);
        const __half* wh = reinterpret_cast<const __half*>(&wv);
        float4 ov;
        __half* oh = reinterpret_cast<__half*>(&ov);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            oh[j] = __float2half(__half2float(ih[j]) * rms * __half2float(wh[j]));
        }
        output_vec[i] = ov;
    }
    for (int i = vec_size * 8 + tid; i < hidden_size; i += blockDim.x) {
        float val = __half2float(row_input[i]) * rms * __half2float(weight[i]);
        row_output[i] = __float2half(val);
    }
}

extern "C" __global__ void rmsnorm_inplace_f16(
    __half* __restrict__ data,
    const __half* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __half* row_data = data + row * hidden_size;

    extern __shared__ float shared[];

    const int vec_size = hidden_size / 8;
    const float4* data_vec = reinterpret_cast<const float4*>(row_data);

    float local_sum = 0.0f;
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 v = data_vec[i];
        const __half* h = reinterpret_cast<const __half*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float val = __half2float(h[j]);
            local_sum += val * val;
        }
    }
    for (int i = vec_size * 8 + tid; i < hidden_size; i += blockDim.x) {
        float val = __half2float(row_data[i]);
        local_sum += val * val;
    }

    float rms = block_reduce_sum(local_sum, shared);
    rms = rsqrtf(rms / (float)hidden_size + eps);

    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    float4* data_vec_mut = reinterpret_cast<float4*>(row_data);

    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 iv = data_vec[i];
        float4 wv = weight_vec[i];
        const __half* ih = reinterpret_cast<const __half*>(&iv);
        const __half* wh = reinterpret_cast<const __half*>(&wv);
        float4 ov;
        __half* oh = reinterpret_cast<__half*>(&ov);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            oh[j] = __float2half(__half2float(ih[j]) * rms * __half2float(wh[j]));
        }
        data_vec_mut[i] = ov;
    }
    for (int i = vec_size * 8 + tid; i < hidden_size; i += blockDim.x) {
        float val = __half2float(row_data[i]) * rms * __half2float(weight[i]);
        row_data[i] = __float2half(val);
    }
}

// ============================================================================
// BF16 variants: input/output/weight in bfloat16, accumulation in f32
// Uses float4 vectorized loads (8 bfloat16s = 16 bytes per load)
// ============================================================================

extern "C" __global__ void rmsnorm_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* row_input = input + row * hidden_size;
    __nv_bfloat16* row_output = output + row * hidden_size;

    extern __shared__ float shared[];

    // float4 = 16 bytes = 8 bfloat16s
    const int vec_size = hidden_size / 8;
    const float4* input_vec = reinterpret_cast<const float4*>(row_input);

    float local_sum = 0.0f;
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 v = input_vec[i];
        const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float val = __bfloat162float(h[j]);
            local_sum += val * val;
        }
    }
    for (int i = vec_size * 8 + tid; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(row_input[i]);
        local_sum += val * val;
    }

    float rms = block_reduce_sum(local_sum, shared);
    rms = rsqrtf(rms / (float)hidden_size + eps);

    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    float4* output_vec = reinterpret_cast<float4*>(row_output);

    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 iv = input_vec[i];
        float4 wv = weight_vec[i];
        const __nv_bfloat16* ih = reinterpret_cast<const __nv_bfloat16*>(&iv);
        const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wv);
        float4 ov;
        __nv_bfloat16* oh = reinterpret_cast<__nv_bfloat16*>(&ov);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            oh[j] = __float2bfloat16(__bfloat162float(ih[j]) * rms * __bfloat162float(wh[j]));
        }
        output_vec[i] = ov;
    }
    for (int i = vec_size * 8 + tid; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(row_input[i]) * rms * __bfloat162float(weight[i]);
        row_output[i] = __float2bfloat16(val);
    }
}

extern "C" __global__ void rmsnorm_inplace_bf16(
    __nv_bfloat16* __restrict__ data,
    const __nv_bfloat16* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __nv_bfloat16* row_data = data + row * hidden_size;

    extern __shared__ float shared[];

    const int vec_size = hidden_size / 8;
    const float4* data_vec = reinterpret_cast<const float4*>(row_data);

    float local_sum = 0.0f;
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 v = data_vec[i];
        const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float val = __bfloat162float(h[j]);
            local_sum += val * val;
        }
    }
    for (int i = vec_size * 8 + tid; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(row_data[i]);
        local_sum += val * val;
    }

    float rms = block_reduce_sum(local_sum, shared);
    rms = rsqrtf(rms / (float)hidden_size + eps);

    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    float4* data_vec_mut = reinterpret_cast<float4*>(row_data);

    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 iv = data_vec[i];
        float4 wv = weight_vec[i];
        const __nv_bfloat16* ih = reinterpret_cast<const __nv_bfloat16*>(&iv);
        const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wv);
        float4 ov;
        __nv_bfloat16* oh = reinterpret_cast<__nv_bfloat16*>(&ov);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            oh[j] = __float2bfloat16(__bfloat162float(ih[j]) * rms * __bfloat162float(wh[j]));
        }
        data_vec_mut[i] = ov;
    }
    for (int i = vec_size * 8 + tid; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(row_data[i]) * rms * __bfloat162float(weight[i]);
        row_data[i] = __float2bfloat16(val);
    }
}
