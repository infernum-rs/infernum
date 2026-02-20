#include <cuda_fp16.h>
#include <cuda_bf16.h>

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
    
    // Compute sum of squares using block reduction
    extern __shared__ float shared[];
    
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = row_input[i];
        local_sum += val * val;
    }
    
    shared[tid] = local_sum;
    __syncthreads();
    
    // Block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    float rms = rsqrtf(shared[0] / (float)hidden_size + eps);
    
    // Apply normalization and weight
    for (int i = tid; i < hidden_size; i += blockDim.x) {
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
    
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = row_data[i];
        local_sum += val * val;
    }
    
    shared[tid] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    float rms = rsqrtf(shared[0] / (float)hidden_size + eps);
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        row_data[i] = row_data[i] * rms * weight[i];
    }
}

// ============================================================================
// F16 variants: input/output/weight in half, accumulation in f32
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
    
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __half2float(row_input[i]);
        local_sum += val * val;
    }
    
    shared[tid] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    float rms = rsqrtf(shared[0] / (float)hidden_size + eps);
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
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
    
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __half2float(row_data[i]);
        local_sum += val * val;
    }
    
    shared[tid] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    float rms = rsqrtf(shared[0] / (float)hidden_size + eps);
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __half2float(row_data[i]) * rms * __half2float(weight[i]);
        row_data[i] = __float2half(val);
    }
}

// ============================================================================
// BF16 variants: input/output/weight in bfloat16, accumulation in f32
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
    
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(row_input[i]);
        local_sum += val * val;
    }
    
    shared[tid] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    float rms = rsqrtf(shared[0] / (float)hidden_size + eps);
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
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
    
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(row_data[i]);
        local_sum += val * val;
    }
    
    shared[tid] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    float rms = rsqrtf(shared[0] / (float)hidden_size + eps);
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(row_data[i]) * rms * __bfloat162float(weight[i]);
        row_data[i] = __float2bfloat16(val);
    }
}
