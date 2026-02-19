// CUDA doesn't have INFINITY in NVRTC, use the IEEE 754 representation
#define INFINITY __int_as_float(0x7f800000)

extern "C" __global__ void softmax_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int row_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    extern __shared__ float shared[];
    
    const float* row_input = input + row * row_size;
    float* row_output = output + row * row_size;
    
    // Step 1: Find max (for numerical stability)
    float local_max = -INFINITY;
    for (int i = tid; i < row_size; i += blockDim.x) {
        local_max = fmaxf(local_max, row_input[i]);
    }
    
    shared[tid] = local_max;
    __syncthreads();
    
    // Reduce to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    
    float max_val = shared[0];
    __syncthreads();
    
    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float exp_val = expf(row_input[i] - max_val);
        row_output[i] = exp_val;  // Temporarily store exp values
        local_sum += exp_val;
    }
    
    shared[tid] = local_sum;
    __syncthreads();
    
    // Reduce to find sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    float sum_val = shared[0];
    
    // Step 3: Normalize
    for (int i = tid; i < row_size; i += blockDim.x) {
        row_output[i] /= sum_val;
    }
}

// Softmax with causal mask for attention
extern "C" __global__ void softmax_causal_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int row_size,
    const int query_idx,      // Which query position this is
    const int position_offset // For KV cache scenarios
) {
    const int row = blockIdx.x;  // Which head
    const int tid = threadIdx.x;
    
    extern __shared__ float shared[];
    
    const float* row_input = input + row * row_size;
    float* row_output = output + row * row_size;
    
    // The causal mask means: for query at position q, 
    // we can only attend to key positions k where k <= q
    const int max_valid_k = query_idx + position_offset + 1;
    
    // Step 1: Find max (only over valid positions)
    float local_max = -INFINITY;
    for (int i = tid; i < row_size && i < max_valid_k; i += blockDim.x) {
        local_max = fmaxf(local_max, row_input[i]);
    }
    
    shared[tid] = local_max;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    
    float max_val = shared[0];
    __syncthreads();
    
    // Step 2: Compute exp and sum (only valid positions)
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        if (i < max_valid_k) {
            float exp_val = expf(row_input[i] - max_val);
            row_output[i] = exp_val;
            local_sum += exp_val;
        } else {
            row_output[i] = 0.0f;  // Masked positions get zero probability
        }
    }
    
    shared[tid] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    float sum_val = shared[0];
    
    // Step 3: Normalize
    for (int i = tid; i < max_valid_k && i < row_size; i += blockDim.x) {
        row_output[i] /= sum_val;
    }
}
