// Warp-level max reduction: propagates both value and index
static __device__ __forceinline__ void warp_reduce_max(float& val, unsigned int& idx) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, val, offset);
        unsigned int other_idx = __shfl_xor_sync(0xffffffff, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

extern "C" __global__ void argmax_last_f32(
    unsigned int* __restrict__ output,
    const float* __restrict__ input,
    const int row_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* row_input = input + row * row_size;

    // Each thread finds local max across its strided elements using float4 loads
    float local_max = -1e38f;
    unsigned int local_idx = 0;

    const int vec_size = row_size / 4;
    const float4* input_vec = reinterpret_cast<const float4*>(row_input);

    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 v = input_vec[i];
        int base = i * 4;
        if (v.x > local_max) { local_max = v.x; local_idx = base; }
        if (v.y > local_max) { local_max = v.y; local_idx = base + 1; }
        if (v.z > local_max) { local_max = v.z; local_idx = base + 2; }
        if (v.w > local_max) { local_max = v.w; local_idx = base + 3; }
    }
    // Handle remaining elements
    for (int i = vec_size * 4 + tid; i < row_size; i += blockDim.x) {
        float val = row_input[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    // Warp-level reduction
    warp_reduce_max(local_max, local_idx);

    const int num_warps = blockDim.x / 32;

    if (num_warps <= 1) {
        // Single warp — no inter-warp reduction needed
        if (tid == 0) {
            output[row] = local_idx;
        }
        return;
    }

    // Inter-warp reduction via shared memory
    extern __shared__ char smem[];
    float* shared_val = reinterpret_cast<float*>(smem);
    unsigned int* shared_idx = reinterpret_cast<unsigned int*>(
        smem + num_warps * sizeof(float));

    const int lane_id = tid % 32;
    const int warp_id = tid / 32;

    if (lane_id == 0) {
        shared_val[warp_id] = local_max;
        shared_idx[warp_id] = local_idx;
    }
    __syncthreads();

    // First warp reduces the per-warp results
    if (warp_id == 0) {
        local_max = (lane_id < num_warps) ? shared_val[lane_id] : -1e38f;
        local_idx = (lane_id < num_warps) ? shared_idx[lane_id] : 0;
        warp_reduce_max(local_max, local_idx);
    }

    if (tid == 0) {
        output[row] = local_idx;
    }
}
