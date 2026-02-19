extern "C" __global__ void argmax_last_f32(
    unsigned int* __restrict__ output,
    const float* __restrict__ input,
    const int row_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ char smem[];
    float* shared_val = (float*)smem;
    unsigned int* shared_idx = (unsigned int*)(smem + blockDim.x * sizeof(float));

    const float* row_input = input + row * row_size;

    // Each thread finds local max across its strided elements
    float local_max = -1e38f;
    unsigned int local_idx = 0;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float val = row_input[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    shared_val[tid] = local_max;
    shared_idx[tid] = local_idx;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_val[tid + stride] > shared_val[tid]) {
                shared_val[tid] = shared_val[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = shared_idx[0];
    }
}
