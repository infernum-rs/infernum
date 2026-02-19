// Fused residual add + RMS normalization.
//
// For each row: computes sum = residual + x, then RMS-normalizes sum.
// Writes both the sum (updated hidden state) and the normed result,
// saving one global memory round-trip compared to separate add + rmsnorm.

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

    // Pass 1: compute sum and accumulate sum-of-squares
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float s = row_residual[i] + row_x[i];
        row_sum[i] = s;
        local_ss += s * s;
    }

    shared[tid] = local_ss;
    __syncthreads();

    // Block reduction for sum-of-squares
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / (float)hidden_size + eps);

    // Pass 2: apply normalization and weight
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        row_norm[i] = row_sum[i] * rms * weight[i];
    }
}
