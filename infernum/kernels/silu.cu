extern "C" __global__ void silu_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

extern "C" __global__ void silu_inplace_f32(
    float* __restrict__ data,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        data[idx] = x / (1.0f + expf(-x));
    }
}

// SiLU with elementwise multiplication: output = silu(a) * b
// Used in SwiGLU: silu(gate) * up
extern "C" __global__ void silu_mul_f32(
    float* __restrict__ output,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = gate[idx];
        float silu_x = x / (1.0f + expf(-x));
        output[idx] = silu_x * up[idx];
    }
}
