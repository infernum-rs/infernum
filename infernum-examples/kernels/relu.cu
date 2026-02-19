// ReLU activation: out[i] = max(0, x[i])
//
// This is a minimal example of a custom CUDA kernel for Infernum.
// See examples/custom_cuda_op.rs for how to load and launch it.

extern "C" __global__ void relu_f32(float *out, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

extern "C" __global__ void relu_inplace_f32(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (x[i] < 0.0f) {
            x[i] = 0.0f;
        }
    }
}
