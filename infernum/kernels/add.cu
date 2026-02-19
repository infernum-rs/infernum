extern "C" __global__ void add_f32(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

extern "C" __global__ void add_inplace_f32(
    float* __restrict__ a,
    const float* __restrict__ b,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += b[idx];
    }
}
