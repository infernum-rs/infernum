extern "C" __global__ void scale_f32(
    float* __restrict__ data,
    const float scale,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// Per-column broadcast scale: data[m, n] *= scales[n]
// data shape: [M, N] row-major, scales shape: [N]
// Grid/block: 1D over total elements (M * N)
extern "C" __global__ void scale_rows_f32(
    float* __restrict__ data,
    const float* __restrict__ scales,
    const int N,
    const int total
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx] *= scales[idx % N];
    }
}
