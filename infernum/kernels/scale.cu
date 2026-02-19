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
