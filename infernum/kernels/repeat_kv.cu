extern "C" __global__ void repeat_kv_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int seq_len,
    const int num_kv_heads,
    const int num_repeats,
    const int head_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int new_num_heads = num_kv_heads * num_repeats;
    const int total = seq_len * new_num_heads * head_dim;
    if (idx < total) {
        const int s = idx / (new_num_heads * head_dim);
        const int remainder = idx % (new_num_heads * head_dim);
        const int new_head = remainder / head_dim;
        const int d = remainder % head_dim;

        const int kv_head = new_head / num_repeats;
        const int src_idx = s * num_kv_heads * head_dim + kv_head * head_dim + d;
        output[idx] = input[src_idx];
    }
}
