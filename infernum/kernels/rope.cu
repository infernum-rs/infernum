extern "C" __global__ void rope_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int position_offset
) {
    // Each block handles one (seq_pos, head) pair
    // Each thread handles one pair of dimensions
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int pair_idx = threadIdx.x;  // Which pair of dimensions (0..head_dim/2)
    
    if (pair_idx >= head_dim / 2) return;
    
    const int pos = position_offset + seq_idx;
    
    // Input layout: (seq_len, num_heads, head_dim)
    // Sequential (half-half) convention: pairs are (x[i], x[i + head_dim/2])
    const int half_dim = head_dim / 2;
    const int base_idx = (seq_idx * num_heads + head_idx) * head_dim;
    const int idx0 = base_idx + pair_idx;
    const int idx1 = base_idx + pair_idx + half_dim;
    
    // Cache layout: (max_seq_len, head_dim/2)
    const int cache_idx = pos * (head_dim / 2) + pair_idx;
    
    float cos_val = cos_cache[cache_idx];
    float sin_val = sin_cache[cache_idx];
    
    float x0 = input[idx0];
    float x1 = input[idx1];
    
    output[idx0] = x0 * cos_val - x1 * sin_val;
    output[idx1] = x0 * sin_val + x1 * cos_val;
}
