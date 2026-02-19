#define INFINITY __int_as_float(0x7f800000)

extern "C" __global__ void fused_decode_attention_f32(
    float* __restrict__ output,       // (1, num_heads, head_dim)
    const float* __restrict__ q,      // (1, num_heads, head_dim)
    const float* __restrict__ k,      // (total_len, num_kv_heads, head_dim)
    const float* __restrict__ v,      // (total_len, num_kv_heads, head_dim)
    const float scale,
    const int total_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;

    // GQA: map query head to KV head
    const int kv_head = head * num_kv_heads / num_heads;

    // Shared memory layout:
    //   [0 .. head_dim)           : Q vector for this head
    //   [head_dim .. head_dim + blockDim.x) : reduction scratch
    extern __shared__ float shared[];
    float* s_q = shared;
    float* s_scratch = shared + head_dim;

    // Load Q into shared memory
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = q[head * head_dim + d];
    }
    __syncthreads();

    // Pass 1: compute scores, find max for softmax
    // Each thread handles a subset of key positions
    float local_max = -INFINITY;
    for (int t = tid; t < total_len; t += blockDim.x) {
        float dot = 0.0f;
        const float* k_ptr = k + t * num_kv_heads * head_dim + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * k_ptr[d];
        }
        dot *= scale;
        local_max = fmaxf(local_max, dot);
    }

    // Reduce max across threads
    s_scratch[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] = fmaxf(s_scratch[tid], s_scratch[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = s_scratch[0];
    __syncthreads();

    // Pass 2: compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (int t = tid; t < total_len; t += blockDim.x) {
        float dot = 0.0f;
        const float* k_ptr = k + t * num_kv_heads * head_dim + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            dot += s_q[d] * k_ptr[d];
        }
        dot *= scale;
        local_sum += expf(dot - max_val);
    }

    s_scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = s_scratch[0];
    __syncthreads();

    // Pass 3: compute weighted output
    // Each thread accumulates partial output across its assigned positions
    // We iterate over output dimensions in the outer loop for better write pattern
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < total_len; t++) {
            // Recompute score for position t
            float dot = 0.0f;
            const float* k_ptr = k + t * num_kv_heads * head_dim + kv_head * head_dim;
            for (int dd = 0; dd < head_dim; dd++) {
                dot += s_q[dd] * k_ptr[dd];
            }
            dot *= scale;
            float weight = expf(dot - max_val) / sum_val;

            const float* v_ptr = v + t * num_kv_heads * head_dim + kv_head * head_dim;
            acc += weight * v_ptr[d];
        }
        output[head * head_dim + d] = acc;
    }
}
