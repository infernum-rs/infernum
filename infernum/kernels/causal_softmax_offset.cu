#define INFINITY __int_as_float(0x7f800000)

// Causal softmax over (batch, seq_q, seq_k) attention scores with position offset.
// For prefill with KV cache: query position i can attend to key positions [0..offset + i + 1].
// One block per (batch, query) row.
// Layout: input/output are (batch * seq_q * seq_k) contiguous, row-major.
extern "C" __global__ void causal_softmax_offset_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int batch,
    const int seq_q,
    const int seq_k,
    const int offset
) {
    const int row_idx = blockIdx.x;  // which (batch, query) pair
    const int query_idx = row_idx % seq_q;
    const int tid = threadIdx.x;

    extern __shared__ float shared[];

    const int row_offset = row_idx * seq_k;
    const float* row_input = input + row_offset;
    float* row_output = output + row_offset;

    const int max_valid_k = offset + query_idx + 1;

    // Step 1: Find max over valid positions
    float local_max = -INFINITY;
    for (int i = tid; i < max_valid_k && i < seq_k; i += blockDim.x) {
        local_max = fmaxf(local_max, row_input[i]);
    }

    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = shared[0];
    __syncthreads();

    // Step 2: Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < seq_k; i += blockDim.x) {
        if (i < max_valid_k) {
            float exp_val = expf(row_input[i] - max_val);
            row_output[i] = exp_val;
            local_sum += exp_val;
        } else {
            row_output[i] = 0.0f;
        }
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float sum_val = shared[0];

    // Step 3: Normalize
    for (int i = tid; i < max_valid_k && i < seq_k; i += blockDim.x) {
        row_output[i] /= sum_val;
    }
}
