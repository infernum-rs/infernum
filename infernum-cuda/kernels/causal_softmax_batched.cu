#define INFINITY __int_as_float(0x7f800000)

// Causal softmax over (batch, seq_q, seq_k) attention scores.
// One block per (batch, query) row. Causal mask: allow k <= query_idx.
// Layout: input/output are (batch * seq * seq) contiguous, row-major.
extern "C" __global__ void causal_softmax_batched_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int batch,
    const int seq
) {
    const int row_idx = blockIdx.x;  // which (batch, query) pair
    const int batch_idx = row_idx / seq;
    const int query_idx = row_idx % seq;
    const int tid = threadIdx.x;

    extern __shared__ float shared[];

    const int row_offset = (batch_idx * seq + query_idx) * seq;
    const float* row_input = input + row_offset;
    float* row_output = output + row_offset;

    const int max_valid_k = query_idx + 1;

    // Step 1: Find max over valid positions
    float local_max = -INFINITY;
    for (int i = tid; i < max_valid_k; i += blockDim.x) {
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
    for (int i = tid; i < seq; i += blockDim.x) {
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
    for (int i = tid; i < max_valid_k; i += blockDim.x) {
        row_output[i] /= sum_val;
    }
}
