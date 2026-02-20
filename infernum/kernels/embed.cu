extern "C" __global__ void embedding_gather_f32(
    float* __restrict__ output,
    const float* __restrict__ embed_table,
    const unsigned int* __restrict__ input_ids,
    const int seq_len,
    const int hidden_size
) {
    const int pos = blockIdx.x;
    const int dim = blockIdx.y * blockDim.x + threadIdx.x;

    if (pos < seq_len && dim < hidden_size) {
        unsigned int token_id = input_ids[pos];
        output[pos * hidden_size + dim] = embed_table[token_id * hidden_size + dim];
    }
}
