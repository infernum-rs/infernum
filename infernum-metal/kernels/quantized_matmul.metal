#include <metal_stdlib>
using namespace metal;

// Params struct shared by all quantized GEMV kernels.
struct QuantizedLinearParams {
    uint N;  // out_features
    uint K;  // in_features
};

// ==========================================================================
// Q8_0 GEMV: one thread per output neuron
// Weight layout: N rows, each row has K/32 blocks of 32 int8 values.
// Scales: N * (K/32) float values.
// ==========================================================================
kernel void gemv_q8_f32(
    device const float*  input         [[buffer(0)]],
    device const uchar*  weight_data   [[buffer(1)]],
    device const float*  weight_scales [[buffer(2)]],
    device float*        output        [[buffer(3)]],
    constant QuantizedLinearParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.N) return;

    const uint K = params.K;
    const uint BLOCK_SIZE = 32;
    const uint num_blocks = K / BLOCK_SIZE;
    const uint quant_bytes_per_row = num_blocks * BLOCK_SIZE;

    float sum = 0.0f;
    const uint q_start = tid * quant_bytes_per_row;
    const uint s_start = tid * num_blocks;

    for (uint b = 0; b < num_blocks; b++) {
        float scale = weight_scales[s_start + b];
        uint block_start = q_start + b * BLOCK_SIZE;
        uint inp_start = b * BLOCK_SIZE;
        for (uint j = 0; j < BLOCK_SIZE; j++) {
            float q = float(as_type<char>(weight_data[block_start + j]));
            sum += input[inp_start + j] * q * scale;
        }
    }

    output[tid] = sum;
}

// ==========================================================================
// Q4_0 GEMV: one thread per output neuron
// Weight layout: N rows, each row has K/32 blocks of 16 packed bytes.
// Nibble unpacking: byte[j] → lo nibble (elem j), hi nibble (elem j+16).
// Both offset by -8.
// ==========================================================================
kernel void gemv_q4_f32(
    device const float*  input         [[buffer(0)]],
    device const uchar*  weight_data   [[buffer(1)]],
    device const float*  weight_scales [[buffer(2)]],
    device float*        output        [[buffer(3)]],
    constant QuantizedLinearParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.N) return;

    const uint K = params.K;
    const uint BLOCK_SIZE = 32;
    const uint HALF_BLOCK = BLOCK_SIZE / 2;
    const uint num_blocks = K / BLOCK_SIZE;
    const uint packed_bytes_per_row = num_blocks * HALF_BLOCK;

    float sum = 0.0f;
    const uint p_start = tid * packed_bytes_per_row;
    const uint s_start = tid * num_blocks;

    for (uint b = 0; b < num_blocks; b++) {
        float scale = weight_scales[s_start + b];
        uint block_start = p_start + b * HALF_BLOCK;
        uint inp_start = b * BLOCK_SIZE;
        for (uint j = 0; j < HALF_BLOCK; j++) {
            uchar byte_val = weight_data[block_start + j];
            float lo = float(int(byte_val & 0x0F) - 8);
            float hi = float(int(byte_val >> 4) - 8);
            sum += input[inp_start + j] * lo * scale;
            sum += input[inp_start + j + 16] * hi * scale;
        }
    }

    output[tid] = sum;
}

// ==========================================================================
// Q4_1 GEMV: one thread per output neuron
// Same as Q4_0 but with per-block min: value = nibble × scale + min.
// ==========================================================================
kernel void gemv_q4_1_f32(
    device const float*  input         [[buffer(0)]],
    device const uchar*  weight_data   [[buffer(1)]],
    device const float*  weight_scales [[buffer(2)]],
    device const float*  weight_mins   [[buffer(3)]],
    device float*        output        [[buffer(4)]],
    constant QuantizedLinearParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.N) return;

    const uint K = params.K;
    const uint BLOCK_SIZE = 32;
    const uint HALF_BLOCK = BLOCK_SIZE / 2;
    const uint num_blocks = K / BLOCK_SIZE;
    const uint packed_bytes_per_row = num_blocks * HALF_BLOCK;

    float sum = 0.0f;
    const uint p_start = tid * packed_bytes_per_row;
    const uint s_start = tid * num_blocks;

    for (uint b = 0; b < num_blocks; b++) {
        float scale = weight_scales[s_start + b];
        float min_val = weight_mins[s_start + b];
        uint block_start = p_start + b * HALF_BLOCK;
        uint inp_start = b * BLOCK_SIZE;
        for (uint j = 0; j < HALF_BLOCK; j++) {
            uchar byte_val = weight_data[block_start + j];
            float lo = float(byte_val & 0x0F);
            float hi = float(byte_val >> 4);
            sum += input[inp_start + j] * (lo * scale + min_val);
            sum += input[inp_start + j + 16] * (hi * scale + min_val);
        }
    }

    output[tid] = sum;
}

// ==========================================================================
// Q6_K GEMV: one thread per output neuron
// Weight layout: N rows, each row has K/256 super-blocks of 210 bytes.
// Super-block: ql[128] + qh[64] + scales[16] + d[2] (f16).
// ==========================================================================
kernel void gemv_q6k_f32(
    device const float*  input         [[buffer(0)]],
    device const uchar*  weight_data   [[buffer(1)]],
    device const float*  unused_scales [[buffer(2)]],
    device float*        output        [[buffer(3)]],
    constant QuantizedLinearParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.N) return;

    const uint K = params.K;
    const uint SUPERBLOCK_ELEMENTS = 256;
    const uint SUPERBLOCK_BYTES = 210;
    const uint num_superblocks = K / SUPERBLOCK_ELEMENTS;
    const uint superblock_bytes_per_row = num_superblocks * SUPERBLOCK_BYTES;

    float sum = 0.0f;
    const uint row_start = tid * superblock_bytes_per_row;

    for (uint sb = 0; sb < num_superblocks; sb++) {
        uint bs = row_start + sb * SUPERBLOCK_BYTES;

        device const uchar* ql = weight_data + bs;
        device const uchar* qh = weight_data + bs + 128;
        device const uchar* sc_bytes = weight_data + bs + 192;

        // Read d as f16 (little-endian)
        ushort d_bits = ushort(weight_data[bs + 208])
                      | (ushort(weight_data[bs + 209]) << 8);
        float d = float(as_type<half>(d_bits));

        uint inp_start = sb * SUPERBLOCK_ELEMENTS;

        for (uint elem = 0; elem < SUPERBLOCK_ELEMENTS; elem++) {
            uint sub_block = elem / 16;
            uint row8 = elem / 32;
            uint col32 = elem % 32;

            // ql extraction
            uint ql_half = row8 / 4;
            uint ql_nibble_sel = (row8 % 4) / 2;
            uint ql_offset = (row8 % 4) % 2;
            uint ql_byte_idx = ql_half * 64 + ql_offset * 32 + col32;
            uchar ql_byte = ql[ql_byte_idx];
            uint ql_val = (ql_nibble_sel == 0)
                        ? uint(ql_byte & 0x0F)
                        : uint(ql_byte >> 4);

            // qh extraction
            uint qh_half = row8 / 4;
            uint qh_shift_sel = row8 % 4;
            uint qh_byte_idx = qh_half * 32 + col32;
            uchar qh_byte = qh[qh_byte_idx];
            uint qh_shift = qh_shift_sel * 2;
            uint qh_val = uint((qh_byte >> qh_shift) & 0x03);

            int q = int(ql_val | (qh_val << 4)) - 32;
            float sc = float(as_type<char>(sc_bytes[sub_block]));
            sum += input[inp_start + elem] * d * sc * float(q);
        }
    }

    output[tid] = sum;
}

// ==========================================================================
// Optimized Q8_0 GEMV: NR output neurons per threadgroup.
//
// NR SIMD-groups share the same threadgroup. All access the same input
// addresses, so the GPU's L1/texture cache naturally provides sharing
// without threadgroup memory overhead. Uses half precision for the
// inner dot product (2× ALU throughput on Apple Silicon).
//
// Grid: (ceil(N/NR), 1, 1), threads_per_group = (NR*32, 1, 1).
// ==========================================================================
constant constexpr uint Q8_ROWS_PER_TG = 4;

kernel void gemv_q8_simd_f32(
    device const float*  input         [[buffer(0)]],
    device const uchar*  weight_data   [[buffer(1)]],
    device const float*  weight_scales [[buffer(2)]],
    device float*        output        [[buffer(3)]],
    constant QuantizedLinearParams& params [[buffer(4)]],
    uint group_id  [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]])
{
    const uint simd_idx = tid / 32;
    const uint lane = tid % 32;

    const uint neuron = group_id * Q8_ROWS_PER_TG + simd_idx;
    if (neuron >= params.N) return;

    const uint K = params.K;
    const uint BLOCK_SIZE = 32;
    const uint num_blocks = K / BLOCK_SIZE;
    const uint quant_bytes_per_row = num_blocks * BLOCK_SIZE;

    device const uchar* row_data = weight_data + neuron * quant_bytes_per_row;
    device const float* row_scales = weight_scales + neuron * num_blocks;

    float partial = 0.0f;

    for (uint b = lane; b < num_blocks; b += 32) {
        float scale = row_scales[b];
        device const uchar* block_ptr = row_data + b * BLOCK_SIZE;
        device const float4* inp_ptr = (device const float4*)(input + b * BLOCK_SIZE);

        device const uint4* w_ptr = (device const uint4*)block_ptr;
        uint4 w0 = w_ptr[0];
        uint4 w1 = w_ptr[1];

        half4 x0 = half4(inp_ptr[0]); half4 x1 = half4(inp_ptr[1]);
        half4 x2 = half4(inp_ptr[2]); half4 x3 = half4(inp_ptr[3]);
        half4 x4 = half4(inp_ptr[4]); half4 x5 = half4(inp_ptr[5]);
        half4 x6 = half4(inp_ptr[6]); half4 x7 = half4(inp_ptr[7]);

        half block_sum = 0.0h;

        char4 c0 = as_type<char4>(w0.x);
        block_sum += x0.x * half(c0.x); block_sum += x0.y * half(c0.y);
        block_sum += x0.z * half(c0.z); block_sum += x0.w * half(c0.w);
        char4 c1 = as_type<char4>(w0.y);
        block_sum += x1.x * half(c1.x); block_sum += x1.y * half(c1.y);
        block_sum += x1.z * half(c1.z); block_sum += x1.w * half(c1.w);
        char4 c2 = as_type<char4>(w0.z);
        block_sum += x2.x * half(c2.x); block_sum += x2.y * half(c2.y);
        block_sum += x2.z * half(c2.z); block_sum += x2.w * half(c2.w);
        char4 c3 = as_type<char4>(w0.w);
        block_sum += x3.x * half(c3.x); block_sum += x3.y * half(c3.y);
        block_sum += x3.z * half(c3.z); block_sum += x3.w * half(c3.w);

        char4 c4 = as_type<char4>(w1.x);
        block_sum += x4.x * half(c4.x); block_sum += x4.y * half(c4.y);
        block_sum += x4.z * half(c4.z); block_sum += x4.w * half(c4.w);
        char4 c5 = as_type<char4>(w1.y);
        block_sum += x5.x * half(c5.x); block_sum += x5.y * half(c5.y);
        block_sum += x5.z * half(c5.z); block_sum += x5.w * half(c5.w);
        char4 c6 = as_type<char4>(w1.z);
        block_sum += x6.x * half(c6.x); block_sum += x6.y * half(c6.y);
        block_sum += x6.z * half(c6.z); block_sum += x6.w * half(c6.w);
        char4 c7 = as_type<char4>(w1.w);
        block_sum += x7.x * half(c7.x); block_sum += x7.y * half(c7.y);
        block_sum += x7.z * half(c7.z); block_sum += x7.w * half(c7.w);

        partial += float(block_sum) * scale;
    }

    float result = simd_sum(partial);
    if (lane == 0) {
        output[neuron] = result;
    }
}

// ==========================================================================
// Optimized Q4_0 GEMV: NR output neurons per threadgroup.
//
// NR SIMD-groups share the same threadgroup. All access the same input
// addresses, so the GPU's L1/texture cache naturally provides sharing.
// Uses half precision for the inner dot product.
//
// Nibble layout: byte[j] → lo nibble = element j, hi nibble = element j+16.
// Both offset by -8.
//
// Grid: (ceil(N/NR), 1, 1), threads_per_group = (NR*32, 1, 1).
// ==========================================================================
constant constexpr uint Q4_ROWS_PER_TG = 4;

kernel void gemv_q4_simd_f32(
    device const float*  input         [[buffer(0)]],
    device const uchar*  weight_data   [[buffer(1)]],
    device const float*  weight_scales [[buffer(2)]],
    device float*        output        [[buffer(3)]],
    constant QuantizedLinearParams& params [[buffer(4)]],
    uint group_id  [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]])
{
    const uint simd_idx = tid / 32;
    const uint lane = tid % 32;

    const uint neuron = group_id * Q4_ROWS_PER_TG + simd_idx;
    if (neuron >= params.N) return;

    const uint K = params.K;
    const uint BLOCK_SIZE = 32;
    const uint HALF_BLOCK = BLOCK_SIZE / 2;
    const uint num_blocks = K / BLOCK_SIZE;
    const uint packed_bytes_per_row = num_blocks * HALF_BLOCK;

    device const uchar* row_data = weight_data + neuron * packed_bytes_per_row;
    device const float* row_scales = weight_scales + neuron * num_blocks;

    float partial = 0.0f;

    for (uint b = lane; b < num_blocks; b += 32) {
        float scale = row_scales[b];
        device const uchar* block_ptr = row_data + b * HALF_BLOCK;

        uint4 packed = ((device const uint4*)block_ptr)[0];

        uint inp_start = b * BLOCK_SIZE;
        device const float4* lo_ptr = (device const float4*)(input + inp_start);
        device const float4* hi_ptr = (device const float4*)(input + inp_start + 16);

        half4 lo_x0 = half4(lo_ptr[0]); half4 lo_x1 = half4(lo_ptr[1]);
        half4 lo_x2 = half4(lo_ptr[2]); half4 lo_x3 = half4(lo_ptr[3]);
        half4 hi_x0 = half4(hi_ptr[0]); half4 hi_x1 = half4(hi_ptr[1]);
        half4 hi_x2 = half4(hi_ptr[2]); half4 hi_x3 = half4(hi_ptr[3]);

        half block_sum = 0.0h;

        uchar4 p0 = as_type<uchar4>(packed.x);
        block_sum += lo_x0.x * half(int(p0.x & 0x0F) - 8);
        block_sum += hi_x0.x * half(int(p0.x >> 4) - 8);
        block_sum += lo_x0.y * half(int(p0.y & 0x0F) - 8);
        block_sum += hi_x0.y * half(int(p0.y >> 4) - 8);
        block_sum += lo_x0.z * half(int(p0.z & 0x0F) - 8);
        block_sum += hi_x0.z * half(int(p0.z >> 4) - 8);
        block_sum += lo_x0.w * half(int(p0.w & 0x0F) - 8);
        block_sum += hi_x0.w * half(int(p0.w >> 4) - 8);

        uchar4 p1 = as_type<uchar4>(packed.y);
        block_sum += lo_x1.x * half(int(p1.x & 0x0F) - 8);
        block_sum += hi_x1.x * half(int(p1.x >> 4) - 8);
        block_sum += lo_x1.y * half(int(p1.y & 0x0F) - 8);
        block_sum += hi_x1.y * half(int(p1.y >> 4) - 8);
        block_sum += lo_x1.z * half(int(p1.z & 0x0F) - 8);
        block_sum += hi_x1.z * half(int(p1.z >> 4) - 8);
        block_sum += lo_x1.w * half(int(p1.w & 0x0F) - 8);
        block_sum += hi_x1.w * half(int(p1.w >> 4) - 8);

        uchar4 p2 = as_type<uchar4>(packed.z);
        block_sum += lo_x2.x * half(int(p2.x & 0x0F) - 8);
        block_sum += hi_x2.x * half(int(p2.x >> 4) - 8);
        block_sum += lo_x2.y * half(int(p2.y & 0x0F) - 8);
        block_sum += hi_x2.y * half(int(p2.y >> 4) - 8);
        block_sum += lo_x2.z * half(int(p2.z & 0x0F) - 8);
        block_sum += hi_x2.z * half(int(p2.z >> 4) - 8);
        block_sum += lo_x2.w * half(int(p2.w & 0x0F) - 8);
        block_sum += hi_x2.w * half(int(p2.w >> 4) - 8);

        uchar4 p3 = as_type<uchar4>(packed.w);
        block_sum += lo_x3.x * half(int(p3.x & 0x0F) - 8);
        block_sum += hi_x3.x * half(int(p3.x >> 4) - 8);
        block_sum += lo_x3.y * half(int(p3.y & 0x0F) - 8);
        block_sum += hi_x3.y * half(int(p3.y >> 4) - 8);
        block_sum += lo_x3.z * half(int(p3.z & 0x0F) - 8);
        block_sum += hi_x3.z * half(int(p3.z >> 4) - 8);
        block_sum += lo_x3.w * half(int(p3.w & 0x0F) - 8);
        block_sum += hi_x3.w * half(int(p3.w >> 4) - 8);

        partial += float(block_sum) * scale;
    }

    float result = simd_sum(partial);
    if (lane == 0) {
        output[neuron] = result;
    }
}
