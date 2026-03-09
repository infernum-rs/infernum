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
// Optimized Q8_0 GEMV: 4 output neurons per threadgroup (4 SIMD-groups).
//
// Each SIMD-group (32 threads) handles one neuron. 4 SIMD-groups share
// the same threadgroup, enabling future input-sharing optimizations.
// Weight loads use uint4 (16 bytes = 16 int8 values) per thread.
//
// Grid: (ceil(N/4), 1, 1), threads_per_group = (128, 1, 1).
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
    const uint simd_idx = tid / 32;  // which of 4 SIMD-groups
    const uint lane = tid % 32;      // lane within SIMD-group

    const uint neuron = group_id * Q8_ROWS_PER_TG + simd_idx;
    if (neuron >= params.N) return;

    const uint K = params.K;
    const uint BLOCK_SIZE = 32;
    const uint num_blocks = K / BLOCK_SIZE;
    const uint quant_bytes_per_row = num_blocks * BLOCK_SIZE;

    device const uchar* row_data = weight_data + neuron * quant_bytes_per_row;
    device const float* row_scales = weight_scales + neuron * num_blocks;

    float partial = 0.0f;

    // Each lane handles blocks [lane, lane+32, ...]
    for (uint b = lane; b < num_blocks; b += 32) {
        float scale = row_scales[b];
        device const uchar* block_ptr = row_data + b * BLOCK_SIZE;
        device const float4* inp_ptr = (device const float4*)(input + b * BLOCK_SIZE);

        // Load 32 bytes as 2 × uint4 (each uint4 = 16 bytes = 16 int8 values)
        device const uint4* w_ptr = (device const uint4*)block_ptr;
        uint4 w0 = w_ptr[0];  // bytes 0-15
        uint4 w1 = w_ptr[1];  // bytes 16-31

        // Load 32 input floats as 8 × float4
        float4 x0 = inp_ptr[0];
        float4 x1 = inp_ptr[1];
        float4 x2 = inp_ptr[2];
        float4 x3 = inp_ptr[3];
        float4 x4 = inp_ptr[4];
        float4 x5 = inp_ptr[5];
        float4 x6 = inp_ptr[6];
        float4 x7 = inp_ptr[7];

        // Unpack and dot-product: extract bytes from uint4 components
        // w0.x has bytes 0-3, w0.y has bytes 4-7, etc.
        float block_sum = 0.0f;

        // Process w0 (16 bytes → 16 int8 values → pairs with x0..x3)
        block_sum += x0.x * float(as_type<char4>(w0.x).x);
        block_sum += x0.y * float(as_type<char4>(w0.x).y);
        block_sum += x0.z * float(as_type<char4>(w0.x).z);
        block_sum += x0.w * float(as_type<char4>(w0.x).w);
        block_sum += x1.x * float(as_type<char4>(w0.y).x);
        block_sum += x1.y * float(as_type<char4>(w0.y).y);
        block_sum += x1.z * float(as_type<char4>(w0.y).z);
        block_sum += x1.w * float(as_type<char4>(w0.y).w);
        block_sum += x2.x * float(as_type<char4>(w0.z).x);
        block_sum += x2.y * float(as_type<char4>(w0.z).y);
        block_sum += x2.z * float(as_type<char4>(w0.z).z);
        block_sum += x2.w * float(as_type<char4>(w0.z).w);
        block_sum += x3.x * float(as_type<char4>(w0.w).x);
        block_sum += x3.y * float(as_type<char4>(w0.w).y);
        block_sum += x3.z * float(as_type<char4>(w0.w).z);
        block_sum += x3.w * float(as_type<char4>(w0.w).w);

        // Process w1 (16 bytes → 16 int8 values → pairs with x4..x7)
        block_sum += x4.x * float(as_type<char4>(w1.x).x);
        block_sum += x4.y * float(as_type<char4>(w1.x).y);
        block_sum += x4.z * float(as_type<char4>(w1.x).z);
        block_sum += x4.w * float(as_type<char4>(w1.x).w);
        block_sum += x5.x * float(as_type<char4>(w1.y).x);
        block_sum += x5.y * float(as_type<char4>(w1.y).y);
        block_sum += x5.z * float(as_type<char4>(w1.y).z);
        block_sum += x5.w * float(as_type<char4>(w1.y).w);
        block_sum += x6.x * float(as_type<char4>(w1.z).x);
        block_sum += x6.y * float(as_type<char4>(w1.z).y);
        block_sum += x6.z * float(as_type<char4>(w1.z).z);
        block_sum += x6.w * float(as_type<char4>(w1.z).w);
        block_sum += x7.x * float(as_type<char4>(w1.w).x);
        block_sum += x7.y * float(as_type<char4>(w1.w).y);
        block_sum += x7.z * float(as_type<char4>(w1.w).z);
        block_sum += x7.w * float(as_type<char4>(w1.w).w);

        partial += block_sum * scale;
    }

    float result = simd_sum(partial);
    if (lane == 0) {
        output[neuron] = result;
    }
}

// ==========================================================================
// Optimized Q4_0 GEMV: 4 output neurons per threadgroup (4 SIMD-groups).
//
// Each SIMD-group handles one neuron. Weight block = 16 packed bytes
// (32 nibbles) loaded as a single uint4. Input loaded as float4.
//
// Nibble layout: byte[j] → lo nibble = element j, hi nibble = element j+16.
// Both offset by -8.
//
// Grid: (ceil(N/4), 1, 1), threads_per_group = (128, 1, 1).
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

        // Load all 16 packed bytes as a single uint4 (16 bytes)
        uint4 packed = ((device const uint4*)block_ptr)[0];

        // Load 32 input floats: first 16 for lo nibbles, next 16 for hi nibbles
        uint inp_start = b * BLOCK_SIZE;
        device const float4* lo_ptr = (device const float4*)(input + inp_start);
        device const float4* hi_ptr = (device const float4*)(input + inp_start + 16);

        float4 lo_x0 = lo_ptr[0];  // input[0..3]
        float4 lo_x1 = lo_ptr[1];  // input[4..7]
        float4 lo_x2 = lo_ptr[2];  // input[8..11]
        float4 lo_x3 = lo_ptr[3];  // input[12..15]
        float4 hi_x0 = hi_ptr[0];  // input[16..19]
        float4 hi_x1 = hi_ptr[1];  // input[20..23]
        float4 hi_x2 = hi_ptr[2];  // input[24..27]
        float4 hi_x3 = hi_ptr[3];  // input[28..31]

        // Unpack nibbles from packed uint4.
        // packed.x = bytes 0-3, packed.y = bytes 4-7,
        // packed.z = bytes 8-11, packed.w = bytes 12-15.
        // byte[j]: lo nibble = element j (offset -8), hi nibble = element j+16 (offset -8).
        float block_sum = 0.0f;

        // Helper: extract lo/hi nibbles from 4 bytes in a uint
        // Byte 0 of u: lo = (u & 0xF) - 8, hi = ((u >> 4) & 0xF) - 8
        // Byte 1 of u: lo = ((u >> 8) & 0xF) - 8, hi = ((u >> 12) & 0xF) - 8
        // etc.

        // Process packed.x (bytes 0-3 → elements 0-3 lo, 16-19 hi)
        uchar4 b0 = as_type<uchar4>(packed.x);
        block_sum += lo_x0.x * float(int(b0.x & 0x0F) - 8);
        block_sum += hi_x0.x * float(int(b0.x >> 4) - 8);
        block_sum += lo_x0.y * float(int(b0.y & 0x0F) - 8);
        block_sum += hi_x0.y * float(int(b0.y >> 4) - 8);
        block_sum += lo_x0.z * float(int(b0.z & 0x0F) - 8);
        block_sum += hi_x0.z * float(int(b0.z >> 4) - 8);
        block_sum += lo_x0.w * float(int(b0.w & 0x0F) - 8);
        block_sum += hi_x0.w * float(int(b0.w >> 4) - 8);

        // Process packed.y (bytes 4-7 → elements 4-7 lo, 20-23 hi)
        uchar4 b1 = as_type<uchar4>(packed.y);
        block_sum += lo_x1.x * float(int(b1.x & 0x0F) - 8);
        block_sum += hi_x1.x * float(int(b1.x >> 4) - 8);
        block_sum += lo_x1.y * float(int(b1.y & 0x0F) - 8);
        block_sum += hi_x1.y * float(int(b1.y >> 4) - 8);
        block_sum += lo_x1.z * float(int(b1.z & 0x0F) - 8);
        block_sum += hi_x1.z * float(int(b1.z >> 4) - 8);
        block_sum += lo_x1.w * float(int(b1.w & 0x0F) - 8);
        block_sum += hi_x1.w * float(int(b1.w >> 4) - 8);

        // Process packed.z (bytes 8-11 → elements 8-11 lo, 24-27 hi)
        uchar4 b2 = as_type<uchar4>(packed.z);
        block_sum += lo_x2.x * float(int(b2.x & 0x0F) - 8);
        block_sum += hi_x2.x * float(int(b2.x >> 4) - 8);
        block_sum += lo_x2.y * float(int(b2.y & 0x0F) - 8);
        block_sum += hi_x2.y * float(int(b2.y >> 4) - 8);
        block_sum += lo_x2.z * float(int(b2.z & 0x0F) - 8);
        block_sum += hi_x2.z * float(int(b2.z >> 4) - 8);
        block_sum += lo_x2.w * float(int(b2.w & 0x0F) - 8);
        block_sum += hi_x2.w * float(int(b2.w >> 4) - 8);

        // Process packed.w (bytes 12-15 → elements 12-15 lo, 28-31 hi)
        uchar4 b3 = as_type<uchar4>(packed.w);
        block_sum += lo_x3.x * float(int(b3.x & 0x0F) - 8);
        block_sum += hi_x3.x * float(int(b3.x >> 4) - 8);
        block_sum += lo_x3.y * float(int(b3.y & 0x0F) - 8);
        block_sum += hi_x3.y * float(int(b3.y >> 4) - 8);
        block_sum += lo_x3.z * float(int(b3.z & 0x0F) - 8);
        block_sum += hi_x3.z * float(int(b3.z >> 4) - 8);
        block_sum += lo_x3.w * float(int(b3.w & 0x0F) - 8);
        block_sum += hi_x3.w * float(int(b3.w >> 4) - 8);

        partial += block_sum * scale;
    }

    float result = simd_sum(partial);
    if (lane == 0) {
        output[neuron] = result;
    }
}
