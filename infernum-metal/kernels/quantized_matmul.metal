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

// ==========================================================================
// F16 activation variants: input is half*, output is half*.
// Eliminates f32-to-half conversion on input and half-to-f32 on output.
// ==========================================================================

kernel void gemv_q8_simd_f16(
    device const half*   input         [[buffer(0)]],
    device const uchar*  weight_data   [[buffer(1)]],
    device const float*  weight_scales [[buffer(2)]],
    device half*         output        [[buffer(3)]],
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
        device const half4* inp_ptr = (device const half4*)(input + b * BLOCK_SIZE);

        device const uint4* w_ptr = (device const uint4*)block_ptr;
        uint4 w0 = w_ptr[0];
        uint4 w1 = w_ptr[1];

        half4 x0 = inp_ptr[0]; half4 x1 = inp_ptr[1];
        half4 x2 = inp_ptr[2]; half4 x3 = inp_ptr[3];
        half4 x4 = inp_ptr[4]; half4 x5 = inp_ptr[5];
        half4 x6 = inp_ptr[6]; half4 x7 = inp_ptr[7];

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
        output[neuron] = half(result);
    }
}

kernel void gemv_q4_simd_f16(
    device const half*   input         [[buffer(0)]],
    device const uchar*  weight_data   [[buffer(1)]],
    device const float*  weight_scales [[buffer(2)]],
    device half*         output        [[buffer(3)]],
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
        device const half4* lo_ptr = (device const half4*)(input + inp_start);
        device const half4* hi_ptr = (device const half4*)(input + inp_start + 16);

        half4 lo_x0 = lo_ptr[0]; half4 lo_x1 = lo_ptr[1];
        half4 lo_x2 = lo_ptr[2]; half4 lo_x3 = lo_ptr[3];
        half4 hi_x0 = hi_ptr[0]; half4 hi_x1 = hi_ptr[1];
        half4 hi_x2 = hi_ptr[2]; half4 hi_x3 = hi_ptr[3];

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
        output[neuron] = half(result);
    }
}

// ==========================================================================
// Optimized GEMV v2 using dot(half4, half4) vectorized dot products.
//
// Key idea: each SIMD-group computes NR output rows. All 32 lanes in
// the SIMD-group share the same input chunk (loaded once), and each lane
// accumulates partial dot products across K/4 chunks using dot(half4,half4).
// simd_sum reduces across lanes at the end.
//
// This gives NR× better input cache reuse than v1 (which does 1 row
// per SIMD-group) and uses vectorized dot products instead of scalar MADs.
//
// Grid: (ceil(N / (NR*nsg)), 1, 1), threads_per_group = (nsg*32, 1, 1).
// Each lane walks K in strides of 32*4 = 128 elements.
// ==========================================================================

constant constexpr uint NR_V2 = 4;      // output rows per SIMD-group
constant constexpr uint NSG_V2 = 2;     // SIMD-groups per threadgroup
constant constexpr uint ROWS_PER_TG_V2 = NR_V2 * NSG_V2;  // 8 rows per TG
constant constexpr uint THREADS_PER_TG_V2 = NSG_V2 * 32;   // 64 threads per TG

// --------------------------------------------------------------------------
// Q8_0 GEMV v2 (vectorized dot product) — f32 output
//
// Each lane processes Q8_0 blocks in strides. For each block (32 elements),
// the lane computes 8 dot(half4, half4) products across NR rows and
// accumulates into per-row partial sums.
// --------------------------------------------------------------------------
kernel void gemv_q8_simd_v2_f32(
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

    const uint row_base = group_id * ROWS_PER_TG_V2 + simd_idx * NR_V2;

    const uint K = params.K;
    const uint BLOCK_SIZE = 32;
    const uint num_blocks = K / BLOCK_SIZE;
    const uint quant_bytes_per_row = num_blocks * BLOCK_SIZE;

    float sums[NR_V2] = {0.0f};

    for (uint b = lane; b < num_blocks; b += 32) {
        // Load input for this block: 32 elements → 8 half4 vectors
        uint inp_start = b * BLOCK_SIZE;
        device const float4* inp_f4 = (device const float4*)(input + inp_start);
        half4 x[8];
        for (uint i = 0; i < 8; i++) {
            x[i] = half4(inp_f4[i]);
        }

        // Process each row
        for (uint r = 0; r < NR_V2; r++) {
            uint neuron = row_base + r;
            if (neuron >= params.N) break;

            float scale = weight_scales[neuron * num_blocks + b];
            device const uchar* block_ptr = weight_data + neuron * quant_bytes_per_row + b * BLOCK_SIZE;

            // Load 32 int8 weights as 8 char4 → convert to 8 half4
            device const uint4* w_ptr = (device const uint4*)block_ptr;
            uint4 w0 = w_ptr[0];
            uint4 w1 = w_ptr[1];

            half4 wh[8];
            wh[0] = half4(as_type<char4>(w0.x));
            wh[1] = half4(as_type<char4>(w0.y));
            wh[2] = half4(as_type<char4>(w0.z));
            wh[3] = half4(as_type<char4>(w0.w));
            wh[4] = half4(as_type<char4>(w1.x));
            wh[5] = half4(as_type<char4>(w1.y));
            wh[6] = half4(as_type<char4>(w1.z));
            wh[7] = half4(as_type<char4>(w1.w));

            // 8 dot products of half4 = 32 multiply-accumulates
            half block_sum = 0.0h;
            for (uint i = 0; i < 8; i++) {
                block_sum += dot(x[i], wh[i]);
            }
            sums[r] += float(block_sum) * scale;
        }
    }

    // Reduce across lanes
    for (uint r = 0; r < NR_V2; r++) {
        float result = simd_sum(sums[r]);
        if (lane == 0) {
            uint neuron = row_base + r;
            if (neuron < params.N) {
                output[neuron] = result;
            }
        }
    }
}

// --------------------------------------------------------------------------
// Q8_0 GEMV v2 — f16 output
// --------------------------------------------------------------------------
kernel void gemv_q8_simd_v2_f16(
    device const half*   input         [[buffer(0)]],
    device const uchar*  weight_data   [[buffer(1)]],
    device const float*  weight_scales [[buffer(2)]],
    device half*         output        [[buffer(3)]],
    constant QuantizedLinearParams& params [[buffer(4)]],
    uint group_id  [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]])
{
    const uint simd_idx = tid / 32;
    const uint lane = tid % 32;

    const uint row_base = group_id * ROWS_PER_TG_V2 + simd_idx * NR_V2;

    const uint K = params.K;
    const uint BLOCK_SIZE = 32;
    const uint num_blocks = K / BLOCK_SIZE;
    const uint quant_bytes_per_row = num_blocks * BLOCK_SIZE;

    float sums[NR_V2] = {0.0f};

    for (uint b = lane; b < num_blocks; b += 32) {
        uint inp_start = b * BLOCK_SIZE;
        device const half4* inp_h4 = (device const half4*)(input + inp_start);
        half4 x[8];
        for (uint i = 0; i < 8; i++) {
            x[i] = inp_h4[i];
        }

        for (uint r = 0; r < NR_V2; r++) {
            uint neuron = row_base + r;
            if (neuron >= params.N) break;

            float scale = weight_scales[neuron * num_blocks + b];
            device const uchar* block_ptr = weight_data + neuron * quant_bytes_per_row + b * BLOCK_SIZE;

            device const uint4* w_ptr = (device const uint4*)block_ptr;
            uint4 w0 = w_ptr[0];
            uint4 w1 = w_ptr[1];

            half4 wh[8];
            wh[0] = half4(as_type<char4>(w0.x));
            wh[1] = half4(as_type<char4>(w0.y));
            wh[2] = half4(as_type<char4>(w0.z));
            wh[3] = half4(as_type<char4>(w0.w));
            wh[4] = half4(as_type<char4>(w1.x));
            wh[5] = half4(as_type<char4>(w1.y));
            wh[6] = half4(as_type<char4>(w1.z));
            wh[7] = half4(as_type<char4>(w1.w));

            half block_sum = 0.0h;
            for (uint i = 0; i < 8; i++) {
                block_sum += dot(x[i], wh[i]);
            }
            sums[r] += float(block_sum) * scale;
        }
    }

    for (uint r = 0; r < NR_V2; r++) {
        float result = simd_sum(sums[r]);
        if (lane == 0) {
            uint neuron = row_base + r;
            if (neuron < params.N) {
                output[neuron] = half(result);
            }
        }
    }
}

// --------------------------------------------------------------------------
// Q4_0 GEMV v2 (vectorized dot product) — f32 output
//
// Q4_0 block: 32 elements in 16 packed bytes.
// byte[j]: lo nibble = element j, hi nibble = element j+16. Both offset -8.
//
// We unpack each byte into two half4 vectors (lo and hi halves) and
// use dot(half4, half4) for vectorized accumulation.
// --------------------------------------------------------------------------
kernel void gemv_q4_simd_v2_f32(
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

    const uint row_base = group_id * ROWS_PER_TG_V2 + simd_idx * NR_V2;

    const uint K = params.K;
    const uint BLOCK_SIZE = 32;
    const uint HALF_BLOCK = 16;
    const uint num_blocks = K / BLOCK_SIZE;
    const uint packed_bytes_per_row = num_blocks * HALF_BLOCK;

    float sums[NR_V2] = {0.0f};

    for (uint b = lane; b < num_blocks; b += 32) {
        uint inp_start = b * BLOCK_SIZE;
        // Load input: lo half (elements 0..15) and hi half (elements 16..31)
        device const float4* lo_f4 = (device const float4*)(input + inp_start);
        device const float4* hi_f4 = (device const float4*)(input + inp_start + 16);
        half4 xl[4], xh[4];
        for (uint i = 0; i < 4; i++) {
            xl[i] = half4(lo_f4[i]);
            xh[i] = half4(hi_f4[i]);
        }

        for (uint r = 0; r < NR_V2; r++) {
            uint neuron = row_base + r;
            if (neuron >= params.N) break;

            float scale = weight_scales[neuron * num_blocks + b];
            device const uchar* block_ptr = weight_data + neuron * packed_bytes_per_row + b * HALF_BLOCK;

            // Load 16 packed bytes → unpack to lo (16 elements) and hi (16 elements)
            uint4 packed = ((device const uint4*)block_ptr)[0];
            uchar4 p[4];
            p[0] = as_type<uchar4>(packed.x);
            p[1] = as_type<uchar4>(packed.y);
            p[2] = as_type<uchar4>(packed.z);
            p[3] = as_type<uchar4>(packed.w);

            half block_sum = 0.0h;
            for (uint i = 0; i < 4; i++) {
                // Lo nibbles → elements i*4 .. i*4+3
                half4 lo_w = half4(
                    half(int(p[i].x & 0x0F) - 8),
                    half(int(p[i].y & 0x0F) - 8),
                    half(int(p[i].z & 0x0F) - 8),
                    half(int(p[i].w & 0x0F) - 8)
                );
                // Hi nibbles → elements i*4+16 .. i*4+19
                half4 hi_w = half4(
                    half(int(p[i].x >> 4) - 8),
                    half(int(p[i].y >> 4) - 8),
                    half(int(p[i].z >> 4) - 8),
                    half(int(p[i].w >> 4) - 8)
                );
                block_sum += dot(xl[i], lo_w);
                block_sum += dot(xh[i], hi_w);
            }
            sums[r] += float(block_sum) * scale;
        }
    }

    for (uint r = 0; r < NR_V2; r++) {
        float result = simd_sum(sums[r]);
        if (lane == 0) {
            uint neuron = row_base + r;
            if (neuron < params.N) {
                output[neuron] = result;
            }
        }
    }
}

// --------------------------------------------------------------------------
// Q4_0 GEMV v2 — f16 output
// --------------------------------------------------------------------------
kernel void gemv_q4_simd_v2_f16(
    device const half*   input         [[buffer(0)]],
    device const uchar*  weight_data   [[buffer(1)]],
    device const float*  weight_scales [[buffer(2)]],
    device half*         output        [[buffer(3)]],
    constant QuantizedLinearParams& params [[buffer(4)]],
    uint group_id  [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]])
{
    const uint simd_idx = tid / 32;
    const uint lane = tid % 32;

    const uint row_base = group_id * ROWS_PER_TG_V2 + simd_idx * NR_V2;

    const uint K = params.K;
    const uint BLOCK_SIZE = 32;
    const uint HALF_BLOCK = 16;
    const uint num_blocks = K / BLOCK_SIZE;
    const uint packed_bytes_per_row = num_blocks * HALF_BLOCK;

    float sums[NR_V2] = {0.0f};

    for (uint b = lane; b < num_blocks; b += 32) {
        uint inp_start = b * BLOCK_SIZE;
        device const half4* lo_h4 = (device const half4*)(input + inp_start);
        device const half4* hi_h4 = (device const half4*)(input + inp_start + 16);
        half4 xl[4], xh[4];
        for (uint i = 0; i < 4; i++) {
            xl[i] = lo_h4[i];
            xh[i] = hi_h4[i];
        }

        for (uint r = 0; r < NR_V2; r++) {
            uint neuron = row_base + r;
            if (neuron >= params.N) break;

            float scale = weight_scales[neuron * num_blocks + b];
            device const uchar* block_ptr = weight_data + neuron * packed_bytes_per_row + b * HALF_BLOCK;

            uint4 packed = ((device const uint4*)block_ptr)[0];
            uchar4 p[4];
            p[0] = as_type<uchar4>(packed.x);
            p[1] = as_type<uchar4>(packed.y);
            p[2] = as_type<uchar4>(packed.z);
            p[3] = as_type<uchar4>(packed.w);

            half block_sum = 0.0h;
            for (uint i = 0; i < 4; i++) {
                half4 lo_w = half4(
                    half(int(p[i].x & 0x0F) - 8),
                    half(int(p[i].y & 0x0F) - 8),
                    half(int(p[i].z & 0x0F) - 8),
                    half(int(p[i].w & 0x0F) - 8)
                );
                half4 hi_w = half4(
                    half(int(p[i].x >> 4) - 8),
                    half(int(p[i].y >> 4) - 8),
                    half(int(p[i].z >> 4) - 8),
                    half(int(p[i].w >> 4) - 8)
                );
                block_sum += dot(xl[i], lo_w);
                block_sum += dot(xh[i], hi_w);
            }
            sums[r] += float(block_sum) * scale;
        }
    }

    for (uint r = 0; r < NR_V2; r++) {
        float result = simd_sum(sums[r]);
        if (lane == 0) {
            uint neuron = row_base + r;
            if (neuron < params.N) {
                output[neuron] = half(result);
            }
        }
    }
}

// ==========================================================================
// Fused SwiGLU + GEMV kernels
//
// These read from a gate_up buffer of 2*K elements (gate in [0..K), up in
// [K..2K)) and compute silu(gate[d]) * up[d] on-the-fly as the effective
// input for the down projection GEMV.  This eliminates the separate SwiGLU
// dispatch.
//
// Params: N = out_features of down_proj, K = intermediate_size (= in_features
// of down_proj).  The gate_up buffer has 2*K elements.
// ==========================================================================

// ==========================================================================
// Dequantize Q8_0 / Q4_0 weights to F32 for batched prefill GEMM.
//
// Grid: 2D (width=K, height=N). Each thread writes one F32 element of the
// dense (N, K) weight matrix. The result is passed directly to
// linear_dense_f32 as weight_t (same layout expected by that kernel).
// ==========================================================================

kernel void dequantize_q8_to_f32(
    device const uchar* data        [[buffer(0)]],
    device const float* scales      [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant QuantizedLinearParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    if (col >= params.K || row >= params.N) return;

    uint num_blocks = params.K / 32;
    float scale = scales[row * num_blocks + col / 32];
    float q = float(as_type<char>(data[row * params.K + col]));
    output[row * params.K + col] = q * scale;
}

kernel void dequantize_q4_to_f32(
    device const uchar* data        [[buffer(0)]],
    device const float* scales      [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant QuantizedLinearParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    if (col >= params.K || row >= params.N) return;

    uint block      = col / 32;
    uint j_in_block = col % 32;
    uint packed_bytes_per_row = params.K / 2;

    uchar byte_val;
    float q;
    if (j_in_block < 16) {
        byte_val = data[row * packed_bytes_per_row + block * 16 + j_in_block];
        q = float(int(byte_val & 0x0F) - 8);
    } else {
        byte_val = data[row * packed_bytes_per_row + block * 16 + (j_in_block - 16)];
        q = float(int(byte_val >> 4) - 8);
    }
    float scale = scales[row * (params.K / 32) + block];
    output[row * params.K + col] = q * scale;
}

// Q4_1: same nibble layout as Q4_0 but unsigned (0..15) with per-block min.
// Formula: output = nibble * scale + min.
kernel void dequantize_q4_1_to_f32(
    device const uchar* data        [[buffer(0)]],
    device const float* scales      [[buffer(1)]],
    device const float* mins        [[buffer(2)]],
    device float*       output      [[buffer(3)]],
    constant QuantizedLinearParams& params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    if (col >= params.K || row >= params.N) return;

    uint block      = col / 32;
    uint j_in_block = col % 32;
    uint packed_bytes_per_row = params.K / 2;

    uchar byte_val;
    float q;
    if (j_in_block < 16) {
        byte_val = data[row * packed_bytes_per_row + block * 16 + j_in_block];
        q = float(byte_val & 0x0F);
    } else {
        byte_val = data[row * packed_bytes_per_row + block * 16 + (j_in_block - 16)];
        q = float(byte_val >> 4);
    }
    uint block_idx = row * (params.K / 32) + block;
    float scale = scales[block_idx];
    float min_val = mins[block_idx];
    output[row * params.K + col] = q * scale + min_val;
}

// ==========================================================================
// GGUF native block format kernels — interleaved [f16 scale | quant data]
//
// These kernels read weights in their native GGUF on-disk block layout,
// avoiding the separate-scales-buffer double cache-miss that the v2 kernels
// suffer from.  When loading a block, the F16 scale and the quantized bytes
// come from the *same* cache line.
//
// Q8_0 block: 34 bytes — [f16 d (2B)] [int8 qs[32] (32B)]
// Q4_0 block: 18 bytes — [f16 d (2B)] [uint8 qs[16] (16B)]
//
// Threadgroup configs (matching llama.cpp):
//   Q8_0: NSG=4 SIMD-groups × NR=2 rows = 8 rows/TG, 128 threads/TG
//   Q4_0: NSG=2 SIMD-groups × NR=4 rows = 8 rows/TG, 64 threads/TG
//
// Grid: (ceil(N / ROWS_PER_TG), 1, 1), threads_per_group = (THREADS_PER_TG, 1, 1).
// ==========================================================================

// ==========================================================================
// Dequantize from native GGUF block format to F32
// Used for M>1 (prefill) when weights are in blocks-only storage.
// Grid: (K, N) — one thread per output element.
// ==========================================================================

kernel void dequantize_q8_blocks_to_f32(
    device const uchar*             blocks [[buffer(0)]],
    device float*                   output [[buffer(1)]],
    constant QuantizedLinearParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    if (col >= params.K || row >= params.N) return;

    uint block_idx = col / 32;
    uint elem = col % 32;
    uint nb = params.K / 32;
    device const uchar* blk = blocks + (row * nb + block_idx) * 34u;
    float scale = float(as_type<half>(*(device const ushort*)blk));
    float q = float(as_type<char>(blk[2 + elem]));
    output[row * params.K + col] = q * scale;
}

kernel void dequantize_q4_blocks_to_f32(
    device const uchar*             blocks [[buffer(0)]],
    device float*                   output [[buffer(1)]],
    constant QuantizedLinearParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    if (col >= params.K || row >= params.N) return;

    uint block_idx = col / 32;
    uint j = col % 32;
    uint nb = params.K / 32;
    device const uchar* blk = blocks + (row * nb + block_idx) * 18u;
    float scale = float(as_type<half>(*(device const ushort*)blk));
    uchar byte_val;
    float q;
    if (j < 16) {
        byte_val = blk[2 + j];
        q = float(int(byte_val & 0x0F) - 8);
    } else {
        byte_val = blk[2 + (j - 16)];
        q = float(int(byte_val >> 4) - 8);
    }
    output[row * params.K + col] = q * scale;
}

constant constexpr uint Q8_BLOCK_BYTES = 34;  // 2 (f16) + 32 (int8)
constant constexpr uint Q4_BLOCK_BYTES = 18;  // 2 (f16) + 16 (nibbles)

// Q8_0 config (matching llama.cpp N_R0_Q8_0=2, N_SG_Q8_0=4)
constant constexpr uint Q8B_NR  = 2;
constant constexpr uint Q8B_NSG = 4;
constant constexpr uint Q8B_ROWS_PER_TG    = Q8B_NR * Q8B_NSG;   // 8
constant constexpr uint Q8B_THREADS_PER_TG = Q8B_NSG * 32;        // 128

// Q4_0 config (matching llama.cpp N_R0_Q4_0=4, N_SG_Q4_0=2)
constant constexpr uint Q4B_NR  = 4;
constant constexpr uint Q4B_NSG = 2;
constant constexpr uint Q4B_ROWS_PER_TG    = Q4B_NR * Q4B_NSG;   // 8
constant constexpr uint Q4B_THREADS_PER_TG = Q4B_NSG * 32;        // 64

// --------------------------------------------------------------------------
// Q8_0 blocks GEMV — f16 input, f16 output
// --------------------------------------------------------------------------
kernel void gemv_q8_blocks_f16(
    device const half*              input  [[buffer(0)]],
    device const uchar*             blocks [[buffer(1)]],
    device half*                    output [[buffer(2)]],
    constant QuantizedLinearParams& params [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]])
{
    const uint simd_idx = tid / 32;
    const uint lane     = tid % 32;
    const uint row_base = group_id * Q8B_ROWS_PER_TG + simd_idx * Q8B_NR;

    const uint nb           = params.K / 32;
    const uint block_stride = nb * Q8_BLOCK_BYTES;

    float sums[Q8B_NR] = {0.0f};

    for (uint b = lane; b < nb; b += 32) {
        // Load input block (32 × f16 = 64 bytes) — shared across all rows
        device const half4* inp = (device const half4*)(input + b * 32);
        half4 x[8];
        for (uint i = 0; i < 8; i++) x[i] = inp[i];

        for (uint r = 0; r < Q8B_NR; r++) {
            uint neuron = row_base + r;
            if (neuron >= params.N) continue;

            // Single contiguous read: [f16 scale | int8[32]]
            device const uchar* blk = blocks + neuron * block_stride + b * Q8_BLOCK_BYTES;
            float scale = float(as_type<half>(*(device const ushort*)blk));
            device const uint4* w_ptr = (device const uint4*)(blk + 2);
            uint4 w0 = w_ptr[0];
            uint4 w1 = w_ptr[1];

            half4 wh[8];
            wh[0] = half4(as_type<char4>(w0.x)); wh[1] = half4(as_type<char4>(w0.y));
            wh[2] = half4(as_type<char4>(w0.z)); wh[3] = half4(as_type<char4>(w0.w));
            wh[4] = half4(as_type<char4>(w1.x)); wh[5] = half4(as_type<char4>(w1.y));
            wh[6] = half4(as_type<char4>(w1.z)); wh[7] = half4(as_type<char4>(w1.w));

            half block_sum = 0.0h;
            for (uint i = 0; i < 8; i++) block_sum += dot(x[i], wh[i]);
            sums[r] += float(block_sum) * scale;
        }
    }

    for (uint r = 0; r < Q8B_NR; r++) {
        float result = simd_sum(sums[r]);
        if (lane == 0) {
            uint neuron = row_base + r;
            if (neuron < params.N) output[neuron] = half(result);
        }
    }
}

// --------------------------------------------------------------------------
// Q8_0 blocks GEMV — f32 input, f32 output
// --------------------------------------------------------------------------
kernel void gemv_q8_blocks_f32(
    device const float*             input  [[buffer(0)]],
    device const uchar*             blocks [[buffer(1)]],
    device float*                   output [[buffer(2)]],
    constant QuantizedLinearParams& params [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]])
{
    const uint simd_idx = tid / 32;
    const uint lane     = tid % 32;
    const uint row_base = group_id * Q8B_ROWS_PER_TG + simd_idx * Q8B_NR;

    const uint nb           = params.K / 32;
    const uint block_stride = nb * Q8_BLOCK_BYTES;

    float sums[Q8B_NR] = {0.0f};

    for (uint b = lane; b < nb; b += 32) {
        device const float4* inp = (device const float4*)(input + b * 32);
        half4 x[8];
        for (uint i = 0; i < 8; i++) x[i] = half4(inp[i]);

        for (uint r = 0; r < Q8B_NR; r++) {
            uint neuron = row_base + r;
            if (neuron >= params.N) continue;

            device const uchar* blk = blocks + neuron * block_stride + b * Q8_BLOCK_BYTES;
            float scale = float(as_type<half>(*(device const ushort*)blk));
            device const uint4* w_ptr = (device const uint4*)(blk + 2);
            uint4 w0 = w_ptr[0];
            uint4 w1 = w_ptr[1];

            half4 wh[8];
            wh[0] = half4(as_type<char4>(w0.x)); wh[1] = half4(as_type<char4>(w0.y));
            wh[2] = half4(as_type<char4>(w0.z)); wh[3] = half4(as_type<char4>(w0.w));
            wh[4] = half4(as_type<char4>(w1.x)); wh[5] = half4(as_type<char4>(w1.y));
            wh[6] = half4(as_type<char4>(w1.z)); wh[7] = half4(as_type<char4>(w1.w));

            half block_sum = 0.0h;
            for (uint i = 0; i < 8; i++) block_sum += dot(x[i], wh[i]);
            sums[r] += float(block_sum) * scale;
        }
    }

    for (uint r = 0; r < Q8B_NR; r++) {
        float result = simd_sum(sums[r]);
        if (lane == 0) {
            uint neuron = row_base + r;
            if (neuron < params.N) output[neuron] = result;
        }
    }
}

// --------------------------------------------------------------------------
// Q4_0 blocks GEMV — f16 input, f16 output
// Q4_0 block: [f16 d (2B)] [nibbles[16] (16B)], total 18 bytes
// byte j: lo nibble = element j, hi nibble = element j+16, both offset -8
// --------------------------------------------------------------------------
kernel void gemv_q4_blocks_f16(
    device const half*              input  [[buffer(0)]],
    device const uchar*             blocks [[buffer(1)]],
    device half*                    output [[buffer(2)]],
    constant QuantizedLinearParams& params [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]])
{
    const uint simd_idx = tid / 32;
    const uint lane     = tid % 32;
    const uint row_base = group_id * Q4B_ROWS_PER_TG + simd_idx * Q4B_NR;

    const uint nb           = params.K / 32;
    const uint block_stride = nb * Q4_BLOCK_BYTES;

    float sums[Q4B_NR] = {0.0f};

    for (uint b = lane; b < nb; b += 32) {
        // Load input lo and hi halves (32 × f16 total)
        device const half4* lo_h = (device const half4*)(input + b * 32);
        device const half4* hi_h = (device const half4*)(input + b * 32 + 16);
        half4 xl[4], xh[4];
        for (uint i = 0; i < 4; i++) { xl[i] = lo_h[i]; xh[i] = hi_h[i]; }

        for (uint r = 0; r < Q4B_NR; r++) {
            uint neuron = row_base + r;
            if (neuron >= params.N) continue;

            // Single read: [f16 scale | nibbles[16]]
            device const uchar* blk = blocks + neuron * block_stride + b * Q4_BLOCK_BYTES;
            float scale = float(as_type<half>(*(device const ushort*)blk));
            uint4 packed = *(device const uint4*)(blk + 2);

            uchar4 p[4];
            p[0] = as_type<uchar4>(packed.x); p[1] = as_type<uchar4>(packed.y);
            p[2] = as_type<uchar4>(packed.z); p[3] = as_type<uchar4>(packed.w);

            half block_sum = 0.0h;
            for (uint i = 0; i < 4; i++) {
                half4 lo_w = half4(half(int(p[i].x & 0x0F)-8), half(int(p[i].y & 0x0F)-8),
                                   half(int(p[i].z & 0x0F)-8), half(int(p[i].w & 0x0F)-8));
                half4 hi_w = half4(half(int(p[i].x >> 4)-8),   half(int(p[i].y >> 4)-8),
                                   half(int(p[i].z >> 4)-8),   half(int(p[i].w >> 4)-8));
                block_sum += dot(xl[i], lo_w) + dot(xh[i], hi_w);
            }
            sums[r] += float(block_sum) * scale;
        }
    }

    for (uint r = 0; r < Q4B_NR; r++) {
        float result = simd_sum(sums[r]);
        if (lane == 0) {
            uint neuron = row_base + r;
            if (neuron < params.N) output[neuron] = half(result);
        }
    }
}

// --------------------------------------------------------------------------
// Q4_0 blocks GEMV — f32 input, f32 output
// --------------------------------------------------------------------------
kernel void gemv_q4_blocks_f32(
    device const float*             input  [[buffer(0)]],
    device const uchar*             blocks [[buffer(1)]],
    device float*                   output [[buffer(2)]],
    constant QuantizedLinearParams& params [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]])
{
    const uint simd_idx = tid / 32;
    const uint lane     = tid % 32;
    const uint row_base = group_id * Q4B_ROWS_PER_TG + simd_idx * Q4B_NR;

    const uint nb           = params.K / 32;
    const uint block_stride = nb * Q4_BLOCK_BYTES;

    float sums[Q4B_NR] = {0.0f};

    for (uint b = lane; b < nb; b += 32) {
        device const float4* lo_f = (device const float4*)(input + b * 32);
        device const float4* hi_f = (device const float4*)(input + b * 32 + 16);
        half4 xl[4], xh[4];
        for (uint i = 0; i < 4; i++) { xl[i] = half4(lo_f[i]); xh[i] = half4(hi_f[i]); }

        for (uint r = 0; r < Q4B_NR; r++) {
            uint neuron = row_base + r;
            if (neuron >= params.N) continue;

            device const uchar* blk = blocks + neuron * block_stride + b * Q4_BLOCK_BYTES;
            float scale = float(as_type<half>(*(device const ushort*)blk));
            uint4 packed = *(device const uint4*)(blk + 2);

            uchar4 p[4];
            p[0] = as_type<uchar4>(packed.x); p[1] = as_type<uchar4>(packed.y);
            p[2] = as_type<uchar4>(packed.z); p[3] = as_type<uchar4>(packed.w);

            half block_sum = 0.0h;
            for (uint i = 0; i < 4; i++) {
                half4 lo_w = half4(half(int(p[i].x & 0x0F)-8), half(int(p[i].y & 0x0F)-8),
                                   half(int(p[i].z & 0x0F)-8), half(int(p[i].w & 0x0F)-8));
                half4 hi_w = half4(half(int(p[i].x >> 4)-8),   half(int(p[i].y >> 4)-8),
                                   half(int(p[i].z >> 4)-8),   half(int(p[i].w >> 4)-8));
                block_sum += dot(xl[i], lo_w) + dot(xh[i], hi_w);
            }
            sums[r] += float(block_sum) * scale;
        }
    }

    for (uint r = 0; r < Q4B_NR; r++) {
        float result = simd_sum(sums[r]);
        if (lane == 0) {
            uint neuron = row_base + r;
            if (neuron < params.N) output[neuron] = result;
        }
    }
}

static inline half silu_h(half x) {
    return x / (half(1.0h) + exp(-x));
}

// --------------------------------------------------------------------------
// Fused SwiGLU + Q8_0 blocks GEMV — f16 input, f16 output
// --------------------------------------------------------------------------
kernel void gemv_swiglu_q8_blocks_f16(
    device const half*              gate_up [[buffer(0)]],
    device const uchar*             blocks  [[buffer(1)]],
    device half*                    output  [[buffer(2)]],
    constant QuantizedLinearParams& params  [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]])
{
    const uint simd_idx = tid / 32;
    const uint lane     = tid % 32;
    const uint row_base = group_id * Q8B_ROWS_PER_TG + simd_idx * Q8B_NR;

    const uint K           = params.K;
    const uint nb          = K / 32;
    const uint block_stride = nb * Q8_BLOCK_BYTES;

    float sums[Q8B_NR] = {0.0f};

    for (uint b = lane; b < nb; b += 32) {
        uint inp_start = b * 32;
        device const half4* gate_h = (device const half4*)(gate_up + inp_start);
        device const half4* up_h   = (device const half4*)(gate_up + K + inp_start);
        half4 x[8];
        for (uint i = 0; i < 8; i++) {
            half4 g = gate_h[i], u = up_h[i];
            x[i] = half4(silu_h(g.x)*u.x, silu_h(g.y)*u.y,
                         silu_h(g.z)*u.z, silu_h(g.w)*u.w);
        }

        for (uint r = 0; r < Q8B_NR; r++) {
            uint neuron = row_base + r;
            if (neuron >= params.N) continue;

            device const uchar* blk = blocks + neuron * block_stride + b * Q8_BLOCK_BYTES;
            float scale = float(as_type<half>(*(device const ushort*)blk));
            device const uint4* w_ptr = (device const uint4*)(blk + 2);
            uint4 w0 = w_ptr[0], w1 = w_ptr[1];

            half4 wh[8];
            wh[0] = half4(as_type<char4>(w0.x)); wh[1] = half4(as_type<char4>(w0.y));
            wh[2] = half4(as_type<char4>(w0.z)); wh[3] = half4(as_type<char4>(w0.w));
            wh[4] = half4(as_type<char4>(w1.x)); wh[5] = half4(as_type<char4>(w1.y));
            wh[6] = half4(as_type<char4>(w1.z)); wh[7] = half4(as_type<char4>(w1.w));

            half block_sum = 0.0h;
            for (uint i = 0; i < 8; i++) block_sum += dot(x[i], wh[i]);
            sums[r] += float(block_sum) * scale;
        }
    }

    for (uint r = 0; r < Q8B_NR; r++) {
        float result = simd_sum(sums[r]);
        if (lane == 0) {
            uint neuron = row_base + r;
            if (neuron < params.N) output[neuron] = half(result);
        }
    }
}

// --------------------------------------------------------------------------
// Fused SwiGLU + Q4_0 blocks GEMV — f16 input, f16 output
// --------------------------------------------------------------------------
kernel void gemv_swiglu_q4_blocks_f16(
    device const half*              gate_up [[buffer(0)]],
    device const uchar*             blocks  [[buffer(1)]],
    device half*                    output  [[buffer(2)]],
    constant QuantizedLinearParams& params  [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]])
{
    const uint simd_idx = tid / 32;
    const uint lane     = tid % 32;
    const uint row_base = group_id * Q4B_ROWS_PER_TG + simd_idx * Q4B_NR;

    const uint K            = params.K;
    const uint nb           = K / 32;
    const uint block_stride = nb * Q4_BLOCK_BYTES;

    float sums[Q4B_NR] = {0.0f};

    for (uint b = lane; b < nb; b += 32) {
        uint inp_start = b * 32;
        device const half4* gate_lo = (device const half4*)(gate_up + inp_start);
        device const half4* gate_hi = (device const half4*)(gate_up + inp_start + 16);
        device const half4* up_lo   = (device const half4*)(gate_up + K + inp_start);
        device const half4* up_hi   = (device const half4*)(gate_up + K + inp_start + 16);
        half4 xl[4], xh[4];
        for (uint i = 0; i < 4; i++) {
            half4 gl = gate_lo[i], ul = up_lo[i];
            xl[i] = half4(silu_h(gl.x)*ul.x, silu_h(gl.y)*ul.y,
                          silu_h(gl.z)*ul.z, silu_h(gl.w)*ul.w);
            half4 gh = gate_hi[i], uh = up_hi[i];
            xh[i] = half4(silu_h(gh.x)*uh.x, silu_h(gh.y)*uh.y,
                          silu_h(gh.z)*uh.z, silu_h(gh.w)*uh.w);
        }

        for (uint r = 0; r < Q4B_NR; r++) {
            uint neuron = row_base + r;
            if (neuron >= params.N) continue;

            device const uchar* blk = blocks + neuron * block_stride + b * Q4_BLOCK_BYTES;
            float scale = float(as_type<half>(*(device const ushort*)blk));
            uint4 packed = *(device const uint4*)(blk + 2);

            uchar4 p[4];
            p[0] = as_type<uchar4>(packed.x); p[1] = as_type<uchar4>(packed.y);
            p[2] = as_type<uchar4>(packed.z); p[3] = as_type<uchar4>(packed.w);

            half block_sum = 0.0h;
            for (uint i = 0; i < 4; i++) {
                half4 lo_w = half4(half(int(p[i].x & 0x0F)-8), half(int(p[i].y & 0x0F)-8),
                                   half(int(p[i].z & 0x0F)-8), half(int(p[i].w & 0x0F)-8));
                half4 hi_w = half4(half(int(p[i].x >> 4)-8),   half(int(p[i].y >> 4)-8),
                                   half(int(p[i].z >> 4)-8),   half(int(p[i].w >> 4)-8));
                block_sum += dot(xl[i], lo_w) + dot(xh[i], hi_w);
            }
            sums[r] += float(block_sum) * scale;
        }
    }

    for (uint r = 0; r < Q4B_NR; r++) {
        float result = simd_sum(sums[r]);
        if (lane == 0) {
            uint neuron = row_base + r;
            if (neuron < params.N) output[neuron] = half(result);
        }
    }
}

// --------------------------------------------------------------------------
// Fused SwiGLU + Q8_0 GEMV v2 — f16 output
// --------------------------------------------------------------------------
kernel void gemv_swiglu_q8_simd_v2_f16(
    device const half*   gate_up       [[buffer(0)]],
    device const uchar*  weight_data   [[buffer(1)]],
    device const float*  weight_scales [[buffer(2)]],
    device half*         output        [[buffer(3)]],
    constant QuantizedLinearParams& params [[buffer(4)]],
    uint group_id  [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]])
{
    const uint simd_idx = tid / 32;
    const uint lane = tid % 32;

    const uint row_base = group_id * ROWS_PER_TG_V2 + simd_idx * NR_V2;

    const uint K = params.K;
    const uint BLOCK_SIZE = 32;
    const uint num_blocks = K / BLOCK_SIZE;
    const uint quant_bytes_per_row = num_blocks * BLOCK_SIZE;

    float sums[NR_V2] = {0.0f};

    for (uint b = lane; b < num_blocks; b += 32) {
        uint inp_start = b * BLOCK_SIZE;
        // Load gate and up values, compute SwiGLU on-the-fly
        device const half4* gate_h4 = (device const half4*)(gate_up + inp_start);
        device const half4* up_h4   = (device const half4*)(gate_up + K + inp_start);
        half4 x[8];
        for (uint i = 0; i < 8; i++) {
            half4 g = gate_h4[i];
            half4 u = up_h4[i];
            x[i] = half4(silu_h(g.x) * u.x, silu_h(g.y) * u.y,
                         silu_h(g.z) * u.z, silu_h(g.w) * u.w);
        }

        for (uint r = 0; r < NR_V2; r++) {
            uint neuron = row_base + r;
            if (neuron >= params.N) break;

            float scale = weight_scales[neuron * num_blocks + b];
            device const uchar* block_ptr = weight_data + neuron * quant_bytes_per_row + b * BLOCK_SIZE;

            device const uint4* w_ptr = (device const uint4*)block_ptr;
            uint4 w0 = w_ptr[0];
            uint4 w1 = w_ptr[1];

            half4 wh[8];
            wh[0] = half4(as_type<char4>(w0.x));
            wh[1] = half4(as_type<char4>(w0.y));
            wh[2] = half4(as_type<char4>(w0.z));
            wh[3] = half4(as_type<char4>(w0.w));
            wh[4] = half4(as_type<char4>(w1.x));
            wh[5] = half4(as_type<char4>(w1.y));
            wh[6] = half4(as_type<char4>(w1.z));
            wh[7] = half4(as_type<char4>(w1.w));

            half block_sum = 0.0h;
            for (uint i = 0; i < 8; i++) {
                block_sum += dot(x[i], wh[i]);
            }
            sums[r] += float(block_sum) * scale;
        }
    }

    for (uint r = 0; r < NR_V2; r++) {
        float result = simd_sum(sums[r]);
        if (lane == 0) {
            uint neuron = row_base + r;
            if (neuron < params.N) {
                output[neuron] = half(result);
            }
        }
    }
}

// --------------------------------------------------------------------------
// Fused SwiGLU + Q4_0 GEMV v2 — f16 output
// --------------------------------------------------------------------------
kernel void gemv_swiglu_q4_simd_v2_f16(
    device const half*   gate_up       [[buffer(0)]],
    device const uchar*  weight_data   [[buffer(1)]],
    device const float*  weight_scales [[buffer(2)]],
    device half*         output        [[buffer(3)]],
    constant QuantizedLinearParams& params [[buffer(4)]],
    uint group_id  [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]])
{
    const uint simd_idx = tid / 32;
    const uint lane = tid % 32;

    const uint row_base = group_id * ROWS_PER_TG_V2 + simd_idx * NR_V2;

    const uint K = params.K;
    const uint BLOCK_SIZE = 32;
    const uint HALF_BLOCK = 16;
    const uint num_blocks = K / BLOCK_SIZE;
    const uint packed_bytes_per_row = num_blocks * HALF_BLOCK;

    float sums[NR_V2] = {0.0f};

    for (uint b = lane; b < num_blocks; b += 32) {
        uint inp_start = b * BLOCK_SIZE;
        // Load gate and up, apply SwiGLU for lo/hi halves of the block
        device const half4* gate_lo = (device const half4*)(gate_up + inp_start);
        device const half4* gate_hi = (device const half4*)(gate_up + inp_start + 16);
        device const half4* up_lo   = (device const half4*)(gate_up + K + inp_start);
        device const half4* up_hi   = (device const half4*)(gate_up + K + inp_start + 16);
        half4 xl[4], xh[4];
        for (uint i = 0; i < 4; i++) {
            half4 g_lo = gate_lo[i];
            half4 u_lo = up_lo[i];
            xl[i] = half4(silu_h(g_lo.x) * u_lo.x, silu_h(g_lo.y) * u_lo.y,
                          silu_h(g_lo.z) * u_lo.z, silu_h(g_lo.w) * u_lo.w);
            half4 g_hi = gate_hi[i];
            half4 u_hi = up_hi[i];
            xh[i] = half4(silu_h(g_hi.x) * u_hi.x, silu_h(g_hi.y) * u_hi.y,
                          silu_h(g_hi.z) * u_hi.z, silu_h(g_hi.w) * u_hi.w);
        }

        for (uint r = 0; r < NR_V2; r++) {
            uint neuron = row_base + r;
            if (neuron >= params.N) break;

            float scale = weight_scales[neuron * num_blocks + b];
            device const uchar* block_ptr = weight_data + neuron * packed_bytes_per_row + b * HALF_BLOCK;

            uint4 packed = ((device const uint4*)block_ptr)[0];
            uchar4 p[4];
            p[0] = as_type<uchar4>(packed.x);
            p[1] = as_type<uchar4>(packed.y);
            p[2] = as_type<uchar4>(packed.z);
            p[3] = as_type<uchar4>(packed.w);

            half block_sum = 0.0h;
            for (uint i = 0; i < 4; i++) {
                half4 lo_w = half4(
                    half(int(p[i].x & 0x0F) - 8),
                    half(int(p[i].y & 0x0F) - 8),
                    half(int(p[i].z & 0x0F) - 8),
                    half(int(p[i].w & 0x0F) - 8)
                );
                half4 hi_w = half4(
                    half(int(p[i].x >> 4) - 8),
                    half(int(p[i].y >> 4) - 8),
                    half(int(p[i].z >> 4) - 8),
                    half(int(p[i].w >> 4) - 8)
                );
                block_sum += dot(xl[i], lo_w);
                block_sum += dot(xh[i], hi_w);
            }
            sums[r] += float(block_sum) * scale;
        }
    }

    for (uint r = 0; r < NR_V2; r++) {
        float result = simd_sum(sums[r]);
        if (lane == 0) {
            uint neuron = row_base + r;
            if (neuron < params.N) {
                output[neuron] = half(result);
            }
        }
    }
}
