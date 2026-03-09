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
