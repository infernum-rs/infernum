// Prefill attention using cuBLAS batched GEMMs + fused causal softmax.
//
// For seq_len > FLASH_THRESHOLD, the 4-query-row Flash Attention kernel
// reads K/V (seq/4) times redundantly.  For moderate seq_len (512–4096)
// where the full attention matrix fits in L2, batched GEMM is faster:
//
//   1. Expand K/V: (kvh, seq, hd) → (nh, seq, hd) for GQA repetition.
//   2. Q×K^T:  (nh, seq, hd) @ (nh, hd, seq) → (nh, seq, seq) F32.
//   3. Causal softmax: in-place on (nh, seq, seq), scale by 1/sqrt(hd).
//   4. A×V:    (nh, seq, seq) @ (nh, seq, hd) → (nh, seq, hd) BF16.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Expand KV: repeat each kv-head (nh/kvh) times along the head axis.
// Input:  (kvh, seq, hd) BF16
// Output: (nh,  seq, hd) BF16
// ---------------------------------------------------------------------------
extern "C" __global__ void expand_kv_bf16(
    __nv_bfloat16*       __restrict__ out,   // (nh,  seq, hd)
    const __nv_bfloat16* __restrict__ in,    // (kvh, seq, hd)
    int nh, int kvh, int seq, int hd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nh * seq * hd;
    if (idx >= total) return;
    int h = idx / (seq * hd);
    int rest = idx % (seq * hd);
    int kv_h = h * kvh / nh;
    out[idx] = in[kv_h * seq * hd + rest];
}

// ---------------------------------------------------------------------------
// Causal softmax in-place on (nh, seq, seq) F32 attention scores.
// Each warp handles one (head, query-row) pair.
// Applies scale = 1/sqrt(head_dim) before softmax, zeros positions > qpos.
// ---------------------------------------------------------------------------
extern "C" __global__ void causal_softmax_f32(
    float* __restrict__ attn,   // (nh, seq, seq) row-major
    int nh, int seq, float scale)
{
    // One warp per (head, query) pair.
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;
    int total_warps = nh * seq;
    if (warp_id >= total_warps) return;

    int head = warp_id / seq;
    int qi   = warp_id % seq;
    float* row = attn + (head * seq + qi) * seq;

    // Each lane handles multiple keys via strided access.
    // Valid positions: k = 0 .. qi (causal mask).
    float max_val = -1e30f;
    for (int k = lane; k < seq; k += 32) {
        float v = (k <= qi) ? row[k] * scale : -1e30f;
        max_val = fmaxf(max_val, v);
    }
    // Warp reduce max
    for (int delta = 16; delta >= 1; delta >>= 1)
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, delta));

    float sum = 0.f;
    for (int k = lane; k < seq; k += 32) {
        float v = (k <= qi) ? expf(row[k] * scale - max_val) : 0.f;
        row[k] = v;          // write exp(score) (masked = 0)
        sum += v;
    }
    // Warp reduce sum
    for (int delta = 16; delta >= 1; delta >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, delta);

    float inv = (sum > 0.f) ? 1.f / sum : 0.f;
    for (int k = lane; k < seq; k += 32)
        row[k] *= inv;
}

// ---------------------------------------------------------------------------
// Cast BF16 matrix to F32 in-place.
// Used to upcast V from BF16 to F32 for the Attn×V GEMM.
// ---------------------------------------------------------------------------
extern "C" __global__ void cast_bf16_mat_to_f32(
    float*               __restrict__ out,   // (n,)
    const __nv_bfloat16* __restrict__ in,    // (n,)
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __bfloat162float(in[idx]);
}

// ---------------------------------------------------------------------------
// Cast F32 matrix to BF16 in-place.
// Used to downcast the Attn×V result from F32 to BF16.
// ---------------------------------------------------------------------------
extern "C" __global__ void cast_f32_mat_to_bf16(
    __nv_bfloat16* __restrict__ out,   // (n,)
    const float*   __restrict__ in,    // (n,)
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2bfloat16(in[idx]);
}

// ---------------------------------------------------------------------------
// Scatter (seq_q, head_dim) F32 → (seq_q, num_heads, head_dim) BF16
// for a specific head index. Enables writing per-head Attn×V results
// back into the interleaved output tensor without a full transpose.
// ---------------------------------------------------------------------------
extern "C" __global__ void scatter_head_f32_to_bf16(
    __nv_bfloat16* __restrict__ out,    // (seq_q, num_heads, head_dim)
    const float*   __restrict__ in,     // (seq_q, head_dim)
    int seq_q, int num_heads, int head_dim, int head_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_q * head_dim;
    if (idx >= total) return;
    int row = idx / head_dim;
    int col = idx % head_dim;
    out[row * num_heads * head_dim + head_idx * head_dim + col] = __float2bfloat16(in[idx]);
}
