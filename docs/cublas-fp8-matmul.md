# cuBLAS FP8 Matmul Implementation Guide

## Problem

The current FP8 (F8E4M3) matmul uses a naive CUDA kernel that manually decodes each
FP8 byte to f32, then does a scalar multiply-accumulate — one thread per output element,
looping over K. This is functionally correct but doesn't use tensor cores. On the L4
(Ada Lovelace, which has FP8 tensor cores), FP8 is actually *slower* than FP16 cuBLAS.

## Goal

Replace the FP8 path in `quantized_matmul` with a cuBLAS-based GEMM that uses hardware
FP8 tensor cores. This should make FP8 meaningfully faster than FP16 on Ada/Hopper GPUs.

## Background

cuBLAS supports FP8 GEMM via `cublasLtMatmul`. The API:
- Takes FP8 inputs (E4M3 for weights, E4M3 or E5M2 for activations)
- Accumulates in f32 internally
- Outputs in f16/bf16/f32
- Accepts per-tensor scale factors as device pointers (`aScale`, `bScale`, `dScale`)

The L4 GPU (compute capability 8.9, Ada Lovelace) supports this natively.

## Implementation Steps

### 1. Add `cublaslt` feature to `cudarc` dependency

In `infernum/Cargo.toml`, add the `cublaslt` feature:

```toml
cudarc = { version = "0.12", features = ["cublas", "cublaslt", "cuda-12060"], optional = true }
```

Check what `cudarc` 0.12 actually exports for cublasLt — it may be under `cudarc::cublaslt`
or may need a different version. Run `cargo doc -p cudarc --open` to inspect available modules.

**If `cudarc` doesn't expose `cublasLtMatmul` with FP8 types**, you have two options:
- Use `cudarc::driver` to call the raw CUDA driver API / load the cublasLt shared library
- Use the `cuda-runtime-sys` or `cublas-sys` crate for raw FFI bindings

### 2. Initialize cublasLt handle in `CudaContext`

File: `infernum/src/cuda/context.rs`

```rust
use cudarc::cublaslt::CudaBlasLT;

pub struct CudaContext {
    device: Arc<CudaDevice>,
    blas: Arc<CudaBlas>,
    blas_lt: Arc<CudaBlasLT>,  // NEW
}
```

Initialize in `CudaContext::new()` alongside the existing `CudaBlas` handle.
Add a `pub fn blas_lt(&self) -> &Arc<CudaBlasLT>` accessor.

### 3. Quantize activations to FP8 on-the-fly

cuBLAS FP8 GEMM expects *both* inputs in FP8. The weights are already FP8, but
activations are f32. You need to:

1. Find the absmax of the activation tensor
2. Compute `scale_a = 448.0 / absmax` (448.0 is the max representable E4M3 value)
3. Quantize: `fp8_val = clamp(round(x * scale_a), -448, 448)` cast to E4M3
4. Pass `1.0 / scale_a` as the activation scale to cuBLAS

This should be a small CUDA kernel. Add it in `infernum/src/cuda/ops/quantized_matmul.rs`
or a new file `fp8_utils.rs`.

Alternatively, for a simpler first pass: cast activations to f16 and use a mixed-precision
GEMM (FP8 weights × f16 activations). cuBLAS supports this and it avoids the dynamic
quantization step. Benchmark both approaches.

### 4. Implement the cuBLAS FP8 matmul function

File: `infernum/src/cuda/ops/quantized_matmul.rs`

Create a new function (or add a branch in the existing `quantized_matmul`):

```rust
fn quantized_matmul_fp8_cublas(
    ctx: &CudaContext,
    input: &CudaTensor<f32>,    // (M, K) activations
    weight: &QuantizedTensor,   // (N, K) FP8 weights
) -> Result<CudaTensor<f32>>    // (M, N) output
```

Key cuBLAS parameters:
- **A** = weight, type `CUDA_R_8F_E4M3`, layout row-major (or column-major transposed)
- **B** = input (quantized to FP8 or cast to f16), type accordingly
- **C/D** = output, type `CUDA_R_32F` or `CUDA_R_16F`
- **computeType** = `CUBLAS_COMPUTE_32F`
- **scaleA** = `1.0 / weight.weight_scale()` (device pointer)
- **scaleB** = `1.0 / activation_scale` (device pointer)
- **scaleD** = `1.0` (device pointer)

**Important layout note**: cuBLAS is column-major. The existing `matmul.rs` already handles
this by swapping A/B and computing `C^T = B^T @ A^T`. Follow the same pattern.

### 5. Wire it into the dispatch

In the `DType::F8E4M3` match arm of `quantized_matmul()`, check if the GPU supports
FP8 tensor cores (compute capability >= 8.9) and dispatch to the cuBLAS path.
Fall back to the existing kernel for older GPUs.

```rust
DType::F8E4M3 => {
    if supports_fp8_tensor_cores(ctx) {
        return quantized_matmul_fp8_cublas(ctx, input, weight);
    }
    // existing naive kernel fallback
    let func = device.get_func(module_name, "matmul_fp8_f32").unwrap();
    // ...
}
```

To query compute capability:
```rust
fn supports_fp8_tensor_cores(ctx: &CudaContext) -> bool {
    // Ada Lovelace (sm_89) and Hopper (sm_90+)
    // cudarc may expose this via device properties
    let (major, minor) = ctx.device().compute_capability();
    major > 8 || (major == 8 && minor >= 9)
}
```

Check how `cudarc` exposes compute capability — it might be through
`CudaDevice::attribute()` with `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`.

### 6. Handle weight layout

cuBLAS may require specific memory alignment or layout (e.g., column-major, or
specific tile formats for tensor core ops). The current `QuantizedTensor` stores
FP8 weights in row-major order (N × K, one byte per element).

cublasLt supports specifying `CUBLASLT_ORDER_ROW` for row-major, which should work
with the existing layout. If not, add a one-time transpose on model load in
`infernum/src/weights/safetensors.rs` where FP8 weights are loaded.

### 7. Testing

Add tests alongside the existing ones in `quantized_matmul.rs`:

1. **Correctness**: Compare cuBLAS FP8 output against the existing naive kernel output
   for the same inputs. They should match within FP8 precision tolerance (~0.1% relative error).
2. **Scale handling**: Test with non-trivial `weight_scale` values.
3. **Edge cases**: Test with M=1 (single-token inference), large K, non-aligned dimensions.

### 8. Benchmark

Update the generate example or add a benchmark to compare:
- FP16 cuBLAS (current default)
- FP8 naive kernel (current)
- FP8 cuBLAS (new)

Expected results on L4:
- FP8 cuBLAS should be ~1.5-2x faster than FP16 cuBLAS
- FP8 cuBLAS should be ~3-4x faster than FP8 naive kernel

## Files to modify

| File | Change |
|---|---|
| `infernum/Cargo.toml` | Add `cublaslt` feature to cudarc |
| `infernum/src/cuda/context.rs` | Add `CudaBlasLT` handle |
| `infernum/src/cuda/ops/quantized_matmul.rs` | Add cuBLAS FP8 path + activation quantization kernel |
| `infernum/src/cuda/quantized.rs` | Possibly add alignment/layout helpers |

## References

- [cuBLAS LtMatmul docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul)
- [NVIDIA FP8 training/inference guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [cudarc crate docs](https://docs.rs/cudarc/)
- [vLLM FP8 implementation](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/fp8/) — uses cutlass, good reference for scale handling

## Pitfalls

1. **cudarc cublasLt support**: Verify that `cudarc` 0.12 actually exposes the FP8 data types
   and matmul descriptors needed. If not, you may need raw FFI or a newer cudarc version.
2. **Scale as device pointer**: cublasLt expects scale factors as *device* pointers (GPU memory),
   not host values. Allocate small GPU buffers for the scale floats.
3. **Alignment**: Tensor core ops may require 16-byte aligned pointers. `cudarc` allocations
   should be aligned by default, but verify.
4. **Column-major gotcha**: Follow the same trick used in `matmul.rs` — swap A and B to
   get row-major output from column-major cuBLAS.
5. **Activation quantization overhead**: If dynamic FP8 quantization of activations is too
   slow (absmax reduction + quantize kernel), consider the f16 activation path first.
