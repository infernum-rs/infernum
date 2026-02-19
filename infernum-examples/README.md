# infernum-examples

Example binaries demonstrating how to use and extend Infernum.

All examples require a CUDA GPU and are built with `--features cuda`.

## Examples

### `generate` — Text generation with Llama

Run inference on a Llama model with configurable sampling parameters, KV cache,
and support for both SafeTensors and GGUF weight formats.

```bash
cargo run --example generate --features cuda -- -m /path/to/model "Hello"
```

### `custom_cuda_op` — Adding a custom CUDA kernel

Shows the full workflow for adding a new GPU operation using CUDA C:

1. Write the kernel in `kernels/relu.cu`
2. `build.rs` compiles it to PTX via `nvcc` at build time
3. Load the PTX with `include_str!` and launch via cudarc

```bash
cargo run --example custom_cuda_op --features cuda
```

### `custom_triton_op` — Adding a custom Triton kernel

Same workflow but using a [Triton](https://github.com/triton-lang/triton) kernel
written in Python. Demonstrates that Triton-compiled PTX can be loaded and
launched identically to CUDA C kernels, with notes on the Triton-specific
details (thread count from `.reqntid`, extra null pointer parameters, raw `u64`
device pointers).

Requires `pip install triton` in the Python environment.

```bash
cargo run --example custom_triton_op --features cuda
```

## Kernel files

| File | Language | Compiled by |
|------|----------|-------------|
| `kernels/relu.cu` | CUDA C | `nvcc --ptx` |
| `kernels/triton_gelu.py` | Triton (Python) | `python3` → `triton.compiler` |
