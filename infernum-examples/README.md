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

### `bench` — Decode throughput benchmark

Measures raw decode throughput (tokens/sec) using greedy decoding with KV cache.
Works with both SafeTensors directories and GGUF files. Runs a short warmup
before the timed run.

```bash
# Default: 128 tokens
cargo run --release --example bench --features cuda -- /path/to/model

# Custom token count
cargo run --release --example bench --features cuda -- /path/to/model 256
```

## Benchmarking against llama.cpp

To compare infernum performance against llama.cpp:

### 1. Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release -j$(nproc)
```

### 2. Convert model to GGUF

```bash
pip install torch transformers gguf

# Convert SafeTensors → GGUF (f16)
python3 /tmp/llama.cpp/convert_hf_to_gguf.py models/llama-3.2-1b \
    --outfile /tmp/llama-3.2-1b-f16.gguf --outtype f16

# Quantize to Q8_0 and Q4_0
/tmp/llama.cpp/build/bin/llama-quantize /tmp/llama-3.2-1b-f16.gguf /tmp/llama-3.2-1b-q8_0.gguf q8_0
/tmp/llama.cpp/build/bin/llama-quantize /tmp/llama-3.2-1b-f16.gguf /tmp/llama-3.2-1b-q4_0.gguf q4_0
```

### 3. Run llama.cpp benchmark

```bash
cd /tmp/llama.cpp
LD_LIBRARY_PATH=build/bin:$LD_LIBRARY_PATH \
  ./build/bin/llama-bench -m /tmp/llama-3.2-1b-q8_0.gguf -p 512 -n 128 -r 3 -ngl 99
```

### 4. Run infernum benchmark

```bash
cargo run --release --example bench --features cuda -- /tmp/llama-3.2-1b-q8_0.gguf 128
```

### Reference results (NVIDIA L4, Llama 3.2 1B)

| Format | llama.cpp decode (tok/s) | infernum decode (tok/s) | Gap |
|--------|--------------------------|-------------------------|-----|
| F32    | 50.1                     | 46.0                    | -8% |
| F16    | 96.3                     | —                       | n/a |
| Q8_0   | 164.6                    | 33.3                    | -80% |
| Q4_0   | 252.9                    | 45.8                    | -82% |

*Measured 2025-02-19 with 8-token prompt, 128 generated tokens, greedy decoding,
KV cache enabled. llama.cpp at commit `11c325c`.*

## Kernel files

| File | Language | Compiled by |
|------|----------|-------------|
| `kernels/relu.cu` | CUDA C | `nvcc --ptx` |
| `kernels/triton_gelu.py` | Triton (Python) | `python3` → `triton.compiler` |
