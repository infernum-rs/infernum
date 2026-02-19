# Infernum

A Rust-based LLM inference server designed to be researcher-friendly, type-safe, and composable.

## Goals

- **Researcher-friendly**: Easy integration of new inference ideas with fast benchmarking
- **AI-assistant friendly**: Clean, readable codebase with consistent patterns
- **Type-safe**: Rust's type system enforces correctness at compile time
- **Composable**: Building blocks that can be mixed and matched

## Status

Phase 1 implementation complete - single GPU Llama inference.

See `docs/plan.md` for the design document and roadmap.

## Features

- **Tensor Foundation**: `CudaTensor` with GPU memory management and host/device transfers
- **Core Ops**: MatMul (cuBLAS), RMSNorm, RoPE, SiLU, Softmax, Attention
- **Block Fusion**: `define_block!`/`define_fusion!` macros for automatic kernel fusion with feature-flag control
- **Weight Loading**: SafeTensors with memory-mapped loading
- **Llama Model**: Full Llama 3.2 architecture with GQA support
- **Tokenizer**: HuggingFace tokenizers integration
- **Generation**: Greedy decoding with CLI example

## Prerequisites

- Rust stable toolchain
- CUDA Toolkit (12.x recommended)
- A Llama model in HuggingFace SafeTensors format

## Building

```bash
# Build without CUDA (for development/CI)
cargo build

# Build with CUDA support
cargo build --features cuda

# Run tests
cargo test --all

# Run linting
cargo clippy -- -W clippy::pedantic
```

## Usage

### Text Generation Example

```bash
# Download a Llama model (e.g., Llama 3.2 1B)
# Place it in a directory with config.json, tokenizer.json, and *.safetensors files

# Run generation
cargo run --example generate --features cuda -- \
  --model /path/to/llama-3.2-1b \
  "Hello, my name is"

# Or set the model path via environment variable
export LLAMA_MODEL_PATH=/path/to/llama-3.2-1b
cargo run --example generate --features cuda -- "Once upon a time"
```

### Options

```
Usage: generate [OPTIONS] [PROMPT]

Options:
  -m, --model <PATH>       Path to model directory
  -n, --max-tokens <N>     Maximum tokens to generate (default: 100)
  -h, --help               Show help message
```

### Block Fusion Example

The fusion framework lets you define blocks with decomposed implementations and
register optimized fused replacements. See `docs/fusion.md` for a full guide.

```bash
# Debug build — always uses decomposed path
cargo run --example fusion_example

# Release build — uses fused kernels after fusion::init()
cargo run --example fusion_example --release

# Force fused even in debug
cargo run --example fusion_example --features force-fuse

# Force decomposed even in release
cargo run --example fusion_example --features no-fuse --release
```

## Architecture

```
infernum/
├── cuda/           # CUDA backend (CudaContext, CudaTensor)
├── ops/            # CUDA kernels (matmul, rmsnorm, rope, silu, softmax, attention)
├── llama/          # Llama model implementation
├── weights/        # SafeTensors weight loading
├── tokenizer/      # Tokenizer integration
├── dtype.rs        # Data types (F32, F16, BF16)
├── tensor.rs       # Tensor trait definition
└── error.rs        # Error types
```

## Current Limitations

- No KV cache (recomputes full sequence each step)
- Single GPU only
- Greedy decoding only (no sampling)
- F32 inference only (no quantization)

## Next Steps (Phase 2)

- KV cache for efficient generation
- Continuous batching
- HTTP server with OpenAI-compatible API
- Multi-GPU support

## License

TODO