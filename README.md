# Infernum

A Rust-based LLM inference server designed to be researcher-friendly, type-safe, and composable.

## Goals

- **Researcher-friendly**: Easy integration of new inference ideas with fast benchmarking
- **AI-assistant friendly**: Clean, readable codebase with consistent patterns
- **Type-safe**: Rust's type system enforces correctness at compile time
- **Composable**: Building blocks that can be mixed and matched

## Status

Active development — single and multi-GPU inference with continuous batching, paged KV cache, and an OpenAI-compatible HTTP server.

See `docs/plan.md` for the design document and roadmap.

## Supported Models

| Family | `model_type` | Crate | Architecture | Notes |
|--------|-------------|-------|--------------|-------|
| Llama | `llama` | `infernum-llama` | Dense | Llama 2, Llama 3, SmolLM2, CodeLlama, etc. |
| Mistral | `mistral` | `infernum-llama` | Dense | Mistral v1/v2/v3, Devstral |
| Mixtral | `mixtral` | `infernum-llama` | MoE | Mixtral 8x7B, 8x22B |
| Qwen | `qwen2` / `qwen3` / `qwen3_moe` | `infernum-qwen` | Dense / MoE | Qwen2/2.5, Qwen3, Qwen3-MoE |
| DeepSeek | `deepseek_v3` | `infernum-deepseek` | MLA + MoE | DeepSeek-V3, DeepSeek-R1 |
| Gemma 2 | `gemma2` | `infernum-gemma` | Dense | Gemma 2 2B/9B/27B |
| Gemma 3 | `gemma3_text` | `infernum-gemma` | Dense | Gemma 3 1B/4B/12B/27B (text decoder) |

Model family is auto-detected from `config.json`.

## Features

- **Tensor Foundation**: `CudaTensor` with GPU memory management, host/device transfers, and buffer pooling
- **Core Ops**: MatMul (cuBLAS), RMSNorm, RoPE (standard and interleaved), SiLU, GeGLU, Softmax, Attention, MLA tensor ops, MoE routing
- **Block Fusion**: `define_block!`/`define_fusion!` macros for automatic kernel fusion with feature-flag control
- **Quantization**: FP8 (E4M3), GPTQ INT4, AWQ INT4, Q8_0, Q4_0, Q6_K — all with on-the-fly dequantization during matmul
- **Weight Loading**: SafeTensors (memory-mapped, sharded) and GGUF (with tokenizer from metadata)
- **KV Cache**: Pre-allocated per-layer cache for incremental decoding, plus paged KV cache for continuous batching
- **Paged Attention**: Block-level KV cache allocation with dynamic slot management
- **Continuous Batching**: Inflight batching with FCFS scheduling, chunked prefill, and iteration-level preemption
- **CUDA Graphs**: Capture/replay for batched decode steps (optional)
- **Multi-GPU**: Tensor parallelism via NCCL (all-reduce), sharded weight loading
- **Sliding Window Attention**: Per-layer window configuration (Mistral, Qwen3, Gemma)
- **Sampling**: Greedy (argmax), nucleus (top-p) with temperature, repetition penalty
- **Chat Templates**: Per-model-family templates (Llama 3, ChatML, Gemma, DeepSeek)
- **HTTP Server**: OpenAI-compatible `/v1/chat/completions` (streaming + non-streaming) and `/v1/models`
- **Tokenizer**: HuggingFace tokenizers and GGUF-embedded tokenizer

## Prerequisites

- Rust stable toolchain
- CUDA Toolkit (12.x recommended)
- Model weights in HuggingFace SafeTensors or GGUF format

## Building

```bash
# Build without CUDA (for development/CI)
cargo build

# Build with CUDA support
cargo build --features cuda

# Run tests
cargo test --all

# Run linting (pedantic)
cargo clippy -- -W clippy::pedantic

# Format
cargo fmt --all
```

## Usage

### Text Generation

```bash
# SafeTensors directory (auto-detects model family)
cargo run --example generate --features cuda -- -m /path/to/model "Hello"

# GGUF file
cargo run --example generate --features cuda -- -m model.gguf "Hello"

# Greedy decoding
cargo run --example generate --features cuda -- -m /path/to/model --greedy "Hello"

# Custom sampling
cargo run --example generate --features cuda -- -m /path/to/model -t 0.8 -p 0.95 "Hello"
```

### HTTP Server

```bash
# Start the server
cargo run --example serve --features cuda -- -m /path/to/model

# Non-streaming request
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'

# Streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

### Multi-GPU

```bash
# Tensor-parallel generation across 2 GPUs
cargo run --example generate_parallel --features nccl -- \
  -m /path/to/model --gpus 2 "Hello"

# Verify single-GPU and multi-GPU produce identical output
cargo run --example verify_parallel --features nccl -- \
  -m /path/to/model --gpus 2
```

### Benchmarking

```bash
# Decode throughput (tokens/sec)
cargo run --release --example bench --features cuda -- /path/to/model 128

# With buffer pool and CUDA graphs
cargo run --release --example bench --features cuda -- /path/to/model 128 --pool --graphs

# Compare against llama.cpp across quantization formats
./bench_comparison.sh
```

See `infernum-examples/README.md` for full details.

## Architecture

```
infernum/               # Core crate (tensor traits, ops, CUDA impls, blocks, weight loading)
infernum-macros/        # Procedural macros (define_block!, define_fusion!)
infernum-llama/         # Llama / Mistral / Mixtral models
infernum-qwen/          # Qwen2, Qwen3, Qwen3-MoE models
infernum-deepseek/      # DeepSeek-V3/R1 (MLA + MoE)
infernum-gemma/         # Gemma 2, Gemma 3 text models
infernum-runtime/       # Engine (token-level) + Scheduler (continuous batching)
infernum-serve/         # HTTP server (Axum, OpenAI-compatible API)
infernum-examples/      # Example binaries (generate, bench, serve, multi-GPU, custom kernels)
```

## License

TODO
