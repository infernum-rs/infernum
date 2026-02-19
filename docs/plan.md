# Infernum: Design & Roadmap

## Project Overview

Infernum is a Rust-based LLM inference server. The name comes from Latin (meaning "inferno") — note that `inferno` is taken on crates.io (it's a flamegraph tool).

### Core Goals

1. **Researcher-friendly**: Make it easy to integrate new inference ideas and benchmark them quickly. Reduce the "time from paper to running inference."

2. **AI-assistant friendly**: Clean, readable codebase with consistent patterns and good examples. Given a Python implementation of some new technique, an AI coding assistant should be able to translate it to Infernum easily.

3. **Type-safe**: Use Rust's type system to enforce correctness at compile time — prevent mixing tensors from different hardware, ensure parallelism-aware ops are used in multi-GPU setups, etc.

4. **Composable**: Building blocks that can be mixed and matched. Researchers should be able to swap out one attention implementation for another without touching the rest of the model.

---

## The Inference Landscape (Context)

### What big companies use

- **Frontier labs (Google, OpenAI, Anthropic)**: Proprietary inference stacks, custom CUDA kernels, optimized for their specific models and hardware
- **Inference providers (OpenRouter, Together, Fireworks)**: Build on open source — primarily vLLM, TensorRT-LLM, or SGLang. Heavily patched and extended.
- **Local inference (Ollama, llama.cpp)**: Optimized for ease of use on consumer hardware

### Why Python works for vLLM

vLLM is mostly Python, which seems wrong for performance-critical code. But it works because:
- 99%+ of compute time is in CUDA kernels (FlashAttention, cuBLAS, Triton)
- Python is just orchestrating — scheduling, KV cache management, batching
- The actual hot path is C++/CUDA
- Developer velocity matters more than shaving microseconds off the control plane

### Why Rust for Infernum

- Procedural macros can analyze code structure at compile time (enables the graph optimization approach)
- Zero-cost abstractions — generics monomorphize
- Memory control needed for GPU memory and KV cache management
- No GIL for the scheduler
- Type system can enforce hardware/parallelism constraints
- Cloudflare built their inference engine (Infire) in Rust for similar reasons

---

## Core Abstractions

### The Hierarchy

```
Op      → Atomic compute (matmul, softmax, rope, etc.)
Block   → Composable architecture unit (attention, FFN)  
Model   → Full model (Llama, Qwen, etc.)
Runtime → Scheduling, batching, KV cache, tokenization, text in/out
Server  → HTTP, OpenAI-compatible API
```

### Tensor as a Trait

The tensor type encodes what hardware data lives on. Different hardware = different types = can't mix them.

```
CudaTensor      → NVIDIA GPU
MetalTensor     → Apple Silicon  
Parallel<T>     → Distributed across multiple GPUs (wraps another tensor type)
```

The `Parallel<T>` wrapper is how we handle tensor parallelism — an op that takes `Parallel<CudaTensor>` must handle multi-GPU communication. An op that only takes `CudaTensor` can't be used in a TP setup (compiler error).

### Ops as Traits

Each operation (MatMul, Softmax, RMSNorm, RoPE, Attention, SiLU, etc.) is a trait with an associated tensor type.

Different implementations exist for:
- Different hardware (CublasMatMul for CUDA, MetalMatMul for Apple)
- Different algorithms (FlashAttention vs naive attention)
- Different optimizations (fused vs unfused)

Key insight: An Op and a Block can be interchangeable from the outside. Sometimes for optimization you create a new Op that could be implemented as a Block (composition of smaller ops), but it's faster as a single fused kernel.

### Blocks Compose Ops

Blocks are the unit of architectural composition. A researcher saying "I want to try a different attention mechanism" is swapping a Block.

The generic constraints on blocks ensure all ops use the same tensor type — you can't accidentally mix CUDA and Metal ops in one block.

### Not All Ops Exist for All Backends

This is fine. If Metal doesn't have FlashAttention yet, you can't build a model that uses FlashAttention on Metal. The compiler will tell you "FlashAttention doesn't exist for MetalTensor."

Community can add ops incrementally. No artificial "Backend" bundle that forces all-or-nothing support.

---

## Block Fusion via Macros

### The Problem

We want readable code (blocks as functions) but also the ability to fuse operations for performance.

### The Solution

Two proc macros — `define_block!` and `define_fusion!` — plus an `inventory`-based registry that connects them at startup.

### How It Works

`define_block!` wraps a function, generating:
- `foo_decomposed(...)` — the original body, always available
- `FOO_FUSED` — a `OnceLock` static for the fused replacement
- `foo(...)` — a dispatcher that checks the static

```rust
infernum::define_block! {
    fn attention(q, k, v, mask) -> Tensor {
        let scores = matmul(q, k_transposed);
        let masked = apply_mask(scores, mask);
        let weights = softmax(masked);
        matmul(weights, v)
    }
}
```

`define_fusion!` registers an optimized replacement via `inventory`:

```rust
infernum::define_fusion! {
    block: ATTENTION_FUSED,
    fn fused_attention(q, k, v, mask) -> Tensor {
        flash_attention_kernel(q, k, v, mask)
    }
}
```

At startup, `fusion::init()` runs all registered initializers, populating the `OnceLock` statics.

### Dispatch Behavior

| Build | Feature flag | Dispatcher calls |
|-------|-------------|-----------------|
| Debug | (default) | Decomposed (zero overhead — never checks static) |
| Release | (default) | Fused if registered, else decomposed |
| Any | `force-fuse` | Fused (useful for testing fused paths in debug) |
| Any | `no-fuse` | Decomposed (useful for debugging in release) |

### The Key Insight

The code is always the source of truth. Debug mode runs your actual code. Production mode uses fused kernels where available. The feature flags give full control for testing and debugging.

See `docs/fusion.md` for a guide on adding fused ops.

---

## Multi-GPU / Tensor Parallelism

### The Approach

Wrap the tensor type: `Parallel<CudaTensor>` contains:
- The local shard on this GPU
- TP rank and size
- NCCL communicator for all-reduce

### Type Safety

Ops that support TP implement for `Parallel<T>`. Ops that don't only implement for `T`.

If you try to build a model with TP using an op that doesn't support it, the compiler catches the type mismatch.

### vLLM's Approach (for reference)

vLLM uses special parallel linear layers (ColumnParallelLinear, RowParallelLinear) baked into model definitions. Parallelism is a runtime config (`tensor_parallel_size=4`), not a type parameter.

For Infernum, we're encoding it in the type system for stronger guarantees, but this is a design choice that could be revisited.

---

## Project Structure

```
infernum/
├── infernum/               # Core crate
│   ├── tensor.rs          # Tensor trait
│   ├── dtype.rs           # Data types (F32, F16, BF16, Q8_0, Q4_0)
│   ├── error.rs           # Error types
│   ├── model.rs           # Model trait + ModelConfig
│   ├── fusion.rs          # Fusion registry (init, FusionInit)
│   ├── sampling.rs        # Sampling parameters
│   ├── cuda/              # CUDA backend
│   │   ├── tensor.rs      # CudaTensor
│   │   ├── quantized.rs   # QuantizedTensor (FP8, Q8, Q4)
│   │   ├── context.rs     # CudaContext (device, cuBLAS, PTX)
│   │   ├── kv_cache.rs    # KvCache
│   │   └── ops/           # CUDA op implementations
│   ├── tokenizer/         # Tokenizer trait + implementations
│   │   ├── llama_tokenizer.rs    # HuggingFace tokenizers
│   │   └── gguf_tokenizer.rs     # Tokenizer from GGUF metadata
│   └── weights/           # Weight loaders
│       ├── safetensors.rs # SafeTensors (memory-mapped)
│       └── gguf.rs        # GGUF format
│
├── infernum-macros/       # Procedural macros (define_block!, define_fusion!)
│
├── infernum-llama/        # Llama model family
│   ├── config.rs          # Parse HF config.json / GGUF metadata
│   └── model.rs           # LlamaModel (Llama 3.2, GQA)
│
├── infernum-runtime/      # Execution runtime
│   ├── engine.rs          # Token-level engine (KV cache, prefill/decode, sampling)
│   └── runtime.rs         # Text-level runtime (tokenize, generate, stream)
│
├── infernum-examples/     # Example binaries
│   └── examples/
│       └── generate.rs    # CLI text generation
│
├── infernum-qwen/         # Qwen model family (planned)
├── infernum-phi/          # Phi model family (planned)
├── infernum-mistral/      # Mistral model family (planned)
└── infernum-serve/        # HTTP server (planned)
```

### Dependency Flow

```
infernum-serve (planned)
    └── infernum-runtime
            ├── infernum
            ├── infernum-llama
            └── (other model crates)
```

---

## Tokenization

The `Tokenizer` trait and concrete implementations live in the **infernum** core crate (`infernum/src/tokenizer/`). Two implementations exist:

- `LlamaTokenizer` — wraps HuggingFace `tokenizers` library, loads from `tokenizer.json`
- `GgufTokenizer` — builds a tokenizer from GGUF file metadata (no separate tokenizer file needed)

The **runtime** consumes the trait for text ↔ token conversion, providing text-level APIs:
- `runtime.generate(prompt, options)` — tokenize, generate, detokenize
- `runtime.generate_streaming(prompt, options)` — stream tokens as text

---

## Engine Internals

### The Model is Dumb

The model just sees tensors. It doesn't know about requests, scheduling, or batching.

```
Model.forward(tokens, positions, kv_cache) -> logits
```

### The Engine Manages State

Currently the engine handles single-request inference:
- KV cache: pre-allocated per model, reset between requests
- Prefill/decode loop: full-sequence prefill, then incremental token-by-token decode
- Sampling: temperature, top-p (nucleus sampling), greedy (argmax)
- Streaming: token-by-token output via channels

### Future: Batching & Scheduling (Phase 6)

When inflight batching is added, the engine will grow:
- Scheduler trait with pluggable strategies (FCFS, continuous batching, fair, priority)
- PagedAttention: pre-allocated KV cache blocks, dynamic slot allocation per request
- Batch assembly: build batched tensors from heterogeneous concurrent requests
- Metrics: latency, throughput, queue depth

---

## Weight Loading

Both formats are implemented with memory-mapped loading:
- **SafeTensors** (HuggingFace standard) — `SafeTensorsLoader`
- **GGUF** (llama.cpp format) — `GgufLoader`, also extracts tokenizer and model config from metadata

Key design: **Load without full host memory**. Files are memory-mapped; weights are copied directly to GPU chunk by chunk without buffering the entire model in RAM.

---

## API Protocol

OpenAI-compatible API is the standard. Everyone speaks it.

Core endpoints:
- `POST /v1/chat/completions` (with streaming via SSE)
- `GET /v1/models`

### Reasoning/Thinking

OpenAI doesn't standardize this. DeepSeek added `reasoning_content` field. Anthropic uses block types. 

For Infernum: start with OpenAI compat, add `reasoning_content` field (DeepSeek style) when supporting reasoning models. Don't overthink it.

---

## Error Handling

**Development**: Panic on everything. Shape mismatch = bug = crash with good error message.

**Later**: Use Result for runtime errors (OOM, CUDA failure). Keep panics for programmer errors.

If shapes don't match, something is deeply wrong. You're not going to "handle" this gracefully.

---

## Implementation Roadmap

### Phase 1: Core Foundations ✅

Single model on single GPU, end to end.

- Tensor trait + `CudaTensor` with GPU memory management
- Core ops: MatMul (cuBLAS), RMSNorm, RoPE, SiLU, Softmax, Attention
- Weight loading from SafeTensors (memory-mapped)
- Llama 3.2 1B model: config parsing, forward pass, GQA
- Tokenizer integration (HuggingFace tokenizers)

**Milestone**: `cargo run --example generate -- "Hello"` produces coherent output. ✅

### Phase 2: KV Cache, Quantization & Runtime ✅

- KV cache for incremental decoding (prefill + decode phases)
- FP8 quantization with quantized matmul kernels
- GGUF weight loading (memory-mapped, with tokenizer from metadata)
- `infernum-runtime` crate: Engine (token-level) + Runtime (text-level)
- Streaming generation with sampling (temperature, top-p)

See `docs/phase2-plan.md` for detailed design.

**Milestone**: Streaming generation with KV cache and quantized models. ✅

### Phase 3: Performance Optimization 🔄

*In progress on `optimizer` branch.*

- `infernum-macros` crate: `define_block!` and `define_fusion!` proc macros
- Fusion registry with `inventory` for cross-crate registration
- Feature flags: `force-fuse` / `no-fuse` for testing
- Fused kernels: attention, add+rmsnorm, SwiGLU
- Kernel performance tuning and benchmarking

**Milestone**: Fused kernels match or beat naive implementations in benchmarks.

### Phase 4: More Model Architectures

Broaden model support beyond Llama.

- **Qwen** (`infernum-qwen`): Qwen 2/2.5 family
- **Phi** (`infernum-phi`): Phi-3/4 family
- **Mistral** (`infernum-mistral`): Mistral/Mixtral family
- Factor out shared patterns (transformer blocks, weight mapping) to reduce per-model boilerplate
- Chat templates per model family

**Milestone**: At least two additional model families running end to end.

### Phase 5: Multi-GPU / Tensor Parallelism

Scale to models that don't fit on a single GPU.

- `Parallel<T>` tensor wrapper encoding TP rank/size in the type
- NCCL integration for all-reduce / all-gather
- Parallel linear layers (column-parallel, row-parallel)
- Sharded weight loading (load directly to target GPU, no host buffering)
- Parallel-aware ops: ops that don't support TP won't compile with `Parallel<T>`

**Milestone**: 70B-class model running across multiple GPUs.

### Phase 6: Inflight Batching

Serve multiple concurrent requests efficiently.

- PagedAttention: pre-allocated KV cache blocks, dynamic slot allocation
- Scheduler trait with pluggable strategies (FCFS, continuous batching, fair, priority)
- Batch assembly: build batched tensors from heterogeneous requests
- Request preemption and memory sharing
- Metrics: latency, throughput, queue depth

**Milestone**: Sustained throughput under concurrent load with bounded latency.

### Phase 7: HTTP Server (`infernum-serve`)

OpenAI-compatible API for production serving.

- Axum HTTP server
- `POST /v1/chat/completions` with SSE streaming
- `GET /v1/models`
- TOML configuration
- Works with OpenAI Python client
- `reasoning_content` field for reasoning models (DeepSeek style)

**Milestone**: HTTP server serving models with OpenAI-compatible API.

### Future

- Benchmarking harness (compare custom vs baseline, memory profiling, report generation)
- Researcher DX: documentation, examples, "implement a paper in an afternoon"
- Speculative decoding
- Metal backend (Apple Silicon)
- Deployment / containerization

---

## Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Rust | Macros, type safety, performance, no GIL |
| Tensor typing | Hardware in type | Prevent mixing at compile time |
| TP approach | `Parallel<T>` wrapper | Type-safe multi-GPU |
| Op granularity | Traits per operation | Swappable implementations |
| Graph optimization | Macro-based | Code is source of truth, graph derived |
| Tokenization | In core crate | Trait in `infernum`, consumed by runtime |
| Error handling | Panics (for now) | Simpler, bugs should crash |
| API | OpenAI-compatible | Industry standard |
| First model | Llama 3.2 1B | Small, well-documented, covers most architectures |

---

## What This Document Doesn't Cover (Needs Elaboration)

- NCCL setup for multi-GPU
- PagedAttention implementation details
- Speculative decoding
- Memory management details for large-scale serving
- Deployment / containerization
