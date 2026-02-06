# Infernum: Design Document

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

## Graph Optimization via Macros

### The Problem

We want readable code (blocks as functions) but also the ability to fuse operations for performance.

### The Solution

Rust macros that look like code but generate both:
1. Executable code (what you wrote, readable, debuggable)
2. A graph representation (for pattern matching and fusion)

### How It Works

User writes:
```rust
infernum::define_block! {
    fn swiglu(x, gate_proj, up_proj, down_proj) {
        let gate = gate_proj(x);
        let up = up_proj(x);
        let activated = silu(gate) * up;
        down_proj(activated)
    }
}
```

Macro generates:
- The actual forward function (exactly what you wrote)
- A graph representation capturing the structure
- Hooks for fusion rule matching

### Fusion Rules

Separately, kernel developers register fusion rules:
```
Pattern: [Linear, SiLU, Mul, Linear] → FusedSwiGLUKernel
```

The optimizer matches patterns in the graph and substitutes fused implementations.

### The Key Insight

The code is always the source of truth. The graph is derived from it. Debug mode runs your actual code. Production mode uses fused kernels where available.

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
│   ├── ops/               # Op traits (MatMul, Attention, etc.)
│   ├── cuda/              # CUDA implementations
│   │   ├── tensor.rs      # CudaTensor
│   │   └── ops/           # CublasMatMul, FlashAttention, etc.
│   ├── block.rs           # Block trait
│   ├── layers/            # Common compositions (Linear, Embedding)
│   ├── weights/           # SafeTensors, GGUF loaders
│   └── tokenizer.rs       # Tokenizer + ChatTemplate traits
│
├── infernum-macros/       # Procedural macros (define_block!, etc.)
│
├── infernum-llama/        # Llama model family
│   ├── config.rs          # Parse HF config.json
│   ├── model.rs           # Llama struct
│   ├── layers.rs          # LlamaAttention, LlamaMLP
│   ├── tokenizer.rs       # LlamaTokenizer impl
│   └── chat_template.rs   # Llama3 chat format
│
├── infernum-qwen/         # Qwen model family (later)
├── infernum-phi/          # Phi model family (later)
│
├── infernum-runtime/      # Execution runtime
│   ├── runtime.rs         # Main interface (text in, text out)
│   ├── engine/            # Internal (token-level)
│   │   ├── engine.rs      
│   │   ├── scheduler.rs   # Scheduler trait + impls
│   │   ├── kv_cache.rs    # KVCachePool
│   │   └── batch.rs       
│   ├── metrics.rs         # Prometheus metrics
│   └── config.rs          
│
└── infernum-serve/        # HTTP server
    ├── server.rs          # Axum setup
    ├── api/
    │   ├── completions.rs # POST /v1/chat/completions
    │   ├── models.rs      # GET /v1/models
    │   └── health.rs      
    └── config.rs          # TOML config
```

### Dependency Flow

```
infernum-serve
    └── infernum-runtime
            ├── infernum
            ├── infernum-llama
            └── (other model crates)
```

---

## Tokenization

Lives in **infernum-runtime**, not a separate crate.

The runtime handles text ↔ token conversion. Internally may be split into text/ and engine/ modules, but consumers see one unified interface.

The model crates (infernum-llama, etc.) implement the Tokenizer and ChatTemplate traits.

Runtime provides:
- `complete(prompt: &str, max_tokens) -> String`
- `chat(messages: &[Message], max_tokens) -> String`
- `chat_stream(messages) -> impl Stream<Item = String>`
- `complete_tokens(tokens: &[u32], max_tokens) -> Vec<u32>` (low-level escape hatch)

---

## Engine Internals

### The Model is Dumb

The model just sees batched tensors. It doesn't know about requests, scheduling, or batching.

```
Model.forward(tokens, positions, kv_cache, slot_mapping) -> logits
```

### The Engine is Smart

The engine handles:
- Scheduler: which requests to run this step
- KV cache pool: memory management, slot allocation
- Batch assembly: build tensors from requests
- Sampling: temperature, top_p, top_k
- Metrics: latency, throughput

### Scheduling Strategies

Pluggable scheduler trait:
- FCFS (simple)
- Continuous batching (vLLM-style)
- Fair scheduling (prevent starvation)
- Priority scheduling

### KV Cache

PagedAttention-style: pre-allocated blocks, requests map to blocks, allows memory sharing and efficient preemption.

---

## Weight Loading

Support multiple formats:
- SafeTensors (HuggingFace standard)
- GGUF (llama.cpp format)

Key requirement: **Load without full host memory**. For large models:
- Memory-map the file
- Copy directly to GPU chunk by chunk
- Never buffer entire model in RAM

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

### Phase 1: Core Foundations (Weeks 1-6)

**Goal**: Single model on single GPU, end to end.

Week 1-2: `infernum` crate
- Tensor trait + CudaTensor
- Basic ops using existing kernels (cuBLAS, FlashAttention)
- Weight loading from safetensors
- Unit tests

Week 3-4: `infernum-llama`
- Config parsing from HF config.json  
- Llama model struct
- Load Llama 3.2 1B weights
- Forward pass (no KV cache)
- Compare outputs with HF transformers

Week 5-6: Generation
- KV cache
- Autoregressive generation loop
- Tokenizer + chat template
- Example binary that generates text

**Milestone**: `cargo run --example generate -- "Hello"` produces coherent output.

### Phase 2: Runtime (Weeks 7-12)

Week 7-8: `infernum-runtime`
- Runtime struct wrapping model + tokenizer
- Text in/out API
- Streaming generation
- Single request handling

Week 9-10: Batching
- Continuous batching
- Scheduler trait + implementation
- Multiple concurrent requests
- Metrics

Week 11-12: `infernum-serve`
- Axum HTTP server
- OpenAI-compatible endpoints
- SSE streaming
- Works with OpenAI Python client

**Milestone**: HTTP server serving Llama with OpenAI-compatible API.

### Phase 3: Researcher DX (Weeks 13-18)

Week 13-14: Macro system
- `define_block!` macro
- Graph representation generation
- Basic fusion rule matching

Week 15-16: Benchmarking
- Built-in benchmark harness
- Compare custom vs baseline
- Memory profiling
- Report generation

Week 17-18: Documentation + examples
- How to add custom ops, blocks, models
- Example: implement a paper's attention variant
- Ensure AI can translate Python → Infernum

**Milestone**: Researcher can implement and benchmark new attention in an afternoon.

### Phase 4: Scale (Week 19+)

- `Parallel<T>` for multi-GPU
- More models (Qwen, Phi, Mistral)  
- Quantization support
- Metal backend
- PagedAttention optimization

---

## Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Rust | Macros, type safety, performance, no GIL |
| Tensor typing | Hardware in type | Prevent mixing at compile time |
| TP approach | `Parallel<T>` wrapper | Type-safe multi-GPU |
| Op granularity | Traits per operation | Swappable implementations |
| Graph optimization | Macro-based | Code is source of truth, graph derived |
| Tokenization | In runtime crate | Single crate for consumers |
| Error handling | Panics (for now) | Simpler, bugs should crash |
| API | OpenAI-compatible | Industry standard |
| First model | Llama 3.2 1B | Small, well-documented, covers most architectures |

---

## What This Document Doesn't Cover (Needs Elaboration)

- Detailed Triton kernel implementation
- NCCL setup for multi-GPU
- Exact macro implementation
- Quantization approach (INT8, INT4, GPTQ, AWQ, etc.)
- Memory management details
- Speculative decoding
- PagedAttention implementation details
- CI/CD, testing strategy
- Deployment / containerization
