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
Model   → Full model (Llama, Qwen, DeepSeek, Gemma, etc.)
Runtime → Scheduling, batching, KV cache, tokenization, text in/out
Server  → HTTP, OpenAI-compatible API
```

### Tensor as a Trait

The tensor type encodes what hardware data lives on. Different hardware = different types = can't mix them.

```
CudaTensor      → NVIDIA GPU
MetalTensor     → Apple Silicon (future)
```

Multi-GPU tensor parallelism is handled at runtime, not in the type system. Each GPU runs the same model code with its own `CudaTensor` weights (sharded at load time), and synchronizes via an `NcclCommunicator` passed to the model.

### Ops as Traits

Each operation (MatMul, Softmax, RMSNorm, RoPE, Attention, SiLU, etc.) is a trait with an associated tensor type.

Different implementations exist for:
- Different hardware (CublasMatMul for CUDA, MetalMatMul for Apple)
- Different algorithms (FlashAttention vs naive attention, fused vs unfused)
- Different data types (f32, f16, bf16, quantized)

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

Runtime tensor parallelism: each GPU gets a full model instance with sharded
weights and an NCCL communicator. The model code is nearly identical to
single-GPU — the only additions are 2 `all_reduce_sum` calls per layer.

- `ShardConfig { rank, world_size }` tells weight loading which slice to take
- `NcclCommunicator` provides all-reduce for row-parallel layer outputs
- Column-parallel layers (Q/K/V, gate, up) split the output dimension — no sync
- Row-parallel layers (O, down) split the input dimension — need all-reduce
- `ShardedModel` orchestrates N shards across N GPUs, wrapping them behind the standard `Model` trait

### Why Not `Parallel<T>`?

An earlier design proposed a `Parallel<T>` tensor wrapper to catch missing
all-reduces at compile time. In practice, there are only 2 sync points per
layer, making this hard to miss. The type-level approach would force generic
plumbing through the entire model or require duplicating single/multi-GPU code
paths, without meaningful safety benefit. The runtime approach (used by vLLM,
TensorRT-LLM, and every other major project) is simpler and sufficient.

---

## Project Structure

```
infernum/
├── infernum/               # Core crate
│   ├── tensor.rs          # Tensor trait
│   ├── dtype.rs           # Data types (F32, F16, BF16, F8E4M3, Q8_0, Q4_0, Q6_K, GPTQ_INT4, AWQ_INT4)
│   ├── error.rs           # Error types
│   ├── model.rs           # Model trait + ModelConfig + ShardedLoadable
│   ├── fusion.rs          # Fusion registry (init, FusionInit)
│   ├── sampling.rs        # Sampling parameters (temperature, top-p, repetition penalty)
│   ├── chat_template.rs   # ChatTemplate trait + ChatMessage
│   ├── cuda/              # CUDA backend
│   │   ├── tensor.rs      # CudaTensor<T>
│   │   ├── quantized.rs   # QuantizedTensor (FP8, GPTQ, AWQ, Q8, Q4, Q6_K)
│   │   ├── context.rs     # CudaContext (device, cuBLAS, PTX)
│   │   ├── kv_cache.rs    # KvCache<T> (pre-allocated per-layer)
│   │   ├── paged_kv_cache.rs  # PagedKvCache<T> (block-level allocation)
│   │   ├── block_allocator.rs # BlockAllocator, BlockTable
│   │   ├── buffer_pool.rs # BufferPool (GPU memory reuse)
│   │   ├── graph.rs       # CudaGraph (capture/replay)
│   │   ├── batched_graph.rs   # BatchedGraphInputs (indirect kernel args)
│   │   ├── shard.rs       # ShardConfig, GpuConfig, shard strategies
│   │   ├── sharded.rs     # ShardedModel (multi-GPU wrapper)
│   │   ├── nccl.rs        # NcclCommunicator (all-reduce, all-gather)
│   │   ├── moe.rs         # MoE routing utilities
│   │   ├── seq_position.rs # SeqPosition (offset tracking)
│   │   └── ops/           # CUDA op implementations
│   │       ├── matmul.rs          # cuBLAS matmul (f32, f16, bf16)
│   │       ├── quantized_matmul.rs # Quantized matmul (Q8, Q4, Q6_K, FP8, GPTQ, AWQ)
│   │       ├── attention.rs       # Causal attention (prefill)
│   │       ├── fused_attention.rs # Fused prefill attention (single kernel)
│   │       ├── paged_attention.rs # Paged decode attention
│   │       ├── rmsnorm.rs         # RMSNorm
│   │       ├── add_rmsnorm.rs     # Fused residual + RMSNorm
│   │       ├── rope.rs            # RoPE (standard + interleaved)
│   │       ├── silu.rs / swiglu.rs / gelu.rs / geglu.rs  # Activations
│   │       ├── softmax.rs         # Softmax + causal softmax
│   │       ├── embed.rs           # Token embedding lookup
│   │       ├── linear.rs          # Linear projection dispatch
│   │       ├── moe_routing.rs     # MoE expert routing (top-k, sigmoid)
│   │       ├── mla_tensor_ops.rs  # MLA-specific tensor operations (DeepSeek)
│   │       └── ...                # cast, scale, transpose, repeat_kv, etc.
│   ├── tokenizer/         # Tokenizer trait + implementations
│   │   ├── llama_tokenizer.rs    # HuggingFace tokenizers
│   │   └── gguf_tokenizer.rs     # Tokenizer from GGUF metadata
│   └── weights/           # Weight loaders
│       ├── safetensors.rs # SafeTensors (memory-mapped, sharded)
│       ├── gguf.rs        # GGUF format (memory-mapped)
│       └── loader.rs      # Unified loader interface
│
├── infernum-macros/       # Procedural macros (define_block!, define_fusion!)
│
├── infernum-llama/        # Llama / Mistral / Mixtral model family
│   ├── config.rs          # LlamaConfig (HF config.json + GGUF metadata)
│   ├── model.rs           # LlamaModel (Dense + MoE, GQA, sliding window)
│   └── chat_templates.rs  # Llama3Template, MistralTemplate
│
├── infernum-qwen/         # Qwen model family
│   ├── config.rs          # QwenConfig (Qwen2/2.5, Qwen3, Qwen3-MoE)
│   ├── model.rs           # QwenModel (Dense + MoE, QK-norm, sliding window)
│   └── chat_templates.rs  # ChatMLTemplate, Qwen3Template
│
├── infernum-deepseek/     # DeepSeek model family
│   ├── config.rs          # DeepSeekConfig (V3/R1)
│   ├── model.rs           # DeepSeekModel (MLA + MoE, sigmoid routing)
│   └── chat_templates.rs  # DeepSeekTemplate
│
├── infernum-gemma/        # Gemma model family
│   ├── config.rs          # GemmaConfig (Gemma 2, Gemma 3 text)
│   ├── model.rs           # GemmaModel (soft-capping, dual-theta RoPE)
│   └── chat_templates.rs  # Gemma2Template, Gemma3Template
│
├── infernum-runtime/      # Execution runtime
│   ├── engine.rs          # Engine (paged KV cache, inflight batching, CUDA graphs)
│   ├── scheduler.rs       # Scheduler (FCFS, block-level memory, chunked prefill)
│   └── runtime.rs         # Runtime (text-level: tokenize, generate, stream)
│
├── infernum-serve/        # HTTP server
│   ├── server.rs          # Server + builder (Axum, model registration)
│   └── types.rs           # OpenAI API types (request/response, SSE chunks)
│
└── infernum-examples/     # Example binaries
    └── examples/
        ├── generate.rs           # CLI text generation (all model families)
        ├── bench.rs              # Decode throughput benchmark
        ├── serve.rs              # HTTP server example
        ├── generate_parallel.rs  # Multi-GPU text generation
        ├── verify_parallel.rs    # Single vs multi-GPU correctness check
        ├── custom_cuda_op.rs     # Adding a custom CUDA C kernel
        └── custom_triton_op.rs   # Adding a custom Triton kernel
```

### Dependency Flow

```
infernum-serve
    └── infernum-runtime
            ├── infernum
            ├── infernum-llama
            ├── infernum-qwen
            ├── infernum-deepseek
            └── infernum-gemma
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
Model.forward_prefill_paged(tokens, paged_kvs, block_table, start_pos) -> logits
Model.forward_batch_decode(token_ids, paged_kvs, block_tables, positions) -> logits
```

### The Engine Manages State

The engine owns the model and paged KV caches, providing inflight (continuous) batching:

- **Paged KV cache**: Block-level allocation via `BlockAllocator`, dynamic slot management per request
- **Scheduler**: FCFS admission with block-level memory accounting, chunked prefill
- **Inflight batching**: Multiple requests share the GPU, with scheduling decisions at every decode step
- **Batched decode**: All active sequences processed in a single forward pass (one weight read)
- **CUDA graphs**: Optional capture/replay for batched decode steps
- **Multi-GPU**: One `PagedKvCache` per device, sharing logical block indices
- **Sampling**: Greedy (argmax), nucleus (top-p) with temperature, repetition penalty
- **Streaming**: Token-by-token output via `TokenSender` trait

### Request Lifecycle

```
Waiting → Prefill → Decode → Finished
```

- **Waiting**: queued, not yet admitted
- **Prefill**: prompt tokens are being processed (possibly chunked)
- **Decode**: generating tokens one at a time
- **Finished**: EOS, max tokens, or cancelled — blocks freed

---

## Weight Loading

Both formats are implemented with memory-mapped loading:
- **SafeTensors** (HuggingFace standard) — `SafeTensorsLoader`, supports sharded files
- **GGUF** (llama.cpp format) — `GgufLoader`, also extracts tokenizer and model config from metadata

Key design: **Load without full host memory**. Files are memory-mapped; weights are copied directly to GPU chunk by chunk without buffering the entire model in RAM.

### Quantization Formats

| Format | Source | Description |
|--------|--------|-------------|
| F32, F16, BF16 | SafeTensors / GGUF | Full precision |
| FP8 (E4M3) | SafeTensors (compressed-tensors) | 8-bit float with per-tensor scale |
| GPTQ INT4 | SafeTensors | Group-quantized 4-bit (128 elements/group) |
| AWQ INT4 | SafeTensors | Group-quantized 4-bit (transposed packing) |
| Q8_0, Q4_0 | GGUF | Block-quantized (32 elements/block) |
| Q6_K | GGUF | K-quant 6-bit (256 elements/super-block) |

All quantized formats use on-the-fly dequantization during matmul — the full-precision expansion never exists in GPU memory.

---

## API Protocol

OpenAI-compatible API is the standard. Everyone speaks it.

Core endpoints:
- `POST /v1/chat/completions` (with streaming via SSE)
- `GET /v1/models`

The `infernum-serve` crate provides a builder-based server that accepts model registrations and handles chat template application, tokenization, and generation.

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
- Q8_0, Q4_0, Q6_K block-quantized matmul kernels
- GPTQ INT4 and AWQ INT4 group-quantized matmul kernels
- `infernum-runtime` crate: Engine (token-level) + Runtime (text-level)
- Streaming generation with sampling (temperature, top-p, repetition penalty)

See `docs/phase2-plan.md` for the original design.

**Milestone**: Streaming generation with KV cache and quantized models. ✅

### Phase 3: Performance Optimization ✅

- `infernum-macros` crate: `define_block!` and `define_fusion!` proc macros
- Fusion registry with `inventory` for cross-crate registration
- Feature flags: `force-fuse` / `no-fuse` for testing
- Fused kernels: attention, add+rmsnorm, SwiGLU
- Buffer pool for GPU memory reuse
- CUDA graph capture/replay for decode steps

**Milestone**: Fused kernels and CUDA graphs for optimized decode throughput. ✅

### Phase 4: Multi-GPU / Tensor Parallelism ✅

- NCCL bindings (`NcclCommunicator`: all-reduce, all-gather)
- `ShardConfig` (rank, world_size) for weight slicing
- Sharded weight loading (column/row slicing in SafeTensors loader)
- `ShardedModel` wrapper: orchestrates N shards across N GPUs
- Model changes: sharded constructors + 2 all-reduce calls per layer

**Milestone**: 70B-class models running across multiple GPUs with `generate_parallel` and `verify_parallel` examples. ✅

### Phase 5: More Model Architectures ✅

- **Llama/Mistral/Mixtral** (`infernum-llama`): Dense + MoE, GQA, sliding window attention
- **Qwen** (`infernum-qwen`): Qwen2/2.5, Qwen3 (QK-norm), Qwen3-MoE (decoder_sparse_step)
- **DeepSeek** (`infernum-deepseek`): V3/R1 with MLA (Q LoRA compression, joint KV projection, interleaved RoPE) + MoE (sigmoid routing with bias correction, shared experts)
- **Gemma** (`infernum-gemma`): Gemma 2 (soft-capping, GeGLU, alternating attention) + Gemma 3 text (QK-norm, dual-theta RoPE)
- Chat templates per model family
- Sliding window attention (per-layer, mask-only phase 1)

**Milestone**: Five model families running end to end. ✅

### Phase 6: Inflight Batching ✅

- Paged KV cache: block-level allocation with `BlockAllocator`
- Scheduler with FCFS admission and block-level memory accounting
- Chunked prefill (configurable `max_prefill_tokens`)
- Batched decode: all active sequences in a single forward pass
- CUDA graph support for batched decode (optional)
- Request lifecycle: Waiting → Prefill → Decode → Finished
- `TokenSender` trait for flexible output delivery

**Milestone**: Sustained throughput under concurrent load with inflight batching. ✅

### Phase 7: HTTP Server ✅

- `infernum-serve` crate with Axum
- `POST /v1/chat/completions` with SSE streaming
- `GET /v1/models`
- Builder-based server with model registration (`ModelEntry`)
- Configurable `BatchConfig` per model
- CORS support, graceful shutdown
- `serve` example binary

**Milestone**: HTTP server serving models with OpenAI-compatible API. ✅

### Future

- Ring-buffer KV cache for sliding window (memory savings from O(seq_len) to O(W))
- Speculative decoding
- Metal backend (Apple Silicon)
- More scheduling strategies (fair, priority)
- Request preemption and memory sharing
- Benchmarking harness (compare custom vs baseline, memory profiling, report generation)
- Researcher DX: documentation, examples, "implement a paper in an afternoon"
- Deployment / containerization

---

## Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Rust | Macros, type safety, performance, no GIL |
| Tensor typing | Hardware in type | Prevent mixing at compile time |
| TP approach | Runtime (NCCL + `ShardConfig`) | Simple, proven, 2 sync points per layer |
| Op granularity | Traits per operation | Swappable implementations |
| Graph optimization | Macro-based | Code is source of truth, graph derived |
| KV cache | Paged blocks | Dynamic allocation, memory-efficient batching |
| Tokenization | In core crate | Trait in `infernum`, consumed by runtime |
| Error handling | Panics (for now) | Simpler, bugs should crash |
| API | OpenAI-compatible | Industry standard |
| First model | Llama 3.2 1B | Small, well-documented, covers most architectures |
