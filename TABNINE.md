# Infernum

## Project Overview

Infernum is a Rust-based LLM inference server designed to be researcher-friendly, AI-assistant friendly, type-safe, and composable. The name comes from Latin (meaning "inferno").

### Core Goals

1. **Researcher-friendly**: Easy integration of new inference ideas with fast benchmarking. Reduce "time from paper to running inference."
2. **AI-assistant friendly**: Clean, readable codebase with consistent patterns. Given a Python implementation of some new technique, an AI coding assistant should be able to translate it to Infernum easily.
3. **Type-safe**: Use Rust's type system to enforce correctness at compile time — prevent mixing tensors from different hardware, ensure parallelism-aware ops are used in multi-GPU setups.
4. **Composable**: Building blocks that can be mixed and matched. Researchers can swap out one attention implementation for another without touching the rest of the model.

### Architecture

The project follows a layered abstraction hierarchy:

```
Op      → Atomic compute (matmul, softmax, rope, etc.)
Block   → Composable architecture unit (attention, FFN)  
Model   → Full model (Llama, Qwen, etc.)
Runtime → Scheduling, batching, KV cache, tokenization, text in/out
Server  → HTTP, OpenAI-compatible API
```

Key design patterns:
- **Tensor as a Trait**: Hardware is encoded in the type (e.g., `CudaTensor`, `MetalTensor`, `Parallel<T>` for multi-GPU)
- **Ops as Traits**: Each operation is a trait with different implementations for different hardware/algorithms
- **Macro-based Graph Optimization**: `define_block!` macros generate both executable code and graph representation for fusion

### Current Status

The project is in early development (Phase 1). Currently contains a minimal skeleton with a placeholder `hello()` function.

### Planned Crate Structure

```
infernum/               # Core crate (tensor traits, ops, CUDA impls, blocks, weight loading)
infernum-macros/        # Procedural macros (define_block!, etc.)
infernum-llama/         # Llama model family
infernum-qwen/          # Qwen model family (later)
infernum-phi/           # Phi model family (later)
infernum-runtime/       # Execution runtime (scheduler, KV cache, batching)
infernum-serve/         # HTTP server (Axum, OpenAI-compatible API)
```

## Building and Running

### Prerequisites

- Rust stable toolchain (managed via `rust-toolchain.toml`)
- Components: `clippy`, `rustfmt`

### Commands

```bash
# Build the project
cargo build

# Run tests
cargo test --all

# Check code with Clippy (pedantic mode, as per CI)
cargo clippy -- -W clippy::pedantic

# Check formatting
cargo fmt --all -- --check

# Format code
cargo fmt --all
```

## Development Conventions

### Code Style

- **Linting**: Clippy with pedantic warnings enabled (`-W clippy::pedantic`)
- **Formatting**: Standard `rustfmt` formatting, enforced in CI
- **Edition**: Rust 2021

### Error Handling

- **Development phase**: Panic on programmer errors (shape mismatches, invariant violations). These are bugs that should crash with good error messages.
- **Later**: Use `Result` for runtime errors (OOM, CUDA failure). Keep panics for programmer errors.

### Testing

- Unit tests are placed in `#[cfg(test)]` modules within source files
- Run all tests with `cargo test --all`

### CI Pipeline

The GitHub Actions CI pipeline runs on every push/PR to `main`:
1. Clippy check with pedantic warnings
2. Format check
3. All tests

### Design Principles

- Code is always the source of truth; graphs are derived from it
- Tensors from different hardware are different types (compile-time safety)
- Ops that don't support tensor parallelism won't compile with `Parallel<T>` (type system enforcement)
- Prefer composition through blocks; fusion happens at the optimization layer

### Documentation

- Design document: `docs/initial-plan.md` (comprehensive architecture and roadmap)
