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
- **Tensor as a Trait**: Hardware is encoded in the type (e.g., `CudaTensor`, `MetalTensor`). Multi-GPU uses runtime TP with NCCL.
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

# Run tests (no CUDA)
cargo test --all

# Run tests (with CUDA) — must use --test-threads=1 to avoid
# CUDA context conflicts between parallel test threads
cargo test --features cuda -- --test-threads=1

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
- Multi-GPU tensor parallelism via runtime NCCL all-reduce (2 sync points per transformer layer)
- Prefer composition through blocks; fusion happens at the optimization layer

### Documentation

- Design document: `docs/plan.md` (architecture and roadmap)
- Short-term / ephemeral planning documents go in `ephemeral-docs/` (gitignored)
- Git worktrees go in `worktrees/` (gitignored)

### Integration Tests

Integration tests live in `infernum-llama/tests/model_integration.rs`. They download real models from HuggingFace and verify end-to-end generation output. They are gated behind the `integration` feature so they don't run during normal `cargo test`.

**Running integration tests:**

```bash
cargo test -p infernum-llama --features integration -- --test-threads=1
```

Models are cached in `~/.cache/infernum/models/` so subsequent runs are fast.

**What's tested:**

- **SmolLM2-360M** (SafeTensors f32) — greedy generation correctness, no NaN/Inf in logits
- **Llama-3.2-1B-Instruct FP8** (compressed-tensors) — no NaN/Inf in logits (generation quality test ignored for now)
- GGUF integration tests are planned but not yet added

**Writing new integration tests:**

- Add a new `mod` block in `model_integration.rs` following the existing pattern (const `REPO`, `model_dir()` helper, test functions).
- Use `download_model(repo_id)` to fetch model files — it handles caching automatically.
- Use `generate_greedy()` for deterministic generation tests.
- Use `#[ignore = "reason"]` for tests that document known issues but shouldn't block CI.
- Only use ungated HuggingFace models (no auth tokens required).

### Implementation Guides

An implementation guide is a design and implementation plan for a feature or optimization. It lives in `ephemeral-docs/` and serves as the single source of truth for the work in progress. Structure:

1. **Overview** — What we're building and why.
2. **Steps** — The implementation broken into incremental steps. Each step should be small enough to verify independently.
3. **Progress** — A log updated after each step is completed with results, observations, and any deviations from the plan.

**Workflow:**

1. Create the implementation guide in `ephemeral-docs/`. The guide **must specify the worktree name and branch** (e.g., `worktrees/my-feature` on branch `my-feature`).
2. Create a git worktree for the work: `git worktree add worktrees/<name> -b <branch-name>`.
3. **ALL implementation work happens exclusively in the worktree, never on `main`.** The main worktree is only for creating the guide, merging, and non-implementation tasks.
4. Implement one step at a time in the worktree. After each step:
   - Build: `cargo build --release --features cuda`
   - Clippy: `cargo clippy -p infernum -p infernum-llama --features cuda -- -W clippy::pedantic`
   - Format: `cargo fmt --all -- --check`
   - Tests: `cargo test --all --features cuda -- --test-threads=1`
   - If all pass: commit and push.
   - Update the progress section in the implementation guide.
5. As a final step, consider whether the work warrants new or updated integration tests (e.g., if it affects model output, adds a new weight format, or changes generation behavior). If so, add an integration test step to the guide.
6. After all steps are complete, create a PR and merge.
