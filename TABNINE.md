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

### Supported Model Families

Llama/Mistral/Mixtral are loaded via `infernum-llama` (architecturally identical, type aliases for API clarity). Qwen models are loaded via `infernum-qwen`. DeepSeek V3/R1 are loaded via `infernum-deepseek`. Gemma models use `infernum-gemma`.

| Family | `model_type` | Crate | Architecture | Notes |
|--------|-------------|-------|--------------|-------|
| Llama | `llama` | `infernum-llama` | Dense | Llama 2, Llama 3, SmolLM2, CodeLlama, etc. |
| Mistral | `mistral` | `infernum-llama` | Dense | Mistral v1/v2/v3, Devstral (code fine-tune) |
| Mixtral | `mixtral` | `infernum-llama` | MoE | Mixtral 8x7B, 8x22B, etc. |
| Qwen | `qwen2` / `qwen3` / `qwen3_moe` | `infernum-qwen` | Dense / MoE | Qwen2/2.5, Qwen3, Qwen3-MoE |
| DeepSeek | `deepseek_v3` | `infernum-deepseek` | MLA + MoE | DeepSeek-V3, DeepSeek-R1 |
| Gemma 2 | `gemma2` | `infernum-gemma` | Dense | Gemma 2 2B/9B/27B |
| Gemma 3 | `gemma3_text` | `infernum-gemma` | Dense | Gemma 3 1B/4B/12B/27B (text decoder) |

### Sliding Window Attention

Sliding window attention (SWA) restricts each query position to attend only to the most recent `W` key positions, instead of the full causal history. This is used by Mistral, Qwen3, and Qwen3-MoE models.

**Current implementation (Phase 1 — mask-only):** The full KV cache is kept, but the attention kernels clamp the iteration window. Functionally correct, no memory savings yet.

**Config fields** (in both `LlamaConfig` and `QwenConfig`):

- `sliding_window: Option<usize>` — window size `W`
- `use_sliding_window: bool` — whether SWA is enabled
- `max_window_layers: Option<usize>` — SWA applies only to layers `[0, max_window_layers)`. Layers at or above this index use full causal attention.

**Helper:** `config.effective_sliding_window(layer_idx) -> Option<usize>` returns the window size for a given layer, or `None` for full attention.

**Phase 2 (planned, separate guide):** Ring-buffer KV cache to reduce memory from O(seq_len) to O(W).

### Planned Crate Structure

```
infernum/               # Core crate (tensor traits, ops, CUDA impls, blocks, weight loading)
infernum-macros/        # Procedural macros (define_block!, etc.)
infernum-deepseek/      # DeepSeek model family (V3, R1 — MLA + MoE)
infernum-llama/         # Llama model family
infernum-qwen/          # Qwen model family (Qwen2/2.5, Qwen3/3.5, Qwen3-MoE)
infernum-gemma/         # Gemma model family (Gemma 2, Gemma 3 text)
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

### Feature Flags

Most CUDA-dependent code is behind feature flags. The `cuda` feature must be enabled to compile or test GPU code.

| Crate | Feature | What it enables |
|---|---|---|
| `infernum` | `cuda` | CUDA tensor, ops, kernels (requires CUDA toolkit) |
| `infernum` | `nccl` | Multi-GPU NCCL support (implies `cuda`) |
| `infernum` | `force-fuse` | Force fused kernels in debug builds |
| `infernum` | `no-fuse` | Disable fused kernels in release builds |
| `infernum-deepseek` | `cuda` | Enables `infernum/cuda` + `infernum-runtime/cuda` |
| `infernum-deepseek` | `nccl` | Multi-GPU (implies `cuda`) |
| `infernum-deepseek` | `integration` | Integration tests — downloads real models (implies `cuda`) |
| `infernum-llama` | `cuda` | Enables `infernum/cuda` + `infernum-runtime/cuda` |
| `infernum-llama` | `nccl` | Multi-GPU (implies `cuda`) |
| `infernum-llama` | `integration` | Integration tests — downloads real models (implies `cuda`) |
| `infernum-qwen` | `cuda` | Enables `infernum/cuda` + `infernum-runtime/cuda` |
| `infernum-qwen` | `nccl` | Multi-GPU (implies `cuda`) |
| `infernum-qwen` | `integration` | Integration tests — downloads real models (implies `cuda`) |
| `infernum-gemma` | `cuda` | Enables `infernum/cuda` + `infernum-runtime/cuda` |
| `infernum-gemma` | `nccl` | Multi-GPU (implies `cuda`) |
| `infernum-gemma` | `integration` | Integration tests — downloads real models (implies `cuda`) |
| `infernum-runtime` | `cuda` | Enables `infernum/cuda` |
| `infernum-runtime` | `nccl` | Multi-GPU (implies `cuda`) |
| `infernum-examples` | `cuda` | All CUDA deps for examples |
| `infernum-examples` | `nccl` | Multi-GPU examples |

```bash
# Run CUDA unit tests (e.g. quantized matmul kernels)
cargo test -p infernum --features cuda

# Run integration tests (downloads models, needs GPU)
cargo test -p infernum-llama --features integration -- --test-threads=1

# Build examples
cargo build -p infernum-examples --features cuda --examples
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

Integration tests live in each model crate's `tests/model_integration.rs`. They download real models from HuggingFace and verify end-to-end generation output. They are gated behind the `integration` feature so they don't run during normal `cargo test`.

**Running integration tests:**

```bash
# Llama/Mixtral/Mistral family
cargo test -p infernum-llama --features integration -- --test-threads=1

# Qwen family
cargo test -p infernum-qwen --features integration -- --test-threads=1

# DeepSeek family
cargo test -p infernum-deepseek --features integration -- --test-threads=1

# Gemma family
cargo test -p infernum-gemma --features integration -- --test-threads=1
```

Models are cached in `~/.cache/infernum/models/` so subsequent runs are fast.

**What's tested (infernum-llama):**

- **SmolLM2-360M** (SafeTensors f32) — greedy generation correctness, no NaN/Inf in logits
- **Llama-3.2-1B-Instruct FP8** (compressed-tensors) — no NaN/Inf in logits (generation quality test ignored for now)
- **Llama-3.2-1B GPTQ** (GPTQ INT4, group_size=128) — greedy generation correctness, no NaN/Inf in logits
- **Mixtral-tiny** (`jamesdborin/tiny-mixtral`, 2-layer MoE, ~988MB f32) — MoE loading/routing plumbing, no NaN/Inf in logits (random weights, no quality check)
- GGUF integration tests are planned but not yet added

**What's tested (infernum-qwen):**

- **Qwen2.5-0.5B** (`Qwen/Qwen2.5-0.5B`, ~987MB bf16) — greedy generation correctness ("Paris"), no NaN/Inf. Tests Q/K/V bias and tied embeddings.
- **Qwen3-0.6B** (`Qwen/Qwen3-0.6B`, ~1.2GB bf16) — greedy generation correctness ("Paris"), no NaN/Inf. Tests QK-norm (RMSNorm on Q/K per-head before RoPE) and explicit `head_dim` override.
- **Qwen3-MoE-tiny** (`yujiepan/qwen3-moe-tiny-random`, ~5MB, random weights) — MoE loading/routing plumbing, no NaN/Inf. Tests `decoder_sparse_step` (mixed dense/MoE layers), 8 experts top-2.

**What's tested (infernum-deepseek):**

- **DeepSeek-V3-tiny** (`yujiepan/deepseek-v3-tiny-random`, ~8.8MB, random weights) — MLA pipeline (Q LoRA compression, joint KV projection, interleaved RoPE, V padding), sigmoid MoE routing with bias correction and grouped top-k, shared expert, dense→MoE layer transition (`first_k_dense_replace=1`). No NaN/Inf.

**What's tested (infernum-gemma):**

- **Gemma 2 tiny** (`yujiepan/gemma-2-tiny-random`, ~2MB f32, random weights) — architecture plumbing, no NaN/Inf. Tests 4 norms/layer, GeGLU, embedding scaling, attention/final logit soft-capping, alternating sliding/full attention.
- **Gemma 3 text tiny** (`katuni4ka/tiny-random-gemma3-text`, ~2MB bf16, random weights) — architecture plumbing, no NaN/Inf. Tests QK-norm, dual-theta RoPE, sliding_window_pattern auto-generation, no soft-capping.

**Ignored (large model) tests:**

Some integration tests are marked `#[ignore]` because they require very large model downloads and GPU memory. They are skipped by default and must be run manually:

```bash
# Run all ignored tests
cargo test -p infernum-llama --features integration -- --ignored --test-threads=1

# Run a specific ignored test module
cargo test -p infernum-llama --features integration -- --ignored --test-threads=1 mixtral_2x7b
```

Currently ignored:

- **Mistral-7B-Instruct-v0.3** (`mistralai/Mistral-7B-Instruct-v0.3`, ~14.5GB bf16, 3 sharded SafeTensors) — validates `model_type: "mistral"` loads correctly via `MistralModel` alias. Requires ~30GB VRAM (loaded as f32).
- **laser-dolphin-mixtral-2x7b-dpo** (`macadeliccc/laser-dolphin-mixtral-2x7b-dpo`, ~24GB bf16, 3 sharded SafeTensors) — validates MoE generation quality with real weights. Requires ~48GB VRAM (loaded as f32); fits on a single A100 80GB.
- **Gemma 2 2B** (`google/gemma-2-2b`, gated, ~5GB bf16) — validates Gemma 2 generation quality. Requires HF auth and ~10GB VRAM.
- **Gemma 3 1B** (`google/gemma-3-1b-it`, gated, ~2GB bf16) — validates Gemma 3 generation quality. Requires HF auth and ~4GB VRAM.

**Writing new integration tests:**

- Add a new `mod` block in `model_integration.rs` following the existing pattern (const `REPO`, `model_dir()` helper, test functions).
- Use `download_model(repo_id)` to fetch model files — it handles caching automatically.
- For sharded models, use `download_model_files(repo_id, &["file1", "file2", ...])` with an explicit file list.
- Use `generate_greedy()` for deterministic generation tests.
- Use `#[ignore = "reason"]` for tests that are too large for CI but validate quality with real weights.
- Only use ungated HuggingFace models (no auth tokens required).

### Examples

The `infernum-examples` crate contains standalone binaries demonstrating usage and extension of Infernum. All require `--features cuda`. See `infernum-examples/README.md` for full details including llama.cpp comparison instructions.

**Key examples:**

| Example | Purpose | Usage |
|---------|---------|-------|
| `generate` | Text generation with configurable sampling (SafeTensors + GGUF) | `cargo run --example generate --features cuda -- -m /path/to/model "Hello"` |
| `bench` | Decode throughput measurement (greedy, KV cache) | `cargo run --release --example bench --features cuda -- /path/to/model 128` |
| `generate_parallel` | Multi-GPU text generation (tensor parallelism) | `cargo run --example generate_parallel --features nccl -- -m /path/to/model --gpus 2 "Hello"` |
| `verify_parallel` | Correctness check: single-GPU vs multi-GPU output match | `cargo run --example verify_parallel --features nccl -- -m /path/to/model --gpus 2` |
| `custom_cuda_op` | Adding a custom CUDA C kernel (write .cu → nvcc → PTX → launch) | `cargo run --example custom_cuda_op --features cuda` |
| `custom_triton_op` | Adding a custom Triton kernel (Python → PTX → launch) | `cargo run --example custom_triton_op --features cuda` |

**`bench` flags:** `--pool` (buffer pool), `--graphs` (CUDA graphs), `--dtype bf16` (cast weights).

**Adding new examples:**

- Add a new `.rs` file in `infernum-examples/examples/`.
- If it needs a custom kernel, add the `.cu` file to `infernum-examples/kernels/` and update `infernum-examples/build.rs`.
- Follow the existing pattern: use `clap::Parser` for CLI args, gate with `#![cfg(feature = "cuda")]`.

### Benchmarking

The `bench_comparison.sh` script compares infernum vs llama.cpp decode throughput on Llama 3.2 1B across multiple quantization formats. It outputs a Markdown table to stdout.

**Prerequisites:**

- CUDA GPU with `nvidia-smi`
- llama.cpp built at `/home/amir/llama.cpp/` (with `llama-bench`, `llama-quantize`)
- Rust toolchain (`cargo`)
- `hf` CLI with token: `pip install 'huggingface_hub[cli]'`
- Python 3 with `torch`, `transformers`, `gguf` packages

**Usage:**

```bash
# Run all format benchmarks
./bench_comparison.sh

# Run only specific formats (comma-separated, case-insensitive)
./bench_comparison.sh --tests fp8,gptq-int4

# Dry run — show plan without executing
./bench_comparison.sh --dry-run

# Combine flags
./bench_comparison.sh --dry-run --tests q8,q4
```

**Available test names:** `f32`, `bf16`, `f16`, `fp8`, `q8`, `q4`, `gptq-int4`, `all` (default).

Only the models and GGUF conversions needed for the selected tests are downloaded/generated. Results are cached so repeated runs are fast.

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
5. As a final step, consider whether the work warrants:
   - **Integration tests** — if it affects model output, adds a new weight format, or changes generation behavior.
   - **Examples** — if it adds a new user-facing capability that benefits from an interactive demo (e.g., a new inference mode, a new kernel type, a new model family).
   If so, add a step to the guide.
6. After all steps are complete, create a PR. **Never merge the PR** — only the human reviews and merges.
