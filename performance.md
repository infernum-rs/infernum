# Infernum Performance Measurement

This document describes the methodology and specific instructions for measuring inference performance across all compute targets: CPU, Metal (Apple Silicon), and CUDA (NVIDIA GPU). The goal is consistent, comparable measurements that can track improvement over time.

---

## Concepts

### Prefill vs Decode

Every inference request has two phases with distinct performance characteristics:

- **Prefill** (prompt processing): All prompt tokens are processed in a single forward pass. Work scales with `O(seq_len²)` in attention and `O(seq_len)` in the FFN. Compute-bound. Measured in **tok/s**.

- **Decode** (token generation): One token generated per step, attending over the accumulated KV cache. Bandwidth-bound — weights are read once per generated token. Measured in **tok/s**.

These phases have different bottlenecks and must always be measured independently.

### Why Measure All Model Families

Infernum implements separate kernel paths per model family:

- **Llama/Mistral/Mixtral** (`infernum-llama`) — dense GQA attention, SwiGLU FFN, RoPE
- **Qwen** (`infernum-qwen`) — adds QK-norm, optional Q/K/V bias, tied embeddings, MoE (Qwen3-MoE)
- **DeepSeek** (`infernum-deepseek`) — MLA compressed KV projection, interleaved RoPE, sigmoid MoE with bias correction
- **Gemma** (`infernum-gemma`) — GeGLU, embedding scaling, soft-capping, alternating SWA/full attention

Each family exercises different kernels. A regression in one family's matmul or attention path will not be caught by benchmarking only Llama. **Every target should measure every supported family.**

### KV Cache Fill Level

Decode latency grows with KV cache depth. The current bench tools use an 8-token warm-up prompt, measuring near-empty-cache throughput. For more realistic numbers, also run with a long prompt (512–2048 tokens) to measure decode throughput at a realistic fill level.

### Short vs Long Context

| Configuration | Prompt length | Decode steps | Captures |
|--------------|--------------|-------------|---------|
| Short | 8 tokens | 256 | Cold-cache decode; weight bandwidth |
| Long | 2048 tokens | 128 | Prefill throughput; warm-cache decode |

### What to Report

For every target/hardware/model/format combination:

- **Prefill throughput** (tok/s)
- **Decode throughput** (tok/s)
- **Hardware** — full description (GPU model, CPU + core count, Mac chip + memory)
- **Infernum commit** and **llama.cpp commit**
- **Date**
- **Ratio** — infernum ÷ llama.cpp (so > 1.0 means we are faster)

---

## Target: CPU

### Overview

CPU inference is bandwidth-bound for decode (DRAM weight reads) and compute-bound for prefill. Family coverage matters on CPU: the matmul kernel path is shared, but attention variants (GQA vs MLA, SWA, soft-capping) and FFN variants (SwiGLU vs GeGLU vs MoE gating) are family-specific.

### Model Family Matrix

| Family | Model | Format | Notes |
|--------|-------|--------|-------|
| Llama | SmolLM2-360M | GGUF Q4_0, Q8_0, F32 | Primary regression model; ungated, small |
| Llama | Llama-3.2-1B | GGUF Q4_0, Q8_0 | Standard reference size |
| Qwen | Qwen2.5-0.5B | SafeTensors F32 | Smallest ungated Qwen; tests Q/K/V bias and tied embeddings |
| Gemma | Gemma-2-2B (tiny random) | SafeTensors F32 | Tests GeGLU, soft-capping, alternating SWA |
| DeepSeek | DeepSeek-V3-tiny (random) | SafeTensors F32 | MLA path; random weights are fine for throughput |

For regular development, SmolLM2-360M and Qwen2.5-0.5B are sufficient. The others catch family-specific regressions and should be run before any significant kernel change.

Larger models (e.g., Llama-3.1-8B Q4) take too long for routine benchmarking on CPU and are only useful for thread-scaling experiments.

### Dtypes

- **F32**: correctness baseline; use for non-Llama families (native format)
- **GGUF Q8_0**: best quality/speed tradeoff; primary Llama CPU format
- **GGUF Q4_0**: maximum decode throughput on memory-limited machines

BF16 is not prioritized for CPU — most x86 hardware lacks native BF16 multiply-accumulate support.

### How to Run

```bash
# Build once
cargo build --release --example bench_cpu --features cpu

# Short context decode
cargo run --release --example bench_cpu --features cpu -- /path/to/model.gguf 256

# Short context prefill
cargo run --release --example bench_cpu --features cpu -- --graph /path/to/safetensors 256

# Long context prefill (measures prompt processing throughput)
cargo run --release --example bench_cpu --features cpu -- --graph /path/to/safetensors 2048

# Thread count override
cargo run --release --example bench_cpu --features cpu -- -j 8 /path/to/model.gguf 256

# Full comparison table vs llama.cpp (Llama family only)
./bench_cpu_comparison.sh
./bench_cpu_comparison.sh --tests q8,q4
./bench_cpu_comparison.sh --threads 8 --n-tokens 256
```

The comparison script only covers the Llama family today. Non-Llama families need to be run manually via `bench_cpu`.

### Checklist

- [ ] Run at both all-cores and a fixed count (e.g., 8) to separate per-thread efficiency from scaling
- [ ] Cover all four families (Llama, Qwen, Gemma, DeepSeek)
- [ ] Measure both decode and prefill
- [ ] Record `lscpu | grep "Model name"` and thread count

---

## Target: Metal (Apple Silicon)

### Overview

Metal uses the GPU on Apple Silicon Macs. Unified memory eliminates explicit host-device transfer; bottlenecks are GPU bandwidth and Metal shader efficiency. The gap vs llama.cpp on Metal is currently large.

The most interesting Metal use case is **large models on high-memory Apple Silicon** — Macs with M-series Ultra chips can run 70B parameter models in BF16 that require a 4-GPU server on CUDA. This is a real differentiator worth optimizing for specifically.

**Current limitation:** `bench_metal` measures decode throughput only. Prefill measurement requires a Metal graph engine (not yet implemented).

### Hardware Tiers

| Tier | Chip | Unified Memory | What fits (BF16) |
|------|------|---------------|-----------------|
| Standard | M3 Pro / M4 Pro | 24–48 GB | Up to ~7B BF16, 13B Q4 |
| High | M3 Max / M4 Max | 64–128 GB | Up to ~30B BF16, 65B Q4 |
| Ultra | M3 Ultra / M4 Ultra | 192 GB+ | Up to ~70B BF16, 405B Q4 (model-dependent) |

Benchmarks should state chip model and memory. The Ultra tier (70B+ in BF16) is the primary performance target.

### Model Family Matrix

Test all families on each available hardware tier.

**All tiers (small models, ungated, < 2GB):**

| Family | Model | Format |
|--------|-------|--------|
| Llama | SmolLM2-360M | GGUF Q4_0, Q8_0, F32 |
| Qwen | Qwen3-0.6B | SafeTensors BF16 |
| Gemma | Gemma-2-2B | SafeTensors BF16 (or GGUF) |
| DeepSeek | DeepSeek-V3-tiny (random) | SafeTensors BF16 |

**High tier (64–128 GB):**

| Family | Model | Format | Memory (approx) |
|--------|-------|--------|----------------|
| Llama | Llama-3.1-8B | BF16 SafeTensors | ~16 GB |
| Llama | Llama-3.1-8B | GGUF Q4_0 | ~4.5 GB |
| Qwen | Qwen3-8B | BF16 SafeTensors | ~16 GB |
| Gemma | Gemma-3-12B | BF16 SafeTensors | ~24 GB |

**Ultra tier (192 GB+):**

| Family | Model | Format | Memory (approx) |
|--------|-------|--------|----------------|
| Llama | Llama-3.1-70B | BF16 SafeTensors | ~140 GB |
| Llama | Llama-3.1-70B | GGUF Q4_0 | ~40 GB |
| Llama | Llama-3.1-70B | GGUF Q8_0 | ~70 GB |
| Qwen | Qwen3-72B | BF16 SafeTensors | ~144 GB |
| Qwen | Qwen3-32B | BF16 SafeTensors | ~64 GB |
| Gemma | Gemma-3-27B | BF16 SafeTensors | ~54 GB |
| Mixtral | Mixtral-8x7B | GGUF Q4_0 | ~26 GB |

Llama-3.1-70B BF16 on an M3/M4 Ultra is the headline number — it represents a workload that fits on a single consumer machine but requires a multi-GPU server on CUDA.

### Dtypes

- **GGUF Q4_0**: primary throughput target; most common user format
- **GGUF Q8_0**: quality reference
- **BF16 SafeTensors**: native quality; the right format for families with native SafeTensors support

### How to Run

```bash
# Build once
cargo build --release --example bench_metal --features metal

# Decode throughput
cargo run --release --example bench_metal --features metal -- /path/to/model 256

# With per-kernel profiling (separate run — ~10% slower, use only to diagnose bottlenecks)
cargo run --release --example bench_metal --features metal -- --profile /path/to/model 256

# Full comparison table vs llama.cpp (SmolLM2-360M, Llama only)
./bench_metal_comparison.sh
./bench_metal_comparison.sh --tests q4,q8
./bench_metal_comparison.sh --n-gen 256

# Manual run for non-Llama families
cargo run --release --example bench_metal --features metal -- /path/to/qwen3-72b 256
cargo run --release --example bench_metal --features metal -- /path/to/gemma-3-27b 256
```

The comparison script covers Llama family only. Non-Llama families require manual runs.

### Checklist

- [ ] Record chip model and memory (`system_profiler SPDisplaysDataType | grep -E "Chipset|VRAM"`)
- [ ] Cover all four families (Llama, Qwen, Gemma, DeepSeek/MLA path)
- [ ] For the Ultra tier: include 70B BF16 decode throughput — this is the headline metric
- [ ] Use `--profile` only on a dedicated diagnostic run, not in reported results
- [ ] Compare all formats against llama.cpp `-ngl 99`
- [ ] Once Metal prefill graph is implemented, add prefill measurements

---

## Target: CUDA (NVIDIA GPU)

### Overview

CUDA has the most complete optimization stack: paged KV cache, CUDA graph capture, cuBLAS/cuBLASLt. Decode is the primary CUDA graph target; prefill is the primary kernel fusion target.

CUDA benchmarks are split across two hardware tiers that correspond to meaningfully different model scales.

### Hardware Tiers

#### Single L4 (24 GB VRAM)

The L4 is the standard development and small-deployment GPU. Its 24 GB limits which models fit at native dtype.

Models that fit in BF16 (≤ ~20 GB budget, leaving headroom for KV cache):

| Family | Model | Format | VRAM (approx) |
|--------|-------|--------|--------------|
| Llama | Llama-3.2-1B | BF16 | ~2 GB |
| Llama | Llama-3.2-1B | FP8, GPTQ INT4, GGUF Q4_0, Q8_0 | < 2 GB |
| Llama | Llama-3.1-8B | BF16 | ~16 GB |
| Llama | Llama-3.1-8B | GPTQ INT4, GGUF Q4_0 | ~4–5 GB |
| Mistral | Mistral-7B-v0.3 | BF16 | ~14 GB |
| Qwen | Qwen3-0.6B | BF16 | ~1.2 GB |
| Qwen | Qwen2.5-7B | BF16 | ~14 GB |
| Qwen | Qwen3-8B | BF16 | ~16 GB |
| Gemma | Gemma-2-9B | BF16 | ~18 GB |
| Gemma | Gemma-3-12B | BF16 | ~24 GB (tight) |
| DeepSeek | DeepSeek-V3-tiny (random) | BF16 | ~9 MB (sanity only) |

Mixtral and large MoE models do not fit on a single L4 at any usable dtype.

**Primary models for the L4:** Llama-3.1-8B BF16 (exercises full-size dense decode) and Qwen3-8B BF16 (exercises Qwen path). Llama-3.2-1B across all formats covers the quantization matrix.

#### 8× H100 80 GB (640 GB total VRAM)

The H100 cluster enables tensor-parallel inference over large models. All models below assume tensor parallelism with `--nccl` features.

| Family | Model | Format | VRAM (approx) | TP needed |
|--------|-------|--------|--------------|-----------|
| Llama | Llama-3.1-70B | BF16 | ~140 GB | TP=2 |
| Llama | Llama-3.1-70B | GPTQ INT4 | ~35 GB | TP=1 |
| Llama | Llama-3.1-405B | GPTQ INT4 | ~200 GB | TP=4 |
| Qwen | Qwen3-72B | BF16 | ~144 GB | TP=2 |
| Qwen | Qwen3-235B-A22B (MoE) | BF16 | ~470 GB | TP=8 |
| Qwen | Qwen3-235B-A22B (MoE) | GPTQ INT4 | ~120 GB | TP=2 |
| Mixtral | Mixtral-8x22B (MoE) | BF16 | ~282 GB | TP=4 |
| Mixtral | Mixtral-8x7B (MoE) | BF16 | ~92 GB | TP=2 |
| DeepSeek | DeepSeek-V3 (MoE) | GGUF Q4_0 | ~335 GB | TP=4–8 |
| DeepSeek | DeepSeek-R1 (MoE) | GGUF Q4_0 | ~335 GB | TP=4–8 |
| Gemma | Gemma-3-27B | BF16 | ~54 GB | TP=1 |

DeepSeek-V3/R1 BF16 does not fit on 8× H100 (~1.3 TB). GGUF Q4 (~335 GB) fits on TP=4-8 and is the recommended format for these models.

**Primary H100 targets:** Llama-3.1-70B BF16 (dense baseline), Qwen3-235B-A22B BF16 (MoE + Qwen path), DeepSeek-V3 Q4 (MLA + large MoE).

### Dtypes (CUDA)

| Format | Models | Notes |
|--------|--------|-------|
| BF16 SafeTensors | All families | Native precision; primary quality reference |
| FP8 SafeTensors | Llama family | compressed-tensors format |
| GGUF Q8_0 | Llama, Qwen | High quality, ~2× size reduction |
| GGUF Q4_0 | Llama, Qwen, DeepSeek | Primary throughput format for large models |
| GPTQ INT4 | Llama, Qwen | Production quantization |
| AWQ INT4 | Llama, Qwen | Alternative production quantization |

F32 is not measured on CUDA — no production workload uses it.

### How to Run

```bash
# Build (single GPU)
cargo build --release --example bench --features cuda

# Build (multi-GPU with NCCL)
cargo build --release --example bench --features nccl

# Decode — production CUDA graph engine path
cargo run --release --example bench --features cuda -- --cuda-graph-engine /path/to/model 256

# Prefill — graph executor
cargo run --release --example bench --features cuda -- --graph /path/to/model 256

# Specific dtype for SafeTensors
cargo run --release --example bench --features cuda -- --cuda-graph-engine --dtype bf16 /path/to/model 256

# Multi-GPU decode (use generate_parallel example for TP)
cargo run --release --example generate_parallel --features nccl -- -m /path/to/model --gpus 8 "Hello"

# Full comparison table vs llama.cpp (Llama-3.2-1B, single GPU)
./bench_comparison.sh
./bench_comparison.sh --tests bf16,gptq-int4
./bench_comparison.sh --dry-run   # preview

# Non-Llama families (manual, no comparison script yet)
cargo run --release --example bench --features cuda -- --cuda-graph-engine /path/to/qwen3-8b 256
cargo run --release --example bench --features cuda -- /path/to/gemma-2-9b 256
cargo run --release --example bench --features cuda -- /path/to/deepseek-v3-q4.gguf 256
```

The CUDA graph engine path warms up during the prefill phase. The reported tok/s covers only the decode steps.

### Family Coverage Matrix (CUDA)

This is the minimum family coverage for a complete benchmark run. A ✓ means the family must be measured. Use the smallest model in the family that runs on the available hardware.

| Family | L4 (representative model) | H100×8 (representative model) |
|--------|--------------------------|-------------------------------|
| Llama | Llama-3.1-8B BF16 | Llama-3.1-70B BF16 |
| Mistral | Mistral-7B-v0.3 BF16 | — (covered by Llama path) |
| Mixtral/MoE | — (doesn't fit) | Mixtral-8x22B BF16 |
| Qwen | Qwen3-8B BF16 | Qwen3-235B-A22B BF16 |
| DeepSeek | DeepSeek-V3-tiny (smoke test) | DeepSeek-V3 Q4 |
| Gemma | Gemma-2-9B BF16 | Gemma-3-27B BF16 |

### Checklist

- [ ] Run `--cuda-graph-engine` for decode (not legacy `--graphs`)
- [ ] Cover all six families (Llama, Mistral, Mixtral, Qwen, DeepSeek, Gemma)
- [ ] Compare all families against llama.cpp — a family that is 2× slower than Llama in ratio terms indicates a family-specific kernel problem
- [ ] Record `nvidia-smi` output: GPU model, VRAM total, driver version
- [ ] For multi-GPU: record which TP degree was used

---

## Cross-Target Comparisons

Use SmolLM2-360M GGUF Q8_0 decode throughput as the single cross-target anchor — it is the only model that can run on all three targets without special hardware. This lets you directly compare CPU vs Metal vs CUDA for the same workload.

For cross-family comparisons within a target: normalize by model parameter count (tok/s per billion parameters) to make different-size models comparable. A 7B Qwen model should be roughly comparable to a 7B Llama model at the same arithmetic intensity.

---

## What We Are Not Measuring (Phase 1)

- **Batching / concurrency**: Single-request throughput only. Server throughput (multiple concurrent requests) is a separate effort.
- **Sampling overhead**: All benchmarks use greedy argmax. Sampling is negligible at this scale.
- **First-token latency (TTFT)**: Captured indirectly by prefill throughput for short prompts but not isolated.
- **Memory usage / OOM boundaries**: Tracked informally but not systematically.
- **Continuous batching**: Not implemented yet.
- **Multi-GPU Metal**: Apple Silicon multi-die setups (Mac Pro) are not supported.

---

## Baseline Reference (fill before starting optimization)

Run the comparison scripts and manual bench commands, then fill in this table. This is the before-state; all optimization work is measured against it.

### CPU (SmolLM2-360M, all-cores, lscpu: _______)

| Format | Op | Infernum (tok/s) | llama.cpp (tok/s) | Ratio |
|--------|----|-----------------|-------------------|-------|
| GGUF Q4_0 | Decode | | | |
| GGUF Q8_0 | Decode | | | |
| GGUF F32 | Decode | | | |
| GGUF Q8_0 | Prefill | | | |

### Metal (chip: _____, memory: _____ GB)

#### SmolLM2-360M

| Format | Family | Infernum (tok/s) | llama.cpp (tok/s) | Ratio |
|--------|--------|-----------------|-------------------|-------|
| GGUF Q4_0 | Llama | | | |
| GGUF Q8_0 | Llama | | | |
| GGUF F32 | Llama | | | |
| BF16 | Qwen | | | |
| BF16 | Gemma | | | |

#### Llama-3.1-70B (Ultra tier only)

| Format | Infernum (tok/s) | llama.cpp (tok/s) | Ratio |
|--------|-----------------|-------------------|-------|
| BF16 | | | |
| GGUF Q4_0 | | | |
| GGUF Q8_0 | | | |

### CUDA — Single L4

| Family | Model | Format | Op | Infernum (tok/s) | llama.cpp (tok/s) | Ratio |
|--------|-------|--------|----|-----------------|-------------------|-------|
| Llama | Llama-3.2-1B | BF16 | Decode | | | |
| Llama | Llama-3.2-1B | BF16 | Prefill | | | |
| Llama | Llama-3.2-1B | GPTQ INT4 | Decode | | | |
| Llama | Llama-3.2-1B | GGUF Q4_0 | Decode | | | |
| Llama | Llama-3.1-8B | BF16 | Decode | | | |
| Llama | Llama-3.1-8B | BF16 | Prefill | | | |
| Mistral | Mistral-7B-v0.3 | BF16 | Decode | | | |
| Qwen | Qwen3-0.6B | BF16 | Decode | | | |
| Qwen | Qwen3-8B | BF16 | Decode | | | |
| Qwen | Qwen2.5-7B | BF16 | Decode | | | |
| Gemma | Gemma-2-9B | BF16 | Decode | | | |
| DeepSeek | DeepSeek-V3-tiny | BF16 | Decode | n/a (random) | n/a | — |

### CUDA — 8× H100 (GPU: H100 80GB SXM, TP=_____)

| Family | Model | Format | Op | TP | Infernum (tok/s) | llama.cpp (tok/s) | Ratio |
|--------|-------|--------|----|-----|-----------------|-------------------|-------|
| Llama | Llama-3.1-70B | BF16 | Decode | 2 | | | |
| Llama | Llama-3.1-70B | BF16 | Prefill | 2 | | | |
| Qwen | Qwen3-72B | BF16 | Decode | 2 | | | |
| Qwen | Qwen3-235B-A22B | BF16 | Decode | 8 | | | |
| Mixtral | Mixtral-8x22B | BF16 | Decode | 4 | | | |
| DeepSeek | DeepSeek-V3 | Q4 | Decode | 4–8 | | | |
| Gemma | Gemma-3-27B | BF16 | Decode | 1 | | | |
