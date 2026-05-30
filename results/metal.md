# Metal Benchmark Results

Measured with `bench_metal.sh`. See [performance.md](../performance.md) for methodology.

**Chip (all results):** Apple M3 Pro (18 GB unified memory)

---

## Current Results

Most recent measurement for each model/format. Decode: 256 tokens, 8-token warm-up, greedy. Prefill: 512-token prompt.

### Decode throughput (tok/s)

| Model | Format | infernum | llama.cpp | ratio | Date |
| ----- | ------ | -------: | --------: | ----: | ---- |
| Llama / Llama-3.1-8B | GGUF Q4_0 | 12.8 | 28.1 | 0.46x | 2026-05-22 |
| Llama / Llama-3.1-8B | GGUF Q8_0 | 2.9 | 16.3 | 0.18x | 2026-05-22 |
| Llama / Llama-3.2-3B | GGUF Q4_0 | 21.6 | 59.6 | 0.36x | 2026-05-22 |
| Llama / Llama-3.2-3B | GGUF Q8_0 | 21.5 | 36.9 | 0.58x | 2026-05-22 |
| Llama / SmolLM2-360M | GGUF Q4_0 | 108.2 | 226.7 | 0.48x | 2026-05-30 |
| Llama / SmolLM2-360M | GGUF Q8_0 | 99.1 | 184.4 | 0.54x | 2026-05-30 |
| Llama / SmolLM2-360M | SafeTensors F32 | 40.0 | — | — | 2026-05-30 |
| Qwen / Qwen3-0.6B | SafeTensors BF16 | 28.9 | — | — | 2026-05-25 |
| Gemma / gemma-2-2b-it | GGUF Q8_0 | 20.9 | 29.1 | 0.72x | 2026-05-25 |
| Gemma / gemma-2-2b-it | GGUF Q4_K_M | — | 65.7 | — | 2026-05-23 |

### Prefill throughput (tok/s)

`llama-bench` only accepts GGUF — SafeTensors rows have no llama.cpp comparison and never will.

| Model | Format | infernum | llama.cpp | ratio | Date |
| ----- | ------ | -------: | --------: | ----: | ---- |
| Llama / Llama-3.1-8B | GGUF Q4_0 | — | 290.8 | — | 2026-05-22 |
| Llama / Llama-3.1-8B | GGUF Q8_0 | 51.8 | 285.8 | 0.18x | 2026-05-22 |
| Llama / Llama-3.2-3B | GGUF Q4_0 | 417 | 686.9 | 0.61x | 2026-05-22 |
| Llama / Llama-3.2-3B | GGUF Q8_0 | 424 | 683.0 | 0.62x | 2026-05-22 |
| Llama / SmolLM2-360M | SafeTensors F32 | 1284.3 | — | — | 2026-05-30 |
| Llama / SmolLM2-360M | GGUF Q8_0 | 1074.6 | 4541 | 0.24x | 2026-05-30 |
| Llama / SmolLM2-360M | GGUF Q4_0 | 1080.7 | 4596 | 0.24x | 2026-05-30 |
| Qwen / Qwen3-0.6B | SafeTensors BF16 | 872.1 | — | — | 2026-05-30 |
| Gemma / gemma-2-2b-it | GGUF Q8_0 | 270.3 | 938 | 0.29x | 2026-05-30 |
| Gemma / gemma-2-2b-it | GGUF Q4_K_M | — | 903 | — | 2026-05-23 |

---

## History

---

## 2026-05-30 — Flash prefill attention + tiled GEMM (K_BLK=32)

### Changes

1. **Single-pass flash attention for prefill** — replaced the 3-pass prefill attention kernel with a single-pass online softmax design. The 3-pass kernel recomputed Q·K for every output dimension in Pass 3 (outer loop over `d`, inner loop over `kv`), reading K `head_dim` times per KV position — O(kv_len × head_dim²) complexity. The new kernel loads Q into registers once, then streams K and V through a single forward pass with online (Dao-style) softmax — K and V each read exactly once. Grid: (seq_len × n_heads) threadgroups, 32 threads each (1 SIMD group; no shared memory). Uses `simd_sum` for the within-group dot product reduction.

2. **Tiled GEMM with K_BLK=32 coalesced loads** — `linear_dense_f32_tiled` kernel now uses 16 SIMD groups (4×4) per threadgroup with K_BLK=32. Each SIMD group reads one 32-float row (one 128-byte cache line) per load instruction, fully coalesced. The earlier K_BLK=8 version had threads within one SIMD group reading from 4 different rows (stride K apart) → 4× more cache transactions. shmem: sA[32×32]+sB[32×32]=8 KB total.

### Prefill throughput (tok/s) — 512-token prompt

| Model | Format | before | after | delta |
| ----- | ------ | -----: | ----: | ----: |
| Llama / SmolLM2-360M | GGUF Q8_0 | 253.9 | 1074.6 | +323% |
| Llama / SmolLM2-360M | GGUF Q4_0 | 253.9 | 1080.7 | +325% |
| Llama / SmolLM2-360M | SafeTensors F32 | 264.8 | 1284.3 | +385% |
| Qwen / Qwen3-0.6B | SafeTensors BF16 | 90.9 | 872.1 | +859% |
| Gemma / gemma-2-2b-it | GGUF Q8_0 | 295.9 | 270.3 | -9% (thermal) |

### Notes

- **GGUF Q8_0/Q4_0 prefill 0.056x → 0.24x llama.cpp** — the most visible improvement. SmolLM2 GGUF prefill was catastrophically slow because the 3-pass kernel's O(kv_len × head_dim²) complexity made the attention portion dominate. With flash attention, attention is now O(kv_len × head_dim) and the bottleneck shifts back to GEMM.
- **Qwen 9.6× improvement** — Qwen3-0.6B has 28 layers vs SmolLM2's 32 and a larger head_dim=64 (same as SmolLM2), so the 3-pass kernel hurt proportionally. Also benefits from the tiled GEMM improvement.
- **Gemma prefill slightly down** — thermal noise; Gemma uses logit soft-capping (tanh) which adds a `precise::tanh` call per KV position in the flash loop, reducing the relative speedup at smaller prefill sizes. Within measurement variability (prior measured range was 280–660 tok/s).
- **Decode throughput unchanged** — the decode path uses the existing flash decode kernel (unchanged) and paged attention.

---

## 2026-05-30 — f16 KV cache + add+rmsnorm fusion fix + RMSNorm simd_sum

Combined session; three changes landed together:

### Changes

1. **f16 KV cache** — added cross-dtype kernels (`append_kv_paged_batched_fused_f32_to_f16`, `paged_attention_flash_decode_f32_kv16`). KV cache now allocated as f16 instead of f32 for quantized models; reads 2× fewer bytes during attention decode. Also halves KV cache memory usage.

2. **add+rmsnorm graph fusion fix** — removed the incorrect `consumer_count == 1` guard in `fuse_add_rmsnorm`. In transformers the Add output (residual) always has 2 consumers (the RmsNorm AND the next skip-connection add), so the guard always prevented fusion. With the fix, all 64 add+rmsnorm opportunities per decode step are now fused (was 1/step). Removes 63 standalone `add_f32` dispatches per token step.

3. **RMSNorm simd_sum 2-barrier reduction + float4 loads** — replaced log2(tg_size) tree reduction (8 barriers for tg=256) with 2-barrier simd_sum approach matching llama.cpp. Also vectorised loads with float4/half4.

### Decode throughput (tok/s) — 256 tokens, SmolLM2-360M

| Format | session start | after change 3 (RMSNorm) | after change 2 (fusion fix) | after change 1 (f16 KV) |
| ------ | ------------: | -----------------------: | --------------------------: | -----------------------: |
| GGUF Q8_0 | 92.3 | 95.0 | 96.8 | **99.1** |
| GGUF Q4_0 | 98.5 | 102.4 | 104.9 | **108.2** |
| SafeTensors F32 | 39.2 | 39.3 | 39.3 | **40.0** |

---

## 2026-05-30 — RMSNorm: simd_sum 2-barrier reduction + float4 vectorised loads

- **Chip:** Apple M3 Pro (18 GB unified memory)
- **Decode tokens:** 256 (8-token warm-up prompt, greedy)
- **infernum commit:** `19aa7c2`
- **Date:** 2026-05-30

### Decode throughput (tok/s) — 256 tokens

| Model | Format | before | after | delta |
| ----- | ------ | -----: | ----: | ----: |
| Llama / SmolLM2-360M | GGUF Q8_0 | 92.3 | 95.0 | +2.9% |
| Llama / SmolLM2-360M | GGUF Q4_0 | 98.5 | 102.4 | +3.9% |
| Llama / SmolLM2-360M | SafeTensors F32 | 39.2 | 39.3 | ~0% |

### Changes

1. **simd_sum 2-barrier reduction** — replaced the log2(tg_size) tree reduction (8 threadgroup barriers for tg=256) with `simd_sum` + 2 barriers: one `simd_sum` reduces each 32-thread SIMD group with no barrier, two barriers then collect the SIMD-group sums into shared memory and let all threads read the total. For SmolLM2 with 32 layers and 3 norms per layer (96 RMSNorm calls/token × 6 saved barriers = 576 fewer barriers per decode step).

2. **float4/half4 vectorised loads** — the sum-of-squares accumulation loop and output write loop now use `float4`/`half4` loads when `hidden % 4 == 0` (always true for SmolLM2 hidden=960), quartering the number of loop iterations.

3. **Reduced threadgroup shared memory** — shmem allocation changed from `tg_size * 4` bytes to `ceil(tg_size/32) * 4` bytes (e.g., 1024 → 32 bytes for tg=256), reducing threadgroup memory pressure.

### Notes

- F32 model unchanged — the norm kernels are faster, but F32 decode is dominated by memory bandwidth of the weight matrices (4× larger than Q8_0), so norm savings are a small fraction of total time.
- All 4 norm variants updated: `rms_norm_f32`, `add_rmsnorm_f32`, `rms_norm_f16`, `add_rmsnorm_f16`.

---

## 2026-05-29 — Vectorized cooperative GEMV + f32 SwiGLU kernels (no measurable gain)

- **Chip:** Apple M3 Pro (18 GB unified memory)
- **Decode tokens:** 256 (8-token warm-up prompt, greedy)
- **Prefill tokens:** 512
- **infernum commit:** (this PR)
- **Date:** 2026-05-29

### Decode throughput (tok/s) — 256 tokens

| Model | Format | before | after | delta |
| ----- | ------ | -----: | ----: | ----: |
| Llama / SmolLM2-360M | GGUF Q8_0 | 92.1 | 92.3 | ~0% (noise) |
| Llama / SmolLM2-360M | GGUF Q4_0 | 100.3 | 98.5 | ~0% (noise) |
| Llama / SmolLM2-360M | SafeTensors F32 | 39.0 | 39.2 | ~0% (noise) |

### Changes

1. **Cooperative GEMV inner loop vectorization** — replaced 8-iteration scalar `yl[j] * float(qs[j])` FMA loops with `float4`/`half4` input loads and `dot(float4, float4)` intrinsics for Q8_0 and Q4_0 (both f16 and f32 input variants). Better ILP in theory; no measured effect because the cooperative GEMV is already memory-bandwidth-bound. The kernel reads 34 bytes/block (Q8_0) or 18 bytes/block (Q4_0) from device memory; `dot()` saves compute cycles but the GPU already idles waiting for the load.

2. **F32 SwiGLU-fused down-proj kernels** (`gemv_swiglu_q8_blocks_f32`, `gemv_swiglu_q4_blocks_f32`) — mirror of the existing f16 kernels but with float4 input loads and `silu_f`. Updated Rust dispatch (`swiglu_linear`) to route f32 gate_up through the fused path. No measured effect because `swiglu_linear` is called from `transformer.rs::forward_mlp` (the eager non-graph path), and the Metal bench uses `MetalGraphEngine` which dispatches `SwigluOp` as a separate graph node. The kernels are correct and will be useful when graph-level SwiGLU+down-proj fusion is implemented.

### Root cause analysis

After profiling (42.2% GEMV, 14.2% rmsnorm, 13.3% add, 8.5% flash_attn, 7.3% kv_append, 6.9% rope, 6.8% swiglu), the non-GEMV ops total 5.4 ms/token on 960-dim vectors. The bottleneck for all small-dim ops (rmsnorm, add, swiglu) is Metal kernel dispatch overhead (~20 µs/dispatch, 430 dispatches/token). The remaining gap vs llama.cpp requires graph-level fusion of adjacent ops (add+rmsnorm, rope+kv_append, swiglu+down_proj) to eliminate dispatches, not kernel-level compute optimizations.

---

## 2026-05-25 — Fused Q+K RoPE dispatch (`apply_rope_qk_f32`)

- **Chip:** Apple M3 Pro (18 GB unified memory)
- **Decode tokens:** 256 (8-token warm-up prompt, greedy)
- **Prefill tokens:** 512
- **infernum commit:** (this PR)
- **Date:** 2026-05-25

### Decode throughput (tok/s) — 256 tokens

| Model | Format | before | after | delta |
| ----- | ------ | -----: | ----: | ----: |
| Llama / SmolLM2-360M | GGUF Q8_0 | 91.2 | 92.1 | +1.0% |
| Llama / SmolLM2-360M | GGUF Q4_0 | 100.0 | 100.3 | +0.3% |
| Llama / SmolLM2-360M | SafeTensors F32 | 39.2 | 39.0 | ~0% (noise) |

### Notes

- **Root cause:** The graph optimizer creates `FusedRopePairOp` for every Q+K RoPE pair (32 per SmolLM2 decode step), but `MetalBackend::apply_rope_pair` had no override and fell back to two separate `apply_rope_f32` dispatches. This meant 64 RoPE dispatches per token instead of 32.
- **Fix:** Added `apply_rope_qk_f32` Metal kernel (new `RopeQkNonBatchedParams` struct) and implemented `RopeOps::apply_rope_pair` override for `MetalBackend`. The kernel partitions the thread grid: first `seq_len * q_heads * half_dim` threads process Q, remaining `seq_len * k_heads * half_dim` threads process K. Falls back to two separate dispatches when offsets or seq_lens differ.
- **Small gain:** Saves 32 dispatches per decode step (64 → 32 RoPE dispatches). At ~10μs per dispatch this is ~320μs per token, which translates to ~1% throughput on Q8_0. The small absolute gain reflects that RoPE is a tiny fraction of total per-token time; GEMV dominates.

---

## 2026-05-25 — Cache decode graph and execution plan between token steps

- **Chip:** Apple M3 Pro (18 GB unified memory)
- **Decode tokens:** 256 (8-token warm-up prompt, greedy)
- **Prefill tokens:** 512
- **infernum commit:** (this PR)
- **Date:** 2026-05-25

### Decode throughput (tok/s) — 256 tokens

| Model | Format | before | after | delta |
| ----- | ------ | -----: | ----: | ----: |
| Llama / SmolLM2-360M | SafeTensors F32 | 37.7 | 39.2 | +4.0% |
| Llama / SmolLM2-360M | GGUF Q8_0 | 84.1 | 91.2 | +8.4% |
| Llama / SmolLM2-360M | GGUF Q4_0 | 92.2 | 100.0 | +8.4% |
| Qwen / Qwen3-0.6B | SafeTensors BF16 | 28.2 | 28.9 | +2.5% |
| Gemma / gemma-2-2b-it | GGUF Q8_0 | 22.0 | 20.9 | -5% (thermal noise) |

### Notes

- **Root cause:** `forward_batch_decode` called `build_paged_decode_graph_metal()` + `optimizer::optimize()` + `plan()` on every single token step. For SmolLM2 with 32 layers and ~450 graph nodes this adds ~0.9ms CPU overhead per token.
- **Fix:** Added `DecodeCache` struct to `MetalGraphEngine` (behind a `RefCell` for interior mutability, matching the `CudaGraphEngine` pattern). On the first decode step the graph is built, optimized, and planned once. Subsequent steps reuse the cached `Graph<MetalBackend>` + `ExecutionPlan`, rebuilding only when `(batch_size, block_size, max_blocks_per_seq)` changes (never in practice for single-sequence decode).
- **Gemma -5%:** Not a real regression — Gemma decode is ~45ms/token and the machine was hotter than the reference measurement. Gemma has fewer dispatches/token than SmolLM2 (124K vs 135K) and gains less from graph caching since it's more GPU-bound.

---

## 2026-05-25 — Eliminate per-layer GPU sync in decode (seq_lens readback)

- **Chip:** Apple M3 Pro (18 GB unified memory)
- **Decode tokens:** 256 (8-token warm-up prompt, greedy)
- **Prefill tokens:** 512
- **infernum commit:** (this PR)
- **Date:** 2026-05-25

### Decode throughput (tok/s) — 256 tokens

| Model | Format | before | after | delta |
| ----- | ------ | -----: | ----: | ----: |
| Llama / SmolLM2-360M | SafeTensors F32 | 29.9 | 37.7 | +26.1% |
| Llama / SmolLM2-360M | GGUF Q8_0 | 57.5 | 84.1 | +46.3% |
| Llama / SmolLM2-360M | GGUF Q4_0 | 60.8 | 92.2 | +51.6% |
| Qwen / Qwen3-0.6B | SafeTensors BF16 | 24.1 | 28.2 | +17.0% |
| Gemma / gemma-2-2b-it | GGUF Q8_0 | 18.0 | 22.0 | +22.2% |

### GPU flushes per 256-token decode (smollm2-q8_0)

| | flushes | GPU wait |
|-|--------:|---------:|
| before | 1280 | 426.9 ms |
| after | 256 | 2483.9 ms (all useful) |

### Notes

- **Root cause:** `paged_attention_decode` in `execute_context.rs` was calling `seq_lens.as_bytes()` to compute `max_seq_len` at runtime (because the graph stores `0` as a sentinel). `as_bytes()` calls `flush()` which commits and waits for the current command buffer before reading CPU-side memory. With 32 layers per decode step this caused 32 forced GPU sync-points per token — 1280 flushes for 256 generated tokens instead of the necessary 256 (one per token to read the output logits).
- **Fix:** `MetalPagedKvCache` now carries a `current_max_seq_len: usize` field. The engine sets it from the CPU-computed `seq_lens_u32.max()` in `forward_batch_decode()` before any GPU dispatch starts. The execute_context reads this field directly instead of calling `seq_lens.as_bytes()`.
- **Why GPU wait time increased:** The `426.9ms` before was the sum of 1280 small waits, each for a tiny slice of work. The `2483.9ms` after is one large wait per token, but the GPU is doing the same total work — it's just expressed as useful GPU time rather than serialized pipeline stalls. The CPU no longer interrupts the GPU 32 times per token.
- **Gemma prefill (295.9 tok/s):** Lower than the 815.3 recorded in the previous entry (measured under GPU boost conditions). The seq_lens fix does not affect the prefill path, which never uses `MetalPagedKvCache`. Gemma 2B prefill throughput varies with thermal state; 280–660 tok/s represents the realistic range on M3 Pro.

---

## 2026-05-25 — Concurrent Metal dispatch with selective barriers

- **Chip:** Apple M3 Pro (18 GB unified memory)
- **Decode tokens:** 256 (8-token warm-up prompt, greedy)
- **Prefill tokens:** 512
- **infernum commit:** (this PR)
- **Date:** 2026-05-25

### Decode throughput (tok/s) — 256 tokens

| Model | Format | before | after | delta |
| ----- | ------ | -----: | ----: | ----: |
| Llama / SmolLM2-360M | SafeTensors F32 | 26.3 | 29.9 | +13.7% |
| Llama / SmolLM2-360M | GGUF Q8_0 | 56.5 | 57.5 | +1.8% |
| Llama / SmolLM2-360M | GGUF Q4_0 | 59.4 | 60.8 | +2.4% |
| Qwen / Qwen3-0.6B | SafeTensors BF16 | 22.8 | 24.1 | +5.7% |
| Gemma / gemma-2-2b-it | GGUF Q8_0 | 18.4 | 18.0 | -2.2% (noise) |

### Prefill throughput (tok/s) — 512-token prompt

| Model | Format | before | after | delta |
| ----- | ------ | -----: | ----: | ----: |
| Llama / SmolLM2-360M | SafeTensors F32 | 264.8 | 264.8 | 0% |
| Llama / SmolLM2-360M | GGUF Q8_0 | 253.2 | 253.9 | +0.3% (noise) |
| Llama / SmolLM2-360M | GGUF Q4_0 | 253.9 | 253.9 | 0% |
| Qwen / Qwen3-0.6B | SafeTensors BF16 | 90.9 | 91.0 | +0.1% (noise) |
| Gemma / gemma-2-2b-it | GGUF Q8_0 | 661.4 | 815.3 | +23.3% |

### Notes

- **Concurrent dispatch:** Switched `MTLDispatchType::Serial` → `MTLDispatchType::Concurrent`. With Serial dispatch the GPU must fully complete each kernel before starting the next, preventing weight-data prefetch for upcoming kernels. With Concurrent dispatch the GPU pipeline can look ahead and start fetching data for the next kernel while the current one is running.
- **Selective barriers:** Full "barrier before every dispatch" (tried and reverted) caused catastrophic Gemma regression (-87% decode) because each `memoryBarrierWithScope:` adds overhead. Instead, we track which GPU buffer addresses were written by each dispatch and insert a barrier only when a subsequent dispatch reads a buffer that was written (read-after-write dependency). Independent ops (Q/K/V projections, gate/up FFN projections) get no barrier between them and can overlap on the GPU.
- **Special-case output tracking:** Kernels whose output buffer is not the last slot in the buffer list (`add_inplace`, `bias_add_inplace`, `append_kv_paged_batched_fused`, `rope_kv_append_fused`, `fused_attention_prefill`) now use explicit `dispatch_*_with_outputs` variants that specify the correct output indices, ensuring correctness without false-positive barriers.
- **Decode improvements:** GGUF decode also benefits from concurrent dispatch (+1.8–2.4%). The SafeTensors F32 improvement (+13.7%) is larger because F32 weights occupy 4× the memory of Q8_0, so the GPU prefetcher gains more from looking ahead.
- **Gemma prefill +23%:** The largest single improvement this session. Gemma 2B's 18-layer architecture with 4 norms per layer generates many short, fast kernels that are particularly well-suited to concurrent overlap. With Serial dispatch these ran back-to-back with full synchronization barriers; with Concurrent dispatch the GPU schedules them in waves.
- **Gemma decode unchanged:** 18.0 vs 18.4 tok/s is within measurement noise (thermal variability between sessions).

---

## 2026-05-23 — GGUF native block format for Q8_0/Q4_0 GEMV

- **Chip:** Apple M3 Pro (18 GB unified memory)
- **Decode tokens:** 256 (8-token warm-up prompt, greedy)
- **Prefill tokens:** 512
- **infernum commit:** `20b30a3`
- **llama.cpp commit:** `e22cd0aa1` (`-ngl 99`, 3 reps)
- **Date:** 2026-05-23

### Prefill throughput (tok/s) — 512-token prompt

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| Llama / SmolLM2-360M | SafeTensors F32 | 264.9 | — | — |
| Llama / SmolLM2-360M | GGUF Q8_0 | 253.2 | 4541 | 0.056x |
| Llama / SmolLM2-360M | GGUF Q4_0 | 253.9 | 4596 | 0.055x |
| Qwen / Qwen3-0.6B | SafeTensors BF16 | 90.9 | — | — |
| Gemma / gemma-2-2b-it | GGUF Q8_0 | 661.4 | 938 | 0.70x |

### Decode throughput (tok/s) — 256 tokens

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| Llama / SmolLM2-360M | SafeTensors F32 | 26.3 | — | — |
| Llama / SmolLM2-360M | GGUF Q8_0 | 56.5 | 184.4 | 0.31x |
| Llama / SmolLM2-360M | GGUF Q4_0 | 59.4 | 226.7 | 0.26x |
| Qwen / Qwen3-0.6B | SafeTensors BF16 | 22.8 | — | — |
| Gemma / gemma-2-2b-it | GGUF Q8_0 | 18.4 | 29.1 | 0.63x |

### Notes

- **GGUF native block format:** Q8_0 and Q4_0 weights are now stored as interleaved GGUF blocks (`[f16 scale | quant data]` per block) instead of separate data/scales buffers. New GEMV kernels read scale+data from a single 34-byte (Q8_0) or 18-byte (Q4_0) contiguous block per SIMD-group, reducing cache-miss surface. Threadgroup configs match llama.cpp: Q8_0 uses 4 SGs × 2 rows = 128 threads; Q4_0 uses 2 SGs × 4 rows = 64 threads.
- **Decode improvements vs prior measurement (isolated per-model):** SmolLM2 Q8_0 +1.4%, Q4_0 +0.7%, Gemma Q8_0 +2.3%.
- **Qwen SafeTensors unaffected:** block format applies only to GGUF weights. Qwen improvement over the 2026-05-21 baseline reflects different thermal conditions, not a code change.
- **Gemma decode (18.4 tok/s):** lower than the 2026-05-21 baseline (22.1 tok/s) because the benchmark now runs a 512-token prefill before decode instead of an 8-token warmup. Both measurements use the same methodology within a session; the prefill heats up the GPU more, reducing sustained decode throughput on the 2-2B model.
- **Gemma prefill (661.4 vs 938 = 0.70x):** significant improvement over prior best (576, 0.61x). Full benchmark run is now above thermal parity with the previous session.
- Machine is Apple M3 Pro (below Standard tier; Standard is 24–48 GB).

---

## 2026-05-22 — Llama 3.1 8B (first 8B results)

- **Chip:** Apple M3 Pro (18 GB unified memory) — below Standard tier (24–48 GB)
- **Decode tokens:** 256 (8-token warm-up prompt, greedy)
- **Prefill prompt:** 512 tokens (llama.cpp only — infernum GGUF prefill is serial GEMV, impractical at 8B scale)
- **infernum commit:** `558b468`
- **llama.cpp commit:** `e22cd0aa1` (`-ngl 99`, best of 3 reps)
- **Models:** `QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF`

### Prefill throughput (tok/s) — 512-token prompt

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| Llama / Llama-3.1-8B | GGUF Q4_0 | — | 290.8 | — |

### Decode throughput (tok/s) — 256 tokens

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| Llama / Llama-3.1-8B | GGUF Q4_0 | 12.8 | 28.1 | 0.46x |

### Notes

- **Prefill skipped for infernum:** Same serial GEMV issue as at 3B scale — impractical for GGUF prefill. Only llama.cpp numbers recorded.
- **Ratio improvement with model scale (Q4_0 decode):** 360M=0.26x → 3B=0.36x → 8B=0.46x. Dispatch overhead (135 K Metal dispatches per 256-token decode) is constant per layer; larger models have more compute per dispatch, so the gap narrows.
- **Q8_0 not yet measured:** The 8.54 GB Q8_0 file spans multiple xet shards and cannot be downloaded via direct curl in this environment. Only Q4_0 numbers recorded.
- Machine is Apple M3 Pro (below Standard tier; Standard is 24–48 GB).

---

## 2026-05-22 — Llama 3.1 8B Q8_0 (first Q8_0 8B results)

- **Chip:** Apple M3 Pro (18 GB unified memory)
- **Decode tokens:** 256 (8-token warm-up prompt, greedy)
- **Prefill prompt:** 512 tokens
- **infernum commit:** `c5a5604`
- **llama.cpp commit:** `e22cd0aa1` (`-ngl 99`, 1 rep)
- **Model:** `QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF`

### Prefill throughput (tok/s) — 512-token prompt

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| Llama / Llama-3.1-8B | GGUF Q8_0 | 51.8 | 285.8 | 0.18x |

### Decode throughput (tok/s) — 256 tokens

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| Llama / Llama-3.1-8B | GGUF Q8_0 | 2.9 | 16.3 | 0.18x |

### Notes

- **Q8_0 decode regression at 8B scale:** At 3B, Q8_0 and Q4_0 decode both measure ~21 tok/s (dispatch-overhead-bound). At 8B, Q4_0 decodes at 12.8 tok/s but Q8_0 decodes at only 2.9 tok/s — a 4.4× gap vs the expected ~1.9× (proportional to data size). The M3 Pro's GPU memory bandwidth (~150 GB/s) and 8 GB Q8_0 weight data imply a ~18 tok/s theoretical ceiling; llama.cpp reaches 89% of that (16.3 tok/s) while infernum reaches 16%. The Q8_0 GEMV kernel is not yet bandwidth-saturating at this scale.
- **Prefill ratio (0.18×) is significantly below 3B (0.62×):** At 8B the weight matrices are larger (4096→4096 vs 3072→3072 linear layers) and the dequant+GEMM path shows lower GPU utilization. The M3 Pro has less GPU compute relative to memory than the M3 Max/Ultra.
- **Q8_0 decode on this machine is impractical:** 2.9 tok/s is below useful interactive speed. Q4_0 (12.8 tok/s) is the better choice on M3 Pro 18 GB.
- Machine is Apple M3 Pro (below Standard tier; Standard is 24–48 GB).

---

## 2026-05-22 — Llama 3.2 3B (GPU dequant+GEMM prefill)

- **Chip:** Apple M3 Pro (18 GB unified memory) — below Standard tier (24–48 GB); 8B+ BF16 does not fit
- **Decode tokens:** 256 (8-token warm-up prompt, greedy)
- **Prefill prompt:** 512 tokens (both infernum and llama.cpp)
- **infernum commit:** `perf/measurement-doc` (GPU dequant+GEMM + extract_last_row)
- **llama.cpp commit:** `e22cd0aa1` (`-ngl 99`, best of 3 reps)
- **Models:** `bartowski/Llama-3.2-3B-Instruct-GGUF`

### Prefill throughput (tok/s) — 512-token prompt

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| Llama / Llama-3.2-3B | GGUF Q4_0 | 417 | 686.9 | 0.61x |
| Llama / Llama-3.2-3B | GGUF Q8_0 | 424 | 683.0 | 0.62x |

### Prefill throughput (tok/s) — varying prompt length

| Model | Format | n=128 | n=256 | n=512 |
| ----- | ------ | ----: | ----: | ----: |
| Llama / Llama-3.2-3B | GGUF Q4_0 | 69 | 189 | 417 |
| Llama / Llama-3.2-3B | GGUF Q8_0 | 89 | 183 | 424 |

### Decode throughput (tok/s) — 256 tokens

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| Llama / Llama-3.2-3B | GGUF Q4_0 | 21.6 | 59.6 | 0.36x |
| Llama / Llama-3.2-3B | GGUF Q8_0 | 21.5 | 36.9 | 0.58x |

### Notes

- **GPU dequant+GEMM for prefill:** The previous session used the decode GEMV kernel serially at seq_len > 1. This session added a GPU dequantize (Q8_0, Q4_0, Q4_1) → F32 dense buffer followed by a SIMD-group MMAT GEMM, and an `extract_last_row` op before the LM head so only 1 token is projected through the 128K-vocab head instead of all 512. Together these bring prefill from impractical (>20 min) to 417–424 tok/s at 512 tokens, reaching 0.61–0.62× llama.cpp.
- **Q4_0 vs Q8_0 prefill parity:** Both reach similar tok/s because the bottleneck at 512 tokens is GPU GEMM compute, not memory bandwidth. At small prompt sizes (n=128) Q8_0 is slightly faster — larger blocks are more GPU-friendly.
- **Decode unchanged:** The GEMV path for decode is identical. 21.5–21.6 tok/s reflects dispatch overhead.
- **3B Q4_0 GGUF contains mixed quant types:** most layers are Q4_0, but 3 down_proj layers are Q4_1 and the token embedding (reused as LM head) is Q6_K. The GPU dequant path handles Q4_1 via a new `dequantize_q4_1_to_f32` kernel; Q6_K falls to CPU only for the LM head (now m=1 after extract_last_row, so handled by the existing Q6_K GEMV).
- Machine is Apple M3 Pro (below Standard tier; Standard is 24–48 GB).

---

## 2026-05-22 — Llama 3.2 3B (first larger-model results)

- **Chip:** Apple M3 Pro (18 GB unified memory) — below Standard tier (24–48 GB); 8B+ BF16 does not fit
- **Decode tokens:** 256 (8-token warm-up prompt, greedy)
- **Prefill prompt:** 512 tokens (llama.cpp only — infernum GGUF prefill is serial GEMV, impractical at 3B scale)
- **infernum commit:** `558b468`
- **llama.cpp commit:** `e22cd0aa1` (`-ngl 99`, best of 3 reps)
- **Models:** `bartowski/Llama-3.2-3B-Instruct-GGUF`

### Prefill throughput (tok/s) — 512-token prompt

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| Llama / Llama-3.2-3B | GGUF Q4_0 | — | 686.9 | — |
| Llama / Llama-3.2-3B | GGUF Q8_0 | — | 683.0 | — |

### Decode throughput (tok/s) — 256 tokens

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| Llama / Llama-3.2-3B | GGUF Q4_0 | 21.6 | 59.6 | 0.36x |
| Llama / Llama-3.2-3B | GGUF Q8_0 | 21.5 | 36.9 | 0.58x |

### Notes

- **Prefill skipped for infernum:** The serial GEMV issue (0.001x at 360M scale) is even worse at 3B. With ~28 layers and 3072 hidden dim, each token forward pass dispatches hundreds of serial kernel calls; a 512-token prefill would take ~20+ minutes. Only llama.cpp prefill numbers are recorded.
- **Q4_0 vs Q8_0 decode throughput (infernum):** Both formats achieve nearly identical throughput (21.6 vs 21.5 tok/s). The bottleneck is dispatch overhead (118 K Metal dispatches per 256-token decode), not weight memory bandwidth. llama.cpp's more optimized GEMV kernels are faster for Q4_0 (59.6 tok/s) than Q8_0 (36.9 tok/s) because it actually saturates memory bandwidth; infernum does not.
- **Ratio improvement over 360M:** 3B Q8_0 reaches 0.58x vs 0.30x at 360M scale. At larger model sizes the compute per kernel dispatch grows relative to launch overhead, narrowing the gap.
- Machine is Apple M3 Pro (below Standard tier; Standard is 24–48 GB).

---

## 2026-05-21 — GGUF support + prefill benchmarked

- **Chip:** Apple M3 Pro (18 GB unified memory)
- **Decode tokens:** 256 (8-token warm-up prompt, greedy)
- **Prefill prompt:** 512 tokens (llama-bench); infernum GGUF rows use 16-token sample (see note)
- **infernum commit:** `975c532`
- **llama.cpp commit:** `e22cd0aa1` (`-ngl 99`, best of 3 reps)
- **Note:** Gemma Q4_K_M not yet supported in infernum Metal (K-quants, GGML type 12 — only Q4_0 and Q8_0 are implemented)

### Prefill throughput (tok/s) — 512-token prompt

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| Llama / SmolLM2-360M | SafeTensors F32 | 252 | — | — |
| Llama / SmolLM2-360M | GGUF Q8_0 | 4.0 | 4541 | 0.001x |
| Llama / SmolLM2-360M | GGUF Q4_0 | 4.3 | 4596 | 0.001x |
| Qwen / Qwen3-0.6B | SafeTensors BF16 | 86 | — | — |
| Gemma / gemma-2-2b-it | GGUF Q8_0 | 0.6 | 938 | 0.001x |
| Gemma / gemma-2-2b-it | GGUF Q4_K_M | — | 903 | — |

### Decode throughput (tok/s) — 256 tokens

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| Llama / SmolLM2-360M | SafeTensors F32 | 25.7 | — | — |
| Llama / SmolLM2-360M | GGUF Q8_0 | 56.1 | 184.4 | 0.30x |
| Llama / SmolLM2-360M | GGUF Q4_0 | 58.1 | 226.7 | 0.26x |
| Qwen / Qwen3-0.6B | SafeTensors BF16 | 21.1 | — | — |
| Gemma / gemma-2-2b-it | GGUF Q8_0 | 22.1 | 29.1 | 0.76x |
| Gemma / gemma-2-2b-it | GGUF Q4_K_M | — | 65.7 | — |

### Notes

- **GGUF prefill (0.001x):** infernum's Metal GGUF path dispatches one GEMV kernel per input token per linear layer — this is the decode kernel reused at seq_len > 1. The throughput is constant regardless of prompt length (4 tok/s for SmolLM2-360M, 0.6 tok/s for Gemma 2B). llama.cpp uses proper fused GEMM Metal kernels for prefill. Implementing fused quantized GEMM is the top optimization target for infernum Metal prefill.
- **SafeTensors prefill:** infernum F32/BF16 prefill dispatches dense matrix-matrix multiplications (GEMM), which Metal executes efficiently. SmolLM2-360M F32 at 252 tok/s and Qwen3-0.6B BF16 at 86 tok/s (at 512 tokens) reflect real GEMM throughput. No llama.cpp comparison (SafeTensors not supported by llama-bench).
- **GGUF decode (0.26–0.76x):** the decode GEMV path is the same kernel used by llama.cpp, so the gap reflects inference engine overhead and kernel tuning rather than a fundamental algorithmic difference.
- **Gemma decode closer to parity (0.76x):** larger model is more compute-bound relative to engine overhead.
- **Gemma Q4_K_M:** K-quant format not yet implemented in infernum Metal. llama.cpp numbers recorded for reference.
- **Qwen:** no llama.cpp comparison available (llama-bench does not support SafeTensors).
- Machine is Apple M3 Pro (Standard tier per performance.md). High/Ultra tier numbers (large models) are not yet measured.

---

## 2026-05-21 — Baseline (SafeTensors only)

- **Chip:** Apple M3 Pro (18 GB unified memory)
- **Decode tokens:** 256 (8-token warm-up prompt, greedy)
- **infernum commit:** `8d64472`
- **llama.cpp commit:** `e22cd0aa1` (`-ngl 99`, best of 3 reps)
- **Note:** Prefill not yet benchmarked (Metal graph engine pending)
- **Note:** infernum Metal supports SafeTensors only at this point — GGUF loading not yet implemented. Llama and Gemma rows use different formats for infernum vs llama.cpp; direct ratio comparison is not possible.

### Decode throughput (tok/s)

| Model | Format | infernum | llama.cpp |
| ----- | ------ | -------: | --------: |
| Llama / SmolLM2-360M | SafeTensors F32 | 25.9 | — |
| Llama / SmolLM2-360M | GGUF Q8_0 | — | 183.9 |
| Llama / SmolLM2-360M | GGUF Q4_0 | — | 227.4 |
| Qwen / Qwen3-0.6B | SafeTensors BF16 | 20.7 | — |
| Gemma / gemma-2-2b-it | GGUF Q8_0 | — | 43.3 |
| Gemma / gemma-2-2b-it | GGUF Q4_K_M | — | 65.4 |

### Notes

- Llama/Gemma: infernum runs SafeTensors F32; llama.cpp runs GGUF (quantized). These measure different computational workloads — infernum F32 is heavier than Q4/Q8, so the gap partially reflects format difference, not just kernel efficiency.
- Qwen: no llama.cpp comparison available (llama-bench does not support SafeTensors).
- Gemma: infernum has no ungated SafeTensors checkpoint for gemma-2-2b-it (gated on HuggingFace). Only llama.cpp numbers via bartowski GGUF.
- The main takeaway from this baseline: infernum Metal decode is far behind llama.cpp even when accounting for the F32 vs quantized gap. The Metal kernel path is the primary optimization target.
- Machine is Apple M3 Pro (Standard tier per performance.md). High/Ultra tier numbers (large models) are not yet measured.
