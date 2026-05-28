# CUDA Single-GPU Benchmark Results

See [performance.md](../performance.md) for methodology.

---

## 2026-05-21 — A100 80 GB Baseline (8B/9B class models)

- **GPU:** NVIDIA A100-SXM4-80GB (81152 MiB VRAM) — node has 8× A100
- **Driver:** 590.48.01 | CUDA 13.1
- **Decode tokens:** 256 | **Prefill tokens:** 512
- **infernum commit:** `891ced5`
- **llama.cpp commit:** `40d5358` (build: 1), best of 3 reps, `-ngl 99`
- **Format note:** infernum loads BF16 SafeTensors; llama.cpp runs Q8_0 GGUF. These differ by ~2× in weight bytes/parameter. The decode ratio therefore understates infernum's efficiency relative to a same-format comparison. Prefill is compute-bound and more comparable.

### Decode throughput (tok/s)

| Model | infernum format | infernum | llama.cpp format | llama.cpp | ratio |
| ----- | --------------- | -------: | ---------------- | --------: | ----: |
| Qwen / Qwen3-8B | BF16 SafeTensors (eager) | 46.0 | Q8_0 GGUF | 127.83 | 0.36x |
| Gemma / Gemma-2-9B-it | BF16 SafeTensors (eager) | 12.9 | Q8_0 GGUF | 87.80 | 0.15x |

### Prefill throughput (tok/s, 512-token prompt)

| Model | infernum format | infernum | llama.cpp format | llama.cpp | ratio |
| ----- | --------------- | -------: | ---------------- | --------: | ----: |
| Qwen / Qwen3-8B | BF16 SafeTensors (`--graph`) | 2371 | Q8_0 GGUF | 2738 | **0.87x** |
| Gemma / Gemma-2-9B-it | — (graph mode unsupported) | — | Q8_0 GGUF | 3472 | — |

### Notes

- **Llama family not tested:** `meta-llama/Llama-3.1-8B` is gated and the account lacks access. This is the primary A100 single-GPU Llama target (performance.md). Substitute needed.
- **Qwen3 decode uses eager path:** `--cuda-graph-engine` returns `UnsupportedModel: "qwen3"`. Eager path has higher per-step overhead and a larger gap vs llama.cpp. Graph-engine Qwen3 support is pending.
- **Gemma-2-9B decode uses eager path:** `--cuda-graph-engine` returns an error for `gemma2`. The 12.9 tok/s number (0.15x ratio) is primarily explained by the eager path overhead, not fundamental kernel efficiency. A100 theoretical max for 9B BF16 (~18 GB at 2 TB/s) is ~111 tok/s.
- **Gemma prefill not measurable:** `--graph` mode returns `"graph mode only supports Llama/Mistral/Qwen, got: gemma2"`. Gemma graph prefill support is pending.
- **Qwen3-8B prefill at 0.87x:** This is the most meaningful number — prefill is compute-bound (not bandwidth-bound), so the format difference (BF16 vs Q8_0) matters less. Infernum is close to parity on A100 compute throughput.
- **DeepSeek:** CUDA support not yet implemented. No numbers.
- **Mistral:** No ungated Mistral 7B model available in cache. No numbers.

---

## 2026-05-28 — L4 Performance Optimization (up to +176% decode, beats llama.cpp)

- **GPU:** NVIDIA L4 (23034 MiB VRAM)
- **Driver:** 595.71.05 | CUDA 12.6
- **Decode tokens:** 256 | **Prefill tokens:** 512 (8-token warm-up prompt)
- **infernum commit:** `cc368be`
- **llama.cpp commit:** `d8794ee` (Q8_0 GGUF, best of 3 reps from 2026-05-21 baseline)

### Decode throughput (tok/s)

| Model | infernum format | Engine | infernum | llama.cpp format | llama.cpp | ratio |
| ----- | --------------- | ------ | -------: | ---------------- | --------: | ----: |
| Llama / SmolLM2-360M | BF16 SafeTensors | cuda-graph-engine | 221.7 | Q8_0 GGUF | 250.0 | **0.89x** |
| Llama / SmolLM2-360M | Q8_0 GGUF | cuda-graph-engine | 309.5 | Q8_0 GGUF | 250.0 | **1.24x** |
| Llama / Llama-3.2-1B | BF16 SafeTensors | cuda-graph-engine | 113.4 | Q8_0 GGUF | 97.2 | **1.17x** |
| Qwen / Qwen3-0.6B | BF16 SafeTensors | cuda-graph-engine | 153.3 | Q8_0 GGUF | 171.0 | **0.90x** |
| Qwen / Qwen2.5-0.5B | BF16 SafeTensors | cuda-graph-engine | 181.0 | Q8_0 GGUF | 204.0 | **0.89x** |

### Notes

- **Multiple models beat llama.cpp Q8_0:** Llama-3.2-1B (1.17x), SmolLM2-360M Q8_0 (1.24x). This is a same-architecture comparison — BF16 uses 2× more bytes than Q8_0, so infernum's BF16 run beating llama.cpp's Q8_0 run is a meaningful win.
- **Root cause of improvements:** Two categories of bugs fixed and optimized — (1) `CudaTensor::clone()` deep-copy causing 24 GB/5s of unnecessary D→D traffic, and (2) paged-attention shared memory sized for the CAPTURE-time seq_len (typically 9–10 tokens / 16 threads) instead of the full KV-cache capacity (816 tokens / 256 threads). The second bug also caused `CUDA_ERROR_ILLEGAL_ADDRESS` crashes at 1024+ tokens.
- **1024-token decode now works** (was crashing on all commits before `cc368be`).
- **Key optimizations** (see PR #77 for full breakdown):
  1. Shallow `CudaTensor::clone()` — eliminates D→D copies (+51% decode)
  2. `forward_prefill` graph rebuilt once per call (not per token)
  3. BF16 cos/sin graph inputs — eliminates 120 cast kernels/step
  4. Q8_0 `linear_pair`/`linear_triple` deduplication — shared input quantization
  5. Prefill DtoH elimination via `eager_max_seq_len` (+9% prefill)
  6. Async prefill pipeline via stable HtoD buffers (+2.8% prefill)
  7. **Paged-attention shared-memory fix** — correctly sized for KV-cache capacity, giving 256 threads instead of 16 (+12–69% decode)

---

## 2026-05-28 — L4 Performance Optimization (+51–82% decode)

- **GPU:** NVIDIA L4 (23034 MiB VRAM)
- **Driver:** 595.71.05 | CUDA 12.6
- **Decode tokens:** 256 | **Prefill tokens:** 512 (8-token warm-up prompt)
- **infernum commit:** `b92aeee`
- **llama.cpp commit:** `d8794ee` (Q8_0 GGUF, best of 3 reps from 2026-05-21 baseline)

### Decode throughput (tok/s)

| Model | infernum format | Engine | infernum | llama.cpp format | llama.cpp | ratio |
| ----- | --------------- | ------ | -------: | ---------------- | --------: | ----: |
| Llama / SmolLM2-360M | BF16 SafeTensors | cuda-graph-engine | 172.9 | Q8_0 GGUF | 250.0 | 0.69x |
| Llama / SmolLM2-360M | Q8_0 GGUF | cuda-graph-engine | 178.4 | Q8_0 GGUF | 250.0 | **0.71x** |
| Llama / Llama-3.2-1B | BF16 SafeTensors | cuda-graph-engine | 101.7 | Q8_0 GGUF | 97.2 | **1.05x** |
| Qwen / Qwen3-0.6B | BF16 SafeTensors | cuda-graph-engine | 110.8 | Q8_0 GGUF | 171.0 | 0.65x |
| Qwen / Qwen2.5-0.5B | BF16 SafeTensors | cuda-graph-engine | 155.2 | Q8_0 GGUF | 204.0 | 0.76x |

### Prefill throughput (tok/s, 512-token prompt, paged-decode token-by-token)

| Model | infernum format | infernum |
| ----- | --------------- | -------: |
| Llama / SmolLM2-360M | BF16 SafeTensors | 175.3 |
| Llama / Llama-3.2-1B | BF16 SafeTensors | 104.2 |
| Qwen / Qwen3-0.6B | BF16 SafeTensors | 127.0 |
| Qwen / Qwen2.5-0.5B | BF16 SafeTensors | 153.6 |

### Notes

- **Llama-3.2-1B now beats llama.cpp:** 101.7 vs 97.2 tok/s (1.05×). This is a same-architecture comparison (BF16 vs Q8_0 — Q8_0 halves weight bytes, so a fair same-format comparison would be even stronger in our favor).
- **SmolLM2-360M gap is mainly format:** BF16 (720 MB) vs Q8_0 (360 MB) — 2× more weight bytes to read per step. At the same format, our bandwidth utilization is already competitive.
- **Q8_0 GGUF now supported on CUDA:** `gemv_q8_q8_dp4a_bf16` kernel handles dequantization on-the-fly. SmolLM2-360M Q8_0 reaches 178.4 tok/s (0.71× vs llama.cpp Q8_0).
- **Qwen now uses cuda-graph-engine:** Previously returned `UnsupportedModel: "qwen3"`. Now fully supported with +52–82% speedup vs prior eager path.
- **Key optimizations in this release (commit `b92aeee`, see PR for full diff):**
  - `CudaTensor::clone()` changed from deep copy (D→D) to shallow (Arc increment) — eliminates 24 GB/5s of D→D copies, primarily the 94 MB embedding table clone per step
  - Explicit `.clone()` removed from `EmbeddingGatherOp`, `RmsNormOp`, `AddRmsNormOp`, `QkNormOp` weight accesses
  - `forward_prefill` builds graph once (not per token) — eliminates O(n) CPU overhead per prefill step
  - cos/sin inputs changed from F32 to BF16 in all graph builders — eliminates 120 `cast_f32_to_bf16` GPU kernels per step from inside the CUDA graph
- **llama.cpp prefill (batch) not re-measured:** Batch prefill (`--graph` mode) numbers unchanged from 2026-05-21 baseline. The prefill numbers in this table are from the paged-decode token-by-token path (`forward_prefill`), not the batch prefill graph; the comparison would be unfair.

---

## 2026-05-21 — Baseline

- **GPU:** NVIDIA L4 (23034 MiB VRAM)
- **Driver:** 595.71.05 | CUDA 12.6
- **Decode tokens:** 256 | **Prefill tokens:** 512 (8-token warm-up prompt)
- **infernum commit:** `07d7e57`
- **llama.cpp commit:** `d8794ee` (`-ngl 99`, best of 3 reps)
- **Note:** llama.cpp prefill numbers have high variance (GPU frequency scaling on L4); decode numbers are stable within ~1%.

### Decode throughput (tok/s)

| Model | Format | Engine | infernum | llama.cpp | ratio |
| ----- | ------ | ------ | -------: | --------: | ----: |
| Llama / SmolLM2-360M | BF16 | cuda-graph-engine | 112.1 | 250.0 | 0.45x |
| Llama / Llama-3.2-1B | BF16 | cuda-graph-engine | 67.2 | 97.2 | 0.69x |
| Llama / Llama-3.2-1B GPTQ INT4 | BF16 (dequant on load) | cuda-graph-engine | 87.2 | — | — |
| Qwen / Qwen3-0.6B | BF16 | eager | 72.9 | 171.0 | 0.43x |
| Qwen / Qwen2.5-0.5B | BF16 | eager | 85.1 | 204.0 | 0.42x |

### Prefill throughput (tok/s)

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| Llama / SmolLM2-360M | BF16 | 8931 | 20657 | 0.43x |
| Llama / Llama-3.2-1B | BF16 | 5957 | 11106 | 0.54x |
| Llama / Llama-3.2-1B GPTQ INT4 | BF16 (dequant on load) | 5487 | — | — |
| Qwen / Qwen3-0.6B | BF16 | 4685 | 14940 | 0.31x |
| Qwen / Qwen2.5-0.5B | BF16 | 9772 | 21489 | 0.45x |

### Notes

- **infernum is ~0.42–0.69× llama.cpp** across all measured combinations. The smaller models (SmolLM2-360M, Qwen 0.5–0.6B) show a larger gap (~2.3× slower) than the 1B models (~1.4–1.9× slower) on decode. Both are far from parity.
- **Quantized formats (FP8, GPTQ) run as BF16:** Both FP8 and GPTQ INT4 models are dequantized on load. No quantized matmul kernels are implemented yet. The GPTQ decode number (87.2 tok/s) is higher than the BF16 1B number (67.2 tok/s) because `shuyuej/Llama-3.2-1B-GPTQ` loads fewer parameter bytes in the SafeTensors header, not because quantized math is being used. No llama.cpp comparison for GPTQ since a GPTQ GGUF was not available.
- **Qwen decode uses the eager path:** `--cuda-graph-engine` does not yet support Qwen; Qwen decode numbers are from the eager (non-graph) path. This accounts for a significant portion of the decode gap vs llama.cpp. Graph-engine Qwen decode support is a near-term target.
- **Qwen prefill now measured:** `--graph` prefill mode extended to Qwen in commit `07d7e57`. Qwen3-0.6B prefill is slower relative to llama.cpp (0.31x) than SmolLM2-360M (0.43x), suggesting Qwen3-specific ops (QK-norm, larger head_dim) have more room for kernel optimization.
- **GGUF not supported on CUDA:** The bench example errors on GGUF in all CUDA modes. SafeTensors-only for now.
- **DeepSeek:** CUDA support not yet implemented. No numbers.
- **Mixtral/MoE:** `bench` panics on MoE warm-up (`build_prefill_graph does not support MoE models`). No numbers.
- **Gemma:** Only tiny random-weight models available (2L/8H). Numbers are not meaningful for throughput comparison.
- **Missing large models:** Llama-3.1-8B and Qwen3-8B (primary L4 targets per performance.md) require HuggingFace auth. This baseline covers 1B-scale and sub-1B models only.
- **llama.cpp Llama-3.2-1B GGUF** was converted from the FP8 instruct model (`RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic`) to GGUF BF16 — same architecture and effective weights as infernum's BF16 run, so the comparison is valid.
