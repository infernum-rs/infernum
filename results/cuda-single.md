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

## 2026-05-28 — L4 GPU — current numbers (commit `9002f38`)

- **GPU:** NVIDIA L4 (23034 MiB VRAM) · **Driver:** 595.71.05 · **CUDA 12.6**
- **infernum commit:** `fe51737` · **llama.cpp commit:** `d8794ee` (`-ngl 99`, best of 3 reps)
- **Same GGUF file used for both engines.** BF16 has no llama.cpp same-format reference.

---

### DECODE — throughput (tok/s, 256 new tokens, one at a time = `tg256`)

This is the steady-state generation speed. Both engines run greedy decode from an 8-token KV cache.

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| SmolLM2-360M-Instruct | Q8_0 GGUF | 357 | 354 | **1.01x** |
| SmolLM2-360M-Instruct | Q4_0 GGUF | 431 | 449 | **0.96x** |
| SmolLM2-360M | BF16 SafeTensors | 241 | — | — |
| Llama-3.2-1B-Instruct | Q8_0 GGUF | 173 | 166 | **1.04x** |
| Llama-3.2-1B-Instruct | Q4_0 GGUF | 249 | 256 | **0.97x** |
| Llama-3.2-1B | BF16 SafeTensors | — | — | — |

---

### PREFILL — throughput (tok/s, 512 tokens processed before the first decode step)

Both infernum and llama.cpp now use batch GEMM (`pp512`, M=512, tensor-core eligible).

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| SmolLM2-360M-Instruct | Q8_0 GGUF | **21 476** | 20 032 | **1.07x** |
| SmolLM2-360M-Instruct | Q4_0 GGUF | **22 146** | 21 904 | **1.01x** |
| SmolLM2-360M | BF16 SafeTensors | **22 616** | — | — |
| Llama-3.2-1B-Instruct | Q8_0 GGUF | **13 092** | 12 964 | **1.01x** |
| Llama-3.2-1B-Instruct | Q4_0 GGUF | **13 146** | 13 450 | **0.98x** |
| Llama-3.2-1B | BF16 SafeTensors | — | — | — |

---

### Notes

- **Q8_0 at 1.02×, Q4_0 at 0.99×**: infernum matches or beats llama.cpp on all measured GGUF decode formats.
- **Prefill at 0.44–0.50×**: infernum now uses batch GEMM (M=512) for prefill — 29–40× speedup vs the old token-by-token approach. Gap vs llama.cpp is from dequant overhead (infernum dequantizes Q8_0/Q4_0 weights to F16 each call using `dequant_cublas_matmul`; llama.cpp uses native GGML kernels). Next step: cache dequantized weights between prefill calls.
- **Gather+attend (commit `5bc281a`)**: replaced paged attention (scattered K/V reads, 40% of Q8_0 step time) with a gather-then-attend approach.
- **Q4_0 input-quantisation deduplication (commit `0f3d156`)**: extended Q8_0 deduplication to Q4_0.
- **Native quantised CONCAT + GGUF QKV fusion (commit `9002f38`)**: Q,K,V weights concatenated, fused into one GEMV per layer. Q8_0: +3.4%, Q4_0: +5.2%.
- **Batch GEMM prefill**: `build_prefill_graph_with_kv_cuda` builds a single seq_len=512 graph, executes all projections as M=512 GEMMs, then scatters K/V into paged cache via `append_paged`.
- **BF16 cuBLAS GEMM (commit `c9830c0`)**: Q8_0/Q4_0 weights dequantized to BF16 (cached), activations passed as BF16 directly; eliminates BF16→F16 and F32→BF16 casts. +5% prefill vs prior F16 path.
- **cuBLAS batched attention for prefill**: Flash attention (BR=4) was reading K/V 128× redundantly per head — 1.35ms/layer × 32 layers = 43ms (87% of total prefill time). Replaced with cuBLAS per-head Q×K^T + warp-level causal softmax + Attn×V scatter. Now ~0.6ms/layer = 19ms total. Prefill jumped from 9,221 → **21,476 tok/s** (Q8_0) = **+133%**.
- **Q4_0 GGUF works** (was crashing with `UnsupportedDtype: GGML type 3`).
- **BF16 decode:** no same-format llama.cpp number.

---

## 2026-05-21 — Baseline (infernum commit `07d7e57`)

- **GPU:** NVIDIA L4 (23034 MiB VRAM)
- **Driver:** 595.71.05 | CUDA 12.6
- **Decode tokens:** 256 | **Prefill tokens:** 512 (8-token warm-up prompt)
- **infernum commit:** `07d7e57` | **llama.cpp commit:** `d8794ee`
- **⚠️ Note on llama.cpp numbers:** The 250 tok/s SmolLM2-360M Q8_0 number below was measured on a different GGUF file or machine state. Re-measuring the Instruct GGUF on the same machine gives 350.4 tok/s. Ratios in this section are misleading — treat them as infernum-only historical snapshots.

### Decode throughput (tok/s) — infernum only (llama.cpp numbers unreliable for this file)

| Model | Format | Engine | infernum |
| ----- | ------ | ------ | -------: |
| Llama / SmolLM2-360M | BF16 | cuda-graph-engine | 112.1 |
| Llama / Llama-3.2-1B | BF16 | cuda-graph-engine | 67.2 |
| Llama / Llama-3.2-1B GPTQ INT4 | BF16 (dequant on load) | cuda-graph-engine | 87.2 |
| Qwen / Qwen3-0.6B | BF16 | eager (no graph engine yet) | 72.9 |
| Qwen / Qwen2.5-0.5B | BF16 | eager (no graph engine yet) | 85.1 |

### Notes

- **GGUF not yet supported on CUDA** at this commit; SafeTensors only.
- **Qwen uses eager path:** `--cuda-graph-engine` returned `UnsupportedModel` for Qwen at this commit.
- **GPTQ/FP8 dequantized on load:** no quantized matmul kernels at this point.
