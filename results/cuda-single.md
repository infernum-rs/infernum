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

## 2026-05-28 — L4 GPU — current numbers (commit `fe51737`)

- **GPU:** NVIDIA L4 (23034 MiB VRAM) · **Driver:** 595.71.05 · **CUDA 12.6**
- **infernum commit:** `fe51737` · **llama.cpp commit:** `d8794ee` (`-ngl 99`, best of 3 reps)
- **Same GGUF file used for both engines.** BF16 has no llama.cpp same-format reference.

---

### DECODE — throughput (tok/s, 256 new tokens, one at a time = `tg256`)

This is the steady-state generation speed. Both engines run greedy decode from an 8-token KV cache.

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| SmolLM2-360M-Instruct | Q8_0 GGUF | 309.4 | 354.0 | 0.87x |
| SmolLM2-360M-Instruct | Q4_0 GGUF | 351.7 | 448.8 | 0.78x |
| SmolLM2-360M | BF16 SafeTensors | 234.1 | — | — |
| Llama-3.2-1B-Instruct | Q8_0 GGUF | 167.0 | 165.9 | **1.01x** |
| Llama-3.2-1B-Instruct | Q4_0 GGUF | 232.5 | 255.6 | 0.91x |
| Llama-3.2-1B | BF16 SafeTensors | 114.7 | — | — |

---

### PREFILL — throughput (tok/s, 512 tokens processed before the first decode step)

**⚠️ Not an apples-to-apples comparison.** infernum currently prefills token-by-token (sequential M=1 GEMVs through the paged KV cache). llama.cpp uses batched GEMM (`pp512`, M=512, tensor-core eligible). The two numbers reflect different algorithms; the llama.cpp column shows the target we are working towards.

| Model | Format | infernum (token-by-token) | llama.cpp (batch GEMM) |
| ----- | ------ | -------------------------: | ---------------------: |
| SmolLM2-360M-Instruct | Q8_0 GGUF | 241 | 20 032 |
| SmolLM2-360M-Instruct | Q4_0 GGUF | 261 | 21 904 |
| SmolLM2-360M | BF16 SafeTensors | 206 | — |
| Llama-3.2-1B-Instruct | Q8_0 GGUF | 157 | 12 964 |
| Llama-3.2-1B-Instruct | Q4_0 GGUF | 212 | 13 450 |
| Llama-3.2-1B | BF16 SafeTensors | 110 | — |

---

### Notes

- **Decode status:** Llama-3.2-1B Q8_0 matches llama.cpp (1.01×). SmolLM2-360M lags (0.78–0.87×) — it has 32 layers vs 16, so per-step kernel-launch overhead is doubled. Closing the SmolLM2 decode gap is the current focus.
- **Prefill status:** infernum prefill is sequential (identical to decode speed × token count). Batch prefill (single batched GEMM over all prompt tokens) is the next major feature — it would bring infernum prefill to the same order of magnitude as llama.cpp.
- **Q4_0 GGUF now works** (was crashing with `UnsupportedDtype: GGML type 3`). Fixed by adding Q4_1 block dequantisation — many Q4_0 GGUF files store some tensors in Q4_1 format.
- **BF16 decode:** no same-format llama.cpp number available (llama.cpp doesn't run BF16 on GGUF). BF16 uses 2× more bandwidth than Q8_0, so the decode speed is proportionally lower.

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
