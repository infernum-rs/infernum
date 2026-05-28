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

## 2026-05-28 — L4 GPU (final, commit `2477f76`)

- **GPU:** NVIDIA L4 (23034 MiB VRAM)
- **Driver:** 595.71.05 | CUDA 12.6
- **Decode tokens:** 256 | **Prefill tokens:** 512 (8-token warm-up prompt)
- **infernum commit:** `2477f76` | **llama.cpp commit:** `d8794ee` (`-ngl 99`, best of 3 reps)
- **Note:** The 2026-05-21 llama.cpp baseline of 250 tok/s (SmolLM2-360M Q8_0) was measured on a different GGUF file or machine state. Re-measuring on the same Instruct GGUF gives 350 tok/s.

### Decode throughput (tok/s) — GGUF, same-format comparison

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| Llama / SmolLM2-360M-Instruct | Q8_0 GGUF | 308.5 | 350.4 | 0.88x |
| Llama / SmolLM2-360M-Instruct | Q4_0 GGUF | 351.7 | 450.0 | 0.78x |

### Decode throughput (tok/s) — BF16 SafeTensors (no same-format llama.cpp number)

| Model | Format | Engine | infernum | llama.cpp (Q8_0 for ref) |
| ----- | ------ | ------ | -------: | -------: |
| Llama / SmolLM2-360M | BF16 SafeTensors | cuda-graph-engine | 233.9 | — |
| Llama / Llama-3.2-1B | BF16 SafeTensors | cuda-graph-engine | 114.8 | (97.2, Llama-3.2-1B Q8_0) |
| Qwen / Qwen3-0.6B | BF16 SafeTensors | cuda-graph-engine | 155.4 | — |
| Qwen / Qwen2.5-0.5B | BF16 SafeTensors | cuda-graph-engine | 182.2 | — |

### Prefill throughput (tok/s, 512 tokens, paged-decode token-by-token path)

| Model | Format | infernum | llama.cpp |
| ----- | ------ | -------: | --------: |
| SmolLM2-360M-Instruct | Q8_0 GGUF | 240.2 | 20195 (batch) |
| SmolLM2-360M-Instruct | Q4_0 GGUF | 261.1 | 21196 (batch) |
| SmolLM2-360M | BF16 SafeTensors | 206.2 | — |
| Llama-3.2-1B | BF16 SafeTensors | 109.9 | — |

### Notes

- **Q8_0 at 0.88×, Q4_0 at 0.78×**: llama.cpp uses highly optimised fused quantised GEMV kernels (custom CUDA, purpose-built for each quant format). Infernum uses cuBLAS for dense weights and generic dp4a kernels for Q8_0/Q4_0. Closing this gap requires a custom BF16 GEMV kernel.
- **Q4_0 now works**: was previously crashing (`UnsupportedDtype: type 3`). Fixed by adding Q4_1 block dequantisation — many Q4_0 GGUF files store some tensors (e.g. down_proj) in Q4_1 format.
- **Prefill is not comparable**: infernum's prefill uses a token-by-token paged-decode loop; llama.cpp uses a batched matmul. The llama.cpp prefill numbers (20k tok/s) reflect batch GEMM with tensor-core utilisation and are a different operation.
- **Key optimisations in this cycle (PR #77):** shallow `CudaTensor::clone()` (eliminated 24 GB/5 s of D→D copies); paged-attention shared-memory correctness fix (+12–69% decode, fixed crash at 1024 tokens); QKV fused GEMV (+5.4%); Q8_0 linear_pair/triple input-quantisation deduplication; BF16 cos/sin inputs; async prefill pipeline.

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
