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

## 2026-05-28 — L4 GPU (final, commit `e791b03`)

- **GPU:** NVIDIA L4 (23034 MiB VRAM)
- **Driver:** 595.71.05 | CUDA 12.6
- **Decode tokens:** 256 | **Prefill tokens:** 512 (8-token warm-up prompt)
- **infernum commit:** `e791b03` | **llama.cpp commit:** `d8794ee` (`-ngl 99`, best of 3 reps)
- **Same model, same GGUF file for both engines.**

### Decode throughput (tok/s, `tg256`) — same model, same GGUF, greedy one-token-at-a-time

| Model | Format | infernum | llama.cpp | ratio |
| ----- | ------ | -------: | --------: | ----: |
| SmolLM2-360M-Instruct | Q8_0 GGUF | 308.5 | 350.4 | 0.88x |
| SmolLM2-360M-Instruct | Q4_0 GGUF | 351.7 | 450.0 | 0.78x |
| Llama-3.2-1B-Instruct | Q8_0 GGUF | 166.7 | 165.7 | **1.01x** |
| Llama-3.2-1B-Instruct | Q4_0 GGUF | 231.7 | 254.9 | 0.91x |

### Decode throughput (tok/s, `tg256`) — BF16 SafeTensors (no llama.cpp same-format reference)

| Model | Format | infernum |
| ----- | ------ | -------: |
| SmolLM2-360M | BF16 SafeTensors | 233.9 |
| Llama-3.2-1B | BF16 SafeTensors | 114.8 |
| Qwen3-0.6B | BF16 SafeTensors | 155.4 |
| Qwen2.5-0.5B | BF16 SafeTensors | 182.2 |

### Prefill throughput (tok/s, 512 tokens, paged-decode token-by-token path)

| Model | Format | infernum | llama.cpp (batch) |
| ----- | ------ | -------: | ----------------: |
| SmolLM2-360M-Instruct | Q8_0 GGUF | 240.2 | 20195 |
| SmolLM2-360M-Instruct | Q4_0 GGUF | 261.1 | 21196 |
| Llama-3.2-1B-Instruct | Q8_0 GGUF | 156.4 | 12861 |
| Llama-3.2-1B-Instruct | Q4_0 GGUF | 211.3 | 12953 |
| SmolLM2-360M | BF16 SafeTensors | 206.2 | — |
| Llama-3.2-1B | BF16 SafeTensors | 109.9 | — |

### Notes

- **Llama-3.2-1B Q8_0 at 1.01×**: infernum matches llama.cpp for this model/format combination. Llama-3.2-1B is a 16-layer model with a larger intermediate-to-hidden ratio — the quantised GEMV profile is different from SmolLM2-360M (32 layers).
- **SmolLM2-360M gap (0.78–0.88×)**: llama.cpp uses heavily tuned custom CUDA kernels for each quantisation format. The gap is larger for Q4_0 than Q8_0 because Q4_0 has smaller blocks (18 bytes vs 34 bytes for Q8_0), so kernel launch overhead is a larger fraction of total time — llama.cpp amortises this better.
- **Q4_0 now works**: was previously crashing (`UnsupportedDtype: GGML type 3`). Fixed by adding Q4_1 block dequantisation (commit `189538d`) — many Q4_0 GGUF files store some tensors in Q4_1 format.
- **Prefill is not comparable**: infernum uses a token-by-token paged-decode loop; llama.cpp uses batched GEMM with tensor cores. llama.cpp's 12–20k tok/s prefill reflects a fundamentally different code path.
- **Key optimisations in this cycle (PR #77):** shallow `CudaTensor::clone()` (eliminated 24 GB/5 s of D→D copies); paged-attention shared-memory correctness fix (+12–69% decode, fixed 1024-token crash); QKV fused GEMV (+5.4%); Q8_0 linear_pair/triple input-quantisation deduplication; BF16 cos/sin inputs; async prefill pipeline.

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
