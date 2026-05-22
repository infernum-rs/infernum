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
| Llama / Llama-3.2-3B | GGUF Q4_0 | 21.6 | 59.6 | 0.36x | 2026-05-22 |
| Llama / Llama-3.2-3B | GGUF Q8_0 | 21.5 | 36.9 | 0.58x | 2026-05-22 |
| Llama / SmolLM2-360M | GGUF Q4_0 | 58.1 | 226.7 | 0.26x | 2026-05-21 |
| Llama / SmolLM2-360M | GGUF Q8_0 | 56.1 | 184.4 | 0.30x | 2026-05-21 |
| Llama / SmolLM2-360M | SafeTensors F32 | 25.7 | — | — | 2026-05-21 |
| Qwen / Qwen3-0.6B | SafeTensors BF16 | 21.1 | — | — | 2026-05-21 |
| Gemma / gemma-2-2b-it | GGUF Q8_0 | 22.1 | 29.1 | 0.76x | 2026-05-21 |
| Gemma / gemma-2-2b-it | GGUF Q4_K_M | — | 65.7 | — | 2026-05-21 |

### Prefill throughput (tok/s)

| Model | Format | infernum | llama.cpp | ratio | Date |
| ----- | ------ | -------: | --------: | ----: | ---- |
| Llama / Llama-3.1-8B | GGUF Q4_0 | — | 290.8 | — | 2026-05-22 |
| Llama / Llama-3.2-3B | GGUF Q4_0 | — | 686.9 | — | 2026-05-22 |
| Llama / Llama-3.2-3B | GGUF Q8_0 | — | 683.0 | — | 2026-05-22 |
| Llama / SmolLM2-360M | SafeTensors F32 | 252 | — | — | 2026-05-21 |
| Llama / SmolLM2-360M | GGUF Q8_0 | 4.0 | 4541 | 0.001x | 2026-05-21 |
| Llama / SmolLM2-360M | GGUF Q4_0 | 4.3 | 4596 | 0.001x | 2026-05-21 |
| Qwen / Qwen3-0.6B | SafeTensors BF16 | 86 | — | — | 2026-05-21 |
| Gemma / gemma-2-2b-it | GGUF Q8_0 | 0.6 | 938 | 0.001x | 2026-05-21 |
| Gemma / gemma-2-2b-it | GGUF Q4_K_M | — | 903 | — | 2026-05-21 |

---

## History

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
- **8B not yet measured:** The M3 Pro 18 GB can fit Llama 3.1 8B in Q4_0/Q8_0 format, but the HuggingFace xet storage system was too slow to download the file in this session (<0.3 MB/s reconstructed rate). A future session will add 8B numbers.
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
