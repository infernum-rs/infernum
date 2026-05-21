# CUDA Single-GPU Benchmark Results

Measured with `bench_cuda_small.sh`. See [performance.md](../performance.md) for methodology.

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
