# CUDA Multi-GPU Benchmark Results

See [performance.md](../performance.md) for methodology.

---

## 2026-05-21 — A100 ×8 Node — Large-Model Baseline (llama.cpp only)

- **Node:** 8× NVIDIA A100-SXM4-80GB (81152 MiB each, 640 GiB total) — NVLink interconnect
- **Driver:** 590.48.01 | CUDA 13.1
- **Decode tokens:** 256 | **Prefill tokens:** 512
- **infernum commit:** `891ced5`
- **llama.cpp commit:** `40d5358` (build: 1), best of 3 reps, `-ngl 99`
- **infernum TP status:** Tensor-parallel inference is not yet implemented in infernum. No infernum numbers appear in this file. All rows are llama.cpp baselines establishing targets for future infernum multi-GPU support.

### Llama-3.1-70B Q4_0 — GPU scaling (llama.cpp)

Model size: 37.22 GiB — **fits on a single A100 (80 GiB)**, so multi-GPU gives no decode speedup.

| GPUs | config | decode (tok/s) | prefill (tok/s, pp512) | notes |
| ---: | ------ | -------------: | ---------------------: | ----- |
| 1 | `CUDA_VISIBLE_DEVICES=0` | 23.58 | 528 | bandwidth-bound on single GPU |
| 2 | `CUDA_VISIBLE_DEVICES=0,1` | 23.62 | — | no speedup — model fits in 1 GPU; NVLink overhead |
| 4 | `CUDA_VISIBLE_DEVICES=0,1,2,3` | 19.21 | 530 | slower — NCCL sync cost exceeds any compute benefit |

**Observation:** Adding GPUs to a model that already fits on one GPU does not improve decode throughput and can degrade it. The true multi-GPU benefit is memory capacity: enabling models that exceed a single GPU's VRAM. At Q4_0 quantization (~0.5 bytes/param), even 70B models fit in 37 GB — well within 80 GB. The benefit of multi-GPU becomes visible only with BF16/FP16 precision (~2 bytes/param), where 70B requires ~140 GB → 2 GPUs minimum.

### Qwen3-72B Q4_K_M — GPU scaling (llama.cpp)

Model size: 44.11 GiB — fits on a single A100. Same pattern as Llama-70B.

| GPUs | config | decode (tok/s) | prefill (tok/s, pp512) | notes |
| ---: | ------ | -------------: | ---------------------: | ----- |
| 1 | `CUDA_VISIBLE_DEVICES=4` | 17.69 | 517 | bandwidth-bound on single GPU |
| 2 | `CUDA_VISIBLE_DEVICES=4,5` | 17.70 | — | no speedup — model fits in 1 GPU |

---

## 2026-05-22 — A100 ×8 Node — Qwen3-8B BF16 TP Scaling (infernum)

- **Node:** 8× NVIDIA A100-SXM4-80GB — NVLink interconnect
- **Driver:** 590.48.01 | CUDA 13.1
- **Model:** Qwen/Qwen3-8B (BF16 SafeTensors) — ~16 GB
- **Decode tokens:** 100 | **Decoding:** greedy (argmax)
- **infernum commit:** `perf/measurement-doc` branch
- **Notes:** TP via `generate_parallel` example; CUDA graph engine (`capture_unsafe=true` path bypassed for TP — all steps run eagerly via `execute()`)

### Qwen3-8B BF16 — TP scaling (infernum)

| GPUs (TP) | decode (tok/s) | wall time (100 tok) | notes |
| --------: | -------------: | ------------------: | ----- |
| 2 | 16.7 | 6.00s | |
| 4 | 10.8 | 9.29s | fixed NCCL collective deadlock (see below) |
| 8 | 5.9 | 16.90s | AllReduce overhead dominates at this model scale |

**TP=4/TP=8 hang fix:** The original code passed rank-0 GPU tensors into per-rank threads. Inside each rank's thread, `dtoh_sync_copy_into` called `cuStreamSynchronize(GPU-0 stream)`. When ranks 0/1 had already submitted NCCL AllReduces to GPU-0's stream, ranks 2/3 blocked in `cuStreamSynchronize` and could not participate — deadlocking the ring. Fix: pre-download all rank-0 GPU tensors to host in `ShardedGraphEngine::forward_batch_decode` before spawning rank threads; each rank receives host data and uploads to its own device via `forward_batch_decode_precomputed`.

**Observation:** Qwen3-8B fits comfortably in a single A100. TP=2 gives the best decode rate; TP=4 and TP=8 show diminishing returns because NCCL AllReduce communication cost grows with world size while the per-GPU compute shrinks. The model is too small to be compute-bound at this scale. Meaningful multi-GPU gains for 8B-class models require batching (multiple concurrent requests), not just TP degree.

---

### What is missing (pending BF16 70B+ access)

The performance.md primary H100×8 targets are:

| Family | Model | Format | Needed VRAM | Status |
| ------ | ----- | ------ | ----------: | ------ |
| Llama | Llama-3.1-70B | BF16 | ~140 GB (2× A100) | Gated — no HF access |
| Qwen | Qwen3-235B-A22B | BF16 | ~470 GB (6× A100) | Not downloaded |
| Qwen | Qwen3-72B | BF16 | ~144 GB (2× A100) | Not downloaded |
| DeepSeek | DeepSeek-V3 | GGUF Q4 | ~335 GB (4–5× A100) | GPTQ INT4 cached (329 GB) but GGUF not available; llama.cpp requires GGUF |

**DeepSeek-V3 note:** The full GPTQ INT4 model (`OPEA/DeepSeek-V3-int4-sym-gptq-inc`, 329 GB, 71 SafeTensors shards) is already in the model cache. However:
- infernum CUDA support for DeepSeek is not yet implemented
- llama.cpp requires GGUF format; GPTQ SafeTensors are not supported

To benchmark DeepSeek-V3 with llama.cpp, the model must first be converted to GGUF. The GPTQ weights could be dequantized to BF16 and re-quantized to GGUF Q4_0, but this is a multi-step process requiring significant compute time and scratch space (~700 GB intermediate BF16 → GGUF pipeline). Tracking as a follow-up.

### Expected numbers (not yet measured)

These are the targets to fill in once access and format issues are resolved:

| Model | Format | GPUs | Expected llama.cpp decode | Expected infernum decode |
| ----- | ------ | ---: | -----------------------: | -----------------------: |
| Llama-3.1-70B | BF16 GGUF | 2 | ~60–80 tok/s | pending TP |
| Qwen3-72B | BF16 GGUF | 2 | ~55–75 tok/s | pending TP |
| Qwen3-235B-A22B | Q4_K_M | 4–6 | ~15–25 tok/s | pending TP |
| DeepSeek-V3 | GGUF Q4 | 4–5 | ~8–15 tok/s | pending TP + DeepSeek CUDA |
