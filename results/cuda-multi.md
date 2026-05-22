# CUDA Multi-GPU Benchmark Results

See [performance.md](../performance.md) for methodology.

## Current Results

**Node:** 8× NVIDIA A100-SXM4-80GB — NVLink interconnect | Driver 590.48.01 | CUDA 13.1  
**Run date:** 2026-05-22 (clean sequential runs, no GPU interference)  
**infernum commit:** `perf/measurement-doc` branch | **llama.cpp commit:** `40d5358` (build 1)

---

### Qwen3-8B — TP scaling decode (greedy argmax, 100 tokens)

infernum: BF16 SafeTensors, CudaGraphEngine (TP=1) or ShardedGraphEngine (TP>1)  
llama.cpp: Q8_0 GGUF, best of 3, `-ngl 99`, `tg100`

| GPUs | infernum BF16 (tok/s) | llama.cpp Q8_0 (tok/s) | ratio |
| ---: | --------------------: | ---------------------: | ----: |
| 1    | 52.6                  | 127.0                  | 0.41x |
| 2    | 17.5                  | 126.9                  | 0.14x |
| 4    | 10.0                  | 112.5                  | 0.09x |
| 8    |  5.6                  | 110.4                  | 0.05x |

**TP decode is structurally eager:** `all_reduce_sum` ops are flagged capture-unsafe — NCCL's AllReduce internally calls `cuMemAlloc` which is illegal inside a CUDA graph capture window. TP=1 uses the CUDA graph fast path; TP>1 runs every decode step eagerly via the interpreter. This is a hard constraint, not a configuration choice.

**Format gap at TP=1:** infernum BF16 (~2 bytes/param) vs llama.cpp Q8_0 (~1 byte/param). Decode is memory-bandwidth-bound. Even at TP=1 with the CUDA graph path, infernum runs at 0.41× of llama.cpp because it moves twice as much weight per step.

**TP degrades throughput:** Qwen3-8B fits on one GPU. NCCL AllReduce overhead exceeds any per-GPU compute savings at single-batch decode, so throughput falls as GPU count increases. Meaningful multi-GPU gains require either (a) models that don't fit on a single GPU, or (b) batching multiple concurrent requests.

---

### infernum GGUF loading (new — 2026-05-22)

Q8_0 GGUF weights are dequantized to BF16 on host before upload. Runtime behaviour is identical to BF16 SafeTensors; the only difference is file size and load time.

| Model | Format | infernum (tok/s) | llama.cpp (tok/s) | ratio |
| ----- | ------ | ---------------: | ----------------: | ----: |
| Qwen3-8B | BF16 SafeTensors | 52.6 | 127.0 | 0.41x |
| Qwen3-8B | Q8_0 GGUF (infernum new) | 53.0 | 127.0 | 0.42x |
| Gemma-2-9B | BF16 SafeTensors | 13.4 | 94.5 | 0.14x |
| Gemma-2-9B | Q8_0 GGUF (infernum new) | 13.4 | 94.5 | 0.14x |

GGUF and SafeTensors are within noise — confirming correct dequantization. The format gap vs llama.cpp remains because llama.cpp runs native Q8_0 GEMV kernels (never dequantizes at inference time).

---

### 8B/9B models — single GPU decode (greedy, 100 tokens)

| Model | Format | infernum (tok/s) | llama.cpp (tok/s) | ratio |
| ----- | ------ | ---------------: | ----------------: | ----: |
| Qwen3-8B | BF16 SafeTensors | 52.6 | 127.0 Q8_0 | 0.41x |
| Gemma-2-9B | BF16 SafeTensors | 13.4 | 94.5 Q8_0 | 0.14x |

Gemma-2-9B gap is wider (0.14×). Gemma has head_dim=256 (vs 128 for Qwen) and 42 layers. The prefill path is also slower than expected (infernum: 137 tok/s vs llama.cpp: 3404 tok/s at 512 tokens) — this is a known gap in infernum's Gemma attention kernel, not a measurement artifact.

---

### Large models — llama.cpp baseline (decode tg256, prefill pp512)

| Model | Format | GPUs | decode (tok/s) | prefill (tok/s) |
| ----- | ------ | ---: | -------------: | --------------: |
| Llama-3.1-70B | Q4_0 | 1 | 30.1 | 530 |
| Llama-3.1-70B | Q4_0 | 2 | 30.1 | 532 |
| Llama-3.1-70B | Q4_0 | 4 | 26.9 | 532 |
| Qwen3-72B | Q4_K_M | 1 | 23.4 | 517 |
| Qwen3-72B | Q4_K_M | 2 | 23.6 | 520 |

Llama-70B (37 GiB) fits on one 80GB A100; adding GPUs gives no decode speedup and degrades it at 4× (NCCL overhead). Qwen3-72B (44 GiB) also fits on one GPU; same pattern.

---

### Pending

| Model | Status |
| ----- | ------ |
| Llama-3.1-70B BF16 TP=2 infernum | No BF16 weights downloaded (gated HF repo) |
| Qwen3-72B BF16 TP=2 infernum | Weights not downloaded |
| DeepSeek-V3 | infernum CUDA not implemented; GGUF not available |

---

## History

## 2026-05-22 — Full re-run (clean sequential), all models

- **Node:** 8× NVIDIA A100-SXM4-80GB — NVLink interconnect
- **Driver:** 590.48.01 | CUDA 13.1
- **infernum commit:** `perf/measurement-doc` branch | **llama.cpp commit:** `40d5358` (build 1)
- All runs sequential (no parallel GPU sharing), GPUs idle between runs

### infernum: Qwen3-8B BF16 — TP scaling (100 decode tokens, argmax)

| GPUs | decode (tok/s) | engine |
| ---: | -------------: | ------ |
| 1 | 52.6 | CudaGraphEngine |
| 2 | 17.5 | ShardedGraphEngine |
| 4 | 10.0 | ShardedGraphEngine |
| 8 |  5.6 | ShardedGraphEngine |

### infernum: single-GPU decode (100 tokens, CudaGraphEngine, argmax)

| Model | Format | decode (tok/s) | prefill pp512 (tok/s) |
| ----- | ------ | -------------: | --------------------: |
| Qwen3-8B | BF16 SafeTensors | 52.6 | 2417 |
| Qwen3-8B | Q8_0 GGUF | 53.0 | — |
| Gemma-2-9B | BF16 SafeTensors | 13.4 | 137 |
| Gemma-2-9B | Q8_0 GGUF | 13.4 | — |

### llama.cpp: Qwen3-8B Q8_0 — GPU scaling (tg100, best of 3)

| GPUs | decode (tok/s) |
| ---: | -------------: |
| 1 | 127.0 ± 3.0 |
| 2 | 126.9 ± 2.8 |
| 4 | 112.5 ± 0.4 |
| 8 | 110.4 ± 0.6 |

### llama.cpp: large models (pp512 + tg256, best of 3)

| Model | Format | GPUs | decode (tok/s) | prefill (tok/s) |
| ----- | ------ | ---: | -------------: | --------------: |
| Gemma-2-9B | Q8_0 | 1 | 94.5 ± 0.8 | 3404 ± 457 |
| Llama-3.1-70B | Q4_0 | 1 | 30.1 ± 0.0 | 530 ± 11 |
| Llama-3.1-70B | Q4_0 | 2 | 30.1 ± 0.0 | 532 ± 6 |
| Llama-3.1-70B | Q4_0 | 4 | 26.9 ± 0.3 | 532 ± 3 |
| Qwen3-72B | Q4_K_M | 1 | 23.4 ± 0.0 | 517 ± 13 |
| Qwen3-72B | Q4_K_M | 2 | 23.6 ± 0.0 | 520 ± 7 |

---

## 2026-05-22 — Qwen3-8B TP scaling (infernum, initial run — later superseded by clean re-run above)

- Results from the initial session had GPU interference from parallel benchmark execution; superseded by the clean run above.
- TP=4 deadlock fix documented here: pre-downloading rank-0 GPU tensors to host before `thread::scope` in `ShardedGraphEngine::forward_batch_decode` eliminates cross-rank `cuStreamSynchronize` deadlock on NCCL AllReduce ring.

---

## 2026-05-21 — A100 ×8 Node — Large-Model Baseline (llama.cpp only)

- **Node:** 8× NVIDIA A100-SXM4-80GB (81152 MiB each, 640 GiB total) — NVLink interconnect
- **Driver:** 590.48.01 | CUDA 13.1
- **Decode tokens:** 256 | **Prefill tokens:** 512
- **infernum commit:** `891ced5`
- **llama.cpp commit:** `40d5358` (build: 1), best of 3 reps, `-ngl 99`

### Llama-3.1-70B Q4_0 — GPU scaling (llama.cpp)

| GPUs | config | decode (tok/s) | prefill (tok/s, pp512) |
| ---: | ------ | -------------: | ---------------------: |
| 1 | `CUDA_VISIBLE_DEVICES=0` | 23.58 | 528 |
| 2 | `CUDA_VISIBLE_DEVICES=0,1` | 23.62 | — |
| 4 | `CUDA_VISIBLE_DEVICES=0,1,2,3` | 19.21 | 530 |

**Note:** 2026-05-21 numbers were measured while other processes may have been running on the same node. The 2026-05-22 clean re-run (30.1 tok/s single GPU) supersedes these.

### Qwen3-72B Q4_K_M — GPU scaling (llama.cpp)

| GPUs | config | decode (tok/s) |
| ---: | ------ | -------------: |
| 1 | `CUDA_VISIBLE_DEVICES=4` | 17.69 |
| 2 | `CUDA_VISIBLE_DEVICES=4,5` | 17.70 |

**Note:** 2026-05-21 numbers were measured while other processes may have been running. The 2026-05-22 clean re-run (23.4 tok/s single GPU) supersedes these.
