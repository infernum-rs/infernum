# CUDA Multi-GPU Benchmark Results

See [performance.md](../performance.md) for methodology.

## Current Results

**Node:** 8× NVIDIA A100-SXM4-80GB — NVLink interconnect | Driver 590.48.01 | CUDA 13.1  
**Run date:** 2026-05-22 — clean sequential runs, all GPUs idle between measurements  
**infernum commit:** `perf/measurement-doc` branch | **llama.cpp commit:** `40d5358` (build 1)

### Single-GPU decode — all models (greedy, 100 tokens)

infernum column is BF16 SafeTensors unless noted. llama.cpp uses best available quantization.

| Model | Params | infernum (tok/s) | llama.cpp (tok/s) | llama.cpp format | ratio |
| ----- | -----: | ---------------: | ----------------: | :--------------- | ----: |
| Qwen3-8B | 8B | 52.6 | 127.0 | Q8_0 | 0.41x |
| Qwen3-8B ¹ | 8B | 53.0 | 127.0 | Q8_0 | 0.42x |
| Gemma-2-9B | 9B | 13.4 | 94.5 | Q8_0 | 0.14x |
| Gemma-2-9B ¹ | 9B | 13.4 | 94.5 | Q8_0 | 0.14x |
| Llama-3.1-70B | 70B | — | 30.1 | Q4_0 | — |
| Qwen3-72B | 72B | — | 23.4 | Q4_K_M | — |

¹ infernum loading Q8_0 GGUF (new feature, 2026-05-22). Dequantizes to BF16 on host before upload — identical runtime behaviour to SafeTensors. llama.cpp runs native Q8_0 GEMV kernels and never dequantizes.

**Main performance gap:** infernum uses BF16 weights (2 bytes/param); llama.cpp uses Q8_0 (1 byte/param). Decode is memory-bandwidth-bound, so the format alone accounts for roughly 2× of the gap. Even with matching Q8_0 GGUF input, infernum's runtime is BF16 (0.42×) while llama.cpp's runtime stays Q8_0.

**Gemma gap is wider** (0.14×): Gemma-2-9B has head_dim=256 (vs 128 for Qwen), 42 layers, and a known slow prefill path in infernum (137 tok/s vs llama.cpp's 3404 tok/s at pp512). The decode gap is also larger than Qwen's, likely due to suboptimal BF16 attention for large head_dim.

**Llama-70B / Qwen3-72B (infernum):** BF16 SafeTensors weights not downloaded; pending.

---

### Qwen3-8B — TP scaling decode (greedy argmax, 100 tokens)

infernum uses ShardedGraphEngine for TP>1 (eager path — see note).  
llama.cpp uses tensor-parallel with best of 3, `-ngl 99`, `tg100`.

| GPUs | infernum BF16 (tok/s) | llama.cpp Q8_0 (tok/s) | ratio |
| ---: | --------------------: | ---------------------: | ----: |
| 1    | 52.6                  | 127.0                  | 0.41x |
| 2    | 17.5                  | 126.9                  | 0.14x |
| 4    | 10.0                  | 112.5                  | 0.09x |
| 8    |  5.6                  | 110.4                  | 0.05x |

**TP decode is structurally eager:** `all_reduce_sum` ops are flagged capture-unsafe because NCCL's AllReduce internally calls `cuMemAlloc`, which is illegal inside a CUDA graph capture window. TP=1 uses the CUDA graph fast path; TP>1 runs every decode step eagerly via the interpreter. This is a hard constraint, not a configuration choice.

**TP degrades throughput for this model:** Qwen3-8B fits on one GPU. NCCL AllReduce overhead exceeds any per-GPU compute savings at single-batch decode, so throughput falls as GPU count increases. Multi-GPU gains require models too large for one GPU, or batching.

**llama.cpp multi-GPU pattern:** llama.cpp uses layer-parallel sharding (not NCCL tensor-parallel for decode), so 2-GPU barely changes decode throughput (127→127) and 4-8 GPU degrades less than infernum (127→112). Same root cause: the model is too small to benefit.

---

### Large models — llama.cpp baseline (pp512 prefill + tg256 decode, best of 3)

| Model | Format | GPUs | decode (tok/s) | prefill pp512 (tok/s) |
| ----- | ------ | ---: | -------------: | --------------------: |
| Llama-3.1-70B | Q4_0 | 1 | 30.1 | 530 |
| Llama-3.1-70B | Q4_0 | 2 | 30.1 | 532 |
| Llama-3.1-70B | Q4_0 | 4 | 26.9 | 532 |
| Qwen3-72B | Q4_K_M | 1 | 23.4 | 517 |
| Qwen3-72B | Q4_K_M | 2 | 23.6 | 520 |

Both models fit on a single 80GB A100 at these quantizations. Adding GPUs provides no decode benefit.

---

### Pending

| Model | Status |
| ----- | ------ |
| Llama-3.1-70B BF16 TP=2 infernum | BF16 weights not downloaded (gated HF repo) |
| Qwen3-72B BF16 TP=2 infernum | BF16 weights not downloaded |
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
