# CUDA Multi-GPU Benchmark Results

See [performance.md](../performance.md) for methodology.

## Current Results

**Node:** 8× NVIDIA A100-SXM4-80GB — NVLink interconnect | Driver 590.48.01 | CUDA 13.1  
**Run date:** 2026-05-22 — clean sequential runs, all GPUs idle between measurements  
**infernum commit:** `perf/measurement-doc` branch | **llama.cpp commit:** `40d5358` (build 1)

### Multi-GPU decode — large models (greedy argmax, 200 tokens)

infernum uses NCCL tensor-parallel (ShardedGraphEngine for TP>1).  
llama.cpp single-GPU is the reference baseline; TP>1 provides no benefit for models that fit on one GPU.

| Model | Format | infernum TP=2 (tok/s) | infernum TP=8 (tok/s) | llama.cpp 1GPU (tok/s) | ratio (TP=2) |
| ----- | ------ | --------------------: | --------------------: | ---------------------: | -----------: |
| Llama-3.1-70B | Q4_0 GGUF | 10.4 | 3.8 | 30.1 | 0.35x |
| Qwen3-72B | Q4_K_M GGUF | 7.3 | 3.5 | 23.4 | 0.31x |
| Qwen3-235B-A22B | Q4_K_M GGUF | — | — | — | — |

**TP decode is structurally eager:** `all_reduce_sum` ops are flagged capture-unsafe because NCCL AllReduce internally calls `cuMemAlloc`, which is illegal inside a CUDA graph capture window. TP=1 uses the CUDA graph fast path; TP>1 runs every decode step eagerly via the interpreter. This is a hard constraint, not a configuration choice.

**TP degrades throughput for models that fit on fewer GPUs:** Both 70B and 72B fit on a single A100-80GB at their GGUF quantizations. NCCL AllReduce overhead exceeds any per-GPU compute savings at single-batch decode, so TP=8 is slower than TP=2. Multi-GPU gains require models too large for fewer GPUs, or batching.

**Footnotes:**
- Llama-3.1-70B Q4_0: on-GPU Q4_0 GEMV kernels (0.5 byte/param), block-sharded across ranks at quantized boundaries.
- Qwen3-72B Q4_K_M: BF16 dequant at load (Q4_K GPU kernel not yet implemented; 2 byte/param on GPU). Some layers have column counts not divisible by `32 × world_size` at TP=8, causing automatic fallback to BF16 sharding. TP=8 slower than TP=2 (NCCL overhead dominates; model fits on 2 GPUs).
- Qwen3-235B-A22B: blocked — see Pending table.

---

### Pending

| Item | Status |
| ---- | ------ |
| Qwen3-235B-A22B Q4_K_M | **Blocked (2 issues).** (1) `qwen3moe` GGUF architecture not yet supported in the graph engine — needs config parsing + stacked-expert weight loading. (2) Q4_K_M has no GPU GEMV kernel; all 128 experts across 94 layers would dequant to BF16 (~530 GB), exceeding 8×80 GB VRAM. Both must be resolved before any TP run is possible. |
| Q4_K GPU kernel | Q4_K_M dequants to BF16 at load; native GPU Q4_K GEMV kernel would reduce Qwen3-72B VRAM usage and unlock Qwen3-235B-A22B inference |
| Qwen3-MoE GGUF support | `qwen3moe` architecture needs: GGUF metadata → `QwenConfig` MoE fields, `safetensors_to_gguf_name` MoE weight mappings (stacked `ffn_{gate,up,down}_exps.weight`), expert-slice loading in the GGUF weight loader |
| CUDA graph TP | NCCL AllReduce blocks CUDA graph capture; TP>1 runs eager |

---

## History

## 2026-05-22 — TP=8 runs for 70B and 72B models; 235B blocked

- **New:** Llama-3.1-70B Q4_0 at TP=8: 3.8 tok/s (vs TP=2 at 10.4 tok/s — NCCL overhead dominates, model fits on 2 GPUs)
- **New:** Qwen3-72B Q4_K_M at TP=8: 3.5 tok/s (vs TP=2 at 7.3 tok/s — same overhead pattern)
- **Fix:** TP=8 alignment check — shapes not divisible by `32 × world_size` now fall back to BF16 sharding instead of panic (affects Qwen3-72B layers with 29568-column weights)
- **Fix:** Split GGUF loading — `GgufLoader::from_file_or_split` auto-detects `NNNNN-of-MMMMM.gguf` shard sets; enables loading Qwen3-235B-A22B's 5-shard Q4_K_M GGUF
- **Blocked:** Qwen3-235B-A22B Q4_K_M — `qwen3moe` GGUF architecture unsupported + Q4_K has no GPU GEMV kernel (BF16 dequant would require ~530 GB, exceeding 640 GB total VRAM)

---

## 2026-05-22 — GGUF quantized loading + TP GGUF (infernum)

- **New:** GGUF weights for Q8_0 and Q4_0 now stay quantized on GPU (previously dequanted to BF16 at load). On-GPU GEMV kernels dequantize on the fly.
- **New:** `--gpus N` flag enables GGUF tensor-parallel loading. Q8_0/Q4_0 weights are block-sharded across ranks without converting to BF16.
- **New:** Q4_K, Q5_K, Q6_K, Q5_0 formats are now recognized; unsupported-for-GPU types (Q4_K, Q5_K) dequant to BF16 at load.
- **New:** Llama-3.1-70B Q4_0 TP=2 and Qwen3-72B Q4_K_M TP=2 now produce results.

### infernum: 70B/72B GGUF TP=2 (200 tokens)

| Model | Format | GPUs | decode (tok/s) |
| ----- | ------ | ---: | -------------: |
| Llama-3.1-70B | Q4_0 GGUF | 2 | 10.4 |
| Qwen3-72B | Q4_K_M GGUF | 2 | 7.3 |

Llama-70B uses on-GPU Q4_0 kernels (0.5 byte/param). Qwen3-72B dequants Q4_K to BF16 at load (Q4_K GPU kernel pending).

---

## 2026-05-22 — Full re-run (clean sequential), all models

- **Node:** 8× NVIDIA A100-SXM4-80GB — NVLink interconnect
- **Driver:** 590.48.01 | CUDA 13.1
- **infernum commit:** `perf/measurement-doc` branch | **llama.cpp commit:** `40d5358` (build 1)
- All runs sequential (no parallel GPU sharing), GPUs idle between runs

### infernum: single-GPU decode (200 tokens, CudaGraphEngine, argmax)

| Model | Format | decode (tok/s) |
| ----- | ------ | -------------: |
| Qwen3-8B | Q8_0 GGUF | 59.4 |
| Gemma-2-9B | Q8_0 GGUF | 14.6 |
| Qwen3-8B | BF16 SafeTensors | 52.6 |
| Gemma-2-9B | BF16 SafeTensors | 13.4 |

### infernum: Qwen3-8B BF16 — TP scaling (100 decode tokens, argmax)

| GPUs | decode (tok/s) | engine |
| ---: | -------------: | ------ |
| 1 | 52.6 | CudaGraphEngine |
| 2 | 17.5 | ShardedGraphEngine |
| 4 | 10.0 | ShardedGraphEngine |
| 8 |  5.6 | ShardedGraphEngine |

### llama.cpp: all models (pp512 prefill + tg256 decode, best of 3)

| Model | Format | GPUs | decode (tok/s) | prefill pp512 (tok/s) |
| ----- | ------ | ---: | -------------: | --------------------: |
| Qwen3-8B | Q8_0 | 1 | 127.0 ± 3.0 | — |
| Gemma-2-9B | Q8_0 | 1 | 94.5 ± 0.8 | 3404 ± 457 |
| Llama-3.1-70B | Q4_0 | 1 | 30.1 ± 0.0 | 530 ± 11 |
| Llama-3.1-70B | Q4_0 | 2 | 30.1 ± 0.0 | 532 ± 6 |
| Llama-3.1-70B | Q4_0 | 4 | 26.9 ± 0.3 | 532 ± 3 |
| Qwen3-72B | Q4_K_M | 1 | 23.4 ± 0.0 | 517 ± 13 |
| Qwen3-72B | Q4_K_M | 2 | 23.6 ± 0.0 | 520 ± 7 |

---

## 2026-05-21 — A100 ×8 Node — Large-Model Baseline (llama.cpp only)

- **Node:** 8× NVIDIA A100-SXM4-80GB (81152 MiB each, 640 GiB total) — NVLink interconnect
- **Driver:** 590.48.01 | CUDA 13.1
- **infernum commit:** `891ced5`
- **llama.cpp commit:** `40d5358` (build: 1), best of 3 reps, `-ngl 99`

**Note:** 2026-05-21 numbers were measured while other processes may have been running on the same node. The 2026-05-22 clean re-run supersedes all of these.

### Llama-3.1-70B Q4_0 — GPU scaling (llama.cpp)

| GPUs | decode (tok/s) |
| ---: | -------------: |
| 1 | 23.58 |
| 2 | 23.62 |
| 4 | 19.21 |

### Qwen3-72B Q4_K_M — GPU scaling (llama.cpp)

| GPUs | decode (tok/s) |
| ---: | -------------: |
| 1 | 17.69 |
| 2 | 17.70 |
