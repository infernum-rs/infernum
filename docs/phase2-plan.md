# Phase 2: KV Cache & Quantization

## Overview

Two independent tracks that can be developed in parallel (separate branches/worktrees).
They converge at the end: quantized weights + KV cache = fast, memory-efficient inference.

### Why parallel works

| Area | Track A: KV Cache | Track B: Quantization |
|------|-------------------|----------------------|
| **Model forward signature** | Changes (adds kv_cache, positions args) | No change |
| **Attention op** | New `attention_with_kv_cache` | No change |
| **Weight loading** | No change | New dtype paths (INT8/INT4) |
| **Linear projection** | No change | New quantized matmul kernels |
| **Tensor types** | No change | New `QuantizedTensor` type |
| **Generation loop** | Rewritten (incremental decode) | No change |
| **RoPE** | position_offset becomes meaningful | No change |

The only merge conflict will be in `model.rs` (forward signature), which is a straightforward resolution:
the merged model accepts both quantized weights and a KV cache.

---

## Track A: KV Cache (branch: `phase-2-kv-cache`)

### Goal

Incremental decoding: process only the new token each step instead of the full sequence.
This changes generation from O(n²) forward passes to O(n).

### Step A1: KV Cache Data Structure

Create `infernum/src/cuda/kv_cache.rs`:

- `KvCache` struct holding pre-allocated K and V buffers per layer
  - Shape: `(num_layers, max_seq_len, num_kv_heads, head_dim)`
  - Tracks current sequence length
- Methods:
  - `new(ctx, num_layers, max_seq_len, num_kv_heads, head_dim)` — allocate
  - `append(layer_idx, k, v)` — write new K/V at current position, advance length
  - `get(layer_idx)` — return K/V slices up to current length
  - `reset()` — reset length to 0 (reuse memory)

### Step A2: Attention with KV Cache

New op `attention_kv` in `infernum/src/cuda/ops/attention.rs`:

```
fn attention_kv(
    q: &CudaTensor<f32>,       // (1, num_heads, head_dim) — single new token
    kv_cache: &mut KvCache,     // accumulated K/V
    layer_idx: usize,
    k_new: &CudaTensor<f32>,   // (1, num_kv_heads, head_dim)
    v_new: &CudaTensor<f32>,   // (1, num_kv_heads, head_dim)
) -> Result<CudaTensor<f32>>   // (1, num_heads, head_dim)
```

- Append k_new/v_new to cache
- Compute attention: Q against all cached K, apply to all cached V
- Causal mask is implicit (cache only contains past tokens)
- No need for explicit mask during incremental decode

### Step A3: Model Forward with KV Cache

Update `LlamaModel` in `infernum-llama/src/model.rs`:

- Add `forward_next_token(&self, token_id: u32, kv_cache: &mut KvCache, position: usize)` method
  - Embeds single token
  - Runs through layers, each updating the KV cache
  - RoPE uses `position` as `position_offset`
  - Returns logits for the single token
- Keep existing `forward(&self, input_ids: &[u32])` for prefill (process full prompt)
  - Prefill populates the KV cache all at once

### Step A4: Two-Phase Generation Loop

Update `generate` / `generate_sampled`:

```
1. Prefill: forward(prompt_tokens) -> populates KV cache, get first next-token logits
2. Decode loop: forward_next_token(last_token, kv_cache, position) -> next logits
```

- Prefill processes the full prompt in one pass (existing attention, fills KV cache)
- Each decode step processes only 1 token (fast)
- RoPE position_offset = current sequence length

### Step A5: Tests & Validation

- Unit test: KV cache append/get round-trip
- Unit test: `attention_kv` matches `attention` for single-step decode
- Integration test: `generate` with KV cache produces identical output to without
- Benchmark: measure tokens/sec improvement (should be dramatic for long sequences)

---

## Track B: Quantization (branch: `phase-2-quantization`)

### Goal

Load and run models stored in INT8 and INT4 quantized formats (GGUF and/or GPTQ).
Reduces memory footprint ~2-4x, enabling larger models on the same GPU.

### Step B1: Quantized Data Types

Add to `infernum/src/dtype.rs`:

- `DType::Q8_0` — 8-bit quantization (block size 32, one f16 scale per block)
- `DType::Q4_0` — 4-bit quantization (block size 32, one f16 scale per block)

### Step B2: Quantized Tensor Storage

Create `infernum/src/cuda/quantized.rs`:

- `QuantizedTensor` struct — stores raw quantized bytes + scale factors on GPU
  - Fields: `data: CudaSlice<u8>`, `scales: CudaSlice<f16>`, `shape`, `dtype`, `block_size`
  - Not parameterized by element type (the dtype field determines interpretation)
- Dequantize-on-the-fly during matmul (don't expand to f32 in memory)

### Step B3: Quantized MatMul Kernel

New op in `infernum/src/cuda/ops/matmul.rs` (or a new `quantized_matmul.rs`):

- `matmul_q8(input: &CudaTensor<f32>, weight: &QuantizedTensor) -> CudaTensor<f32>`
  - Input activations remain f32
  - Weights are dequantized per-block inside the kernel
  - Output is f32
- `matmul_q4(input: &CudaTensor<f32>, weight: &QuantizedTensor) -> CudaTensor<f32>`
  - Same pattern, 4-bit dequantization

### Step B4: GGUF Weight Loader

Create `infernum/src/weights/gguf.rs`:

- Parse GGUF file format (header, metadata, tensor descriptors)
- Memory-map the file (same pattern as SafeTensors loader)
- Load tensors as `QuantizedTensor` when quantized, `CudaTensor<f32>` when not
- Read model config from GGUF metadata (hidden_size, num_layers, etc.)

### Step B5: Model Integration

Update `infernum-llama/src/model.rs`:

- `LlamaModel` weight fields become an enum or generic over precision:
  ```rust
  enum LinearWeight {
      F32(CudaTensor<f32>),
      Quantized(QuantizedTensor),
  }
  ```
- `linear()` dispatches to `matmul` or `matmul_q8`/`matmul_q4` based on weight type
- Embedding and norm weights stay f32 (negligible memory, quantization hurts quality)
- Load from GGUF: `LlamaModel::from_gguf(ctx, path)`

### Step B6: Tests & Validation

- Unit test: Q8 matmul matches f32 matmul within tolerance (~1e-3)
- Unit test: Q4 matmul matches f32 matmul within tolerance (~1e-2)
- Unit test: GGUF loader reads known test file correctly
- Integration test: quantized model produces coherent text
- Benchmark: measure memory reduction and throughput vs f32

---

## Merge & Integration

After both tracks are complete:

1. Merge Track A into main
2. Merge Track B into main (resolve `model.rs` conflicts)
3. Combined result: quantized model with KV cache
4. Validate: GGUF model + KV cache generation produces coherent text

---

## Validation Checkpoints

1. KV cache generate produces identical output to full-recompute generate
2. KV cache gives measurable speedup (>5x for 100-token generation)
3. Q8 model generates coherent text with <2x perplexity increase
4. Q4 model generates coherent text with <3x perplexity increase
5. Q4 model fits in ~half the VRAM of f32 model

---

## What Comes Next (Phase 3)

- Continuous batching (multiple concurrent requests)
- `infernum-runtime` crate (scheduler, text in/out API)
- `infernum-serve` crate (Axum HTTP server, OpenAI-compatible API)
- PagedAttention (dynamic KV cache memory management)
