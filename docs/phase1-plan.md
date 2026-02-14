# Phase 1: Single GPU Llama Inference

## Target

`cargo run --example generate -- "Hello"` produces coherent text using Llama 3.2 1B on CUDA.

---

## Steps

### Step 1: Tensor Foundation
- Define `Tensor` trait (shape, dtype, basic operations)
- Implement `CudaTensor` using cudarc (Rust CUDA bindings)
- GPU memory allocation/deallocation
- Host ↔ Device transfers

### Step 2: Core Ops
- `MatMul` via cuBLAS
- `RMSNorm`
- `RoPE` (rotary positional embeddings)
- `SiLU` activation
- `Softmax`
- `Attention` (naive implementation, no KV cache)

### Step 3: Weight Loading
- SafeTensors parser
- Memory-mapped loading (don't buffer entire model in RAM)
- Map HuggingFace weight names → Infernum layer names

### Step 4: Llama Model
- Parse HuggingFace `config.json` for model hyperparameters
- Implement Llama architecture:
  - `LlamaAttention` block
  - `LlamaMLP` block (SwiGLU)
  - `LlamaDecoderLayer`
  - `LlamaModel` (embed → layers → norm → lm_head)
- Validate outputs against HuggingFace transformers

### Step 5: Tokenizer
- Integrate `tokenizers` crate (HuggingFace's Rust tokenizer)
- Load Llama tokenizer from HuggingFace repo
- Encode/decode text ↔ tokens

### Step 6: Generation Loop
- Greedy decoding (argmax)
- Recompute full sequence each step (no KV cache)
- Stop on EOS token
- CLI binary that ties it all together

---

## Key Dependencies

- `cudarc` - Rust CUDA bindings
- `safetensors` - weight loading
- `tokenizers` - HuggingFace tokenizers
- `serde`/`serde_json` - config parsing

## Validation Checkpoints

1. CudaTensor can round-trip data to GPU
2. MatMul matches cuBLAS reference
3. Single layer output matches HuggingFace
4. Full model logits match HuggingFace (within fp tolerance)
5. Generated text is coherent

---

## What Comes Next (Phase 2)

- KV cache for efficient generation
- Continuous batching
- Runtime and HTTP server
