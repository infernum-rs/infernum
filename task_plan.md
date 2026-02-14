# Task: Create `infernum-runtime` Crate

## Goal
Create the `infernum-runtime` crate with model-generic `Engine` (token-level) and `Runtime` (text-level) abstractions. Adapt the generate example to use `Runtime`.

## Key Design Decisions
- **Engine** works on tokens, generic over `Model` trait
- **Runtime** works on text, generic over `Model` + `Tokenizer` traits  
- **KvCache stays in `infernum` core** — it's a CUDA data structure used by `attention_kv` op (can't move without circular deps or splitting the op)
- **Quantization stays in core** — tensor/weight concern
- **Multi-model**: each Runtime instance wraps one model; multiple can coexist
- **Model trait** defined in `infernum` core (model crates implement it)
- **Tokenizer trait** defined in `infernum` core (model crates implement it)
- **SamplingParams** moves to `infernum` core (shared across models)

## Phases

### Phase 1: Define traits in `infernum` core — `in_progress`
- Add `Model` trait to `infernum/src/model.rs`
- Add `Tokenizer` trait to `infernum/src/tokenizer/mod.rs`  
- Implement `Tokenizer` for `LlamaTokenizer`
- Move `SamplingParams` to `infernum` core
- Files: `infernum/src/model.rs`, `infernum/src/tokenizer/mod.rs`, `infernum/src/lib.rs`

### Phase 2: Create `infernum-runtime` crate scaffold — `pending`
- Create `infernum-runtime/Cargo.toml`, `src/lib.rs`
- Add to workspace
- Module structure: `engine.rs`, `runtime.rs`

### Phase 3: Implement Model trait for LlamaModel — `pending`
- Implement the `Model` trait on `LlamaModel`

### Phase 4: Implement Engine — `pending`
- `Engine<M: Model>` — generic over model, manages KvCache
- Methods: `generate`, `generate_sampled` (token-level)
- Move generation logic from `LlamaModel` into `Engine`

### Phase 5: Implement Runtime — `pending`
- `Runtime<M: Model, T: Tokenizer>` — wraps Engine + Tokenizer
- Methods: `generate(&str) -> String`

### Phase 6: Adapt generate example — `pending`
- Update to use `Runtime`

### Phase 7: Clean up LlamaModel — `pending`
- Remove generation methods from LlamaModel (now in Engine)
- Keep forward methods (Model trait impl)

### Phase 8: Verify — `pending`
- cargo build, clippy, fmt

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|