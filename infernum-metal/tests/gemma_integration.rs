//! Integration tests for Gemma model family on the Metal backend.
//!
//! Downloads real models from HuggingFace and verifies generation output.
//! Gated behind the `integration` feature so they don't run during normal
//! `cargo test`. Run with:
//!
//!   `cargo test -p infernum-metal --features integration gemma`
//!
//! Models are cached in `~/.cache/infernum/models/`, so subsequent runs are fast.
#![cfg(feature = "integration")]

mod test_helpers;

use std::path::PathBuf;

use infernum::tokenizer::LlamaTokenizer;
use infernum_gemma::{GemmaMetalGraphEngine, GemmaMetalGraphEngineExt as _};
use infernum_metal::MetalContext;
use infernum_runtime::Runtime;

use test_helpers::{download_model, greedy_options};

/// Load a Gemma model and generate text with greedy decoding.
fn generate_greedy(model_dir: &PathBuf, prompt: &str, max_tokens: usize) -> String {
    let ctx = MetalContext::new();
    let engine = GemmaMetalGraphEngine::from_pretrained(ctx, model_dir)
        .expect("Failed to load GemmaMetalGraphEngine");
    let tokenizer = LlamaTokenizer::from_pretrained(model_dir).expect("Failed to load tokenizer");

    let runtime = Runtime::new(engine, tokenizer).expect("Failed to create runtime");
    runtime
        .generate(prompt, &greedy_options(max_tokens))
        .expect("Generation failed")
}

// ─── Gemma 2 tiny random (random weights — architecture plumbing test) ────────

/// Gemma 2 tiny random — ~2MB f32, random weights.
/// Tests 4 norms/layer, GeGLU activation, embedding scaling, attention/final
/// logit soft-capping, and alternating sliding/full attention.
/// No quality assertion since weights are random.
mod gemma2_tiny_random {
    use super::*;

    const REPO: &str = "yujiepan/gemma-2-tiny-random";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    /// Successful 1-token generation without panic confirms no NaN/Inf in
    /// logits after all four norms, soft-capping, and sliding-window attention.
    #[test]
    fn no_nan_in_logits() {
        let model_dir = model_dir();
        let ctx = MetalContext::new();
        let engine = GemmaMetalGraphEngine::from_pretrained(ctx, &model_dir)
            .expect("Failed to load GemmaMetalGraphEngine (Gemma 2 tiny)");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let runtime = Runtime::new(engine, tokenizer).expect("Failed to create runtime");
        runtime
            .generate("Hello world", &greedy_options(1))
            .expect("Generation failed — possible NaN/Inf in Gemma 2 logits");
    }

    /// Multi-token generation exercises the full prefill→decode transition
    /// through alternating sliding-window and full-attention layers.
    #[test]
    fn no_oob_tokens() {
        let output = generate_greedy(&model_dir(), "Hello world", 5);
        // Successful decode of 5 tokens confirms all IDs are in-vocab.
        let _ = output;
    }

    /// Generates 10 tokens and checks the output is non-empty and valid UTF-8.
    /// No quality check — random weights produce arbitrary text.
    #[test]
    fn loads_and_generates() {
        let output = generate_greedy(&model_dir(), "The capital of France is", 10);
        assert!(
            !output.is_empty(),
            "Expected non-empty output from Gemma 2 tiny random model"
        );
        println!("Gemma 2 tiny random output: {output}");
    }
}

// ─── Gemma 3 text tiny random (random weights — architecture plumbing test) ──

/// Gemma 3 text tiny random — ~2MB bf16, random weights.
/// Tests QK-norm (RMSNorm on Q/K per-head before RoPE), dual-theta RoPE,
/// sliding_window_pattern auto-generation, and absence of soft-capping.
/// No quality assertion since weights are random.
mod gemma3_text_tiny_random {
    use super::*;

    const REPO: &str = "katuni4ka/tiny-random-gemma3-text";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    /// Successful 1-token generation without panic confirms no NaN/Inf in
    /// logits after QK-norm and dual-theta RoPE on random weights.
    #[test]
    fn no_nan_in_logits() {
        let model_dir = model_dir();
        let ctx = MetalContext::new();
        let engine = GemmaMetalGraphEngine::from_pretrained(ctx, &model_dir)
            .expect("Failed to load GemmaMetalGraphEngine (Gemma 3 text tiny)");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let runtime = Runtime::new(engine, tokenizer).expect("Failed to create runtime");
        runtime
            .generate("Hello world", &greedy_options(1))
            .expect("Generation failed — possible NaN/Inf in Gemma 3 logits");
    }

    /// Multi-token generation exercises decode through QK-norm and dual-theta RoPE.
    #[test]
    fn no_oob_tokens() {
        let output = generate_greedy(&model_dir(), "Hello world", 5);
        // Successful decode of 5 tokens confirms all IDs are in-vocab.
        let _ = output;
    }

    /// Generates 10 tokens and checks the output is non-empty and valid UTF-8.
    /// No quality check — random weights produce arbitrary text.
    #[test]
    fn loads_and_generates() {
        let output = generate_greedy(&model_dir(), "The capital of France is", 10);
        assert!(
            !output.is_empty(),
            "Expected non-empty output from Gemma 3 text tiny random model"
        );
        println!("Gemma 3 text tiny random output: {output}");
    }
}
