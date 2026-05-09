//! Integration tests for Qwen model family on the Metal backend.
//!
//! Downloads real models from HuggingFace and verifies generation output.
//! Gated behind the `integration` feature so they don't run during normal
//! `cargo test`. Run with:
//!
//!   `cargo test -p infernum-metal --features integration qwen`
//!
//! Models are cached in `~/.cache/infernum/models/`, so subsequent runs are fast.
#![cfg(feature = "integration")]

mod test_helpers;

use std::path::PathBuf;

use infernum::tokenizer::LlamaTokenizer;
use infernum_metal::MetalContext;
use infernum_qwen::{QwenMetalGraphEngine, QwenMetalGraphEngineExt as _};
use infernum_runtime::Runtime;

use test_helpers::{download_model, greedy_options};

/// Load a Qwen model and generate text with greedy decoding.
fn generate_greedy(model_dir: &PathBuf, prompt: &str, max_tokens: usize) -> String {
    let ctx = MetalContext::new();
    let engine = QwenMetalGraphEngine::from_pretrained(ctx, model_dir)
        .expect("Failed to load QwenMetalGraphEngine");
    let tokenizer = LlamaTokenizer::from_pretrained(model_dir).expect("Failed to load tokenizer");

    let runtime = Runtime::new(engine, tokenizer).expect("Failed to create runtime");
    runtime
        .generate(prompt, &greedy_options(max_tokens))
        .expect("Generation failed")
}

// ─── Qwen2.5 tiny (random weights — Qwen2 architecture plumbing) ─────────────

/// Qwen2.5-tiny-random — ~4.9MB random weights, Qwen2 architecture.
/// Validates basic weight loading, Q/K/V bias, tied embeddings, and the
/// prefill→decode transition on the Metal backend. No quality assertion.
mod qwen2_5_tiny_random {
    use super::*;

    const REPO: &str = "yujiepan/qwen2.5-tiny-random";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    /// Successful 1-token generation without panic confirms no NaN/Inf in
    /// logits (argmax panics on all-NaN input; Runtime surfaces it as an error).
    #[test]
    fn no_nan_in_logits() {
        let model_dir = model_dir();
        let ctx = MetalContext::new();
        let engine = QwenMetalGraphEngine::from_pretrained(ctx, &model_dir)
            .expect("Failed to load QwenMetalGraphEngine");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let runtime = Runtime::new(engine, tokenizer).expect("Failed to create runtime");
        runtime
            .generate("Hello world", &greedy_options(1))
            .expect("Generation failed — possible NaN/Inf in logits");
    }

    /// Generate 5 tokens and verify every ID is within vocab.
    /// A successful decode means all token IDs were in-range.
    #[test]
    fn no_oob_tokens() {
        let _ = generate_greedy(&model_dir(), "Hello world", 5);
    }

    /// Generates 10 tokens through both prefill and multiple decode steps.
    /// Checks that the backend doesn't collapse into degenerate repetition.
    #[test]
    fn multi_step_generates() {
        let output = generate_greedy(&model_dir(), "The capital of France is", 10);
        // Random weights produce arbitrary text; we only require non-degeneracy.
        assert!(
            !output.contains("is is is is"),
            "Model produced degenerate repeated output: {output:?}"
        );
        println!("Qwen2.5-tiny-random output: {output}");
    }
}

// ─── Qwen3-MoE tiny (random weights — MoE routing plumbing) ──────────────────

/// Qwen3-MoE-tiny — random weights, ~5MB.
/// Validates MoE routing plumbing (`decoder_sparse_step`, top-2 experts).
/// No quality assertion since weights are random.
mod qwen3_moe_tiny {
    use super::*;

    const REPO: &str = "yujiepan/qwen3-moe-tiny-random";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    #[test]
    fn no_nan_in_logits() {
        let model_dir = model_dir();
        let ctx = MetalContext::new();
        let engine = QwenMetalGraphEngine::from_pretrained(ctx, &model_dir)
            .expect("Failed to load QwenMetalGraphEngine (Qwen3-MoE-tiny)");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let runtime = Runtime::new(engine, tokenizer).expect("Failed to create runtime");
        runtime
            .generate("Hello", &greedy_options(1))
            .expect("Generation failed — possible NaN/Inf in MoE logits");
    }

    #[test]
    fn no_oob_tokens() {
        let _ = generate_greedy(&model_dir(), "Hello", 5);
    }

    #[test]
    fn multi_step_generates() {
        let output = generate_greedy(&model_dir(), "The capital of France is", 10);
        assert!(
            !output.contains("is is is is"),
            "Model produced degenerate repeated output: {output:?}"
        );
        println!("Qwen3-MoE-tiny output: {output}");
    }
}

// ─── Qwen2.5-0.5B (real weights — generation quality) ────────────────────────

/// Qwen2.5-0.5B — real pretrained weights, ~987MB bf16.
/// Validates generation quality, Q/K/V bias, and tied embeddings.
/// Marked `#[ignore]` because the download is too large for routine CI.
/// Run manually with:
///   cargo test -p infernum-metal --features integration -- --ignored qwen2_5_0_5b
mod qwen2_5_0_5b {
    use super::*;

    const REPO: &str = "Qwen/Qwen2.5-0.5B";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    #[test]
    #[ignore = "large download (~987MB); run manually to validate generation quality"]
    fn capital_of_france() {
        let output = generate_greedy(&model_dir(), "The capital of France is", 10);
        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in output, got: {output:?}"
        );
        assert!(
            !output.contains("is is is"),
            "Model produced degenerate repeated output: {output:?}"
        );
    }
}
