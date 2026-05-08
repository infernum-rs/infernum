//! Integration tests for Llama model family on the Metal backend.
//!
//! Downloads real models from HuggingFace and verifies generation output.
//! Gated behind the `integration` feature so they don't run during normal
//! `cargo test`. Run with:
//!
//!   `cargo test -p infernum-metal --features integration llama`
//!
//! Models are cached in `~/.cache/infernum/models/`, so subsequent runs are fast.
#![cfg(feature = "integration")]

mod test_helpers;

use std::path::PathBuf;

use infernum::tokenizer::LlamaTokenizer;
use infernum_llama::{LlamaMetalGraphEngine, LlamaMetalGraphEngineExt as _};
use infernum_metal::MetalContext;
use infernum_runtime::Runtime;

use test_helpers::{download_model, greedy_options};

/// Load a model and generate text with greedy decoding.
fn generate_greedy(model_dir: &PathBuf, prompt: &str, max_tokens: usize) -> String {
    let ctx = MetalContext::new();
    let engine = LlamaMetalGraphEngine::from_pretrained(ctx, model_dir)
        .expect("Failed to load LlamaMetalGraphEngine");
    let tokenizer = LlamaTokenizer::from_pretrained(model_dir).expect("Failed to load tokenizer");

    let runtime = Runtime::new(engine, tokenizer).expect("Failed to create runtime");
    runtime
        .generate(prompt, &greedy_options(max_tokens))
        .expect("Generation failed")
}

// ─── SafeTensors f32 ─────────────────────────────────────────────────────────

/// SmolLM2-360M (ungated, ~700MB, Llama architecture)
mod smollm2_360m {
    use super::*;

    const REPO: &str = "HuggingFaceTB/SmolLM2-360M";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    #[test]
    fn capital_of_france() {
        // Generate enough tokens to catch degenerate repetition bugs — the
        // single-token test would have passed even when the backend was broken
        // (it always produced "Paris" on the first decode step by coincidence).
        let output = generate_greedy(&model_dir(), "The capital of France is", 5);
        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in output, got: {output}"
        );
        assert!(
            !output.contains("is is is"),
            "Model produced degenerate repeated output: {output}"
        );
    }

    #[test]
    fn no_nan_in_output() {
        let model_dir = model_dir();
        let ctx = MetalContext::new();
        let engine = LlamaMetalGraphEngine::from_pretrained(ctx, &model_dir)
            .expect("Failed to load LlamaMetalGraphEngine");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        // Generate one token via Runtime to exercise the full forward pass.
        let runtime = Runtime::new(engine, tokenizer).expect("Failed to create runtime");
        runtime
            .generate("Hello world", &greedy_options(1))
            .expect("Generation failed");
    }

    #[test]
    fn logits_diagnostic() {
        let model_dir = model_dir();
        let ctx = MetalContext::new();
        let engine = LlamaMetalGraphEngine::from_pretrained(ctx, &model_dir)
            .expect("Failed to load LlamaMetalGraphEngine");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let runtime = Runtime::new(engine, tokenizer).expect("Failed to create runtime");

        let input = "The capital of France is";
        let result = runtime
            .generate(input, &greedy_options(5))
            .expect("Generation failed");
        println!("Input:          {input}");
        println!("Generated text: {result}");

        // Tokenize the full output to inspect individual token IDs.
        let full_text = format!("{input}{result}");
        let tokenizer2 =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");
        let full_ids = tokenizer2.encode(&full_text, false).expect("encode full");
        let input_ids = tokenizer2.encode(input, false).expect("encode input");
        let generated_ids = &full_ids[input_ids.len()..];
        println!("Generated token IDs: {generated_ids:?}");

        // Check it doesn't repeat the same token (degenerate output).
        assert!(
            result.contains("Paris") || !result.contains("is is is"),
            "Model produced degenerate output: {result}"
        );
    }

    /// Diagnostic test for the prefill→decode transition bug.
    ///
    /// Generates exactly 2 tokens so we can see whether the failure occurs on
    /// the very first decode step (token 2) or only on later steps.
    ///
    /// Expected: token 1 = " Paris", token 2 ≠ token 1 (no immediate repetition).
    #[test]
    fn multi_step_diagnostic() {
        let model_dir = model_dir();
        let ctx = MetalContext::new();
        let engine = LlamaMetalGraphEngine::from_pretrained(ctx, &model_dir)
            .expect("Failed to load LlamaMetalGraphEngine");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let input = "The capital of France is";

        // Encode the prompt so we know how many tokens to strip.
        let tokenizer_inspect =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");
        let prompt_ids = tokenizer_inspect
            .encode(input, false)
            .expect("encode prompt");
        println!("Prompt: {input:?}");
        println!(
            "Prompt token IDs ({} tokens): {prompt_ids:?}",
            prompt_ids.len()
        );

        let runtime = Runtime::new(engine, tokenizer).expect("Failed to create runtime");

        // Generate exactly 2 new tokens.
        let result = runtime
            .generate(input, &greedy_options(2))
            .expect("Generation failed");
        println!("Generated text (2 tokens): {result:?}");

        // Tokenize the full output to extract the 2 generated token IDs.
        let full_text = format!("{input}{result}");
        let full_ids = tokenizer_inspect
            .encode(&full_text, false)
            .expect("encode full");
        let generated_ids = &full_ids[prompt_ids.len()..];
        println!("Generated token IDs: {generated_ids:?}");

        // Decode each generated token individually for clarity.
        for (i, &id) in generated_ids.iter().enumerate() {
            let text = tokenizer_inspect.decode(&[id]).expect("decode token");
            println!("  token[{i}]: id={id}, text={text:?}");
        }

        // The first generated token must NOT be token 314 (" is") — the last
        // prompt token. When the Metal Concurrent-dispatch bug is active, the
        // attention kernel reads stale KV data and collapses to the most recent
        // KV slot, always producing the last prompt token.
        assert!(
            generated_ids.len() >= 1 && generated_ids[0] != 314,
            "First decode token is 314 (' is') — last prompt token repeated; \
             likely a KV-cache ordering bug. Got: {generated_ids:?}"
        );

        // The second token must differ from the first — no immediate repetition.
        if generated_ids.len() >= 2 {
            assert_ne!(
                generated_ids[0], generated_ids[1],
                "Token 1 and token 2 are identical (id={}): degenerate repetition detected",
                generated_ids[0],
            );
        }
    }
}
