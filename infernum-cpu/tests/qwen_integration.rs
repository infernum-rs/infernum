//! Integration tests for Qwen model family on the CPU backend.
//!
//! Downloads real models from HuggingFace and verifies generation output.
//! Gated behind the `integration` feature so they don't run during normal
//! `cargo test`. Run with:
//!
//!   cargo test -p infernum-cpu --features integration qwen
//!
//! Models are cached in `~/.cache/infernum/models/`, so subsequent runs are fast.
#![cfg(feature = "integration")]

mod test_helpers;

use std::path::PathBuf;

use infernum::tokenizer::LlamaTokenizer;
use infernum_qwen::{QwenGraphEngine, QwenGraphEngineExt as _};

use test_helpers::download_model;

// ─── Qwen2.5-0.5B via QwenGraphEngine ────────────────────────────────────────

/// End-to-end generation test using `QwenGraphEngine` on real Qwen2.5-0.5B
/// SafeTensors weights (~987MB bf16). Verifies that the graph-mode prefill +
/// autoregressive decode loop produces correct output and in-vocab tokens.
mod qwen25_05b_graph_engine {
    use super::*;

    const REPO: &str = "Qwen/Qwen2.5-0.5B";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    /// Graph-mode generation produces "Paris" for the standard prompt.
    #[test]
    fn capital_of_france() {
        let model_dir = model_dir();
        let engine =
            QwenGraphEngine::from_pretrained(&model_dir).expect("Failed to load QwenGraphEngine");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let prompt_ids = tokenizer
            .encode("The capital of France is", true)
            .expect("tokenize");
        let eos = engine.config().eos_token_id;

        let output_ids = engine
            .generate(&prompt_ids, 30, eos)
            .expect("graph generation failed");

        let new_ids = &output_ids[prompt_ids.len()..];
        let output = tokenizer.decode(new_ids).expect("decode");

        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in QwenGraphEngine output, got: {output}"
        );
    }

    /// Prefill + one decode step produces tokens within the vocab range.
    #[test]
    fn no_oob_tokens() {
        let model_dir = model_dir();
        let engine =
            QwenGraphEngine::from_pretrained(&model_dir).expect("Failed to load QwenGraphEngine");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let prompt_ids = tokenizer.encode("Hello world", true).expect("tokenize");
        let eos = engine.config().eos_token_id;
        let vocab_size = engine.config().vocab_size;

        let output_ids = engine
            .generate(&prompt_ids, 1, eos)
            .expect("graph generation failed");

        assert!(
            output_ids.len() >= prompt_ids.len(),
            "Output shorter than prompt: {} < {}",
            output_ids.len(),
            prompt_ids.len()
        );

        for &tok in &output_ids {
            assert!(
                (tok as usize) < vocab_size,
                "Generated token {tok} is out of vocab range {vocab_size}"
            );
        }
    }
}
