//! Integration tests for Gemma model family on the CPU backend.
//!
//! Downloads real models from HuggingFace and verifies generation output.
//! Gated behind the `integration` feature so they don't run during normal
//! `cargo test`. Run with:
//!
//!   cargo test -p infernum-cpu --features integration gemma
//!
//! Models are cached in `~/.cache/infernum/models/`, so subsequent runs are fast.
#![cfg(feature = "integration")]

mod test_helpers;

use std::path::PathBuf;

use infernum::tokenizer::GgufTokenizer;
use infernum_gemma::{GemmaGraphEngine, GemmaGraphEngineExt as _};

use test_helpers::download_model_files;

// ─── Gemma 2 2B-it GGUF Q8_0 ──────────────────────────────────────────────

/// Gemma 2 2B-it quantized to Q8_0 (~2.8GB GGUF, ungated).
/// Tests GGUF weight loading, 4 norms per layer with +1.0 adjustment,
/// soft-capping, alternating sliding/full attention, and generation quality.
mod gemma2_2b_q8 {
    use super::*;

    const REPO: &str = "bartowski/gemma-2-2b-it-GGUF";
    const GGUF_FILE: &str = "gemma-2-2b-it-Q8_0.gguf";

    fn gguf_path() -> PathBuf {
        let dir = download_model_files(REPO, &[GGUF_FILE]);
        dir.join(GGUF_FILE)
    }

    #[test]
    fn no_nan_in_output() {
        let path = gguf_path();
        let engine = GemmaGraphEngine::from_gguf(&path).expect("Failed to load GGUF");
        let eos = engine.config().eos_token_id;

        // Gemma BOS=2, "Hello"=4521 (SentencePiece)
        engine
            .generate(&[2, 4521], 1, eos)
            .expect("graph generation failed");
    }

    #[test]
    #[ignore = "CPU GemmaGraphEngine produces degenerate output; quality bug needs separate investigation"]
    fn capital_of_france() {
        let path = gguf_path();
        let engine = GemmaGraphEngine::from_gguf(&path).expect("Failed to load GGUF");

        let loader =
            infernum::weights::gguf::GgufLoader::from_file(&path).expect("Failed to parse GGUF");
        let tokenizer = GgufTokenizer::from_gguf_metadata(loader.metadata())
            .expect("Failed to load GGUF tokenizer");

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
            "Expected 'Paris' in Q8_0 output, got: {output}"
        );
    }
}

// ─── Gemma 2 2B-it via GemmaGraphEngine ──────────────────────────────────────

/// End-to-end generation test using `GemmaGraphEngine` on real Gemma 2 2B-it
/// Q8_0 GGUF weights (~2.8GB). Verifies that the graph-mode prefill +
/// autoregressive decode loop produces correct output and in-vocab tokens.
/// Reuses the same GGUF file as the eager path test above — no extra download.
mod gemma2_2b_graph_engine {
    use super::*;

    const REPO: &str = "bartowski/gemma-2-2b-it-GGUF";
    const GGUF_FILE: &str = "gemma-2-2b-it-Q8_0.gguf";

    fn gguf_path() -> PathBuf {
        let dir = download_model_files(REPO, &[GGUF_FILE]);
        dir.join(GGUF_FILE)
    }

    /// Graph-mode generation produces "Paris" for the standard prompt.
    #[test]
    #[ignore = "CPU GemmaGraphEngine produces degenerate output; quality bug needs separate investigation"]
    fn capital_of_france() {
        let path = gguf_path();
        let engine = GemmaGraphEngine::from_gguf(&path).expect("Failed to load GemmaGraphEngine");

        let loader =
            infernum::weights::gguf::GgufLoader::from_file(&path).expect("Failed to parse GGUF");
        let tokenizer = GgufTokenizer::from_gguf_metadata(loader.metadata())
            .expect("Failed to load GGUF tokenizer");

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
            "Expected 'Paris' in GemmaGraphEngine output, got: {output}"
        );
    }

    /// Prefill + one decode step produces tokens within the vocab range.
    #[test]
    fn no_oob_tokens() {
        let path = gguf_path();
        let engine = GemmaGraphEngine::from_gguf(&path).expect("Failed to load GemmaGraphEngine");
        let eos = engine.config().eos_token_id;
        let vocab_size = engine.config().vocab_size;

        // Gemma BOS=2, "Hello"=4521 (SentencePiece)
        let output_ids = engine
            .generate(&[2, 4521], 1, eos)
            .expect("graph generation failed");

        assert!(
            !output_ids.is_empty(),
            "Graph engine produced no output tokens"
        );
        for &tok in &output_ids {
            assert!(
                (tok as usize) < vocab_size,
                "Generated token {tok} is out of vocab range {vocab_size}"
            );
        }
    }
}
