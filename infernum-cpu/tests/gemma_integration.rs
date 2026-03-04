//! Integration tests for Gemma model family on the CPU backend (GGUF).
//!
//! Downloads real GGUF models from HuggingFace and verifies generation output.
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
use infernum_cpu::CpuBackend;
use infernum_gemma::GemmaModel;

use test_helpers::{download_model_files, greedy_options};

// ─── Gemma 2 2B-it GGUF Q8_0 ──────────────────────────────────────────────

/// Gemma 2 2B-it quantized to Q8_0 (~2.8GB GGUF, ungated).
/// Tests GGUF weight loading, 4 norms per layer with +1.0 adjustment,
/// soft-capping, alternating sliding/full attention, and generation quality.
mod gemma2_2b_q8 {
    use super::*;

    use infernum::Tokenizer as _;
    use infernum_runtime::Runtime;

    const REPO: &str = "bartowski/gemma-2-2b-it-GGUF";
    const GGUF_FILE: &str = "gemma-2-2b-it-Q8_0.gguf";

    fn gguf_path() -> PathBuf {
        let dir = download_model_files(REPO, &[GGUF_FILE]);
        dir.join(GGUF_FILE)
    }

    #[test]
    fn no_nan_in_output() {
        let path = gguf_path();
        let model = GemmaModel::<CpuBackend>::from_gguf(&(), &path).expect("Failed to load GGUF");

        // Gemma BOS=2, "Hello"=4521 (SentencePiece)
        let input_ids = vec![2, 4521];

        let hidden = model.forward_full(&input_ids).expect("Forward pass failed");
        let logits_vec = hidden.to_f32_vec();

        let nan_count = logits_vec.iter().filter(|x: &&f32| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x: &&f32| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }

    #[test]
    fn capital_of_france() {
        let path = gguf_path();
        let model = GemmaModel::<CpuBackend>::from_gguf(&(), &path).expect("Failed to load GGUF");

        let loader =
            infernum::weights::gguf::GgufLoader::from_file(&path).expect("Failed to parse GGUF");
        let tokenizer = GgufTokenizer::from_gguf_metadata(loader.metadata())
            .expect("Failed to load GGUF tokenizer");

        let runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
        let output = runtime
            .generate("The capital of France is", &greedy_options(30))
            .expect("Generation failed");
        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in Q8_0 output, got: {output}"
        );
    }
}
