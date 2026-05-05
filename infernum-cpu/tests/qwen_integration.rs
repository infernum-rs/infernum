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

use infernum::tokenizer::{GgufTokenizer, LlamaTokenizer};
use infernum_cpu::CpuBackend;
use infernum_qwen::{QwenGraphEngine, QwenGraphEngineExt as _, QwenModel};

use test_helpers::{download_model, download_model_files, greedy_options};

// ─── Qwen2.5-0.5B-Instruct GGUF Q8_0 ──────────────────────────────────────

/// Qwen2.5-0.5B-Instruct quantized to Q8_0 (~507MB GGUF).
/// Tests GGUF weight loading, Q/K/V biases, and generation quality.
mod qwen25_05b_q8 {
    use super::*;

    use infernum_runtime::Runtime;

    const REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
    const GGUF_FILE: &str = "qwen2.5-0.5b-instruct-q8_0.gguf";

    fn gguf_path() -> PathBuf {
        let dir = download_model_files(REPO, &[GGUF_FILE]);
        dir.join(GGUF_FILE)
    }

    #[test]
    fn no_nan_in_output() {
        let path = gguf_path();
        let model = QwenModel::<CpuBackend>::from_gguf(&(), &path).expect("Failed to load GGUF");

        // Use BOS + common token IDs (Qwen2.5 uses tiktoken-based vocab)
        // BOS=151643, "Hello"=9707
        let input_ids = vec![151643, 9707];

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
        let model = QwenModel::<CpuBackend>::from_gguf(&(), &path).expect("Failed to load GGUF");

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
