//! Integration tests for Llama model family on the CPU backend.
//!
//! Downloads real models from HuggingFace and verifies generation output.
//! Gated behind the `integration` feature so they don't run during normal
//! `cargo test`. Run with:
//!
//!   cargo test -p infernum-cpu --features integration llama
//!
//! Models are cached in `~/.cache/infernum/models/`, so subsequent runs are fast.
#![cfg(feature = "integration")]

mod test_helpers;

use std::path::PathBuf;

use infernum::tokenizer::LlamaTokenizer;
use infernum_cpu::CpuBackend;
use infernum_llama::LlamaModel;
use infernum_runtime::Runtime;

use test_helpers::{download_model, download_model_files, greedy_options};

/// Load a model and generate text with greedy decoding.
fn generate_greedy(model_dir: &PathBuf, prompt: &str, max_tokens: usize) -> String {
    let model =
        LlamaModel::<CpuBackend>::from_pretrained(&(), model_dir).expect("Failed to load model");
    let tokenizer = LlamaTokenizer::from_pretrained(model_dir).expect("Failed to load tokenizer");

    let runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
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
        let output = generate_greedy(&model_dir(), "The capital of France is", 30);
        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in output, got: {output}"
        );
    }

    #[test]
    fn no_nan_in_output() {
        let model_dir = model_dir();
        let model = LlamaModel::<CpuBackend>::from_pretrained(&(), &model_dir)
            .expect("Failed to load model");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        // Run a raw forward pass and check logits for NaN/Inf
        let hidden = model.forward_full(&input_ids).expect("Forward pass failed");
        let logits_vec = hidden.to_f32_vec();

        let nan_count = logits_vec.iter().filter(|x: &&f32| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x: &&f32| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }
}

// ─── SmolLM2-360M GGUF Q8_0 ─────────────────────────────────────────────────

/// SmolLM2-360M quantized to Q8_0 (~380MB GGUF).
/// Tests that GGUF quantized weight loading works on CPU and produces
/// finite logits. Uses raw token IDs (SmolLM2's GPT2 BPE tokenizer
/// is not supported by the GGUF tokenizer).
mod smollm2_360m_q8 {
    use super::*;

    const REPO: &str = "bartowski/SmolLM2-360M-Instruct-GGUF";
    const GGUF_FILE: &str = "SmolLM2-360M-Instruct-Q8_0.gguf";

    fn gguf_path() -> PathBuf {
        let dir = download_model_files(REPO, &[GGUF_FILE]);
        dir.join(GGUF_FILE)
    }

    #[test]
    fn no_nan_in_output() {
        let path = gguf_path();
        let model = LlamaModel::<CpuBackend>::from_gguf(&(), &path).expect("Failed to load GGUF");

        // Use raw token IDs: "Hello" in GPT2 BPE (SmolLM2 vocab)
        // BOS=0, "Hello"=15339 (common GPT2 token)
        let input_ids = vec![0, 15339];

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
        // Only download tokenizer files (not the full SafeTensors model)
        let tokenizer_dir = download_model_files(
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            &["tokenizer.json", "tokenizer_config.json"],
        );
        let model = LlamaModel::<CpuBackend>::from_gguf(&(), &path).expect("Failed to load GGUF");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&tokenizer_dir).expect("Failed to load tokenizer");

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

// ─── SmolLM2-360M GGUF Q4_0 (mixed Q4_0/Q4_1) ──────────────────────────────

/// SmolLM2-360M quantized to Q4_0 (~200MB GGUF).
/// This GGUF contains a mix of Q4_0 and Q4_1 tensors (llama.cpp uses Q4_1
/// for certain layers). Tests that both quant formats load and produce
/// finite logits and correct generation.
mod smollm2_360m_q4 {
    use super::*;

    const REPO: &str = "bartowski/SmolLM2-360M-Instruct-GGUF";
    const GGUF_FILE: &str = "SmolLM2-360M-Instruct-Q4_0.gguf";

    fn gguf_path() -> PathBuf {
        let dir = download_model_files(REPO, &[GGUF_FILE]);
        dir.join(GGUF_FILE)
    }

    #[test]
    fn no_nan_in_output() {
        let path = gguf_path();
        let model = LlamaModel::<CpuBackend>::from_gguf(&(), &path).expect("Failed to load GGUF");

        // BOS=0, "Hello"=15339 (GPT2 BPE token in SmolLM2 vocab)
        let input_ids = vec![0, 15339];

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
        let tokenizer_dir = download_model_files(
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            &["tokenizer.json", "tokenizer_config.json"],
        );
        let model = LlamaModel::<CpuBackend>::from_gguf(&(), &path).expect("Failed to load GGUF");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&tokenizer_dir).expect("Failed to load tokenizer");

        let runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
        let output = runtime
            .generate("The capital of France is", &greedy_options(30))
            .expect("Generation failed");
        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in Q4_0 output, got: {output}"
        );
    }
}
