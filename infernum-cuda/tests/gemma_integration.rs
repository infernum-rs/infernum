//! Integration tests for Gemma model family (Gemma 2, Gemma 3 text).
//!
//! Run with:
//!   cargo test -p infernum-cuda --features integration -- --test-threads=1 gemma
#![cfg(feature = "integration")]

mod test_helpers;

use std::path::PathBuf;

use infernum::tokenizer::LlamaTokenizer;
use infernum_cuda::cuda::CudaContext;
use infernum_gemma::{GemmaCudaGraphEngine, GemmaCudaGraphEngineExt as _};
use infernum_runtime::Runtime;

use test_helpers::{download_model, greedy_options};

/// Load a model and generate text with greedy decoding.
fn generate_greedy(model_dir: &PathBuf, prompt: &str, max_tokens: usize) -> String {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let model =
        GemmaCudaGraphEngine::from_pretrained(ctx, model_dir).expect("Failed to load model");
    let tokenizer = LlamaTokenizer::from_pretrained(model_dir).expect("Failed to load tokenizer");

    let runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
    runtime
        .generate(prompt, &greedy_options(max_tokens))
        .expect("Generation failed")
}

// ─── Gemma 2 tiny random (random weights, architecture test) ────────────────

mod gemma2_tiny_random {
    use super::*;

    const REPO: &str = "yujiepan/gemma-2-tiny-random";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    #[test]
    fn loads_and_generates() {
        let output = generate_greedy(&model_dir(), "Hello", 10);
        assert!(
            !output.is_empty(),
            "Expected non-empty output from Gemma 2 tiny model"
        );
        println!("Gemma 2 tiny random output: {output}");
    }

    #[test]
    fn no_nan_in_output() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model_dir = model_dir();
        let model = GemmaCudaGraphEngine::from_pretrained(ctx, &model_dir)
            .expect("Failed to load model");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        let logits = model.forward(&input_ids).expect("Forward pass failed");
        let logits_vec: Vec<f32> = logits.tensor().to_vec().expect("Failed to read logits");

        let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }
}

// ─── Gemma 3 text tiny random (random weights, architecture test) ───────────

mod gemma3_text_tiny_random {
    use super::*;

    const REPO: &str = "katuni4ka/tiny-random-gemma3-text";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    #[test]
    fn loads_and_generates() {
        let output = generate_greedy(&model_dir(), "Hello", 10);
        assert!(
            !output.is_empty(),
            "Expected non-empty output from Gemma 3 text tiny model"
        );
        println!("Gemma 3 text tiny random output: {output}");
    }

    #[test]
    fn no_nan_in_output() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model_dir = model_dir();
        let model = GemmaCudaGraphEngine::from_pretrained(ctx, &model_dir)
            .expect("Failed to load model");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        let logits = model.forward(&input_ids).expect("Forward pass failed");
        let logits_vec: Vec<f32> = logits.tensor().to_vec().expect("Failed to read logits");

        let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }
}

// ─── Gemma 2 2B (ungated, ignored, quality check) ──────────────────────────

mod gemma2_2b {
    use super::*;

    const REPO: &str = "unsloth/gemma-2-2b";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    #[test]
    #[ignore = "5GB model, needs ~10GB VRAM — run manually with --ignored"]
    fn greedy_generation_quality() {
        let output = generate_greedy(&model_dir(), "The capital of France is", 30);
        assert!(
            output.contains("city"),
            "Expected continuation containing 'city', got: {output}"
        );
        let unique_words: std::collections::HashSet<&str> = output.split_whitespace().collect();
        assert!(
            unique_words.len() >= 15,
            "Output looks repetitive (only {} unique words): {output}",
            unique_words.len()
        );
    }

    #[test]
    #[ignore = "5GB model, needs ~10GB VRAM — run manually with --ignored"]
    fn no_nan_in_output() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model_dir = model_dir();
        let model = GemmaCudaGraphEngine::from_pretrained(ctx, &model_dir)
            .expect("Failed to load model");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        let logits = model.forward(&input_ids).expect("Forward pass failed");
        let logits_vec: Vec<f32> = logits.tensor().to_vec().expect("Failed to read logits");

        let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }
}

// ─── Gemma 3 1B (ungated, ignored, quality check) ──────────────────────────

mod gemma3_1b {
    use super::*;

    const REPO: &str = "unsloth/gemma-3-1b-it";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    #[test]
    #[ignore = "2GB model, needs ~4GB VRAM — run manually with --ignored"]
    fn capital_of_france() {
        let output = generate_greedy(&model_dir(), "The capital of France is", 30);
        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in output, got: {output}"
        );
    }

    #[test]
    #[ignore = "2GB model, needs ~4GB VRAM — run manually with --ignored"]
    fn no_nan_in_output() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model_dir = model_dir();
        let model = GemmaCudaGraphEngine::from_pretrained(ctx, &model_dir)
            .expect("Failed to load model");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        let logits = model.forward(&input_ids).expect("Forward pass failed");
        let logits_vec: Vec<f32> = logits.tensor().to_vec().expect("Failed to read logits");

        let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }
}

// ─── Gemma 2 tiny random SafeTensors (CUDA) ─────────────────────────────────

/// Gemma 2 tiny random (SafeTensors). Tests 4 norms/layer, GeGLU, soft-capping,
/// and alternating sliding/full attention using the CUDA graph engine.
/// (Previously tested via GGUF Q8_0; the CUDA graph engine uses SafeTensors.)
mod gemma2_tiny_safetensors {
    use super::*;

    const REPO: &str = "yujiepan/gemma-2-tiny-random";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    #[test]
    fn no_nan_in_output() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model_dir = model_dir();
        let model = GemmaCudaGraphEngine::from_pretrained(ctx, &model_dir)
            .expect("Failed to load model");

        // Gemma BOS=2, "Hello"=4521 (SentencePiece)
        let input_ids = vec![2_u32, 4521];

        let logits = model.forward(&input_ids).expect("Forward pass failed");
        let logits_vec: Vec<f32> = logits.tensor().to_vec().expect("Failed to read logits");

        let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }

    #[test]
    fn loads_and_generates() {
        let output = generate_greedy(&model_dir(), "The capital of France is", 10);
        assert!(!output.is_empty(), "Expected non-empty generation output");
    }
}
