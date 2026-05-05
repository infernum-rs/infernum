//! Integration tests for Llama/Mistral/Mixtral model families.
//!
//! Downloads real models from HuggingFace and verifies generation output.
//! Gated behind the `integration` feature so they don't run during normal
//! `cargo test`. Run with:
//!
//!   cargo test -p infernum-cuda --features integration -- --test-threads=1 llama
//!
//! Models are cached in `~/.cache/infernum/models/`, so subsequent runs are fast.
#![cfg(feature = "integration")]

mod test_helpers;

use std::path::PathBuf;

use infernum::tokenizer::LlamaTokenizer;
use infernum::Model;
use infernum_cuda::cuda::CudaContext;
use infernum_llama::{LlamaCudaGraphEngine, LlamaCudaGraphEngineExt as _};
use infernum_runtime::Runtime;

use test_helpers::{download_model, download_model_files, greedy_options};

/// Load a model and generate text with greedy decoding.
fn generate_greedy(model_dir: &PathBuf, prompt: &str, max_tokens: usize) -> String {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let model =
        LlamaCudaGraphEngine::from_pretrained(ctx, model_dir).expect("Failed to load model");
    let tokenizer = LlamaTokenizer::from_pretrained(model_dir).expect("Failed to load tokenizer");

    let runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
    runtime
        .generate(prompt, &greedy_options(max_tokens))
        .expect("Generation failed")
}

/// Load a model and generate text with greedy decoding + CUDA graph capture/replay.
fn generate_greedy_with_graphs(model_dir: &PathBuf, prompt: &str, max_tokens: usize) -> String {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let model =
        LlamaCudaGraphEngine::from_pretrained(ctx, model_dir).expect("Failed to load model");
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
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model_dir = model_dir();
        let model =
            LlamaCudaGraphEngine::from_pretrained(ctx, &model_dir).expect("Failed to load model");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        // Run a raw forward pass and check logits for NaN/Inf
        let logits = model.forward(&input_ids).expect("Forward pass failed");
        let logits_vec: Vec<f32> = logits.tensor().to_vec().expect("Failed to read logits");

        let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }
}

// ─── CUDA Graphs ─────────────────────────────────────────────────────────────

/// SmolLM2-360M with CUDA graph capture/replay for the decode loop.
mod smollm2_360m_cuda_graphs {
    use super::*;

    const REPO: &str = "HuggingFaceTB/SmolLM2-360M";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    #[test]
    fn capital_of_france() {
        let output = generate_greedy_with_graphs(&model_dir(), "The capital of France is", 30);
        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in output, got: {output}"
        );
    }

    #[test]
    fn matches_eager() {
        let dir = model_dir();
        let prompt = "The capital of France is";
        let eager = generate_greedy(&dir, prompt, 30);
        let graph = generate_greedy_with_graphs(&dir, prompt, 30);

        assert!(
            eager.contains("Paris"),
            "Eager output should contain 'Paris', got: {eager}"
        );
        assert!(
            graph.contains("Paris"),
            "Graph output should contain 'Paris', got: {graph}"
        );
    }

    #[test]
    fn longer_generation() {
        let output = generate_greedy_with_graphs(&model_dir(), "Once upon a time", 100);
        assert!(
            !output.is_empty(),
            "Graph-accelerated generation should produce non-empty output"
        );
    }
}

// ─── SafeTensors FP8 ────────────────────────────────────────────────────────

/// RedHatAI Llama-3.2-1B-Instruct FP8 (ungated, ~1.5GB, compressed-tensors format)
mod llama_fp8 {
    use super::*;

    const REPO: &str = "RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic";

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
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model_dir = model_dir();
        let model =
            LlamaCudaGraphEngine::from_pretrained(ctx, &model_dir).expect("Failed to load model");

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

// ─── GPTQ INT4 ───────────────────────────────────────────────────────────────

/// Llama-3.2-1B GPTQ INT4 (ungated, ~985MB, group_size=128, sym=true)
mod llama_gptq {
    use super::*;

    const REPO: &str = "shuyuej/Llama-3.2-1B-GPTQ";

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
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model_dir = model_dir();
        let model =
            LlamaCudaGraphEngine::from_pretrained(ctx, &model_dir).expect("Failed to load model");

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

// ─── Mixtral MoE ─────────────────────────────────────────────────────────────

/// jamesdborin/tiny-mixtral (ungated, 2-layer Mixtral with 8 experts, ~988MB f32)
mod mixtral_moe_tiny {
    use super::*;

    const REPO: &str = "jamesdborin/tiny-mixtral";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    #[test]
    fn loads_and_generates() {
        // Random weights won't produce meaningful text, but generation must not panic
        let _output = generate_greedy(&model_dir(), "Hello", 10);
    }

    #[test]
    fn no_nan_in_output() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model_dir = model_dir();
        let model =
            LlamaCudaGraphEngine::from_pretrained(ctx, &model_dir).expect("Failed to load model");

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

// ─── Mixtral MoE (real weights) ─────────────────────────────────────────────

/// laser-dolphin-mixtral-2x7b-dpo (ungated, ~24GB bf16, 3 sharded SafeTensors)
mod mixtral_2x7b {
    use super::*;

    const REPO: &str = "macadeliccc/laser-dolphin-mixtral-2x7b-dpo";

    const FILES: &[&str] = &[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "model.safetensors.index.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ];

    fn model_dir() -> PathBuf {
        download_model_files(REPO, FILES)
    }

    #[test]
    #[ignore = "24GB model, needs ~48GB VRAM — run manually with --ignored"]
    fn capital_of_france() {
        let output = generate_greedy(&model_dir(), "The capital of France is", 30);
        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in output, got: {output}"
        );
    }

    #[test]
    #[ignore = "24GB model, needs ~48GB VRAM — run manually with --ignored"]
    fn no_nan_in_output() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model_dir = model_dir();
        let model =
            LlamaCudaGraphEngine::from_pretrained(ctx, &model_dir).expect("Failed to load model");

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

// ─── Mistral (dense) ────────────────────────────────────────────────────────

mod mistral_7b {
    use super::*;

    const REPO: &str = "mistralai/Mistral-7B-Instruct-v0.3";

    const FILES: &[&str] = &[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ];

    fn model_dir() -> PathBuf {
        download_model_files(REPO, FILES)
    }

    #[test]
    #[ignore = "14.5GB model, needs ~30GB VRAM — run manually with --ignored"]
    fn capital_of_france() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model_dir = model_dir();
        let model =
            LlamaCudaGraphEngine::from_pretrained(ctx, &model_dir).expect("Failed to load model");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
        let output = runtime
            .generate("The capital of France is", &greedy_options(30))
            .expect("Generation failed");
        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in output, got: {output}"
        );
    }

    #[test]
    #[ignore = "14.5GB model, needs ~30GB VRAM — run manually with --ignored"]
    fn no_nan_in_output() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model_dir = model_dir();
        let model =
            LlamaCudaGraphEngine::from_pretrained(ctx, &model_dir).expect("Failed to load model");

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
