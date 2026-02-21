//! Integration tests that download real models and verify generation output.
//!
//! Gated behind the `integration` feature so they don't run during normal
//! `cargo test`. Run with:
//!
//!   cargo test -p infernum-llama --features integration -- --test-threads=1
//!
//! Models are cached in `~/.cache/infernum/models/`, so subsequent runs are fast.
#![cfg(feature = "integration")]

use std::fs;
use std::path::PathBuf;

use infernum::cuda::CudaContext;
use infernum::tokenizer::LlamaTokenizer;
use infernum::GenerateOptions;
use infernum_llama::LlamaModel;
use infernum_runtime::Runtime;

/// Files needed from a HuggingFace repo to load a SafeTensors model.
const REQUIRED_FILES: &[&str] = &[
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
];

/// Download a single file from HuggingFace Hub to `dest`.
///
/// Skips the download if `dest` already exists.
fn download_file(repo_id: &str, filename: &str, dest: &PathBuf) {
    if dest.exists() {
        return;
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).expect("Failed to create cache directory");
    }

    let url = format!("https://huggingface.co/{repo_id}/resolve/main/{filename}");
    let response = ureq::get(&url).call().unwrap_or_else(|e| {
        panic!("Failed to download {repo_id}/{filename}: {e}");
    });

    let body = response
        .into_body()
        .with_config()
        .limit(5 * 1024 * 1024 * 1024) // 5 GB
        .read_to_vec()
        .unwrap_or_else(|e| panic!("Failed to read response body for {filename}: {e}"));

    fs::write(dest, &body).unwrap_or_else(|e| panic!("Failed to write {}: {e}", dest.display()));
}

/// Download a model from HuggingFace Hub and return the local directory path.
///
/// Files are cached in `~/.cache/infernum/models/<org>/<model>/`.
fn download_model(repo_id: &str) -> PathBuf {
    download_model_files(repo_id, REQUIRED_FILES)
}

/// Download specific files from a HuggingFace model repo.
///
/// Use this instead of [`download_model`] for sharded models or repos
/// with non-standard file layouts.
fn download_model_files(repo_id: &str, files: &[&str]) -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let cache_dir = PathBuf::from(home)
        .join(".cache")
        .join("infernum")
        .join("models")
        .join(repo_id);

    for filename in files {
        download_file(repo_id, filename, &cache_dir.join(filename));
    }

    cache_dir
}

/// Greedy generation options (deterministic, no sampling).
fn greedy_options(max_tokens: usize) -> GenerateOptions {
    GenerateOptions {
        max_new_tokens: max_tokens,
        sampling: None,
        use_kv_cache: true,
        ..GenerateOptions::default()
    }
}

/// Load a model and generate text with greedy decoding.
fn generate_greedy(model_dir: &PathBuf, prompt: &str, max_tokens: usize) -> String {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let model = LlamaModel::<f32>::from_pretrained(&ctx, model_dir).expect("Failed to load model");
    let tokenizer = LlamaTokenizer::from_pretrained(model_dir).expect("Failed to load tokenizer");

    let mut runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
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
            LlamaModel::<f32>::from_pretrained(&ctx, &model_dir).expect("Failed to load model");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        // Run a raw forward pass and check logits for NaN/Inf
        let logits = model.forward(&input_ids).expect("Forward pass failed");
        let logits_vec = logits.to_vec().expect("Failed to read logits");

        let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
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
            LlamaModel::<f32>::from_pretrained(&ctx, &model_dir).expect("Failed to load model");

        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");
        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        let logits = model.forward(&input_ids).expect("Forward pass failed");
        let logits_vec = logits.to_vec().expect("Failed to read logits");

        let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }
}

// ─── GGUF ────────────────────────────────────────────────────────────────────

// TODO: Add GGUF integration test once we pick an ungated GGUF model.
// Will test LlamaModel::from_gguf() and GgufTokenizer.

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
            LlamaModel::<f32>::from_pretrained(&ctx, &model_dir).expect("Failed to load model");

        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");
        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        let logits = model.forward(&input_ids).expect("Forward pass failed");
        let logits_vec = logits.to_vec().expect("Failed to read logits");

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
            LlamaModel::<f32>::from_pretrained(&ctx, &model_dir).expect("Failed to load model");

        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");
        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        let logits = model.forward(&input_ids).expect("Forward pass failed");
        let logits_vec = logits.to_vec().expect("Failed to read logits");

        let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }
}

// ─── Mixtral MoE (real weights) ─────────────────────────────────────────────

/// laser-dolphin-mixtral-2x7b-dpo (ungated, ~24GB bf16, 3 sharded SafeTensors)
///
/// Real Mixtral-architecture MoE model with 2 experts (top-2), 32 layers.
/// Requires ~48GB VRAM (loaded as f32) — fits on a single A100 80GB or 2+ GPUs.
/// Run manually with:
///   cargo test -p infernum-llama --features integration -- --ignored --test-threads=1 mixtral_2x7b
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
            LlamaModel::<f32>::from_pretrained(&ctx, &model_dir).expect("Failed to load model");

        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");
        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        let logits = model.forward(&input_ids).expect("Forward pass failed");
        let logits_vec = logits.to_vec().expect("Failed to read logits");

        let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }
}
