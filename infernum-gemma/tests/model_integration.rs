//! Integration tests that download real models and verify generation output.
//!
//! Gated behind the `integration` feature so they don't run during normal
//! `cargo test`. Run with:
//!
//!   cargo test -p infernum-gemma --features integration -- --test-threads=1
//!
//! Models are cached in `~/.cache/infernum/models/`, so subsequent runs are fast.
#![cfg(feature = "integration")]

use std::fs;
use std::path::PathBuf;

use infernum::cuda::CudaContext;
use infernum::tokenizer::LlamaTokenizer;
use infernum::GenerateOptions;
use infernum::Model;
use infernum_gemma::GemmaModel;
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
/// Streams directly to disk to avoid buffering large files in memory.
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

    let tmp_dest = dest.with_extension("tmp");
    let mut file = fs::File::create(&tmp_dest)
        .unwrap_or_else(|e| panic!("Failed to create {}: {e}", tmp_dest.display()));
    std::io::copy(&mut response.into_body().as_reader(), &mut file)
        .unwrap_or_else(|e| panic!("Failed to download {filename}: {e}"));
    fs::rename(&tmp_dest, dest)
        .unwrap_or_else(|e| panic!("Failed to rename {}: {e}", tmp_dest.display()));
}

/// Download a model from HuggingFace Hub and return the local directory path.
///
/// Files are cached in `~/.cache/infernum/models/<org>/<model>/`.
fn download_model(repo_id: &str) -> PathBuf {
    download_model_files(repo_id, REQUIRED_FILES)
}

/// Download specific files from a HuggingFace model repo.
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
    let model = GemmaModel::<f32>::from_pretrained(&ctx, model_dir).expect("Failed to load model");
    let tokenizer = LlamaTokenizer::from_pretrained(model_dir).expect("Failed to load tokenizer");

    let mut runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
    runtime
        .generate(prompt, &greedy_options(max_tokens))
        .expect("Generation failed")
}

// ─── Gemma 2 tiny random (random weights, architecture test) ────────────────

/// yujiepan/gemma-2-tiny-random (ungated, ~2MB, 2 layers, hidden_size=8,
/// head_dim=32, sliding_window=4096, softcapping, random weights)
///
/// Tests Gemma 2 architecture: 4 norms/layer, GeGLU, embedding scaling,
/// query_pre_attn_scalar, attention logit softcapping, final logit softcapping,
/// alternating sliding/full attention. Random weights — no quality check.
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
        let model =
            GemmaModel::<f32>::from_pretrained(&ctx, &model_dir).expect("Failed to load model");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        let logits = model.forward(&input_ids).expect("Forward pass failed");
        let logits_vec: Vec<f32> = logits.to_vec().expect("Failed to read logits");

        let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }
}

// ─── Gemma 3 text tiny random (random weights, architecture test) ───────────

/// katuni4ka/tiny-random-gemma3-text (ungated, ~2MB, 2 layers, hidden_size=8,
/// head_dim=16, sliding_window=512, sliding_window_pattern=6, QK-norm,
/// dual-theta RoPE, random weights)
///
/// Tests Gemma 3 text architecture: QK-norm, dual RoPE theta (local + global),
/// no softcapping, sliding_window_pattern auto-generation. Random weights —
/// no quality check.
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
        let model =
            GemmaModel::<f32>::from_pretrained(&ctx, &model_dir).expect("Failed to load model");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        let logits = model.forward(&input_ids).expect("Forward pass failed");
        let logits_vec: Vec<f32> = logits.to_vec().expect("Failed to read logits");

        let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }
}

// ─── Gemma 2 2B (gated, ignored, quality check) ────────────────────────────

/// google/gemma-2-2b (gated, ~5GB bf16) — validates Gemma 2 generation quality
/// with real weights. Requires HF auth and ~10GB VRAM.
mod gemma2_2b {
    use super::*;

    const REPO: &str = "google/gemma-2-2b";

    fn model_dir() -> PathBuf {
        download_model_files(
            REPO,
            &[
                "config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )
    }

    #[test]
    #[ignore = "Gated model, requires HF auth and ~10GB VRAM"]
    fn capital_of_france() {
        let output = generate_greedy(&model_dir(), "The capital of France is", 30);
        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in output, got: {output}"
        );
    }
}

// ─── Gemma 3 1B (gated, ignored, quality check) ────────────────────────────

/// google/gemma-3-1b-it (gated, ~2GB bf16) — validates Gemma 3 generation
/// quality with real weights. Requires HF auth and ~4GB VRAM.
mod gemma3_1b {
    use super::*;

    const REPO: &str = "google/gemma-3-1b-it";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    #[test]
    #[ignore = "Gated model, requires HF auth and ~4GB VRAM"]
    fn capital_of_france() {
        let output = generate_greedy(&model_dir(), "The capital of France is", 30);
        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in output, got: {output}"
        );
    }
}
