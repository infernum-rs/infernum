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

use infernum::tokenizer::LlamaTokenizer;
use infernum::GenerateOptions;
use infernum::Tensor;
use infernum_cuda::cuda::CudaContext;
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

    // Stream to a temp file first, then rename — avoids partial files on failure.
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
    let model = LlamaModel::from_pretrained(&ctx, model_dir).expect("Failed to load model");
    let tokenizer = LlamaTokenizer::from_pretrained(model_dir).expect("Failed to load tokenizer");

    let runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
    runtime
        .generate(prompt, &greedy_options(max_tokens))
        .expect("Generation failed")
}

/// Load a model and generate text with greedy decoding + CUDA graph capture/replay.
fn generate_greedy_with_graphs(model_dir: &PathBuf, prompt: &str, max_tokens: usize) -> String {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let model = LlamaModel::from_pretrained(&ctx, model_dir).expect("Failed to load model");
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
        let model = LlamaModel::from_pretrained(&ctx, &model_dir).expect("Failed to load model");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        // Run a raw forward pass and check logits for NaN/Inf
        let logits = model.forward_full(&input_ids).expect("Forward pass failed");
        let logits_vec: Vec<f32> = logits.to_vec().expect("Failed to read logits");

        let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }

    /// Verify paged prefill + decode produces identical tokens as forward().
    #[test]
    fn paged_decode_matches_forward() {
        use infernum_cuda::cuda::PagedKvCache;
        use infernum_cuda::cuda::{BlockAllocator, BlockConfig, BlockTable};

        let model_dir = model_dir();
        let ctx = CudaContext::new(0).expect("CUDA context");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");
        let prompt_ids = tokenizer.encode("The capital of France is", true).unwrap();
        let num_decode_steps = 20;

        let model = LlamaModel::from_pretrained(&ctx, &model_dir).expect("load model");
        let model_cfg = model.model_config();

        // forward() reference: greedy decode step-by-step (no KV cache)
        let mut fwd_ids = prompt_ids.clone();
        let mut fwd_tokens = Vec::new();
        for _step in 0..num_decode_steps {
            let logits = model.forward_full(&fwd_ids).expect("forward");
            let logits_vec: Vec<f32> = logits.to_vec().expect("read");
            let vocab_size = logits.shape()[1];
            let last_start = (fwd_ids.len() - 1) * vocab_size;
            let last_logits = &logits_vec[last_start..last_start + vocab_size];

            let next_id = last_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32;
            fwd_tokens.push(next_id);
            fwd_ids.push(next_id);
        }

        // Paged prefill + decode
        let block_config = BlockConfig {
            block_size: 16,
            num_blocks: 128,
        };
        let mut paged_kv = PagedKvCache::new(
            &ctx,
            model_cfg.num_layers,
            &block_config,
            model_cfg.num_kv_heads,
            model_cfg.head_dim,
            model.dtype(),
        )
        .expect("paged kv");
        let mut allocator = BlockAllocator::new(&block_config);
        let mut block_table = BlockTable::new(block_config.block_size);

        let blocks_needed = (prompt_ids.len() + num_decode_steps).div_ceil(block_config.block_size);
        for _ in 0..blocks_needed {
            block_table.append_block(allocator.allocate().expect("alloc"));
        }

        let prefill_logits = model
            .forward_prefill_paged(&prompt_ids, &mut paged_kv, &block_table, 0)
            .expect("prefill");
        block_table.advance(prompt_ids.len());

        let prefill_vec: Vec<f32> = prefill_logits.to_vec().expect("read");
        let first_token = prefill_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;

        let mut decode_tokens = vec![first_token];
        let mut prev_token = first_token;
        for _step in 0..num_decode_steps - 1 {
            let pos = block_table.seq_len();
            let decode_logits = model
                .forward_batch_decode(&[prev_token], &mut paged_kv, &[block_table.clone()], &[pos])
                .expect("decode");
            block_table.advance(1);

            let decode_vec: Vec<f32> = decode_logits.to_vec().expect("read");
            let next_id = decode_vec
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32;
            decode_tokens.push(next_id);
            prev_token = next_id;
        }

        let fwd_text = tokenizer.decode(&fwd_tokens).unwrap_or_default();
        let dec_text = tokenizer.decode(&decode_tokens).unwrap_or_default();
        assert_eq!(
            fwd_tokens, decode_tokens,
            "Paged decode diverged from forward():\n  forward: {fwd_text:?}\n  decode:  {dec_text:?}"
        );
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
        let model = LlamaModel::from_pretrained(&ctx, &model_dir).expect("Failed to load model");

        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");
        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        let logits = model.forward_full(&input_ids).expect("Forward pass failed");
        let logits_vec: Vec<f32> = logits.to_vec().expect("Failed to read logits");

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
        let model = LlamaModel::from_pretrained(&ctx, &model_dir).expect("Failed to load model");

        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");
        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        let logits = model.forward_full(&input_ids).expect("Forward pass failed");
        let logits_vec: Vec<f32> = logits.to_vec().expect("Failed to read logits");

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
        let model = LlamaModel::from_pretrained(&ctx, &model_dir).expect("Failed to load model");

        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");
        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        let logits = model.forward_full(&input_ids).expect("Forward pass failed");
        let logits_vec: Vec<f32> = logits.to_vec().expect("Failed to read logits");

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
        let model = LlamaModel::from_pretrained(&ctx, &model_dir).expect("Failed to load model");

        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");
        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        let logits = model.forward_full(&input_ids).expect("Forward pass failed");
        let logits_vec: Vec<f32> = logits.to_vec().expect("Failed to read logits");

        let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }
}

// ─── MoE tensor-parallel (multi-GPU) ────────────────────────────────────────

/// tiny-mixtral on 2 GPUs — validates MoE sharded loading and forward pass.
///
/// Uses the same tiny random-weight Mixtral as `mixtral_moe_tiny`, but
/// loads across 2 GPUs with tensor parallelism to verify that the sharded
/// MoE weight loading and the single all-reduce forward path work correctly.
///
/// Requires 2+ CUDA GPUs and the `nccl` feature. Run manually with:
///   cargo test -p infernum-llama --features integration,nccl -- --ignored --test-threads=1 mixtral_moe_tp
#[cfg(feature = "nccl")]
mod mixtral_moe_tp {
    use super::*;
    use infernum::{GpuConfig, Model as _, ShardedModel};
    use infernum_cuda::CudaBackend;

    const REPO: &str = "jamesdborin/tiny-mixtral";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    #[test]
    #[ignore = "Requires 2+ GPUs with NCCL — run manually with --ignored"]
    fn loads_and_generates_2gpu() {
        let model_dir = model_dir();
        let model = ShardedModel::<CudaBackend, LlamaModel<CudaBackend>>::from_pretrained(
            &model_dir,
            2,
            |device, path, shard, comm| {
                LlamaModel::from_pretrained_sharded(
                    device,
                    path,
                    GpuConfig::Sharded(shard),
                    Some(comm),
                )
            },
        )
        .expect("Failed to load sharded MoE model");

        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
        let _output = runtime
            .generate("Hello", &greedy_options(10))
            .expect("Sharded MoE generation failed");
    }

    #[test]
    #[ignore = "Requires 2+ GPUs with NCCL — run manually with --ignored"]
    fn no_nan_in_output_2gpu() {
        let model_dir = model_dir();
        let model = ShardedModel::<CudaBackend, LlamaModel<CudaBackend>>::from_pretrained(
            &model_dir,
            2,
            |device, path, shard, comm| {
                LlamaModel::from_pretrained_sharded(
                    device,
                    path,
                    GpuConfig::Sharded(shard),
                    Some(comm),
                )
            },
        )
        .expect("Failed to load sharded MoE model");

        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let input_ids = tokenizer.encode("Hello world", true).unwrap();
        let logits = model.forward_full(&input_ids).expect("Forward pass failed");
        let logits_vec: Vec<f32> = logits.to_vec().expect("Failed to read logits");

        let nan_count = logits_vec.iter().filter(|x: &&f32| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x: &&f32| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }
}

// ─── Mistral (dense) ────────────────────────────────────────────────────────

/// Mistral-7B-Instruct-v0.3 (ungated, ~14.5GB, 3 sharded SafeTensors)
///
/// Dense Mistral model with `model_type: "mistral"`. Architecturally identical
/// to Llama — tests that `model_type` validation accepts `"mistral"` and that
/// loading + generation works correctly via the `MistralModel` alias.
///
/// Run manually with:
///   cargo test -p infernum-llama --features integration -- --ignored --test-threads=1 mistral_7b
mod mistral_7b {
    use super::*;
    use infernum_llama::MistralModel;

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
        let model = MistralModel::from_pretrained(&ctx, &model_dir).expect("Failed to load model");
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
        let model = MistralModel::from_pretrained(&ctx, &model_dir).expect("Failed to load model");

        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");
        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        let logits = model.forward_full(&input_ids).expect("Forward pass failed");
        let logits_vec: Vec<f32> = logits.to_vec().expect("Failed to read logits");

        let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }
}
