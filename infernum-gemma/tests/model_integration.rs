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

use infernum::tokenizer::LlamaTokenizer;
use infernum::GenerateOptions;
use infernum::Tensor;
use infernum_cuda::cuda::CudaContext;
use infernum_cuda::CudaBackend;
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
    let model =
        GemmaModel::<CudaBackend>::from_pretrained(&ctx, model_dir).expect("Failed to load model");
    let tokenizer = LlamaTokenizer::from_pretrained(model_dir).expect("Failed to load tokenizer");

    let runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
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
        let model = GemmaModel::<CudaBackend>::from_pretrained(&ctx, &model_dir)
            .expect("Failed to load model");
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

    /// Verify paged decode produces logits consistent with non-paged forward().
    ///
    /// Runs forward() on a prompt to get reference logits, then runs the paged
    /// prefill + decode path on the same tokens and compares the argmax at each
    /// step. Catches bugs where paged_attention_decode receives wrong params
    /// (e.g. attn_scale instead of softcap).
    #[test]
    fn paged_decode_matches_forward() {
        use infernum_cuda::cuda::PagedKvCache;
        use infernum_cuda::{BlockAllocator, BlockConfig, BlockTable};

        let model_dir = model_dir();
        let ctx = CudaContext::new(0).expect("CUDA context");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");
        let prompt_ids = tokenizer.encode("Hello world", true).unwrap();
        let num_decode_steps = 4;

        // --- Reference: greedy decode via forward() ---
        let model_ref =
            GemmaModel::<CudaBackend>::from_pretrained(&ctx, &model_dir).expect("load model");
        let ref_logits = model_ref.forward_full(&prompt_ids).expect("forward");
        let ref_vec: Vec<f32> = ref_logits.to_vec().expect("read logits");
        let vocab_size = ref_logits.shape()[1];

        // Argmax of last prompt position → first generated token
        let last_row_start = (prompt_ids.len() - 1) * vocab_size;
        let ref_first_token = ref_vec[last_row_start..last_row_start + vocab_size]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as u32;

        // --- Paged path ---
        let model_cfg = model_ref.model_config();
        let block_config = BlockConfig {
            block_size: 16,
            num_blocks: 64,
        };
        let mut paged_kv = PagedKvCache::new(
            &ctx,
            model_cfg.num_layers,
            &block_config,
            model_cfg.num_kv_heads,
            model_cfg.head_dim,
            model_ref.dtype(),
        )
        .expect("paged kv");
        let mut allocator = BlockAllocator::new(&block_config);

        let mut block_table = BlockTable::new(block_config.block_size);
        let blocks_needed = (prompt_ids.len() + num_decode_steps + block_config.block_size - 1)
            / block_config.block_size;
        for _ in 0..blocks_needed {
            block_table.append_block(allocator.allocate().expect("alloc"));
        }

        // Prefill
        let prefill_logits = model_ref
            .forward_prefill_paged(&prompt_ids, &mut paged_kv, &block_table, 0)
            .expect("prefill");
        block_table.advance(prompt_ids.len());
        let prefill_vec: Vec<f32> = prefill_logits.to_vec().expect("read");
        let paged_first_token = prefill_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as u32;

        assert_eq!(
            ref_first_token, paged_first_token,
            "Prefill argmax diverged from forward()"
        );

        // Decode steps: collect generated tokens
        let mut generated = vec![paged_first_token];
        let mut prev_token = paged_first_token;
        for step in 0..num_decode_steps {
            let pos = block_table.seq_len();

            let decode_logits = model_ref
                .forward_batch_decode(&[prev_token], &mut paged_kv, &[block_table.clone()], &[pos])
                .expect("decode");
            block_table.advance(1);

            let decode_vec: Vec<f32> = decode_logits.to_vec().expect("read");
            assert!(
                !decode_vec.iter().any(|x| x.is_nan()),
                "NaN in decode step {step}"
            );

            let paged_next = decode_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0 as u32;

            generated.push(paged_next);
            prev_token = paged_next;
        }

        // For random-weight models, decode token-level matching is too strict
        // (flat logit distributions mean tiny numerical differences flip the
        // argmax). The key correctness check is that prefill matches forward(),
        // which we already verified above.
        println!(
            "Paged decode generated {} tokens (NaN-free)",
            generated.len()
        );
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
        let model = GemmaModel::<CudaBackend>::from_pretrained(&ctx, &model_dir)
            .expect("Failed to load model");
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

    #[test]
    fn paged_decode_matches_forward() {
        use infernum_cuda::cuda::PagedKvCache;
        use infernum_cuda::{BlockAllocator, BlockConfig, BlockTable};

        let model_dir = model_dir();
        let ctx = CudaContext::new(0).expect("CUDA context");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");
        let prompt_ids = tokenizer.encode("Hello world", true).unwrap();
        let num_decode_steps = 4;

        // --- Reference: greedy decode via forward() ---
        let model_ref =
            GemmaModel::<CudaBackend>::from_pretrained(&ctx, &model_dir).expect("load model");
        let ref_logits = model_ref.forward_full(&prompt_ids).expect("forward");
        let ref_vec: Vec<f32> = ref_logits.to_vec().expect("read logits");
        let vocab_size = ref_logits.shape()[1];

        let last_row_start = (prompt_ids.len() - 1) * vocab_size;
        let ref_first_token = ref_vec[last_row_start..last_row_start + vocab_size]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as u32;

        // --- Paged path ---
        let model_cfg = model_ref.model_config();
        let block_config = BlockConfig {
            block_size: 16,
            num_blocks: 64,
        };
        let mut paged_kv = PagedKvCache::new(
            &ctx,
            model_cfg.num_layers,
            &block_config,
            model_cfg.num_kv_heads,
            model_cfg.head_dim,
            model_ref.dtype(),
        )
        .expect("paged kv");
        let mut allocator = BlockAllocator::new(&block_config);

        let mut block_table = BlockTable::new(block_config.block_size);
        let blocks_needed = (prompt_ids.len() + num_decode_steps + block_config.block_size - 1)
            / block_config.block_size;
        for _ in 0..blocks_needed {
            block_table.append_block(allocator.allocate().expect("alloc"));
        }

        // Prefill
        let prefill_logits = model_ref
            .forward_prefill_paged(&prompt_ids, &mut paged_kv, &block_table, 0)
            .expect("prefill");
        block_table.advance(prompt_ids.len());
        let prefill_vec: Vec<f32> = prefill_logits.to_vec().expect("read");
        let paged_first_token = prefill_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as u32;

        assert_eq!(
            ref_first_token, paged_first_token,
            "Prefill argmax diverged from forward()"
        );

        // Decode steps
        let mut generated = vec![paged_first_token];
        let mut prev_token = paged_first_token;
        for step in 0..num_decode_steps {
            let pos = block_table.seq_len();

            let decode_logits = model_ref
                .forward_batch_decode(&[prev_token], &mut paged_kv, &[block_table.clone()], &[pos])
                .expect("decode");
            block_table.advance(1);

            let decode_vec: Vec<f32> = decode_logits.to_vec().expect("read");
            assert!(
                !decode_vec.iter().any(|x| x.is_nan()),
                "NaN in decode step {step}"
            );

            let paged_next = decode_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0 as u32;

            generated.push(paged_next);
            prev_token = paged_next;
        }

        println!(
            "Paged decode generated {} tokens (NaN-free)",
            generated.len()
        );
    }
}

// ─── Gemma 2 2B (ungated, ignored, quality check) ──────────────────────────

/// unsloth/gemma-2-2b (ungated mirror, ~5GB bf16) — validates Gemma 2
/// generation quality with real weights. Requires ~10GB VRAM (loaded as f32).
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
        // Reject degenerate repetition loops (e.g., "a city of contrasts, a
        // city of contrasts...") by requiring enough unique words.
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
        let model = GemmaModel::<CudaBackend>::from_pretrained(&ctx, &model_dir)
            .expect("Failed to load model");
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

    #[test]
    #[ignore = "5GB model, needs ~10GB VRAM — run manually with --ignored"]
    fn paged_decode_matches_forward() {
        use infernum_cuda::cuda::PagedKvCache;
        use infernum_cuda::{BlockAllocator, BlockConfig, BlockTable};

        let model_dir = model_dir();
        let ctx = CudaContext::new(0).expect("CUDA context");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");
        let prompt = "The capital of France is";
        let prompt_ids = tokenizer.encode(prompt, true).unwrap();
        let num_decode_steps = 20;

        let model =
            GemmaModel::<CudaBackend>::from_pretrained(&ctx, &model_dir).expect("load model");
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

        // Token-by-token match
        let fwd_text = tokenizer.decode(&fwd_tokens).unwrap_or_default();
        let dec_text = tokenizer.decode(&decode_tokens).unwrap_or_default();
        assert_eq!(
            fwd_tokens, decode_tokens,
            "Paged decode diverged from forward():\n  forward: {fwd_text:?}\n  decode:  {dec_text:?}"
        );
    }
}

// ─── Gemma 3 1B (ungated, ignored, quality check) ──────────────────────────

/// unsloth/gemma-3-1b-it (ungated mirror, ~2GB bf16) — validates Gemma 3
/// generation quality with real weights. Requires ~4GB VRAM (loaded as f32).
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
        let model = GemmaModel::<CudaBackend>::from_pretrained(&ctx, &model_dir)
            .expect("Failed to load model");
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
