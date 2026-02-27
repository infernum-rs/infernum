//! Integration tests that download real models and verify generation output.
//!
//! Gated behind the `integration` feature so they don't run during normal
//! `cargo test`. Run with:
//!
//!   cargo test -p infernum-deepseek --features integration -- --test-threads=1
//!
//! Models are cached in `~/.cache/infernum/models/`, so subsequent runs are fast.
#![cfg(feature = "integration")]

use std::fs;
use std::path::PathBuf;

use infernum::tokenizer::LlamaTokenizer;
use infernum::GenerateOptions;
use infernum_cuda::cuda::CudaContext;
use infernum_cuda::CudaBackend;
use infernum_deepseek::DeepSeekModel;
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
    let model = DeepSeekModel::<CudaBackend>::from_pretrained(&ctx, model_dir)
        .expect("Failed to load model");
    let tokenizer = LlamaTokenizer::from_pretrained(model_dir).expect("Failed to load tokenizer");

    let runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
    runtime
        .generate(prompt, &greedy_options(max_tokens))
        .expect("Generation failed")
}

// ─── DeepSeek-V3 tiny (random weights, MLA + MoE plumbing test) ─────────────

/// yujiepan/deepseek-v3-tiny-random (ungated, ~8.8MB, 2 layers, 256 experts
/// top-8, MLA with q_lora_rank=16, kv_lora_rank=16, random weights)
///
/// Tests MLA projection pipeline, sigmoid routing, shared expert,
/// dense→MoE transition (first_k_dense_replace=1). Random weights
/// produce garbage output so we only check no NaN/Inf.
mod deepseek_v3_tiny {
    use super::*;

    const REPO: &str = "yujiepan/deepseek-v3-tiny-random";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    #[test]
    fn loads_and_generates() {
        let output = generate_greedy(&model_dir(), "The capital of France is", 10);
        assert!(
            !output.is_empty(),
            "Expected non-empty output from DeepSeek model"
        );
        println!("DeepSeek-V3-tiny output: {output}");
    }

    #[test]
    fn no_nan_in_output() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model_dir = model_dir();
        let model = DeepSeekModel::<CudaBackend>::from_pretrained(&ctx, &model_dir)
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

    /// Tests batched decode via the `Model::forward_batch_decode` trait
    /// method with device tensors. This validates the batched forward
    /// path used by the inflight batching engine.
    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn batched_decode_no_nan() {
        use infernum::backend::TensorFactory;
        use infernum::Model;
        use infernum_cuda::cuda::PagedKvCache;
        use infernum_cuda::{BlockAllocator, BlockConfig, BlockTable};

        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model_dir = model_dir();
        let model = DeepSeekModel::<CudaBackend>::from_pretrained(&ctx, &model_dir)
            .expect("Failed to load model");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let model_config = model.model_config();
        let block_size = 16;
        let num_blocks = 256;
        let block_config = BlockConfig {
            block_size,
            num_blocks,
        };
        let mut paged_kv = PagedKvCache::new(
            &ctx,
            model_config.num_layers,
            &block_config,
            model_config.num_kv_heads,
            model_config.head_dim,
            model.dtype(),
        )
        .expect("Failed to create paged KV cache");
        let mut allocator = BlockAllocator::new(&block_config);

        // Prefill a prompt first
        let input_ids = tokenizer.encode("The capital of France is", true).unwrap();
        let num_prompt_blocks = input_ids.len().div_ceil(block_size);
        let mut bt = BlockTable::new(block_size);
        for _ in 0..num_prompt_blocks {
            let block_idx = allocator.allocate().expect("Failed to allocate block");
            bt.append_block(block_idx);
        }
        let _prefill_logits = model
            .forward_prefill_paged(&input_ids, &mut paged_kv, &bt, 0)
            .expect("Prefill failed");
        bt.advance(input_ids.len());

        // Allocate one more block for the decode token
        if bt.needs_new_block() {
            let block_idx = allocator
                .allocate()
                .expect("Failed to allocate decode block");
            bt.append_block(block_idx);
        }

        // Prepare tensor-based batched decode step
        let batch_size = 1;
        let max_blocks_per_seq = bt.blocks().len();
        let position = input_ids.len() as i32;
        let seq_len = position + 1;
        let max_seq_len = seq_len as usize;

        let mut block_tables_flat: Vec<i32> = bt.blocks().iter().map(|&b| b as i32).collect();
        block_tables_flat.resize(max_blocks_per_seq, 0);

        let token_ids_t =
            <CudaBackend as TensorFactory>::from_u32_slice(&ctx, &[batch_size], &[2_u32])
                .expect("upload token_ids");
        let block_tables_t = <CudaBackend as TensorFactory>::from_i32_slice(
            &ctx,
            &[batch_size * max_blocks_per_seq],
            &block_tables_flat,
        )
        .expect("upload block_tables");
        let seq_lens_t =
            <CudaBackend as TensorFactory>::from_i32_slice(&ctx, &[batch_size], &[seq_len])
                .expect("upload seq_lens");
        let positions_t =
            <CudaBackend as TensorFactory>::from_i32_slice(&ctx, &[batch_size], &[position])
                .expect("upload positions");

        let mut runtime_state = infernum_cuda::CudaRuntimeState::test_placeholder();

        let logits = Model::forward_batch_decode(
            &model,
            &token_ids_t,
            &mut paged_kv,
            &mut runtime_state,
            &block_tables_t,
            &seq_lens_t,
            &positions_t,
            batch_size,
            max_blocks_per_seq,
            max_seq_len,
        )
        .expect("Batched decode failed");

        let logits_vec: Vec<f32> = logits.tensor().to_vec().expect("Failed to read logits");
        let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }
}
