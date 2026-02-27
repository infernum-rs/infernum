//! Integration tests for Qwen model family (Qwen2/2.5, Qwen3, Qwen3-MoE).
//!
//! Run with:
//!   cargo test -p infernum-cuda --features integration -- --test-threads=1 qwen
#![cfg(feature = "integration")]

mod test_helpers;

use std::path::PathBuf;

use infernum::tokenizer::LlamaTokenizer;
use infernum::Tensor;
use infernum_cuda::cuda::CudaContext;
use infernum_cuda::CudaBackend;
use infernum_qwen::QwenModel;
use infernum_runtime::Runtime;

use test_helpers::{download_model, greedy_options};

/// Load a model and generate text with greedy decoding.
fn generate_greedy(model_dir: &PathBuf, prompt: &str, max_tokens: usize) -> String {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let model =
        QwenModel::<CudaBackend>::from_pretrained(&ctx, model_dir).expect("Failed to load model");
    let tokenizer = LlamaTokenizer::from_pretrained(model_dir).expect("Failed to load tokenizer");

    let runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
    runtime
        .generate(prompt, &greedy_options(max_tokens))
        .expect("Generation failed")
}

// ─── Qwen2.5-0.5B (bf16, Q/K/V bias, tied embeddings) ──────────────────────

mod qwen2_5_0_5b {
    use super::*;

    const REPO: &str = "Qwen/Qwen2.5-0.5B";

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
        let model = QwenModel::<CudaBackend>::from_pretrained(&ctx, &model_dir)
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

    /// Verify paged prefill + decode produces identical tokens as forward().
    #[test]
    fn paged_decode_matches_forward() {
        use infernum_cuda::cuda::PagedKvCache;
        use infernum_cuda::{BlockAllocator, BlockConfig, BlockTable};

        let model_dir = model_dir();
        let ctx = CudaContext::new(0).expect("CUDA context");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");
        let prompt_ids = tokenizer.encode("The capital of France is", true).unwrap();
        let num_decode_steps = 20;

        let model =
            QwenModel::<CudaBackend>::from_pretrained(&ctx, &model_dir).expect("load model");
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
            "Paged decode diverged from forward():
  forward: {fwd_text:?}
  decode:  {dec_text:?}"
        );
    }
}

// ─── Qwen3-0.6B (bf16, QK-norm, no bias, tied embeddings) ──────────────────

mod qwen3_0_6b {
    use super::*;

    const REPO: &str = "Qwen/Qwen3-0.6B";

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
        let model = QwenModel::<CudaBackend>::from_pretrained(&ctx, &model_dir)
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

// ─── Qwen3-MoE tiny (random weights, MoE plumbing test) ─────────────────────

mod qwen3_moe_tiny {
    use super::*;

    const REPO: &str = "yujiepan/qwen3-moe-tiny-random";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    #[test]
    fn loads_and_generates() {
        let output = generate_greedy(&model_dir(), "The capital of France is", 10);
        assert!(
            !output.is_empty(),
            "Expected non-empty output from MoE model"
        );
        println!("Qwen3-MoE-tiny output: {output}");
    }

    #[test]
    fn no_nan_in_output() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model_dir = model_dir();
        let model = QwenModel::<CudaBackend>::from_pretrained(&ctx, &model_dir)
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
