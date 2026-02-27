//! Integration tests for DeepSeek model family (V3, R1).
//!
//! Run with:
//!   cargo test -p infernum-cuda --features integration -- --test-threads=1 deepseek
#![cfg(feature = "integration")]

mod test_helpers;

use std::path::PathBuf;

use infernum::tokenizer::LlamaTokenizer;
use infernum_cuda::cuda::CudaContext;
use infernum_cuda::CudaBackend;
use infernum_deepseek::DeepSeekModel;
use infernum_runtime::Runtime;

use test_helpers::{download_model, greedy_options};

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
    /// method with device tensors.
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
