//! Integration tests that download real models and verify generation output.
//!
//! Gated behind the `integration` feature so they don't run during normal
//! `cargo test`. Run with:
//!
//!   cargo test -p infernum-deepseek --features integration -- --test-threads=1
//!
//! Models are cached in `~/.cache/infernum/models/`, so subsequent runs are fast.
#![cfg(feature = "integration")]

mod test_helpers;

use infernum::tokenizer::LlamaTokenizer;
use infernum::Model as _;
use infernum_cuda::CudaContext;
use infernum_deepseek::DeepSeekCudaEngine;
use infernum_runtime::Runtime;

// ─── DeepSeek-V3 tiny (random weights, MLA + MoE plumbing test) ─────────────

/// `yujiepan/deepseek-v3-tiny-random` (ungated, ~8.8MB, 2 layers, 256 experts
/// top-8, MLA with `q_lora_rank=16`, `kv_lora_rank=16`, random weights)
///
/// Tests MLA projection pipeline, sigmoid routing, shared expert,
/// dense→MoE transition (`first_k_dense_replace=1`). Random weights
/// produce garbage output so we only check no `NaN`/`Inf`.
mod deepseek_v3_tiny {
    use std::path::PathBuf;

    use super::*;

    const REPO: &str = "yujiepan/deepseek-v3-tiny-random";

    fn model_dir() -> PathBuf {
        test_helpers::download_model(REPO)
    }

    #[test]
    fn no_nan_in_prefill() {
        let model_dir = model_dir();
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model =
            DeepSeekCudaEngine::from_pretrained(ctx, &model_dir).expect("Failed to load model");

        let logits = model.forward(&[1, 2, 3, 4]).expect("Forward pass failed");
        let logits_vec: Vec<f32> = logits
            .tensor()
            .to_vec::<f32>()
            .expect("Failed to read logits");

        let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }

    #[test]
    fn generates_tokens() {
        let model_dir = model_dir();
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let model =
            DeepSeekCudaEngine::from_pretrained(ctx, &model_dir).expect("Failed to load model");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
        let output = runtime
            .generate("Hello", &test_helpers::greedy_options(8))
            .expect("Generation failed");

        assert!(
            !output.is_empty(),
            "Expected non-empty output from DeepSeek model"
        );
        println!("DeepSeek-V3-tiny output: {output}");
    }
}
