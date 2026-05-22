//! Tensor-parallel integration tests: single-GPU vs multi-GPU output match.
//!
//! Downloads a real model from HuggingFace and verifies that greedy decoding
//! produces identical tokens on 1 GPU and on 2 GPUs (tensor parallel).
//!
//! These tests require at least 2 GPUs and NCCL support, so they are marked
//! `#[ignore]` and must be run explicitly:
//!
//!   cargo test -p infernum-cuda --features integration-nccl -- \
//!     --ignored --test-threads=1 tp_
//!
//! Models are cached in `~/.cache/infernum/models/`.
#![cfg(all(feature = "nccl", feature = "integration"))]

mod test_helpers;

use std::path::PathBuf;

use cudarc::driver::CudaDevice;
use infernum::tokenizer::LlamaTokenizer;
use infernum_cuda::cuda::CudaContext;
use infernum_llama::{
    LlamaCudaGraphEngine, LlamaCudaGraphEngineExt as _, LlamaShardedGraphEngine,
    LlamaShardedGraphEngineExt as _,
};
use infernum_runtime::Runtime;

use test_helpers::{download_model, greedy_options};

fn generate_single_gpu(model_dir: &PathBuf, prompt: &str, max_tokens: usize) -> String {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let model =
        LlamaCudaGraphEngine::from_pretrained(ctx, model_dir).expect("Failed to load model");
    let tokenizer = LlamaTokenizer::from_pretrained(model_dir).expect("Failed to load tokenizer");
    let runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
    runtime
        .generate(prompt, &greedy_options(max_tokens))
        .expect("Generation failed")
}

fn generate_two_gpu(model_dir: &PathBuf, prompt: &str, max_tokens: usize) -> String {
    let model = LlamaShardedGraphEngine::from_pretrained(2, model_dir)
        .expect("Failed to load sharded model");
    let tokenizer = LlamaTokenizer::from_pretrained(model_dir).expect("Failed to load tokenizer");
    let runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
    runtime
        .generate(prompt, &greedy_options(max_tokens))
        .expect("Generation failed")
}

/// SmolLM2-360M: single-GPU vs 2-GPU tensor parallel output must match exactly.
///
/// Requires 2+ GPUs. Run with --ignored.
mod smollm2_360m_tp {
    use super::*;

    const REPO: &str = "HuggingFaceTB/SmolLM2-360M";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    #[test]
    #[ignore = "requires 2+ GPUs; run manually: cargo test -p infernum-cuda --features integration-nccl -- --ignored --test-threads=1"]
    fn single_vs_two_gpu_greedy_match() {
        let n_devices = CudaDevice::count().unwrap() as usize;
        assert!(
            n_devices >= 2,
            "need at least 2 GPUs, found {n_devices}; skip or add more GPUs"
        );

        let model_dir = model_dir();
        let prompt = "The capital of France is";
        let max_tokens = 20;

        let single = generate_single_gpu(&model_dir, prompt, max_tokens);
        let parallel = generate_two_gpu(&model_dir, prompt, max_tokens);

        assert_eq!(
            single, parallel,
            "single-GPU and 2-GPU outputs differ\n  single:   {single:?}\n  parallel: {parallel:?}"
        );
    }
}
