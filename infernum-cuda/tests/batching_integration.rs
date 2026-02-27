//! Integration tests for inflight batching with a real model.
//!
//! Uses SmolLM2-360M (same as llama integration tests) to verify the
//! [`Engine`] produces correct output under various concurrency patterns.
//!
//! Run with:
//!   cargo test -p infernum-cuda --features integration -- --test-threads=1 batching
#![cfg(feature = "integration")]

mod test_helpers;

use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use infernum::tokenizer::LlamaTokenizer;
use infernum_cuda::cuda::CudaContext;
use infernum_llama::LlamaModel;
use infernum_runtime::{BatchConfig, Engine, FinishReason, GenerationEvent};

use test_helpers::{download_model, greedy_options};

const REPO: &str = "HuggingFaceTB/SmolLM2-360M";

fn model_dir() -> PathBuf {
    download_model(REPO)
}

fn batch_config() -> BatchConfig {
    BatchConfig {
        max_batch_size: 8,
        max_prefill_tokens: 512,
        block_size: 16,
        num_blocks: 256,
    }
}

/// Collect all tokens from a generation event channel.
fn collect_tokens(rx: mpsc::Receiver<GenerationEvent>) -> (Vec<u32>, FinishReason) {
    let mut tokens = Vec::new();
    let mut reason = FinishReason::Length;
    for event in rx {
        match event {
            GenerationEvent::Token(id) => tokens.push(id),
            GenerationEvent::Finished(r) => {
                reason = r;
                break;
            }
            GenerationEvent::Error(e) => panic!("Generation error: {e}"),
        }
    }
    (tokens, reason)
}

// ---------------------------------------------------------------------------
// Test 1: Single request — greedy generation correctness
// ---------------------------------------------------------------------------

#[test]
fn single_request() {
    let dir = model_dir();
    let ctx = CudaContext::new(0).expect("CUDA context");
    let model = LlamaModel::from_pretrained(&ctx, &dir).expect("load model");
    let tokenizer = LlamaTokenizer::from_pretrained(&dir).expect("load tokenizer");

    let engine = Engine::with_config(model, batch_config()).expect("engine");
    let input_ids = tokenizer
        .encode("The capital of France is", true)
        .expect("encode");
    let result = engine
        .generate(&input_ids, &greedy_options(20))
        .expect("gen");

    let text = tokenizer.decode(&result).expect("decode");
    assert!(text.contains("Paris"), "Expected 'Paris' in output: {text}");
}

// ---------------------------------------------------------------------------
// Test 2: Concurrent identical requests produce identical output
// ---------------------------------------------------------------------------

#[test]
fn concurrent_identical_requests() {
    let dir = model_dir();
    let ctx = CudaContext::new(0).expect("CUDA context");
    let model = LlamaModel::from_pretrained(&ctx, &dir).expect("load model");
    let tokenizer = LlamaTokenizer::from_pretrained(&dir).expect("load tokenizer");
    let engine = Engine::with_config(model, batch_config()).expect("engine");

    let prompt = "The capital of France is";
    let input_ids = tokenizer.encode(prompt, true).expect("encode");
    let options = greedy_options(15);

    let num_concurrent = 4;
    let mut receivers = Vec::new();

    for _ in 0..num_concurrent {
        let (tx, rx) = mpsc::channel();
        engine.submit(input_ids.clone(), options.clone(), tx);
        receivers.push(rx);
    }

    let results: Vec<(Vec<u32>, FinishReason)> =
        receivers.into_iter().map(collect_tokens).collect();

    // All should produce the same tokens (greedy, deterministic)
    let first = &results[0].0;
    for (i, (tokens, _)) in results.iter().enumerate().skip(1) {
        assert_eq!(
            first, tokens,
            "Request {i} produced different tokens than request 0"
        );
    }

    let text = tokenizer.decode(first).expect("decode");
    assert!(
        !text.is_empty(),
        "Concurrent requests should produce non-empty output"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Concurrent requests with different prompts
// ---------------------------------------------------------------------------

#[test]
fn concurrent_different_prompts() {
    let dir = model_dir();
    let ctx = CudaContext::new(0).expect("CUDA context");
    let model = LlamaModel::from_pretrained(&ctx, &dir).expect("load model");
    let tokenizer = LlamaTokenizer::from_pretrained(&dir).expect("load tokenizer");
    let engine = Engine::with_config(model, batch_config()).expect("engine");

    let prompts = [
        "The capital of France is",
        "Water boils at",
        "The largest planet in the solar system is",
    ];

    let mut receivers = Vec::new();
    let mut all_input_ids = Vec::new();

    for prompt in &prompts {
        let input_ids = tokenizer.encode(prompt, true).expect("encode");
        let (tx, rx) = mpsc::channel();
        engine.submit(input_ids.clone(), greedy_options(20), tx);
        all_input_ids.push(input_ids);
        receivers.push(rx);
    }

    for (i, rx) in receivers.into_iter().enumerate() {
        let (tokens, _reason) = collect_tokens(rx);
        assert!(
            !tokens.is_empty(),
            "Request {i} ({}) produced no tokens",
            prompts[i]
        );

        // Verify no garbage — all token IDs should be in valid range
        for &tok in &tokens {
            assert!(
                tok < 100_000,
                "Request {i}: suspicious token ID {tok} (likely corruption)"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 4: Staggered arrival — join running batch
// ---------------------------------------------------------------------------

#[test]
fn staggered_arrival() {
    let dir = model_dir();
    let ctx = CudaContext::new(0).expect("CUDA context");
    let model = LlamaModel::from_pretrained(&ctx, &dir).expect("load model");
    let tokenizer = LlamaTokenizer::from_pretrained(&dir).expect("load tokenizer");
    let engine = Engine::with_config(model, batch_config()).expect("engine");

    let prompt_a = "The capital of France is";
    let prompt_b = "Water boils at a temperature of";
    let ids_a = tokenizer.encode(prompt_a, true).expect("encode");
    let ids_b = tokenizer.encode(prompt_b, true).expect("encode");

    // Submit request A
    let (tx_a, rx_a) = mpsc::channel();
    engine.submit(ids_a.clone(), greedy_options(20), tx_a);

    // Wait a bit for A to start generating
    thread::sleep(Duration::from_millis(100));

    // Submit request B (should join mid-batch)
    let (tx_b, rx_b) = mpsc::channel();
    engine.submit(ids_b.clone(), greedy_options(20), tx_b);

    let (tokens_a, _) = collect_tokens(rx_a);
    let (tokens_b, _) = collect_tokens(rx_b);

    assert!(!tokens_a.is_empty(), "Request A produced no tokens");
    assert!(!tokens_b.is_empty(), "Request B produced no tokens");

    // Verify A and B are different (different prompts)
    assert_ne!(
        tokens_a, tokens_b,
        "Different prompts should produce different tokens"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Early cancellation — drop receiver
// ---------------------------------------------------------------------------

#[test]
fn early_cancellation() {
    let dir = model_dir();
    let ctx = CudaContext::new(0).expect("CUDA context");
    let model = LlamaModel::from_pretrained(&ctx, &dir).expect("load model");
    let tokenizer = LlamaTokenizer::from_pretrained(&dir).expect("load tokenizer");
    let engine = Engine::with_config(model, batch_config()).expect("engine");

    let input_ids = tokenizer
        .encode("Count from one to one hundred:", true)
        .expect("encode");

    // Submit a request with many tokens
    let (tx, rx) = mpsc::channel();
    engine.submit(input_ids.clone(), greedy_options(100), tx);

    // Receive a few tokens then drop
    let mut received = 0;
    for event in &rx {
        if let GenerationEvent::Token(_) = event {
            received += 1;
            if received >= 3 {
                break;
            }
        }
    }
    drop(rx);

    // Wait a moment for cleanup
    thread::sleep(Duration::from_millis(200));

    // Now submit another request — should work fine (blocks freed)
    let (tx2, rx2) = mpsc::channel();
    engine.submit(
        tokenizer.encode("Hello", true).expect("encode"),
        greedy_options(10),
        tx2,
    );
    let (tokens, _) = collect_tokens(rx2);
    assert!(
        !tokens.is_empty(),
        "Post-cancellation request should succeed"
    );
}

// ---------------------------------------------------------------------------
// Test 6: Memory pressure — small block pool
// ---------------------------------------------------------------------------

#[test]
fn memory_pressure() {
    let dir = model_dir();
    let ctx = CudaContext::new(0).expect("CUDA context");
    let model = LlamaModel::from_pretrained(&ctx, &dir).expect("load model");
    let tokenizer = LlamaTokenizer::from_pretrained(&dir).expect("load tokenizer");

    // Very small block pool — enough for about 2 short sequences
    let config = BatchConfig {
        max_batch_size: 4,
        max_prefill_tokens: 512,
        block_size: 16,
        num_blocks: 16, // 16 blocks * 16 tokens = 256 token capacity total
    };
    let engine = Engine::with_config(model, config).expect("engine");

    // Submit 3 requests with short prompts
    let input = tokenizer.encode("Hello", true).expect("encode");
    let mut receivers = Vec::new();
    for _ in 0..3 {
        let (tx, rx) = mpsc::channel();
        engine.submit(input.clone(), greedy_options(10), tx);
        receivers.push(rx);
    }

    // All should eventually complete (scheduler waits for blocks)
    for (i, rx) in receivers.into_iter().enumerate() {
        let (tokens, _) = collect_tokens(rx);
        assert!(
            !tokens.is_empty(),
            "Request {i} under memory pressure should still produce tokens"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 7: Sequential engine still works unchanged
// ---------------------------------------------------------------------------

#[test]
fn sequential_engine_regression() {
    let dir = model_dir();
    let ctx = CudaContext::new(0).expect("CUDA context");
    let model = LlamaModel::from_pretrained(&ctx, &dir).expect("load model");
    let tokenizer = LlamaTokenizer::from_pretrained(&dir).expect("load tokenizer");

    let engine = Engine::new(model).expect("engine");
    let input_ids = tokenizer
        .encode("The capital of France is", true)
        .expect("encode");
    let result = engine
        .generate(&input_ids, &greedy_options(20))
        .expect("gen");

    let text = tokenizer.decode(&result).expect("decode");
    assert!(
        text.contains("Paris"),
        "Sequential engine regression: expected 'Paris' in output: {text}"
    );
}
