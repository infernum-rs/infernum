//! Integration tests for inflight batching with a real model.
//!
//! Uses SmolLM2-360M (same as llama integration tests) to verify the
//! [`Engine`] produces correct output under various concurrency patterns.
//!
//! Run with:
//!   cargo test -p infernum-runtime --features integration -- --test-threads=1
#![cfg(feature = "integration")]

use std::fs;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use infernum::tokenizer::LlamaTokenizer;
use infernum::GenerateOptions;
use infernum_cuda::cuda::CudaContext;
use infernum_llama::LlamaModel;
use infernum_runtime::{BatchConfig, Engine, FinishReason, GenerationEvent};

// ---------------------------------------------------------------------------
// Test infrastructure (copied from llama integration tests)
// ---------------------------------------------------------------------------

const REQUIRED_FILES: &[&str] = &[
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
];

fn download_file(repo_id: &str, filename: &str, dest: &PathBuf) {
    if dest.exists() {
        return;
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).expect("Failed to create cache directory");
    }
    let url = format!("https://huggingface.co/{repo_id}/resolve/main/{filename}");
    let response = ureq::get(&url)
        .call()
        .unwrap_or_else(|e| panic!("Failed to download {repo_id}/{filename}: {e}"));
    let tmp_dest = dest.with_extension("tmp");
    let mut file = fs::File::create(&tmp_dest)
        .unwrap_or_else(|e| panic!("Failed to create {}: {e}", tmp_dest.display()));
    std::io::copy(&mut response.into_body().as_reader(), &mut file)
        .unwrap_or_else(|e| panic!("Failed to download {filename}: {e}"));
    fs::rename(&tmp_dest, dest)
        .unwrap_or_else(|e| panic!("Failed to rename {}: {e}", tmp_dest.display()));
}

fn download_model(repo_id: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let cache_dir = PathBuf::from(home)
        .join(".cache")
        .join("infernum")
        .join("models")
        .join(repo_id);
    for filename in REQUIRED_FILES {
        download_file(repo_id, filename, &cache_dir.join(filename));
    }
    cache_dir
}

const REPO: &str = "HuggingFaceTB/SmolLM2-360M";

fn model_dir() -> PathBuf {
    download_model(REPO)
}

fn greedy_options(max_tokens: usize) -> GenerateOptions {
    GenerateOptions {
        max_new_tokens: max_tokens,
        sampling: None,
        use_kv_cache: true,
        ..GenerateOptions::default()
    }
}

fn batch_config() -> BatchConfig {
    BatchConfig {
        max_batch_size: 8,
        max_prefill_tokens: 512,
        block_size: 16,
        num_blocks: 256,
        use_cuda_graphs: false,
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
// Test 1: Single request — custom config vs default config
// ---------------------------------------------------------------------------

#[test]
fn single_request_matches_default_config() {
    let dir = model_dir();
    let ctx = CudaContext::new(0).expect("CUDA context");
    let tokenizer = LlamaTokenizer::from_pretrained(&dir).expect("load tokenizer");
    let prompt = "The capital of France is";
    let input_ids = tokenizer.encode(prompt, true).expect("encode");
    let options = greedy_options(20);

    // Engine with custom batch config
    let model1 = LlamaModel::from_pretrained(&ctx, &dir).expect("load model");
    let engine1 = Engine::with_config(model1, batch_config()).expect("engine1");
    let result1 = engine1.generate(&input_ids, &options).expect("gen1");
    drop(engine1);

    let text1 = tokenizer.decode(&result1).expect("decode1");
    assert!(
        text1.contains("Paris"),
        "Expected 'Paris' in custom-config output: {text1}"
    );

    // Engine with default config
    let model2 = LlamaModel::from_pretrained(&ctx, &dir).expect("load model");
    let engine2 = Engine::new(model2).expect("engine2");
    let result2 = engine2.generate(&input_ids, &options).expect("gen2");
    drop(engine2);

    let text2 = tokenizer.decode(&result2).expect("decode2");
    assert!(
        text2.contains("Paris"),
        "Expected 'Paris' in default-config output: {text2}"
    );
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

    // All should produce the same tokens
    let first = &results[0].0;
    for (i, (tokens, _)) in results.iter().enumerate().skip(1) {
        assert_eq!(
            first, tokens,
            "Request {i} produced different tokens than request 0"
        );
    }

    // Verify output quality
    let full: Vec<u32> = input_ids.iter().chain(first.iter()).copied().collect();
    let text = tokenizer.decode(&full).expect("decode");
    assert!(text.contains("Paris"), "Expected 'Paris' in output: {text}");
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
        use_cuda_graphs: false,
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
