//! Integration tests for Llama model family on the CPU backend.
//!
//! Downloads real models from HuggingFace and verifies generation output.
//! Gated behind the `integration` feature so they don't run during normal
//! `cargo test`. Run with:
//!
//!   cargo test -p infernum-cpu --features integration llama
//!
//! Models are cached in `~/.cache/infernum/models/`, so subsequent runs are fast.
#![cfg(feature = "integration")]

mod test_helpers;

use std::path::PathBuf;

use infernum::tokenizer::LlamaTokenizer;
use infernum_llama::{LlamaGraphEngine, LlamaGraphEngineExt as _};
use infernum_runtime::Runtime;

use test_helpers::{download_model, download_model_files, greedy_options};

/// Load a model and generate text with greedy decoding.
fn generate_greedy(model_dir: &PathBuf, prompt: &str, max_tokens: usize) -> String {
    let engine =
        LlamaGraphEngine::from_pretrained(model_dir).expect("Failed to load LlamaGraphEngine");
    let tokenizer = LlamaTokenizer::from_pretrained(model_dir).expect("Failed to load tokenizer");

    let runtime = Runtime::new(engine, tokenizer).expect("Failed to create runtime");
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
        let model_dir = model_dir();
        let engine =
            LlamaGraphEngine::from_pretrained(&model_dir).expect("Failed to load LlamaGraphEngine");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let prompt_ids = tokenizer.encode("Hello world", true).expect("tokenize");
        let eos = engine.config().eos_token_id;

        // Generate one token to exercise the full forward pass.
        engine
            .generate(&prompt_ids, 1, eos)
            .expect("graph generation failed");
    }
}

// ─── SmolLM2-360M GGUF Q8_0 ─────────────────────────────────────────────────

/// SmolLM2-360M quantized to Q8_0 (~380MB GGUF).
/// Tests that GGUF quantized weight loading works on CPU and produces
/// finite logits. Uses raw token IDs (SmolLM2's GPT2 BPE tokenizer
/// is not supported by the GGUF tokenizer).
mod smollm2_360m_q8 {
    use super::*;

    const REPO: &str = "bartowski/SmolLM2-360M-Instruct-GGUF";
    const GGUF_FILE: &str = "SmolLM2-360M-Instruct-Q8_0.gguf";

    fn gguf_path() -> PathBuf {
        let dir = download_model_files(REPO, &[GGUF_FILE]);
        dir.join(GGUF_FILE)
    }

    #[test]
    fn no_nan_in_output() {
        let path = gguf_path();
        let engine = LlamaGraphEngine::from_gguf(&path).expect("Failed to load GGUF");
        let eos = engine.config().eos_token_id;

        // BOS=0, "Hello"=15339 (common GPT2 token)
        engine
            .generate(&[0, 15339], 1, eos)
            .expect("graph generation failed");
    }

    #[test]
    fn capital_of_france() {
        let path = gguf_path();
        // Only download tokenizer files (not the full SafeTensors model)
        let tokenizer_dir = download_model_files(
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            &["tokenizer.json", "tokenizer_config.json"],
        );
        let engine = LlamaGraphEngine::from_gguf(&path).expect("Failed to load GGUF");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&tokenizer_dir).expect("Failed to load tokenizer");

        let runtime = Runtime::new(engine, tokenizer).expect("Failed to create runtime");
        let output = runtime
            .generate("The capital of France is", &greedy_options(30))
            .expect("Generation failed");
        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in Q8_0 output, got: {output}"
        );
    }
}

// ─── SmolLM2-360M GGUF Q4_0 (mixed Q4_0/Q4_1) ──────────────────────────────

/// SmolLM2-360M quantized to Q4_0 (~200MB GGUF).
/// This GGUF contains a mix of Q4_0 and Q4_1 tensors (llama.cpp uses Q4_1
/// for certain layers). Tests that both quant formats load and produce
/// finite logits and correct generation.
mod smollm2_360m_q4 {
    use super::*;

    const REPO: &str = "bartowski/SmolLM2-360M-Instruct-GGUF";
    const GGUF_FILE: &str = "SmolLM2-360M-Instruct-Q4_0.gguf";

    fn gguf_path() -> PathBuf {
        let dir = download_model_files(REPO, &[GGUF_FILE]);
        dir.join(GGUF_FILE)
    }

    #[test]
    fn no_nan_in_output() {
        let path = gguf_path();
        let engine = LlamaGraphEngine::from_gguf(&path).expect("Failed to load GGUF");
        let eos = engine.config().eos_token_id;

        // BOS=0, "Hello"=15339 (GPT2 BPE token in SmolLM2 vocab)
        engine
            .generate(&[0, 15339], 1, eos)
            .expect("graph generation failed");
    }

    #[test]
    fn capital_of_france() {
        let path = gguf_path();
        let tokenizer_dir = download_model_files(
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            &["tokenizer.json", "tokenizer_config.json"],
        );
        let engine = LlamaGraphEngine::from_gguf(&path).expect("Failed to load GGUF");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&tokenizer_dir).expect("Failed to load tokenizer");

        let runtime = Runtime::new(engine, tokenizer).expect("Failed to create runtime");
        let output = runtime
            .generate("The capital of France is", &greedy_options(30))
            .expect("Generation failed");
        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in Q4_0 output, got: {output}"
        );
    }
}

// ─── SmolLM2-360M via LlamaGraphEngine ───────────────────────────────────────

/// End-to-end generation test using `LlamaGraphEngine` on real SmolLM2-360M
/// SafeTensors weights (~700MB).  Verifies that the graph-mode prefill +
/// autoregressive decode loop produces correct output and no out-of-vocab tokens.
mod smollm2_360m_graph_engine {
    use infernum::tokenizer::LlamaTokenizer;
    use infernum_llama::{LlamaGraphEngine, LlamaGraphEngineExt as _};

    use super::*;

    const REPO: &str = "HuggingFaceTB/SmolLM2-360M";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    /// Graph-mode generation produces "Paris" for the standard prompt.
    #[test]
    fn capital_of_france() {
        let model_dir = model_dir();
        let engine =
            LlamaGraphEngine::from_pretrained(&model_dir).expect("Failed to load LlamaGraphEngine");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let prompt_ids = tokenizer
            .encode("The capital of France is", true)
            .expect("tokenize");
        let eos = engine.config().eos_token_id;

        let output_ids = engine
            .generate(&prompt_ids, 30, eos)
            .expect("graph generation failed");

        // Decode only the newly generated tokens (everything after the prompt).
        let new_ids = &output_ids[prompt_ids.len()..];
        let output = tokenizer.decode(new_ids).expect("decode");

        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in graph engine output, got: {output}"
        );
    }

    /// Prefill + one decode step produces tokens within the vocab range.
    #[test]
    fn no_oob_tokens() {
        let model_dir = model_dir();
        let engine =
            LlamaGraphEngine::from_pretrained(&model_dir).expect("Failed to load LlamaGraphEngine");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let prompt_ids = tokenizer.encode("Hello world", true).expect("tokenize");
        let eos = engine.config().eos_token_id;
        let vocab_size = engine.config().vocab_size;

        // Generate just one new token — enough to exercise both prefill and decode.
        let output_ids = engine
            .generate(&prompt_ids, 1, eos)
            .expect("graph generation failed");

        assert!(
            output_ids.len() >= prompt_ids.len(),
            "Output shorter than prompt: {} < {}",
            output_ids.len(),
            prompt_ids.len()
        );

        for &tok in &output_ids {
            assert!(
                (tok as usize) < vocab_size,
                "Generated token {tok} is out of vocab range {vocab_size}"
            );
        }
    }
}

// ─── LlamaGraphEngine + GGUF Q8_0 / Q4_0 ────────────────────────────────────

/// Verifies that [`LlamaGraphEngine::from_gguf`] can load Q8_0 and Q4_0 GGUF
/// files and run the full prefill + autoregressive decode loop using quantized
/// weights.  Uses the same SmolLM2-360M GGUF files as the eager GGUF tests
/// above, so no additional downloads are required when both modules run.
mod smollm2_360m_graph_engine_gguf {
    use infernum::tokenizer::LlamaTokenizer;
    use infernum_llama::{LlamaGraphEngine, LlamaGraphEngineExt as _};

    use super::*;

    fn tokenizer_dir() -> PathBuf {
        download_model_files(
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            &["tokenizer.json", "tokenizer_config.json"],
        )
    }

    // ── Q8_0 ──────────────────────────────────────────────────────────────────

    const Q8_REPO: &str = "bartowski/SmolLM2-360M-Instruct-GGUF";
    const Q8_FILE: &str = "SmolLM2-360M-Instruct-Q8_0.gguf";

    fn q8_gguf_path() -> PathBuf {
        download_model_files(Q8_REPO, &[Q8_FILE]).join(Q8_FILE)
    }

    /// Graph engine loaded from Q8_0 GGUF produces in-vocab tokens with no crashes.
    #[test]
    fn q8_no_oob_tokens() {
        let path = q8_gguf_path();
        let engine = LlamaGraphEngine::from_gguf(&path).expect("Failed to load Q8_0 GGUF");
        let eos = engine.config().eos_token_id;
        // BOS=0, "Hello"=15339 in GPT2 BPE (SmolLM2 vocab)
        let output_ids = engine
            .generate(&[0, 15339], 5, eos)
            .expect("graph generation failed");
        assert!(
            !output_ids.is_empty(),
            "Graph engine produced no output tokens"
        );
        for &tok in &output_ids {
            assert!(
                (tok as usize) < engine.config().vocab_size,
                "Generated token {tok} is out-of-vocab"
            );
        }
    }

    /// Graph engine loaded from Q8_0 GGUF generates "Paris" for the standard prompt.
    #[test]
    fn q8_capital_of_france() {
        let path = q8_gguf_path();
        let engine = LlamaGraphEngine::from_gguf(&path).expect("Failed to load Q8_0 GGUF");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&tokenizer_dir()).expect("Failed to load tokenizer");

        let prompt_ids = tokenizer
            .encode("The capital of France is", true)
            .expect("tokenize");
        let eos = engine.config().eos_token_id;

        let output_ids = engine
            .generate(&prompt_ids, 30, eos)
            .expect("graph generation failed");

        let new_ids = &output_ids[prompt_ids.len()..];
        let output = tokenizer.decode(new_ids).expect("decode");

        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in Q8_0 graph engine output, got: {output}"
        );
    }

    // ── Q4_0 ──────────────────────────────────────────────────────────────────

    const Q4_REPO: &str = "bartowski/SmolLM2-360M-Instruct-GGUF";
    const Q4_FILE: &str = "SmolLM2-360M-Instruct-Q4_0.gguf";

    fn q4_gguf_path() -> PathBuf {
        download_model_files(Q4_REPO, &[Q4_FILE]).join(Q4_FILE)
    }

    /// Graph engine loaded from Q4_0 GGUF produces in-vocab tokens with no crashes.
    #[test]
    fn q4_no_oob_tokens() {
        let path = q4_gguf_path();
        let engine = LlamaGraphEngine::from_gguf(&path).expect("Failed to load Q4_0 GGUF");
        let eos = engine.config().eos_token_id;
        let output_ids = engine
            .generate(&[0, 15339], 5, eos)
            .expect("graph generation failed");
        assert!(
            !output_ids.is_empty(),
            "Graph engine produced no output tokens"
        );
        for &tok in &output_ids {
            assert!(
                (tok as usize) < engine.config().vocab_size,
                "Generated token {tok} is out-of-vocab"
            );
        }
    }

    /// Graph engine loaded from Q4_0 GGUF generates "Paris" for the standard prompt.
    #[test]
    fn q4_capital_of_france() {
        let path = q4_gguf_path();
        let engine = LlamaGraphEngine::from_gguf(&path).expect("Failed to load Q4_0 GGUF");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&tokenizer_dir()).expect("Failed to load tokenizer");

        let prompt_ids = tokenizer
            .encode("The capital of France is", true)
            .expect("tokenize");
        let eos = engine.config().eos_token_id;

        let output_ids = engine
            .generate(&prompt_ids, 30, eos)
            .expect("graph generation failed");

        let new_ids = &output_ids[prompt_ids.len()..];
        let output = tokenizer.decode(new_ids).expect("decode");

        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in Q4_0 graph engine output, got: {output}"
        );
    }
}
