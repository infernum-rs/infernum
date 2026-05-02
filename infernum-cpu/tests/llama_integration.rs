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
use infernum_cpu::CpuBackend;
use infernum_llama::LlamaModel;
use infernum_runtime::Runtime;

use test_helpers::{download_model, download_model_files, greedy_options};

/// Load a model and generate text with greedy decoding.
fn generate_greedy(model_dir: &PathBuf, prompt: &str, max_tokens: usize) -> String {
    let model =
        LlamaModel::<CpuBackend>::from_pretrained(&(), model_dir).expect("Failed to load model");
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
        let model_dir = model_dir();
        let model = LlamaModel::<CpuBackend>::from_pretrained(&(), &model_dir)
            .expect("Failed to load model");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&model_dir).expect("Failed to load tokenizer");

        let input_ids = tokenizer.encode("Hello world", true).unwrap();

        // Run a raw forward pass and check logits for NaN/Inf
        let hidden = model.forward_full(&input_ids).expect("Forward pass failed");
        let logits_vec = hidden.to_f32_vec();

        let nan_count = logits_vec.iter().filter(|x: &&f32| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x: &&f32| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
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
        let model = LlamaModel::<CpuBackend>::from_gguf(&(), &path).expect("Failed to load GGUF");

        // Use raw token IDs: "Hello" in GPT2 BPE (SmolLM2 vocab)
        // BOS=0, "Hello"=15339 (common GPT2 token)
        let input_ids = vec![0, 15339];

        let hidden = model.forward_full(&input_ids).expect("Forward pass failed");
        let logits_vec = hidden.to_f32_vec();

        let nan_count = logits_vec.iter().filter(|x: &&f32| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x: &&f32| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }

    #[test]
    fn capital_of_france() {
        let path = gguf_path();
        // Only download tokenizer files (not the full SafeTensors model)
        let tokenizer_dir = download_model_files(
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            &["tokenizer.json", "tokenizer_config.json"],
        );
        let model = LlamaModel::<CpuBackend>::from_gguf(&(), &path).expect("Failed to load GGUF");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&tokenizer_dir).expect("Failed to load tokenizer");

        let runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
        let output = runtime
            .generate("The capital of France is", &greedy_options(30))
            .expect("Generation failed");
        assert!(
            output.contains("Paris"),
            "Expected 'Paris' in Q8_0 output, got: {output}"
        );
    }
}

// ─── SmolLM2-360M graph execution vs eager ──────────────────────────────────

/// Verifies that the CPU graph executor produces identical per-position
/// argmax predictions as the eager CPU forward pass on SmolLM2-360M.
mod smollm2_360m_graph {
    use infernum::graph::{plan as compile_plan, Arena};
    use infernum::tokenizer::LlamaTokenizer;
    use infernum::transformer::build_rope_cache;
    use infernum_cpu::executor::execute;
    use infernum_cpu::{CpuBackend, CpuTensor};
    use infernum_llama::graph_builder::{build_prefill_graph, load_graph_weights_safetensors};
    use infernum_llama::LlamaModel;

    use super::*;

    const REPO: &str = "HuggingFaceTB/SmolLM2-360M";

    fn model_dir() -> PathBuf {
        download_model(REPO)
    }

    /// Graph prefill logits match eager logits (same argmax per position).
    #[test]
    #[ignore = "requires model download (~700MB)"]
    fn graph_matches_eager_logits() {
        let model_dir = model_dir();

        // ── Eager forward pass ────────────────────────────────────────────────
        let eager_model =
            LlamaModel::<CpuBackend>::from_pretrained(&(), &model_dir).expect("eager model load");
        let tokenizer = LlamaTokenizer::from_pretrained(&model_dir).expect("tokenizer load");
        let prompt = "The capital of France is";
        let input_ids = tokenizer.encode(prompt, true).expect("tokenize");
        let seq_len = input_ids.len();

        let eager_logits_tensor = eager_model.forward_full(&input_ids).expect("eager forward");
        let eager_logits = eager_logits_tensor.to_f32_vec();
        let config = eager_model.config().clone();
        let vocab_size = config.vocab_size;

        // ── Graph build + weight load ─────────────────────────────────────────
        let (graph, _weight_ids) =
            build_prefill_graph::<CpuBackend>(&config, seq_len, infernum::dtype::DType::F32);

        let weights =
            load_graph_weights_safetensors(&graph, &model_dir, &config).expect("weight loading");

        // ── RoPE inputs ───────────────────────────────────────────────────────
        let head_dim = config.head_dim();
        let (cos_full, sin_full) = build_rope_cache::<CpuBackend>(
            &(),
            head_dim,
            seq_len,
            config.rope_theta,
            None,
            infernum::dtype::DType::F32,
        )
        .expect("rope cache");

        // ── Execute graph ─────────────────────────────────────────────────────
        let token_tensor = CpuTensor::from_u32(&[seq_len], &input_ids);
        let inputs = vec![token_tensor, cos_full, sin_full];

        let ep = compile_plan(&graph);
        let mut arena = Arena::new(ep.arena_size);
        let output_nodes = graph.output_ids().to_vec();

        let outputs = execute(
            &ep,
            graph.nodes(),
            &mut arena,
            &weights,
            &inputs,
            &output_nodes,
            None,
        )
        .expect("graph execute");

        let graph_logits = outputs[0].to_f32_vec();

        // ── Verify: argmax must match at every position ───────────────────────
        assert_eq!(
            graph_logits.len(),
            eager_logits.len(),
            "logit tensor length mismatch"
        );
        for pos in 0..seq_len {
            let row = &graph_logits[pos * vocab_size..(pos + 1) * vocab_size];
            let graph_tok = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            let eager_row = &eager_logits[pos * vocab_size..(pos + 1) * vocab_size];
            let eager_tok = eager_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            assert_eq!(
                graph_tok, eager_tok,
                "argmax mismatch at position {pos}: graph={graph_tok} eager={eager_tok}"
            );
        }
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
        let model = LlamaModel::<CpuBackend>::from_gguf(&(), &path).expect("Failed to load GGUF");

        // BOS=0, "Hello"=15339 (GPT2 BPE token in SmolLM2 vocab)
        let input_ids = vec![0, 15339];

        let hidden = model.forward_full(&input_ids).expect("Forward pass failed");
        let logits_vec = hidden.to_f32_vec();

        let nan_count = logits_vec.iter().filter(|x: &&f32| x.is_nan()).count();
        let inf_count = logits_vec.iter().filter(|x: &&f32| x.is_infinite()).count();

        assert_eq!(nan_count, 0, "Found {nan_count} NaN values in logits");
        assert_eq!(inf_count, 0, "Found {inf_count} Inf values in logits");
    }

    #[test]
    fn capital_of_france() {
        let path = gguf_path();
        let tokenizer_dir = download_model_files(
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            &["tokenizer.json", "tokenizer_config.json"],
        );
        let model = LlamaModel::<CpuBackend>::from_gguf(&(), &path).expect("Failed to load GGUF");
        let tokenizer =
            LlamaTokenizer::from_pretrained(&tokenizer_dir).expect("Failed to load tokenizer");

        let runtime = Runtime::new(model, tokenizer).expect("Failed to create runtime");
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
    use infernum_llama::LlamaGraphEngine;

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
