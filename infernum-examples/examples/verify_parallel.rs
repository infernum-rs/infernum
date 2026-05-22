//! Correctness verification: compare single-GPU vs multi-GPU output.
//!
//! Loads the same model on 1 GPU and on N GPUs (tensor-parallel), runs greedy
//! decoding with the same prompt, and verifies that the generated tokens match
//! exactly. Auto-detects model family (Llama/Mistral/Mixtral/Qwen/Gemma).
//!
//! Usage:
//!   cargo run --example verify_parallel --features nccl -- \
//!     -m /path/to/model --gpus 2

#![cfg(feature = "nccl")]

use std::path::Path;
use std::time::Instant;

use clap::Parser;
use cudarc::driver::CudaDevice;
use serde::Deserialize;

use infernum::tokenizer::LlamaTokenizer;
use infernum::{GenerateOptions, Result};
use infernum_cuda::cuda::CudaContext;
use infernum_gemma::{
    GemmaCudaGraphEngine, GemmaCudaGraphEngineExt as _, GemmaShardedGraphEngine,
    GemmaShardedGraphEngineExt as _,
};
use infernum_llama::{
    LlamaCudaGraphEngine, LlamaCudaGraphEngineExt as _, LlamaShardedGraphEngine,
    LlamaShardedGraphEngineExt as _,
};
use infernum_qwen::{
    QwenCudaGraphEngine, QwenCudaGraphEngineExt as _, QwenShardedGraphEngine,
    QwenShardedGraphEngineExt as _,
};
use infernum_runtime::Runtime;

/// Verify that multi-GPU TP produces identical output to single-GPU.
#[derive(Parser)]
#[command(name = "verify_parallel")]
struct Cli {
    /// Path to SafeTensors model directory
    #[arg(
        short,
        long,
        env = "LLAMA_MODEL_PATH",
        default_value = "models/llama-3.2-1b"
    )]
    model: String,

    /// Text prompt
    #[arg(default_value = "The capital of France is")]
    prompt: String,

    /// Number of GPUs for the parallel run
    #[arg(long, default_value_t = 2)]
    gpus: usize,

    /// Maximum tokens to generate
    #[arg(short = 'n', long, default_value_t = 20)]
    max_tokens: usize,

    /// Maximum KV cache sequence length (default: min(model max, 4096))
    #[arg(long)]
    max_seq_len: Option<usize>,
}

/// Peek at just the `model_type` field from config.json.
#[derive(Deserialize)]
struct ModelTypeProbe {
    #[serde(default = "default_model_type")]
    model_type: String,
}

fn default_model_type() -> String {
    "llama".to_string()
}

fn detect_model_type(model_path: &str) -> Result<String> {
    let config_path = Path::new(model_path).join("config.json");
    let content = std::fs::read_to_string(&config_path)?;
    let probe: ModelTypeProbe = serde_json::from_str(&content)?;
    Ok(probe.model_type)
}

fn greedy_generate<M: infernum::Model + Send + 'static>(
    model: M,
    tokenizer: LlamaTokenizer,
    max_seq_len: Option<usize>,
    prompt: &str,
    max_tokens: usize,
) -> Result<String>
where
    M::B: infernum::DecodeBufferOps,
{
    let runtime = Runtime::with_max_seq_len(model, tokenizer, max_seq_len)?;
    let t0 = Instant::now();
    let output = runtime.generate(
        prompt,
        &GenerateOptions {
            max_new_tokens: max_tokens,
            eos_token_id: None,
            sampling: None,
            use_kv_cache: true,
        },
    )?;
    println!("  Output: {output}");
    println!("  Generated in {:.2}s", t0.elapsed().as_secs_f64());
    Ok(output)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let world_size = cli.gpus;
    let model_dir = Path::new(&cli.model);

    let n_devices = CudaDevice::count().unwrap() as usize;
    if n_devices < world_size {
        eprintln!(
            "Need {} GPUs but only {} available — skipping",
            world_size, n_devices
        );
        return Ok(());
    }

    let model_type = detect_model_type(&cli.model)?;
    let family = match model_type.as_str() {
        "llama" | "mistral" | "mixtral" => "llama",
        "qwen2" | "qwen3" | "qwen3_moe" => "qwen",
        "gemma2" | "gemma3_text" => "gemma",
        other => {
            return Err(infernum::Error::UnsupportedModel(format!(
                "Unsupported model_type: \"{other}\""
            )))
        }
    };

    {
        let tokenizer = LlamaTokenizer::from_pretrained(&cli.model)?;
        let prompt_tokens = tokenizer.encode(&cli.prompt, true)?;
        println!("Prompt: {:?} ({} tokens)", cli.prompt, prompt_tokens.len());
    }

    // --- Single-GPU baseline ---
    let single_output = {
        println!("\n=== Single GPU ===");
        let t0 = Instant::now();
        let ctx = CudaContext::new(0)?;
        let tokenizer = LlamaTokenizer::from_pretrained(&cli.model)?;

        match family {
            "qwen" => {
                let model = QwenCudaGraphEngine::from_pretrained(ctx, model_dir)?;
                println!("  Loaded in {:.2}s", t0.elapsed().as_secs_f64());
                greedy_generate(
                    model,
                    tokenizer,
                    cli.max_seq_len,
                    &cli.prompt,
                    cli.max_tokens,
                )?
            }
            "gemma" => {
                let model = GemmaCudaGraphEngine::from_pretrained(ctx, model_dir)?;
                println!("  Loaded in {:.2}s", t0.elapsed().as_secs_f64());
                greedy_generate(
                    model,
                    tokenizer,
                    cli.max_seq_len,
                    &cli.prompt,
                    cli.max_tokens,
                )?
            }
            _ => {
                let model = LlamaCudaGraphEngine::from_pretrained(ctx, model_dir)?;
                println!("  Loaded in {:.2}s", t0.elapsed().as_secs_f64());
                greedy_generate(
                    model,
                    tokenizer,
                    cli.max_seq_len,
                    &cli.prompt,
                    cli.max_tokens,
                )?
            }
        }
    };

    // --- Multi-GPU ---
    let parallel_output = {
        println!("\n=== {world_size} GPUs (tensor parallel) ===");
        let t0 = Instant::now();
        let tokenizer = LlamaTokenizer::from_pretrained(&cli.model)?;

        match family {
            "qwen" => {
                let model = QwenShardedGraphEngine::from_pretrained(world_size, model_dir)?;
                println!("  Loaded in {:.2}s", t0.elapsed().as_secs_f64());
                greedy_generate(
                    model,
                    tokenizer,
                    cli.max_seq_len,
                    &cli.prompt,
                    cli.max_tokens,
                )?
            }
            "gemma" => {
                let model = GemmaShardedGraphEngine::from_pretrained(world_size, model_dir)?;
                println!("  Loaded in {:.2}s", t0.elapsed().as_secs_f64());
                greedy_generate(
                    model,
                    tokenizer,
                    cli.max_seq_len,
                    &cli.prompt,
                    cli.max_tokens,
                )?
            }
            _ => {
                let model = LlamaShardedGraphEngine::from_pretrained(world_size, model_dir)?;
                println!("  Loaded in {:.2}s", t0.elapsed().as_secs_f64());
                greedy_generate(
                    model,
                    tokenizer,
                    cli.max_seq_len,
                    &cli.prompt,
                    cli.max_tokens,
                )?
            }
        }
    };

    // --- Compare ---
    println!("\n=== Comparison ===");
    if single_output == parallel_output {
        println!("PASS: outputs match exactly");
    } else {
        println!("FAIL: outputs differ");
        println!("  Single GPU: {single_output}");
        println!("  {world_size} GPUs:    {parallel_output}");

        let tokenizer = LlamaTokenizer::from_pretrained(&cli.model)?;
        let single_toks = tokenizer.encode(&single_output, false)?;
        let parallel_toks = tokenizer.encode(&parallel_output, false)?;
        let max_len = single_toks.len().max(parallel_toks.len());
        for i in 0..max_len {
            let s = single_toks.get(i).copied();
            let p = parallel_toks.get(i).copied();
            if s != p {
                println!("  First difference at token {i}: single={s:?} parallel={p:?}");
                break;
            }
        }

        std::process::exit(1);
    }

    Ok(())
}
