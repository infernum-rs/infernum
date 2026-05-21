//! Multi-GPU text generation example
//!
//! Loads a SafeTensors model across multiple GPUs using tensor parallelism
//! and generates text. Auto-detects model family (Llama/Mistral/Mixtral/Qwen/Gemma).
//!
//! Usage:
//!   cargo run --example generate_parallel --features nccl --
//!     -m /path/to/model --gpus 2 "Hello"

#![cfg(feature = "nccl")]

use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use clap::Parser;
use serde::Deserialize;

use infernum::tokenizer::LlamaTokenizer;
use infernum::{GenerateOptions, Model as _, Result, SamplingParams};
use infernum_gemma::{GemmaShardedGraphEngine, GemmaShardedGraphEngineExt as _};
use infernum_llama::{LlamaShardedGraphEngine, LlamaShardedGraphEngineExt as _};
use infernum_qwen::{QwenShardedGraphEngine, QwenShardedGraphEngineExt as _};
use infernum_runtime::Runtime;

/// Multi-GPU text generation with tensor parallelism
#[derive(Parser)]
#[command(name = "generate_parallel")]
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
    #[arg(default_value = "Hello")]
    prompt: String,

    /// Number of GPUs to use
    #[arg(long, default_value_t = 2)]
    gpus: usize,

    /// Maximum tokens to generate
    #[arg(short = 'n', long, default_value_t = 100)]
    max_tokens: usize,

    /// Use greedy (argmax) decoding instead of sampling
    #[arg(long)]
    greedy: bool,

    /// Sampling temperature
    #[arg(short, long, default_value_t = 0.7)]
    temperature: f32,

    /// Nucleus sampling threshold
    #[arg(short = 'p', long, default_value_t = 0.9)]
    top_p: f32,

    /// RNG seed for sampling
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Repetition penalty factor
    #[arg(short = 'r', long, default_value_t = 1.1)]
    repetition_penalty: f32,

    /// Number of recent tokens to consider for repetition penalty
    #[arg(long, default_value_t = 64)]
    repetition_penalty_window: usize,

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

fn run_parallel<M: infernum::Model + Send + 'static>(
    model: M,
    tokenizer: LlamaTokenizer,
    world_size: usize,
    cli: &Cli,
) -> Result<()>
where
    M::B: infernum::DecodeBufferOps,
{
    let cfg = model.config();
    println!(
        "Model loaded ({} layers, head_dim={}, vocab {})",
        cfg.num_layers,
        cfg.head_dim,
        tokenizer.vocab_size(),
    );

    let options = GenerateOptions {
        max_new_tokens: cli.max_tokens,
        eos_token_id: None,
        sampling: if cli.greedy {
            None
        } else {
            Some(SamplingParams {
                temperature: cli.temperature,
                top_p: cli.top_p,
                seed: cli.seed,
                repetition_penalty: cli.repetition_penalty,
                repetition_penalty_window: cli.repetition_penalty_window,
            })
        },
        use_kv_cache: true,
    };

    if let Some(ref params) = options.sampling {
        println!(
            "Sampling: temperature={}, top_p={}, seed={}",
            params.temperature, params.top_p, params.seed,
        );
    } else {
        println!("Decoding: greedy (argmax)");
    }

    let runtime = Runtime::with_max_seq_len(model, tokenizer, cli.max_seq_len)?;

    print!("{}", cli.prompt);
    io::stdout().flush()?;

    let start = Instant::now();
    let output_tokens = runtime.generate_stream(&cli.prompt, &options)?;
    let elapsed = start.elapsed();

    let prompt_len = runtime.tokenizer().encode(&cli.prompt, true)?.len();
    let generated = output_tokens.len() - prompt_len;

    println!();
    println!(
        "Generated {} tokens in {:.2}s ({:.1} tokens/sec) on {} GPUs",
        generated,
        elapsed.as_secs_f64(),
        generated as f64 / elapsed.as_secs_f64(),
        world_size,
    );

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let world_size = cli.gpus;
    let model_dir = Path::new(&cli.model);

    println!(
        "Loading model from: {} across {} GPUs",
        cli.model, world_size,
    );

    let model_type = detect_model_type(&cli.model)?;
    let tokenizer = LlamaTokenizer::from_pretrained(&cli.model)?;

    let t0 = Instant::now();

    match model_type.as_str() {
        "llama" | "mistral" | "mixtral" => {
            let model = LlamaShardedGraphEngine::from_pretrained(world_size, model_dir)?;
            println!("Loaded in {:.2}s", t0.elapsed().as_secs_f64());
            run_parallel(model, tokenizer, world_size, &cli)
        }
        "qwen2" | "qwen3" | "qwen3_moe" => {
            let model = QwenShardedGraphEngine::from_pretrained(world_size, model_dir)?;
            println!("Loaded in {:.2}s", t0.elapsed().as_secs_f64());
            run_parallel(model, tokenizer, world_size, &cli)
        }
        "gemma2" | "gemma3_text" => {
            let model = GemmaShardedGraphEngine::from_pretrained(world_size, model_dir)?;
            println!("Loaded in {:.2}s", t0.elapsed().as_secs_f64());
            run_parallel(model, tokenizer, world_size, &cli)
        }
        other => Err(infernum::Error::UnsupportedModel(format!(
            "Unsupported model_type: \"{other}\". Supported: llama, mistral, mixtral, qwen2, qwen3, qwen3_moe, gemma2, gemma3_text"
        ))),
    }
}
