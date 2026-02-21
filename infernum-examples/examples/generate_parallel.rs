//! Multi-GPU text generation example
//!
//! Loads a SafeTensors model across multiple GPUs using tensor parallelism
//! and generates text.
//!
//! Usage:
//!   cargo run --example generate_parallel --features nccl --
//!     -m /path/to/llama --gpus 2 "Hello"

use std::io::{self, Write};
use std::time::Instant;

use clap::Parser;

use infernum::tokenizer::LlamaTokenizer;
use infernum::{GenerateOptions, Model as _, Result, SamplingParams, ShardedModel};
use infernum_llama::LlamaModel;
use infernum_runtime::Runtime;

/// Multi-GPU text generation with Llama models (tensor parallelism)
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let world_size = cli.gpus;

    println!(
        "Loading model from: {} across {} GPUs",
        cli.model, world_size
    );

    let t0 = Instant::now();
    let model = ShardedModel::<LlamaModel<f32>>::from_pretrained(&cli.model, world_size)?;
    let cfg = model.config();
    println!(
        "Model loaded in {:.2}s ({} layers, head_dim={})",
        t0.elapsed().as_secs_f64(),
        cfg.num_layers,
        cfg.head_dim,
    );

    let tokenizer = LlamaTokenizer::from_pretrained(&cli.model)?;
    println!("Tokenizer loaded (vocab {})", tokenizer.vocab_size());

    // Build generation options
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
        use_cuda_graphs: false,
    };

    if let Some(ref params) = options.sampling {
        println!(
            "Sampling: temperature={}, top_p={}, seed={}",
            params.temperature, params.top_p, params.seed,
        );
    } else {
        println!("Decoding: greedy (argmax)");
    }

    let mut runtime = Runtime::new(model, tokenizer)?;

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
