//! Multi-GPU text generation example
//!
//! Loads a SafeTensors model across multiple GPUs using tensor parallelism
//! and generates text.
//!
//! Usage:
//!   cargo run --example generate_parallel --features nccl --
//!     -m /path/to/llama --gpus 2 "Hello"

use std::io::{self, Write};
use std::thread;
use std::time::Instant;

use clap::Parser;
use cudarc::driver::CudaDevice;

use infernum::cuda::CudaContext;
use infernum::tokenizer::LlamaTokenizer;
use infernum::{GenerateOptions, GpuConfig, NcclCommunicator, Result, SamplingParams, ShardConfig};
use infernum_llama::LlamaModel;
use infernum_runtime::ParallelRuntime;

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

    // Create NCCL communicators (one per GPU)
    let devices: Vec<_> = (0..world_size)
        .map(|i| CudaDevice::new(i).expect("failed to create CUDA device"))
        .collect();
    let comms = NcclCommunicator::from_devices(devices)?;
    println!(
        "NCCL communicators created in {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    // Load sharded models in parallel (one thread per GPU)
    let t0 = Instant::now();
    let model_path = &cli.model;

    let models: Vec<(CudaContext, LlamaModel<f32>)> = thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                s.spawn(move || {
                    let ctx = CudaContext::new(rank)?;
                    let gpu_config = GpuConfig::Sharded(ShardConfig { rank, world_size });
                    let model = LlamaModel::<f32>::from_pretrained_sharded(
                        &ctx, model_path, gpu_config, comm,
                    )?;
                    Ok::<_, infernum::Error>((ctx, model))
                })
            })
            .collect();

        handles
            .into_iter()
            .map(|h| h.join().expect("GPU thread panicked"))
            .collect::<Result<Vec<_>>>()
    })?;

    println!(
        "Models loaded in {:.2}s ({} layers, {} hidden)",
        t0.elapsed().as_secs_f64(),
        models[0].1.config().num_hidden_layers,
        models[0].1.config().hidden_size,
    );

    // Create tokenizer
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

    // Create parallel runtime and generate
    let mut runtime = ParallelRuntime::new(models, tokenizer)?;

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
