//! Correctness verification: compare single-GPU vs multi-GPU output.
//!
//! Loads the same model on 1 GPU and on N GPUs (tensor-parallel), runs greedy
//! decoding with the same prompt, and verifies that the generated tokens match
//! exactly.
//!
//! Usage:
//!   cargo run --example verify_parallel --features nccl -- \
//!     -m /path/to/llama --gpus 2

use std::time::Instant;

use clap::Parser;
use cudarc::driver::CudaDevice;

use infernum::cuda::CudaContext;
use infernum::tokenizer::LlamaTokenizer;
use infernum::{GenerateOptions, Model as _, Result, ShardedModel};
use infernum_llama::LlamaModel;
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let n_devices = CudaDevice::count().unwrap() as usize;
    if n_devices < cli.gpus {
        eprintln!(
            "Need {} GPUs but only {} available — skipping",
            cli.gpus, n_devices
        );
        return Ok(());
    }

    let options = GenerateOptions {
        max_new_tokens: cli.max_tokens,
        eos_token_id: None,
        sampling: None, // greedy for deterministic comparison
        use_kv_cache: true,
        use_cuda_graphs: false,
    };

    // Print prompt info
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
        let model = LlamaModel::<f32>::from_pretrained(&ctx, &cli.model)?;
        println!(
            "Loaded in {:.2}s ({} layers, hidden={})",
            t0.elapsed().as_secs_f64(),
            model.config().num_hidden_layers,
            model.config().hidden_size,
        );

        let tokenizer = LlamaTokenizer::from_pretrained(&cli.model)?;
        let mut runtime = Runtime::new(model, tokenizer)?;
        let t0 = Instant::now();
        let output = runtime.generate(&cli.prompt, &options)?;
        let elapsed = t0.elapsed();

        println!("Output: {output}");
        println!("Generated in {:.2}s", elapsed.as_secs_f64());
        output
    };

    // --- Multi-GPU ---
    let world_size = cli.gpus;
    let parallel_output = {
        println!("\n=== {world_size} GPUs (tensor parallel) ===");
        let t0 = Instant::now();
        let model = ShardedModel::<LlamaModel<f32>>::from_pretrained(&cli.model, world_size)?;
        let cfg = model.config();
        println!(
            "Loaded in {:.2}s ({} layers, head_dim={})",
            t0.elapsed().as_secs_f64(),
            cfg.num_layers,
            cfg.head_dim,
        );

        let tokenizer = LlamaTokenizer::from_pretrained(&cli.model)?;
        let mut runtime = Runtime::new(model, tokenizer)?;
        let t0 = Instant::now();
        let output = runtime.generate(&cli.prompt, &options)?;
        let elapsed = t0.elapsed();

        println!("Output: {output}");
        println!("Generated in {:.2}s", elapsed.as_secs_f64());
        output
    };

    // --- Compare ---
    println!("\n=== Comparison ===");
    if single_output == parallel_output {
        println!("✓ PASS: outputs match exactly");
    } else {
        println!("✗ FAIL: outputs differ");
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
