//! Minimal benchmark for infernum decode throughput
//! Measures raw forward_next_token performance (no tokenizer decode needed)
//!
//! Usage:
//!   cargo run --release --example bench --features cuda -- models/llama-3.2-1b 256
//!   cargo run --release --example bench --features cuda -- --dtype bf16 models/llama-3.2-1b 256
//!   cargo run --release --example bench --features cuda -- --pool models/llama-3.2-1b 256

#![cfg(feature = "cuda")]

use std::time::Instant;

use clap::Parser;

use infernum::cuda::CudaContext;
use infernum::model::Model;
use infernum::GenerateOptions;
use infernum_llama::LlamaModel;
use infernum_runtime::Engine;

#[derive(Parser)]
#[command(name = "bench")]
struct Cli {
    /// Path to model directory or .gguf file
    model: String,

    /// Number of tokens to generate
    #[arg(default_value_t = 128)]
    n_gen: usize,

    /// Compute dtype: f32 or bf16 (only for SafeTensors models)
    #[arg(long, default_value = "f32")]
    dtype: String,

    /// Enable buffer pool
    #[arg(long)]
    pool: bool,

    /// Enable CUDA graph capture/replay for the decode loop
    #[arg(long)]
    graphs: bool,
}

fn bench_model<M: Model>(
    model: M,
    ctx: &CudaContext,
    n_gen: usize,
    use_cuda_graphs: bool,
) -> infernum::Result<()> {
    let mut engine = Engine::new(ctx, model)?;

    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let options = GenerateOptions {
        max_new_tokens: n_gen,
        eos_token_id: None,
        sampling: None,
        use_kv_cache: true,
        use_cuda_graphs,
    };

    // Warmup
    let _ = engine.generate(
        &prompt,
        &GenerateOptions {
            max_new_tokens: 2,
            eos_token_id: None,
            sampling: None,
            use_kv_cache: true,
            use_cuda_graphs: false,
        },
    )?;

    // Benchmark
    let start = Instant::now();
    let tokens = engine.generate(&prompt, &options)?;
    let elapsed = start.elapsed();

    let generated = tokens.len() - prompt.len();
    let tok_s = generated as f64 / elapsed.as_secs_f64();

    println!(
        "{generated} tokens in {:.2}s = {:.1} tok/s",
        elapsed.as_secs_f64(),
        tok_s,
    );

    if let Some(pool) = ctx.buffer_pool() {
        eprintln!(
            "Pool stats: {} hits, {} misses, {:.1} MB cached",
            pool.hits(),
            pool.misses(),
            pool.free_bytes() as f64 / (1024.0 * 1024.0),
        );
    }

    Ok(())
}

fn main() -> infernum::Result<()> {
    let cli = Cli::parse();
    let mut ctx = CudaContext::new(0)?;

    if cli.pool {
        ctx.enable_buffer_pool();
        eprintln!("Buffer pool: ENABLED");
    } else {
        eprintln!("Buffer pool: disabled");
    }

    if cli.graphs {
        eprintln!("CUDA graphs: ENABLED");
    } else {
        eprintln!("CUDA graphs: disabled");
    }

    let is_gguf = cli.model.ends_with(".gguf");

    match cli.dtype.as_str() {
        "f32" => {
            let model = if is_gguf {
                LlamaModel::from_gguf(&ctx, &cli.model)?
            } else {
                LlamaModel::<f32>::from_pretrained(&ctx, &cli.model)?
            };
            eprintln!(
                "Model loaded: {} layers, {} hidden, dtype=f32",
                model.config().num_hidden_layers,
                model.config().hidden_size,
            );
            bench_model(model, &ctx, cli.n_gen, cli.graphs)
        }
        "bf16" => {
            assert!(
                !is_gguf,
                "bf16 dtype is only supported for SafeTensors models"
            );
            let model = LlamaModel::<infernum::dtype::BF16>::from_pretrained(&ctx, &cli.model)?;
            eprintln!(
                "Model loaded: {} layers, {} hidden, dtype=bf16",
                model.config().num_hidden_layers,
                model.config().hidden_size,
            );
            bench_model(model, &ctx, cli.n_gen, cli.graphs)
        }
        other => panic!("Unsupported dtype: {other}. Use f32 or bf16."),
    }
}
