//! Minimal benchmark for infernum decode throughput
//! Measures raw forward_next_token performance (no tokenizer decode needed)
//!
//! Usage:
//!   cargo run --release --example bench --features cuda -- models/llama-3.2-1b 256

#![cfg(feature = "cuda")]

use std::path::Path;
use std::time::Instant;

use clap::Parser;
use serde::Deserialize;

use infernum::GenerateOptions;
use infernum_cuda::cuda::CudaContext;
use infernum_cuda::Model;
use infernum_deepseek::DeepSeekModel;
use infernum_gemma::GemmaModel;
use infernum_llama::LlamaModel;
use infernum_qwen::QwenModel;
use infernum_runtime::Engine;

#[derive(Parser)]
#[command(name = "bench")]
struct Cli {
    /// Path to model directory or .gguf file
    model: String,

    /// Number of tokens to generate
    #[arg(default_value_t = 128)]
    n_gen: usize,

    /// Enable CUDA graph capture/replay for the decode loop
    #[arg(long)]
    graphs: bool,
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

fn detect_model_type(model_path: &str) -> infernum::Result<String> {
    let config_path = Path::new(model_path).join("config.json");
    let data = std::fs::read_to_string(&config_path).map_err(|e| {
        infernum::Error::Io(std::io::Error::new(
            e.kind(),
            format!("Failed to read {}: {e}", config_path.display()),
        ))
    })?;
    let probe: ModelTypeProbe = serde_json::from_str(&data).map_err(|e| {
        infernum::Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to parse config.json: {e}"),
        ))
    })?;
    Ok(probe.model_type)
}

fn bench_model<M: Model + Send + 'static>(
    model: M,
    n_gen: usize,
    use_cuda_graphs: bool,
) -> infernum::Result<()> {
    let prompt = vec![1u32, 15043, 29892, 920, 526, 366, 2599, 13];

    let engine = Engine::new(model)?;

    let options = GenerateOptions {
        max_new_tokens: n_gen,
        use_cuda_graphs,
        ..GenerateOptions::default()
    };

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

    Ok(())
}

fn bench_with_info<M: Model + Send + 'static>(
    model: M,
    num_layers: usize,
    hidden_size: usize,
    dtype: &str,
    n_gen: usize,
    use_cuda_graphs: bool,
) -> infernum::Result<()> {
    eprintln!(
        "Model loaded: {} layers, {} hidden, dtype={}",
        num_layers, hidden_size, dtype,
    );
    bench_model(model, n_gen, use_cuda_graphs)
}

fn main() -> infernum::Result<()> {
    let cli = Cli::parse();
    let ctx = CudaContext::new(0)?;

    if cli.graphs {
        eprintln!("CUDA graphs: ENABLED");
    } else {
        eprintln!("CUDA graphs: disabled");
    }

    let is_gguf = cli.model.ends_with(".gguf");
    let model_type = if is_gguf {
        "llama".to_string()
    } else {
        detect_model_type(&cli.model)?
    };
    let family = match model_type.as_str() {
        "llama" | "mistral" | "mixtral" => "llama",
        "qwen2" | "qwen3" | "qwen3_moe" => "qwen",
        "deepseek_v3" => "deepseek",
        "gemma2" | "gemma3_text" => "gemma",
        other => panic!("Unsupported model_type: {other}"),
    };

    let result = match family {
        "llama" => {
            let model = if is_gguf {
                LlamaModel::from_gguf(&ctx, &cli.model)?
            } else {
                LlamaModel::from_pretrained(&ctx, &cli.model)?
            };
            let dtype = format!("{}", model.dtype());
            let (nl, hs) = (model.config().num_hidden_layers, model.config().hidden_size);
            bench_with_info(model, nl, hs, &dtype, cli.n_gen, cli.graphs)
        }
        "qwen" => {
            let model = QwenModel::from_pretrained(&ctx, &cli.model)?;
            let dtype = format!("{}", model.dtype());
            let (nl, hs) = (model.config().num_hidden_layers, model.config().hidden_size);
            bench_with_info(model, nl, hs, &dtype, cli.n_gen, cli.graphs)
        }
        "deepseek" => {
            let model = DeepSeekModel::from_pretrained(&ctx, &cli.model)?;
            let dtype = format!("{}", model.dtype());
            let (nl, hs) = (model.config().num_hidden_layers, model.config().hidden_size);
            bench_with_info(model, nl, hs, &dtype, cli.n_gen, cli.graphs)
        }
        "gemma" => {
            let model = GemmaModel::from_pretrained(&ctx, &cli.model)?;
            let dtype = format!("{}", model.dtype());
            let (nl, hs) = (model.config().num_hidden_layers, model.config().hidden_size);
            bench_with_info(model, nl, hs, &dtype, cli.n_gen, cli.graphs)
        }
        other => panic!("Unsupported family: {other}"),
    };

    if let Some(pool) = ctx.buffer_pool() {
        eprintln!(
            "Pool stats: {} hits, {} misses, {:.1} MB cached",
            pool.hits(),
            pool.misses(),
            pool.free_bytes() as f64 / (1024.0 * 1024.0),
        );
    }

    result
}
