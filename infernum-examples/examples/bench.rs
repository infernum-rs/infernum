//! Minimal benchmark for infernum decode throughput
//! Measures raw forward_next_token performance (no tokenizer decode needed)
//!
//! Usage:
//!   cargo run --release --example bench --features cuda -- models/llama-3.2-1b 256
//!   cargo run --release --example bench --features cuda -- --dtype bf16 models/llama-3.2-1b 256

#![cfg(feature = "cuda")]

use std::path::Path;
use std::time::Instant;

use clap::Parser;
use serde::Deserialize;

use infernum::cuda::CudaContext;
use infernum::model::Model;
use infernum::GenerateOptions;
use infernum_deepseek::DeepSeekModel;
use infernum_gemma::GemmaModel;
use infernum_llama::LlamaModel;
use infernum_qwen::QwenModel;
use infernum_runtime::{BatchConfig, Engine};

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

    /// Enable CUDA graph capture/replay for the decode loop
    #[arg(long)]
    graphs: bool,

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

fn detect_model_type(model_path: &str) -> infernum::Result<String> {
    let config_path = Path::new(model_path).join("config.json");
    let content = std::fs::read_to_string(&config_path)?;
    let probe: ModelTypeProbe = serde_json::from_str(&content)?;
    Ok(probe.model_type)
}

fn bench_model<M: Model + Send + 'static>(
    model: M,
    n_gen: usize,
    use_cuda_graphs: bool,
    max_seq_len: Option<usize>,
) -> infernum::Result<()> {
    let _ = max_seq_len; // TODO: re-add max_seq_len support to Engine
    let config = BatchConfig {
        use_cuda_graphs,
        ..BatchConfig::default()
    };
    let engine = Engine::with_config(model, config)?;

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

    Ok(())
}

fn bench_with_info<M: Model + Send + 'static>(
    model: M,
    num_layers: usize,
    hidden_size: usize,
    dtype: &str,
    n_gen: usize,
    use_cuda_graphs: bool,
    max_seq_len: Option<usize>,
) -> infernum::Result<()> {
    eprintln!(
        "Model loaded: {} layers, {} hidden, dtype={}",
        num_layers, hidden_size, dtype,
    );
    bench_model(model, n_gen, use_cuda_graphs, max_seq_len)
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

    let result = match (cli.dtype.as_str(), family) {
        ("f32", "llama") => {
            let model = if is_gguf {
                LlamaModel::from_gguf(&ctx, &cli.model)?
            } else {
                LlamaModel::<f32>::from_pretrained(&ctx, &cli.model)?
            };
            let (nl, hs) = (model.config().num_hidden_layers, model.config().hidden_size);
            bench_with_info(model, nl, hs, "f32", cli.n_gen, cli.graphs, cli.max_seq_len)
        }
        ("bf16", "llama") => {
            assert!(
                !is_gguf,
                "bf16 dtype is only supported for SafeTensors models"
            );
            let model = LlamaModel::<infernum::dtype::BF16>::from_pretrained(&ctx, &cli.model)?;
            let (nl, hs) = (model.config().num_hidden_layers, model.config().hidden_size);
            bench_with_info(
                model,
                nl,
                hs,
                "bf16",
                cli.n_gen,
                cli.graphs,
                cli.max_seq_len,
            )
        }
        ("f32", "qwen") => {
            let model = QwenModel::<f32>::from_pretrained(&ctx, &cli.model)?;
            let (nl, hs) = (model.config().num_hidden_layers, model.config().hidden_size);
            bench_with_info(model, nl, hs, "f32", cli.n_gen, cli.graphs, cli.max_seq_len)
        }
        ("bf16", "qwen") => {
            let model = QwenModel::<infernum::dtype::BF16>::from_pretrained(&ctx, &cli.model)?;
            let (nl, hs) = (model.config().num_hidden_layers, model.config().hidden_size);
            bench_with_info(
                model,
                nl,
                hs,
                "bf16",
                cli.n_gen,
                cli.graphs,
                cli.max_seq_len,
            )
        }
        ("f32", "deepseek") => {
            let model = DeepSeekModel::<f32>::from_pretrained(&ctx, &cli.model)?;
            let (nl, hs) = (model.config().num_hidden_layers, model.config().hidden_size);
            bench_with_info(model, nl, hs, "f32", cli.n_gen, cli.graphs, cli.max_seq_len)
        }
        ("bf16", "deepseek") => {
            let model = DeepSeekModel::<infernum::dtype::BF16>::from_pretrained(&ctx, &cli.model)?;
            let (nl, hs) = (model.config().num_hidden_layers, model.config().hidden_size);
            bench_with_info(
                model,
                nl,
                hs,
                "bf16",
                cli.n_gen,
                cli.graphs,
                cli.max_seq_len,
            )
        }
        ("f32", "gemma") => {
            let model = GemmaModel::<f32>::from_pretrained(&ctx, &cli.model)?;
            let (nl, hs) = (model.config().num_hidden_layers, model.config().hidden_size);
            bench_with_info(model, nl, hs, "f32", cli.n_gen, cli.graphs, cli.max_seq_len)
        }
        ("bf16", "gemma") => {
            let model = GemmaModel::<infernum::dtype::BF16>::from_pretrained(&ctx, &cli.model)?;
            let (nl, hs) = (model.config().num_hidden_layers, model.config().hidden_size);
            bench_with_info(
                model,
                nl,
                hs,
                "bf16",
                cli.n_gen,
                cli.graphs,
                cli.max_seq_len,
            )
        }
        (other, _) => panic!("Unsupported dtype: {other}. Use f32 or bf16."),
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
