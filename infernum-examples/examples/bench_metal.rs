//! Minimal benchmark for infernum Metal decode throughput
//! Measures raw forward pass performance with greedy decoding (no sampling)
//!
//! Usage:
//!   cargo run --release --example bench_metal --features metal -- /path/to/safetensors_dir 128
//!
//! Note: GGUF loading is not yet supported on Metal. Pass a SafeTensors directory.

use std::path::Path;
use std::time::Instant;

use clap::Parser;
use serde::Deserialize;

use infernum::GenerateOptions;
use infernum_gemma::{GemmaMetalGraphEngine, GemmaMetalGraphEngineExt as _};
use infernum_llama::{LlamaMetalGraphEngine, LlamaMetalGraphEngineExt as _};
use infernum_metal::{MetalBackend, MetalContext};
use infernum_qwen::{QwenMetalGraphEngine, QwenMetalGraphEngineExt as _};
use infernum_runtime::Engine;

#[derive(Parser)]
#[command(name = "bench_metal")]
struct Cli {
    /// Path to SafeTensors model directory
    model: String,

    /// Number of tokens to generate
    #[arg(default_value_t = 128)]
    n_gen: usize,

    /// Enable per-kernel GPU timing (slower, but shows time breakdown)
    #[arg(long)]
    profile: bool,
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

fn bench_model<M: infernum::Model<B = MetalBackend> + Send + 'static>(
    model: M,
    n_gen: usize,
    ctx: &MetalContext,
) -> infernum::Result<()> {
    // 8-token prompt (same as other benchmarks for comparability)
    let prompt = vec![1u32, 15043, 29892, 920, 526, 366, 2599, 13];

    let engine = Engine::new(model)?;

    let options = GenerateOptions {
        max_new_tokens: n_gen,
        ..GenerateOptions::default()
    };

    // Reset stats so we only measure the generation phase
    ctx.reset_dispatch_stats();

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

    ctx.print_dispatch_stats();

    Ok(())
}

fn main() -> infernum::Result<()> {
    let cli = Cli::parse();

    if cli.model.ends_with(".gguf") {
        eprintln!("ERROR: GGUF loading is not yet supported on Metal. Pass a SafeTensors directory.");
        std::process::exit(1);
    }

    let mut ctx = MetalContext::new();
    if cli.profile {
        ctx.set_profile_per_kernel(true);
    }

    let model_type = detect_model_type(&cli.model)?;
    let family = match model_type.as_str() {
        "llama" | "mistral" | "mixtral" => "llama",
        "qwen2" | "qwen3" | "qwen3_moe" => "qwen",
        "gemma2" | "gemma3_text" => "gemma",
        other => {
            eprintln!("ERROR: Unsupported model_type: {other}");
            std::process::exit(1);
        }
    };

    eprintln!(
        "Loading: {} (Metal, SafeTensors, {model_type})",
        cli.model,
    );

    let model_dir = Path::new(&cli.model);

    match family {
        "llama" => {
            let model = LlamaMetalGraphEngine::from_pretrained(ctx.clone(), model_dir)?;
            let cfg = model.config();
            eprintln!(
                "Model: {} layers, {} hidden",
                cfg.num_hidden_layers, cfg.hidden_size,
            );
            bench_model(model, cli.n_gen, &ctx)
        }
        "qwen" => {
            let model = QwenMetalGraphEngine::from_pretrained(ctx.clone(), model_dir)?;
            let cfg = model.config();
            eprintln!(
                "Model: {} layers, {} hidden",
                cfg.num_hidden_layers, cfg.hidden_size,
            );
            bench_model(model, cli.n_gen, &ctx)
        }
        "gemma" => {
            let model = GemmaMetalGraphEngine::from_pretrained(ctx.clone(), model_dir)?;
            let cfg = model.config();
            eprintln!(
                "Model: {} layers, {} hidden",
                cfg.num_hidden_layers, cfg.hidden_size,
            );
            bench_model(model, cli.n_gen, &ctx)
        }
        other => panic!("Unsupported family: {other}"),
    }
}
