//! Minimal benchmark for infernum Metal decode and prefill throughput.
//! Measures raw forward pass performance with greedy decoding (no sampling).
//!
//! Usage:
//!   cargo run --release --example bench_metal --features metal -- /path/to/model 256
//!
//! Accepts either a SafeTensors directory or a GGUF file (.gguf).
//! Outputs labeled lines:
//!   prefill: N tokens in Xs = Y tok/s
//!   decode: N tokens in Xs = Y tok/s

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
    /// Path to SafeTensors model directory or GGUF file
    model: String,

    /// Number of tokens to generate (decode benchmark)
    #[arg(default_value_t = 128)]
    n_gen: usize,

    /// Number of prompt tokens for prefill benchmark (0 = skip)
    #[arg(long, default_value_t = 512)]
    n_prompt: usize,

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

fn detect_model_type_safetensors(model_path: &str) -> infernum::Result<String> {
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

fn detect_family_gguf(gguf_path: &Path) -> infernum::Result<&'static str> {
    let loader = infernum::weights::gguf::GgufLoader::from_file(infernum::path_to_utf8(gguf_path)?)?;
    let arch = loader
        .metadata()
        .get("general.architecture")
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_else(|| "llama".to_string());
    let family = match arch.as_str() {
        "llama" | "mistral" | "mixtral" => "llama",
        "gemma2" | "gemma3" => "gemma",
        other => {
            eprintln!("ERROR: Unsupported GGUF architecture: {other}");
            std::process::exit(1);
        }
    };
    Ok(family)
}

/// Benchmark prefill: time one forward pass over `n_prompt` tokens.
///
/// Performs one warmup call first to ensure Metal shaders are compiled,
/// then measures a single timed pass. Returns tokens/second.
fn bench_prefill<M: infernum::Model<B = MetalBackend>>(
    model: &M,
    n_prompt: usize,
    ctx: &MetalContext,
) -> infernum::Result<f64> {
    let prompt: Vec<u32> = (1..=u32::try_from(n_prompt).expect("n_prompt fits u32")).collect();

    // Warmup — compiles Metal shaders on first execution.
    ctx.reset_dispatch_stats();
    let _ = model.forward(&prompt)?;

    ctx.reset_dispatch_stats();
    let start = Instant::now();
    let _ = model.forward(&prompt)?;
    let elapsed = start.elapsed();

    Ok(n_prompt as f64 / elapsed.as_secs_f64())
}

fn bench_model<M: infernum::Model<B = MetalBackend> + Send + 'static>(
    model: M,
    n_gen: usize,
    n_prompt: usize,
    ctx: &MetalContext,
) -> infernum::Result<()> {
    // ── Prefill benchmark ────────────────────────────────────────────────────
    if n_prompt > 0 {
        let tok_s = bench_prefill(&model, n_prompt, ctx)?;
        println!(
            "prefill: {n_prompt} tokens = {tok_s:.1} tok/s",
        );
    }

    // ── Decode benchmark ─────────────────────────────────────────────────────
    // 8-token prompt (same as other benchmarks for comparability)
    let prompt = vec![1u32, 15043, 29892, 920, 526, 366, 2599, 13];

    let engine = Engine::new(model)?;

    let options = GenerateOptions {
        max_new_tokens: n_gen,
        ..GenerateOptions::default()
    };

    ctx.reset_dispatch_stats();

    let start = Instant::now();
    let tokens = engine.generate(&prompt, &options)?;
    let elapsed = start.elapsed();

    let generated = tokens.len() - prompt.len();
    let tok_s = generated as f64 / elapsed.as_secs_f64();

    println!(
        "decode: {generated} tokens in {:.2}s = {tok_s:.1} tok/s",
        elapsed.as_secs_f64(),
    );

    ctx.print_dispatch_stats();

    Ok(())
}

fn main() -> infernum::Result<()> {
    let cli = Cli::parse();

    let mut ctx = MetalContext::new();
    if cli.profile {
        ctx.set_profile_per_kernel(true);
    }

    let model_path = &cli.model;

    if model_path.ends_with(".gguf") {
        let gguf_path = Path::new(model_path);
        let family = detect_family_gguf(gguf_path)?;

        eprintln!("Loading: {model_path} (Metal, GGUF, {family})");

        match family {
            "llama" => {
                let model = LlamaMetalGraphEngine::from_gguf(ctx.clone(), gguf_path)?;
                let cfg = model.config();
                eprintln!("Model: {} layers, {} hidden", cfg.num_hidden_layers, cfg.hidden_size);
                bench_model(model, cli.n_gen, cli.n_prompt, &ctx)
            }
            "gemma" => {
                let model = GemmaMetalGraphEngine::from_gguf(ctx.clone(), gguf_path)?;
                let cfg = model.config();
                eprintln!("Model: {} layers, {} hidden", cfg.num_hidden_layers, cfg.hidden_size);
                bench_model(model, cli.n_gen, cli.n_prompt, &ctx)
            }
            other => panic!("Unsupported GGUF family: {other}"),
        }
    } else {
        let model_type = detect_model_type_safetensors(model_path)?;
        let family = match model_type.as_str() {
            "llama" | "mistral" | "mixtral" => "llama",
            "qwen2" | "qwen3" | "qwen3_moe" => "qwen",
            "gemma2" | "gemma3_text" => "gemma",
            other => {
                eprintln!("ERROR: Unsupported model_type: {other}");
                std::process::exit(1);
            }
        };

        eprintln!("Loading: {model_path} (Metal, SafeTensors, {model_type})");

        let model_dir = Path::new(model_path);

        match family {
            "llama" => {
                let model = LlamaMetalGraphEngine::from_pretrained(ctx.clone(), model_dir)?;
                let cfg = model.config();
                eprintln!("Model: {} layers, {} hidden", cfg.num_hidden_layers, cfg.hidden_size);
                bench_model(model, cli.n_gen, cli.n_prompt, &ctx)
            }
            "qwen" => {
                let model = QwenMetalGraphEngine::from_pretrained(ctx.clone(), model_dir)?;
                let cfg = model.config();
                eprintln!("Model: {} layers, {} hidden", cfg.num_hidden_layers, cfg.hidden_size);
                bench_model(model, cli.n_gen, cli.n_prompt, &ctx)
            }
            "gemma" => {
                let model = GemmaMetalGraphEngine::from_pretrained(ctx.clone(), model_dir)?;
                let cfg = model.config();
                eprintln!("Model: {} layers, {} hidden", cfg.num_hidden_layers, cfg.hidden_size);
                bench_model(model, cli.n_gen, cli.n_prompt, &ctx)
            }
            other => panic!("Unsupported family: {other}"),
        }
    }
}
