//! Minimal benchmark for infernum CPU decode throughput
//! Measures raw forward pass performance with greedy decoding (no sampling)
//!
//! Usage:
//!   cargo run --release --example bench_cpu --features cpu -- /path/to/model.gguf 128
//!   cargo run --release --example bench_cpu --features cpu -- /path/to/safetensors_dir 128

use std::path::Path;
use std::time::Instant;

use clap::Parser;
use serde::Deserialize;

use infernum::GenerateOptions;
use infernum_cpu::CpuBackend;
use infernum_gemma::GemmaModel;
use infernum_llama::LlamaModel;
use infernum_qwen::QwenModel;
use infernum_runtime::Engine;

#[derive(Parser)]
#[command(name = "bench_cpu")]
struct Cli {
    /// Path to model directory (SafeTensors) or .gguf file
    model: String,

    /// Number of tokens to generate
    #[arg(default_value_t = 128)]
    n_gen: usize,

    /// Number of threads (default: all cores)
    #[arg(short = 'j', long)]
    threads: Option<usize>,
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

/// Detect architecture from GGUF metadata.
fn detect_gguf_arch(path: &str) -> infernum::Result<String> {
    let loader = infernum::weights::gguf::GgufLoader::from_file(path)?;
    let arch = loader
        .metadata()
        .get("general.architecture")
        .and_then(infernum::GgufValue::as_str)
        .unwrap_or("llama")
        .to_string();
    Ok(arch)
}

fn bench_model<M: infernum::Model<B = CpuBackend> + Send + 'static>(
    model: M,
    n_gen: usize,
) -> infernum::Result<()> {
    // 8-token prompt (same as GPU bench for comparability)
    let prompt = vec![1u32, 15043, 29892, 920, 526, 366, 2599, 13];

    let engine = Engine::new(model)?;

    let options = GenerateOptions {
        max_new_tokens: n_gen,
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

fn main() -> infernum::Result<()> {
    let cli = Cli::parse();

    if let Some(threads) = cli.threads {
        // Set thread count for rayon / internal parallelism if supported
        std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
    }

    let is_gguf = cli.model.ends_with(".gguf");

    let (family, model_type) = if is_gguf {
        let arch = detect_gguf_arch(&cli.model)?;
        let family = match arch.as_str() {
            "llama" => "llama",
            "qwen2" => "qwen",
            "gemma2" | "gemma3" => "gemma",
            other => panic!("Unsupported GGUF architecture: {other}"),
        };
        (family, arch)
    } else {
        let mt = detect_model_type(&cli.model)?;
        let family = match mt.as_str() {
            "llama" | "mistral" | "mixtral" => "llama",
            "qwen2" | "qwen3" | "qwen3_moe" => "qwen",
            "gemma2" | "gemma3_text" => "gemma",
            other => panic!("Unsupported model_type: {other}"),
        };
        (family, mt)
    };

    eprintln!(
        "Loading: {} (CPU, {}, {})",
        cli.model,
        if is_gguf { "GGUF" } else { "SafeTensors" },
        model_type,
    );

    match family {
        "llama" => {
            let model = if is_gguf {
                LlamaModel::<CpuBackend>::from_gguf(&(), Path::new(&cli.model))?
            } else {
                LlamaModel::<CpuBackend>::from_pretrained(&(), &cli.model)?
            };
            let cfg = model.config();
            eprintln!(
                "Model: {} layers, {} hidden, dtype={}",
                cfg.num_hidden_layers,
                cfg.hidden_size,
                model.dtype(),
            );
            bench_model(model, cli.n_gen)
        }
        "qwen" => {
            let model = if is_gguf {
                QwenModel::<CpuBackend>::from_gguf(&(), Path::new(&cli.model))?
            } else {
                QwenModel::<CpuBackend>::from_pretrained(&(), &cli.model)?
            };
            let cfg = model.config();
            eprintln!(
                "Model: {} layers, {} hidden, dtype={}",
                cfg.num_hidden_layers,
                cfg.hidden_size,
                model.dtype(),
            );
            bench_model(model, cli.n_gen)
        }
        "gemma" => {
            let model = if is_gguf {
                GemmaModel::<CpuBackend>::from_gguf(&(), Path::new(&cli.model))?
            } else {
                GemmaModel::<CpuBackend>::from_pretrained(&(), &cli.model)?
            };
            let cfg = model.config();
            eprintln!(
                "Model: {} layers, {} hidden, dtype={}",
                cfg.num_hidden_layers,
                cfg.hidden_size,
                model.dtype(),
            );
            bench_model(model, cli.n_gen)
        }
        other => panic!("Unsupported family: {other}"),
    }
}
