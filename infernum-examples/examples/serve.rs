//! HTTP server example
//!
//! Serves a model via the `OpenAI`-compatible Chat Completions API.
//!
//! # Usage
//!
//! ```text
//! cargo run --example serve --features cuda -- -m /path/to/model
//!
//! # Non-streaming
//! curl http://localhost:8080/v1/chat/completions \
//!   -H "Content-Type: application/json" \
//!   -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'
//!
//! # Streaming
//! curl http://localhost:8080/v1/chat/completions \
//!   -H "Content-Type: application/json" \
//!   -d '{"model":"default","messages":[{"role":"user","content":"Hello"}],"stream":true}'
//!
//! # List models
//! curl http://localhost:8080/v1/models
//! ```

use std::path::Path;

use clap::Parser;
use serde::Deserialize;

use infernum::cuda::CudaContext;
use infernum::tokenizer::LlamaTokenizer;
use infernum::{ChatTemplate, Result};
use infernum_deepseek::{DeepSeekModel, DeepSeekTemplate};
use infernum_llama::{Llama3Template, LlamaModel, MistralTemplate};
use infernum_qwen::{ChatMLTemplate, QwenModel};
use infernum_serve::{ModelEntry, Server};

/// Serve a model via the `OpenAI`-compatible Chat Completions API
#[derive(Parser)]
#[command(name = "serve")]
struct Cli {
    /// Path to model directory (`SafeTensors`)
    #[arg(short, long, env = "LLAMA_MODEL_PATH")]
    model: String,

    /// Model name to register (used in API requests)
    #[arg(long, default_value = "default")]
    name: String,

    /// Port to listen on
    #[arg(short, long, default_value_t = 8080)]
    port: u16,
}

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

fn select_template(model_type: &str) -> Box<dyn ChatTemplate> {
    match model_type {
        "mistral" => Box::new(MistralTemplate),
        "qwen2" | "qwen3" | "qwen3_moe" => Box::new(ChatMLTemplate),
        "deepseek_v3" => Box::new(DeepSeekTemplate),
        _ => Box::new(Llama3Template),
    }
}

fn print_usage(port: u16, name: &str) {
    eprintln!();
    eprintln!("Test with:");
    eprintln!("  curl http://localhost:{port}/v1/chat/completions \\");
    eprintln!(r#"    -H "Content-Type: application/json" \"#);
    eprintln!(
        "    -d '{{\"model\":\"{name}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}],\"stream\":true}}'"
    );
    eprintln!();
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    eprintln!("Loading model from: {}", cli.model);

    let ctx = CudaContext::new(0)?;
    eprintln!("CUDA context initialized");

    let model_type = detect_model_type(&cli.model)?;
    let tokenizer = LlamaTokenizer::from_pretrained(&cli.model)?;
    let template = select_template(&model_type);

    eprintln!("Model type: {model_type}");

    let entry = match model_type.as_str() {
        "llama" | "mistral" | "mixtral" => {
            let model = LlamaModel::<f32>::from_pretrained(&ctx, &cli.model)?;
            ModelEntry::new(&cli.name, model, tokenizer, template)
        }
        "qwen2" | "qwen3" | "qwen3_moe" => {
            let model = QwenModel::<f32>::from_pretrained(&ctx, &cli.model)?;
            ModelEntry::new(&cli.name, model, tokenizer, template)
        }
        "deepseek_v3" => {
            let model = DeepSeekModel::<f32>::from_pretrained(&ctx, &cli.model)?;
            ModelEntry::new(&cli.name, model, tokenizer, template)
        }
        other => {
            return Err(infernum::Error::UnsupportedModel(format!(
                "Unsupported model_type: {other}"
            )));
        }
    };

    eprintln!("Starting server on 0.0.0.0:{}", cli.port);
    print_usage(cli.port, &cli.name);

    let server = Server::builder()
        .add_model(entry)
        .bind(([0, 0, 0, 0], cli.port))
        .build();

    server
        .run()
        .await
        .map_err(|e| infernum::Error::Io(std::io::Error::other(e.to_string())))?;

    Ok(())
}
