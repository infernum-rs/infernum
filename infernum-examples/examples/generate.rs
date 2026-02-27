//! Text generation example supporting Llama, Qwen, and Gemma model families
//!
//! Usage:
//!   # SafeTensors directory (auto-detects model family):
//!   cargo run --example generate --features cuda -- -m /path/to/model "Hello"
//!   # GGUF file (Llama family only):
//!   cargo run --example generate --features cuda -- -m model.gguf "Hello"
//!   # Greedy decoding:
//!   cargo run --example generate --features cuda -- -m model.gguf --greedy "Hello"
//!   # Custom sampling:
//!   cargo run --example generate --features cuda -- -m model.gguf -t 0.8 -p 0.95 "Hello"

use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use clap::Parser;
use serde::Deserialize;

use infernum::tokenizer::{GgufTokenizer, LlamaTokenizer};
use infernum::Tokenizer as _;
use infernum::{GenerateOptions, Result, SamplingParams};
use infernum_cuda::cuda::CudaContext;
use infernum_cuda::CudaBackend;
use infernum_deepseek::DeepSeekModel;
use infernum_gemma::GemmaModel;
use infernum_llama::LlamaModel;
use infernum_qwen::QwenModel;
use infernum_runtime::{Engine, Runtime};

/// Text generation with Llama and Qwen models
///
/// Automatically detects .gguf files and loads tokenizer from GGUF metadata.
/// For SafeTensors directories, expects tokenizer.json alongside the weights.
/// The model family (Llama/Mistral/Mixtral vs Qwen) is auto-detected from config.json.
/// Uses KV cache and nucleus sampling by default.
#[derive(Parser)]
#[command(name = "generate")]
struct Cli {
    /// Path to model directory or .gguf file
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

    /// Maximum tokens to generate
    #[arg(short = 'n', long, default_value_t = 100)]
    max_tokens: usize,

    /// Use greedy (argmax) decoding instead of sampling
    #[arg(long)]
    greedy: bool,

    /// Disable KV cache (recompute full sequence each step)
    #[arg(long)]
    no_kv_cache: bool,

    /// Sampling temperature
    #[arg(short, long, default_value_t = 0.7)]
    temperature: f32,

    /// Nucleus sampling threshold
    #[arg(short = 'p', long, default_value_t = 0.9)]
    top_p: f32,

    /// RNG seed for sampling
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Repetition penalty factor (1.0 = disabled, >1.0 penalises repeated tokens)
    #[arg(short = 'r', long, default_value_t = 1.1)]
    repetition_penalty: f32,

    /// Number of recent tokens to consider for repetition penalty
    #[arg(long, default_value_t = 64)]
    repetition_penalty_window: usize,

    /// Maximum KV cache sequence length (default: min(model max, 4096))
    #[arg(long)]
    max_seq_len: Option<usize>,
}

/// Abstraction over tokenizer backends so we can use either one.
enum Tokenizer {
    HuggingFace(LlamaTokenizer),
    Gguf(GgufTokenizer),
}

impl infernum::Tokenizer for Tokenizer {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        match self {
            Self::HuggingFace(t) => t.encode(text, add_bos),
            Self::Gguf(t) => t.encode(text, add_bos),
        }
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        match self {
            Self::HuggingFace(t) => t.decode(ids),
            Self::Gguf(t) => t.decode(ids),
        }
    }

    fn decode_token(&self, id: u32) -> Result<String> {
        match self {
            Self::HuggingFace(t) => t.decode_token(id),
            Self::Gguf(t) => t.decode_token(id),
        }
    }

    fn eos_token_id(&self) -> u32 {
        match self {
            Self::HuggingFace(t) => t.eos_token_id(),
            Self::Gguf(t) => t.eos_token_id(),
        }
    }
}

impl Tokenizer {
    fn vocab_size(&self) -> usize {
        match self {
            Self::HuggingFace(t) => t.vocab_size(),
            Self::Gguf(t) => t.vocab_size(),
        }
    }
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

fn detect_model_type(model_path: &str) -> Result<String> {
    let config_path = Path::new(model_path).join("config.json");
    let content = std::fs::read_to_string(&config_path)?;
    let probe: ModelTypeProbe = serde_json::from_str(&content)?;
    Ok(probe.model_type)
}

/// Run generation with a model that implements the `Model` trait.
fn run_generate<M: infernum::Model + Send + 'static>(
    model: M,
    tokenizer: Tokenizer,
    num_layers: usize,
    hidden_size: usize,
    cli: &Cli,
) -> Result<()>
where
    M::B: infernum::DecodeBufferOps,
{
    println!(
        "Model loaded ({} layers, {} hidden, vocab {})",
        num_layers,
        hidden_size,
        tokenizer.vocab_size(),
    );

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
        use_kv_cache: !cli.no_kv_cache,
    };

    if let Some(ref params) = options.sampling {
        println!(
            "Sampling: temperature={}, top_p={}, seed={}, repetition_penalty={} (window={})",
            params.temperature,
            params.top_p,
            params.seed,
            params.repetition_penalty,
            params.repetition_penalty_window,
        );
    } else {
        println!("Decoding: greedy (argmax)");
    }
    println!(
        "KV cache: {}",
        if options.use_kv_cache {
            "enabled"
        } else {
            "disabled"
        }
    );

    print!("{}", cli.prompt);
    io::stdout().flush()?;

    let start = Instant::now();

    let (output_tokens, prompt_len) = if !options.use_kv_cache {
        let engine = Engine::new(model)?;
        let input_ids = tokenizer.encode(&cli.prompt, true)?;
        let prompt_len = input_ids.len();

        let mut opts = options.clone();
        opts.eos_token_id = Some(tokenizer.eos_token_id());

        let tokens = engine.generate(&input_ids, &opts)?;

        for &tok in &tokens[prompt_len..] {
            let text = tokenizer.decode_token(tok)?;
            print!("{text}");
            io::stdout().flush()?;
        }

        (tokens, prompt_len)
    } else {
        let runtime = Runtime::with_max_seq_len(model, tokenizer, cli.max_seq_len)?;
        let prompt_len = runtime.tokenizer().encode(&cli.prompt, true)?.len();
        let tokens = runtime.generate_stream(&cli.prompt, &options)?;
        (tokens, prompt_len)
    };

    let elapsed = start.elapsed();
    let generated = output_tokens.len() - prompt_len;

    println!();
    println!(
        "Generated {} tokens in {:.2}s ({:.1} tokens/sec)",
        generated,
        elapsed.as_secs_f64(),
        generated as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let is_gguf = cli.model.ends_with(".gguf");

    println!("Loading model from: {}", cli.model);

    let ctx = CudaContext::new(0)?;
    println!("CUDA context initialized");

    if is_gguf {
        // GGUF is Llama-family only
        let model = LlamaModel::<CudaBackend>::from_gguf(&ctx, &cli.model)?;
        let gguf_loader = infernum_cuda::GgufLoader::from_file(&cli.model)?;
        let tokenizer = Tokenizer::Gguf(GgufTokenizer::from_gguf_metadata(gguf_loader.metadata())?);
        let num_layers = model.config().num_hidden_layers;
        let hidden_size = model.config().hidden_size;
        run_generate(model, tokenizer, num_layers, hidden_size, &cli)
    } else {
        let model_type = detect_model_type(&cli.model)?;
        let tokenizer = Tokenizer::HuggingFace(LlamaTokenizer::from_pretrained(&cli.model)?);

        match model_type.as_str() {
            "llama" | "mistral" | "mixtral" => {
                let model = LlamaModel::<CudaBackend>::from_pretrained(&ctx, &cli.model)?;
                let num_layers = model.config().num_hidden_layers;
                let hidden_size = model.config().hidden_size;
                run_generate(model, tokenizer, num_layers, hidden_size, &cli)
            }
            "qwen2" | "qwen3" | "qwen3_moe" => {
                let model = QwenModel::<CudaBackend>::from_pretrained(&ctx, &cli.model)?;
                let num_layers = model.config().num_hidden_layers;
                let hidden_size = model.config().hidden_size;
                run_generate(model, tokenizer, num_layers, hidden_size, &cli)
            }
            "deepseek_v3" => {
                let model = DeepSeekModel::<CudaBackend>::from_pretrained(&ctx, &cli.model)?;
                let num_layers = model.config().num_hidden_layers;
                let hidden_size = model.config().hidden_size;
                run_generate(model, tokenizer, num_layers, hidden_size, &cli)
            }
            "gemma2" | "gemma3_text" => {
                let model = GemmaModel::<CudaBackend>::from_pretrained(&ctx, &cli.model)?;
                let num_layers = model.config().num_hidden_layers;
                let hidden_size = model.config().hidden_size;
                run_generate(model, tokenizer, num_layers, hidden_size, &cli)
            }
            other => Err(infernum::Error::UnsupportedModel(format!(
                "Unsupported model_type: \"{other}\". Supported: llama, mistral, mixtral, qwen2, qwen3, qwen3_moe, deepseek_v3, gemma2, gemma3_text"
            ))),
        }
    }
}
