//! CPU text generation example (multi-architecture, SafeTensors + GGUF)
//!
//! Usage:
//!   cargo run --example generate_cpu --features cpu -- -m /path/to/model "Hello"
//!   cargo run --example generate_cpu --features cpu -- -m model.Q8_0.gguf "Hello"
//!   cargo run --example generate_cpu --features cpu -- -m /path/to/model --greedy "Hello"
//!   cargo run --example generate_cpu --features cpu -- -m /path/to/model -t 0.8 -p 0.95 "Hello"

use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use clap::Parser;

use infernum::tokenizer::{GgufTokenizer, LlamaTokenizer};
use infernum::Tokenizer as _;
use infernum::{GenerateOptions, Result, SamplingParams};
use infernum_cpu::CpuBackend;
use infernum_gemma::GemmaModel;
use infernum_llama::LlamaModel;
use infernum_qwen::QwenModel;
use infernum_runtime::{Engine, Runtime};

/// CPU text generation with LLMs (SafeTensors + GGUF)
///
/// Supports Llama, Mistral, Mixtral, Qwen, and Gemma model families.
/// SafeTensors models load as f32 (Llama family only).
/// GGUF models load quantized weights (Q8_0, Q4_0) with SIMD-accelerated
/// dequantize-on-the-fly inference. Architecture is auto-detected from the
/// GGUF metadata.
/// Uses KV cache and nucleus sampling by default.
#[derive(Parser)]
#[command(name = "generate_cpu")]
struct Cli {
    /// Path to model directory (SafeTensors) or .gguf file
    #[arg(short, long, default_value = "models/smollm2-360m")]
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

/// Run generation with a loaded model and tokenizer.
fn run_model<M>(model: M, tokenizer: Tokenizer, cli: &Cli) -> Result<()>
where
    M: infernum::Model<B = CpuBackend> + Send + 'static,
{
    let model_config = model.config();
    println!(
        "Model loaded ({} layers, vocab {})",
        model_config.num_layers,
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

/// Detect architecture from GGUF metadata.
fn detect_gguf_arch(path: &str) -> Result<String> {
    let loader = infernum::weights::gguf::GgufLoader::from_file(path)?;
    let arch = loader
        .metadata()
        .get("general.architecture")
        .and_then(infernum::GgufValue::as_str)
        .unwrap_or("llama")
        .to_string();
    Ok(arch)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let is_gguf = cli.model.ends_with(".gguf");

    if is_gguf {
        let arch = detect_gguf_arch(&cli.model)?;
        println!("Loading model from: {} (CPU, GGUF, arch={arch})", cli.model);

        let loader = infernum::weights::gguf::GgufLoader::from_file(&cli.model)?;
        let tokenizer = Tokenizer::Gguf(GgufTokenizer::from_gguf_metadata(loader.metadata())?);

        match arch.as_str() {
            "llama" => {
                let model = LlamaModel::<CpuBackend>::from_gguf(&(), Path::new(&cli.model))?;
                run_model(model, tokenizer, &cli)
            }
            "qwen2" => {
                let model = QwenModel::<CpuBackend>::from_gguf(&(), Path::new(&cli.model))?;
                run_model(model, tokenizer, &cli)
            }
            "gemma2" | "gemma3" => {
                let model = GemmaModel::<CpuBackend>::from_gguf(&(), Path::new(&cli.model))?;
                run_model(model, tokenizer, &cli)
            }
            _ => {
                eprintln!("Unsupported GGUF architecture: {arch}");
                eprintln!("Supported: llama, qwen2, gemma2, gemma3");
                std::process::exit(1);
            }
        }
    } else {
        println!("Loading model from: {} (CPU, SafeTensors)", cli.model);
        let model = LlamaModel::<CpuBackend>::from_pretrained(&(), &cli.model)?;
        let tokenizer = Tokenizer::HuggingFace(LlamaTokenizer::from_pretrained(&cli.model)?);
        run_model(model, tokenizer, &cli)
    }
}
