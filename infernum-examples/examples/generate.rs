//! Text generation example using Llama
//!
//! Usage:
//!   # SafeTensors directory:
//!   cargo run --example generate --features cuda -- -m /path/to/llama "Hello"
//!   # GGUF file:
//!   cargo run --example generate --features cuda -- -m model.gguf "Hello"
//!   # Greedy decoding:
//!   cargo run --example generate --features cuda -- -m model.gguf --greedy "Hello"
//!   # Custom sampling:
//!   cargo run --example generate --features cuda -- -m model.gguf -t 0.8 -p 0.95 "Hello"

use std::env;
use std::io::{self, Write};
use std::time::Instant;

use infernum::cuda::CudaContext;
use infernum::tokenizer::{GgufTokenizer, LlamaTokenizer};
use infernum::Tokenizer as _;
use infernum::Result;
use infernum_llama::{LlamaModel, SamplingParams};
use infernum_runtime::Runtime;

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

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let cli = parse_args(&args);

    let is_gguf = cli.model_path.ends_with(".gguf");

    println!("Loading model from: {}", cli.model_path);

    // Initialize CUDA
    let ctx = CudaContext::new(0)?;
    println!("CUDA context initialized");

    // Load model + tokenizer based on file type
    let (model, tokenizer) = if is_gguf {
        let model = LlamaModel::from_gguf(&ctx, &cli.model_path)?;
        let gguf_loader = infernum::GgufLoader::from_file(&cli.model_path)?;
        let tokenizer = Tokenizer::Gguf(GgufTokenizer::from_gguf_metadata(gguf_loader.metadata())?);
        (model, tokenizer)
    } else {
        let model = LlamaModel::from_pretrained(&ctx, &cli.model_path)?;
        let tokenizer = Tokenizer::HuggingFace(LlamaTokenizer::from_pretrained(&cli.model_path)?);
        (model, tokenizer)
    };

    println!(
        "Model loaded ({} layers, {} hidden, vocab {})",
        model.config().num_hidden_layers,
        model.config().hidden_size,
        tokenizer.vocab_size(),
    );

    // Print decoding strategy
    if !cli.greedy {
        println!(
            "Sampling: temperature={}, top_p={}, seed={}",
            cli.sampling.temperature, cli.sampling.top_p, cli.sampling.seed
        );
    } else {
        println!("Decoding: greedy (argmax)");
    }
    println!(
        "KV cache: {}",
        if cli.no_kv_cache { "disabled" } else { "enabled" }
    );

    print!("{}", cli.prompt);
    io::stdout().flush()?;

    let start = Instant::now();

    let (output_tokens, prompt_len) = if cli.no_kv_cache {
        // Naive generation: recompute full sequence each step (no KV cache)
        let input_ids = tokenizer.encode(&cli.prompt, true)?;
        let prompt_len = input_ids.len();
        let eos = Some(tokenizer.eos_token_id());

        let tokens = if cli.greedy {
            model.generate(&input_ids, cli.max_tokens, eos)?
        } else {
            model.generate_sampled(&input_ids, cli.max_tokens, eos, &cli.sampling)?
        };

        // Stream output tokens
        for &tok in &tokens[prompt_len..] {
            let text = tokenizer.decode_token(tok)?;
            print!("{text}");
            io::stdout().flush()?;
        }

        (tokens, prompt_len)
    } else {
        // KV-cached generation via Runtime
        let runtime = Runtime::new(ctx, model, tokenizer);

        let params = if cli.greedy {
            None
        } else {
            Some(&cli.sampling)
        };

        let prompt_len = runtime.tokenizer().encode(&cli.prompt, true)?.len();
        let tokens = runtime.generate_streaming(&cli.prompt, cli.max_tokens, params)?;
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

struct CliArgs {
    model_path: String,
    prompt: String,
    max_tokens: usize,
    greedy: bool,
    no_kv_cache: bool,
    sampling: SamplingParams,
}

fn parse_args(args: &[String]) -> CliArgs {
    let mut model_path = String::new();
    let mut prompt = String::new();
    let mut max_tokens = 100;
    let mut greedy = false;
    let mut no_kv_cache = false;
    let mut temperature: Option<f32> = None;
    let mut top_p: Option<f32> = None;
    let mut seed: Option<u64> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                if i < args.len() {
                    model_path = args[i].clone();
                }
            }
            "--max-tokens" | "-n" => {
                i += 1;
                if i < args.len() {
                    max_tokens = args[i].parse().unwrap_or(100);
                }
            }
            "--temperature" | "-t" => {
                i += 1;
                if i < args.len() {
                    temperature = Some(args[i].parse().unwrap_or(0.7));
                }
            }
            "--top-p" | "-p" => {
                i += 1;
                if i < args.len() {
                    top_p = Some(args[i].parse().unwrap_or(0.9));
                }
            }
            "--seed" | "-s" => {
                i += 1;
                if i < args.len() {
                    seed = Some(args[i].parse().unwrap_or(42));
                }
            }
            "--greedy" => {
                greedy = true;
            }
            "--no-kv-cache" => {
                no_kv_cache = true;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            arg if !arg.starts_with('-') => {
                // First non-flag positional: if it looks like a model path
                // (ends in .gguf or is a directory path), treat it as model.
                // Otherwise treat it as the prompt.
                if model_path.is_empty()
                    && (arg.ends_with(".gguf") || std::path::Path::new(arg).is_dir())
                {
                    model_path = arg.to_string();
                } else if prompt.is_empty() {
                    prompt = arg.to_string();
                }
            }
            _ => {}
        }
        i += 1;
    }

    if model_path.is_empty() {
        model_path =
            env::var("LLAMA_MODEL_PATH").unwrap_or_else(|_| "models/llama-3.2-1b".to_string());
    }

    if prompt.is_empty() {
        prompt = "Hello".to_string();
    }

    let defaults = SamplingParams::default();
    let sampling = SamplingParams {
        temperature: temperature.unwrap_or(defaults.temperature),
        top_p: top_p.unwrap_or(defaults.top_p),
        seed: seed.unwrap_or(defaults.seed),
    };

    CliArgs {
        model_path,
        prompt,
        max_tokens,
        greedy,
        no_kv_cache,
        sampling,
    }
}

fn print_usage() {
    eprintln!("Usage: generate [OPTIONS] [MODEL_PATH] [PROMPT]");
    eprintln!();
    eprintln!("Arguments:");
    eprintln!("  [MODEL_PATH]             Path to model directory or .gguf file");
    eprintln!("  [PROMPT]                 Text prompt (default: \"Hello\")");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  -m, --model <PATH>       Path to model directory or .gguf file");
    eprintln!("  -n, --max-tokens <N>     Maximum tokens to generate (default: 100)");
    eprintln!("  -t, --temperature <F>    Sampling temperature (default: 0.7)");
    eprintln!("  -p, --top-p <F>          Nucleus sampling threshold (default: 0.9)");
    eprintln!("  -s, --seed <N>           RNG seed for sampling (default: 42)");
    eprintln!("      --greedy             Use greedy (argmax) decoding instead of sampling");
    eprintln!("      --no-kv-cache        Disable KV cache (recompute full sequence each step)");
    eprintln!("  -h, --help               Show this help message");
    eprintln!();
    eprintln!("Automatically detects .gguf files and loads tokenizer from GGUF metadata.");
    eprintln!("For SafeTensors directories, expects tokenizer.json alongside the weights.");
    eprintln!("Uses KV cache and nucleus sampling by default.");
    eprintln!("Pass --greedy for deterministic argmax.");
    eprintln!();
    eprintln!("Examples:");
    eprintln!(
        "  cargo run --example generate --features cuda -- model.Q4_0.gguf \"Hello, world!\""
    );
    eprintln!(
        "  cargo run --example generate --features cuda -- -m /path/to/model --greedy \"Hello\""
    );
    eprintln!("  cargo run --example generate --features cuda -- -m model.gguf -t 0.8 -p 0.95 \"Once upon a time\"");
}
