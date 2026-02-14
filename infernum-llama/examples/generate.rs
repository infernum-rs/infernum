//! Text generation example using Llama
//!
//! Usage:
//!   cargo run --example generate --features cuda -- --model /path/to/llama "Hello"
//!   cargo run --example generate --features cuda -- --model /path/to/llama --greedy "Hello"
//!   cargo run --example generate --features cuda -- --model /path/to/llama -t 0.8 -p 0.95 "Hello"

use std::env;
use std::io::{self, Write};

use infernum::cuda::CudaContext;
use infernum::tokenizer::LlamaTokenizer;
use infernum::Result;
use infernum_llama::{LlamaModel, SamplingParams};
use infernum_runtime::Runtime;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let cli = parse_args(&args);

    println!("Loading model from: {}", cli.model_path);

    // Initialize CUDA
    let ctx = CudaContext::new(0)?;
    println!("CUDA context initialized");

    // Load tokenizer
    let tokenizer = LlamaTokenizer::from_pretrained(&cli.model_path)?;
    println!("Tokenizer loaded (vocab size: {})", tokenizer.vocab_size());

    // Load model
    let model = LlamaModel::from_pretrained(&ctx, &cli.model_path)?;
    println!(
        "Model loaded ({} layers, {} hidden)",
        model.config().num_hidden_layers,
        model.config().hidden_size
    );

    // Create runtime
    let runtime = Runtime::new(ctx, model, tokenizer);

    // Generate
    let params = if cli.greedy {
        None
    } else {
        Some(&cli.sampling)
    };

    if !cli.greedy {
        let s = params.unwrap();
        println!(
            "Sampling: temperature={}, top_p={}, seed={}",
            s.temperature, s.top_p, s.seed
        );
    } else {
        println!("Decoding: greedy (argmax)");
    }

    print!("{}", cli.prompt);
    io::stdout().flush()?;

    let output_tokens = runtime.generate_streaming(&cli.prompt, cli.max_tokens, params)?;

    println!();
    println!("Generated {} tokens total", output_tokens.len());

    Ok(())
}

struct CliArgs {
    model_path: String,
    prompt: String,
    max_tokens: usize,
    greedy: bool,
    sampling: SamplingParams,
}

fn parse_args(args: &[String]) -> CliArgs {
    let mut model_path = String::new();
    let mut prompt = String::new();
    let mut max_tokens = 100;
    let mut greedy = false;
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
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            arg if !arg.starts_with('-') && prompt.is_empty() => {
                prompt = arg.to_string();
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
        sampling,
    }
}

fn print_usage() {
    eprintln!("Usage: generate [OPTIONS] [PROMPT]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  -m, --model <PATH>       Path to model directory (default: $LLAMA_MODEL_PATH or models/llama-3.2-1b)");
    eprintln!("  -n, --max-tokens <N>     Maximum tokens to generate (default: 100)");
    eprintln!("  -t, --temperature <F>    Sampling temperature (default: 0.7)");
    eprintln!("  -p, --top-p <F>          Nucleus sampling threshold (default: 0.9)");
    eprintln!("  -s, --seed <N>           RNG seed for sampling (default: 42)");
    eprintln!("      --greedy             Use greedy (argmax) decoding instead of sampling");
    eprintln!("  -h, --help               Show this help message");
    eprintln!();
    eprintln!("Uses KV cache and nucleus sampling by default.");
    eprintln!("Pass --greedy for deterministic argmax.");
    eprintln!();
    eprintln!("Examples:");
    eprintln!(
        "  cargo run --example generate --features cuda -- -m /path/to/model \"Hello, world!\""
    );
    eprintln!(
        "  cargo run --example generate --features cuda -- -m /path/to/model --greedy \"Hello\""
    );
    eprintln!("  cargo run --example generate --features cuda -- -m /path/to/model -t 0.8 -p 0.95 \"Once upon a time\"");
}
